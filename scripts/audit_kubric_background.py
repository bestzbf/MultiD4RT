"""Verify the dome mask against raw renderer coordinates on a fixed plan."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import attach_kubric_dense_recipe, materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples, read_data_config
from multicamera4rc.evaluate import fixed_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config',required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    config=read_data_config(args.data_config)
    for source in config['sources']:source['split']='val'
    dataset=create_dataset(config)
    plan=json.loads(Path(args.plan).read_text())
    records=[]
    for index,case in enumerate(plan['cases']):
        source=dataset.sources[case['layout']]
        clip=source['adapter'].load_clip(case['sequence'],case['times']*source['cameras'],
            layout_num_cameras=source['cameras'],layout_num_times=source['num_times'],sampling_seed=case['sampling_seed'])
        sample=prepare_clip(clip,image_mode='native',num_queries=0)
        images=sample['images'].clone()
        attach_kubric_dense_recipe(clip,sample,image_mode='native',include_visibility=True,background_policy='objects_ground')
        assert torch.equal(images,sample['images'])
        batch=collate_adapter_samples([sample]); sources=fixed_sources(batch,4)
        target=materialize_dense_targets(batch,sources)
        valid=target['depth_valid'][0].numpy()
        excluded=target['excluded_background'][0].numpy()
        old_valid=batch['targets']['depth_valid'][0].numpy()
        normalised_base=target['ray'][...,:3]*target['depth'][...,None]+target['ray'][...,3:]
        normalised_range=normalised_base.norm(dim=-1)
        torch.testing.assert_close(normalised_range[target['depth_valid']].mean(),torch.tensor(1.),atol=2e-6,rtol=0)
        expected=np.zeros_like(valid); objects=np.zeros_like(valid); ground=np.zeros_like(valid)
        raw_distance=[]; old_distance=[]
        E=np.asarray(clip.extrinsics[0],dtype='f8')
        reference_center=np.linalg.inv(E)[:3,3]
        with h5py.File(clip.metadata['h5_path'],'r') as f:
            for s,(c,t) in enumerate(zip(sample['camera_id'].tolist(),sample['time_idx'].tolist())):
                depth=f['depth'][t,c];ids=f['segmentation'][t,c];h,w=depth.shape
                yy,xx=np.mgrid[:h,:w]
                rays=np.stack((xx+.5,yy+.5,np.ones_like(xx)),-1)@np.linalg.inv(f['intrinsics'][c]).T
                rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
                pose=f['extrinsics'][c]
                world=(rays*depth[...,None])@pose[:3,:3].T+pose[:3,3]
                mask=(ids>0)|((ids==0)&(abs(world[...,2])<=.05))
                expected[s,:h,:w]=mask
                objects[s,:h,:w]=ids>0
                ground[s,:h,:w]=(ids==0)&(abs(world[...,2])<=.05)
                distance=np.linalg.norm(world-reference_center,axis=-1)
                raw_distance.append(distance[mask]);old_distance.append(distance.ravel())
        assert np.array_equal(valid,expected & old_valid),'Raw H5 world-plane mask disagrees'
        assert np.array_equal(excluded,old_valid&~expected)
        assert valid[objects].all(),'An object pixel was lost'
        assert valid[ground].all(),'A ground pixel was lost'
        wanted=float(np.concatenate(raw_distance).mean());actual=float(target['scene_scale'][0])
        assert abs(wanted-actual)<2e-5,(wanted,actual)
        source_bad=target['excluded_background'][:,sources[0]]
        assert not target['source_pixel_valid'][source_bad].any()
        for key in ['trajectory_valid','displacement_valid']:
            assert not target[key][source_bad[:,:,None].expand_as(target[key])].any()
        assert not target['static_mask'][source_bad].any()
        assert not target['visibility_valid'][source_bad[:,:,None,None].expand_as(target['visibility_valid'])].any()
        record=dict(case=index,sequence=case['sequence'],layout=[source['cameras'],source['num_times']],
            original_pixels=int(old_valid.sum()),excluded_pixels=int(excluded.sum()),
            excluded_fraction=float(excluded.sum()/old_valid.sum()),
            objects_retained=int(objects.sum()),ground_retained=int(ground.sum()),
            old_scale_m=float(np.concatenate(old_distance).mean()),filtered_scale_m=actual,
            raw_h5_mask_exact=True,all_target_masks_passed=True)
        records.append(record);print(json.dumps(record),flush=True)
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(dict(passed=len(records)==len(plan['cases']),cases=records),indent=2)+'\n')


if __name__=='__main__':main()
