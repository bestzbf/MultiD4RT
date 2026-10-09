"""Static view grouping preserves measured geometry and invents no motion."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.datasets.partial_dense import attach_partial_dense_recipe
from multicamera4rc.datasets.static_views import select_static_views


def configuration(tmp_path, count=10, choices=None, pairs=True):
    h,w=27,39
    path=tmp_path/'static.h5'
    K=np.array([[20.,0.,19.],[0.,20.,13.],[0.,0.,1.]],np.float32)
    E=np.repeat(np.eye(4,dtype=np.float32)[None],count,0)
    E[:,0,3]=-np.arange(count)*.05
    with h5py.File(path,'w') as f:
        g=f.create_group('train/scene')
        g.create_dataset('images',data=np.stack([np.full((h,w,3),10+20*i,np.uint8) for i in range(count)]))
        g.create_dataset('depths',data=np.full((count,h,w),2.,np.float32))
        g.create_dataset('intrinsics',data=np.repeat(K[None],count,0))
        g.create_dataset('extrinsics',data=E)
        if pairs:
            neighbors=np.array([[i]+[j for j in range(count) if j!=i] for i in range(count)],np.int64)
            g.create_dataset('pairs',data=neighbors)
    return dict(image_mode='native',supervision='dense',sampling='coverage',epoch_size=1,num_times=1,seed=49,
        sources=[dict(adapter='mugen_h5',root=str(path),name='blended',num_times=1,
            cameras=[4,8] if choices is None else choices,view_mode='static_multiview',
            options=dict(dataset_name='blended',generate_static_tracks=False,prefer_precomputed_tracks=False,
                         train_only_val_ratio=0.,verbose=False))])


def test_static_grouping_preserves_geometry_and_uses_one_shared_time(tmp_path):
    torch.set_num_threads(1)
    dataset=create_dataset(configuration(tmp_path))
    assert len(dataset)==dataset.coverage_size==10
    samples=[dataset[i] for i in range(10)]
    assert {len(s['images']) for s in samples}=={4,8}
    assert {s['static_multiview']['view_indices'][0] for s in samples}==set(range(10))
    for index,sample in enumerate(samples):
        views=sample['static_multiview']['view_indices']; C=len(views)
        assert len(set(views))==C
        assert sample['camera_id'].tolist()==list(range(C))
        assert sample['time_idx'].tolist()==[0]*C and sample['query_time_idx'].tolist()==[0]
        assert sample['images'].shape==(C,3,28,42)
        assert sample['images'][:,0,0,0].mul(255).round().int().tolist()==[10+20*v for v in views]
        assert dataset[index]['static_multiview']==sample['static_multiview']
        raw=dataset.sources[0]['adapter'].load_clip('scene',views)
        temporal=prepare_clip(raw,num_queries=0,image_mode='native')
        attach_partial_dense_recipe(raw,temporal,kind='geometry',image_mode='native')
        old=materialize_dense_targets(collate_adapter_samples([temporal]),torch.arange(C)[None])
        new=materialize_dense_targets(collate_adapter_samples([sample]),torch.arange(C)[None])
        for key in ['depth','K','c2w','ray','scene_scale']:
            torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)
        assert new['trajectory'].shape==(1,C,1,28,42,3)
        assert new['trajectory_valid'].sum()==new['source_pixel_valid'].sum()==C*27*39
        assert not new['displacement_valid'].any() and not new['static_mask'].any()
        assert not new['depth_valid'][...,27:,:].any() and not new['depth_valid'][...,:,39:].any()
        radii=new['trajectory'][new['trajectory_valid']].norm(dim=-1)
        torch.testing.assert_close(radii.mean(),torch.tensor(1.),rtol=1e-6,atol=1e-6)


def test_covisibility_cannot_reintroduce_excluded_frames(tmp_path):
    config=configuration(tmp_path)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(root=config['sources'][0]['root'],sequences={
        'scene':dict(valid_frames=[0,2,3,6,9])})))
    config['sources'][0]['manifest']=str(manifest)
    dataset=create_dataset(config)
    assert len(dataset)==5
    for i in range(5):
        sample=dataset[i]
        assert len(sample['images'])==4
        assert set(sample['static_multiview']['view_indices'])<=set([0,2,3,6,9])


def test_packed_pair_geometry_verification_rejects_wrong_world_depth(tmp_path):
    config=configuration(tmp_path,choices=4)
    with h5py.File(config['sources'][0]['root'],'r+') as f:
        E=f['train/scene/extrinsics'][1];E[2,3]=1.
        f['train/scene/extrinsics'][1]=E
    sample=create_dataset(config)[0]
    assert len(sample['images'])==4 and 1 not in sample['static_multiview']['view_indices']
    assert sample['static_multiview']['selection']=='packed_depth_covisibility'


def test_static_validation_relative_output_path_keeps_heldout_manifest(tmp_path,monkeypatch):
    from scripts.evaluate_static_multiview import validation_dataset
    config=configuration(tmp_path,count=4)
    with h5py.File(config['sources'][0]['root'],'r+') as f:
        f.copy('train/scene','train/held')
        f['train/held/images'][...]=f['train/held/images'][...]+1
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(root=config['sources'][0]['root'],
        sequences={'scene':dict(length=4,physical_scene='training_room')},
        validation_sequences={'held':dict(length=4,physical_scene='heldout_room')})))
    config['sources'][0]['manifest']=str(manifest)
    (tmp_path/'config.json').write_text(json.dumps(config));monkeypatch.chdir(tmp_path)
    dataset,splits=validation_dataset('config.json',Path('validation'))
    assert dataset.sources[0]['pool']==[('held',4)]
    assert splits['blended']['scene_overlap']==0
    assert dataset[0]['images'].shape==(4,3,28,42)


@pytest.mark.parametrize('pairs,count,expected',[(True,3,3),(False,10,1)])
def test_short_or_isolated_neighborhood_never_duplicates_views(tmp_path,pairs,count,expected):
    dataset=create_dataset(configuration(tmp_path,count=count,pairs=pairs))
    assert len(dataset)==count
    for i in range(count):
        sample=dataset[i]
        assert len(sample['images'])==expected
        assert len(set(sample['static_multiview']['view_indices']))==expected
        assert sample['query_time_idx'].tolist()==[0]


@pytest.mark.parametrize('changes',[
    dict(cameras=[]),dict(cameras=[4,4]),dict(cameras=[True,8]),dict(cameras=[0,8]),
    dict(num_times=2),dict(dense_kind='replica_flow'),dict(view_mode='temporal'),
    dict(options=dict(dataset_name='pointodyssey',generate_static_tracks=False,verbose=False)),
])
def test_static_contract_rejects_invalid_or_dynamic_grouping(tmp_path,changes):
    config=configuration(tmp_path)
    config['sources'][0].update(changes)
    with pytest.raises(ValueError): create_dataset(config)


def test_static_multiview_parent_h5_is_safe_for_spawn_worker(tmp_path):
    torch.set_num_threads(1)
    dataset=create_dataset(configuration(tmp_path,count=4,choices=4))
    expected=dataset[0]
    loader=DataLoader(dataset,batch_size=1,num_workers=1,multiprocessing_context='spawn',
                      collate_fn=collate_adapter_samples,timeout=30)
    actual=next(iter(loader))
    torch.testing.assert_close(actual['images'][0],expected['images'])
    assert actual['static_multiview']==[expected['static_multiview']]


def test_renderer_pose_neighbors_respect_direction_and_admitted_pool():
    E=np.repeat(np.eye(4)[None],10,axis=0)
    E[9,:3,:3]=np.diag([-1.,1.,-1.])  # opposite viewing direction
    centers=np.zeros((10,3));centers[:,0]=np.arange(10)
    adapter=SimpleNamespace(records={'scene':dict(E=E,camera_centers=centers)})
    views,info=select_static_views(adapter,'scene',[0,2,3,4,5,6,9],0,(4,8),random.Random(4))
    assert views[0]==0 and len(set(views))==4
    assert set(views)<=set([0,2,3,4,5,6])
    assert info['selection']=='renderer_pose_neighbors' and info['available_neighbors']==5


def test_static_anchor_must_be_admitted():
    with pytest.raises(ValueError,match='admitted'):
        select_static_views(SimpleNamespace(),'scene',[1,2,3],0,(4,),random.Random(0))


def test_renderer_depth_frustum_rejects_nonoverlapping_view():
    K=np.array([[20.,0.,15.5],[0.,20.,15.5],[0.,0.,1.]])
    E=np.repeat(np.eye(4)[None],3,axis=0)
    E[1,0,3]=20.;E[2,0,3]=.1
    record=dict(E=E,camera_centers=-E[:,:3,3],K=K,native_hw=(32,32))
    clip=SimpleNamespace(depths=[np.full((32,32),2.)],intrinsics=K[None],extrinsics=E[:1])
    adapter=SimpleNamespace(records={'scene':record},load_clip=lambda seq,ids:clip)
    views,info=select_static_views(adapter,'scene',[0,1,2],0,(4,),random.Random(0))
    assert views==[0,2] and info['actual_cameras']==2
    assert info['selection']=='renderer_depth_frustum' and info['available_neighbors']==1
