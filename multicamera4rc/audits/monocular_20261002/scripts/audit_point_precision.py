"""Explain PointOdyssey release precision using original NPZ and local H5."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True);args=p.parse_args()
    raw=json.loads((args.report_dir/'pointodyssey.json').read_text())
    result=dict(dataset='PointOdyssey',samples=[],
        source_reference='https://github.com/y-zheng18/point_odyssey/blob/main/utils/gen_tracking.py',
        interpretation='Native cameras, trajectories and image coordinates were half-precision quantized; the official generator also snaps UV near segmentation boundaries. Strict sub-0.05px checks in the initial report are diagnostics, not an eligibility gate for this source.')
    for sample in raw['samples']:
        seq=sample['sequence'];base=args.root/'train'/seq
        with h5py.File(base/'anno.h5','r') as f,np.load(base/'anno.npz',allow_pickle=False) as original:
            fields={}
            for name in ['intrinsics','extrinsics']:
                a=np.asarray(f[name]);b=original[name]
                fields[name]=dict(shape=list(a.shape),h5_dtype=str(a.dtype),npz_dtype=str(b.dtype),
                    h5_matches_original_npz=bool(np.array_equal(a,b)),
                    max_h5_original_difference=float(np.max(np.abs(a-b))),
                    fp16_exact_fraction=float((a==a.astype(np.float16).astype(a.dtype)).mean()))
            uv=np.asarray(f['trajs_2d'][[0,len(f['trajs_2d'])//2,len(f['trajs_2d'])-1]])
            fields['trajs_2d_sample']=dict(h5_dtype=str(uv.dtype),sampled_frames=3,
                fp16_exact_fraction=float((uv==uv.astype(np.float16).astype(uv.dtype)).mean()))
            item=dict(sequence=seq,fields=fields,
                observed_reprojection_p95_px=sample['reprojection_px_given_visible']['p95'],
                observed_depth_relative_p95=sample['visible_depth_3x3_relative_error']['p95'],
                usable_for_dense_geometry_with_valid_masks=bool(sample['checks']['finite_valid_3d'] and
                    sample['checks']['visible_vertex_depth_agrees'] and sample['checks']['native_depth_available']),
                authoritative_native_track_only=True,full_per_pixel_dynamic_trajectories=False)
            result['samples'].append(item)
        print(json.dumps({'sequence':seq,'original_cameras_equal':all(fields[n]['h5_matches_original_npz'] for n in ['intrinsics','extrinsics'])}),flush=True)
    passed=bool(result['samples']) and all(s['usable_for_dense_geometry_with_valid_masks'] and
        all(s['fields'][n]['h5_matches_original_npz'] for n in ['intrinsics','extrinsics']) for s in result['samples'])
    result['stage']='sampled_native_precision_consistent' if passed else 'needs_review'
    (args.report_dir/'pointodyssey_precision_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['stage'],flush=True)


if __name__=='__main__':main()
