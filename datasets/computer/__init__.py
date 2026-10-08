"""
D4RT offline precompute module.

Derives normals, trajs_2d, trajs_3d_world, valids, visibs from
depth + intrinsics + extrinsics for static-scene datasets.

Core algorithms:
    depth_to_normals.compute_normals  - depth + K → normal map
    depth_to_tracks.compute_tracks    - depth + K + E → tracks

Run scripts (one per dataset):
    run_scannet.py
    run_co3dv2.py
    run_blendedmvs.py
    run_mvssynth.py
"""
