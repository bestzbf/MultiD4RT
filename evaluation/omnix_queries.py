"""Query-only memory adapter for the pristine OmniX trajectory generator.

All encoder observations, ST attention and learned heads are unchanged. The
original final trajectory function is evaluated on selected source images;
that function has no attention or mixing between its source-image entries.
The original target axis and physical reference-time indices are preserved.
Two unused, quadratic auxiliary allocations are omitted in a private AST
copy. No official source checkout is edited. Actual release-output parity is
required before the benchmark runner accepts this adapter.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
from pathlib import Path
import textwrap
from types import MethodType, SimpleNamespace

import torch

from .common import digest


def omit_unused_allocations(instance, mode):
    original = instance.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    removed,projected = [],[]

    class Omit(ast.NodeTransformer):
        def visit_Assign(self, node):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if mode=='st' and names & {'out_feature','indices_feat'}:
                removed.append(ast.dump(node,include_attributes=False))
                return None
            if mode=='st' and names=={'motion_vec'}:
                if ast.unparse(node.value)!='self.motion_proj_list[idx](dynamic_token).float()':
                    raise ValueError('Official independent motion projection changed')
                projected.append(ast.dump(node,include_attributes=False))
                node.value=ast.Name(id='dynamic_token',ctx=ast.Load())
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if mode=='st' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute):
                obj = node.value.func.value
                if isinstance(obj,ast.Name) and obj.id=='out_feature' and node.value.func.attr=='scatter_':
                    removed.append(ast.dump(node,include_attributes=False))
                    return None
                if isinstance(obj,ast.Name) and obj.id=='out_dynamic_token_list' and node.value.func.attr=='append':
                    if len(node.value.args)!=1 or ast.unparse(node.value.args[0])!='dynamic_token':
                        raise ValueError('Official ST feature capture changed')
                    projected.append(ast.dump(node,include_attributes=False))
                    node.value.args=[ast.parse('self.motion_proj_list[len(out_dynamic_token_list)](dynamic_token).float()',mode='eval').body]
            return self.generic_visit(node)

        def visit_If(self, node):
            test = ast.unparse(node.test)
            if mode=='model' and test=='self.offset_head.pred_conf':
                # This block only copies already-computed confidence to every
                # target frame. Neither confidence nor the copy affects XYZ.
                if {n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)} != {'sampling_conf','trajectory_conf'}:
                    raise ValueError('Official auxiliary confidence block changed')
                removed.append(ast.dump(node,include_attributes=False))
                return None
            node = self.generic_visit(node)
            if not node.body: node.body=[ast.Pass()]
            return node

    modified = Omit().visit(tree)
    if len(removed)!=(4 if mode=='st' else 1):
        raise ValueError(f'Unexpected number of unused allocations: {mode}, {len(removed)}')
    if mode=='st' and any(isinstance(n,ast.Name) and n.id in {'out_feature','indices_feat'} for n in ast.walk(modified)):
        raise ValueError('The discarded ST feature is used by an output')
    ast.fix_missing_locations(modified)
    namespace = dict(original.__func__.__globals__)
    source_file = Path(inspect.getsourcefile(original))
    exec(compile(modified,str(source_file)+'#query-only-auxiliary', 'exec'),namespace)
    instance.forward=MethodType(namespace['forward'],instance)
    if mode=='st' and len(projected)!=2:
        raise ValueError('Expected exactly one early projection and one later projection removal')
    return dict(source_sha256=digest(source_file),removed_ast=removed,early_motion_projection_ast=projected)


def chunk_independent_mlp(module,chunk=8192):
    """Original tokenwise MLP, with bounded temporaries and identical weights."""
    original=module.forward

    def forward(_module,x):
        if torch.is_grad_enabled():
            raise RuntimeError('The bounded MLP adapter is inference-only')
        flat=x.reshape(-1,x.shape[-1])
        if flat.shape[0]<=chunk:
            return original(x)
        piece=original(flat[:chunk])
        output=torch.empty((len(flat),piece.shape[-1]),device=piece.device,dtype=piece.dtype)
        output[:chunk]=piece
        for start in range(chunk,len(flat),chunk):
            output[start:start+chunk]=original(flat[start:start+chunk])
        return output.reshape(*x.shape[:-1],output.shape[-1])

    module.forward=MethodType(forward,module)
    return dict(chunk=chunk,source_sha256=digest(inspect.getsourcefile(original)),
                original_function='tokenwise fc1/activation/dropout/fc2/dropout; no attention or token mixing')


def stream_motion_max_pool(head,chunk=8):
    """Max over all target times, avoiding four enormous projected tensors."""
    original=head.forward
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    changes=[]

    class Pool(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='motion_feature_mp':
                code=ast.unparse(node.value)
                if code=='[self.motion_proj_mp(motion_field) for motion_field in motion_field_list]':
                    changes.append(ast.dump(node,include_attributes=False))
                    node.value=ast.parse('[self._comparison_pool_motion(motion_field) for motion_field in motion_field_list]',mode='eval').body
                elif code=='torch.max(motion_feature_mp, dim=2)[0]':
                    changes.append(ast.dump(node,include_attributes=False))
                    return None
            return self.generic_visit(node)

    tree=Pool().visit(tree)
    if len(changes)!=2: raise ValueError('Official target-time max-pooling changed')
    ast.fix_missing_locations(tree)
    namespace=dict(original.__func__.__globals__)
    source=Path(inspect.getsourcefile(original))
    exec(compile(tree,str(source)+'#bounded-motion-pool','exec'),namespace)

    def pool(_head,field):
        pooled=None
        for start in range(0,field.shape[2],chunk):
            projected=_head.motion_proj_mp(field[:,:,start:start+chunk].contiguous())
            reduced=projected.max(dim=2)[0]
            pooled=reduced if pooled is None else torch.maximum(pooled,reduced)
        return pooled

    head._comparison_pool_motion=MethodType(pool,head)
    head.forward=MethodType(namespace['forward'],head)
    return dict(source_sha256=digest(source),changed_ast=changes,target_chunk=chunk,
                proof='channelwise max commutes with concatenation; streaming max includes every original target slot')


@dataclass
class OmniXScene:
    geometry: dict
    image_size: tuple[int,int]
    frames: int
    points: torch.Tensor
    offsets: dict
    fields: list
    dynamic: dict
    time_info: torch.Tensor
    cache_key: tuple=()
    cache: torch.Tensor|None=None


class OmniXQueries:
    supports_visibility=False

    def __init__(self,model):
        if model.pretrain or model.enable_gs:
            raise ValueError('Expected the complete released tracking network, without GS')
        self.model=model
        self.config=SimpleNamespace(patch_size=14)
        self.original_trajectory=model.trajectory_mod.forward
        pointwise_heads=all(isinstance(p,torch.nn.Sequential) and len(p)==3
                           and isinstance(p[0],torch.nn.Linear) and isinstance(p[1],torch.nn.ReLU)
                           and not p[1].inplace and isinstance(p[2],torch.nn.Linear)
                           for p in model.st_mod.motion_proj_list)
        if model.st_mod.use_gs_field or model.st_mod.motion_dim!=9 or not pointwise_heads:
            raise ValueError('Early projection requires released, independent pointwise motion heads without GS')
        self.provenance=dict(st_auxiliary=omit_unused_allocations(model.st_mod,'st'),
            model_auxiliary=omit_unused_allocations(model,'model'),
            trajectory_source_sha256=digest(inspect.getsourcefile(self.original_trajectory)),
            target_mapping='canonical first-camera slots for shared global_time_idx; preserves official synced-video semantics',
            source_selection='original source-independent final trajectory function, evaluated only for requested sources',
            preserves='all encoder inputs, every ST target slot, attention, learned heads, original physical reference times')
        self.provenance['tokenwise_mlp']=[chunk_independent_mlp(block.mlp) for block in model.st_mod.blocks]
        self.provenance['offset_motion_max_pool']=stream_motion_max_pool(model.offset_head)

    @torch.inference_mode()
    def encode(self,images,camera_id,time_idx):
        if images.shape[0]!=1:
            raise ValueError('Evaluation expects one complete sequence at a time')
        s=images.shape[1]
        cameras=torch.unique_consecutive(camera_id[0])
        frames=int(time_idx.max())+1
        if s!=len(cameras)*frames or not torch.equal(time_idx[0],torch.arange(frames,device=images.device).repeat(len(cameras))):
            raise ValueError('OmniX comparison requires complete synchronized camera-major order')
        camera_slots=torch.arange(len(cameras),device=images.device).repeat_interleave(frames)
        info=torch.stack((torch.arange(s,device=images.device),camera_slots,time_idx[0],time_idx[0]),-1)[None]
        captured={}

        def capture(_module,points,offsets,**kwargs):
            captured.update(points=points,offsets=offsets,fields=kwargs['motion_field_list'],
                            dynamic=kwargs['dynamic_token_info'],time_info=kwargs['time_info'])
            return dict(pts3d_trajectory=None,pts3d_dynamic_score=None)

        original=self.model.trajectory_mod.forward
        self.model.trajectory_mod.forward=MethodType(capture,self.model.trajectory_mod)
        try:
            output=self.model(dict(image=images,image_info=info))
        finally:
            self.model.trajectory_mod.forward=original
        geometry=dict(base_points=output['pts3d'],c2w=output['camera_pose'],K=output['intrinsics'])
        return OmniXScene(geometry,tuple(images.shape[-2:]),frames,**captured)

    @torch.inference_mode()
    def decode(self,scene,query_time_idx,source_obs_idx,return_visibility=False):
        if return_visibility:
            raise ValueError('OmniX has no official occlusion predictor')
        key=tuple(source_obs_idx[0].tolist())
        if scene.cache_key!=key:
            sources=source_obs_idx[0]
            s=scene.points.shape[1]
            def select(value):
                if torch.is_tensor(value) and value.ndim>=2 and value.shape[1]==s:
                    return value[:,sources]
                raise ValueError('Unexpected official source-array shape')
            offsets={k:select(v) for k,v in scene.offsets.items()}
            dynamic={k:select(v) for k,v in scene.dynamic.items()}
            # Keep the full original target axis; normalizing by reference time
            # must happen before discarding other cameras' target slots.
            result=self.original_trajectory(scene.points[:,sources],offsets,
                motion_field_list=[f[:,sources] for f in scene.fields],
                dynamic_token_info=dynamic,time_info=scene.time_info[:,sources])
            scene.cache=result['pts3d_trajectory'][:,:,:scene.frames].contiguous()
            scene.cache_key=key
        return dict(trajectory=scene.cache[:,:,query_time_idx[0]])


def make_info(c,t,device):
    slots=torch.arange(c*t,device=device)
    frame=torch.arange(t,device=device).repeat(c)
    camera=torch.arange(c,device=device).repeat_interleave(t)
    return torch.stack((slots,camera,frame,frame),-1)[None],camera[None],frame[None]


@torch.inference_mode()
def parity_case(model,images,c,t):
    images=images[:,:c,:t] if images.ndim==6 else images
    images=images.reshape(1,c*t,*images.shape[-3:])
    info,camera,times=make_info(c,t,images.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        expected=model(dict(image=images,image_info=info))
    # Save only the values used by our benchmark. Every other native output
    # remains unnecessary to the downstream query adapter.
    expected={k:expected[k] for k in ['trajectory','pts3d','camera_pose','intrinsics']}
    st_forward,model_forward=model.st_mod.forward,model.forward
    mlp_forward=[block.mlp.forward for block in model.st_mod.blocks]
    offset_forward=model.offset_head.forward
    try:
        wrapper=OmniXQueries(model)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            scene=wrapper.encode(images,camera,times)
            pieces=[]
            for start in range(0,c*t,4):
                sources=torch.arange(start,min(start+4,c*t),device=images.device)[None]
                pieces.append(wrapper.decode(scene,times[:,:t],sources)['trajectory'])
            actual=torch.cat(pieces,1)
        errors={}
        for name,value,truth in [('trajectory',actual,expected['trajectory'][:,:,:t]),
                                 ('pts3d',scene.geometry['base_points'],expected['pts3d']),
                                 ('camera_pose',scene.geometry['c2w'],expected['camera_pose']),
                                 ('intrinsics',scene.geometry['K'],expected['intrinsics'])]:
            torch.testing.assert_close(value,truth,rtol=2e-5,atol=5e-6)
            errors[name]=float((value.float()-truth.float()).abs().max())
        return dict(cameras=c,frames=t,source_chunk=4,maximum_absolute_errors=errors,passed=True,
                    adapter_provenance=wrapper.provenance)
    finally:
        model.st_mod.forward,model.forward=st_forward,model_forward
        for block,forward in zip(model.st_mod.blocks,mlp_forward): block.mlp.forward=forward
        model.offset_head.forward=offset_forward
        if hasattr(model.offset_head,'_comparison_pool_motion'): del model.offset_head._comparison_pool_motion
