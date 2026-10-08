
import json,pathlib,sys,torch
from multicamera4rc.losses import MultiCamera4RCLoss,_resize_scalar
import scripts.evaluate_static_multiview as evaluator
root=pathlib.Path('/data8/zbf/multicamera4rc_runs')
out=root/'blended_scene_audit_20261008/depth_bias'
plan=json.loads((out/'fixed_plan.json').read_text())
labels=['53000','55000','55500_replay1']
class DiagnosticLoss(MultiCamera4RCLoss):
 calls=0
 def forward(self,prediction,target):
  result=super().forward(prediction,target)
  p=_resize_scalar(prediction['depth'],tuple(target['depth'].shape[-2:])).float()
  g=target['depth'].float()
  mask=target['depth_valid'].bool()
  ratio=(p[mask]/g[mask])
  # A diagnostic only. Official metrics always keep the original prediction.
  global_scale=torch.median(g[mask]/p[mask])
  views=[]
  for i in range(p.shape[1]):
   valid=mask[0,i];pv=p[0,i][valid];gv=g[0,i][valid]
   residual=(pv-gv)/gv
   scale=torch.median(gv/pv)
   views.append(dict(view=plan['cases'][self.calls%len(plan['cases'])]['view_indices'][i],
    valid_pixels=len(pv),abs_rel=float(residual.abs().mean()),
    signed_rel_mean=float(residual.mean()),signed_rel_median=float(residual.median()),
    rel_error_p90=float(torch.quantile(residual.abs(),.9)),
    rel_error_p98=float(torch.quantile(residual.abs(),.98)),
    median_gt_over_prediction=float(scale),
    per_view_median_scale_aligned_abs_rel=float(((pv*scale-gv).abs()/gv).mean()),
    clip_median_scale_aligned_abs_rel=float(((pv*global_scale-gv).abs()/gv).mean())))
  case=plan['cases'][self.calls%len(plan['cases'])]
  row=dict(checkpoint=labels[self.calls//len(plan['cases'])],case=case,
    depth_abs_rel=float(((p[mask]-g[mask]).abs()/g[mask]).mean()),
    signed_rel_mean=float((ratio-1).mean()),signed_rel_median=float((ratio-1).median()),
    median_gt_over_prediction=float(global_scale),
    clip_median_scale_aligned_abs_rel=float(((p[mask]*global_scale-g[mask]).abs()/g[mask]).mean()),
    views=views)
  with (out/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  DiagnosticLoss.calls+=1
  return result
evaluator.MultiCamera4RCLoss=DiagnosticLoss
refs={'53000':'syn4d_activation_20261007/validation/baseline_53000/static/evaluation.json','55000':'syn4d_complete_20261008/validation/baseline_55000/static/evaluation.json','55500_replay1':'syn4d_replay_20261008/validation/extended_55500_try1/static/evaluation.json'}
checkpoints=[k+'='+next(iter(json.loads((root/v).read_text())['checkpoints'].values()))['path'] for k,v in refs.items()]
sys.argv=['audit','--train-config','/data9/zbf/my_dfrt_multicamera4rc/outputs/training_preflight/static_multiview_20261002/full_mixture_static_multiview.yaml','--plan',str(out/'fixed_plan.json'),'--output',str(out),'--memory-fraction','.12','--checkpoints',*checkpoints]
evaluator.main()
