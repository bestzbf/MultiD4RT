"""Check fused whole-query camera attention against the dense reference."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from multicamera4rc.attention import relation_attention


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.manual_seed(53);torch.set_num_threads(2)
    report=dict(device=torch.cuda.get_device_name(),dtype='bfloat16',samples=[])
    for d,fp32_qk in ((64,False),(96,False),(64,True),(96,True)):
        q=torch.randn(2,3,77,d,device='cuda',dtype=torch.float32 if fp32_qk else torch.bfloat16)
        k=torch.randn(2,3,99,d,device='cuda',dtype=q.dtype)
        v=torch.randn_like(k);weight=torch.tensor([0.,.2,-.7],device='cuda')
        cq=torch.arange(77,device='cuda')[None].repeat(2,1)%4
        ck=torch.arange(99,device='cuda')[None].repeat(2,1)%4;cq[:,0]=-1
        values=[x.clone().requires_grad_() for x in (q,k,v,weight)]
        reference=[x.clone().requires_grad_() for x in (q,k,v,weight)]
        with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            actual=relation_attention(*values[:3],cq,ck,values[3],query_chunk_size=None)
        qr,kr,vr,wr=reference
        same=(cq[:,:,None]==ck[:,None,:]) & (cq[:,:,None]>=0)
        bias=same[:,None].to(q.dtype)*wr.to(q.dtype)[None,:,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
            expected=F.scaled_dot_product_attention(qr,kr,vr,attn_mask=bias)
        probe=torch.randn_like(actual)
        (actual.float()*probe.float()).sum().backward()
        (expected.float()*probe.float()).sum().backward()
        error=float((actual.detach().float()-expected.detach().float()).norm()/expected.detach().float().norm())
        gradient_errors=[float((a.grad.float()-b.grad.float()).norm()/b.grad.float().norm().clamp_min(1e-8))
            for a,b in zip(values,reference)]
        assert error<.015 and max(gradient_errors)<.03,(d,error,gradient_errors)
        assert torch.isfinite(values[3].grad).all() and values[3].grad[0].abs()>0
        report['samples'].append(dict(head_dim=d,fp32_qk_under_autocast=fp32_qk,forward_relative_l2=error,
            q_k_v_weight_gradient_relative_l2=gradient_errors,forced_flash_backend_passed=True,
            zero_initialized_weight_gradient=float(values[3].grad[0])))
    report['passed']=True;args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
