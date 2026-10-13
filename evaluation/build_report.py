"""Build offline, self-contained comparison tables from complete reports."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path


LABELS={'multid4rt':'MultiD4RT · 60k','omnix':'OmniX · release'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args();root=args.directory
    sintel=json.loads((root/'sintel16/comparison.json').read_text())
    tap=json.loads((root/'tapvid16_pstudio_full_eval/comparison.json').read_text())
    mv=json.loads((root/'multiview46/comparison.json').read_text())
    if any(r['stage']!='passed' or not r['identical_rgb'] for r in [sintel,tap,mv]):
        raise ValueError('All full paired comparisons must pass before publishing tables')
    rows=[]
    def add(dataset,metric,unit,direction,counts,values,protocol):
        rows.append(dict(dataset=dataset,metric=metric,unit=unit,direction=direction,samples=counts,
            multid4rt=values['multid4rt'],omnix=values['omnix'],protocol=protocol))
    add('PStudio · 单相机','APD3D','%', '↑','106 视频 / 954 × T16',
        {m:100*v['apd3d'] for m,v in tap['methods'].items()},'RGB + 2D 查询；官方 depth-adaptive 阈值、每段全局 median scale；视频宏平均')
    add('PStudio · 单相机','EPE','m','↓','106 视频 / 954 × T16',
        {m:v['epe_m'] for m,v in tap['methods'].items()},'预测目标相机坐标；与 APD 相同尺度，GT 可见点欧氏误差；视频宏平均')
    for title,key,unit,scale,direction in [
        ('AbsRel · S','depth_S_Abs Rel','',1,'↓'),('AbsRel · SS','depth_SS_Abs Rel','',1,'↓'),
        ('δ₁ · S','depth_S_δ < 1.25','%',100,'↑'),('δ₁ · SS','depth_SS_δ < 1.25','%',100,'↑'),
        ('RMSE · S','depth_S_RMSE','m',1,'↓'),('RMSE · SS','depth_SS_RMSE','m',1,'↓')]:
        add('Sintel · 深度',title,unit,direction,'14 序列 / 37 × T16',
            {m:scale*v[key] for m,v in sintel['methods'].items()},'4RC/CUT3R 官方函数；GT <70m；每段一个 S 或 SS 对齐；片段宏平均')
    add('Sintel · 位姿','RPE-R · 全 37 段','°','↓','37 × T16',
        {m:v['all37_rotation_rpe_rmse_deg'] for m,v in sintel['supplement'].items()},'evo 相邻帧旋转 RMSE；无需轨迹尺度对齐；双方旋转做有界 SO(3) 数值校正')
    for title,key in [('ATE · Sim3','ate_rmse_m'),('RPE-T · Sim3','rpe_translation_rmse_m')]:
        add('Sintel · 位姿',title,'m','↓','16 个 GT 非退化片段',
            {m:v[key] for m,v in sintel['methods'].items()},'evo；GT 相机中心秩不足的 21 段同等排除；不是全 37 段位姿平均')
    names={'panoptic':'Panoptic · 多相机','dexycb':'DexYCB · 多相机','mv-kubric':'MV-Kubric · 多相机'}
    for name,label in names.items():
        pair={r['method']:r for r in mv['rows'] if r['dataset']==name}
        count=pair['multid4rt']['sequences']
        times='T150' if name=='panoptic' else 'T24/25' if name=='dexycb' else 'T24'
        condition=f'{count} 完整序列 / C4{times}'
        for title,key,unit,direction in [('Δavg','delta_avg_percent','%','↑'),
            ('MTE','mte','cm' if name!='mv-kubric' else '原始场景单位 ×100','↓'),
            ('ATE · tracking','ate','cm' if name!='mv-kubric' else '原始场景单位 ×100','↓'),
            ('Survival','survival_percent','%','↑')]:
            add(label,title,unit,direction,condition,{m:v[key] for m,v in pair.items()},
                '官方固定查询、完整长度、官方 scorer；RGB 网络 + 标定坐标对齐 + 给定 3D 查询；any 组' if name=='panoptic' else
                '官方固定查询、完整长度、官方 scorer；RGB 网络 + 标定坐标对齐 + 给定 3D 查询；dynamic-static-mean 组')
        add(label,'AJ3D（OmniX 无官方遮挡头）','%','↑',condition,
            {m:v['aj3d_percent'] for m,v in pair.items()},'OmniX 存档 all-visible 占位不作为遮挡预测指标')
    sweep={m:json.loads((root/'camera_sweep'/m/'results.json').read_text()) for m in LABELS}
    if any(v['stage']!='passed' or len(v['records'])!=12 for v in sweep.values()):
        raise ValueError('Camera diagnostic incomplete')
    a,b=[{(r['scene'],r['cameras']):r for r in v['records']} for v in sweep.values()]
    if a.keys()!=b.keys(): raise ValueError('Different camera diagnostic cases')
    for key in a:
        if any(a[key][k]!=b[key][k] for k in ['query_points_sha256','point_ids','official_cache_sha256','frames']):
            raise ValueError('Camera sweep changed fixed query identity')
        if a[key]['diagnostics']['preprocessing']!=b[key]['diagnostics']['preprocessing']:
            raise ValueError('Camera sweep changed RGB')
    sweep_rows=[]
    for c in [4,6,8,10]:
        item=dict(cameras=c,frames=8,sequences=3)
        for method,report in sweep.items():
            subset=[r for r in report['records'] if r['cameras']==c]
            metrics={}
            for title,key in [('delta_avg_percent','average_pts_within_thresh'),('mte','mte_visible')]:
                values=[r['metrics'][f'3dpt/model__{key}__any'] for r in subset]
                if any(v is None for v in values): raise ValueError('Undefined camera diagnostic metric')
                metrics[title]=sum(values)/len(values)
            item[method]=metrics
        sweep_rows.append(item)
    diagnostics=json.loads((root/'tapvid16_pstudio_full_eval/error_diagnostics.json').read_text())
    result=dict(stage='passed',created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        camera_sweep=sweep_rows,error_diagnostics={m:v['mean'] for m,v in diagnostics['methods'].items()},
        sources=dict(omnix='https://github.com/yanqinJiang/OmniX',tapvid3d='https://github.com/google-deepmind/tapnet',
                     mvtracker='https://github.com/ethz-vlg/mvtracker',depth='https://github.com/Luo-Yihang/4RC',
                     evo='https://github.com/MichaelGrupp/evo',d4rt='https://d4rt-paper.github.io/'),
        limitations=[
            '这是共同预处理和明确固定协议下的发布权重重跑；OmniX 的论文专用完整评测 wrapper 未发布，不能把论文表值直接当作这里的基线。',
            'D4RT 官方页面未找到可运行模型代码/权重；未将 OpenD4RT 等非官方复现冒充官方 D4RT。',
            'N3DV 是新视角渲染任务；本轮没有把几何/追踪输出转换成 PSNR、SSIM、LPIPS，未运行渲染对比。',
            '不报告公平速度对比：OmniX CUDA attention/deformable kernels 在 B300 上使用等价 PyTorch 实现。',
            '相机数量诊断只用固定 1、15、30 三个 MV-Kubric 场景、T8、固定 first-frame 查询与四个参考视角 GT 可见性；它与完整 C4T24 官方测试分开。',
            'OmniX 的内存适配保留全部输入、attention、原目标槽和时间语义；实际 C2T4/C4T8/C4T24 与原完整输出差异为 0；40 个已完成预测通过逐数组与评分一致性核验复用。'])
    (root/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (root/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=json.dumps(result,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MultiD4RT 60k · 共同协议评测</title>
<style>body{margin:0;background:#10141b;color:#e5eaf2;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1240px;margin:40px auto;padding:0 28px}h1{font-size:30px;margin:0 0 8px}h2{font-size:21px;margin-top:34px}.sub,small{color:#a4b0c4}a{color:#87bcff}.tag{display:inline-block;background:#163c2e;color:#91e3b9;border-radius:20px;padding:4px 12px;margin:12px 6px 18px 0}select{background:#19212d;color:white;border:1px solid #344458;border-radius:8px;padding:8px 14px;margin:10px 0}table{width:100%;border-collapse:collapse;background:#151d28;border-radius:10px;overflow:hidden}td,th{padding:10px 13px;border-bottom:1px solid #293647;text-align:left}th{font-weight:600;color:#b8c6d9}td.num{text-align:right;font-variant-numeric:tabular-nums}.win{color:#a5e6bd;font-weight:650}.note{background:#172233;border-left:3px solid #5c9ee5;padding:14px 18px;margin:18px 0}.protocol{font-size:12px;color:#a9b6c9}details{margin:20px 0}li{margin:8px 0}.links{display:flex;gap:18px;flex-wrap:wrap}.scroll{overflow:auto}footer{margin:36px 0;color:#8e9bb0;font-size:12px}@media(max-width:700px){main{padding:0 14px}h1{font-size:23px}td,th{padding:8px;min-width:70px}}</style>
<main><h1>MultiD4RT 60k · 与 OmniX 共同协议评测</h1><div class="sub">固定发布权重 · 相同 RGB、查询、GT 和评分器 · 2026-10-13</div><span class="tag">PStudio 106/106 视频</span><span class="tag">多相机 46/46 序列</span><span class="tag">相机数量 24/24 实测组合</span>
<p>本页比较真实推理结果。多相机测试的输入条件包含标定坐标对齐与给定三维查询，单目 PStudio 使用二维查询；这两个条件单独解读。</p>
<div class="links"><a href="metrics.csv">下载指标 CSV</a><a href="results.json">完整汇总 JSON</a><a href="multiview46/comparison.json">多相机核验与结果</a><a href="source/omnix_queries_bounded_parity.json">OmniX 原输出一致性核验</a></div>
<h2>完整测试集对照</h2><label>评测项 <select id="dataset"><option value="">全部</option></select></label><div class="scroll"><table><thead><tr><th>数据与指标</th><th>样本</th><th>MultiD4RT 60k</th><th>OmniX release</th><th>协议</th></tr></thead><tbody id="metrics"></tbody></table></div>
<h2>相机数量诊断 · 固定三场景，T8</h2><p class="sub">固定首帧三维查询及四个参考视角的 GT 可见性，只增加输入 RGB 相机。以下为 any 组均值；完整官方 MV-Kubric 表使用 dynamic-static-mean 组。</p><div class="scroll"><table><thead><tr><th>相机数量</th><th>60k Δavg ↑</th><th>OmniX Δavg ↑</th><th>60k MTE ↓</th><th>OmniX MTE ↓</th></tr></thead><tbody id="cameras"></tbody></table></div><small>MTE 单位为原始 Kubric 场景单位 ×100；诊断不替代完整 30 序列测试。</small>
<h2>PStudio 误差分解</h2><div id="diagnostic" class="note"></div><p class="sub">起始与后续误差使用相同的片段尺度；增量误差包含预测相机误差。这些是诊断数值，未修改官方 APD 或 EPE，也没有向模型提供 GT 三维锚点。</p>
<details><summary>协议、来源及适用范围</summary><ul id="limits"></ul><div id="sources" class="links"></div><p>逐序列原始结果：<a href="sintel16/multid4rt/results.json">Sintel 60k</a> · <a href="sintel16/omnix/results.json">Sintel OmniX</a> · <a href="tapvid16_pstudio_full_eval/multid4rt/results.json">PStudio 60k</a> · <a href="tapvid16_pstudio_full_eval/omnix/results.json">PStudio OmniX</a> · <a href="multiview46/multid4rt/score/per_sequence.csv">多相机 60k</a> · <a href="multiview46/omnix/score/per_sequence.csv">多相机 OmniX</a></p><p>同一 Sintel 对照另附等场景宏平均：<a href="sintel16/comparison.json">完整统计及退化 GT 清单</a>。原始 PStudio queries_xyt 字段的重投影审计单独保存；当前 T16 协议从首帧可见 XYZ 投影生成 XY，不消费原始 queries_xyt。</p></details><footer>仅对应发布的 60,000 步固定权重，不代表后续训练 checkpoint。更高为优 ↑，更低为优 ↓；— 表示没有对应预测头或无法得到同协议结果。</footer></main>
<script>const D=__DATA__;const fmt=(x,u)=>x===null?'—':x.toFixed(u==='%'?2:4)+(u?' '+u:'');function cell(text,cls=''){let e=document.createElement('td');e.textContent=text;e.className=cls;return e}function render(){let body=document.getElementById('metrics');body.replaceChildren();D.rows.filter(r=>!dataset.value||r.dataset===dataset.value).forEach(r=>{let tr=document.createElement('tr'),name=cell(r.dataset+' · '+r.metric+' '+r.direction);tr.append(name,cell(r.samples));let sign=r.direction==='↑'?1:-1;['multid4rt','omnix'].forEach(m=>{let other=m==='multid4rt'?'omnix':'multid4rt',win=r[m]!==null&&r[other]!==null&&sign*(r[m]-r[other])>0;tr.append(cell(fmt(r[m],r.unit),'num'+(win?' win':'')))});tr.append(cell(r.protocol,'protocol'));body.append(tr)})}const dataset=document.getElementById('dataset');[...new Set(D.rows.map(r=>r.dataset))].forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;dataset.append(o)});dataset.onchange=render;render();D.camera_sweep.forEach(r=>{let tr=document.createElement('tr');tr.append(cell(r.cameras+' 相机 × '+r.frames+' 帧'));['delta_avg_percent','mte'].forEach(k=>['multid4rt','omnix'].forEach(m=>tr.append(cell(r[m][k].toFixed(2)+(k==='delta_avg_percent'?'%':''),'num'))));document.getElementById('cameras').append(tr)});let e=D.error_diagnostics;document.getElementById('diagnostic').textContent='起始三维位置 EPE：60k '+(e.multid4rt.source_epe_m*1000).toFixed(1)+' mm，OmniX '+(e.omnix.source_epe_m*1000).toFixed(1)+' mm。后续位置 EPE：'+(e.multid4rt.future_epe_m*1000).toFixed(1)+' / '+(e.omnix.future_epe_m*1000).toFixed(1)+' mm；运动增量误差：'+(e.multid4rt.future_displacement_error_m*1000).toFixed(1)+' / '+(e.omnix.future_displacement_error_m*1000).toFixed(1)+' mm。';D.limitations.forEach(t=>{let li=document.createElement('li');li.textContent=t;document.getElementById('limits').append(li)});Object.entries(D.sources).forEach(([name,url])=>{let a=document.createElement('a');a.href=url;a.textContent=name;a.target='_blank';a.rel='noopener';document.getElementById('sources').append(a)});</script></html>'''
    (root/'comparison.html').write_text(template.replace('__DATA__',payload))
    print(root/'comparison.html')


if __name__=='__main__': main()
