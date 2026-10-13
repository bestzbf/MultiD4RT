"""Copy compact benchmark evidence, excluding weights and large predictions."""
import argparse
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='B300zbf')
    p.add_argument('--campaign',default='/data6/benchmarks/evaluation/multid4rt60k_comparison_20261013')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    code='''from pathlib import Path
import tarfile
p=Path(CAMPAIGN)
paths=[]
for folder in [p/'sintel16',p/'tapvid16_pstudio_full_eval']:
 for f in folder.glob('*.json'): paths.append((f,str(f.relative_to(p))))
 for method in ['multid4rt','omnix']:
  f=folder/method/'results.json'
  if f.exists():paths.append((f,str(f.relative_to(p))))
for f in (p/'tapvid16_pstudio_full_eval/legacy_raw_intrinsics').rglob('*.json'):
 paths.append((f,str(f.relative_to(p))))
for suffix in ['*.json','*.md','*.png']:
 for f in (p/'pstudio_diagnosis_20261013').rglob(suffix):
  paths.append((f,str(f.relative_to(p))))
for method in ['multid4rt','omnix']:
 folder=p/'multiview46'/method
 for name in ['run.json','progress.json','cache_migration.json','score/results.json','score/per_sequence.csv']:
  f=folder/name
  if f.exists(): paths.append((f,str(f.relative_to(p))))
 for f in folder.glob('*/*.diagnostics.json'): paths.append((f,str(f.relative_to(p))))
 f=p/'camera_sweep'/method/'results.json'
 if f.exists(): paths.append((f,str(f.relative_to(p))))
f=p/'multiview46/comparison.json'
if f.exists():paths.append((f,str(f.relative_to(p))))
for name in ['omnix_queries_parity.json','omnix_queries_compact_parity.json','omnix_queries_bounded_parity.json']:
 f=p/name
 if f.exists():paths.append((f,'source/'+name))
for f in (p/'code/evaluation/references').glob('*.json'):paths.append((f,'source/'+f.name))
paths.append((Path('/data6/benchmarks/evaluation/protocols/tracking-main.json'),'multiview46/plan.json'))
with tarfile.open(p/'compact_reports.tar.gz','w:gz') as tar:
 for f,name in paths:tar.add(f,arcname=name,recursive=False)
print(len(paths),'report files')
'''.replace('CAMPAIGN',repr(args.campaign))
    subprocess.run(['ssh',args.host,'python -'],input=code,text=True,check=True)
    args.output.mkdir(parents=True,exist_ok=True)
    archive=args.output/'compact_reports.tar.gz'
    remote=f'{args.host}:{shlex.quote(args.campaign+"/compact_reports.tar.gz")}'
    subprocess.run(['scp','-q',remote,str(archive)],check=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                raise ValueError('Unexpected archive entry')
        tar.extractall(args.output)


if __name__=='__main__': main()
