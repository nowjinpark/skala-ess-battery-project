"""Package reviewed DAY 1+2 local deliverables; exclude raw files and runtime."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=ROOT/'results/day2'
 for name in ['validation.json','report_validation.json','notebook_validation.json']:
  q=json.loads((d/name).read_text());assert q['status']=='passed',name
 pdf=ROOT/'output/pdf/DS-MINI-Result-울산_1반-박진원.pdf';nb=ROOT/'notebooks/02_DAY2_Modeling.ipynb'
 assert json.loads((d/'report_validation.json').read_text())['sha256']==sha(pdf)
 nq=json.loads((d/'notebook_validation.json').read_text());assert nq['sha256']==sha(nb)
 assert (ROOT/'docs/final_day2_review.md').exists()
 files=[ROOT/n for n in ['README.md','requirements.txt','.gitignore','data/source_manifest.json','data/DATA_SOURCE.md']]
 for folder in ['src','notebooks','data/processed','results','docs','references','assets','output/pdf']:
  files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(ROOT).parts))
 # Previous monochrome previews are historical output, not needed in the new bundle.
 files=[p for p in files if 'figures_bw' not in p.parts]
 target=ROOT/'output/DAY2_분석자료.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(set(files)):z.write(p,p.relative_to(ROOT))
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  assert not any(x.startswith(('data/raw/','.venv/','tmp/','.cache/')) for x in z.namelist())
  count=len(z.namelist())
 print(f'{count} files; {target.stat().st_size:,} bytes; {target}')
if __name__=='__main__':main()
