"""Generate and compile the plain presentation assignment from its Markdown."""
import importlib.util,re,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('report_tex',ROOT/'scripts/render_research_report.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.MD=m.OUT/'PHAN_CONG_THUYET_TRINH.md';m.TEX=m.OUT/'PHAN_CONG_THUYET_TRINH.tex'
base=m.inline

def inline(text):
    pieces=re.split(r'(\*\*.*?\*\*)',text)
    return ''.join(r'\textbf{'+base(x[2:-2])+'}' if x.startswith('**') and x.endswith('**') else base(x) for x in pieces)
m.inline=inline
base_widths=m.widths
m.widths=lambda headers,rows: [.12,.33,.43,.12] if headers[0]=='Người' and len(headers)==4 else base_widths(headers,rows)
tex=m.generate();s=tex.read_text()
s=re.sub(r'\\hypersetup\{pdftitle=.*?pdfauthor=\{\}\}',r'\\hypersetup{pdftitle={Phân công thuyết trình: Hoàng, Bảo, Nguyên},pdfauthor={}}',s)
s=s.replace(r'\date{31/10/2026}',r'\date{03/11/2026}')
s=s.replace('% Generated from BAO_CAO_TONG_HOP.md. Edit Markdown, then regenerate.','% Generated from PHAN_CONG_THUYET_TRINH.md.')
s=s.replace('BAO_CAO_TONG_HOP_PHOBERT_VIHSD.tex','PHAN_CONG_THUYET_TRINH.tex')
# This short assignment needs no separate title or contents page.
s=re.sub(r'\\begingroup\n\\small\n\\makeatletter.*?\\clearpage\n','',s,count=1,flags=re.S)
s=s.replace(r'\linespread{1.08}',r'\linespread{1.02}').replace(r'\setlength{\parskip}{3pt}',r'\setlength{\parskip}{1pt}')
s=s.replace(r'\begin{document}',r'\usepackage{enumitem}'+'\n'+r'\setlist[itemize]{itemsep=2pt,parsep=0pt,topsep=4pt,leftmargin=*}'+'\n'+r'\begin{document}'+'\n'+r'\raggedright')
tex.write_text(s)
engine=shutil.which('tectonic') or str(Path.home()/'.cache/tmg-report-tools/tectonic-0.17.0/tectonic')
subprocess.run([engine,'--keep-logs','--outdir',str(m.PDF_DIR),tex.name],cwd=m.OUT,check=True)
print('SOURCE',tex);print('PDF',m.PDF_DIR/(tex.stem+'.pdf'))
