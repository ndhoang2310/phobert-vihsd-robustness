"""Read the canonical Markdown, emit plain LaTeX, then compile with a TeX engine.

Usage: .venv/bin/python scripts/render_research_report.py [--tex-only]
The report is edited in reports/final_report/BAO_CAO_TONG_HOP.md, not report_blocks.json.
No HTML or ReportLab rendering is involved.
"""
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/final_report'
MD=OUT/'BAO_CAO_TONG_HOP.md'
TEX=OUT/'BAO_CAO_TONG_HOP_PHOBERT_VIHSD.tex'
PDF_DIR=ROOT/'output/pdf'


def escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','±':r'\ensuremath{\pm}','×':r'\ensuremath{\times}','ε':r'\ensuremath{\varepsilon}','Δ':r'\ensuremath{\Delta}','→':r'\ensuremath{\rightarrow}','≥':r'\ensuremath{\geq}','≤':r'\ensuremath{\leq}'}
    return ''.join(replacements.get(c,c) for c in text)


def inline(text):
    # Preserve math, links, paths and full hashes as explicit TeX constructs.
    pattern=r'(\$[^$\n]+\$|\[[^\]]+\]\([^)]+\)|`[^`]+`|\b[a-f0-9]{40,64}\b|(?:reports|scripts|src|\.venv)/[^\s,;)]*)'
    result=[];previous=0
    for m in re.finditer(pattern,text):
        result.append(escape(text[previous:m.start()]));value=m.group()
        if value.startswith('$'):result.append(value)
        elif value.startswith('['):
            link=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',value);label,url=link.groups()
            if label==url:result.append(r'\url{'+url+'}')
            else:result.append(r'\href{'+url+'}{'+escape(label)+'}')
        elif value.startswith('`'):
            code=value[1:-1]
            result.append(r'\texttt{'+escape(code)+'}')
        elif re.fullmatch('[a-f0-9]{40,64}',value):result.append(r'{\ttfamily\scriptsize\seqsplit{'+value+'}}')
        else:result.append(r'\nolinkurl{'+value+'}')
        previous=m.end()
    result.append(escape(text[previous:]));return ''.join(result)


def widths(headers,rows):
    n=len(headers)
    if n==2:
        if headers[0]=='Hồ sơ':return [.54,.46]
        return [.31,.69]
    if n==3:return [.23,.39,.38]
    if n==4:return [.31,.23,.23,.23]
    if n==5:return [.28,.18,.18,.18,.18]
    if headers[0]=='Phương pháp' and 'Profile' in headers:return [.23,.21,.14,.14,.14,.14]
    if headers[0]=='Sample ID':return [.20,.24,.14,.14,.14,.14]
    if headers[0]=='Mức yêu cầu':return [.14,.15,.15,.186,.186,.188]
    return [1/n]*n


def make_table(lines):
    def cells(line):return [x.strip() for x in line.strip().strip('|').split('|')]
    headers=cells(lines[0]);rows=[cells(x) for x in lines[2:]];n=len(headers)
    if any(len(r)!=n for r in rows):raise ValueError('Nonrectangular Markdown table')
    w=widths(headers,rows)
    spec='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+f'{x:.5f}'+r'\reporttablewidth}' for x in w)+'@{}'
    head=' & '.join(r'\textbf{'+inline(t)+'}' for t in headers)+r' \\'
    body=[r'\begingroup',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\setlength{\reporttablewidth}{\dimexpr\linewidth-'+str(2*(n-1))+r'\tabcolsep\relax}',r'\renewcommand{\arraystretch}{1.15}',r'\begin{longtable}{'+spec+'}',r'\toprule',head,r'\midrule',r'\endfirsthead',r'\toprule',head,r'\midrule',r'\endhead',r'\bottomrule',r'\endlastfoot']
    body+=[' & '.join(inline(t) for t in row)+r' \\' for row in rows]
    body +=[r'\end{longtable}',r'\endgroup']
    return '\n'.join(body)


def generate():
    lines=MD.read_text().splitlines();title=lines[0].removeprefix('# ').strip();parts=[];appendix=False;i=1
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('## '):
            heading=line[3:]
            if heading.startswith('Phụ lục '):
                if not appendix:parts.append(r'\appendix');appendix=True
                heading=re.sub(r'^Phụ lục [A-Z]\.\s*','',heading)
                parts.append(r'\FloatBarrier\section{'+inline(heading)+'}')
            elif re.match(r'^\d+\. ',heading):parts.append(r'\FloatBarrier\section{'+inline(re.sub(r'^\d+\.\s*','',heading))+'}')
            else:parts +=[r'\section*{'+inline(heading)+'}',r'\addcontentsline{toc}{section}{'+inline(heading)+'}']
            i+=1;continue
        if line.startswith('### '):parts.append(r'\Needspace{5\baselineskip}\subsection*{'+inline(line[4:])+'}');i+=1;continue
        if line=='$$':
            equation=[];i+=1
            while i<len(lines) and lines[i].strip()!='$$':equation.append(lines[i]);i+=1
            if i==len(lines):raise ValueError('Unclosed display math')
            parts.append('\\[\n'+'\n'.join(equation)+'\n\\]');i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            if i==len(lines):raise ValueError('Unclosed code fence')
            parts.extend([r'\begin{Verbatim}[fontsize=\footnotesize]','\n'.join(code),r'\end{Verbatim}']);i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):table.append(lines[i]);i+=1
            parts.append(make_table(table));continue
        if line.startswith('- '):
            items=[]
            while i<len(lines) and lines[i].strip().startswith('- '):items.append(inline(lines[i].strip()[2:]));i+=1
            parts.extend([r'\begin{itemize}']+[r'\item '+x for x in items]+[r'\end{itemize}']);continue
        figure=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if figure:
            caption,path=figure.groups();assert (OUT/path).is_file()
            caption=re.sub(r'^Hình \d+\.\s*','',caption)
            parts.extend([r'\begin{figure}[htbp]',r'\centering',r'\includegraphics[width=\linewidth,height=.34\textheight,keepaspectratio]{'+path+'}',r'\caption{'+inline(caption)+'}',r'\end{figure}']);i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||- |!\[|\$\$|```)',lines[i].strip()):paragraph.append(lines[i].strip());i+=1
        parts.append(inline(' '.join(paragraph))+'\n')
    preamble=r'''% Generated from BAO_CAO_TONG_HOP.md. Edit Markdown, then regenerate.
% Build from this directory: tectonic --keep-logs BAO_CAO_TONG_HOP_PHOBERT_VIHSD.tex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=24mm]{geometry}
\usepackage{fontspec}
\setmainfont{Noto Serif}
\setmonofont{DejaVu Sans Mono}[Scale=MatchLowercase]
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,longtable,array}
\usepackage{xurl,seqsplit}
\usepackage{needspace,fancyvrb}
\usepackage[section]{placeins}
\usepackage[hidelinks,unicode]{hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\newlength{\reporttablewidth}
\renewcommand{\contentsname}{Mục lục}
\renewcommand{\figurename}{Hình}
\renewcommand{\tablename}{Bảng}
\setlength{\parindent}{1.2em}
\setlength{\parskip}{3pt}
\setlength{\emergencystretch}{3em}
\linespread{1.08}
\raggedbottom
\pagestyle{plain}
\hypersetup{pdftitle={Đánh giá và tăng cường độ bền vững của PhoBERT trên ViHSD},pdfauthor={}}
\title{TITLE}
\author{}
\date{31/10/2026}
\begin{document}
\maketitle
\begingroup
\small
\makeatletter
\renewcommand*\l@section{\@dottedtocline{1}{0em}{2em}}
\makeatother
\tableofcontents
\endgroup
\clearpage
'''.replace('TITLE',escape(title))
    TEX.write_text(preamble+'\n\n'.join(parts)+'\n\\end{document}\n')
    return TEX


def compile_tex(tex):
    PDF_DIR.mkdir(parents=True,exist_ok=True)
    engine=shutil.which('tectonic')
    cached=Path.home()/'.cache/tmg-report-tools/tectonic-0.17.0/tectonic'
    if not engine and cached.is_file():engine=str(cached)
    if engine:command=[engine,'--keep-logs','--keep-intermediates','--outdir',str(PDF_DIR),tex.name]
    else:
        engine=shutil.which('xelatex')
        if not engine:raise RuntimeError('Install Tectonic or XeLaTeX; no fallback to non-TeX PDF generation.')
        command=[engine,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(PDF_DIR),tex.name]
    for _ in range(1 if 'tectonic' in Path(engine).name else 3):
        subprocess.run(command,cwd=OUT,check=True)
    pdf=PDF_DIR/(tex.stem+'.pdf');assert pdf.is_file()
    build={'source_markdown':str(MD.relative_to(ROOT)),'source_tex':str(tex.relative_to(ROOT)),'pdf':str(pdf.relative_to(ROOT)),'engine':Path(engine).name,'command':command,'markdown_sha256':hashlib.sha256(MD.read_bytes()).hexdigest(),'tex_sha256':hashlib.sha256(tex.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (OUT/'LATEX_BUILD.json').write_text(json.dumps(build,ensure_ascii=False,indent=2)+'\n')
    return pdf


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tex-only',action='store_true');args=p.parse_args()
    tex=generate();print('LaTeX source:',tex,flush=True)
    if not args.tex_only:print('Compiled PDF:',compile_tex(tex),flush=True)
if __name__=='__main__':main()
