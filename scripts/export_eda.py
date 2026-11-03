"""Export an offline, self-contained HTML report from verified EDA artifacts."""
from pathlib import Path
import base64
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
report = ROOT / 'reports/eda/EDA_REPORT.md'
text = report.read_text(encoding='utf-8')
text += '\n\n' + (ROOT / 'reports/DATA_REVIEW.md').read_text(encoding='utf-8')
if (ROOT / 'reports/eda/UNK_NOTE.md').exists():
    text += '\n\n' + (ROOT / 'reports/eda/UNK_NOTE.md').read_text(encoding='utf-8')
renderer = MarkdownIt('commonmark').enable('table')
body = renderer.render(text)
for picture in (ROOT / 'reports/eda').glob('*.png'):
    data = base64.b64encode(picture.read_bytes()).decode()
    body = body.replace('src="' + picture.name + '"', 'src="data:image/png;base64,' + data + '"')
css = '''
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#edf1f5;color:#243546;font:16px/1.7 system-ui,-apple-system,sans-serif}main{max-width:1140px;margin:36px auto;background:white;padding:40px 48px;border-top:6px solid #3979a8;box-shadow:0 4px 24px #23394d12}h1{font-size:30px;line-height:1.3;margin:0 0 28px}h2{font-size:23px;margin:36px 0 16px;border-bottom:1px solid #dce3ea;padding-bottom:10px}h3{font-size:19px}p,li{max-width:100%}img{display:block;width:100%;height:auto;margin:28px auto;border:1px solid #e4e9ee}table{border-collapse:collapse;width:100%;font-size:15px;margin:20px 0}th{background:#edf3f8;text-align:left}th,td{padding:11px 13px;border-bottom:1px solid #dce3ea}code{background:#edf2f6;border-radius:3px;padding:2px 5px;overflow-wrap:anywhere}a{color:#1c659b}footer{margin-top:40px;color:#536476;font-size:14px}@media(max-width:700px){main{margin:0;padding:24px 18px}h1{font-size:25px}table{font-size:12px}th,td{padding:6px}}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0;max-width:none}img,table{break-inside:avoid}h2{break-after:avoid}}
'''
page = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>EDA ViHSD · PhoBERT baseline</title><style>{css}</style></head><body><main>{body}<footer>Báo cáo offline tự chứa ảnh. Số liệu được tạo từ pipeline TMG; raw data và source hashes được lưu riêng trong manifest.</footer></main></body></html>'''
(ROOT / 'reports/EDA.html').write_text(page,encoding='utf-8')
print(ROOT / 'reports/EDA.html')
