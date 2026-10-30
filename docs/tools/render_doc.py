#!/usr/bin/env python3
"""Fill ${dotted.keys} in an HTML template and print it with headless Chrome."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chrome import print_pdf, pdf_pages
TOKEN=re.compile(r"\$\{([A-Za-z_][\w.-]*)\}")
BAD=('TBD','XX','TODO')
def load(path):
    return json.loads(Path(path).read_text())
def get(root, key):
    cur=root
    for part in key.split('.'):
        if isinstance(cur, dict) and part in cur: cur=cur[part]
        else: raise KeyError(key)
    if isinstance(cur,(dict,list)): return json.dumps(cur, ensure_ascii=False)
    if cur is None: return 'unavailable'
    return str(cur)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('template'); ap.add_argument('output'); ap.add_argument('--data',action='append',required=True); ap.add_argument('--max-pages',type=int,default=5)
    a=ap.parse_args(); data={}
    for spec in a.data:
        p=Path(spec); obj=load(p); stem=p.stem
        data[stem]=obj
        if isinstance(obj,dict): data.update(obj)
    html=Path(a.template).read_text(encoding='utf-8')
    missing=[]
    def sub(m):
        try:
            return get(data,m.group(1))
        except KeyError:
            missing.append(m.group(1)); return m.group(0)
    html=TOKEN.sub(sub,html)
    if missing: raise SystemExit('unresolved placeholders: '+', '.join(sorted(set(missing))))
    if '${' in html: raise SystemExit('unresolved placeholder syntax remains')
    for bad in BAD:
        if bad in html: raise SystemExit(f'forbidden marker {bad}')
    tmp=Path(a.output).with_suffix('.filled.html'); tmp.write_text(html,encoding='utf-8')
    try:
        print_pdf(tmp,Path(a.output),page_css='@page { size: A4; margin: 12mm; }')
    finally: tmp.unlink(missing_ok=True)
    pages=pdf_pages(Path(a.output))
    if pages is not None and pages>a.max_pages: raise SystemExit(f'page count {pages} exceeds {a.max_pages}')
    print(f'wrote {a.output}'+(f' ({pages} pages)' if pages else ''))
if __name__=='__main__': main()
