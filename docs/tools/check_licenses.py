#!/usr/bin/env python3
"""Report installed GPU-image packages and vendored viewer files against the licence manifest.

The SSH command is read-only. It is deliberately opt-in because it needs tyrone access.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ssh', action='store_true', help='run the prescribed read-only pip scan over SSH')
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    manifest=json.loads((ROOT/'docs/licenses.yaml').read_text())
    names={n.strip().lower() for group in ('product','system','base_image') for x in manifest.get(group,[]) for n in x['name'].split(',')}
    def norm(s):
        return s.lower().replace('_','-').split('==')[0]
    vendored=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'viewer/vendor').rglob('*') if p.is_file())
    notices=[f for f in vendored if '/LICENSES/' in f]
    report={'manifest_names':sorted(names),'vendored_files':vendored,'remote_packages':None,'unverified':[], 'diff': {'vendored_without_notice':[f for f in vendored if '/LICENSES/' not in f]}, 'license_notice_files': notices}
    if args.ssh:
        cmd="ssh -o BatchMode=yes sher@10.1.18.190 'docker run --rm namma-space-gpu:0.2 pip list --not-required --format=json'"
        p=subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if p.returncode:
            print(p.stderr.strip(), file=sys.stderr); return p.returncode
        # NVIDIA's base image prints a licence banner before pip's JSON.
        raw=p.stdout; start=raw.find('[')
        if start < 0: raise SystemExit('pip scan returned no JSON array')
        report['remote_packages']=json.loads(raw[start:])
        installed={norm(p.get('name','')) for p in report['remote_packages']}
        manifest_norm={norm(x) for x in names}
        report['diff'].update({'installed_not_in_manifest':sorted(installed-manifest_norm), 'manifest_not_seen_in_pip':sorted(manifest_norm-installed)})
    else:
        report['unverified'].append('remote GPU image scan not run; use --ssh')
    if args.json: print(json.dumps(report, indent=2))
    else:
        print('Vendored viewer files:')
        for f in vendored: print('  '+f)
        print('Vendored files without an adjacent notice (review against LICENSES.md):', ', '.join(report['diff']['vendored_without_notice']) or 'none')
        if report['remote_packages'] is None: print('GPU image: unverified (run with --ssh)')
        else:
            print('GPU image top-level packages:')
            for p in report['remote_packages']: print(f"  {p.get('name')}=={p.get('version')}")
            print('Diff:')
            print('  installed but absent from manifest:', ', '.join(report['diff']['installed_not_in_manifest']) or 'none')
            print('  manifest components not seen as top-level packages:', ', '.join(report['diff']['manifest_not_seen_in_pip']) or 'none')
        print('\nManifest components:', len(names))
    return 0
if __name__=='__main__': raise SystemExit(main())
