#!/usr/bin/env python3
"""Generate LICENSES.md from docs/licenses.yaml (JSON is valid YAML)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
data = json.loads((ROOT / "docs/licenses.yaml").read_text())
out = ["# Namma Space licences and credits", "", "Generated from `docs/licenses.yaml`. Verdicts describe this submission's use.", ""]
for title, key in (("Product path", "product"), ("Evaluation data and credits", "evaluation"), ("System libraries", "system"), ("Installed in the GPU image but not used by our code", "base_image")):
    out += [f"## {title}", "", "| Component | Version | Licence | Role | Verdict | Source |", "|---|---|---|---|---|---|"]
    for x in data[key]:
        out.append(f"| {x['name']} | {x.get('version','')} | {x['license']} | {x.get('role','')} | {x['verdict']} | [{x['source']}]({x['source']}) |")
    out.append("")
out += ["## Rejected or dropped", "", "| Item | Reason |", "|---|---|"]
for x in data["rejected"]: out.append(f"| {x['name']} | {x['reason']} |")
out += ["", "## Disclosures", ""] + [f"- {x}" for x in data["disclosures"]] + ["", "## Repository licence", "", "The repository code is Apache-2.0; see `LICENSE`.", ""]
(ROOT / "LICENSES.md").write_text("\n".join(out), encoding="utf-8")
print(f"wrote {ROOT/'LICENSES.md'}")
