"""Build the GT-FENG reference and answer tables for the Medicaid instance (Feng, 2026, "Right Number, Wrong State?").

usage: python build_tables.py SUPP_DIR [OUT_DIR]
SUPP_DIR is the supplementary folder of that paper (data/gold, outputs/).

Coordinates: quantity (pregnancy, infant, parent_caretaker), jurisdiction (51), time (2025 or a MACStats edition
2018-2024), definition (canon = as defined in the question, alt = the other 5-point disregard convention).
Each quantity's rules are scoped to questions about that quantity (column asked_quantity).
Levels: 1 = July 2025 values; 2 = editions 2021/2023/2024 and federal anchors; 3 = the other editions;
4 = every other number in the asked state's data-book rows (scoped to that question).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

SUPP = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
ANCHORS = (100, 133, 138)
RUNS = {"claude-sonnet-5-5": "outputs/v2/claude-sonnet-5-5.jsonl", "gpt-5.6-sol": "outputs/v1/gpt-5.6-sol.jsonl"}


def stale(path):
    g = pd.read_csv(path)
    out = {}
    for r in g.itertuples():
        d = json.loads(r.stale_values) if isinstance(r.stale_values, str) and r.stale_values else {}
        for k, v in d.items():
            out[(r.construct, r.state, int(k[:4]), "alt" if k.endswith("_alt") else "canon")] = int(v)
    return g, out


gold, full = stale(SUPP / "data/gold/gold_v1.1.csv")
_, part = stale(SUPP / "data/gold/gold_v1.csv")
rows = []
for r in gold.itertuples():
    rows.append((r.construct, r.state, 2025, "canon", int(r.gold_value), 1, r.construct, None))
    rows.append((r.construct, r.state, 2025, "alt", int(r.alternate_value), 1, r.construct, None))
for (q, s, t, d), v in full.items():
    rows.append((q, s, t, d, v, 2 if (q, s, t, d) in part else 3, q, None))
for a in ANCHORS:
    rows.append(("federal anchor", "US", 2025, "federal", a, 2, None, None))

# level 4: numbers in the asked state's data-book cells that no rule of the asked quantity already sets
long = pd.read_csv(SUPP / "data/gold/macpac_long_raw.csv")
cells = {}
for r in long.itertuples():
    for n in re.findall(r"\d+", str(r.raw_cell)):
        for x in (int(n), int(n) + 5, int(n) - 5):
            cells.setdefault(r.state, set()).add(x)
own = {}
for q, s, t, d, v, *_ in rows:
    own.setdefault((q, s), set()).add(v)
for r in gold.itertuples():
    for v in sorted(cells.get(r.state, ()) - own[(r.construct, r.state)]):
        rows.append(("data-book cell", r.state, 2025, "cell", v, 4, r.construct, r.state))
refs = pd.DataFrame(rows, columns=["quantity", "jurisdiction", "time", "definition", "value", "level",
                                   "asked_quantity", "asked_jurisdiction"])

ans = []
for model, path in RUNS.items():
    for line in open(SUPP / path):
        x = json.loads(line)
        ans.append({"model": model, "quantity": x["construct"], "jurisdiction": x["state"], "time": 2025,
                    "definition": "canon", "repeat": x["rep"], "answer": x["pred_value"]})
OUT.mkdir(parents=True, exist_ok=True)
refs.to_csv(OUT / "references.csv", index=False)
pd.DataFrame(ans).to_csv(OUT / "answers.csv", index=False)
print(f"references: {len(refs)} rules; answers: {len(ans)} responses -> {OUT}")
