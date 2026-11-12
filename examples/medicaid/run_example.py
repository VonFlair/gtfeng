"""Reproduce the GT-FENG standard report for the Medicaid instance (paper Section 5, Table 3).

usage: python run_example.py   (run build_tables.py first)
"""
import math
from pathlib import Path

import pandas as pd

import gtfeng

HERE = Path(__file__).parent
refs = pd.read_csv(HERE / "references.csv")
answers = pd.read_csv(HERE / "answers.csv")

result = gtfeng.evaluate(
    answers, refs, groups=["model"],
    # level 1 is the naive check: other states' July 2025 values. The asked state's own other values enter at
    # level 2, and other states' earlier values are never used.
    level_by_shift={"D": 2, "T": 2, "TD": 2, "JT": math.inf, "JTD": math.inf},
    scope={"asked_quantity": "quantity", "asked_jurisdiction": "jurisdiction"},
)
report = result.report(families={"cross_jurisdiction": ["J", "JD"]})
pd.set_option("display.width", 200)
print(report[["model", "level", "wrong", "U", "shift_ambiguous", "source_ambiguous", "mean_sources",
              "O_cross_jurisdiction"]].to_string(index=False))
