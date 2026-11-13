"""Reproduce the GT-FENG standard report for the Medicaid instance (paper Section 6, Table 3).

usage: python run_example.py   (the tables are included; build_tables.py rebuilds them)
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
    # When each kind of match starts to count:
    #   J, JD      another state's 2025 limit: from level 1 (the default)
    #   D, T, TD   the asked state's own other definition or year: from level 2
    #   JT, JTD    another state's earlier year: never (math.inf means "at no level")
    level_by_shift={"D": 2, "T": 2, "TD": 2, "JT": math.inf, "JTD": math.inf},
    # A documented rule is compared only with questions about the same quantity; the extra data-book numbers
    # added at level 4 are compared only with questions about their own state.
    scope={"asked_quantity": "quantity", "asked_jurisdiction": "jurisdiction"},
)
report = result.report(families={"cross_jurisdiction": ["J", "JD"]})
pd.set_option("display.width", 200)
print(report[["model", "level", "wrong", "U", "shift_ambiguous", "source_ambiguous", "mean_sources",
              "O_cross_jurisdiction"]].to_string(index=False))
print()
print(result.feng_value(["J", "JD"]).to_string(index=False))
