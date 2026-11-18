"""Reproduce the GT-FENG unemployment insurance instance (paper Section 7, Tables 6 and 16).

usage: python run_example.py   (references.csv and answers.csv are included)
Weekly benefit amounts are in dollars and the taxable wage base in hundreds of dollars.
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
    # as in the Medicaid example: the asked state's own other definition or edition counts from level 2,
    # another state's earlier edition never counts
    level_by_shift={"D": 2, "T": 2, "TD": 2, "JT": math.inf, "JTD": math.inf},
    scope={"asked_quantity": "quantity", "asked_jurisdiction": "jurisdiction"},
)
pd.set_option("display.width", 200)
print(result.report(families={"cross_jurisdiction": ["J", "JD"]})[
    ["model", "level", "correct", "wrong", "U", "source_ambiguous", "O_cross_jurisdiction"]].to_string(index=False))
print(result.feng_value(["J", "JD"], bootstrap=10000, cluster="jurisdiction").to_string(index=False))
