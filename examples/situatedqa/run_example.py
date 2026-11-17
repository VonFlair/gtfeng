# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""Reproduce the GT-FENG check on the geographic SituatedQA test set (paper Section 7, Table 6 of arXiv:2611.18279v2).

usage: python run_example.py   (run build_tables.py first)

Coordinates: the question (quantity) and the location (jurisdiction). Time and definition are fixed, so every
match has shift J. Several gold strings for one question and location are all accepted (Rule 1).
"""
import re
import string
import unicodedata
from pathlib import Path

import pandas as pd

import gtfeng

HERE = Path(__file__).parent


def normalize(value):
    """SituatedQA's evaluation normalization, after removing text in parentheses; None for an empty answer."""
    if not isinstance(value, str):
        return None
    s = re.sub(r"\([^()]*\)", " ", value)
    s = unicodedata.normalize("NFD", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
    return s or None


refs = pd.read_csv(HERE / "references.csv", dtype=str)
answers = pd.read_csv(HERE / "answers.csv", dtype={"quantity": str})
# locations are compared after the same normalization ("USA " and "USA" are one location)
for table in (refs, answers):
    table["jurisdiction"] = table["jurisdiction"].map(normalize)
    table["time"], table["definition"] = "2021-02", "as asked"

# scope: a gold answer is compared only with questions about the same question id.
# asked_as keeps rows that differ only in how the location is written (e.g. a trailing space) as separate items.
result = gtfeng.evaluate(answers, refs, groups=["model", "asked_as"], compare=normalize,
                         scope={"asked_quantity": "quantity"})
items = result.items
rows = []
for model, it in items.groupby("model"):
    wrong = it[it.label != gtfeng.CORRECT]
    matched = wrong[wrong.label != gtfeng.UNMATCHED]
    rows.append({"model": model, "stable": len(it), "correct": len(it) - len(wrong), "wrong": len(wrong),
                 "U": int((wrong.label == gtfeng.UNMATCHED).sum()), "shift_J": len(matched),
                 "source_ambiguous": int((matched.n_sources >= 2).sum())})
print(pd.DataFrame(rows).to_string(index=False))
