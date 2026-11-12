# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""The GT-FENG standard report and the citation helpers."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd

from .rules import CORRECT, UNMATCHED

VERSION = "0.1.0"
SIGNATURE = f"Evaluated with GT-FENG v{VERSION} (Feng, 2026)."
BIBTEX = """@article{feng2026gtfeng,
  title  = {GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models},
  author = {Feng, Jiayu},
  year   = {2026}
}"""
TEXT = "Feng, J. (2026). GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models."


def cite(style: str = "bibtex") -> str:
    """Citation for GT-FENG: style "bibtex" (default) or "text"."""
    if style not in ("bibtex", "text"):
        raise ValueError('style must be "bibtex" or "text"')
    return BIBTEX if style == "bibtex" else TEXT


def standard_report(items: pd.DataFrame, groups: Sequence[str],
                    families: Mapping[str, Iterable[str]] | None = None) -> pd.DataFrame:
    """One row per group and reference level, with the seven quantities of the standard report:

    1. correct      items whose answer is correct
    2. U            wrong items with no match
    3. shift_X      wrong items labeled with the single shift X
    4. shift_ambiguous   wrong items whose matches carry two or more shifts
    5. source_ambiguous  wrong items whose matches come from two or more sources
    6. mean_sources      mean number of sources per matched item
    7. O_<family>   wrong items grounded in the family (every shift in M belongs to it)
    """
    families = {name: frozenset(shifts) for name, shifts in (families or {}).items()}
    keys = list(groups) + ["level"]
    rows = []
    for key_values, group in items.groupby(keys, sort=True):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        wrong = group[group.label != CORRECT]
        matched = wrong[wrong.label != UNMATCHED]
        single_shift = matched[matched.n_shifts == 1]

        row = dict(zip(keys, key_values))
        row["items"] = len(group)
        row["correct"] = int((group.label == CORRECT).sum())
        row["wrong"] = len(wrong)
        row["U"] = int((wrong.label == UNMATCHED).sum())
        for shift, count in single_shift.label.value_counts().items():
            row[f"shift_{shift}"] = int(count)
        row["shift_ambiguous"] = int((matched.n_shifts >= 2).sum())
        row["source_ambiguous"] = int((matched.n_sources >= 2).sum())
        row["mean_sources"] = round(float(matched.n_sources.mean()), 2) if len(matched) else float("nan")
        for name, family in families.items():
            # an item is grounded in a family when every shift in its M belongs to the family
            grounded = matched.shifts.map(lambda shifts, family=family: set(shifts) <= family)
            row[f"O_{name}"] = int(grounded.sum())
        rows.append(row)

    report = pd.DataFrame(rows)
    shift_columns = sorted(c for c in report.columns if c.startswith("shift_") and c != "shift_ambiguous")
    report[shift_columns] = report[shift_columns].fillna(0).astype(int)
    columns = (keys + ["items", "correct", "wrong", "U"] + shift_columns
               + ["shift_ambiguous", "source_ambiguous", "mean_sources"] + [f"O_{name}" for name in families])
    report = report[columns]
    report.attrs["generated_by"] = SIGNATURE
    return report


def to_markdown(report: pd.DataFrame) -> str:
    """A report as a Markdown table, signed with the GT-FENG version."""
    header = "| " + " | ".join(report.columns) + " |"
    divider = "|" + "---|" * len(report.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in report.itertuples(index=False)]
    return "\n".join([header, divider, *body]) + "\n\n" + SIGNATURE + "\n"
