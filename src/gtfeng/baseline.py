# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""Rule 7 (local perturbation baseline) and the FENG value.

For a family F (a set of shifts, e.g. {"J", "JD"} for another jurisdiction):

    W  wrong items at a reference level
    O  wrong items grounded in F: they have matches, and every shift in M belongs to F
    N  the number of grounded items that nearby values produce. Each wrong answer A is replaced by A + i for every
       increment i in I; increments that reach the correct value are dropped for that item. N is the sum over wrong
       items of the share of nearby values that are grounded in F.
    E  = O - N, the excess over the baseline
    p  the share of random perturbations (one increment per item, drawn independently) whose grounded count
       reaches O

    FENG value = (O - N) / (W - N)

The FENG value is 0 when wrong answers are grounded in F as often as nearby values are, 1 when every wrong answer
is grounded in F, and negative when they are grounded less often than nearby values. It has the chance-corrected
form of Cohen's kappa; the match event and the local baseline come from GT-FENG.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .rules import CORRECT, find_matches

# Increments must leave out any known definition shift. The Medicaid instance leaves out 5 (the 5% disregard).
DEFAULT_INCREMENTS = (-4, -3, -2, -1, 1, 2, 3, 4)


def _grounded(shifts: Iterable[str], family: frozenset) -> bool:
    shifts = set(shifts)
    return bool(shifts) and shifts <= family


def feng_values(result, family: Iterable[str], *, increments: Sequence[float] = DEFAULT_INCREMENTS,
                permutations: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Rule 7 and the FENG value, one row per group and reference level. Numeric answers only."""
    context = result._context
    if context is None:
        raise ValueError("this Result was not produced by gtfeng.evaluate")
    family = frozenset(family)
    increments = [i for i in increments if i != 0]
    if not increments:
        raise ValueError("increments must contain at least one nonzero value")
    references = context["references"]
    coordinates = context["coordinates"]
    item_columns = list(result.groups) + coordinates
    rng = np.random.default_rng(seed)

    # Nearby values of each wrong item, with their matches (computed once, then filtered by level).
    nearby = {}
    wrong_rows = result.items[result.items.label != CORRECT]
    for row in wrong_rows.drop_duplicates(item_columns).to_dict("records"):
        answer = row["answer"]
        if isinstance(answer, bool) or not isinstance(answer, (int, float, np.integer, np.floating)):
            raise ValueError(f"Rule 7 needs numeric answers; got {answer!r} for {row}")
        item = {c: row[c] for c in item_columns}
        asked_index = tuple(row[c] for c in coordinates)
        gold = references.gold(asked_index)
        values = []
        for i in increments:
            key = context["compare"](float(answer) + i)
            if key == gold:   # an increment that reaches the correct value is dropped for this item
                continue
            values.append(find_matches(key, item, asked_index, references, coordinates,
                                       context["initials"], context["level_by_shift"]))
        nearby[tuple(item.values())] = values

    rows = []
    keys = list(result.groups) + ["level"]
    for key_values, group in result.items.groupby(keys, sort=True):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        level = key_values[-1]
        wrong = group[group.label != CORRECT]
        observed = int(sum(_grounded(s, family) for s in wrong.shifts))
        hits = []
        for row in wrong[item_columns].itertuples(index=False, name=None):
            hits.append(np.array([_grounded({m.shift for m in ms if m.level <= level}, family)
                                  for ms in nearby[tuple(row)]], dtype=float))
        null = float(sum(h.mean() for h in hits if len(h)))
        draws = np.zeros(permutations)
        for h in hits:
            if len(h):
                draws += h[rng.integers(len(h), size=permutations)]
        n_wrong = len(wrong)
        rows.append({**dict(zip(keys, key_values)),
                     "W": n_wrong, "O": observed, "N": round(null, 2), "E": round(observed - null, 2),
                     "feng_value": round((observed - null) / (n_wrong - null), 3) if n_wrong > null else float("nan"),
                     "p": float((draws >= observed).mean())})
    return pd.DataFrame(rows)
