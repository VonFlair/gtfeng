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

Confidence interval: the stable items of each group are resampled with replacement, the FENG value is recomputed on
every resample, and the interval is the percentile interval of the resampled values. Resample d uses the same items
at every reference level, so `feng_bootstrap` also gives intervals for differences between levels.
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


def _item_scores(result, family: Iterable[str], increments: Sequence[float]):
    """Per group and level, in the order of result.items: for every stable item, (w, o, hits).

    w is 1 for a wrong item, o is 1 for a wrong item grounded in F, and hits marks which of the item's nearby
    values are grounded in F (empty for a correct item).
    """
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
            if key in gold:   # an increment that reaches the correct value is dropped for this item
                continue
            values.append(find_matches(key, item, asked_index, references, coordinates,
                                       context["initials"], context["level_by_shift"]))
        nearby[tuple(item.values())] = values

    keys = list(result.groups) + ["level"]
    scores = {}
    for key_values, group in result.items.groupby(keys, sort=True):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        level = key_values[-1]
        w, o, hits = [], [], []
        for row in group[item_columns + ["label", "shifts"]].itertuples(index=False, name=None):
            item, item_label, shifts = row[:-2], row[-2], row[-1]
            if item_label == CORRECT:
                w.append(0), o.append(0), hits.append(np.zeros(0))
                continue
            w.append(1)
            o.append(int(_grounded(shifts, family)))
            hits.append(np.array([_grounded({m.shift for m in ms if m.level <= level}, family)
                                  for ms in nearby[tuple(item)]], dtype=float))
        scores[key_values] = (np.array(w), np.array(o), hits)
    return keys, scores


def _feng(w, o, n):
    """FENG value from totals; NaN where W <= N."""
    w, o, n = (np.asarray(x, dtype=float) for x in (w, o, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w > n, (o - n) / (w - n), np.nan)


def feng_bootstrap(result, family: Iterable[str], *, increments: Sequence[float] = DEFAULT_INCREMENTS,
                   resamples: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap resamples of the FENG value: one row per group, level, and resample.

    The stable items of each group are resampled with replacement; resample d uses the same items at every level.
    """
    keys, scores = _item_scores(result, family, increments)
    rng = np.random.default_rng(seed)
    indices = {}
    frames = []
    for key_values, (w, o, hits) in scores.items():
        n = np.array([h.mean() if len(h) else 0.0 for h in hits])
        group = key_values[:-1]
        if group not in indices:
            indices[group] = rng.integers(len(w), size=(resamples, len(w)))
        idx = indices[group]
        frame = pd.DataFrame({k: [v] * resamples for k, v in zip(keys, key_values)})
        frame["resample"] = np.arange(resamples)
        frame["feng_value"] = _feng(w[idx].sum(1), o[idx].sum(1), n[idx].sum(1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def feng_values(result, family: Iterable[str], *, increments: Sequence[float] = DEFAULT_INCREMENTS,
                permutations: int = 10000, seed: int = 0, bootstrap: int = 0,
                ci: float = 0.95) -> pd.DataFrame:
    """Rule 7 and the FENG value, one row per group and reference level. Numeric answers only.

    bootstrap  number of bootstrap resamples. When positive, adds ci_low and ci_high, the percentile interval at
               level `ci` (see feng_bootstrap, run with the same seed).
    """
    keys, scores = _item_scores(result, family, increments)
    rng = np.random.default_rng(seed)
    rows = []
    for key_values, (w, o, hits) in scores.items():
        wrong = [h for h, wi in zip(hits, w) if wi]
        observed = int(o.sum())
        null = float(sum(h.mean() for h in wrong if len(h)))
        draws = np.zeros(permutations)
        for h in wrong:
            if len(h):
                draws += h[rng.integers(len(h), size=permutations)]
        n_wrong = int(w.sum())
        rows.append({**dict(zip(keys, key_values)),
                     "W": n_wrong, "O": observed, "N": round(null, 2), "E": round(observed - null, 2),
                     "feng_value": round((observed - null) / (n_wrong - null), 3) if n_wrong > null else float("nan"),
                     "p": float((draws >= observed).mean())})
    out = pd.DataFrame(rows)
    if bootstrap > 0:
        boot = feng_bootstrap(result, family, increments=increments, resamples=bootstrap, seed=seed)
        alpha = (1 - ci) / 2
        bounds = boot.groupby(keys).feng_value.quantile([alpha, 1 - alpha]).unstack()
        bounds.columns = ["ci_low", "ci_high"]
        out = out.merge(bounds.round(3).reset_index(), on=keys, how="left")
    return out
