# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""GT-FENG evaluation: the entry point `evaluate` and its `Result`.

The steps follow the paper:
    1. keep stable items (every repeat gives the same answer);
    2. Rule 1: is the answer correct? A correct answer is labeled "correct" at every level;
    3. for a wrong answer, Rules 2 and 4: find the documented rules that set the same value, and their shifts;
    4. for a wrong answer, Rules 3, 5 and 6: label it at each reference level.

Glossary
    item     one group (e.g. one model) x one asked index; a dict of group and coordinate values
    level    level k means the reference set R_k; a documented rule or match belongs to R_k when its level <= k
    source   documented rules with equal coordinates except time (plus scope columns) form one source
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from .references import ReferenceSet
from .report import standard_report, to_markdown
from .rules import CORRECT, coordinate_initials, count_sources, find_matches, is_correct, label
from .values import normalize

DEFAULT_COORDINATES = ("quantity", "jurisdiction", "time", "definition")


@dataclass
class Result:
    """Output of `evaluate`.

    items     one row per stable item and reference level: label, shifts, number of sources, matched indices.
    unstable  items left out because their repeats disagree or their answer is missing.
    """

    items: pd.DataFrame
    unstable: pd.DataFrame
    groups: tuple = field(default_factory=tuple)

    def report(self, families: Mapping[str, Iterable[str]] | None = None) -> pd.DataFrame:
        """The standard report; `families` maps a name to a set of shifts, e.g. {"cross": ["J", "JD"]}."""
        return standard_report(self.items, self.groups, families)

    def report_markdown(self, families: Mapping[str, Iterable[str]] | None = None) -> str:
        """The standard report as a signed Markdown table."""
        return to_markdown(self.report(families))


def evaluate(answers: pd.DataFrame, references: pd.DataFrame, *,
             coordinates: Sequence[str] = DEFAULT_COORDINATES,
             time_coordinate: str | None = "time",
             answer: str = "answer",
             value: str = "value",
             level: str | None = "level",
             level_by_shift: Mapping[str, float] | None = None,
             scope: Mapping[str, str] | None = None,
             groups: Sequence[str] = (),
             decimals: int | None = None,
             compare: Callable | None = None) -> Result:
    """Evaluate model answers against documented rules (GT-FENG Rules 1-6).

    answers          one row per response: the coordinate columns, `answer`, and any `groups` columns
                     (e.g. "model"). Repeated responses to the same question are allowed.
    references       one row per documented rule: the coordinate columns and `value`. Optional: a `level`
                     column (the first reference level that contains the rule, default 1) and `scope` columns.
    time_coordinate  the coordinate that holds time; documented rules that differ only in it share a source.
    level_by_shift   extension: the earliest level at which a shift may enter. math.inf leaves a shift out.
                     Example: {"D": 2, "T": 2, "TD": 2} holds the asked jurisdiction's own other values back
                     until level 2.
    scope            extension: {references column: answers coordinate}. A documented rule with a value in that
                     column applies only to items whose coordinate equals it.
    decimals         round numbers to this many places before comparing.
    compare          a custom function that turns a value into a comparison key (replaces values.normalize).
    """
    coordinates = list(coordinates)
    groups = tuple(groups)
    level_by_shift = dict(level_by_shift or {})
    scope = dict(scope or {})
    compare = compare or (lambda v: normalize(v, decimals))
    # Checked first: a setup error, not a data error.
    initials = coordinate_initials(coordinates)
    _check_columns(answers, coordinates + [answer], "answers")
    _check_columns(references, coordinates + [value] + list(scope), "references")

    coordinate_order = [initials[c] for c in coordinates]
    reference_set = ReferenceSet(references, coordinates=coordinates, value=value, level=level,
                                 time_coordinate=time_coordinate, scope=scope, compare=compare)
    levels = _reference_levels(reference_set, level_by_shift)

    stable, unstable = _split_by_stability(answers, list(groups) + coordinates, answer, compare)
    rows = []
    for item, raw_answer, answer_key in stable:
        asked_index = tuple(item[c] for c in coordinates)

        # Rule 1
        correct = is_correct(answer_key, asked_index, reference_set)

        # Rules 2 and 4 (each match carries its shift and its level)
        matches = [] if correct else find_matches(answer_key, item, asked_index, reference_set,
                                                  coordinates, initials, level_by_shift)

        # Rule 6: label the answer once per reference level
        for current_level in levels:
            if correct:
                answer_label, shifts, in_level = CORRECT, (), []
            else:
                in_level = [m for m in matches if m.level <= current_level]
                # Rules 3 and 5
                answer_label, shifts = label(in_level, coordinate_order)
            rows.append({**item,
                         "answer": raw_answer,
                         "level": _tidy_level(current_level),
                         "label": answer_label,
                         "shifts": shifts,
                         "n_shifts": len(shifts),
                         "n_sources": count_sources(in_level),
                         "matches": tuple(m.rule.index for m in in_level)})
    return Result(items=pd.DataFrame(rows), unstable=pd.DataFrame(unstable), groups=groups)


def _check_columns(table: pd.DataFrame, required: Sequence[str], name: str) -> None:
    for column in required:
        if column not in table.columns:
            raise KeyError(f"{name} lacks column {column!r}")


def _split_by_stability(answers: pd.DataFrame, item_columns: list, answer: str, compare: Callable):
    """An item is stable when every repeat gives the same, non-missing answer.

    Returns (stable, unstable): stable is a list of (item, first raw answer, comparison key).
    """
    stable, unstable = [], []
    for key_values, responses in answers.groupby(item_columns, sort=False, dropna=False):
        item = dict(zip(item_columns, key_values))
        keys = [compare(v) for v in responses[answer]]
        if None in keys or len(set(keys)) != 1:
            unstable.append({**item, "answers": list(responses[answer])})
        else:
            stable.append((item, responses[answer].iloc[0], keys[0]))
    return stable, unstable


def _reference_levels(reference_set: ReferenceSet, level_by_shift: Mapping[str, float]) -> list[float]:
    """Every level at which some documented rule or shift enters: the k of R1, R2, ..."""
    levels = set(reference_set.levels()) | {v for v in level_by_shift.values() if v != math.inf}
    return sorted(levels) or [1.0]


def _tidy_level(level: float):
    """Show level 2.0 as 2 in the output."""
    return int(level) if float(level).is_integer() else level
