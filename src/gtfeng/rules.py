# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""GT-FENG Rules 1-6, one function per rule, in the order of the paper.

Rule 1  Correct      the answer equals the gold value at the asked index.
Rule 2  Matches      otherwise, M = every other documented rule whose value equals the answer.
Rule 3  Unmatched    if M is empty, the answer is labeled U.
Rule 4  Shift        the coordinates in which a match differs from the asked index, as initials (J, T, JD, ...).
Rule 5  Label        one shift -> that shift; two or more shifts -> "ambiguous {...}".
Rule 6  Levels       nested reference sets R1 ⊆ R2 ⊆ ...; a match belongs to R_k when its level is at most k.

Two package extensions, not in the paper's rules: `scope` limits a documented rule to some items, and
`level_by_shift` sets the earliest level at which a shift may enter (a short way to define R1, R2, ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .references import DocumentedRule, ReferenceSet

CORRECT = "correct"
UNMATCHED = "U"
# The label "ambiguous" marks shift ambiguity (Rule 5). Source ambiguity is reported separately, as n_sources >= 2.
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Match:
    """A documented rule that sets the same value as a wrong answer."""

    rule: DocumentedRule
    shift: str      # Rule 4
    level: float    # Rule 6: the first reference level that contains this match


def coordinate_initials(coordinates: Sequence[str]) -> dict[str, str]:
    """Each coordinate is named by its initial in shifts: quantity -> Q, jurisdiction -> J, and so on."""
    initials = {c: c.strip()[0].upper() for c in coordinates}
    if len(set(initials.values())) != len(initials):
        raise ValueError(f"coordinate initials must be unique, got {initials}; rename a coordinate")
    return initials


def is_correct(answer_key, asked_index: tuple, references: ReferenceSet) -> bool:
    """Rule 1."""
    return answer_key == references.gold(asked_index)


def shift(rule_index: tuple, asked_index: tuple, coordinates: Sequence[str], initials: Mapping[str, str]) -> str:
    """Rule 4: the initials of the coordinates in which a documented rule differs from the asked index."""
    return "".join(initials[c] for c, theirs, asked in zip(coordinates, rule_index, asked_index) if theirs != asked)


def find_matches(answer_key, item: Mapping, asked_index: tuple, references: ReferenceSet,
                 coordinates: Sequence[str], initials: Mapping[str, str],
                 level_by_shift: Mapping[str, float]) -> list[Match]:
    """Rule 2: every other documented rule that sets the answer's value, with its shift (Rule 4) and level (Rule 6).

    Package extensions: a rule whose scope excludes this item is skipped, and `level_by_shift` can delay a match
    to a later level (math.inf leaves it out of every level).
    """
    matches = []
    for rule in references.rules_with_value(answer_key):
        if rule.index == asked_index:      # the asked index itself is the gold, not a match
            continue
        if not rule.applies_to(item):      # scope extension
            continue
        rule_shift = shift(rule.index, asked_index, coordinates, initials)
        level = max(rule.level, level_by_shift.get(rule_shift, 1.0))
        if level != math.inf:
            matches.append(Match(rule=rule, shift=rule_shift, level=level))
    return matches


def label(matches: Sequence[Match], coordinate_order: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Rules 3 and 5 for a wrong answer: return (label, shifts present in M)."""
    # Sort shifts by coordinate order, so the label reads {J, JD, T} and never {T, J, DJ}.
    position = {initial: i for i, initial in enumerate(coordinate_order)}
    shifts = tuple(sorted({m.shift for m in matches}, key=lambda s: [position[letter] for letter in s]))
    if not matches:
        return UNMATCHED, shifts
    if len(shifts) == 1:
        return shifts[0], shifts
    return f"{AMBIGUOUS} {{{', '.join(shifts)}}}", shifts


def count_sources(matches: Sequence[Match]) -> int:
    """Distinct sources in M. Two documented rules share a source when they differ only in time."""
    return len({m.rule.source for m in matches})
