# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""The reference set R: every documented rule, indexed by its coordinates.

A documented rule sets one value at one index, for example (quantity="pregnancy", jurisdiction="Iowa",
time=2025, definition="canon") -> 205. The gold value G of a question is the value of the documented rule at the
question's index.

Naming: a "documented rule" is a row of the references table. The paper's "Rules 1-6" are the evaluation steps in
rules.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd

from .values import is_missing


@dataclass(frozen=True)
class DocumentedRule:
    """One row of the references table."""

    index: tuple          # coordinate values, in the order of `coordinates`
    value_key: object     # comparison key of the rule's value (see values.normalize)
    level: float          # the first reference level that contains this rule (level k means R_k)
    source: tuple         # all coordinates except time, plus the scope columns: equal sources = one source
    scope: dict           # {answers coordinate: required value}; empty when the rule applies to every item

    def applies_to(self, item: Mapping) -> bool:
        """A scoped rule applies only to items whose coordinates equal its scope values."""
        return all(item[coordinate] == required for coordinate, required in self.scope.items())


class ReferenceSet:
    """All documented rules, with lookup by index (for the gold value) and by value (for matches)."""

    def __init__(self, table: pd.DataFrame, *, coordinates: Sequence[str], value: str, level: str | None,
                 time_coordinate: str | None, scope: Mapping[str, str], compare: Callable):
        source_columns = [c for c in coordinates if c != time_coordinate] + list(scope)
        self.rules: list[DocumentedRule] = []
        for row in table.to_dict("records"):
            value_key = compare(row[value])
            if value_key is None:  # a rule without a value cannot be gold and cannot match anything
                continue
            self.rules.append(DocumentedRule(
                index=tuple(row[c] for c in coordinates),
                value_key=value_key,
                level=float(row[level]) if level and level in row else 1.0,
                source=tuple(None if is_missing(row[c]) else row[c] for c in source_columns),
                scope={item_column: row[ref_column] for ref_column, item_column in scope.items()
                       if not is_missing(row[ref_column])},
            ))
        self._gold: dict[tuple, object] = {}
        self._by_value: dict[object, list[DocumentedRule]] = {}
        for rule in self.rules:
            self._gold.setdefault(rule.index, rule.value_key)  # if an index appears twice, the first row is gold
            self._by_value.setdefault(rule.value_key, []).append(rule)

    def gold(self, index: tuple):
        """G at an index: the value of the documented rule at that index."""
        if index not in self._gold:
            raise KeyError(f"no documented rule with a value at the asked index {index}; "
                           "every question needs a gold value in the references table")
        return self._gold[index]

    def rules_with_value(self, value_key) -> list[DocumentedRule]:
        """Every documented rule whose value equals `value_key`."""
        return self._by_value.get(value_key, [])

    def levels(self) -> list[float]:
        """The levels that occur in the references table, in increasing order."""
        return sorted({rule.level for rule in self.rules})
