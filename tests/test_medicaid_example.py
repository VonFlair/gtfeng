"""Regression test: the package reproduces the paper's Medicaid standard report (Section 5, Table 3)."""
import math
import unittest
from pathlib import Path

import pandas as pd

import gtfeng

EX = Path(__file__).resolve().parents[1] / "examples" / "medicaid"
EXPECTED = {  # (model, level): (U, shift_ambiguous, source_ambiguous, O cross-jurisdiction)
    ("claude-sonnet-5-5", 1): (25, 13, 30, 37), ("claude-sonnet-5-5", 2): (15, 28, 30, 16),
    ("claude-sonnet-5-5", 3): (11, 33, 35, 10), ("claude-sonnet-5-5", 4): (6, 35, 36, 8),
    ("gpt-5.6-sol", 1): (21, 21, 33, 47), ("gpt-5.6-sol", 2): (17, 29, 36, 32),
    ("gpt-5.6-sol", 3): (15, 32, 38, 25), ("gpt-5.6-sol", 4): (9, 40, 41, 15),
}


@unittest.skipUnless((EX / "references.csv").exists(), "example tables not built")
class TestMedicaid(unittest.TestCase):
    def test_table3(self):
        res = gtfeng.evaluate(
            pd.read_csv(EX / "answers.csv"), pd.read_csv(EX / "references.csv"), groups=["model"],
            level_by_shift={"D": 2, "T": 2, "TD": 2, "JT": math.inf, "JTD": math.inf},
            scope={"asked_quantity": "quantity", "asked_jurisdiction": "jurisdiction"})
        rep = res.report(families={"cross": ["J", "JD"]}).set_index(["model", "level"])
        for k, (u, sa, so, o) in EXPECTED.items():
            r = rep.loc[k]
            self.assertEqual((r.U, r.shift_ambiguous, r.source_ambiguous, r.O_cross), (u, sa, so, o), k)


if __name__ == "__main__":
    unittest.main()
