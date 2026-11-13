"""Regression test: the package reproduces the paper's Medicaid standard report (Section 6, Table 3)."""
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

    def test_rule7_feng_value(self):
        """Rule 7 with increments +-1..+-4 reproduces the local perturbation baseline of the Medicaid instance."""
        res = gtfeng.evaluate(
            pd.read_csv(EX / "answers.csv"), pd.read_csv(EX / "references.csv"), groups=["model"],
            level_by_shift={"D": 2, "T": 2, "TD": 2, "JT": math.inf, "JTD": math.inf},
            scope={"asked_quantity": "quantity", "asked_jurisdiction": "jurisdiction"})
        fv = res.feng_value(["J", "JD"], permutations=1000).set_index(["model", "level"])
        expected = {  # (model, level): (W, O, N, FENG value)
            ("claude-sonnet-5-5", 1): (62, 37, 27.21, 0.281), ("claude-sonnet-5-5", 4): (62, 8, 19.46, -0.270),
            ("gpt-5.6-sol", 1): (68, 47, 26.41, 0.495), ("gpt-5.6-sol", 4): (68, 15, 19.79, -0.099),
        }
        for k, (w, o, n, f) in expected.items():
            r = fv.loc[k]
            self.assertEqual((r.W, r.O, r.N, r.feng_value), (w, o, n, f), k)


if __name__ == "__main__":
    unittest.main()
