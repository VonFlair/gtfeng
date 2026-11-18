"""Regression test: the package reproduces the unemployment insurance instance (paper Section 7, Tables 6 and 16)."""
import math
import unittest
from pathlib import Path

import pandas as pd

import gtfeng

HERE = Path(__file__).resolve().parents[1] / "examples" / "unemployment_insurance"


class TestUnemploymentInsurance(unittest.TestCase):
    def test_report_and_feng(self):
        res = gtfeng.evaluate(pd.read_csv(HERE / "answers.csv"), pd.read_csv(HERE / "references.csv"),
                              groups=["model"],
                              level_by_shift={"D": 2, "T": 2, "TD": 2, "JT": math.inf, "JTD": math.inf},
                              scope={"asked_quantity": "quantity", "asked_jurisdiction": "jurisdiction"})
        rep = res.report(families={"cross": ["J", "JD"]}).set_index(["model", "level"])
        self.assertEqual(tuple(rep.loc[("gpt-5.6-sol", 1), ["correct", "wrong", "O_cross"]]), (82, 66, 17))
        self.assertEqual(tuple(rep.loc[("claude-sonnet-5-5", 4), ["wrong", "O_cross", "source_ambiguous"]]), (47, 4, 7))
        fv = res.feng_value(["J", "JD"], permutations=200).set_index(["model", "level"])
        self.assertAlmostEqual(fv.loc[("gemma-4-31b-it", 1)].feng_value, 0.387, places=3)
        self.assertAlmostEqual(fv.loc[("claude-sonnet-5-5", 4)].feng_value, -0.026, places=3)


if __name__ == "__main__":
    unittest.main()
