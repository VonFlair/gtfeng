import math
import unittest

import pandas as pd

import gtfeng


def refs():
    rows = [
        # quantity, jurisdiction, time, definition, value
        ("limit", "A", 2025, "x", 50), ("limit", "A", 2018, "x", 40),
        ("limit", "B", 2025, "x", 40), ("limit", "B", 2018, "x", 60),
        ("limit", "C", 2025, "x", 60),
        ("cap", "A", 2025, "x", 10), ("cap", "B", 2025, "x", 45), ("cap", "B", 2018, "x", 45),
    ]
    return pd.DataFrame(rows, columns=["quantity", "jurisdiction", "time", "definition", "value"])


def ask(q, j, a, model="m", reps=1):
    return [{"model": model, "quantity": q, "jurisdiction": j, "time": 2025, "definition": "x", "answer": a}] * reps


class TestRules(unittest.TestCase):
    def run_one(self, rows, **kw):
        res = gtfeng.evaluate(pd.DataFrame(rows), refs(), groups=["model"], **kw)
        return res

    def label(self, q, j, a, level=None, **kw):
        it = self.run_one(ask(q, j, a), **kw).items
        if level is not None:
            it = it[it.level == level]
        return it.iloc[-1]

    def test_correct(self):
        self.assertEqual(self.label("limit", "A", 50).label, "correct")

    def test_unmatched(self):
        self.assertEqual(self.label("limit", "A", 55).label, "U")

    def test_shift_ambiguity(self):
        r = self.label("limit", "A", 40)
        self.assertEqual(r.label, "ambiguous {J, T}")
        self.assertEqual(r.n_sources, 2)

    def test_joint_shift_and_sources(self):
        r = self.label("limit", "A", 60)
        self.assertEqual(r.shifts, ("J", "JT"))
        self.assertEqual(r.n_sources, 2)  # C and B are different jurisdictions

    def test_one_source_two_shifts(self):
        r = self.label("cap", "A", 45)  # B set 45 in 2018 and 2025: one source, shifts J and JT
        self.assertEqual(r.shifts, ("J", "JT"))
        self.assertEqual(r.n_sources, 1)

    def test_levels_by_shift(self):
        kw = {"level_by_shift": {"T": 2, "JT": math.inf}}
        self.assertEqual(self.label("limit", "A", 40, level=1, **kw).label, "J")
        self.assertEqual(self.label("limit", "A", 40, level=2, **kw).label, "ambiguous {J, T}")
        self.assertEqual(self.label("limit", "A", 60, level=2, **kw).label, "J")

    def test_unstable_dropped(self):
        rows = ask("limit", "A", 40) + ask("limit", "A", 60)
        res = self.run_one(rows)
        self.assertTrue(res.items.empty)
        self.assertEqual(len(res.unstable), 1)

    def test_report_monotone_and_family(self):
        rows = ask("limit", "A", 40) + ask("limit", "B", 60) + ask("limit", "C", 40)
        rep = self.run_one(rows, level_by_shift={"T": 2, "JT": math.inf}).report(
            families={"cross": ["J", "JD"]})
        o = rep.set_index("level")["O_cross"]
        self.assertGreaterEqual(o.loc[1], o.loc[2])

    def test_value_types(self):
        r = pd.DataFrame([("duty", "A", 2025, "x", True), ("duty", "B", 2025, "x", False),
                          ("list", "A", 2025, "x", ["a", "b"]), ("list", "B", 2025, "x", ["b", "c"]),
                          ("list", "A", 2018, "x", ["b", "a", "c"])],
                         columns=["quantity", "jurisdiction", "time", "definition", "value"])
        a = pd.DataFrame(ask("duty", "A", False) + ask("list", "A", ["c", "b"]))
        it = gtfeng.evaluate(a, r, groups=["model"]).items.set_index("quantity")
        self.assertEqual(it.loc["duty", "label"], "J")
        self.assertEqual(it.loc["list", "label"], "J")

    def test_citation_and_signature(self):
        self.assertIn("Feng, Jiayu", gtfeng.cite())
        self.assertTrue(gtfeng.cite("text").startswith("Feng, J. (2026)"))
        res = self.run_one(ask("limit", "A", 40))
        self.assertIn("GT-FENG", res.report().attrs["generated_by"])
        md = res.report_markdown()
        self.assertTrue(md.rstrip().endswith("(Feng, 2026)."))

    def test_missing_values(self):
        self.assertIsNone(gtfeng.normalize(pd.NA))
        self.assertIsNone(gtfeng.normalize(float("nan")))
        self.assertIsNone(gtfeng.normalize(["a", None]))
        self.assertEqual(gtfeng.normalize(205), gtfeng.normalize(205.0))
        res = self.run_one(ask("limit", "A", pd.NA, reps=2))
        self.assertTrue(res.items.empty)
        self.assertEqual(len(res.unstable), 1)

    def test_initials_must_be_unique(self):
        with self.assertRaises(ValueError):
            gtfeng.evaluate(pd.DataFrame(), refs(), coordinates=["time", "territory"])


if __name__ == "__main__":
    unittest.main()
