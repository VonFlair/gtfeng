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


class TestFengValue(unittest.TestCase):
    def setUp(self):
        self.refs = pd.DataFrame(
            [("limit", "A", 2025, "x", 50), ("limit", "B", 2025, "x", 40)],
            columns=["quantity", "jurisdiction", "time", "definition", "value"])

    def ask(self, answer):
        return pd.DataFrame([{"quantity": "limit", "jurisdiction": "A", "time": 2025, "definition": "x",
                              "answer": answer}])

    def test_grounded_answer_beyond_baseline(self):
        # 40 matches B's rule; the nearby values 39 and 41 match nothing, so N = 0 and the FENG value is 1.
        fv = gtfeng.evaluate(self.ask(40), self.refs).feng_value(["J"], increments=[-1, 1], permutations=100)
        row = fv.iloc[0]
        self.assertEqual((row.W, row.O, row.N, row.feng_value, row.p), (1, 1, 0.0, 1.0, 0.0))

    def test_increment_reaching_gold_is_dropped(self):
        # 39 + 1 = 40 matches B; 39 + 11 = 50 is the correct value and is dropped, so N = 1 / 1 = 1.
        fv = gtfeng.evaluate(self.ask(39), self.refs).feng_value(["J"], increments=[1, 11], permutations=100)
        row = fv.iloc[0]
        self.assertEqual((row.O, row.N, row.E), (0, 1.0, -1.0))

    def test_numeric_answers_only(self):
        refs = pd.DataFrame([("rule", "A", 2025, "x", "yes"), ("rule", "B", 2025, "x", "no")],
                            columns=["quantity", "jurisdiction", "time", "definition", "value"])
        answers = pd.DataFrame([{"quantity": "rule", "jurisdiction": "A", "time": 2025, "definition": "x",
                                 "answer": "no"}])
        with self.assertRaises(ValueError):
            gtfeng.evaluate(answers, refs).feng_value(["J"])


class TestSeveralGoldValues(unittest.TestCase):
    def test_any_documented_value_at_the_asked_index_is_correct(self):
        # Two rows at index (q, A) give two accepted spellings; either one is correct (Rule 1).
        refs = pd.DataFrame([("q", "A", 2021, "x", "cbs"), ("q", "A", 2021, "x", "on cbs"),
                             ("q", "B", 2021, "x", "channel 5")],
                            columns=["quantity", "jurisdiction", "time", "definition", "value"])
        answers = pd.DataFrame([{"quantity": "q", "jurisdiction": "A", "time": 2021, "definition": "x", "answer": a,
                                 "model": a} for a in ("on cbs", "channel 5")])
        items = gtfeng.evaluate(answers, refs, groups=["model"]).items.set_index("model")
        self.assertEqual(items.loc["on cbs", "label"], "correct")
        self.assertEqual(items.loc["channel 5", "label"], "J")


if __name__ == "__main__":
    unittest.main()
