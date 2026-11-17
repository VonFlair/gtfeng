"""Regression test: the package reproduces the paper's SituatedQA check (Section 7, Table 6 of arXiv:2611.18279v2).

Needs the tables built by examples/situatedqa/build_tables.py (the SituatedQA data are not redistributed)."""
import runpy
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

EX = Path(__file__).resolve().parents[1] / "examples" / "situatedqa"
EXPECTED = ["claude-sonnet-5-5     373      146    227 222        5                 0",
            "gpt-5.6-sol     476      187    289 277       12                 2"]


@unittest.skipUnless((EX / "references.csv").exists(), "SituatedQA tables not built")
class TestSituatedQA(unittest.TestCase):
    def test_table6(self):
        out = StringIO()
        with redirect_stdout(out):
            runpy.run_path(str(EX / "run_example.py"))
        text = " ".join(out.getvalue().split())
        for line in EXPECTED:
            self.assertIn(" ".join(line.split()), text)


if __name__ == "__main__":
    unittest.main()
