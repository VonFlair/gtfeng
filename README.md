# gtfeng

**GT-FENG (General Theory of Factual Evaluation and Normative Grounding)** evaluates language-model answers about facts whose correct value is set by a rule: an income limit, a filing deadline, a legal duty. The same answer can be right under one jurisdiction, date, or definition and wrong under another. `gtfeng` scores each answer against the rule that applies and lists every documented rule that matches a wrong answer.

## Install

```bash
pip install gtfeng          # after release on PyPI
pip install .               # from this folder
```

Requires Python 3.9+ and pandas.

## Quick start

```python
import pandas as pd
import gtfeng

references = pd.DataFrame(
    [("limit", "A", 2025, "x", 50), ("limit", "A", 2018, "x", 40), ("limit", "B", 2025, "x", 40)],
    columns=["quantity", "jurisdiction", "time", "definition", "value"])
answers = pd.DataFrame(
    [{"model": "m", "quantity": "limit", "jurisdiction": "A", "time": 2025, "definition": "x", "answer": 40}])

result = gtfeng.evaluate(answers, references, groups=["model"])
result.items[["jurisdiction", "answer", "label", "n_sources"]]
#   jurisdiction  answer             label  n_sources
#              A      40  ambiguous {J, T}          2
result.report(families={"cross_jurisdiction": ["J", "JD"]})
```

The answer 40 is wrong for A in 2025. It matches B's current rule (shift **J**, another jurisdiction) and A's 2018 rule (shift **T**, another time), so GT-FENG labels it ambiguous and reports both sources.

## Inputs

| Table | Required columns | Optional columns |
|---|---|---|
| `references` | the coordinates (default `quantity`, `jurisdiction`, `time`, `definition`) and `value` | `level` (reference level, default 1); scope columns |
| `answers` | the coordinates and `answer` | group columns such as `model`; repeated rows for repeated responses |

Coordinates are configurable (`coordinates=[...]`); each must have a distinct initial, which names its shift. Values can be numbers, Booleans, strings, or lists compared as sets.

## What it computes

| Paper | `gtfeng` |
|---|---|
| Stability | an item enters only when all repeats give the same answer (`result.unstable` lists the rest) |
| Rule 1, correct | `label == "correct"` |
| Rules 2–3, matches and U | `matches`; `label == "U"` when nothing matches |
| Rule 4, shift | `shifts`, e.g. `J`, `T`, `JD`, `JT` |
| Rule 5, label and ambiguity | single shift, or `ambiguous {...}` (shift ambiguity); `n_sources` counts distinct sources, so source ambiguity is `n_sources >= 2` (documented rules that differ only in time are one source) |
| Rule 6, reference levels | `level` column (level k means R_k); extension `level_by_shift`, e.g. `{"T": 2}` holds matches with shift T back until level 2 |
| Standard report | `result.report(families=...)`: correct, U, single-shift labels, shift-ambiguous, source-ambiguous, mean sources per matched item, and O for each family |

Extension `scope={"ref_column": "answer_coordinate"}` restricts a documented rule to items whose coordinate equals the rule's value in that column.

Version 0.1 implements Rules 1–6 and the standard report. Rule 7 (the local perturbation baseline) will follow in a later version.

## Example: Medicaid income limits

`examples/medicaid/` reproduces the paper's Medicaid instance (two models, 51 jurisdictions, three quantities, reference levels R1–R4):

```bash
cd examples/medicaid
python run_example.py
```

The test suite checks that the output equals the paper's Table 3.

## Citation

`gtfeng.cite()` returns the BibTeX entry (`gtfeng.cite("text")` a plain reference), and `result.report_markdown()` signs every report with "Evaluated with GT-FENG v0.1.0 (Feng, 2026)." If you use GT-FENG, please cite:

```bibtex
@article{feng2026gtfeng,
  title  = {GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models},
  author = {Feng, Jiayu},
  year   = {2026}
}
```

## License

The code is licensed under the [Apache License 2.0](LICENSE). Anyone who redistributes it, modified or not, must keep the `LICENSE` and `NOTICE` files, which carry the GT-FENG name and its author. The license grants no right to use the name GT-FENG for other products.

The example data in `examples/medicaid/` are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The gold values are derived from MACPAC MACStats Exhibits 35–36.
