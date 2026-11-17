# gtfeng

**GT-FENG (General Theory of Factual Evaluation and Normative Grounding)** evaluates language-model answers about facts whose correct value is set by a rule: an income limit, a filing deadline, a legal duty. The same answer can be right under one jurisdiction, date, or definition and wrong under another. `gtfeng` scores each answer against the rule that applies and lists every documented rule that matches a wrong answer. Paper: [arXiv:2611.18279](https://arxiv.org/abs/2611.18279).

Use it when you evaluate model answers about numbers, dates, or yes/no facts set by law or policy, and you want to know whether a wrong answer is the correct value for another state, another year, or another definition.

## Install

```bash
pip install gtfeng          # from PyPI
pip install .               # from a clone of this repository
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

## Key terms

| Term | Meaning | Example (quick start) |
|---|---|---|
| match | a documented rule, other than the one asked about, that gives the same value as a wrong answer | 40 matches B's 2025 rule and A's 2018 rule |
| U | a wrong answer with no match | an answer of 45 |
| shift | the coordinates in which a match differs from the question, written as their initials | B 2025 has shift J (another jurisdiction); A 2018 has shift T (another time) |
| source | where a match comes from; matches that differ only in year count as one source | A's 2018 and A's 2021 rules would be one source |
| reference level | how many documented rules a wrong answer is checked against; R1 is the smallest set, and each later level adds rules. Set with the `level` column or `level_by_shift` (default: everything at R1) | in the Medicaid example, R1 holds other states' current limits and R2 adds the asked state's earlier ones |
| ambiguous | the matches have two or more different shifts (shift ambiguity); matches from two or more sources are source ambiguity | 40 is `ambiguous {J, T}` |
| family | a set of shifts treated as one kind of error | {J, JD}: the value belongs to another jurisdiction |
| grounded in a family | every match of the wrong answer has a shift in the family | 40 is not grounded in {J, JD}, because its match A 2018 has shift T |

## Inputs

| Table | Required columns | Optional columns |
|---|---|---|
| `references` | the coordinates (default `quantity`, `jurisdiction`, `time`, `definition`) and `value` | `level` (reference level, default 1); scope columns |
| `answers` | the coordinates and `answer` | group columns such as `model`; repeated rows for repeated responses |

Coordinates are configurable (`coordinates=[...]`); each must have a distinct initial, which names its shift. Values can be numbers, Booleans, strings, or lists compared as sets.

## What it computes

Rule numbers follow the [GT-FENG paper](https://arxiv.org/abs/2611.18279).

| Paper | `gtfeng` |
|---|---|
| Stability | an item enters only when all repeats give the same answer (`result.unstable` lists the rest) |
| Rule 1, correct | `label == "correct"`; several rows at the asked index (e.g. spellings of one answer) are all accepted |
| Rules 2–3, matches and U | `matches`; `label == "U"` when nothing matches |
| Rule 4, shift | `shifts`, e.g. `J`, `T`, `JD`, `JT` |
| Rule 5, label and ambiguity | single shift, or `ambiguous {...}` (shift ambiguity); `n_sources` counts distinct sources, so source ambiguity is `n_sources >= 2` (documented rules that differ only in time are one source) |
| Rule 6, reference levels | `level` column (level k means R_k); extension `level_by_shift`, e.g. `{"T": 2}` holds matches with shift T back until level 2 |
| Rule 7, local perturbation baseline | `result.feng_value(family)`: O, N, E, the FENG value, and p; `bootstrap=10000` adds a 95% confidence interval |
| Standard report | `result.report(families=...)`: correct, U, single-shift labels, shift-ambiguous, source-ambiguous, mean sources per matched item, and O for each family |

Extension `scope={"ref_column": "answer_coordinate"}` restricts a documented rule to items whose coordinate equals the rule's value in that column.

## Rule 7 and the FENG value

For a family of shifts F, `result.feng_value(F)` compares how often wrong answers are grounded in F with how often nearby values are:

```python
result.feng_value(["J", "JD"], bootstrap=10000)   # one row per group and reference level
#  W  O      N     E  feng_value       p  ci_low  ci_high
```

| Column | Meaning |
|---|---|
| `W` | wrong items |
| `O` | wrong items grounded in F (they have matches, and every match has a shift in F) |
| `N` | local perturbation baseline: each wrong answer A is replaced by A + i for every increment i (default ±1 to ±4; increments that reach the correct value are dropped), and N sums each item's share of nearby values grounded in F |
| `E` | O − N |
| `feng_value` | (O − N) / (W − N); NaN when N ≥ W |
| `p` | share of random perturbations (one increment per item) whose grounded count reaches O |
| `ci_low`, `ci_high` | with `bootstrap=B`: percentile interval (level `ci`, default 0.95) of the FENG value over B resamples of the group's stable items |

The FENG value is 0 when wrong answers are grounded in F as often as nearby values, 1 when every wrong answer is grounded in F, and negative when nearby values are grounded more often. It has the chance-corrected form of Cohen's kappa; the match event and the local baseline come from GT-FENG. Rule 7 needs numeric answers, and the increments must leave out any known definition shift (the Medicaid example leaves out 5).

`result.feng_bootstrap(F)` returns the resampled FENG values themselves. Resample d uses the same items at every reference level, so the difference between two levels has an interval too:

```python
boot = result.feng_bootstrap(["J", "JD"])
wide = boot.pivot_table(index=["model", "resample"], columns="level", values="feng_value")
(wide[1] - wide[4]).groupby("model").quantile([0.025, 0.975])   # drop from R1 to R4
```

Report the FENG value next to accuracy: it describes the structure of wrong answers and does not rank models.

## Example: Medicaid income limits

[`examples/medicaid/`](https://github.com/VonFlair/gtfeng/tree/main/examples/medicaid) (in the GitHub repository) reproduces the paper's Medicaid instance (two models, 51 jurisdictions, three quantities). The data come from [*Right Number, Wrong State?*](https://arxiv.org/abs/2610.09458) (Feng, 2026); if you use them, please cite that paper as well. Its four reference levels follow Table 2 of the paper:

| Level | A wrong answer is also checked against |
|---|---|
| R1 | other states' July 2025 limits, with and without the 5-point income disregard |
| R2 | the asked state's own limit under the other definition, three earlier editions, and the federal anchors 100, 133 and 138 (percent of the federal poverty level) |
| R3 | every edition of the asked state's limits, 2018–2024 |
| R4 | every other number in the asked state's rows of the MACStats data book, 2018–2025 |

```bash
cd examples/medicaid
python run_example.py
```

The input tables are included (`build_tables.py` rebuilds them from the original paper's supplementary materials), and `run_example.py` shows how to set reference levels with `level_by_shift` and limit rules with `scope`. The test suite checks that the output equals Table 3 of arXiv:2611.18279v2 and that Rule 7 reproduces the local perturbation baseline of the Medicaid instance.

## Example: SituatedQA

[`examples/situatedqa/`](https://github.com/VonFlair/gtfeng/tree/main/examples/situatedqa) reproduces the paper's check on the geographic questions of SituatedQA (Zhang and Choi, 2021): text answers, one coordinate (location), several accepted spellings per gold answer. The SituatedQA data are not included; the example explains how to download them.

## Citation

`gtfeng.cite()` returns the BibTeX entry (`gtfeng.cite("text")` a plain reference), and `result.report_markdown()` signs every report with "Evaluated with GT-FENG v0.1.1 (Feng, 2026)." If you use GT-FENG, please cite:

```bibtex
@article{feng2026gtfeng,
  title   = {GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models},
  author  = {Feng, Jiayu},
  journal = {arXiv preprint arXiv:2611.18279},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.18279}
}
```

## License

The code is licensed under the [Apache License 2.0](https://github.com/VonFlair/gtfeng/blob/main/LICENSE). Anyone who redistributes it, modified or not, must keep the `LICENSE` and `NOTICE` files, which carry the GT-FENG name and its author. The license grants no right to use the name GT-FENG for other products.

The example data in `examples/medicaid/` are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The gold values are derived from MACPAC MACStats Exhibits 35–36.
