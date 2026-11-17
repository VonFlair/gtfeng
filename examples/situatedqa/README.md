# SituatedQA example

Reproduces the GT-FENG check on the geographic test set of SituatedQA (paper Section 7, Table 6 of arXiv:2611.18279v2): 506
question–location pairs, two models, two repeats each.

The SituatedQA data are not included. Download `data/qa_data/geo.test.jsonl` from
[github.com/mikejqzhang/SituatedQA](https://github.com/mikejqzhang/SituatedQA) and cite
Zhang and Choi (2021), "SituatedQA: Incorporating Extra-Linguistic Contexts into QA" (EMNLP 2021).
The model responses are in `responses/` (CC BY 4.0); if you use them, please cite GT-FENG
([arXiv:2611.18279](https://arxiv.org/abs/2611.18279)) as well.

```bash
python build_tables.py path/to/geo.test.jsonl responses
python run_example.py
```

Each question is asked "As of February 2021", the date of SituatedQA's Wikipedia snapshot. Answers and gold strings
are compared after removing text in parentheses and applying the normalization of SituatedQA's evaluation script.
