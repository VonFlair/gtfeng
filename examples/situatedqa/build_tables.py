# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""Build the GT-FENG tables for the SituatedQA check (paper Section 7, Table 6 of arXiv:2611.18279v2).

usage: python build_tables.py GEO_TEST_JSONL RUN_DIR [OUT_DIR]
GEO_TEST_JSONL  SituatedQA data/qa_data/geo.test.jsonl (download it from github.com/mikejqzhang/SituatedQA;
                it is not included here)
RUN_DIR         folder with claude-sonnet-5-5.jsonl and gpt-5.6-sol.jsonl (model responses)

Writes references.csv (one row per gold string: question id, location, value, and the question it applies to)
and answers.csv (one row per response: model, question id, location as written, repeat, answer).
"""
import json
import sys
from pathlib import Path

import pandas as pd

MODELS = ["claude-sonnet-5-5", "gpt-5.6-sol"]


def fix_mojibake(text):
    """Undo UTF-8 text that came back decoded as cp1250 (one Claude answer: 'AndrĂ©s' -> 'Andrés')."""
    if text and ("Ă" in text or "Ã" in text):
        try:
            return text.encode("cp1250").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def main(data_path, run_dir, out_dir="."):
    rows = [json.loads(l) for l in Path(data_path).read_text().splitlines() if l.strip()]
    refs = [{"quantity": str(r["id"]), "jurisdiction": r["location"], "value": a, "asked_quantity": str(r["id"])}
            for r in rows for a in r["answer"]]
    answers = []
    for model in MODELS:
        latest = {}
        for line in (Path(run_dir) / f"{model}.jsonl").read_text().splitlines():
            j = json.loads(line)
            if j.get("error") is None:      # latest error-free record per item and repeat
                latest[(str(j["id"]), j["location"], j["rep"])] = j
        for (qid, loc, rep), j in sorted(latest.items()):
            answers.append({"model": model, "quantity": qid, "jurisdiction": loc, "asked_as": loc, "repeat": rep,
                            "answer": fix_mojibake(j.get("pred_answer"))})
    out = Path(out_dir)
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False)
    pd.DataFrame(answers).to_csv(out / "answers.csv", index=False)
    print(f"{len(refs)} reference rows, {len(answers)} responses")


if __name__ == "__main__":
    main(*sys.argv[1:])
