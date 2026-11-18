"""Parse the DOL "Significant Provisions of State Unemployment Insurance Laws" PDFs (January and July editions,
2018-2026) into one table of documented values.

Columns read from each state row: minimum and maximum weekly benefit amount (WBA) and wages subject to tax (the
taxable wage base). A WBA cell "$56 - $128" gives the amount without and with dependents' allowances (DA); a cell
"$767 or $844" gives two statutory alternatives. The first value has definition "base", the second "alt".
Footnote superscripts (font size < 6.5) are dropped. Column bands are located from the header words on each page.

usage: python3 scripts/ui_parse_sigpros.py data/ui_sigpros results/ui/ui_documented_values.csv
"""
import csv
import re
import sys
from pathlib import Path

import pymupdf as fitz

STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND "
             "OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
MONTH = {"January": 1, "July": 7}
ROW_LABELS = STATES | {"PR", "VI"}   # PR and VI delimit rows but are not part of the 51 jurisdictions


def spans(page):
    """Words (x0, y0, x1, text) built from characters, so text that runs across two columns is split at spaces."""
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["size"] < 6.5:
                    continue
                word = []
                for ch in s["chars"] + [{"c": " ", "bbox": None}]:
                    if ch["c"].isspace():
                        if word:
                            out.append((word[0]["bbox"][0], word[0]["bbox"][1], word[-1]["bbox"][2],
                                        "".join(c["c"] for c in word)))
                        word = []
                    else:
                        word.append(ch)
    return out


ANCHORS = {"earn": ("Qualify",), "comp": ("Computation",), "min": ("Mini",), "max": ("Maxi",),
           "disr": ("Disregard",), "calc": ("Calculation",), "weeks": ("Payable",), "payroll": ("Payroll",),
           "wage": ("Subject",), "rates": ("Rates",)}


def bands(sp):
    """Centre x of every column header."""
    head = [s for s in sp if s[1] < 200]
    centres = {}
    for k, prefixes in ANCHORS.items():
        c = sorted(((s[0] + s[2]) / 2) for s in head if s[3].startswith(prefixes))
        if c:
            centres[k] = c[-1] if k == "rates" else c[0]
    # the WBA sub-headers are the "Maxi" nearest to the left of "Disregarded" and the "Mini" nearest to its left
    centres["max"] = max((s[0] + s[2]) / 2 for s in head if s[3].startswith("Maxi") and (s[0] + s[2]) / 2 < centres["disr"])
    centres["min"] = max((s[0] + s[2]) / 2 for s in head if s[3].startswith("Mini") and (s[0] + s[2]) / 2 < centres["max"])
    return centres


# A cell is left-aligned near its header: a word whose left edge lies well left of the header centre is the tail of
# the neighbouring column (e.g. "DA up to $72" in Computation, "$1,500 in any qtr" in Size of Payroll).
GUARD = {"min": 18, "wage": 22}


def column(span, centres):
    x = (span[0] + span[2]) / 2
    k = min(centres, key=lambda c: abs(centres[c] - x))
    if k in GUARD and span[0] < centres[k] - GUARD[k]:
        return None
    return k


def dollars(text):
    """Dollar values in order; in a range "$56- 128" the second number carries no dollar sign."""
    text = re.sub(r"(\$\s?[\d,]+(?:\.\d+)?)\s*-\s*(?=[\d])", r"\1 - $", text)
    return [float(v.replace(",", "")) for v in re.findall(r"\$\s?([\d,]+(?:\.\d+)?)", text)]


def parse(pdf):
    m = re.match(r"(January|July)(\d{4})", Path(pdf).stem)
    edition = f"{m.group(2)}-{MONTH[m.group(1)]:02d}"
    rows = []
    bd = None
    for page in fitz.open(pdf):
        sp = spans(page)
        labels = sorted((s[1], s[3]) for s in sp if s[0] < 35 and s[3] in ROW_LABELS)
        if not labels:
            continue
        ys = [y for y, _ in labels]
        header = [s for s in sp if s[1] < 200 and s[3].startswith(("Mini", "Maxi"))]
        if header:   # a page with the table header; continuation pages reuse the previous bands
            bd = bands(sp)
            top = max(s[1] + 8 for s in sp if s[1] < ys[0] - 15 and s[3].startswith(("Mini", "Maxi", "Rate", "ployer")))
        else:
            top = 30
        for i, (y, st) in enumerate(labels):
            lo = top if i == 0 else (ys[i - 1] + y) / 2
            hi = (y + ys[i + 1]) / 2 if i + 1 < len(ys) else y + 40   # the last row ends before the footnotes
            cell = {}
            row = [s for s in sp if lo <= s[1] < hi and s[0] >= 35]
            for k in ("min", "max", "wage"):
                words = sorted((s for s in row if column(s, bd) == k), key=lambda s: (s[1], s[0]))
                cell[k] = " ".join(w[3] for w in words)
            if st not in STATES:
                continue
            rows.append({"edition": edition, "state": st, **{f"{k}_text": v for k, v in cell.items()}})
    return rows


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for pdf in sorted(src.glob("*.pdf")):
        for r in parse(pdf):
            for q, key in (("wba_min", "min_text"), ("wba_max", "max_text"), ("wage_base", "wage_text")):
                vals = dollars(r[key])
                if not vals and q != "wage_base" and re.match(r"^\d[\d,]*\s*-\s*\d", r[key]):
                    vals = dollars("$" + r[key])   # the source omits "$" in a few WBA cells, e.g. "68-76"
                for d, v in zip(("base", "alt"), vals[:2]):
                    records.append({"edition": r["edition"], "state": r["state"], "quantity": q, "definition": d,
                                    "value": v, "cell_text": r[key]})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"{len(records)} values written to {out}")


if __name__ == "__main__":
    main()
