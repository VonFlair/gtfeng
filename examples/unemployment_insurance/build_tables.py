"""Build the GT-FENG reference set and gold table for the unemployment insurance (UI) instance.

Source: results/ui/ui_documented_values.csv (parsed from the DOL "Significant Provisions of State UI Laws",
January and July editions 2018-2026; scripts/ui_parse_sigpros.py).

Coordinates: quantity (wba_max, wba_min, wage_base), jurisdiction (50 states and DC, full names), time (edition
"YYYY-MM"), definition ("base" = without dependents' allowance or the standard rate group; "alt" = with dependents'
allowance or an alternative rate group). Both maxima of a weekly benefit cell "$767 or $844" are "base". The asked index is (q, state, "2026-01", "base").
Unit: dollars for the weekly benefit amounts and hundreds of dollars for the taxable wage base, so that the same
increments (±1 to ±4 units) give nearby values on a comparable scale.

Reference levels (mirroring the Medicaid instance):
  R1  every state's January 2026 values, both definitions (matches with shift J, JD)
  R2  the asked state's other definition and the editions 2024-01 to 2025-07, plus the federal anchor (FUTA wage
      base, $7,000 = 70 hundreds)
  R3  the asked state's editions 2018-01 to 2023-07
  R4  every value of the other two quantities in the asked state's rows, any edition (recorded as quantity
      "report cell", definition "cell")

usage: python3 scripts/ui_build_tables.py
"""
from pathlib import Path

import pandas as pd

NAMES = dict(AL="Alabama", AK="Alaska", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado",
             CT="Connecticut", DE="Delaware", DC="District of Columbia", FL="Florida", GA="Georgia", HI="Hawaii",
             ID="Idaho", IL="Illinois", IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana",
             ME="Maine", MD="Maryland", MA="Massachusetts", MI="Michigan", MN="Minnesota", MS="Mississippi",
             MO="Missouri", MT="Montana", NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey",
             NM="New Mexico", NY="New York", NC="North Carolina", ND="North Dakota", OH="Ohio", OK="Oklahoma",
             OR="Oregon", PA="Pennsylvania", RI="Rhode Island", SC="South Carolina", SD="South Dakota",
             TN="Tennessee", TX="Texas", UT="Utah", VT="Vermont", VA="Virginia", WA="Washington",
             WV="West Virginia", WI="Wisconsin", WY="Wyoming")
ASKED_TIME = "2026-01"
QUANTITIES = ["wba_max", "wba_min", "wage_base"]
UNIT = {"wba_max": 1.0, "wba_min": 1.0, "wage_base": 100.0}


def level_of(edition):
    if edition == ASKED_TIME:
        return 1
    return 2 if edition >= "2024-01" else 3


def main():
    d = pd.read_csv("results/ui/ui_documented_values.csv")
    d["jurisdiction"] = d.state.map(NAMES)
    assert d.jurisdiction.notna().all()
    d["value"] = d.apply(lambda r: round(r.value / UNIT[r.quantity], 2), axis=1)
    # A weekly benefit cell "$767 or $844" (Colorado, Minnesota) gives two maxima under different wage formulas, both
    # without dependents' allowances: both are the rule's value, so both carry definition "base" and both are correct.
    # Minnesota's 2018-2019 editions print the two formula maxima as a range ("$447- 693", footnote 6).
    alt_formula = d.quantity.str.startswith("wba") & (d.cell_text.str.contains(r"\bor\b", regex=True)
                                                      | ((d.state == "MN") & (d.quantity == "wba_max")))
    d.loc[alt_formula, "definition"] = "base"
    d = d.drop_duplicates(["quantity", "jurisdiction", "edition", "definition", "value"])

    rows = []
    for r in d.itertuples():
        rows.append({"quantity": r.quantity, "jurisdiction": r.jurisdiction, "time": r.edition,
                     "definition": r.definition, "value": r.value, "level": level_of(r.edition),
                     "asked_quantity": r.quantity, "asked_jurisdiction": None})
    rows.append({"quantity": "federal anchor", "jurisdiction": "US", "time": ASKED_TIME, "definition": "federal",
                 "value": 70.0, "level": 2, "asked_quantity": "wage_base", "asked_jurisdiction": None})
    # R4: values of the other two quantities in the asked state's rows (in the asked quantity's unit)
    for state, g in d.groupby("jurisdiction"):
        for q in QUANTITIES:
            others = g[g.quantity != q]
            for v in sorted(set((others.value * others.quantity.map(UNIT) / UNIT[q]).round(2))):
                rows.append({"quantity": "report cell", "jurisdiction": state, "time": ASKED_TIME,
                             "definition": "cell", "value": v, "level": 4, "asked_quantity": q,
                             "asked_jurisdiction": state})
    refs = pd.DataFrame(rows)
    out = Path("results/ui")
    refs.to_csv(out / "ui_references.csv", index=False)
    gold = d[(d.edition == ASKED_TIME) & (d.definition == "base")][["quantity", "jurisdiction", "value"]]
    gold = (gold.groupby(["quantity", "jurisdiction"], sort=True).value
            .agg(gold="first", gold_values=lambda v: " ".join(str(x) for x in v)).reset_index())
    assert len(gold) == 153, len(gold)
    gold.to_csv(out / "ui_gold.csv", index=False)
    print(refs.groupby("level").size().to_dict(), "gold items:", len(gold))


if __name__ == "__main__":
    main()
