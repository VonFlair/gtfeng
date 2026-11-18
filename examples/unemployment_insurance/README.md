# Unemployment insurance example

The second instance of the GT-FENG paper (Section 7): 51 jurisdictions × 3 quantities (maximum and minimum weekly
benefit amount without dependents' allowances, taxable wage base for standard-rated employers), asked as of January
2026, answered twice by nine models.

- `references.csv`: documented values from the U.S. Department of Labor, *Significant Provisions of State
  Unemployment Insurance Laws*, January and July editions 2018–2026 (a U.S. government work), with reference levels
  R1–R4. Weekly benefit amounts are in dollars, the wage base in hundreds of dollars.
- `gold.csv`: the asked index of every question. Colorado and Minnesota publish two maximum weekly benefit amounts
  under different wage formulas, and both count as correct.
- `answers.csv`: the stable and unstable answers of the nine models (CC BY 4.0).
- `parse_sigpros.py` and `build_tables.py` rebuild the tables from the DOL PDFs; `prompts.py` holds the prompts.
- `run_example.py` prints the standard report and the FENG value with cluster-bootstrap intervals.
