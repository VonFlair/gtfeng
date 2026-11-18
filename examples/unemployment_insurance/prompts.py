"""Prompts for the unemployment insurance (UI) instance. Same structure as the Medicaid prompts of our earlier study
(Right Number, Wrong State?, src/jg/run.py, prompt v1): one system prompt, one template with a reference date, one
question per quantity, and the same strict JSON schema."""
import pandas as pd

REFERENCE_DATE = "January 2026"
SYSTEM = ("You are answering factual questions about U.S. state unemployment insurance law. "
          "Answer from your own knowledge. Respond with JSON only.")
QUESTION = {
    "wba_max": ("what is the maximum weekly benefit amount for regular unemployment insurance in {state}, "
                "excluding any dependents' allowance?"),
    "wba_min": ("what is the minimum weekly benefit amount for regular unemployment insurance in {state}, "
                "excluding any dependents' allowance?"),
    "wage_base": ("what is the taxable wage base for state unemployment insurance in {state}, that is, the annual "
                  "wages per employee subject to the state unemployment tax for standard-rated employers?"),
}
TEMPLATE = ("As of {ref}, {question} Express the answer in U.S. dollars. "
            'Return JSON of the form {{"state": "<state>", "value": <number>, "unit": "USD"}}.')
SCHEMA = {"type": "object", "properties": {"state": {"type": "string"}, "value": {"type": "number"},
                                           "unit": {"type": "string"}}, "required": ["state", "value", "unit"]}
STRICT_SCHEMA = {**SCHEMA, "additionalProperties": False}


def prompt_for(state, quantity):
    return TEMPLATE.format(ref=REFERENCE_DATE, question=QUESTION[quantity].format(state=state))


def items(gold_csv="results/ui/ui_gold.csv"):
    g = pd.read_csv(gold_csv)
    return [{"key": f"{r.jurisdiction}|{r.quantity}", "state": r.jurisdiction, "construct": r.quantity,
             "system": SYSTEM, "user": prompt_for(r.jurisdiction, r.quantity), "schema": STRICT_SCHEMA}
            for r in g.itertuples()]
