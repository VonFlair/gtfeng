# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""GT-FENG: General Theory of Factual Evaluation and Normative Grounding.

    import gtfeng
    result = gtfeng.evaluate(answers, references, groups=["model"])
    result.report(families={"cross_jurisdiction": ["J", "JD"]})
    print(gtfeng.cite())   # BibTeX for the GT-FENG paper

Modules, in the order of the paper:
    values      how an answer and a gold value are compared
    references  the reference set R of documented rules
    rules       Rules 1-6, one function each
    core        evaluate(): runs the rules on every stable item
    report      the standard report and citation helpers
"""
from .core import DEFAULT_COORDINATES, Result, evaluate
from .report import VERSION, cite
from .rules import AMBIGUOUS, CORRECT, UNMATCHED
from .values import normalize

__version__ = VERSION
__all__ = ["evaluate", "cite", "Result", "normalize", "DEFAULT_COORDINATES", "CORRECT", "UNMATCHED", "AMBIGUOUS",
           "__version__"]
