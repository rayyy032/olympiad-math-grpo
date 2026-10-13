"""Dataset builders: GSM8K / MATH difficulty-gradient mix + offline synthetic.

``dataset`` config values:
  * ``synthetic``    -> tiny offline math QA list (smoke tests, no network).
  * ``gsm8k``         -> GSM8K (grade-school, single difficulty level).
  * ``math``          -> Hendrycks MATH, all levels 1-5.
  * ``math_l45``      -> MATH levels 4-5 only (competition-grade training mix).
  * ``aime``          -> AI-MO AIME validation set (olympiad, 90 problems).
  * ``amc``           -> AI-MO AMC validation set (83 problems).
  * ``math_l5``       -> AI-MO MATH level-5 validation set (721 problems).
  * ``olympiad``      -> AIME + AMC + MATH-L5 concatenated (OOD stress suite).
  * ``gsm8k_math``    -> GSM8K + MATH mixed, stratified by MATH level 1-5.
  * ``/path.jsonl``   -> local JSONL with ``{"question":..., "answer":...}`` and
    optional ``"data_source"`` / ``"level"`` fields.

Each sample is stored as a dict with ``prompt`` (list of chat messages),
``ground_truth`` (string), ``data_source`` and ``level``.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from src.rewards import extract_last_boxed

_SYSTEM_PROMPT = (
    "Please solve the math problem step by step and put the final numeric "
    "answer inside \\boxed{}."
)

# Tiny offline fallback used by smoke tests so training runs with no dataset
# download and no HF authentication.
_SYNTHETIC = [
    ("John has 5 apples and buys 3 more. How many apples does he have?", "8", "synthetic", "1"),
    ("Compute 12 * 12.", "144", "synthetic", "1"),
    ("What is 7 + 6?", "13", "synthetic", "1"),
    ("Half of 50 is what number?", "25", "synthetic", "1"),
    ("A train travels 60 miles per hour for 2 hours. How far?", "120", "synthetic", "1"),
]


def _make_sample(question: str, answer: str, data_source: str, level) -> dict:
    gt = _extract_gt(answer)
    return {
        "prompt": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ground_truth": gt,
        "data_source": data_source,
        "level": level,
    }


def _extract_gt(answer: str) -> str:
    """Normalize a ground-truth string into a pairable numeric/LaTeX token."""
    text = answer.strip()
    m = re.search(r"####\s*(.+)$", text)
    if m:
        text = m.group(1).strip()
    return text


def build_synthetic(n: int) -> List[dict]:
    samples = [_make_sample(q, a, ds, lv) for q, a, ds, lv in _SYNTHETIC]
    # repeat to satisfy requested size if needed
    while len(samples) < n:
        samples.extend([_make_sample(q, a, ds, lv) for q, a, ds, lv in _SYNTHETIC])
    return samples[:n]


def _load_gsm8k(split: str = "train") -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for ex in ds:
        out.append(_make_sample(ex["question"], ex["answer"], "gsm8k", 1))
    return out


_MATH_CONFIGS = ("algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra",
                 "precalculus")


def _load_math(split: str = "train", levels: Optional[set] = None) -> List[dict]:
    """Hendrycks MATH, via the maintained ``EleutherAI/hendrycks_math`` mirror
    (the original ``hendrycks/competition_math`` repo is no longer accessible).

    ``level`` is a string like ``"Level 5"``; only its trailing digit is kept
    so it matches the ``{"4", "5"}`` filter used for the hard-math mix.  The
    ground truth lives inside ``\\boxed{...}`` in the full solution text, so it
    must be extracted; passing the whole solution would make every answer
    unmatchable.
    """
    from datasets import load_dataset
    out = []
    for cfg_name in _MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg_name, split=split)
        for ex in ds:
            lv = re.sub(r"[^0-9]", "", str(ex.get("level", "1"))) or "1"
            if levels is not None and lv not in levels:
                continue
            gt = extract_last_boxed(ex["solution"]) or _extract_gt(ex["solution"])
            out.append(_make_sample(ex["problem"], gt, "math", lv))
    return out


# ---------------------------------------------------------------------------
# AI-MO olympiad validation sets — the reference problems behind the Kaggle
# AIMO / AIME / AMC competitions.  These are competition-grade: far harder than
# GSM8K, used here as the out-of-domain stress test for a policy trained on
# MATH levels 4-5.
# ---------------------------------------------------------------------------
_AIMO_DATASETS = {
    "aime":    ("AI-MO/aimo-validation-aime",         "aime"),
    "amc":     ("AI-MO/aimo-validation-amc",          "amc"),
    "math_l5": ("AI-MO/aimo-validation-math-level-5", "math_l5"),
}


def _fmt_answer(value) -> str:
    """Normalize a raw answer field (str / int / float) into a clean token."""
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _load_aimo(key: str) -> List[dict]:
    """Load one AI-MO validation set.

    Prefers the explicit ``answer`` field; AMC has one (a float), AIME/MATH-L5
    have string answers.  Falls back to ``\\boxed{}`` extraction from
    ``solution`` if no usable ``answer`` field exists.
    """
    from datasets import load_dataset
    repo, source = _AIMO_DATASETS[key]
    ds = load_dataset(repo, split="train")
    out = []
    for ex in ds:
        ans = ex.get("answer")
        if ans is not None and str(ans).strip() != "":
            gt = _fmt_answer(ans)
        else:
            gt = extract_last_boxed(ex.get("solution", "")) or ""
        if gt == "":
            continue
        out.append(_make_sample(ex["problem"], gt, source, "5"))
    return out


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(_make_sample(
                rec["question"], rec["answer"],
                rec.get("data_source", "local"), rec.get("level", "1")))
    return out


def build_dataset(dataset: str, num_samples: int, split: str = "train",
                  seed: int = 42, levels: Optional[set] = None) -> List[dict]:
    """Build and shuffle a training list, truncated to ``num_samples``."""
    if dataset == "synthetic":
        samples = build_synthetic(num_samples)
    elif dataset == "gsm8k":
        samples = _load_gsm8k(split)
    elif dataset == "math":
        samples = _load_math(split, levels)
    elif dataset == "math_l45":
        # competition-grade training mix: MATH level 4-5 only
        samples = _load_math(split, {"4", "5"})
    elif dataset in _AIMO_DATASETS:
        samples = _load_aimo(dataset)
    elif dataset == "olympiad":
        # AIME + AMC + MATH-L5: the out-of-domain stress-test suite
        samples = (_load_aimo("aime") + _load_aimo("amc")
                   + _load_aimo("math_l5"))
    elif dataset in ("gsm8k_math", "mix"):
        gsm = _load_gsm8k(split)
        math = _load_math(split, levels)
        samples = gsm + math
    elif os.path.isfile(dataset):
        samples = _load_jsonl(dataset)
    else:
        raise ValueError(f"Unknown dataset spec: {dataset!r}")

    import random
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:num_samples]
