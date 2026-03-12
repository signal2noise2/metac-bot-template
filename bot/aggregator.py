from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Any

from bot.state import PredictionSample


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-6), 1 - 1e-6)


def _logit(p: float) -> float:
    p = _clip_probability(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def summarize_prediction(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, (int, float)):
        p = float(prediction)
        return {
            "prediction_type": "binary_probability",
            "method": "single_prediction",
            "count": 1,
            "probabilities": [p],
            "aggregate_probability": p,
            "median_probability": p,
            "mean_probability": p,
            "min_probability": p,
            "max_probability": p,
            "spread": 0.0,
            "stdev": 0.0,
            "trim_fraction": 0.0,
        }

    return {
        "prediction_type": "structured_or_nonbinary",
        "method": "single_prediction",
        "count": 1,
        "value": prediction,
    }


def aggregate_prediction_samples(
    samples: list[PredictionSample],
    *,
    method: str = "trimmed_mean_logit",
    trim_fraction: float = 0.2,
) -> dict[str, Any]:
    if not samples:
        return {
            "prediction_type": "binary_probability",
            "method": method,
            "count": 0,
            "probabilities": [],
            "aggregate_probability": None,
            "median_probability": None,
            "mean_probability": None,
            "min_probability": None,
            "max_probability": None,
            "spread": None,
            "stdev": None,
            "trim_fraction": trim_fraction,
        }

    probs = sorted(_clip_probability(sample.prediction_value) for sample in samples)
    count = len(probs)
    med = median(probs)
    avg = mean(probs)
    min_p = probs[0]
    max_p = probs[-1]
    spread = max_p - min_p
    stdev = pstdev(probs) if count > 1 else 0.0

    trim_n = int(count * max(0.0, min(trim_fraction, 0.49)))
    trimmed_probs = probs[trim_n : count - trim_n] if count - 2 * trim_n > 0 else probs

    if method == "mean":
        aggregate = mean(probs)
    elif method == "median":
        aggregate = med
    elif method == "trimmed_mean":
        aggregate = mean(trimmed_probs)
    elif method == "median_logit":
        aggregate = _sigmoid(median([_logit(p) for p in probs]))
    else:
        logits = [_logit(p) for p in trimmed_probs]
        aggregate = _sigmoid(mean(logits))

    return {
        "prediction_type": "binary_probability",
        "method": method,
        "count": count,
        "probabilities": [round(p, 6) for p in probs],
        "aggregate_probability": round(float(aggregate), 6),
        "median_probability": round(float(med), 6),
        "mean_probability": round(float(avg), 6),
        "min_probability": round(float(min_p), 6),
        "max_probability": round(float(max_p), 6),
        "spread": round(float(spread), 6),
        "stdev": round(float(stdev), 6),
        "trim_fraction": trim_fraction,
    }