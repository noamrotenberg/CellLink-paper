#!/usr/bin/env python3
"""Aggregate bootstrap metric samples from multiple evaluation files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_metrics(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    try:
        with path.open(encoding="utf-8") as input_file:
            values = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON metrics file {path}: {error}") from error

    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    if not all(isinstance(sample, dict) for sample in values):
        raise ValueError(f"Every sample in {path} must be a JSON object.")

    metric_names = tuple(sorted(values[0]))
    if not metric_names:
        raise ValueError(f"Samples in {path} must contain at least one metric.")
    for sample_index, sample in enumerate(values):
        if tuple(sorted(sample)) != metric_names:
            raise ValueError(
                f"Sample {sample_index} in {path} does not have metrics {metric_names}."
            )
        for metric_name in metric_names:
            value = sample[metric_name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Sample {sample_index} metric {metric_name!r} in {path} "
                    "must be numeric."
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Sample {sample_index} metric {metric_name!r} in {path} "
                    "must be finite and non-negative."
                )
    return values, metric_names


def aggregate(paths: list[Path]) -> tuple[np.ndarray, tuple[str, ...]]:
    if not paths:
        raise ValueError("At least one metrics file is required.")

    loaded = [load_metrics(path) for path in paths]
    sample_count = len(loaded[0][0])
    metric_names = loaded[0][1]
    for path, (samples, names) in zip(paths[1:], loaded[1:]):
        if len(samples) != sample_count:
            raise ValueError(
                f"{path} has {len(samples)} samples; expected {sample_count}."
            )
        if names != metric_names:
            raise ValueError(
                f"{path} has metrics {names}; expected {metric_names}."
            )

    # Shape: files x samples x metrics. Harmonic means are calculated across
    # the file axis, preserving one aggregate value per sample and metric.
    values = np.asarray(
        [[[sample[name] for name in metric_names] for sample in samples]
         for samples, _ in loaded],
        dtype=float,
    )
    reciprocal_sum = np.sum(np.divide(1.0, values, out=np.zeros_like(values), where=values != 0), axis=0)
    zero_count = np.count_nonzero(values == 0, axis=0)
    harmonic_means = np.divide(
        len(paths),
        reciprocal_sum,
        out=np.zeros_like(reciprocal_sum),
        where=(zero_count == 0) & (reciprocal_sum != 0),
    )
    return harmonic_means, metric_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_files", nargs="+", type=Path,
                        help="JSON sample metrics files to aggregate.")
    args = parser.parse_args()

    try:
        sample_metrics, metric_names = aggregate(args.metrics_files)
    except ValueError as error:
        parser.error(str(error))

    lower, upper = np.percentile(sample_metrics, (2.5, 97.5), axis=0)
    print(f"Aggregated {sample_metrics.shape[0]} samples across {len(args.metrics_files)} files.")
    for index, metric_name in enumerate(metric_names):
        print(
            f"{metric_name}: 95% CI = [{lower[index]:.6f}, {upper[index]:.6f}]"
        )


if __name__ == "__main__":
    main()
