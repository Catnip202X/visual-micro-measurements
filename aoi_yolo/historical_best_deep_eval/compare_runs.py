from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def cliffs_delta(run1: list[float], run2: list[float]) -> float | None:
    if not run1 or not run2:
        return None
    left = np.asarray(run1, dtype=float)[:, None]
    right = np.asarray(run2, dtype=float)[None, :]
    comparisons = left - right
    return float(
        (np.count_nonzero(comparisons > 0) - np.count_nonzero(comparisons < 0))
        / comparisons.size
    )


def bootstrap_differences(
    run1: list[float],
    run2: list[float],
    *,
    samples: int = 10_000,
    seed: int = 20260721,
) -> dict[str, float]:
    first = np.asarray(run1, dtype=float)
    second = np.asarray(run2, dtype=float)
    if not len(first) or not len(second):
        raise ValueError("bootstrap requires non-empty independent runs")
    generator = np.random.default_rng(seed)
    mean_differences = np.empty(samples, dtype=float)
    median_differences = np.empty(samples, dtype=float)
    for index in range(samples):
        sample1 = generator.choice(first, size=len(first), replace=True)
        sample2 = generator.choice(second, size=len(second), replace=True)
        mean_differences[index] = np.mean(sample1) - np.mean(sample2)
        median_differences[index] = np.median(sample1) - np.median(sample2)
    return {
        "mean_difference": float(np.mean(first) - np.mean(second)),
        "mean_ci95_low": float(np.percentile(mean_differences, 2.5)),
        "mean_ci95_high": float(np.percentile(mean_differences, 97.5)),
        "median_difference": float(np.median(first) - np.median(second)),
        "median_ci95_low": float(np.percentile(median_differences, 2.5)),
        "median_ci95_high": float(np.percentile(median_differences, 97.5)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _contingency_test(first: Counter, second: Counter) -> dict:
    labels = sorted(set(first) | set(second))
    table = np.asarray([[first[label] for label in labels], [second[label] for label in labels]], dtype=float)
    keep = np.sum(table, axis=0) > 0
    table = table[:, keep]
    labels = [label for label, selected in zip(labels, keep) if selected]
    if table.shape[1] < 2 or np.any(np.sum(table, axis=1) == 0):
        return {"test": "chi_square", "statistic": None, "p_value": None, "degrees_of_freedom": None, "categories": labels}
    statistic, p_value, degrees, _ = stats.chi2_contingency(table)
    return {
        "test": "chi_square",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees),
        "categories": labels,
        "run_1_counts": [int(value) for value in table[0]],
        "run_2_counts": [int(value) for value in table[1]],
    }


def _two_proportion(success1: int, total1: int, success2: int, total2: int) -> dict:
    if total1 == 0 or total2 == 0:
        return {"difference": None, "ci95_low": None, "ci95_high": None, "z": None, "p_value": None}
    rate1, rate2 = success1 / total1, success2 / total2
    difference = rate1 - rate2
    standard_error_ci = math.sqrt(rate1 * (1 - rate1) / total1 + rate2 * (1 - rate2) / total2)
    pooled = (success1 + success2) / (total1 + total2)
    standard_error_null = math.sqrt(pooled * (1 - pooled) * (1 / total1 + 1 / total2))
    if standard_error_null == 0:
        z_value, p_value = 0.0, 1.0
    else:
        z_value = difference / standard_error_null
        p_value = 2.0 * stats.norm.sf(abs(z_value))
    return {
        "run_1_rate": rate1,
        "run_2_rate": rate2,
        "difference": difference,
        "ci95_low": difference - 1.959963984540054 * standard_error_ci,
        "ci95_high": difference + 1.959963984540054 * standard_error_ci,
        "z": float(z_value),
        "p_value": float(p_value),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def compare_runs(
    run1_timing: list[dict],
    run2_timing: list[dict],
    run1_score: dict,
    run2_score: dict,
    output_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260721,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    values1 = [float(row["total_production_ms"]) for row in run1_timing]
    values2 = [float(row["total_production_ms"]) for row in run2_timing]
    mann_whitney = stats.mannwhitneyu(values1, values2, alternative="two-sided")
    kolmogorov = stats.ks_2samp(values1, values2, alternative="two-sided", method="auto")
    timing = {
        "run_1_images": len(values1),
        "run_2_images": len(values2),
        "mann_whitney_u": float(mann_whitney.statistic),
        "mann_whitney_p": float(mann_whitney.pvalue),
        "ks_statistic": float(kolmogorov.statistic),
        "ks_p": float(kolmogorov.pvalue),
        "cliffs_delta": cliffs_delta(values1, values2),
        **bootstrap_differences(values1, values2, samples=bootstrap_samples, seed=seed),
    }
    bracket_test = _contingency_test(
        Counter(row["timing_bracket"] for row in run1_timing),
        Counter(row["timing_bracket"] for row in run2_timing),
    )
    composition_test = _contingency_test(
        Counter(row["context"] for row in run1_timing),
        Counter(row["context"] for row in run2_timing),
    )

    metric_tests: list[dict] = []
    for label in ("pinhole", "scratch"):
        first = run1_score["by_class"][label]
        second = run2_score["by_class"][label]
        for metric, denominator_terms in (("recall", ("tp", "fn")), ("precision", ("tp", "fp"))):
            total1 = sum(int(first[term]) for term in denominator_terms)
            total2 = sum(int(second[term]) for term in denominator_terms)
            test = _two_proportion(int(first["tp"]), total1, int(second["tp"]), total2)
            metric_tests.append({
                "label": label,
                "metric": metric,
                "run_1_successes": int(first["tp"]),
                "run_1_trials": total1,
                "run_2_successes": int(second["tp"]),
                "run_2_trials": total2,
                **test,
            })
    raw_p_values = [row["p_value"] if row["p_value"] is not None else 1.0 for row in metric_tests]
    for row, adjusted in zip(metric_tests, holm_adjust(raw_p_values)):
        row["holm_p"] = adjusted

    result = {
        "timing": timing,
        "bracket_distribution": bracket_test,
        "composition_distribution": composition_test,
        "per_defect_metric_tests": metric_tests,
        "interpretation_rule": (
            "Assess composition before attributing differences to model instability; "
            "Holm-adjusted per-defect tests control family-wise error across recall and precision comparisons."
        ),
    }
    _write_csv(output_dir / "per_defect_metric_tests.csv", metric_tests)
    (output_dir / "between_run_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
