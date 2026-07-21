from __future__ import annotations

from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .cache import ImagePrediction
from .dataset import ImageRecord


TIMING_FIELDS = (
    "model_preprocess_ms",
    "inference_ms",
    "model_postprocess_ms",
    "conversion_cascade_ms",
    "total_production_ms",
)


def defect_context(counts: dict[str, int]) -> str:
    pinhole = counts.get("pinhole", 0) > 0
    scratch = counts.get("scratch", 0) > 0
    if pinhole and scratch:
        return "mixed"
    if pinhole:
        return "pinhole-only"
    if scratch:
        return "scratch-only"
    return "no-target"


def _edge_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def bracket_labels(edges: tuple[float, ...]) -> list[str]:
    labels: list[str] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if math.isinf(upper):
            labels.append(f">={_edge_text(lower)} ms")
        elif lower == 0:
            labels.append(f"<{_edge_text(upper)} ms")
        else:
            labels.append(f"{_edge_text(lower)}-{_edge_text(upper)} ms")
    return labels


def timing_bracket(value: float, edges: tuple[float, ...]) -> str:
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if lower <= value < upper:
            return bracket_labels(edges)[index]
    raise ValueError(f"timing value {value} lies outside configured brackets")


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {key: None for key in ("mean", "sd", "min", "p50", "p95", "p99", "max")} | {"n": 0}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _correlation(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.ptp(np.asarray(x, dtype=float)) == 0 or np.ptp(np.asarray(y, dtype=float)) == 0:
        return None, None
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _plot_timing(enriched: list[dict], brackets: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(
        [row["index"] for row in enriched],
        [row["total_production_ms"] for row in enriched],
        linewidth=1.0,
        color="#1f77b4",
    )
    axes[0].set_title("Production-path processing time by image")
    axes[0].set_xlabel("Measured image order")
    axes[0].set_ylabel("Milliseconds")
    axes[0].grid(alpha=0.25)
    axes[1].bar(
        [row["bracket"] for row in brackets],
        [row["count"] for row in brackets],
        color="#ff7f0e",
    )
    axes[1].set_title("Images per fixed processing-time bracket")
    axes[1].set_xlabel("Production processing time")
    axes[1].set_ylabel("Image count")
    axes[1].tick_params(axis="x", rotation=25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def analyze_timing(
    cached_rows: list[ImagePrediction],
    manifest: list[ImageRecord],
    edges: tuple[float, ...],
    output_dir: Path,
) -> dict:
    if [row.stem for row in cached_rows] != [row.stem for row in manifest]:
        raise ValueError("timing cache does not exactly match manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched: list[dict] = []
    for cached, record in zip(cached_rows, manifest):
        row = asdict(cached.timing)
        target_counts = record.target_object_counts
        row.update({
            "context": defect_context(target_counts),
            "pinhole_objects": int(target_counts.get("pinhole", 0)),
            "scratch_objects": int(target_counts.get("scratch", 0)),
            "target_objects": int(sum(target_counts.values())),
            "pinhole_present": int(target_counts.get("pinhole", 0) > 0),
            "scratch_present": int(target_counts.get("scratch", 0) > 0),
            "pixel_count": int(record.width * record.height),
            "timing_bracket": timing_bracket(cached.timing.total_production_ms, edges),
        })
        enriched.append(row)

    summaries = [
        {"timing_field": field, **_summary(row[field] for row in enriched)}
        for field in TIMING_FIELDS
    ]
    labels = bracket_labels(edges)
    brackets = []
    for label in labels:
        count = sum(row["timing_bracket"] == label for row in enriched)
        brackets.append({
            "bracket": label,
            "count": count,
            "percent": 100.0 * count / len(enriched) if enriched else 0.0,
        })
    context_order = ("pinhole-only", "scratch-only", "mixed", "no-target")
    by_context = []
    for context in context_order:
        values = [row["total_production_ms"] for row in enriched if row["context"] == context]
        if values:
            by_context.append({"context": context, **_summary(values)})

    correlation_variables = (
        "pinhole_present", "scratch_present", "pinhole_objects", "scratch_objects",
        "target_objects", "raw_predictions", "kept_predictions", "width", "height", "pixel_count",
    )
    total = [row["total_production_ms"] for row in enriched]
    correlations = []
    for variable in correlation_variables:
        coefficient, p_value = _correlation([float(row[variable]) for row in enriched], total)
        correlations.append({
            "variable": variable,
            "spearman_rho": coefficient,
            "p_value": p_value,
            "n": len(enriched),
        })
    context_samples = [
        [row["total_production_ms"] for row in enriched if row["context"] == context]
        for context in context_order
    ]
    context_samples = [sample for sample in context_samples if sample]
    if len(context_samples) >= 2:
        kruskal = stats.kruskal(*context_samples)
        context_test = {
            "test": "kruskal_wallis",
            "statistic": float(kruskal.statistic),
            "p_value": float(kruskal.pvalue),
            "groups": len(context_samples),
        }
    else:
        context_test = {"test": "kruskal_wallis", "statistic": None, "p_value": None, "groups": len(context_samples)}

    _write_csv(output_dir / "timing_enriched.csv", enriched)
    _write_csv(output_dir / "timing_summary.csv", summaries)
    _write_csv(output_dir / "timing_brackets.csv", brackets)
    _write_csv(output_dir / "timing_by_context.csv", by_context)
    _write_csv(output_dir / "timing_correlations.csv", correlations)
    _plot_timing(enriched, brackets, output_dir / "timing_graph.png")
    result = {
        "images": len(enriched),
        "summaries": summaries,
        "brackets": brackets,
        "by_context": by_context,
        "correlations": correlations,
        "context_distribution_test": context_test,
    }
    (output_dir / "timing_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
