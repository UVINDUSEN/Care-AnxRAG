from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .coverage import audit_corpus_coverage
from .evaluation import BenchmarkItem, evaluate_ablation, load_benchmark
from .reproducibility import build_experiment_snapshot
from .runtime import Runtime


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_timings(runtime: Runtime, items: list[BenchmarkItem]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        answer = runtime.rag.answer(item.question, include_debug=True)
        rows.append(
            {
                "id": item.id,
                "answer_timings_ms": dict(answer.timings_ms),
                "retrieval_timings_ms": (
                    dict(answer.retrieval.timings_ms)
                    if answer.retrieval is not None
                    else {}
                ),
                "abstained": answer.abstained,
                "safety_level": answer.safety_level.value,
            }
        )

    stages = sorted(
        {
            stage
            for row in rows
            for timings_key in ("answer_timings_ms", "retrieval_timings_ms")
            for stage in row[timings_key]
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for stage in stages:
        values = [
            float(row[timings_key][stage])
            for row in rows
            for timings_key in ("answer_timings_ms", "retrieval_timings_ms")
            if stage in row[timings_key]
        ]
        if values:
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
            summary[stage] = {
                "count": float(len(values)),
                "mean_ms": sum(values) / len(values),
                "median_ms": ordered[len(ordered) // 2],
                "p95_ms": ordered[p95_index],
            }

    return {"count": len(rows), "per_item": rows, "stage_summary": summary}


def run_experiment_bundle(
    runtime: Runtime,
    benchmark_path: Path,
    output_dir: Path,
    *,
    code_revision: str,
) -> dict[str, Any]:
    """Run the reproducible CARE-AnxRAG research bundle.

    The target directory is immutable-by-default. A pre-existing non-empty
    directory is rejected so a completed research run cannot be silently
    overwritten.
    """
    benchmark_path = benchmark_path.resolve()
    output_dir = output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Experiment output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_benchmark(benchmark_path)
    ablation_reports = evaluate_ablation(
        runtime.retriever,
        runtime.rag,
        items,
    )
    ablation = {
        label: report.as_dict()
        for label, report in ablation_reports.items()
    }
    coverage = audit_corpus_coverage(runtime.database).as_dict()
    timings = _collect_timings(runtime, items)
    snapshot = build_experiment_snapshot(
        runtime,
        code_revision=code_revision,
        benchmark_path=benchmark_path,
    )

    payloads = {
        "ablation.json": ablation,
        "coverage.json": coverage,
        "timings.json": timings,
        "snapshot.json": snapshot,
    }
    for filename, payload in payloads.items():
        _write_json(output_dir / filename, payload)

    artifacts = {
        filename: {
            "sha256": _sha256_file(output_dir / filename),
            "size_bytes": (output_dir / filename).stat().st_size,
        }
        for filename in sorted(payloads)
    }
    manifest = {
        "bundle_version": 1,
        "code_revision": code_revision,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256_file(benchmark_path),
        "item_count": len(items),
        "ablation_modes": list(ablation),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest)

    return {
        "output_dir": str(output_dir),
        "manifest": manifest,
    }
