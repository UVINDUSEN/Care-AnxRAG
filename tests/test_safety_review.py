from __future__ import annotations

import csv
from pathlib import Path

import pytest

from care_anxrag.models import SafetyLevel
from care_anxrag.safety_evaluation import load_safety_benchmark
from care_anxrag.safety_review import (
    compile_adjudicated_safety_benchmark,
    validate_safety_splits,
    write_safety_review_package,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_safety_review_package_keeps_gold_labels_blank(tmp_path: Path) -> None:
    output = tmp_path / "review"

    manifest = write_safety_review_package(output)

    development = _read_rows(output / "development.csv")
    test_rows = _read_rows(output / "test.csv")

    assert manifest["status"] == "awaiting_human_review"
    assert development
    assert test_rows == []
    assert all(row["final_expected_level"] == "" for row in development)
    assert all(row["reviewer_1_id"] == "" for row in development)
    assert all(row["reviewer_2_id"] == "" for row in development)
    assert all(row["adjudicated"] == "" for row in development)


def test_safety_compiler_rejects_missing_independent_reviewers(
    tmp_path: Path,
) -> None:
    package = tmp_path / "review"
    write_safety_review_package(package)
    rows = _read_rows(package / "development.csv")
    rows[0]["final_expected_level"] = "normal"
    rows[0]["reviewer_1_id"] = "reviewer-a"
    rows[0]["reviewer_2_id"] = ""
    rows[0]["adjudicated"] = "true"
    adjudicated = tmp_path / "adjudicated.csv"
    _write_rows(adjudicated, [rows[0]])

    with pytest.raises(ValueError, match="two distinct reviewer IDs"):
        compile_adjudicated_safety_benchmark(
            adjudicated,
            tmp_path / "compiled.jsonl",
        )


def test_safety_compiler_writes_reviewed_jsonl(tmp_path: Path) -> None:
    package = tmp_path / "review"
    write_safety_review_package(package)
    rows = _read_rows(package / "development.csv")
    row = rows[0]
    row["final_expected_level"] = "crisis"
    row["reviewer_1_id"] = "reviewer-a"
    row["reviewer_2_id"] = "reviewer-b"
    row["adjudicated"] = "true"
    adjudicated = tmp_path / "adjudicated.csv"
    _write_rows(adjudicated, [row])
    output = tmp_path / "compiled.jsonl"

    compiled = compile_adjudicated_safety_benchmark(adjudicated, output)

    assert len(compiled) == 1
    assert compiled[0].expected_level == SafetyLevel.CRISIS
    assert compiled[0].annotator_ids == ["reviewer-a", "reviewer-b"]
    assert compiled[0].adjudicated is True
    loaded = load_safety_benchmark(output)
    assert loaded == compiled


def test_safety_split_validator_rejects_repeated_prompt(
    tmp_path: Path,
) -> None:
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    dev.write_text(
        '{"id":"dev-1","text":"Same prompt","expected_level":"normal",'
        '"stratum":"normal","split":"development",'
        '"annotator_ids":["a","b"],"adjudicated":true}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"id":"test-1","text":" same   PROMPT ","expected_level":"normal",'
        '"stratum":"normal","split":"test",'
        '"annotator_ids":["c","d"],"adjudicated":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Repeated safety benchmark text"):
        validate_safety_splits(dev, test)
