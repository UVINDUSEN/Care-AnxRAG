from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Protocol, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field

from .embeddings import Embedder
from .models import RelationLabel, SearchHit
from .nli import NliClassifier
from .rerank import Reranker
from .util import cosine_similarity


class _ReviewedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    stratum: str = "unspecified"
    split: str = "unassigned"
    annotator_ids: list[str] = Field(default_factory=list)
    adjudicated: bool = False


class EmbeddingBenchmarkItem(_ReviewedItem):
    query: str
    positive_text: str
    negative_texts: list[str] = Field(default_factory=list)


class RerankerBenchmarkItem(_ReviewedItem):
    query: str
    positive_text: str
    negative_texts: list[str] = Field(default_factory=list)


class NliBenchmarkItem(_ReviewedItem):
    premise: str
    hypothesis: str
    expected_label: RelationLabel


@dataclass(slots=True)
class RankingEvaluationReport:
    count: int
    top1_accuracy: float
    mrr: float
    per_item: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "top1_accuracy": self.top1_accuracy,
            "mrr": self.mrr,
            "per_item": self.per_item,
        }


@dataclass(slots=True)
class NliEvaluationReport:
    count: int
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]
    per_item: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
            "per_item": self.per_item,
        }


def _positive_rank(scores: Sequence[float]) -> int:
    if not scores:
        raise ValueError("At least one candidate score is required")
    ranked = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )
    return ranked.index(0) + 1


def _ranking_report(
    rows: list[dict[str, Any]],
) -> RankingEvaluationReport:
    count = len(rows)
    top1 = sum(row["positive_rank"] == 1 for row in rows)
    reciprocal_ranks = [
        1.0 / int(row["positive_rank"])
        for row in rows
    ]
    return RankingEvaluationReport(
        count=count,
        top1_accuracy=top1 / count if count else 0.0,
        mrr=(
            sum(reciprocal_ranks) / count
            if count
            else 0.0
        ),
        per_item=rows,
    )


def evaluate_embedding_model(
    embedder: Embedder,
    items: Iterable[EmbeddingBenchmarkItem],
) -> RankingEvaluationReport:
    rows: list[dict[str, Any]] = []

    for item in items:
        candidates = [
            item.positive_text,
            *item.negative_texts,
        ]
        vectors = embedder.embed(
            [item.query, *candidates]
        )
        query_vector = vectors[0]
        scores = [
            cosine_similarity(
                query_vector,
                candidate_vector,
            )
            for candidate_vector in vectors[1:]
        ]
        rank = _positive_rank(scores)
        rows.append(
            {
                "id": item.id,
                "positive_rank": rank,
                "positive_score": float(scores[0]),
                "candidate_count": len(candidates),
                "stratum": item.stratum,
                "split": item.split,
            }
        )

    return _ranking_report(rows)


def _text_hits(
    texts: Sequence[str],
) -> list[SearchHit]:
    return [
        cast(
            SearchHit,
            SimpleNamespace(
                chunk=SimpleNamespace(text=text)
            ),
        )
        for text in texts
    ]


def evaluate_reranker(
    reranker: Reranker,
    items: Iterable[RerankerBenchmarkItem],
) -> RankingEvaluationReport:
    rows: list[dict[str, Any]] = []

    for item in items:
        candidates = [
            item.positive_text,
            *item.negative_texts,
        ]
        scores = list(
            reranker.score(
                item.query,
                _text_hits(candidates),
            )
        )
        if len(scores) != len(candidates):
            raise RuntimeError(
                "Reranker returned an unexpected number of scores: "
                f"expected {len(candidates)}, received {len(scores)}"
            )
        rank = _positive_rank(scores)
        rows.append(
            {
                "id": item.id,
                "positive_rank": rank,
                "positive_score": float(scores[0]),
                "candidate_count": len(candidates),
                "stratum": item.stratum,
                "split": item.split,
            }
        )

    return _ranking_report(rows)


def evaluate_nli_model(
    nli: NliClassifier,
    items: Iterable[NliBenchmarkItem],
) -> NliEvaluationReport:
    benchmark = list(items)
    pairs = [
        (item.premise, item.hypothesis)
        for item in benchmark
    ]
    predictions = nli.classify_text_pairs(pairs)

    if len(predictions) != len(benchmark):
        raise RuntimeError(
            "NLI classifier returned an unexpected number of predictions: "
            f"expected {len(benchmark)}, received {len(predictions)}"
        )

    labels = [
        RelationLabel.CONTRADICTION,
        RelationLabel.ENTAILMENT,
        RelationLabel.NEUTRAL,
    ]
    confusion = {
        expected.value: {
            predicted.value: 0
            for predicted in labels
        }
        for expected in labels
    }

    rows: list[dict[str, Any]] = []
    for item, (predicted, confidence) in zip(
        benchmark,
        predictions,
        strict=True,
    ):
        confusion[item.expected_label.value][predicted.value] += 1
        rows.append(
            {
                "id": item.id,
                "expected_label": item.expected_label.value,
                "predicted_label": predicted.value,
                "confidence": float(confidence),
                "correct": predicted == item.expected_label,
                "stratum": item.stratum,
                "split": item.split,
            }
        )

    count = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return NliEvaluationReport(
        count=count,
        accuracy=correct / count if count else 0.0,
        confusion_matrix=confusion,
        per_item=rows,
    )
