from __future__ import annotations

from care_anxrag.component_evaluation import (
    EmbeddingBenchmarkItem,
    NliBenchmarkItem,
    RerankerBenchmarkItem,
    evaluate_embedding_model,
    evaluate_nli_model,
    evaluate_reranker,
)
from care_anxrag.models import RelationLabel


class StubEmbedder:
    model_id = "stub-embedding"

    def embed(self, texts):
        vectors = {
            "query-good": [1.0, 0.0],
            "positive-good": [1.0, 0.0],
            "negative-good": [0.0, 1.0],
            "query-rank2": [1.0, 0.0],
            "positive-rank2": [0.6, 0.8],
            "negative-rank2": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


class StubReranker:
    def score(self, query, hits):
        return [
            1.0 if "relevant" in hit.chunk.text else 0.1
            for hit in hits
        ]


class StubNli:
    def classify_text_pairs(self, pairs):
        outputs = []
        for premise, hypothesis in pairs:
            if "supports" in premise and "supports" in hypothesis:
                outputs.append((RelationLabel.ENTAILMENT, 0.95))
            elif "supports" in premise and "not supported" in hypothesis:
                outputs.append((RelationLabel.CONTRADICTION, 0.90))
            else:
                outputs.append((RelationLabel.NEUTRAL, 0.80))
        return outputs


def test_embedding_evaluation_reports_top1_and_mrr() -> None:
    report = evaluate_embedding_model(
        StubEmbedder(),
        [
            EmbeddingBenchmarkItem(
                id="e1",
                query="query-good",
                positive_text="positive-good",
                negative_texts=["negative-good"],
            ),
            EmbeddingBenchmarkItem(
                id="e2",
                query="query-rank2",
                positive_text="positive-rank2",
                negative_texts=["negative-rank2"],
            ),
        ],
    )

    assert report.count == 2
    assert report.top1_accuracy == 0.5
    assert report.mrr == 0.75
    assert report.per_item[0]["positive_rank"] == 1
    assert report.per_item[1]["positive_rank"] == 2


def test_reranker_evaluation_reports_positive_rank() -> None:
    report = evaluate_reranker(
        StubReranker(),
        [
            RerankerBenchmarkItem(
                id="r1",
                query="panic treatment",
                positive_text="relevant panic treatment evidence",
                negative_texts=[
                    "unrelated genetics evidence",
                    "another unrelated document",
                ],
            )
        ],
    )

    assert report.count == 1
    assert report.top1_accuracy == 1.0
    assert report.mrr == 1.0
    assert report.per_item[0]["positive_rank"] == 1


def test_nli_evaluation_reports_confusion_matrix() -> None:
    report = evaluate_nli_model(
        StubNli(),
        [
            NliBenchmarkItem(
                id="n1",
                premise="The study supports CBT.",
                hypothesis="The study supports CBT.",
                expected_label=RelationLabel.ENTAILMENT,
            ),
            NliBenchmarkItem(
                id="n2",
                premise="The study supports CBT.",
                hypothesis="The treatment is not supported.",
                expected_label=RelationLabel.CONTRADICTION,
            ),
            NliBenchmarkItem(
                id="n3",
                premise="Different population.",
                hypothesis="CBT improves symptoms.",
                expected_label=RelationLabel.NEUTRAL,
            ),
        ],
    )

    assert report.count == 3
    assert report.accuracy == 1.0
    assert report.confusion_matrix["entailment"]["entailment"] == 1
    assert report.confusion_matrix["contradiction"]["contradiction"] == 1
    assert report.confusion_matrix["neutral"]["neutral"] == 1


def test_component_benchmarks_preserve_review_metadata() -> None:
    item = NliBenchmarkItem(
        id="reviewed",
        premise="premise",
        hypothesis="hypothesis",
        expected_label=RelationLabel.NEUTRAL,
        stratum="different_follow_up",
        split="test",
        annotator_ids=["reviewer-a", "reviewer-b"],
        adjudicated=True,
    )

    assert item.stratum == "different_follow_up"
    assert item.split == "test"
    assert item.adjudicated is True
