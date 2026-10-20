"""Unit tests for the opt-in clinician-facing summary; not clinical validation."""

import json

from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.grounding import ClaimGroundingVerifier
from care_anxrag.models import RelationLabel
from care_anxrag.ollama_summarizer import OllamaClinicalSummarizer
from care_anxrag.rag import CareAnxRag
from care_anxrag.runtime import _build_generator
from test_answer_selection import hit, retrieval


QUESTION = "What evidence supports CBT for anxiety?"
SOURCE = "Cognitive behavioral therapy (CBT) can reduce anxiety symptoms."
SUMMARY = "Cognitive behavioral therapy may reduce symptoms of anxiety."


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": json.dumps(self.data)}}


class StubNli:
    def __init__(self, label=RelationLabel.ENTAILMENT, probability=0.96):
        self.label = label
        self.probability = probability
        self.pairs = []

    def classify_text_pairs(self, pairs):
        self.pairs.extend(pairs)
        return [(self.label, self.probability) for _ in pairs]


def _mock_reply(monkeypatch, sentence, source_id="S1"):
    def fake_post(url, **kwargs):
        assert url.endswith("/api/chat")
        assert kwargs["json"]["model"] == "gemma3:4b"
        return FakeResponse(
            {"claims": [{"source_id": source_id, "sentence": sentence}]}
        )
    monkeypatch.setattr("care_anxrag.ollama_summarizer.httpx.post", fake_post)


def test_natural_summary_keeps_citation_and_preselected_evidence(monkeypatch):
    _mock_reply(monkeypatch, SUMMARY)
    hits = [hit(SOURCE)]
    payload = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b").generate(
        QUESTION, hits, retrieval(QUESTION, hits)
    )
    assert payload.answer.startswith("Research summary\n\n")
    assert SUMMARY.rstrip(".") + " [S1]." in payload.answer
    assert "[S1]" in payload.answer
    assert payload.cited_source_ids == ["S1"]
    assert payload.source_evidence == {"S1": SOURCE}

    nli = StubNli()
    verifier = ClaimGroundingVerifier(
        nli, threshold=0.65, allow_entailed_paraphrases=True
    )
    assert verifier.verify(payload, hits).supported
    assert len(nli.pairs) == 1


def test_legacy_exact_source_verifier_stays_strict(monkeypatch):
    _mock_reply(monkeypatch, SUMMARY)
    hits = [hit(SOURCE)]
    payload = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b").generate(
        QUESTION, hits, retrieval(QUESTION, hits)
    )
    assert not ClaimGroundingVerifier(StubNli()).verify(payload, hits).supported


def test_unrelated_claim_fails_conservative_guard_before_nli(monkeypatch):
    _mock_reply(monkeypatch, "CBT may prevent all future panic attacks.")
    hits = [hit(SOURCE)]
    payload = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b").generate(
        QUESTION, hits, retrieval(QUESTION, hits)
    )
    # Panic-disorder content is not in the original anxiety source statement.
    # The generator itself must reject the candidate and fall back unchanged.
    assert payload.answer == "- " + SOURCE + " [S1]"
    assert payload.source_evidence == {}


def test_new_number_and_missing_negation_are_rejected_before_grounding(monkeypatch):
    assert not OllamaClinicalSummarizer._conservative(
        "Cognitive behavioral therapy reduces symptoms by 90%.",
        SOURCE,
    )
    assert not OllamaClinicalSummarizer._conservative(
        "Digital CBT significantly improves panic disorder outcomes.",
        "Digital CBT did not significantly improve panic disorder outcomes.",
    )
    assert not OllamaClinicalSummarizer._conservative(
        "CBT cures anxiety disorder.",
        SOURCE,
    )


def test_malicious_source_ids_fail_closed_to_extractive(monkeypatch):
    _mock_reply(monkeypatch, "CBT prevents all anxiety.", "S999")
    hits = [hit(SOURCE)]
    payload = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b").generate(
        QUESTION, hits, retrieval(QUESTION, hits)
    )
    expected = EvidenceOnlyGenerator().generate(
        QUESTION, hits, retrieval(QUESTION, hits)
    )
    assert payload.answer == expected.answer
    assert "S999" not in payload.answer


def test_failed_semantic_grounding_falls_back_in_rag(monkeypatch, settings):
    _mock_reply(monkeypatch, SUMMARY)
    settings.generator_provider = "ollama_summary"
    hits = [hit(SOURCE)]
    class StaticRetriever:
        def retrieve(self, question):
            return retrieval(question, hits)

    rag = CareAnxRag(
        settings,
        StaticRetriever(),
        OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b"),
        ClaimGroundingVerifier(
            StubNli(RelationLabel.NEUTRAL, 0.99),
            allow_entailed_paraphrases=True,
        ),
    )
    answer = rag.answer(QUESTION)
    assert not answer.abstained
    assert answer.answer == "- " + SOURCE + " [S1]"
    assert answer.citations[0].citation_id == "S1"


def test_existing_ollama_evidence_mode_unchanged(settings):
    from care_anxrag.ollama_organizer import OllamaEvidenceOrganizer
    settings.generator_provider = "ollama_evidence"
    assert isinstance(_build_generator(settings), OllamaEvidenceOrganizer)
    settings.generator_provider = "ollama_summary"
    assert isinstance(_build_generator(settings), OllamaClinicalSummarizer)
