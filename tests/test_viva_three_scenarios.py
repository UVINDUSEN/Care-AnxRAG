"""Three viva demonstrations through the real RAG answer pipeline.

Synthetic evidence tests the engineering contract only. They are not live
611-chunk corpus results and are not a clinical accuracy benchmark.
"""
from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.grounding import ClaimGroundingVerifier
from care_anxrag.models import EvidenceLevel, SafetyLevel
from care_anxrag.nli import HeuristicNliClassifier
from care_anxrag.rag import CareAnxRag
from test_answer_selection import StaticRetriever, hit, retrieval


COMPARISON = (
    "What are the differences between psychological and pharmacological "
    "treatment approaches for anxiety disorders?"
)
PSYCH = (
    "Psychological interventions are essential treatments for anxiety disorders "
    "and refer primarily to talk therapy with professionals or supervised lay therapists."
)
MED = (
    "Antidepressant medications, such as selective serotonin reuptake inhibitors "
    "(SSRIs), can also be useful in treating adults with anxiety disorders."
)


def _rag(settings, question, hits):
    return CareAnxRag(
        settings,
        StaticRetriever(retrieval(question, hits)),
        EvidenceOnlyGenerator(),
        ClaimGroundingVerifier(HeuristicNliClassifier()),
    )


def test_supported_clinical_comparison_cites_both_verbatim_who_statements(settings):
    evidence = hit(
        "There are several effective treatments for anxiety disorders. "
        + PSYCH + " " + MED, title="Anxiety disorders",
    )
    evidence.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    evidence.chunk.source_name = "WHO synthetic fixture (not live WHO retrieval)"
    evidence.chunk.url = "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders"

    result = _rag(settings, COMPARISON, [evidence]).answer(COMPARISON)
    assert result.abstained is False
    assert result.safety_level == SafetyLevel.NORMAL
    assert result.abstention_reason is None
    assert result.answer == f"- {PSYCH} [S1]\\n- {MED} [S1]".replace("\\n", "\n")
    assert len(result.citations) == 1
    assert result.citations[0].citation_id == "S1"
    assert result.citations[0].chunk_id == evidence.chunk.chunk_id
    assert result.citations[0].source_name == evidence.chunk.source_name
    assert result.citations[0].url == evidence.chunk.url


def test_unsupported_universal_cure_claim_abstains_with_no_citations(settings):
    question = "Does CBT permanently cure social anxiety disorder in every patient?"
    # The one available passage does not establish permanence or universality.
    irrelevant = hit(
        "Social anxiety disorder can interfere with work and relationships.",
        title="Social anxiety disorder",
    )
    result = _rag(settings, question, [irrelevant]).answer(question)
    assert result.abstained is True
    assert result.abstention_reason == "insufficient_answer_evidence"
    assert result.safety_level == SafetyLevel.NORMAL
    assert result.citations == []
    assert "permanently cure" not in result.answer.lower()


def test_unrelated_question_is_rejected_before_evidence_retrieval(runtime):
    question = "How do I repair a refrigerator compressor?"
    result = runtime.rag.answer(question)
    assert result.abstained is True
    assert result.abstention_reason == "outside_anxiety_scope"
    assert result.safety_level == SafetyLevel.NORMAL
    assert result.citations == []
    assert "compressor" not in result.answer.lower()


def test_synthetic_self_harm_routes_to_crisis_without_medical_citations(runtime):
    result = runtime.rag.answer("i have thoughts about kiling my self")
    assert result.abstained is True
    assert result.safety_level == SafetyLevel.CRISIS
    assert result.abstention_reason == "self_harm_or_suicide_signal"
    assert result.citations == []
