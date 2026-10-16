"""Synthetic source fixtures for answer selection; not clinical evidence or gold labels."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from care_anxrag.api import create_app
from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.grounding import ClaimGroundingVerifier
from care_anxrag.models import (
    ChunkRecord,
    DocumentStatus,
    EvidenceLevel,
    KnowledgeLayer,
    RetrievalResult,
    SearchHit,
)
from care_anxrag.nli import HeuristicNliClassifier
from care_anxrag.query import QueryAnalyzer
from care_anxrag.rag import CareAnxRag


QUESTION = "What does the evidence say about cognitive behavioural therapy for anxiety?"


def hit(text: str, index: int = 1, *, title: str = "Synthetic evidence") -> SearchHit:
    return SearchHit(
        chunk=ChunkRecord(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            version_id=f"version-{index}",
            source_id=f"source-{index}",
            source_name=f"Synthetic source {index}",
            title=title,
            layer=KnowledgeLayer.CLINICAL_CORE,
            status=DocumentStatus.ACTIVE,
            section_path="evidence",
            section_heading="Evidence",
            ordinal=0,
            text=text,
            text_hash="synthetic",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            authority_score=0.9,
            evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
            evidence_score=0.9,
            metadata={"synthetic": True},
        ),
        care_score=0.9,
        relevance_score=0.9,
    )


def retrieval(question: str, hits: list[SearchHit], **overrides) -> RetrievalResult:
    return RetrievalResult(
        query_analysis=QueryAnalyzer().analyze(question),
        hits=hits,
        confidence=0.9,
        **overrides,
    )


class StaticRetriever:
    """Supply a fixed retrieval result to exercise real presentation and grounding."""

    def __init__(self, result: RetrievalResult):
        self.result = result

    def retrieve(self, question: str) -> RetrievalResult:
        return self.result


def rag_for(settings, result: RetrievalResult) -> CareAnxRag:
    return CareAnxRag(
        settings,
        StaticRetriever(result),
        EvidenceOnlyGenerator(),
        ClaimGroundingVerifier(HeuristicNliClassifier()),
    )


def test_findings_are_selected_instead_of_more_lexically_similar_objectives():
    objective = (
        "The purpose of this study was to investigate the effectiveness of cognitive "
        "behavioural therapy for anxiety."
    )
    finding = "CBT reduced anxiety symptoms compared with the control condition."
    hits = [hit(f"{objective} {finding}")]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"
    assert payload.cited_source_ids == ["S1"]
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(payload, hits).supported


@pytest.mark.parametrize("text", [
    "The aim was to determine whether CBT reduced anxiety symptoms.",
    "Patients with anxiety were randomly assigned to CBT or a control condition.",
    "Internet-based cognitive behavioral therapy programs are offered with varying therapist support.",
    "We hypothesize that CBT will improve anxiety symptoms.",
    "This study evaluated CBT effectiveness for anxiety using symptom scores.",
    "CBT effectiveness for anxiety was assessed using symptom questionnaires.",
])
def test_objectives_methods_and_delivery_background_are_not_effectiveness_answers(text):
    hits = [hit(text)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_scans_remaining_context_and_keeps_original_citation_ids(settings):
    hits = [hit("The purpose was to study CBT for anxiety.", i) for i in range(1, 4)]
    finding = "CBT can reduce anxiety symptoms."
    hits.append(hit(finding, 4))
    result = rag_for(settings, retrieval(QUESTION, hits)).answer(QUESTION)

    assert not result.abstained
    assert result.answer == f"- {finding} [S4]"
    assert [(c.citation_id, c.chunk_id) for c in result.citations] == [("S4", "chunk-4")]


def test_duplicate_source_sentences_are_not_repeated():
    finding = "CBT can reduce anxiety symptoms."
    hits = [hit(finding, 1), hit(finding, 2)]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"
    assert payload.cited_source_ids == ["S1"]


@pytest.mark.parametrize("finding", [
    "CBT did not reduce anxiety symptoms compared with the control condition.",
    "The evidence for CBT effectiveness in anxiety remains uncertain.",
    "CBT had no effect on anxiety symptoms.",
    "CBT reduced anxiety scores by 2.5 points, but the estimate was imprecise.",
])
def test_negative_and_uncertain_findings_remain_verbatim(finding):
    hits = [hit(finding)]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"


def test_population_qualification_is_kept_in_the_selected_sentence():
    finding = "CBT reduced anxiety symptoms in adolescents in this pilot trial."
    hits = [hit(finding)]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert finding in payload.answer


@pytest.mark.parametrize("title,text", [
    ("CBT for adolescents", "CBT reduced anxiety symptoms."),
    ("Synthetic evidence", "Participants were adolescents. CBT reduced anxiety symptoms."),
])
def test_population_scope_cannot_be_lost_when_extracting_a_finding(title, text):
    hits = [hit(text, title=title)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_adolescent_finding_does_not_answer_an_explicit_adult_question(settings):
    question = "Does CBT reduce anxiety in adults?"
    hits = [hit("CBT reduced anxiety symptoms in adolescents in this pilot trial.")]
    result = rag_for(settings, retrieval(question, hits)).answer(question)

    assert result.abstained
    assert result.abstention_reason == "insufficient_answer_evidence"
    assert result.citations == []


def test_unrelated_finding_cannot_borrow_treatment_context_from_an_objective():
    hits = [hit(
        "The aim was to study CBT for anxiety. Medication reduced anxiety symptoms."
    )]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_population_scope_from_ingestion_facets_is_preserved():
    evidence = hit("CBT reduced anxiety symptoms.")
    evidence.chunk.metadata["clinical_facets"] = {"populations": ["children_and_adolescents"]}
    hits = [evidence]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_unrecognized_population_metadata_does_not_crash():
    finding = "CBT can reduce anxiety symptoms."
    evidence = hit(finding)
    evidence.chunk.metadata["clinical_facets"] = {"populations": [{"unknown": "value"}]}
    hits = [evidence]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"


def test_findings_do_not_borrow_the_requested_outcome_from_an_objective(settings):
    question = "Does CBT improve quality of life for anxiety?"
    hits = [hit(
        "The aim was to assess CBT effects on quality of life for anxiety. "
        "CBT reduced anxiety symptoms."
    )]
    result = rag_for(settings, retrieval(question, hits)).answer(question)

    assert result.abstained
    assert result.abstention_reason == "insufficient_answer_evidence"


def test_general_information_queries_still_return_source_definitions():
    question = "What are anxiety disorders?"
    definition = "Anxiety disorders involve persistent fear or worry."
    hits = [hit(definition)]
    payload = EvidenceOnlyGenerator().generate(question, hits, retrieval(question, hits))

    assert payload.answer == f"- {definition} [S1]"


def test_descriptive_treatment_question_can_return_a_discussed_intervention():
    question = "What evidence-based psychological intervention is discussed for panic disorder?"
    discussion = "The fixture discusses cognitive behavioural therapy for panic disorder."
    hits = [hit(discussion)]
    payload = EvidenceOnlyGenerator().generate(question, hits, retrieval(question, hits))

    assert payload.answer == f"- {discussion} [S1]"


def test_helpful_question_cannot_be_answered_with_delivery_background():
    question = "Is CBT helpful for anxiety?"
    hits = [hit("Internet-based CBT programs for anxiety vary in therapist support.")]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, hits, retrieval(question, hits))


@pytest.mark.parametrize("finding", [
    "In this trial CBT did not reduce anxiety symptoms.",
    "CBT was not effective for anxiety in this trial.",
    "CBT effectiveness for anxiety remains uncertain.",
])
def test_actual_negative_result_is_preferred_over_generic_background(finding):
    background = "CBT is an effective treatment for anxiety."
    hits = [hit(f"{background} {finding}")]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"


@pytest.mark.parametrize("finding", [
    "CBT reduced anxiety symptoms vs. placebo, but the estimate was imprecise.",
    "CBT did not improve anxiety outcomes, e.g. symptom severity, in this trial.",
])
def test_abbreviations_do_not_strip_comparators_or_qualifiers(settings, finding):
    hits = [hit(finding)]
    result = rag_for(settings, retrieval(QUESTION, hits)).answer(QUESTION)

    assert not result.abstained
    assert result.answer == f"- {finding} [S1]"
    assert result.citations[0].chunk_id == "chunk-1"


def test_objectives_only_returns_specific_api_abstention(runtime):
    hits = [hit("The purpose was to study CBT for anxiety.")]
    runtime.rag = rag_for(runtime.settings, retrieval(QUESTION, hits))
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post("/v1/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["abstention_reason"] == "insufficient_answer_evidence"
    assert body["citations"] == []
    assert "findings" in body["answer"]
    assert "format" not in body["answer"]


def test_existing_unresolved_conflict_abstention_is_preserved(settings):
    hits = [hit("CBT can reduce anxiety symptoms.")]
    result = rag_for(settings, retrieval(
        QUESTION, hits, should_abstain=True,
        abstention_reason="unresolved_high_confidence_evidence_conflict",
    )).answer(QUESTION)

    assert result.abstained
    assert result.abstention_reason == "unresolved_high_confidence_evidence_conflict"
    assert result.citations == []


# ---------------------------------------------------------------------------
# Viva regressions: question-directed factuality vs merely topical citations.
# Synthetic engineering examples; no clinician-reviewed labels or outcomes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,irrelevant", [
    (
        "What treatments are supported for generalized anxiety disorder?",
        "Generalized anxiety disorder (GAD) is a prevalent anxiety disorder "
        "characterized by uncontrollable worry, muscle tension, and irritability.",
    ),
    (
        "What treatments are supported for generalized anxiety disorder?",
        "Generalized anxiety disorder (GAD) has important implications "
        "for patients and healthcare resources.",
    ),
    (
        "How is panic disorder diagnosed?",
        "Panic disorder occurs in 1-3% of adolescents and may persist.",
    ),
    (
        "How is panic disorder diagnosed?",
        "Cognitive behavioral therapy (CBT) is efficacious for panic disorder.",
    ),
    (
        "What evidence contradicts an intervention recommendation for GAD?",
        "However, there is a lack of research exploring online check-ins for GAD.",
    ),
    (
        "What evidence contradicts an intervention recommendation for GAD?",
        "CBT is an effective treatment for generalized anxiety disorder.",
    ),
])
def test_clinician_queries_do_not_return_topical_non_answers(question, irrelevant):
    rows = [hit(irrelevant)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


@pytest.mark.parametrize("question,statement", [
    (
        "What treatments are supported for generalized anxiety disorder?",
        "CBT was effective for adults with generalized anxiety disorder.",
    ),
    (
        "What does the evidence say about CBT for social anxiety disorder?",
        "CBT reduced symptom severity in adults with social anxiety disorder.",
    ),
    (
        "Compare exposure therapy and CBT for panic disorder.",
        "CBT and exposure therapy showed no significant difference for panic disorder.",
    ),
    (
        "What evidence contradicts an intervention recommendation for GAD?",
        "CBT did not reduce anxiety symptoms in adults with generalized anxiety disorder.",
    ),
    (
        "How is panic disorder diagnosed?",
        "Panic disorder diagnosis requires assessment of recurrent unexpected panic attacks.",
    ),
])
def test_clinician_questions_can_select_direct_source_sentences(question, statement):
    rows = [hit(statement)]
    payload = EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))
    assert payload.answer == f"- {statement} [S1]"
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(payload, rows).supported


def test_comparison_requires_comparative_outcome_not_two_treatment_mentions():
    question = "Compare exposure therapy and CBT for panic disorder."
    rows = [hit("CBT and exposure therapy are treatments for panic disorder.")]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_recent_review_requires_relevant_treatment_finding_and_recent_review_level():
    from datetime import UTC, datetime
    from care_anxrag.models import EvidenceLevel

    question = "What recent systematic reviews address anxiety-disorder treatment?"
    biomarker = hit(
        "This systematic review examined associations of inflammatory biomarkers "
        "and anxiety disorders."
    )
    biomarker.chunk.evidence_level = EvidenceLevel.SYSTEMATIC_REVIEW
    biomarker.chunk.published_at = datetime(2026, 7, 9, tzinfo=UTC)
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, [biomarker], retrieval(question, [biomarker]))

    therapy = hit(
        "CBT reduced symptom severity for adults with anxiety disorders "
        "in the included studies."
    )
    therapy.chunk.evidence_level = EvidenceLevel.SYSTEMATIC_REVIEW
    therapy.chunk.published_at = datetime(2026, 7, 9, tzinfo=UTC)
    generated = EvidenceOnlyGenerator().generate(question, [therapy], retrieval(question, [therapy]))
    assert generated.cited_source_ids == ["S1"]


def test_authoritative_medication_query_excludes_unrelated_secondary_disease_reviews():
    from care_anxrag.models import EvidenceLevel

    question = "What do authoritative sources say about medication for anxiety?"
    copd = hit("SSRIs may help anxiety symptoms in chronic obstructive pulmonary disease.")
    copd.chunk.evidence_level = EvidenceLevel.META_ANALYSIS
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, [copd], retrieval(question, [copd]))

    who = hit(
        "Antidepressant medications, such as selective serotonin reuptake "
        "inhibitors (SSRIs), can also be useful in treating adults with anxiety disorders."
    )
    who.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    generated = EvidenceOnlyGenerator().generate(question, [who], retrieval(question, [who]))
    assert generated.cited_source_ids == ["S1"]
