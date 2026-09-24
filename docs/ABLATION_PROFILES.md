# Retrieval ablation profiles

CARE-AnxRAG uses `care_full` by default. The other profiles are research-only ablations for measuring the contribution of retrieval and evidence-assessment stages while keeping the same frozen corpus and extractive answer renderer.

| Profile | Dense | BM25 | RRF | CrossEncoder | CARE score | NLI conflict | Relevance gate | Calibrated abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `b0_dense` | yes | no | single-list rank | no | no | no | no | no |
| `b1_lexical` | no | yes | single-list rank | no | no | no | no | no |
| `b2_hybrid_rrf` | yes | yes | yes | no | no | no | no | no |
| `b3_hybrid_rerank` | yes | yes | yes | yes | no | no | no | no |
| `b4_care` | yes | yes | yes | yes | yes | no | no | no |
| `b5_conflict` | yes | yes | yes | yes | yes | yes | no | no |
| `care_full` | yes | yes | yes | yes | yes | yes | yes | yes |

Safety routing remains enabled for every profile. Active-version filtering also remains enabled because withdrawn/superseded evidence must never become answerable merely for an ablation experiment.

## Usage

Run the same frozen benchmark once per profile:

```bash
CARE_RETRIEVAL_PROFILE=b0_dense care-anxrag evaluate data/benchmark/anxiety-dev.jsonl --project-root .
CARE_RETRIEVAL_PROFILE=b1_lexical care-anxrag evaluate data/benchmark/anxiety-dev.jsonl --project-root .
CARE_RETRIEVAL_PROFILE=b2_hybrid_rrf care-anxrag evaluate data/benchmark/anxiety-dev.jsonl --project-root .
CARE_RETRIEVAL_PROFILE=b3_hybrid_rerank care-anxrag evaluate data/benchmark/anxiety-dev.jsonl --project-root .
CARE_RETRIEVAL_PROFILE=b4_care care-anxrag evaluate data/benchmark/anxiety-dev.jsonl --project-root .
CARE_RETRIEVAL_PROFILE=b5_conflict care-anxrag evaluate data/benchmark/anxiety-dev.jsonl --project-root .
CARE_RETRIEVAL_PROFILE=care_full care-anxrag evaluate data/benchmark/anxiety-dev.jsonl --project-root .
```

Tune thresholds only on the development split. Freeze the selected configuration before running the locked test split.

## Interpretation

The non-full profiles intentionally omit the production relevance/sufficiency/abstention guardrails described in the stage matrix. They exist only to measure incremental contribution. Do not deploy them as the clinical-facing configuration.
