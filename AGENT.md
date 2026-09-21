# Agent Guide

This repo is an academic benchmark for comparing RAG paradigms on one fixed cybersecurity corpus. The authoritative design is `docs/IMPLEMENTATION_PLAN.md`; the current implementation details and pinned model names are in `config.yaml` and `.github/copilot-instructions.md`.

## Project In One Minute

The project evaluates multiple retrieval-augmented generation systems over a frozen corpus about CVE-2024-3400 and related edge-device vulnerabilities. Each system answers the same dev/test questions, writes resumable JSON results, then local and LLM judges compute correctness, retrieval, faithfulness, latency, and efficiency metrics.

Core rule: all systems share the same corpus, chunks, generation prompt, LLM budget, result schema, and output format. Differences should come from the RAG paradigm, not accidental plumbing changes.

## Current Runtime

- Python 3.11.
- Package code lives under `src/ragc`.
- Tests use `pytest` with `pythonpath = src`.
- Main LLM is configured in `config.yaml` as Gemini `gemini-3.5-flash-lite`.
- Judge LLM is configured in `config.yaml` as Groq `openai/gpt-oss-20b`.
- Never hard-code model ids, paths, chunk sizes, thresholds, or budgets outside `config.yaml` unless a test fixture explicitly needs it.
- Real API keys go in `.env`; do not print or commit them.

## Important Terminology

- **Corpus**: fixed source documents under `data/raw/`, processed into `data/processed/docs.jsonl` and `data/processed/chunks.jsonl`.
- **Doc**: parser output for one source document. It has `doc_id`, `source_type`, `title`, `url`, `date`, `cve_ids`, and Markdown-like `text`.
- **Chunk**: retrieval unit with id like `nvd_CVE-2024-3400#000`. Defined in `ragc.schemas.Chunk`.
- **Hit**: retrieval result with `chunk_id`, `score`, `rank`, and `via`.
- **EvidenceItem**: exact evidence shown to final generation. Citations must use chunk ids.
- **SystemResult**: saved answer payload for every system/question run.
- **S-1**: no-retrieval baseline.
- **S0**: naive dense retrieve-and-read.
- **S1**: pre-retrieval rewrite + HyDE + RRF.
- **S2**: post-retrieval hybrid + rerank + filter + compression.
- **S3**: decomposition plus S2-style post-retrieval.
- **S4**: LLM-centric gate, notes, critique, revision.
- **S5**: single-agent search loop.
- **S6a**: acquisition-driven multi-agent.
- **S6b**: utilization-driven multi-agent.
- **S6c**: hybrid dynamic multi-agent with source routing.
- **A1, A2a, A2b**: ablations for S6c routing and S2 post-retrieval steps.
- **FakeLLM**: deterministic local LLM stub for tests and dry runs.
- **CallBudget**: per-question LLM call limit, usually 8 calls with 1 reserved for final generation.
- **RRF**: Reciprocal Rank Fusion, used to merge ranked dense/BM25 lists.
- **KEV / NVD / ATT&CK**: CISA Known Exploited Vulnerabilities, NIST vulnerability data, and MITRE ATT&CK technique documents.

## Folder Map

- `.github/copilot-instructions.md`: high-priority project notes for AI agents. It currently overrides older plan model names.
- `docs/`: implementation plan and architecture diagrams. Read `docs/IMPLEMENTATION_PLAN.md` before changing benchmark behavior.
- `config.yaml`: central runtime config for models, retrieval parameters, paths, budget, chunking, and thresholds.
- `corpus_manifest.yaml`: source list for corpus fetching.
- `data/raw/`: frozen downloaded source files.
- `data/processed/`: generated `docs.jsonl` and `chunks.jsonl`.
- `data/index/`: local Chroma and BM25 indexes.
- `data/questions/questions.jsonl`: frozen dev/test question catalog.
- `src/ragc/config.py`: YAML loader.
- `src/ragc/schemas.py`: Pydantic data contracts shared across the pipeline.
- `src/ragc/llm/`: LLM clients, cache, key pool, quota handling, FakeLLM.
- `src/ragc/corpus/`: fetchers, parsers, chunker, index builder.
- `src/ragc/retrieval/`: dense, BM25, hybrid, reranker, retriever facade.
- `src/ragc/generation/`: fixed prompts, evidence rendering, answer generation, citation parsing.
- `src/ragc/systems/`: implementations of S-1 through S6c and ablations.
- `src/ragc/eval/`: question validation, local metrics, judge, aggregation, plots.
- `scripts/`: CLI entry points for corpus build, model checks, runs, judging, local eval, reports.
- `ui/` and `ui_server.py`: static RAG Explorer UI served by FastAPI.
- `tests/`: task-card tests and regression tests.
- `cache/`: LLM cache and key-pool state. Runtime artifact; do not delete unless asked.
- `results/`: system outputs, judged rows, tables, figures. Runs are resumable; do not delete unless asked.

## Setup Commands

Create and activate a venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create local env file:

```bash
cp .env.example .env
```

Fill `.env` with `GEMINI_API_KEY1`, `GROQ_API_KEY1`, or plain `GEMINI_API_KEY` / `GROQ_API_KEY`.

## Common Commands

Run all tests:

```bash
pytest
```

Run a focused test:

```bash
pytest tests/test_t08_retrieval.py
```

Check available provider models:

```bash
python scripts/check_models.py --provider both
```

Pin the newest available Gemini flash-lite model:

```bash
python scripts/check_models.py --provider gemini --pin
```

List corpus manifest targets:

```bash
python scripts/fetch_corpus.py --dry-run
```

Check vendor/threat-intel URLs:

```bash
python scripts/fetch_corpus.py --check-urls
```

Fetch, parse, chunk, index, and build questions:

```bash
python scripts/fetch_corpus.py
```

Rebuild all corpus artifacts and indexes:

```bash
python scripts/build_all.py --force
```

Run one system on a few dev questions with FakeLLM:

```bash
python scripts/run_system.py --system S0 --split dev --limit 3 --fake
```

Run one real system on selected test questions:

```bash
python scripts/run_system.py --system S2 --split test --qids t01,t02
```

Run the 24-question subset when quota is tight:

```bash
python scripts/run_system.py --system S6c --split test --subset test24
```

Run pointwise judging:

```bash
python scripts/run_judge.py --system S0 --split test
```

Compute deterministic local metrics:

```bash
python scripts/run_eval_local.py --system S0 --split test --dummy-nli
```

Build report tables:

```bash
python scripts/make_report.py
```

Start the UI:

```bash
PYTHONPATH=src uvicorn ui_server:app --reload
```

Open `http://127.0.0.1:8000/`. The UI uses `FakeLLM` and existing chunks when available; otherwise it falls back to demo chunks.

## Implementation Guardrails

- Preserve `System.run(qid, question) -> SystemResult` for every system.
- Register new or renamed systems in `src/ragc/systems/registry.py`.
- Keep final generation through `ragc.generation.generator.generate`.
- Keep LLM calls inside `ragc.llm` clients or `System.call_json`; do not call provider SDKs from systems.
- Use `FakeLLM` in tests before real provider calls.
- Keep runs resumable: scripts skip existing result files and append judge outputs.
- Respect `CallBudget`; systems should stop agent loops and proceed to final generation when budget is tight.
- Preserve citation format `[chunk_id]`, where ids look like `doc_id#000`.
- Treat `data/questions/questions.jsonl` test rows as frozen benchmark data.
- Do not casually change corpus chunking, retrieval thresholds, or judge prompts; those affect comparability.
- Do not delete `cache/` or `results/` unless the user explicitly asks.
- Be careful with the dirty worktree. Assume existing modifications are user work and avoid reverting them.

## Development Notes

The retrieval layer is designed to keep tests runnable without heavy downloads. `TextEmbedder` can use sentence-transformers when `RAGC_USE_ST_EMBEDDINGS=1`, but falls back to stable hashed embeddings. `Reranker` tries to load the configured transformer model and falls back to a lexical dummy scorer if loading fails.

For corpus changes, regenerate in order: raw/processed corpus, chunks, indexes, lock, questions validation. For system behavior changes, add or update focused tests under `tests/` and run at least the affected test plus `pytest` when feasible.

