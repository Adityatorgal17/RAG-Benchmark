## Project Overview

This is a **KAG (Knowledge-Augmented Generation) benchmark** project for comparing RAG (Retrieval-Augmented Generation) paradigms on a fixed cybersecurity corpus about CVE-2024-3400 and related edge-device vulnerabilities.

**Key Concept**: Evaluate 10 different RAG systems + 3 ablation variants against the same questions to compare their performance systematically.

## Core Structure

### 📋 Primary Documentation

- **`docs/IMPLEMENTATION_PLAN.md`** - Authoritative implementation guide with detailed specifications for each system
- **`AGENT.md`** - Current implementation details and pinned model names
- **`config.yaml`** - Central configuration for models, retrieval parameters, and runtime settings
- **`.github/copilot-instructions.md`** - High-priority guidelines for AI agents

### 🏗️ System Architecture

**Package Structure**:

```
src/ragc/
├── systems/           # 10+ system implementations
├── llm/              # Groq/Gemini clients with caching
├── retrieval/        # Dense + BM25 + hybrid search
├── generation/       # Evidence formatting + citation parsing
├── eval/             # Judgment and metrics
├── schemas.py        # Pydantic data contracts
└── config.py         # YAML configuration loader
```

**The 10 Main RAG Systems** (all implementing `System.run(qid, question) -> SystemResult`):

1. **S-1**: No-retrieval baseline (LLM answers directly from knowledge)
2. **S0**: Naive RAG (dense retrieval + concatenation)
3. **S1**: Pre-retrieval query rewriting + HyDE + RRF fusion
4. **S2**: Post-retrieval system (hybrid + rerank + filter + compress)
5. **S3**: Query decomposition + post-retrieval refinement
6. **S4**: LLM-centric gate + Chain-of-Note + self-critique + revision
7. **S5**: Single-agent search loop with scratchpad reasoning
8. **S6a**: Acquisition-driven multi-agent (retrieval-decision + query + sufficiency)
9. **S6b**: Utilization-driven multi-agent (evaluator + note-taker + recon)
10. **S6c**: Hybrid dynamic multi-agent with source routing
11. **Plus 3 ablations**: A1 (routing disabled), A2a (filter+compress disabled), A2b (filter enabled, compress disabled)

### 🔧 Key Design Principles

**Strict Rules** (from copilot-instructions.md):

- All systems share identical corpus, chunks, generation prompts, LLM budget, and output format
- Differences must come from RAG paradigm design, not plumbing changes
- Every LLM call goes through `ragc.llm` clients (cache + rate limit + retries)
- Runs are resumable (skip existing result files)
- Retrieval, reranker, and NLI tests use the configured real local models; `FakeLLM` is only used where tests must avoid external Gemini/Groq API calls.
- Final generation goes through `ragc.generation.generator.generate`

**The Big Picture**:

- **Corpus**: Fixed cybersecurity documents about CVE-2024-3400
- **Questions**: 40 frozen benchmark questions in `data/questions/questions.jsonl`
- **Evaluation**: Multi-level judging (correctness, retrieval metrics, faithfulness, latency, efficiency)
- **Goal**: Academic comparison with numbers to support the survey paper "Knowledge intensive agents"

### 🛠️ Runtime Environment

**Configuration** (`config.yaml`):

```yaml
main_llm:
  provider: groq
  model: openai/gpt-oss-20b
  temperature: 0
  budget: {max_llm_calls: 8, reserve_final: 1}
judge_llm:
  provider: gemini
  model: gemini-3.5-flash-lite
embedding:
  model: BAAI/bge-small-en-v1.5
reranker:
  model: BAAI/bge-reranker-base
```

Dense retrieval, in-memory retrieval tests, reranking, and NLI faithfulness intentionally
require the configured transformer models. `embeddings.py` uses `sentence-transformers`
for `BAAI/bge-small-en-v1.5`, and `reranker.py` / `nli_faith.py` use `transformers`;
there is no hashed, dummy, lightweight, or lexical model fallback for benchmark behavior.

Retrieval and agent-control values are configured in `config.yaml`, not inside the system
implementations. Tune values such as `retrieval.s0_k`, `retrieval.rrf_k`,
`retrieval.dense_overfetch_multiplier`, each system's `retrieval.s*` settings,
`post.tau`, `post.compress_tokens`, `post.min_keep`, `context.max_tokens`, and
`budget.max_llm_calls` there when you need different benchmark settings.

**Common Commands**:

```bash
# Run all tests
pytest

# Test a specific system without spending external LLM quota
python scripts/run_system.py --system S0 --split dev --limit 3 --fake

# Real test run (with API keys)
python scripts/run_system.py --system S2 --split test --qids t01,t02
```

### 📊 Expected Output

Each system generates JSON results with:

- **Answer**: Final generated response
- **Abstention**: Whether answer starts with "INSUFFICIENT_EVIDENCE"
- **Citations**: Chunk IDs used (`[chunk_id]` format)
- **Evidence**: Top evidence shown to generator
- **Usage**: Token counts, latency, call metrics
- **Trace**: Agent steps and LLM interactions

### 🎯 Mission

This is an **academic prototype** that provides:

1. **Reproducible experiments** for comparing RAG paradigms
2. **Standardized evaluation** across 10+ approaches
3. **Rich metrics** beyond just accuracy (latency, efficiency, faithfulness)
4. ** extensible framework** for adding new RAG patterns

The project implements the empirical comparison promised in the "Knowledge intensive agents" survey paper, providing the first experimental validation of these RAG paradigms on a real cybersecurity corpus.

**Key takeaway**: This isn't a typical application - it's a **research infrastructure** for studying how different RAG approaches perform in a controlled, comparable environment.
