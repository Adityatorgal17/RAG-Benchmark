# CLAUDE.md 

---

## Project Identity

**Name**: KAG (Knowledge-Augmented Generation) Benchmark
**Purpose**: Empirical comparison of 13 RAG paradigms on a fixed cybersecurity corpus
**Paper Foundation**: "Knowledge Intensive Agents" (Liu et al., AI Open 7, 2026)
**Scope**: Training-free, prompt-based implementations (not trained models)
**Language**: Python 3.11, plain scripts, no frameworks
**Status**: All 13 systems implemented, tested, and benchmarking in progress

---

## Fixed Decisions (DO NOT CHANGE)

### Models

- **Main LLM**: Groq `openai/gpt-oss-20b` (was Gemini in plan, changed in config.yaml)
- **Judge LLM**: Groq `llama-3.3-70b-versatile` (evaluation only)
- **Embeddings**: BAAI/bge-small-en-v1.5 (384-d)
- **Reranker**: BAAI/bge-reranker-base (cross-encoder)
- **Vector DB**: ChromaDB (persistent, local, cosine)
- **Sparse**: BM25Okapi with custom CVE tokenizer
- **Fusion**: Reciprocal Rank Fusion (k=60)

### Data & Chunking

- **Corpus**: CVE-2024-3400 and related vulnerability documents
- **Chunk Size**: 180 words, 30-word overlap, 25-word minimum (~250 tokens)
- **Context Cap**: 2,000 tokens max evidence per generation
- **Questions**: 40 (10 dev, 30 test)
- **Call Budget**: 8 LLM calls per question, 1 reserved for final generation

### Output Format

- **Generation Prompt**: Fixed across all systems (Appendix A, P-GEN)
- **Result Fields**: system, qid, question, answer, retrieval_results, latency_s, llm_calls, errors
- **Storage**: `results/raw/{system}/{qid}.json` per-run, `results/benchmark.jsonl` aggregated

---

## System Specifications

### 13 Systems Overview

- **S-1**: No-retrieval baseline (knowledge only)
- **S0**: Naive RAG (dense top-5, concat, generate)
- **S1**: Pre-retrieval (query rewrite + HyDE + RRF)
- **S2**: Post-retrieval (hybrid + rerank + filter + compress)
- **S3**: Pre+Post (decompose + rerank + compress)
- **S4**: LLM-centric (gate + chain-of-note + critique + revise)
- **S5**: Single-agent (scratchpad reasoning with search tool)
- **S6a**: Multi-agent acquisition (decision + query + sufficiency)
- **S6b**: Multi-agent utilization (evaluator + note-taker + generator)
- **S6c**: Multi-agent hybrid (planner + source-router + refiner + decider)
- **A1**: Ablation of S6c (routing disabled)
- **A2a**: Ablation of S2 (rerank only, no compression)
- **A2b**: Ablation of S2 (no compression)

Each system implements `System.run(qid, question) -> SystemResult`

### System Class Locations

```python
src/ragc/systems/
├── base.py              # System(ABC) base class
├── s0_naive.py          # S0NaiveSystem
├── s1_pre_retrieval.py  # S1PreRetrievalSystem
├── s2_post_retrieval.py # S2PostRetrievalSystem
├── s3_decomp.py         # S3DecompositionSystem
├── s4_llm_centric.py    # S4LLMCentricSystem
├── s5_single_agent.py   # S5SingleAgentSystem
├── s6a_acquisition.py   # S6aAcquisitionSystem
├── s6b_utilization.py   # S6bUtilizationSystem
├── s6c_hybrid.py        # S6cHybridSystem
├── registry.py          # System factory
└── ablations.py         # A1, A2a, A2b
```

---

## Key Implementation Facts

### All LLM Calls Go Through Clients

- **File**: `src/ragc/llm/clients.py`
- **Classes**: GroqClient, ProviderLLM
- **Features**: Caching, rate limiting, token counting, error handling
- **Never**: Call Groq SDK directly; always use client
- **Cache**: SQLite at `cache/llm_cache.sqlite` (prompt-based dedup)

### API Key Management

- **File**: `src/ragc/llm/keypool.py`
- **Class**: KeyPool
- **Keys**: 6 independent accounts (GROQ_API_KEY, GROQ_API_KEY1-5 in .env)
- **Rate Limit**: 25 RPM per key, 150 RPM total (free tier)
- **Fallback**: `run_with_fallback()` cycles through keys on QuotaExhausted
- **Blacklist**: 24-hour exhaustion blacklist in `cache/keypool_state.json`
- **Rate Limiting in Benchmark**: Added delays to prevent simultaneous quota exhaustion
  - Per-question delay: 0.1-1.0s (scaled by LLM calls)
  - Between-system delay: 3.0s (keypool cooldown)
  - Error delay: 1.0s (prevent thrashing)

### Retrieval Architecture

- **Dense**: Query → bge-small embeddings → ChromaDB cosine search
- **Sparse**: Query → BM25 tokenization → Okapi BM25 ranking
- **Hybrid**: RRF fusion of dense + sparse with k=60
- **Reranking**: Cross-encoder for top-k documents (systems S2, S3, S6c)
- **Source Filtering**: Metadata filter for source-aware retrieval (S6c)

### Evaluation Flow

- **Per-Run**: Token F1, recall@5, citation count, errors
- **Aggregation**: `scripts/benchmark_all_systems.py` collects to `benchmark.jsonl`
- **Summary**: Metrics aggregated per-system in `benchmark_summary_fixed.json`
- **Faithfulness**: NLI check with local cross-encoder (post-analysis)

---

## Recent Fixes & Discoveries

### Issue: API Quota Exhaustion (Sept 26-27)

**Symptoms**:

- S4, S5, S6a, S6b, S6c, A1 failed with "All configured keys are exhausted"
- Only 483/520 benchmark runs completed
- Simple systems (S0-S3, A2a, A2b) succeeded

**Root Cause**:

- Not code bugs (all systems proven working with FakeLLM + real API)
- Infrastructure constraint: 1,200 API calls in ~10 minutes exceeded 150 RPM capacity
- All 6 keys hit quota simultaneously → benchmark crashed

**Solution Implemented** (in `scripts/benchmark_all_systems.py`):

```python
import time  # Added

# Per-question rate limiting (0.3s per LLM call)
call_delay = max(0.1, min(1.0, llm_calls * 0.3))
time.sleep(call_delay)

# Between-system cooldown (3.0s)
if system_idx > 0:
    time.sleep(3.0)

# Error recovery delay (1.0s)
except Exception:
    time.sleep(1.0)
```

**Effect**: Spreads 1,200 calls over 45-60 minutes instead of 10 minutes → stays under quota

**Verification**: All 6 systems tested individually with real API → all work

---

## Testing Strategy

### Unit Tests (FakeLLM - No API Calls)

- **File**: `tests/test_t11_system.py`
- **Purpose**: Validate system logic before API testing
- **Speed**: <1 minute for all 13 systems
- **Coverage**: All systems + all system types (retrieval, agent, etc.)
- **Result**: All 13 systems pass FakeLLM tests

### Integration Tests (Real API, Limited Scale)

- **Command**: `python3 scripts/run_system.py --system S5 --split dev --limit 1`
- **Purpose**: Validate system works with real API
- **Scale**: 1-5 questions per system
- **Cost**: ~10-50 API calls per system

### Full Benchmark (All Systems, All Questions)

- **Command**: `python3 scripts/benchmark_all_systems.py --num-questions 40`
- **Purpose**: Generate complete metrics for all 13 systems
- **Scale**: 13 systems × 40 questions = 520 runs
- **Cost**: ~1,200 API calls total
- **Duration**: 45-60 minutes with rate limiting
- **Output**: `results/benchmark.jsonl`, `results/benchmark_summary_fixed.json`

---

## Common Tasks & Patterns

### Test a Single System

```python
# In your code:
from src.ragc.systems.registry import get_system_factory
from src.ragc.llm.clients import GroqClient
from src.ragc.retrieval.retriever import HybridRetriever

# Initialize
client = GroqClient()
retriever = HybridRetriever()
factory = get_system_factory("S5")
system = factory(cfg, client, retriever, None)

# Run
result = system.run("t01", "What is CVE-2024-3400?")
print(result)
```

### Add a New System

1. Create `src/ragc/systems/s_mynew.py`
2. Inherit from `System` base class
3. Implement `run(qid, question) -> SystemResult`
4. Register in `systems/registry.py`
5. Add test in `tests/test_t11_system.py`
6. Run: `pytest tests/test_t11_system.py::test_mynew -v`

### Debug a Failed Run

1. Check `results/raw/{system}/{qid}.json` for error message
2. Test system in isolation: `python3 scripts/run_system.py --system S5 --qid t01`
3. If API error: Check keypool state, API quota
4. If logic error: Run with FakeLLM first, then with real API

### Run Benchmark Efficiently

```bash
# Fast validation (10 questions, ~20 minutes)
rm -rf results/raw cache/keypool_state.json
python3 scripts/benchmark_all_systems.py --num-questions 10

# Full benchmark (40 questions, ~60 minutes)
rm -rf results/raw cache/keypool_state.json
python3 scripts/benchmark_all_systems.py --num-questions 40
```

---

## Code Quality Standards

### LLM Clients

- ✅ All calls through `ProviderLLM.generate()`
- ✅ JSON mode explicitly requested for structured outputs
- ✅ Token count tracked
- ✅ Caching enabled (dedupe by prompt)
- ✅ Rate limiting enforced

### System Implementation

- ✅ All inherit from `System` base class
- ✅ Implement `run(qid, question) -> SystemResult`
- ✅ Use `self.call_json()` for LLM calls (enforces budget)
- ✅ Track errors in `self.errors` list
- ✅ Return valid `SystemResult` even on partial failure

### Results Storage

- ✅ Save to `results/raw/{system}/{qid}.json`
- ✅ Check if exists before overwriting (resumable)
- ✅ Include all metadata (system, qid, question, answer, etc.)
- ✅ Never delete `results/` or `cache/` without asking

### Tests

- ✅ Use FakeLLM before real API
- ✅ Test happy path, error paths, edge cases
- ✅ Compare against baseline if applicable

---

## Known Constraints

1. **No Training**: All implementations are prompt-based (frozen LLM)
2. **No Async**: Sequential execution only (simplicity, easier debugging)
3. **No Web Frameworks**: Plain Python scripts (except UI server which is simple Flask-like)
4. **One Corpus**: All 520 chunks from CVE-2024-3400 + related
5. **One Generation Prompt**: Same P-GEN prompt for all systems
6. **API Quota**: Free tier limits (150 RPM total, 6 keys)
7. **No GPU**: CPU only (embeddings, reranker on CPU)

---

## When in Doubt

1. **Always check** `docs/IMPLEMENTATION_PLAN.md` (authoritative decisions)
2. **Always check** `.github/copilot-instructions.md` (high-priority constraints)
3. **Always test** with FakeLLM first, then real API
4. **Never** change models without explicit request
5. **Never** commit `.env` or cache files
6. **Always** leave results/ and cache/ intact unless asked to clean

---

## Quick Reference

| Item                | Location                                          |
| ------------------- | ------------------------------------------------- |
| All system code     | `src/ragc/systems/`                             |
| LLM clients         | `src/ragc/llm/clients.py`                       |
| API keys            | `.env` (not committed)                          |
| Configuration       | `config.yaml`                                   |
| Corpus              | `data/raw/` and `data/processed/chunks.jsonl` |
| Results             | `results/raw/{system}/{qid}.json`               |
| Cache               | `cache/llm_cache.sqlite` (prompt dedup)         |
| Tests               | `tests/test_*.py`                               |
| Benchmark script    | `scripts/benchmark_all_systems.py`              |
| UI server           | `ui_server.py`                                  |
| Implementation plan | `docs/IMPLEMENTATION_PLAN.md`                   |
| This document       | `CLAUDE.md`                                     |
