# Project Glossary

Complete reference for all RAG concepts, models, and evaluation metrics used in this project.

---

## RAG Paradigms (Core Concepts)

### Naive RAG
- **Definition**: Retrieve most relevant documents, concatenate into context, ask LLM to generate answer
- **Systems**: S0
- **Pros**: Simple, baseline
- **Cons**: Noisy context, no iterative refinement
- **Key**: Single retrieval → single generation

### Retrieval-Centric RAG
Optimizes what, where, and when to retrieve before or after fetching documents.

#### Pre-Retrieval
- **Definition**: Query optimization before retrieval (rewriting, expansion, hypothetical documents)
- **Systems**: S1
- **Techniques**:
  - Query Rewriting: Transform ambiguous queries into better search terms
  - HyDE (Hypothetical Document Embeddings): Generate imaginary relevant documents, use their embeddings for retrieval
  - RRF (Reciprocal Rank Fusion): Combine rankings from multiple retrieval methods
- **Use Case**: Improve retrieval quality by fixing query problems first

#### Post-Retrieval
- **Definition**: Evidence refinement after retrieval (filtering, compression, reranking)
- **Systems**: S2
- **Techniques**:
  - Reranking: Use cross-encoder to re-score retrieved documents
  - Filtering: Remove irrelevant chunks based on semantic similarity
  - Sentence Compression: Shorten evidence to most critical parts
- **Use Case**: Remove noise from retrieved context

#### Combined (Pre + Post)
- **Definition**: Both pre-retrieval optimization and post-retrieval refinement
- **Systems**: S3
- **Example**: Query decomposition → retrieval → reranking → compression
- **Use Case**: Maximize retrieval quality at both ends

### LLM-Centric RAG
Lets LLM decide when/what to retrieve during reasoning.

- **Definition**: LLM controls retrieval flow (gate, chain-of-note, self-critique, revision)
- **Systems**: S4
- **Key Techniques**:
  - Gate: LLM decides if question answerable from memory or needs retrieval
  - Chain-of-Note: LLM explicitly reasons through retrieved notes
  - Self-Critique: LLM evaluates its own answer quality
  - Revision: Iterative refinement based on self-critique
- **Pros**: Adaptive retrieval, only retrieves when needed
- **Cons**: Requires more LLM calls

### Agentic RAG
Agents with explicit roles manage retrieval and generation workflows.

#### Single-Agent
- **Definition**: One agent with scratchpad reasoning and search tool
- **Systems**: S5
- **Workflow**: Agent thinks → decides action (search/answer) → tool execution → repeat
- **Tool**: Hybrid search (dense + sparse)
- **Pros**: Transparent reasoning loop
- **Cons**: Limited parallelization

#### Multi-Agent (Acquisition-Driven)
- **Definition**: Agents specialized in deciding what/where/when to retrieve
- **Systems**: S6a
- **Roles**:
  - Decision Agent: Decides if retrieval needed
  - Query Agent: Generates search queries
  - Sufficiency Agent: Checks if enough evidence gathered
- **Key**: Focuses on optimal evidence acquisition
- **Use Case**: Complex questions needing multiple search rounds

#### Multi-Agent (Utilization-Driven)
- **Definition**: Agents specialized in refining and organizing evidence for generation
- **Systems**: S6b
- **Roles**:
  - Evaluator: Rates evidence relevance
  - Note-Taker: Synthesizes notes from evidence
  - Generator: Creates final answer
  - Reconciler: Ensures consistency
- **Key**: Focuses on optimal evidence usage
- **Use Case**: Complex synthesis requiring careful organization

#### Multi-Agent (Hybrid/Dynamic)
- **Definition**: Agents with dynamic routing and source-aware retrieval
- **Systems**: S6c
- **Roles**:
  - Planner: Routes to specific sources (NVD, CISA, vendor, threat-intel)
  - Refiner: Iteratively refines queries based on retrieved evidence
  - Decider: Decides when to stop retrieving
- **Key**: Source-routing for domain-specific retrieval
- **Use Case**: Multi-source corpus requiring strategic source selection

---

## Embedding Models

### BAAI/bge-small-en-v1.5
- **Full Name**: BAAI BGE (Base General Embeddings) Small English v1.5
- **Dimensions**: 384
- **Parameters**: ~33M
- **Architecture**: Sentence Transformer (BiEncoders)
- **Library**: sentence-transformers
- **Use Case**: Dense vector retrieval for chunked documents
- **Performance**: High quality-to-size ratio, fast on CPU
- **Why Chosen**: Strong retrieval quality, small RAM footprint, works on CPU
- **Key Capability**: Captures semantic similarity between queries and documents

---

## Vector Databases & Retrieval

### ChromaDB
- **Type**: Embedded vector database (no server required)
- **Persistence**: SQLite-backed local storage at `data/index/chroma/`
- **Collection**: `cve_corpus` (single collection for all 520 chunks)
- **Distance Metric**: Cosine similarity
- **Special Feature**: Metadata filtering for source-aware retrieval
- **Why Chosen**: Lightweight, local, supports filtering
- **Use**: Store and search dense embeddings from bge-small

### BM25 (Okapi BM25)
- **Type**: Sparse retrieval / lexical search
- **Implementation**: rank-bm25 library with custom CVE tokenizer
- **Library**: rank-bm25
- **Purpose**: Exact-match retrieval for identifiers (CVE-XXXX-YYYY, version strings)
- **Custom Tokenizer**: Keeps CVE IDs and version strings intact (not splitting them)
- **Persistence**: Pickled at `data/index/bm25.pkl`
- **Why Separate from Dense**: Dense embeddings blur exact identifiers; BM25 captures them
- **Score Type**: TF-IDF-based ranking

### Reciprocal Rank Fusion (RRF)
- **Type**: Hybrid fusion algorithm
- **Input**: Rankings from dense (bge-small) and sparse (BM25) retrieval
- **Formula**: `score = sum(1 / (k + rank))` for each method
- **k Parameter**: 60 (balances contributions)
- **Output**: Single merged ranking
- **Why Used**: Combines strengths (dense: semantic, sparse: exact-match) without needing separate training
- **Systems Using**: S0, S2, S3, S5, S6a, S6b, S6c, A1

---

## Reranking & Filtering

### BAAI/bge-reranker-base
- **Type**: Cross-Encoder (not bi-encoder like embeddings)
- **Architecture**: BERT-based sequence classification
- **Input**: (query, document) pairs
- **Output**: Logit → sigmoid(logit) = relevance score [0, 1]
- **Library**: transformers (torch on CPU)
- **RAM**: ~1.1 GB when loaded
- **Inference Speed**: ~100 pairs/second on CPU
- **Purpose**: Re-score retrieved documents with higher precision
- **Why Different from BGE**: BGE is symmetric (query vs doc), reranker is asymmetric (query context matters)
- **Systems Using**: S2, S3, S6c, A1, A2a (reranker only), A2b

---

## Evaluation Metrics

### Token-Level F1 Score
- **Definition**: Harmonic mean of precision and recall at token level
- **Calculation**: Compares tokens in LLM answer against tokens in reference answer
- **Range**: [0, 1], where 1 = perfect match
- **Why Used**: Standard NLP metric, measures exact answer content overlap
- **Interpretation**:
  - 0.0 = Completely wrong answer
  - 0.1-0.2 = Partially relevant
  - 0.3+ = Reasonably good
  - 0.5+ = Very good
- **Limitation**: Doesn't capture semantic paraphrasing (e.g., "bug" vs "vulnerability")
- **Reference**: BLEU, ROUGE family

### Recall@5
- **Definition**: Fraction of relevant chunks appearing in top-5 retrieved results
- **Calculation**: `relevant_in_top5 / total_relevant_chunks`
- **Range**: [0, 1]
- **Why Used**: Measures retrieval quality (does the system find needed evidence?)
- **Interpretation**:
  - 0.0 = No relevant chunks retrieved
  - 0.5 = Half the relevant evidence found
  - 1.0 = All relevant evidence in top 5
- **Limitation**: Depends on gold annotation (which chunks are "relevant")

### Faithfulness (via NLI)
- **Definition**: Does the answer follow logically from retrieved evidence?
- **Model**: cross-encoder/nli-deberta-v3-xsmall (local NLI classifier)
- **Labels**: ENTAILMENT, NEUTRAL, CONTRADICTION
- **Interpretation**:
  - ENTAILMENT = Answer supported by evidence ✅
  - NEUTRAL = Answer neither supported nor contradicted
  - CONTRADICTION = Answer contradicts evidence ❌
- **Why Used**: Quality check (preventing hallucinations)
- **Evaluation Only**: Not used during system execution, only in post-analysis

### Latency
- **Definition**: End-to-end time to answer one question (wall-clock seconds)
- **Includes**: All retrieval, LLM calls, parsing, post-processing
- **Excludes**: Initial model loading
- **Typical Range**: 5-15 seconds per question
- **Why Tracked**: Practical system viability (slower = fewer users served)

### LLM Call Count
- **Definition**: Number of LLM API calls per question
- **Budget**: 8 calls max per system per question (1 reserved for final generation)
- **Actual Usage**: Varies by system
  - S0: 1 call (just generation)
  - S1-S3: 1-2 calls (pre-/post-processing + generation)
  - S4-S6: 3-5 calls (reasoning + refinement + generation)
- **Why Tracked**: Cost proxy (more calls = more expensive)

---

## System Outputs

### SystemResult
- **Fields**:
  - `system`: System ID (S0, S1, etc.)
  - `qid`: Question ID (t01, t02, etc.)
  - `question`: The question text
  - `answer`: LLM-generated answer
  - `retrieval_results`: List of retrieved chunks with scores
  - `latency_s`: Wall-clock execution time
  - `llm_calls`: Total LLM calls made
  - `errors`: List of error messages (if any)
- **Stored**: `results/raw/{system}/{qid}.json`

### Benchmark Summary
- **Aggregates results across all questions for each system**
- **Per-System Metrics**:
  - `n`: Total questions run
  - `n_successful`: Questions with valid answers
  - `n_error`: Questions with errors
  - `n_abstain`: Questions where system declined to answer
  - `latency_s`: Average latency
  - `llm_calls`: Average LLM calls
  - `token_f1`: Mean F1 score
  - `error_rate`: Percentage of runs with errors
  - `abstain_rate`: Percentage of abstentions
- **Stored**: `results/benchmark_summary_fixed.json`

---

## Data Pipeline

### Raw Corpus
- **Sources**:
  - **NVD** (National Vulnerability Database): Official CVE descriptions
  - **CISA KEV** (Known Exploited Vulnerabilities): List of actively exploited CVEs
  - **Palo Alto**: Vendor-specific vulnerability research
  - **Threat Intel**: External threat intelligence
- **Location**: `data/raw/{nvd,kev,vendor,threatintel}/`
- **Format**: JSON documents with metadata (source_type, title, CVE IDs)

### Chunking Strategy
- **Target Size**: 180 words per chunk (~250 tokens)
- **Overlap**: 30 words between consecutive chunks (context preservation)
- **Minimum Chunk Size**: 25 words (prevent tiny fragments)
- **Prefix**: Chunks include `title - section` header for context
- **Processed Storage**: `data/processed/chunks.jsonl` (520 chunks total)
- **Why 180 Words**: Balances granularity (enough context to answer) with coverage (5 chunks fit context cap)

### Questions
- **Total**: 40 questions
  - 10 Development (d01-d10): For prompt debugging
  - 30 Test (t01-t30): For reporting metrics
- **Domain**: Security (CVE, vulnerability, exploitation)
- **Answerable**: All questions answerable from corpus
- **Format**: Plain text questions
- **Storage**: `data/questions/questions.jsonl`

---

## LLM Configuration

### Main LLM (Groq)
- **Model**: Groq `openai/gpt-oss-20b`
- **Type**: Open-source LLM via Groq inference API
- **Temperature**: 0 (deterministic, no randomness)
- **JSON Mode**: Enabled for structured outputs (reasoning steps, actions)
- **Cost**: Free tier (limited RPM)
- **Rate Limit**: 25 requests per minute per API key
- **Why Groq**: Free, fast inference, good for benchmarking

### Judge LLM (Evaluation Only)
- **Model**: Groq `llama-3.3-70b-versatile`
- **Purpose**: Evaluate answer quality (F1, citation counting)
- **Temperature**: 0
- **Why Different Model**: Reduces self-preference bias (main LLM evaluating its own answers)
- **Use**: Post-benchmark analysis only, not during system execution

---

## Context & Tokens

### Context Window
- **Total Tokens Available**: ~32k (typical for modern LLMs)
- **Reserved for Output**: ~4k (answer generation)
- **Available for Prompt+Evidence**: ~28k
- **Evidence Cap**: 2,000 tokens (~8,000 characters)
- **Why 2,000 Tokens**: Allows ~5-7 chunks in context (180 words × 250 tokens ≈ 1,250 tokens)

### Token Counting
- **Method**: Estimated as `characters / 4` (rough approximation)
- **Actual Token Count**: Varies by tokenizer and model
- **Used For**: Context budgeting in systems S2 and S6c

---

## Common Abbreviations

| Abbr | Full Form |
|------|-----------|
| RAG | Retrieval-Augmented Generation |
| LLM | Large Language Model |
| RRF | Reciprocal Rank Fusion |
| HyDE | Hypothetical Document Embeddings |
| NLI | Natural Language Inference |
| CVE | Common Vulnerabilities and Exposures |
| BM25 | Okapi BM25 ranking algorithm |
| RPM | Requests Per Minute |
| F1 | Harmonic mean of Precision and Recall |
| BLEU | Bilingual Evaluation Understudy |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation |
| QA | Question Answering |
| IR | Information Retrieval |

---

## Quick Reference by Question Type

### "Which CVE should I use?"
→ Check corpus. All questions answerable from `data/raw/` sources.

### "What model is used for embeddings?"
→ BAAI/bge-small-en-v1.5 (384-d, sentence-transformers)

### "How much does it cost?"
→ Free (using free-tier API keys), but rate-limited (25 RPM per key)

### "How long does full benchmark take?"
→ 45-60 minutes with rate limiting (45+ minute spreads 1,200 API calls to stay under quota)

### "Can I run without internet?"
→ Embeddings and reranker are local (offline). LLM calls require API (online).

### "Why 13 systems, not 10?"
→ 10 main + 3 ablation variants (A1, A2a, A2b) to isolate design decisions

