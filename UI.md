# RAG Pipeline Explorer - UI Documentation

The RAG Pipeline Explorer is an interactive web-based tool for running RAG queries and visualizing pipeline execution.

---

## 🚀 Quick Start

### Start the Server
```bash
# Activate virtual environment
source .venv/bin/activate

# Start UI server (runs on http://localhost:8000)
python ui_server.py
```

### Access the UI
```
Open browser: http://localhost:8000
```

---

## 📖 User Guide

### 1. Select a RAG System

The interface displays 13 buttons for all available systems:

| Button | System | Paradigm |
|--------|--------|----------|
| **S-1** | No Retrieval Baseline | Reference (parametric only) |
| **S0** | Naive RAG | Basic retrieval + generation |
| **S1** | Pre-Retrieval | Query rewriting + HyDE + RRF |
| **S2** | Post-Retrieval | Hybrid + rerank + filter + compress |
| **S3** | Pre+Post Retrieval | Decomposition + refinement |
| **S4** | LLM-Centric | Gate + chain-of-note + critique |
| **S5** | Single-Agent | Search loop with scratchpad |
| **S6a** | Multi-Agent (Acq.) | Decision + query + sufficiency agents |
| **S6b** | Multi-Agent (Util.) | Evaluator + note-taker + generator |
| **S6c** | Multi-Agent (Hybrid) | Planner + router + refiner + decider |
| **A1** | Ablation | S6c without routing |
| **A2a** | Ablation | S2 rerank-only |
| **A2b** | Ablation | S2 without compression |

### 2. Enter a Question

Type any security-related question in the text input field:

**Example Questions**:
- "What is CVE-2024-3400?"
- "How does CVE-2024-3400 affect edge devices?"
- "What are the recommended mitigations for this vulnerability?"
- "Which vendors are affected by this CVE?"

### 3. Run the Pipeline

Click the **"Run Pipeline"** button to execute the selected system on your question.

### 4. View Results

The UI displays results in real-time:

#### Left Panel: System Information
- **System ID**: Selected RAG system
- **Status**: Running / Complete
- **Execution Time**: Wall-clock seconds
- **LLM Calls**: Number of API calls made
- **Tokens Used**: Estimated token consumption

#### Main Panel: Pipeline Execution Steps
- **Step-by-Step Breakdown**: Each major operation
  - Retrieval (if applicable)
  - Reasoning (if agentic)
  - Generation
- **Retrieved Evidence**: Chunks used for context
- **Intermediate Reasoning**: Agent thoughts (if S5+)
- **Final Answer**: Generated answer from system

#### Right Panel: Metrics
- **Latency**: Time taken (seconds)
- **Chunk Usage**: Number of retrieved chunks
- **Confidence**: LLM confidence in answer (if available)

### 5. Compare Systems

Run the same question on multiple systems to:
- Compare answer quality
- Compare execution times
- See different retrieval strategies
- Understand system differences

---

## 🎨 Interface Features

### Dark Mode (Default)
- Retro pixel aesthetic
- White text on dark background
- Optimized for readability

### System Buttons
- **Colored by Paradigm**:
  - Gray: Baseline (S-1)
  - Blue: Naive (S0)
  - Green: Retrieval-Centric (S1-S3, A2a, A2b)
  - Orange: LLM-Centric (S4)
  - Red: Agentic (S5-S6c, A1)
- **Hover**: Shows system description
- **Active**: Highlights selected system

### Results Panel
- **Collapsible Sections**: Each stage of pipeline
- **Evidence Highlighting**: Retrieved chunks in different colors
- **Code Block**: Raw JSON results (optional)
- **Copy Button**: Copy answer to clipboard

---

## 🔧 Advanced Features

### View Raw Results

Click **"Show JSON"** to see the complete raw result structure:

```json
{
  "system": "S5",
  "qid": "ui_query",
  "question": "What is CVE-2024-3400?",
  "answer": "CVE-2024-3400 is a vulnerability in...",
  "retrieval_results": [
    {
      "chunk_id": 0,
      "score": 0.92,
      "text": "CVE-2024-3400 is a critical vulnerability..."
    }
  ],
  "latency_s": 8.5,
  "llm_calls": 3,
  "errors": [],
  "agent_steps": [
    {
      "step": 1,
      "thought": "I need to search for...",
      "action": "search",
      "query": "CVE-2024-3400"
    }
  ]
}
```

### Batch Run

Run same question on all systems (not in main UI yet):

```bash
python3 scripts/ui_validation_test.py
```

This tests the UI backend on multiple systems and questions.

---

## 🐛 Troubleshooting

### Server Won't Start

**Error**: `Address already in use`
```bash
# Kill existing server on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python ui_server.py --port 8001
```

**Error**: `Module not found`
```bash
# Make sure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

### API Keys Missing

**Error**: `GROQ_API_KEY not found`
```bash
# Create .env file
cp .env.example .env  # or create manually
# Add your API keys
```

### Slow Response

**Normal**: First system run is slower (model loading, embedding cache)
**Expected**: S5-S6c take 10-15 seconds (multiple LLM calls)
**Optimization**: Use S0-S3 for faster responses

### Empty Results

**Check**:
1. Question is entered before running
2. System is selected
3. API keys are configured
4. Check browser console for errors (F12)

### Question Not Answerable

Some questions may not be answerable from the corpus.

**Try**: Rephrase the question or select a different system

---

## 📊 Typical Execution Times

| System | Type | Typical Time | LLM Calls |
|--------|------|--------------|-----------|
| S-1 | Baseline | 1-2s | 1 |
| S0 | Naive | 2-3s | 1 |
| S1 | Pre-Retrieval | 3-4s | 2 |
| S2 | Post-Retrieval | 4-5s | 1 |
| S3 | Pre+Post | 5-6s | 2 |
| S4 | LLM-Centric | 8-10s | 4 |
| S5 | Single-Agent | 10-15s | 3-5 |
| S6a | Multi-Agent A | 12-15s | 4-6 |
| S6b | Multi-Agent U | 10-13s | 5-6 |
| S6c | Multi-Agent H | 12-18s | 5-7 |
| A1 | Ablation | 10-15s | 4-5 |
| A2a | Ablation | 4-5s | 1 |
| A2b | Ablation | 5-6s | 1 |

---

## 🔌 Backend Architecture

### Server Code
**File**: `ui_server.py`
- Lightweight Flask-like server
- JSON API endpoints
- Handles system instantiation and execution
- No async (simple sequential processing)

### Frontend Code
**Files**: 
- `ui/index.html` - HTML structure
- `ui/app.js` - JavaScript logic and DOM manipulation
- `ui/styles.css` - Styling and layout

### API Endpoints

#### POST `/api/run`
```json
Request:
{
  "system": "S5",
  "question": "What is CVE-2024-3400?"
}

Response:
{
  "system": "S5",
  "qid": "ui_query",
  "question": "What is CVE-2024-3400?",
  "answer": "...",
  "retrieval_results": [...],
  "latency_s": 8.5,
  "llm_calls": 3,
  "errors": []
}
```

#### GET `/api/systems`
```json
Response:
{
  "systems": [
    {
      "id": "S-1",
      "name": "No Retrieval Baseline",
      "paradigm": "Reference",
      "color": "gray"
    },
    ...
  ]
}
```

---

## 🎯 Use Cases

### 1. Educational: Understand RAG Paradigms
Run the same question on S0 vs S1 vs S4 vs S5 to see how different paradigms work.

### 2. Debugging: Understand System Behavior
Run your question on a specific system to see:
- What retrieval happened?
- What reasoning was done?
- What evidence was used?
- How long did it take?

### 3. Benchmarking: Compare Systems
Run same questions on multiple systems to:
- Compare answer quality
- Compare speed
- Compare retrieval strategies

### 4. Testing: Validate Implementation
Run example questions to validate that system works correctly.

---

## 📝 Notes

- **UI Queries**: Questions run via UI are saved with `qid: "ui_query"` (not part of benchmark)
- **No Results Storage**: UI queries are not persisted to results/ (benchmarks only)
- **API Rate Limiting**: UI doesn't apply rate limiting (for interactive use)
- **Development Mode**: No authentication or permission control

---

## 🚀 Future Enhancements

Possible future features:
- [ ] Batch run interface (run all systems at once)
- [ ] Question library (pre-defined benchmark questions)
- [ ] Result comparison view (side-by-side system answers)
- [ ] Visualization of retrieval rankings
- [ ] Interactive parameter tuning
- [ ] Results export (CSV, PDF)
- [ ] Performance dashboards
- [ ] User accounts and history

---

## 💡 Tips

1. **Fast Iteration**: S0 and S2 are fastest (1-2 seconds)
2. **Best Quality**: S2 and S3 have good balance of speed and quality
3. **Most Interesting**: S5 and S6c show agent reasoning
4. **Baseline Comparison**: Compare system against S-1 to see impact of retrieval
5. **Troubleshooting**: Run FakeLLM version to isolate API issues

