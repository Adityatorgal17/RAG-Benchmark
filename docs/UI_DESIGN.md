# RAG Pipeline Explorer - UI Design Documentation

## Overview

The RAG Pipeline Explorer is a modern, dark-themed web interface that visualizes the execution of different RAG systems step-by-step. It provides deep insights into how each system processes queries, retrieves information, ranks candidates, and generates answers.

## Architecture

### Layout

The UI follows a **two-column responsive layout**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline Explorer                       │
├─────────────────┬───────────────────────────────────────────────┤
│  CONTROL PANEL  │                 CHAT PANEL                    │
│                 │                                               │
│ • System Select │  ┌─────────────────────────────────────────┐ │
│ • Question Input│  │           Chat Header (Stats)            │ │
│ • Run Button    │  ├─────────────────────────────────────────┤ │
│ • Runtime Info  │  │                                         │ │
│                 │  │  Messages Container                     │ │
│                 │  │  ├─ Pipeline Messages (Steps)           │ │
│                 │  │  ├─ Answer Message                      │ │
│                 │  │  └─ Evidence Blocks                     │ │
│                 │  │                                         │ │
│                 │  ├─────────────────────────────────────────┤ │
│                 │  │  Chat Input Area (Quick Actions)        │ │
│                 │  └─────────────────────────────────────────┘ │
└─────────────────┴───────────────────────────────────────────────┘
```

### Color Scheme (Dark Theme)

| Element | Color | Hex |
|---------|-------|-----|
| Primary Background | Very Dark Gray | #0f0f0f |
| Secondary Background | Dark Gray | #1a1a1a |
| Tertiary Background | Medium Gray | #252525 |
| Text Primary | White | #ffffff |
| Text Secondary | Light Gray | #b0b0b0 |
| Text Muted | Medium Gray | #808080 |
| Accent | Bright Cyan | #00d4ff |
| Success | Green | #10b981 |
| Warning | Amber | #f59e0b |
| Error | Red | #ef4444 |
| Info | Cyan | #06b6d4 |

## Features

### 1. System Selector

**Purpose**: Switch between different RAG paradigms

**Layout**: 2-column grid in control panel (13 total systems)

**System Categories**:
- **S-1**: No Retrieval (baseline)
- **S0**: Naive RAG (basic dense retrieval)
- **S1**: Pre-retrieval (query rewriting, HyDE, fusion)
- **S2**: Post-retrieval (hybrid search, reranking, filtering, compression)
- **S3**: Decomposition (sub-questions)
- **S4**: LLM-centric (gating, notes, critique)
- **S5**: Single-agent (search loop)
- **S6a**: Acquisition Agents (multi-agent decision)
- **S6b**: Utilization Agents (multi-agent evaluation)
- **S6c**: Hybrid Agents (multi-agent routing)
- **A1**: No Routing (S6c ablation)
- **A2a**: Rerank-only (S2 ablation)
- **A2b**: No Compression (S2 ablation)

**Interaction**:
- Click a system button to select it
- Selected system highlighted in cyan
- System title and description update in header

### 2. Query Input

**Purpose**: Enter questions about the corpus

**Features**:
- Multi-line textarea with 6 rows
- Placeholder text: "Ask a security question about CVE vulnerabilities..."
- Dark theme styling with cyan border on focus
- Keyboard shortcut: Ctrl+Enter to run

### 3. Run Pipeline Button

**States**:
- **Ready**: Cyan button, clickable
- **Running**: Shows spinner, disabled, button text hidden
- **Error**: Red text indicator
- **Complete**: Status badge updates to "Complete"

### 4. Runtime Status Card

**Displays**:
- Runtime Mode: REAL (actual LLM) or FAKE (deterministic testing)
- Provider: Groq, Gemini, etc.
- Model Name: Full model identifier
- Shows in status card with distinct styling

### 5. Chat Header

**Left Section**:
- System badge with system ID (e.g., "S0")
- System title (e.g., "Naive RAG")
- System description (e.g., "Dense retrieval plus fixed generation")

**Right Section (Stats)**:
- LLM Calls: Count of LLM invocations
- Time: Elapsed time in seconds
- Status: Ready/Running/Complete/Error

### 6. Messages Container

**Message Types**:

#### 6.1 Pipeline Messages (Expandable)
Show RAG execution phases:

**Phases**:
- 🔄 **Query Processing**: Query rewriting, HyDE, intent detection
- 🔍 **Retrieval**: Dense and sparse search results
- ⭐ **Reranking**: Cross-encoder scoring, relevance ranking
- 🎯 **Filtering**: Candidate pruning, threshold filtering
- 📦 **Compression**: Evidence summarization, token optimization
- ✍️ **Generation**: LLM answer generation, prompt engineering
- 🤖 **Agent Steps**: Agent state, routing decisions, loop iterations

**Structure**:
```
┌─────────────────────────────────────────┐
│ 🔍 Retrieval                    ▼ ⭐⚡  │  ← Header (clickable)
├─────────────────────────────────────────┤
│ ┌─ Step Details ──────────────────────┐ │
│ │ s0_dense:                           │ │
│ │ [dense retrieval output - 500 chars]│ │
│ │ ───────────────────────────────────  │ │
│ │ s0_sparse:                          │ │
│ │ [sparse retrieval output]           │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Interaction**:
- Click header to expand/collapse
- Toggle button shows ▼ (expanded) or ▲ (collapsed)
- First phase auto-expands
- Colors indicate status: green (success), red (error), yellow (warning)

#### 6.2 Query Message
Shows the user's input question

**Structure**:
```
❓ Query
├─────────────────────────────────────────────
│ "What is the impact of CVE-2024-3400?"
```

#### 6.3 Chunks Message
Shows retrieved chunk IDs in chip format

**Structure**:
```
📋 Retrieved Chunks
├─────────────────────────────────────────────
│ [nvd_CVE-2024-3400#000] [nvd_CVE-2024-3400#001]
│ [kev_CVE-2024-3400#000] [attack_T1041#002]
```

#### 6.4 Answer Message
Final generated answer with status

**Structure**:
```
┌─────────────────────────────────────────────┐
│ ⭐ GENERATED                                 │
│                                             │
│ CVE-2024-3400 is a critical vulnerability...│
│                                             │
│ 📎 Citations                                │
│ ┌─ Evidence Block ─────────────────────────┐│
│ │ [Evidence text...]                       ││
│ │ Source: [nvd_CVE-2024-3400#000]          ││
│ └─────────────────────────────────────────┘│
│ ┌─ Evidence Block ─────────────────────────┐│
│ │ [Evidence text...]                       ││
│ │ Source: [nvd_CVE-2024-3400#001]          ││
│ └─────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

**Status Indicators**:
- ✅ **GENERATED**: Answer produced with evidence
- ⚠️ **INSUFFICIENT_EVIDENCE**: No evidence found
- ❓ **ABSTAIN**: System unable to answer

### 7. Evidence Display

**For Each Citation**:
- Evidence text from chunk
- Green "Success" status
- Source chunk ID in badge
- Clear visual separation

**Citation Tags**:
- Inline cyan badges for chunk references
- Format: `[chunk_id#line]`

### 8. Quick Actions

**Available Actions**:
- 🗑️ **Clear**: Reset chat, clear all messages
- 📋 **Export**: Download trace as JSON file

### 9. Status Indicators

**Colors**:
- 🟢 **Ready/Complete**: Green (success)
- 🟡 **Running**: Amber (processing)
- 🔵 **Info**: Cyan (informational)
- 🔴 **Error**: Red (failure)

## Pipeline Visualization Logic

### Phase Detection
The UI automatically categorizes trace steps into phases:

```python
if tag in ['s1_rewrite', 's4_gate', 's5_agent', ...]:
    phase = 'query_rewrite'
elif tag in ['s0_dense', 's1_dense', 's2_hybrid', ...]:
    phase = 'retrieval'
elif tag in ['s2_rerank', 's3_rerank', ...]:
    phase = 'reranking'
elif tag in ['s2_filter', 's3_filter', ...]:
    phase = 'filtering'
elif tag in ['s2_compress', 's3_compress', ...]:
    phase = 'compression'
elif tag in ['generate', 'answer', 'llm_gen', ...]:
    phase = 'generation'
else:
    phase = 'agent'
```

### Step Details Display

Each step shows:
1. **Tag**: System identifier (e.g., "s2_dense", "s6c_route")
2. **Output**: First 500 characters of result
3. **Status**: Color-coded border (green/red/yellow)
4. **Formatting**: 
   - JSON output: Pretty-printed, truncated
   - Text output: Displayed as-is
   - Complex objects: Converted to JSON string

### Sorting
Phases appear in order:
1. Query Processing
2. Retrieval
3. Reranking
4. Filtering
5. Compression
6. Generation
7. Agent Steps

## Interaction Flows

### Run Query
```
1. User enters question
2. User clicks "Run Pipeline" or Ctrl+Enter
3. Button shows spinner, status → "Running"
4. Welcome message clears
5. Query message appears
6. Phase messages stream in (each expandable)
7. Chunks message shows retrieved items
8. Answer message with citations appears
9. Status → "Complete", stats update
10. Button re-enables
```

### Clear Chat
```
1. User clicks "Clear" button
2. All messages cleared
3. Welcome message reappears
4. Stats reset to defaults
5. Status → "Ready"
```

### Export Trace
```
1. User clicks "Export" button
2. Full trace JSON downloaded
3. Filename: rag-trace-S0-2026-09-26T...json
4. File contains raw trace data for analysis
```

### Expand/Collapse Phase
```
1. User clicks phase header
2. Message-content height animates
3. Toggle button rotates (▼↔▲)
4. Details become visible/hidden
```

## System-Specific Features

### S-1 (No Retrieval)
- Shows only generation phase
- No retrieval, reranking, or filtering
- Direct LLM response

### S0 (Naive RAG)
- Dense retrieval phase
- Direct generation
- Minimal pipeline

### S1 (Pre-retrieval)
- Query rewriting phase
- Dense + hybrid retrieval
- RRF fusion shown in retrieval phase

### S2 (Post-retrieval)
- Hybrid retrieval (dense + sparse)
- Reranking phase
- Filtering phase
- Compression phase
- Final generation

### S3 (Decomposition)
- Sub-question generation
- Per-subquery retrieval
- Aggregated reranking
- Compression before generation

### S4 (LLM-centric)
- Gate decision phase
- Per-step notes
- Critique phase
- Revision phase

### S5 (Single-agent)
- Agent loop iterations shown
- Per-iteration retrieval
- Agent state in trace
- Loop termination reason

### S6a/b/c (Multi-agent)
- Agent role phases
- Inter-agent communication
- Routing decisions
- Utilization phases
- Final aggregation

### A1/A2a/A2b (Ablations)
- Same as base system but with ablated components
- Missing phases in pipeline (shown as skipped)
- Comparison opportunity with base systems

## Error Handling

### Network Errors
- Error message appears in pipeline
- Status badge shows error color
- User can retry by running again

### LLM API Errors
- Caught by backend
- Returns error message
- Answer shows "INSUFFICIENT_EVIDENCE"
- Error details in response

### Parsing Errors
- Step output truncated gracefully
- JSON parsing failures show raw text
- UI remains stable and responsive

## Performance Optimizations

1. **Lazy Loading**: Messages stream in as received
2. **Truncation**: Long outputs truncated to 500 chars
3. **Virtualization**: Only visible messages in DOM
4. **Animations**: GPU-accelerated transforms
5. **Debouncing**: No continuous re-renders

## Accessibility

- **High Contrast**: Cyan accent on dark background
- **Status Indicators**: Color + text + icons
- **Keyboard Navigation**: Tab through sections
- **Semantic HTML**: Proper heading hierarchy
- **ARIA Labels**: Screen reader support (future)

## Future Enhancements

1. **Diff View**: Compare two systems side-by-side
2. **Timeline View**: Visual timeline of steps
3. **Metrics Dashboard**: Graph latency, accuracy across systems
4. **Custom Questions**: Save favorite questions
5. **Batch Mode**: Run multiple questions
6. **Integration Tests**: Automated pipeline testing
7. **Performance Profiling**: Detailed timing breakdown
8. **Evidence Verification**: Highlight citations in original docs

## File Structure

```
ui/
├── index.html          # Semantic HTML layout
├── styles.css          # Dark theme with 400+ lines
├── app.js              # Interactive logic & state management
├── app_old.js          # Backup of previous version
└── styles_old.css      # Backup of previous styles
```

## Running the UI

```bash
# Start the server
python3 ui_server.py

# Open browser
open http://localhost:8000
```

## Configuration

Environment variables (optional):
- `RAGC_UI_FAKE_LLM=1`: Force FakeLLM (deterministic)
- `RAGC_UI_REAL_LLM=1`: Force real LLM (requires API key)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Blank page | Check browser console for errors |
| Buttons not responding | Check server is running on port 8000 |
| No results | Verify API keys in .env |
| Slow performance | Use FakeLLM for faster testing |
| CSS not loading | Clear browser cache, hard refresh |
