# Quick Commands Guide

All commands should be run from the project root directory: `/home/aditya17/Aditya/IIITK/sem7/DL/KAG`

## 1. Initial Setup

### Create Virtual Environment

```bash
python3 -m venv .venv
```

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup API Keys

```bash
# Create .env file with your Groq API keys:
cat > .env << 'EOF'
GROQ_API_KEY=your_first_key_here
GROQ_API_KEY1=your_second_key_here
GROQ_API_KEY2=your_third_key_here
GROQ_API_KEY3=your_fourth_key_here
GROQ_API_KEY4=your_fifth_key_here
GROQ_API_KEY5=your_sixth_key_here
EOF

# Make sure .env is not committed:
echo ".env" >> .gitignore
```

---

## 2. Running Benchmarks

### Run Full Benchmark (All 13 Systems × 40 Questions)

```bash
# Clean previous results
rm -rf results/raw cache/keypool_state.json

# Run full benchmark (approx 45-60 minutes with rate limiting)
python3 scripts/benchmark_all_systems.py --num-questions 40
```

### Run Quick Validation Test (All 13 Systems × 10 Questions)

```bash
# Clean previous results
rm -rf results/raw cache/keypool_state.json

# Run validation benchmark (approx 15-20 minutes)
python3 scripts/benchmark_all_systems.py --num-questions 10
```

### Run Single System on Dev Questions

```bash
# Test a specific system (S0, S1, S2, ..., S6c, A1, A2a, A2b, S-1)
python3 scripts/run_system.py --system S0 --split dev --limit 5
```

### Run Single System with Fake LLM (No API Calls)

```bash
# Perfect for testing logic without consuming API quota
python3 scripts/run_system.py --system S5 --limit 1 --fake
```

### Test Specific Question with Specific System

```bash
python3 scripts/run_system.py --system S2 --qid t01
```

---

## 3. Running the UI

### Start the Web UI Server

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Start server (runs on http://localhost:8000)
python ui_server.py
```

### Access the UI

```
Open browser and visit: http://localhost:8000
```

### Run UI Tests

```bash
python3 scripts/ui_validation_test.py
```

---

## 4. Running Tests

### Run All Unit Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_t05_parsers.py -v
```

### Run Tests with Coverage

```bash
pytest --cov=src/ragc
```

### Test Without API Calls (FakeLLM)

```bash
# All systems validated with fake LLM
pytest tests/test_t11_system.py -v
```

---

## 5. Analyzing Results

### View Benchmark Summary

```bash
# After running benchmark, view aggregated metrics
cat results/benchmark_summary_fixed.json | python3 -m json.tool
```

### View Full Benchmark Results (JSONL)

```bash
# Each line is one run (system, question, metrics)
head -20 results/benchmark.jsonl
wc -l results/benchmark.jsonl  # Total number of runs
```

### Export Results to CSV

```bash
# Already done, but check metrics by system
cat results/exports/metrics_by_system.csv
```

### Check Specific Question Results

```bash
# View results for question t01 across all systems
grep '"qid": "t01"' results/benchmark.jsonl
```

---

## 6. Troubleshooting

### Check API Key Configuration

```bash
python3 scripts/check_models.py
```

### Clear Cache and Results

```bash
# Clears LLM cache and benchmark results
rm -rf cache/*.sqlite cache/*.json results/raw/
```

### Reset API Key Quota State

```bash
# Clear the 24-hour exhaustion blacklist
rm cache/keypool_state.json
```

### View System-Specific Results

```bash
# For debugging, check raw results for a system
ls results/raw/S5/  # Shows all questions that were run
cat results/raw/S5/t01.json | python3 -m json.tool
```

---

## 7. Configuration

### Modify System Parameters

Edit `config.yaml`:

- Retrieval parameters (dense_k, bm25_k, final_k)
- Chunking parameters (words, overlap, min_words)
- Model names
- Rate limiting settings

### Check Current Configuration

```bash
cat config.yaml
```

---

## 8. Typical Workflow

### First Time Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Create .env with your API keys
```

### Development/Testing

```bash
# Test system logic without API cost
python3 scripts/run_system.py --system S5 --limit 1 --fake

# Run unit tests
pytest tests/test_t11_system.py -v
```

### Run Full Benchmark

```bash
# Clean old results
rm -rf results/raw cache/keypool_state.json

# Run benchmark (takes ~45-60 minutes)
python3 scripts/benchmark_all_systems.py --num-questions 40

# Check results
cat results/benchmark_summary_fixed.json | python3 -m json.tool
```

### Visualize Results with UI

```bash
# In another terminal, start the UI
python ui_server.py

# Open http://localhost:8000
# Select a system and enter a question to see pipeline execution
```

---

## 9. Performance Notes

### API Rate Limiting

- 6 independent Groq API keys
- Each key: 25 requests/minute (free tier)
- Total capacity: 150 requests/minute
- Benchmark applies intelligent rate limiting (0.1-1.0s delays)
- **Recommendation**: Use rate limiting for full benchmarks to avoid quota exhaustion

### Typical Benchmark Times

- Full benchmark (40 questions): 45-60 minutes with rate limiting
- Quick validation (10 questions): 15-20 minutes
- Single system: 2-5 minutes depending on system complexity

### Estimated API Calls

- Full benchmark: ~1,200-1,300 LLM calls total
- With 150 RPM capacity: ~8 minutes of continuous calling
- With rate limiting: spread over ~45-60 minutes = safe margin
