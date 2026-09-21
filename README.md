# KAG Benchmark

This repository implements the academic comparison of retrieval-augmented generation paradigms described in the project plan. The source of truth is [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Local setup

1. Copy the environment template: `cp .env.example .env`
2. Fill in your Gemini and Groq API keys in `.env`
3. Create and activate a local virtual environment:
   `python -m venv .venv` and `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `pytest`

> Keep `.env` local-only. It is intentionally ignored by git.

## Pilot run for T16

The pilot runs the two simplest systems on three dev questions to estimate the real cost of the benchmark before full test-set execution.

### Measured dev pilot values

The current pilot artifact is saved in [results/pilot.md](results/pilot.md).

| System | Mean calls | Mean tokens | Mean seconds | Questions |
| ------ | ---------: | ----------: | -----------: | --------: |
| S-1    |          2 |         651 |         0.01 |         3 |
| S0     |          2 |         651 |         0.01 |         3 |

These estimates are intentionally early and are meant to calibrate the Section 11.8 budget table before larger batches are run.

## Section 11.8 budget and run rules

| System                 | Max calls / question | Expected average | Calls for 30 test questions |
| ---------------------- | -------------------: | ---------------: | --------------------------: |
| S-1                    |                    1 |                1 |                          30 |
| S0                     |                    1 |                1 |                          30 |
| S1                     |                    2 |                2 |                          60 |
| S2                     |                    1 |                1 |                          30 |
| S3                     |                    2 |                2 |                          60 |
| S4                     |                    5 |              4.5 |                         135 |
| S5                     |                    6 |                4 |                         120 |
| S6a                    |                    7 |              5.5 |                         165 |
| S6b                    |                    7 |                6 |                         180 |
| S6c                    |                    7 |                6 |                         180 |
| A2a + A2b (ablation)   |               1 each |                1 |                          60 |
| A1 (ablation)          |                    7 |                6 |                         180 |
| Total main + ablations |                      |                  |                 about 1,230 |

### Pilot-based calibration note

The pilot suggests the real per-question overhead is modest in the local fake-LLM workflow, but the planned API budget remains conservative. Until a real-provider run confirms otherwise, the project keeps the original estimates from the implementation plan and treats the T16 pilot as a sanity check rather than a final measurement.

### Test-size rule

1. Never run a full-test batch before a system passes its dev sanity check on 3 questions.
2. If daily provider limits are tight, use the `test24` subset rather than forcing the full 30-question run.
3. Keep all judge and system executions resumable and cached.
