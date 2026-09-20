
Project: academic comparison of RAG paradigms (see docs/IMPLEMENTATION_PLAN.md).

- The plan is authoritative. Do not change models, libraries, parameters, prompts or file names.
- Python 3.11, package src/ragc, tests in tests/ with pytest. No async, no web frameworks.
- All LLM calls go through ragc.llm clients (cache, rate limit, retries). Never call an SDK directly elsewhere.
- Every system implements System.run(qid, question) -> SystemResult and respects CallBudget (8 calls, 1 reserved for final generation).
- Write unit tests using FakeLLM before running anything against the real APIs.
- Runs are resumable: never delete results/ or cache/ unless asked.
- Read keys from .env (GEMINI_API_KEY, GROQ_API_KEY); never print or commit them.
- Keep code simple and commented; this is an academic prototype, not production.
