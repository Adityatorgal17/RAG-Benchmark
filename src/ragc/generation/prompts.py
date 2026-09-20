"""Prompt templates used by the generator and agent systems.

These are the fixed text templates required by the benchmark so all systems share the
same generation behavior and cite evidence in the same format.
"""

P_GEN = """You are a cybersecurity analyst assistant. Answer the question using ONLY the evidence items provided.
Rules:
1. End every factual statement with its supporting citation(s) in the form [chunk_id], copying ids exactly as given.
2. Cover every part of the question that the evidence supports (for example versions, exploitation status, behaviour, mitigation).
3. If evidence items conflict, state both versions with their citations and say which source is more recent or more authoritative.
4. If the evidence does not support part of the question, name that part as unsupported. If nothing in the evidence answers the question, reply exactly: INSUFFICIENT_EVIDENCE
5. Maximum 180 words. No preamble, no closing remarks.

Question: {question}

Evidence:
{evidence_block}
{extra_block}
"""

P_GEN_NOEVID = """You are a cybersecurity analyst assistant. No documents are available; answer from your own knowledge.
Rules: Maximum 180 words, no preamble. If you are not sure of a specific fact (version numbers, dates, scores, identifiers), do not guess. If you cannot reliably answer any part of the question, reply exactly: INSUFFICIENT_EVIDENCE. Otherwise answer only the parts you are sure of and say which parts you cannot answer.

Question: {question}
"""
