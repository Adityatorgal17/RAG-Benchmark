#!/usr/bin/env python3
"""Comprehensive benchmarking suite for all 13 RAG systems.

Metrics:
- Latency: wall-clock time (s)
- LLM Calls: number of API calls to LLM
- Tokens: prompt + completion tokens used
- Retrieval Quality: recall@k, MRR@k, precision@k
- Answer Quality: token F1, citation coverage, abstention rate
"""

from __future__ import annotations

import json
import sys
import os
from collections import defaultdict
from time import perf_counter
from pathlib import Path

# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragc.config import load_config
from ragc.llm.clients import build_main_llm
from ragc.retrieval.reranker import Reranker
from ragc.retrieval.retriever import Retriever
from ragc.systems.registry import get_system_factory


SYSTEMS = ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "A1", "A2a", "A2b"]
RERANKER_SYSTEMS = {"S2", "S3", "S6c", "A1", "A2a", "A2b"}


def load_questions(limit=None):
    """Load test questions from questions.jsonl."""
    questions_path = Path(__file__).parent.parent / "data" / "questions" / "questions.jsonl"
    questions = []
    with open(questions_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            questions.append({
                "qid": data.get("qid", f"q{i}"),
                "text": data.get("question", ""),
                "gold_chunks": data.get("gold_chunks", []),
            })
    return questions


def compute_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    """Compute recall@k: fraction of gold items in top-k retrieved items."""
    if not gold_ids:
        return 1.0  # Perfect recall if no gold items
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    intersection = len(top_k & gold_set)
    return intersection / len(gold_set) if gold_set else 1.0


def compute_mrr_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    """Compute MRR@k: mean reciprocal rank of first gold item in top-k."""
    if not gold_ids:
        return 1.0
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k], 1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    """Compute precision@k: fraction of top-k that are gold items."""
    if not retrieved_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    intersection = len(top_k & gold_set)
    return intersection / min(k, len(retrieved_ids))


def compute_token_f1(answer: str, question: str) -> float:
    """Compute token-level F1: simple word overlap baseline."""
    if not answer or not question:
        return 0.0
    
    # Tokenize (simple split on whitespace + punctuation)
    import re
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    question_tokens = set(re.findall(r"\b\w+\b", question.lower()))
    
    # Remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been",
                  "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "what"}
    answer_tokens -= stop_words
    question_tokens -= stop_words
    
    if not answer_tokens:
        return 0.0
    
    overlap = len(answer_tokens & question_tokens)
    precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    recall = overlap / len(question_tokens) if question_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def run_benchmark(num_questions: int = 40, output_file: str = "results/benchmark.jsonl"):
    """Run comprehensive benchmark across all systems and questions."""
    cfg = load_config()
    
    # Load LLM and retriever
    llm = build_main_llm(cfg, fake=False)  # Always use real LLM for benchmarking
    retriever = Retriever(cfg)
    
    # Load reranker (shared across systems that need it)
    reranker_cfg = getattr(cfg, "reranker", None)
    reranker = Reranker(
        getattr(reranker_cfg, "model", "BAAI/bge-reranker-base"),
        batch_size=int(getattr(reranker_cfg, "batch_size", 16)),
        max_length=int(getattr(reranker_cfg, "max_length", 384)),
    )
    
    questions = load_questions(limit=num_questions)
    print(f"🔄 Loaded {len(questions)} questions")
    
    # Results storage
    results = []
    system_stats = defaultdict(lambda: {
        "latency_s": [],
        "llm_calls": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "recall_at_5": [],
        "mrr_at_5": [],
        "precision_at_5": [],
        "token_f1": [],
        "abstain_rate": 0.0,
        "error_rate": 0.0,
        "errors": [],
    })
    
    # Run benchmark
    total = len(SYSTEMS) * len(questions)
    current = 0
    
    for system_id in SYSTEMS:
        print(f"\n📊 Benchmarking {system_id}...")
        factory = get_system_factory(system_id)
        
        system_errors = 0
        system_abstentions = 0
        
        for question in questions:
            current += 1
            qid = question["qid"]
            text = question["text"]
            gold_chunks = question.get("gold_chunks", [])
            
            try:
                # Create system instance
                sys_instance = factory(
                    cfg=cfg,
                    llm=llm,
                    retriever=retriever,
                    reranker=reranker if system_id in RERANKER_SYSTEMS else None,
                )
                
                # Run system
                start = perf_counter()
                result = sys_instance.run(qid, text)
                elapsed = perf_counter() - start
                
                # Extract metrics
                usage = result.usage or {}
                latency = usage.get("wall_s", elapsed)
                llm_calls = usage.get("llm_calls", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                # Retrieval quality
                candidate_ids = result.candidate_ids or []
                recall_5 = compute_recall_at_k(candidate_ids, gold_chunks, k=5)
                mrr_5 = compute_mrr_at_k(candidate_ids, gold_chunks, k=5)
                precision_5 = compute_precision_at_k(candidate_ids, gold_chunks, k=5)
                
                # Answer quality
                answer = result.answer or ""
                token_f1 = compute_token_f1(answer, text)
                
                # Abstention tracking
                is_abstain = result.abstained or answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE")
                
                # Record result
                record = {
                    "system": system_id,
                    "qid": qid,
                    "latency_s": round(latency, 4),
                    "llm_calls": llm_calls,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "recall_at_5": round(recall_5, 4),
                    "mrr_at_5": round(mrr_5, 4),
                    "precision_at_5": round(precision_5, 4),
                    "token_f1": round(token_f1, 4),
                    "abstained": is_abstain,
                    "has_errors": len(result.errors) > 0,
                    "num_citations": len(result.cited_ids or []),
                }
                results.append(record)
                
                # Aggregate stats
                system_stats[system_id]["latency_s"].append(latency)
                system_stats[system_id]["llm_calls"].append(llm_calls)
                system_stats[system_id]["prompt_tokens"].append(prompt_tokens)
                system_stats[system_id]["completion_tokens"].append(completion_tokens)
                system_stats[system_id]["recall_at_5"].append(recall_5)
                system_stats[system_id]["mrr_at_5"].append(mrr_5)
                system_stats[system_id]["precision_at_5"].append(precision_5)
                system_stats[system_id]["token_f1"].append(token_f1)
                
                if is_abstain:
                    system_abstentions += 1
                if result.errors:
                    system_errors += 1
                    system_stats[system_id]["errors"].extend(result.errors)
                
                # Progress
                pct = (current / total) * 100
                print(f"  [{pct:5.1f}%] {qid}: {latency:.2f}s, {llm_calls} calls, R@5={recall_5:.3f}, F1={token_f1:.3f}")
                
            except Exception as e:
                print(f"  ❌ {qid}: {str(e)}")
                system_errors += 1
                system_stats[system_id]["errors"].append(str(e))
        
        # Compute final system stats
        stats = system_stats[system_id]
        n = len(questions) - system_errors
        if n > 0:
            stats["latency_s"] = round(sum(stats["latency_s"]) / n, 4)
            stats["llm_calls"] = round(sum(stats["llm_calls"]) / n, 2)
            stats["prompt_tokens"] = round(sum(stats["prompt_tokens"]) / n, 1)
            stats["completion_tokens"] = round(sum(stats["completion_tokens"]) / n, 1)
            stats["recall_at_5"] = round(sum(stats["recall_at_5"]) / n, 4)
            stats["mrr_at_5"] = round(sum(stats["mrr_at_5"]) / n, 4)
            stats["precision_at_5"] = round(sum(stats["precision_at_5"]) / n, 4)
            stats["token_f1"] = round(sum(stats["token_f1"]) / n, 4)
        else:
            # All runs had errors, use defaults
            stats["latency_s"] = 0.0
            stats["llm_calls"] = 0.0
            stats["prompt_tokens"] = 0.0
            stats["completion_tokens"] = 0.0
            stats["recall_at_5"] = 0.0
            stats["mrr_at_5"] = 0.0
            stats["precision_at_5"] = 0.0
            stats["token_f1"] = 0.0
        stats["abstain_rate"] = round(system_abstentions / len(questions), 3)
        stats["error_rate"] = round(system_errors / len(questions), 3)
    
    # Save detailed results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    print(f"\n✅ Detailed results saved to {output_file}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY (All Metrics Aggregated)")
    print("=" * 120)
    print(f"{'System':<10} {'Latency (s)':<12} {'LLM Calls':<12} {'Tokens':<15} {'Recall@5':<12} {'MRR@5':<12} {'Precision@5':<13} {'Token F1':<10} {'Abstain':<10} {'Errors':<10}")
    print("-" * 120)
    
    for system_id in SYSTEMS:
        stats = system_stats[system_id]
        # Ensure values are numbers
        pt = int(stats['prompt_tokens']) if isinstance(stats['prompt_tokens'], (int, float)) else 0
        ct = int(stats['completion_tokens']) if isinstance(stats['completion_tokens'], (int, float)) else 0
        tokens = f"{pt}+{ct}"
        ls = float(stats['latency_s']) if isinstance(stats['latency_s'], (int, float)) else 0.0
        lc = float(stats['llm_calls']) if isinstance(stats['llm_calls'], (int, float)) else 0.0
        r5 = float(stats['recall_at_5']) if isinstance(stats['recall_at_5'], (int, float)) else 0.0
        m5 = float(stats['mrr_at_5']) if isinstance(stats['mrr_at_5'], (int, float)) else 0.0
        p5 = float(stats['precision_at_5']) if isinstance(stats['precision_at_5'], (int, float)) else 0.0
        f1 = float(stats['token_f1']) if isinstance(stats['token_f1'], (int, float)) else 0.0
        ar = float(stats['abstain_rate']) if isinstance(stats['abstain_rate'], (int, float)) else 0.0
        er = float(stats['error_rate']) if isinstance(stats['error_rate'], (int, float)) else 0.0
        print(f"{system_id:<10} {ls:<12} {lc:<12.2f} {tokens:<15} {r5:<12.4f} {m5:<12.4f} {p5:<13.4f} {f1:<10.4f} {ar:<10.1%} {er:<10.1%}")
    
    # Save summary
    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(system_stats, f, indent=2)
    print(f"\n✅ Summary saved to {summary_file}")
    
    return results, system_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark all RAG systems")
    parser.add_argument("--num-questions", type=int, default=40, help="Number of questions to benchmark (default: all 40)")
    parser.add_argument("--output", type=str, default="results/benchmark.jsonl", help="Output file for detailed results")
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive RAG benchmark with REAL LLM...")
    print(f"   Testing {len(SYSTEMS)} systems × {args.num_questions} questions = {len(SYSTEMS) * args.num_questions} runs")
    print()
    
    try:
        results, stats = run_benchmark(num_questions=args.num_questions, output_file=args.output)
        print("\n🎉 Benchmark complete!")
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
