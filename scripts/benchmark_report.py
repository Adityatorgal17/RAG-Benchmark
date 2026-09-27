#!/usr/bin/env python3
"""Generate comprehensive benchmark report from benchmark results.

Generates:
- Summary statistics table
- System ranking by various metrics
- Comparative analysis
- Visualization-ready CSV exports
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_benchmark_results(jsonl_path: str) -> list[dict[str, Any]]:
    """Load benchmark results from JSONL file."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_aggregate_stats(results: list[dict]) -> dict[str, dict]:
    """Compute per-system aggregate statistics."""
    stats = defaultdict(lambda: {
        "count": 0,
        "latency_s": [],
        "llm_calls": [],
        "tokens": [],
        "recall_at_5": [],
        "mrr_at_5": [],
        "precision_at_5": [],
        "token_f1": [],
        "abstain_count": 0,
        "error_count": 0,
        "avg_citations": [],
    })
    
    for result in results:
        system = result["system"]
        stats[system]["count"] += 1
        stats[system]["latency_s"].append(result["latency_s"])
        stats[system]["llm_calls"].append(result["llm_calls"])
        stats[system]["tokens"].append(result["prompt_tokens"] + result["completion_tokens"])
        stats[system]["recall_at_5"].append(result["recall_at_5"])
        stats[system]["mrr_at_5"].append(result["mrr_at_5"])
        stats[system]["precision_at_5"].append(result["precision_at_5"])
        stats[system]["token_f1"].append(result["token_f1"])
        stats[system]["avg_citations"].append(result["num_citations"])
        
        if result.get("abstained"):
            stats[system]["abstain_count"] += 1
        if result.get("has_errors"):
            stats[system]["error_count"] += 1
    
    # Compute means and other aggregate metrics
    aggregate = {}
    for system, data in stats.items():
        n = data["count"]
        aggregate[system] = {
            "n_questions": n,
            "avg_latency_s": round(sum(data["latency_s"]) / n, 4),
            "avg_llm_calls": round(sum(data["llm_calls"]) / n, 2),
            "avg_tokens": round(sum(data["tokens"]) / n, 1),
            "avg_recall_at_5": round(sum(data["recall_at_5"]) / n, 4),
            "avg_mrr_at_5": round(sum(data["mrr_at_5"]) / n, 4),
            "avg_precision_at_5": round(sum(data["precision_at_5"]) / n, 4),
            "avg_token_f1": round(sum(data["token_f1"]) / n, 4),
            "abstain_rate": round(data["abstain_count"] / n, 3),
            "error_rate": round(data["error_count"] / n, 3),
            "avg_citations_per_answer": round(sum(data["avg_citations"]) / n, 2),
            "min_latency_s": round(min(data["latency_s"]), 4),
            "max_latency_s": round(max(data["latency_s"]), 4),
            "std_latency_s": round(
                (sum((x - aggregate[system].get("avg_latency_s", 0))**2 for x in data["latency_s"]) / n) ** 0.5
                if n > 0 else 0, 4
            ) if "avg_latency_s" in aggregate.get(system, {}) else 0,
        }
    
    return aggregate


def print_summary_table(aggregate: dict[str, dict]):
    """Print formatted summary table."""
    print("\n" + "=" * 160)
    print("COMPREHENSIVE BENCHMARK SUMMARY - ALL 13 SYSTEMS")
    print("=" * 160)
    print(f"{'System':<10} {'n':<4} {'Latency(s)':<14} {'LLM Calls':<12} {'Tokens':<12} {'Recall@5':<12} {'MRR@5':<12} {'Prec@5':<12} {'TokenF1':<10} {'Citations':<11} {'Abstain':<10} {'Error':<8}")
    print("-" * 160)
    
    for system in ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "A1", "A2a", "A2b"]:
        if system not in aggregate:
            continue
        
        stats = aggregate[system]
        latency_str = f"{stats['avg_latency_s']:.2f}s"
        tokens_str = f"{int(stats['avg_tokens'])}"
        
        print(
            f"{system:<10} "
            f"{stats['n_questions']:<4} "
            f"{latency_str:<14} "
            f"{stats['avg_llm_calls']:<12.1f} "
            f"{tokens_str:<12} "
            f"{stats['avg_recall_at_5']:<12.4f} "
            f"{stats['avg_mrr_at_5']:<12.4f} "
            f"{stats['avg_precision_at_5']:<12.4f} "
            f"{stats['avg_token_f1']:<10.4f} "
            f"{stats['avg_citations_per_answer']:<11.2f} "
            f"{stats['abstain_rate']:<10.1%} "
            f"{stats['error_rate']:<8.1%}"
        )
    
    print("=" * 160)


def print_rankings(aggregate: dict[str, dict]):
    """Print system rankings by various metrics."""
    print("\n" + "=" * 120)
    print("SYSTEM RANKINGS BY METRIC")
    print("=" * 120)
    
    # Ranking functions
    metrics = [
        ("Speed (Lowest Latency)", "avg_latency_s", lambda x: -x),
        ("Efficiency (Lowest LLM Calls)", "avg_llm_calls", lambda x: -x),
        ("Token Efficiency (Lowest Tokens)", "avg_tokens", lambda x: -x),
        ("Retrieval Quality (Highest Recall@5)", "avg_recall_at_5", lambda x: x),
        ("Retrieval Ranking (Highest MRR@5)", "avg_mrr_at_5", lambda x: x),
        ("Precision@5", "avg_precision_at_5", lambda x: x),
        ("Answer Quality (Highest Token F1)", "avg_token_f1", lambda x: x),
        ("Citation Coverage (Avg Citations)", "avg_citations_per_answer", lambda x: x),
        ("Reliability (Lowest Abstention)", "abstain_rate", lambda x: -x),
    ]
    
    for metric_name, key, sort_fn in metrics:
        ranked = sorted(
            [(sys, aggregate[sys][key]) for sys in aggregate if key in aggregate[sys]],
            key=lambda x: sort_fn(x[1]),
            reverse=True
        )
        
        print(f"\n🏆 {metric_name}:")
        for i, (system, value) in enumerate(ranked[:5], 1):
            if key.endswith("_rate"):
                print(f"  {i}. {system:<8} {value:>8.1%}")
            elif key in {"avg_citations_per_answer"}:
                print(f"  {i}. {system:<8} {value:>8.2f}")
            elif key in {"avg_latency_s", "avg_token_f1", "avg_recall_at_5", "avg_mrr_at_5", "avg_precision_at_5"}:
                print(f"  {i}. {system:<8} {value:>8.4f}")
            else:
                print(f"  {i}. {system:<8} {value:>8.1f}")


def generate_csv_exports(aggregate: dict[str, dict], output_dir: str):
    """Generate CSV files for charting and analysis."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Metrics CSV
    metrics_csv = Path(output_dir) / "metrics_by_system.csv"
    with open(metrics_csv, "w") as f:
        f.write("System,n_questions,avg_latency_s,avg_llm_calls,avg_tokens,avg_recall_at_5,avg_mrr_at_5,avg_precision_at_5,avg_token_f1,avg_citations,abstain_rate,error_rate\n")
        for system in ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "A1", "A2a", "A2b"]:
            if system not in aggregate:
                continue
            stats = aggregate[system]
            f.write(
                f"{system},"
                f"{stats['n_questions']},"
                f"{stats['avg_latency_s']},"
                f"{stats['avg_llm_calls']},"
                f"{stats['avg_tokens']},"
                f"{stats['avg_recall_at_5']},"
                f"{stats['avg_mrr_at_5']},"
                f"{stats['avg_precision_at_5']},"
                f"{stats['avg_token_f1']},"
                f"{stats['avg_citations_per_answer']},"
                f"{stats['abstain_rate']},"
                f"{stats['error_rate']}\n"
            )
    
    print(f"\n✅ Metrics CSV exported to {metrics_csv}")
    
    # Retrieval quality CSV
    retrieval_csv = Path(output_dir) / "retrieval_quality.csv"
    with open(retrieval_csv, "w") as f:
        f.write("System,Recall@5,MRR@5,Precision@5\n")
        for system in ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "A1", "A2a", "A2b"]:
            if system not in aggregate:
                continue
            stats = aggregate[system]
            f.write(f"{system},{stats['avg_recall_at_5']},{stats['avg_mrr_at_5']},{stats['avg_precision_at_5']}\n")
    
    print(f"✅ Retrieval quality CSV exported to {retrieval_csv}")
    
    # Efficiency CSV
    efficiency_csv = Path(output_dir) / "efficiency.csv"
    with open(efficiency_csv, "w") as f:
        f.write("System,Latency(s),LLM_Calls,Tokens,Citations\n")
        for system in ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "A1", "A2a", "A2b"]:
            if system not in aggregate:
                continue
            stats = aggregate[system]
            f.write(
                f"{system},"
                f"{stats['avg_latency_s']},"
                f"{stats['avg_llm_calls']},"
                f"{stats['avg_tokens']},"
                f"{stats['avg_citations_per_answer']}\n"
            )
    
    print(f"✅ Efficiency CSV exported to {efficiency_csv}")


def main(results_file: str = "results/benchmark.jsonl"):
    """Generate comprehensive benchmark report."""
    path = Path(results_file)
    
    if not path.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📊 Loading benchmark results from {results_file}...")
    results = load_benchmark_results(str(path))
    
    if not results:
        print("❌ No results found in file")
        return
    
    print(f"✅ Loaded {len(results)} result records")
    
    # Compute statistics
    aggregate = compute_aggregate_stats(results)
    
    # Print reports
    print_summary_table(aggregate)
    print_rankings(aggregate)
    
    # Export CSVs
    generate_csv_exports(aggregate, "results/exports")
    
    # Save JSON summary
    summary_file = path.parent / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n✅ Summary saved to {summary_file}")
    
    print("\n🎉 Benchmark report complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results", type=str, default="results/benchmark.jsonl", help="Path to benchmark.jsonl")
    
    args = parser.parse_args()
    main(args.results)
