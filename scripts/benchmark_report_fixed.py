#!/usr/bin/env python3
"""
Fixed benchmark report generator - handles error runs correctly.
Excludes failed runs from metrics, reports success rates separately.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

def load_benchmark_results(results_file):
    """Load results from JSONL file"""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def compute_aggregate_stats(results):
    """Compute aggregated stats, excluding error runs"""
    stats = defaultdict(lambda: {
        "latency_s": [],
        "llm_calls": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "recall_at_5": [],
        "mrr_at_5": [],
        "precision_at_5": [],
        "token_f1": [],
        "citations": [],
        "successful_runs": 0,
        "total_runs": 0,
        "error_runs": 0,
        "abstain_runs": 0,
    })
    
    for result in results:
        system_id = result["system"]
        stats[system_id]["total_runs"] += 1
        
        # Skip if there was an error
        if result.get("has_errors"):
            stats[system_id]["error_runs"] += 1
            continue
        
        stats[system_id]["successful_runs"] += 1
        stats[system_id]["latency_s"].append(result["latency_s"])
        stats[system_id]["llm_calls"].append(result["llm_calls"])
        stats[system_id]["prompt_tokens"].append(result["prompt_tokens"])
        stats[system_id]["completion_tokens"].append(result["completion_tokens"])
        stats[system_id]["recall_at_5"].append(result["recall_at_5"])
        stats[system_id]["mrr_at_5"].append(result["mrr_at_5"])
        stats[system_id]["precision_at_5"].append(result["precision_at_5"])
        stats[system_id]["token_f1"].append(result["token_f1"])
        stats[system_id]["citations"].append(result.get("num_citations", 0))
        
        # Count abstentions
        if result.get("abstained", False):
            stats[system_id]["abstain_runs"] += 1
    
    # Compute final averages
    final_stats = {}
    for system_id, data in stats.items():
        n = data["successful_runs"]
        final_stats[system_id] = {
            "n": data["total_runs"],
            "n_successful": n,
            "n_error": data["error_runs"],
            "n_abstain": data["abstain_runs"],
            "latency_s": round(sum(data["latency_s"]) / n, 4) if n > 0 else 0.0,
            "llm_calls": round(sum(data["llm_calls"]) / n, 2) if n > 0 else 0.0,
            "prompt_tokens": round(sum(data["prompt_tokens"]) / n, 1) if n > 0 else 0.0,
            "completion_tokens": round(sum(data["completion_tokens"]) / n, 1) if n > 0 else 0.0,
            "recall_at_5": round(sum(data["recall_at_5"]) / n, 4) if n > 0 else 0.0,
            "mrr_at_5": round(sum(data["mrr_at_5"]) / n, 4) if n > 0 else 0.0,
            "precision_at_5": round(sum(data["precision_at_5"]) / n, 4) if n > 0 else 0.0,
            "token_f1": round(sum(data["token_f1"]) / n, 4) if n > 0 else 0.0,
            "citations": round(sum(data["citations"]) / n, 2) if n > 0 else 0.0,
            "error_rate": round(data["error_runs"] / data["total_runs"], 3) if data["total_runs"] > 0 else 0.0,
            "abstain_rate": round(data["abstain_runs"] / data["total_runs"], 3) if data["total_runs"] > 0 else 0.0,
        }
    
    return final_stats

def print_summary_table(stats):
    """Print comprehensive summary table"""
    print("\n" + "=" * 140)
    print("COMPREHENSIVE BENCHMARK SUMMARY - ALL SYSTEMS (ERRORS EXCLUDED FROM METRICS)")
    print("=" * 140)
    print(f"{'System':<7} {'n':<4} {'OK':<4} {'Err':<4} {'Latency(s)':<12} {'LLM Calls':<11} {'Tokens':<13} {'Recall@5':<10} {'MRR@5':<10} {'TokenF1':<9} {'Error%':<8}")
    print("-" * 140)
    
    for system_id in sorted(stats.keys()):
        data = stats[system_id]
        tokens = f"{int(data['prompt_tokens'])}+{int(data['completion_tokens'])}"
        latency_str = f"{data['latency_s']:.4f}" if data['latency_s'] > 0 else "N/A"
        error_pct = f"{data['error_rate']*100:.1f}%"
        
        print(
            f"{system_id:<7} {data['n']:<4} {data['n_successful']:<4} {data['n_error']:<4} "
            f"{latency_str:<12} {data['llm_calls']:<11.2f} {tokens:<13} "
            f"{data['recall_at_5']:<10.4f} {data['mrr_at_5']:<10.4f} {data['token_f1']:<9.4f} {error_pct:<8}"
        )

def print_rankings(stats):
    """Print rankings by metric"""
    print("\n" + "=" * 140)
    print("SYSTEM RANKINGS BY METRIC (ONLY SUCCESSFUL RUNS COUNTED)")
    print("=" * 140)
    
    # Filter to systems with at least 1 successful run
    valid_stats = {s: d for s, d in stats.items() if d['n_successful'] > 0}
    
    print("\n🏆 Speed (Lowest Latency):")
    ranked = sorted(valid_stats.items(), key=lambda x: x[1]['latency_s'])
    for i, (sys_id, data) in enumerate(ranked[:5], 1):
        print(f"  {i}. {sys_id:<7} {data['latency_s']:.4f}s ({data['n_successful']}/{data['n']} runs)")
    
    print("\n🏆 Efficiency (Lowest LLM Calls):")
    ranked = sorted(valid_stats.items(), key=lambda x: x[1]['llm_calls'])
    for i, (sys_id, data) in enumerate(ranked[:5], 1):
        print(f"  {i}. {sys_id:<7} {data['llm_calls']:.2f} calls ({data['n_successful']}/{data['n']} runs)")
    
    print("\n🏆 Token Efficiency (Lowest Tokens):")
    ranked = sorted(valid_stats.items(), key=lambda x: x[1]['prompt_tokens'] + x[1]['completion_tokens'])
    for i, (sys_id, data) in enumerate(ranked[:5], 1):
        total_tokens = data['prompt_tokens'] + data['completion_tokens']
        print(f"  {i}. {sys_id:<7} {total_tokens:.0f} tokens ({data['n_successful']}/{data['n']} runs)")
    
    print("\n🏆 Retrieval Quality (Highest Recall@5):")
    ranked = sorted(valid_stats.items(), key=lambda x: x[1]['recall_at_5'], reverse=True)
    for i, (sys_id, data) in enumerate(ranked[:5], 1):
        print(f"  {i}. {sys_id:<7} {data['recall_at_5']:.4f} ({data['n_successful']}/{data['n']} runs)")
    
    print("\n🏆 Answer Quality (Highest Token F1):")
    ranked = sorted(valid_stats.items(), key=lambda x: x[1]['token_f1'], reverse=True)
    for i, (sys_id, data) in enumerate(ranked[:5], 1):
        print(f"  {i}. {sys_id:<7} {data['token_f1']:.4f} ({data['n_successful']}/{data['n']} runs)")
    
    print("\n🏆 Citation Coverage (Avg Citations):")
    ranked = sorted(valid_stats.items(), key=lambda x: x[1]['citations'], reverse=True)
    for i, (sys_id, data) in enumerate(ranked[:5], 1):
        print(f"  {i}. {sys_id:<7} {data['citations']:.2f} citations ({data['n_successful']}/{data['n']} runs)")
    
    print("\n🏆 Reliability (Lowest Error Rate):")
    ranked = sorted(valid_stats.items(), key=lambda x: x[1]['error_rate'])
    for i, (sys_id, data) in enumerate(ranked[:5], 1):
        error_pct = f"{data['error_rate']*100:.1f}%"
        print(f"  {i}. {sys_id:<7} {error_pct} error rate ({data['n_successful']}/{data['n']} runs)")
    
    # Systems with 0 successful runs
    failed_systems = {s: d for s, d in stats.items() if d['n_successful'] == 0}
    if failed_systems:
        print("\n❌ Failed Systems (0 Successful Runs):")
        for sys_id, data in sorted(failed_systems.items()):
            print(f"  {sys_id:<7} {data['n_error']}/{data['n']} runs failed ({data['error_rate']*100:.1f}% error rate)")

def generate_csv_exports(stats):
    """Export metrics to CSV files"""
    os.makedirs("results/exports", exist_ok=True)
    
    # All metrics
    with open("results/exports/metrics_by_system.csv", "w") as f:
        f.write("System,Runs,Successful,Errors,Latency(s),LLM_Calls,Prompt_Tokens,Completion_Tokens,Recall@5,MRR@5,TokenF1,Citations,Error_Rate,Abstain_Rate\n")
        for sys_id in sorted(stats.keys()):
            d = stats[sys_id]
            f.write(f"{sys_id},{d['n']},{d['n_successful']},{d['n_error']},{d['latency_s']},{d['llm_calls']},{d['prompt_tokens']},{d['completion_tokens']},{d['recall_at_5']},{d['mrr_at_5']},{d['token_f1']},{d['citations']},{d['error_rate']},{d['abstain_rate']}\n")
    
    # Retrieval quality
    with open("results/exports/retrieval_quality.csv", "w") as f:
        f.write("System,Runs,Successful,Recall@5,MRR@5,Precision@5\n")
        for sys_id in sorted(stats.keys()):
            d = stats[sys_id]
            f.write(f"{sys_id},{d['n']},{d['n_successful']},{d['recall_at_5']},{d['mrr_at_5']},{d['precision_at_5']}\n")
    
    # Efficiency
    with open("results/exports/efficiency.csv", "w") as f:
        f.write("System,Runs,Successful,Latency(s),LLM_Calls,Total_Tokens,TokenF1,Citations\n")
        for sys_id in sorted(stats.keys()):
            d = stats[sys_id]
            total_tokens = d['prompt_tokens'] + d['completion_tokens']
            f.write(f"{sys_id},{d['n']},{d['n_successful']},{d['latency_s']},{d['llm_calls']},{total_tokens},{d['token_f1']},{d['citations']}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark report from JSONL results")
    parser.add_argument("--results", default="results/benchmark.jsonl", help="Path to benchmark JSONL file")
    args = parser.parse_args()
    
    # Load and process
    print(f"📊 Loading benchmark results from {args.results}...")
    results = load_benchmark_results(args.results)
    print(f"✅ Loaded {len(results)} result records\n")
    
    # Compute stats
    stats = compute_aggregate_stats(results)
    
    # Print reports
    print_summary_table(stats)
    print_rankings(stats)
    
    # Export CSVs
    generate_csv_exports(stats)
    print("\n✅ Metrics CSV exported to results/exports/metrics_by_system.csv")
    print("✅ Retrieval quality CSV exported to results/exports/retrieval_quality.csv")
    print("✅ Efficiency CSV exported to results/exports/efficiency.csv")
    
    # Save updated summary
    with open("results/benchmark_summary_fixed.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Summary saved to results/benchmark_summary_fixed.json")
    
    print("\n🎉 Fixed benchmark report complete!")
