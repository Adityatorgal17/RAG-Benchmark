#!/usr/bin/env python3
"""
Comprehensive validation test for all 13 RAG systems
Tests with 4 diverse questions to expose key paradigm differences
"""

import sys
import json
import time
import os
from pathlib import Path

# Set up paths
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragc.config import load_config
from ragc.schemas import SystemResult
from ragc.systems.registry import get_system_factory
from ragc.llm.clients import build_main_llm
from ragc.retrieval.retriever import Retriever
from ragc.retrieval.reranker import Reranker

# Load config
config = load_config()

# Test Questions - Strategically chosen to expose paradigm differences
TEST_QUESTIONS = {
    "Q1_simple_factual": {
        "question": "What is CVE-2024-20359 and what is its severity?",
        "reason": "Tests basic factual retrieval. All systems should excel. Baseline expectation.",
        "category": "Simple Factual",
        "expects_multi_doc": False,
    },
    "Q2_synthesis": {
        "question": "What are the critical vulnerabilities in enterprise VPN appliances and how are they being actively exploited by threat actors?",
        "reason": "Tests multi-document synthesis and complex reasoning. Exposes retrieval quality and reasoning depth.",
        "category": "Multi-Step Synthesis",
        "expects_multi_doc": True,
    },
    "Q3_comparison": {
        "question": "Compare the severity and prevalence of Cisco ASA and Palo Alto vulnerabilities. Which represents greater current threat?",
        "reason": "Tests multi-document comparison and judgment. Distinguishes analytical systems from simple retrieval.",
        "category": "Comparison/Analysis",
        "expects_multi_doc": True,
    },
    "Q4_sparse": {
        "question": "What are the primary defense mechanisms and detection techniques for the T1041 command and control technique?",
        "reason": "Tests knowledge gaps and graceful degradation. Checks routing decisions under uncertainty.",
        "category": "Sparse Information",
        "expects_multi_doc": False,
    },
}

SYSTEMS = [
    "S-1",  # Baseline
    "S0",   # Naive retrieval
    "S1",   # Pre-retrieval
    "S2",   # Post-retrieval
    "S3",   # Decomposition
    "S4",   # LLM-centric
    "S5",   # Single-agent
    "S6a",  # Acquisition agents
    "S6b",  # Utilization agents
    "S6c",  # Hybrid agents
    "A1",   # Ablation 1
    "A2a",  # Ablation 2a
    "A2b",  # Ablation 2b
]

def run_validation():
    """Run comprehensive validation test"""
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": TEST_QUESTIONS,
        "systems": {},
        "summary": {},
    }
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RAG SYSTEM VALIDATION TEST")
    print("="*80)
    print(f"\nRunning {len(SYSTEMS)} systems × {len(TEST_QUESTIONS)} questions")
    print(f"Total runs: {len(SYSTEMS) * len(TEST_QUESTIONS)}\n")
    
    # Load env and build shared dependencies
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    
    # Build LLM and retrieval components once
    llm = build_main_llm(config, fake=False)
    retriever = Retriever(config)
    reranker = Reranker(config)
    
    # Test each system
    for system_id in SYSTEMS:
        print(f"\n{'='*80}")
        print(f"Testing: {system_id}")
        print(f"{'='*80}")
        
        results["systems"][system_id] = {
            "name": system_id,
            "questions": {},
            "stats": {
                "total": len(TEST_QUESTIONS),
                "passed": 0,
                "failed": 0,
                "partial": 0,
                "avg_latency": 0,
                "total_llm_calls": 0,
            }
        }
        
        latencies = []
        total_calls = 0
        
        # Test each question
        for q_id, q_data in TEST_QUESTIONS.items():
            question = q_data["question"]
            
            print(f"\n  [{q_id}] {q_data['category']}")
            print(f"  Question: {question[:70]}...")
            
            try:
                # Create system using factory
                factory = get_system_factory(system_id)
                system = factory(cfg=config, llm=llm, retriever=retriever, reranker=reranker)
                
                # Run system
                start_time = time.time()
                result = system.run(q_id, question)
                elapsed = time.time() - start_time
                
                latencies.append(elapsed)
                total_calls += result.usage.get("llm_calls", 0)
                
                # Evaluate result
                passed = False
                reason = ""
                
                if result.status == "GENERATED" and result.answer:
                    # Answer was generated
                    passed = True
                    reason = f"Generated answer ({len(result.answer)} chars)"
                elif result.status == "ABSTAINED":
                    reason = "System abstained (no sufficient evidence)"
                    passed = False
                elif result.status == "ERROR":
                    reason = f"Error: {result.error_type}"
                    passed = False
                else:
                    reason = f"Unexpected status: {result.status}"
                    passed = False
                
                # Record result
                results["systems"][system_id]["questions"][q_id] = {
                    "status": result.status,
                    "answer_length": len(result.answer) if result.answer else 0,
                    "latency": round(elapsed, 2),
                    "llm_calls": result.usage.get("llm_calls", 0),
                    "tokens": result.usage.get("total_tokens", 0),
                    "passed": passed,
                    "reason": reason,
                    "error": result.error_type if result.status == "ERROR" else None,
                }
                
                # Update stats
                if passed:
                    results["systems"][system_id]["stats"]["passed"] += 1
                else:
                    results["systems"][system_id]["stats"]["failed"] += 1
                
                status_icon = "✅" if passed else "❌"
                print(f"  {status_icon} {reason} ({elapsed:.2f}s, {result.usage.get('llm_calls', 0)} calls)")
                
            except Exception as e:
                print(f"  ❌ Exception: {str(e)[:100]}")
                results["systems"][system_id]["questions"][q_id] = {
                    "status": "EXCEPTION",
                    "passed": False,
                    "reason": str(e)[:200],
                    "error": str(type(e).__name__),
                }
                results["systems"][system_id]["stats"]["failed"] += 1
        
        # Calculate averages
        if latencies:
            results["systems"][system_id]["stats"]["avg_latency"] = round(sum(latencies) / len(latencies), 2)
        results["systems"][system_id]["stats"]["total_llm_calls"] = total_calls
    
    # Calculate overall summary
    total_passed = sum(s["stats"]["passed"] for s in results["systems"].values())
    total_run = sum(s["stats"]["total"] for s in results["systems"].values())
    
    results["summary"] = {
        "total_runs": total_run,
        "total_passed": total_passed,
        "total_failed": total_run - total_passed,
        "pass_rate": f"{(total_passed/total_run)*100:.1f}%",
    }
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Results: {total_passed}/{total_run} passed ({(total_passed/total_run)*100:.1f}%)\n")
    
    # Save results
    results_file = Path(__file__).parent.parent / "results" / "validation_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}\n")
    
    return results

if __name__ == "__main__":
    results = run_validation()
