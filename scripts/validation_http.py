#!/usr/bin/env python3
"""
Validation test via HTTP API - tests all 13 systems with 4 strategic questions
"""

import json
import time
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

TEST_QUESTIONS = {
    "Q1_simple": {
        "question": "What is CVE-2024-20359 and what is its severity?",
        "reason": "Tests basic factual retrieval - all systems should excel",
        "category": "Simple Factual",
    },
    "Q2_synthesis": {
        "question": "What are the critical vulnerabilities in enterprise VPN appliances and how are they being actively exploited by threat actors?",
        "reason": "Tests multi-document synthesis and complex reasoning",
        "category": "Multi-Step Synthesis",
    },
    "Q3_comparison": {
        "question": "Compare the severity and prevalence of Cisco ASA and Palo Alto vulnerabilities. Which represents greater current threat?",
        "reason": "Tests multi-document comparison and analytical judgment",
        "category": "Comparison/Analysis",
    },
    "Q4_sparse": {
        "question": "What are the primary defense mechanisms and detection techniques for the T1041 command and control technique?",
        "reason": "Tests knowledge gaps and graceful degradation",
        "category": "Sparse Information",
    },
}

SYSTEMS = [
    "S-1", "S0", "S1", "S2", "S3", "S4", "S5",
    "S6a", "S6b", "S6c", "A1", "A2a", "A2b"
]

def run_query(system_id, question):
    """Run a query against a system via HTTP API"""
    try:
        response = requests.post(
            f"{API_URL}/api/run",
            json={"system": system_id, "question": question},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}

def run_validation():
    """Run comprehensive validation test"""
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_url": API_URL,
        "questions": TEST_QUESTIONS,
        "systems": {},
    }
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RAG SYSTEM VALIDATION (HTTP API)")
    print("="*80)
    print(f"Testing {len(SYSTEMS)} systems × {len(TEST_QUESTIONS)} questions = {len(SYSTEMS)*len(TEST_QUESTIONS)} runs\n")
    
    for system_id in SYSTEMS:
        print(f"\nTesting: {system_id}")
        print("-" * 60)
        
        results["systems"][system_id] = {
            "questions": {},
            "stats": {"total": 0, "passed": 0, "failed": 0, "avg_latency": 0.0}
        }
        
        latencies = []
        
        for q_id, q_data in TEST_QUESTIONS.items():
            print(f"  [{q_id}] ", end="", flush=True)
            
            start = time.time()
            result = run_query(system_id, q_data["question"])
            elapsed = time.time() - start
            latencies.append(elapsed)
            
            # Evaluate
            passed = False
            status = result.get("status", "UNKNOWN")
            
            if status == "GENERATED":
                passed = True
                answer_len = len(result.get("answer", ""))
                print(f"✅ PASS ({elapsed:.2f}s, {answer_len} chars)")
            elif status == "ABSTAINED":
                print(f"⚠️  ABSTAIN ({elapsed:.2f}s)")
            elif status == "ERROR":
                print(f"❌ ERROR ({elapsed:.2f}s): {result.get('error_type', 'Unknown')}")
            else:
                print(f"❓ {status} ({elapsed:.2f}s)")
            
            results["systems"][system_id]["questions"][q_id] = {
                "status": status,
                "latency": round(elapsed, 2),
                "passed": passed,
                "answer_length": len(result.get("answer", "")) if result.get("answer") else 0,
            }
            
            results["systems"][system_id]["stats"]["total"] += 1
            if passed:
                results["systems"][system_id]["stats"]["passed"] += 1
            else:
                results["systems"][system_id]["stats"]["failed"] += 1
        
        # Calculate average latency
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            results["systems"][system_id]["stats"]["avg_latency"] = round(avg_latency, 2)
            pass_rate = results["systems"][system_id]["stats"]["passed"] / results["systems"][system_id]["stats"]["total"]
            print(f"\n  Summary: {results['systems'][system_id]['stats']['passed']}/{results['systems'][system_id]['stats']['total']} passed, Avg: {avg_latency:.2f}s")
    
    # Overall summary
    total_passed = sum(s["stats"]["passed"] for s in results["systems"].values())
    total_tests = sum(s["stats"]["total"] for s in results["systems"].values())
    
    print("\n" + "="*80)
    print(f"VALIDATION COMPLETE: {total_passed}/{total_tests} passed ({(total_passed/total_tests)*100:.1f}%)")
    print("="*80)
    
    # Save results
    out_file = Path(__file__).parent.parent / "results" / "validation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {out_file}\n")
    
    return results

if __name__ == "__main__":
    results = run_validation()
