#!/usr/bin/env python3
"""
UI Validation Test - Test all 13 RAG systems with strategic questions
"""
import requests
import json
import time
from datetime import datetime

# Test configuration
API_BASE = "http://localhost:8000"

# 5 strategic test questions covering different RAG paradigms
TEST_QUESTIONS = [
    {
        "id": "Q1",
        "text": "What is CVE-2024-21762?",
        "category": "Simple Factual Lookup",
        "why": "Tests basic dense retrieval and factual recall. Should be straightforward for all systems.",
        "expected_to_pass": ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "A1", "A2a", "A2b"]
    },
    {
        "id": "Q2",
        "text": "How are critical vulnerabilities in enterprise VPN appliances being actively exploited by threat actors?",
        "category": "Complex Multi-step Reasoning",
        "why": "Tests decomposition and multi-step reasoning. Favors S3, S6a, S6b (multi-agent). S0 may struggle.",
        "expected_to_pass": ["S1", "S2", "S3", "S6a", "S6b", "S6c"]
    },
    {
        "id": "Q3",
        "text": "Compare how T1041 (Exfiltration Over C2) and T1105 (Ingress Tool Transfer) are used together in attack chains",
        "category": "Comparative Analysis & Synthesis",
        "why": "Tests ability to synthesize multiple sources and compare. Reranking (S2, A2a) and multi-agent (S6a, S6b) should excel.",
        "expected_to_pass": ["S2", "S3", "S6a", "S6b", "A2a", "A2b"]
    },
    {
        "id": "Q4",
        "text": "Which CVEs have been most actively exploited in the wild according to threat intelligence?",
        "category": "Ranking & Filtering",
        "why": "Tests reranking and evidence quality filtering. S2 (reranker), S6a, S6b (utilization) should perform best.",
        "expected_to_pass": ["S2", "S6a", "S6b", "A2a", "A2b"]
    },
    {
        "id": "Q5",
        "text": "What are the most important things to do for defense?",
        "category": "Abstention/Gate Logic",
        "why": "Vague question testing gate logic. S4, S5 should recognize and either abstract or refuse.",
        "expected_to_pass": ["S-1", "S4", "S5"]
    }
]

# All 13 systems
SYSTEMS = ["S-1", "S0", "S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S6c", "A1", "A2a", "A2b"]

def run_test(system_id, question_text):
    """Run a single test against the API"""
    try:
        payload = {
            "system": system_id,
            "question": question_text
        }
        
        response = requests.post(
            f"{API_BASE}/api/run",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "PASS",
                "data": data,
                "llm_calls": data.get("usage", {}).get("llm_calls", 0),
                "latency": data.get("latency_s", 0),
                "has_error": data.get("has_errors", False),
                "answer": data.get("answer", "")[:100],  # First 100 chars
                "error_msg": data.get("errors", [])
            }
        else:
            return {
                "status": "FAIL",
                "error": f"HTTP {response.status_code}",
                "response": response.text[:200]
            }
    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e)[:100]
        }

def main():
    print("\n" + "="*80)
    print("RAG SYSTEMS UI VALIDATION TEST")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Store results
    results = {
        "timestamp": datetime.now().isoformat(),
        "questions": TEST_QUESTIONS,
        "system_results": {}
    }
    
    # Test each system with all questions
    total_tests = len(SYSTEMS) * len(TEST_QUESTIONS)
    current_test = 0
    
    for q_idx, question in enumerate(TEST_QUESTIONS):
        print(f"\n{'─'*80}")
        print(f"QUESTION {question['id']}: {question['category']}")
        print(f"{'─'*80}")
        print(f"Q: {question['text']}")
        print(f"Why: {question['why']}\n")
        
        for s_idx, system in enumerate(SYSTEMS):
            current_test += 1
            progress = f"[{current_test}/{total_tests}]"
            
            print(f"{progress} Testing {system:5s}... ", end="", flush=True)
            
            result = run_test(system, question['text'])
            
            # Store result
            key = f"{question['id']}_{system}"
            results["system_results"][key] = {
                "question_id": question['id'],
                "system": system,
                "result": result
            }
            
            # Print status
            if result['status'] == 'PASS':
                error_indicator = "❌ ERROR" if result.get('has_error') else "✅ OK"
                print(f"{error_indicator} | Calls: {result.get('llm_calls', 0):2d} | Time: {result.get('latency', 0):6.2f}s")
            else:
                print(f"❌ FAIL | {result.get('error', 'Unknown error')}")
            
            time.sleep(0.5)  # Small delay between tests
    
    print(f"\n{'='*80}")
    print("TEST EXECUTION COMPLETE")
    print(f"{'='*80}\n")
    
    # Save detailed results
    results_file = "/home/aditya17/Aditya/IIITK/sem7/DL/KAG/results/ui_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Detailed results saved to: {results_file}\n")
    
    return results

if __name__ == "__main__":
    main()
