#!/bin/bash
# Monitor benchmark progress
cd /home/aditya17/Aditya/IIITK/sem7/DL/KAG

echo "Benchmark Progress Monitor"
echo "=========================="

while true; do
    completed=$(find results/raw -name "*.json" 2>/dev/null | wc -l)
    total=$((13 * 40))  # 13 systems × 40 questions
    percent=$((completed * 100 / total))
    
    echo "Completed: $completed/$total ($percent%)"
    
    # Show last system being benchmarked
    if [ $completed -gt 0 ]; then
        last=$(tail -1 benchmark_run.log 2>/dev/null | grep -oP "Benchmarking \K[^ ]*" || echo "Running...")
        echo "Current system: $last"
    fi
    
    # Check if benchmark is still running
    if pgrep -f "benchmark_all_systems.py" > /dev/null; then
        echo "Status: RUNNING..."
        sleep 30
    else
        echo "Status: COMPLETED!"
        break
    fi
    
    echo "---"
done

# Show final summary
echo ""
echo "Final Results:"
head -30 results/benchmark_summary_fixed.json 2>/dev/null || echo "Summary file not yet created"
