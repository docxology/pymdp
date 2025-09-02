#!/bin/bash

# PyMDP Textbook Examples Runner
# ===============================
# Consolidated script to run all examples with logging and comprehensive reporting
#
# Usage: bash run_all.sh [--verbose] [--help]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
VERBOSE=false
for arg in "$@"; do
    case $arg in
        --verbose) VERBOSE=true ;;
        --help)
            echo "PyMDP Textbook Examples Runner"
            echo "Usage: bash run_all.sh [--verbose] [--help]"
            echo ""
            echo "Options:"
            echo "  --verbose    Show detailed output during execution"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# Helper functions
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    print_colored "$BLUE" "============================================================"
    print_colored "$BLUE" "$1"
    print_colored "$BLUE" "============================================================"
}

# Check dependencies
if [ ! -f "01_probability_basics.py" ]; then
    print_colored "$RED" "Error: Must be run from examples directory"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    print_colored "$RED" "Error: python3 not found"
    exit 1
fi

# Setup
mkdir -p logs
START_TIME=$(date +%s)
LOG_FILE="logs/run_all_$(date +%Y%m%d_%H%M%S).log"

# Example list with categories
examples=(
    "01_probability_basics.py|Foundation|Probability distributions and basic operations"
    "02_bayes_rule.py|Foundation|Bayes rule and belief updating with VFE"
    "03_observation_models.py|Foundation|Building observation models (A matrices)"
    "03_observation_models_refactored.py|Foundation|Refactored observation models using PyMDP core"
    "04_state_inference.py|Inference|Inferring hidden states from observations"
    "05_sequential_inference.py|Inference|Sequential inference over time with VFE"
    "06_multi_factor_models.py|Inference|Multi-factor state space models"
    "07_transition_models.py|Dynamics|Building transition models (B matrices)"
    "08_preferences_and_control.py|Control|EFE-based preferences and action selection"
    "09_policy_inference.py|Control|Multi-step policy inference and planning"
    "10_simple_pomdp.py|POMDP|Complete POMDP with active inference"
    "10_simple_pomdp_backup.py|POMDP|Backup simple POMDP variant"
    "11_gridworld_pomdp.py|POMDP|Grid world navigation POMDP"
    "12_tmaze_pomdp.py|POMDP|T-maze decision making POMDP"
)

# Initialize counters
successful=0
failed=0
total=${#examples[@]}
auth_failed=0
out_failed=0

# Header
print_header "PyMDP Textbook Examples - Complete Run"
echo "Start time: $(date)"
echo "Total examples: $total"
echo "Timeout per example: 180 seconds"
echo "Log file: $LOG_FILE"

if $VERBOSE; then
    print_colored "$YELLOW" "Running in VERBOSE mode"
fi

echo ""

# Initialize log
{
    echo "PyMDP Examples Execution Log - $(date)"
    echo "========================================"
    echo ""
} > "$LOG_FILE"

# Verification helpers
verify_outputs() {
    local out_dir="$1"
    local example_name="$2"
    local ok=true

    if [ ! -d "$out_dir" ]; then
        print_colored "$YELLOW" "    Warning: No output directory found"
        ok=false
    else
        # Non-empty files check
        local zero_count
        zero_count=$(find "$out_dir" -type f -size 0 2>/dev/null | wc -l)
        if [ "$zero_count" -gt 0 ]; then
            print_colored "$YELLOW" "    Warning: $zero_count zero-byte output files detected"
            ok=false
        fi

        # Validate JSON files (if any) can be parsed
        local json_files
        IFS=$'\n' read -r -d '' -a json_files < <(find "$out_dir" -type f -name "*.json" -print0 | xargs -0 -I{} echo {} && printf '\0') || true
        if [ "${#json_files[@]}" -gt 0 ]; then
            for jf in "${json_files[@]}"; do
                if ! python3 - <<PY 2>/dev/null
import json,sys
with open("$jf","r") as f:
    json.load(f)
PY
                then
                    print_colored "$YELLOW" "    Warning: Invalid JSON: $jf"
                    ok=false
                fi
            done
        fi
    fi

    $ok && return 0 || return 1
}

verify_authenticity() {
    local log_file="$1"
    local suspicious=(
        "Fallback"
        "Using fallback function"
        "Proceeding with educational demonstration"
        "PyMDP inference error"
        "PyMDP error"
    )
    for pat in "${suspicious[@]}"; do
        if grep -qi -- "$pat" "$log_file"; then
            return 1
        fi
    done
    return 0
}

# Run each example
for i in "${!examples[@]}"; do
    # Parse example info
    example_info="${examples[i]}"
    filename=$(echo "$example_info" | cut -d'|' -f1)
    category=$(echo "$example_info" | cut -d'|' -f2)
    description=$(echo "$example_info" | cut -d'|' -f3)
    example_num=$((i + 1))
    
    print_colored "$BLUE" "[$example_num/$total] $filename"
    echo "  Category: $category"
    echo "  Description: $description"
    
    start_time=$(date +%s)
    log_file="logs/${filename%.py}.log"
    
    if $VERBOSE; then
        echo "  Command: python3 $filename"
        echo "  Log: $log_file"
    fi
    
    # Run example with timeout and capture result
    if timeout 180 python3 "$filename" > "$log_file" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_colored "$GREEN" "  ✓ SUCCESS (${duration}s)"
        
        # Check outputs
        output_dir="outputs/${filename%.py}"
        if [ -d "$output_dir" ]; then
            file_count=$(find "$output_dir" -type f | wc -l)
            echo "    Output files: $file_count in $output_dir/"
        else
            print_colored "$YELLOW" "    Warning: No output directory found"
        fi
        
        ((successful++))
        status="SUCCESS"
    else
        exit_code=$?
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        if [ $exit_code -eq 124 ]; then
            print_colored "$RED" "  ✗ TIMEOUT (${duration}s)"
            status="TIMEOUT"
        else
            print_colored "$RED" "  ✗ FAILED (${duration}s)"
            status="FAILED"
        fi
        
        if $VERBOSE; then
            echo "    See log: $log_file"
        fi
        
        ((failed++))
    fi
    
    # Output verification
    if ! verify_outputs "$output_dir" "$filename"; then
        ((out_failed++))
        echo "    Output verification: ${RED}FAILED${NC}"
    else
        echo "    Output verification: ${GREEN}OK${NC}"
    fi

    # Authenticity verification (real PyMDP methods used)
    if ! verify_authenticity "$log_file"; then
        ((auth_failed++))
        echo "    Authenticity check: ${YELLOW}POTENTIAL FALLBACK/ERROR DETECTED${NC}"
        if $VERBOSE; then
            echo "    See log: $log_file"
        fi
    else
        echo "    Authenticity check: ${GREEN}OK${NC}"
    fi

    # Log result
    printf "%-30s %-12s %-8s %4ds  %s\n" \
        "$filename" "$category" "$status" "$duration" "$description" >> "$LOG_FILE"
    
    echo ""
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

print_header "EXECUTION SUMMARY"

echo "Execution completed: $(date)"
echo "Total time: ${TOTAL_DURATION}s"
echo ""

print_colored "$GREEN" "Successful: $successful/$total"
if [ $failed -gt 0 ]; then
    print_colored "$RED" "Failed: $failed/$total"
fi

if [ $out_failed -gt 0 ]; then
    print_colored "$YELLOW" "Output verification warnings: $out_failed"
fi

if [ $auth_failed -gt 0 ]; then
    print_colored "$YELLOW" "Authenticity warnings: $auth_failed (possible fallbacks/errors detected)"
fi

echo ""
print_colored "$BLUE" "Detailed Results:"
printf "%-30s %-12s %-8s %6s  %s\n" "Example" "Category" "Status" "Time" "Description"
echo "==================================================================================="
cat "$LOG_FILE" | tail -n +4

# Output structure
echo ""
print_colored "$BLUE" "Output Structure:"
if [ -d "outputs" ]; then
    find outputs -maxdepth 1 -type d -name "*_*" | sort | while read -r dir; do
        if [ -d "$dir" ]; then
            file_count=$(find "$dir" -type f | wc -l)
            total_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            printf "  %-30s %3d files (%s)\n" "$(basename "$dir")" "$file_count" "$total_size"
        fi
    done
    
    echo ""
    total_outputs=$(find outputs -type f | wc -l)
    total_size=$(du -sh outputs 2>/dev/null | cut -f1 || echo "0")
    echo "  Total: $total_outputs files ($total_size)"
fi

# Final result
echo ""
if [ $failed -eq 0 ] && [ $auth_failed -eq 0 ] && [ $out_failed -eq 0 ]; then
    print_colored "$GREEN" "🎉 All examples completed successfully!"
    print_colored "$GREEN" "The complete PyMDP textbook learning path is working."
    echo ""
    print_colored "$BLUE" "Next steps:"
    echo "  • Explore individual example outputs in outputs/"
    echo "  • Review comprehensive visualizations"
    echo "  • Check execution logs in logs/"
    exit 0
else
    print_colored "$YELLOW" "Some checks reported issues (runtime failures: $failed, output warnings: $out_failed, authenticity warnings: $auth_failed)"
    print_colored "$YELLOW" "Check individual logs and outputs for details."
    exit 1
fi
