#!/usr/bin/env bash
set -euo pipefail

# System tests for fish.py
# Each test invokes the tool directly and checks for a zero exit code.

PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift
    echo -n "TEST: ${name} ... "
    if "$@" > /dev/null 2>&1; then
        echo "OK"
        PASS=$((PASS + 1))
    else
        echo "FAIL (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

# Show help (no arguments)
run_test "help flag" python fish.py --help

# Search stations by river name
run_test "station-list by river" python fish.py --station-list "Loire"

# Search stations by station name
run_test "station-list by station name" python fish.py --station-list "Paris"

# Plot a specific station by code (Seine at Paris-Austerlitz)
run_test "station by code" python fish.py --station F700000103

# Location-based lookup
run_test "location Paris" python fish.py Paris

# Location-based lookup with --date (past date)
run_test "location with past date" python fish.py Lyon --date 2025-01-15

# Station with --date
run_test "station with past date" python fish.py --station F700000103 --date 2025-01-15

# Location-based lookup with --tomorrow
run_test "location with tomorrow" python fish.py Paris --tomorrow

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"

if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
