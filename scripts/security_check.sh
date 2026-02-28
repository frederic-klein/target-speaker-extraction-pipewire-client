#!/usr/bin/env bash
set -euo pipefail

echo "=== Security Check: TSE-PipeWire ==="
PASS=0
FAIL=0

# Check 1: No network imports in runtime code
echo -n "[1/6] Checking for network imports in src/... "
if grep -rE 'import (requests|urllib|httpx|http\.client|socket|aiohttp)' src/; then
    echo "FAIL: Network library imported in runtime code"
    FAIL=$((FAIL + 1))
else
    echo "PASS"
    PASS=$((PASS + 1))
fi

# Check 2: No hardcoded URLs in runtime code
echo -n "[2/6] Checking for URLs in src/... "
if grep -rE 'https?://' src/; then
    echo "FAIL: URL found in runtime code"
    FAIL=$((FAIL + 1))
else
    echo "PASS"
    PASS=$((PASS + 1))
fi

# Check 3: Checksums verification module exists
echo -n "[3/6] Checking model integrity module exists... "
if [ -f src/tse_pipewire/model_integrity.py ]; then
    echo "PASS"
    PASS=$((PASS + 1))
else
    echo "FAIL: src/tse_pipewire/model_integrity.py not found"
    FAIL=$((FAIL + 1))
fi

# Check 4: No pickle/torch.load in runtime code
echo -n "[4/6] Checking for unsafe deserialization in src/... "
if grep -rE '(pickle\.load|torch\.load|joblib\.load)' src/; then
    echo "FAIL: Unsafe deserialization found in runtime code"
    FAIL=$((FAIL + 1))
else
    echo "PASS"
    PASS=$((PASS + 1))
fi

# Check 5: requirements-locked.txt exists
echo -n "[5/6] Checking requirements-locked.txt exists... "
if [ -f requirements-locked.txt ]; then
    echo "PASS"
    PASS=$((PASS + 1))
else
    echo "FAIL: requirements-locked.txt not found"
    FAIL=$((FAIL + 1))
fi

# Check 6: pip-audit (if available)
echo -n "[6/6] Running pip-audit... "
if command -v pip-audit &>/dev/null; then
    if pip-audit 2>/dev/null; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL: Vulnerabilities found"
        FAIL=$((FAIL + 1))
    fi
else
    echo "SKIP (pip-audit not installed)"
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
