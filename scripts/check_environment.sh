#!/usr/bin/env bash
# check_environment.sh
# Validates all prerequisites for the llm-characterization harness.
# Run this before building or running experiments.
# Does NOT modify any system settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LLAMA_BENCH="${1:-/home/ihtgoot/project/llama.cpp/build/bin/llama-bench}"
MODEL_PATH="${2:-/home/ihtgoot/.cache/huggingface/hub/models--ggml-org--gemma-3-270m-it-GGUF/snapshots/e7647be17ae1108f2f605ed061ca0608b171afff/gemma-3-270m-it-Q8_0.gguf}"

PASS=0
FAIL=0
WARN=0

ok()   { echo "  [OK]   $*"; ((PASS++)); }
fail() { echo "  [FAIL] $*"; ((FAIL++)); }
warn() { echo "  [WARN] $*"; ((WARN++)); }

echo "=== llm-characterization environment check ==="
echo ""

# ── 1. llama-bench ────────────────────────────────────────────────────────
echo "[1] llama-bench binary"
if [[ -f "$LLAMA_BENCH" ]]; then
    ok "Exists: $LLAMA_BENCH"
    if [[ -x "$LLAMA_BENCH" ]]; then
        ok "Executable"
    else
        fail "Not executable: $LLAMA_BENCH"
    fi

    # Try running --version
    LIBDIR="$(dirname "$LLAMA_BENCH")"
    VER=$(LD_LIBRARY_PATH="$LIBDIR" "$LLAMA_BENCH" --version 2>&1 | head -2 || true)
    if [[ -n "$VER" ]]; then
        ok "Version: $(echo "$VER" | tr '\n' ' ')"
    else
        warn "Could not determine version"
    fi
else
    fail "Not found: $LLAMA_BENCH"
fi
echo ""

# ── 2. Model file ─────────────────────────────────────────────────────────
echo "[2] Model file"
if [[ -f "$MODEL_PATH" ]]; then
    SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo "unknown")
    ok "Exists: $MODEL_PATH ($SIZE bytes)"
else
    warn "Default model not found: $MODEL_PATH"
    warn "Update config/experiment.yaml with a valid model path"
fi
echo ""

# ── 3. perf ───────────────────────────────────────────────────────────────
echo "[3] perf"
if command -v perf &>/dev/null; then
    PERF_VER=$(perf --version 2>/dev/null || echo "unknown")
    ok "perf found: $PERF_VER"

    PARANOIA=$(sysctl -n kernel.perf_event_paranoia 2>/dev/null || echo "unknown")
    if [[ "$PARANOIA" == "unknown" ]]; then
        warn "Cannot read perf_event_paranoia (try: sysctl kernel.perf_event_paranoia)"
    elif [[ "$PARANOIA" -le 1 ]]; then
        ok "perf_event_paranoia=$PARANOIA (hardware events available)"
    else
        warn "perf_event_paranoia=$PARANOIA — hardware events may require sudo"
        warn "To fix: echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoia"
    fi

    # Probe hardware events
    HW_TEST=$(perf stat -e cycles /bin/true 2>&1 || true)
    if echo "$HW_TEST" | grep -q "cycles"; then
        if echo "$HW_TEST" | grep -q "<not supported>"; then
            warn "Hardware events: NOT supported (blocked by paranoia or permissions)"
        else
            ok "Hardware events: accessible (cycles visible)"
        fi
    else
        warn "Hardware events: probe inconclusive"
    fi

    # Software events (always available)
    SW_TEST=$(perf stat -e context-switches /bin/true 2>&1 || true)
    if echo "$SW_TEST" | grep -q "context-switches"; then
        ok "Software events: context-switches accessible"
    else
        warn "Software events: probe failed"
    fi

    # Uncore events
    UNCORE_TEST=$(perf stat -e uncore_imc_free_running/data_read/ /bin/true 2>&1 || true)
    if echo "$UNCORE_TEST" | grep -q "<not supported>"; then
        warn "Uncore IMC events: NOT accessible (requires root)"
    elif echo "$UNCORE_TEST" | grep -q "uncore_imc"; then
        ok "Uncore IMC events: accessible"
    else
        warn "Uncore IMC events: unavailable or blocked"
    fi
else
    fail "perf not found in PATH"
    warn "Install: sudo pacman -S perf  (or equivalent for your distro)"
fi
echo ""

# ── 4. System tools ───────────────────────────────────────────────────────
echo "[4] Required tools"
for cmd in taskset lscpu cmake; do
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd found"
    else
        warn "$cmd not found (optional but recommended)"
    fi
done
echo ""

# ── 5. CPU topology ───────────────────────────────────────────────────────
echo "[5] CPU topology"
CPU_MODEL=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)
CPU_COUNT=$(nproc)
ok "CPU: $CPU_MODEL"
ok "Logical CPUs: $CPU_COUNT"
GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [[ "$GOVERNOR" != "performance" ]]; then
    warn "CPU governor: $GOVERNOR (not 'performance' — frequency scaling may affect results)"
    warn "To fix (sets performance governor): echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
else
    ok "CPU governor: $GOVERNOR"
fi
TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "unknown")
if [[ "$TURBO" == "0" ]]; then
    warn "Turbo boost: ENABLED (may cause frequency variability)"
elif [[ "$TURBO" == "1" ]]; then
    ok "Turbo boost: disabled (stable frequency)"
else
    warn "Turbo boost: unknown"
fi
echo ""

# ── 6. Memory ─────────────────────────────────────────────────────────────
echo "[6] Memory"
MEM_TOTAL_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
MEM_AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
MEM_TOTAL_GB=$((MEM_TOTAL_KB / 1024 / 1024))
MEM_AVAIL_GB=$((MEM_AVAIL_KB / 1024 / 1024))
ok "Total RAM: ${MEM_TOTAL_GB} GiB"
if [[ "$MEM_AVAIL_GB" -lt 2 ]]; then
    warn "Available RAM: ${MEM_AVAIL_GB} GiB — may be too low for large models"
else
    ok "Available RAM: ${MEM_AVAIL_GB} GiB"
fi
echo ""

# ── 7. Google Benchmark (for microbench) ───────────────────────────────────
echo "[7] Google Benchmark"
GBENCH_HEADER="/usr/local/include/benchmark/benchmark.h"
GBENCH_LIB="/usr/local/lib/libbenchmark.a"
if [[ -f "$GBENCH_HEADER" ]]; then
    ok "Header found: $GBENCH_HEADER"
else
    warn "Header not found: $GBENCH_HEADER (memory-microbench will be skipped)"
fi
if [[ -f "$GBENCH_LIB" ]]; then
    ok "Library found: $GBENCH_LIB"
else
    warn "Library not found: $GBENCH_LIB"
fi
echo ""

# ── 8. Build directory ────────────────────────────────────────────────────
echo "[8] Build"
BUILD_DIR="$PROJECT_DIR/build"
if [[ -d "$BUILD_DIR" ]]; then
    if [[ -f "$BUILD_DIR/llm-charharness" ]]; then
        ok "Harness already built: $BUILD_DIR/llm-charharness"
    else
        warn "Build directory exists but harness not yet compiled"
        warn "Run: cmake -S . -B build && cmake --build build"
    fi
else
    warn "Build directory not found. Run:"
    warn "  cmake -S . -B build && cmake --build build"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "=== Summary ==="
echo "  PASSED   : $PASS"
echo "  WARNINGS : $WARN"
echo "  FAILURES : $FAIL"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
    echo "RESULT: FAIL — fix failures before running experiments"
    exit 1
elif [[ "$WARN" -gt 0 ]]; then
    echo "RESULT: WARN — environment is usable but check warnings above"
    exit 0
else
    echo "RESULT: PASS — environment is ready"
    exit 0
fi
