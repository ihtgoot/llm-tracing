#!/usr/bin/env bash
# collect_system_info.sh
# Dumps comprehensive system environment info to stdout (JSON-like text).
# Run before experiments to create a permanent environment snapshot.
# Does NOT modify any system settings.

set -euo pipefail

LLAMA_BENCH="${1:-/home/ihtgoot/project/llama.cpp/build/bin/llama-bench}"
LIBDIR="$(dirname "$LLAMA_BENCH")"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "{"
echo "  \"collected_at\": \"$TIMESTAMP\","

# ── CPU ───────────────────────────────────────────────────────────────────
CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
CPU_SOCKETS=$(grep -c "physical id" /proc/cpuinfo 2>/dev/null | sort -u | wc -l || echo 1)
CPU_PHYS=$(grep "cpu cores" /proc/cpuinfo | head -1 | awk '{print $NF}' || echo "?")
CPU_LOGICAL=$(nproc)
CPU_FAMILY=$(grep "cpu family" /proc/cpuinfo | head -1 | awk '{print $NF}')
CPU_MODEL_NUM=$(grep "^model\s*:" /proc/cpuinfo | head -1 | awk '{print $NF}')
CPU_STEPPING=$(grep "stepping" /proc/cpuinfo | head -1 | awk '{print $NF}')
CPU_MICROCODE=$(grep "microcode" /proc/cpuinfo | head -1 | awk '{print $NF}')

echo "  \"cpu\": {"
echo "    \"model\": \"$CPU_MODEL\","
echo "    \"family\": $CPU_FAMILY,"
echo "    \"model_number\": $CPU_MODEL_NUM,"
echo "    \"stepping\": $CPU_STEPPING,"
echo "    \"microcode\": \"$CPU_MICROCODE\","
echo "    \"physical_cores_per_socket\": \"$CPU_PHYS\","
echo "    \"logical_cpus\": $CPU_LOGICAL,"

# P/E core topology (i7-14700HX heuristic: core_id < 32 = P-core)
P_CORES=0
E_CORES=0
for i in $(seq 0 $((CPU_LOGICAL-1))); do
    CTYPE=$(cat /sys/devices/system/cpu/cpu${i}/topology/core_id 2>/dev/null || echo "?")
    if [[ "$CTYPE" != "?" ]]; then
        if [[ "$CTYPE" -lt 32 ]]; then
            ((P_CORES++))
        else
            ((E_CORES++))
        fi
    fi
done
echo "    \"p_cores_logical\": $P_CORES,"
echo "    \"e_cores_logical\": $E_CORES,"
echo "    \"p_e_heuristic\": \"core_id<32=P-core (Raptor Lake i7-14700HX)\","

# Frequency
CPU_MAX_KHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
CPU_MIN_KHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq 2>/dev/null || echo 0)
GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
TURBO_NO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "unknown")
echo "    \"max_freq_khz\": $CPU_MAX_KHZ,"
echo "    \"min_freq_khz\": $CPU_MIN_KHZ,"
echo "    \"scaling_governor\": \"$GOVERNOR\","
echo "    \"intel_pstate_no_turbo\": \"$TURBO_NO\","

# Cache
L1D=$(lscpu 2>/dev/null | grep "L1d cache" | awk -F: '{print $2}' | xargs || echo "?")
L1I=$(lscpu 2>/dev/null | grep "L1i cache" | awk -F: '{print $2}' | xargs || echo "?")
L2=$(lscpu 2>/dev/null | grep "L2 cache" | awk -F: '{print $2}' | xargs || echo "?")
L3=$(lscpu 2>/dev/null | grep "L3 cache" | awk -F: '{print $2}' | xargs || echo "?")
echo "    \"l1d_cache\": \"$L1D\","
echo "    \"l1i_cache\": \"$L1I\","
echo "    \"l2_cache\": \"$L2\","
echo "    \"l3_cache\": \"$L3\""
echo "  },"

# ── Memory ────────────────────────────────────────────────────────────────
MEM_TOTAL=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
MEM_FREE=$(awk '/MemFree/ {print $2}' /proc/meminfo)
MEM_AVAIL=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
SWAP_TOTAL=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)

echo "  \"memory\": {"
echo "    \"mem_total_kb\": $MEM_TOTAL,"
echo "    \"mem_free_kb\": $MEM_FREE,"
echo "    \"mem_available_kb\": $MEM_AVAIL,"
echo "    \"swap_total_kb\": $SWAP_TOTAL"
echo "  },"

# ── Kernel / OS ───────────────────────────────────────────────────────────
KERNEL=$(uname -r)
OS_PRETTY=$(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"')
HOSTNAME=$(hostname)
ARCH=$(uname -m)

echo "  \"kernel\": {"
echo "    \"version\": \"$KERNEL\","
echo "    \"arch\": \"$ARCH\","
echo "    \"hostname\": \"$HOSTNAME\","
echo "    \"os\": \"$OS_PRETTY\""
echo "  },"

# ── llama.cpp ─────────────────────────────────────────────────────────────
LLAMA_VER=$(LD_LIBRARY_PATH="$LIBDIR" "$LLAMA_BENCH" --version 2>&1 | head -3 || echo "unavailable")
LLAMA_COMMIT=$(echo "$LLAMA_VER" | grep "version:" | grep -oP '\(\K[^)]+' || echo "unknown")
LLAMA_BUILD_NUM=$(echo "$LLAMA_VER" | grep "version:" | awk '{print $2}' || echo "0")
LLAMA_COMPILER=$(echo "$LLAMA_VER" | grep "built with" || echo "unknown")

# Build type from CMakeCache
LLAMA_BUILDDIR="$(dirname "$LIBDIR")"
CMAKE_CACHE="$LLAMA_BUILDDIR/CMakeCache.txt"
BUILD_TYPE="unknown"
if [[ -f "$CMAKE_CACHE" ]]; then
    BUILD_TYPE=$(grep "CMAKE_BUILD_TYPE:" "$CMAKE_CACHE" | cut -d= -f2 || echo "unknown")
fi

echo "  \"llama_cpp\": {"
echo "    \"binary\": \"$LLAMA_BENCH\","
echo "    \"build_commit\": \"$LLAMA_COMMIT\","
echo "    \"build_number\": \"$LLAMA_BUILD_NUM\","
echo "    \"build_type\": \"$BUILD_TYPE\","
echo "    \"compiler\": \"$LLAMA_COMPILER\","
echo "    \"cmake_cache\": \"$CMAKE_CACHE\""
echo "  },"

# ── perf ─────────────────────────────────────────────────────────────────
PERF_VER=$(perf --version 2>/dev/null || echo "not found")
PARANOIA=$(sysctl -n kernel.perf_event_paranoia 2>/dev/null || echo "-1")
HW_PROBE=$(perf stat -e cycles /bin/true 2>&1 || true)
if echo "$HW_PROBE" | grep -q "<not supported>"; then
    PERF_HW="false"
elif echo "$HW_PROBE" | grep -q "cycles"; then
    PERF_HW="true"
else
    PERF_HW="unknown"
fi

UNCORE_PROBE=$(perf stat -e uncore_imc_free_running/data_read/ /bin/true 2>&1 || true)
if echo "$UNCORE_PROBE" | grep -qE "<not supported>|Permission denied|No such file"; then
    PERF_UNCORE="false"
else
    PERF_UNCORE="true"
fi

echo "  \"perf\": {"
echo "    \"version\": \"$PERF_VER\","
echo "    \"perf_event_paranoia\": $PARANOIA,"
echo "    \"hw_events_available\": $PERF_HW,"
echo "    \"uncore_events_available\": $PERF_UNCORE"
echo "  },"

# ── NUMA ──────────────────────────────────────────────────────────────────
NUMA_NODES=$(ls /sys/devices/system/node/ 2>/dev/null | grep -c "^node[0-9]" || echo 1)
echo "  \"numa\": {"
echo "    \"nodes\": $NUMA_NODES"
echo "  }"

echo "}"
