# llm-characterization

Reproducible baseline characterization harness for CPU-based LLM inference.
Wraps `llama-bench` as a black-box workload and adds external telemetry collection.

Part of the **Kernel-Telemetry-Guided Runtime Adaptation for Stable CPU LLM Inference** research project.

---

## Quick Start

```bash
# 1. Check prerequisites
bash scripts/check_environment.sh

# 2. Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)

# 3. Run one experiment (first milestone: 1 model, 4 threads, 2 warmup + 5 measurements)
./build/llm-charharness --config config/experiment.yaml

# 4. Inspect results
cat results/raw/runs.csv
cat results/aggregate/summary.csv
cat results/metadata/*.json
```

---

## Architecture

```
Layer A  — llama-bench (inference workload, unmodified)
Layer B  — C++ harness (orchestration: fork/exec, affinity, timing)
Layer C  — Telemetry: /proc/<pid>/status + perf stat -p <pid>
```

The harness treats `llama-bench` as the system under test. It does **not** reimplement inference.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| CPU | x86-64, Linux |
| llama-bench | `/home/ihtgoot/project/llama.cpp/build/bin/llama-bench` |
| perf | `perf_event_paranoia ≤ 1` for hardware counters |
| Model | Any GGUF model; see `config/experiment.yaml` |
| Compiler | GCC or Clang with C++17 |
| Google Benchmark | `/usr/local/lib/libbenchmark.a` (for memory microbench) |

---

## Directory Structure

```
llm-characterization/
├── CMakeLists.txt          Build system
├── config/
│   └── experiment.yaml     Experiment configuration
├── src/
│   ├── main.cpp            Orchestration loop
│   ├── config.cpp/.hpp     YAML config parser (no external dependencies)
│   ├── launcher.cpp/.hpp   fork/exec llama-bench, CPU affinity
│   ├── process_monitor.cpp/.hpp   /proc/<pid>/status sampler
│   ├── perf_monitor.cpp/.hpp      perf stat subprocess
│   ├── system_info.cpp/.hpp       Environment metadata collection
│   ├── result_writer.cpp/.hpp     CSV + JSON output
│   └── types.hpp           Shared data types
├── microbench/
│   └── memory_bench.cpp    Google Benchmark memory hierarchy microbench
├── scripts/
│   ├── check_environment.sh   Prerequisite validator
│   └── collect_system_info.sh System snapshot to JSON
├── results/
│   ├── raw/runs.csv           One row per run (including warmup)
│   ├── aggregate/summary.csv  Statistics (mean/median/stddev/p50/p95/p99)
│   └── metadata/              System info + experiment config JSON
└── docs/
    └── methodology.md         Measurement methodology and limitations
```

---

## Measured Metrics

| Metric | Source | Label in CSV |
|--------|--------|-------------|
| Tokens/sec | llama-bench JSON | `bench_avg_ts [source:llama-bench]` |
| Run time (ns) | llama-bench JSON | `bench_avg_ns [source:llama-bench]` |
| Process wall time | CLOCK_MONOTONIC | `wall_elapsed_ns [source:harness]` |
| Peak RSS | /proc/status VmHWM | `mem_peak_hwm_kb [source:/proc/status VmHWM]` |
| Anonymous memory | /proc/status RssAnon | `mem_peak_anon_kb` |
| File-backed memory | /proc/status RssFile | `mem_peak_file_kb` |
| CPU cycles | perf stat | `perf_cycles` |
| Instructions | perf stat | `perf_instructions` |
| Cache misses | perf stat | `perf_cache_misses` |
| Context switches | perf stat | `perf_context_switches` |
| CPU migrations | perf stat | `perf_cpu_migrations` |

All measurement sources are **explicitly labeled** in CSV column headers and aggregate metric names.

---

## Enabling perf Hardware Counters

By default `perf_event_paranoia` may block hardware counters. To enable:

```bash
# Check current value
sysctl kernel.perf_event_paranoia

# Enable hardware events without root (temporary, resets on reboot)
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoia
```

The harness probes this at startup and warns if hardware counters are unavailable.
Software events (context-switches, cpu-migrations) are always collected.

---

## Important Notes

### Debug Build Warning

The installed llama.cpp (`build/bin/llama-bench`) is a **Debug build** (no optimization).
All inference throughput numbers will be significantly lower than a Release build.
This is recorded in `llama_build_type` in all experiment metadata.

### Memory: model size ≠ memory usage

`model_size_bytes` in the CSV is the GGUF file size on disk.
`mem_peak_hwm_kb` is the actual resident memory peak.
With `mmap` loading, only accessed pages are resident — memory usage is always lower than file size initially.

---

## Expanding the Experiment Matrix

Edit `config/experiment.yaml`:

```yaml
experiments:
  threads:
    - 1
    - 4
    - 8
    - 16          # Step 13: add more thread counts

  affinity:
    - "none"
    - "0-7"       # Step 14+: add P-core pinning
    - "16-23"     # E-core pinning

  models:
    - name: "my-model-Q4"
      path: "/path/to/model-q4.gguf"
      quantization: "Q4_K_M"
    - name: "my-model-Q8"
      path: "/path/to/model-q8.gguf"
      quantization: "Q8_0"
```

---

## Memory Microbenchmark

Separately characterizes the memory hierarchy:

```bash
./build/memory-microbench --benchmark_format=json > results/raw/memory_microbench.json
```

Benchmarks: sequential read, sequential write, random read, pointer-chase latency, memcpy.
Working sets: 16 KiB to 512 MiB (L1 → L2 → L3 → DRAM).

**Results are kept separate from inference results.**

---

## See Also

- [`docs/methodology.md`](docs/methodology.md) — Detailed measurement methodology, limitations, validation checklist
- [`config/experiment.yaml`](config/experiment.yaml) — Experiment configuration with comments
# llm-tracing
# llm-tracing
# llm-tracing
