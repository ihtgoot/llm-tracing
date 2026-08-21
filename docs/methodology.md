# llm-characterization — Measurement Methodology

## Overview

This document describes the measurement methodology for the **Baseline Characterization Harness** for CPU-based LLM inference. Its purpose is to establish a reproducible, scientifically valid dataset before implementing ProfInfer eBPF telemetry.

The research question being answered by this dataset:

> How do (model × quantization × thread count × CPU affinity × workload size) affect inference latency, throughput, and memory/cache behavior on a hybrid Intel CPU?

---

## System Under Test

| Property | Value |
|----------|-------|
| CPU | Intel Core i7-14700HX (Raptor Lake-HX) |
| Topology | 8 P-cores (HT → 16 logical) + 12 E-cores = 28 logical |
| L3 cache | 33 MiB shared |
| RAM | 16 GiB |
| OS | CachyOS Linux (Arch-based), kernel 7.1.8 |
| llama.cpp | commit `4801e3c56` (build 10362) |
| **Build type** | **Debug** — see §Known Limitations |

---

## Measurement Layers

The harness implements a strict 3-layer separation to avoid mixing measurement sources:

### Layer A — Inference Workload

`llama-bench` is the **only inference engine**. It is treated as a black box. The harness does not reimplement inference.

`llama-bench` is invoked with explicit arguments:
```
llama-bench -m <model> -t <threads> -p <n_prompt> -n <n_gen> -r 1 -o json
```

`-r 1` means llama-bench does one internal repetition per harness invocation. The harness controls repetitions externally to allow telemetry collection per run.

### Layer B — Experiment Orchestration (C++ harness)

The harness:
1. Parses `config/experiment.yaml`
2. Sets CPU affinity via `sched_setaffinity()` in the child process before exec
3. fork/exec's `llama-bench`
4. Notifies telemetry (Layer C) of the child PID immediately after fork
5. Waits for exit via `waitpid()`
6. Parses `llama-bench` JSON from stdout
7. Collects telemetry results
8. Writes to CSV/JSON

### Layer C — Telemetry

| Source | Metric | Notes |
|--------|--------|-------|
| `llama-bench` JSON stdout | tokens/sec, avg_ns, samples_ns[] | **Internal timing — labeled `[source:llama-bench]`** |
| Wall clock (`CLOCK_MONOTONIC`) | Process lifetime | **External — includes model load** |
| `/proc/<pid>/status` | VmRSS, VmHWM, VmSize, RssAnon, RssFile | Polled every 100ms in background thread |
| `perf stat -p <pid>` | cycles, instructions, cache-refs, cache-misses, ctx-switches, migrations | Attached to workload PID |

---

## Timing Measurements

### What is measured

| Field | Source | Boundary |
|-------|--------|---------|
| `bench_avg_ns` | llama-bench internal | prompt processing OR token generation (separate tests) |
| `bench_avg_ts` | llama-bench internal | tokens/second for the measured phase |
| `wall_elapsed_ns` | harness CLOCK_MONOTONIC | fork() → waitpid() — includes model load, warmup, and test |

### What is NOT done
- Internal llama-bench timing is NOT mixed with external wall-clock timing
- No attempt to subtract model load time from wall-clock timing
- Phase separation (load vs. PP vs. TG) is derived from `llama-bench`'s own JSON fields (`n_prompt`, `n_gen`)

---

## Memory Measurements

### Sources
- `/proc/<pid>/status`: sampled every 100ms during the process lifetime
- Values in **kilobytes (kB)** as reported by the Linux kernel

### Fields and their meanings

| Field | Kernel name | Meaning |
|-------|-------------|---------|
| `mem_peak_hwm_kb` | `VmHWM` | **Kernel-maintained peak RSS** — the definitive resident memory peak. Read from last sample before exit. |
| `mem_peak_rss_kb` | `VmRSS` | Maximum `VmRSS` observed across all samples. May differ from VmHWM if peak occurred between samples. |
| `mem_peak_vm_size_kb` | `VmSize` | Maximum virtual address space size. Includes mmap'd regions that may not be resident. |
| `mem_peak_anon_kb` | `RssAnon` | Maximum anonymous (heap + stack) resident memory. Represents allocator allocations and KV cache. |
| `mem_peak_file_kb` | `RssFile` | Maximum file-backed resident memory. For mmap load mode, this is the model weights loaded from disk. |

### Important distinctions

| Concept | How measured |
|---------|-------------|
| Model file size | `stat` on the GGUF file — NOT memory usage |
| Memory-mapped weights | `RssFile` — loaded-on-access pages |
| Resident heap/KV-cache | `RssAnon` |
| True peak RSS | `VmHWM` |
| Virtual address space | `VmSize` — includes all mmap regions |

**Model file size ≠ memory usage.** With mmap loading, only accessed pages are resident. With mlock, all pages are pinned.

---

## perf Measurements

### How perf is attached

```
perf stat -p <child_pid> -e <event_list>
```

Attaches to the workload PID after fork, before the workload has initialized. There is an unavoidable startup gap of ~10–50ms between the harness spawning perf and perf actually attaching. For inference runs of 30–300 seconds, this gap is negligible.

### Event selection

| Event | Type | Notes |
|-------|------|-------|
| `cycles` | Hardware | CPU cycles consumed by workload threads |
| `instructions` | Hardware | Retired instructions |
| `cache-references` | Hardware | LLC (L3) cache references |
| `cache-misses` | Hardware | LLC cache misses |
| `branch-instructions` | Hardware | Branch instructions |
| `branch-misses` | Hardware | Branch mispredictions |
| `context-switches` | Software | OS context switches (always available) |
| `cpu-migrations` | Software | Times process migrated to a different CPU |

### Hardware event availability

Hardware events require `perf_event_paranoia ≤ 1`. The harness probes this at startup.

If hardware events are unavailable:
- The harness continues with software events only
- Hardware counter fields in CSV are left empty (not 0, not fabricated)
- A warning is printed at startup

```bash
# Enable hardware events without root:
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoia
```

### DRAM bandwidth (uncore IMC events)

The i7-14700HX exposes `uncore_imc_free_running/data_read|write|total/` events. These are available only with root or `CAP_PERFMON`. They are **not enabled by default** in the config (`perf_uncore_events: false`).

If DRAM bandwidth is not measured, this is explicitly recorded. No bandwidth estimate is derived from execution time.

---

## CPU Affinity

CPU affinity is applied via `sched_setaffinity()` in the child process before `execv()`. This avoids the extra process overhead of using `taskset`.

### P-core / E-core classification

`/sys/devices/system/cpu/cpu*/topology/core_type` returns N/A on this kernel build. Classification uses the `core_id` heuristic:

- **core_id < 32** → P-core (CPUs 0–15 on i7-14700HX)
- **core_id ≥ 32** → E-core (CPUs 16–27 on i7-14700HX)

This is **documented as a heuristic**. Future kernel builds may expose `core_type` directly.

### Affinity configurations

| Config string | CPUs | Core type |
|---------------|------|-----------|
| `none` | all 0–27 | mixed |
| `0-7` | CPUs 0–7 | 4 P-core pairs (HT) |
| `0-3` | CPUs 0–3 | 2 P-core pairs |
| `16-23` | CPUs 16–23 | 4 E-cores |

---

## Warmup and Measurement Runs

Every configuration runs:
1. `warmup_runs` executions — recorded in CSV with `is_warmup=1`
2. `measurement_runs` executions — used for aggregate statistics

Warmup runs are **never discarded silently**. They are written to `results/raw/runs.csv` with `is_warmup=1`.

Rationale: the first run after a cold start may load the model from disk into the page cache. Subsequent runs benefit from OS page cache. The warmup/measurement separation allows downstream analysis to study cache warming effects.

---

## Statistical Summary

For each (model, threads, affinity) configuration, the aggregate CSV contains:

| Statistic | Description |
|-----------|-------------|
| mean | Arithmetic mean of measurement runs |
| median | 50th percentile |
| stddev | Sample standard deviation |
| min / max | Extremes |
| p50 | 50th percentile (= median) |
| p95 | 95th percentile — important for latency stability |
| p99 | 99th percentile — tail latency |

Raw per-run data is always preserved in `results/raw/runs.csv`.

---

## Known Limitations

### 1. Debug Build ⚠️

**The existing llama.cpp installation is a Debug build** (`CMAKE_BUILD_TYPE=Debug`, `-g`, no optimization). Performance numbers from this build are **not representative of production inference**. Throughput may be 3–10× lower than a Release build.

Every result records `llama_build_type=Debug`. This limitation is fundamental to the current installation and must be noted in any analysis.

**Mitigation**: rebuild llama.cpp as Release or RelWithDebInfo for scientifically meaningful throughput numbers. The harness supports any llama-bench binary — update `llama_bench_binary` in `experiment.yaml`.

### 2. CPU Frequency Scaling

The CPU governor is `powersave`. CPU frequency scales dynamically based on load. This introduces measurement variability.

The harness records `scaling_governor` in experiment metadata. The harness does **not** change the governor automatically. To reduce variability:
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 3. Turbo Boost

Turbo boost is enabled (`intel_pstate/no_turbo=0`). The CPU may boost to 5.5 GHz for short bursts then throttle. This can cause run-to-run timing variability.

### 4. Thermal Throttling

On a laptop, sustained inference loads may cause thermal throttling. The harness does not measure CPU temperature. If p95/p99 latencies are significantly higher than median, thermal throttling is a candidate explanation.

### 5. perf Attachment Gap

`perf stat -p <pid>` attaches after the child process starts. There is an inherent ~10–50ms gap where the early startup of llama-bench is not captured. For runs lasting 30+ seconds, this is negligible.

### 6. /proc Sampling Granularity

Memory is sampled every 100ms. The true peak may occur between samples. `VmHWM` (kernel-maintained peak) is more reliable than `max(VmRSS)` for capturing the true resident memory peak.

### 7. DRAM Bandwidth Not Measured by Default

Uncore IMC events require root. DRAM bandwidth is not measured in unprivileged mode. This is a documented limitation — not a fabricated alternative.

### 8. Single NUMA Node

The i7-14700HX has a single NUMA node. NUMA-related memory effects are not a factor in this experimental setup.

---

## Validation Checklist

Before declaring results trustworthy, verify:

- [ ] PID in metadata matches observed llama-bench process during run
- [ ] `VmHWM` in results is plausible (285 MB model → expect 300–600 MB depending on KV cache)
- [ ] perf counter values are nonzero for successful runs
- [ ] CPU affinity in result matches config (verify with `/proc/<pid>/status Cpus_allowed`)
- [ ] No orphaned llama-bench processes after harness exits (`ps aux | grep llama-bench`)
- [ ] Raw CSV has one row per run including warmup runs
- [ ] Metadata JSON contains build_commit and build_type
- [ ] Failed runs recorded with failure_reason, not silently omitted
- [ ] p95/p99 latencies are within 2× of median (if not, investigate noise sources)

---

## Research Context

This dataset is a **baseline characterization study**. It establishes the relationship between inference configuration and performance under normal conditions.

The eventual research progression:

```
Baseline characterization (THIS HARNESS)
        ↓
Identify telemetry/performance relationships
        ↓
Implement ProfInfer eBPF telemetry
        ↓
Validate eBPF telemetry against baseline
        ↓
Introduce controlled memory contention
        ↓
Characterize latency instability
        ↓
Develop runtime adaptation controller
        ↓
Evaluate adaptive policy
```

The harness is designed to remain useful after ProfInfer is implemented — the CSV output format includes sufficient fields to correlate with eBPF telemetry.
