// types.hpp
// Shared data types for the llm-characterization harness.
// All measurements are clearly labelled with their source.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace llmchar {

// ── Source tags ────────────────────────────────────────────────────────────
// Every timing field must carry its source to avoid mixing measurement methods.
// "internal"  = from llama-bench JSON output (llama.cpp's own timers)
// "external"  = measured by this harness (wall clock / proc / perf)

// ── Memory measurements from /proc/<pid>/status ────────────────────────────
struct ProcMemSample {
    uint64_t vm_rss_kb      = 0; ///< Current RSS (kB)
    uint64_t vm_hwm_kb      = 0; ///< Peak RSS ever (kB) — VmHWM
    uint64_t vm_size_kb     = 0; ///< Virtual memory size (kB)
    uint64_t rss_anon_kb    = 0; ///< Anonymous (heap/stack) resident (kB)
    uint64_t rss_file_kb    = 0; ///< File-backed (mmap'd model weights) resident (kB)
    uint64_t rss_shmem_kb   = 0; ///< Shared memory resident (kB)
    int64_t  timestamp_us   = 0; ///< Microseconds since epoch at sample time
};

// ── Aggregate memory across all samples for one run ────────────────────────
struct MemMeasurement {
    // Peak values (maximums observed across all samples)
    uint64_t peak_rss_kb          = 0; ///< Max VmRSS across all samples (kB)
    uint64_t peak_hwm_kb          = 0; ///< VmHWM at process exit (kB) — kernel-maintained peak
    uint64_t peak_vm_size_kb      = 0; ///< Max virtual size (kB)
    uint64_t peak_anon_kb         = 0; ///< Max anonymous resident (kB)
    uint64_t peak_file_kb         = 0; ///< Max file-backed resident (kB)
    int      sample_count         = 0; ///< How many /proc samples were taken
    bool     available            = false;
};

// ── perf stat counters for one run ─────────────────────────────────────────
struct PerfMeasurement {
    // Hardware events (may be unavailable at paranoia > 1)
    std::optional<int64_t> cycles;              ///< CPU cycles consumed by workload
    std::optional<int64_t> instructions;        ///< Instructions retired
    std::optional<int64_t> cache_references;    ///< LLC references
    std::optional<int64_t> cache_misses;        ///< LLC misses
    std::optional<int64_t> branch_instructions; ///< Branch instructions
    std::optional<int64_t> branch_misses;       ///< Branch mispredictions

    // Software events (always available)
    std::optional<int64_t> context_switches;    ///< Context switches (SW event)
    std::optional<int64_t> cpu_migrations;      ///< CPU migrations (SW event)

    // Derived (when both cycles and instructions available)
    std::optional<double>  ipc;                 ///< Instructions per cycle

    bool     hardware_available = false; ///< Were hardware counters accessible?
    bool     available          = false; ///< Was perf stat available at all?
    std::string raw_output;             ///< Raw perf stat stderr for audit
};

// ── Timing from llama-bench JSON (internal source) ─────────────────────────
struct LlamaBenchResult {
    // These come directly from llama-bench JSON — "internal" source
    int64_t  avg_ns        = 0;   ///< Average test time in nanoseconds (internal)
    int64_t  stddev_ns     = 0;   ///< Std dev of test time (internal)
    double   avg_ts        = 0.0; ///< Average tokens/sec (internal)
    double   stddev_ts     = 0.0; ///< Std dev of tokens/sec (internal)
    std::vector<int64_t> samples_ns;  ///< Per-repetition ns (internal raw data)
    std::vector<double>  samples_ts;  ///< Per-repetition tok/s (internal raw data)

    // Model info from llama-bench JSON
    std::string model_filename;
    std::string model_type;   ///< e.g. "gemma3 270M Q8_0"
    int64_t     model_size    = 0; ///< Model file size in bytes (NOT memory usage)
    int64_t     model_n_params= 0;
    int         n_threads     = 0;
    int         n_batch       = 0;
    int         n_ubatch      = 0;
    int         n_prompt      = 0; ///< 0 for gen-only runs
    int         n_gen         = 0; ///< 0 for prompt-only runs
    std::string build_commit;
    int         build_number  = 0;
    std::string cpu_info;
    std::string test_time;    ///< ISO8601 from llama-bench
    std::string load_mode;    ///< mmap/mlock/none

    bool parsed = false;
};

// ── Wall-clock timing (external source) ────────────────────────────────────
struct WallTiming {
    int64_t start_ns          = 0; ///< Process start (monotonic, external)
    int64_t end_ns            = 0; ///< Process end (monotonic, external)
    int64_t elapsed_ns        = 0; ///< end - start (external wall clock)
    bool    available         = false;
};

// ── Complete result for one run (one execution of llama-bench) ─────────────
struct RunResult {
    std::string experiment_id; ///< e.g. "exp_20260815_103000"
    int         run_id        = 0; ///< 0-indexed within experiment
    bool        is_warmup     = false;
    bool        success       = false;
    std::string failure_reason;

    // Configuration recorded for traceability
    std::string model_name;
    std::string model_path;
    std::string quantization;
    int         threads       = 0;
    std::string cpu_affinity;  ///< e.g. "0-7" or "none"
    int         context_size  = 0;
    int         batch_size    = 0;
    int         ubatch_size   = 0;
    int         n_prompt      = 0;
    int         n_gen         = 0;

    // Child process info
    int         child_pid     = -1;
    int         exit_code     = -1;

    // Measurements
    WallTiming        wall;    ///< External wall-clock
    LlamaBenchResult  bench;   ///< llama-bench internal timing + model info
    MemMeasurement    mem;     ///< /proc RSS measurements
    PerfMeasurement   perf;    ///< perf stat counters

    // Timestamp
    std::string timestamp_iso; ///< Wall clock ISO8601 at run start
};

// ── Aggregate statistics for a set of runs ─────────────────────────────────
struct AggregateStats {
    double mean   = 0.0;
    double median = 0.0;
    double stddev = 0.0;
    double min    = 0.0;
    double max    = 0.0;
    double p50    = 0.0;
    double p95    = 0.0;
    double p99    = 0.0;
    int    n      = 0;     ///< Number of samples
};

// ── Experiment-level configuration (loaded from YAML) ──────────────────────
struct ModelConfig {
    std::string name;
    std::string path;
    std::string quantization; ///< Informational label (model is pre-quantized)
};

struct ExperimentConfig {
    std::string llama_bench_binary;
    std::string llama_cli_binary;  ///< For future use with llama-cli
    std::string output_dir;

    std::vector<ModelConfig> models;
    std::vector<int>         thread_counts;
    std::vector<std::string> cpu_affinities; ///< "none" | "0-3" | "4-7" etc.

    int warmup_runs      = 2;
    int measurement_runs = 5;
    int timeout_seconds  = 600; ///< Per-run timeout

    // Workload parameters
    int n_prompt  = 512;  ///< Tokens for prompt processing test
    int n_gen     = 128;  ///< Tokens to generate
    int batch_size    = 512;
    int ubatch_size   = 512;
    std::string load_mode = "mmap"; ///< mmap|mlock|none

    // perf configuration
    bool perf_enabled          = true;
    bool perf_hardware_events  = true; ///< Try hardware counters (may require root)
    bool perf_uncore_events    = false; ///< Try uncore DRAM events (requires root)
};

// ── System environment snapshot ────────────────────────────────────────────
struct CpuCoreInfo {
    int  cpu_id   = 0;
    int  core_id  = 0;
    bool is_p_core = false; ///< Heuristic: core_id < 32 → P-core for i7-14700HX
    int  min_freq_khz = 0;
    int  max_freq_khz = 0;
    std::string scaling_governor;
};

struct SystemInfo {
    std::string cpu_model;
    int         cpu_count       = 0;
    int         p_core_count    = 0;
    int         e_core_count    = 0;
    std::vector<CpuCoreInfo> cores;
    bool        turbo_enabled   = false;
    std::string scaling_governor; ///< Governor of cpu0 (representative)

    std::string kernel_version;
    std::string os_distribution;
    uint64_t    ram_total_kb    = 0;
    uint64_t    ram_available_kb= 0;

    std::string llama_build_commit;
    int         llama_build_number = 0;
    std::string llama_build_type;
    std::string llama_binary_path;
    std::string compiler_version;

    std::string perf_version;
    int         perf_event_paranoia = -1; ///< -1 if unreadable
    bool        perf_hw_available   = false;
    bool        perf_uncore_available = false;

    std::string l1d_cache;
    std::string l1i_cache;
    std::string l2_cache;
    std::string l3_cache;

    std::string timestamp_iso; ///< When collected
};

} // namespace llmchar
