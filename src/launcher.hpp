// launcher.hpp
// Launches llama-bench as a child process with controlled configuration.
// Handles: CPU affinity, environment setup, stdout capture, timeout.
//
// Design:
//   - The launcher is responsible for the entire lifecycle of one llama-bench run.
//   - It fork/exec's llama-bench with explicit arguments.
//   - CPU affinity is applied to the child process using sched_setaffinity()
//     before exec (not taskset, to avoid an extra process layer).
//   - stdout is captured via pipe to get the JSON output.
//   - stderr is forwarded to our stderr (llama.cpp logs are useful for debugging).
//   - A timeout thread kills the child if it runs too long.

#pragma once

#include "types.hpp"
#include <string>
#include <vector>
#include <functional>

namespace llmchar {

struct LaunchResult {
    bool    success         = false;
    pid_t   child_pid       = -1;
    int     exit_code       = -1;
    bool    timed_out       = false;
    std::string stdout_output;   ///< Captured stdout (llama-bench JSON)
    std::string failure_reason;

    // Wall-clock timing (external — NOT llama-bench internal timing)
    int64_t wall_start_ns   = 0; ///< CLOCK_MONOTONIC nanoseconds at fork
    int64_t wall_end_ns     = 0; ///< CLOCK_MONOTONIC nanoseconds at waitpid
};

struct LaunchConfig {
    std::string binary_path;     ///< Full path to llama-bench
    std::string model_path;
    int         threads         = 4;
    int         n_prompt        = 512;
    int         n_gen           = 128;
    int         batch_size      = 512;
    int         ubatch_size     = 512;
    int         repetitions     = 1;  ///< llama-bench -r (internal repetitions)
    std::string load_mode       = "mmap";
    std::string cpu_affinity    = "none"; ///< "none" | "0-3" | "0,2,4" etc.
    int         timeout_seconds = 600;
    std::string lib_dir;  ///< LD_LIBRARY_PATH (directory containing llama libs)

    // Extra args (for future extensibility)
    std::vector<std::string> extra_args;
};

/// Build the CPU set from an affinity string like "0-3" or "0,2,4,6" or "none".
/// Returns false if the affinity string is invalid.
bool parse_cpu_affinity(const std::string& affinity_str,
                        std::vector<int>& cpu_list);

/// Launch llama-bench with the given configuration.
/// Blocks until the process exits or times out.
/// The on_pid callback is called immediately after fork with the child PID
/// (used by the caller to start telemetry monitors).
LaunchResult launch_llama_bench(
    const LaunchConfig& cfg,
    std::function<void(pid_t)> on_pid_ready = nullptr);

/// Parse llama-bench JSON output into a LlamaBenchResult.
/// Returns a result with parsed=false if parsing fails.
LlamaBenchResult parse_llama_bench_json(const std::string& json_output);

} // namespace llmchar
