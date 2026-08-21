// perf_monitor.hpp
// Runs `perf stat` as a child process attached to a target PID.
// Collects hardware and software performance counters.
//
// Graceful degradation:
// - If perf is unavailable: marks as unavailable, continues.
// - If hardware events are blocked (paranoia > 1): software events still collected.
// - If a specific counter shows "<not supported>": recorded as std::nullopt.
//
// We NEVER fabricate counter values. Missing data is always explicitly marked.

#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace llmchar {

class PerfMonitor {
public:
    /// hw_available: whether to try hardware events (cycles, instructions, etc.)
    /// uncore_available: whether to try uncore IMC events (requires root)
    PerfMonitor(bool hw_available, bool uncore_available);

    /// Build the event string for the perf stat command.
    std::string build_event_string() const;

    /// Start perf stat attached to target_pid.
    /// Returns true if perf was started successfully.
    bool start(pid_t target_pid);

    /// Stop perf stat and collect results.
    /// Must be called after the target process exits.
    void stop();

    /// Get the parsed measurement. Call after stop().
    PerfMeasurement get_result() const;

    PerfMonitor(const PerfMonitor&) = delete;
    PerfMonitor& operator=(const PerfMonitor&) = delete;

private:
    bool parse_perf_output(const std::string& output, PerfMeasurement& m) const;
    std::optional<int64_t> parse_counter(const std::string& output,
                                          const std::string& event_name) const;

    bool hw_available_;
    bool uncore_available_;
    pid_t perf_pid_   = -1;
    int   pipe_fd_    = -1; // read end of stderr pipe
    PerfMeasurement result_;
};

} // namespace llmchar
