// process_monitor.hpp
// Background thread that samples /proc/<pid>/status at regular intervals.
// Provides peak RSS, VmHWM (kernel-maintained peak), virtual size,
// and anonymous/file-backed breakdown.
//
// NOTE: All values are in kilobytes as reported by the kernel.
// Do NOT confuse these with model file size on disk.

#pragma once

#include "types.hpp"
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

namespace llmchar {

class ProcessMonitor {
public:
    /// Create a monitor for the given PID.
    /// sample_interval_ms: how often to sample /proc/<pid>/status.
    explicit ProcessMonitor(pid_t pid, int sample_interval_ms = 100);
    ~ProcessMonitor();

    /// Start background sampling thread.
    void start();

    /// Stop sampling and join the thread.
    void stop();

    /// Get the aggregate memory measurement.
    /// Call after stop().
    MemMeasurement get_result() const;

    /// Get all raw samples (for debugging / post-hoc analysis).
    std::vector<ProcMemSample> get_samples() const;

    ProcessMonitor(const ProcessMonitor&) = delete;
    ProcessMonitor& operator=(const ProcessMonitor&) = delete;

private:
    void sample_loop();
    bool read_proc_status(ProcMemSample& s) const;

    pid_t           pid_;
    int             interval_ms_;
    std::atomic<bool> running_{false};
    std::thread       thread_;

    mutable std::mutex          mutex_;
    std::vector<ProcMemSample>  samples_;
};

/// One-shot read of /proc/<pid>/status — convenience for post-mortem.
MemMeasurement read_proc_status_once(pid_t pid);

} // namespace llmchar
