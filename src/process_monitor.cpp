// process_monitor.cpp
// Samples /proc/<pid>/status on a background thread.
// Tracks VmRSS, VmHWM, VmSize, RssAnon, RssFile across the process lifetime.

#include "process_monitor.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <ctime>

#include <sys/types.h>
#include <signal.h>

namespace llmchar {

namespace {

static uint64_t parse_kb(const std::string& val) {
    // val is e.g. "12345 kB" or just "12345"
    try { return std::stoull(val); } catch(...) { return 0; }
}

static int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        system_clock::now().time_since_epoch()).count();
}

} // namespace

// ── ProcessMonitor ─────────────────────────────────────────────────────────

ProcessMonitor::ProcessMonitor(pid_t pid, int sample_interval_ms)
    : pid_(pid), interval_ms_(sample_interval_ms) {}

ProcessMonitor::~ProcessMonitor() {
    stop();
}

void ProcessMonitor::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&ProcessMonitor::sample_loop, this);
}

void ProcessMonitor::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
}

bool ProcessMonitor::read_proc_status(ProcMemSample& s) const {
    std::string path = "/proc/" + std::to_string(pid_) + "/status";
    std::ifstream f(path);
    if (!f.is_open()) return false; // process may have exited

    s.timestamp_us = now_us();
    std::string line;
    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val_str = line.substr(colon + 1);
        // Strip leading whitespace and " kB" suffix
        size_t start = val_str.find_first_not_of(" \t");
        if (start != std::string::npos) val_str = val_str.substr(start);
        size_t space = val_str.find(' ');
        if (space != std::string::npos) val_str = val_str.substr(0, space);

        if      (key == "VmRSS")   s.vm_rss_kb    = parse_kb(val_str);
        else if (key == "VmHWM")   s.vm_hwm_kb    = parse_kb(val_str);
        else if (key == "VmSize")  s.vm_size_kb   = parse_kb(val_str);
        else if (key == "RssAnon") s.rss_anon_kb  = parse_kb(val_str);
        else if (key == "RssFile") s.rss_file_kb  = parse_kb(val_str);
        else if (key == "RssShmem")s.rss_shmem_kb = parse_kb(val_str);
    }
    return true;
}

void ProcessMonitor::sample_loop() {
    while (running_.load(std::memory_order_acquire)) {
        ProcMemSample s{};
        bool ok = read_proc_status(s);
        if (ok && s.vm_rss_kb > 0) {
            std::lock_guard<std::mutex> lk(mutex_);
            samples_.push_back(s);
        }
        if (!ok) {
            // Process exited — stop sampling
            break;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(interval_ms_));
    }
}

MemMeasurement ProcessMonitor::get_result() const {
    std::lock_guard<std::mutex> lk(mutex_);
    MemMeasurement m;
    m.sample_count = static_cast<int>(samples_.size());
    if (samples_.empty()) {
        m.available = false;
        return m;
    }
    m.available = true;

    for (auto& s : samples_) {
        m.peak_rss_kb    = std::max(m.peak_rss_kb,    s.vm_rss_kb);
        m.peak_hwm_kb    = std::max(m.peak_hwm_kb,    s.vm_hwm_kb);
        m.peak_vm_size_kb= std::max(m.peak_vm_size_kb, s.vm_size_kb);
        m.peak_anon_kb   = std::max(m.peak_anon_kb,   s.rss_anon_kb);
        m.peak_file_kb   = std::max(m.peak_file_kb,   s.rss_file_kb);
    }
    // VmHWM is kernel-maintained — the last sample has the definitive peak
    m.peak_hwm_kb = samples_.back().vm_hwm_kb;

    return m;
}

std::vector<ProcMemSample> ProcessMonitor::get_samples() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return samples_;
}

// ── One-shot convenience ────────────────────────────────────────────────────

MemMeasurement read_proc_status_once(pid_t pid) {
    std::string path = "/proc/" + std::to_string(pid) + "/status";
    std::ifstream f(path);
    MemMeasurement result;
    if (!f.is_open()) return result;

    result.available = true;
    std::string line;
    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val_str = line.substr(colon + 1);
        size_t start = val_str.find_first_not_of(" \t");
        if (start != std::string::npos) val_str = val_str.substr(start);
        size_t space = val_str.find(' ');
        if (space != std::string::npos) val_str = val_str.substr(0, space);

        uint64_t v = 0;
        try { v = std::stoull(val_str); } catch(...) {}

        if      (key == "VmRSS")   result.peak_rss_kb     = v;
        else if (key == "VmHWM")   result.peak_hwm_kb     = v;
        else if (key == "VmSize")  result.peak_vm_size_kb  = v;
        else if (key == "RssAnon") result.peak_anon_kb    = v;
        else if (key == "RssFile") result.peak_file_kb    = v;
    }
    result.sample_count = 1;
    return result;
}

} // namespace llmchar
