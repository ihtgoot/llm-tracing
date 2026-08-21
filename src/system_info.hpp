// system_info.hpp
// Collect and expose system environment metadata for experiment reproducibility.

#pragma once

#include "types.hpp"
#include <string>

namespace llmchar {

/// Collect all system info. This is a one-time snapshot at experiment start.
SystemInfo collect_system_info(const std::string& llama_bench_binary);

/// Write SystemInfo as formatted JSON to a file.
void write_system_info_json(const SystemInfo& info, const std::string& path);

/// Print human-readable summary of system info.
void print_system_info(const SystemInfo& info);

/// Check perf_event_paranoia — returns -1 if unreadable.
int get_perf_event_paranoia();

/// Probe whether perf hardware counters are actually accessible.
/// Runs a tiny perf stat and checks for "<not supported>" in output.
bool probe_perf_hardware_events();

/// Probe whether uncore IMC events are accessible.
bool probe_perf_uncore_events();

} // namespace llmchar
