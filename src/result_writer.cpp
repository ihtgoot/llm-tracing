// result_writer.cpp
// Writes experiment results to structured CSV and JSON files.
// CSV: one row per run (all raw data, including warmup runs flagged)
// JSON: experiment metadata and aggregates

#include "result_writer.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sys/stat.h>

namespace llmchar {

namespace {

static void ensure_dir(const std::string& path) {
    std::string tmp = path;
    for (size_t i = 1; i < tmp.size(); ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            mkdir(tmp.c_str(), 0755);
            tmp[i] = '/';
        }
    }
    mkdir(tmp.c_str(), 0755);
}

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

static std::string csv_escape(const std::string& s) {
    // Wrap in quotes if contains comma, quote, or newline
    if (s.find(',') == std::string::npos &&
        s.find('"') == std::string::npos &&
        s.find('\n') == std::string::npos) {
        return s;
    }
    std::string r = "\"";
    for (char c : s) {
        if (c == '"') r += "\"\"";
        else r += c;
    }
    r += "\"";
    return r;
}

} // anonymous namespace

// ── ResultWriter ────────────────────────────────────────────────────────────

ResultWriter::ResultWriter(const std::string& output_dir)
    : output_dir_(output_dir) {}

bool ResultWriter::initialize() {
    ensure_dir(output_dir_);
    ensure_dir(output_dir_ + "/raw");
    ensure_dir(output_dir_ + "/aggregate");
    ensure_dir(output_dir_ + "/metadata");

    raw_csv_path_       = output_dir_ + "/raw/runs.csv";
    aggregate_csv_path_ = output_dir_ + "/aggregate/summary.csv";
    return true;
}

std::string ResultWriter::opt_str(const std::optional<int64_t>& v) {
    return v.has_value() ? std::to_string(*v) : "";
}

std::string ResultWriter::opt_double_str(const std::optional<double>& v) {
    if (!v.has_value()) return "";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << *v;
    return ss.str();
}

void ResultWriter::ensure_raw_header() {
    if (raw_header_written_) return;
    std::ofstream f(raw_csv_path_, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "[writer] Cannot open " << raw_csv_path_ << "\n";
        return;
    }
    // CSV header — one row per llama-bench run
    f << "timestamp"
      << ",experiment_id"
      << ",run_id"
      << ",is_warmup"
      << ",success"
      << ",failure_reason"

      // Configuration
      << ",model_name"
      << ",model_path"
      << ",model_size_bytes"
      << ",model_type"
      << ",model_n_params"
      << ",quantization"
      << ",threads"
      << ",cpu_affinity"
      << ",n_prompt"
      << ",n_gen"
      << ",batch_size"
      << ",ubatch_size"
      << ",load_mode"

      // External wall-clock timing (source: harness)
      << ",wall_elapsed_ns"
      << ",exit_code"

      // llama-bench internal timing (source: llama-bench JSON)
      // These are for the token-generation test (n_gen > 0)
      << ",bench_avg_ns"          // average of internal repetitions, nanoseconds
      << ",bench_stddev_ns"       // std dev of internal repetitions
      << ",bench_avg_ts"          // average tokens/sec
      << ",bench_stddev_ts"       // std dev tokens/sec
      << ",bench_samples_ns"      // raw per-rep ns (semicolon-separated)
      << ",bench_samples_ts"      // raw per-rep tok/s (semicolon-separated)
      << ",bench_build_commit"

      // Memory measurements (source: /proc/<pid>/status)
      // VmHWM = kernel-maintained peak RSS (most reliable)
      // VmRSS = current RSS at peak sample
      << ",mem_peak_rss_kb"       // max VmRSS across samples (kB)
      << ",mem_peak_hwm_kb"       // VmHWM at process exit (kB) — definitive peak
      << ",mem_peak_vm_size_kb"   // max virtual memory size (kB)
      << ",mem_peak_anon_kb"      // max anonymous resident (heap+stack) (kB)
      << ",mem_peak_file_kb"      // max file-backed resident (mmap'd weights) (kB)
      << ",mem_sample_count"      // how many /proc samples taken

      // perf hardware counters (source: perf stat)
      // Empty = <not supported> or perf unavailable
      << ",perf_cycles"
      << ",perf_instructions"
      << ",perf_cache_references"
      << ",perf_cache_misses"
      << ",perf_branch_instructions"
      << ",perf_branch_misses"
      << ",perf_ipc"              // instructions / cycles

      // perf software counters
      << ",perf_context_switches"
      << ",perf_cpu_migrations"

      << "\n";
    raw_header_written_ = true;
}

void ResultWriter::write_run(const RunResult& run) {
    ensure_raw_header();
    std::ofstream f(raw_csv_path_, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "[writer] Cannot write to " << raw_csv_path_ << "\n";
        return;
    }

    // Build semicolon-separated samples_ns and samples_ts
    std::string samples_ns_str, samples_ts_str;
    for (size_t i = 0; i < run.bench.samples_ns.size(); ++i) {
        if (i) samples_ns_str += ";";
        samples_ns_str += std::to_string(run.bench.samples_ns[i]);
    }
    for (size_t i = 0; i < run.bench.samples_ts.size(); ++i) {
        if (i) samples_ts_str += ";";
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << run.bench.samples_ts[i];
        samples_ts_str += ss.str();
    }

    auto d2s = [](double v, int p = 6) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(p) << v;
        return ss.str();
    };

    f << csv_escape(run.timestamp_iso)
      << "," << csv_escape(run.experiment_id)
      << "," << run.run_id
      << "," << (run.is_warmup ? "1" : "0")
      << "," << (run.success ? "1" : "0")
      << "," << csv_escape(run.failure_reason)

      << "," << csv_escape(run.model_name)
      << "," << csv_escape(run.model_path)
      << "," << run.bench.model_size
      << "," << csv_escape(run.bench.model_type)
      << "," << run.bench.model_n_params
      << "," << csv_escape(run.quantization)
      << "," << run.threads
      << "," << csv_escape(run.cpu_affinity)
      << "," << run.bench.n_prompt
      << "," << run.bench.n_gen
      << "," << run.batch_size
      << "," << run.ubatch_size
      << "," << csv_escape(run.bench.load_mode)

      << "," << (run.wall.available ? std::to_string(run.wall.elapsed_ns) : "")
      << "," << run.exit_code

      << "," << run.bench.avg_ns
      << "," << run.bench.stddev_ns
      << "," << d2s(run.bench.avg_ts, 4)
      << "," << d2s(run.bench.stddev_ts, 4)
      << "," << csv_escape(samples_ns_str)
      << "," << csv_escape(samples_ts_str)
      << "," << csv_escape(run.bench.build_commit)

      << "," << (run.mem.available ? std::to_string(run.mem.peak_rss_kb)      : "")
      << "," << (run.mem.available ? std::to_string(run.mem.peak_hwm_kb)      : "")
      << "," << (run.mem.available ? std::to_string(run.mem.peak_vm_size_kb)  : "")
      << "," << (run.mem.available ? std::to_string(run.mem.peak_anon_kb)     : "")
      << "," << (run.mem.available ? std::to_string(run.mem.peak_file_kb)     : "")
      << "," << run.mem.sample_count

      << "," << opt_str(run.perf.cycles)
      << "," << opt_str(run.perf.instructions)
      << "," << opt_str(run.perf.cache_references)
      << "," << opt_str(run.perf.cache_misses)
      << "," << opt_str(run.perf.branch_instructions)
      << "," << opt_str(run.perf.branch_misses)
      << "," << opt_double_str(run.perf.ipc)

      << "," << opt_str(run.perf.context_switches)
      << "," << opt_str(run.perf.cpu_migrations)

      << "\n";
}

AggregateStats ResultWriter::compute_stats(const std::vector<double>& values) {
    AggregateStats s;
    if (values.empty()) return s;
    s.n = static_cast<int>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    s.min = sorted.front();
    s.max = sorted.back();

    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    s.mean = sum / sorted.size();

    // Median (p50)
    size_t mid = sorted.size() / 2;
    s.median = (sorted.size() % 2 == 0)
        ? (sorted[mid-1] + sorted[mid]) / 2.0
        : sorted[mid];
    s.p50 = s.median;

    // Percentiles
    auto percentile = [&](double p) {
        double idx = p * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = idx - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    };
    s.p95 = percentile(0.95);
    s.p99 = percentile(0.99);

    // Std dev
    double sq_sum = 0;
    for (double v : values) {
        double d = v - s.mean;
        sq_sum += d * d;
    }
    s.stddev = std::sqrt(sq_sum / values.size());

    return s;
}

void ResultWriter::ensure_aggregate_header() {
    if (aggregate_header_written_) return;
    std::ofstream f(aggregate_csv_path_, std::ios::app);
    if (!f.is_open()) return;
    f << "experiment_id"
      << ",model_name"
      << ",quantization"
      << ",threads"
      << ",cpu_affinity"
      << ",n_warmup"
      << ",n_measured"
      << ",metric"
      << ",mean"
      << ",median"
      << ",stddev"
      << ",min"
      << ",max"
      << ",p50"
      << ",p95"
      << ",p99"
      << "\n";
    aggregate_header_written_ = true;
}

void ResultWriter::write_aggregate(
    const std::string& experiment_id,
    const std::string& model_name,
    const std::string& quantization,
    int threads,
    const std::string& affinity,
    const std::vector<RunResult>& all_runs,
    int warmup_count)
{
    ensure_aggregate_header();
    std::ofstream f(aggregate_csv_path_, std::ios::app);
    if (!f.is_open()) return;

    // Filter to measurement runs only (skip warmup)
    std::vector<const RunResult*> measurement_runs;
    for (auto& r : all_runs) {
        if (!r.is_warmup && r.success) measurement_runs.push_back(&r);
    }
    if (measurement_runs.empty()) return;

    int n_measured = static_cast<int>(measurement_runs.size());
    std::string prefix =
        csv_escape(experiment_id) + "," +
        csv_escape(model_name) + "," +
        csv_escape(quantization) + "," +
        std::to_string(threads) + "," +
        csv_escape(affinity) + "," +
        std::to_string(warmup_count) + "," +
        std::to_string(n_measured) + ",";

    auto write_metric = [&](const std::string& name, const std::vector<double>& vals) {
        if (vals.empty()) return;
        auto s = compute_stats(vals);
        auto d2s = [](double v) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << v;
            return ss.str();
        };
        f << prefix << csv_escape(name)
          << "," << d2s(s.mean)
          << "," << d2s(s.median)
          << "," << d2s(s.stddev)
          << "," << d2s(s.min)
          << "," << d2s(s.max)
          << "," << d2s(s.p50)
          << "," << d2s(s.p95)
          << "," << d2s(s.p99)
          << "\n";
    };

    // Collect metric vectors
    // Source label is embedded in metric name

    // bench_avg_ts: token/s from llama-bench internal timing
    {
        std::vector<double> ts;
        for (auto* r : measurement_runs) ts.push_back(r->bench.avg_ts);
        write_metric("bench_tokens_per_sec [source:llama-bench]", ts);
    }
    // bench_avg_ns: average run time from llama-bench
    {
        std::vector<double> ns;
        for (auto* r : measurement_runs) ns.push_back(static_cast<double>(r->bench.avg_ns));
        write_metric("bench_avg_ns [source:llama-bench]", ns);
    }
    // wall_elapsed_ns: external wall-clock
    {
        std::vector<double> ns;
        for (auto* r : measurement_runs) {
            if (r->wall.available) ns.push_back(static_cast<double>(r->wall.elapsed_ns));
        }
        write_metric("wall_elapsed_ns [source:harness]", ns);
    }
    // Memory: peak_hwm_kb (definitive from kernel)
    {
        std::vector<double> v;
        for (auto* r : measurement_runs) {
            if (r->mem.available) v.push_back(static_cast<double>(r->mem.peak_hwm_kb));
        }
        write_metric("mem_peak_hwm_kb [source:/proc/status VmHWM]", v);
    }
    // Memory: peak_rss_kb
    {
        std::vector<double> v;
        for (auto* r : measurement_runs) {
            if (r->mem.available) v.push_back(static_cast<double>(r->mem.peak_rss_kb));
        }
        write_metric("mem_peak_rss_kb [source:/proc/status VmRSS]", v);
    }
    // perf cycles
    {
        std::vector<double> v;
        for (auto* r : measurement_runs) {
            if (r->perf.cycles) v.push_back(static_cast<double>(*r->perf.cycles));
        }
        write_metric("perf_cycles [source:perf stat]", v);
    }
    // perf cache misses
    {
        std::vector<double> v;
        for (auto* r : measurement_runs) {
            if (r->perf.cache_misses) v.push_back(static_cast<double>(*r->perf.cache_misses));
        }
        write_metric("perf_cache_misses [source:perf stat]", v);
    }
    // perf context switches
    {
        std::vector<double> v;
        for (auto* r : measurement_runs) {
            if (r->perf.context_switches) v.push_back(static_cast<double>(*r->perf.context_switches));
        }
        write_metric("perf_context_switches [source:perf stat]", v);
    }
    // perf cpu migrations
    {
        std::vector<double> v;
        for (auto* r : measurement_runs) {
            if (r->perf.cpu_migrations) v.push_back(static_cast<double>(*r->perf.cpu_migrations));
        }
        write_metric("perf_cpu_migrations [source:perf stat]", v);
    }
}

void ResultWriter::write_experiment_metadata(
    const std::string& experiment_id,
    const SystemInfo& sys,
    const ExperimentConfig& cfg)
{
    std::string path = output_dir_ + "/metadata/" + experiment_id + ".json";
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[writer] Cannot write metadata to " << path << "\n";
        return;
    }

    auto q = [](const std::string& s) {
        return "\"" + json_escape(s) + "\"";
    };

    f << "{\n";
    f << "  \"experiment_id\": " << q(experiment_id) << ",\n";
    f << "  \"system\": {\n";
    f << "    \"cpu_model\": " << q(sys.cpu_model) << ",\n";
    f << "    \"cpu_count\": " << sys.cpu_count << ",\n";
    f << "    \"p_core_count\": " << sys.p_core_count << ",\n";
    f << "    \"e_core_count\": " << sys.e_core_count << ",\n";
    f << "    \"scaling_governor\": " << q(sys.scaling_governor) << ",\n";
    f << "    \"turbo_enabled\": " << (sys.turbo_enabled ? "true" : "false") << ",\n";
    f << "    \"l3_cache\": " << q(sys.l3_cache) << ",\n";
    f << "    \"ram_total_kb\": " << sys.ram_total_kb << ",\n";
    f << "    \"kernel_version\": " << q(sys.kernel_version) << ",\n";
    f << "    \"os_distribution\": " << q(sys.os_distribution) << ",\n";
    f << "    \"llama_build_commit\": " << q(sys.llama_build_commit) << ",\n";
    f << "    \"llama_build_number\": " << sys.llama_build_number << ",\n";
    f << "    \"llama_build_type\": " << q(sys.llama_build_type) << ",\n";
    f << "    \"llama_binary_path\": " << q(sys.llama_binary_path) << ",\n";
    f << "    \"compiler_version\": " << q(sys.compiler_version) << ",\n";
    f << "    \"perf_version\": " << q(sys.perf_version) << ",\n";
    f << "    \"perf_event_paranoia\": " << sys.perf_event_paranoia << ",\n";
    f << "    \"perf_hw_available\": " << (sys.perf_hw_available ? "true" : "false") << ",\n";
    f << "    \"perf_uncore_available\": " << (sys.perf_uncore_available ? "true" : "false") << "\n";
    f << "  },\n";
    f << "  \"config\": {\n";
    f << "    \"llama_bench_binary\": " << q(cfg.llama_bench_binary) << ",\n";
    f << "    \"warmup_runs\": " << cfg.warmup_runs << ",\n";
    f << "    \"measurement_runs\": " << cfg.measurement_runs << ",\n";
    f << "    \"n_prompt\": " << cfg.n_prompt << ",\n";
    f << "    \"n_gen\": " << cfg.n_gen << ",\n";
    f << "    \"batch_size\": " << cfg.batch_size << ",\n";
    f << "    \"ubatch_size\": " << cfg.ubatch_size << ",\n";
    f << "    \"load_mode\": " << q(cfg.load_mode) << ",\n";
    f << "    \"models\": [\n";
    for (size_t i = 0; i < cfg.models.size(); ++i) {
        f << "      {\"name\":" << q(cfg.models[i].name)
          << ",\"path\":" << q(cfg.models[i].path)
          << ",\"quantization\":" << q(cfg.models[i].quantization) << "}";
        if (i+1 < cfg.models.size()) f << ",";
        f << "\n";
    }
    f << "    ],\n";
    f << "    \"thread_counts\": [";
    for (size_t i = 0; i < cfg.thread_counts.size(); ++i) {
        if (i) f << ",";
        f << cfg.thread_counts[i];
    }
    f << "],\n";
    f << "    \"cpu_affinities\": [";
    for (size_t i = 0; i < cfg.cpu_affinities.size(); ++i) {
        if (i) f << ",";
        f << q(cfg.cpu_affinities[i]);
    }
    f << "]\n";
    f << "  }\n";
    f << "}\n";

    std::cout << "[writer] Metadata written to " << path << "\n";
}

} // namespace llmchar
