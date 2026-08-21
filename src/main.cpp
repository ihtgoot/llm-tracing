// main.cpp
// Entry point for the llm-characterization harness.
// Orchestrates: config loading → system info → experiment matrix → result writing.
//
// Usage:
//   ./llm-charharness [--config path/to/experiment.yaml] [--sysinfo-only]
//
// Layer B orchestration: for each (model × threads × affinity), runs
// warmup_runs + measurement_runs repetitions, collecting external telemetry
// (wall time, /proc RSS, perf stat) around each llama-bench invocation.

#include "config.hpp"
#include "system_info.hpp"
#include "launcher.hpp"
#include "process_monitor.hpp"
#include "perf_monitor.hpp"
#include "result_writer.hpp"
#include "types.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <csignal>
#include <atomic>
#include <thread>

namespace {

// ── Global signal handling ─────────────────────────────────────────────────
// Catches SIGTERM/SIGINT so we can clean up and flush results.
std::atomic<bool> g_shutdown{false};

void signal_handler(int sig) {
    std::cerr << "\n[main] Caught signal " << sig << " — shutting down gracefully...\n";
    g_shutdown.store(true, std::memory_order_release);
}

// ── Timestamp helpers ──────────────────────────────────────────────────────

static std::string iso8601_now() {
    time_t t = time(nullptr);
    struct tm tm_buf;
    gmtime_r(&t, &tm_buf);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

static std::string make_experiment_id() {
    time_t t = time(nullptr);
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char buf[32];
    strftime(buf, sizeof(buf), "exp_%Y%m%d_%H%M%S", &tm_buf);
    return buf;
}

// ── One-run execution ──────────────────────────────────────────────────────

static llmchar::RunResult execute_one_run(
    const llmchar::ExperimentConfig& cfg,
    const llmchar::ModelConfig& model,
    int threads,
    const std::string& affinity,
    const std::string& experiment_id,
    int run_id,
    bool is_warmup,
    const llmchar::SystemInfo& sysinfo)
{
    llmchar::RunResult result;
    result.experiment_id = experiment_id;
    result.run_id        = run_id;
    result.is_warmup     = is_warmup;
    result.model_name    = model.name;
    result.model_path    = model.path;
    result.quantization  = model.quantization;
    result.threads       = threads;
    result.cpu_affinity  = affinity;
    result.n_prompt      = cfg.n_prompt;
    result.n_gen         = cfg.n_gen;
    result.batch_size    = cfg.batch_size;
    result.ubatch_size   = cfg.ubatch_size;
    result.timestamp_iso = iso8601_now();

    std::cout << "\n[run " << run_id << (is_warmup ? " WARMUP" : "") << "] "
              << model.name << " t=" << threads << " affinity=" << affinity << "\n";

    // Build launch config
    llmchar::LaunchConfig launch;
    launch.binary_path      = cfg.llama_bench_binary;
    launch.model_path       = model.path;
    launch.threads          = threads;
    launch.n_prompt         = cfg.n_prompt;
    launch.n_gen            = cfg.n_gen;
    launch.batch_size       = cfg.batch_size;
    launch.ubatch_size      = cfg.ubatch_size;
    launch.repetitions      = 1;  // harness controls repetitions externally
    launch.load_mode        = cfg.load_mode;
    launch.cpu_affinity     = affinity;
    launch.timeout_seconds  = cfg.timeout_seconds;

    // lib_dir: directory containing llama shared libraries
    {
        std::string bin = cfg.llama_bench_binary;
        auto slash = bin.rfind('/');
        if (slash != std::string::npos)
            launch.lib_dir = bin.substr(0, slash);
    }

    // Set up perf monitor (before fork, so it's ready when PID arrives)
    std::unique_ptr<llmchar::PerfMonitor> perf_mon;
    if (cfg.perf_enabled) {
        perf_mon = std::make_unique<llmchar::PerfMonitor>(
            sysinfo.perf_hw_available,
            sysinfo.perf_uncore_available && cfg.perf_uncore_events);
    }

    std::unique_ptr<llmchar::ProcessMonitor> proc_mon;

    // Launch llama-bench; telemetry starts in on_pid_ready callback
    auto on_pid_ready = [&](pid_t child_pid) {
        result.child_pid = child_pid;
        std::cout << "  [pid=" << child_pid << "]\n";

        // Start /proc memory sampler
        proc_mon = std::make_unique<llmchar::ProcessMonitor>(child_pid, 100);
        proc_mon->start();

        // Start perf stat attached to child PID
        if (perf_mon) {
            if (!perf_mon->start(child_pid)) {
                std::cerr << "  [perf] Failed to start perf stat — skipping hw counters\n";
                perf_mon.reset();
            } else {
                std::cout << "  [perf] perf stat attached to pid " << child_pid << "\n";
            }
        }
    };

    llmchar::LaunchResult launch_result = llmchar::launch_llama_bench(launch, on_pid_ready);

    // Stop telemetry
    if (proc_mon) proc_mon->stop();
    if (perf_mon) perf_mon->stop();

    // Collect results
    result.exit_code = launch_result.exit_code;
    result.success   = launch_result.success;
    result.failure_reason = launch_result.failure_reason;

    // Wall clock
    result.wall.start_ns   = launch_result.wall_start_ns;
    result.wall.end_ns     = launch_result.wall_end_ns;
    result.wall.elapsed_ns = launch_result.wall_end_ns - launch_result.wall_start_ns;
    result.wall.available  = (launch_result.wall_start_ns > 0);

    // llama-bench JSON output
    if (!launch_result.stdout_output.empty()) {
        result.bench = llmchar::parse_llama_bench_json(launch_result.stdout_output);
        if (!result.bench.parsed) {
            std::cerr << "  [json] WARNING: Failed to parse llama-bench JSON output\n";
            std::cerr << "  [json] Raw output (first 500 chars):\n"
                      << launch_result.stdout_output.substr(0, 500) << "\n";
        }
    }

    // Memory
    if (proc_mon) {
        result.mem = proc_mon->get_result();
        if (result.mem.available) {
            std::cout << "  [mem] peak_hwm=" << result.mem.peak_hwm_kb << " kB"
                      << " peak_rss=" << result.mem.peak_rss_kb << " kB"
                      << " (" << result.mem.sample_count << " samples)\n";
        }
    }

    // perf
    if (perf_mon) {
        result.perf = perf_mon->get_result();
        if (result.perf.available) {
            if (result.perf.cycles) {
                std::cout << "  [perf] cycles=" << *result.perf.cycles
                          << " instructions=" << (result.perf.instructions ? std::to_string(*result.perf.instructions) : "N/A")
                          << " cache_misses=" << (result.perf.cache_misses ? std::to_string(*result.perf.cache_misses) : "N/A")
                          << "\n";
            } else {
                std::cout << "  [perf] hardware counters unavailable (paranoia too high?)\n";
            }
            if (result.perf.context_switches) {
                std::cout << "  [perf] ctx_switches=" << *result.perf.context_switches
                          << " migrations=" << (result.perf.cpu_migrations ? std::to_string(*result.perf.cpu_migrations) : "N/A")
                          << "\n";
            }
        }
    }

    // Print summary
    if (result.success && result.bench.parsed) {
        double wall_s = static_cast<double>(result.wall.elapsed_ns) / 1e9;
        std::cout << "  [bench] avg_ts=" << std::fixed << std::setprecision(2)
                  << result.bench.avg_ts << " tok/s"
                  << " avg_ns=" << result.bench.avg_ns
                  << " wall=" << std::fixed << std::setprecision(1) << wall_s << "s\n";
    } else if (!result.success) {
        std::cerr << "  [FAILED] " << result.failure_reason << "\n";
    }

    return result;
}

// ── Experiment loop ────────────────────────────────────────────────────────

static void run_experiment_matrix(
    const llmchar::ExperimentConfig& cfg,
    const llmchar::SystemInfo& sysinfo,
    llmchar::ResultWriter& writer)
{
    std::string experiment_id = make_experiment_id();
    std::cout << "\n=== Experiment ID: " << experiment_id << " ===\n";

    writer.write_experiment_metadata(experiment_id, sysinfo, cfg);

    int global_run_id = 0;

    for (const auto& model : cfg.models) {
        if (g_shutdown.load()) break;

        std::cout << "\n--- Model: " << model.name << " ---\n";
        std::cout << "    Path: " << model.path << "\n";
        std::cout << "    Quantization: " << model.quantization << "\n";

        for (int threads : cfg.thread_counts) {
            if (g_shutdown.load()) break;

            for (const auto& affinity : cfg.cpu_affinities) {
                if (g_shutdown.load()) break;

                std::cout << "\n>>> threads=" << threads
                          << " affinity=" << affinity << "\n";

                std::vector<llmchar::RunResult> all_runs;

                // Warmup runs
                for (int w = 0; w < cfg.warmup_runs && !g_shutdown.load(); ++w) {
                    auto r = execute_one_run(cfg, model, threads, affinity,
                                             experiment_id, global_run_id++,
                                             /*is_warmup=*/true, sysinfo);
                    all_runs.push_back(r);
                    writer.write_run(r);
                }

                // Measurement runs
                for (int m = 0; m < cfg.measurement_runs && !g_shutdown.load(); ++m) {
                    auto r = execute_one_run(cfg, model, threads, affinity,
                                             experiment_id, global_run_id++,
                                             /*is_warmup=*/false, sysinfo);
                    all_runs.push_back(r);
                    writer.write_run(r);
                }

                // Write aggregate for this (model, threads, affinity) combination
                writer.write_aggregate(experiment_id, model.name, model.quantization,
                                       threads, affinity, all_runs, cfg.warmup_runs);

                std::cout << ">>> Configuration complete. "
                          << cfg.measurement_runs << " measurement runs written.\n";
            }
        }
    }
}

// ── Usage ──────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --config <path>     Path to experiment.yaml (default: config/experiment.yaml)\n"
              << "  --sysinfo-only      Print system info and exit\n"
              << "  --output-dir <dir>  Override output directory\n"
              << "  --help              Show this help\n";
}

} // anonymous namespace

// ── main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT,  signal_handler);

    std::string config_path   = "config/experiment.yaml";
    std::string output_dir_override;
    bool        sysinfo_only  = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--sysinfo-only") {
            sysinfo_only = true;
        } else if (arg == "--config" && i+1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--output-dir" && i+1 < argc) {
            output_dir_override = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== llm-characterization harness ===\n";
    std::cout << "    Started: " << iso8601_now() << "\n\n";

    // ── Load config ──────────────────────────────────────────────────────
    llmchar::ExperimentConfig cfg;
    try {
        cfg = llmchar::load_config(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[main] Config error: " << e.what() << "\n";
        return 1;
    }

    if (!output_dir_override.empty()) {
        cfg.output_dir = output_dir_override;
    }

    llmchar::print_config_summary(cfg);

    // ── Collect system info ──────────────────────────────────────────────
    std::cout << "\nCollecting system information...\n";
    llmchar::SystemInfo sysinfo = llmchar::collect_system_info(cfg.llama_bench_binary);
    llmchar::print_system_info(sysinfo);

    // Write system info separately
    {
        llmchar::ResultWriter tmp_writer(cfg.output_dir);
        tmp_writer.initialize();
        llmchar::write_system_info_json(sysinfo, cfg.output_dir + "/metadata/system_info.json");
    }

    if (sysinfo_only) {
        std::cout << "\n[main] --sysinfo-only: exiting.\n";
        return 0;
    }

    // ── Validate prerequisites ───────────────────────────────────────────
    // Check llama-bench binary exists
    if (access(cfg.llama_bench_binary.c_str(), X_OK) != 0) {
        std::cerr << "[main] ERROR: llama-bench not executable: "
                  << cfg.llama_bench_binary << "\n";
        return 1;
    }

    // Check model files exist
    bool all_models_ok = true;
    for (auto& m : cfg.models) {
        if (access(m.path.c_str(), R_OK) != 0) {
            std::cerr << "[main] ERROR: Model not readable: " << m.path << "\n";
            all_models_ok = false;
        }
    }
    if (!all_models_ok) return 1;

    // Warn about debug build
    if (sysinfo.llama_build_type == "Debug") {
        std::cerr << "\n*** WARNING: llama.cpp is a DEBUG build. ***\n"
                  << "*** Performance results will NOT be representative of a Release build. ***\n"
                  << "*** All results record build_type=Debug for traceability. ***\n\n";
    }

    // Warn if perf hardware counters unavailable
    if (cfg.perf_enabled && !sysinfo.perf_hw_available) {
        std::cerr << "*** WARNING: perf hardware counters unavailable. ***\n"
                  << "*** Run with sudo or: echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoia ***\n"
                  << "*** Software counters (context-switches, migrations) will still be collected. ***\n\n";
    }

    // ── Initialize result writer ─────────────────────────────────────────
    llmchar::ResultWriter writer(cfg.output_dir);
    if (!writer.initialize()) {
        std::cerr << "[main] Failed to initialize output directory: " << cfg.output_dir << "\n";
        return 1;
    }
    std::cout << "\nOutput directory: " << cfg.output_dir << "\n";

    // ── Run experiment matrix ─────────────────────────────────────────────
    run_experiment_matrix(cfg, sysinfo, writer);

    std::cout << "\n=== Experiment complete ===\n";
    std::cout << "    Raw data     : " << cfg.output_dir << "/raw/runs.csv\n";
    std::cout << "    Aggregates   : " << cfg.output_dir << "/aggregate/summary.csv\n";
    std::cout << "    Metadata     : " << cfg.output_dir << "/metadata/\n";
    std::cout << "    Finished     : " << iso8601_now() << "\n";

    if (g_shutdown.load()) {
        std::cout << "[main] Experiment was interrupted by signal.\n";
        return 1;
    }
    return 0;
}
