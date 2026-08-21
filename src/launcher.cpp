// launcher.cpp
// Launches llama-bench as a child process and manages its lifecycle.

#include "launcher.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <atomic>
#include <thread>

#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <fcntl.h>
#include <sched.h>
#include <errno.h>

namespace llmchar {

namespace {

// ── Timing ─────────────────────────────────────────────────────────────────

static int64_t monotonic_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

// ── JSON parsing helpers ────────────────────────────────────────────────────

// Extract value for a JSON key from a flat JSON line.
// Works on both the formatted and single-line JSON formats.
// Returns empty string if key not found.
static std::string json_get_string(const std::string& json,
                                    const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    // Skip whitespace and colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) ++pos;
    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        // String value
        ++pos;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    } else {
        // Numeric or boolean value — read until comma, }, or ]
        size_t end = pos;
        while (end < json.size() && json[end] != ',' && json[end] != '}' &&
               json[end] != ']' && json[end] != '\n') {
            ++end;
        }
        std::string val = json.substr(pos, end - pos);
        // trim whitespace
        size_t a = val.find_first_not_of(" \t");
        size_t b = val.find_last_not_of(" \t");
        if (a == std::string::npos) return "";
        return val.substr(a, b-a+1);
    }
}

static int64_t json_get_int(const std::string& json, const std::string& key,
                              int64_t def = 0) {
    std::string v = json_get_string(json, key);
    if (v.empty()) return def;
    try { return std::stoll(v); } catch(...) { return def; }
}

static double json_get_double(const std::string& json, const std::string& key,
                               double def = 0.0) {
    std::string v = json_get_string(json, key);
    if (v.empty()) return def;
    try { return std::stod(v); } catch(...) { return def; }
}

// Extract a JSON array of numbers: [...] associated with the given key.
static std::vector<int64_t> json_get_int_array(const std::string& json,
                                                 const std::string& key) {
    std::vector<int64_t> result;
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return result;
    size_t end = json.find(']', pos);
    if (end == std::string::npos) return result;
    std::string arr = json.substr(pos+1, end-pos-1);
    std::istringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // trim
        size_t a = token.find_first_not_of(" \t\n\r");
        size_t b = token.find_last_not_of(" \t\n\r");
        if (a == std::string::npos) continue;
        std::string t = token.substr(a, b-a+1);
        try { result.push_back(std::stoll(t)); } catch(...) {}
    }
    return result;
}

static std::vector<double> json_get_double_array(const std::string& json,
                                                   const std::string& key) {
    std::vector<double> result;
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return result;
    size_t end = json.find(']', pos);
    if (end == std::string::npos) return result;
    std::string arr = json.substr(pos+1, end-pos-1);
    std::istringstream ss(arr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t a = token.find_first_not_of(" \t\n\r");
        size_t b = token.find_last_not_of(" \t\n\r");
        if (a == std::string::npos) continue;
        std::string t = token.substr(a, b-a+1);
        try { result.push_back(std::stod(t)); } catch(...) {}
    }
    return result;
}

} // anonymous namespace

// ── CPU affinity parsing ────────────────────────────────────────────────────

bool parse_cpu_affinity(const std::string& affinity_str,
                        std::vector<int>& cpu_list) {
    cpu_list.clear();
    if (affinity_str == "none" || affinity_str.empty()) return true;

    // Parse "0-3" or "0,1,2,3" or "0-3,8-11"
    std::istringstream ss(affinity_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t dash = token.find('-');
        if (dash != std::string::npos) {
            // Range: "0-3"
            try {
                int lo = std::stoi(token.substr(0, dash));
                int hi = std::stoi(token.substr(dash+1));
                if (lo > hi || lo < 0 || hi > 511) return false;
                for (int i = lo; i <= hi; ++i) cpu_list.push_back(i);
            } catch(...) { return false; }
        } else {
            // Single: "4"
            try {
                int c = std::stoi(token);
                if (c < 0 || c > 511) return false;
                cpu_list.push_back(c);
            } catch(...) { return false; }
        }
    }
    return true;
}

// ── JSON parser for llama-bench output ────────────────────────────────────

LlamaBenchResult parse_llama_bench_json(const std::string& json_output) {
    LlamaBenchResult r;
    if (json_output.empty()) return r;

    // llama-bench outputs a JSON array of objects.
    // Each object is one test (pp or tg).
    // We need to find each object. We extract the full JSON as a flat string
    // and search for each key.

    // Simple approach: flatten the whole JSON and look for each key.
    // This works because our JSON has one array of objects.
    std::string flat = json_output;
    // Remove newlines for easier searching
    std::replace(flat.begin(), flat.end(), '\n', ' ');

    r.build_commit   = json_get_string(flat, "build_commit");
    r.build_number   = static_cast<int>(json_get_int(flat, "build_number"));
    r.cpu_info       = json_get_string(flat, "cpu_info");
    r.model_filename = json_get_string(flat, "model_filename");
    r.model_type     = json_get_string(flat, "model_type");
    r.model_size     = json_get_int(flat, "model_size");
    r.model_n_params = json_get_int(flat, "model_n_params");
    r.n_threads      = static_cast<int>(json_get_int(flat, "n_threads"));
    r.n_batch        = static_cast<int>(json_get_int(flat, "n_batch"));
    r.n_ubatch       = static_cast<int>(json_get_int(flat, "n_ubatch"));
    r.load_mode      = json_get_string(flat, "load_mode");
    r.test_time      = json_get_string(flat, "test_time");

    // For n_prompt and n_gen: the JSON has separate objects for pp and tg.
    // Find the object with n_prompt > 0 → that's the pp (prompt processing) test.
    // Find the object with n_gen > 0 → that's the tg (token generation) test.
    //
    // We use a simple object-by-object scan.
    // Each object starts with '{' and ends with '}'.
    // We collect avg_ns, stddev_ns, avg_ts, stddev_ts, samples_ns, samples_ts
    // from the LAST object (generation is the last one in llama-bench output).

    // Find all objects in the array
    struct BenchEntry {
        int n_prompt, n_gen;
        int64_t avg_ns, stddev_ns;
        double avg_ts, stddev_ts;
        std::vector<int64_t> samples_ns;
        std::vector<double> samples_ts;
    };

    std::vector<BenchEntry> entries;
    size_t pos = 0;
    while (pos < flat.size()) {
        size_t obj_start = flat.find('{', pos);
        if (obj_start == std::string::npos) break;
        // Find matching closing brace (naive — no nested objects)
        int depth = 0;
        size_t obj_end = obj_start;
        for (size_t i = obj_start; i < flat.size(); ++i) {
            if (flat[i] == '{') ++depth;
            else if (flat[i] == '}') {
                --depth;
                if (depth == 0) { obj_end = i; break; }
            }
        }
        if (obj_end == obj_start) break;

        std::string obj = flat.substr(obj_start, obj_end - obj_start + 1);
        BenchEntry e;
        e.n_prompt    = static_cast<int>(json_get_int(obj, "n_prompt"));
        e.n_gen       = static_cast<int>(json_get_int(obj, "n_gen"));
        e.avg_ns      = json_get_int(obj, "avg_ns");
        e.stddev_ns   = json_get_int(obj, "stddev_ns");
        e.avg_ts      = json_get_double(obj, "avg_ts");
        e.stddev_ts   = json_get_double(obj, "stddev_ts");
        e.samples_ns  = json_get_int_array(obj, "samples_ns");
        e.samples_ts  = json_get_double_array(obj, "samples_ts");
        entries.push_back(e);

        pos = obj_end + 1;
    }

    if (entries.empty()) {
        std::cerr << "[launcher] Warning: failed to parse any JSON objects from llama-bench output\n";
        return r;
    }

    // We want the generation (tg) entry: n_gen > 0
    // If no tg entry, take the last entry
    BenchEntry* gen_entry = nullptr;
    BenchEntry* pp_entry  = nullptr;
    for (auto& e : entries) {
        if (e.n_gen > 0)    gen_entry = &e;
        if (e.n_prompt > 0) pp_entry  = &e;
    }

    // Use generation as primary metric
    BenchEntry* primary = gen_entry ? gen_entry : &entries.back();
    r.n_prompt   = pp_entry  ? pp_entry->n_prompt  : 0;
    r.n_gen      = gen_entry ? gen_entry->n_gen     : 0;
    r.avg_ns     = primary->avg_ns;
    r.stddev_ns  = primary->stddev_ns;
    r.avg_ts     = primary->avg_ts;
    r.stddev_ts  = primary->stddev_ts;
    r.samples_ns = primary->samples_ns;
    r.samples_ts = primary->samples_ts;

    r.parsed = (!r.model_filename.empty() || !r.model_type.empty());
    return r;
}

// ── Main launch function ───────────────────────────────────────────────────

LaunchResult launch_llama_bench(
    const LaunchConfig& cfg,
    std::function<void(pid_t)> on_pid_ready)
{
    LaunchResult result;

    // Determine lib directory for LD_LIBRARY_PATH
    std::string lib_dir = cfg.lib_dir;
    if (lib_dir.empty()) {
        // Default: same directory as the binary
        auto slash = cfg.binary_path.rfind('/');
        if (slash != std::string::npos) {
            lib_dir = cfg.binary_path.substr(0, slash);
        }
    }

    // Parse CPU affinity
    std::vector<int> cpu_list;
    if (!parse_cpu_affinity(cfg.cpu_affinity, cpu_list)) {
        result.failure_reason = "Invalid CPU affinity string: " + cfg.cpu_affinity;
        return result;
    }

    // Build argument list for llama-bench
    // llama-bench args: -m model -t threads -p n_prompt -n n_gen -b batch -ub ubatch -r reps --load-mode mode -o json
    std::vector<std::string> args_str;
    args_str.push_back(cfg.binary_path);
    args_str.push_back("-m"); args_str.push_back(cfg.model_path);
    args_str.push_back("-t"); args_str.push_back(std::to_string(cfg.threads));

    // Run prompt processing and generation in separate tests:
    // -p <n_prompt> -n <n_gen> will run both
    // Use -pg 0,<n_gen> to run gen only, or -p <n> for pp only
    // For baseline: run both pp and tg
    if (cfg.n_prompt > 0) {
        args_str.push_back("-p"); args_str.push_back(std::to_string(cfg.n_prompt));
    }
    if (cfg.n_gen > 0) {
        args_str.push_back("-n"); args_str.push_back(std::to_string(cfg.n_gen));
    }
    args_str.push_back("-b");  args_str.push_back(std::to_string(cfg.batch_size));
    args_str.push_back("-ub"); args_str.push_back(std::to_string(cfg.ubatch_size));
    args_str.push_back("-r");  args_str.push_back(std::to_string(cfg.repetitions));
    args_str.push_back("--load-mode"); args_str.push_back(cfg.load_mode);
    args_str.push_back("-o");  args_str.push_back("json");

    for (auto& a : cfg.extra_args) args_str.push_back(a);

    // Build argv array (const char* pointers into args_str)
    std::vector<const char*> argv;
    for (auto& s : args_str) argv.push_back(s.c_str());
    argv.push_back(nullptr);

    // Create stdout pipe
    int stdout_pipe[2];
    if (pipe(stdout_pipe) != 0) {
        result.failure_reason = "pipe() failed: " + std::string(strerror(errno));
        return result;
    }
    // Make read-end non-blocking initially (we'll block after child exits)
    fcntl(stdout_pipe[0], F_SETFD, FD_CLOEXEC);

    // Record wall start
    result.wall_start_ns = monotonic_ns();

    pid_t child = fork();
    if (child < 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        result.failure_reason = "fork() failed: " + std::string(strerror(errno));
        return result;
    }

    if (child == 0) {
        // ── Child process ──────────────────────────────────────────────────

        // Set CPU affinity before exec
        if (!cpu_list.empty()) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (int c : cpu_list) CPU_SET(c, &cpuset);
            if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
                fprintf(stderr, "[launcher] sched_setaffinity failed: %s\n",
                        strerror(errno));
                // Non-fatal — continue with default affinity
            }
        }

        // Redirect stdout to write end of pipe
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        // Set LD_LIBRARY_PATH so llama shared libs are found
        if (!lib_dir.empty()) {
            std::string existing = "";
            const char* cur = getenv("LD_LIBRARY_PATH");
            if (cur) existing = std::string(":") + cur;
            setenv("LD_LIBRARY_PATH", (lib_dir + existing).c_str(), 1);
        }

        // Exec llama-bench
        execv(cfg.binary_path.c_str(), const_cast<char* const*>(argv.data()));
        fprintf(stderr, "[launcher] execv failed: %s\n", strerror(errno));
        _exit(127);
    }

    // ── Parent process ─────────────────────────────────────────────────────
    close(stdout_pipe[1]); // close write end in parent
    result.child_pid = child;

    // Notify caller of child PID (so telemetry can start)
    if (on_pid_ready) {
        on_pid_ready(child);
    }

    // Timeout watchdog thread
    std::atomic<bool> timed_out{false};
    std::thread watchdog([&]() {
        std::this_thread::sleep_for(
            std::chrono::seconds(cfg.timeout_seconds));
        if (!timed_out.load()) {
            timed_out.store(true);
            kill(child, SIGTERM);
            std::this_thread::sleep_for(std::chrono::seconds(5));
            kill(child, SIGKILL); // force kill if SIGTERM ignored
        }
    });

    // Read stdout while child runs (non-blocking read loop)
    // We buffer output here; llama-bench only writes JSON on exit.
    std::string stdout_data;
    {
        char buf[8192];
        ssize_t n;
        while ((n = read(stdout_pipe[0], buf, sizeof(buf))) > 0) {
            stdout_data.append(buf, n);
        }
        close(stdout_pipe[0]);
    }

    // Wait for child
    int status = 0;
    waitpid(child, &status, 0);

    result.wall_end_ns = monotonic_ns();

    // Cancel watchdog
    timed_out.store(true); // signal watchdog to not kill
    watchdog.detach(); // detach; it will expire naturally

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
        result.success   = (result.exit_code == 0);
        if (!result.success && result.failure_reason.empty()) {
            result.failure_reason = "llama-bench exited with code " +
                                    std::to_string(result.exit_code);
        }
    } else if (WIFSIGNALED(status)) {
        result.exit_code = -WTERMSIG(status);
        if (timed_out.load()) {
            result.timed_out = true;
            result.failure_reason = "Process timed out after " +
                                    std::to_string(cfg.timeout_seconds) + "s";
        } else {
            result.failure_reason = "Process killed by signal " +
                                    std::to_string(WTERMSIG(status));
        }
    }

    result.stdout_output = stdout_data;
    return result;
}

} // namespace llmchar
