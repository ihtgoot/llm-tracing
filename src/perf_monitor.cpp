// perf_monitor.cpp
// Manages a `perf stat -p <pid>` subprocess.
// perf stat attaches to the target PID and collects counters for its lifetime.
// Output is captured from perf's stderr (perf stat always writes to stderr).
//
// Event selection:
//   Hardware: cycles, instructions, cache-references, cache-misses,
//             branch-instructions, branch-misses
//   Software: context-switches, cpu-migrations
//   Uncore (root only): uncore_imc_free_running/data_read/ etc.
//
// Parsing: we look for lines of the form:
//   "    12,345,678      cycles"
//   "    <not supported> cache-references"
//   "    <not counted>   instructions"

#include "perf_monitor.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <regex>

#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>

namespace llmchar {

namespace {

// Strip commas and spaces from a number string like "1,234,567"
static int64_t parse_counter_value(const std::string& s) {
    std::string clean;
    for (char c : s) {
        if (c >= '0' && c <= '9') clean += c;
    }
    if (clean.empty()) return 0;
    try { return std::stoll(clean); } catch(...) { return 0; }
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b-a+1);
}

// Read all data from a file descriptor into a string
static std::string read_fd(int fd) {
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        out.append(buf, n);
    }
    return out;
}

} // anonymous namespace

// ── PerfMonitor ────────────────────────────────────────────────────────────

PerfMonitor::PerfMonitor(bool hw_available, bool uncore_available)
    : hw_available_(hw_available), uncore_available_(uncore_available) {}

std::string PerfMonitor::build_event_string() const {
    std::vector<std::string> events;

    if (hw_available_) {
        // Hardware events — these require paranoia <= 1
        events.push_back("cycles");
        events.push_back("instructions");
        events.push_back("cache-references");
        events.push_back("cache-misses");
        events.push_back("branch-instructions");
        events.push_back("branch-misses");
    }

    // Software events — always available (they're SW counters)
    events.push_back("context-switches");
    events.push_back("cpu-migrations");

    if (uncore_available_) {
        // DRAM bandwidth events — require root
        events.push_back("uncore_imc_free_running/data_read/");
        events.push_back("uncore_imc_free_running/data_write/");
        events.push_back("uncore_imc_free_running/data_total/");
    }

    std::string s;
    for (size_t i = 0; i < events.size(); ++i) {
        if (i) s += ",";
        s += events[i];
    }
    return s;
}

bool PerfMonitor::start(pid_t target_pid) {
    // Create a pipe to capture perf's stderr
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        std::cerr << "[perf] Failed to create pipe: " << strerror(errno) << "\n";
        return false;
    }

    // Set O_CLOEXEC on read end so it doesn't leak into perf's exec
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);

    std::string events = build_event_string();

    perf_pid_ = fork();
    if (perf_pid_ < 0) {
        std::cerr << "[perf] fork failed: " << strerror(errno) << "\n";
        close(pipefd[0]);
        close(pipefd[1]);
        return false;
    }

    if (perf_pid_ == 0) {
        // Child: redirect stderr to write end of pipe
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);

        // Redirect stdout to /dev/null
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) dup2(devnull, STDOUT_FILENO);

        // perf stat -p <pid> -e <events>
        // -x , → CSV output for easier parsing (alternative to human-readable)
        std::string pid_str = std::to_string(target_pid);
        execlp("perf", "perf", "stat",
               "-p", pid_str.c_str(),
               "-e", events.c_str(),
               nullptr);
        // If exec fails:
        fprintf(stderr, "[perf] exec failed: %s\n", strerror(errno));
        _exit(127);
    }

    // Parent: close write end, keep read end
    close(pipefd[1]);
    pipe_fd_ = pipefd[0];
    result_.available = true;
    result_.hardware_available = hw_available_;
    return true;
}

void PerfMonitor::stop() {
    if (perf_pid_ <= 0) return;

    // Send SIGINT to perf so it dumps its stats and exits cleanly
    kill(perf_pid_, SIGINT);

    // Read perf's stderr output (may block briefly until perf exits)
    std::string output;
    if (pipe_fd_ >= 0) {
        output = read_fd(pipe_fd_);
        close(pipe_fd_);
        pipe_fd_ = -1;
    }

    // Wait for perf to exit
    int status = 0;
    waitpid(perf_pid_, &status, 0);
    perf_pid_ = -1;

    result_.raw_output = output;
    parse_perf_output(output, result_);
}

std::optional<int64_t> PerfMonitor::parse_counter(
    const std::string& output,
    const std::string& event_name) const
{
    // perf stat human-readable format (default):
    //
    //      12,345,678      cycles
    //   <not supported>    cache-references
    //   <not counted>      instructions
    //
    // We search for a line ending with the event name (possibly with unit info)

    std::istringstream ss(output);
    std::string line;
    while (std::getline(ss, line)) {
        // Check if this line mentions the event
        if (line.find(event_name) == std::string::npos) continue;

        // Check for not-supported / not-counted
        if (line.find("<not supported>") != std::string::npos) return std::nullopt;
        if (line.find("<not counted>")   != std::string::npos) return std::nullopt;

        // Extract the first token (the number, possibly with commas)
        std::string trimmed = trim(line);
        size_t space = trimmed.find_first_of(" \t");
        if (space == std::string::npos) continue;
        std::string num_str = trimmed.substr(0, space);

        int64_t val = parse_counter_value(num_str);
        if (val > 0 || num_str == "0") return val;
    }
    return std::nullopt;
}

bool PerfMonitor::parse_perf_output(const std::string& output,
                                     PerfMeasurement& m) const {
    if (output.empty()) {
        m.available = false;
        return false;
    }

    m.available = true;

    if (hw_available_) {
        m.cycles              = parse_counter(output, "cycles");
        m.instructions        = parse_counter(output, "instructions");
        m.cache_references    = parse_counter(output, "cache-references");
        m.cache_misses        = parse_counter(output, "cache-misses");
        m.branch_instructions = parse_counter(output, "branch-instructions");
        m.branch_misses       = parse_counter(output, "branch-misses");
        m.hardware_available  = m.cycles.has_value();

        if (m.cycles && m.instructions && *m.cycles > 0) {
            m.ipc = static_cast<double>(*m.instructions) /
                    static_cast<double>(*m.cycles);
        }
    }

    m.context_switches = parse_counter(output, "context-switches");
    m.cpu_migrations   = parse_counter(output, "cpu-migrations");

    return true;
}

PerfMeasurement PerfMonitor::get_result() const {
    return result_;
}

} // namespace llmchar
