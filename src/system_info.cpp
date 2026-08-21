// system_info.cpp
// Collects environment metadata: CPU topology, memory, kernel, compiler, perf.
// Uses /sys, /proc, and external commands (lscpu, perf, llama-bench --version).
// Does NOT modify any system settings.

#include "system_info.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <unistd.h>
#include <sys/wait.h>
#include <sys/utsname.h>

namespace llmchar {

namespace {

// ── File/Shell helpers ──────────────────────────────────────────────────────

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    // strip trailing whitespace/newline
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s;
}

// Run a command and capture stdout+stderr merged. Returns empty string on failure.
static std::string run_cmd(const std::string& cmd) {
    std::string full = cmd + " 2>&1";
    FILE* p = popen(full.c_str(), "r");
    if (!p) return "";
    std::ostringstream ss;
    char buf[512];
    while (fgets(buf, sizeof(buf), p)) ss << buf;
    pclose(p);
    return ss.str();
}

// Run command, return first non-empty line
static std::string run_cmd_line1(const std::string& cmd) {
    std::string out = run_cmd(cmd);
    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
            line.pop_back();
        if (!line.empty()) return line;
    }
    return "";
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b-a+1);
}

static int safe_int(const std::string& s, int def=0) {
    try { return std::stoi(s); } catch(...) { return def; }
}

// ── CPU topology from /sys ──────────────────────────────────────────────────

// Raptor Lake heuristic: P-cores have core_id < 32, E-cores have core_id >= 32.
// This is derived from empirical inspection of i7-14700HX topology.
// The sysfs core_type file returned N/A on this kernel build.
static bool is_p_core_heuristic(int core_id) {
    return core_id < 32;
}

static std::vector<CpuCoreInfo> collect_cpu_cores() {
    std::vector<CpuCoreInfo> cores;

    // Discover how many CPUs exist
    int max_cpu = 0;
    {
        std::string online = read_file("/sys/devices/system/cpu/online");
        // Format: "0-27" or "0,1,2-5,..."
        if (!online.empty()) {
            // Parse the last range endpoint
            // Simple approach: just try cpu0..cpu63
            for (int i = 0; i < 256; ++i) {
                std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/core_id";
                std::ifstream f(path);
                if (!f.is_open()) {
                    if (i > 0) break; // no more CPUs
                    continue;
                }
                max_cpu = i;
            }
        }
    }

    for (int i = 0; i <= max_cpu; ++i) {
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(i);

        // Check if CPU exists
        std::ifstream core_id_f(base + "/topology/core_id");
        if (!core_id_f.is_open()) continue;

        CpuCoreInfo c;
        c.cpu_id = i;

        {
            std::string s;
            core_id_f >> s;
            c.core_id = safe_int(s);
        }

        c.is_p_core = is_p_core_heuristic(c.core_id);

        c.min_freq_khz = safe_int(read_file(base + "/cpufreq/cpuinfo_min_freq"));
        c.max_freq_khz = safe_int(read_file(base + "/cpufreq/cpuinfo_max_freq"));
        c.scaling_governor = read_file(base + "/cpufreq/scaling_governor");

        cores.push_back(c);
    }
    return cores;
}

// ── Memory info from /proc/meminfo ─────────────────────────────────────────

static void collect_mem_info(SystemInfo& info) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key, val, unit;
        ss >> key >> val;
        if (key == "MemTotal:") info.ram_total_kb     = std::stoull(val);
        if (key == "MemAvailable:") info.ram_available_kb = std::stoull(val);
    }
}

// ── Kernel version from uname ──────────────────────────────────────────────

static std::string get_kernel_version() {
    struct utsname u;
    if (uname(&u) == 0) return std::string(u.release);
    return read_file("/proc/version");
}

// ── OS distribution from /etc/os-release ──────────────────────────────────

static std::string get_os_distro() {
    std::ifstream f("/etc/os-release");
    std::string line;
    while (std::getline(f, line)) {
        if (line.substr(0, 12) == "PRETTY_NAME=") {
            std::string val = line.substr(12);
            // strip quotes
            if (!val.empty() && val.front() == '"') val = val.substr(1);
            if (!val.empty() && val.back() == '"')  val.pop_back();
            return val;
        }
    }
    return "Unknown";
}

// ── CPU model from /proc/cpuinfo ───────────────────────────────────────────

static std::string get_cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.substr(0, 10) == "model name") {
            auto colon = line.find(':');
            if (colon != std::string::npos) return trim(line.substr(colon+1));
        }
    }
    return "Unknown";
}

// ── Cache sizes from lscpu ─────────────────────────────────────────────────

static void collect_cache_info(SystemInfo& info) {
    std::string out = run_cmd("lscpu 2>/dev/null");
    std::istringstream ss(out);
    std::string line;
    while (std::getline(ss, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon+1));
        if (key == "L1d cache") info.l1d_cache = val;
        else if (key == "L1i cache") info.l1i_cache = val;
        else if (key == "L2 cache") info.l2_cache = val;
        else if (key == "L3 cache") info.l3_cache = val;
    }
}

// ── Turbo boost ────────────────────────────────────────────────────────────

static bool get_turbo_enabled() {
    std::string s = read_file("/sys/devices/system/cpu/intel_pstate/no_turbo");
    if (s == "0") return true;   // no_turbo=0 → turbo enabled
    if (s == "1") return false;  // no_turbo=1 → turbo disabled
    // Try cpufreq boost
    s = read_file("/sys/devices/system/cpu/cpufreq/boost");
    return (s == "1");
}

// ── llama-bench version ────────────────────────────────────────────────────

static void collect_llama_info(SystemInfo& info, const std::string& llama_bench_binary) {
    std::string bindir = llama_bench_binary;
    auto slash = bindir.rfind('/');
    if (slash != std::string::npos) bindir = bindir.substr(0, slash);

    info.llama_binary_path = llama_bench_binary;

    // llama-bench does NOT have --version. Use llama-cli if available.
    // If not, parse info from CMakeCache + git log.
    std::string llama_cli = bindir + "/llama-cli";
    std::string ld_prefix = "LD_LIBRARY_PATH=" + bindir + " ";

    {
        std::string cmd = ld_prefix + llama_cli + " --version 2>&1 | head -3";
        std::string out = run_cmd(cmd);
        std::istringstream ss(out);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.find("version:") != std::string::npos) {
                auto paren_open  = line.find('(');
                auto paren_close = line.find(')');
                if (paren_open != std::string::npos && paren_close != std::string::npos) {
                    info.llama_build_commit = line.substr(paren_open+1, paren_close-paren_open-1);
                }
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    std::istringstream ls(line.substr(colon+1));
                    ls >> info.llama_build_number;
                }
            } else if (line.find("built with") != std::string::npos) {
                info.compiler_version = trim(line);
            }
        }
    }

    // Fallback: read from CMakeCache.txt for build info
    std::string cmake_cache = bindir + "/../CMakeCache.txt";
    std::ifstream cf(cmake_cache);
    if (cf.is_open()) {
        std::string cline;
        while (std::getline(cf, cline)) {
            if (cline.substr(0, 17) == "CMAKE_BUILD_TYPE:") {
                auto eq = cline.find('=');
                if (eq != std::string::npos)
                    info.llama_build_type = trim(cline.substr(eq+1));
            }
            if (info.compiler_version.empty() && cline.find("CMAKE_CXX_COMPILER:") != std::string::npos) {
                // Get compiler ID from a different key
            }
        }
    }

    // If build_commit still empty, try git
    if (info.llama_build_commit.empty()) {
        std::string repo_dir = bindir + "/../..";
        std::string git_out = run_cmd("git -C " + repo_dir + " rev-parse --short HEAD 2>/dev/null");
        if (!git_out.empty() && git_out.find("fatal") == std::string::npos) {
            info.llama_build_commit = trim(git_out);
        }
    }
    if (info.llama_build_number == 0) {
        // Try reading from a build number file or CMakeCache
        std::string bn = run_cmd("grep 'LLAMA_BUILD_NUMBER' " + cmake_cache + " 2>/dev/null | cut -d= -f2");
        if (!bn.empty()) {
            try { info.llama_build_number = std::stoi(trim(bn)); } catch(...) {}
        }
    }
    if (info.llama_build_type.empty()) info.llama_build_type = "Unknown";
    if (info.compiler_version.empty()) {
        info.compiler_version = run_cmd_line1("c++ --version 2>/dev/null | head -1");
    }
}

// ── perf version ───────────────────────────────────────────────────────────

static std::string get_perf_version() {
    return run_cmd_line1("perf --version 2>/dev/null");
}

// ── ISO8601 timestamp ──────────────────────────────────────────────────────

static std::string iso8601_now() {
    time_t t = time(nullptr);
    struct tm tm_buf;
    gmtime_r(&t, &tm_buf);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

// ── JSON escaping ──────────────────────────────────────────────────────────

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

} // anonymous namespace

// ── Public API ─────────────────────────────────────────────────────────────

int get_perf_event_paranoia() {
    std::string s = read_file("/proc/sys/kernel/perf_event_paranoia");
    if (s.empty()) return -1;
    try { return std::stoi(s); } catch(...) { return -1; }
}

bool probe_perf_hardware_events() {
    // Run a tiny perf stat on /bin/true and check output
    std::string out = run_cmd("perf stat -e cycles,instructions /bin/true 2>&1");
    if (out.empty()) return false;
    // If hardware counters are blocked: output contains "<not supported>" or "Permission denied"
    if (out.find("<not supported>") != std::string::npos) return false;
    if (out.find("Permission denied") != std::string::npos) return false;
    if (out.find("perf_event_open") != std::string::npos) return false;
    // Check that cycles shows a number
    return (out.find("cycles") != std::string::npos);
}

bool probe_perf_uncore_events() {
    std::string out = run_cmd(
        "perf stat -e uncore_imc_free_running/data_read/ /bin/true 2>&1");
    if (out.empty()) return false;
    if (out.find("<not supported>") != std::string::npos) return false;
    if (out.find("Permission denied") != std::string::npos) return false;
    if (out.find("No such file") != std::string::npos) return false;
    return (out.find("uncore_imc") != std::string::npos);
}

SystemInfo collect_system_info(const std::string& llama_bench_binary) {
    SystemInfo info;
    info.timestamp_iso = iso8601_now();

    info.cpu_model      = get_cpu_model();
    info.kernel_version = get_kernel_version();
    info.os_distribution = get_os_distro();

    collect_mem_info(info);
    collect_cache_info(info);

    info.cores = collect_cpu_cores();
    info.cpu_count = static_cast<int>(info.cores.size());
    info.p_core_count = 0;
    info.e_core_count = 0;
    for (auto& c : info.cores) {
        if (c.is_p_core) ++info.p_core_count;
        else             ++info.e_core_count;
    }

    if (!info.cores.empty()) {
        info.scaling_governor = info.cores[0].scaling_governor;
    }
    info.turbo_enabled = get_turbo_enabled();

    collect_llama_info(info, llama_bench_binary);

    info.perf_version        = get_perf_version();
    info.perf_event_paranoia = get_perf_event_paranoia();
    info.perf_hw_available   = probe_perf_hardware_events();
    info.perf_uncore_available = probe_perf_uncore_events();

    return info;
}

void write_system_info_json(const SystemInfo& info, const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: cannot write system info to " << path << "\n";
        return;
    }

    auto q = [](const std::string& s) -> std::string {
        return "\"" + json_escape(s) + "\"";
    };

    f << "{\n";
    f << "  \"collected_at\": " << q(info.timestamp_iso) << ",\n";
    f << "  \"cpu_model\": " << q(info.cpu_model) << ",\n";
    f << "  \"cpu_count\": " << info.cpu_count << ",\n";
    f << "  \"p_core_count\": " << info.p_core_count << ",\n";
    f << "  \"e_core_count\": " << info.e_core_count << ",\n";
    f << "  \"p_e_heuristic\": \"core_id<32=P-core (i7-14700HX Raptor Lake)\",\n";
    f << "  \"scaling_governor\": " << q(info.scaling_governor) << ",\n";
    f << "  \"turbo_enabled\": " << (info.turbo_enabled ? "true" : "false") << ",\n";
    f << "  \"l1d_cache\": " << q(info.l1d_cache) << ",\n";
    f << "  \"l1i_cache\": " << q(info.l1i_cache) << ",\n";
    f << "  \"l2_cache\": " << q(info.l2_cache) << ",\n";
    f << "  \"l3_cache\": " << q(info.l3_cache) << ",\n";
    f << "  \"ram_total_kb\": " << info.ram_total_kb << ",\n";
    f << "  \"ram_available_kb\": " << info.ram_available_kb << ",\n";
    f << "  \"kernel_version\": " << q(info.kernel_version) << ",\n";
    f << "  \"os_distribution\": " << q(info.os_distribution) << ",\n";
    f << "  \"llama_bench_binary\": " << q(info.llama_binary_path) << ",\n";
    f << "  \"llama_build_commit\": " << q(info.llama_build_commit) << ",\n";
    f << "  \"llama_build_number\": " << info.llama_build_number << ",\n";
    f << "  \"llama_build_type\": " << q(info.llama_build_type) << ",\n";
    f << "  \"compiler_version\": " << q(info.compiler_version) << ",\n";
    f << "  \"perf_version\": " << q(info.perf_version) << ",\n";
    f << "  \"perf_event_paranoia\": " << info.perf_event_paranoia << ",\n";
    f << "  \"perf_hw_available\": " << (info.perf_hw_available ? "true" : "false") << ",\n";
    f << "  \"perf_uncore_available\": " << (info.perf_uncore_available ? "true" : "false") << ",\n";
    f << "  \"cpu_cores\": [\n";
    for (size_t i = 0; i < info.cores.size(); ++i) {
        const auto& c = info.cores[i];
        f << "    {\"cpu_id\":" << c.cpu_id
          << ",\"core_id\":" << c.core_id
          << ",\"is_p_core\":" << (c.is_p_core ? "true" : "false")
          << ",\"max_freq_khz\":" << c.max_freq_khz
          << ",\"scaling_governor\":" << q(c.scaling_governor)
          << "}";
        if (i+1 < info.cores.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

void print_system_info(const SystemInfo& info) {
    std::cout << "=== System Environment ===\n";
    std::cout << "  CPU          : " << info.cpu_model << "\n";
    std::cout << "  Cores        : " << info.cpu_count
              << " (" << info.p_core_count << " P + " << info.e_core_count << " E)\n";
    std::cout << "  L3 cache     : " << info.l3_cache << "\n";
    std::cout << "  RAM          : " << (info.ram_total_kb / 1024 / 1024) << " GiB total\n";
    std::cout << "  Governor     : " << info.scaling_governor << "\n";
    std::cout << "  Turbo        : " << (info.turbo_enabled ? "enabled" : "disabled") << "\n";
    std::cout << "  Kernel       : " << info.kernel_version << "\n";
    std::cout << "  OS           : " << info.os_distribution << "\n";
    std::cout << "  llama commit : " << info.llama_build_commit
              << " (build " << info.llama_build_number << ")\n";
    std::cout << "  llama build  : " << info.llama_build_type << "\n";
    if (info.llama_build_type == "Debug") {
        std::cout << "  *** WARNING: Debug build — performance not representative of Release ***\n";
    }
    std::cout << "  Compiler     : " << info.compiler_version << "\n";
    std::cout << "  perf         : " << info.perf_version << "\n";
    std::cout << "  perf paranoia: " << info.perf_event_paranoia << "\n";
    std::cout << "  perf HW evts : " << (info.perf_hw_available ? "YES" : "NO (run with sudo or lower paranoia)") << "\n";
    std::cout << "  perf uncore  : " << (info.perf_uncore_available ? "YES" : "NO (requires root)") << "\n";
    std::cout << "==========================\n";
}

} // namespace llmchar
