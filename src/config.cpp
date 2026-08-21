// config.cpp
// YAML subset parser for experiment.yaml.
// Handles: key: value scalars, lists (- item), nested mappings, comments (#).
// Does NOT require external yaml-cpp — intentionally self-contained.

#include "config.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <functional>

namespace llmchar {

namespace {

// ── String utilities ────────────────────────────────────────────────────────

static std::string ltrim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
        [](unsigned char c){ return !std::isspace(c); }));
    return s;
}

static std::string rtrim(std::string s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
        [](unsigned char c){ return !std::isspace(c); }).base(), s.end());
    return s;
}

static std::string trim(const std::string& s) {
    return ltrim(rtrim(s));
}

// Strip inline comment
static std::string strip_comment(const std::string& s) {
    // Only strip # not inside quotes (simple heuristic — our values don't have #)
    auto pos = s.find('#');
    if (pos == std::string::npos) return s;
    return rtrim(s.substr(0, pos));
}

// Strip surrounding quotes from a scalar value
static std::string unquote(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// Count leading spaces (indentation level)
static int indent_of(const std::string& line) {
    int n = 0;
    for (char c : line) {
        if (c == ' ') ++n;
        else break;
    }
    return n;
}

// ── Minimal YAML tokenizer ──────────────────────────────────────────────────
// We represent the YAML as a flat list of (indent, key, value) entries
// where list items have key="" and value=item text.


static std::vector<YamlEntry> tokenize(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    std::vector<YamlEntry> entries;
    std::string line;
    while (std::getline(f, line)) {
        // Strip \r for Windows line endings
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::string raw = line;
        std::string trimmed = trim(raw);

        // Skip blank lines and comments
        if (trimmed.empty() || trimmed[0] == '#') continue;

        int ind = indent_of(raw);
        std::string content = strip_comment(trimmed);

        YamlEntry e;
        e.indent = ind;

        if (content.substr(0, 2) == "- ") {
            // List item
            e.is_list_item = true;
            e.value = unquote(trim(content.substr(2)));
        } else if (content == "-") {
            // Empty list item
            e.is_list_item = true;
            e.value = "";
        } else {
            // Key: value
            auto colon = content.find(':');
            if (colon == std::string::npos) continue; // skip malformed
            e.key   = trim(content.substr(0, colon));
            e.value = unquote(trim(content.substr(colon + 1)));
        }
        entries.push_back(e);
    }
    return entries;
}

// ── Simple navigation helpers ───────────────────────────────────────────────

// Find the value for a top-level key
static std::string find_scalar(const std::vector<YamlEntry>& entries,
                                const std::string& key,
                                const std::string& default_val = "") {
    for (auto& e : entries) {
        if (!e.is_list_item && e.key == key && e.indent == 0) {
            return e.value;
        }is_list_item
    }
    return default_val;
}

// Collect list items that appear directly under a key at a given parent indent
// (i.e. list items at parent_indent+2 following the key line)
static std::vector<std::string> find_list(const std::vector<YamlEntry>& entries,
                                          const std::string& parent_key,
                                          int parent_indent) {
    std::vector<std::string> result;
    bool in_section = false;
    int expected_indent = -1;

    for (auto& e : entries) {
        if (!e.is_list_item && e.key == parent_key && e.indent == parent_indent) {
            in_section = true;
            expected_indent = -1;
            continue;
        }
        if (in_section) {
            if (e.is_list_item) {
                if (expected_indent == -1) expected_indent = e.indent;
                if (e.indent == expected_indent) {
                    result.push_back(e.value);
                } else if (e.indent < expected_indent) {
                    break; // left the list
                }
                // deeper indent: skip (sub-item of list)
            } else {
                if (expected_indent != -1 && e.indent <= parent_indent) break;
                if (expected_indent == -1 && e.indent <= parent_indent) break;
            }
        }
    }
    return result;
}

// Collect nested mapping entries under a key (returning sub-entries)
static std::vector<YamlEntry> find_section(const std::vector<YamlEntry>& entries,
                                            const std::string& parent_key,
                                            int parent_indent) {
    std::vector<YamlEntry> result;
    bool in_section = false;

    for (auto& e : entries) {
        if (!e.is_list_item && e.key == parent_key && e.indent == parent_indent) {
            in_section = true;
            continue;
        }
        if (in_section) {
            if (e.indent <= parent_indent) break;
            result.push_back(e);
        }
    }
    return result;
}

// Helper: find scalar in section at any indent matching key
static std::string section_find(const std::vector<YamlEntry>& sec,
                                 const std::string& key,
                                 const std::string& default_val = "") {
    for (auto& e : sec) {
        if (!e.is_list_item && e.key == key) return e.value;
    }
    return default_val;
}

static int to_int(const std::string&to_int s, int def = 0) {
    if (s.empty()) return def;
    try { return std::stoi(s); }
    catch (...) { return def; }
}

static bool to_bool(const std::string& s, bool def = false) {
    if (s == "true" || s == "yes" || s == "1") return true;
    if (s == "false" || s == "no" || s == "0") return false;
    return def;
}

} // anonymous namespace

// ── Public API ─────────────────────────────────────────────────────────────

ExperimentConfig load_config(const std::string& path) {
    auto entries = tokenize(path);
    ExperimentConfig cfg;

    // Top-level scalars
    cfg.llama_bench_binary = find_scalar(entries, "llama_bench_binary");
    cfg.llama_cli_binary   = find_scalar(entries, "llama_cli_binary", "");
    cfg.output_dir         = find_scalar(entries, "output_dir", "results");

    // --- experiments section ---
    auto exp_sec = find_section(entries, "experiments", 0);
    // Indent of items inside experiments: typically 2
    // Find the indent of first entry in exp_sec
    int exp_indent = exp_sec.empty() ? 2 : exp_sec[0].indent;

    // threads list
    {
        auto tlist = find_list(exp_sec, "threads", exp_indent);
        for (auto& t : tlist) {
            int v = to_int(t, -1);
            if (v > 0) cfg.thread_counts.push_back(v);
        }
        if (cfg.thread_counts.empty()) cfg.thread_counts.push_back(4);
    }

    // affinity list
    {
        auto alist = find_list(exp_sec, "affinity", exp_indent);
        for (auto& a : alist) cfg.cpu_affinities.push_back(a);
        if (cfg.cpu_affinities.empty()) cfg.cpu_affinities.push_back("none");
    }

    // Simple scalars under experiments
    cfg.warmup_runs      = to_int(section_find(exp_sec, "warmup_runs"), 2);
    cfg.measurement_runs = to_int(section_find(exp_sec, "measurement_runs"), 5);
    cfg.timeout_seconds  = to_int(section_find(exp_sec, "timeout_seconds"), 600);
    cfg.n_prompt         = to_int(section_find(exp_sec, "n_prompt"), 512);
    cfg.n_gen            = to_int(section_find(exp_sec, "n_gen"), 128);
    cfg.batch_size       = to_int(section_find(exp_sec, "batch_size"), 512);
    cfg.ubatch_size      = to_int(section_find(exp_sec, "ubatch_size"), 512);
    cfg.load_mode        = section_find(exp_sec, "load_mode", "mmap");
    cfg.perf_enabled     = to_bool(section_find(exp_sec, "perf_enabled", "true"), true);
    cfg.perf_hardware_events = to_bool(section_find(exp_sec, "perf_hardware_events", "true"), true);
    cfg.perf_uncore_events   = to_bool(section_find(exp_sec, "perf_uncore_events", "false"), false);

    // models list (each is a mapping with name/path/quantization)
    // Format:
    //   models:
    //     - name: "foo"
    //       path: "/path/to/model.gguf"
    //       quantization: "Q8_0"
    //
    // We collect all list items in the models block, then look for the
    // sub-keys that follow each list item marker.
    {
        // Find the models section inside experiments
        auto models_sec = find_section(exp_sec, "models", exp_indent);

        // Walk through looking for list items followed by key-value sub-entries
        ModelConfig current;
        bool in_model = false;

        for (size_t i = 0; i < models_sec.size(); ++i) {
            auto& e = models_sec[i];
            if (e.is_list_item) {
                // Save previous model if valid
                if (in_model && !current.path.empty()) {
                    cfg.models.push_back(current);
                }
                current = ModelConfig{};
                // The list item value itself might be a name or empty
                in_model = true;
                // If the list item has a value like "- name: foo", that's unusual YAML;
                // our format uses separate sub-keys
            } else if (in_model && !e.is_list_item) {
                if (e.key == "name")         current.name         = e.value;
                else if (e.key == "path")         current.path         = e.value;
                else if (e.key == "quantization") current.quantization = e.value;
            }
        }
        if (in_model && !current.path.empty()) {
            cfg.models.push_back(current);
        }

        // Fill in default name from path if missing
        for (auto& m : cfg.models) {
            if (m.name.empty()) {
                auto pos = m.path.rfind('/');
                m.name = (pos != std::string::npos) ? m.path.substr(pos+1) : m.path;
            }
        }
    }

    // Validate required fields
    if (cfg.llama_bench_binary.empty()) {
        throw std::runtime_error("Config missing required field: llama_bench_binary");
    }
    if (cfg.models.empty()) {
        throw std::runtime_error("Config has no models defined");
    }

    return cfg;
}

void print_config_summary(const ExperimentConfig& cfg) {
    std::cout << "=== Experiment Configuration ===\n";
    std::cout << "  llama-bench : " << cfg.llama_bench_binary << "\n";
    std::cout << "  output_dir  : " << cfg.output_dir << "\n";
    std::cout << "  Models      :\n";
    for (auto& m : cfg.models) {
        std::cout << "    - " << m.name << " [" << m.quantization << "] -> " << m.path << "\n";
    }
    std::cout << "  Threads     : ";
    for (int t : cfg.thread_counts) std::cout << t << " ";
    std::cout << "\n";
    std::cout << "  Affinities  : ";
    for (auto& a : cfg.cpu_affinities) std::cout << a << " ";
    std::cout << "\n";
    std::cout << "  Warmup runs : " << cfg.warmup_runs << "\n";
    std::cout << "  Measure runs: " << cfg.measurement_runs << "\n";
    std::cout << "  n_prompt    : " << cfg.n_prompt << "\n";
    std::cout << "  n_gen       : " << cfg.n_gen << "\n";
    std::cout << "  perf        : " << (cfg.perf_enabled ? "enabled" : "disabled") << "\n";
    std::cout << "================================\n";
}

} // namespace llmchar
