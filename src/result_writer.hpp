// result_writer.hpp
// Writes experiment results to CSV and JSON files.
// Design: one CSV row per run (raw data); JSON for metadata and aggregates.
// Never silently drops data — failures are recorded explicitly.

#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace llmchar {

class ResultWriter {
public:
    /// Initialize the writer with the output directory path.
    /// Creates subdirectories raw/, aggregate/, metadata/ if needed.
    explicit ResultWriter(const std::string& output_dir);

    /// Initialize output files (write CSV headers).
    /// Must be called before any write operations.
    bool initialize();

    /// Write one RunResult as a CSV row to raw/runs.csv.
    void write_run(const RunResult& run);

    /// Write experiment metadata as JSON (system info + config).
    void write_experiment_metadata(const std::string& experiment_id,
                                   const SystemInfo& sys,
                                   const ExperimentConfig& cfg);

    /// Write aggregate statistics CSV for a set of measurement runs.
    /// warmup_runs are included in the file but flagged separately.
    void write_aggregate(const std::string& experiment_id,
                         const std::string& model_name,
                         const std::string& quantization,
                         int threads,
                         const std::string& affinity,
                         const std::vector<RunResult>& all_runs,
                         int warmup_count);

    const std::string& output_dir() const { return output_dir_; }

private:
    std::string output_dir_;
    std::string raw_csv_path_;
    std::string aggregate_csv_path_;
    bool        raw_header_written_      = false;
    bool        aggregate_header_written_= false;

    void ensure_raw_header();
    void ensure_aggregate_header();

    static AggregateStats compute_stats(const std::vector<double>& values);
    static std::string opt_str(const std::optional<int64_t>& v);
    static std::string opt_double_str(const std::optional<double>& v);
};

} // namespace llmchar
