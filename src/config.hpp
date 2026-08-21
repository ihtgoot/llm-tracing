// config.hpp
// Simple YAML subset parser for experiment.yaml.
// We parse only the specific keys we need — no general YAML support required.
// Supports: key: value, lists (- item), and nested blocks (indentation-based).

#pragma once

#include "types.hpp"
#include <string>
#include <stdexcept>

namespace llmchar {

/// Load experiment configuration from a YAML file.
/// Throws std::runtime_error on parse failure.
ExperimentConfig load_config(const std::string& path);

/// Print config summary for verification.
void print_config_summary(const ExperimentConfig& cfg);

} // namespace llmchar
