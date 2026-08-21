#!/usr/bin/env python3
"""
run_topology_memory_experiment.py
Runs the exact core topology & memory allocation experiment requested:
Configs: 1e, 1p, 2e, 2p, 4e, 4p, 12e, 8p, 1e+1p, 1e+2p, 2p+1e, 2e+2p, 4e+4p, default
For each config:
  - 3 runs using mlock (Pre-allocated memory before inference)
  - 3 runs using mmap (Self-allocated / lazy memory mapping)
Plots comparative throughput & latency figures.
"""

import os
import sys
import subprocess
import csv
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MODEL_PATH = "/home/ihtgoot/.cache/huggingface/hub/models--ggml-org--gemma-3-270m-it-GGUF/snapshots/e7647be17ae1108f2f605ed061ca0608b171afff/gemma-3-270m-it-Q8_0.gguf"
HARNESS_BIN   = "./build/llm-charharness"
RELEASE_BENCH = "/home/ihtgoot/project/llama.cpp/build-release/bin/llama-bench"

CONFIG_SUITE = [
    # (Label, affinity_mask, threads)
    ("1p",        "0",                  1),
    ("1e",        "16",                 1),
    ("2p",        "0,2",                2),
    ("2e",        "16,17",              2),
    ("4p",        "0,2,4,6",            4),
    ("4e",        "16-19",              4),
    ("8p",        "0,2,4,6,8,10,12,14", 8),
    ("12e",       "16-27",              12),
    ("1e+1p",     "16,0",               2),
    ("1e+2p",     "16,0,2",             3),
    ("2p+1e",     "0,2,16",             3),
    ("2e+2p",     "16,17,0,2",          4),
    ("4e+4p",     "16-19,0,2,4,6",      8),
    ("whole_cpu", "0-27",               28),
    ("default",   "none",               4),
]

def write_yaml(output_dir, affinity, threads, load_mode, yaml_path):
    content = f"""llama_bench_binary: "{RELEASE_BENCH}"
output_dir: "{output_dir}"

experiments:
  models:
    - name: "gemma-3-270m-Q8_0"
      path: "{MODEL_PATH}"
      quantization: "Q8_0"

  threads:
    - {threads}

  affinity:
    - "{affinity}"

  warmup_runs: 1
  measurement_runs: 3
  timeout_seconds: 300

  n_prompt: 512
  n_gen: 128
  batch_size: 512
  ubatch_size: 512
  load_mode: "{load_mode}"

  perf_enabled: true
  perf_hardware_events: true
  perf_uncore_events: false
"""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)

def run_experiment(config_path, output_dir):
    cmd = [HARNESS_BIN, "--config", config_path, "--output-dir", output_dir]
    print(f"--> Running: {output_dir}")
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(f"    ERROR running {output_dir}")

def parse_mean_toks(summary_csv):
    if not os.path.exists(summary_csv):
        return 0.0
    with open(summary_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['metric'].startswith('bench_tokens_per_sec'):
                return float(r['mean'])
    return 0.0

def generate_plot(results_dict):
    labels = [c[0] for c in CONFIG_SUITE]
    prealloc_vals = [results_dict[c[0]]["mlock"] for c in CONFIG_SUITE]
    selfalloc_vals = [results_dict[c[0]]["mmap"] for c in CONFIG_SUITE]

    x = range(len(labels))
    width = 0.35

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.autolayout': True,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], prealloc_vals, width, label='Pre-Allocated Memory (mlock)', color='indigo', alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], selfalloc_vals, width, label='Self-Allocated Memory (mmap)', color='mediumseagreen', alpha=0.85)

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_title('Topology & Memory Allocation Strategy Impact (Gemma-3 270M Q8_0)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.legend()

    # Add numeric labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    out_img = os.path.join(plots_dir, "plot_topology_prealloc_vs_selfalloc.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"\nSaved topology comparison bar plot: {out_img}")

    # Generate Line Graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(labels, prealloc_vals, marker='o', linewidth=2.5, color='indigo', label='Pre-Allocated Memory (mlock)')
    ax.plot(labels, selfalloc_vals, marker='s', linewidth=2.5, color='mediumseagreen', linestyle='--', label='Self-Allocated Memory (mmap)')

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_xlabel('Core Topology Configuration')
    ax.set_title('Topology & Memory Allocation Line Trend (Gemma-3 270M Q8_0)')
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.legend()

    # Annotate points
    for i, (p, s) in enumerate(zip(prealloc_vals, selfalloc_vals)):
        ax.annotate(f"{p:.1f}", (labels[i], p), textcoords="offset points", xytext=(0,6), ha='center', fontsize=8, color='indigo')
        ax.annotate(f"{s:.1f}", (labels[i], s), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=8, color='darkgreen')

    fig.tight_layout()
    out_line_img = os.path.join(plots_dir, "plot_topology_prealloc_vs_selfalloc_line.png")
    plt.savefig(out_line_img, dpi=300)
    plt.close()
    print(f"Saved topology comparison line plot: {out_line_img}")

def main():
    if not os.path.exists(HARNESS_BIN):
        print("Harness binary missing!")
        sys.exit(1)

    results_data = {c[0]: {"mlock": 0.0, "mmap": 0.0} for c in CONFIG_SUITE}

    print("=========================================================================")
    print("Executing Topology & Memory Allocation Experiment Suite")
    print("Configs:", [c[0] for c in CONFIG_SUITE])
    print("=========================================================================")

    for label, mask, threads in CONFIG_SUITE:
        # Pre-allocated memory (mlock)
        out_mlock = f"results/topology_study/{label}_mlock"
        yaml_mlock = f"config/topology_study/{label}_mlock.yaml"
        write_yaml(out_mlock, mask, threads, "mlock", yaml_mlock)
        run_experiment(yaml_mlock, out_mlock)
        results_data[label]["mlock"] = parse_mean_toks(f"{out_mlock}/aggregate/summary.csv")

        # Self-allocated memory (mmap)
        out_mmap = f"results/topology_study/{label}_mmap"
        yaml_mmap = f"config/topology_study/{label}_mmap.yaml"
        write_yaml(out_mmap, mask, threads, "mmap", yaml_mmap)
        run_experiment(yaml_mmap, out_mmap)
        results_data[label]["mmap"] = parse_mean_toks(f"{out_mmap}/aggregate/summary.csv")

    generate_plot(results_data)

    os.makedirs("results/topology_study", exist_ok=True)
    with open("results/topology_study/summary.json", "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    main()
