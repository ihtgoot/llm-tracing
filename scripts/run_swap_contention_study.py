#!/usr/bin/env python3
"""
run_swap_contention_study.py
Executes the Swap & Memory Placement Study across:
- 14 Core Topology Configs: 1p, 1e, 2p, 2e, 4p, 4e, 8p, 12e, 1e+1p, 1e+2p, 2p+1e, 2e+2p, 4e+4p, default
- 3 Quantization Precisions: Q8_0, Q4_0, Q4_K_M
- 4 Memory/Swap Placement Modes:
  1. pure_ram       (Pre-allocated & locked in physical DRAM via mlock)
  2. pure_swap      (100% of model buffer evicted to Swap via Linux MADV_PAGEOUT)
  3. default_os     (Default lazy mmap demand paging)
  4. hybrid_50_50   (50% in physical RAM, 50% evicted to Swap via MADV_PAGEOUT)
Generates high-resolution comparative Matplotlib plots in results/plots/ and docs/plots/
"""

import os
import sys
import subprocess
import csv
import json
import mmap
import ctypes
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MODELS = [
    {
        "name": "gemma-3-270m-Q8_0",
        "path": "/home/ihtgoot/.cache/huggingface/hub/models--ggml-org--gemma-3-270m-it-GGUF/snapshots/e7647be17ae1108f2f605ed061ca0608b171afff/gemma-3-270m-it-Q8_0.gguf",
        "quantization": "Q8_0"
    },
    {
        "name": "gemma-3-270m-Q4_0",
        "path": "/home/ihtgoot/project/models/gemma-3-270m-it-Q4_0.gguf",
        "quantization": "Q4_0"
    },
    {
        "name": "gemma-3-270m-Q4_K_M",
        "path": "/home/ihtgoot/project/models/gemma-3-270m-it-Q4_K_M.gguf",
        "quantization": "Q4_K_M"
    }
]

HARNESS_BIN   = "./build/llm-charharness"
RELEASE_BENCH = "/home/ihtgoot/project/llama.cpp/build-release/bin/llama-bench"

CONFIG_SUITE = [
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
    ("default",   "none",               4),
]

MEMORY_MODES = [
    ("pure_ram",     "mlock", 0.0),
    ("default_os",   "mmap",  0.0),
    ("hybrid_50_50", "mmap",  0.5),
    ("pure_swap",    "mmap",  1.0),
]

def evict_file_pages(model_path, evict_fraction):
    if evict_fraction <= 0.0:
        return
    try:
        file_size = os.path.getsize(model_path)
        fd = os.open(model_path, os.O_RDWR)
        mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_WRITE)
        # Touch pages to populate
        for i in range(0, file_size, 4096):
            _ = mm[i]
        libc = ctypes.CDLL(None)
        start_offset = int(file_size * (1.0 - evict_fraction))
        length = int(file_size * evict_fraction)
        # MADV_PAGEOUT = 21 on Linux
        ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)) + start_offset)
        libc.madvise(ptr, ctypes.c_size_t(length), ctypes.c_int(21))
        mm.close()
        os.close(fd)
    except Exception as e:
        print(f"Warning: Failed to evict pages ({e})")

def write_yaml(output_dir, model, threads, affinity, load_mode, yaml_path):
    content = f"""llama_bench_binary: "{RELEASE_BENCH}"
output_dir: "{output_dir}"

experiments:
  models:
    - name: "{model['name']}"
      path: "{model['path']}"
      quantization: "{model['quantization']}"

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
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(f"    ERROR executing {output_dir}")

def parse_mean_toks(summary_csv):
    if not os.path.exists(summary_csv):
        return 0.0
    with open(summary_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['metric'].startswith('bench_tokens_per_sec'):
                return float(r['mean'])
    return 0.0

def save_plots(fig, base_name):
    for d in ["results/plots", "docs/plots", "/home/ihtgoot/.gemini/antigravity-ide/brain/8d77327f-512d-44a2-abab-7ac41ed1e6a9/plots"]:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, base_name), dpi=300)
    print(f"Saved plot: {base_name} across output directories")

def plot_swap_modes_summary(results_data):
    labels = [c[0] for c in CONFIG_SUITE]
    
    # Plot 1: Line Graph for Q8_0 across 4 Swap Modes
    fig, ax = plt.subplots(figsize=(13, 6.5))
    q8_data = results_data["Q8_0"]
    colors = {
        "pure_ram": "indigo",
        "default_os": "blue",
        "hybrid_50_50": "darkorange",
        "pure_swap": "crimson"
    }
    style_names = {
        "pure_ram": "Purely on RAM (mlock)",
        "default_os": "Default OS (mmap)",
        "hybrid_50_50": "50-50 RAM/Swap Hybrid",
        "pure_swap": "Purely on Swap (100% Pageout)"
    }

    for mode, _, _ in MEMORY_MODES:
        vals = [q8_data[mode][c[0]] for c in CONFIG_SUITE]
        ax.plot(labels, vals, marker='o', linewidth=2.5, color=colors[mode], label=style_names[mode])

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_xlabel('CPU Core Topology Configuration')
    ax.set_title('Impact of Memory/Swap Placement across Topologies (Gemma-3 270M Q8_0)')
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_plots(fig, "plot_swap_modes_across_topologies_line.png")
    plt.close(fig)

    # Plot 2: Bar Graph for Q8_0 across 4 Swap Modes
    x = range(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6.5))
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    for idx, (mode, _, _) in enumerate(MEMORY_MODES):
        vals = [q8_data[mode][c[0]] for c in CONFIG_SUITE]
        ax.bar([i + offsets[idx] for i in x], vals, width, label=style_names[mode], color=colors[mode], alpha=0.85)

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_title('Memory Placement & Swap Pressure Bar Comparison (Q8_0)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_plots(fig, "plot_swap_modes_across_topologies_bar.png")
    plt.close(fig)

    # Plot 3: Quantization Sensitivity under Pure Swap vs Pure RAM
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for m in MODELS:
        q_label = m["quantization"]
        ram_vals  = [results_data[q_label]["pure_ram"][c[0]] for c in CONFIG_SUITE]
        swap_vals = [results_data[q_label]["pure_swap"][c[0]] for c in CONFIG_SUITE]
        ax.plot(labels, ram_vals, marker='o', linewidth=2.2, label=f"{q_label} (Pure RAM)")
        ax.plot(labels, swap_vals, marker='x', linewidth=2.2, linestyle='--', label=f"{q_label} (Pure Swap)")

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_xlabel('CPU Core Topology Configuration')
    ax.set_title('Quantization Sensitivity under Pure RAM vs. Pure Swap Contention')
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_plots(fig, "plot_swap_quantization_comparison_line.png")
    plt.close(fig)

def main():
    if not os.path.exists(HARNESS_BIN):
        print("Harness missing!")
        sys.exit(1)

    print("=========================================================================")
    print("Executing Swap & Memory Placement Study")
    print("Topologies: 14 | Quantizations: 3 | Memory Modes: 4 (RAM, OS, 50-50, Swap)")
    print("=========================================================================")

    results_data = {m["quantization"]: {mode[0]: {} for mode in MEMORY_MODES} for m in MODELS}

    for m in MODELS:
        q_label = m["quantization"]
        print(f"\n--- Model Quantization: {q_label} ---")
        for mode_name, load_mode, evict_frac in MEMORY_MODES:
            print(f"  > Memory Mode: {mode_name} (load_mode={load_mode}, evict={evict_frac*100}%)")
            for label, mask, threads in CONFIG_SUITE:
                out_dir = f"results/swap_study/{q_label}_{mode_name}_{label}"
                yaml_p  = f"config/swap_study/{q_label}_{mode_name}_{label}.yaml"
                write_yaml(out_dir, m, threads, mask, load_mode, yaml_p)
                
                # Evict file pages to swap if fraction > 0
                if evict_frac > 0.0:
                    evict_file_pages(m["path"], evict_frac)

                run_experiment(yaml_p, out_dir)
                results_data[q_label][mode_name][label] = parse_mean_toks(f"{out_dir}/aggregate/summary.csv")

    os.makedirs("results/swap_study", exist_ok=True)
    with open("results/swap_study/summary.json", "w") as f:
        json.dump(results_data, f, indent=2)

    plot_swap_modes_summary(results_data)

    print("\nSwap & Memory Placement Study completed successfully!")

if __name__ == "__main__":
    main()
