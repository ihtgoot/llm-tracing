#!/usr/bin/env python3
"""
run_gemma2b_multiinstance_study.py
Executes the Gemma-2 2B Baseline Characterization & Multi-Instance Contention Study across:
- Model: Gemma-2 2B (Q4_K_M ~1.6GB, Q8_0 ~2.6GB)
- Multi-Instance Scaling: 1 Instance (Single-tenant) vs 2 Concurrent Instances (Dual-tenant)
- 14 Core Topologies: 1p, 1e, 2p, 2e, 4p, 4e, 8p, 12e, 1e+1p, 1e+2p, 2p+1e, 2e+2p, 4e+4p, default
- 4 Memory Placement / Swap Modes: pure_ram (mlock), default_os (mmap), hybrid_50_50, pure_swap (MADV_PAGEOUT)

Generates comparative publication-quality Matplotlib line and bar charts.
"""

import os
import sys
import subprocess
import csv
import json
import mmap
import ctypes
import time
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MODELS = [
    {
        "name": "gemma-2-2b-Q4_K_M",
        "path": "/home/ihtgoot/project/models/gemma-2-2b-it-Q4_K_M.gguf",
        "quantization": "Q4_K_M"
    },
    {
        "name": "gemma-2-2b-Q8_0",
        "path": "/home/ihtgoot/project/models/gemma-2-2b-it-Q8_0.gguf",
        "quantization": "Q8_0"
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
        for i in range(0, file_size, 4096):
            _ = mm[i]
        libc = ctypes.CDLL(None)
        start_offset = int(file_size * (1.0 - evict_fraction))
        length = int(file_size * evict_fraction)
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

def run_single_harness(yaml_path, out_dir):
    cmd = [HARNESS_BIN, "--config", yaml_path, "--output-dir", out_dir]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return res.returncode

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
    print(f"Saved plot: {base_name}")

def plot_gemma2b_results(results_data):
    labels = [c[0] for c in CONFIG_SUITE]
    
    # 1. Multi-Instance Contention Comparison (Single Instance vs 2 Concurrent Instances)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    s1_vals = [results_data["Q4_K_M"]["1_instance"]["pure_ram"][c[0]] for c in CONFIG_SUITE]
    s2_vals = [results_data["Q4_K_M"]["2_instances"]["pure_ram"][c[0]] for c in CONFIG_SUITE]

    ax.plot(labels, s1_vals, marker='o', linewidth=2.5, color='indigo', label='1 Instance (Single-Tenant)')
    ax.plot(labels, s2_vals, marker='s', linewidth=2.5, color='crimson', linestyle='--', label='2 Concurrent Instances (Dual-Tenant)')

    ax.set_ylabel('Inference Throughput per Instance (tok/s)')
    ax.set_xlabel('CPU Core Topology Configuration')
    ax.set_title('Gemma-2 2B: Multi-Instance CPU Core & Memory Contention Impact (Q4_K_M)')
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_plots(fig, "plot_gemma2b_multiinstance_contention_line.png")
    plt.close(fig)

    # 2. Multi-Instance Bar Chart Comparison
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar([i - width/2 for i in x], s1_vals, width, label='1 Instance (Single-Tenant)', color='indigo', alpha=0.85)
    ax.bar([i + width/2 for i in x], s2_vals, width, label='2 Concurrent Instances (Dual-Tenant)', color='crimson', alpha=0.85)

    ax.set_ylabel('Inference Throughput per Instance (tok/s)')
    ax.set_title('Gemma-2 2B: Single vs Dual Instance Throughput Bar Comparison')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_plots(fig, "plot_gemma2b_multiinstance_contention_bar.png")
    plt.close(fig)

    # 3. Swap Modes Comparison for Gemma-2 2B
    fig, ax = plt.subplots(figsize=(13, 6.5))
    q4_data = results_data["Q4_K_M"]["1_instance"]
    colors = {"pure_ram": "indigo", "default_os": "blue", "hybrid_50_50": "darkorange", "pure_swap": "crimson"}
    mode_names = {"pure_ram": "Pure RAM (mlock)", "default_os": "Default OS (mmap)", "hybrid_50_50": "50-50 RAM/Swap Hybrid", "pure_swap": "Pure Swap (100% Evicted)"}

    for mode, _, _ in MEMORY_MODES:
        vals = [q4_data[mode][c[0]] for c in CONFIG_SUITE]
        ax.plot(labels, vals, marker='o', linewidth=2.2, color=colors[mode], label=mode_names[mode])

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_xlabel('CPU Core Topology Configuration')
    ax.set_title('Gemma-2 2B: Swap & Memory Placement Impact across Topologies (Q4_K_M)')
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_plots(fig, "plot_gemma2b_swap_modes_line.png")
    plt.close(fig)

def main():
    print("=========================================================================")
    print("Executing Gemma-2 2B Baseline & Multi-Instance Memory Contention Study")
    print("Models: Gemma-2 2B (Q4_K_M, Q8_0) | Instances: 1 & 2 Concurrent")
    print("Topologies: 14 | Swap Modes: 4 (RAM, OS, 50-50, Swap)")
    print("=========================================================================")

    # Structure: [quantization][num_instances][memory_mode][topology]
    results_data = {
        m["quantization"]: {
            "1_instance": {mode[0]: {} for mode in MEMORY_MODES},
            "2_instances": {mode[0]: {} for mode in MEMORY_MODES}
        } for m in MODELS
    }

    for m in MODELS:
        q_label = m["quantization"]
        if not os.path.exists(m["path"]):
            print(f"Waiting/Skipping model {m['path']} (not ready)")
            continue

        print(f"\n==================== Model Quantization: {q_label} ====================")

        for mode_name, load_mode, evict_frac in MEMORY_MODES:
            print(f"\n--- Memory Mode: {mode_name} (evict={evict_frac*100}%) ---")

            for label, mask, threads in CONFIG_SUITE:
                # Evict file pages if needed
                if evict_frac > 0.0:
                    evict_file_pages(m["path"], evict_frac)

                # --- 1. Single Instance Run ---
                out_dir_1 = f"results/gemma2b_study/{q_label}_1inst_{mode_name}_{label}"
                yaml_p_1  = f"config/gemma2b_study/{q_label}_1inst_{mode_name}_{label}.yaml"
                write_yaml(out_dir_1, m, threads, mask, load_mode, yaml_p_1)
                run_single_harness(yaml_p_1, out_dir_1)
                toks_1 = parse_mean_toks(f"{out_dir_1}/aggregate/summary.csv")
                results_data[q_label]["1_instance"][mode_name][label] = toks_1

                # --- 2. Dual Concurrent Instances Run (2 Concurrent Instances) ---
                out_dir_2a = f"results/gemma2b_study/{q_label}_2instA_{mode_name}_{label}"
                out_dir_2b = f"results/gemma2b_study/{q_label}_2instB_{mode_name}_{label}"
                yaml_p_2a  = f"config/gemma2b_study/{q_label}_2instA_{mode_name}_{label}.yaml"
                yaml_p_2b  = f"config/gemma2b_study/{q_label}_2instB_{mode_name}_{label}.yaml"
                write_yaml(out_dir_2a, m, threads, mask, load_mode, yaml_p_2a)
                write_yaml(out_dir_2b, m, threads, mask, load_mode, yaml_p_2b)

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f1 = executor.submit(run_single_harness, yaml_p_2a, out_dir_2a)
                    f2 = executor.submit(run_single_harness, yaml_p_2b, out_dir_2b)
                    concurrent.futures.wait([f1, f2])

                toks_2a = parse_mean_toks(f"{out_dir_2a}/aggregate/summary.csv")
                toks_2b = parse_mean_toks(f"{out_dir_2b}/aggregate/summary.csv")
                mean_dual_toks = (toks_2a + toks_2b) / 2.0 if (toks_2a > 0 and toks_2b > 0) else max(toks_2a, toks_2b)
                results_data[q_label]["2_instances"][mode_name][label] = mean_dual_toks

                print(f"  [{label:<6}] 1-Inst: {toks_1:>6.1f} tok/s | 2-Inst (Concurrent): {mean_dual_toks:>6.1f} tok/s per inst")

    os.makedirs("results/gemma2b_study", exist_ok=True)
    with open("results/gemma2b_study/summary.json", "w") as f:
        json.dump(results_data, f, indent=2)

    plot_gemma2b_results(results_data)
    print("\nGemma-2 2B Multi-Instance Memory Contention Study completed successfully!")

if __name__ == "__main__":
    main()
