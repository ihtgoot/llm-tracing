#!/usr/bin/env python3
"""
run_advanced_topology_matrix.py
Executes two comprehensive experimental studies:
Study 1: 14 Topology Configs x Quantization (Q8_0, Q4_0, Q4_K_M)
Study 2: 14 Topology Configs x Vector Acceleration (AVX-Native vs No-AVX Fallback)
Generates high-resolution comparative Matplotlib plots in results/plots/ and docs/plots/
"""

import os
import sys
import subprocess
import csv
import json
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
NOAVX_BENCH   = "/home/ihtgoot/project/llama.cpp/build-noavx/bin/llama-bench"

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

def write_yaml(binary_path, output_dir, model, threads, affinity, load_mode="mmap", warmup=1, measurements=3, yaml_path=""):
    content = f"""llama_bench_binary: "{binary_path}"
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

  warmup_runs: {warmup}
  measurement_runs: {measurements}
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

def plot_quant_study(quant_results):
    labels = [c[0] for c in CONFIG_SUITE]
    q8_vals = [quant_results["Q8_0"][c[0]] for c in CONFIG_SUITE]
    q4_vals = [quant_results["Q4_0"][c[0]] for c in CONFIG_SUITE]
    q4k_vals = [quant_results["Q4_K_M"][c[0]] for c in CONFIG_SUITE]

    # Line Plot
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(labels, q4_vals, marker='s', linewidth=2.5, color='crimson', label='Q4_0 (4-bit Standard)')
    ax.plot(labels, q4k_vals, marker='^', linewidth=2.5, color='darkgreen', label='Q4_K_M (4-bit Mixture)')
    ax.plot(labels, q8_vals, marker='o', linewidth=2.5, color='darkblue', label='Q8_0 (8-bit Baseline)')

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_xlabel('CPU Core Topology Configuration')
    ax.set_title('Impact of Core Topology across Quantization Precision (Gemma-3 270M)')
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_plots(fig, "plot_topology_quantization_line.png")
    plt.close(fig)

    # Bar Plot
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar([i - width for i in x], q4_vals, width, label='Q4_0 (4-bit)', color='crimson', alpha=0.85)
    ax.bar(x, q4k_vals, width, label='Q4_K_M (4-bit K)', color='darkgreen', alpha=0.85)
    ax.bar([i + width for i in x], q8_vals, width, label='Q8_0 (8-bit)', color='darkblue', alpha=0.85)

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_title('Topology Throughput Comparison across Quantizations (Gemma-3 270M)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_plots(fig, "plot_topology_quantization_bar.png")
    plt.close(fig)

def plot_avx_study(avx_results):
    labels = [c[0] for c in CONFIG_SUITE]
    avx_on_vals  = [avx_results["AVX_ON"][c[0]] for c in CONFIG_SUITE]
    avx_off_vals = [avx_results["AVX_OFF"][c[0]] for c in CONFIG_SUITE]

    # Line Plot
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(labels, avx_on_vals, marker='o', linewidth=2.5, color='forestgreen', label='AVX-Native Release (-march=native)')
    ax.plot(labels, avx_off_vals, marker='x', linewidth=2.5, color='firebrick', linestyle='--', label='No-AVX Fallback (SSE4.2/F16C/BMI2)')

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_xlabel('CPU Core Topology Configuration')
    ax.set_title('Topology Performance: AVX-Native vs. No-AVX Fallback (Gemma-3 270M Q8_0)')
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_plots(fig, "plot_topology_avx_line.png")
    plt.close(fig)

    # Bar Plot
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar([i - width/2 for i in x], avx_on_vals, width, label='AVX-Native Release', color='forestgreen', alpha=0.85)
    ax.bar([i + width/2 for i in x], avx_off_vals, width, label='No-AVX Fallback', color='firebrick', alpha=0.85)

    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_title('Vector Acceleration Impact across 14 Core Topologies')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_plots(fig, "plot_topology_avx_bar.png")
    plt.close(fig)

def main():
    if not os.path.exists(HARNESS_BIN):
        print("Harness missing!")
        sys.exit(1)

    print("=========================================================================")
    print("Study 1: Core Topology x Quantization Precision (Q8_0, Q4_0, Q4_K_M)")
    print("=========================================================================")

    quant_results = {m["quantization"]: {} for m in MODELS}

    for m in MODELS:
        q_label = m["quantization"]
        print(f"\n--- Model Quantization: {q_label} ---")
        for label, mask, threads in CONFIG_SUITE:
            out_dir = f"results/advanced_topology/quant_{q_label}_{label}"
            yaml_p  = f"config/advanced_topology/quant_{q_label}_{label}.yaml"
            write_yaml(RELEASE_BENCH, out_dir, m, threads, mask, load_mode="mmap", yaml_path=yaml_p)
            run_experiment(yaml_p, out_dir)
            quant_results[q_label][label] = parse_mean_toks(f"{out_dir}/aggregate/summary.csv")

    os.makedirs("results/advanced_topology", exist_ok=True)
    with open("results/advanced_topology/quant_summary.json", "w") as f:
        json.dump(quant_results, f, indent=2)

    plot_quant_study(quant_results)

    print("\n=========================================================================")
    print("Study 2: Core Topology x Vector Acceleration (AVX Native vs No-AVX)")
    print("=========================================================================")

    avx_results = {"AVX_ON": {}, "AVX_OFF": {}}
    m_q8 = MODELS[0]

    for label, mask, threads in CONFIG_SUITE:
        # AVX ON
        out_on  = f"results/advanced_topology/avx_on_{label}"
        yaml_on = f"config/advanced_topology/avx_on_{label}.yaml"
        write_yaml(RELEASE_BENCH, out_on, m_q8, threads, mask, load_mode="mmap", yaml_path=yaml_on)
        run_experiment(yaml_on, out_on)
        avx_results["AVX_ON"][label] = parse_mean_toks(f"{out_on}/aggregate/summary.csv")

        # AVX OFF
        out_off  = f"results/advanced_topology/avx_off_{label}"
        yaml_off = f"config/advanced_topology/avx_off_{label}.yaml"
        write_yaml(NOAVX_BENCH, out_off, m_q8, threads, mask, load_mode="mmap", yaml_path=yaml_off)
        run_experiment(yaml_off, out_off)
        avx_results["AVX_OFF"][label] = parse_mean_toks(f"{out_off}/aggregate/summary.csv")

    with open("results/advanced_topology/avx_summary.json", "w") as f:
        json.dump(avx_results, f, indent=2)

    plot_avx_study(avx_results)

    print("\nAll advanced topology experiments completed successfully!")

if __name__ == "__main__":
    main()
