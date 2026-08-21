#!/usr/bin/env python3
"""
generate_plots_and_report.py
Parses experiment matrix results and microbench JSON to generate
matplotlib visualization plots saved in results/plots/ and outputs summary data.
"""

import os
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Non-interactive backend

PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def read_summary_csv(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

def read_raw_csv(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

# -------------------------------------------------------------------
# Plot 1: Quantization & Thread Scaling (Q8_0, Q4_0, Q4_K_M)
# -------------------------------------------------------------------
def plot_quant_thread_scaling():
    summary_path = "results/matrix_threads_quant/aggregate/summary.csv"
    records = read_summary_csv(summary_path)
    if not records:
        print(f"Skipping plot_quant_thread_scaling: {summary_path} not found")
        return

    data = {} # (model_name, threads) -> mean_ts
    for r in records:
        if r['metric'].startswith('bench_tokens_per_sec'):
            m = r['model_name']
            t = int(r['threads'])
            data[(m, t)] = float(r['mean'])

    models = sorted(list(set(m for m, t in data.keys())))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {'gemma-3-270m-Q8_0': 'darkblue', 'gemma-3-270m-Q4_0': 'crimson', 'gemma-3-270m-Q4_K_M': 'darkgreen'}
    markers = {'gemma-3-270m-Q8_0': 'o', 'gemma-3-270m-Q4_0': 's', 'gemma-3-270m-Q4_K_M': '^'}

    for m in models:
        threads_for_m = sorted([t for (mod, t) in data.keys() if mod == m])
        toks = [data[(m, t)] for t in threads_for_m]
        ax.plot(threads_for_m, toks, marker=markers.get(m, 'o'), color=colors.get(m, 'black'),
                linewidth=2, label=m)

    ax.set_xlabel('CPU Thread Count')
    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_title('Gemma-3 270M Throughput across Quantizations & Threads')
    ax.axvline(x=8, color='gray', linestyle='--', alpha=0.6, label='8 Physical P-cores')
    ax.axvline(x=16, color='red', linestyle='--', alpha=0.6, label='16 P-core HT threads')
    ax.legend()

    fig.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, "plot1_quant_thread_scaling.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved {plot_file}")

# -------------------------------------------------------------------
# Plot 2: Branch Misses & Cache Misses vs Core Topology
# -------------------------------------------------------------------
def plot_branch_and_cache_misses():
    summary_path = "results/matrix_affinity/aggregate/summary.csv"
    records = read_summary_csv(summary_path)
    if not records:
        print("Skipping plot_branch_and_cache_misses: summary missing")
        return

    aff_cache = {}
    for r in records:
        if r['metric'].startswith('perf_cache_misses'):
            aff = r['cpu_affinity']
            t = r['threads']
            key = f"{aff} (t={t})"
            aff_cache[key] = float(r['mean']) / 1e6 # Millions of misses

    if not aff_cache:
        return

    keys = list(aff_cache.keys())
    values = [aff_cache[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(keys, values, color='coral', alpha=0.85)
    ax.set_xlabel('LLC Cache Misses (Millions)')
    ax.set_title('LLC Cache Misses across CPU Core Affinities (P-cores vs E-cores)')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f'{w:.1f}M', va='center', fontsize=9)

    fig.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, "plot2_cache_misses_affinity.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved {plot_file}")

# -------------------------------------------------------------------
# Plot 3: Memory Load Modes (mmap vs mlock vs none)
# -------------------------------------------------------------------
def plot_memory_modes():
    modes = ["mmap", "mlock", "none"]
    hwm_vals = []
    anon_vals = []
    file_vals = []

    for mode in modes:
        raw_path = f"results/matrix_memory_{mode}/raw/runs.csv"
        records = read_raw_csv(raw_path)
        if records:
            r = [x for x in records if x['is_warmup'] == '0'][-1]
            hwm_vals.append(float(r['mem_peak_hwm_kb']) / 1024.0)
            anon_vals.append(float(r['mem_peak_anon_kb']) / 1024.0)
            file_vals.append(float(r['mem_peak_file_kb']) / 1024.0)
        else:
            hwm_vals.append(0)
            anon_vals.append(0)
            file_vals.append(0)

    x = range(len(modes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width for i in x], hwm_vals, width, label='Peak RSS (VmHWM)', color='darkred')
    ax.bar(x, anon_vals, width, label='Heap/KV-Cache (RssAnon)', color='darkorange')
    ax.bar([i + width for i in x], file_vals, width, label='Mapped Weights (RssFile)', color='navy')

    ax.set_ylabel('Memory Footprint (MiB)')
    ax.set_title('Memory Footprint by Model Load Mode (Gemma-3 270M Q8_0)')
    ax.set_xticks(x)
    ax.set_xticklabels(['mmap (Lazy Load)', 'mlock (Pinned RAM)', 'none (Heap Buffer)'])
    ax.legend()

    fig.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, "plot3_memory_modes.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved {plot_file}")

# -------------------------------------------------------------------
# Plot 4: AVX Native vs No-AVX Vectorization Impact
# -------------------------------------------------------------------
def plot_avx_impact():
    on_summary = read_summary_csv("results/matrix_avx_on/aggregate/summary.csv")
    off_summary = read_summary_csv("results/matrix_avx_off/aggregate/summary.csv")

    if not on_summary or not off_summary:
        print("Skipping plot_avx_impact: summary data missing")
        return

    threads_on = {}
    for r in on_summary:
        if r['metric'].startswith('bench_tokens_per_sec'):
            threads_on[int(r['threads'])] = float(r['mean'])

    threads_off = {}
    for r in off_summary:
        if r['metric'].startswith('bench_tokens_per_sec'):
            threads_off[int(r['threads'])] = float(r['mean'])

    common_threads = sorted(list(set(threads_on.keys()).intersection(set(threads_off.keys()))))
    toks_on = [threads_on[t] for t in common_threads]
    toks_off = [threads_off[t] for t in common_threads]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(common_threads, toks_on, marker='o', linewidth=2, color='green', label='AVX-Native Release (-march=native)')
    ax.plot(common_threads, toks_off, marker='s', linewidth=2, color='crimson', linestyle='--', label='No-AVX Fallback (SSE4.2/F16C/BMI2)')

    ax.set_xlabel('CPU Thread Count')
    ax.set_ylabel('Inference Throughput (tokens/sec)')
    ax.set_title('Vector Acceleration Impact: AVX-Native vs. No-AVX Fallback')
    ax.set_xticks(common_threads)
    ax.legend()

    fig.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, "plot4_avx_impact.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved {plot_file}")

# -------------------------------------------------------------------
# Plot 5: Google Benchmark Memory Hierarchy Characterization
# -------------------------------------------------------------------
def plot_memory_microbench():
    json_path = "results/microbench_memory.json"
    if not os.path.exists(json_path):
        print(f"Skipping plot_memory_microbench: {json_path} not found")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    benches = data.get("benchmarks", [])
    if not benches:
        return

    seq_read = {}
    seq_write = {}
    memcpy_bw = {}
    ptr_chase = {}

    for b in benches:
        name = b["name"]
        bytes_per_sec = b.get("bytes_per_second", 0)
        items_per_sec = b.get("items_per_second", 0)

        label = b.get("label", "")
        if "KiB" not in label:
            continue
        size_kib = int(label.split()[0])

        if name.startswith("BM_SeqRead"):
            seq_read[size_kib] = bytes_per_sec / 1e9
        elif name.startswith("BM_SeqWrite"):
            seq_write[size_kib] = bytes_per_sec / 1e9
        elif name.startswith("BM_Memcpy"):
            memcpy_bw[size_kib] = bytes_per_sec / 1e9
        elif name.startswith("BM_PtrChase"):
            if items_per_sec > 0:
                ptr_chase[size_kib] = 1e9 / items_per_sec

    sizes = sorted(seq_read.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sizes, [seq_read[s] for s in sizes], marker='o', label='Seq Read BW (GB/s)', color='darkblue')
    ax1.plot(sizes, [seq_write[s] for s in sizes], marker='s', label='Seq Write BW (GB/s)', color='darkgreen')
    ax1.plot(sizes, [memcpy_bw[s] for s in sizes], marker='^', label='Memcpy BW (GB/s)', color='purple')
    ax1.set_xscale('log')
    ax1.set_xlabel('Working Set Size (KiB)')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Memory Hierarchy Bandwidth (L1 -> L2 -> L3 -> DRAM)')
    ax1.legend()

    if ptr_chase:
        chase_sizes = sorted(ptr_chase.keys())
        ax2.plot(chase_sizes, [ptr_chase[s] for s in chase_sizes], marker='d', color='crimson', label='Pointer Chase Latency')
        ax2.set_xscale('log')
        ax2.set_xlabel('Working Set Size (KiB)')
        ax2.set_ylabel('Latency per Access (ns)')
        ax2.set_title('Pointer Chase Access Latency')
        ax2.legend()

    fig.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, "plot5_memory_microbench.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved {plot_file}")

def main():
    plot_quant_thread_scaling()
    plot_branch_and_cache_misses()
    plot_memory_modes()
    plot_avx_impact()
    plot_memory_microbench()

if __name__ == "__main__":
    main()
