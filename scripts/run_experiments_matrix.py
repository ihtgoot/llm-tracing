#!/usr/bin/env python3
"""
run_experiments_matrix.py
Executes the full experiment matrix for Gemma-3 270M with 10 measurement runs per combination:
- Quantizations: Q8_0, Q4_0, Q4_K_M
- Thread Counts: 1, 2, 4, 8, 12, 16, 20, 24, 28
- CPU Affinities: none, 0-3 (2P-HT), 0,2,4,6 (4P-NoHT), 0-7 (4P-HT), 0-15 (8P-HT), 16-19 (4E), 16-23 (8E), 16-27 (12E), 0-3,16-19 (Hybrid)
- Load Modes: mmap, mlock, none
- Vector ISAs: AVX-Native Release vs No-AVX
"""

import os
import sys
import subprocess

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

def write_yaml(binary_path, output_dir, models, threads, affinities, load_mode="mmap", warmup=2, measurements=10, yaml_path=""):
    models_str = "\n".join([
        f"    - name: \"{m['name']}\"\n      path: \"{m['path']}\"\n      quantization: \"{m['quantization']}\""
        for m in models
    ])
    threads_str = "\n".join([f"    - {t}" for t in threads])
    aff_str = "\n".join([f"    - \"{a}\"" for a in affinities])

    content = f"""llama_bench_binary: "{binary_path}"
output_dir: "{output_dir}"

experiments:
  models:
{models_str}

  threads:
{threads_str}

  affinity:
{aff_str}

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

def run_harness(config_path, output_dir):
    cmd = [HARNESS_BIN, "--config", config_path, "--output-dir", output_dir]
    print(f"\n=======================================================")
    print(f"Running experiment matrix: {output_dir}")
    print(f"=======================================================")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"ERROR: Experiment {output_dir} failed with exit code {res.returncode}")

def main():
    if not os.path.exists(HARNESS_BIN):
        print("Build harness first!")
        sys.exit(1)

    # 1. Thread Scaling & Quantization Matrix (10 repetitions)
    write_yaml(
        binary_path=RELEASE_BENCH,
        output_dir="results/matrix_threads_quant",
        models=MODELS,
        threads=[1, 2, 4, 8, 12, 16, 20, 24, 28],
        affinities=["none"],
        measurements=10,
        yaml_path="config/exp_threads_quant.yaml"
    )
    run_harness("config/exp_threads_quant.yaml", "results/matrix_threads_quant")

    # 2. CPU Affinity & Core Heterogeneity (P-cores vs E-cores vs HT, 10 repetitions)
    affinities_suite = [
        "none",        # Default OS scheduling
        "0-3",         # 2 P-cores (4 HT threads)
        "0,2,4,6",     # 4 physical P-cores (No HT)
        "0-7",         # 4 P-cores (8 HT threads)
        "0-15",        # 8 P-cores (16 HT threads - Full P-cluster)
        "16-19",       # 4 E-cores
        "16-23",       # 8 E-cores
        "16-27",       # 12 E-cores (Full E-cluster)
        "0-3,16-19"    # Hybrid (2 P-cores + 4 E-cores)
    ]
    write_yaml(
        binary_path=RELEASE_BENCH,
        output_dir="results/matrix_affinity",
        models=MODELS[:2], # Q8_0 and Q4_0
        threads=[4, 8],
        affinities=affinities_suite,
        measurements=10,
        yaml_path="config/exp_affinity.yaml"
    )
    run_harness("config/exp_affinity.yaml", "results/matrix_affinity")

    # 3. Memory Load Mode Sweep (mmap vs mlock vs none, 10 repetitions)
    for mode in ["mmap", "mlock", "none"]:
        write_yaml(
            binary_path=RELEASE_BENCH,
            output_dir=f"results/matrix_memory_{mode}",
            models=[MODELS[0]], # Q8_0
            threads=[4, 8],
            affinities=["0-7", "16-23"],
            load_mode=mode,
            measurements=10,
            yaml_path=f"config/exp_memory_{mode}.yaml"
        )
        run_harness(f"config/exp_memory_{mode}.yaml", f"results/matrix_memory_{mode}")

    # 4. Vector Acceleration (AVX Native vs No-AVX, 10 repetitions)
    write_yaml(
        binary_path=RELEASE_BENCH,
        output_dir="results/matrix_avx_on",
        models=[MODELS[0]],
        threads=[1, 2, 4, 8, 16],
        affinities=["0-7"],
        measurements=10,
        yaml_path="config/exp_avx_on.yaml"
    )
    run_harness("config/exp_avx_on.yaml", "results/matrix_avx_on")

    write_yaml(
        binary_path=NOAVX_BENCH,
        output_dir="results/matrix_avx_off",
        models=[MODELS[0]],
        threads=[1, 2, 4, 8, 16],
        affinities=["0-7"],
        measurements=10,
        yaml_path="config/exp_avx_off.yaml"
    )
    run_harness("config/exp_avx_off.yaml", "results/matrix_avx_off")

    print("\nAll 10-repetition experiment matrices completed successfully!")

if __name__ == "__main__":
    main()
