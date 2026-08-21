// memory_bench.cpp
// Google Benchmark microbenchmark for memory hierarchy characterization.
//
// Purpose: characterize the underlying memory subsystem INDEPENDENTLY from
// LLM inference. This is NOT for measuring llama.cpp — it measures how the
// hardware behaves for sequential/random access at different working set sizes.
//
// Working set sizes cover: L1 → L2 → L3 → DRAM transition points.
// i7-14700HX caches:
//   L1d:  768 KiB  (per-core)   → ~38 KiB per core (approximate)
//   L2:   28 MiB total (11 instances, ~2.5 MiB per cluster)
//   L3:   33 MiB  (shared)
//   DRAM: 16 GiB
//
// Results are SEPARATE from llama.cpp inference results.
// Do NOT mix these with inference benchmarks.

#include <benchmark/benchmark.h>

#include <vector>
#include <random>
#include <cstring>
#include <cstdint>
#include <numeric>
#include <algorithm>

namespace {

// ── Working set sizes ──────────────────────────────────────────────────────
// Chosen to stress each level of the memory hierarchy.
// 16 KiB:   well within L1d
// 64 KiB:   L1d boundary
// 256 KiB:  L2 (per-core P-core L2 is ~2 MiB)
// 1 MiB:    L2 zone
// 4 MiB:    L2/L3 boundary
// 16 MiB:   L3
// 33 MiB:   L3 boundary / DRAM onset
// 128 MiB:  clearly DRAM
// 512 MiB:  large DRAM

static const std::vector<int64_t> kWorkingSizes = {
    16LL    * 1024,
    64LL    * 1024,
    256LL   * 1024,
    1LL     * 1024 * 1024,
    4LL     * 1024 * 1024,
    16LL    * 1024 * 1024,
    33LL    * 1024 * 1024,
    128LL   * 1024 * 1024,
    512LL   * 1024 * 1024,
};

// ── Sequential read ────────────────────────────────────────────────────────
// Measures throughput of linear sequential reads.
// Representative of: reading model weight tensors sequentially.

static void BM_SeqRead(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    // Allocate with alignment for SIMD
    std::vector<uint8_t> buf(size, 0xAB);

    uint64_t checksum = 0; // prevent optimization

    for (auto _ : state) {
        for (size_t i = 0; i < size; i += 64) {
            // Touch one byte per cache line (64 bytes)
            checksum += buf[i];
        }
        benchmark::DoNotOptimize(checksum);
    }

    benchmark::DoNotOptimize(checksum);

    // Report throughput: bytes processed per second
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
    state.SetLabel(std::to_string(size / 1024) + " KiB");
}

// ── Sequential write ───────────────────────────────────────────────────────
// Measures write bandwidth.

static void BM_SeqWrite(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> buf(size, 0);

    for (auto _ : state) {
        // Sequential write touching each cache line
        for (size_t i = 0; i < size; i += 64) {
            buf[i] = static_cast<uint8_t>(i & 0xFF);
        }
        benchmark::DoNotOptimize(buf.data());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
    state.SetLabel(std::to_string(size / 1024) + " KiB");
}

// ── Random read ────────────────────────────────────────────────────────────
// Measures latency-dominated random access.
// Representative of: KV-cache scatter/gather, attention computation.

static void BM_RandomRead(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    const size_t n_cachelines = size / 64;
    if (n_cachelines == 0) {
        state.SkipWithError("Size too small");
        return;
    }

    std::vector<uint8_t> buf(size, 0xCD);

    // Pre-compute shuffled access order
    std::vector<size_t> indices(n_cachelines);
    std::iota(indices.begin(), indices.end(), 0);
    {
        std::mt19937_64 rng(42); // fixed seed for reproducibility
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    uint64_t checksum = 0;
    const size_t n_accesses = std::min(n_cachelines,
                                        static_cast<size_t>(1 << 20)); // cap iterations

    for (auto _ : state) {
        for (size_t i = 0; i < n_accesses; ++i) {
            checksum += buf[indices[i % n_cachelines] * 64];
        }
        benchmark::DoNotOptimize(checksum);
    }

    benchmark::DoNotOptimize(checksum);

    // Report as accesses per second (each access = 1 cache line = 64 bytes)
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n_accesses) * 64);
    state.SetLabel(std::to_string(size / 1024) + " KiB");
}

// ── Pointer chase (latency chain) ──────────────────────────────────────────
// Measures true memory latency with no prefetch hiding.
// Each load depends on the previous result → no instruction-level parallelism.
// This directly measures memory hierarchy latency.

struct Node {
    Node* next;
    char  pad[56]; // pad to 64 bytes (one cache line per node)
};

static void BM_PtrChase(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    const size_t n_nodes = size / sizeof(Node);
    if (n_nodes < 4) {
        state.SkipWithError("Size too small for pointer chase");
        return;
    }

    // Allocate nodes
    std::vector<Node> nodes(n_nodes);

    // Build a random cycle through all nodes
    std::vector<size_t> order(n_nodes);
    std::iota(order.begin(), order.end(), 0);
    {
        std::mt19937_64 rng(1234);
        std::shuffle(order.begin(), order.end(), rng);
    }
    for (size_t i = 0; i < n_nodes; ++i) {
        nodes[order[i]].next = &nodes[order[(i+1) % n_nodes]];
    }

    const size_t n_chases = std::min(n_nodes, static_cast<size_t>(1 << 20));

    Node* p = &nodes[0];
    for (auto _ : state) {
        for (size_t i = 0; i < n_chases; ++i) {
            p = p->next; // dependent load chain
        }
        benchmark::DoNotOptimize(p);
    }

    // Report latency per access
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n_chases));
    state.SetLabel(std::to_string(size / 1024) + " KiB");
}

// ── memcpy bandwidth ───────────────────────────────────────────────────────
// Measures combined read+write bandwidth (like weight copying to KV cache).

static void BM_Memcpy(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    std::vector<uint8_t> src(size, 0xAB);
    std::vector<uint8_t> dst(size, 0);

    for (auto _ : state) {
        memcpy(dst.data(), src.data(), size);
        benchmark::DoNotOptimize(dst.data());
    }

    // memcpy reads and writes: 2x bytes
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size) * 2);
    state.SetLabel(std::to_string(size / 1024) + " KiB");
}

} // anonymous namespace

// ── Register benchmarks ───────────────────────────────────────────────────
// Each benchmark runs for all working set sizes.
// Output includes bytes/sec which Google Benchmark converts to bandwidth.

BENCHMARK(BM_SeqRead)
    ->ArgsProduct({kWorkingSizes})
    ->MinWarmUpTime(0.5)
    ->MinTime(1.0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SeqWrite)
    ->ArgsProduct({kWorkingSizes})
    ->MinWarmUpTime(0.5)
    ->MinTime(1.0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RandomRead)
    ->ArgsProduct({kWorkingSizes})
    ->MinWarmUpTime(0.5)
    ->MinTime(1.0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_PtrChase)
    ->ArgsProduct({kWorkingSizes})
    ->MinWarmUpTime(0.5)
    ->MinTime(1.0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Memcpy)
    ->ArgsProduct({kWorkingSizes})
    ->MinWarmUpTime(0.5)
    ->MinTime(1.0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
