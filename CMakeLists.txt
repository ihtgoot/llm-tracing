cmake_minimum_required(VERSION 3.16)
project(llm_characterization CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to RelWithDebInfo so we get useful debug info but also -O2
# (the harness itself should be fast; the workload is llama-bench)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ── Warnings ──────────────────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic)

# ── Main harness executable ───────────────────────────────────────────────
add_executable(llm-charharness
    src/main.cpp
    src/config.cpp
    src/system_info.cpp
    src/launcher.cpp
    src/process_monitor.cpp
    src/perf_monitor.cpp
    src/result_writer.cpp
)

# Threads (for ProcessMonitor background thread)
find_package(Threads REQUIRED)
target_link_libraries(llm-charharness PRIVATE Threads::Threads)

target_include_directories(llm-charharness PRIVATE src/)

# ── Memory microbenchmark (Google Benchmark) ──────────────────────────────
# Google Benchmark is installed as a static lib at /usr/local/lib/
find_library(BENCHMARK_LIB benchmark HINTS /usr/local/lib)

if(BENCHMARK_LIB)
    message(STATUS "Google Benchmark found: ${BENCHMARK_LIB}")
    add_executable(memory-microbench
        microbench/memory_bench.cpp
    )
    target_include_directories(memory-microbench PRIVATE /usr/local/include)
    target_link_libraries(memory-microbench PRIVATE
        ${BENCHMARK_LIB}
        Threads::Threads
    )
    # Benchmark needs -O2 or better to be meaningful
    target_compile_options(memory-microbench PRIVATE -O2)
else()
    message(WARNING "Google Benchmark not found — memory-microbench target skipped")
endif()

# ── Installation ──────────────────────────────────────────────────────────
install(TARGETS llm-charharness DESTINATION bin)

# ── Summary ───────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== llm-characterization build targets ===")
message(STATUS "  llm-charharness    — main experiment harness")
if(BENCHMARK_LIB)
message(STATUS "  memory-microbench  — memory hierarchy microbench")
endif()
message(STATUS "")
