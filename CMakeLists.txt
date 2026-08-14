cmake_minimum_required(VERSION 3.16)
project(mpmc_map LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MPMC_SANITIZE "Build tests with undefined-behavior sanitizer" OFF)

add_compile_options(-Wall -Wextra)

find_package(Threads REQUIRED)

# ---------------- Library (header-only) ----------------
add_library(mpmc INTERFACE)
target_include_directories(mpmc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mpmc INTERFACE Threads::Threads)

if(MPMC_SANITIZE)
  # clang 17 on MinGW: UBSan works; ASan does not (interception failure).
  # MSVC: ASan works with ASAN_WIN_CONTINUE_ON_INTERCEPTION_FAILURE=1
  # (see docs/PERF_en.md for the toolchain matrix).
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=undefined)
endif()

# ---------------- Negative compile check (configure-time) ----------------
# The containers must REJECT keys with a throwing copy constructor.
try_compile(MPMC_COMPILE_FAIL_UNEXPECTED
  ${CMAKE_CURRENT_BINARY_DIR}/compile_fail_check
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/compile_fail.cpp
  CMAKE_FLAGS "-DCMAKE_CXX_STANDARD=11"
  OUTPUT_VARIABLE MPMC_COMPILE_FAIL_OUT)
if(MPMC_COMPILE_FAIL_UNEXPECTED)
  message(FATAL_ERROR "negative compile test FAILED: throwing-copy keys must not compile")
endif()
message(STATUS "negative compile check passed: throwing-copy keys rejected as expected")

# ---------------- Tests ----------------
enable_testing()

function(add_mpmc_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE mpmc)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

add_mpmc_test(test_reclamation)
add_mpmc_test(stress_reclamation 1000000)
add_mpmc_test(test_unordered)
add_mpmc_test(stress_unordered 1000000)
add_mpmc_test(test_tree)
add_mpmc_test(stress_tree 1000000)
add_mpmc_test(dbg_degenerate)
add_mpmc_test(repro_conc 50)
add_mpmc_test(test_pool)
add_mpmc_test(api_traits_check)

# ---------------- Benchmarks (not ctest targets) ----------------
add_executable(bench_reclamation bench/bench_reclamation.cpp)
target_link_libraries(bench_reclamation PRIVATE mpmc)
add_executable(bench_unordered bench/bench_unordered.cpp)
target_link_libraries(bench_unordered PRIVATE mpmc)
add_executable(bench_tree bench/bench_tree.cpp)
target_link_libraries(bench_tree PRIVATE mpmc)
add_executable(microbench bench/microbench.cpp)
target_link_libraries(microbench PRIVATE mpmc)
