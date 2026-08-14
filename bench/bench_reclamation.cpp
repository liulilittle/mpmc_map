/**
 * @file bench_reclamation.cpp
 * @brief A/B benchmark: hazard-pointer vs epoch reclamation.
 *
 * The same container type (mpmc::unordered_map<int, int>) and the same
 * deterministic workload are run with each reclamation scheme at 1, 2, 4
 * and 8 threads. The reclamation cost shows up in every traversal
 * (hazard: per-step protect + re-validate; epoch: per-operation
 * enter/exit against a shared global counter) and on retirement (hazard:
 * per-object retire-list scan; epoch: batched reclaim at generation
 * boundaries).
 *
 * The numbers decide reclamation::default_scheme (see docs/PERF_en.md).
 *
 * Usage: bench_reclamation [ops_per_thread]
 */

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "bench_common.hpp"
#include "mpmc/mpmc_unordered_map.hpp"

namespace
{
    // same container, two schemes
    using hazard_map = mpmc::unordered_map<
        int, int, std::hash<int>, std::equal_to<int>,
        std::allocator<std::pair<const int, int>>,
        mpmc::reclamation::hazard_pointer>;
    using epoch_map = mpmc::unordered_map<
        int, int, std::hash<int>, std::equal_to<int>,
        std::allocator<std::pair<const int, int>>, mpmc::reclamation::epoch>;

    // mutex-guarded std baseline for scale context
    struct std_map_guard
    {
        std::mutex mutex;
        std::unordered_map<int, int> map;

        void insert(const std::pair<int, int>& kv)
        {
            std::lock_guard<std::mutex> lock(mutex);
            map.insert(kv);
        }

        void erase(int k)
        {
            std::lock_guard<std::mutex> lock(mutex);
            map.erase(k);
        }

        void find(int k)
        {
            std::lock_guard<std::mutex> lock(mutex);
            map.find(k);
        }
    };
}

int main(int argc, char** argv)
{
    const unsigned long long ops =
        argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2000000ULL;
    const unsigned max_threads = 8;
    for (unsigned mode = 0; mode < 2; ++mode)
    {
        bench::compare_table<hazard_map, epoch_map>(
            "bench_reclamation: hazard vs epoch",
            "hazard ops/s", "epoch ops/s", max_threads, ops, mode);
        bench::compare_table<hazard_map, std_map_guard>(
            "bench_reclamation: hazard vs std+mutex",
            "hazard ops/s", "std+mutex ops/s", max_threads, ops, mode);
    }
    std::printf("bench_reclamation done\n");
    return 0;
}
