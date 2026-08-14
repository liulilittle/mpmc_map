/**
 * @file bench_unordered.cpp
 * @brief Throughput benchmark: mpmc::unordered_map vs std::unordered_map
 *        behind a mutex.
 *
 * Two workloads (mixed churn and read-heavy) at 1, 2, 4 and 8 threads.
 * The mpmc container uses the default reclamation scheme.
 *
 * Usage: bench_unordered [ops_per_thread]
 */

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "bench_common.hpp"
#include "mpmc/mpmc_unordered_map.hpp"

namespace
{
    using mpmc_map_t = mpmc::unordered_map<int, int>;

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
        bench::compare_table<mpmc_map_t, std_map_guard>(
            "bench_unordered: mpmc vs std+mutex",
            "mpmc ops/s", "std+mutex ops/s", max_threads, ops, mode);
    }
    std::printf("bench_unordered done\n");
    return 0;
}
