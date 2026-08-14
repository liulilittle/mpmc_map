/**
 * @file bench_common.hpp
 * @brief Shared benchmark harness for the mpmc_map container benchmarks.
 *
 * A workload is a deterministic per-thread op stream over a private key
 * range (churn: inserts and erases never collide across threads) plus
 * reads of a small shared range. The harness runs the workload on a
 * container and reports total operations per second.
 *
 * Modes:
 *   mode 0  mixed:      40% insert, 40% erase, 20% find (private),
 *                        plus a shared find every 8th op
 *   mode 1  read-heavy: 10% insert, 10% erase, 80% find (private),
 *                        plus a shared find every 4th op
 *
 * Containers only need insert(key,value) / erase(key) / find(key) — the
 * same code drives mpmc containers and mutex-guarded std containers.
 */

#ifndef MPMC_BENCH_COMMON_HPP
#define MPMC_BENCH_COMMON_HPP

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

namespace bench
{
    struct rng
    {
        std::uint64_t s;

        explicit rng(std::uint64_t seed)
            : s(seed)
        {
        }

        std::uint64_t next()
        {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            return s;
        }
    };

    const unsigned PRIVATE_RANGE = 2048; // per-thread churn range
    const unsigned SHARED_RANGE = 256;   // shared read-only range

    template <typename Container>
    long long run_workload(Container& c, unsigned threads,
                           unsigned long long ops_per_thread, unsigned mode)
    {
        std::vector<std::thread> ts;
        auto start = std::chrono::steady_clock::now();
        for (unsigned t = 0; t < threads; ++t)
        {
            ts.emplace_back([&c, t, ops_per_thread, mode]() {
                rng r(0x9e3779b97f4a7c15ULL ^ t);
                const int base = static_cast<int>(t * PRIVATE_RANGE);
                for (unsigned long long i = 0; i < ops_per_thread; ++i)
                {
                    const int k = base + static_cast<int>(r.next() % PRIVATE_RANGE);
                    const unsigned op =
                        static_cast<unsigned>(r.next() % 10);
                    if (mode == 0)
                    {
                        if (op < 4)
                        {
                            c.insert(std::make_pair(k, k));
                        }
                        else if (op < 8)
                        {
                            c.erase(k);
                        }
                        else
                        {
                            c.find(k);
                        }
                        if ((i & 7) == 0)
                        {
                            c.find(static_cast<int>(r.next() % SHARED_RANGE));
                        }
                    }
                    else
                    {
                        if (op < 1)
                        {
                            c.insert(std::make_pair(k, k));
                        }
                        else if (op < 2)
                        {
                            c.erase(k);
                        }
                        else
                        {
                            c.find(k);
                        }
                        if ((i & 3) == 0)
                        {
                            c.find(static_cast<int>(r.next() % SHARED_RANGE));
                        }
                    }
                }
            });
        }
        for (auto& t : ts)
        {
            t.join();
        }
        auto end = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(end - start).count();
        const double total = static_cast<double>(threads) * ops_per_thread;
        return secs > 0 ? static_cast<long long>(total / secs) : 0;
    }

    template <typename A, typename B>
    void compare_table(const char* title, const char* a_name,
                       const char* b_name, unsigned max_threads,
                       unsigned long long ops_per_thread, unsigned mode)
    {
        std::printf("%s (mode %u, ops/thread %llu)\n", title, mode,
                    (unsigned long long)ops_per_thread);
        std::printf("threads | %-16s | %-16s | speedup\n", a_name, b_name);
        std::printf("--------+------------------+------------------+--------\n");
        for (unsigned threads = 1; threads <= max_threads; threads *= 2)
        {
            A a;
            B b;
            const long long rate_a = run_workload(a, threads, ops_per_thread, mode);
            const long long rate_b = run_workload(b, threads, ops_per_thread, mode);
            const double speedup = rate_b > 0
                                       ? static_cast<double>(rate_a) / rate_b
                                       : 0.0;
            std::printf("%7u | %16lld | %16lld | %6.2fx\n", threads, rate_a,
                        rate_b, speedup);
            std::fflush(stdout);
        }
        std::printf("\n");
    }
}

#endif // MPMC_BENCH_COMMON_HPP
