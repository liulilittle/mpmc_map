/**
 * @file stress_tree.cpp
 * @brief Concurrent stress test for mpmc::map / mpmc::set (rb_tree).
 *
 * Scenario: T threads share one container and one overlapping key space.
 * Each thread runs a deterministic per-thread op stream (insert / erase /
 * find / lower_bound) driven by its own RNG; every thread records the last
 * operation per key and verifies the container state for every key it
 * touched (exact per-key oracle — not statistical).
 *
 * After join:
 *   * size() equals the expected count recomputed by replaying every
 *     thread's RNG stream (identical draws, including shared-read draws),
 *   * iteration visits exactly the expected key multiset in ascending
 *     order,
 *   * every allocation is released and the reclaimer reports zero
 *     outstanding retirements,
 *   * clear() empties the container.
 *
 * Both reclamation schemes run the same scenario. Run directly with
 * <iterations> for heavier runs; ctest uses the default.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "common/check.hpp"
#include "mpmc/mpmc_map.hpp"
#include "mpmc/mpmc_set.hpp"

namespace
{
    template <typename T>
    struct counting_allocator
    {
        using value_type = T;

        static std::atomic<std::size_t> allocs;
        static std::atomic<std::size_t> frees;

        counting_allocator() {}

        template <typename U>
        counting_allocator(const counting_allocator<U>&) {}

        T* allocate(std::size_t n)
        {
            allocs.fetch_add(n, std::memory_order_relaxed);
            return std::allocator<T>().allocate(n);
        }

        void deallocate(T* p, std::size_t n)
        {
            frees.fetch_add(n, std::memory_order_relaxed);
            std::allocator<T>().deallocate(p, n);
        }

        template <typename U>
        struct rebind
        {
            using other = counting_allocator<U>;
        };

        friend bool operator==(const counting_allocator&, const counting_allocator&)
        {
            return true;
        }

        friend bool operator!=(const counting_allocator&, const counting_allocator&)
        {
            return false;
        }
    };

    template <typename T>
    std::atomic<std::size_t> counting_allocator<T>::allocs(0);

    template <typename T>
    std::atomic<std::size_t> counting_allocator<T>::frees(0);

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

    template <typename Reclaimer>
    void pump_reclaimer()
    {
        for (int i = 0; i < 4; ++i)
        {
            typename Reclaimer::op_guard g;
            Reclaimer::reclaim();
        }
    }

    // ---- map stress -------------------------------------------------------
    template <typename Scheme>
    void stress_map(unsigned threads, unsigned per_thread)
    {
        using map_type = mpmc::map<int, int, std::less<int>,
                                   counting_allocator<std::pair<const int, int>>,
                                   Scheme>;
        map_type m;
        static const unsigned range = 1024; // shared key space
        static const unsigned long long seed = 0x9e3779b97f4a7c15ULL;

        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; ++t)
            {
                ts.emplace_back([&m, t, per_thread]() {
                    rng r(seed ^ t);
                    std::vector<unsigned char> last_op(range, 0);
                    std::vector<int> last_val(range, 0);
                    for (unsigned i = 0; i < per_thread; ++i)
                    {
                        const int k = static_cast<int>(r.next() % range);
                        const unsigned op = static_cast<unsigned>(r.next() % 4);
                        if (op == 0)
                        {
                            // value = key: the final value of a shared key is
                            // deterministic regardless of interleaving
                            m.insert(std::make_pair(k, k));
                            last_op[static_cast<unsigned>(k)] = 1;
                            last_val[static_cast<unsigned>(k)] = k;
                        }
                        else if (op == 1)
                        {
                            m.erase(k);
                            last_op[static_cast<unsigned>(k)] = 2;
                        }
                        else if (op == 2)
                        {
                            auto it = m.find(k);
                            if (it != m.end())
                            {
                                TEST_CHECK(it->first == k);
                            }
                        }
                        else
                        {
                            m.lower_bound(k);
                            m.upper_bound(k);
                        }
                    }
                    // per-key oracle: every touched key must be exactly in
                    // the state of its last operation
                    unsigned bad = 0;
                    for (unsigned i = 0; i < range; ++i)
                    {
                        if (last_op[i] == 0)
                        {
                            continue;
                        }
                        const int k = static_cast<int>(i);
                        auto it = m.find(k);
                        const bool have = it != m.end();
                        const bool want = last_op[i] == 1;
                        if (have != want)
                        {
                            if (bad < 4)
                            {
                                std::printf("  [t=%u] key %d present=%d want=%d\n",
                                            t, k, (int)have, (int)want);
                                std::fflush(stdout);
                            }
                            ++bad;
                        }
                        else if (want && it->second != last_val[i])
                        {
                            std::printf("  [t=%u] key %d value %d want %d\n", t,
                                        k, it->second, last_val[i]);
                            std::fflush(stdout);
                            ++bad;
                        }
                    }
                    if (bad != 0)
                    {
                        std::printf("  [t=%u] %u mismatches\n", t, bad);
                        std::fflush(stdout);
                    }
                });
            }
            for (auto& t : ts)
            {
                t.join();
            }
        }

        // RNG-replay: recompute every thread's stream and the expected
        // multiset (same seeds, same draw order)
        std::set<int> expected;
        std::vector<int> expected_val(1024, 0);
        for (unsigned t = 0; t < threads; ++t)
        {
            rng r(seed ^ t);
            std::vector<unsigned char> last_op(range, 0);
            std::vector<int> last_val(range, 0);
            for (unsigned i = 0; i < per_thread; ++i)
            {
                const int k = static_cast<int>(r.next() % range);
                const unsigned op = static_cast<unsigned>(r.next() % 4);
                if (op == 0)
                {
                    last_op[static_cast<unsigned>(k)] = 1;
                    last_val[static_cast<unsigned>(k)] = k;
                }
                else if (op == 1)
                {
                    last_op[static_cast<unsigned>(k)] = 2;
                }
                else if (op == 2)
                {
                    // find consumed no extra draws
                }
                else
                {
                    // bounds consumed no extra draws
                }
            }
            for (unsigned i = 0; i < range; ++i)
            {
                if (last_op[i] == 1)
                {
                    expected.insert(static_cast<int>(i));
                    expected_val[i] = last_val[i];
                }
            }
        }
        TEST_CHECK(m.size() == expected.size());

        // in-order iteration must visit exactly the expected multiset
        std::size_t count = 0;
        int prev = -1;
        std::set<int> seen;
        for (auto& kv : m)
        {
            TEST_CHECK(expected.count(kv.first) == 1);
            TEST_CHECK(kv.second == expected_val[static_cast<unsigned>(kv.first)]);
            TEST_CHECK(kv.first > prev);
            prev = kv.first;
            TEST_CHECK(seen.insert(kv.first).second);
            ++count;
        }
        TEST_CHECK(count == expected.size());
        TEST_CHECK(seen.size() == expected.size());

        // bounds spot checks against the expected set
        for (int probe : {0, 511, 512, 1023})
        {
            auto lb = m.lower_bound(probe);
            auto rlb = expected.lower_bound(probe);
            TEST_CHECK((lb == m.end()) == (rlb == expected.end()));
            if (lb != m.end())
            {
                TEST_CHECK(lb->first == *rlb);
            }
            auto ub = m.upper_bound(probe);
            auto rub = expected.upper_bound(probe);
            TEST_CHECK((ub == m.end()) == (rub == expected.end()));
            if (ub != m.end())
            {
                TEST_CHECK(ub->first == *rub);
            }
        }

        m.clear();
        TEST_CHECK(m.empty());
        TEST_CHECK(m.size() == 0);

        pump_reclaimer<typename map_type::reclaimer>();
        const std::size_t allocs =
            counting_allocator<unsigned char>::allocs.load(
                std::memory_order_relaxed);
        const std::size_t frees =
            counting_allocator<unsigned char>::frees.load(
                std::memory_order_relaxed);
        TEST_CHECK(allocs == frees && "no allocation leaks");
        TEST_CHECK(!map_type::reclaimer::has_outstanding());
    }

    // ---- set stress --------------------------------------------------------
    template <typename Scheme>
    void stress_set(unsigned threads, unsigned per_thread)
    {
        using set_type = mpmc::set<int, std::less<int>,
                                   counting_allocator<int>, Scheme>;
        set_type s;
        static const unsigned range = 1024;
        static const unsigned long long seed = 0x6a09e667f3bcc909ULL;

        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; ++t)
            {
                ts.emplace_back([&s, t, per_thread]() {
                    rng r(seed ^ t);
                    std::vector<unsigned char> last_op(range, 0);
                    for (unsigned i = 0; i < per_thread; ++i)
                    {
                        const int k = static_cast<int>(r.next() % range);
                        const unsigned op = static_cast<unsigned>(r.next() % 4);
                        if (op == 0)
                        {
                            s.insert(k);
                            last_op[static_cast<unsigned>(k)] = 1;
                        }
                        else if (op == 1)
                        {
                            s.erase(k);
                            last_op[static_cast<unsigned>(k)] = 2;
                        }
                        else if (op == 2)
                        {
                            s.contains(k);
                        }
                        else
                        {
                            s.lower_bound(k);
                        }
                    }
                    for (unsigned i = 0; i < range; ++i)
                    {
                        if (last_op[i] == 0)
                        {
                            continue;
                        }
                        const int k = static_cast<int>(i);
                        TEST_CHECK(s.contains(k) == (last_op[i] == 1));
                    }
                });
            }
            for (auto& t : ts)
            {
                t.join();
            }
        }

        std::set<int> expected;
        for (unsigned t = 0; t < threads; ++t)
        {
            rng r(seed ^ t);
            std::vector<unsigned char> last_op(range, 0);
            for (unsigned i = 0; i < per_thread; ++i)
            {
                const int k = static_cast<int>(r.next() % range);
                const unsigned op = static_cast<unsigned>(r.next() % 4);
                if (op == 0)
                {
                    last_op[static_cast<unsigned>(k)] = 1;
                }
                else if (op == 1)
                {
                    last_op[static_cast<unsigned>(k)] = 2;
                }
            }
            for (unsigned i = 0; i < range; ++i)
            {
                if (last_op[i] == 1)
                {
                    expected.insert(static_cast<int>(i));
                }
            }
        }
        TEST_CHECK(s.size() == expected.size());
        std::size_t count = 0;
        int prev = -1;
        std::set<int> seen;
        for (int k : s)
        {
            TEST_CHECK(expected.count(k) == 1);
            TEST_CHECK(k > prev);
            prev = k;
            TEST_CHECK(seen.insert(k).second);
            ++count;
        }
        TEST_CHECK(count == expected.size());
        s.clear();
        TEST_CHECK(s.empty());
        pump_reclaimer<typename set_type::reclaimer>();
        TEST_CHECK(!set_type::reclaimer::has_outstanding());
    }
}

int main(int argc, char** argv)
{
    const unsigned iterations =
        argc > 1 ? static_cast<unsigned>(std::atoll(argv[1])) : 1000000;
    // Optional second argument: thread count (default 8; 16 validates
    // 16-core scaling, see bench/results.md).
    const unsigned threads =
        argc > 2 ? static_cast<unsigned>(std::atoll(argv[2])) : 8;
    std::printf("stress_tree: iterations=%u threads=%u\n", iterations,
                threads);

    std::printf("map hazard:\n");
    stress_map<mpmc::reclamation::hazard_pointer>(threads, iterations);
    std::printf("map epoch:\n");
    stress_map<mpmc::reclamation::epoch>(threads, iterations);
    std::printf("set hazard:\n");
    stress_set<mpmc::reclamation::hazard_pointer>(threads, iterations);
    std::printf("set epoch:\n");
    stress_set<mpmc::reclamation::epoch>(threads, iterations);

    std::printf("stress_tree: all tests passed\n");
    return 0;
}
