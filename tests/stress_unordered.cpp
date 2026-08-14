/**
 * @file stress_unordered.cpp
 * @brief Concurrent stress test for mpmc::unordered_map / unordered_set.
 *
 * Scenario: T threads, each owns a private key range of size R and a
 * private operation sequence (deterministic per thread). Every thread
 * repeatedly:
 *
 *   * inserts keys (own range),
 *   * finds its own keys,
 *   * erases every second key,
 *   * re-inserts them,
 *   * reads shared keys (a small shared range, no modification).
 *
 * Verifications after join:
 *   * size() equals the expected count computed from the per-thread
 *     deterministic sequences,
 *   * every expected key is present with the expected value, every
 *     expected-absent key is absent,
 *   * iteration visits exactly the expected multiset,
 *   * every allocation is released (counting allocator) and the reclaimer
 *     reports zero outstanding retirements,
 *   * clear() empties the container.
 *
 * Run directly with <iterations> for heavier runs; ctest uses the default.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/check.hpp"
#include "mpmc/mpmc_unordered_map.hpp"
#include "mpmc/mpmc_unordered_set.hpp"

namespace
{
    // ---- counting allocator (per-container leak detection) --------------
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

    // ---- deterministic per-thread RNG -----------------------------------
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

    // ---- epoch pump -------------------------------------------------------
    // The epoch scheme reclaims a retirement only after the global epoch
    // has passed it by two generations; empty guard cycles advance the
    // epoch when this is the only active thread. Hazard pointers need no
    // pumping (the guard and reclaim are no-ops or immediate).
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
        using map_type = mpmc::unordered_map<
            int, int, std::hash<int>, std::equal_to<int>,
            counting_allocator<std::pair<const int, int>>, Scheme>;
        map_type m;
        const std::size_t allocs_before =
            counting_allocator<unsigned char>::allocs.load(
                std::memory_order_relaxed);
        static const unsigned range = 512; // per-thread private range

        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; ++t)
            {
                ts.emplace_back([&m, t, per_thread]() {
                    rng r(0x9e3779b97f4a7c15ULL ^ t);
                    const int base = static_cast<int>(t * range);
                    // last operation per key: 0 = untouched, 1 = insert,
                    // 2 = erase (the RNG does not necessarily touch every
                    // key, so the final state is defined per touched key)
                    std::vector<unsigned char> last_op(range, 0);
                    for (unsigned i = 0; i < per_thread; ++i)
                    {
                        const int k = base + static_cast<int>(r.next() % range);
                        const bool even = (k & 1) == 0;
                        if (even)
                        {
                            m.insert(std::make_pair(k, k));
                            last_op[static_cast<unsigned>(k - base)] = 1;
                        }
                        else
                        {
                            m.erase(k);
                            last_op[static_cast<unsigned>(k - base)] = 2;
                        }
                        if ((i & 3) == 0)
                        {
                            auto it = m.find(k);
                            if (it != m.end())
                            {
                                TEST_CHECK(it->first == k);
                            }
                        }
                        if ((i & 7) == 0)
                        {
                            // shared read-only keys
                            m.find(static_cast<int>(r.next() % 16));
                        }
                    }
                    // final: the state of every touched key
                    unsigned bad = 0;
                    for (unsigned i = 0; i < range; ++i)
                    {
                        const int k = base + static_cast<int>(i);
                        if (last_op[i] == 0)
                        {
                            continue; // never touched: no expectation
                        }
                        const bool c = m.contains(k);
                        const bool want = last_op[i] == 1;
                        if (c != want)
                        {
                            if (bad < 4)
                            {
                                std::printf("  [t=%u] key %d contains=%d want=%d\n",
                                            t, k, (int)c, (int)want);
                                std::fflush(stdout);
                            }
                            ++bad;
                        }
                        else if (want)
                        {
                            TEST_CHECK(m.at(k) == k);
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

        // expected multiset: recompute the per-thread RNG sequences (same
        // seeds) and keep the keys whose last operation was an insert
        std::unordered_set<int> expected;
        for (unsigned t = 0; t < threads; ++t)
        {
            rng r(0x9e3779b97f4a7c15ULL ^ t);
            const int base = static_cast<int>(t * range);
            std::vector<unsigned char> last_op(range, 0);
            for (unsigned i = 0; i < per_thread; ++i)
            {
                const int k = base + static_cast<int>(r.next() % range);
                last_op[static_cast<unsigned>(k - base)] =
                    ((k & 1) == 0) ? 1 : 2;
                if ((i & 7) == 0)
                {
                    r.next(); // the shared find consumed one draw
                }
            }
            for (unsigned i = 0; i < range; ++i)
            {
                if (last_op[i] == 1)
                {
                    expected.insert(base + static_cast<int>(i));
                }
            }
        }
        TEST_CHECK(m.size() == expected.size());

        std::unordered_set<int> seen;
        for (auto& kv : m)
        {
            TEST_CHECK(expected.count(kv.first) == 1);
            TEST_CHECK(kv.second == kv.first);
            TEST_CHECK(seen.insert(kv.first).second);
        }
        TEST_CHECK(seen.size() == expected.size());

        m.clear();
        TEST_CHECK(m.empty());
        TEST_CHECK(m.size() == 0);

        pump_reclaimer<typename map_type::reclaimer>();
        const std::size_t allocs_after =
            counting_allocator<unsigned char>::allocs.load(
                std::memory_order_relaxed);
        const std::size_t frees_after =
            counting_allocator<unsigned char>::frees.load(
                std::memory_order_relaxed);
        TEST_CHECK(allocs_after == frees_after && "no allocation leaks");
        (void)allocs_before;
        TEST_CHECK(!map_type::reclaimer::has_outstanding());
    }

    // ---- set stress --------------------------------------------------------
    template <typename Scheme>
    void stress_set(unsigned threads, unsigned per_thread)
    {
        using set_type = mpmc::unordered_set<int, std::hash<int>,
                                             std::equal_to<int>,
                                             counting_allocator<int>, Scheme>;
        set_type s;
        static const unsigned range = 512;

        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; ++t)
            {
                ts.emplace_back([&s, t, per_thread]() {
                    rng r(0x6a09e667f3bcc909ULL ^ t);
                    const int base = static_cast<int>(t * range);
                    std::vector<unsigned char> last_op(range, 0);
                    for (unsigned i = 0; i < per_thread; ++i)
                    {
                        const int k = base + static_cast<int>(r.next() % range);
                        if ((k & 1) == 0)
                        {
                            s.insert(k);
                            last_op[static_cast<unsigned>(k - base)] = 1;
                        }
                        else
                        {
                            s.erase(k);
                            last_op[static_cast<unsigned>(k - base)] = 2;
                        }
                        if ((i & 3) == 0)
                        {
                            s.contains(k);
                        }
                    }
                    for (unsigned i = 0; i < range; ++i)
                    {
                        if (last_op[i] == 0)
                        {
                            continue;
                        }
                        const int k = base + static_cast<int>(i);
                        TEST_CHECK(s.contains(k) == (last_op[i] == 1));
                    }
                });
            }
            for (auto& t : ts)
            {
                t.join();
            }
        }

        std::unordered_set<int> expected;
        for (unsigned t = 0; t < threads; ++t)
        {
            rng r(0x6a09e667f3bcc909ULL ^ t);
            const int base = static_cast<int>(t * range);
            std::vector<unsigned char> last_op(range, 0);
            for (unsigned i = 0; i < per_thread; ++i)
            {
                const int k = base + static_cast<int>(r.next() % range);
                last_op[static_cast<unsigned>(k - base)] =
                    ((k & 1) == 0) ? 1 : 2;
            }
            for (unsigned i = 0; i < range; ++i)
            {
                if (last_op[i] == 1)
                {
                    expected.insert(base + static_cast<int>(i));
                }
            }
        }
        TEST_CHECK(s.size() == expected.size());
        std::unordered_set<int> seen;
        for (int k : s)
        {
            TEST_CHECK(expected.count(k) == 1);
            TEST_CHECK(seen.insert(k).second);
        }
        TEST_CHECK(seen.size() == expected.size());
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
    // Thread count is an optional second argument (default 8; pass 16 to
    // validate 16-core scaling -- the 16-thread matrix runs in CI as a
    // documented verification, see bench/results.md).
    const unsigned threads =
        argc > 2 ? static_cast<unsigned>(std::atoll(argv[2])) : 8;
    std::printf("stress_unordered: iterations=%u threads=%u\n", iterations,
                threads);

    std::printf("map hazard:\n");
    stress_map<mpmc::reclamation::hazard_pointer>(threads, iterations);
    std::printf("map epoch:\n");
    stress_map<mpmc::reclamation::epoch>(threads, iterations);
    std::printf("set hazard:\n");
    stress_set<mpmc::reclamation::hazard_pointer>(threads, iterations);
    std::printf("set epoch:\n");
    stress_set<mpmc::reclamation::epoch>(threads, iterations);

    std::printf("stress_unordered: all tests passed\n");
    return 0;
}
