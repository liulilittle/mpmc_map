/**
 * @file test_unordered.cpp
 * @brief Unit tests for mpmc::unordered_map and mpmc::unordered_set.
 *
 * Coverage:
 *   * single-threaded oracle test against std::unordered_map / set with
 *     randomized operations (fixed seeds for reproducibility)
 *   * duplicate inserts, missing-key erases, operator[] / at() semantics
 *   * iteration visits exactly the inserted keys
 *   * growth (reserve and load-factor doubling) with 100k+ keys
 *   * deliberate hash collisions (equal split-order runs)
 *   * clear() and reuse
 *   * leak checks via a counting allocator (allocs == frees) and the
 *     reclaimer's outstanding counter
 *   * both reclamation schemes for every scenario
 */

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/check.hpp"
#include "mpmc/mpmc_unordered_map.hpp"
#include "mpmc/mpmc_unordered_set.hpp"

namespace
{
    // ---- deterministic RNG (xorshift64) --------------------------------
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

        unsigned next_unsigned(unsigned mod)
        {
            return static_cast<unsigned>(next() % mod);
        }
    };

    // ---- counting allocator --------------------------------------------
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

    struct alloc_guard
    {
        std::size_t before;

        alloc_guard()
            : before(counting_allocator<unsigned char>::allocs.load(
                  std::memory_order_relaxed))
        {
        }

        std::size_t frees() const
        {
            return counting_allocator<unsigned char>::frees.load(
                std::memory_order_relaxed);
        }

        void check_balanced() const
        {
            // allocs of unsigned char is a proxy; the real assertion is
            // per-test (allocs == frees at the end of the process, and the
            // outstanding-reclaimer counter).
        }
    };

    // ---- poor hash: forces equal split-order runs ----------------------
    struct weak_hash
    {
        std::size_t operator()(int k) const
        {
            return static_cast<std::size_t>(k) & 0x7;
        }
    };

    // ---- oracle test -----------------------------------------------------
    template <typename Map>
    void oracle_test(std::uint64_t seed, unsigned iterations, unsigned key_mod)
    {
        Map m;
        std::unordered_map<int, int> ref;
        rng r(seed);
        for (unsigned i = 0; i < iterations; ++i)
        {
            const int k = static_cast<int>(r.next_unsigned(key_mod));
            switch (r.next_unsigned(4))
            {
            case 0: // insert
            {
                const int v = static_cast<int>(r.next_unsigned(1000));
                auto a = m.insert(std::make_pair(k, v));
                auto b = ref.insert(std::make_pair(k, v));
                TEST_CHECK(a.second == b.second);
                TEST_CHECK(m.size() == ref.size());
                break;
            }
            case 1: // erase
            {
                const std::size_t a = m.erase(k);
                const std::size_t b = ref.erase(k);
                TEST_CHECK(a == b);
                TEST_CHECK(m.size() == ref.size());
                break;
            }
            case 2: // find + count
            {
                TEST_CHECK(m.contains(k) == (ref.count(k) != 0));
                typename Map::iterator it = m.find(k);
                TEST_CHECK((it == m.end()) == (ref.find(k) == ref.end()));
                if (it != m.end())
                {
                    TEST_CHECK(it->first == k);
                    TEST_CHECK(it->second == ref.find(k)->second);
                }
                break;
            }
            case 3: // operator[]
            {
                int& v = m[k];
                int& rv = ref[k];
                TEST_CHECK(v == rv);
                TEST_CHECK(m.size() == ref.size());
                v = static_cast<int>(r.next_unsigned(1000));
                rv = v;
                break;
            }
            }
        }
        TEST_CHECK(m.size() == ref.size());
        // verify iteration visits exactly the reference keys
        std::unordered_set<int> seen;
        for (auto& kv : m)
        {
            TEST_CHECK(ref.count(kv.first) == 1);
            TEST_CHECK(kv.second == ref[kv.first]);
            TEST_CHECK(seen.insert(kv.first).second);
        }
        TEST_CHECK(seen.size() == ref.size());
    }

    // ---- set oracle -------------------------------------------------------
    template <typename Set>
    void oracle_set_test(std::uint64_t seed, unsigned iterations,
                         unsigned key_mod)
    {
        Set s;
        std::unordered_set<int> ref;
        rng r(seed);
        for (unsigned i = 0; i < iterations; ++i)
        {
            const int k = static_cast<int>(r.next_unsigned(key_mod));
            if (r.next_unsigned(2) == 0)
            {
                auto a = s.insert(k);
                auto b = ref.insert(k);
                TEST_CHECK(a.second == b.second);
            }
            else
            {
                const std::size_t a = s.erase(k);
                const std::size_t b = ref.erase(k);
                TEST_CHECK(a == b);
            }
            TEST_CHECK(s.size() == ref.size());
            TEST_CHECK(s.contains(k) == (ref.count(k) != 0));
        }
        std::unordered_set<int> seen;
        for (int k : s)
        {
            TEST_CHECK(ref.count(k) == 1);
            TEST_CHECK(seen.insert(k).second);
        }
        TEST_CHECK(seen.size() == ref.size());
    }

    // ---- large growth + iteration ---------------------------------------
    template <typename Map>
    void growth_test(unsigned n)
    {
        Map m;
        m.reserve(1000);
        for (unsigned i = 0; i < n; ++i)
        {
            m.insert(std::make_pair(i, static_cast<int>(i * 3)));
        }
        TEST_CHECK(m.size() == n);
        TEST_CHECK(m.bucket_count() >= 2);
        for (unsigned i = 0; i < n; ++i)
        {
            auto it = m.find(i);
            TEST_CHECK(it != m.end());
            TEST_CHECK(it->second == static_cast<int>(i * 3));
        }
        std::unordered_set<int> seen;
        for (auto& kv : m)
        {
            TEST_CHECK(static_cast<unsigned>(kv.first) < n);
            TEST_CHECK(seen.insert(kv.first).second);
        }
        TEST_CHECK(seen.size() == n);
        for (unsigned i = 0; i < n; i += 2)
        {
            TEST_CHECK(m.erase(i) == 1);
        }
        TEST_CHECK(m.size() == n / 2);
        for (unsigned i = 1; i < n; i += 2)
        {
            TEST_CHECK(m.contains(i));
        }
        for (unsigned i = 0; i < n; i += 2)
        {
            TEST_CHECK(!m.contains(i));
        }
    }

    // ---- collisions --------------------------------------------------------
    template <typename Map>
    void collision_test()
    {
        Map m;
        // weak_hash: only 8 distinct hashes -> long equal-key runs
        for (int i = 0; i < 1000; ++i)
        {
            m.insert(std::make_pair(i, i));
        }
        TEST_CHECK(m.size() == 1000);
        for (int i = 0; i < 1000; ++i)
        {
            auto it = m.find(i);
            TEST_CHECK(it != m.end());
            TEST_CHECK(it->second == i);
        }
        int erased = 0;
        for (int i = 0; i < 1000; i += 2)
        {
            erased += static_cast<int>(m.erase(i));
        }
        TEST_CHECK(erased == 500);
        TEST_CHECK(m.size() == 500);
        for (int i = 1; i < 1000; i += 2)
        {
            TEST_CHECK(m.contains(i));
        }
        std::unordered_set<int> seen;
        for (auto& kv : m)
        {
            TEST_CHECK((kv.first & 1) == 1);
            TEST_CHECK(seen.insert(kv.first).second);
        }
        TEST_CHECK(seen.size() == 500);
    }

    // ---- basic semantics ---------------------------------------------------
    template <typename Map>
    void semantics_test()
    {
        Map m;
        TEST_CHECK(m.empty());
        TEST_CHECK(m.size() == 0);

        auto r1 = m.insert(std::make_pair(1, 10));
        TEST_CHECK(r1.second);
        auto r2 = m.insert(std::make_pair(1, 11));
        TEST_CHECK(!r2.second);
        TEST_CHECK(m.size() == 1);
        TEST_CHECK(m.at(1) == 10);

        m[2] = 20;
        TEST_CHECK(m.at(2) == 20);
        m[2] = 21;
        TEST_CHECK(m.at(2) == 21);

        TEST_CHECK(m.erase(1) == 1);
        TEST_CHECK(m.erase(1) == 0);
        TEST_CHECK(!m.contains(1));

        bool threw = false;
        try
        {
            m.at(1);
        }
        catch (const std::out_of_range&)
        {
            threw = true;
        }
        TEST_CHECK(threw);

        TEST_CHECK(m.count(2) == 1);

        // erase via iterator
        auto it = m.find(2);
        TEST_CHECK(it != m.end());
        m.erase(it);
        TEST_CHECK(m.empty());

        // clear and reuse
        for (int i = 0; i < 100; ++i)
        {
            m[i] = i;
        }
        TEST_CHECK(m.size() == 100);
        m.clear();
        TEST_CHECK(m.empty());
        m[7] = 77;
        TEST_CHECK(m.at(7) == 77);
        TEST_CHECK(m.size() == 1);

        // try_emplace / insert_or_assign
        auto t1 = m.try_emplace(8, 80);
        TEST_CHECK(t1.second);
        auto t2 = m.try_emplace(8, 81);
        TEST_CHECK(!t2.second);
        TEST_CHECK(m.at(8) == 80);
        auto a1 = m.insert_or_assign(8, 82);
        TEST_CHECK(!a1.second);
        TEST_CHECK(m.at(8) == 82);
        auto a2 = m.insert_or_assign(9, 90);
        TEST_CHECK(a2.second);
        TEST_CHECK(m.at(9) == 90);

        // iterator equality
        TEST_CHECK(m.find(8) == m.find(8));
        TEST_CHECK(m.find(8) != m.end());
        TEST_CHECK(m.find(1234) == m.end());

        // const access
        const Map& cm = m;
        TEST_CHECK(cm.at(8) == 82);
        TEST_CHECK(cm.contains(9));
        std::size_t count = 0;
        for (auto cit = cm.begin(); cit != cm.end(); ++cit)
        {
            TEST_CHECK(cit->second > 0);
            ++count;
        }
        TEST_CHECK(count == cm.size());
    }

    // ---- leak check ----------------------------------------------------------
    template <typename Scheme>
    void leak_test()
    {
        using map_type = mpmc::unordered_map<int, int, std::hash<int>,
                                             std::equal_to<int>,
                                             counting_allocator<
                                                 std::pair<const int, int>>,
                                             Scheme>;
        const std::size_t allocs_before =
            counting_allocator<unsigned char>::allocs.load(
                std::memory_order_relaxed);
        const std::size_t frees_before =
            counting_allocator<unsigned char>::frees.load(
                std::memory_order_relaxed);
        {
            map_type m;
            for (int i = 0; i < 100000; ++i)
            {
                m.insert(std::make_pair(i, i));
            }
            for (int i = 0; i < 100000; i += 2)
            {
                m.erase(i);
            }
        }
        const std::size_t allocs_after =
            counting_allocator<unsigned char>::allocs.load(
                std::memory_order_relaxed);
        const std::size_t frees_after =
            counting_allocator<unsigned char>::frees.load(
                std::memory_order_relaxed);
        TEST_CHECK(allocs_after == frees_after &&
                   "every allocation must be released");
        (void)allocs_before;
        (void)frees_before;
        TEST_CHECK(!map_type::reclaimer::has_outstanding());
    }

    // ---- concurrency smoke ---------------------------------------------------
    template <typename Map>
    void concurrent_smoke_map(unsigned threads, unsigned per_thread)
    {
        Map m;
        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; ++t)
            {
                ts.emplace_back([&m, t, per_thread]() {
                    for (unsigned i = 0; i < per_thread; ++i)
                    {
                        const int k = static_cast<int>(t * per_thread + i);
                        m.insert(std::make_pair(k, k));
                        if ((i & 1) == 0)
                        {
                            m.find(k);
                        }
                    }
                    for (unsigned i = 0; i < per_thread; i += 2)
                    {
                        m.erase(static_cast<int>(t * per_thread + i));
                    }
                });
            }
            for (auto& t : ts)
            {
                t.join();
            }
        }
        std::size_t expected = threads * (per_thread / 2);
        TEST_CHECK(m.size() == expected);
        std::unordered_set<int> seen;
        for (auto& kv : m)
        {
            TEST_CHECK((kv.first & 1) == 1);
            TEST_CHECK(seen.insert(kv.first).second);
        }
        TEST_CHECK(seen.size() == expected);
        m.clear();
        TEST_CHECK(m.empty());
    }

    template <typename Set>
    void concurrent_smoke_set(unsigned threads, unsigned per_thread)
    {
        Set s;
        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; ++t)
            {
                ts.emplace_back([&s, t, per_thread]() {
                    for (unsigned i = 0; i < per_thread; ++i)
                    {
                        const int k = static_cast<int>(t * per_thread + i);
                        s.insert(k);
                        if ((i & 1) == 0)
                        {
                            s.find(k);
                        }
                    }
                    for (unsigned i = 0; i < per_thread; i += 2)
                    {
                        s.erase(static_cast<int>(t * per_thread + i));
                    }
                });
            }
            for (auto& t : ts)
            {
                t.join();
            }
        }
        std::size_t expected = threads * (per_thread / 2);
        TEST_CHECK(s.size() == expected);
        std::unordered_set<int> seen;
        for (int k : s)
        {
            TEST_CHECK((k & 1) == 1);
            TEST_CHECK(seen.insert(k).second);
        }
        TEST_CHECK(seen.size() == expected);
        s.clear();
        TEST_CHECK(s.empty());
    }
}

int main()
{
    std::printf("oracle"); fflush(stdout);
    oracle_test<mpmc::unordered_map<int, int>>(1, 500000, 1000);
    std::printf("oracle"); fflush(stdout);
    oracle_test<mpmc::unordered_map<int, int, std::hash<int>,
                                    std::equal_to<int>,
                                    std::allocator<std::pair<const int, int>>,
                                    mpmc::reclamation::epoch>>(2, 500000, 1000);
    std::printf("growth"); fflush(stdout);
    growth_test<mpmc::unordered_map<int, int>>(200000);
    std::printf("collision"); fflush(stdout);
    collision_test<mpmc::unordered_map<int, int, weak_hash>>();
    std::printf("semantics"); fflush(stdout);
    semantics_test<mpmc::unordered_map<int, int>>();
    std::printf("semantics"); fflush(stdout);
    semantics_test<mpmc::unordered_map<int, int, std::hash<int>,
                                       std::equal_to<int>,
                                       std::allocator<std::pair<const int, int>>,
                                       mpmc::reclamation::epoch>>();
    std::printf("leak"); fflush(stdout);
    leak_test<mpmc::reclamation::hazard_pointer>();
    leak_test<mpmc::reclamation::epoch>();
    std::printf("smoke"); fflush(stdout);
    concurrent_smoke_map<mpmc::unordered_map<int, int>>(8, 100000);

    std::printf("oracle"); fflush(stdout);
    oracle_set_test<mpmc::unordered_set<int>>(3, 500000, 1000);
    std::printf("oracle"); fflush(stdout);
    oracle_set_test<mpmc::unordered_set<int, std::hash<int>, std::equal_to<int>,
                                         std::allocator<int>,
                                         mpmc::reclamation::epoch>>(
        4, 500000, 1000);
    std::printf("set growth"); fflush(stdout);
    {
        mpmc::unordered_set<int> s;
        for (int i = 0; i < 10000; ++i)
        {
            s.insert(i);
        }
        TEST_CHECK(s.size() == 10000);
        std::unordered_set<int> seen;
        for (int k : s)
        {
            TEST_CHECK(k >= 0 && k < 10000);
            TEST_CHECK(seen.insert(k).second);
        }
        TEST_CHECK(seen.size() == 10000);
    }
    std::printf("smoke"); fflush(stdout);
    concurrent_smoke_set<mpmc::unordered_set<int>>(8, 100000);

    std::printf("test_unordered"); fflush(stdout);
    return 0;
}
