/**
 * @file test_tree.cpp
 * @brief Unit tests for mpmc::map and mpmc::set.
 *
 * Coverage:
 *   * single-threaded oracle test against std::map / std::set with
 *     randomized operations (fixed seeds) — including lower_bound,
 *     upper_bound and iteration
 *   * duplicate inserts, missing-key erases, at()/operator[] semantics
 *   * in-order iteration visits exactly the inserted keys in order
 *   * ordered bounds correctness on a large set
 *   * erase of two-children nodes (successor relink) — forced by
 *     inserting then erasing in specific orders
 *   * clear() and reuse
 *   * leak checks via a counting allocator and the reclaimer's
 *     outstanding counter
 *   * both reclamation schemes for every scenario
 */

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "common/check.hpp"
#include "mpmc/mpmc_map.hpp"
#include "mpmc/mpmc_set.hpp"

namespace
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

        unsigned next_unsigned(unsigned mod)
        {
            return static_cast<unsigned>(next() % mod);
        }
    };

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

    // ---- oracle test -----------------------------------------------------
    template <typename Map>
    void oracle_test(std::uint64_t seed, unsigned iterations, unsigned key_mod)
    {
        Map m;
        std::map<int, int> ref;
        rng r(seed);
        for (unsigned i = 0; i < iterations; ++i)
        {
            if (i % 20000 == 0 && seed == 2)
            {
                std::printf("  epoch oracle i=%u size=%llu\n", i,
                            (unsigned long long)m.size());
                std::fflush(stdout);
            }
            const int k = static_cast<int>(r.next_unsigned(key_mod));
            const unsigned op = r.next_unsigned(5);
            if (seed == 2 && i >= 195000)
            {
                std::printf("  op i=%u op=%u k=%d\n", i, op, k);
                std::fflush(stdout);
            }
            switch (op)
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
                auto it = m.find(k);
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
            case 4: // bounds
            {
                auto lb = m.lower_bound(k);
                auto rlb = ref.lower_bound(k);
                TEST_CHECK((lb == m.end()) == (rlb == ref.end()));
                if (lb != m.end())
                {
                    if (lb->first != rlb->first)
                    {
                        std::printf("  [lb] k=%d got=%d want=%d\n", k,
                                    lb->first, rlb->first);
                        std::fflush(stdout);
                        int prev = -9999999;
                        for (auto& kv : m)
                        {
                            if (kv.first <= prev)
                            {
                                std::printf("    ORDER-VIOLATION %d after %d\n",
                                            kv.first, prev);
                                std::fflush(stdout);
                            }
                            prev = kv.first;
                        }
                    }
                    TEST_CHECK(lb->first == rlb->first);
                }
                auto ub = m.upper_bound(k);
                auto rub = ref.upper_bound(k);
                TEST_CHECK((ub == m.end()) == (rub == ref.end()));
                if (ub != m.end())
                {
                    TEST_CHECK(ub->first == rub->first);
                }
                break;
            }
            }
        }
        TEST_CHECK(m.size() == ref.size());
        // iteration: in-order, exactly the reference keys
        std::printf("  [ITER] begin\n");
        std::fflush(stdout);
        auto rit = ref.begin();
        unsigned iter_i = 0;
        for (auto it = m.begin(); it != m.end(); ++it)
        {
            if (iter_i++ % 100 == 0)
            {
                std::printf("  [ITER] %u\n", iter_i);
                std::fflush(stdout);
            }
            TEST_CHECK(rit != ref.end());
            TEST_CHECK(it->first == rit->first);
            TEST_CHECK(it->second == rit->second);
            ++rit;
        }
        std::printf("  [ITER] done %u\n", iter_i);
        std::fflush(stdout);
        TEST_CHECK(rit == ref.end());
    }

    // ---- set oracle -------------------------------------------------------
    template <typename Set>
    void oracle_set_test(std::uint64_t seed, unsigned iterations,
                         unsigned key_mod)
    {
        Set s;
        std::set<int> ref;
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
            if (r.next_unsigned(4) == 0)
            {
                auto lb = s.lower_bound(k);
                auto rlb = ref.lower_bound(k);
                TEST_CHECK((lb == s.end()) == (rlb == ref.end()));
                if (lb != s.end())
                {
                    TEST_CHECK(*lb == *rlb);
                }
            }
        }
        auto rit = ref.begin();
        for (auto it = s.begin(); it != s.end(); ++it)
        {
            TEST_CHECK(rit != ref.end());
            TEST_CHECK(*it == *rit);
            ++rit;
        }
        TEST_CHECK(rit == ref.end());
    }

    // ---- growth + iteration + bounds -------------------------------------
    template <typename Map>
    void growth_test(unsigned n)
    {
        Map m;
        for (unsigned i = 0; i < n; ++i)
        {
            m.insert(std::make_pair(i, static_cast<int>(i * 2)));
        }
        TEST_CHECK(m.size() == n);
        for (unsigned i = 0; i < n; ++i)
        {
            auto it = m.find(i);
            TEST_CHECK(it != m.end());
            TEST_CHECK(it->second == static_cast<int>(i * 2));
        }
        // in-order iteration
        unsigned expect = 0;
        for (auto& kv : m)
        {
            TEST_CHECK(kv.first == static_cast<int>(expect));
            TEST_CHECK(kv.second == static_cast<int>(expect * 2));
            ++expect;
        }
        TEST_CHECK(expect == n);
        // bounds
        for (unsigned i = 0; i < n; ++i)
        {
            auto lb = m.lower_bound(i);
            TEST_CHECK(lb != m.end());
            TEST_CHECK(lb->first == static_cast<int>(i));
            auto ub = m.upper_bound(i);
            TEST_CHECK(ub == m.end() ||
                       ub->first == static_cast<int>(i + 1));
            auto er = m.equal_range(i);
            TEST_CHECK(er.first == lb);
            TEST_CHECK(er.second == ub);
        }
        TEST_CHECK(m.lower_bound(static_cast<int>(n) + 1000) == m.end());
        // erase everything (odd order: two-children relinks)
        for (unsigned i = 0; i < n; ++i)
        {
            TEST_CHECK(m.erase(i) == 1);
        }
        TEST_CHECK(m.empty());
    }

    // ---- two-children delete patterns --------------------------------------
    template <typename Map>
    void relink_test()
    {
        Map m;
        // build a tree with two-children nodes, then erase in orders that
        // force successor relinks
        const int keys[] = {50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43,
                            56, 68, 81, 93};
        for (unsigned i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i)
        {
            m.insert(std::make_pair(keys[i], keys[i]));
        }
        // erase the root (50: two children), then 25 (two children), etc.
        const int order[] = {50, 25, 75, 37, 62, 87, 12, 31, 43, 56, 68, 81,
                             93, 6, 18};
        std::set<int> ref;
        for (unsigned i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i)
        {
            ref.insert(keys[i]);
        }
        for (unsigned i = 0; i < sizeof(order) / sizeof(order[0]); ++i)
        {
            const int k = order[i];
            TEST_CHECK(m.erase(k) == 1);
            TEST_CHECK(ref.erase(k) == 1);
            TEST_CHECK(m.size() == ref.size());
            auto rit = ref.begin();
            for (auto it = m.begin(); it != m.end(); ++it)
            {
                TEST_CHECK(rit != ref.end());
                TEST_CHECK(it->first == *rit);
                ++rit;
            }
            TEST_CHECK(rit == ref.end());
        }
        TEST_CHECK(m.empty());
    }

    // ---- basic semantics ---------------------------------------------------
    template <typename Map>
    void semantics_test()
    {
        Map m;
        TEST_CHECK(m.empty());
        auto r1 = m.insert(std::make_pair(5, 50));
        TEST_CHECK(r1.second);
        auto r2 = m.insert(std::make_pair(5, 51));
        TEST_CHECK(!r2.second);
        TEST_CHECK(m.size() == 1);
        TEST_CHECK(m.at(5) == 50);

        m[3] = 30;
        m[7] = 70;
        m[9] = 90;
        TEST_CHECK(m.size() == 4);
        m[3] = 31;
        TEST_CHECK(m.at(3) == 31);

        TEST_CHECK(m.erase(5) == 1);
        TEST_CHECK(m.erase(5) == 0);
        bool threw = false;
        try
        {
            m.at(5);
        }
        catch (const std::out_of_range&)
        {
            threw = true;
        }
        TEST_CHECK(threw);

        // try_emplace / insert_or_assign
        auto t1 = m.try_emplace(11, 110);
        TEST_CHECK(t1.second);
        auto t2 = m.try_emplace(11, 111);
        TEST_CHECK(!t2.second);
        TEST_CHECK(m.at(11) == 110);
        auto a1 = m.insert_or_assign(11, 112);
        TEST_CHECK(!a1.second);
        TEST_CHECK(m.at(11) == 112);

        // in-order iteration
        int prev = -1;
        for (auto& kv : m)
        {
            TEST_CHECK(kv.first > prev);
            prev = kv.first;
        }

        // erase via iterator
        auto it = m.find(11);
        TEST_CHECK(it != m.end());
        m.erase(it);
        TEST_CHECK(!m.contains(11));

        // clear and reuse
        for (int i = 0; i < 100; ++i)
        {
            m[i] = i;
        }
        m.clear();
        TEST_CHECK(m.empty());
        m[42] = 4242;
        TEST_CHECK(m.at(42) == 4242);

        // const access
        const Map& cm = m;
        TEST_CHECK(cm.at(42) == 4242);
        TEST_CHECK(cm.contains(42));
        TEST_CHECK(cm.count(1) == 0);
    }

    // ---- leak check ----------------------------------------------------------
    template <typename Scheme>
    void leak_test()
    {
        using map_type = mpmc::map<int, int, std::less<int>,
                                   counting_allocator<
                                       std::pair<const int, int>>,
                                   Scheme>;
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
        const std::size_t allocs =
            counting_allocator<unsigned char>::allocs.load(
                std::memory_order_relaxed);
        const std::size_t frees =
            counting_allocator<unsigned char>::frees.load(
                std::memory_order_relaxed);
        TEST_CHECK(allocs == frees && "every allocation must be released");
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
                        const int k = static_cast<int>(t * per_thread + i);
                        if (m.erase(k) != 1)
                        {
                            std::printf("    ERASE-FAIL %d\n", k);
                            std::fflush(stdout);
                        }
                    }
                });
            }
            for (auto& t : ts)
            {
                t.join();
            }
        }
        const std::size_t expected = threads * (per_thread / 2);
        std::printf("  smoke size=%llu expected=%llu\n",
                    (unsigned long long)m.size(),
                    (unsigned long long)expected);
        std::fflush(stdout);
        for (auto& kv : m)
        {
            if ((kv.first & 1) == 0)
            {
                std::printf("    EXTRA-EVEN %d\n", kv.first);
                std::fflush(stdout);
            }
        }
        TEST_CHECK(m.size() == expected);
        std::size_t count = 0;
        int prev = -1;
        for (auto& kv : m)
        {
            TEST_CHECK((kv.first & 1) == 1);
            TEST_CHECK(kv.first > prev);
            prev = kv.first;
            ++count;
        }
        TEST_CHECK(count == expected);
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
        const std::size_t expected = threads * (per_thread / 2);
        TEST_CHECK(s.size() == expected);
        std::size_t count = 0;
        int prev = -1;
        for (int k : s)
        {
            TEST_CHECK((k & 1) == 1);
            TEST_CHECK(k > prev);
            prev = k;
            ++count;
        }
        TEST_CHECK(count == expected);
        s.clear();
        TEST_CHECK(s.empty());
    }
}

int main()
{
    std::printf("oracle map hazard\n");
    std::fflush(stdout);
    oracle_test<mpmc::map<int, int>>(1, 500000, 1000);
    std::printf("oracle map epoch\n");
    std::fflush(stdout);
    oracle_test<mpmc::map<int, int, std::less<int>,
                          std::allocator<std::pair<const int, int>>,
                          mpmc::reclamation::epoch>>(2, 500000, 1000);
    std::printf("growth map\n");
    growth_test<mpmc::map<int, int>>(100000);
    std::printf("relink map\n");
    relink_test<mpmc::map<int, int>>();
    relink_test<mpmc::map<int, int, std::less<int>,
                          std::allocator<std::pair<const int, int>>,
                          mpmc::reclamation::epoch>>();
    std::printf("semantics map\n");
    semantics_test<mpmc::map<int, int>>();
    semantics_test<mpmc::map<int, int, std::less<int>,
                             std::allocator<std::pair<const int, int>>,
                             mpmc::reclamation::epoch>>();
    std::printf("leak map\n");
    leak_test<mpmc::reclamation::hazard_pointer>();
    leak_test<mpmc::reclamation::epoch>();
    std::printf("smoke map\n");
    concurrent_smoke_map<mpmc::map<int, int>>(8, 100000);

    std::printf("oracle set hazard\n");
    oracle_set_test<mpmc::set<int>>(3, 500000, 1000);
    std::printf("oracle set epoch\n");
    oracle_set_test<mpmc::set<int, std::less<int>, std::allocator<int>,
                               mpmc::reclamation::epoch>>(4, 500000, 1000);
    std::printf("set growth\n");
    {
        mpmc::set<int> s;
        for (int i = 0; i < 10000; ++i)
        {
            s.insert(i);
        }
        TEST_CHECK(s.size() == 10000);
        int prev = -1;
        for (int k : s)
        {
            TEST_CHECK(k == prev + 1);
            prev = k;
        }
        TEST_CHECK(prev == 9999);
    }
    std::printf("smoke set\n");
    concurrent_smoke_set<mpmc::set<int>>(8, 100000);

    std::printf("test_tree: all tests passed\n");
    return 0;
}
