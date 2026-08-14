/**
 * @file test_pool.cpp
 * @brief Tests for the per-instantiation node pool and allocator
 *        interactions (the D2 optimization layer).
 *
 * Coverage:
 *   * two containers of the same instantiation (shared reclaimer and
 *     shared node pool) with the SAME allocator: churn across both, the
 *     pool reuses blocks, every allocation is eventually released;
 *   * two containers with DIFFERENT allocator instances (stateful,
 *     counting per instance): the pool must never cross-free — each
 *     block is released through the allocator it was allocated with;
 *   * clear() + reuse and destroy-after-churn leak checks.
 */

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <memory>

#include "common/check.hpp"
#include "mpmc/mpmc_unordered_map.hpp"
#include "mpmc/mpmc_unordered_set.hpp"
#include "mpmc/mpmc_map.hpp"
#include "mpmc/mpmc_set.hpp"

namespace
{
    // Stateful counting allocator: each instance counts its own
    // allocate/deallocate calls.
    template <typename T>
    struct stateful_allocator
    {
        using value_type = T;
        std::atomic<std::size_t>* allocs;
        std::atomic<std::size_t>* frees;

        explicit stateful_allocator(std::atomic<std::size_t>* a,
                                    std::atomic<std::size_t>* f)
            : allocs(a)
            , frees(f)
        {
        }

        template <typename U>
        stateful_allocator(const stateful_allocator<U>& o)
            : allocs(o.allocs)
            , frees(o.frees)
        {
        }

        T* allocate(std::size_t n)
        {
            allocs->fetch_add(n, std::memory_order_relaxed);
            return std::allocator<T>().allocate(n);
        }

        void deallocate(T* p, std::size_t n)
        {
            frees->fetch_add(n, std::memory_order_relaxed);
            std::allocator<T>().deallocate(p, n);
        }

        template <typename U>
        struct rebind
        {
            using other = stateful_allocator<U>;
        };

        friend bool operator==(const stateful_allocator& a,
                               const stateful_allocator& b)
        {
            return a.allocs == b.allocs;
        }

        friend bool operator!=(const stateful_allocator& a,
                               const stateful_allocator& b)
        {
            return !(a == b);
        }
    };

    template <typename Scheme>
    void same_allocator_test()
    {
        // Two containers of one instantiation share the node pool; the
        // churn across both must balance allocations exactly.
        std::atomic<std::size_t> allocs(0);
        std::atomic<std::size_t> frees(0);
        using pair_t = std::pair<const int, int>;
        using alloc_t = stateful_allocator<pair_t>;
        using map2_type = mpmc::unordered_map<int, int, std::hash<int>,
                                              std::equal_to<int>, alloc_t,
                                              Scheme>;
        {
            map2_type a(alloc_t(&allocs, &frees));
            map2_type b(alloc_t(&allocs, &frees));
            for (int i = 0; i < 20000; ++i)
            {
                a.insert(std::make_pair(i, i));
                b.insert(std::make_pair(i + 20000, i));
                if ((i & 1) == 0)
                {
                    a.erase(i);
                    b.erase(i + 20000);
                }
            }
            TEST_CHECK(a.size() == 10000);
            TEST_CHECK(b.size() == 10000);
        }
        TEST_CHECK(allocs.load(std::memory_order_relaxed) ==
                   frees.load(std::memory_order_relaxed) &&
                   "pool must not leak across containers of one type");
        TEST_CHECK(!map2_type::reclaimer::has_outstanding());
    }

    template <typename Scheme>
    void distinct_allocator_test()
    {
        // Different allocator instances must never cross-free: the pool's
        // acquire checks alloc_owner and releases foreign blocks through
        // their own allocator.
        using pair_t = std::pair<const int, int>;
        using alloc_t = stateful_allocator<pair_t>;
        using map_type = mpmc::unordered_map<int, int, std::hash<int>,
                                             std::equal_to<int>, alloc_t,
                                             Scheme>;
        std::atomic<std::size_t> a_allocs(0), a_frees(0);
        std::atomic<std::size_t> b_allocs(0), b_frees(0);
        {
            map_type a(alloc_t(&a_allocs, &a_frees));
            map_type b(alloc_t(&b_allocs, &b_frees));
            for (int i = 0; i < 20000; ++i)
            {
                a.insert(std::make_pair(i, i));
                b.insert(std::make_pair(i, i)); // same keys: same pool churn
                if ((i & 1) == 0)
                {
                    a.erase(i);
                    b.erase(i);
                }
            }
            TEST_CHECK(a.size() == 10000);
            TEST_CHECK(b.size() == 10000);
        }
        // The dtor drains the shared pool: every block returns to the
        // allocator it came from.
        TEST_CHECK(a_allocs.load(std::memory_order_relaxed) ==
                   a_frees.load(std::memory_order_relaxed));
        TEST_CHECK(b_allocs.load(std::memory_order_relaxed) ==
                   b_frees.load(std::memory_order_relaxed));
        TEST_CHECK(a_allocs.load(std::memory_order_relaxed) > 0);
        TEST_CHECK(!map_type::reclaimer::has_outstanding());
    }

    template <typename Scheme>
    void tree_pool_test()
    {
        using map_type = mpmc::map<int, int, std::less<int>,
                                   std::allocator<std::pair<const int, int>>,
                                   Scheme>;
        std::atomic<std::size_t> allocs(0);
        std::atomic<std::size_t> frees(0);
        using pair_t = std::pair<const int, int>;
        using alloc_t = stateful_allocator<pair_t>;
        using map2_type = mpmc::map<int, int, std::less<int>, alloc_t,
                                    Scheme>;
        {
            map_type a;
            map2_type b(alloc_t(&allocs, &frees));
            for (int i = 0; i < 20000; ++i)
            {
                a.insert(std::make_pair(i, i));
                b.insert(std::make_pair(i, i));
                if ((i & 1) == 0)
                {
                    a.erase(i);
                    b.erase(i);
                }
            }
        TEST_CHECK(a.size() == 10000);
        TEST_CHECK(b.size() == 10000);
        a.clear();
        TEST_CHECK(a.empty());
        }
        std::printf("  tree_pool: allocs=%llu frees=%llu\n",
                    (unsigned long long)allocs.load(std::memory_order_relaxed),
                    (unsigned long long)frees.load(std::memory_order_relaxed));
        std::fflush(stdout);
        TEST_CHECK(allocs.load(std::memory_order_relaxed) ==
                   frees.load(std::memory_order_relaxed));
        TEST_CHECK(!map2_type::reclaimer::has_outstanding());
    }
template <typename Scheme>
    void get_allocator_test()
    {
        // get_allocator() must return the allocator the container was
        // constructed with — not a default-constructed one. The stateful
        // allocator has no default constructor, so the old bug
        // (`return allocator_type();`) was a compile error here; with a
        // default-constructible stateful allocator it would silently
        // return the wrong instance.
        std::atomic<std::size_t> allocs(0), frees(0);
        using pair_t = std::pair<const int, int>;
        using map_alloc_t = stateful_allocator<pair_t>;
        using set_alloc_t = stateful_allocator<int>;
        {
            mpmc::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
                                map_alloc_t, Scheme> um(
                map_alloc_t(&allocs, &frees));
            TEST_CHECK(um.get_allocator() == map_alloc_t(&allocs, &frees) &&
                       "get_allocator must return the stored allocator");
            mpmc::unordered_set<int, std::hash<int>, std::equal_to<int>,
                                set_alloc_t, Scheme> us(
                set_alloc_t(&allocs, &frees));
            TEST_CHECK(us.get_allocator() == set_alloc_t(&allocs, &frees));
            mpmc::map<int, int, std::less<int>, map_alloc_t, Scheme> m(
                map_alloc_t(&allocs, &frees));
            TEST_CHECK(m.get_allocator() == map_alloc_t(&allocs, &frees));
            mpmc::set<int, std::less<int>, set_alloc_t, Scheme> s(
                set_alloc_t(&allocs, &frees));
            TEST_CHECK(s.get_allocator() == set_alloc_t(&allocs, &frees));
        }
        TEST_CHECK(allocs.load(std::memory_order_relaxed) ==
                   frees.load(std::memory_order_relaxed) &&
                   "get_allocator test must not leak");
    }
}

int main()
{
    std::printf("test_pool: get_allocator (hazard)\n");
    get_allocator_test<mpmc::reclamation::hazard_pointer>();
    std::printf("test_pool: get_allocator (epoch)\n");
    get_allocator_test<mpmc::reclamation::epoch>();
    std::printf("test_pool: same-allocator (hazard)\n");
    same_allocator_test<mpmc::reclamation::hazard_pointer>();
    std::printf("test_pool: same-allocator (epoch)\n");
    same_allocator_test<mpmc::reclamation::epoch>();
    std::printf("test_pool: distinct-allocators (hazard)\n");
    distinct_allocator_test<mpmc::reclamation::hazard_pointer>();
    std::printf("test_pool: distinct-allocators (epoch)\n");
    distinct_allocator_test<mpmc::reclamation::epoch>();
    std::printf("test_pool: tree pool (hazard)\n");
    tree_pool_test<mpmc::reclamation::hazard_pointer>();
    std::printf("test_pool: tree pool (epoch)\n");
    tree_pool_test<mpmc::reclamation::epoch>();
    std::printf("test_pool: all tests passed\n");
    return 0;
}
