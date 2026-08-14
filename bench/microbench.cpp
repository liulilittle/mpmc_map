/**
 * @file microbench.cpp
 * @brief Cost-component microbenchmarks (single-threaded).
 *
 * Isolates the per-operation costs that make up the container overhead:
 *
 *   A. epoch guard enter/exit          (per-operation reclamation cost)
 *   B. hazard guard + protect/clear    (same, hazard scheme)
 *   C. mpmc::map vs std::map           (insert / erase / find)
 *   D. mpmc::unordered_map vs std::unordered_map (insert / erase / find)
 *   E. raw node allocation cost        (new/delete of a node-sized block)
 *
 * All numbers are nanoseconds per operation, Release build, single
 * thread. The container workloads use a bounded key space so the
 * containers stay at a realistic size.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <unordered_map>

#include "mpmc/mpmc_map.hpp"
#include "mpmc/mpmc_unordered_map.hpp"

namespace
{
    template <typename F>
    double ns_per_op(F fn, unsigned long long iters)
    {
        fn(); // warmup
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned long long i = 0; i < iters; ++i)
        {
            fn();
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count() /
               static_cast<double>(iters);
    }

    template <typename F>
    void report(const char* name, F fn, unsigned long long iters)
    {
        std::printf("  %-46s %8.2f ns/op\n", name, ns_per_op(fn, iters));
        std::fflush(stdout);
    }
}

int main(int argc, char** argv)
{
    const unsigned long long iters =
        argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2000000ULL;
    const unsigned long long big =
        argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 200000ULL;
    std::printf("microbench: iters=%llu container_ops=%llu\n",
                (unsigned long long)iters, (unsigned long long)big);

    // ---- A: epoch guard -------------------------------------------------
    {
        using map_type = mpmc::map<int, int>;
        using reclaimer = typename map_type::reclaimer;
        report("A epoch guard enter/exit", [&]() {
            typename reclaimer::op_guard g;
        }, iters);
    }

    // ---- B: hazard guard + protects --------------------------------------
    {
        using map_type = mpmc::unordered_map<
            int, int, std::hash<int>, std::equal_to<int>,
            std::allocator<std::pair<const int, int>>,
            mpmc::reclamation::hazard_pointer>;
        using reclaimer = typename map_type::reclaimer;
        static int dummy = 0;
        report("B hazard guard + 2 protects", [&]() {
            typename reclaimer::op_guard g;
            reclaimer::protect(0, &dummy);
            reclaimer::protect(1, &dummy);
            reclaimer::clear(0);
            reclaimer::clear(1);
        }, iters);
    }

    // ---- C: mpmc::map vs std::map ----------------------------------------
    {
        mpmc::map<int, int> mm;
        for (int i = 0; i < 10000; ++i)
        {
            mm.insert(std::make_pair(i, i));
        }
        std::map<int, int> sm;
        for (int i = 0; i < 10000; ++i)
        {
            sm.insert(std::make_pair(i, i));
        }
        std::atomic<int> sink{0};
        report("C mpmc::map insert (10k tree)", [&]() {
            static int k = 10000;
            mm.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        report("C std::map insert (10k tree)", [&]() {
            static int k = 10000;
            sm.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        report("C mpmc::map find", [&]() {
            sink.fetch_add(mm.find(5000) != mm.end() ? 1 : 0,
                           std::memory_order_relaxed);
        }, iters);
        report("C std::map find", [&]() {
            sink.fetch_add(sm.find(5000) != sm.end() ? 1 : 0,
                           std::memory_order_relaxed);
        }, iters);
        report("C mpmc::map erase", [&]() {
            static int k = 10000;
            mm.erase(k);
            ++k;
        }, iters);
        report("C std::map erase", [&]() {
            static int k = 10000;
            sm.erase(k);
            ++k;
        }, iters);
        (void)sink;
    }

    // ---- D: mpmc::unordered_map vs std::unordered_map --------------------
    {
        mpmc::unordered_map<int, int> mu;
        for (int i = 0; i < 10000; ++i)
        {
            mu.insert(std::make_pair(i, i));
        }
        std::unordered_map<int, int> su;
        for (int i = 0; i < 10000; ++i)
        {
            su.insert(std::make_pair(i, i));
        }
        std::atomic<int> sink{0};
        report("D mpmc::unordered insert (growing)", [&]() {
            static int k = 10000;
            mu.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        report("D std::unordered insert (growing)", [&]() {
            static int k = 10000;
            su.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        report("D mpmc::unordered insert (steady 10k)", [&]() {
            static int k = 10000;
            mu.insert(std::make_pair(k % 10000, k));
            ++k;
        }, iters);
        report("D std::unordered insert (steady 10k)", [&]() {
            static int k = 10000;
            su.insert(std::make_pair(k % 10000, k));
            ++k;
        }, iters);
        report("D mpmc::unordered find", [&]() {
            sink.fetch_add(mu.find(5000) != mu.end() ? 1 : 0,
                           std::memory_order_relaxed);
        }, iters);
        report("D std::unordered find", [&]() {
            sink.fetch_add(su.find(5000) != su.end() ? 1 : 0,
                           std::memory_order_relaxed);
        }, iters);
        report("D mpmc::unordered erase (growing)", [&]() {
            static int k = 10000;
            mu.erase(k);
            ++k;
        }, iters);
        report("D std::unordered erase (growing)", [&]() {
            static int k = 10000;
            su.erase(k);
            ++k;
        }, iters);
        report("D mpmc::unordered erase (steady 10k)", [&]() {
            static int k = 10000;
            mu.erase(k % 10000);
            ++k;
        }, iters);
        report("D std::unordered erase (steady 10k)", [&]() {
            static int k = 10000;
            su.erase(k % 10000);
            ++k;
        }, iters);
        (void)sink;
    }

    // ---- D2: growth-isolated insert ----------------------------------------
    {
        // huge load factor: growth never triggers — isolates the pure
        // insert path (locate + scan + CAS + alloc)
        mpmc::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
                            std::allocator<std::pair<const int, int>>,
                            mpmc::reclamation::epoch>
            mu_nogrow;
        mu_nogrow.max_load_factor(1000.0f);
        for (int i = 0; i < 10000; ++i)
        {
            mu_nogrow.insert(std::make_pair(i, i));
        }
        report("D2 mpmc::unordered insert (fresh, NO growth)", [&]() {
            static int k = 10000;
            mu_nogrow.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        // growth-only: default load factor, fresh keys
        mpmc::unordered_map<int, int> mu_grow;
        for (int i = 0; i < 10000; ++i)
        {
            mu_grow.insert(std::make_pair(i, i));
        }
        report("D2 mpmc::unordered insert (fresh, growth)", [&]() {
            static int k = 10000;
            mu_grow.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        // pre-reserved: table already at final size, no growth during the
        // measured phase — isolates the steady insert at load ~1.0
        mpmc::unordered_map<int, int> mu_res;
        mu_res.reserve(1u << 21);
        for (int i = 0; i < 10000; ++i)
        {
            mu_res.insert(std::make_pair(i, i));
        }
        report("D2 mpmc::unordered insert (fresh, pre-reserved)", [&]() {
            static int k = 10000;
            mu_res.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        std::printf("    (reserved bucket_count=%llu)\n",
                    (unsigned long long)mu_res.bucket_count());
        // cache-miss hypothesis: same fresh-key insert on a small table
        // (64k cells = 512KB, L2-resident) vs the 16MB table above
        mpmc::unordered_map<int, int> mu_small;
        mu_small.reserve(1u << 16);
        report("D2 mpmc::unordered insert (fresh, 64k table)", [&]() {
            static int k = 100000;
            mu_small.insert(std::make_pair(k, k));
            ++k;
        }, iters / 8);
        std::printf("    (small bucket_count=%llu)\n",
                    (unsigned long long)mu_small.bucket_count());
        // find on the same tables: isolates the bucket-cell / walk cost
        report("D2 mpmc::unordered find (16MB table)", [&]() {
            static int k = 100000;
            mu_res.find(k);
            ++k;
        }, iters);
        report("D2 mpmc::unordered find (2MB table)", [&]() {
            static int k = 100000;
            mu_small.find(k);
            ++k;
        }, iters);
        // std on the same big-table setup for a fair comparison
        std::unordered_map<int, int> su_big;
        su_big.reserve(1u << 21);
        for (int i = 0; i < 10000; ++i)
        {
            su_big.insert(std::make_pair(i, i));
        }
        report("D2 std::unordered insert (fresh, pre-reserved)", [&]() {
            static int k = 100000;
            su_big.insert(std::make_pair(k, k));
            ++k;
        }, iters);
        report("D2 std::unordered find (16MB table)", [&]() {
            static int k = 100000;
            su_big.find(k);
            ++k;
        }, iters);
        // duplicate insert (no CAS, no count, no grow) on both tables:
        // splits locate+scan cost from the fresh-insert extras
        report("D2 mpmc::unordered dup-insert (16MB table)", [&]() {
            static int k = 0;
            mu_res.insert(std::make_pair(k % 10000, k));
            ++k;
        }, iters);
        mpmc::unordered_map<int, int> mu_dup;
        for (int i = 0; i < 10000; ++i)
        {
            mu_dup.insert(std::make_pair(i, i));
        }
        report("D2 mpmc::unordered dup-insert (small table)", [&]() {
            static int k = 0;
            mu_dup.insert(std::make_pair(k % 10000, k));
            ++k;
        }, iters);
        report("D2 std::unordered dup-insert (16MB table)", [&]() {
            static int k = 0;
            su_big.insert(std::make_pair(k % 10000, k));
            ++k;
        }, iters);
        // steady churn: insert fresh + erase, table stays at final size
        report("D2 mpmc::unordered churn (16MB table)", [&]() {
            static int k = 100000;
            mu_res.insert(std::make_pair(k, k));
            mu_res.erase(k);
            ++k;
        }, iters);
        report("D2 std::unordered churn (16MB table)", [&]() {
            static int k = 100000;
            su_big.insert(std::make_pair(k, k));
            su_big.erase(k);
            ++k;
        }, iters);
    }

    // ---- E: raw allocation ------------------------------------------------
    {
        struct node
        {
            std::atomic<void*> next;
            int payload;
        };
        report("E new/delete node-sized block", [&]() {
            node* n = new node();
            delete n;
        }, iters);
        report("E new without delete (heap growth)", [&]() {
            static std::vector<node*> keep;
            node* n = new node();
            keep.push_back(n);
        }, iters);
        std::printf("    (kept %zu blocks)\n", 0);
        report("E std::allocator allocate/deallocate (no reuse)", [&]() {
            static std::allocator<node> a;
            node* n = a.allocate(1);
            a.deallocate(n, 1);
        }, iters);
    }

    std::printf("microbench done\n");
    return 0;
}
