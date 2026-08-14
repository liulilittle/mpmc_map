/**
 * @file dbg_degenerate.cpp
 * @brief Red-black balance regression test (single-threaded).
 *
 * Replays a deterministic randomized op stream (the same oracle scenario
 * the unit tests use) and verifies after EVERY operation that:
 *
 *   * the red-black invariants hold (no RED node with a RED child, and
 *     every root-to-leaf path carries the same black-height) — via the
 *     container's audit hook dbg_rb_valid(),
 *   * the tree depth never exceeds the red-black bound
 *     2 * log2(size + 1) + 2 — via dbg_max_depth().
 *
 * These checks catch rebalancing regressions that the order/size oracle
 * cannot see (a valid-BST-but-unbalanced tree passes the oracle but
 * degrades under concurrency). The scenario was derived from a real bug:
 * the old successor-relink fixup left an extra black per erase, which
 * slowly turned the tree into a degenerate chain.
 */

#include <cstdio>
#include <cstdlib>

#include "mpmc/mpmc_map.hpp"

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
}

int main(int argc, char** argv)
{
    unsigned iterations = 200000;
    if (argc > 1)
    {
        iterations = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    }
    mpmc::map<int, int> m;
    rng r(1);
    unsigned max_depth = 0;
    for (unsigned i = 0; i < iterations; ++i)
    {
        const int k = static_cast<int>(r.next_unsigned(1000));
        const unsigned op = r.next_unsigned(5);
        switch (op)
        {
        case 0:
            m.insert(std::make_pair(k, static_cast<int>(r.next_unsigned(1000))));
            break;
        case 1:
            m.erase(k);
            break;
        case 2:
            m.find(k);
            break;
        case 3:
            m[k] = static_cast<int>(r.next_unsigned(1000));
            break;
        case 4:
            m.lower_bound(k);
            m.upper_bound(k);
            break;
        }
        const unsigned d = m.dbg_max_depth();
        if (d > max_depth)
        {
            max_depth = d;
        }
        if (!m.dbg_rb_valid())
        {
            std::fprintf(stderr, "[RB-BROKEN] i=%u op=%u k=%d size=%llu\n", i,
                         op, k, (unsigned long long)m.size());
            std::fflush(stderr);
            return 1;
        }
        const unsigned long long n = m.size();
        unsigned lg = 0;
        for (unsigned long long x = n + 1; x > 1; x >>= 1)
        {
            ++lg;
        }
        const unsigned bound = 2 * lg + 2;
        if (d > bound)
        {
            std::fprintf(stderr,
                         "[DEGEN] i=%u op=%u k=%d depth=%u size=%llu bound=%u\n",
                         i, op, k, d, n, bound);
            std::fflush(stderr);
            return 1;
        }
    }
    std::printf("dbg_degenerate: PASS max_depth=%u\n", max_depth);
    return 0;
}
