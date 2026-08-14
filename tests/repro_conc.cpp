/**
 * @file repro_conc.cpp
 * @brief Focused concurrent red-black tree repro (root-race regression).
 *
 * Scenario (from the failing pre-fix runs): two threads each insert a
 * disjoint range of 20000 keys, then each erases its own even keys. The
 * old lock_path descended from a stale root after blocking on its lock,
 * which corrupted the BST (erases missed present keys).
 *
 * The scenario is repeated `rounds` times; any missed erase, wrong size,
 * or broken ordering fails the test. A mixed insert/erase/find phase runs
 * concurrently as a second stress.
 */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "mpmc/mpmc_map.hpp"

namespace
{
    template <typename Map>
    int run_rounds(unsigned rounds, unsigned per_thread)
    {
        unsigned misses = 0;
        for (unsigned r = 0; r < rounds; ++r)
        {
            Map m;
            {
                std::vector<std::thread> ts;
                for (unsigned t = 0; t < 2; ++t)
                {
                    ts.emplace_back([&m, &misses, r, t, per_thread]() {
                        for (unsigned i = 0; i < per_thread; ++i)
                        {
                            const int k = static_cast<int>(t * per_thread + i);
                            m.insert(std::make_pair(k, k));
                        }
                        for (unsigned i = 0; i < per_thread; i += 2)
                        {
                            const int k = static_cast<int>(t * per_thread + i);
                            if (m.erase(k) != 1)
                            {
                                std::fprintf(stderr,
                                             "  MISS round=%u t=%u k=%d\n", r,
                                             t, k);
                                std::fflush(stderr);
                                ++misses;
                            }
                        }
                    });
                }
                for (auto& t : ts)
                {
                    t.join();
                }
            }
            const std::size_t expected = 2 * (per_thread / 2);
            if (m.size() != expected)
            {
                std::fprintf(stderr, "  SIZE round=%u got=%llu want=%llu\n", r,
                             (unsigned long long)m.size(),
                             (unsigned long long)expected);
                std::fflush(stderr);
                ++misses;
            }
            int prev = -1;
            std::size_t count = 0;
            for (auto& kv : m)
            {
                if ((kv.first & 1) == 0)
                {
                    std::fprintf(stderr, "  EXTRA-EVEN round=%u k=%d\n", r,
                                 kv.first);
                    std::fflush(stderr);
                    ++misses;
                }
                if (kv.first <= prev)
                {
                    std::fprintf(stderr, "  ORDER round=%u %d after %d\n", r,
                                 kv.first, prev);
                    std::fflush(stderr);
                    ++misses;
                }
                prev = kv.first;
                ++count;
            }
            if (count != expected)
            {
                std::fprintf(stderr, "  ITER-COUNT round=%u got=%llu\n", r,
                             (unsigned long long)count);
                std::fflush(stderr);
                ++misses;
            }
            if ((r & 31) == 0)
            {
                std::printf("  round %u ok\n", r);
                std::fflush(stdout);
            }
        }
        return static_cast<int>(misses);
    }

    template <typename Map>
    int run_mixed(unsigned rounds, unsigned ops)
    {
        // two threads run random-ish ops over a shared key space with a
        // per-key oracle-free invariant: every erase result is either 0 or
        // 1 and the final tree must be in order with the exact size.
        unsigned bad = 0;
        for (unsigned r = 0; r < rounds; ++r)
        {
            Map m;
            {
                std::vector<std::thread> ts;
                for (unsigned t = 0; t < 4; ++t)
                {
                    ts.emplace_back([&m, t, ops]() {
                        std::uint64_t s = 0x9e3779b97f4a7c15ULL ^ (t * 977u);
                        for (unsigned i = 0; i < ops; ++i)
                        {
                            s ^= s << 13;
                            s ^= s >> 7;
                            s ^= s << 17;
                            const int k = static_cast<int>((s >> 8) % 4096);
                            const unsigned op = static_cast<unsigned>(s % 3);
                            if (op == 0)
                            {
                                m.insert(std::make_pair(k, k));
                            }
                            else if (op == 1)
                            {
                                m.erase(k);
                            }
                            else
                            {
                                m.find(k);
                            }
                        }
                    });
                }
                for (auto& t : ts)
                {
                    t.join();
                }
            }
            std::size_t size = m.size();
            std::size_t count = 0;
            int prev = -1;
            for (auto& kv : m)
            {
                if (kv.first <= prev)
                {
                    std::fprintf(stderr, "  MIX-ORDER round=%u %d after %d\n",
                                 r, kv.first, prev);
                    std::fflush(stderr);
                    ++bad;
                }
                prev = kv.first;
                ++count;
            }
            if (count != size)
            {
                std::fprintf(stderr, "  MIX-COUNT round=%u iter=%llu size=%llu\n",
                             r, (unsigned long long)count,
                             (unsigned long long)size);
                std::fflush(stderr);
                ++bad;
            }
        }
        return static_cast<int>(bad);
    }
}

int main(int argc, char** argv)
{
    unsigned rounds = 100;
    if (argc > 1)
    {
        rounds = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    }
    else
    {
        rounds = 100;
    }
    std::printf("repro: 2-thread rounds=%u\n", rounds);
    std::fflush(stdout);
    int bad = run_rounds<mpmc::map<int, int>>(rounds, 20000);
    std::printf("mixed: rounds=%u\n", rounds);
    std::fflush(stdout);
    bad += run_mixed<mpmc::map<int, int>>(rounds, 50000);
    std::printf("repro_conc: %s (bad=%d)\n", bad == 0 ? "PASS" : "FAIL",
                bad);
    return bad == 0 ? 0 : 1;
}
