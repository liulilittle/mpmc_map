/**
 * @file stress_reclamation.cpp
 * @brief Concurrent stress test of both reclamation schemes.
 *
 * Scenario: a single shared atomic slot holds a pointer to the "current"
 * object. Retirer threads CAS the slot to a fresh object and retire the
 * replaced one; reader threads load the slot, publish it to a hazard slot
 * (hazard scheme), re-validate, and dereference it (checking a magic word).
 *
 * Invariants under stress:
 *
 *   1. No reader ever observes a freed object (magic word always intact) -
 *      this is exactly the use-after-free that both schemes must rule out.
 *   2. Every retired object is freed exactly once (the free callback checks
 *      the magic word; the final counters must match: freed == retired).
 *   3. After every thread joined and the reclaimer is pumped (epoch scheme:
 *      the global epoch must pass each retirement by two generations), zero
 *      retirements are outstanding.
 */

#include <atomic>
#include <mutex>
#include <vector>
#include "common/check.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <thread>
#include <vector>

#include "mpmc/detail/reclamation.hpp"

namespace
{
    struct object
    {
        explicit object()
            : uid(uid_counter().fetch_add(1, std::memory_order_relaxed))
            , magic(MAGIC)
        {
        }

        static std::atomic<unsigned>& uid_counter()
        {
            static std::atomic<unsigned> c(1);
            return c;
        }

        unsigned uid;
        unsigned magic;
        static const unsigned MAGIC = 0x600d600d;
    };

    struct stats_t
    {
        std::atomic<std::uint64_t> retired;
        std::atomic<std::uint64_t> freed;
        std::atomic<std::uint64_t> reads;
        std::atomic<std::uint64_t> bad_reads;

        stats_t()
            : retired(0)
            , freed(0)
            , reads(0)
            , bad_reads(0)
        {
        }
    };

    template <typename Scheme>
    struct harness
    {
        using tag = harness<Scheme>;
        using reclaimer = typename Scheme::template reclaimer<tag>;

        static std::atomic<object*> slot;
        static stats_t stats;
        static std::mutex freed_mutex;
        static std::unordered_map<object*, unsigned> freed_set;
        static std::vector<std::pair<void*, const char*>> retire_log;

        static void log_retire(void* p, const char* who)
        {
            std::lock_guard<std::mutex> lk(freed_mutex);
            retire_log.push_back(std::make_pair(p, who));
        }

        static void free_action(void* ptr)
        {
            object* o = static_cast<object*>(ptr);
            {
                std::lock_guard<std::mutex> lk(freed_mutex);
                std::unordered_map<object*, unsigned>::iterator it =
                    freed_set.find(o);
                if (it != freed_set.end() && it->second == o->uid)
                {
                    std::printf("REAL DOUBLE FREE %p uid=%u\n", (void*)o, o->uid);
                    for (std::size_t j = 0; j < retire_log.size(); ++j)
                    {
                        if (retire_log[j].first == o)
                        {
                            std::printf("  retired by %s\n", retire_log[j].second);
                        }
                    }
                    std::abort();
                }
                freed_set[o] = o->uid;
            }
            TEST_CHECK(o->magic == object::MAGIC); // no use-after-free garbage
            stats.freed.fetch_add(1, std::memory_order_relaxed);
            delete o;
        }

        static void retirer(unsigned iterations)
        {
            typename reclaimer::op_guard g;
            for (unsigned i = 0; i < iterations; ++i)
            {
                object* fresh = new object();
                // The CAS-compare is against an address; if the object at
                // `old` were freed and the address reused (ABA), the CAS
                // would succeed against the new object and retire it while
                // it is live. Publishing `old` as a hazard BEFORE the CAS
                // prevents its reclamation, which rules ABA out - the same
                // protocol the containers use for their unlink CASes.
                for (;;)
                {
                    object* old = slot.load(std::memory_order_acquire);
                    reclaimer::protect(0, old);
                    if (slot.load(std::memory_order_acquire) != old)
                    {
                        continue; // replaced under us: reload and retry
                    }
                    if (slot.compare_exchange_weak(old, fresh,
                                                   std::memory_order_acq_rel))
                    {
                        if (old != nullptr)
                        {
                            stats.retired.fetch_add(1, std::memory_order_relaxed);
                            log_retire(old, "retirer");
                            reclaimer::retire(old, {&harness::free_action});
                        }
                        break;
                    }
                }
            }
            reclaimer::clear(0);
        }

        static void reader(unsigned iterations)
        {
            typename reclaimer::op_guard g;
            for (unsigned i = 0; i < iterations; ++i)
            {
                object* p = slot.load(std::memory_order_acquire);
                if (p == nullptr)
                {
                    continue;
                }
                reclaimer::protect(0, p);
                if (slot.load(std::memory_order_acquire) != p)
                {
                    continue; // replaced concurrently; retry next iteration
                }
                stats.reads.fetch_add(1, std::memory_order_relaxed);
                if (p->magic != object::MAGIC)
                {
                    stats.bad_reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
            reclaimer::clear(0);
        }

        static void run(unsigned iterations)
        {
            const unsigned RETIRERS = 4;
            const unsigned READERS = 4;
            {
                std::vector<std::thread> ts;
                for (unsigned i = 0; i < RETIRERS; ++i)
                {
                    ts.emplace_back(retirer, iterations);
                }
                for (unsigned i = 0; i < READERS; ++i)
                {
                    ts.emplace_back(reader, iterations);
                }
                for (auto& t : ts)
                {
                    t.join();
                }
            }
            // Pump the reclamation (a no-op for hazard pointers) and verify.
            for (int i = 0; i < 8; ++i)
            {
                typename reclaimer::op_guard g;
                reclaimer::reclaim();
            }
            const std::uint64_t retired = stats.retired.load(
                std::memory_order_relaxed);
            const std::uint64_t freed = stats.freed.load(
                std::memory_order_relaxed);
            const std::uint64_t bad = stats.bad_reads.load(
                std::memory_order_relaxed);
            std::printf("  retired=%llu freed=%llu reads=%llu bad_reads=%llu\n",
                        (unsigned long long)retired, (unsigned long long)freed,
                        (unsigned long long)stats.reads.load(std::memory_order_relaxed),
                        (unsigned long long)bad);
            TEST_CHECK(bad == 0 && "use-after-free detected");
            TEST_CHECK(freed == retired && "every retirement freed exactly once");
            TEST_CHECK(!reclaimer::has_outstanding() && "nothing outstanding");
        }
    };

    template <typename Scheme>
    std::atomic<object*> harness<Scheme>::slot(nullptr);

    template <typename Scheme>
    stats_t harness<Scheme>::stats;

    template <typename Scheme>
    std::mutex harness<Scheme>::freed_mutex;

    template <typename Scheme>
    std::unordered_map<object*, unsigned> harness<Scheme>::freed_set;

    template <typename Scheme>
    std::vector<std::pair<void*, const char*>> harness<Scheme>::retire_log;
}

int main(int argc, char** argv)
{
    const unsigned iterations =
        argc > 1 ? static_cast<unsigned>(std::atoll(argv[1])) : 1000000;
    std::printf("stress_reclamation: iterations=%u\n", iterations);

    std::printf("hazard pointers:\n");
    harness<mpmc::reclamation::hazard_pointer>::run(iterations);

    std::printf("epoch:\n");
    harness<mpmc::reclamation::epoch>::run(iterations);

    std::printf("stress_reclamation: all tests passed\n");
    return 0;
}
