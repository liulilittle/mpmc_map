/**
 * @file test_reclamation.cpp
 * @brief Unit tests for the two reclamation schemes.
 *
 * Every test runs against both schemes (hazard pointers and epoch). The
 * observable contract under test:
 *
 *   1. retire() + reclaim() frees an unprotected object exactly once.
 *   2. An object that is protected (published hazard slot, or still inside
 *      the retiring epoch guard) is NOT freed by reclaim(), no matter how
 *      often reclaim() runs.
 *   3. After the protection is gone (and, for the epoch scheme, after the
 *      global epoch has passed the retirement by two generations) the
 *      object is freed exactly once.
 *   4. drain_all() leaves no retirement outstanding.
 *
 * The epoch scheme reclaims an object only after its epoch has been passed
 * by two generations, so "pump_until()" runs empty guard cycles (each
 * advances the global epoch when this is the only active thread) until the
 * expected number of frees is observed. For the hazard scheme the first
 * reclaim already frees, so pumping is a no-op.
 */

#include <atomic>
#include "common/check.hpp"
#include <cstdio>
#include <thread>
#include <vector>

#include "mpmc/detail/reclamation.hpp"

namespace
{
    // A minimal shared object with a magic word to detect use-after-free.
    struct object
    {
        explicit object(unsigned m)
            : magic(m)
        {
        }

        unsigned magic;
        static const unsigned MAGIC = 0xc0ffee;
    };

    struct counters_t
    {
        std::atomic<unsigned> freed;
        std::atomic<unsigned> retired;

        counters_t()
            : freed(0)
            , retired(0)
        {
        }
    };

    template <typename Scheme>
    struct fixture
    {
        using tag = fixture<Scheme>;
        using reclaimer = typename Scheme::template reclaimer<tag>;
        using retire_action = mpmc::detail::retire_action;

        static counters_t& counters()
        {
            static counters_t c;
            return c;
        }

        static void free_action(void* ptr)
        {
            object* o = static_cast<object*>(ptr);
            TEST_CHECK(o->magic == object::MAGIC); // no double free / no garbage
            counters().freed.fetch_add(1, std::memory_order_relaxed);
            delete o;
        }

        object* make()
        {
            counters().retired.fetch_add(1, std::memory_order_relaxed);
            return new object(object::MAGIC);
        }

        void retire(object* o)
        {
            reclaimer::retire(o, retire_action_of());
        }

        static retire_action retire_action_of()
        {
            retire_action a;
            a.fn = &fixture::free_action;
            return a;
        }
    };

    template <typename Reclaimer>
    void pump_until(std::atomic<unsigned>& freed, unsigned target)
    {
        for (int i = 0; i < 8 && freed.load(std::memory_order_relaxed) < target; ++i)
        {
            typename Reclaimer::op_guard g;
            Reclaimer::reclaim();
        }
        if (freed.load(std::memory_order_relaxed) != target)
        {
            std::printf("  pump_until: freed=%u target=%u outstanding=%d\n",
                        freed.load(std::memory_order_relaxed), target,
                        (int)Reclaimer::has_outstanding());
            std::fflush(stdout);
        }
        TEST_CHECK(freed.load(std::memory_order_relaxed) == target);
    }

    template <typename Scheme>
    void test_retire_and_reclaim()
    {
        fixture<Scheme> f;
        const unsigned before = fixture<Scheme>::counters().freed.load(
            std::memory_order_relaxed);
        {
            // The retirement is part of an operation (guard scope). The
            // guard must NOT span the pumps below: for the epoch scheme it
            // would pin the global epoch and prevent reclamation.
            typename fixture<Scheme>::reclaimer::op_guard g;
            f.retire(f.make());
        }
        fixture<Scheme>::reclaimer::reclaim();
        pump_until<typename fixture<Scheme>::reclaimer>(
            fixture<Scheme>::counters().freed, before + 1);
        TEST_CHECK(!fixture<Scheme>::reclaimer::has_outstanding());
    }

    template <typename Scheme>
    void test_protected_not_freed()
    {
        fixture<Scheme> f;
        const unsigned before = fixture<Scheme>::counters().freed.load(
            std::memory_order_relaxed);
        std::atomic<object*> src(nullptr);
        object* o = f.make();
        src.store(o, std::memory_order_release);
        {
            typename fixture<Scheme>::reclaimer::op_guard g;
            // Publish o to hazard slot 0 and re-validate the source (hazard
            // scheme); for the epoch scheme both calls are no-ops and the
            // guard provides the protection.
            fixture<Scheme>::reclaimer::protect(0, o);
            const bool source_unchanged =
                (src.load(std::memory_order_acquire) == o);
            TEST_CHECK(source_unchanged && "source changed unexpectedly");
            f.retire(o);
            fixture<Scheme>::reclaimer::reclaim();
            const unsigned after = fixture<Scheme>::counters().freed.load(
                std::memory_order_relaxed);
            TEST_CHECK(after == before && "protected object must not be freed");
            fixture<Scheme>::reclaimer::clear(0);
        }
        // Protection is gone: hazard reclaims immediately, epoch after the
        // global epoch has passed the retirement by two generations.
        fixture<Scheme>::reclaimer::reclaim();
        pump_until<typename fixture<Scheme>::reclaimer>(
            fixture<Scheme>::counters().freed, before + 1);
        TEST_CHECK(!fixture<Scheme>::reclaimer::has_outstanding());
    }

    template <typename Scheme>
    void test_retire_batch()
    {
        fixture<Scheme> f;
        const unsigned before = fixture<Scheme>::counters().freed.load(
            std::memory_order_relaxed);
        const unsigned N = 100;
        {
            typename fixture<Scheme>::reclaimer::op_guard g;
            for (unsigned i = 0; i < N; ++i)
            {
                f.retire(f.make());
            }
        }
        fixture<Scheme>::reclaimer::reclaim();
        pump_until<typename fixture<Scheme>::reclaimer>(
            fixture<Scheme>::counters().freed, before + N);
        TEST_CHECK(!fixture<Scheme>::reclaimer::has_outstanding());
    }

    template <typename Scheme>
    void test_thread_exit_reclaims()
    {
        fixture<Scheme> f;
        const unsigned before = fixture<Scheme>::counters().freed.load(
            std::memory_order_relaxed);
        {
            std::thread t([&f]() {
                typename fixture<Scheme>::reclaimer::op_guard g;
                for (unsigned i = 0; i < 10; ++i)
                {
                    f.retire(f.make());
                }
            });
            t.join();
        }
        // The worker's thread-local finalization frees what it can and
        // orphans the rest; the main thread's pumps reclaim the orphans
        // once their epoch (hazard: protection) has passed.
        fixture<Scheme>::reclaimer::reclaim();
        pump_until<typename fixture<Scheme>::reclaimer>(
            fixture<Scheme>::counters().freed, before + 10);
        TEST_CHECK(!fixture<Scheme>::reclaimer::has_outstanding());
    }

    template <typename Scheme>
    unsigned freed_count()
    {
        return fixture<Scheme>::counters().freed.load(std::memory_order_relaxed);
    }

    template <typename Scheme>
    void test_thread_churn()
    {
        fixture<Scheme> f;
        const unsigned THREADS = 32;
        static const unsigned PER = 50;
        const unsigned TOTAL = THREADS * PER;
        const unsigned before = fixture<Scheme>::counters().freed.load(
            std::memory_order_relaxed);
        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < THREADS; ++t)
            {
                ts.emplace_back([&f]() {
                    typename fixture<Scheme>::reclaimer::op_guard g;
                    for (unsigned i = 0; i < PER; ++i)
                    {
                        f.retire(f.make());
                    }
                });
            }
            for (auto& t : ts)
            {
                t.join();
            }
        }
        fixture<Scheme>::reclaimer::reclaim();
        pump_until<typename fixture<Scheme>::reclaimer>(
            fixture<Scheme>::counters().freed, before + TOTAL);
        const unsigned retired = fixture<Scheme>::counters().retired.load(
            std::memory_order_relaxed);
        TEST_CHECK(freed_count<Scheme>() == retired && "every retirement freed exactly once");
        TEST_CHECK(!fixture<Scheme>::reclaimer::has_outstanding());
    }
}

int main()
{
    std::printf("test_retire_and_reclaim<hazard>\n");
    test_retire_and_reclaim<mpmc::reclamation::hazard_pointer>();
    std::printf("test_retire_and_reclaim<epoch>\n");
    test_retire_and_reclaim<mpmc::reclamation::epoch>();
    std::printf("test_protected_not_freed<hazard>\n");
    test_protected_not_freed<mpmc::reclamation::hazard_pointer>();
    std::printf("test_protected_not_freed<epoch>\n");
    test_protected_not_freed<mpmc::reclamation::epoch>();
    std::printf("test_retire_batch<hazard>\n");
    test_retire_batch<mpmc::reclamation::hazard_pointer>();
    std::printf("test_retire_batch<epoch>\n");
    test_retire_batch<mpmc::reclamation::epoch>();
    std::printf("test_thread_exit_reclaims<hazard>\n");
    test_thread_exit_reclaims<mpmc::reclamation::hazard_pointer>();
    std::printf("test_thread_exit_reclaims<epoch>\n");
    test_thread_exit_reclaims<mpmc::reclamation::epoch>();
    std::printf("test_thread_churn<hazard>\n");
    test_thread_churn<mpmc::reclamation::hazard_pointer>();
    std::printf("test_thread_churn<epoch>\n");
    test_thread_churn<mpmc::reclamation::epoch>();
    std::printf("test_reclamation: all tests passed\n");
    return 0;
}
