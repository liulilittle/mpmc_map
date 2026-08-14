#ifndef MPMC_EPOCH_HPP
#define MPMC_EPOCH_HPP

/**
 * @file epoch.hpp
 * @brief Epoch-based memory reclamation (Fraser, 2004; Harris 2001-style
 *        global epoch with per-thread retire lists).
 *
 * Scheme:
 *
 *   * The global epoch counter takes three values 0..2. A thread enters an
 *     epoch at the start of every operation (guard RAII) and leaves it at
 *     the end. enter() records the current global epoch and increments the
 *     per-epoch active count; exit() decrements it.
 *   * The global epoch advances (CAS, best effort) only when the active
 *     count of the current epoch is zero -- i.e. the last thread of that
 *     epoch left. Hence, once the epoch has advanced past an epoch, its
 *     active count stays zero forever (no thread ever enters an epoch other
 *     than the current one), and every thread that was active in it has
 *     finished the operation in which it could dereference anything.
 *   * A thread retires an object with a tag equal to its current epoch; an
 *     object retired in epoch e is freed only when the global epoch has
 *     passed e (global == e+2 mod 3), which proves that no thread can still
 *     be active in e, and therefore no thread can still dereference the
 *     object (the object was unlinked before retirement, so pointers to it
 *     were read only by threads active in epochs <= e).
 *
 * The ABA case on the 3-value counter is harmless: advancing is advisory
 * (it only enables frees of older epochs), and a stale CAS that matches a
 * re-wrapped value advances an epoch whose frees are still guarded by the
 * two-generation rule.
 *
 * Memory ordering: global loads are acquire, the advance CAS is acq_rel,
 * active-count increments are relaxed, decrements are release (the advance
 * check must observe the last leaver's decrement).
 *
 * Objects retired by a thread that exits before its retirements are
 * reclaimable move to the shared late-retire chain (tagged); any later
 * reclaim in any thread, and every container destructor, drains it, so
 * nothing leaks when threads churn.
 */

#include <atomic>
#include <cstddef>
#include <new>
#include <vector>

#include "hazard_pointer.hpp" // retire_action

#ifndef MPMC_EPOCH_RECLAIM_THRESHOLD
#define MPMC_EPOCH_RECLAIM_THRESHOLD 256
#endif

namespace mpmc
{
    namespace detail
    {
        /**
         * @brief The epoch reclaimer; Tag identifies one (container
         *        instantiation, reclamation scheme) pair.
         */
        template <typename Tag>
        class epoch_reclaimer
        {
        public:
            epoch_reclaimer() = delete;

            /**
             * @brief The scheme needs no load-re-validation in traversals:
             *        the per-operation guard already proves no concurrent
             *        operation can dereference a retired object, so a stale
             *        read is safe (the CAS/scan layers handle staleness).
             */
            static const bool NEEDS_REVALIDATION = false;

            static const unsigned EPOCHS = 3;

            /**
             * @brief RAII guard: enter the current epoch on construction,
             *        leave it on destruction.
             *
             * Every container operation runs inside a guard; while a thread
             * is inside a guard it may dereference any pointer it read
             * during this operation, and retired objects stay alive until
             * no thread can be in the epoch they were retired from.
             */
            class guard
            {
            public:
                /**
                 * @brief Enter the current global epoch.
                 *
                 * Records the epoch (a load of the global counter) and
                 * increments its active counter so the epoch cannot
                 * advance while this guard is alive. Every container
                 * operation holds one guard for its whole duration.
                 */
                guard()
                {
                    enter();
                }

                /**
                 * @brief Leave the epoch entered at construction.
                 *
                 * Decrements the active counter; if this was the last
                 * active thread of the epoch, tries to advance the global
                 * epoch (making two-generation-old retirements
                 * reclaimable). 1:1 pairing with the constructor's
                 * enter() -- exactly one exit per guard.
                 */
                ~guard()
                {
                    exit();
                }

                guard(const guard&) = delete;
                guard& operator=(const guard&) = delete;
            };

            struct thread_state
            {
                /**
                 * @brief Runs finalize_thread() when the owning thread
                 *        exits.
                 *
                 * A member of thread_state so that the thread's
                 * retirements are finalized (reclaimable items freed,
                 * the rest pushed to the shared late-retire chain) as
                 * part of the thread_local teardown, before the vectors
                 * themselves are destroyed.
                 */
                struct exit_guard
                {
                    ~exit_guard()
                    {
                        epoch_reclaimer::finalize_thread();
                    }
                };

                unsigned local_epoch;
                unsigned last_drain_epoch;
                std::size_t retired_total;
                std::vector<void*> retired[EPOCHS];
                std::vector<retire_action> retired_actions[EPOCHS];
                exit_guard g;

                /**
                 * @brief Zero-initialize the per-thread reclamation state.
                 *
                 * local_epoch starts at 0 (the first enter() overwrites
                 * it); last_drain_epoch starts at ~0u so the first
                 * reclaim() drains once even if the epoch is still 0
                 * (harmless: the chain is empty at startup).
                 * retired_total starts at 0; all three tag vectors are
                 * empty.
                 */
                thread_state()
                    : local_epoch(0)
                    , last_drain_epoch(~0u)
                    , retired_total(0)
                {
                }
            };

            static thread_local thread_state tls_;

            /**
             * @brief Retire an object (unlinked already); tagged with the
             *        current epoch; a reclaim runs when the local list
             *        exceeds the threshold.
             *
             * Memory exhaustion is not recoverable, but the two vectors are
             * kept in lockstep by construction: the pointer is pushed first,
             * and if the action push throws, the pointer is popped again --
             * an allocation failure leaks the object but never orphanes it
             * from its action (which would crash free_tagged). No reserve
             * is used: reserve(size + 1) at size == capacity reallocates to
             * exactly size + 1, making EVERY push O(n) -- under a long-lived
             * guard the epoch cannot advance, every retire lands in one tag
             * vector while reclaim() only frees the other tags, and the
             * O(n^2) copy work livelocks the thread inside the reallocation
             * loop. Plain push_back has amortized O(1) growth.
             */
            static void retire(void* p, retire_action action)
            {
                thread_state& ts = tls_;
                outstanding().fetch_add(1, std::memory_order_relaxed);
                const unsigned e = ts.local_epoch % EPOCHS;
                try
                {
                    ts.retired[e].push_back(p);
                    ts.retired_actions[e].push_back(action);
                }
                catch (...)
                {
                    // push_back of a pointer is strong-guarantee: a throw
                    // leaves the vector unchanged. If the action push threw,
                    // the pointer push succeeded and must be undone so the
                    // two vectors stay index-aligned (free_tagged reads both
                    // by index).
                    if (ts.retired[e].size() > ts.retired_actions[e].size())
                    {
                        ts.retired[e].pop_back();
                    }
                    outstanding().fetch_sub(1, std::memory_order_acq_rel);
                    throw;
                }
                if (++ts.retired_total > MPMC_EPOCH_RECLAIM_THRESHOLD)
                {
                    reclaim();
                }
            }

            /**
             * @brief Per-operation guard alias (uniform scheme interface).
             */
            using op_guard = guard;

            /**
             * @brief No-op in the epoch scheme: the per-operation guard
             *        already brackets every dereference. Exists so that the
             *        container code is identical for both reclamation
             *        schemes.
             */
            static void protect(unsigned, void*)
            {
            }

            /**
             * @brief No-op in the epoch scheme (see protect()).
             */
            static void clear(unsigned)
            {
            }

            /**
             * @brief True when no retirement is pending anywhere in this
             *        instantiation (used by leak assertions after all user
             *        threads have joined).
             */
            static bool has_outstanding()
            {
                return outstanding().load(std::memory_order_acquire) != 0;
            }

            /**
             * @brief Free this thread's retirements tagged two epochs before
             *        the current global epoch (the only tags that are proven
             *        safe), and drain a bounded number of late-retire items.
             */
            static void reclaim()
            {
                thread_state& ts = tls_;
                const unsigned g = global().load(std::memory_order_acquire);
                const unsigned tag = (g + 1) % EPOCHS; // g - 2 (mod 3)
                free_tagged(ts, tag);
                // Drain the shared late-retire chain only when it may hold
                // newly-reclaimable items. Two conditions must hold:
                //   (1) the chain is non-empty -- the exchange inside
                //       drain_late is a global locked RMW on a cache line
                //       shared by every thread, and under sustained
                //       multi-thread activity reclaim() runs on nearly every
                //       retire, so an empty-chain exchange would serialize
                //       all threads on that one line for nothing (a relaxed
                //       head load is free: read-only line, no contention);
                //   (2) the global epoch has advanced since this thread last
                //       drained -- drain_late only frees orphans whose tag
                //       matches (g+1)%3; when the epoch is frozen, re-draining
                //       just walks and re-pushes the same chain, O(chain) per
                //       retire for nothing. Deferring the drain to the next
                //       epoch advance is safe: orphaned retirements stay on
                //       the chain until their tag becomes reclaimable, and
                //       drain_late_all (destruction) frees everything.
                if (g != ts.last_drain_epoch &&
                    late()->head.load(std::memory_order_relaxed) != nullptr)
                {
                    drain_late(tag);
                    ts.last_drain_epoch = g;
                }
                ts.retired_total = 0;
                for (unsigned i = 0; i < EPOCHS; ++i)
                {
                    ts.retired_total += ts.retired[i].size();
                }
            }

            /**
             * @brief Free everything (container destruction contract: no
             *        thread may be inside an operation, and every user
             *        thread has been joined, so its thread_local state has
             *        already run finalize_thread()).
             */
            static void drain_all()
            {
                thread_state& ts = tls_;
                for (unsigned e = 0; e < EPOCHS; ++e)
                {
                    free_tagged(ts, e);
                }
                ts.retired_total = 0;
                drain_late_all();
            }

        private:
            /**
             * @brief The global 3-value epoch counter.
             *
             * Advanced by the last thread leaving an epoch (see exit()).
             * Acquire loads: the counter's release-store in exit() (the
             * CAS) publishes the epoch advance.
             */
            static std::atomic<unsigned>& global()
            {
                static std::atomic<unsigned> g(0);
                return g;
            }

            /**
             * @brief The active-thread counter of one epoch.
             *
             * Three slots (one per epoch value). enter() increments the
             * current epoch's slot; exit() decrements and, seeing 0,
             * attempts the global advance. Relaxed increments (the
             * acquire load of the epoch in enter() orders the
             * increment); release decrement publishes the guard's reads
             * to the advancing thread.
             */
            static std::atomic<unsigned>& active(unsigned e)
            {
                static std::atomic<unsigned> a[EPOCHS];
                return a[e % EPOCHS];
            }

            /**
             * @brief Global count of objects retired but not yet freed.
             *
             * Incremented in retire(), decremented wherever an object is
             * actually freed (free_tagged, drain_late, drain_all).
             * Used by has_outstanding() for leak assertions after all
             * user threads have joined. Relaxed increments; acq_rel
             * decrements so the assertion sees every free.
             */
            static std::atomic<std::size_t>& outstanding()
            {
                static std::atomic<std::size_t> n(0);
                return n;
            }

            /**
             * @brief Enter the current global epoch (guard entry).
             *
             * Load the epoch (acquire: pairs with the advancing CAS's
             * release), bump its active counter (relaxed -- the acquire
             * load already ordered us), and remember the epoch for the
             * matching exit(). Every container operation calls this
             * exactly once per op_guard.
             */
            static void enter()
            {
                thread_state& ts = tls_;
                const unsigned g = global().load(std::memory_order_acquire);
                active(g).fetch_add(1, std::memory_order_relaxed);
                ts.local_epoch = g;
            }

            /**
             * @brief Leave the epoch entered by enter() (guard exit).
             *
             * Decrement the active counter with release ordering (this
             * publishes everything the guard's operation read, so a
             * thread that later advances the epoch knows no reader of
             * that generation is still active). If this was the last
             * active thread, load the global and CAS it forward if it
             * has not advanced already -- the single place the epoch
             * moves. Exactly one exit per enter (1:1 pairing).
             */
            static void exit()
            {
                thread_state& ts = tls_;
                const unsigned local = ts.local_epoch;
                if (active(local).fetch_sub(1, std::memory_order_release) == 1)
                {
                    // I was the last thread of the current epoch: try to
                    // advance so that older retirements become reclaimable.
                    unsigned g = global().load(std::memory_order_acquire);
                    if (g == local)
                    {
                        global().compare_exchange_strong(
                            g, (g + 1) % EPOCHS, std::memory_order_acq_rel);
                    }
                }
            }

            /**
             * @brief Free every retirement of one tag of one thread.
             *
             * Runs each retire_action (which self-frees the object,
             * typically returning it to the node pool), decrements the
             * outstanding counter, and clears the tag's vectors. Called
             * by reclaim() with the tag two epochs old -- the only tag
             * proven safe (no guard of that generation can still be
             * active) -- and by drain_all/finalize_thread for every tag.
             *
             * @param ts  the thread whose tag vectors to free
             * @param tag the tag index (0..EPOCHS-1)
             */
            static void free_tagged(thread_state& ts, unsigned tag)
            {
                std::vector<void*>& ptrs = ts.retired[tag];
                std::vector<retire_action>& acts = ts.retired_actions[tag];
                ts.retired_total -= ptrs.size();
                for (std::size_t i = 0; i < ptrs.size(); ++i)
                {
                    acts[i].run(ptrs[i]);
                    outstanding().fetch_sub(1, std::memory_order_acq_rel);
                }
                ptrs.clear();
                acts.clear();
            }

            // ---- Late-retire chain (objects retired by exited threads) ----
            struct orphan
            {
                orphan* next;
                void* ptr;
                retire_action action;
                unsigned tag;
            };

            #if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
            /**
             * @brief Cache-line-aligned holder for the late-retire chain
             *        head.
             *
             * Alignment keeps the head atomic on its own cache line so
             * exchange/CAS traffic does not false-share with unrelated
             * data.
             */
            struct alignas(64) late_retire_holder
            {
                std::atomic<orphan*> head;

                /**
                 * @brief Construct an empty chain (head = null).
                 */
                late_retire_holder()
                    : head(nullptr)
                {
                }
            };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

            /**
             * @brief The shared late-retire chain (heap-allocated,
             *        cache-line aligned).
             *
             * The chain stores retirements pushed by threads that exited
             * while some of their objects were not yet reclaimable; any
             * later reclaim() and every container destructor drain it.
             * All pushes are per-element head CASes (cycle-free by
             * construction -- a whole-chain push would race and could
             * cross tail links into a cycle).
             *
             * @return pointer to the singleton holder (never null after
             *         first call; the lambda initializer runs once,
             *         thread-safe via the function-local static)
             */
            static late_retire_holder* late()
            {
                static late_retire_holder* const h = []() {
                    const std::size_t align = MPMC_CACHE_LINE;
                    void* raw = ::operator new(sizeof(late_retire_holder) + align);
                    unsigned char* base = static_cast<unsigned char*>(raw);
                    const std::uintptr_t addr =
                        reinterpret_cast<std::uintptr_t>(base);
                    const std::size_t off = (align - (addr % align)) % align;
                    late_retire_holder* p =
                        ::new (static_cast<void*>(base + off))
                            late_retire_holder();
                    return p;
                }();
                return h;
            }

            /**
             * @brief Thread-exit finalization: free what is safe, push
             *        the rest onto the late-retire chain.
             *
             * Runs once per thread (from thread_state's exit_guard, as
             * part of the thread_local teardown). First reclaim() frees
             * this thread's two-generations-old retirements; the
             * survivors are moved to the shared chain one orphan at a
             * time (per-element head CAS, cycle-free), tagged with the
             * tag they were retired into, so any thread's later
             * reclaim() -- and the container destructor's drain_late_all
             * -- can free them once their generation has passed. After
             * this, the thread's vectors are empty and safe to destroy.
             */
            static void finalize_thread()
            {
                thread_state& ts = tls_;
                reclaim();
                // Move the not-yet-reclaimable retirements to the shared
                // chain, tagged, so they can be reclaimed by any thread
                // (including the container destructor) once their epoch has
                // passed.
                orphan* head = nullptr;
                for (unsigned e = 0; e < EPOCHS; ++e)
                {
                    for (std::size_t i = 0; i < ts.retired[e].size(); ++i)
                    {
                        orphan* o = new orphan();
                        o->next = head;
                        o->ptr = ts.retired[e][i];
                        o->action = ts.retired_actions[e][i];
                        o->tag = e;
                        head = o;
                    }
                    ts.retired[e].clear();
                    ts.retired_actions[e].clear();
                }
                ts.retired_total = 0;
                if (head != nullptr)
                {
                    // Push the local chain onto the late chain one orphan
                    // at a time (a single-node Treiber push per element):
                    // a whole-chain push is either tail-anchored (two
                    // pushers can cross their tail links and form a cycle,
                    // because the CAS validates only its own expected
                    // value) or head-anchored (it overwrites the chain's
                    // second link and orphans the rest). Each node here is
                    // exclusively ours, so the per-node push is cycle-free.
                    orphan* cur = head;
                    while (cur != nullptr)
                    {
                        orphan* nx = cur->next;
                        orphan* expected =
                            late()->head.load(std::memory_order_relaxed);
                        for (;;)
                        {
                            cur->next = expected;
                            if (late()->head.compare_exchange_weak(
                                    expected, cur, std::memory_order_acq_rel))
                            {
                                break;
                            }
                        }
                        cur = nx;
                    }
                }
            }

            /**
             * @brief Free late-retire items whose tag is reclaimable under
             *        the current global epoch; keep the rest on the chain.
             */
            static void drain_late(unsigned current_tag)
            {
                orphan* head = late()->head.exchange(nullptr,
                                                     std::memory_order_acq_rel);
                orphan* kept_head = nullptr;
                orphan* kept_tail = nullptr;
                while (head != nullptr)
                {
                    orphan* nx = head->next;
                    if (head->tag == current_tag)
                    {
                        head->action.run(head->ptr);
                        outstanding().fetch_sub(1, std::memory_order_acq_rel);
                        delete head;
                    }
                    else
                    {
                        if (kept_tail == nullptr)
                        {
                            kept_head = head;
                        }
                        else
                        {
                            kept_tail->next = head;
                        }
                        kept_tail = head;
                    }
                    head = nx;
                }
                if (kept_head != nullptr)
                {
                    // Re-push the kept chain one orphan at a time (same
                    // cycle-free argument as finalize_thread). Terminate
                    // the chain first: its current tail still points into
                    // the freed part of the original chain.
                    kept_tail->next = nullptr;
                    orphan* cur = kept_head;
                    while (cur != nullptr)
                    {
                        orphan* nx = cur->next;
                        orphan* expected =
                            late()->head.load(std::memory_order_relaxed);
                        for (;;)
                        {
                            cur->next = expected;
                            if (late()->head.compare_exchange_weak(
                                    expected, cur, std::memory_order_acq_rel))
                            {
                                break;
                            }
                        }
                        cur = nx;
                    }
                }
            }

            /**
             * @brief Free every late-retire item (destruction contract).
             */
            static void drain_late_all()
            {
                for (;;)
                {
                    orphan* head = late()->head.exchange(nullptr,
                                                         std::memory_order_acq_rel);
                    if (head == nullptr)
                    {
                        return;
                    }
                    while (head != nullptr)
                    {
                        orphan* nx = head->next;
                        head->action.run(head->ptr);
                        outstanding().fetch_sub(1, std::memory_order_acq_rel);
                        delete head;
                        head = nx;
                    }
                }
            }
        };

        // C++11: out-of-class definition of the thread_local static member.
        template <typename Tag>
        thread_local typename epoch_reclaimer<Tag>::thread_state
            epoch_reclaimer<Tag>::tls_;
    }
}

#endif // MPMC_EPOCH_HPP
