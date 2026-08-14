#ifndef MPMC_HAZARD_POINTER_HPP
#define MPMC_HAZARD_POINTER_HPP

/**
 * @file hazard_pointer.hpp
 * @brief Hazard-pointer memory reclamation (Michael, 2004).
 *
 * Protocol (the same as in the reference mpmc_queue, generalized here):
 *
 *   1. A thread that intends to dereference a shared pointer publishes the
 *      pointer to one of its hazard slots BEFORE the dereference, then
 *      re-validates the source it read the pointer from. If the source
 *      changed, it retries; it never dereferences a pointer it did not
 *      re-validate.
 *   2. A thread retires an object only after it has unlinked it from every
 *      shared structure. The object is freed by a scan only when no hazard
 *      slot of any live thread publishes it.
 *
 * Because an object is never freed while published, use-after-free is
 * impossible; the publish-then-revalidate order closes the window where a
 * pointer is freed between the publisher's read and its publication.
 *
 * Memory ordering: hazard publications use release stores, scans use
 * acquire loads (a scan must observe the latest publications before it is
 * safe to free), hazard clears use relaxed stores, and the unlink CASes in
 * the containers use acq_rel.
 *
 * The registry is static per instantiation (the Tag parameter): one
 * registry is shared by every instance of the same container instantiation,
 * exactly like the reference mpmc_queue. It is heap-allocated once and
 * never freed (process lifetime), which removes the destruction-order
 * problem between thread_local states and static objects.
 *
 * A thread acquires its hazard slot on first use and keeps it for its
 * lifetime (thread_local). Threads beyond MPMC_MAX_THREADS spin until a
 * slot is released (a released slot implies its previous owner finished
 * all dereferences). Objects that are still protected when their retiring
 * thread exits are pushed to the shared late-retire chain; any later scan
 * (or any container destructor) drains that chain, so nothing leaks when
 * threads churn.
 *
 * Every retired object must be self-freeing: retire() stores a function
 * pointer supplied by the object's own code (a static member of the node
 * type that destroys the payload and returns the node to the allocator
 * instance that allocated it). This keeps reclamation correct across
 * container instances and instantiations.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <thread>
#include <vector>

#include "cache_line.hpp"

#ifndef MPMC_MAX_THREADS
#define MPMC_MAX_THREADS 128
#endif

#ifndef MPMC_HAZARD_POINTERS
#define MPMC_HAZARD_POINTERS 4
#endif

#ifndef MPMC_HAZARD_RETIRE_THRESHOLD
#define MPMC_HAZARD_RETIRE_THRESHOLD (2 * MPMC_MAX_THREADS * MPMC_HAZARD_POINTERS)
#endif

namespace mpmc
{
    namespace detail
    {
        /**
         * @brief Self-freeing deletion action for a retired object.
         *
         * fn(ptr) must destroy and release the object exactly once. The
         * action is type-erased so that objects of different container
         * instantiations (and different allocator instances) can share one
         * registry.
         */
        struct retire_action
        {
            void (*fn)(void* ptr);

            /**
             * @brief Invoke the stored free action.
             *
             * @param ptr the object to free (must have been unlinked
             *            from its container)
             */
            void run(void* ptr) const
            {
                fn(ptr);
            }
        };

        /**
         * @brief The hazard-pointer reclaimer; Tag identifies one
         *        (container instantiation, reclamation scheme) pair.
         *
         * All state is static; the container calls the static members and
         * addresses its hazard slots 0..MPMC_HAZARD_POINTERS-1.
         */
        template <typename Tag>
        class hazard_pointer_reclaimer
        {
        public:
            hazard_pointer_reclaimer() = delete;

            /**
             * @brief The scheme needs load-re-validation in traversals:
             *        a pointer is published to a hazard slot and the source
             *        is re-read to rule out ABA before it is dereferenced.
             */
            static const bool NEEDS_REVALIDATION = true;

            /**
             * @brief No-op per-operation guard.
             *
             * Hazard pointers protect per dereference, so an operation
             * needs no epoch-style bracketing; the type exists so that the
             * container code is identical for both reclamation schemes.
             * The user-provided (empty) constructor suppresses
             * unused-variable warnings under -Wall.
             */
            struct op_guard
            {
                op_guard() {}
                ~op_guard() {}
            };

            /**
             * @brief Per-thread state, finalized at thread exit.
             *
             * The guard member runs finalize_thread() when the thread exits:
             * unprotected retirements are freed, protected ones move to the
             * late-retire chain, the hazard slot is cleared and released.
             */
            struct thread_state
            {
                /**
                 * @brief Runs finalize_thread() at thread exit.
                 *
                 * Part of the thread_local teardown: frees unprotected
                 * retirements, moves protected ones to the shared
                 * late-retire chain, releases the hazard slot.
                 */
                struct guard
                {
                    ~guard()
                    {
                        hazard_pointer_reclaimer::finalize_thread();
                    }
                };

                std::int32_t slot; // -1 until first use
                std::vector<void*> retired;
                std::vector<retire_action> retired_actions;
                guard g;

                /**
                 * @brief Zero-initialize the per-thread state.
                 *
                 * slot starts -1 (no hazard slot acquired yet); the
                 * retirement vectors are empty. A hazard slot is
                 * acquired lazily on the first protect().
                 */
                thread_state()
                    : slot(-1)
                {
                }
            };

            static thread_local thread_state tls_;

            // ---- Static registry (per instantiation, process lifetime) ----
            // Heap-allocated once, never freed: the thread_local guard may
            // run after every container instance is gone and must still find
            // the registry intact. Alignment: C++11 ::operator new
            // guarantees only max_align_t; the slot array must be
            // cache-line aligned so that adjacent slots do not share lines
            // (false sharing).
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
            struct alignas(MPMC_CACHE_LINE) hazard_slot
            {
                std::atomic<std::uint32_t> owner; // 0 == free
                std::atomic<void*> hp[MPMC_HAZARD_POINTERS];

                hazard_slot()
                    : owner(0)
                {
                    for (unsigned i = 0; i < MPMC_HAZARD_POINTERS; ++i)
                    {
                        hp[i].store(nullptr, std::memory_order_relaxed);
                    }
                }
            };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

            /**
             * @brief Compact registry of the slots owned by live threads.
             *
             * is_protected() must answer "does any live thread publish X?"
             * for every retired object; scanning all 128 slots per object is
             * the dominant reclamation cost, so only the slots of live
             * threads are scanned (slots[0..count)).
             *
             * Registration: reserve the entry index first (fetch_add returns
             * a unique k), then publish the entry with a relaxed store. A
             * scanner that observes the new count may read slots[k] before
             * the store lands; a stale value is always conservative (-1, an
             * index of another still-owned slot, or a released slot).
             *
             * Unregistration: move the last entry into the removed slot's
             * position, then publish the decreased count (both relaxed
             * stores happen before the count drop). A scanner holding the
             * old count may read stale entries that resolve to a
             * currently-owned slot (checked against its live hazards) or to
             * a released slot (skipped). It can never miss a hazard that is
             * actually in use, so premature reclamation is impossible.
             */
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
            struct alignas(MPMC_CACHE_LINE) active_registry
            {
                std::atomic<unsigned> count;
                std::atomic<std::int32_t> slots[MPMC_MAX_THREADS];

                /**
                 * @brief Construct an empty registry (no active slots).
                 *
                 * count = 0; every slot entry is -1 (the released
                 * marker) so a scanner can never mistake an empty
                 * entry for a live hazard.
                 */
                active_registry()
                    : count(0)
                {
                    for (unsigned i = 0; i < MPMC_MAX_THREADS; ++i)
                    {
                        slots[i].store(-1, std::memory_order_relaxed);
                    }
                }
            };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

            /**
             * @brief Node of the late-retire chain.
             *
             * Allocated only on the thread-exit path (when an object is
             * still protected at thread exit), never on the data path.
             */
            struct orphan
            {
                orphan* next;
                void* ptr;
                retire_action action;
            };

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
            struct alignas(MPMC_CACHE_LINE) late_retire_holder
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
             * @brief The shared hazard-slot array (one per thread, up to
             *        MPMC_MAX_THREADS).
             *
             * Cache-line aligned so threads never false-share slot
             * lines. One-time aligned allocation inside the function-
             * local static initializer.
             *
             * @return pointer to the first slot (never null)
             */
            static hazard_slot* slots()
            {
                // One-time construction inside the static initializer; the
                // loop must NOT run on every call.
                static hazard_slot* const s = []() {
                    hazard_slot* p = static_cast<hazard_slot*>(
                        aligned_alloc_raw(sizeof(hazard_slot) * MPMC_MAX_THREADS));
                    for (unsigned i = 0; i < MPMC_MAX_THREADS; ++i)
                    {
                        ::new (static_cast<void*>(&p[i])) hazard_slot();
                    }
                    return p;
                }();
                return s;
            }

            /**
             * @brief The shared registry of slots in use.
             *
             * @return pointer to the singleton registry (never null)
             */
            static active_registry* active()
            {
                static active_registry* const r = []() {
                    active_registry* p = static_cast<active_registry*>(
                        aligned_alloc_raw(sizeof(active_registry)));
                    ::new (static_cast<void*>(p)) active_registry();
                    return p;
                }();
                return r;
            }

            /**
             * @brief The shared late-retire chain.
             *
             * Holds retirements pushed by threads that exited while some
             * of their objects were still protected; any later scan and
             * the container destructor drain it.
             *
             * @return pointer to the singleton chain holder
             */
            static late_retire_holder* late()
            {
                static late_retire_holder* const h = []() {
                    late_retire_holder* p = static_cast<late_retire_holder*>(
                        aligned_alloc_raw(sizeof(late_retire_holder)));
                    ::new (static_cast<void*>(p)) late_retire_holder();
                    return p;
                }();
                return h;
            }

            /**
             * @brief Cache-line-aligned raw allocation.
             *
             * Over-allocates by one cache line and aligns the returned
             * pointer, so the object never shares a line with
             * neighbouring allocations.
             *
             * @param size the object size (alignment padding added)
             * @return an aligned pointer (never null; allocation
             *         failure throws bad_alloc)
             */
            static void* aligned_alloc_raw(std::size_t size)
            {
                const std::size_t align = MPMC_CACHE_LINE;
                unsigned char* base =
                    static_cast<unsigned char*>(::operator new(size + align));
                const std::uintptr_t addr =
                    reinterpret_cast<std::uintptr_t>(base);
                const std::size_t off = (align - (addr % align)) % align;
                return base + off;
            }

            /**
             * @brief This thread's hazard slot; acquired on first use.
             *
             * If every slot is busy (more concurrent threads than
             * MPMC_MAX_THREADS), spins until one is released -- a documented
             * capacity bound, not a failure.
             */
            static hazard_slot* slot()
            {
                thread_state& ts = tls_;
                if (ts.slot < 0)
                {
                    const std::uint32_t self = thread_owner_id();
                    for (;;)
                    {
                        for (unsigned i = 0; i < MPMC_MAX_THREADS; ++i)
                        {
                            std::uint32_t expected = 0;
                            if (slots()[i].owner.compare_exchange_strong(
                                    expected, self, std::memory_order_acq_rel))
                            {
                                ts.slot = static_cast<std::int32_t>(i);
                                register_active(i);
                                return &slots()[i];
                            }
                        }
                        std::this_thread::yield();
                    }
                }
                return &slots()[ts.slot];
            }

            /**
             * @brief Publish `p` to hazard slot `idx`.
             *
             * Release: a concurrent scan (acquire load) must observe "this
             * thread now protects p" before the thread dereferences p. The
             * caller must re-validate the source it read `p` from.
             */
            static void protect(unsigned idx, void* p)
            {
                slot()->hp[idx].store(p, std::memory_order_release);
            }

            /**
             * @brief Clear hazard slot `idx`.
             *
             * Relaxed suffices (only affects scan outcomes).
             */
            static void clear(unsigned idx)
            {
                slot()->hp[idx].store(nullptr, std::memory_order_relaxed);
            }

            /**
             * @brief Retire an object; a scan runs when the local list
             *        exceeds the threshold.
             *
             * The object must already be unlinked from every shared
             * structure. Memory exhaustion is not recoverable, but the two
             * vectors are kept in lockstep by construction: the pointer is
             * pushed first, and if the action push throws, the pointer is
             * popped again -- an allocation failure leaks the object but
             * never orphanes it from its action (which would crash the
             * scan). No reserve is used: reserve(size + 1) at size ==
             * capacity reallocates to exactly size + 1, making EVERY push
             * O(n) -- when a scan cannot free anything (every hazard slot
             * occupied), the list grows without bound and the O(n^2) copy
             * work livelocks the thread inside the reallocation loop. Plain
             * push_back has amortized O(1) growth.
             */
            static void retire(void* p, retire_action action)
            {
                thread_state& ts = tls_;
                outstanding().fetch_add(1, std::memory_order_relaxed);
                try
                {
                    ts.retired.push_back(p);
                    ts.retired_actions.push_back(action);
                }
                catch (...)
                {
                    // push_back of a pointer is strong-guarantee: a throw
                    // leaves the vector unchanged. If the action push threw,
                    // the pointer push succeeded and must be undone so the
                    // two vectors stay index-aligned (scan reads both by
                    // index).
                    if (ts.retired.size() > ts.retired_actions.size())
                    {
                        ts.retired.pop_back();
                    }
                    outstanding().fetch_sub(1, std::memory_order_acq_rel);
                    throw;
                }
                if (ts.retired.size() > MPMC_HAZARD_RETIRE_THRESHOLD)
                {
                    scan();
                }
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
             * @brief Free every retirement of this thread that is not
             *        protected by any live thread's hazard slot.
             *
             * Delegates to scan(); exists so the uniform reclaimer
             * interface (retire/reclaim/drain_all) matches the epoch
             * scheme. Called when the per-thread retired list exceeds
             * MPMC_HAZARD_RETIRE_THRESHOLD and from finalize_thread.
             */
            static void reclaim()
            {
                scan();
            }

            /**
             * @brief Free this thread's unprotected retirements and drain
             *        the shared late-retire chain.
             *
             * First the late chain is drained (objects no longer
             * protected by any live slot are freed; the still-protected
             * ones are re-pushed one at a time -- cycle-free by
             * construction). Then this thread's own retired list is
             * compacted in place: protected entries move to the front,
             * unprotected ones are freed (their actions run) and the
             * outstanding counter is decremented. The list is resized to
             * the kept count, so memory stays bounded by the protected
             * set. O(list + chain) per call.
             */
            static void scan()
            {
                drain_late_retire();
                thread_state& ts = tls_;
                std::size_t kept = 0;
                const std::size_t n = ts.retired.size();
                for (std::size_t i = 0; i < n; ++i)
                {
                    void* cur = ts.retired[i];
                    if (is_protected(cur))
                    {
                        ts.retired[kept] = cur;
                        ts.retired_actions[kept] = ts.retired_actions[i];
                        ++kept;
                    }
                    else
                    {
                        ts.retired_actions[i].run(cur);
                        outstanding().fetch_sub(1, std::memory_order_acq_rel);
                    }
                }
                ts.retired.resize(kept);
                ts.retired_actions.resize(kept);
            }

            /**
             * @brief Free everything pending in this thread's list and on
             *        the late-retire chain.
             *
             * Container destruction contract: no thread may be inside an
             * operation, and every user thread has been joined, so its
             * thread_local state has already run finalize_thread(). Under
             * that contract this frees every retirement of the
             * instantiation.
             */
            static void drain_all()
            {
                drain_late_retire();
                thread_state& ts = tls_;
                for (std::size_t i = 0; i < ts.retired.size(); ++i)
                {
                    ts.retired_actions[i].run(ts.retired[i]);
                    outstanding().fetch_sub(1, std::memory_order_acq_rel);
                }
                ts.retired.clear();
                ts.retired_actions.clear();
            }

        private:
            static std::atomic<std::size_t>& outstanding()
            {
                static std::atomic<std::size_t> n(0);
                return n;
            }

            static std::uint32_t thread_owner_id()
            {
                static std::atomic<std::uint32_t> next(0);
                // Owner ids never reuse 0 (the free marker).
                return next.fetch_add(1, std::memory_order_relaxed) + 1;
            }

            static void register_active(std::int32_t slot_idx)
            {
                active_registry* r = active();
                const unsigned k = r->count.fetch_add(1, std::memory_order_acq_rel);
                r->slots[k].store(slot_idx, std::memory_order_relaxed);
            }

            static void unregister_active(std::int32_t slot_idx)
            {
                active_registry* r = active();
                const unsigned n = r->count.load(std::memory_order_acquire);
                std::int32_t k = -1;
                for (unsigned j = 0; j < n; ++j)
                {
                    if (r->slots[j].load(std::memory_order_acquire) == slot_idx)
                    {
                        k = static_cast<std::int32_t>(j);
                        break;
                    }
                }
                if (k < 0)
                {
                    return;
                }
                const std::int32_t last =
                    r->slots[n - 1].load(std::memory_order_relaxed);
                r->slots[k].store(last, std::memory_order_relaxed);
                r->slots[n - 1].store(-1, std::memory_order_relaxed);
                r->count.fetch_sub(1, std::memory_order_acq_rel);
            }

            /**
             * @brief Thread-exit finalization: free what is safe, push
             *        the rest onto the late-retire chain, release the
             *        hazard slot.
             *
             * Runs once per thread (from thread_state's guard as part of
             * the thread_local teardown). First scan() frees everything
             * no longer protected; the survivors (still protected by
             * other live threads) become orphans pushed onto the shared
             * late-retire chain one at a time (per-element head CAS,
             * cycle-free) so they outlive this thread's state. Finally
             * the hazard slot is cleared (owner = 0, release) and
             * unregistered from the active registry. A thread may have
             * retired objects without ever acquiring a slot, hence the
             * slot check comes last.
             */
            static void finalize_thread()
            {
                thread_state& ts = tls_;
                // Free what is safe; move the rest to the late-retire chain
                // so they outlive this thread (the thread_local state is
                // about to be destroyed). A thread may retire objects
                // without ever acquiring a hazard slot, so the slot check
                // comes after the reclamation.
                scan();
                if (!ts.retired.empty())
                {
                    orphan* head = nullptr;
                    for (std::size_t i = 0; i < ts.retired.size(); ++i)
                    {
                        orphan* o = new orphan();
                        o->next = head;
                        o->ptr = ts.retired[i];
                        o->action = ts.retired_actions[i];
                        head = o;
                    }
                    ts.retired.clear();
                    ts.retired_actions.clear();
                    orphan* cur = head;
                    while (cur != nullptr)
                    {
                        orphan* nx = cur->next;
                        orphan* expected = late()->head.load(std::memory_order_relaxed);
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
                if (ts.slot >= 0)
                {
                    slots()[ts.slot].owner.store(0, std::memory_order_release);
                    unregister_active(ts.slot);
                    ts.slot = -1;
                }
            }

            /**
             * @brief Whether any live thread protects `cur`.
             *
             * Scans the active-slot registry: every registered slot's
             * owner is checked (a released owner means its publications
             * are meaningless) and each of its MPMC_HAZARD_POINTERS
             * slots is compared against `cur`. Stale registry reads are
             * conservative (they can only over-report protection, never
             * under-report a live hazard), so premature reclamation is
             * impossible.
             *
             * @param cur the retired object pointer to test
             * @return true when some live hazard slot currently holds it
             */
            static bool is_protected(void* cur)
            {
                active_registry* r = active();
                const unsigned n = r->count.load(std::memory_order_acquire);
                for (unsigned k = 0; k < n; ++k)
                {
                    const std::int32_t idx =
                        r->slots[k].load(std::memory_order_acquire);
                    if (idx < 0)
                    {
                        continue; // released concurrently
                    }
                    const hazard_slot& s = slots()[idx];
                    if (s.owner.load(std::memory_order_acquire) == 0)
                    {
                        continue; // released: its publications are meaningless
                    }
                    for (unsigned h = 0; h < MPMC_HAZARD_POINTERS; ++h)
                    {
                        if (s.hp[h].load(std::memory_order_acquire) == cur)
                        {
                            return true;
                        }
                    }
                }
                return false;
            }

            /**
             * @brief Drain the shared late-retire chain.
             *
             * Exchanges the whole chain out (one global RMW), walks it:
             * orphans no longer protected by any live slot are freed
             * (action runs, outstanding decremented, orphan deleted);
             * still-protected orphans are re-pushed to the head one at a
             * time (per-element CAS, cycle-free -- the chain tail is
             * null-terminated first because it still points into the
             * freed prefix). Called from scan() and drain_all(); the
             * exchange is skipped implicitly by callers when the chain
             * is known empty via a relaxed head load (scan path) to
             * avoid hammering the shared line.
             */
            static void drain_late_retire()
            {
                orphan* head = late()->head.exchange(nullptr,
                                                     std::memory_order_acq_rel);
                orphan* kept_head = nullptr;
                orphan* kept_tail = nullptr;
                while (head != nullptr)
                {
                    orphan* nx = head->next;
                    if (is_protected(head->ptr))
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
                    else
                    {
                        head->action.run(head->ptr);
                        outstanding().fetch_sub(1, std::memory_order_acq_rel);
                        delete head;
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
                        orphan* expected = late()->head.load(std::memory_order_relaxed);
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
        };

        // C++11: out-of-class definition of the thread_local static member.
        template <typename Tag>
        thread_local typename hazard_pointer_reclaimer<Tag>::thread_state
            hazard_pointer_reclaimer<Tag>::tls_;
    }
}

#endif // MPMC_HAZARD_POINTER_HPP

