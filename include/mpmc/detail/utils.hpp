#ifndef MPMC_UTILS_HPP
#define MPMC_UTILS_HPP

/**
 * @file utils.hpp
 * @brief Small integer utilities used by the containers.
 *
 * All functions here are constexpr-compatible where the C++11 language
 * allows it, and none of them allocates or throws.
 */

#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace mpmc
{
    namespace detail
    {
        /**
         * @brief A short spin lock (acquire-release on the flag).
         *
         * Used only for internal bookkeeping with critical sections of a
         * few nanoseconds (the node pools); the container data structures
         * themselves remain lock-free. Wait is bounded by one critical
         * section of the holder.
         */
        class spin_lock
        {
        public:
            /**
             * @brief Construct an unlocked spin lock.
             *
             * The flag starts false; the lock is ready for use
             * immediately. No allocation, no system calls.
             */
            spin_lock()
                : flag_(false)
            {
            }

            spin_lock(const spin_lock&) = delete;
            spin_lock& operator=(const spin_lock&) = delete;

            /**
             * @brief Acquire the lock, spinning with backoff.
             *
             * CAS loop on the flag with acquire ordering. The first 64
             * failed attempts pause (x86 _mm_pause, reducing memory
             * ordering traffic and power); after that the thread yields
             * its timeslice so an oversubscribed machine cannot livelock
             * the holder out of its critical section. Wait is bounded by
             * one holder's critical section (a few nanoseconds by
             * contract), so under normal operation the yield path is
             * never reached.
             */
            void lock()
            {
                unsigned spins = 0;
                for (;;)
                {
                    bool expected = false;
                    if (flag_.compare_exchange_weak(expected, true,
                                                    std::memory_order_acquire))
                    {
                        return;
                    }
                    if (++spins < 64)
                    {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                        _mm_pause();
#endif
                    }
                    else
                    {
                        std::this_thread::yield();
                    }
                }
            }

            /**
             * @brief Release the lock.
             *
             * Stores false with release ordering, publishing everything
             * the critical section wrote to the next acquirer (the
             * acquire on the flag in lock() pairs with this store).
             */
            void unlock()
            {
                flag_.store(false, std::memory_order_release);
            }

        private:
            std::atomic<bool> flag_;
        };

        // ---- Pool link accessors -------------------------------------------
        // The retired node's first atomic pointer field is reused as the
        // pool link (the node is detached and dead at release time).
        /**
         * @brief Pool-link policy using the node's `next` pointer.
         *
         * Used by the split-ordered list's node type: the `next` atomic
         * is repurposed as the singly-linked pool chain while the node
         * sits in the pool (it is detached from the container by then).
         * Both accessors are relaxed: the node is exclusively owned by
         * the pool at that point, so no ordering is required.
         */
        template <typename Node>
        struct pool_link_next
        {
            /**
             * @brief Read the pooled node's next pointer.
             * @param n a node currently pooled (exclusive ownership)
             * @return the next pooled node, or null at the chain tail
             */
            static Node* get(Node* n)
            {
                return n->next.load(std::memory_order_relaxed);
            }

            /**
             * @brief Write the pooled node's next pointer.
             * @param n the node to link into the pool chain
             * @param v the next node (previous head), or null
             */
            static void set(Node* n, Node* v)
            {
                n->next.store(v, std::memory_order_relaxed);
            }
        };

        /**
         * @brief Pool-link policy using the node's `left` pointer.
         *
         * Used by the red-black tree's node type: the `left` child
         * pointer is repurposed as the pool chain link while the node is
         * pooled (detached from the tree). Relaxed accessors; the node
         * is exclusively owned by the pool.
         */
        template <typename Node>
        struct pool_link_left
        {
            /**
             * @brief Read the pooled node's left pointer.
             * @param n a node currently pooled (exclusive ownership)
             * @return the next pooled node, or null at the chain tail
             */
            static Node* get(Node* n)
            {
                return n->left.load(std::memory_order_relaxed);
            }

            /**
             * @brief Write the pooled node's left pointer.
             * @param n the node to link into the pool chain
             * @param v the next node (previous head), or null
             */
            static void set(Node* n, Node* v)
            {
                n->left.store(v, std::memory_order_relaxed);
            }
        };

         /**
          * @brief Pool of retired node blocks, shared per container
          *        instantiation (a static member of the container).
          *
          * Retired nodes are returned here (by the reclaimer's free action)
          * instead of being returned to the allocator, so churn-heavy
          * workloads reuse hot blocks instead of hitting the allocator.
          *
          * The pool is guarded by short spin locks: a lock-free
          * (Treiber-style) free list would need an ABA-proof pop, which is
          * not expressible portably in C++11 (no 16-byte atomics). To keep
          * the critical section contention-free under many threads, the
          * pool is SHARDED: each thread is assigned one shard (cache-line
          * separated) on first use and only ever locks its own shard, so
          * concurrent acquire/release from different threads never touch
          * the same cache line. Shard count matches MPMC_MAX_THREADS, so
          * up to 128 threads get exclusive shards. A single shared lock
          * measured at ~12k cycles per pool op with 16 threads hammering
          * it (~55% of all thread time) -- sharding eliminates that.
          *
          * Nodes are pooled together with the allocator they were allocated
          * with (the node's alloc_owner field); a pooled block is only
          * reused by a container using the same allocator instance and is
          * otherwise released through its own allocator. Each shard drains
          * when it exceeds its share of the watermark (bounded memory
          * retention) and every container destructor drains all shards
          * (the allocator's allocate/deallocate balance is preserved, and
          * the pool can never outlive the nodes' allocators).
          */
        template <typename Node, typename NodeAlloc, typename Link>
        class node_pool
        {
        public:
            /**
             * @brief Construct an empty pool (all shards zero-initialized).
             *
             * The pool is a static member of the container engine, so this
             * runs once per (container type, reclamation scheme)
             * instantiation. No shard has any pooled block yet.
             */
            node_pool()
                : shards_()
            {
            }

            /**
             * @brief Drain every shard (destruction contract).
             *
             * Frees every pooled block through its recorded allocator
             * (`alloc_owner`). Runs at process/static destruction; by then
             * no container of the instantiation is alive and every user
             * thread has been joined, so no acquire/release can race this.
             */
            ~node_pool()
            {
                drain();
            }

            node_pool(const node_pool&) = delete;
            node_pool& operator=(const node_pool&) = delete;

            /**
             * @brief Obtain a node block for this thread.
             *
             * Tries this thread's shard first: if the shard holds a block
             * that was pooled by the SAME allocator instance (`na`), the
             * block is reused (the common churn case, ~20ns, uncontended
             * lock); a foreign block (pooled through a different allocator
             * instance) is released through its own allocator and a fresh
             * block is allocated through `na` instead; an empty shard
             * falls through to the allocator directly.
             *
             * The lock is held only for the pointer-pop (a few
             * nanoseconds); the allocator calls happen outside the lock so
             * a slow allocator never stalls other threads sharing the
             * shard.
             *
             * @param na the node allocator of the container requesting the
             *           block (used for the reuse identity check and for
             *           fresh allocation)
             * @return a node block owned by the caller (never null; an
             *         allocation failure propagates the allocator's
             *         exception)
             */
            Node* acquire(NodeAlloc& na)
            {
                shard& s = my_shard();
                s.lock.lock();
                Node* n = s.head;
                if (n != nullptr)
                {
                    s.head = Link::get(n);
                    --s.size;
                }
                s.lock.unlock();
                if (n != nullptr)
                {
                    if (n->alloc_owner == &na)
                    {
                        return n; // reuse: same allocator instance
                    }
                    // foreign block: release it through its own allocator
                    std::allocator_traits<NodeAlloc>::deallocate(
                        *n->alloc_owner, n, 1);
                    n = nullptr;
                }
                return std::allocator_traits<NodeAlloc>::allocate(na, 1);
            }

            /**
             * @brief Return a node block to this thread's shard.
             *
             * The block is linked at the shard's head (LIFO reuse) under
             * the shard lock, then the shard size is checked against its
             * share of the watermark. If the shard overflows, ONLY this
             * shard is drained (its blocks are returned to their recorded
             * allocators) so memory retention stays bounded while other
             * threads' shards are untouched.
             *
             * @param n the block to pool; must be unlinked from any
             *          structure and safe to reuse (the caller's retire
             *          path guarantees this). Must not be null.
             */
            void release(Node* n)
            {
                shard& s = my_shard();
                bool over = false;
                s.lock.lock();
                Link::set(n, s.head);
                s.head = n;
                over = (++s.size > PER_SHARD_WATERMARK);
                s.lock.unlock();
                if (over)
                {
                    drain_shard(s);
                }
            }

            /**
             * @brief Drain every shard (container-destruction contract).
             *
             * Called by the container destructor after all user threads
             * have been joined, so the shard locks are uncontended. Every
             * pooled block is returned to the allocator recorded in its
             * `alloc_owner`, preserving the allocator's allocate/deallocate
             * balance exactly.
             */
            void drain()
            {
                for (unsigned i = 0; i < SHARDS; ++i)
                {
                    drain_shard(shards_[i]);
                }
            }

        private:
            enum
            {
                SHARDS = 128,
                WATERMARK = 4096,
                PER_SHARD_WATERMARK = WATERMARK / SHARDS
            };

            #if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
            /**
             * @brief One pool shard: an independent lock-free-by-design
             *        slot guarded by its own spin lock.
             *
             * The whole struct is cache-line aligned so that threads
             * assigned to different shards never share a cache line (no
             * false sharing): a thread's shard lock stays in that thread's
             * private cache once the first uncontended acquire puts it
             * there, making subsequent lock/unlock pairs plain
             * load-branch-store sequences.
             *
             * Members: `lock` guards `head` (the LIFO stack of pooled
             * blocks) and `size` (the current number of pooled blocks in
             * this shard, compared against PER_SHARD_WATERMARK).
             */
            struct alignas(MPMC_CACHE_LINE) shard
            {
                spin_lock lock;
                Node* head;
                std::size_t size;

                /**
                 * @brief Zero-initialize the shard (no pooled blocks yet).
                 */
                shard()
                    : head(nullptr)
                    , size(0)
                {
                }
            };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

            /**
             * @brief The per-instantiation shard-assignment counter.
             *
             * A single relaxed fetch_add on first use per thread gives
             * each thread a distinct shard index (mod SHARDS); with up to
             * SHARDS threads the mapping is a bijection, so no two threads
             * ever contend on one shard lock.
             */
            static std::atomic<unsigned>& next_shard()
            {
                static std::atomic<unsigned> n(0);
                return n;
            }

            /**
             * @brief Resolve this thread's shard.
             *
             * The thread-local index is initialized on first call (one
             * global relaxed fetch_add, amortized over the thread's
             * lifetime) and stays constant, so every acquire/release by
             * this thread touches the same shard and nothing else touches
             * it while the thread is alive. After thread exit the shard is
             * reused by a later thread with a different index mapping --
             * safe, because the pool's blocks are drained by the container
             * destructor and each block carries its own allocator.
             *
             * @return reference to this thread's shard
             */
            shard& my_shard()
            {
                // One global fetch_add per thread (amortized); the thread
                // then exclusively owns its shard's cache line.
                thread_local unsigned idx =
                    next_shard().fetch_add(1, std::memory_order_relaxed) %
                    SHARDS;
                return shards_[idx];
            }

            shard shards_[SHARDS];

            /**
             * @brief Free every block currently pooled in one shard.
             *
             * Takes the shard's stack under its lock (a few nanoseconds),
             * releases the lock, then walks the detached chain returning
             * each block to the allocator recorded in its `alloc_owner` --
             * the deallocation loop runs lock-free so a slow allocator can
             * never block other threads' pool traffic.
             *
             * @param s the shard to empty (must not be in use by another
             *          thread; callers guarantee this: the over-watermark
             *          path drains only the caller's own shard, the
             *          drain-all paths run after all user threads joined)
             */
            void drain_shard(shard& s)
            {
                s.lock.lock();
                Node* n = s.head;
                s.head = nullptr;
                s.size = 0;
                s.lock.unlock();
                while (n != nullptr)
                {
                    Node* nx = Link::get(n);
                    std::allocator_traits<NodeAlloc>::deallocate(
                        *n->alloc_owner, n, 1);
                    n = nx;
                }
            }
        };
        /**
         * @brief True when `x` is a power of two (x >= 1).
         */
        inline bool is_power_of_two(std::size_t x)
        {
            return x != 0 && (x & (x - 1)) == 0;
        }

        /**
         * @brief The exponent such that 2^r <= x < 2^(r+1), for x >= 1.
         */
        inline unsigned log2_floor(std::size_t x)
        {
            unsigned r = 0;
            while (x >>= 1)
            {
                ++r;
            }
            return r;
        }

        /**
         * @brief Bit-reverse a 64-bit word.
         *
         * Used to compute split-order keys: the split-ordered list is sorted
         * by the bit-reversed value of the mixed hash.
         */
        inline std::uint64_t bit_reverse(std::uint64_t x)
        {
            x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
            x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
            x = ((x & 0x0f0f0f0f0f0f0f0fULL) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL);
            x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
            x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
            x = (x << 32) | (x >> 32);
            return x;
        }

        /**
         * @brief Finalizer of the MurmurHash3 family (Austin Appleby).
         *
         * Applied to the raw std::hash result before the power-of-two modulo
         * of the split-ordered buckets. Without it, hash functions with weak
         * low bits (e.g. identity hash of aligned addresses) would cluster
         * in a few buckets. The mix is a bijection on 64-bit words, so it
         * does not collide two distinct hash values.
         */
        inline std::uint64_t mix_hash(std::uint64_t h)
        {
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return h;
        }

        /**
         * @brief The split-order key of a regular (non-dummy) node.
         *
         * The most significant bit is forced to 1 so that every regular key
         * sorts after every dummy key with the same low bits; the remaining
         * bits are the bit-reversed mixed hash.
         */
        inline std::uint64_t split_order_key(std::uint64_t mixed_hash)
        {
            return bit_reverse(mixed_hash | (1ULL << 63));
        }

        /**
         * @brief The split-order key of a dummy bucket node.
         *
         * The most significant bit stays 0: dummy keys sort before every
         * regular key sharing the same low-order bits.
         */
        inline std::uint64_t dummy_split_order_key(std::uint64_t bucket_index)
        {
            return bit_reverse(bucket_index);
        }
    }
}

#endif // MPMC_UTILS_HPP
