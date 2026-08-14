#ifndef MPMC_SPLIT_ORDERED_TABLE_HPP
#define MPMC_SPLIT_ORDERED_TABLE_HPP

/**
 * @file split_ordered_table.hpp
 * @brief Lock-free extensible hash table: the split-ordered list
 *        (Shalev & Shavit, PODC 2003 / JACM 2006).
 *
 * All entries live in ONE lock-free sorted list, ordered by the
 * split-order key: the bit-reversed mixed hash with the MSB forced on
 * (regular keys are odd) and the bit-reversed bucket index for dummies
 * (dummy keys are even). The bucket array is only an index of lazily
 * created dummy nodes; growing the table means directing new bucket
 * pointers into the list, never moving items, so resize is fully
 * lock-free.
 *
 * Operations (all linearizable):
 *
 *   find(k)   -- protect the bucket table, follow the bucket's dummy, walk
 *               the list while the split-order key is smaller, then compare
 *               keys (KeyEqual) inside the run of nodes with the same
 *               split-order key (hash collisions form such runs).
 *   insert(k) -- locate the position and CAS the new node in (linearization
 *               point). A new node of an existing split-order key is
 *               inserted at the head of its run.
 *   erase(k)  -- locate, CAS the "removed" tag into the node's next pointer
 *               (linearization point), then unlink with a second CAS
 *               (helped by any other thread), then retire the node.
 *
 * Memory safety: every dereferenced pointer is published to a hazard slot
 * and re-validated against its source (store-then-reload); every unlink CAS
 * compares a node that is concurrently published, which rules out address
 * reuse (ABA). With the epoch scheme the per-operation guard brackets the
 * whole operation and protect/clear are no-ops.
 *
 * Tag bits in next pointers: bit 0 marks a bucket (dummy) node, bit 1
 * marks a logically removed node. Dummy nodes are never removed.
 *
 * This header is the engine for mpmc::unordered_map and
 * mpmc::unordered_set; the public headers add the std-compatible API.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "cache_line.hpp"
#include "reclamation.hpp"
#include "utils.hpp"

namespace mpmc
{
    namespace detail
    {
        // ---- Tag bits of next pointers --------------------------------
        /**
         * Tag bits stored in the low two bits of every `next` pointer.
         *
         * BUCKET (bit 0) marks the pointer as targeting a bucket dummy
         * node (used only during grow to publish cells); MARKED (bit 1)
         * marks the node as logically removed -- erasers set it before
         * unlinking, and no insert may follow a marked node (it would be
         * orphaned when the eraser detaches it). All pointers read from
         * the list must be cleared before dereference; the raw tagged
         * value is kept for CAS expected values and marked checks.
         */
        namespace tags
        {
            const std::uintptr_t BUCKET = 1; // next targets a dummy node
            const std::uintptr_t MARKED = 2; // node logically removed
            const std::uintptr_t MASK = 3;

            /**
             * @brief Clear both tag bits of a pointer.
             *
             * @tparam T the pointee type (any node type)
             * @param p a raw tagged pointer (possibly with BUCKET/MARKED)
             * @return the untagged pointer, safe to dereference; null if
             *         p was a pure tag (0x0 | BUCKET etc.)
             */
            template <typename T>
            inline T* clear(T* p)
            {
                return reinterpret_cast<T*>(
                    reinterpret_cast<std::uintptr_t>(p) & ~MASK);
            }

            /**
             * @brief Test the MARKED bit.
             *
             * @tparam T the pointee type
             * @param p a raw tagged pointer
             * @return true when the node is logically removed (marked)
             */
            template <typename T>
            inline bool is_marked(T* p)
            {
                return (reinterpret_cast<std::uintptr_t>(p) & MARKED) != 0;
            }

            /**
             * @brief Set the MARKED bit (logical removal).
             *
             * @tparam T the pointee type
             * @param p a raw tagged pointer
             * @return p with the MARKED bit set (BUCKET bit untouched)
             */
            template <typename T>
            inline T* set_marked(T* p)
            {
                return reinterpret_cast<T*>(
                    reinterpret_cast<std::uintptr_t>(p) | MARKED);
            }

            /**
             * @brief Clear only the MARKED bit.
             *
             * Used by helper-unlink CASes: a marked node's successor is
             * CAS'd over it with the mark cleared. The BUCKET bit (if
             * any) is preserved.
             *
             * @tparam T the pointee type
             * @param p a raw tagged pointer
             * @return p with the MARKED bit cleared
             */
            template <typename T>
            inline T* clear_mark(T* p)
            {
                return reinterpret_cast<T*>(
                    reinterpret_cast<std::uintptr_t>(p) & ~MARKED);
            }
        }

        // ---- Payload ------------------------------------------------------
        // map flavor: value_type = pair<const K, V>; set flavor: K.
        template <typename Key, typename Value>
        struct payload_holder
        {
            using type = std::pair<const Key, Value>;
        };

        template <typename Key>
        struct payload_holder<Key, void>
        {
            using type = Key;
        };

        /**
         * @brief Lock-free extensible hash table engine.
         *
         * Template parameters:
         *   Key        -- key type
         *   Value      -- mapped type, or void for the set flavor
         *   Hash       -- hash functor (must not throw)
         *   KeyEqual   -- equality functor (must not throw)
         *   Allocator  -- allocator of value_type (rebound for nodes)
         *   Reclamation-- reclamation scheme (hazard_pointer or epoch)
         */
        template <typename Key, typename Value, typename Hash,
                  typename KeyEqual, typename Allocator, typename Reclamation>
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
        class split_ordered_table
        {
        public:
            using self = split_ordered_table;
            using reclaimer = typename Reclamation::template reclaimer<self>;
            using value_type = typename payload_holder<Key, Value>::type;
            using allocator_type = Allocator;

            // ---- Allocators ------------------------------------------------
            struct node;
            using node_alloc = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<node>;

            struct bucket_table;
            using table_alloc = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<bucket_table>;
            using cell_alloc = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<std::atomic<node*>>;

            // ---- Node ------------------------------------------------------
            /**
             * @brief A split-ordered list node.
             *
             * `next` carries the tag bits (see tags); `so_key` is the
             * split-order key (bit-reversed mixed hash for regular nodes,
             * bit-reversed bucket index for dummies) that defines the
             * list order; `payload` holds the key/value (or key for
             * sets); `alloc_owner` records the allocator instance the
             * node was allocated with so the pool can return it to the
             * right allocator.
             */
            struct node
            {
                std::atomic<node*> next; // tag bits: see tags
                std::uint64_t so_key;
                value_type payload;
                node_alloc* alloc_owner;

                /**
                 * @brief Construct an unlinked node (next = null).
                 *
                 * The payload is NOT constructed here -- the caller
                 * placement-news it with the actual key/value; the empty
                 * payload is only used for bucket dummies (created via
                 * alloc_node + direct field init).
                 */
                node()
                    : next(nullptr)
                    , so_key(0)
                    , alloc_owner(nullptr)
                {
                }

                /**
                 * @brief The node's immutable key (payload first member).
                 *
                 * The key is set at construction and never modified, so
                 * reads are safe without synchronization (relaxed loads
                 * in the walk paths rely on this immutability).
                 *
                 * @return const reference to the key
                 */
                const Key& key() const
                {
                    return key_of_payload(payload);
                }
            };

            using node_pool = detail::node_pool<node, node_alloc,
                                                 detail::pool_link_next<node>>;

            /**
             * @brief Reclaimer free action for regular nodes.
             *
             * Destroys the payload (mapped value destructor runs here)
             * and returns the block to the node pool. Runs on the
             * reclaimer's thread (the eraser, the exiting thread's
             * finalize, or the container destructor) -- never inside a
             * concurrent operation.
             *
             * @param ptr the node pointer being reclaimed
             */
            static void free_node_action(void* ptr)
            {
                node* n = static_cast<node*>(ptr);
                n->payload.~value_type();
                pool_.release(n);
            }

            // ---- Bucket table ----------------------------------------------
            /**
             * @brief The bucket indirection table (doubling).
             *
             * `size` is a power of two; `buckets` is a heap array of
             * size cells of atomic node pointers. A cell either points
             * at the bucket's dummy node or is null (bucket not yet
             * materialized -- the grow path publishes an all-null upper
             * half and materializes buckets lazily on first touch).
             * The allocators are stored so that stateful (non-default-
             * constructible) allocator types work.
             */
            struct bucket_table
            {
                std::size_t size;               // power of two, >= 2
                std::atomic<node*>* buckets;    // cells; null = uninitialized
                table_alloc self_alloc;
                cell_alloc cell_alloc_;

                // The allocators are passed in so that stateful allocator
                // types (no default constructor) work.
                /**
                 * @brief Construct an empty table (no cells yet).
                 *
                 * @param ta the allocator for the table struct itself
                 * @param ca the allocator for the bucket cell array
                 */
                bucket_table(const table_alloc& ta, const cell_alloc& ca)
                    : size(0)
                    , buckets(nullptr)
                    , self_alloc(ta)
                    , cell_alloc_(ca)
                {
                }

                /**
                 * @brief Reclaimer free action for bucket tables.
                 *
                 * Deallocates the cell array and the table struct
                 * through their recorded allocators. Runs only after no
                 * thread can reference the table anymore (the reclaimer
                 * guarantees this).
                 *
                 * @param ptr the bucket_table pointer being reclaimed
                 */
                static void free_table(void* ptr)
                {
                    bucket_table* t = static_cast<bucket_table*>(ptr);
                    std::allocator_traits<cell_alloc>::deallocate(
                        t->cell_alloc_, t->buckets, t->size);
                    std::allocator_traits<table_alloc>::deallocate(
                        t->self_alloc, t, 1);
                }
            };

            // ---- Hazard slot usage -----------------------------------------
            enum
            {
                HP_TABLE = 0,
                HP_PREV = 1,
                HP_CUR = 2
            };

            // ---- Construction / destruction --------------------------------
            explicit split_ordered_table(const allocator_type& a,
                                         const Hash& h, const KeyEqual& e,
                                         float max_load_factor)
                : na_(a)
                , hash_(h)
                , eq_(e)
                 , count_(0)
                 , growing_(false)
                , max_load_factor_(max_load_factor)
                , inv_max_load_factor_(max_load_factor > 0.0f
                                       ? 1.0f / max_load_factor : 1.0f)
                , cached_threshold_(0)
            {
                table_.store(make_table(2), std::memory_order_release);
                // bucket 0's dummy is the list head: create it eagerly.
                node* head = alloc_node();
                head->so_key = dummy_split_order_key(0);
                bucket_table* t = table_.load(std::memory_order_acquire);
                t->buckets[0].store(head, std::memory_order_relaxed);
            }

            ~split_ordered_table()
            {
                // Destruction contract: no thread may be inside an
                // operation and every user thread has been joined (see the
                // public container docs).
                // Order matters: walk the list FIRST (every linked node is
                // still alive), then drain the reclaimer (unlinked nodes
                // and old tables), then release the current table, then
                // drain the node pool (the pool is shared per
                // instantiation, so this releases pooled blocks that may
                // belong to other containers of the same type; the block's
                // own allocator is used, so the balance is preserved).
                bucket_table* t = table_.load(std::memory_order_acquire);
                node* cur = t->buckets[0].load(std::memory_order_acquire);
                while (cur != nullptr)
                {
                    node* nx = cur->next.load(std::memory_order_acquire);
                    node* real = tags::clear(nx);
                    if (!tags::is_marked(nx))
                    {
                        free_node_action(cur);
                    }
                    cur = real;
                }
                reclaimer::drain_all();
                bucket_table::free_table(t);
                pool_.drain();
            }

            split_ordered_table(const split_ordered_table&) = delete;
            split_ordered_table& operator=(const split_ordered_table&) = delete;

            // ---- Public engine operations ----------------------------------
            /**
             * @brief The current number of elements.
             *
             * A monotonically correct count via the shared atomic; every
             * successful insert/erase updates it exactly once.
             *
             * @return element count (0 for an empty table)
             */
            std::size_t size() const
            {
                return count_.load(std::memory_order_acquire);
            }

            /**
             * @brief Whether the table has no elements.
             *
             * @return true when size() == 0
             */
            bool empty() const
            {
                return size() == 0;
            }

            /**
             * @brief The configured maximum load factor.
             *
             * Growth triggers when count exceeds size * max_load_factor.
             *
             * @return the stored max load factor
             */
            float max_load_factor() const
            {
                return max_load_factor_;
            }

            /**
             * @brief The current bucket count (power of two).
             *
             * @return the current table's bucket count (>= 2)
             */
            std::size_t bucket_count() const
            {
                return table_.load(std::memory_order_acquire)->size;
            }

            /**
             * @brief Pre-grow until the bucket count supports `n` elements
             *        at the configured load factor.
             *
             * Grows by forced grow_once calls; concurrent growth is
             * serialized by the growing_ flag, so reserve is
             * multi-thread safe. Useful before write-heavy phases to
             * avoid growth stalls (the grow path is fully lock-free but
             * the table copy is O(bucket count)).
             *
             * @param n the number of elements to support
             */
            void reserve(std::size_t n)
            {
                const std::size_t target =
                    static_cast<std::size_t>(
                        static_cast<double>(n) * inv_max_load_factor_ + 0.999999);
                while (bucket_count() < target)
                {
                    grow_once(true);
                }
            }

        public:
            // ---- Key / split-order helpers --------------------------------
            /**
             * @brief Mix the user hash into a uniform 64-bit value.
             *
             * The mix spreads the raw hash so bucket selection and
             * split-order key generation behave uniformly.
             *
             * @param k the key
             * @return the mixed hash value
             */
            std::uint64_t mixed_hash(const Key& k) const
            {
                return mix_hash(static_cast<std::uint64_t>(hash_(k)));
            }

            /**
             * @brief The split-order key of a regular node.
             *
             * Bit-reverses the mixed hash with the top bit forced on
             * (odd so_key: sorts AFTER every dummy, whose so_key is
             * even/bit-reversed-bucket). Key is immutable after
             * construction.
             *
             * @param mixed the mixed hash of the key
             * @return the node's so_key
             */
            std::uint64_t so_key_of(std::uint64_t mixed) const
            {
                return bit_reverse(mixed | (1ULL << 63));
            }

            /**
             * @brief Key equality via the configured predicate.
             *
             * @param a first key
             * @param b second key
             * @return true when a and b compare equal
             */
            bool key_equal(const Key& a, const Key& b) const
            {
                return eq_(a, b);
            }

            /**
             * @brief The configured hash function (observers).
             *
             * @return const reference to the stored hasher
             */
            const Hash& hash_function() const
            {
                return hash_;
            }

            /**
             * @brief The configured key-equality predicate (observers).
             *
             * @return const reference to the stored equality functor
             */
            const KeyEqual& key_eq() const
            {
                return eq_;
            }

            /**
             * @brief Change the max load factor.
             *
             * Affects future growth decisions (grow triggers when count
             * exceeds size * f). A non-positive f is clamped to a
             * no-grow-effective 1.0 inverse. Note: this mutates shared
             * state without synchronization -- call it before concurrent
             * use or while no other thread is operating.
             *
             * @param f the new max load factor
             */
            void set_max_load_factor(float f)
            {
                max_load_factor_ = f;
                inv_max_load_factor_ = f > 0.0f ? 1.0f / f : 1.0f;
            }

            /**
             * @brief Extract the key from a map-flavor payload.
             *
             * @tparam K key type
             * @tparam V mapped type
             * @param p the pair payload
             * @return const reference to p.first (the immutable key)
             */
            template <typename K, typename V>
            static const K& key_of_payload(const std::pair<const K, V>& p)
            {
                return p.first;
            }

            /**
             * @brief Identity key access for a set-flavor payload.
             *
             * @param k the key payload
             * @return const reference to k itself
             */
            static const Key& key_of_payload(const Key& k)
            {
                return k;
            }

            // ---- Allocation ------------------------------------------------
            /**
             * @brief Obtain a node block, constructed and allocator-tagged.
             *
             * Pops from the (sharded) node pool or falls through to the
             * allocator; placement-news a node with an EMPTY payload (the
             * caller fills the payload with the real key/value) and
             * records the allocator instance for later pooled reuse.
             *
             * @return a fresh node (never null)
             */
            node* alloc_node() const
            {
                node* n = pool_.acquire(na_);
                ::new (static_cast<void*>(n)) node();
                n->alloc_owner = &na_;
                return n;
            }

            /**
             * @brief Allocate and initialize a bucket table with `size`
             *        all-null cells.
             *
             * Exception-safe: if the cell array allocation throws, the
             * table struct is deallocated and the exception propagates
             * (no leak).
             *
             * @param size the (power-of-two) bucket count
             * @return the new table (cells all null)
             */
            bucket_table* make_table(std::size_t size)
            {
                table_alloc ta = na_;
                cell_alloc ca = na_;
                bucket_table* t =
                    std::allocator_traits<table_alloc>::allocate(ta, 1);
                ::new (static_cast<void*>(t)) bucket_table(ta, ca);
                t->size = size;
                try
                {
                    t->buckets = std::allocator_traits<cell_alloc>::allocate(
                        ca, size);
                }
                catch (...)
                {
                    std::allocator_traits<table_alloc>::deallocate(ta, t, 1);
                    throw;
                }
                for (std::size_t i = 0; i < size; ++i)
                {
                    t->buckets[i].store(nullptr, std::memory_order_relaxed);
                }
                return t;
            }

            /**
             * @brief Deallocate a bucket table and its cells (synchronous
             *        path, used by the destructor).
             *
             * @param t the table to free (must be unreferenced by all
             *          threads)
             */
            static void destroy_table(bucket_table* t)
            {
                std::allocator_traits<cell_alloc>::deallocate(
                    t->cell_alloc_, t->buckets, t->size);
                std::allocator_traits<table_alloc>::deallocate(
                    t->self_alloc, t, 1);
            }

            /**
             * @brief Reclaimer free action for retired tables.
             *
             * @param ptr the bucket_table pointer being reclaimed
             */
            static void retire_table_action(void* ptr)
            {
                bucket_table::free_table(ptr);
            }

            // ---- Protection helpers ----------------------------------------
            // Publishes the (cleared) pointer read from `src` and, for
            // schemes that need it, re-validates the source; returns the
            // cleared node and the tagged value. The re-validation guards
            // against ABA for hazard pointers; the epoch scheme skips it
            // (the per-operation guard already proves the node stays
            // alive, and the CAS/scan layers handle stale reads).
            static node* protect_from(std::atomic<node*>& src, unsigned slot,
                                      node*& tagged_out)
            {
                if (!reclaimer::NEEDS_REVALIDATION)
                {
                    node* p = src.load(std::memory_order_acquire);
                    tagged_out = p;
                    return tags::clear(p);
                }
                for (;;)
                {
                    node* p = src.load(std::memory_order_acquire);
                    node* c = tags::clear(p);
                    reclaimer::protect(slot, c);
                    if (src.load(std::memory_order_acquire) == p)
                    {
                        tagged_out = p;
                        return c;
                    }
                }
            }

            // ---- The core walk ----------------------------------------------
            // Walks the list from the protected dummy of the bucket of
            // `mixed` to the first node with so_key >= cmp_key (or null).
            // Marked nodes are unlinked on the way (helping). Returns
            // (prev, cur), both published in their hazard slots; prev is
            // never marked, cur may be marked.
            void locate(std::uint64_t mixed, std::uint64_t cmp_key,
                        node*& prev_out, node*& cur_out) const
            {
                for (;;)
                {
                    bucket_table* t = table_.load(std::memory_order_acquire);
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_TABLE, t);
                        if (table_.load(std::memory_order_acquire) != t)
                        {
                            continue;
                        }
                    }
                    const std::size_t b =
                        static_cast<std::size_t>(mixed) & (t->size - 1);
                    node* d = t->buckets[b].load(std::memory_order_acquire);
                    if (d == nullptr)
                    {
                        initialize_bucket(t, b);
                        continue;
                    }
                    node* prev = d; // dummy: never removed
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_PREV, prev);
                    }
                    node* tagged = nullptr;
                    node* cur = protect_from(prev->next, HP_CUR, tagged);
                    for (;;)
                    {
                        if (cur == nullptr || cur->so_key >= cmp_key)
                        {
                            prev_out = prev;
                            cur_out = cur;
                            return;
                        }
                        // Single acquire load of cur->next: the cleared
                        // pointer is used for prefetch address and marked
                        // check; the raw tagged value for CAS expected.
                        // (O-series: previously two loads -- relaxed for
                        // prefetch address + acquire for the value; on x86
                        // both are plain `mov`, so the relaxed load saved
                        // nothing and cost an extra L1 hit.)
                        node* raw_next = cur->next.load(
                            std::memory_order_acquire);
                        node* nx = tags::clear(raw_next);
                        // Prefetch the next node's cache line while we
                        // process cur. The walk is a dependent load chain
                        // (cur->next -> next->next), so starting the fetch
                        // early overlaps the L1/L2 miss with the CAS or
                        // protect_from work below.
                        if (nx != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(nx),
                                         _MM_HINT_T0);
#endif
                        }
                        if (tags::is_marked(raw_next))
                        {
                            // help unlink the marked node (the marker owns
                            // its reclamation)
                            node* expected = tagged;
                            node* replacement = tags::clear_mark(nx);
                            prev->next.compare_exchange_weak(
                                expected, replacement,
                                std::memory_order_acq_rel);
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                        else
                        {
                            prev = cur;
                            if (reclaimer::NEEDS_REVALIDATION)
                            {
                                reclaimer::protect(HP_PREV, cur);
                            }
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                    }
                }
            }

            // ---- Bucket initialization --------------------------------------
            // Creates the dummy of bucket b (split-order key = bit-reversed
            // b) and links it into the list via the bucket's parent
            // (recursively), then publishes the bucket cell. A dummy that
            // loses the cell CAS stays linked as a harmless node: it is
            // never searched from a bucket and never removed.
            static std::size_t parent_bucket(std::size_t b)
            {
                std::size_t h = 1;
                while ((h << 1) <= b)
                {
                    h <<= 1;
                }
                return b - h;
            }

            void initialize_bucket(bucket_table* t, std::size_t b) const
            {
                if (t->buckets[b].load(std::memory_order_acquire) != nullptr)
                {
                    return;
                }
                if (b != 0)
                {
                    initialize_bucket(t, parent_bucket(b));
                }
                node* d = alloc_node();
                d->so_key = dummy_split_order_key(b);
                node* start =
                    t->buckets[parent_bucket(b)].load(std::memory_order_acquire);
                insert_from(start, d);
                node* expected = nullptr;
                if (!t->buckets[b].compare_exchange_strong(
                        expected, d, std::memory_order_acq_rel))
                {
                    // lost the race: another dummy for this bucket was
                    // published; this dummy stays as an unreachable list
                    // node until destruction (benign, documented).
                }
            }

            // Inserts n before the first node with so_key >= n->so_key,
            // starting the walk at `start` (a dummy; never removed).
            void insert_from(node* start, node* n) const
            {
                for (;;)
                {
                    node* prev = start;
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_PREV, prev);
                    }
                    node* tagged = nullptr;
                    node* cur = protect_from(prev->next, HP_CUR, tagged);
                    for (;;)
                    {
                        if (cur == nullptr || cur->so_key >= n->so_key)
                        {
                            n->next.store(cur, std::memory_order_relaxed);
                            node* expected = tagged;
                            if (tags::is_marked(expected))
                            {
                                break; // predecessor marked: retry
                            }
                            if (prev->next.compare_exchange_weak(
                                    expected, n, std::memory_order_acq_rel))
                            {
                                return;
                            }
                            break; // retry from the top
                        }
                        // Single acquire load of cur->next: the cleared
                        // pointer is used for prefetch address and marked
                        // check; the raw tagged value for CAS expected.
                        node* raw_next = cur->next.load(
                            std::memory_order_acquire);
                        node* nx = tags::clear(raw_next);
                        // Prefetch the next node's cache line while we
                        // process cur. The walk is a dependent load chain
                        // (cur->next -> next->next), so starting the fetch
                        // early overlaps the L1/L2 miss with the CAS or
                        // protect_from work below.
                        if (nx != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(nx),
                                         _MM_HINT_T0);
#endif
                        }
                        if (tags::is_marked(raw_next))
                        {
                            node* expected = tagged;
                            node* replacement = tags::clear_mark(nx);
                            prev->next.compare_exchange_weak(
                                expected, replacement,
                                std::memory_order_acq_rel);
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                        else
                        {
                            prev = cur;
                            if (reclaimer::NEEDS_REVALIDATION)
                            {
                                reclaimer::protect(HP_PREV, cur);
                            }
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                    }
                }
            }

            // ---- Insert -----------------------------------------------------
            // Inserts a node whose key equals k. Returns INSERT_OK on
            // success (the count has been incremented); returns
            // INSERT_DUPLICATE when the key is already present -- in that
            // case the node was never linked and the caller must free it.
            enum insert_result
            {
                INSERT_OK = 0,
                INSERT_DUPLICATE = 1
            };

            int insert_node(node* n, std::uint64_t mixed, std::uint64_t sk,
                            const Key& k)
            {
                n->so_key = sk;
                for (;;)
                {
                    node* prev = nullptr;
                    node* cur = nullptr;
                    locate(mixed, sk, prev, cur);
                    if (cur != nullptr && cur->so_key == sk)
                    {
                        // Fast path: first node in the run -- check for
                        // duplicate before entering the scan loop (common
                        // case: run length 1, no hash collisions).
                        node* fnx = cur->next.load(
                            std::memory_order_acquire);
                        if (!tags::is_marked(fnx) &&
                            key_equal(cur->key(), k))
                        {
                            return INSERT_DUPLICATE;
                        }
                        // Slow path: scan the run of equal split-order keys
                        node* rprev = prev;
                        node* rcur = cur;
                        node* tagged = nullptr;
                        for (;;)
                        {
                            if (rcur == nullptr || rcur->so_key != sk)
                            {
                                break; // key absent: insertion point found
                            }
                            node* rnx =
                                rcur->next.load(std::memory_order_acquire);
                            if (tags::clear(rnx) != nullptr)
                            {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        tags::clear(rnx)),
                                    _MM_HINT_T0);
#endif
                            }
                            if (tags::is_marked(rnx))
                            {
                                // help unlink, then continue the scan
                                node* expected = tagged;
                                node* replacement = tags::clear_mark(rnx);
                                rprev->next.compare_exchange_weak(
                                    expected, replacement,
                                    std::memory_order_acq_rel);
                                rcur = protect_from(rprev->next, HP_CUR,
                                                    tagged);
                                continue;
                            }
                            if (key_equal(rcur->key(), k))
                            {
                                return INSERT_DUPLICATE;
                            }
                            rprev = rcur;
                            if (reclaimer::NEEDS_REVALIDATION)
                            {
                                reclaimer::protect(HP_PREV, rcur);
                            }
                            rcur = protect_from(rprev->next, HP_CUR, tagged);
                        }
                    }
                    // insert before the first node with so_key >= sk
                    // (the head of the run, or the located position).
                    // The predecessor must not be marked: inserting after
                    // a marked node would hang the new node off a node that
                    // its eraser may detach at any moment (the new node
                    // would become unreachable). The CAS therefore compares
                    // the unmarked tagged value; a marked predecessor makes
                    // it fail and the operation retries.
                    n->next.store(cur, std::memory_order_relaxed);
                    node* expected = prev->next.load(std::memory_order_acquire);
                    if (tags::is_marked(expected))
                    {
                        continue; // predecessor became marked: retry
                    }
                    if (tags::clear(expected) == cur)
                    {
                        if (prev->next.compare_exchange_weak(
                                expected, n, std::memory_order_acq_rel))
                        {
                            count_.fetch_add(1, std::memory_order_release);
                            return INSERT_OK;
                        }
                    }
                    // else: list changed; retry the whole operation
                }
            }

            // ---- Find -------------------------------------------------------
            // Returns true and publishes `out` = the node when k is
            // present (published in HP_CUR at return).
            bool find_node(const Key& k, std::uint64_t mixed, std::uint64_t sk,
                           node*& out) const
            {
                for (;;)
                {
                    node* prev = nullptr;
                    node* cur = nullptr;
                    locate(mixed, sk, prev, cur);
                    if (cur == nullptr || cur->so_key != sk)
                    {
                        return false;
                    }
                    // Fast path: first node in the run is the target
                    // (no hash collisions -> run length 1, the common case).
                    node* rnx = cur->next.load(std::memory_order_acquire);
                    if (!tags::is_marked(rnx) && key_equal(cur->key(), k))
                    {
                        out = cur;
                        return true;
                    }
                    // Slow path: scan the run
                    node* rprev = prev;
                    node* rcur = cur;
                    node* tagged = nullptr;
                    for (;;)
                    {
                        if (rcur == nullptr || rcur->so_key != sk)
                        {
                            return false;
                        }
                        rnx = rcur->next.load(std::memory_order_acquire);
                        if (tags::clear(rnx) != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    tags::clear(rnx)),
                                _MM_HINT_T0);
#endif
                        }
                        if (tags::is_marked(rnx))
                        {
                            node* expected = tagged;
                            node* replacement = tags::clear_mark(rnx);
                            rprev->next.compare_exchange_weak(
                                expected, replacement,
                                std::memory_order_acq_rel);
                            rcur = protect_from(rprev->next, HP_CUR, tagged);
                            continue;
                        }
                        if (key_equal(rcur->key(), k))
                        {
                            out = rcur;
                            return true;
                        }
                        rprev = rcur;
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_PREV, rcur);
                        }
                        rcur = protect_from(rprev->next, HP_CUR, tagged);
                    }
                }
            }

            // find_value_ptr(): locates k and returns a pointer to its
            // payload; the pointer is formed while the node is still
            // protected. Using the pointer after a concurrent erase is the
            // caller's contract (weakly consistent references).
            value_type* find_value_ptr(const Key& k, std::uint64_t mixed,
                                       std::uint64_t sk) const
            {
                typename reclaimer::op_guard g;
                node* n = nullptr;
                if (!find_node(k, mixed, sk, n))
                {
                    return nullptr;
                }
                return &n->payload;
            }

            // destroy_node(): releases a node that was never linked (e.g.
            // the loser of a duplicate-key insert). Never call it on a
            // linked or retired node.
            static void destroy_node(node* n)
            {
                free_node_action(n);
            }

            // ---- Erase ------------------------------------------------------
            // Removes k; returns true when it was present.
            bool erase_node(const Key& k, std::uint64_t mixed, std::uint64_t sk)
            {
                for (;;)
                {
                    node* prev = nullptr;
                    node* cur = nullptr;
                    locate(mixed, sk, prev, cur);
                    if (cur == nullptr || cur->so_key != sk)
                    {
                        return false;
                    }
                    // Fast path: first node in the run is the target
                    // (common case: run length 1, no hash collisions).
                    {
                        node* fnx = cur->next.load(
                            std::memory_order_acquire);
                        if (!tags::is_marked(fnx) &&
                            key_equal(cur->key(), k))
                        {
                            // mark the target (linearization point)
                            if (cur->next.compare_exchange_strong(
                                    fnx, tags::set_marked(fnx),
                                    std::memory_order_acq_rel))
                            {
                                // Unlink: try the prev from locate() first.
                                // The node is now marked, so no insert can
                                // link after it and no one can re-insert it.
                                // If prev->next still points to cur, we can
                                // CAS directly -- avoiding a full re-locate.
                                unlink_marked(prev, cur, mixed, sk);
                                retire_node(cur);
                                count_.fetch_sub(1,
                                                 std::memory_order_release);
                                return true;
                            }
                            // CAS failed: someone else marked or changed
                            // next -- fall through to slow path
                        }
                    }
                    // Slow path: scan the run for the key
                    node* rprev = prev;
                    node* rcur = cur;
                    node* tagged = nullptr;
                    for (;;)
                    {
                        if (rcur == nullptr || rcur->so_key != sk)
                        {
                            return false;
                        }
                        node* rnx =
                            rcur->next.load(std::memory_order_acquire);
                        if (tags::clear(rnx) != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    tags::clear(rnx)),
                                _MM_HINT_T0);
#endif
                        }
                        if (tags::is_marked(rnx))
                        {
                            node* expected = tagged;
                            node* replacement = tags::clear_mark(rnx);
                            rprev->next.compare_exchange_weak(
                                expected, replacement,
                                std::memory_order_acq_rel);
                            rcur = protect_from(rprev->next, HP_CUR, tagged);
                            continue;
                        }
                        if (key_equal(rcur->key(), k))
                        {
                            break; // found: rcur is the target, rprev its
                                   // predecessor
                        }
                        rprev = rcur;
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_PREV, rcur);
                        }
                        rcur = protect_from(rprev->next, HP_CUR, tagged);
                    }
                    // mark the target (linearization point)
                    node* tnext = rcur->next.load(std::memory_order_acquire);
                    if (tags::is_marked(tnext))
                    {
                        continue; // someone else marked it: retry
                    }
                    if (!rcur->next.compare_exchange_strong(
                            tnext, tags::set_marked(tnext),
                            std::memory_order_acq_rel))
                    {
                        continue;
                    }
                    // Unlink the marked node, retrying until it is
                    // detached (helpers may win the unlink CAS for us).
                    // The node is retired only AFTER the unlink: retiring a
                    // still-linked node would let reclamation free memory
                    // that traversals can still reach (use-after-free).
                    unlink_marked(rprev, rcur, mixed, sk);
                    retire_node(rcur);
                    count_.fetch_sub(1, std::memory_order_release);
                    return true;
                }
            }

            void retire_node(node* n)
            {
                retire_action action;
                action.fn = &free_node_action;
                reclaimer::retire(n, action);
            }

            // ---- Unlink a marked node ---------------------------------------
            // After the eraser has marked `target` (set the MARKED bit on
            // its next pointer), the node must be physically unlinked from
            // the list. `hint_prev` is the predecessor that locate() found
            // before the mark CAS -- it MIGHT still be the predecessor, but
            // concurrent insertions or helper-unlinks may have changed it.
            //
            // Fast path: try to CAS hint_prev->next from target to target's
            // unmarked successor. If that fails (hint_prev->next no longer
            // equals target -- someone inserted after hint_prev, or hint_prev
            // was helped to skip target), fall through to the general
            // re-locate + scan-by-pointer loop.
            //
            // The scan-by-pointer loop is required because the run may
            // contain nodes with the same split-order key that sort before
            // the target; only a pointer comparison (not key equality)
            // identifies the exact marked node.
            void unlink_marked(node* hint_prev, node* target,
                               std::uint64_t mixed, std::uint64_t sk)
            {
                // Fast path: hint_prev->next still == target (marked)?
                node* tnext_raw = target->next.load(
                    std::memory_order_acquire);
                node* replacement = tags::clear_mark(tnext_raw);
                node* expected_hint = target;
                if (hint_prev->next.compare_exchange_strong(
                        expected_hint, replacement,
                        std::memory_order_acq_rel))
                {
                    return; // unlinked on the first try
                }
                // Slow path: re-locate and scan by pointer until unlinked
                for (;;)
                {
                    node* p = nullptr;
                    node* c = nullptr;
                    locate(mixed, sk, p, c);
                    if (c == nullptr || c->so_key != sk)
                    {
                        break; // target not in this run: already unlinked
                    }
                    node* rp = p;
                    node* rc = c;
                    node* rtag = nullptr;
                    bool found = false;
                    for (;;)
                    {
                        if (rc == nullptr || rc->so_key != sk)
                        {
                            break; // target absent: unlinked
                        }
                        if (rc == target)
                        {
                            found = true;
                            break;
                        }
                        rp = rc;
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_PREV, rc);
                        }
                        rc = protect_from(rp->next, HP_CUR, rtag);
                    }
                    if (!found)
                    {
                        break; // already unlinked by a helper
                    }
                    node* expected = rc;
                    node* repl = tags::clear_mark(
                        rc->next.load(std::memory_order_acquire));
                    if (rp->next.compare_exchange_weak(
                            expected, repl,
                            std::memory_order_acq_rel))
                    {
                        break; // unlinked
                    }
                    // list changed: re-locate and retry
                }
            }

            // ---- Resize -----------------------------------------------------
            // grow_once doubles the bucket table. The growing_ flag is a
            // spinlock: only one thread resizes at a time, but it does NOT
            // block insert/find/erase -- they proceed on the old table
            // concurrently. The flag prevents two threads from racing the
            // table_ CAS (which would waste one thread's make_table alloc).
            // After the winner publishes the new table, all subsequent
            // operations use it; the loser's maybe_grow() call is a no-op
            // (the growing_ CAS fails -> return immediately).
            //
            // Multi-grow: when count_ far exceeds the threshold (e.g. 8
            // threads inserting concurrently), a single doubling may not
            // suffice. The loop re-checks count_ vs the new table size and
            // grows again if needed, so one grow_once call can double
            // multiple times to catch up. This reduces the number of
            // growing_ CAS acquisitions under burst insertions.
            void grow_once(bool force = false)
            {
                bool expected_flag = false;
                if (!growing_.compare_exchange_strong(
                        expected_flag, true, std::memory_order_acq_rel))
                {
                    return;
                }
                try
                {
                    for (;;)
                    {
                        bucket_table* t = table_.load(std::memory_order_acquire);
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_TABLE, t);
                            if (table_.load(std::memory_order_acquire) != t)
                            {
                                continue;
                            }
                        }
                        const std::size_t sz = t->size;
                        if (!force &&
                            count_.load(std::memory_order_acquire) <=
                                static_cast<std::size_t>(
                                    t->size * inv_max_load_factor_))
                        {
                            break; // no growth needed
                        }
                        if (sz >= (1u << 30))
                        {
                            break; // pragmatic ceiling (2^30 buckets)
                        }
                        bucket_table* nt = make_table(sz * 2);
                        for (std::size_t i = 0; i < sz; ++i)
                        {
                            nt->buckets[i].store(
                                t->buckets[i].load(std::memory_order_acquire),
                                std::memory_order_relaxed);
                        }
                        if (table_.compare_exchange_strong(t, nt,
                                                           std::memory_order_acq_rel))
                        {
                            retire_node_table(t);
                            // Loop: re-check if another grow is needed
                            // (count_ may have increased during the copy).
                            continue;
                        }
                        destroy_table(nt); // never published: free directly
                    }
                }
                catch (...)
                {
                    // An allocation failure (or any other exception) must
                    // not leave the grow flag stuck: the container would
                    // silently stop growing forever. Reset and rethrow.
                    growing_.store(false, std::memory_order_release);
                    throw;
                }
                growing_.store(false, std::memory_order_release);
            }

            void retire_node_table(bucket_table* t)
            {
                retire_action action;
                action.fn = &retire_table_action;
                reclaimer::retire(t, action);
            }

            /**
             * @brief Grow-check on the insert path (amortized O(1)).
             *
             * Fast path: relaxed loads of count_ and the cached
             * threshold; only when count_ exceeds the cache does the
             * slow path load table_ (acquire, likely cache miss) and
             * verify. The cached threshold is always <= the true
             * threshold (a stale value only errs toward the slow path,
             * never skips a needed grow), and growth itself is a
             * performance optimization, not a correctness gate -- the
             * table works at any load factor.
             *
             * Called after every successful insert (and by reserve via
             * grow_once directly).
             */
            void maybe_grow()
            {
                // Fast path: relaxed load of count_ + cached threshold.
                // Avoids loading table_ (acquire, likely cache miss on a
                // separate cache line) on every insert. Only when count_
                // exceeds the cached threshold do we load table_ and verify.
                //
                // cached_threshold_ is a relaxed atomic: it is always <=
                // the true threshold for the current table. A stale value
                // (from a previous smaller table) makes us enter the slow
                // path more often -- never less. Correctness is preserved
                // because growth is a performance optimization, not a
                // correctness gate: the table works at any load factor.
                std::size_t cached = cached_threshold_.load(
                    std::memory_order_relaxed);
                if (cached > 0)
                {
                    std::size_t cnt = count_.load(
                        std::memory_order_relaxed);
                    if (cnt <= cached)
                    {
                        return;
                    }
                }
                // Slow path: load table_ and compute the actual threshold
                bucket_table* t = table_.load(std::memory_order_acquire);
                std::size_t threshold = static_cast<std::size_t>(
                    t->size * inv_max_load_factor_);
                cached_threshold_.store(threshold,
                                        std::memory_order_relaxed);
                if (count_.load(std::memory_order_acquire) > threshold)
                {
                    grow_once();
                    // After grow, refresh the cached threshold from the
                    // (potentially) new table
                    bucket_table* nt =
                        table_.load(std::memory_order_acquire);
                    cached_threshold_.store(
                        static_cast<std::size_t>(
                            nt->size * inv_max_load_factor_),
                        std::memory_order_relaxed);
                }
            }

            // ---- Clear -------------------------------------------------------
            // Removes every regular node (exclusive access contract: no
            // other operation may run concurrently). Nodes are destroyed
            // directly; the buckets and dummies stay. Marked nodes are
            // skipped: their erasing thread already retired them.
            void clear_impl()
            {
                typename reclaimer::op_guard g;
                bucket_table* t = table_.load(std::memory_order_acquire);
                node* prev = t->buckets[0].load(std::memory_order_acquire);
                node* cur = tags::clear(prev->next.load(std::memory_order_acquire));
                while (cur != nullptr)
                {
                    node* raw_next = cur->next.load(std::memory_order_acquire);
                    node* nx = tags::clear(raw_next);
                    if (!tags::is_marked(raw_next) &&
                        (cur->so_key & 1) != 0)
                    {
                        prev->next.store(nx, std::memory_order_relaxed);
                        free_node_action(cur);
                    }
                    else
                    {
                        prev = cur;
                    }
                    cur = nx;
                }
                count_.store(0, std::memory_order_release);
            }

            // ---- Iteration support ------------------------------------------
            // Iterators are weakly consistent: every operation re-locates
            // the element from the bucket table, so it stays correct under
            // concurrent inserts and erases. The node contents are copied
            // while the node is protected.
            struct node_info
            {
                std::uint64_t mixed;
                std::uint64_t sk;
                Key key;

                node_info()
                    : mixed(0)
                    , sk(0)
                    , key()
                {
                }
            };

            // first_node(): the first regular (odd split-order key) unmarked
            // node, starting from bucket 0's dummy (the list head).
            bool first_node(node_info& out) const
            {
                typename reclaimer::op_guard g;
                for (;;)
                {
                    bucket_table* t = table_.load(std::memory_order_acquire);
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_TABLE, t);
                        if (table_.load(std::memory_order_acquire) != t)
                        {
                            continue;
                        }
                    }
                    node* d = t->buckets[0].load(std::memory_order_acquire);
                    if (d == nullptr)
                    {
                        return false;
                    }
                    node* prev = d;
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_PREV, prev);
                    }
                    node* tagged = nullptr;
                    node* cur = protect_from(prev->next, HP_CUR, tagged);
                    for (;;)
                    {
                        if (cur == nullptr)
                        {
                            return false;
                        }
                        node* raw_next = cur->next.load(
                            std::memory_order_acquire);
                        node* nx = tags::clear(raw_next);
                        if (nx != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(nx),
                                         _MM_HINT_T0);
#endif
                        }
                        if (!tags::is_marked(raw_next) && (cur->so_key & 1) != 0)
                        {
                            out.mixed = bit_reverse(cur->so_key) & ~(1ULL << 63);
                            out.sk = cur->so_key;
                            out.key = cur->key();
                            return true;
                        }
                        if (tags::is_marked(raw_next))
                        {
                            node* expected = tagged;
                            node* replacement = tags::clear_mark(nx);
                            prev->next.compare_exchange_weak(
                                expected, replacement,
                                std::memory_order_acq_rel);
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                        else
                        {
                            prev = cur;
                            if (reclaimer::NEEDS_REVALIDATION)
                            {
                                reclaimer::protect(HP_PREV, cur);
                            }
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                    }
                }
            }

            // next_node(): the next regular unmarked node after the node
            // described by (mixed, sk, key). Two phases: first skip the
            // members of the equal split-order-key run up to and including
            // the current node (by key), then return the first unmarked
            // regular node that follows it -- either the next member of the
            // same run or the first node of the next run. Restarting the
            // skip from the run head on every call is O(run length) per
            // step; runs are normally tiny (hash collisions), and the
            // worst case is the same as a std chaining bucket.
            bool next_node(std::uint64_t mixed, std::uint64_t sk,
                           const Key& key, node_info& out) const
            {
                typename reclaimer::op_guard g;
                node* prev = nullptr;
                node* cur = nullptr;
                for (;;)
                {
                    locate(mixed, sk, prev, cur);
                    if (cur == nullptr)
                    {
                        return false;
                    }
                    node* tagged = nullptr;
                    // phase 1: skip to the current node
                    for (;;)
                    {
                        if (cur == nullptr || cur->so_key != sk)
                        {
                            // current node absent (erased): its run has no
                            // anchor; skip to the run's successor (weak
                            // consistency; the run's remaining members are
                            // not visited)
                            return false;
                        }
                        node* raw_next = cur->next.load(
                            std::memory_order_acquire);
                        node* nx = tags::clear(raw_next);
                        if (nx != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(nx),
                                         _MM_HINT_T0);
#endif
                        }
                        if (!tags::is_marked(raw_next) &&
                            key_equal(cur->key(), key))
                        {
                            break; // at the current node
                        }
                        if (tags::is_marked(raw_next))
                        {
                            node* expected = tagged;
                            node* replacement = tags::clear_mark(nx);
                            prev->next.compare_exchange_weak(
                                expected, replacement,
                                std::memory_order_acq_rel);
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                        else
                        {
                            prev = cur;
                            if (reclaimer::NEEDS_REVALIDATION)
                            {
                                reclaimer::protect(HP_PREV, cur);
                            }
                            cur = protect_from(prev->next, HP_CUR, tagged);
                        }
                    }
                    // phase 2: advance past the current node.
                    // Single-load pattern (O-series): each iteration loads
                    // cur->next once (acquire). The cleared pointer serves
                    // as the prefetch address; the raw tagged value serves
                    // the is_marked check and CAS expected. No second load
                    // at the bottom -- the marked check for the new cur is
                    // handled at the top of the next iteration, and the
                    // prefetch is issued from the same top-of-loop load.
                    for (;;)
                    {
                        node* raw_next = cur->next.load(
                            std::memory_order_acquire);
                        node* nx = tags::clear(raw_next);
                        if (nx != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(nx),
                                         _MM_HINT_T0);
#endif
                        }
                        if (tags::is_marked(raw_next))
                        {
                            node* expected = tagged;
                            node* replacement = tags::clear_mark(nx);
                            prev->next.compare_exchange_weak(
                                expected, replacement,
                                std::memory_order_acq_rel);
                            cur = protect_from(prev->next, HP_CUR, tagged);
                            continue;
                        }
                        prev = cur;
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_PREV, cur);
                        }
                        cur = protect_from(prev->next, HP_CUR, tagged);
                        if (cur == nullptr)
                        {
                            return false;
                        }
                        const bool is_regular = (cur->so_key & 1) != 0;
                        if (cur->so_key == sk || is_regular)
                        {
                            out.mixed = bit_reverse(cur->so_key) & ~(1ULL << 63);
                            out.sk = cur->so_key;
                            out.key = cur->key();
                            return true;
                        }
                    }
                }
            }

        private:
            // ---- Members -----------------------------------------------------
            // Mutable: const operations of a lock-free container still
            // allocate (bucket dummies), hash, and compare; the functor and
            // allocator instances must be thread-safe for concurrent use
            // (std::allocator and stateless functors are).
            static node_pool pool_;
            mutable node_alloc na_;
            mutable Hash hash_;
            mutable KeyEqual eq_;
            alignas(MPMC_CACHE_LINE) std::atomic<bucket_table*> table_;
            alignas(MPMC_CACHE_LINE) std::atomic<std::size_t> count_;
            alignas(MPMC_CACHE_LINE) std::atomic<bool> growing_;
            float max_load_factor_;
            float inv_max_load_factor_;
            // Cached growth threshold: updated by whichever thread last
            // loaded table_. Avoids reloading table_ on every insert's
            // maybe_grow fast path. Relaxed atomic: a stale value is
            // safe (it is always <= true threshold, so we err toward
            // the slow path, never skip a needed grow).
            alignas(MPMC_CACHE_LINE) std::atomic<std::size_t> cached_threshold_;
        };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

        // The node pool is shared by every container of this instantiation
        // (it outlives any single container, so a reclaimer free action can
        // never touch a destroyed pool).
        template <typename Key, typename Value, typename Hash,
                  typename KeyEqual, typename Allocator, typename Reclamation>
        typename split_ordered_table<Key, Value, Hash, KeyEqual, Allocator,
                                     Reclamation>::node_pool
            split_ordered_table<Key, Value, Hash, KeyEqual, Allocator,
                                Reclamation>::pool_;
    }
}

#endif // MPMC_SPLIT_ORDERED_TABLE_HPP
