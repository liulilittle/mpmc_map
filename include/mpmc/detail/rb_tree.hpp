#ifndef MPMC_RB_TREE_HPP
#define MPMC_RB_TREE_HPP

/**
 * @file rb_tree.hpp
 * @brief Concurrent red-black tree (map/set engine).
 *
 * Concurrency model:
 *
 *   * Search (find/contains/lower_bound/upper_bound/iteration) is
 *     lock-free: it descends with plain atomic reads and never blocks.
 *     Every child-pointer read is bracketed by two reads of the node's
 *     lock bit: a mutator holds a node's lock for the whole time it
 *     mutates that node's fields, so a search that observed the node
 *     unlocked before AND after reading a child pointer read a stable
 *     pointer. A search that meets a locked node retries from the root.
 *     The lock bracketing makes intermediate mutator states invisible to
 *     searches, which keeps searches linearizable.
 *   * Updates (insert/erase) take node locks hand-over-hand from the
 *     root down the search path (strict parent-before-child order), then
 *     rebalance with the classic CLRS fixups. Fixup steps additionally
 *     lock the uncle/sibling/nephew nodes involved. Every lock edge is a
 *     tree edge directed root-to-leaf, so the lock-order graph is
 *     acyclic and deadlock is impossible.
 *   * Deletes of two-child nodes relink the in-order successor node into
 *     the deleted node's position (the key of pair<const K, V> is
 *     immutable, so values cannot be swapped); the successor's path is
 *     locked as an extension of the target's path.
 *
 * The fixups are driven by the locked path (a fixed-size stack), not by
 * parent pointers: the tree stores no parent pointers at all, which
 * removes the class of races on parent links entirely.
 *
 * Removed nodes are unlinked while locked and then retired through the
 * reclamation scheme; searches hold the per-operation guard, so a removed
 * node is never dereferenced after reclamation.
 *
 * This header is the engine for mpmc::map and mpmc::set; the public
 * headers add the std-compatible API.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#include "cache_line.hpp"
#include "reclamation.hpp"
#include "utils.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#ifndef MPMC_TREE_MAX_HEIGHT
#define MPMC_TREE_MAX_HEIGHT 256
#endif

namespace mpmc
{
    namespace detail
    {
        // ---- Payload ------------------------------------------------------
        template <typename Key, typename Value>
        struct tree_payload_holder
        {
            using type = std::pair<const Key, Value>;
        };

        template <typename Key>
        struct tree_payload_holder<Key, void>
        {
            using type = Key;
        };

        /**
         * @brief Concurrent red-black tree engine.
         */
        template <typename Key, typename Value, typename Compare,
                  typename Allocator, typename Reclamation>
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
        class rb_tree
        {
        public:
            using self = rb_tree;
            using reclaimer = typename Reclamation::template reclaimer<self>;
            using value_type = typename tree_payload_holder<Key, Value>::type;
            using allocator_type = Allocator;

            enum color
            {
                RED = 0,
                BLACK = 1
            };

            // ---- Node ------------------------------------------------------
            struct node;
            using node_alloc = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<node>;
            using node_pool = detail::node_pool<node, node_alloc,
                                                 detail::pool_link_left<node>>;

            struct node
            {
                // Hot fields (search path): left, right, locked, color.
                // These are touched on every node visited during lock-free
                // search. Keeping them in the first cache line minimizes
                // cache-line traffic under read contention.
                std::atomic<node*> left;
                std::atomic<node*> right;
                std::atomic<bool> locked; // spin bit: 1 = held by a mutator
                unsigned char color;
                // Cold fields: payload (may be large) and alloc_owner
                // (only used on allocation/free). Placed after the hot
                // fields so they don't push hot fields to a second line.
                value_type payload;
                node_alloc* alloc_owner;
                /**
                 * @brief Construct an unlinked node.
                 *
                 * Hot fields (left/right/locked/color) are initialized
                 * here; the payload is NOT constructed (the caller
                 * placement-news it with the real key/value) -- the
                 * empty payload is valid only for zero-size payloads.
                 */
                node()
                    : left(nullptr)
                    , right(nullptr)
                    , locked(false)
                    , color(RED)
                    , alloc_owner(nullptr)
                {
                }

                /**
                 * @brief The node's immutable key.
                 *
                 * Set at construction, never modified -- the lock-bit
                 * bracket's compare() between two lock loads relies on
                 * this immutability (no re-read needed).
                 *
                 * @return const reference to the key
                 */
                const Key& key() const
                {
                    return key_of_payload(payload);
                }

                /**
                 * @brief Reclaimer free action for tree nodes.
                 *
                 * Destroys the payload and returns the block to the node
                 * pool. Runs on the eraser's thread or at finalize/
                 * destruction -- never inside a concurrent operation.
                 *
                 * @param ptr the node pointer being reclaimed
                 */
                static void free_node(void* ptr)
                {
                    node* n = static_cast<node*>(ptr);
                    n->payload.~value_type();
                    pool_.release(n);
                }
            };

            enum
            {
                HP_ROOT = 0,
                HP_CUR = 1
            };

            // ---- Construction / destruction --------------------------------
            explicit rb_tree(const allocator_type& a, const Compare& c)
                : na_(a)
                , cmp_(c)
                , root_(nullptr)
                , count_(0)
            {
            }

            ~rb_tree()
            {
                // Destruction contract: no thread may be inside an
                // operation and every user thread has been joined. Free
                // every linked node, then drain the reclaimer (unlinked
                // nodes), then drain the node pool (shared per
                // instantiation; the blocks' own allocators are used).
                node* cur = root_.load(std::memory_order_acquire);
                free_subtree(cur);
                reclaimer::drain_all();
                pool_.drain();
            }

            rb_tree(const rb_tree&) = delete;
            rb_tree& operator=(const rb_tree&) = delete;

            // ---- Public engine operations ----------------------------------
            /**
             * @brief The current element count.
             *
             * @return count via the shared atomic (acquire)
             */
            std::size_t size() const
            {
                return count_.load(std::memory_order_acquire);
            }

            /**
             * @brief Whether the tree has no elements.
             *
             * @return true when size() == 0
             */
            bool empty() const
            {
                return size() == 0;
            }

            /**
             * @brief The configured comparison predicate (observers).
             *
             * @return const reference to the stored comparator
             */
            const Compare& key_comp() const
            {
                return cmp_;
            }

        private:
            // ---- Key helpers ------------------------------------------------
            static const Key& key_of_payload(const Key& k)
            {
                return k;
            }

            template <typename K, typename V>
            static const K& key_of_payload(const std::pair<const K, V>& p)
            {
                return p.first;
            }

            // Three-valued comparison: -1 / 0 / +1. Inlined for the
            // common case (int / pointer keys) where the comparator is
            // std::less and the branch predictor handles the two-way
            // check well. The call sites on the search and update paths
            // are the hottest lines in the tree.
            inline int compare(const Key& a, const Key& b) const
            {
                if (cmp_(a, b))
                {
                    return -1;
                }
                if (cmp_(b, a))
                {
                    return 1;
                }
                return 0;
            }

            /**
             * @brief Red-black black-ness test (null counts as black).
             *
             * @param n the node to test (may be null)
             * @return true when n is null or black
             */
            static bool is_black(node* n)
            {
                return n == nullptr || n->color == BLACK;
            }

            // ---- Locks --------------------------------------------------------
            static void lock_node(node* n)
            {
                unsigned spins = 0;
                for (;;)
                {
                    bool expected = false;
                    if (n->locked.compare_exchange_weak(expected, true,
                                                        std::memory_order_acq_rel))
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
             * @brief Release a node's lock.
             *
             * Stores false with release ordering, publishing the
             * mutator's child-pointer/color writes to searches waiting
             * on the lock-bit bracket (their acquire load pairs with
             * this store). The 1:1 partner of lock_node.
             *
             * @param n the locked node
             */
            static void unlock_node(node* n)
            {
                n->locked.store(false, std::memory_order_release);
            }

        public:
            // ---- Allocation ----------------------------------------------------
            node* alloc_node() const
            {
                node* n = pool_.acquire(na_);
                ::new (static_cast<void*>(n)) node();
                n->alloc_owner = &na_;
                return n;
            }

            /**
             * @brief Free every node of a subtree (destruction path).
             *
             * Iterative in-order free with an explicit stack: no
             * recursion and no link rewriting, so every node is freed
             * exactly once (a fold-based walk that reuses the same
             * leftmost leaf for every level overwrites earlier folds
             * and orphans whole subtrees -- that bug class is why the
             * explicit stack is mandatory).
             *
             * @param cur the subtree root (may be null)
             */
            static void free_subtree(node* cur)
            {
                // Iterative in-order free with an explicit stack: no
                // recursion and no link rewriting, so every node is freed
                // exactly once (a fold-based walk that reuses the same
                // leftmost leaf for every level overwrites earlier folds
                // and orphans whole subtrees).
                std::vector<node*> stack;
                while (cur != nullptr || !stack.empty())
                {
                    while (cur != nullptr)
                    {
                        stack.push_back(cur);
                        cur = cur->left.load(std::memory_order_relaxed);
                    }
                    cur = stack.back();
                    stack.pop_back();
                    node* right = cur->right.load(std::memory_order_relaxed);
                    node::free_node(cur);
                    cur = right;
                }
            }

            // ---- Lock-free search ----------------------------------------------
            // Returns 1 and `out` = the node when k is present; returns 0
            // with `out` = the last node visited when absent.
            int search(const Key& k, node*& out) const
            {
                for (;;)
                {
                    node* cur = root_.load(std::memory_order_acquire);
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_ROOT, cur);
                        if (root_.load(std::memory_order_acquire) != cur)
                        {
                            continue;
                        }
                    }
                    if (cur == nullptr)
                    {
                        out = nullptr;
                        return 0;
                    }
                    node* parent = nullptr;
                    bool retry = false;
                    while (cur != nullptr)
                    {
                        // Bracket the child-pointer read with two lock-bit
                        // loads (acquire). A mutator publishes child writes
                        // under the lock; if the node is unlocked on both
                        // sides the child pointer was stable.
                        //
                        // The compare() call between the two checks only
                        // reads cur->key() (immutable -- set at construction)
                        // and the comparator (stateless for std::less), so
                        // it cannot observe a torn write. The match path
                        // (c==0) returns immediately: the first lock check
                        // suffices because the key is immutable and the
                        // node was unlocked.
                        if (cur->locked.load(std::memory_order_acquire))
                        {
                            retry = true;
                            break;
                        }
                        const int c = compare(k, cur->key());
                        if (c == 0)
                        {
                            out = cur;
                            return 1;
                        }
                        // Relaxed load: the first locked.load(acquire) above
                        // already established a happens-before from any
                        // mutator that locked+unlocked this node, so the
                        // child pointer is visible. The second locked load
                        // (below) verifies the child was stable between the
                        // two checks. If a mutator was concurrently writing,
                        // the second check catches it (retry from root).
                        node* child = (c < 0)
                                          ? cur->left.load(std::memory_order_relaxed)
                                          : cur->right.load(std::memory_order_relaxed);
                        // Prefetch the child's cache line while we validate
                        // the bracket. The child's locked field (early in
                        // the node struct) is what the next iteration loads
                        // first; fetching it now overlaps the child cache
                        // miss with the second acquire barrier below.
                        if (child != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(child),
                                         _MM_HINT_T0);
#endif
                        }
                        if (cur->locked.load(std::memory_order_acquire))
                        {
                            retry = true;
                            break;
                        }
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_CUR, child);
                        }
                        parent = cur;
                        cur = child;
                    }
                    if (!retry)
                    {
                        out = parent;
                        return 0;
                    }
                }
            }

            // ---- Locked path -----------------------------------------------------
            struct path_stack
            {
                node* nodes[MPMC_TREE_MAX_HEIGHT];
                unsigned depth;

                /**
                 * @brief Fixed-capacity stack of locked path nodes.
                 *
                 * Sized to MPMC_TREE_MAX_HEIGHT (the maximum possible
                 * RB height for the supported tree size), so no
                 * allocation happens on the update path. The locked
                 * nodes are unlocked in reverse order (leaf to root)
                 * via unlock_path.
                 */
                path_stack()
                    : depth(0)
                {
                }

                /**
                 * @brief Push a node onto the path.
                 *
                 * @param n the newly locked node (must fit: depth <
                 *          MPMC_TREE_MAX_HEIGHT by the height bound)
                 */
                void push(node* n)
                {
                    nodes[depth++] = n;
                }

                /**
                 * @brief The most recently pushed node (the path leaf).
                 *
                 * @return nodes[depth - 1]; precondition: depth > 0
                 */
                node* top() const
                {
                    return nodes[depth - 1];
                }

                /**
                 * @brief The node at path position i.
                 *
                 * @param i depth from the root (0 = root)
                 * @return nodes[i]; precondition: i < depth
                 */
                node* at(unsigned i) const
                {
                    return nodes[i];
                }
            };

            // Locks the path root -> ... -> the node where the comparison
            // against k first yields 0, or the node that would be k's
            // parent. Returns:
            //   1: the path to a node with key == k (in path.top())
            //   0: the path to the insertion/removal parent; `side` says
            //      which child is the position (0 = left, 1 = right)
            //  -1: the tree is empty or changed shape beyond the stack
            //      capacity - the caller unlocks and retries.
            int lock_path(const Key& k, path_stack& path, int& side)
            {
                for (;;)
                {
                    path.depth = 0;
                    node* cur = root_.load(std::memory_order_acquire);
                    if (cur == nullptr)
                    {
                        return -1;
                    }
                    lock_node(cur);
                    // Re-verify the root: while we waited for its lock
                    // another mutator may have rotated the root (or erased
                    // it), leaving `cur` a stale root. Descending from a
                    // stale root would walk a subtree that is no longer the
                    // whole tree - inserts would corrupt the BST ordering
                    // and erases would miss present keys. The root is the
                    // only node whose subtree can change while we wait,
                    // because every child pointer below it is read while
                    // its parent (which we already hold) is locked.
                    if (root_.load(std::memory_order_acquire) != cur)
                    {
                        unlock_node(cur);
                        continue;
                    }
                    path.push(cur);
                    for (;;)
                    {
                        const int c = compare(k, cur->key());
                        if (c == 0)
                        {
                            return 1;
                        }
                        node* child = (c < 0)
                                          ? cur->left.load(std::memory_order_relaxed)
                                          : cur->right.load(std::memory_order_relaxed);
                        if (child == nullptr)
                        {
                            side = (c < 0) ? 0 : 1;
                            return 0;
                        }
                        if (path.depth >= MPMC_TREE_MAX_HEIGHT)
                        {
                            unlock_path(path);
                            return -1;
                        }
                        lock_node(child);
                        path.push(child);
                        cur = child;
                    }
                }
            }

            /**
             * @brief Unlock every node on the path, leaf to root.
             *
             * Releases locks in reverse acquisition order; the RB
             * fixups unlock nodes progressively as they climb, so at
             * any point the path holds exactly the still-locked prefix.
             *
             * @param path the locked path stack to release
             */
            static void unlock_path(path_stack& path)
            {
                while (path.depth > 0)
                {
                    unlock_node(path.nodes[--path.depth]);
                }
            }

            // ---- Rotation internals + rewiring ------------------------------------
            // The rotation swaps the child pointers of `p` and `n` (n is
            // p's child on `side`); the caller rewires the parent side.
            static void rotate_inner(node* p, node* n, int side)
            {
                if (side == 0)
                {
                    // right-rotation: n = p->left
                    node* middle = n->right.load(std::memory_order_relaxed);
                    n->right.store(p, std::memory_order_relaxed);
                    p->left.store(middle, std::memory_order_relaxed);
                }
                else
                {
                    // left-rotation: n = p->right
                    node* middle = n->left.load(std::memory_order_relaxed);
                    n->left.store(p, std::memory_order_relaxed);
                    p->right.store(middle, std::memory_order_relaxed);
                }
            }

            // Rewires the child pointer of `parent` that currently points
            // to `old_child` so that it points to `new_child`.
            static void rewire_child(node* parent, node* old_child,
                                     node* new_child)
            {
                if (parent->left.load(std::memory_order_relaxed) == old_child)
                {
                    parent->left.store(new_child, std::memory_order_relaxed);
                }
                else
                {
                    parent->right.store(new_child, std::memory_order_relaxed);
                }
            }

            // Rotates `p` with its child on `side`; `p`'s position in the
            // path is `p_index` (or NOT_IN_PATH when p is not on the path,
            // in which case the caller performs the parent rewire).
            // Updates the parent's child pointer (or the root pointer when
            // p is the root).
            static const unsigned NOT_IN_PATH = ~0u;

            void rotate(node* p, int side, path_stack& path, unsigned p_index)
            {
                node* n = (side == 0) ? p->left.load(std::memory_order_relaxed)
                                      : p->right.load(std::memory_order_relaxed);
                rotate_inner(p, n, side);
                if (p_index == NOT_IN_PATH)
                {
                    return; // the caller rewires the parent side
                }
                if (p_index > 0)
                {
                    rewire_child(path.nodes[p_index - 1], p, n);
                }
                else
                {
                    root_.store(n, std::memory_order_release);
                }
            }

            // ---- Insert fixup -------------------------------------------------------
            // The path [root .. parent, n] is locked; n is the new red
            // node already linked under `parent`.
            void fixup_insert(path_stack& path)
            {
                while (path.depth >= 3)
                {
                    node* x = path.top();
                    node* p = path.at(path.depth - 2);
                    if (p->color != RED)
                    {
                        return;
                    }
                    node* g = path.at(path.depth - 3);
                    bool p_is_left = false;
                    node* uncle = nullptr;
                    if (p == g->left.load(std::memory_order_relaxed))
                    {
                        p_is_left = true;
                        uncle = g->right.load(std::memory_order_relaxed);
                    }
                    else
                    {
                        p_is_left = false;
                        uncle = g->left.load(std::memory_order_relaxed);
                    }
                    if (uncle != nullptr && uncle->color == RED)
                    {
                        // case 1: recolor; climb two levels (the popped
                        // entries are unlocked - the fixup no longer needs
                        // them)
                        lock_node(uncle);
                        p->color = BLACK;
                        uncle->color = BLACK;
                        g->color = RED;
                        unlock_node(uncle);
                        unlock_node(path.at(path.depth - 1)); // x
                        unlock_node(path.at(path.depth - 2)); // p
                        path.depth -= 2;
                        continue;
                    }
                    // cases 2/3: the uncle is black or null
                    if (x == (p_is_left
                                  ? p->right.load(std::memory_order_relaxed)
                                  : p->left.load(std::memory_order_relaxed)))
                    {
                        // case 2: rotate p up, then re-examine
                        rotate(p, p_is_left ? 1 : 0, path, path.depth - 2); // INS-C2
                        // the parent of x changed: x is now p's parent; the
                        // new p sits on the SAME side of g as the old p
                        path.nodes[path.depth - 2] = x;
                        path.nodes[path.depth - 1] = p;
                        x = p;
                        p = path.at(path.depth - 2);
                    }
                    // case 3: rotate g up and recolor; the rotated g and
                    // the popped bottom entry are unlocked (no longer on
                    // the path)
                    p->color = BLACK;
                    g->color = RED;
                    rotate(g, p_is_left ? 0 : 1, path, path.depth - 3); // INS-C3
                    unlock_node(g);
                    path.nodes[path.depth - 3] = p;
                    unlock_node(path.nodes[path.depth - 1]);
                    path.depth -= 2;
                    return;
                }
            }

            // ---- Delete fixup ---------------------------------------------------------
            // The path [root .. x's parent] is locked; `x` is the node
            // that replaced the removed one (may be null); `x_side` is the
            // side of x (or of the removed node) under the path top.
            // CLRS RB-DELETE-FIXUP with the sibling and nephews locked in
            // root-to-leaf order.
            void fixup_delete(path_stack& path, node* x, int x_side)
            {
                for (;;)
                {
                    if (path.depth == 0)
                    {
                        // x is the root position: the double black is
                        // resolved by blackening the root. Lock the root
                        // first - it is not locked by this operation (the
                        // case-2 climb unlocks the popped parent), so an
                        // unlocked write could race with a concurrent
                        // erase that detaches and retires it.
                        node* r = root_.load(std::memory_order_acquire);
                        if (r != nullptr)
                        {
                            lock_node(r);
                            r->color = BLACK;
                            unlock_node(r);
                        }
                        return;
                    }
                    if (x != nullptr && x->color == RED)
                    {
                        // x's parent (path.top()) is locked by us, so x is
                        // still linked and cannot be retired concurrently.
                        x->color = BLACK;
                        return;
                    }
                    node* p = path.top();
                    node* w = (x_side == 0)
                                  ? p->right.load(std::memory_order_relaxed)
                                  : p->left.load(std::memory_order_relaxed);
                    if (w != nullptr && w->color == RED)
                    {
                        // case 1: recolor and rotate p (the sibling is the
                        // child on the side opposite to x); w stays locked
                        // on the path and is released by the caller's
                        // unlock_path
                        lock_node(w);
                        w->color = BLACK;
                        p->color = RED;
                        rotate(p, 1 - x_side, path, path.depth - 1); // DEL-C1
                        path.nodes[path.depth - 1] = w;
                        path.push(p);
                        continue;
                    }
                    // w: black or null
                    node* near = (w == nullptr)
                                     ? nullptr
                                     : ((x_side == 0) ? w->left.load(
                                                            std::memory_order_relaxed)
                                                      : w->right.load(
                                                            std::memory_order_relaxed));
                    node* far = (w == nullptr)
                                    ? nullptr
                                    : ((x_side == 0) ? w->right.load(
                                                           std::memory_order_relaxed)
                                                     : w->left.load(
                                                           std::memory_order_relaxed));
                    bool w_locked = false;
                    if (w != nullptr && !is_black(near) && is_black(far))
                    {
                        // case 3: recolor near, rotate w; then case 4
                        lock_node(w);
                        lock_node(near);
                        near->color = BLACK;
                        w->color = RED;
                        rotate(w, x_side == 0 ? 0 : 1, path, NOT_IN_PATH); // DEL-C3
                        rewire_child(p, w, near);
                        unlock_node(w);
                        w = near;
                        w_locked = true;
                        far = (x_side == 0)
                                  ? w->right.load(std::memory_order_relaxed)
                                  : w->left.load(std::memory_order_relaxed);
                    }
                    if (w == nullptr || is_black(w))
                    {
                        if (w == nullptr || is_black(far))
                        {
                        // case 2: recolor w, climb (the popped parent is
                        // unlocked - the fixup no longer needs it)
                        if (w != nullptr)
                        {
                            if (!w_locked)
                            {
                                lock_node(w);
                            }
                            w->color = RED;
                            if (!w_locked)
                            {
                                unlock_node(w);
                            }
                        }
                        node* oldp = p;
                        x = oldp;
                        path.depth -= 1;
                        unlock_node(oldp);
                        if (path.depth >= 1)
                        {
                            node* gp = path.top();
                            x_side =
                                (gp->left.load(std::memory_order_relaxed) ==
                                 x)
                                    ? 0
                                    : 1;
                        }
                        continue;
                        }
                        // case 4: w black, far red; w stays locked on the
                        // path and is released by the caller's unlock_path
                        if (!w_locked)
                        {
                            lock_node(w);
                        }
                        if (far != nullptr)
                        {
                            lock_node(far);
                        }
                        w->color = p->color;
                        p->color = BLACK;
                        if (far != nullptr)
                        {
                            far->color = BLACK;
                            unlock_node(far);
                        }
                        rotate(p, 1 - x_side, path, path.depth - 1); // DEL-C4
                        path.nodes[path.depth - 1] = w;
                        unlock_node(p); // the rotated parent: no longer needed
                        return;
                    }
                    return; // w red handled above; unreachable
                }
            }

        public:
            // ---- Insert ---------------------------------------------------------------
            // Inserts a node whose key equals k; returns true on success
            // (the count is incremented). On duplicate the node is not
            // freed; the caller frees it.
            bool insert_node(node* n, const Key& k)
            {
                for (;;)
                {
                    path_stack path;
                    int side = 0;
                    int r = lock_path(k, path, side);
                    if (r == 1)
                    {
                        unlock_path(path);
                        return false; // duplicate
                    }
                    if (r < 0)
                    {
                        // empty tree (or path overflow): CAS the root
                        node* expected = nullptr;
                        if (root_.compare_exchange_strong(
                                expected, n, std::memory_order_acq_rel))
                        {
                            n->color = BLACK;
                            count_.fetch_add(1, std::memory_order_release);
                            return true;
                        }
                        continue; // empty-tree race: retry
                    }
                    node* parent = path.top();
                    // the new node is locked before publication: the fixup
                    // may rotate it (case 2) and the unlock_path releases it
                    lock_node(n);
                    if (side == 0)
                    {
                        parent->left.store(n, std::memory_order_relaxed);
                    }
                    else
                    {
                        parent->right.store(n, std::memory_order_relaxed);
                    }
                    n->color = RED;
                    path.push(n);
                    fixup_insert(path);
                    node* r2 = root_.load(std::memory_order_acquire);
                    if (r2 != nullptr)
                    {
                        if (path.depth == 0)
                        {
                            lock_node(r2);
                            r2->color = BLACK;
                            unlock_node(r2);
                        }
                        else
                        {
                            r2->color = BLACK; // r2 == path.nodes[0]: held
                        }
                    }
                    unlock_path(path);
                    count_.fetch_add(1, std::memory_order_release);
                    return true;
                }
            }

            // ---- Erase ---------------------------------------------------------------
            // Removes k; returns true when it was present. The removed
            // node is retired before returning.
            bool erase_node(const Key& k)
            {
                for (;;)
                {
                retry:
                    path_stack path;
                    int side = 0;
                    int r = lock_path(k, path, side);
                    if (r == 0)
                    {
                        unlock_path(path);
                        return false; // absent
                    }
                    if (r < 0)
                    {
                        return false; // empty tree
                    }
                    node* target = path.top();
                    node* left = target->left.load(std::memory_order_relaxed);
                    node* right = target->right.load(std::memory_order_relaxed);
                    node* x = nullptr;
                    int x_side = 0;
                    if (left == nullptr || right == nullptr)
                    {
                        // at most one child: unlink directly. x (the child,
                        // possibly null) takes the target's position, so the
                        // fixup's x_side is the target's side under its
                        // parent - not the child's side under the target
                        // (for a leaf the child is null and its "side" is
                        // meaningless).
                        node* child = left != nullptr ? left : right;
                        x = child;
                        x_side =
                            (path.depth >= 2 &&
                             path.at(path.depth - 2)
                                     ->left.load(std::memory_order_relaxed) ==
                                 target)
                                ? 0
                                : 1;
                        if (path.depth >= 2)
                        {
                            rewire_child(path.at(path.depth - 2), target,
                                         child);
                        }
                        else
                        {
                            root_.store(child, std::memory_order_release);
                        }
                        path.depth -= 1; // drop the target
                        unlock_node(target); // detached: release its lock
                        if (target->color == BLACK)
                        {
                            fixup_delete(path, x, x_side);
                        }
                        node* r2 = root_.load(std::memory_order_acquire);
                        if (r2 != nullptr)
                        {
                            if (path.depth == 0)
                            {
                                lock_node(r2);
                                r2->color = BLACK;
                                unlock_node(r2);
                            }
                            else
                            {
                                r2->color = BLACK; // r2 == path.nodes[0]
                            }
                        }
                        unlock_path(path);
                        retire_node(target);
                        count_.fetch_sub(1, std::memory_order_release);
                        return true;
                    }
                    // two children: relink the in-order successor (the
                    // leftmost of the right subtree) into the target's
                    // position; lock the extension path
                    unsigned target_index = path.depth - 1;
                    node* succ = right;
                    path.push(succ);
                    lock_node(succ);
                    while (succ->left.load(std::memory_order_relaxed) != nullptr)
                    {
                        if (path.depth >= MPMC_TREE_MAX_HEIGHT)
                        {
                            unlock_path(path);
                            goto retry;
                        }
                        succ = succ->left.load(std::memory_order_relaxed);
                        lock_node(succ);
                        path.push(succ);
                    }
                    node* succ_parent = path.at(path.depth - 2);
                    node* succ_right =
                        succ->right.load(std::memory_order_relaxed);
                    const unsigned char succ_color = succ->color;
                    // unlink the successor
                    if (succ_parent == target)
                    {
                        target->right.store(succ_right,
                                            std::memory_order_relaxed);
                        x_side = 1;
                    }
                    else
                    {
                        succ_parent->left.store(succ_right,
                                                std::memory_order_relaxed);
                        x_side = 0;
                    }
                    // the successor takes the target's place
                    if (target_index >= 1)
                    {
                        rewire_child(path.at(target_index - 1), target, succ);
                    }
                    else
                    {
                        root_.store(succ, std::memory_order_release);
                    }
                    succ->left.store(left, std::memory_order_relaxed);
                    // when the successor IS the target's right child, its
                    // right pointer must keep its own old right subtree
                    // (target->right was already rewired to succ_right);
                    // assigning `right` (= the successor itself) would
                    // create a self-loop
                    succ->right.store((succ == right) ? succ_right : right,
                                      std::memory_order_relaxed);
                    succ->color = target->color;
                    // Detach the target's children: they alias the
                    // successor's (the successor inherited them).
                    target->left.store(nullptr, std::memory_order_relaxed);
                    target->right.store(nullptr, std::memory_order_relaxed);
                    // Rebuild the fixup path through the successor: it now
                    // occupies the target's entry. When the successor was
                    // NOT the target's right child, the double black sits at
                    // the successor's old position (succ_parent's left slot,
                    // now succ_right) and the fixup climbs from the
                    // successor's parent upward; the extension's
                    // intermediates are released. When the successor WAS
                    // the target's right child, the double black sits on
                    // the successor's own right slot and the fixup climbs
                    // from the successor itself (see below).
                    if (succ == right)
                    {
                        path.nodes[target_index] = succ;
                        path.depth = target_index + 1;
                    }
                    else
                    {
                        // the fixup path keeps the tree order: the successor
                        // occupies the target's entry, followed by the
                        // extension's intermediates and the successor's
                        // parent on top (x's parent); only the successor's
                        // old trailing entry is dropped
                        path.nodes[target_index] = succ;
                        path.depth -= 1;
                    }
                    if (succ == right)
                    {
                        // The successor was the target's right child: the
                        // double black sits where the successor used to be,
                        // i.e. on the successor's own right slot (succ_right,
                        // possibly null) at the successor's own level. The
                        // fixup therefore climbs from the successor - NOT
                        // from the target's parent - and the successor
                        // stays locked on the path until unlock_path.
                        x = succ_right;
                        x_side = 1;
                    }
                    else
                    {
                        x = succ_right;
                    }
                    if (succ_color == BLACK)
                    {
                        fixup_delete(path, x, x_side);
                    }
                    // the detached target is not on the fixup path:
                    // release its lock before retiring it
                    unlock_node(target);
                    node* r2 = root_.load(std::memory_order_acquire);
                    if (r2 != nullptr)
                    {
                        if (path.depth == 0)
                        {
                            lock_node(r2);
                            r2->color = BLACK;
                            unlock_node(r2);
                        }
                        else
                        {
                            r2->color = BLACK; // r2 == path.nodes[0]
                        }
                    }
                    unlock_path(path);
                    retire_node(target);
                    count_.fetch_sub(1, std::memory_order_release);
                    return true;
                }
            }

            /**
             * @brief Hand a tree node to the reclaimer.
             *
             * The node must already be unlinked from the tree; its
             * payload is destroyed by node::free_node when the
             * reclamation scheme proves no thread can reach it anymore.
             *
             * @param n the unlinked node to retire
             */
            void retire_node(node* n)
            {
                retire_action action;
                action.fn = &node::free_node;
                reclaimer::retire(n, action);
            }

        public:
            // ---- Node management (used by the public container headers) ----
            static void destroy_node(node* n)
            {
                node::free_node(n);
            }

            // Audit hook: wait for quiescence, then validate the BST structure
            bool dbg_validate() const
            {
                // wait until no node in the tree is locked
                for (unsigned round = 0; round < 100000; ++round)
                {
                    node* cur = root_.load(std::memory_order_acquire);
                    bool any_locked = false;
                    node* q[65536];
                    unsigned qh = 0, qt = 0;
                    if (cur != nullptr)
                    {
                        q[qt++] = cur;
                    }
                    while (qh < qt)
                    {
                        node* n = q[qh++];
                        if (n->locked.load(std::memory_order_acquire))
                        {
                            any_locked = true;
                            break;
                        }
                        node* l = n->left.load(std::memory_order_acquire);
                        node* r = n->right.load(std::memory_order_acquire);
                        if (l != nullptr && qt < 65536)
                        {
                            q[qt++] = l;
                        }
                        if (r != nullptr && qt < 65536)
                        {
                            q[qt++] = r;
                        }
                    }
                    if (!any_locked)
                    {
                        break;
                    }
                    std::this_thread::yield();
                }
                // structural check
                node* q[65536];
                unsigned qh = 0, qt = 0;
                node* cur = root_.load(std::memory_order_acquire);
                unsigned visited = 0;
                if (cur != nullptr)
                {
                    q[qt++] = cur;
                }
                while (qh < qt)
                {
                    node* n = q[qh++];
                    ++visited;
                    node* l = n->left.load(std::memory_order_relaxed);
                    node* r = n->right.load(std::memory_order_relaxed);
                    if (l != nullptr)
                    {
                        if (!(l->key() < n->key()))
                        {
                            std::fprintf(stderr, "[INVALID] node %d left %d\n",
                                         n->key(), l->key());
                            std::fflush(stderr);
                            return false;
                        }
                        q[qt++] = l;
                    }
                    if (r != nullptr)
                    {
                        if (!(n->key() < r->key()))
                        {
                            std::fprintf(stderr, "[INVALID] node %d right %d\n",
                                         n->key(), r->key());
                            std::fflush(stderr);
                            return false;
                        }
                        q[qt++] = r;
                    }
                }
                if (visited != count_.load(std::memory_order_relaxed))
                {
                    std::fprintf(stderr, "[INVALID] visited=%u size=%llu\n",
                                 visited,
                                 (unsigned long long)count_.load(
                                     std::memory_order_relaxed));
                    std::fflush(stderr);
                    return false;
                }
                // in-order check (global ordering)
                int prev = 0;
                bool first = true;
                node* in = root_.load(std::memory_order_acquire);
                node* stack[256];
                unsigned sp = 0;
                while (in != nullptr || sp > 0)
                {
                    while (in != nullptr)
                    {
                        if (sp >= 256)
                        {
                            std::fprintf(stderr, "[INVALID] in-order stack\n");
                            std::fflush(stderr);
                            return false;
                        }
                        stack[sp++] = in;
                        in = in->left.load(std::memory_order_relaxed);
                    }
                    in = stack[--sp];
                    if (!first && !(prev < in->key()))
                    {
                        std::fprintf(stderr, "[INVALID] order %d after %d\n",
                                     in->key(), prev);
                        std::fflush(stderr);
                        return false;
                    }
                    first = false;
                    prev = in->key();
                    in = in->right.load(std::memory_order_relaxed);
                }
                return true;
            }

            // Audit hook: full red-black invariant check - no RED node with
            // a RED child, and every root-to-leaf path carries the same
            // black-height (iterative; tolerant of locked nodes).
            bool dbg_rb_valid() const
            {
                struct frame
                {
                    node* n;
                    int bh; // black-height of the path so far
                };
                node* cur = root_.load(std::memory_order_acquire);
                if (cur == nullptr)
                {
                    return true;
                }
                if (cur->color != BLACK)
                {
                    std::fprintf(stderr, "[RB] root %d not black\n",
                                 cur->key());
                    std::fflush(stderr);
                    return false;
                }
                frame stack[1024];
                unsigned sp = 0;
                stack[sp++] = frame{cur, 1};
                int target_bh = -1;
                while (sp > 0)
                {
                    frame f = stack[--sp];
                    if (f.n->color == RED)
                    {
                        node* l = f.n->left.load(std::memory_order_relaxed);
                        node* r = f.n->right.load(std::memory_order_relaxed);
                        if ((l != nullptr && l->color == RED) ||
                            (r != nullptr && r->color == RED))
                        {
                            std::fprintf(stderr, "[RB] red-red at %d\n",
                                         f.n->key());
                            std::fflush(stderr);
                            return false;
                        }
                    }
                    int nbh = f.bh + (f.n->color == BLACK ? 1 : 0);
                    node* l = f.n->left.load(std::memory_order_relaxed);
                    node* r = f.n->right.load(std::memory_order_relaxed);
                    if (l == nullptr && r == nullptr)
                    {
                        if (target_bh < 0)
                        {
                            target_bh = nbh;
                        }
                        else if (nbh != target_bh)
                        {
                            std::fprintf(stderr,
                                         "[RB] black-height %d != %d at %d\n",
                                         nbh, target_bh, f.n->key());
                            std::fflush(stderr);
                            return false;
                        }
                    }
                    else
                    {
                        if (l != nullptr)
                        {
                            stack[sp++] = frame{l, nbh};
                        }
                        if (r != nullptr)
                        {
                            stack[sp++] = frame{r, nbh};
                        }
                    }
                }
                return true;
            }

            // Audit hook: dump the tree: key, color, children, and each
            // leaf's black-height.
            void dbg_dump_rb() const
            {
                struct frame
                {
                    node* n;
                    int bh;
                };
                node* cur = root_.load(std::memory_order_acquire);
                frame stack[1024];
                unsigned sp = 0;
                if (cur != nullptr)
                {
                    stack[sp++] = frame{cur, 0};
                }
                while (sp > 0)
                {
                    frame f = stack[--sp];
                    const int nbh =
                        f.bh + (f.n->color == BLACK ? 1 : 0);
                    node* l = f.n->left.load(std::memory_order_relaxed);
                    node* r = f.n->right.load(std::memory_order_relaxed);
                    std::fprintf(stderr,
                                 "[RB] k=%d c=%s l=%d(%s) r=%d(%s) bh=%d\n",
                                 f.n->key(), f.n->color == BLACK ? "B" : "R",
                                 l ? l->key() : -1,
                                 l ? (l->color == BLACK ? "B" : "R") : "-",
                                 r ? r->key() : -1,
                                 r ? (r->color == BLACK ? "B" : "R") : "-",
                                 nbh);
                    if (l != nullptr)
                    {
                        stack[sp++] = frame{l, nbh};
                    }
                    if (r != nullptr)
                    {
                        stack[sp++] = frame{r, nbh};
                    }
                }
            }

            // Audit hook: maximum root-to-leaf depth (0 for an empty tree)
            unsigned dbg_max_depth() const
            {
                node* cur = root_.load(std::memory_order_acquire);
                unsigned best = 0;
                struct frame
                {
                    node* n;
                    unsigned d;
                };
                frame stack[1024];
                unsigned sp = 0;
                if (cur != nullptr)
                {
                    stack[sp++] = frame{cur, 1};
                }
                while (sp > 0)
                {
                    frame f = stack[--sp];
                    if (f.d > best)
                    {
                        best = f.d;
                    }
                    node* l = f.n->left.load(std::memory_order_relaxed);
                    node* r = f.n->right.load(std::memory_order_relaxed);
                    if (l != nullptr)
                    {
                        stack[sp++] = frame{l, f.d + 1};
                    }
                    if (r != nullptr)
                    {
                        stack[sp++] = frame{r, f.d + 1};
                    }
                }
                return best;
            }

            // ---- Find ---------------------------------------------------------------
            bool find_node(const Key& k, node*& out) const
            {
                typename reclaimer::op_guard g;
                node* found = nullptr;
                if (search(k, found) == 1)
                {
                    out = found;
                    return true;
                }
                return false;
            }

            /**
             * @brief Find a key and return a pointer to its payload.
             *
             * Runs under an op_guard (per-operation reclamation
             * bracket); the returned pointer stays valid only while the
             * guard is alive (i.e. inside the calling operation).
             *
             * @param k the key to look up
             * @return pointer to the payload of the matching node, or
             *         null when absent
             */
            value_type* find_value_ptr(const Key& k) const
            {
                typename reclaimer::op_guard g;
                node* found = nullptr;
                if (search(k, found) == 1)
                {
                    return &found->payload;
                }
                return nullptr;
            }

            struct node_info
            {
                Key key;
            };

            // ---- Bounds ---------------------------------------------------------------
            // lower_bound: the smallest node with key >= k; upper_bound:
            // the smallest node with key > k (`strict` = true). The key is
            // copied while the node is protected.
            bool bound_node(const Key& k, bool strict, node_info& out) const
            {
                typename reclaimer::op_guard g;
                for (;;)
                {
                    node* cur = root_.load(std::memory_order_acquire);
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_ROOT, cur);
                        if (root_.load(std::memory_order_acquire) != cur)
                        {
                            continue;
                        }
                    }
                    node* candidate = nullptr;
                    while (cur != nullptr)
                    {
                        if (cur->locked.load(std::memory_order_acquire))
                        {
                            break;
                        }
                        const int c = compare(k, cur->key());
                        node* child = nullptr;
                        if (strict)
                        {
                            if (c < 0)
                            {
                                candidate = cur;
                                child = cur->left.load(std::memory_order_relaxed);
                            }
                            else
                            {
                                child = cur->right.load(std::memory_order_relaxed);
                            }
                        }
                        else
                        {
                            if (c <= 0)
                            {
                                candidate = cur;
                                child = cur->left.load(std::memory_order_relaxed);
                            }
                            else
                            {
                                child = cur->right.load(std::memory_order_relaxed);
                            }
                        }
                        // Prefetch the child's cache line while we
                        // validate the bracket (same as search()).
                        if (child != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(child),
                                         _MM_HINT_T0);
#endif
                        }
                        if (cur->locked.load(std::memory_order_acquire))
                        {
                            break;
                        }
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_CUR, child);
                        }
                        cur = child;
                    }
                    if (candidate != nullptr)
                    {
                        out.key = candidate->key();
                        return true;
                    }
                    return false;
                }
            }

            // ---- Iteration ---------------------------------------------------------------
            bool first_node(node_info& out) const
            {
                typename reclaimer::op_guard g;
                for (;;)
                {
                    node* cur = root_.load(std::memory_order_acquire);
                    if (reclaimer::NEEDS_REVALIDATION)
                    {
                        reclaimer::protect(HP_ROOT, cur);
                        if (root_.load(std::memory_order_acquire) != cur)
                        {
                            continue;
                        }
                    }
                    while (cur != nullptr)
                    {
                        if (cur->locked.load(std::memory_order_acquire))
                        {
                            break;
                        }
                        node* child = cur->left.load(std::memory_order_relaxed);
                        // Prefetch the child's cache line while we
                        // validate the bracket (same as search()).
                        if (child != nullptr)
                        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                            _mm_prefetch(reinterpret_cast<const char*>(child),
                                         _MM_HINT_T0);
#endif
                        }
                        if (cur->locked.load(std::memory_order_acquire))
                        {
                            break;
                        }
                        if (child == nullptr)
                        {
                            out.key = cur->key();
                            return true;
                        }
                        if (reclaimer::NEEDS_REVALIDATION)
                        {
                            reclaimer::protect(HP_CUR, child);
                        }
                        cur = child;
                    }
                    if (cur == nullptr)
                    {
                        return false;
                    }
                }
            }

            // next_node(k): the smallest key strictly greater than k.
            bool next_node(const Key& k, node_info& out) const
            {
                return bound_node(k, true, out);
            }

            // ---- Clear ---------------------------------------------------------------
            void clear_impl()
            {
                // exclusive access contract
                node* cur = root_.exchange(nullptr, std::memory_order_acq_rel);
                free_subtree(cur);
                count_.store(0, std::memory_order_release);
            }

        private:
            static node_pool pool_;
            mutable node_alloc na_;
            mutable Compare cmp_;
            alignas(MPMC_CACHE_LINE) std::atomic<node*> root_;
            alignas(MPMC_CACHE_LINE) std::atomic<std::size_t> count_;
        };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

        // The node pool is shared by every container of this instantiation
        // (it outlives any single container, so a reclaimer free action can
        // never touch a destroyed pool).
        template <typename Key, typename Value, typename Compare,
                  typename Allocator, typename Reclamation>
        typename rb_tree<Key, Value, Compare, Allocator, Reclamation>::node_pool
            rb_tree<Key, Value, Compare, Allocator, Reclamation>::pool_;
    }
}

#endif // MPMC_RB_TREE_HPP
