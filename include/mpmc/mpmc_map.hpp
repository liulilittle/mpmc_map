#ifndef MPMC_MAP_HPP
#define MPMC_MAP_HPP

/**
 * @file mpmc_map.hpp
 * @brief Concurrent ordered map with a std::map-compatible API.
 *
 * The engine is the concurrent red-black tree (detail/rb_tree.hpp):
 * searches are lock-free, updates use hand-over-hand node locks from the
 * root down the search path (deadlock-free by construction), and removed
 * nodes are reclaimed through the reclamation scheme.
 *
 * API notes (deviations from std::map are documented here and in
 * docs/API_en.md):
 *
 *   * Iterators are weakly consistent: every dereference and increment
 *     re-searches the tree, concurrent inserts and erases are tolerated,
 *     and using a reference after the element was erased is undefined.
 *   * swap(), hint-emplace and node handles are not provided.
 *   * clear() requires exclusive access (no concurrent operations).
 *   * The Compare functor must not throw and must be safe for concurrent
 *     use; the key type must be nothrow-copy-constructible and
 *     nothrow-destructible (enforced by static_assert).
 *   * size() is exact (atomic counter).
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "detail/rb_tree.hpp"

namespace mpmc
{
    template <class Key, class T, class Compare = std::less<Key>,
              class Allocator = std::allocator<std::pair<const Key, T>>,
              class Reclamation = reclamation::default_scheme>
    class map
    {
        static_assert(std::is_nothrow_copy_constructible<Key>::value,
                      "mpmc::map requires a nothrow-copy-constructible key");
        static_assert(std::is_nothrow_destructible<Key>::value,
                      "mpmc::map requires a nothrow-destructible key");
        static_assert(std::is_nothrow_destructible<T>::value,
                      "mpmc::map requires a nothrow-destructible mapped "
                      "type");

        using engine = detail::rb_tree<Key, T, Compare, Allocator,
                                       Reclamation>;
        using node = typename engine::node;

    public:
        using key_type = Key;
        using mapped_type = T;
        using value_type = std::pair<const Key, T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using key_compare = Compare;
        using allocator_type = Allocator;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reclaimer = typename engine::reclaimer;

        class iterator;
        class const_iterator;

        // ---- Construction ---------------------------------------------------
        /**
         * @brief Construct an empty map with a default-constructed
         *        comparator and (default-constructed) allocator.
         *
         * Thread-safe to use immediately; no threads may be inside
         * operations when the map is destroyed (see the class docs).
         */
        map()
            : alloc_()
            , impl_(alloc_, Compare())
        {
        }

        /**
         * @brief Construct an empty map with a specific comparison
         *        functor and a default-constructed allocator.
         *
         * @param comp the strict weak ordering used for all key
         *              comparisons and lookups
         */
        explicit map(const Compare& comp)
            : alloc_()
            , impl_(alloc_, comp)
        {
        }

        /**
         * @brief Construct an empty map with a specific allocator and
         *        the default comparator.
         *
         * @param alloc the allocator used for all node allocation; the
         *              same instance is returned by get_allocator()
         */
        explicit map(const allocator_type& alloc)
            : alloc_(alloc)
            , impl_(alloc_, Compare())
        {
        }

        /**
         * @brief Construct an empty map with a specific comparison
         *        functor and allocator.
         *
         * @param comp  the strict weak ordering used for all key
         *              comparisons and lookups
         * @param alloc the allocator used for all node allocation; the
         *              same instance is returned by get_allocator()
         */
        map(const Compare& comp, const allocator_type& alloc)
            : alloc_(alloc)
            , impl_(alloc_, comp)
        {
        }

        /**
         * @brief Copy construction is not supported (deleted).
         *
         * The map is not copyable; move construction is also not
         * provided. Share the map through references or pointers.
         */
        map(const map&) = delete;

        /**
         * @brief Copy assignment is not supported (deleted).
         */
        map& operator=(const map&) = delete;

        /**
         * @brief Destroy the map and free every node.
         *
         * Contract: no thread may be inside an operation and every user
         * thread must have been joined before destruction. All
         * remaining retirements are reclaimed (drain_all) and the node
         * pool is drained, so the allocator balance is exact after this
         * call.
         */
        ~map()
        {
        }

        /**
         * @brief Audit hook: wait for quiescence and validate the tree
         *        structure.
         *
         * Intended for testing and debugging only. May block briefly
         * while waiting for in-flight updates to complete.
         *
         * @return true if the tree structure is valid, false otherwise.
         */
        bool dbg_validate() const
        {
            return impl_.dbg_validate();
        }

        /**
         * @brief Audit hook: return the current maximum tree depth.
         *
         * Intended for testing and debugging only; used to bound
         * degenerate trees (a valid red-black tree has depth at most
         * 2*log2(n+1)+2).
         *
         * @return the number of levels in the longest root-to-leaf
         *         path (0 for an empty tree).
         */
        unsigned dbg_max_depth() const
        {
            return impl_.dbg_max_depth();
        }

        /**
         * @brief Audit hook: validate the red-black coloring
         *        invariants (root black, no red-red, equal black
         *        heights).
         *
         * Intended for testing and debugging only.
         *
         * @return true if the invariants hold, false otherwise.
         */
        bool dbg_rb_valid() const
        {
            return impl_.dbg_rb_valid();
        }

        /**
         * @brief Audit hook: print the tree (key, color, children, and
         *        each leaf's black-height) to stderr.
         *
         * Intended for testing and debugging only.
         */
        void dbg_dump_rb() const
        {
            impl_.dbg_dump_rb();
        }


        // ---- Capacity --------------------------------------------------------
        /**
         * @brief Return whether the map is empty.
         *
         * Thread-safe: the result is exact at the instant of the call
         * but may change immediately afterwards under concurrent
         * modification.
         *
         * @return true if the map contains no elements, false otherwise.
         */
        bool empty() const
        {
            return impl_.empty();
        }

        /**
         * @brief Return the number of elements currently in the map.
         *
         * Exact (backed by an atomic counter), unlike most lock-free
         * containers. Thread-safe: the count is exact at the instant of
         * the call but may change immediately afterwards.
         *
         * @return the current element count.
         */
        size_type size() const
        {
            return impl_.size();
        }

        /**
         * @brief Return an upper bound on the number of elements the map
         *        can ever hold.
         *
         * @return the theoretical maximum element count.
         */
        size_type max_size() const
        {
            return static_cast<size_type>(-1) / sizeof(value_type);
        }

        // ---- Modifiers -------------------------------------------------------
        /**
         * @brief Insert an element constructed in place from args if the
         *        key is not already present.
         *
         * The element is constructed with the map's allocator; if the key
         * is already present the element is destroyed and not inserted.
         * O(log n) time.
         *
         * @param args arguments forwarded to the value_type constructor
         *
         * @return a pair holding an iterator to the inserted or existing
         *         element and a bool that is true when a new element was
         *         inserted.
         */
        template <class... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            node* n = impl_.alloc_node();
            try
            {
                ::new (static_cast<void*>(&n->payload))
                    value_type(std::forward<Args>(args)...);
            }
            catch (...)
            {
                engine::destroy_node(n);
                throw;
            }
            const Key k = n->key();
            if (impl_.insert_node(n, k))
            {
                return std::make_pair(iterator(this, k), true);
            }
            engine::destroy_node(n);
            return std::make_pair(iterator(this, k), false);
        }

        /**
         * @brief Insert a copy of v if its key is not already present.
         *
         * Equivalent to emplace(v). O(log n) time.
         *
         * @param v the value to insert
         *
         * @return a pair holding an iterator to the inserted or existing
         *         element and a bool that is true when a new element was
         *         inserted.
         */
        std::pair<iterator, bool> insert(const value_type& v)
        {
            return emplace(v);
        }

        /**
         * @brief Insert v, moved from, if its key is not already present.
         *
         * Equivalent to emplace(std::move(v)). O(log n) time.
         *
         * @param v the value to move into the map
         *
         * @return a pair holding an iterator to the inserted or existing
         *         element and a bool that is true when a new element was
         *         inserted.
         */
        std::pair<iterator, bool> insert(value_type&& v)
        {
            return emplace(std::move(v));
        }

        /**
         * @brief Insert an element constructed from p if its key is not
         *        already present.
         *
         * Equivalent to emplace(std::forward<P>(p)). O(log n) time.
         *
         * @param p a value_type or a type convertible to it
         *
         * @return a pair holding an iterator to the inserted or existing
         *         element and a bool that is true when a new element was
         *         inserted.
         */
        template <class P>
        std::pair<iterator, bool> insert(P&& p)
        {
            return emplace(std::forward<P>(p));
        }

        /**
         * @brief Insert an element with key k, constructing the mapped
         *        value in place from args, if k is not already present.
         *
         * Unlike emplace, this does not construct (or copy) the key when
         * it is already present, so the key is never moved-from. O(log
         * n) time.
         *
         * @param k    the key to insert if absent
         * @param args arguments forwarded to the mapped type's constructor
         *
         * @return a pair holding an iterator to the inserted or existing
         *         element and a bool that is true when a new element was
         *         inserted.
         */
        template <class... Args>
        std::pair<iterator, bool> try_emplace(const key_type& k, Args&&... args)
        {
            return emplace(std::piecewise_construct,
                           std::forward_as_tuple(k),
                           std::forward_as_tuple(std::forward<Args>(args)...));
        }

        /**
         * @brief Insert an element with key k if absent, or assign obj to
         *        the mapped value of the existing element.
         *
         * O(log n) time.
         *
         * @param k   the key to insert or assign
         * @param obj the value to insert or assign
         *
         * @return a pair holding an iterator to the inserted or existing
         *         element and a bool that is true when a new element was
         *         inserted.
         */
        template <class M>
        std::pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj)
        {
            std::pair<iterator, bool> r = try_emplace(k, std::forward<M>(obj));
            if (!r.second)
            {
                value_type* p = impl_.find_value_ptr(k);
                if (p != nullptr)
                {
                    p->second = std::forward<M>(obj);
                }
            }
            return r;
        }

        /**
         * @brief Return a reference to the mapped value for k, inserting a
         *        default-constructed mapped value if k is absent.
         *
         * If k is absent a new element is inserted. The returned reference
         * is weakly consistent: it is invalidated if the element is
         * concurrently erased, and no thread may be writing to the same
         * element concurrently. O(log n) time.
         *
         * @param k the key to look up (and insert if absent)
         *
         * @return a reference to the mapped value for k.
         */
        mapped_type& operator[](const key_type& k)
        {
            value_type* p = impl_.find_value_ptr(k);
            if (p != nullptr)
            {
                return p->second;
            }
            node* n = impl_.alloc_node();
            ::new (static_cast<void*>(&n->payload))
                value_type(std::piecewise_construct, std::forward_as_tuple(k),
                           std::tuple<>());
            if (impl_.insert_node(n, k))
            {
                return n->payload.second;
            }
            engine::destroy_node(n);
            p = impl_.find_value_ptr(k);
            return p->second;
        }

        /**
         * @brief Return a reference to the mapped value for k.
         *
         * The returned reference is weakly consistent: it is invalidated
         * if the element is concurrently erased, and no thread may be
         * writing to the same element concurrently.
         *
         * @param k the key to look up
         *
         * @return a reference to the mapped value for k.
         *
         * @throws std::out_of_range if k is not present in the map.
         */
        mapped_type& at(const key_type& k)
        {
            value_type* p = impl_.find_value_ptr(k);
            if (p == nullptr)
            {
                throw std::out_of_range("mpmc::map::at");
            }
            return p->second;
        }

        /**
         * @brief Return a const reference to the mapped value for k.
         *
         * See the non-const overload for the weak-consistency caveat.
         *
         * @param k the key to look up
         *
         * @return a const reference to the mapped value for k.
         *
         * @throws std::out_of_range if k is not present in the map.
         */
        const mapped_type& at(const key_type& k) const
        {
            value_type* p = impl_.find_value_ptr(k);
            if (p == nullptr)
            {
                throw std::out_of_range("mpmc::map::at");
            }
            return p->second;
        }

        /**
         * @brief Erase the element whose key is k, if present.
         *
         * O(log n) time.
         *
         * @param k the key of the element to erase
         *
         * @return the number of elements erased (1 if k was present, else
         *         0).
         */
        size_type erase(const key_type& k)
        {
            return impl_.erase_node(k) ? 1 : 0;
        }

        /**
         * @brief Erase the element pointed to by pos.
         *
         * No-op when pos is end(). Returns the iterator that followed pos,
         * which may be end(). O(log n) time.
         *
         * @param pos a valid iterator into this map
         *
         * @return the iterator following the erased element (or end()).
         */
        iterator erase(const_iterator pos)
        {
            iterator next = const_iterator_to_iterator(pos);
            ++next;
            if (!pos.is_end())
            {
                impl_.erase_node(pos.key());
            }
            return next;
        }

        /**
         * @brief Remove all elements from the map.
         *
         * Requires exclusive access: no other thread may be inside an
         * operation when clear() runs (see the class docs). O(n) time.
         */
        void clear()
        {
            impl_.clear_impl();
        }

        // ---- Lookup ----------------------------------------------------------
        /**
         * @brief Find the element whose key is k.
         *
         * O(log n) time. The returned iterator is weakly consistent:
         * dereferencing it after the element was concurrently erased is
         * undefined.
         *
         * @param k the key to search for
         *
         * @return an iterator to the element, or end() if k is absent.
         */
        iterator find(const key_type& k)
        {
            typename engine::node* n = nullptr;
            if (impl_.find_node(k, n))
            {
                return iterator(this, k);
            }
            return end();
        }

        /**
         * @brief Find the element whose key is k (const overload).
         *
         * O(log n) time.
         *
         * @param k the key to search for
         *
         * @return a const iterator to the element, or end() if k is
         *         absent.
         */
        const_iterator find(const key_type& k) const
        {
            typename engine::node* n = nullptr;
            if (impl_.find_node(k, n))
            {
                return const_iterator(this, k);
            }
            return end();
        }

        /**
         * @brief Return whether the map contains an element with key k.
         *
         * O(log n) time.
         *
         * @param k the key to search for
         *
         * @return true if k is present, false otherwise.
         */
        bool contains(const key_type& k) const
        {
            typename engine::node* n = nullptr;
            return impl_.find_node(k, n);
        }

        /**
         * @brief Return the number of elements with key k.
         *
         * Since duplicate keys cannot exist, this is 0 or 1. O(log n)
         * time.
         *
         * @param k the key to search for
         *
         * @return 1 if k is present, 0 otherwise.
         */
        size_type count(const key_type& k) const
        {
            return contains(k) ? 1 : 0;
        }

        // ---- Bounds ------------------------------------------------------------
        /**
         * @brief Find the first element whose key is not less than k
         *        (i.e. the smallest element with key >= k).
         *
         * O(log n) time.
         *
         * @param k the key to search for
         *
         * @return an iterator to the lower bound, or end() if every key
         *         is less than k.
         */
        iterator lower_bound(const key_type& k)
        {
            typename engine::node_info info;
            if (impl_.bound_node(k, false, info))
            {
                return iterator(this, info.key);
            }
            return end();
        }

        /**
         * @brief Find the first element whose key is not less than k
         *        (const overload).
         *
         * O(log n) time.
         *
         * @param k the key to search for
         *
         * @return a const iterator to the lower bound, or end() if every
         *         key is less than k.
         */
        const_iterator lower_bound(const key_type& k) const
        {
            typename engine::node_info info;
            if (impl_.bound_node(k, false, info))
            {
                return const_iterator(this, info.key);
            }
            return end();
        }

        /**
         * @brief Find the first element whose key is greater than k.
         *
         * O(log n) time.
         *
         * @param k the key to search for
         *
         * @return an iterator to the upper bound, or end() if no key is
         *         greater than k.
         */
        iterator upper_bound(const key_type& k)
        {
            typename engine::node_info info;
            if (impl_.bound_node(k, true, info))
            {
                return iterator(this, info.key);
            }
            return end();
        }

        /**
         * @brief Find the first element whose key is greater than k
         *        (const overload).
         *
         * O(log n) time.
         *
         * @param k the key to search for
         *
         * @return a const iterator to the upper bound, or end() if no
         *         key is greater than k.
         */
        const_iterator upper_bound(const key_type& k) const
        {
            typename engine::node_info info;
            if (impl_.bound_node(k, true, info))
            {
                return const_iterator(this, info.key);
            }
            return end();
        }

        /**
         * @brief Return a range containing all elements with key k.
         *
         * The range is empty when k is absent. O(log n) time.
         *
         * @param k the key to search for
         *
         * @return a pair of iterators (lower_bound(k), upper_bound(k)).
         */
        std::pair<iterator, iterator> equal_range(const key_type& k)
        {
            iterator lb = lower_bound(k);
            iterator ub = upper_bound(k);
            return std::make_pair(lb, ub);
        }

        /**
         * @brief Return a range containing all elements with key k
         *        (const overload).
         *
         * The range is empty when k is absent. O(log n) time.
         *
         * @param k the key to search for
         *
         * @return a pair of const iterators (lower_bound(k),
         *         upper_bound(k)).
         */
        std::pair<const_iterator, const_iterator> equal_range(
            const key_type& k) const
        {
            const_iterator lb = lower_bound(k);
            const_iterator ub = upper_bound(k);
            return std::make_pair(lb, ub);
        }

        // ---- Observers ---------------------------------------------------------
        /**
         * @brief Return a copy of the allocator the map was constructed
         *        with.
         *
         * Stateful allocators are supported: the exact instance passed at
         * construction is stored and returned.
         *
         * @return the map's allocator.
         */
        allocator_type get_allocator() const
        {
            return alloc_;
        }

        /**
         * @brief Return a copy of the key comparison functor used to
         *        order the elements.
         *
         * @return the map's key_comp.
         */
        key_compare key_comp() const
        {
            return impl_.key_comp();
        }

        // ---- Iterators --------------------------------------------------------
        /**
         * @brief Return an iterator to the first element (the minimum
         *        key).
         *
         * Iterators are weakly consistent: elements erased concurrently
         * during iteration are skipped. O(log n) time (locating the first
         * element).
         *
         * @return an iterator to the first element, or end() if the map is
         *         empty.
         */
        iterator begin()
        {
            typename engine::node_info info;
            if (impl_.first_node(info))
            {
                return iterator(this, info.key);
            }
            return end();
        }

        /**
         * @brief Return a const iterator to the first element (the
         *        minimum key).
         *
         * @return a const iterator to the first element, or end() if the
         *         map is empty.
         */
        const_iterator begin() const
        {
            typename engine::node_info info;
            if (impl_.first_node(info))
            {
                return const_iterator(this, info.key);
            }
            return end();
        }

        /**
         * @brief Return a const iterator to the first element.
         *
         * @return a const iterator to the first element, or cend() if the
         *         map is empty.
         */
        const_iterator cbegin() const
        {
            return begin();
        }

        /**
         * @brief Return a past-the-end iterator.
         *
         * @return an iterator that denotes the end of the element
         *         sequence.
         */
        iterator end()
        {
            return iterator(this);
        }

        /**
         * @brief Return a const past-the-end iterator.
         *
         * @return a const iterator that denotes the end of the element
         *         sequence.
         */
        const_iterator end() const
        {
            return const_iterator(this);
        }

        /**
         * @brief Return a const past-the-end iterator.
         *
         * @return a const iterator that denotes the end of the element
         *         sequence.
         */
        const_iterator cend() const
        {
            return end();
        }

        // ---- Iterators --------------------------------------------------------
        class iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = typename map::value_type;
            using difference_type = typename map::difference_type;
            using pointer = typename map::pointer;
            using reference = typename map::reference;

            /**
             * @brief Default-construct a singular iterator.
             *
             * A singular iterator holds no container reference; it must be
             * assigned from a valid iterator before use. Dereferencing or
             * incrementing a singular iterator is undefined.
             */
            iterator()
                : owner_(nullptr)
                , is_end_(true)
            {
            }

            /**
             * @brief Dereference the iterator.
             *
             * The element is re-located on each dereference; if it was
             * concurrently erased the behavior is undefined (see the class
             * docs).
             *
             * @return a reference to the element.
             */
            reference operator*() const
            {
                return *resolve();
            }

            /**
             * @brief Dereference the iterator as a pointer.
             *
             * @return a pointer to the element.
             */
            pointer operator->() const
            {
                return resolve();
            }

            /**
             * @brief Advance the iterator to the next element.
             *
             * After the last element the iterator becomes end(). Elements
             * erased concurrently are skipped. O(log n) time.
             *
             * @return a reference to this iterator.
             */
            iterator& operator++()
            {
                advance();
                return *this;
            }

            /**
             * @brief Advance the iterator to the next element (postfix).
             *
             * @return a copy of the iterator before the advance.
             */
            iterator operator++(int)
            {
                iterator tmp = *this;
                advance();
                return tmp;
            }

            /**
             * @brief Compare two iterators for equality.
             *
             * Two iterators compare equal when they refer to the same
             * element, or when both are end() / singular. Comparing
             * iterators from different maps is false (unless both are
             * singular).
             *
             * @param o the iterator to compare with
             *
             * @return true if the iterators are equal, false otherwise.
             */
            bool operator==(const iterator& o) const
            {
                if (owner_ != o.owner_ || is_end_ != o.is_end_)
                {
                    return false;
                }
                if (is_end_)
                {
                    return true;
                }
                return !owner_->key_comp()(key_, o.key_) &&
                       !owner_->key_comp()(o.key_, key_);
            }

            /**
             * @brief Compare two iterators for inequality.
             *
             * @param o the iterator to compare with
             *
             * @return true if the iterators are not equal, false
             *         otherwise.
             */
            bool operator!=(const iterator& o) const
            {
                return !(*this == o);
            }

        private:
            friend class map;
            friend class const_iterator;

            explicit iterator(const map* owner)
                : owner_(owner)
                , is_end_(true)
            {
            }

            iterator(const map* owner, const Key& k)
                : owner_(owner)
                , is_end_(false)
                , key_(k)
            {
            }

            bool is_end() const
            {
                return is_end_;
            }

            const Key& key() const
            {
                return key_;
            }

            value_type* resolve() const
            {
                return owner_->impl_.find_value_ptr(key_);
            }

            void advance()
            {
                typename engine::node_info info;
                if (owner_->impl_.next_node(key_, info))
                {
                    key_ = info.key;
                }
                else
                {
                    is_end_ = true;
                }
            }

            const map* owner_;
            bool is_end_;
            Key key_;
        };

        class const_iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = typename map::value_type;
            using difference_type = typename map::difference_type;
            using pointer = typename map::const_pointer;
            using reference = typename map::const_reference;

            /**
             * @brief Default-construct a singular const iterator.
             *
             * A singular iterator holds no container reference; it must be
             * assigned from a valid iterator before use. Dereferencing or
             * incrementing a singular iterator is undefined.
             */
            const_iterator()
                : owner_(nullptr)
                , is_end_(true)
            {
            }

            /**
             * @brief Convert a mutable iterator to a const iterator.
             *
             * @param it the iterator to convert from
             */
            const_iterator(const iterator& it)
                : owner_(it.owner_)
                , is_end_(it.is_end_)
                , key_(it.key_)
            {
            }

            /**
             * @brief Dereference the iterator.
             *
             * The element is re-located on each dereference; if it was
             * concurrently erased the behavior is undefined (see the class
             * docs).
             *
             * @return a const reference to the element.
             */
            reference operator*() const
            {
                return *resolve();
            }

            /**
             * @brief Dereference the iterator as a pointer.
             *
             * @return a const pointer to the element.
             */
            pointer operator->() const
            {
                return resolve();
            }

            /**
             * @brief Advance the iterator to the next element.
             *
             * After the last element the iterator becomes end(). Elements
             * erased concurrently are skipped. O(log n) time.
             *
             * @return a reference to this iterator.
             */
            const_iterator& operator++()
            {
                advance();
                return *this;
            }

            /**
             * @brief Advance the iterator to the next element (postfix).
             *
             * @return a copy of the iterator before the advance.
             */
            const_iterator operator++(int)
            {
                const_iterator tmp = *this;
                advance();
                return tmp;
            }

            /**
             * @brief Compare two const iterators for equality.
             *
             * @param o the const iterator to compare with
             *
             * @return true if the iterators are equal, false otherwise.
             */
            bool operator==(const const_iterator& o) const
            {
                if (owner_ != o.owner_ || is_end_ != o.is_end_)
                {
                    return false;
                }
                if (is_end_)
                {
                    return true;
                }
                return !owner_->key_comp()(key_, o.key_) &&
                       !owner_->key_comp()(o.key_, key_);
            }

            /**
             * @brief Compare two const iterators for inequality.
             *
             * @param o the const iterator to compare with
             *
             * @return true if the iterators are not equal, false
             *         otherwise.
             */
            bool operator!=(const const_iterator& o) const
            {
                return !(*this == o);
            }

        private:
            friend class map;

            explicit const_iterator(const map* owner)
                : owner_(owner)
                , is_end_(true)
            {
            }

            const_iterator(const map* owner, const Key& k)
                : owner_(owner)
                , is_end_(false)
                , key_(k)
            {
            }

            bool is_end() const
            {
                return is_end_;
            }

            const Key& key() const
            {
                return key_;
            }

            const value_type* resolve() const
            {
                return owner_->impl_.find_value_ptr(key_);
            }

            void advance()
            {
                typename engine::node_info info;
                if (owner_->impl_.next_node(key_, info))
                {
                    key_ = info.key;
                }
                else
                {
                    is_end_ = true;
                }
            }

            const map* owner_;
            bool is_end_;
            Key key_;
        };

        /**
         * @brief Convert a const iterator to a mutable iterator.
         *
         * Private helper used by erase(const_iterator) to obtain the
         * mutable iterator form of a const iterator.
         *
         * @param c the const iterator to convert
         *
         * @return a mutable iterator referring to the same element (or the
         *         same end position).
         */
        static iterator const_iterator_to_iterator(const const_iterator& c)
        {
            return iterator(c.owner_, c.key_);
        }

    private:
        allocator_type alloc_;
        engine impl_;
    };
}

#endif // MPMC_MAP_HPP
