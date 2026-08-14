#ifndef MPMC_UNORDERED_SET_HPP
#define MPMC_UNORDERED_SET_HPP

/**
 * @file mpmc_unordered_set.hpp
 * @brief Lock-free MPMC hash set with a std::unordered_set-compatible API.
 *
 * The underlying engine is the split-ordered list
 * (detail/split_ordered_table.hpp); see mpmc_unordered_map.hpp for the
 * API notes (weakly consistent iterators, exclusive-access clear(),
 * non-throwing hash/equality, exact size()).
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "detail/split_ordered_table.hpp"

namespace mpmc
{
    template <class Key, class Hash = std::hash<Key>,
              class KeyEqual = std::equal_to<Key>,
              class Allocator = std::allocator<Key>,
              class Reclamation = reclamation::default_scheme>
    class unordered_set
    {
        // See mpmc_unordered_map.hpp: the functors must not throw and must
        // be safe for concurrent use (documented, not static-asserted).
        static_assert(std::is_nothrow_copy_constructible<Key>::value,
                      "mpmc::unordered_set requires a nothrow-copy-"
                      "constructible key");
        static_assert(std::is_nothrow_destructible<Key>::value,
                      "mpmc::unordered_set requires a nothrow-destructible "
                      "key");

        using engine = detail::split_ordered_table<Key, void, Hash, KeyEqual,
                                                   Allocator, Reclamation>;
        using node = typename engine::node;

    public:
        using key_type = Key;
        using value_type = Key;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using hasher = Hash;
        using key_equal = KeyEqual;
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
         * @brief Construct an empty set with the default hasher, key
         *        equality and (default-constructed) allocator.
         *
         * Thread-safe to use immediately; no threads may be inside
         * operations when the set is destroyed (see the class docs).
         */
        unordered_set()
            : alloc_()
            , impl_(alloc_, Hash(), KeyEqual(), 1.0f)
        {
        }

        /**
         * @brief Construct an empty set with a specific allocator.
         *
         * @param alloc the allocator used for all node allocation; the
         *              same instance is returned by get_allocator()
         */
        explicit unordered_set(const allocator_type& alloc)
            : alloc_(alloc)
            , impl_(alloc_, Hash(), KeyEqual(), 1.0f)
        {
        }

        /**
         * @brief Construct a set with a requested bucket count and
         *        optional custom hasher / equality / allocator.
         *
         * Reserves `n` buckets up front so the first `n` inserts do not
         * trigger table growth (useful for latency-sensitive startup).
         *
         * @param n     the initial bucket count hint (0 disables the
         *              pre-reservation)
         * @param hf    the hash function
         * @param eql   the key equality predicate
         * @param alloc the node allocator
         */
        explicit unordered_set(size_type n, const hasher& hf = hasher(),
                               const key_equal& eql = key_equal(),
                               const allocator_type& alloc = allocator_type())
            : alloc_(alloc)
            , impl_(alloc_, hf, eql, 1.0f)
        {
            reserve(n);
        }

        /**
         * @brief Construct a set with a requested bucket count and a
         *        specific allocator (default hasher/equality).
         *
         * @param n     the initial bucket count hint
         * @param alloc the node allocator
         */
        unordered_set(size_type n, const allocator_type& alloc)
            : unordered_set(n, hasher(), key_equal(), alloc)
        {
        }

        /**
         * @brief Construct a set with a requested bucket count, a custom
         *        hasher and a specific allocator (default equality).
         *
         * @param n     the initial bucket count hint
         * @param hf    the hash function
         * @param alloc the node allocator
         */
        unordered_set(size_type n, const hasher& hf, const allocator_type& alloc)
            : unordered_set(n, hf, key_equal(), alloc)
        {
        }

        unordered_set(const unordered_set&) = delete;
        unordered_set& operator=(const unordered_set&) = delete;

        /**
         * @brief Destroy the set and free every node.
         *
         * Contract: no thread may be inside an operation and every user
         * thread must have been joined before destruction. All remaining
         * retirements are reclaimed (drain_all) and the node pool is
         * drained, so the allocator balance is exact after this call.
         */
        ~unordered_set()
        {
        }

        // ---- Capacity --------------------------------------------------------
        /**
         * @brief Return whether the set is empty.
         *
         * Thread-safe: the result is exact at the instant of the call but
         * may change immediately afterwards under concurrent modification.
         *
         * @return true if the set contains no elements, false otherwise.
         */
        bool empty() const
        {
            return impl_.empty();
        }

        /**
         * @brief Return the number of elements currently in the set.
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
         * @brief Return an upper bound on the number of elements the set
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
         * The element is constructed with the set's allocator; if the key
         * is already present the element is destroyed and not inserted.
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
            const std::uint64_t mixed = impl_.mixed_hash(k);
            const std::uint64_t sk = impl_.so_key_of(mixed);
            if (impl_.insert_node(n, mixed, sk, k) == engine::INSERT_OK)
            {
                impl_.maybe_grow();
                return std::make_pair(iterator(this, k, mixed, sk), true);
            }
            engine::destroy_node(n);
            return std::make_pair(iterator(this, k, mixed, sk), false);
        }

        /**
         * @brief Insert a copy of v if its key is not already present.
         *
         * Equivalent to emplace(v). O(1) expected time.
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
         * Equivalent to emplace(std::move(v)). O(1) expected time.
         *
         * @param v the value to move into the set
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
         * @brief Erase the element whose key is k, if present.
         *
         * O(1) expected time.
         *
         * @param k the key of the element to erase
         *
         * @return the number of elements erased (1 if k was present, else
         *         0).
         */
        size_type erase(const key_type& k)
        {
            const std::uint64_t mixed = impl_.mixed_hash(k);
            return impl_.erase_node(k, mixed, impl_.so_key_of(mixed))
                       ? 1
                       : 0;
        }

        /**
         * @brief Erase the element pointed to by pos.
         *
         * No-op when pos is end(). Returns the iterator that followed pos,
         * which may be end(). O(1) expected time.
         *
         * @param pos a valid iterator into this set
         *
         * @return the iterator following the erased element (or end()).
         */
        iterator erase(const_iterator pos)
        {
            iterator next = const_iterator_to_iterator(pos);
            ++next;
            if (!pos.is_end())
            {
                impl_.erase_node(pos.key(), pos.mixed(), pos.sk());
            }
            return next;
        }

        /**
         * @brief Remove all elements from the set.
         *
         * Requires exclusive access: no other thread may be inside an
         * operation when clear() runs (see the class docs).
         */
        void clear()
        {
            impl_.clear_impl();
        }

        // ---- Lookup ----------------------------------------------------------
        /**
         * @brief Find the element whose key is k.
         *
         * O(1) expected time. The returned iterator is weakly consistent:
         * dereferencing it after the element was concurrently erased is
         * undefined (it will be skipped).
         *
         * @param k the key to search for
         *
         * @return an iterator to the element, or end() if k is absent.
         */
        iterator find(const key_type& k)
        {
            const std::uint64_t mixed = impl_.mixed_hash(k);
            const std::uint64_t sk = impl_.so_key_of(mixed);
            typename engine::node* n = nullptr;
            if (impl_.find_node(k, mixed, sk, n))
            {
                return iterator(this, k, mixed, sk);
            }
            return end();
        }

        /**
         * @brief Find the element whose key is k (const overload).
         *
         * O(1) expected time.
         *
         * @param k the key to search for
         *
         * @return a const iterator to the element, or end() if k is
         *         absent.
         */
        const_iterator find(const key_type& k) const
        {
            const std::uint64_t mixed = impl_.mixed_hash(k);
            const std::uint64_t sk = impl_.so_key_of(mixed);
            typename engine::node* n = nullptr;
            if (impl_.find_node(k, mixed, sk, n))
            {
                return const_iterator(this, k, mixed, sk);
            }
            return end();
        }

        /**
         * @brief Return whether the set contains an element with key k.
         *
         * O(1) expected time.
         *
         * @param k the key to search for
         *
         * @return true if k is present, false otherwise.
         */
        bool contains(const key_type& k) const
        {
            typename engine::node* n = nullptr;
            const std::uint64_t mixed = impl_.mixed_hash(k);
            return impl_.find_node(k, mixed, impl_.so_key_of(mixed), n);
        }

        /**
         * @brief Return the number of elements with key k.
         *
         * Since duplicate keys cannot exist, this is 0 or 1. O(1) expected
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

        // ---- Observers -------------------------------------------------------
        /**
         * @brief Return a copy of the allocator the set was constructed
         *        with.
         *
         * Stateful allocators are supported: the exact instance passed at
         * construction is stored and returned.
         *
         * @return the set's allocator.
         */
        allocator_type get_allocator() const
        {
            return alloc_;
        }

        /**
         * @brief Return a copy of the hash function used to map keys to
         *        buckets.
         *
         * @return the set's hasher.
         */
        hasher hash_function() const
        {
            return impl_.hash_function();
        }

        /**
         * @brief Return a copy of the key equality predicate.
         *
         * @return the set's key_equal.
         */
        key_equal key_eq() const
        {
            return impl_.key_eq();
        }

        /**
         * @brief Return the current load factor (size / bucket_count).
         *
         * @return the load factor, or 0.0f when there are no buckets.
         */
        float load_factor() const
        {
            const std::size_t bc = impl_.bucket_count();
            return bc == 0 ? 0.0f : static_cast<float>(impl_.size()) / bc;
        }

        /**
         * @brief Return the current maximum load factor.
         *
         * Growth is triggered when the load factor would exceed this value.
         *
         * @return the maximum load factor.
         */
        float max_load_factor() const
        {
            return impl_.max_load_factor();
        }

        /**
         * @brief Set the maximum load factor.
         *
         * The load factor at which the table grows. The argument must be
         * positive; no rehash happens immediately.
         *
         * @param f the new maximum load factor.
         */
        void max_load_factor(float f)
        {
            impl_.set_max_load_factor(f);
        }

        /**
         * @brief Reserve buckets for at least n elements without
         *        triggering table growth.
         *
         * Pre-allocating the bucket table avoids grow-time latency during
         * a burst of inserts. O(n) expected time.
         *
         * @param n the number of elements to reserve capacity for.
         */
        void reserve(size_type n)
        {
            impl_.reserve(n);
        }

        /**
         * @brief Return the number of buckets in the bucket table.
         *
         * @return the current bucket count.
         */
        size_type bucket_count() const
        {
            return impl_.bucket_count();
        }

        // ---- Iterators --------------------------------------------------------
        /**
         * @brief Return an iterator to the first element.
         *
         * Iterators are weakly consistent: elements erased concurrently
         * during iteration are skipped. O(1) expected time (locating the
         * first element).
         *
         * @return an iterator to the first element, or end() if the set is
         *         empty.
         */
        iterator begin()
        {
            typename engine::node_info info;
            if (impl_.first_node(info))
            {
                return iterator(this, info.key, info.mixed, info.sk);
            }
            return end();
        }

        /**
         * @brief Return a const iterator to the first element.
         *
         * @return a const iterator to the first element, or end() if the
         *         set is empty.
         */
        const_iterator begin() const
        {
            typename engine::node_info info;
            if (impl_.first_node(info))
            {
                return const_iterator(this, info.key, info.mixed, info.sk);
            }
            return end();
        }

        /**
         * @brief Return a const iterator to the first element.
         *
         * @return a const iterator to the first element, or cend() if the
         *         set is empty.
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
            using value_type = typename unordered_set::value_type;
            using difference_type = typename unordered_set::difference_type;
            using pointer = typename unordered_set::pointer;
            using reference = typename unordered_set::reference;

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
                , mixed_(0)
                , sk_(0)
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
             * erased concurrently are skipped. O(1) expected time.
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
             * iterators from different sets is false (unless both are
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
                return sk_ == o.sk_ && owner_->key_eq()(key_, o.key_);
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
            friend class unordered_set;
            friend class const_iterator;

            explicit iterator(const unordered_set* owner)
                : owner_(owner)
                , is_end_(true)
                , mixed_(0)
                , sk_(0)
            {
            }

            iterator(const unordered_set* owner, const Key& k,
                     std::uint64_t mixed, std::uint64_t sk)
                : owner_(owner)
                , is_end_(false)
                , key_(k)
                , mixed_(mixed)
                , sk_(sk)
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

            std::uint64_t mixed() const
            {
                return mixed_;
            }

            std::uint64_t sk() const
            {
                return sk_;
            }

            value_type* resolve() const
            {
                return owner_->impl_.find_value_ptr(key_, mixed_, sk_);
            }

            void advance()
            {
                typename engine::node_info info;
                if (owner_->impl_.next_node(mixed_, sk_, key_, info))
                {
                    key_ = info.key;
                    mixed_ = info.mixed;
                    sk_ = info.sk;
                }
                else
                {
                    is_end_ = true;
                }
            }

            const unordered_set* owner_;
            bool is_end_;
            Key key_;
            std::uint64_t mixed_;
            std::uint64_t sk_;
        };

        class const_iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = typename unordered_set::value_type;
            using difference_type = typename unordered_set::difference_type;
            using pointer = typename unordered_set::const_pointer;
            using reference = typename unordered_set::const_reference;

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
                , mixed_(0)
                , sk_(0)
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
                , mixed_(it.mixed_)
                , sk_(it.sk_)
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
             * erased concurrently are skipped. O(1) expected time.
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
                return sk_ == o.sk_ && owner_->key_eq()(key_, o.key_);
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
            friend class unordered_set;

            explicit const_iterator(const unordered_set* owner)
                : owner_(owner)
                , is_end_(true)
                , mixed_(0)
                , sk_(0)
            {
            }

            const_iterator(const unordered_set* owner, const Key& k,
                           std::uint64_t mixed, std::uint64_t sk)
                : owner_(owner)
                , is_end_(false)
                , key_(k)
                , mixed_(mixed)
                , sk_(sk)
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

            std::uint64_t mixed() const
            {
                return mixed_;
            }

            std::uint64_t sk() const
            {
                return sk_;
            }

            const value_type* resolve() const
            {
                return owner_->impl_.find_value_ptr(key_, mixed_, sk_);
            }

            void advance()
            {
                typename engine::node_info info;
                if (owner_->impl_.next_node(mixed_, sk_, key_, info))
                {
                    key_ = info.key;
                    mixed_ = info.mixed;
                    sk_ = info.sk;
                }
                else
                {
                    is_end_ = true;
                }
            }

            const unordered_set* owner_;
            bool is_end_;
            Key key_;
            std::uint64_t mixed_;
            std::uint64_t sk_;
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
            return iterator(c.owner_, c.key_, c.mixed_, c.sk_);
        }

    private:
        allocator_type alloc_;
        engine impl_;
    };
}

#endif // MPMC_UNORDERED_SET_HPP
