/**
 * @file compile_fail.cpp
 * @brief Negative compile test: the containers must REJECT key types whose
 *        copy constructor can throw (the iterators and the engine rely on
 *        nothrow key copies inside protected regions).
 *
 * This file is not built into an executable; it is compiled by the
 * configure-time try_compile check in CMakeLists.txt, which fails the
 * configuration when it unexpectedly compiles.
 */

#include "mpmc/mpmc_map.hpp"
#include "mpmc/mpmc_set.hpp"
#include "mpmc/mpmc_unordered_map.hpp"
#include "mpmc/mpmc_unordered_set.hpp"

namespace
{
    struct throwing_key
    {
        throwing_key() {}
        throwing_key(const throwing_key&) noexcept(false) {}
        throwing_key& operator=(const throwing_key&) { return *this; }
    };

    struct bad_hash
    {
        std::size_t operator()(const throwing_key&) const { return 0; }
    };

    struct bad_eq
    {
        bool operator()(const throwing_key&, const throwing_key&) const
        {
            return true;
        }
    };

    mpmc::unordered_map<throwing_key, int, bad_hash, bad_eq> map;
    mpmc::unordered_set<throwing_key, bad_hash, bad_eq> set;
    mpmc::map<throwing_key, int> tree_map;
    mpmc::set<throwing_key> tree_set;
}

int main()
{
    map.emplace(throwing_key(), 1);
    set.emplace(throwing_key());
    tree_map.emplace(throwing_key(), 1);
    tree_set.emplace(throwing_key());
    return 0;
}
