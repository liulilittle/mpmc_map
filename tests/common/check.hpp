#ifndef MPMC_TEST_CHECK_HPP
#define MPMC_TEST_CHECK_HPP

/**
 * @file check.hpp
 * @brief Always-active test assertions.
 *
 * Standard <cassert> is compiled out when NDEBUG is defined (the default
 * in CMake Release builds). Test binaries must verify the same invariants
 * in every build type, so this header provides TEST_CHECK, which prints
 * the failing expression and aborts, and is never compiled out.
 */

#include <cstdio>
#include <cstdlib>

namespace mpmc_test
{
    inline void fail(const char* file, int line, const char* expr)
    {
        std::fprintf(stderr, "FAILED [%s:%d] %s\n", file, line, expr);
        std::fflush(stderr);
        std::abort();
    }
}

#define TEST_CHECK(cond) \
    do                   \
    {                    \
        if (!(cond))     \
        {                \
            mpmc_test::fail(__FILE__, __LINE__, #cond); \
        }                \
    } while (0)

#endif // MPMC_TEST_CHECK_HPP
