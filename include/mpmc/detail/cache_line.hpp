#ifndef MPMC_CACHE_LINE_HPP
#define MPMC_CACHE_LINE_HPP

/**
 * @file cache_line.hpp
 * @brief Cache-line utilities shared by the lock-free containers.
 *
 * The pad type lets hot fields sit on separate cache lines so that threads
 * writing different fields of the same object do not invalidate each other's
 * lines (false sharing).
 */

#ifndef MPMC_CACHE_LINE
#define MPMC_CACHE_LINE 64
#endif

namespace mpmc
{
    namespace detail
    {
        /**
         * @brief A block of padding bytes of one cache line.
         *
         * Embed an instance before or after a field, or use it with
         * alignas() to place a field on its own cache line.
         */
        struct alignas(MPMC_CACHE_LINE) cache_line_pad
        {
            char bytes[MPMC_CACHE_LINE];
        };
    }
}

#endif // MPMC_CACHE_LINE_HPP
