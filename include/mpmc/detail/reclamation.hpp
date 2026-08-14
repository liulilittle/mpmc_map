#ifndef MPMC_RECLAMATION_HPP
#define MPMC_RECLAMATION_HPP

/**
 * @file reclamation.hpp
 * @brief Reclamation-scheme selection for the containers.
 *
 * Two schemes are implemented and unit-tested identically:
 *
 *   * reclamation::hazard_pointer -- Michael-style hazard pointers. Each
 *     pointer dereference in a traversal is published to a per-thread
 *     hazard slot and re-validated; an object is freed only after a scan
 *     shows that no slot publishes it.
 *   * reclamation::epoch -- a three-value global epoch with per-operation
 *     enter/exit and per-thread retire lists; an object is freed only
 *     after the global epoch has passed the epoch it was retired in by
 *     two generations.
 *
 * The containers are parameterized on the scheme (default
 * reclamation::default_scheme). The default is a compile-time decision
 * made from the benchmark results in docs/PERF_en.md; both schemes pass
 * the same unit, stress and leak test suites.
 */

#include "epoch.hpp"
#include "hazard_pointer.hpp"

namespace mpmc
{
    namespace reclamation
    {
        /**
         * @brief Hazard-pointer reclamation scheme.
         */
        struct hazard_pointer
        {
            template <typename Tag>
            using reclaimer = detail::hazard_pointer_reclaimer<Tag>;
        };

        /**
         * @brief Epoch-based reclamation scheme.
         */
        struct epoch
        {
            template <typename Tag>
            using reclaimer = detail::epoch_reclaimer<Tag>;
        };

        /**
         * @brief The scheme used when a container does not specify one.
         *
         * Chosen from the A/B benchmark results (see docs/PERF_en.md):
         * on the reference machine epoch is faster than hazard pointers
         * for read-heavy workloads (the per-step protect/re-validate cost
         * of hazard pointers outweighs epoch's per-operation enter/exit)
         * and at high thread counts for mixed workloads; the two schemes
         * are within ~15% of each other at 1-2 threads on mixed churn.
         */
        using default_scheme = epoch;
    }
}

#endif // MPMC_RECLAMATION_HPP
