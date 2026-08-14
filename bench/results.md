# Benchmark results

Machine: Windows x64, clang++ 17 (MinGW) Release `-O2`. All numbers in
operations per second (higher is better); "speedup" = A / B.
Methodology and interpretation: [docs/PERF_en.md](../docs/PERF_en.md).

The reference machine's background load fluctuates; multi-thread
absolute numbers vary between runs. Tables below are single
representative runs unless marked as a range. The qualitative
conclusions (epoch over hazard for reads; tree updates serialized;
unordered scales with cores) were stable in every run.

## Reclamation A/B (bench_reclamation, mode 0 = mixed, mode 1 = read-heavy)

| threads | hazard m0 | epoch m0 | speedup | hazard m1 | epoch m1 | speedup |
|---------|-----------|----------|---------|-----------|----------|---------|
| 1       | 22.50M    | 18.51M   | 1.22x   | 19.26M    | 34.52M   | 0.56x   |
| 2       | 10.76M    | 16.42M   | 0.66x   | 32.14M    | 58.51M   | 0.55x   |
| 4       | 6.91M     | 16.69M   | 0.41x   | 25.60M    | 45.29M   | 0.57x   |
| 8       | 4.22M     | 6.70M    | 0.63x   | 15.57M    | 24.59M   | 0.63x   |

Repeated runs confirm the pattern: epoch is faster for read-heavy at
every thread count and for mixed from 4 threads up; hazard leads only
at 1 thread mixed (up to ~38%, typically ~7-16%). `default_scheme` is
therefore `epoch`.

## Unordered vs std+mutex (bench_unordered)

Representative run (2M ops/thread):

| threads | mixed mpmc | mixed std | speedup | read mpmc | read std | speedup |
|---------|------------|-----------|---------|-----------|----------|---------|
| 1       | 21.5M      | 35.4M     | 0.61x   | 46.4M     | 74.7M    | 0.62x   |
| 2       | 26.6M      | 28.4M     | 0.94x   | 76.3M     | 57.2M    | 1.33x   |
| 4       | 27.1M      | 25.0M     | 1.09x   | 145.1M    | 38.9M    | 3.73x   |
| 8       | 18.5M      | 14.7M     | 1.26x   | 252.3M    | 25.0M    | 10.11x  |

Ranges across earlier runs (quiet to heavy load):

| threads | mixed | read-heavy |
|---------|-------|-----------|
| 1       | 0.57-0.62x | 0.57-0.67x |
| 2       | 0.63-0.94x | 1.10-1.33x |
| 4       | 0.63-1.75x | 2.02-3.73x |
| 8       | 0.50-2.48x | 2.42-12.91x |

Historical note: an earlier measurement recorded 0.025M ops/s at 8
threads mixed (0.00x). That was the epoch reclaimer's `drain_late`
amplification, since root-caused and fixed (see
[docs/PERF_en.md](../docs/PERF_en.md), "8-thread mixed mode"); the
resize path was measured normal throughout (~13 grows, no CAS herd).

## Ordered vs std+mutex (bench_tree)

Representative run (1M ops/thread):

| threads | mixed mpmc | mixed std | speedup | read mpmc | read std | speedup |
|---------|------------|-----------|---------|-----------|----------|---------|
| 1       | 16.5M      | 18.4M     | 0.90x   | 19.1M     | 48.0M    | 0.40x   |
| 2       | 8.9M       | 15.4M     | 0.58x   | 15.4M     | 45.8M    | 0.34x   |
| 4       | 4.4M       | 11.6M     | 0.38x   | 13.2M     | 22.7M    | 0.58x   |
| 8       | 2.6M       | 6.3M      | 0.41x   | 9.2M      | 11.2M    | 0.83x   |

The update path locks every node on the descent (root-to-leaf,
hand-over-hand) and holds the root's lock for the whole update, so
updates are fully serialized; reads are lock-free and never block. The
read-heavy 1-2 thread gap (0.34-0.40x) is the per-search cost of
lock-free linearizability (two lock-bit acquire loads per node plus
epoch guard enter/exit). The gap closes as threads grow (0.83x at 8
threads read-heavy, trend continuing).

## Microbenchmarks (single-threaded, load-tolerant)

| measurement | value |
|-------------|-------|
| epoch guard enter/exit | ~4.7-5.0 ns/op |
| hazard guard + 2 protects | ~8.0 ns/op |
| unordered dup-insert (warm, pooled) | ~41.5 ns/op (81.6 without pool) |
| unordered steady insert | ~37-42 ns/op |
| unordered find (warm, small table) | ~2.7-3.2 ns/op |
| tree find | ~35-40 ns/op |
| tree insert | ~143 ns/op |
| tree erase | ~74 ns/op |

## Scaling check (mixed workload, epoch scheme)

| threads | ops/s |
|---------|-------|
| 4       | 40.2M |
| 8       | 39.7M |
| 16      | 41.5M |

Aggregate mixed throughput is flat from 4 to 16 threads (a single
shared lock measured ~12k cycles per pool operation at 16 threads and
was the previous bottleneck; the pool is now sharded). Read-only at 16
threads: 742M ops/s. The residual per-op cost is the algorithmic churn
cost of the shared lock-free list.

## Verification record

- 16-thread matrix (both engines, both reclamation schemes), stress
  with counting-allocator balance and per-key oracles: passed under
  Release, UBSan and MSVC AddressSanitizer; endurance runs of 5M
  iterations per thread passed.
- Release and UBSan: 10/10 test suites pass. MSVC ASan: 10/10.
  MSVC `/W4 /WX /utf-8`: all test sources compile with zero warnings.
- Leak checks: `stress_reclamation` retired == freed (3,999,999) with
  zero bad reads; `test_pool` allocs == frees (both engines, both
  schemes).

## Notable fixes and optimizations (summary)

These changes are documented in detail in
[docs/PERF_en.md](../docs/PERF_en.md) and
[docs/DESIGN_en.md](../docs/DESIGN_en.md); this list records their
verification:

- **Epoch `drain_late` gating** -- root cause of the 8-thread mixed
  collapse (~25k ops/s). `reclaim()` now drains the shared late-retire
  chain only when it is non-empty and the global epoch has advanced.
  Restored 8-thread mixed throughput to ~7.5M ops/s.
- **Sharded node pool** -- the pool's single spinlock was the 16-thread
  bottleneck (~55% of thread time). 128 cache-line-separated shards,
  one per thread. 16-thread mixed: 4.0M -> 41.5M ops/s; 8-thread mixed
  7.5M -> 18.5M ops/s.
- **`retired_total` accuracy** -- the epoch reclaim threshold counter
  is now decremented on frees and recomputed after partial reclaims
  (previously it over-counted, causing extra reclaims with empty
  scans).
- **Single-load walk pattern + software prefetch** -- every
  split-ordered list walk loads `cur->next` once per iteration and
  prefetches the next node; tree search prefetches the child between
  the two lock-bit loads. Unordered find ~2.7-3.2 ns/op; tree find
  ~35-40 ns/op.
- **operator[] direct return** -- no second lookup on the insert-success
  path (both maps).
- **Relaxed tree child loads + cache-line-aware node layout** -- fewer
  fences per node on the search path; hot fields in the first cache
  line.
- **get_allocator() fix** -- returns the stored allocator instance
  (stateful allocators supported); regression test added.
- **API conformance** -- full audit of all four containers against the
  API docs; copy/move traits now enforced by compile-time static
  asserts.
- **MSVC warning-clean gate** -- headers are pure ASCII; all test
  sources compile with `/W4 /WX`; the only suppressed warning (C4324)
  is the intentional cache-line alignment.
