# Performance

Machine: Windows x64, clang++ 17 (MinGW) Release `-O2`, ninja build.
All numbers are operations per second (higher is better); "speedup" =
A / B. Raw tables are kept in `bench/results.md`; the benchmarks in
`bench/` rebuild with the rest of the project.

The reference machine's background load fluctuates (35-90% depending on
the moment), so multi-thread absolute numbers vary between runs. The
tables below are ranges across runs; the qualitative conclusions were
stable in every run. Single-threaded microbenchmark numbers are
load-tolerant and precise. Compare in a quiet window.

## Optimizations in place

The following optimizations are present in the code measured below.
They are grouped by mechanism rather than by chronology; each one is
described in [docs/DESIGN_en.md](docs/DESIGN_en.md).

### Reclamation and the guard path

* **Per-node reclamation pooling** (both engines): retired nodes are
  recycled through a per-instantiation pool instead of returning to the
  allocator on every churn. Duplicate insert into a warm unordered
  table: 81.6 to 41.5 ns/op.
* **Epoch fast walk**: the `NEEDS_REVALIDATION` compile-time trait
  removes all hazard-slot `protect()` calls and re-validation loads
  when the epoch scheme is active (the per-operation guard already
  proves safety), and the container code paths are identical for both
  schemes.
* **Epoch enter/exit** is one atomic load + one relaxed increment on
  entry and one release decrement on exit; no per-dereference cost.

### Resize path (unordered)

* **Cached growth threshold**: the common insert path reads a relaxed
  cached threshold instead of loading `table_` (a separate cache line,
  acquire). The cache is always a lower bound of the true threshold, so
  a stale value only errs toward the slow path.
* **Multi-grow**: one `growing_` acquisition loops to double multiple
  times when `count_` far exceeds the threshold (burst-insert
  catch-up), reducing CAS acquisitions.

### List walks (unordered)

* **Single-load pattern everywhere**: every dependent-load-chain walk
  (`locate`, `insert_from`, `first_node`, `next_node` phases 1/2, and
  the `find_node`/`insert_node`/`erase_node` slow-path run scans) loads
  `cur->next` exactly once per iteration. A second load was previously
  used in several loops for the prefetch address or the new-node marked
  check; on x86 a relaxed and an acquire atomic load compile to the
  same instruction, so the extra load was pure cost.
* **Software prefetch** (`_mm_prefetch`, T0) in every walk loop: the
  next node's cache line is fetched before the acquire load/use, hiding
  L2/L3 latency behind the CAS or protection work. The prefetch address
  is the tag-cleared pointer (the raw value may carry the mark bit).
  Compiles to no-ops on non-x86.
* **Fast paths**: find, insert and erase all check the first node of
  the target run before entering the scan loop; in the common case
  (run length 1, no collision) the loop is skipped entirely.
* **operator[] direct return**: on a successful insert the mapped value
  is returned directly from the just-linked node -- no second lookup.
* **clear() single load**: the clear walk reads `cur->next` once per
  node.

### Tree

* **Lock-bit bracket validation**: the lock-bit protocol itself (see
  DESIGN) was analyzed and confirmed; the `compare()` between the two
  lock-bit loads is safe because the key is immutable.
* **Relaxed child-pointer loads**: the child pointer is loaded with
  `memory_order_relaxed`; the first `locked.load(acquire)` already
  publishes it (the mutator wrote it under the lock and released via
  `locked.store(false, release)`), and the second acquire verifies
  stability. This removes one redundant fence per node visited.
* **Cache-line-aware node layout**: hot search fields (`left`, `right`,
  `locked`, `color`) occupy the start of the node's first cache line;
  cold fields (`payload`, `alloc_owner`) follow. A search visits only
  the hot fields of each node.
* **Software prefetch** in `search`, `bound_node`, `first_node`: the
  child node is prefetched between the child-pointer relaxed load and
  the second lock-bit acquire, hiding 50-100 cycles of L2/L3 latency
  behind the bracket validation. Microbenchmark: tree find ~35-39 ns/op
  (see below).

### Reclamation correctness (affects throughput under churn)

* **`retired_total` accuracy**: the per-thread counter that gates the
  `reclaim()` threshold is decremented when objects are freed and
  recomputed from the surviving tagged lists after partial reclaims.
  Previously it only grew, so `reclaim()` fired more often than
  necessary and scanned already-empty lists.
* **`drain_late` gating**: `reclaim()` drains the shared late-retire
  chain only when the chain is non-empty (a contention-free relaxed
  load) and the global epoch has advanced since this thread last
  drained. Under sustained multi-thread activity the epoch can stay
  frozen for long stretches; without the gate, every per-retire reclaim
  performs a global locked exchange on the chain head, and after
  threads exit and push their backlogs, survivors re-walk and re-push
  the whole chain on every retire. The gating is what keeps the epoch
  scheme's per-retire reclaim cost O(1) in steady state (see
  [8-thread mixed mode](#8-thread-mixed-mode) below).

### Node pool

* **Sharded pool**: 128 cache-line-separated shards, one per thread,
  so concurrent pool traffic never touches the same cache line. A
  single shared lock measured ~12,100 cycles per pool operation at 16
  threads (about 55% of all thread time); sharding removed that
  bottleneck (see [16-thread scaling](#16-thread-scaling) below).

## Methodology

Two workloads run at 1, 2, 4 and 8 threads (and 16 for the scaling
check):

* **mixed** -- 40% insert / 40% erase / 20% find on a per-thread
  private key range (no cross-thread conflicts) plus a find of a small
  shared range every 8th operation;
* **read-heavy** -- 10% insert / 10% erase / 80% find on the private
  range plus a shared find every 4th operation.

Each cell is one deterministic, reproducible run (fixed per-thread RNG
seeds); the reference numbers use the default `epoch` reclamation
scheme.

## Reclamation A/B (mpmc::unordered_map, hazard vs epoch)

| threads | mixed hazard | mixed epoch | speedup | read hazard | read epoch | speedup |
|---------|-------------|-------------|---------|-------------|------------|---------|
| 1       | 21.4-28.8M  | 18.5-20.9M  | 1.02-1.38x | 19.3-30.6M  | 34.5-43.3M  | 0.56-0.76x |
| 2       | 10.8-26.2M  | 16.4-27.9M  | 0.66-1.07x | 32.1-48.6M  | 58.5-76.7M  | 0.55-0.72x |
| 4       | 6.9-31.6M   | 16.7-32.5M  | 0.41-1.02x | 25.6-84.7M  | 45.3-131.5M | 0.48-0.72x |
| 8       | 4.2-16.0M   | 6.7-22.6M   | 0.63-1.02x | 15.6-151.2M | 24.6-224.9M | 0.55-0.71x |

Epoch wins read-heavy workloads at every thread count and mixed
workloads from 4 threads up (in quiet runs); hazard leads only at 1-2
threads on mixed churn, by up to ~38% (1 thread) but typically
~7-16%. This is why `default_scheme` is `epoch`
(see [docs/RECLAMATION_en.md](docs/RECLAMATION_en.md)).

## Unordered containers vs std behind a mutex

`mpmc::unordered_map` (default scheme) vs `std::unordered_map` guarded
by a single `std::mutex` (range across runs; the lower bound is under
heavy background load):

| threads | mixed | read-heavy |
|---------|-------|-----------|
| 1       | 0.57-0.62x | 0.57-0.67x |
| 2       | 0.63-0.75x | 1.19-1.25x |
| 4       | 0.70-1.75x | 1.62-3.21x |
| 8       | 0.56-2.48x | 2.22-12.91x |

At 1 thread the lock-free machinery (atomics, tagging, reclamation
guards) costs roughly 40% versus the plain std implementation; from 4
threads in quiet runs the mutex serialization dominates and the
lock-free table pulls ahead. Under heavy external load the lock-free
container's 4-8 thread mixed-mode gain compresses first (its scaling
suffers when the machine is saturated).

A fresh run at 2M ops/thread gives a representative single sample:

| threads | mixed mpmc | mixed std | speedup | read mpmc | read std | speedup |
|---------|------------|-----------|---------|-----------|----------|---------|
| 1       | 21.5M      | 35.4M     | 0.61x   | 46.4M     | 74.7M    | 0.62x   |
| 2       | 26.6M      | 28.4M     | 0.94x   | 76.3M     | 57.2M    | 1.33x   |
| 4       | 27.1M      | 25.0M     | 1.09x   | 145.1M    | 38.9M    | 3.73x   |
| 8       | 18.5M      | 14.7M     | 1.26x   | 252.3M    | 25.0M    | 10.11x  |

Mixed mode overtakes `std`+mutex from 4 threads; read-heavy overtakes
from 2 threads.

### 8-thread mixed mode

An earlier version of the library collapsed to ~25k ops/s at 8 threads
with 40% insert / 40% erase. The cause was misattributed to resize
contention for a while; instrumentation proved it was the epoch
reclaimer's `drain_late` amplification:

1. Under sustained multi-thread activity the global epoch almost never
   advances (threads' op guards overlap, so the active count never
   reaches zero) -- `free_tagged` never frees, and the per-thread
   retired vectors grow without bound.
2. `reclaim()` re-derives `retired_total` from the vector sizes after
   each call, so the threshold stays permanently exceeded and
   `reclaim()` fires on every retire.
3. Every `reclaim()` called `drain_late()`, whose chain-head exchange
   is a global locked read-modify-write on one cache line shared by all
   threads. Measured: ~264k cycles (~66 us) per call, `drain_late`
   consuming ~66% of all thread time while the chain was empty.
4. When threads exit, `finalize_thread` pushes their whole backlog onto
   the chain as orphans; the surviving threads' per-retire reclaims
   then exchange, walk and re-push the entire chain -- O(chain) per
   retire -- and the orphans' tags never match the frozen current
   epoch, so they are re-pushed forever.

The fix gates `drain_late` on the chain being non-empty and the epoch
having advanced since this thread last drained (see
[Optimizations in place](#optimizations-in-place)). Deferred drains are
safe: orphans stay on the chain until their tag becomes reclaimable,
and the destruction path frees everything.

Result: 8-thread mixed throughput went from ~25k to ~7.5M ops/s, and
after the pool sharding (below) to ~18.5M ops/s (1.26x vs std+mutex).
The hazard scheme never had the drop: its scan is threshold-gated on a
per-thread list that shrinks as nodes become unprotected -- it has no
frozen-epoch equivalent.

### 16-thread scaling

The node pool was the last structural bottleneck after the `drain_late`
fix: its single shared spinlock measured ~12,100 cycles per pool
operation at 16 threads (~55% of all thread time). The pool is now
sharded into 128 cache-line-separated shards, one per thread (assigned
by one amortized global fetch-add on first use; the shard index is
thread-local, the pool remains a per-instantiation static). Allocator
balance is preserved: each block still carries its `alloc_owner`, each
shard drains at its share of the watermark, and the container
destructor drains all shards.

Result (mixed workload, epoch scheme, 200k ops/thread):

| threads | ops/s |
|---------|-------|
| 4       | 40.2M |
| 8       | 39.7M |
| 16      | 41.5M |

Aggregate mixed throughput is flat from 4 to 16 threads; read-only
throughput at 16 threads is 742M ops/s. The residual per-op cost is the
algorithmic churn cost of the shared lock-free list (CAS, guards,
pool); no single dominant lever remains.

## Ordered containers vs std behind a mutex

`mpmc::map` (default scheme) vs `std::map` guarded by a single
`std::mutex`:

| threads | mixed | read-heavy |
|---------|-------|-----------|
| 1       | 0.75-0.88x | 0.37-0.42x |
| 2       | 0.55-0.61x | 0.35-0.37x |
| 4       | 0.38-0.47x | 0.53-0.57x |
| 8       | 0.28-0.36x | 0.61-0.83x |

The read-heavy 1-thread gap (0.40x) is the inherent per-search cost of
lock-free linearizability: two acquire loads of the lock bit per node
visited, plus epoch guard enter/exit. `std::map`+mutex at 1 thread pays
only one uncontended mutex lock+unlock (~20 ns). As the thread count
grows the mutex serializes and the lock-free tree closes the gap (0.83x
at 8 threads read-heavy, trend continuing upward).

## Microbenchmarks (single-threaded)

| measurement | value |
|-------------|-------|
| epoch guard enter/exit | ~4.7-5.0 ns/op |
| unordered find (warm, small table) | ~2.7-3.2 ns/op |
| tree find | ~35-40 ns/op |

## Interpretation (read this before choosing the ordered containers)

The red-black tree's update path locks every node on the descent
(root-to-leaf, hand-over-hand) and holds the root's lock for the whole
update, so **updates are fully serialized at a higher per-op cost than
a single-mutex std::map** -- the tree is slower than std+mutex for
update-heavy workloads at every measured thread count. Its value is
elsewhere:

* searches are **lock-free**: they never block, and they never make
  writers wait -- a mutex-protected map forces readers and writers to
  wait for each other;
* there is no single contended cache line on the read path: readers
  scale with the number of cores once the workload is read-dominated.

Choose `mpmc::map` when reads dominate, latencies must never be blocked
by a writer, or thread counts are high; choose `std::map` + mutex (or a
read-write lock) for small, update-heavy workloads.

The unordered containers do not have the tree's serialization caveat:
they are lock-free on every path, including resize, and in quiet runs
beat the mutex baseline from 4 threads in both workloads (their scaling
under external saturation degrades first -- measure in a quiet window).

## Reproducing

```sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++
cmake --build build
./build/bench_reclamation 2000000   # hazard vs epoch A/B
./build/bench_unordered   2000000   # mpmc vs std+mutex
./build/bench_tree        1000000   # ordered, same comparison
```

Machine dependence: absolute numbers and crossover points vary with
core count and atomics cost; the qualitative conclusions above held
across repeated runs on the reference machine.
