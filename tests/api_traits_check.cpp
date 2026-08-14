#include <type_traits>
#include "mpmc/mpmc_unordered_map.hpp"
#include "mpmc/mpmc_unordered_set.hpp"
#include "mpmc/mpmc_map.hpp"
#include "mpmc/mpmc_set.hpp"

using UM = mpmc::unordered_map<int, int>;
using US = mpmc::unordered_set<int>;
using M  = mpmc::map<int, int>;
using S  = mpmc::set<int>;

static_assert(!std::is_copy_constructible<UM>::value, "UM copy ctor must be deleted");
static_assert(!std::is_copy_assignable<UM>::value,    "UM copy assign must be deleted");
static_assert(!std::is_move_constructible<UM>::value, "UM move ctor must be absent");
static_assert(!std::is_move_assignable<UM>::value,    "UM move assign must be absent");
static_assert(!std::is_copy_constructible<US>::value, "US copy ctor must be deleted");
static_assert(!std::is_copy_assignable<US>::value,    "US copy assign must be deleted");
static_assert(!std::is_move_constructible<US>::value, "US move ctor must be absent");
static_assert(!std::is_move_assignable<US>::value,    "US move assign must be absent");
static_assert(!std::is_copy_constructible<M>::value,  "M copy ctor must be deleted");
static_assert(!std::is_copy_assignable<M>::value,     "M copy assign must be deleted");
static_assert(!std::is_move_constructible<M>::value,  "M move ctor must be absent");
static_assert(!std::is_move_assignable<M>::value,     "M move assign must be absent");
static_assert(!std::is_copy_constructible<S>::value,  "S copy ctor must be deleted");
static_assert(!std::is_copy_assignable<S>::value,     "S copy assign must be deleted");
static_assert(!std::is_move_constructible<S>::value,  "S move ctor must be absent");
static_assert(!std::is_move_assignable<S>::value,     "S move assign must be absent");

static_assert(std::is_default_constructible<UM>::value, "UM default ctor");
static_assert(std::is_default_constructible<US>::value, "US default ctor");
static_assert(std::is_default_constructible<M>::value,  "M default ctor");
static_assert(std::is_default_constructible<S>::value,  "S default ctor");

int main() { return 0; }