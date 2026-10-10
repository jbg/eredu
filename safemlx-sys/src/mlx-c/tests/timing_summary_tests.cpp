#include "doctest/doctest.h"
#include "mlx/backend/common/timing_intervals.h"

TEST_CASE("timing interval union preserves overlap gaps and callback order") {
  // Overlap [1,3]/[2,5], touching [5,6], gap [6,8], then [8,10].
  const auto stats = mlx::core::summarize_timing_intervals(
      {{2, 5}, {8, 10}, {1, 3}, {5, 6}}, 8, 0);
  CHECK(stats == mlx::core::TimingSummary{8, 7, 9, 4, 0});
  const auto nested = mlx::core::summarize_timing_intervals(
      {{1, 9}, {3, 4}, {1, 9}}, 17, 0);
  CHECK(nested == mlx::core::TimingSummary{17, 8, 8, 3, 0});
}

TEST_CASE("timing interval union excludes reserved slots and reports missing timestamps") {
  const auto stats = mlx::core::summarize_timing_intervals(
      {{0, 0}, {2, 4}, {0, 0}}, 2, 1);
  CHECK(stats == mlx::core::TimingSummary{2, 2, 2, 1, 1});
  CHECK(mlx::core::summarize_timing_intervals({}, 0, 0) ==
        mlx::core::TimingSummary{0, 0, 0, 0, 0});
}
