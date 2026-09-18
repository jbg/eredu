// Optional kernel-only timing; caller allocations and graph/stream costs are
// outside the clock. This is neither a heap bound nor an admission benchmark.
#include "cpu_argsort_oracle.h"
#include "mlx/backend/cpu/argsort_f32.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
  using clock = std::chrono::steady_clock;
  constexpr unsigned repeats = 5;
  const char* names[]{"random_finite", "sorted", "reverse", "equal", "dense_ties", "organ_pipe", "float_bits"};
  std::cout << "vocabulary,profile,repetitions,heap_us,stable_sort_us\n";
  for (const size_t width : {16384, 65536, 128256, 262144}) {
    for (unsigned profile = 0; profile < 7; ++profile) {
      const auto values = cpu_argsort_test::values(width, profile);
      std::vector<uint32_t> actual(width), expected(width);
      std::chrono::duration<double, std::micro> heap{}, stable{};
      for (unsigned iteration = 0; iteration < repeats; ++iteration) {
        const auto run_heap = [&] {
          const auto start = clock::now();
          mlx::core::cpu::detail::argsort_row(values.data(), 1, actual.data(), 1, width);
          heap += clock::now() - start;
        };
        const auto run_stable = [&] {
          const auto start = clock::now();
          cpu_argsort_test::ordinary_sort(values.data(), 1, expected);
          stable += clock::now() - start;
        };
        if (iteration % 2 == 0) { run_heap(); run_stable(); }
        else { run_stable(); run_heap(); }
        if (actual != expected) throw std::runtime_error("CPU Argsort benchmark parity failed");
      }
      std::cout << width << ',' << names[profile] << ',' << repeats << ','
                << std::setprecision(10) << heap.count() / repeats << ','
                << stable.count() / repeats << '\n';
    }
  }
}
