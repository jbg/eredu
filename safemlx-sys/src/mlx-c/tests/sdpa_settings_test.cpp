// A standalone cold-process test: mutate the environment before starting any
// threads, then prove that the native getter retains its first selection.
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "mlx/utils.h"

static void set(const char* value) {
#ifdef _WIN32
  _putenv_s("MLX_SDPA_BLOCKS", value);
#else
  setenv("MLX_SDPA_BLOCKS", value, 1);
#endif
}
int main(int argc, char** argv) {
  if (argc != 3) return 1;
  set(argv[1]);
  const int expected = std::stoi(argv[2]);
  auto check = [expected]() {
    try {
      const int actual = mlx::core::env::sdpa_blocks_override();
      if (expected < 0 || actual != expected) std::abort();
    } catch (const std::invalid_argument&) {
      if (expected >= 0) std::abort();
    }
  };
  check();
  set("4096");
  check();
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) threads.emplace_back(check);
  for (auto& thread : threads) thread.join();
}
