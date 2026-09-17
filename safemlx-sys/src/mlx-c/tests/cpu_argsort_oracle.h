// Test/benchmark oracle for the pre-existing CPU stable_sort comparator.
#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace cpu_argsort_test {
inline uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
inline float from_bits(uint32_t x) { return std::bit_cast<float>(x); }

inline void ordinary_sort(
    const float* values, int64_t stride, std::vector<uint32_t>& ids) {
  std::iota(ids.begin(), ids.end(), uint32_t(0));
  std::stable_sort(ids.begin(), ids.end(), [values, stride](uint32_t a, uint32_t b) {
    const auto x = values[static_cast<int64_t>(a) * stride];
    const auto y = values[static_cast<int64_t>(b) * stride];
    if (std::isnan(x)) return false;
    if (std::isnan(y)) return true;
    return x < y || (x == y && a < b);
  });
}

inline std::vector<uint32_t> ordinary(
    const float* values, int64_t stride, size_t count) {
  std::vector<uint32_t> ids(count);
  ordinary_sort(values, stride, ids);
  return ids;
}

// Stable seed and exact bit patterns; no standard-library distribution policy.
inline std::vector<float> values(size_t count, unsigned profile) {
  std::vector<float> result(count);
  uint32_t state = 0x4e554d31;
  for (size_t i = 0; i < count; ++i) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    switch (profile) {
      case 0: result[i] = static_cast<float>(state % 100003) * 0.125f; break;
      case 1: result[i] = static_cast<float>(i); break;
      case 2: result[i] = static_cast<float>(count - i); break;
      case 3: result[i] = 7.0f; break;
      case 4: result[i] = static_cast<float>((i * 17) % 31) - 15.0f; break;
      case 5: result[i] = static_cast<float>(std::min(i, count - i)); break;
      default: result[i] = from_bits(state); break;
    }
  }
  return result;
}
} // namespace cpu_argsort_test
