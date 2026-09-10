// Fixed four-level FP32 row reduction shared with the CPU normalization mechanism.
template <typename Pointer>
inline float cascade_sum_f32(Pointer values, size_t size) {
  constexpr size_t lanes = 4;
  constexpr size_t parallel = 4;
  const size_t vectors = size / lanes;
  const size_t steps = vectors / parallel;
  size_t ceil_log = 0;
  for (size_t n = steps > 0 ? steps - 1 : 0; n; n >>= 1) {
    ++ceil_log;
  }
  const size_t power = metal::max(size_t(4), ceil_log / 4);
  const size_t chunk = size_t(1) << power;
  float sums[4][parallel][lanes] = {};
  size_t i = 0;
  for (; i + chunk <= steps;) {
    for (size_t end = i + chunk; i < end; ++i) {
      for (size_t p = 0; p < parallel; ++p) {
        for (size_t lane = 0; lane < lanes; ++lane) {
          sums[0][p][lane] += values[(i * parallel + p) * lanes + lane];
        }
      }
    }
    for (size_t level = 1; level < 4; ++level) {
      for (size_t p = 0; p < parallel; ++p) {
        for (size_t lane = 0; lane < lanes; ++lane) {
          sums[level][p][lane] += sums[level - 1][p][lane];
          sums[level - 1][p][lane] = 0;
        }
      }
      if (i & ((chunk - 1) << (level * power))) {
        break;
      }
    }
  }
  for (; i < steps; ++i) {
    for (size_t p = 0; p < parallel; ++p) {
      for (size_t lane = 0; lane < lanes; ++lane) {
        sums[0][p][lane] += values[(i * parallel + p) * lanes + lane];
      }
    }
  }
  for (size_t level = 1; level < 4; ++level) {
    for (size_t p = 0; p < parallel; ++p) {
      for (size_t lane = 0; lane < lanes; ++lane) {
        sums[0][p][lane] += sums[level][p][lane];
      }
    }
  }
  for (size_t v = steps * parallel; v < vectors; ++v) {
    for (size_t lane = 0; lane < lanes; ++lane) {
      sums[0][0][lane] += values[v * lanes + lane];
    }
  }
  for (size_t p = 1; p < parallel; ++p) {
    for (size_t lane = 0; lane < lanes; ++lane) {
      sums[0][0][lane] += sums[0][p][lane];
    }
  }
  float sum = 0;
  for (size_t tail = vectors * lanes; tail < size; ++tail) {
    sum += values[tail];
  }
  for (size_t lane = 0; lane < lanes; ++lane) {
    sum += sums[0][0][lane];
  }
  return sum;
}
