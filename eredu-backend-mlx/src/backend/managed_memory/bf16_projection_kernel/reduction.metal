uint lane = thread_position_in_grid.x;
uint col = thread_position_in_grid.y;
uint row = thread_position_in_grid.z;
uint width = input_shape[input_ndim - 1];
uint outputs = threads_per_grid.y;
uint group = GROUPED ? uint(groups[row]) : 0;
size_t base = (size_t(group) * outputs + col) * width;
float sum = 0.0f;
if (COLUMNS) {
  if (lane < 4) {
    for (uint k = lane; k + 3 - lane < width; k += 4)
      sum += float(input[size_t(row) * width + k]) * float(weight[base + k]);
  }
  if (lane == 0) {
    for (uint k = (width / 4) * 4; k < width; ++k)
      sum += float(input[size_t(row) * width + k]) * float(weight[base + k]);
  }
  float s1 = simd_shuffle(sum, 1), s2 = simd_shuffle(sum, 2), s3 = simd_shuffle(sum, 3);
  if (lane == 0)
    output[size_t(row) * outputs + col] = bfloat16_t(((sum + s1) + s2) + s3);
  return;
}
for (uint k = lane; k < width; k += 32)
  sum += float(input[size_t(row) * width + k]) * float(weight[base + k]);
sum += simd_shuffle_down(sum, 16);
sum += simd_shuffle_down(sum, 8);
sum += simd_shuffle_down(sum, 4);
float s1 = simd_shuffle(sum, 1), s2 = simd_shuffle(sum, 2), s3 = simd_shuffle(sum, 3);
if (lane == 0)
  output[size_t(row) * outputs + col] = bfloat16_t((sum + s1) + (s2 + s3));
