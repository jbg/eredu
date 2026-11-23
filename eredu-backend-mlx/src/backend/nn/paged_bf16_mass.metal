// Extend the same four interleaved sums used by contiguous BF16 softmax.
#pragma clang fp reassociate(off)
uint lane = thread_position_in_grid.x % 4;
uint row = thread_position_in_grid.x / 4;
uint rows = uint(geometry[0]);
uint width = uint(geometry[1]);
uint start = uint(geometry[2]);
if (row >= rows) return;

size_t base = size_t(row) * width;
float sum = previous[size_t(row) * 4 + lane];
for (uint key = (lane + 4 - start % 4) % 4; key < width; key += 4) {
    float score = scores[base + key];
    if (score != -INFINITY) sum += stable_exp_f32(score - maximum[row]);
}
mass[size_t(row) * 4 + lane] = sum;
