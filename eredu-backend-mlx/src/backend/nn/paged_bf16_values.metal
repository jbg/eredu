// Extend the contiguous BF16 value product in the original reduction lanes.
#pragma clang fp reassociate(off)
uint feature = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
uint lane = thread_position_in_grid.z;
uint rows = uint(geometry[0]);
uint width = uint(geometry[1]);
uint first_lane = uint(geometry[2]);
uint interleaved = uint(geometry[3]);
uint queries = uint(geometry[4]);
uint heads_per_value_head = uint(geometry[5]);
uint features = uint(geometry[6]);
if (feature >= features || row >= rows) return;

size_t output_index = (size_t(row) * features + feature) * LANES + lane;
float sum = previous[output_index];
size_t probability_base = size_t(row) * width;
size_t value_base = size_t(row / queries / heads_per_value_head) * width * features + feature;
uint first = (lane + LANES - first_lane) % LANES;
uint main_width = LANES == 1 ? width : interleaved;
for (uint key = first; key < main_width; key += LANES) {
    sum += float(probabilities[probability_base + key]) * float(values[value_base + size_t(key) * features]);
}
if (lane == 0) {
    for (uint key = main_width; key < width; ++key) {
        sum += float(probabilities[probability_base + key]) * float(values[value_base + size_t(key) * features]);
    }
}
output[output_index] = sum;
