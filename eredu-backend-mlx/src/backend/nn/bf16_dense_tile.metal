// Each SIMD group reuses two weight rows across the input tile and reuses
// each input value across two output columns. Each dot retains its lane order.
uint lane = thread_position_in_threadgroup.x;
uint first_column = thread_position_in_grid.y * 2;
uint first_row = thread_position_in_grid.z * TILE_ROWS;
size_t weight_base = size_t(first_column) * WIDTH;
float2 sums[TILE_ROWS] = {};
uint valid_rows = min(uint(TILE_ROWS), uint(ROWS) - first_row);
if (valid_rows == TILE_ROWS) {
    for (uint k = lane; k < WIDTH; k += 32) {
        float2 w = float2(float(weight[weight_base + k]),
                          float(weight[weight_base + WIDTH + k]));
        #pragma unroll
        for (uint row = 0; row < TILE_ROWS; ++row) {
            sums[row] += float(input[size_t(first_row + row) * WIDTH + k]) * w;
        }
    }
} else {
    for (uint k = lane; k < WIDTH; k += 32) {
        float2 w = float2(float(weight[weight_base + k]),
                          float(weight[weight_base + WIDTH + k]));
        for (uint row = 0; row < valid_rows; ++row) {
            sums[row] += float(input[size_t(first_row + row) * WIDTH + k]) * w;
        }
    }
}

#pragma unroll
for (uint row = 0; row < TILE_ROWS; ++row) {
    float2 sum = sums[row];
    sum += simd_shuffle_down(sum, 16);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    float2 s1 = simd_shuffle(sum, 1);
    float2 s2 = simd_shuffle(sum, 2);
    float2 s3 = simd_shuffle(sum, 3);
    if (lane == 0 && row < valid_rows) {
        size_t offset = size_t(first_row + row) * OUTPUTS + first_column;
        float2 merged = (sum + s1) + (s2 + s3);
        output[offset] = bfloat16_t(merged.x);
        output[offset + 1] = bfloat16_t(merged.y);
    }
}
