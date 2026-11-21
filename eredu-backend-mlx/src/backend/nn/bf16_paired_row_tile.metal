// Each eight-lane subgroup shares activation loads across two output columns.
// Each output keeps the 32 ordered FP32 chains of the one-column row tile.
constexpr uint COLUMNS = 2;
constexpr uint TILE_ROWS = 4;
uint lane = thread_position_in_threadgroup.x;
uint chain = lane % 8;
uint first_column = (thread_position_in_grid.y * 4 + lane / 8) * COLUMNS;
uint first_row = thread_position_in_grid.z * TILE_ROWS;
uint valid_rows = min(uint(TILE_ROWS), uint(ROWS) - first_row);
device const auto* weights = reinterpret_cast<device const packed_ushort4*>(weight);
device const auto* activations = reinterpret_cast<device const packed_ushort4*>(input);
constexpr uint vectors = WIDTH / 4;
size_t weight_base = size_t(first_column) * vectors;
size_t input_base = size_t(first_row) * vectors;
float4 sums[TILE_ROWS][COLUMNS] = {};
if (valid_rows == TILE_ROWS) {
    #pragma clang loop unroll_count(UNROLL)
    for (uint k = chain; k < vectors; k += 8) {
        float4 w[COLUMNS];
        #pragma unroll
        for (uint c = 0; c < COLUMNS; ++c)
            w[c] = as_type<float4>(uint4(ushort4(weights[weight_base + size_t(c) * vectors + k])) << 16);
        #pragma unroll
        for (uint row = 0; row < TILE_ROWS; ++row) {
            float4 x = as_type<float4>(uint4(ushort4(activations[input_base + size_t(row) * vectors + k])) << 16);
            #pragma unroll
            for (uint c = 0; c < COLUMNS; ++c) sums[row][c] += x * w[c];
        }
    }
} else {
    for (uint k = chain; k < vectors; k += 8) {
        float4 w[COLUMNS];
        #pragma unroll
        for (uint c = 0; c < COLUMNS; ++c)
            w[c] = as_type<float4>(uint4(ushort4(weights[weight_base + size_t(c) * vectors + k])) << 16);
        for (uint row = 0; row < valid_rows; ++row) {
            float4 x = as_type<float4>(uint4(ushort4(activations[input_base + size_t(row) * vectors + k])) << 16);
            #pragma unroll
            for (uint c = 0; c < COLUMNS; ++c) sums[row][c] += x * w[c];
        }
    }
}
#pragma unroll
for (uint row = 0; row < TILE_ROWS; ++row) {
    #pragma unroll
    for (uint c = 0; c < COLUMNS; ++c) {
        float4 sum = sums[row][c];
        sum += simd_shuffle_down(sum, 4);
        sum += simd_shuffle_down(sum, 2);
        sum += simd_shuffle_down(sum, 1);
        if (chain == 0 && row < valid_rows) {
            output[size_t(first_row + row) * OUTPUTS + first_column + c] =
                bfloat16_t((sum.x + sum.y) + (sum.z + sum.w));
        }
    }
}
