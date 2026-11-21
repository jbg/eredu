// Each eight-lane subgroup computes one output column. Four adjacent BF16
// values give each thread four independent chains of the original row dot.
// The packed vector type also accepts contiguous views at BF16-aligned offsets.
uint lane = thread_position_in_threadgroup.x;
uint chain_group = lane % 8;
uint column = thread_position_in_grid.y * 4 + lane / 8;
uint first_row = thread_position_in_grid.z * TILE_ROWS;
uint valid_rows = min(uint(TILE_ROWS), uint(ROWS) - first_row);
device const auto* weights = reinterpret_cast<device const packed_ushort4*>(weight);
device const auto* activations = reinterpret_cast<device const packed_ushort4*>(input);
constexpr uint vectors = WIDTH / 4;
size_t weight_base = size_t(column) * vectors;
float4 sums[TILE_ROWS] = {};

if (valid_rows == TILE_ROWS) {
    #pragma clang loop unroll_count(TILE_ROWS == 1 ? 8 : 2)
    for (uint k = chain_group; k < vectors; k += 8) {
        float4 w = as_type<float4>(uint4(ushort4(weights[weight_base + k])) << 16);
        #pragma unroll
        for (uint row = 0; row < TILE_ROWS; ++row) {
            size_t offset = size_t(first_row + row) * vectors + k;
            float4 x = as_type<float4>(uint4(ushort4(activations[offset])) << 16);
            sums[row] += x * w;
        }
    }
} else {
    #pragma unroll 2
    for (uint k = chain_group; k < vectors; k += 8) {
        float4 w = as_type<float4>(uint4(ushort4(weights[weight_base + k])) << 16);
        for (uint row = 0; row < valid_rows; ++row) {
            size_t offset = size_t(first_row + row) * vectors + k;
            float4 x = as_type<float4>(uint4(ushort4(activations[offset])) << 16);
            sums[row] += x * w;
        }
    }
}

// Shuffles by 4, 2 and 1 now combine original chains 16, 8 and 4 apart.
// The final scalar merge has the same pairing as the original four lanes.
#pragma unroll
for (uint row = 0; row < TILE_ROWS; ++row) {
    float4 sum = sums[row];
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 1);
    if (chain_group == 0 && row < valid_rows) {
        output[size_t(first_row + row) * OUTPUTS + column] =
            bfloat16_t((sum.x + sum.y) + (sum.z + sum.w));
    }
}
