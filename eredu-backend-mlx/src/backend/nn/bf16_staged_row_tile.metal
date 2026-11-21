// Four activation rows are shared across 32 output columns.
// Each output retains the same 32 FP32 chains and merge order as the paired tile.
constexpr uint TILE_ROWS = 4;
constexpr uint COLS = 2;
constexpr uint CHUNK = 256;
constexpr uint GROUPS = 4;
uint lane = thread_position_in_threadgroup.x;
uint chain = lane % 8;
uint column = (thread_position_in_grid.y * 4 + lane / 8) * COLS;
uint row0 = thread_position_in_grid.z * TILE_ROWS;
uint valid_rows = min(uint(TILE_ROWS), uint(ROWS) - row0);
constexpr uint vectors = WIDTH / 4;
device const auto* w = reinterpret_cast<device const packed_ushort4*>(weight);
device const auto* x = reinterpret_cast<device const packed_ushort4*>(input);
threadgroup float4 staged[TILE_ROWS][CHUNK];
uint thread_index = thread_position_in_threadgroup.y * 32 + lane;
float4 sum[TILE_ROWS][COLS] = {};
for (uint tile = 0; tile < vectors; tile += CHUNK) {
    uint valid_vectors = min(uint(CHUNK), vectors - tile);
    for (uint i = thread_index; i < TILE_ROWS * CHUNK; i += GROUPS * 32) {
        uint r = i / CHUNK;
        uint k = i % CHUNK;
        if (r < valid_rows && k < valid_vectors) {
            staged[r][k] = as_type<float4>(uint4(ushort4(x[size_t(row0 + r) * vectors + tile + k])) << 16);
        } else {
            staged[r][k] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll_count(2)
    for (uint k = chain; k < valid_vectors; k += 8) {
        float4 weights[COLS];
        #pragma unroll
        for (uint c = 0; c < COLS; ++c) {
            weights[c] = as_type<float4>(uint4(ushort4(w[size_t(column + c) * vectors + tile + k])) << 16);
        }
        #pragma unroll
        for (uint r = 0; r < TILE_ROWS; ++r) {
            float4 value = staged[r][k];
            #pragma unroll
            for (uint c = 0; c < COLS; ++c) {
                sum[r][c] += value * weights[c];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma unroll
for (uint r = 0; r < TILE_ROWS; ++r) {
    #pragma unroll
    for (uint c = 0; c < COLS; ++c) {
        float4 s = sum[r][c];
        s += simd_shuffle_down(s, 4);
        s += simd_shuffle_down(s, 2);
        s += simd_shuffle_down(s, 1);
        if (chain == 0 && r < valid_rows) {
            output[size_t(row0 + r) * OUTPUTS + column + c] =
                bfloat16_t((s.x + s.y) + (s.z + s.w));
        }
    }
}
