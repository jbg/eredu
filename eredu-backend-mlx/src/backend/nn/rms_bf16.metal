#pragma clang fp contract(off)
uint row = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
uint chunk = thread_position_in_threadgroup.z;
size_t base = size_t(row) * WIDTH;
SquaredBfloat squared{input + base};
threadgroup float partial[CHUNKS == 4 ? 4 * 32 : 1];
float inverse;
if (CHUNKS == 4) {
    // Each SIMD group covers one complete chunk in all 32 cascade chains.
    float sum = 0.0f;
    #pragma unroll
    for (uint step = 0; step < 16; ++step) {
        sum += squared[(chunk * 16 + step) * 32 + lane];
    }
    partial[chunk * 32 + lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (chunk == 0) {
        // No tail or higher cascade level contributes at this width. Preserve
        // the ordered merges of the chunks, the four chains and the eight lanes.
        float chain = ((partial[lane] + partial[32 + lane]) +
                       partial[64 + lane]) + partial[96 + lane];
        uint component = lane % LANES;
        float merged = simd_shuffle(chain, component);
        #pragma unroll
        for (uint p = 1; p < 4; ++p) {
            merged += simd_shuffle(chain, p * LANES + component);
        }
        float total = 0.0f;
        #pragma unroll
        for (uint i = 0; i < LANES; ++i) {
            total += simd_shuffle(merged, i);
        }
        partial[lane] = 1.0f / metal::precise::sqrt(total / float(WIDTH) + epsilon[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inverse = partial[lane];
} else {
    float sum = parallel_cascade_sum_f32<LANES>(squared, WIDTH, lane);
    inverse = 1.0f / metal::precise::sqrt(sum / float(WIDTH) + epsilon[0]);
}
for (uint i = chunk * 32 + lane; i < WIDTH; i += CHUNKS * 32) {
    // Learned gain consumes the rounded normalized value.
    bfloat16_t normalized = bfloat16_t(float(input[base + i]) * inverse);
    output[base + i] = WEIGHTED ? bfloat16_t(float(normalized) * float(gain[i])) : normalized;
}
