#pragma clang fp contract(off)
uint row = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
size_t base = size_t(row) * WIDTH;
float sum = parallel_cascade_sum_f32<LANES>(SquaredBfloat{input + base}, WIDTH, lane);
float variance = sum / float(WIDTH);
float inverse = 1.0f / metal::precise::sqrt(variance + epsilon[0]);
for (uint i = lane; i < WIDTH; i += 32) {
    // Learned gain consumes the rounded normalized value.
    bfloat16_t normalized = bfloat16_t(float(input[base + i]) * inverse);
    output[base + i] = WEIGHTED ? bfloat16_t(float(normalized) * float(gain[i])) : normalized;
}
