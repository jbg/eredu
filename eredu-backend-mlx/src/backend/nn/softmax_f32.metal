// One SIMD group cooperates on each row. Exponentials retain their explicit
// FP32 rounding, and four lanes retain the original ordered accumulation chains.
uint row = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
uint width = uint(dimensions[0]);
if (row >= uint(dimensions[1])) return;
size_t base = size_t(row) * width;
float maximum = -INFINITY;
bool has_nan = false;
for (uint i = lane; i < width; i += 32) {
    float value = input[base + i];
    has_nan |= isnan(value);
    maximum = max(maximum, value);
}
maximum = simd_any(has_nan) ? NAN : simd_max(maximum);
float sum = 0.0f;
if (lane < 4) {
    for (uint i = lane; i < width; i += 4) {
        float value = stable_exp_f32(input[base + i] - maximum);
        output[base + i] = value;
        sum += value;
    }
}
float s0 = simd_shuffle(sum, 0);
float s1 = simd_shuffle(sum, 1);
float s2 = simd_shuffle(sum, 2);
float s3 = simd_shuffle(sum, 3);
float inverse = 1.0f / ((s0 + s2) + (s1 + s3));
// Normalizing lanes read values written by the four accumulation lanes.
threadgroup_barrier(mem_flags::mem_device);
for (uint i = lane; i < width; i += 32) {
    output[base + i] *= inverse;
}
