// One SIMD group cooperates on each row, with four interleaved FP32 sums.
#pragma clang fp reassociate(off)
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
        float centered = input[base + i] - maximum;
        float value = PAIRWISE ? stable_exp_f32(centered) : metal::fast::exp(centered);
        output[base + i] = value;
        sum += value;
    }
}
float s0 = simd_shuffle(sum, 0);
float s1 = simd_shuffle(sum, 1);
float s2 = simd_shuffle(sum, 2);
float s3 = simd_shuffle(sum, 3);
float inverse = PAIRWISE
    ? metal::precise::divide(1.0f, (s0 + s2) + (s1 + s3))
    : metal::fast::divide(1.0f, ((s0 + s1) + s2) + s3);
// Normalizing lanes read values written by the four accumulation lanes.
threadgroup_barrier(mem_flags::mem_device);
for (uint i = lane; i < width; i += 32) {
    output[base + i] *= inverse;
}
