// One SIMD group per row. Lanes stage the same four-element masses used by
// the scalar kernel so the compensated sum still follows column order.
uint lane = thread_position_in_threadgroup.x;
uint width = uint(shape[0]);
uint groups = (width + 3) / 4;
size_t base = size_t(threadgroup_position_in_grid.y) * width;
threadgroup float masses[GROUP_CAPACITY];

float peak = -INFINITY;
for (uint i = lane; i < width; i += 32) {
    peak = max(peak, input[base + i]);
}
peak = simd_max(peak);

for (uint group = lane; group < groups; group += 32) {
    uint i = group * 4;
    float4 values = 0.0f;
    for (uint j = 0; j < 4 && i + j < width; ++j) {
        values[j] = vector_exp_f32(input[base + i + j] - peak);
        out[base + i + j] = values[j];
    }
    masses[group] = softmax_mass(values);
}
simdgroup_barrier(mem_flags::mem_threadgroup);

float inverse = 0.0f;
if (lane == 0) {
    float high = 0.0f;
    float low = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        add_softmax_mass(masses[group], high, low);
    }
    inverse = softmax_inverse(high, low);
}
inverse = simd_shuffle(inverse, 0);

// Each lane normalizes the same output positions it previously wrote.
for (uint group = lane; group < groups; group += 32) {
    uint i = group * 4;
    for (uint j = 0; j < 4 && i + j < width; ++j) {
        out[base + i + j] *= inverse;
    }
}
