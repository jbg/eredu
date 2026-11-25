// Lanes stage four-element masses for each row. The compensated sum follows
// column order regardless of the number of SIMD groups loading its scores.
uint lane = thread_position_in_threadgroup.x;
uint width = uint(shape[0]);
uint groups = (width + 3) / 4;
size_t base = size_t(threadgroup_position_in_grid.y) * width;
threadgroup float masses[GROUP_CAPACITY];
threadgroup float maxima[THREADS / 32];
threadgroup float row_inverse;

float peak = -INFINITY;
for (uint i = lane; i < width; i += THREADS) {
    peak = max(peak, input[base + i]);
}
peak = simd_max(peak);
if (THREADS > 32) {
    if (lane % 32 == 0) maxima[lane / 32] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak = maxima[0];
    for (uint group = 1; group < THREADS / 32; ++group) peak = max(peak, maxima[group]);
}

for (uint group = lane; group < groups; group += THREADS) {
    uint i = group * 4;
    float4 values = 0.0f;
    for (uint j = 0; j < 4 && i + j < width; ++j) {
        values[j] = vector_exp_f32(input[base + i + j] - peak);
        out[base + i + j] = values[j];
    }
    masses[group] = softmax_mass(values);
}
if (THREADS == 32) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
} else {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float inverse = 0.0f;
if (lane == 0) {
    float high = 0.0f;
    float low = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        add_softmax_mass(masses[group], high, low);
    }
    inverse = softmax_inverse(high, low);
    if (THREADS > 32) row_inverse = inverse;
}
if (THREADS == 32) {
    inverse = simd_shuffle(inverse, 0);
} else {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inverse = row_inverse;
}

// Each lane normalizes the same output positions it previously wrote.
for (uint group = lane; group < groups; group += THREADS) {
    uint i = group * 4;
    for (uint j = 0; j < 4 && i + j < width; ++j) {
        out[base + i + j] *= inverse;
    }
}
