#pragma clang fp reassociate(off)
uint row = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
uint width = uint(dimensions[3]);
size_t base = size_t(row) * width;
threadgroup float values[CAPACITY];

int64_t mask_base = 0;
int64_t mask_step = 0;
if (MASK_MODE != 0) {
    uint query = row % uint(dimensions[2]);
    uint head_row = row / uint(dimensions[2]);
    uint head = head_row % uint(dimensions[1]);
    uint batch = head_row / uint(dimensions[1]);
    mask_base = (mask_shape[0] == 1 ? 0 : batch * mask_strides[0])
              + (mask_shape[1] == 1 ? 0 : head * mask_strides[1])
              + (mask_shape[2] == 1 ? 0 : query * mask_strides[2]);
    mask_step = mask_shape[3] == 1 ? 0 : mask_strides[3];
}

float maximum = -INFINITY;
bool has_nan = false;
for (uint i = lane; i < width; i += 32) {
    float value = float(bfloat16_t(float(scores[base + i]) * scale[0]));
    if (MASK_MODE == 1) {
        if (!bool(mask[mask_base + i * mask_step])) value = -INFINITY;
    } else if (MASK_MODE == 2) {
        float addend = float(bfloat16_t(float(mask[mask_base + i * mask_step])));
        value = float(bfloat16_t(value + addend));
    }
    values[i] = value;
    maximum = max(maximum, value);
    has_nan |= isnan(value);
}
maximum = simd_any(has_nan) ? NAN : simd_max(maximum);
for (uint i = lane; i < width; i += 32) {
    values[i] = stable_exp_f32(values[i] - maximum);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// The four FP32 chains and their pairwise merge match the ordered softmax.
float sum = 0.0f;
if (lane < 4) {
    for (uint i = lane; i < width; i += 4) {
        sum += values[i];
    }
}
float s0 = simd_shuffle(sum, 0);
float s1 = simd_shuffle(sum, 1);
float s2 = simd_shuffle(sum, 2);
float s3 = simd_shuffle(sum, 3);
float inverse = metal::precise::divide(1.0f, (s0 + s2) + (s1 + s3));
for (uint i = lane; i < width; i += 32) {
    probabilities[base + i] = bfloat16_t(values[i] * inverse);
}
