// Each threadgroup owns one complete score row. Keep each of the original
// FP32 arithmetic boundaries while retaining scores and probabilities locally.
uint tid = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint heads = shape[0], rows = shape[1], width = shape[2];
uint kv_heads = shape[3], positions = shape[4], outputs = shape[5];
uint query_position = row % rows;
uint head_row = row / rows;
uint head = head_row % heads;
uint batch = head_row / heads;
uint kv_row = batch * kv_heads + head / (heads / kv_heads);
size_t query_base = size_t(row) * width;
size_t key_base = size_t(kv_row) * positions * width;
threadgroup float probabilities[CAPACITY];
threadgroup float masses[CAPACITY / 4];
threadgroup float group_maxima[THREADS / 32];
threadgroup float row_inverse;

int64_t mask_base = 0;
int64_t mask_step = 0;
if (MASK_MODE != 0) {
    mask_base = (mask_shape[0] == 1 ? 0 : batch * mask_strides[0])
              + (mask_shape[1] == 1 ? 0 : head * mask_strides[1])
              + (mask_shape[2] == 1 ? 0 : query_position * mask_strides[2]);
    mask_step = mask_shape[3] == 1 ? 0 : mask_strides[3];
}

uint score_lane = tid % 16;
uint score_end = width / 16 * 16;
for (uint position = tid / 16; position < positions; position += THREADS / 16) {
    if (MASK_MODE == 1 && !bool(mask[mask_base + position * mask_step])) {
        if (score_lane == 0) probabilities[position] = -INFINITY;
        continue;
    }
    size_t key = key_base + size_t(position) * width;
    float sum = 0.0f;
    for (uint i = score_lane; i < score_end; i += 16) {
        sum = fma(query[query_base + i], keys[key + i], sum);
    }
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    float pair = simd_shuffle_down(sum, 1);
    float third = simd_shuffle_down(sum, 2);
    float fourth = simd_shuffle_down(sum, 3);
    if (score_lane == 0) {
        float score = (sum + pair) + (third + fourth);
        for (uint i = score_end; i < width; ++i) {
            score = fma(query[query_base + i], keys[key + i], score);
        }
        score *= scale[0];
        if (MASK_MODE == 2) {
            score += float(mask[mask_base + position * mask_step]);
        }
        probabilities[position] = score;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float peak = -INFINITY;
for (uint i = tid; i < positions; i += THREADS) {
    peak = max(peak, probabilities[i]);
}
peak = simd_max(peak);
if (tid % 32 == 0) group_maxima[tid / 32] = peak;
threadgroup_barrier(mem_flags::mem_threadgroup);
peak = group_maxima[0];
for (uint group = 1; group < THREADS / 32; ++group) peak = max(peak, group_maxima[group]);

uint groups = (positions + 3) / 4;
for (uint group = tid; group < groups; group += THREADS) {
    uint i = group * 4;
    float4 exps = 0.0f;
    for (uint j = 0; j < 4 && i + j < positions; ++j) {
        exps[j] = vector_exp_f32(probabilities[i + j] - peak);
        probabilities[i + j] = exps[j];
    }
    masses[group] = softmax_mass(exps);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float high = 0.0f, low = 0.0f;
    for (uint group = 0; group < groups; ++group) add_softmax_mass(masses[group], high, low);
    row_inverse = softmax_inverse(high, low);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < positions; i += THREADS) probabilities[i] *= row_inverse;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint value_lane = tid % VALUE_LANES;
size_t value_base = size_t(kv_row) * outputs * positions;
for (uint column = tid / VALUE_LANES; column < outputs; column += THREADS / VALUE_LANES) {
    size_t value = value_base + size_t(column) * positions;
    float sum = 0.0f;
    for (uint i = value_lane; i < positions; i += VALUE_LANES) {
        sum = fma(probabilities[i], values[value + i], sum);
    }
    if (VALUE_LANES == 16) {
        sum += simd_shuffle_down(sum, 8);
        sum += simd_shuffle_down(sum, 4);
    }
    float pair = simd_shuffle_down(sum, 1);
    float third = simd_shuffle_down(sum, 2);
    float fourth = simd_shuffle_down(sum, 3);
    if (value_lane == 0) out[size_t(row) * outputs + column] = (sum + pair) + (third + fourth);
}
