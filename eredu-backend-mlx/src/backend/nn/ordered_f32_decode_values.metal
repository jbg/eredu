// Each group merges the original ordered softmax masses and consumes only
// its own value columns. Tiling leaves enough independent groups for decode.
uint tid = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint heads = shape[0], kv_heads = shape[3], positions = shape[4], outputs = shape[5];
uint batch = row / heads, kv_head = (row % heads) / (heads / kv_heads);
size_t score_base = size_t(row) * positions;
threadgroup float probabilities[CAPACITY];
threadgroup float masses[CAPACITY / 4];
threadgroup float group_maxima[THREADS / 32];
threadgroup float row_inverse;

float peak = -INFINITY;
for (uint i = tid; i < positions; i += THREADS) peak = max(peak, scores[score_base + i]);
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
        exps[j] = vector_exp_f32(scores[score_base + i + j] - peak);
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

// Interleave two value columns so adjacent lanes load adjacent channels.
// The sixteen independent accumulators keep the separate product's order.
uint lane = (tid % 32) / 2;
uint column = threadgroup_position_in_grid.z * (THREADS / 16) + (tid / 32) * 2 + tid % 2;
if (column < outputs) {
    size_t value = batch * values_strides[0] + kv_head * values_strides[1]
                 + column * values_strides[2];
    float sum = 0.0f;
    for (uint i = lane; i < positions; i += 16) {
        sum = fma(probabilities[i], values[value + i * values_strides[3]], sum);
    }
    sum += simd_shuffle_down(sum, 16);
    sum += simd_shuffle_down(sum, 8);
    float pair = simd_shuffle_down(sum, 2);
    float third = simd_shuffle_down(sum, 4);
    float fourth = simd_shuffle_down(sum, 6);
    if (lane == 0) out[size_t(row) * outputs + column] = (sum + pair) + (third + fourth);
}
