uint position = thread_position_in_grid.x / 16;
uint lane = thread_position_in_grid.x % 16;
uint row = thread_position_in_grid.y;
uint heads = shape[0], width = shape[2], kv_heads = shape[3], positions = shape[4];
uint batch = row / heads, head = row % heads;
uint kv_head = head / (heads / kv_heads);
size_t q = batch * query_strides[0] + head * query_strides[1];
size_t k = batch * keys_strides[0] + kv_head * keys_strides[1] + position * keys_strides[2];
uint end = width / 16 * 16;
float sum = 0.0f;
for (uint i = lane; i < end; i += 16) {
    sum = fma(query[q + i * query_strides[3]], keys[k + i * keys_strides[3]], sum);
}
sum += simd_shuffle_down(sum, 8);
sum += simd_shuffle_down(sum, 4);
float pair = simd_shuffle_down(sum, 1);
float third = simd_shuffle_down(sum, 2);
float fourth = simd_shuffle_down(sum, 3);
if (lane == 0 && position < positions) {
    float score = (sum + pair) + (third + fourth);
    for (uint i = end; i < width; ++i) {
        score = fma(query[q + i * query_strides[3]], keys[k + i * keys_strides[3]], score);
    }
    score *= scale[0];
    if (MASK_MODE != 0) {
        size_t index = (mask_shape[0] == 1 ? 0 : batch * mask_strides[0])
                     + (mask_shape[1] == 1 ? 0 : head * mask_strides[1])
                     + (mask_shape[3] == 1 ? 0 : position * mask_strides[3]);
        if (MASK_MODE == 1 && !bool(mask[index])) score = -INFINITY;
        if (MASK_MODE == 2) score += float(mask[index]);
    }
    out[size_t(row) * positions + position] = score;
}
