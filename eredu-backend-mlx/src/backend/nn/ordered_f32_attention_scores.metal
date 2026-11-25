uint position = thread_position_in_grid.x / 16;
uint lane = thread_position_in_grid.x % 16;
uint row = thread_position_in_grid.y;
uint heads = shape[0], rows = shape[1], width = shape[2], kv_heads = shape[3], positions = shape[4];
if (position >= positions) return;
uint batch = row / (heads * rows), head = row / rows % heads;
uint query_position = row % rows;
size_t mask_index = 0;
if (MASK_MODE != 0) {
    mask_index = (mask_shape[0] == 1 ? 0 : batch * mask_strides[0])
               + (mask_shape[1] == 1 ? 0 : head * mask_strides[1])
               + (mask_shape[2] == 1 ? 0 : query_position * mask_strides[2])
               + (mask_shape[3] == 1 ? 0 : position * mask_strides[3]);
}
// A boolean mask excludes the entire 16-lane dot product.
if (MASK_MODE == 1 && !bool(mask[mask_index])) {
    if (lane == 0) out[size_t(row) * positions + position] = -INFINITY;
    return;
}
uint kv_head = head / (heads / kv_heads);
size_t q = batch * query_strides[0] + head * query_strides[1] + query_position * query_strides[2];
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
if (lane == 0) {
    float score = (sum + pair) + (third + fourth);
    for (uint i = end; i < width; ++i) {
        score = fma(query[q + i * query_strides[3]], keys[k + i * keys_strides[3]], score);
    }
    score *= scale[0];
    if (MASK_MODE == 2) score += float(mask[mask_index]);
    out[size_t(row) * positions + position] = score;
}
