// Adjacent lanes read adjacent value channels. Each lane keeps the four
// ordered FP32 chains for one output and rounds once after merging them.
uint column = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
uint head = thread_position_in_grid.z;
uint bank = (head / HEADS) * KV_HEADS + (head % HEADS) / REPEATS;
size_t values_base = size_t(bank) * WIDTH * OUTPUTS + column;
size_t input_base = (size_t(head) * ROWS + row) * WIDTH;
constexpr uint full = (WIDTH / 4) * 4;
float4 sum = 0.0f;
for (uint i = 0; i < full; i += 4) {
    float4 p = float4(float(input[input_base + i]), float(input[input_base + i + 1]),
                      float(input[input_base + i + 2]), float(input[input_base + i + 3]));
    size_t offset = values_base + size_t(i) * OUTPUTS;
    float4 v = float4(float(values[offset]), float(values[offset + OUTPUTS]),
                      float(values[offset + 2 * OUTPUTS]), float(values[offset + 3 * OUTPUTS]));
    sum += p * v;
}
// The scalar column product appends the incomplete quartet to its first chain.
for (uint i = full; i < WIDTH; ++i) {
    sum.x += float(input[input_base + i]) * float(values[values_base + size_t(i) * OUTPUTS]);
}
output[(size_t(head) * ROWS + row) * OUTPUTS + column] =
    bfloat16_t(((sum.x + sum.y) + sum.z) + sum.w);
