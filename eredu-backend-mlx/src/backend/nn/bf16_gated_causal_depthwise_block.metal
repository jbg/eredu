#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constexpr uint HISTORY_ROWS = TAPS - 1;
uint channel = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
uint batch = thread_position_in_grid.z;
size_t history_base = size_t(batch) * HISTORY_ROWS * CHANNELS + channel;
size_t projected_base = size_t(batch) * ROWS * 3 * CHANNELS + channel;

// Read a row in the logical concatenation of history and rounded gated inputs.
auto gated = [&](uint position) {
    if (position < HISTORY_ROWS) {
        return HAS_HISTORY ? history[history_base + size_t(position) * CHANNELS]
                           : bfloat16_t(0.0f);
    }
    size_t index = projected_base + size_t(position - HISTORY_ROWS) * 3 * CHANNELS;
    return bfloat16_t(float(projected[index]) * float(projected[index + 2 * CHANNELS]));
};

float sums[4] = {};
#pragma unroll
for (uint tap = 0; tap < TAPS; ++tap) {
    uint lane = tap < (TAPS / 4) * 4 ? tap % 4 : 0;
    sums[lane] += float(gated(row + tap)) * float(weights[size_t(channel) * TAPS + tap]);
}
bfloat16_t convolved = bfloat16_t(((sums[0] + sums[1]) + sums[2]) + sums[3]);
float gate = float(projected[projected_base + size_t(row) * 3 * CHANNELS + CHANNELS]);
output[(size_t(batch) * ROWS + row) * CHANNELS + channel] = bfloat16_t(gate * float(convolved));

if (row == 0) {
    for (uint h = 0; h < HISTORY_ROWS; ++h) {
        next_history[history_base + size_t(h) * CHANNELS] = gated(ROWS + h);
    }
}
