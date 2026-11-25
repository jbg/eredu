#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

uint channel = thread_position_in_grid.x;
uint batch = thread_position_in_grid.y;
size_t output_index = size_t(batch) * CHANNELS + channel;
size_t projected_index = size_t(batch) * 3 * CHANNELS + channel;
size_t history_index = size_t(batch) * (TAPS - 1) * CHANNELS + channel;
size_t weight_index = size_t(channel) * TAPS;

// Preserve the activation-dtype rounding on the input gate and convolution result.
T current = T(
    float(projected[projected_index]) * float(projected[projected_index + 2 * CHANNELS]));
float sums[4] = {};
for (uint tap = 0; tap < TAPS; ++tap) {
    float value = tap == TAPS - 1 ? float(current) : float(history[history_index + tap * size_t(CHANNELS)]);
    uint lane = tap < (TAPS / 4) * 4 ? tap % 4 : 0;
    sums[lane] += value * float(weights[weight_index + tap]);
}
T convolved = T(((sums[0] + sums[1]) + sums[2]) + sums[3]);
output[output_index] = T(float(projected[projected_index + CHANNELS]) * float(convolved));

for (uint tap = 0; tap < TAPS - 2; ++tap) {
    next_history[history_index + tap * size_t(CHANNELS)] =
        history[history_index + (tap + 1) * size_t(CHANNELS)];
}
next_history[history_index + (TAPS - 2) * size_t(CHANNELS)] = current;
