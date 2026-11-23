// Apply the contiguous softmax reciprocal and round one probability per key.
uint flat = thread_position_in_grid.x;
uint width = uint(geometry[1]);
uint row = flat / width;
if (row >= uint(geometry[0])) return;

size_t base = size_t(row) * 4;
float inverse = bf16_softmax_inverse(mass[base], mass[base + 1], mass[base + 2], mass[base + 3]);
float score = scores[flat];
probabilities[flat] = inverse > 0.0f && isfinite(inverse) && score != -INFINITY
    ? bfloat16_t(stable_exp_f32(score - maximum[row]) * inverse)
    : bfloat16_t(0.0f);
