#pragma clang fp contract(off)
uint index = thread_position_in_grid.x;
if (index >= uint(position[1])) return;
constexpr uint half_width = DIMENSIONS / 2;
constexpr uint work = half_width + WIDTH - DIMENSIONS;
uint row = index / work;
uint column = index % work;
size_t base = size_t(row) * WIDTH;
if (column >= half_width) {
  uint tail = DIMENSIONS + column - half_width;
  output[base + tail] = input[base + tail];
  return;
}
// arange<F32> rounds the start before adding the sequence index.
float offset_position = float(position[0]) + float(row % uint(position[2]));
float angle = offset_position * frequencies[column];
float cosine = metal::precise::cos(angle) * amplitude;
float sine = metal::precise::sin(angle) * amplitude;
if (INPUT_TABLE) {
  cosine = float(bfloat16_t(cosine));
  sine = float(bfloat16_t(sine));
}
uint left = TRADITIONAL ? 2 * column : column;
uint right = TRADITIONAL ? 2 * column + 1 : column + half_width;
// Separate rounded products prevent contraction into an FMA.
float first_cosine = float(input[base + left]) * cosine;
float second_sine = float(input[base + right]) * sine;
float second_cosine = float(input[base + right]) * cosine;
float first_sine = float(input[base + left]) * sine;
output[base + left] = T(first_cosine - second_sine);
output[base + right] = T(second_cosine + first_sine);
