uint row = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;
uint simd_lane = lane % 32;
uint simd_group = lane / 32;
constexpr uint lanes = 32 * SIMD_GROUPS;
size_t offset = size_t(row) * WIDTH;
threadgroup float squares[WIDTH];
threadgroup uint exponents[SIMD_GROUPS];
threadgroup uint invalid[SIMD_GROUPS];
threadgroup ulong sums[SIMD_GROUPS];
threadgroup float shared_inverse;

uint largest_exponent = 0;
bool admissible = true;
for (uint i = lane; i < WIDTH; i += lanes) {
    float x = input[offset + i];
    float square = x * x;
    squares[i] = square;
    uint bits = as_type<uint>(square);
    uint exponent = bits >> 23;
    largest_exponent = max(largest_exponent, exponent);
    admissible &= (bits == 0 || (exponent >= 40 && exponent <= 220));
}
largest_exponent = simd_max(largest_exponent);
admissible = simd_all(admissible);
if (simd_lane == 0) {
    exponents[simd_group] = largest_exponent;
    invalid[simd_group] = uint(!admissible);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint i = 0; i < SIMD_GROUPS; ++i) {
    largest_exponent = max(largest_exponent, exponents[i]);
    admissible &= invalid[i] == 0;
}
ulong local = 0;
if (admissible && largest_exponent != 0) {
    for (uint i = lane; i < WIDTH; i += lanes) {
        local += rms_square_units(as_type<uint>(squares[i]), largest_exponent);
    }
    ulong units = sum_rms_lane_units(local);
    if (simd_lane == 0) sums[simd_group] = units;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane == 0) {
    float mean = -1.0f;
    if (admissible) {
        if (largest_exponent == 0) {
            mean = 0.0f;
        } else {
            // Every group uses the same fixed-point units. The disjoint
            // partial sums combine exactly before the rounding check.
            ulong units = 0;
            for (uint i = 0; i < SIMD_GROUPS; ++i) units += sums[i];
            mean = certify_rms_units(units, WIDTH, largest_exponent);
        }
    }
    if (mean < 0.0f) mean = ordered_rms_mean_from_squares(squares, WIDTH);
    shared_inverse = metal::precise::divide(1.0f, metal::precise::sqrt(mean + epsilon[0]));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = lane; i < WIDTH; i += lanes) {
    out[offset + i] = (input[offset + i] * shared_inverse) * gain[i];
}
