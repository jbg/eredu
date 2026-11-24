#pragma clang fp contract(off)

inline ulong round_rms_units(ulong units, uint shift) {
    ulong half_minus_one = (ulong(1) << (shift - 1)) - 1;
    return (units + half_minus_one + ((units >> shift) & 1)) >> shift;
}

// The caller supplies at most 8192 rounded squares and a power-of-two width.
// Return a negative sentinel when the sum's FP32 rounding cannot be certified.
inline float certified_rms_mean(threadgroup const float *squares, uint width, uint lane) {
    uint largest_exponent = 0;
    bool admissible = true;
    for (uint i = lane; i < width; i += 32) {
        uint bits = as_type<uint>(squares[i]);
        uint exponent = bits >> 23;
        largest_exponent = max(largest_exponent, exponent);
        // Keep the legacy arithmetic and its divisions in the normal
        // range. A nonzero square then has a quantum of at least 2^-110.
        admissible &= (bits == 0 || (exponent >= 40 && exponent <= 220));
    }
    if (!simd_all(admissible)) return -1.0f;
    largest_exponent = simd_max(largest_exponent);
    if (largest_exponent == 0) return 0.0f;

    // Use units of 2^(largest_exponent - 127 - 40). Each square contributes
    // at most 41 bits and truncates by less than one unit.
    ulong local = 0;
    for (uint i = lane; i < width; i += 32) {
        uint bits = as_type<uint>(squares[i]);
        uint exponent = bits >> 23;
        uint distance = largest_exponent - exponent;
        if (distance <= 40) {
            ulong mantissa = (bits & 0x7fffffu) | 0x800000u;
            local += distance <= 17 ? mantissa << (17 - distance)
                                    : mantissa >> (distance - 17);
        }
    }
    // Split the lane sums so both uint SIMD reductions fit without wrap.
    ulong units = (ulong(simd_sum(uint(local >> 26))) << 26)
                + ulong(simd_sum(uint(local & 0x3ffffffu)));
    ulong upper_sum = units + width;
    uint width_shift = ctz(width);

    // TwoSum recovers the rounding error of high + square. Adding low to
    // that error is the only inexact step in the two-word accumulation;
    // FastTwoSum then stores the rounded result and its residual exactly.
    // Each addend of the inexact step is bounded by 2u times the upper
    // sum, with u = 2^-24, hence 4 * width * u^2 bounds the relative error
    // accumulated over the row. The positive partial sums stay within
    // that envelope. Dividing either word by width is exact in this range.
    uint error_shift = 46 - width_shift;
    ulong error = (upper_sum + (ulong(1) << error_shift) - 1) >> error_shift;
    ulong lower = units - error;
    ulong upper = upper_sum + error;

    // The largest square alone contributes at least 2^40 units.
    uint top = 63 - clz(uint(upper >> 32));
    if ((lower >> top) == 0) return -1.0f;
    uint shift = top - 23;
    // The legacy correction rounds (high + low) / width. The exact
    // power-of-two scaling commutes with FP32 rounding in this range.
    ulong rounded = round_rms_units(lower, shift);
    if (rounded != round_rms_units(upper, shift)) return -1.0f;
    uint scale_exponent = largest_exponent - 40 + shift - width_shift;
    return float(uint(rounded)) * as_type<float>(scale_exponent << 23);
}
