#pragma clang fp contract(off)

// Both the staged fallback and the scalar path consume the rounded
// squares in input order.
inline void accumulate_ordered_rms_square(float y, thread float &high, thread float &low) {
    float s = high + y;
    float v = s - high;
    float e = (high - (s - v)) + (y - v) + low;
    high = s + e;
    low = e - (high - s);
}

inline float finish_ordered_rms_mean(float high, float low, uint width) {
    float mean = metal::precise::divide(high, float(width));
    mean += metal::precise::divide(fma(-mean, float(width), high) + low, float(width));
    return mean;
}

inline float ordered_rms_mean_from_squares(threadgroup const float *squares, uint width) {
    float high = 0.0f;
    float low = 0.0f;
    for (uint i = 0; i < width; ++i) {
        accumulate_ordered_rms_square(squares[i], high, low);
    }
    return finish_ordered_rms_mean(high, low, width);
}

inline ulong rms_square_units(uint bits, uint largest_exponent) {
    uint distance = largest_exponent - (bits >> 23);
    if (bits == 0 || distance > 40) return 0;
    ulong mantissa = (bits & 0x7fffffu) | 0x800000u;
    return distance <= 17 ? mantissa << (17 - distance)
                          : mantissa >> (distance - 17);
}

inline ulong sum_rms_lane_units(ulong local) {
    // Split lane sums so both uint SIMD reductions fit without wrap.
    return (ulong(simd_sum(uint(local >> 26))) << 26)
            + ulong(simd_sum(uint(local & 0x3ffffffu)));
}

inline ulong round_rms_units(ulong units, uint shift) {
    ulong half_minus_one = (ulong(1) << (shift - 1)) - 1;
    return (units + half_minus_one + ((units >> shift) & 1)) >> shift;
}

inline float certify_rms_units(ulong units, uint width, uint largest_exponent) {
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
        local += rms_square_units(as_type<uint>(squares[i]), largest_exponent);
    }
    return certify_rms_units(sum_rms_lane_units(local), width, largest_exponent);
}
