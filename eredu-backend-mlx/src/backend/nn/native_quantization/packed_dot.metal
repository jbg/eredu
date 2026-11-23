#pragma clang fp contract(off)

inline float packed_half(device const uchar* p) {
    const ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

inline int4 q5_dots(device const int8_t* codes, device const uchar* w) {
    const uint high = uint(w[2]) | (uint(w[3]) << 8) | (uint(w[4]) << 16) | (uint(w[5]) << 24);
    int4 dots = 0;
    for (uint i = 0; i < 16; ++i) {
        const uint q = w[6 + i];
        const int lo = int((q & 15) | (((high >> i) & 1) << 4)) - 16;
        const int hi = int((q >> 4) | (((high >> (i + 16)) & 1) << 4)) - 16;
        dots[i / 4] += lo * int(codes[i]) + hi * int(codes[i + 16]);
    }
    return dots;
}

inline float min_k_block(device const int8_t* codes, float input_scale,
                         device const int* sums, device const uchar* w,
                         bool fifth_bit, float sum) {
    const float scale = packed_half(w) * input_scale;
    const float min_scale = packed_half(w + 2) * input_scale;
    device const uchar* scales = w + 4;
    device const uchar* quants = w + (fifth_bit ? 48 : 16);
    int min_dot = 0;
    for (uint pair = 0; pair < 4; ++pair) {
        int pair_dot = 0;
        for (uint nibble = 0; nibble < 2; ++nibble) {
            const uint group = pair * 2 + nibble;
            const uint packed = scales[group + 4];
            const uint s = group < 4 ? (scales[group] & 63) :
                           (packed & 15) | ((scales[group - 4] >> 6) << 4);
            const uint m = group < 4 ? (packed & 63) :
                           (packed >> 4) | ((scales[group] >> 6) << 4);
            int dot = 0;
            for (uint j = 0; j < 32; ++j) {
                const uint lo = (quants[pair * 32 + j] >> (4 * nibble)) & 15;
                const uint hi = fifth_bit ? (((w[16 + j] >> group) & 1) << 4) : 0;
                dot += int(lo | hi) * int(codes[group * 32 + j]);
            }
            pair_dot += int(s) * dot;
            min_dot += int(m) * sums[group];
        }
        sum = fma(float(pair_dot), scale, sum);
    }
    return fma(-float(min_dot), min_scale, sum);
}

inline float q6_k_block(device const int8_t* codes, float input_scale, device const uchar* w,
                        bool tiled, float sum) {
    const float scale = packed_half(w + 208) * input_scale;
    int dot = 0;
    for (uint group = 0; group < 16; ++group) {
        const uint section = group / 8;
        const uint part = (group % 8) / 2;
        const uint begin = (group % 2) * 16;
        int local = 0;
        for (uint i = begin; i < begin + 16; ++i) {
            const uint lo = w[section * 64 + (part % 2) * 32 + i];
            const uint hi = w[128 + section * 32 + i];
            const int code = int(((lo >> (4 * (part / 2))) & 15) | (((hi >> (2 * part)) & 3) << 4));
            local += (code - 32) * int(codes[section * 128 + part * 32 + i]);
        }
        dot += local * int(as_type<char>(w[192 + group]));
    }
    return tiled ? fma(float(dot), scale, sum) : sum + float(dot) * scale;
}
