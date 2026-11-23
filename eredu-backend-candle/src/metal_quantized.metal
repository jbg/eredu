// Packed GGML weights stay in their original byte layout throughout execution.
struct Q8Activation32 {
    float scale;
    char codes[32];
};

struct Q8Activation256 {
    float scale;
    char codes[256];
    int sums[8];
};

static_assert(sizeof(Q8Activation32) == 36, "32-element activation block ABI");
static_assert(sizeof(Q8Activation256) == 292, "256-element activation block ABI");

kernel void quantize_activation32(
    device const float* input [[buffer(0)]],
    device Q8Activation32* output [[buffer(1)]],
    uint block [[thread_position_in_grid]])
{
    input += size_t(block) * 32;
    float maximum = 0.f;
    for (uint i = 0; i < 32; ++i) maximum = max(maximum, abs(input[i]));
    const float scale = precise::divide(maximum, 127.f);
    const float inverse = scale == 0.f ? 0.f : precise::divide(1.f, scale);
    output[block].scale = float(half(scale));
    for (uint i = 0; i < 32; ++i) {
        output[block].codes[i] = char(rint(input[i] * inverse));
    }
}

kernel void quantize_activation256(
    device const float* input [[buffer(0)]],
    device Q8Activation256* output [[buffer(1)]],
    uint block [[thread_position_in_grid]])
{
    input += size_t(block) * 256;
    float peak = 0.f;
    for (uint i = 0; i < 256; ++i) {
        if (abs(input[i]) > abs(peak)) peak = input[i];
    }
    const float inverse = peak == 0.f ? 0.f : precise::divide(-127.f, peak);
    output[block].scale = peak == 0.f ? 0.f : precise::divide(1.f, inverse);
    for (uint group = 0; group < 8; ++group) {
        int sum = 0;
        for (uint i = 0; i < 32; ++i) {
            const uint column = group * 32 + i;
            const char code = char(rint(input[column] * inverse));
            output[block].codes[column] = code;
            sum += int(code);
        }
        output[block].sums[group] = sum;
    }
}

inline float packed_half(device const uchar* p) {
    const ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

inline int4 q5_dots(device const Q8Activation32& act, device const uchar* w) {
    const uint high = uint(w[2]) | (uint(w[3]) << 8) | (uint(w[4]) << 16) | (uint(w[5]) << 24);
    int4 dots = 0;
    for (uint i = 0; i < 16; ++i) {
        const uint q = w[6 + i];
        const int lo = int((q & 15) | (((high >> i) & 1) << 4)) - 16;
        const int hi = int((q >> 4) | (((high >> (i + 16)) & 1) << 4)) - 16;
        dots[i / 4] += lo * int(act.codes[i]) + hi * int(act.codes[i + 16]);
    }
    return dots;
}

struct PackedProjection {
    uint blocks;
    uint row_bytes;
    uint format;
    uint tiled_rows;
};

enum PackedFormat : uint { Q5_0 = 0, Q8_0 = 1, Q4_K = 2, Q5_K = 3, Q6_K = 4 };

kernel void packed_matmul32(
    device const Q8Activation32* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant PackedProjection& spec [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    input += size_t(index.y) * spec.blocks;
    weights += size_t(index.x) * spec.row_bytes;
    float sum = 0.f;
    if (spec.format == Q8_0) {
        for (uint b = 0; b < spec.blocks; ++b) {
            device const uchar* w = weights + size_t(b) * 34;
            int dot = 0;
            for (uint i = 0; i < 32; ++i) {
                dot += int(input[b].codes[i]) * int(as_type<char>(w[2 + i]));
            }
            sum = fma(float(dot), input[b].scale * packed_half(w), sum);
        }
    } else {
        float4 even = 0.f;
        float4 odd = 0.f;
        uint b = 0;
        for (; b + 1 < spec.blocks; b += 2) {
            device const uchar* w = weights + size_t(b) * 22;
            even = fma(float4(q5_dots(input[b], w)), input[b].scale * packed_half(w), even);
            odd = fma(float4(q5_dots(input[b + 1], w + 22)),
                      input[b + 1].scale * packed_half(w + 22), odd);
        }
        sum = ((even.x + even.y) + (even.z + even.w)) +
              ((odd.x + odd.y) + (odd.z + odd.w));
        if (b < spec.blocks) {
            device const uchar* w = weights + size_t(b) * 22;
            const int4 dots = q5_dots(input[b], w);
            sum = fma(float(dots.x + dots.y + dots.z + dots.w), input[b].scale * packed_half(w), sum);
        }
    }
    output[size_t(index.y) * grid.x + index.x] = sum;
}

inline float min_k_block(device const Q8Activation256& input, device const uchar* w,
                         bool fifth_bit, float sum) {
    const float scale = packed_half(w) * input.scale;
    const float min_scale = packed_half(w + 2) * input.scale;
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
                dot += int(lo | hi) * int(input.codes[group * 32 + j]);
            }
            pair_dot += int(s) * dot;
            min_dot += int(m) * input.sums[group];
        }
        sum = fma(float(pair_dot), scale, sum);
    }
    return fma(-float(min_dot), min_scale, sum);
}

inline float q6_k_block(device const Q8Activation256& input, device const uchar* w,
                        bool tiled, float sum) {
    const float scale = packed_half(w + 208) * input.scale;
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
            local += (code - 32) * int(input.codes[section * 128 + part * 32 + i]);
        }
        dot += local * int(as_type<char>(w[192 + group]));
    }
    return tiled ? fma(float(dot), scale, sum) : sum + float(dot) * scale;
}

kernel void packed_matmul256(
    device const Q8Activation256* input [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant PackedProjection& spec [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    input += size_t(index.y) * spec.blocks;
    weights += size_t(index.x) * spec.row_bytes;
    const uint block_bytes = spec.row_bytes / spec.blocks;
    float sum = 0.f;
    for (uint b = 0; b < spec.blocks; ++b) {
        device const uchar* w = weights + size_t(b) * block_bytes;
        if (spec.format == Q6_K) {
            sum = q6_k_block(input[b], w, index.y < spec.tiled_rows, sum);
        } else {
            sum = min_k_block(input[b], w, spec.format == Q5_K, sum);
        }
    }
    output[size_t(index.y) * grid.x + index.x] = sum;
}


kernel void packed_embedding(
    device const uchar* weights [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint4& spec [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    device const uchar* block = weights + size_t(ids[index.y]) * spec.x
        + (index.x / spec.z) * spec.w;
    const uint column = index.x % spec.z;
    float value;
    switch (spec.y) {
    case Q5_0: {
        const uint high = (block[2 + column / 8] >> (column % 8)) & 1;
        const uint code = ((block[6 + column % 16] >> (4 * (column / 16))) & 15)
            | (high << 4);
        value = packed_half(block) * float(int(code) - 16);
        break;
    }
    case Q8_0:
        value = packed_half(block) * float(as_type<char>(block[2 + column]));
        break;
    case Q6_K: {
        const uint section = column / 128;
        const uint part = (column / 32) % 4;
        const uint lane = column % 32;
        const uint low = (block[section * 64 + (part % 2) * 32 + lane] >> (4 * (part / 2))) & 15;
        const uint high = (block[128 + section * 32 + lane] >> (2 * part)) & 3;
        const float scale = packed_half(block + 208) *
            float(as_type<char>(block[192 + section * 8 + part * 2 + lane / 16]));
        value = scale * float(int(low | (high << 4)) - 32);
        break;
    }
    default: {
        const bool q5 = spec.y == Q5_K;
        const uint group = column / 32;
        const uint lane = column % 32;
        device const uchar* scales = block + 4;
        const uint scale = group < 4 ? scales[group] & 63 :
            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        const uint minimum = group < 4 ? scales[group + 4] & 63 :
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
        const uint low = (block[(q5 ? 48 : 16) + (group / 2) * 32 + lane]
            >> (4 * (group % 2))) & 15;
        const uint high = q5 ? (block[16 + lane] >> group) & 1 : 0;
        value = (packed_half(block) * float(scale)) * float(low | (high << 4))
            - (packed_half(block + 2) * float(minimum));
        break;
    }
    }
    output[size_t(index.y) * grid.x + index.x] = value;
}
