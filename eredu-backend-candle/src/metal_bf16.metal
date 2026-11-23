#include <metal_stdlib>
using namespace metal;

// Keep multiplication and lane reductions as distinct FP32 operations.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

struct WideFloat {
    float high;
    float low;
};

inline WideFloat two_sum(float a, float b) {
    const float sum = a + b;
    const float back = sum - a;
    return {sum, (a - (sum - back)) + (b - back)};
}

inline WideFloat add_wide(WideFloat a, WideFloat b) {
    const WideFloat sum = two_sum(a.high, b.high);
    return two_sum(sum.high, (a.low + b.low) + sum.low);
}

inline WideFloat multiply_wide(WideFloat a, WideFloat b) {
    const float product = a.high * b.high;
    const float cross = (a.high * b.low) + (a.low * b.high);
    return two_sum(product, fma(a.high, b.high, -product) + cross);
}

// Reduce to [-ln(2)/2, ln(2)/2] and evaluate the Taylor polynomial with
// split FP32 coefficients. Keep the low part through the final sum so the
// exponential rounds before probability normalization reaches a BF16 midpoint.
inline float ordered_exp(float input) {
    if (!(abs(input) < 80.f)) return precise::exp(input);
    const float exponent = rint(input * 0x1.715476p+0f);
    const float high = fma(-exponent, 0x1.62e400p-1f, input);
    const WideFloat remainder = two_sum(high, -exponent * 0x1.7f7d1cp-20f);
    constexpr WideFloat coefficients[13] = {
        {0x1.000000p+0f, 0x0.0p+0f},
        {0x1.000000p+0f, 0x0.0p+0f},
        {0x1.000000p-1f, 0x0.0p+0f},
        {0x1.555556p-3f, -0x1.555556p-28f},
        {0x1.555556p-5f, -0x1.555556p-30f},
        {0x1.111112p-7f, -0x1.dddddep-32f},
        {0x1.6c16c2p-10f, -0x1.27d27ep-35f},
        {0x1.a01a02p-13f, -0x1.7f97fap-39f},
        {0x1.a01a02p-16f, -0x1.7f97fap-42f},
        {0x1.71de3ap-19f, 0x1.55b1ccp-45f},
        {0x1.27e4fcp-22f, -0x1.10ec14p-47f},
        {0x1.ae6456p-26f, 0x1.fd5138p-52f},
        {0x1.1eed8ep-29f, 0x1.ff1b12p-54f}
    };
    WideFloat result = coefficients[12];
    for (int i = 11; i >= 0; --i) {
        result = add_wide(multiply_wide(result, remainder), coefficients[i]);
    }
    return ldexp(result.high, int(exponent));
}

struct ProductShape {
    uint rows;
    uint width;
    uint columns;
    uint sequential;
    uint transposed;
    uint interleaved;
    uint row_end;
    uint tail_end;
};

kernel void bf16_product(
    device const bfloat* left [[buffer(0)]],
    device const bfloat* right [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant ProductShape& shape [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const bool scalar = shape.sequential || !shape.transposed;
    const uint row = index.y;
    const uint col = scalar ? index.x : index.x / 32;
    const size_t item = size_t(row) * shape.columns + col;
    const uint batch = row / shape.rows;
    left += size_t(row) * shape.width;
    right += size_t(batch) * shape.width * shape.columns;
    if (shape.transposed) {
        right += size_t(col) * shape.width;
    } else {
        right += col;
    }
    const uint stride = shape.transposed ? 1 : shape.columns;

    if (scalar) {
        float sum[4] = {0.f, 0.f, 0.f, 0.f};
        const uint limit = shape.interleaved;
        for (uint k = 0; k < limit; ++k) {
            sum[k % 4] += float(left[k]) * float(right[size_t(k) * stride]);
        }
        for (uint k = limit; k < shape.width; ++k) {
            sum[0] += float(left[k]) * float(right[size_t(k) * stride]);
        }
        output[item] = bfloat(((sum[0] + sum[1]) + sum[2]) + sum[3]);
        return;
    }

    float sum = 0.f;
    const uint aligned = shape.row_end;
    for (uint k = lane; k < aligned; k += 32) {
        sum += float(left[k]) * float(right[k]);
    }
    sum += simd_shuffle_down(sum, 16);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    const float s1 = simd_shuffle(sum, 1);
    const float s2 = simd_shuffle(sum, 2);
    const float s3 = simd_shuffle(sum, 3);
    if (lane != 0) return;

    float result = (sum + s1) + (s2 + s3);
    const uint tail_end = shape.tail_end;
    float tail[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint k = aligned; k < tail_end; k += 8) {
        for (uint i = 0; i < 4; ++i) {
            tail[i] += float(left[k + i]) * float(right[k + i]);
            tail[i] += float(left[k + i + 4]) * float(right[k + i + 4]);
        }
    }
    result += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for (uint k = tail_end; k < shape.width; ++k) {
        result += float(left[k]) * float(right[k]);
    }
    output[item] = bfloat(result);
}

inline float softmax_inverse(float4 lanes, uint width) {
    const float mass = width < 4 ? (lanes[0] + lanes[1]) + lanes[2] :
                                  (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    return precise::divide(1.f, mass);
}

kernel void bf16_softmax(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    input += size_t(row) * width;
    output += size_t(row) * width;
    float peak = -INFINITY;
    for (uint i = 0; i < width; ++i) peak = max(peak, float(input[i]));
    float lanes[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint i = 0; i < width; ++i) lanes[i % 4] += stable_exp_f32(float(input[i]) - peak);
    const float inverse = softmax_inverse(float4(lanes[0], lanes[1], lanes[2], lanes[3]), width);
    for (uint i = 0; i < width; ++i) {
        output[i] = bfloat(stable_exp_f32(float(input[i]) - peak) * inverse);
    }
}

kernel void bf16_page_mass(
    device const float* scores [[buffer(0)]],
    device const float* maximum [[buffer(1)]],
    device const float* previous [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint2& page [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    const uint lane = i % 4;
    const uint row = i / 4;
    float sum = previous[i];
    if (maximum[row] != -0x1.fffffep127f) {
        const uint first = (lane + 4 - page.y % 4) % 4;
        scores += size_t(row) * page.x;
        for (uint c = first; c < page.x; c += 4) {
            sum += stable_exp_f32(scores[c] - maximum[row]);
        }
    }
    output[i] = sum;
}

kernel void bf16_page_probabilities(
    device const float* scores [[buffer(0)]],
    device const float* maximum [[buffer(1)]],
    device const float* mass [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint2& page [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    const uint row = i / page.x;
    const device float* lanes = mass + size_t(row) * 4;
    const float inverse = softmax_inverse(float4(lanes[0], lanes[1], lanes[2], lanes[3]), page.y);
    output[i] = maximum[row] == -0x1.fffffep127f ? bfloat(0.f) :
        bfloat(stable_exp_f32(scores[i] - maximum[row]) * inverse);
}

struct ValueShape {
    uint width;
    uint first_lane;
    uint interleaved;
    uint queries;
    uint features;
    uint lanes;
};

kernel void bf16_page_values(
    device const bfloat* probabilities [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const float* previous [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ValueShape& shape [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    const uint lane = i % shape.lanes;
    const uint feature = (i / shape.lanes) % shape.features;
    const uint row = i / shape.lanes / shape.features;
    const uint bank = row / shape.queries;
    probabilities += size_t(row) * shape.width;
    values += size_t(bank) * shape.width * shape.features + feature;

    float sum = previous[i];
    if (shape.lanes == 1) {
        for (uint c = 0; c < shape.width; ++c) {
            sum += float(probabilities[c]) * values[size_t(c) * shape.features];
        }
    } else {
        const uint aligned = shape.interleaved;
        const uint first = (lane + shape.lanes - shape.first_lane) % shape.lanes;
        for (uint c = first; c < aligned; c += 4) {
            sum += float(probabilities[c]) * values[size_t(c) * shape.features];
        }
        if (lane == 0) {
            for (uint c = aligned; c < shape.width; ++c) {
                sum += float(probabilities[c]) * values[size_t(c) * shape.features];
            }
        }
    }
    output[i] = sum;
}

kernel void ordered_rms_inverse(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& args [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint width = args.x;
    const uint row = index / 32;
    input += size_t(row) * width;
    const uint aligned = (width / 32) * 32;
    float sum = 0.f;
    for (uint k = lane; k < aligned; k += 32) {
        const float value = input[k];
        sum += value * value;
    }
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 16);
    const float s1 = simd_shuffle(sum, 1);
    const float s2 = simd_shuffle(sum, 2);
    const float s3 = simd_shuffle(sum, 3);
    if (lane != 0) return;
    sum = (sum + s1) + (s2 + s3);
    for (uint k = aligned; k < width; ++k) {
        const float value = input[k];
        sum += value * value;
    }
    const float scale = 1.f / float(width);
    output[row] = 1.f / precise::sqrt(sum * scale + as_type<float>(args.y));
}

// Eight interleaved components in each of four independently cascaded chains.
kernel void cascaded_rms_inverse(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& args [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint width = args.x;
    const uint row = index / 32;
    input += size_t(row) * width;
    const uint vectors = width / 8;
    const uint steps = vectors / 4;
    const uint log_steps = steps > 1 ? 32 - clz(steps - 1) : 0;
    const uint power = max(4u, log_steps / 4);
    const uint chunk = 1u << power;
    float sums[4] = {};
    uint i = 0;
    while (i + chunk <= steps) {
        for (const uint end = i + chunk; i < end; ++i) {
            const float x = input[i * 32 + lane];
            sums[0] += x * x;
        }
        for (uint level = 1; level < 4; ++level) {
            sums[level] += sums[level - 1];
            sums[level - 1] = 0.f;
            if (i & ((chunk - 1) << (level * power))) break;
        }
    }
    for (; i < steps; ++i) {
        const float x = input[i * 32 + lane];
        sums[0] += x * x;
    }
    for (uint level = 1; level < 4; ++level) sums[0] += sums[level];
    if (lane < 8) {
        for (uint v = steps * 4; v < vectors; ++v) {
            const float x = input[v * 8 + lane];
            sums[0] += x * x;
        }
    }
    const uint c = lane % 8;
    float merged = simd_shuffle(sums[0], c);
    for (uint chain = 1; chain < 4; ++chain) {
        merged += simd_shuffle(sums[0], chain * 8 + c);
    }
    float sum = 0.f;
    for (uint k = vectors * 8; k < width; ++k) {
        const float x = input[k];
        sum += x * x;
    }
    for (uint k = 0; k < 8; ++k) sum += simd_shuffle(merged, k);
    if (lane == 0) {
        const float variance = precise::divide(sum, float(width));
        output[row] = precise::divide(1.f, precise::sqrt(variance + as_type<float>(args.y)));
    }
}

kernel void bf16_silu(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    const float value = float(input[index]);
    output[index] = bfloat(value / (1.f + ordered_exp(-value)));
}
