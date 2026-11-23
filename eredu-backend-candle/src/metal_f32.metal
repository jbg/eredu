kernel void recurrent_rotary(
    device const float* input [[buffer(0)]],
    device const float2* tables [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint2& config [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    const uint pairs = grid.x;
    const uint pair = index.x;
    const uint row = index.y;
    const bool adjacent = config.x != 0;
    const float2 trig = tables[size_t(row % config.y) * pairs + pair];
    const size_t base = size_t(row) * (2 * pairs);
    const size_t a = base + (adjacent ? 2 * pair : pair);
    const size_t b = adjacent ? a + 1 : a + pairs;
    const float x = input[a];
    const float y = input[b];
    output[a] = fma(x, trig.y, -(y * trig.x));
    output[b] = fma(x, trig.x, y * trig.y);
}

inline float rotary_round(float value, uint rounding) {
    if (rounding == 1) return float(half(value));
    if (rounding != 2 || !isfinite(value)) return value;
    const uint bits = as_type<uint>(value);
    return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
}

kernel void explicit_rotary(
    device const float* source [[buffer(0)]],
    device const float2* table [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint* args [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    const uint tokens = args[0];
    const uint heads = args[1];
    const uint dimensions = args[2];
    const bool adjacent = args[3] != 0;
    const uint rounding = args[4];
    const size_t start = size_t(index.y) * grid.x;
    const float current = source[start + index.x];
    if (index.x >= dimensions) {
        output[start + index.x] = current;
        return;
    }
    const uint half_width = dimensions / 2;
    const bool first = adjacent ? (index.x % 2 == 0) : (index.x < half_width);
    const uint other = adjacent ? (index.x ^ 1u) :
        (first ? index.x + half_width : index.x - half_width);
    const float neighbor = source[start + other];
    const size_t table_row = size_t(index.y / (heads * tokens)) * tokens + index.y % tokens;
    const float2 trig = table[table_row * dimensions + index.x];
    if (rounding != 0) {
        const float x = rotary_round(current * rotary_round(trig.x, rounding), rounding);
        const float y = rotary_round((first ? -neighbor : neighbor) * rotary_round(trig.y, rounding), rounding);
        output[start + index.x] = x + y;
    } else {
        output[start + index.x] = first ? fma(current, trig.x, -neighbor * trig.y) :
            fma(neighbor, trig.y, current * trig.x);
    }
}

// Port of the FP32 vector exponential used by the CPU mechanism.
// See THIRD-PARTY-NOTICES for Arm Optimized Routines and ggml attribution.
inline float vector_exp_f32(float x) {
    const float biased = fma(x, 0x1.715476p0f, 12582912.f);
    const float n = biased - 12582912.f;
    if (!isfinite(x) || abs(n) > 126.f) return ordered_exp(x);
    const float b = fma(-n, as_type<float>(0x35bfbe8eu),
                        fma(-n, as_type<float>(0x3f317200u), x));
    const float scale = as_type<float>((as_type<uint>(biased) << 23) + 0x3f800000u);
    const float b2 = b * b;
    const float lo = fma(b, as_type<float>(0x3e2aaf33u), as_type<float>(0x3efffedbu));
    const float hi = fma(b, as_type<float>(0x3c072010u), as_type<float>(0x3d2b9f17u));
    const float adjustment = fma(fma(hi, b2, lo), b2, as_type<float>(0x3f7ffff6u) * b);
    return fma(scale, adjustment, scale);
}

inline float divide_wide(float numerator, WideFloat x) {
    const float quotient = numerator / x.high;
    // Recover the residual lost when rounding the denominator to float.
    const float error = fma(-quotient, x.high, numerator) - quotient * x.low;
    return quotient + error / x.high;
}

kernel void f32_sigmoid(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    output[index] = precise::divide(1.f, 1.f + ordered_exp(-input[index]));
}

kernel void f32_silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const float x = input[index];
    const float exp_value = (index % width) < (width / 4) * 4 ?
        vector_exp_f32(-x) : ordered_exp(-x);
    output[index] = precise::divide(x, 1.f + exp_value);
}

kernel void f32_rms_inverse(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& args [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    const uint width = args.x;
    input += size_t(row) * width;
    WideFloat sum = {0.f, 0.f};
    for (uint i = 0; i < width; ++i) {
        const float x = input[i];
        sum = add_wide(sum, {x * x, 0.f});
    }
    const float rounded = sum.high / float(width);
    const float error = fma(-rounded, float(width), sum.high) + sum.low;
    const float mean = rounded + error / float(width);
    output[row] = 1.f / precise::sqrt(mean + as_type<float>(args.y));
}

kernel void f32_attention_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    input += size_t(row) * width;
    output += size_t(row) * width;
    float peak = -INFINITY;
    for (uint i = 0; i < width; ++i) peak = max(peak, input[i]);
    WideFloat mass = {0.f, 0.f};
    for (uint i = 0; i < width; i += 4) {
        float4 values = 0.f;
        for (uint j = 0; j < 4 && i + j < width; ++j) {
            values[j] = vector_exp_f32(input[i + j] - peak);
            output[i + j] = values[j];
        }
        mass = add_wide(mass, {(values.x + values.y) + (values.z + values.w), 0.f});
    }
    const float inverse = divide_wide(1.f, mass);
    for (uint i = 0; i < width; ++i) output[i] *= inverse;
}

struct FloatProduct {
    uint rows;
    uint width;
    uint columns;
    uint strided_rhs;
    uint lanes;
    uint pad;
};

kernel void f32_dot(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FloatProduct& shape [[buffer(3)]],
    uint2 index [[thread_position_in_grid]])
{
    const uint row = index.y;
    const uint column = index.x;
    const uint batch = row / shape.rows;
    left += size_t(row) * shape.width;
    right += size_t(batch) * shape.width * shape.columns;
    right += shape.strided_rhs ? column : size_t(column) * shape.width;
    const uint stride = shape.strided_rhs ? shape.columns : 1;
    const uint end = shape.pad ? shape.width : (shape.width / shape.lanes) * shape.lanes;
    float lanes[16] = {};
    for (uint i = 0; i < end; ++i) {
        const uint lane = i % shape.lanes;
        lanes[lane] = fma(left[i], right[size_t(i) * stride], lanes[lane]);
    }
    if (shape.lanes == 16) {
        for (uint i = 0; i < 8; ++i) lanes[i] += lanes[i + 8];
        for (uint i = 0; i < 4; ++i) lanes[i] += lanes[i + 4];
    }
    float result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (uint i = end; i < shape.width; ++i) {
        result = fma(left[i], right[size_t(i) * stride], result);
    }
    output[size_t(row) * shape.columns + column] = result;
}

kernel void gelu_half_table(device const float* input [[buffer(0)]],
                            device const float* table [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    float x = input[i];
    out[i] = x <= -10.f ? 0.f : (x < 10.f ? table[as_type<ushort>(half(x))] : x);
}

inline float fused_half(float x, float y, float acc) {
    float product = x * y;
    float sum = product + acc;
    float part = sum - product;
    float residual = (product - (sum - part)) + (acc - part);
    uint bits = as_type<uint>(sum);
    if (isfinite(sum) && !(bits & 1) && residual != 0.f) {
        bits = (sum < 0.f) == (residual < 0.f) ? bits + 1 : bits - 1;
    }
    return float(half(as_type<float>(bits)));
}

kernel void binary16_projection(device const half* input [[buffer(0)]],
                                device const half* weight [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint2& args [[buffer(3)]],
                                uint2 pos [[thread_position_in_grid]]) {
    uint width = args.x, outputs = args.y;
    device const half* a = input + size_t(pos.y) * width;
    device const half* b = weight + size_t(pos.x) * width;
    float lane[8] = {};
    for (uint i = 0; i < width; ++i) {
        lane[i % 8] = fused_half(float(a[i]), float(b[i]), lane[i % 8]);
    }
    float4 m = float4(lane[0], lane[1], lane[2], lane[3]) +
               float4(lane[4], lane[5], lane[6], lane[7]);
    out[size_t(pos.y) * outputs + pos.x] = (m.x + m.y) + (m.z + m.w);
}

inline float blocked_moment(device const float* row, uint width, float center, bool squared) {
    float4 sums = 0.f;
    if (width == 32) {
        for (uint lane = 0; lane < 4; ++lane) {
            float terms[4];
            for (uint group = 0; group < 4; ++group) {
                float x = row[4 * group + lane];
                float y = row[4 * (group + 4) + lane];
                terms[group] = squared ? fma(y - center, y - center, (x - center) * (x - center)) : x;
            }
            sums[lane] = (terms[0] + terms[2]) + (terms[1] + terms[3]);
            if (!squared) {
                for (uint group = 4; group < 8; ++group) sums[lane] += row[4 * group + lane];
            }
        }
    } else {
        float head[16];
        for (uint lane = 0; lane < 16; ++lane) {
            float x = row[lane], y = row[16 + lane];
            float a = squared ? (x - center) * (x - center) : x + y;
            float b = squared ? (y - center) * (y - center) : 0.f;
            for (uint block = 32; block + 32 < width; block += 32) {
                x = row[block + lane];
                y = row[block + 16 + lane];
                a = squared ? fma(x - center, x - center, a) : a + x;
                b = squared ? fma(y - center, y - center, b) : b + y;
            }
            x = row[width - 32 + lane];
            y = row[width - 16 + lane];
            head[lane] = squared ? fma(y - center, y - center, fma(x - center, x - center, a) + b)
                                 : ((a + b) + x) + y;
        }
        for (uint i = 0; i < 4; ++i) {
            sums[i] = (head[i] + head[i + 8]) + (head[i + 4] + head[i + 12]);
        }
    }
    return (sums.x + sums.y) + (sums.z + sums.w);
}

kernel void blocked_layer_norm(device const float* input [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant uint2& args [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    uint width = args.x;
    input += size_t(i) * width;
    out += size_t(i) * width;
    float mean = precise::divide(blocked_moment(input, width, 0.f, false), float(width));
    float variance = precise::divide(blocked_moment(input, width, mean, true), float(width));
    float inv = precise::divide(1.f, precise::sqrt(variance + as_type<float>(args.y)));
    for (uint j = 0; j < width; ++j) out[j] = (input[j] - mean) * inv;
}

kernel void f32_softplus(device const float* input [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& beta_bits [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    float beta = as_type<float>(beta_bits);
    float value = input[i];
    float scaled = value * beta;
    out[i] = scaled > 20.f ? value : precise::divide(precise::log(1.f + precise::exp(scaled)), beta);
}

kernel void ordered_l2_norm(device const float* input [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint* args [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    uint width = args[0];
    float epsilon = as_type<float>(args[1]);
    input += size_t(i) * width;
    out += size_t(i) * width;
    WideFloat sum = {0.f, 0.f};
    for (uint j = 0; j < width; ++j) sum = add_wide(sum, {input[j] * input[j], 0.f});
    float squared = sum.high + sum.low;
    float norm = args[2] ? max(precise::sqrt(squared), epsilon) : precise::sqrt(squared + epsilon);
    float inverse = precise::divide(1.f, norm);
    for (uint j = 0; j < width; ++j) out[j] = input[j] * inverse;
}

inline float gated_delta_dot(device const float* state, uint stride, device const float* vector, uint width) {
    uint aligned = width / 16 * 16;
    float lanes[16] = {};
    for (uint i = 0; i < aligned; ++i) lanes[i % 16] = fma(state[i * stride], vector[i], lanes[i % 16]);
    for (uint i = 0; i < 8; ++i) lanes[i] += lanes[i + 8];
    for (uint i = 0; i < 4; ++i) lanes[i] += lanes[i + 4];
    float result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (uint i = aligned; i < width; ++i) result = fma(state[i * stride], vector[i], result);
    return result;
}

kernel void gated_delta_scan(device const float* data [[buffer(0)]],
                              device const float* initial [[buffer(1)]],
                              device float* result [[buffer(2)]],
                              constant uint* args [[buffer(3)]],
                              uint2 pos [[thread_position_in_grid]]) {
    uint sequence = args[0], heads = args[1], keys = args[2], values = args[3];
    uint decay_width = args[4];
    uint batch_head = pos.y, channel = pos.x;
    uint features = keys * 2 + values + decay_width + 1;
    size_t column = size_t(batch_head) * keys * values + channel;
    device float* state = result + column;
    for (uint key = 0; key < keys; ++key) state[key * values] = initial[column + key * values];
    device float* output = result + size_t(args[5]);
    for (uint token = 0; token < sequence; ++token) {
        size_t group = (size_t(batch_head / heads) * sequence + token) * heads + batch_head % heads;
        device const float* query = data + group * features;
        device const float* key = query + keys;
        device const float* value = key + keys;
        device const float* decay = value + values;
        float beta = decay[decay_width];
        if (decay_width == 1) {
            float factor = precise::exp(decay[0]);
            for (uint i = 0; i < keys; ++i) state[i * values] *= factor;
        } else {
            for (uint i = 0; i < keys; ++i) state[i * values] *= precise::exp(decay[i]);
        }
        float delta = (value[channel] - gated_delta_dot(state, values, key, keys)) * beta;
        for (uint i = 0; i < keys; ++i) state[i * values] = fma(key[i], delta, state[i * values]);
        output[group * values + channel] = gated_delta_dot(state, values, query, keys);
    }
}

inline size_t integer_view_index(uint index, constant uint* args) {
    size_t offset = 0;
    for (uint axis = args[1]; axis > 0; --axis) {
        const uint width = args[2 * axis];
        offset += size_t(index % width) * args[2 * axis + 1];
        index /= width;
    }
    return offset;
}

kernel void equal_i32(
    device const int* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant uint* args [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    output[index] = uchar(input[integer_view_index(index, args)] == as_type<int>(args[0]));
}

kernel void maximum_i32(
    device const int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint* args [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    output[index] = max(input[integer_view_index(index, args)], as_type<int>(args[0]));
}

kernel void widen_i32(
    device const int* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant uint* args [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    output[index] = long(input[integer_view_index(index, args)]);
}

kernel void narrow_i64(device const long* input [[buffer(0)]],
                       device int* out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    out[i] = int(input[i]);
}

kernel void rounded_operand_projection(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    input += size_t(index.y) * width;
    weights += size_t(index.x) * width;
    WideFloat sum = {0.f, 0.f};
    for (uint i = 0; i < width; ++i) {
        sum = add_wide(sum, {input[i] * weights[i], 0.f});
    }
    output[size_t(index.y) * grid.x + index.x] = sum.high;
}

kernel void f32_depthwise_convolution(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint* args [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    const uint sequence = args[0];
    const uint length = args[1];
    const uint taps = args[2];
    const uint dilation = args[3];
    const uint separate_products = args[4];
    const size_t row = size_t(index.y / length) * sequence + index.y % length;
    input += row * grid.x + index.x;
    weights += size_t(index.x) * taps;
    float sum = 0.f;
    for (uint tap = 0; tap < taps; ++tap) {
        const float x = input[size_t(tap) * dilation * grid.x];
        sum = tap < separate_products ? sum + x * weights[tap] : fma(x, weights[tap], sum);
    }
    output[size_t(index.y) * grid.x + index.x] = sum;
}

// One ordered selection per row avoids the threadgroup-width limit of
// a bitonic sort. Each pass resumes just below its preceding result.
kernel void f32_top_indices(
    device const float* scores [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint2& geometry [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    scores += size_t(row) * geometry.x;
    output += size_t(row) * geometry.y;
    float upper = INFINITY;
    uint preceding = 0;
    for (uint slot = 0; slot < geometry.y; ++slot) {
        uint best = 0xffffffff;
        float peak = -INFINITY;
        for (uint index = 0; index < geometry.x; ++index) {
            const float score = isnan(scores[index]) ? -INFINITY : scores[index];
            if (slot == 0 || score < upper || (score == upper && index > preceding)) {
                if (best == 0xffffffff || score > peak) {
                    peak = score;
                    best = index;
                }
            }
        }
        upper = peak;
        preceding = best;
        output[slot] = best;
    }
}
