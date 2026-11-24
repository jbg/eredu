uint section = chunk_in_block >> 3u;
uint scale_group = chunk_in_block & 7u;
int scale = int(as_type<char>(weight[base + 192u + section * 8u + scale_group]));
float scale_factor = qk_half(weight, base + 208u) * float(scale);
uint quarter = scale_group >> 1u;
uint offset = (scale_group & 1u) * 16u;
uint high_shift = quarter * 2u;
uint low_shift = (quarter & 2u) * 2u;
// The wire blocks have a 210-byte stride. Use byte-aligned four-value loads.
device const packed_uchar4* low = reinterpret_cast<device const packed_uchar4*>(
    weight + base + section * 64u + (quarter & 1u) * 32u + offset);
device const packed_uchar4* high = reinterpret_cast<device const packed_uchar4*>(
    weight + base + 128u + section * 32u + offset);

#pragma unroll
for (uint v = 0u; v < 4u; ++v) {
    uint4 ql = (uint4(uchar4(low[v])) >> low_shift) & 15u;
    uint4 qh = (uint4(uchar4(high[v])) >> high_shift) & 3u;
    half4 decoded = half4(scale_factor * float4(int4(ql | (qh << 4u)) - 32));
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        values[v * 4u + j] = decoded[j];
    }
}
