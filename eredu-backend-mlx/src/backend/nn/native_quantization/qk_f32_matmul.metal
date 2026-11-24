// A 64-column by 32-row output tile. Specialize on the weight geometry and
// keep the prompt row count dynamic so partial tiles reuse the same pipeline.
constexpr uint NR0 = 64u;
constexpr uint NR1 = 32u;
constexpr uint NK = 32u;
constexpr uint blocks = WIDTH / 256u;
uint rows = uint(input_shape[0]);
uint tid = uint(thread_index_in_threadgroup);
uint simd = uint(simdgroup_index_in_threadgroup);
uint r0 = uint(threadgroup_position_in_grid.y) * NR0;
uint r1 = uint(threadgroup_position_in_grid.z) * NR1;
uint nr0 = min(NR0, uint(OUTPUTS) - r0);
uint nr1 = min(NR1, rows - r1);
uint il0 = tid & 1u;
uint lr0 = min(tid >> 1u, nr0 - 1u);
uint lr1 = min(tid >> 2u, nr1 - 1u);
uint iy = 8u * (tid & 3u);
threadgroup half sa[2048];
alignas(8) threadgroup half sb[1024];
simdgroup_half8x8 ma[4];
simdgroup_half8x8 mb[2];
simdgroup_float8x8 mc[8];

#pragma unroll
for (uint i = 0u; i < 8u; ++i) {
    mc[i].thread_elements()[0] = 0.0f;
    mc[i].thread_elements()[1] = 0.0f;
}

uint row_base = (r0 + lr0) * blocks * QK_BLOCK_BYTES;
for (uint loop_k = 0u; loop_k < WIDTH; loop_k += NK) {
    half values[16];
    qk_dequantize_16(weight, row_base, (loop_k >> 4u) + il0, values);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        uint sx = 2u * il0 + i / 8u;
        uint sy = (tid >> 1u) / 8u;
        uint lx = (tid >> 1u) & 7u;
        uint ly = i & 7u;
        uint ib = 8u * sx + sy;
        sa[64u * ib + 8u * ly + lx] = values[i];
    }
    uint input_base = (r1 + lr1) * WIDTH + loop_k + iy;
    uint sx = tid & 3u;
    uint sy = (tid >> 2u) / 8u;
    uint ly = (tid >> 2u) & 7u;
    uint ib = 4u * sx + sy;
    // Contiguous input views need only float alignment.
    device const packed_float4* src =
        reinterpret_cast<device const packed_float4*>(input + input_base);
    threadgroup half4* dest = reinterpret_cast<threadgroup half4*>(sb + 64u * ib + 8u * ly);
    dest[0] = half4(float4(src[0]));
    dest[1] = half4(float4(src[1]));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const half* lsma = sa + 4u * 64u * (simd & 1u);
    threadgroup const half* lsmb = sb + 2u * 64u * (simd >> 1u);
    // The staging barrier makes both tiles visible. Every lane in each SIMD
    // group then executes the same matrix loads and multiply-accumulates.
    #pragma unroll
    for (uint ik = 0u; ik < NK / 8u; ++ik) {
        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            simdgroup_load(ma[i], lsma + 64u * i, 8u, 0u, false);
        }
        #pragma unroll
        for (uint i = 0u; i < 2u; ++i) {
            simdgroup_load(mb[i], lsmb + 64u * i, 8u, 0u, false);
        }
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i & 3u], mc[i]);
        }
        lsma += 8u * 64u;
        lsmb += 4u * 64u;
    }
}

uint fragment_r0 = 32u * (simd & 1u);
uint fragment_r1 = 16u * (simd >> 1u);
threadgroup float result[NR0 * NR1];
threadgroup float* tile = result + fragment_r0 + fragment_r1 * NR0;
#pragma unroll
for (uint i = 0u; i < 8u; ++i) {
    simdgroup_store(mc[i], tile + 8u * (i & 3u) + 8u * NR0 * (i / 4u), NR0, 0u, false);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0u) {
    for (uint index = tid; index < nr0 * nr1; index += 32u) {
        uint row = index / nr0;
        uint col = index - row * nr0;
        out[(r1 + row) * OUTPUTS + r0 + col] = result[row * NR0 + col];
    }
}
