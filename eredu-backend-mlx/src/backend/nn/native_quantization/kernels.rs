use super::*;

pub(super) fn iq_linear_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_linear_be_{big_endian}").to_lowercase(),
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint item = thread_position_in_grid.y;",
            "uint row = item / OUT_DIM;",
            "uint out_col = item % OUT_DIM;",
            "uint physical_row = ROW_START + out_col;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "float acc = 0.0f;",
            "for (uint col = lane; col < IN_DIM; col += 32) {",
            " acc += float(input[row * IN_DIM + col]) * iq_value(weight, row_base, col);",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0) out[row * OUT_DIM + out_col] = T(total);"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

pub(super) fn iq_batch_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_batch_be_{big_endian}").to_lowercase(),
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "uint physical_row = ROW_START + out_col;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "float acc[BATCH_TILE];",
            "for (uint r = 0; r < BATCH_TILE; ++r) acc[r] = 0.0f;",
            "for (uint col = lane; col < IN_DIM; col += 32) {",
            " float w = iq_value(weight, row_base, col);",
            " for (uint r = 0; r < BATCH_TILE; ++r) {",
            "  uint row = first_row + r;",
            "  if (row < ROWS) acc[r] += float(input[row * IN_DIM + col]) * w;",
            " }",
            "}",
            "for (uint r = 0; r < BATCH_TILE; ++r) {",
            " float total = simd_sum(acc[r]);",
            " uint row = first_row + r;",
            " if (lane == 0 && row < ROWS) out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

pub(super) fn iq_embedding_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_embedding_be_{big_endian}").to_lowercase(),
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "out[elem] = iq_value(weight, row_base, col);"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

pub(super) fn iq_grouped_kernel(
    format: NativeQuantizationFormat,
    big_endian: bool,
) -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        format!("native_{format:?}_grouped_be_{big_endian}").to_lowercase(),
        ["input", "weight", "group_ids"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint item = thread_position_in_grid.y;",
            "uint row = item / OUT_DIM;",
            "uint out_col = item % OUT_DIM;",
            "uint group = uint(group_ids[row]);",
            "uint physical_row = group * PHYSICAL_ROWS + ROW_START + out_col;",
            "uint row_base = physical_row * BLOCKS * BLOCK_BYTES;",
            "float acc = 0.0f;",
            "for (uint col = lane; col < IN_DIM; col += 32) {",
            " acc += float(input[row * IN_DIM + col]) * iq_value(weight, row_base, col);",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0) out[row * OUT_DIM + out_col] = T(total);"
        ),
        iq_metal_header(format, big_endian),
        true,
        false,
    )
}

pub(super) fn q4k_linear_kernel() -> Result<MetalKernel, Exception> {
    // The lane decomposition and two-row SIMD reuse follow llama.cpp's Metal
    // Q4_K matrix-vector kernel (MIT), adapted to MLX custom-kernel arguments,
    // activation dtypes, row views, and output allocation.
    MetalKernel::new(
        "native_q4k_decode_2row",
        ["input", "weight"],
        ["out"],
        concat!(
            // Four eight-lane quads divide the packed blocks between the SIMD
            // lanes. Each SIMD group computes two output rows, reusing the
            // activation values and their sums across both rows.
            "uint lane = thread_position_in_grid.x;",
            "uint output_pair = thread_position_in_grid.y;",
            "uint first_out = output_pair * ROWS_PER_SIMD;",
            "uint ix = lane / 8u;",
            "uint it = lane % 8u;",
            "uint iq = it / 4u;",
            "uint ir = it % 4u;",
            "float sums[ROWS_PER_SIMD];",
            "for (uint row = 0; row < ROWS_PER_SIMD; ++row) sums[row] = 0.0f;",
            "for (uint block = ix; block < BLOCKS; block += 4u) {",
            " uint input_base = block * 256u + 64u * iq + 8u * ir;",
            " float yl[16];",
            " float yh[16];",
            " float4 sumy = float4(0.0f);",
            " for (uint i = 0; i < 8u; ++i) {",
            "  yl[i] = float(input[input_base + i]);",
            "  yl[i + 8u] = float(input[input_base + 32u + i]);",
            "  yh[i] = float(input[input_base + 128u + i]);",
            "  yh[i + 8u] = float(input[input_base + 160u + i]);",
            "  sumy[0] += yl[i];",
            "  sumy[1] += yl[i + 8u];",
            "  sumy[2] += yh[i];",
            "  sumy[3] += yh[i + 8u];",
            " }",
            " for (uint row = 0; row < ROWS_PER_SIMD; ++row) {",
            "  uint out_col = first_out + row;",
            "  if (out_col >= OUT_DIM) continue;",
            "  uint physical_row = ROW_START + out_col;",
            "  uint base = (physical_row * BLOCKS + block) * 144u;",
            "  const device ushort* sc = (const device ushort*)(weight + base + 4u) + iq;",
            "  const device ushort* q1 = (const device ushort*)(weight + base + 16u) + 16u * iq + 4u * ir;",
            "  const device ushort* q2 = q1 + 32u;",
            "  ushort sc16[4];",
            "  thread uchar* sc8 = (thread uchar*)sc16;",
            "  sc16[0] = sc[0] & 0x3f3fu;",
            "  sc16[1] = sc[2] & 0x3f3fu;",
            "  sc16[2] = ((sc[4] >> 0u) & 0x0f0fu) | ((sc[0] & 0xc0c0u) >> 2u);",
            "  sc16[3] = ((sc[4] >> 4u) & 0x0f0fu) | ((sc[2] & 0xc0c0u) >> 2u);",
            "  float4 acc1 = float4(0.0f);",
            "  float4 acc2 = float4(0.0f);",
            "  for (uint i = 0; i < 4u; ++i) {",
            "   acc1[0] += yl[2u*i] * float(q1[i] & 0x000fu);",
            "   acc1[1] += yl[2u*i + 1u] * float(q1[i] & 0x0f00u);",
            "   acc1[2] += yl[2u*i + 8u] * float(q1[i] & 0x00f0u);",
            "   acc1[3] += yl[2u*i + 9u] * float(q1[i] & 0xf000u);",
            "   acc2[0] += yh[2u*i] * float(q2[i] & 0x000fu);",
            "   acc2[1] += yh[2u*i + 1u] * float(q2[i] & 0x0f00u);",
            "   acc2[2] += yh[2u*i + 8u] * float(q2[i] & 0x00f0u);",
            "   acc2[3] += yh[2u*i + 9u] * float(q2[i] & 0xf000u);",
            "  }",
            "  float d = float(*(const device half*)(weight + base));",
            "  float dm = float(*(const device half*)(weight + base + 2u));",
            "  sums[row] += d * (",
            "      (acc1[0] + acc1[1] * (1.0f/256.0f)) * float(sc8[0]) +",
            "      (acc1[2] + acc1[3] * (1.0f/256.0f)) * float(sc8[1]) * (1.0f/16.0f) +",
            "      (acc2[0] + acc2[1] * (1.0f/256.0f)) * float(sc8[4]) +",
            "      (acc2[2] + acc2[3] * (1.0f/256.0f)) * float(sc8[5]) * (1.0f/16.0f)) -",
            "      dm * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +",
            "            sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));",
            " }",
            "}",
            "for (uint row = 0; row < ROWS_PER_SIMD; ++row) {",
            " float total = simd_sum(sums[row]);",
            " uint out_col = first_out + row;",
            " if (lane == 0u && out_col < OUT_DIM) out[out_col] = T(total);",
            "}"
        ),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q5k_linear_kernel() -> Result<MetalKernel, Exception> {
    // Port of llama.cpp's MIT-licensed Q5_K Metal matrix-vector lane layout,
    // adapted to MLX custom-kernel arguments and output dtypes.
    MetalKernel::new(
        "native_q5k_decode",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint tid = lane / 4u;",
            "uint ix = lane % 4u;",
            "uint iq = tid / 4u;",
            "uint ir = tid % 4u;",
            "uint l0 = 8u * ir;",
            "uint q_offset = 32u * iq + l0;",
            "uint input_offset = 64u * iq + l0;",
            "uint hm1 = 1u << (2u * iq);",
            "uint hm2 = hm1 << 1u;",
            "uint hm3 = hm1 << 4u;",
            "uint hm4 = hm2 << 4u;",
            "float sum = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint row_base = physical_row * BLOCKS * 176u;",
            " for (uint block = ix; block < BLOCKS; block += 4u) {",
            "  uint base = row_base + block * 176u;",
            "  const device uchar* q1 = weight + base + 48u + q_offset;",
            "  const device uchar* q2 = q1 + 64u;",
            "  const device uchar* qh = weight + base + 16u + l0;",
            "  const device ushort* scales = (const device ushort*)(weight + base + 4u) + iq;",
            "  ushort sc16[4];",
            "  thread uchar* sc8 = (thread uchar*)sc16;",
            "  sc16[0] = scales[0] & 0x3f3fu;",
            "  sc16[1] = scales[2] & 0x3f3fu;",
            "  sc16[2] = ((scales[4] >> 0u) & 0x0f0fu) | ((scales[0] & 0xc0c0u) >> 2u);",
            "  sc16[3] = ((scales[4] >> 4u) & 0x0f0fu) | ((scales[2] & 0xc0c0u) >> 2u);",
            "  uint input_base = block * 256u + input_offset;",
            "  float yl[16];",
            "  float yh[16];",
            "  float4 sumy = float4(0.0f);",
            "  for (uint i = 0; i < 8u; ++i) {",
            "   yl[i] = float(input[input_base + i]);",
            "   yl[i + 8u] = float(input[input_base + 32u + i]);",
            "   yh[i] = float(input[input_base + 128u + i]);",
            "   yh[i + 8u] = float(input[input_base + 160u + i]);",
            "   sumy[0] += yl[i]; sumy[1] += yl[i + 8u];",
            "   sumy[2] += yh[i]; sumy[3] += yh[i + 8u];",
            "  }",
            "  float4 acc1 = float4(0.0f);",
            "  float4 acc2 = float4(0.0f);",
            "  for (uint i = 0; i < 8u; ++i) {",
            "   uint h = uint(qh[i]);",
            "   acc1[0] += yl[i] * float(q1[i] & 0x0fu);",
            "   acc1[1] += yl[i + 8u] * float(q1[i] & 0xf0u);",
            "   acc1[2] += yh[i] * float(q2[i] & 0x0fu);",
            "   acc1[3] += yh[i + 8u] * float(q2[i] & 0xf0u);",
            "   acc2[0] += (h & hm1) != 0u ? yl[i] : 0.0f;",
            "   acc2[1] += (h & hm2) != 0u ? yl[i + 8u] : 0.0f;",
            "   acc2[2] += (h & hm3) != 0u ? yh[i] : 0.0f;",
            "   acc2[3] += (h & hm4) != 0u ? yh[i + 8u] : 0.0f;",
            "  }",
            "  float d = float(*(const device half*)(weight + base));",
            "  float dm = float(*(const device half*)(weight + base + 2u));",
            "  sum += d * (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0]) +",
            "              float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1]) +",
            "              float(sc8[4]) * (acc1[2] + 16.0f * acc2[2]) +",
            "              float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3])) -",
            "         dm * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +",
            "               sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));",
            " }",
            "}",
            "float total = simd_sum(sum);",
            "if (lane == 0u && out_col < OUT_DIM) out[out_col] = T(total);"
        ),
        "",
        true,
        false,
    )
}

pub(super) fn q6k_linear_kernel() -> Result<MetalKernel, Exception> {
    // Port of llama.cpp's MIT-licensed Q6_K Metal matrix-vector lane layout.
    // Each SIMD group evaluates two weight rows while sharing activations.
    MetalKernel::new(
        "native_q6k_decode_2row",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint output_pair = thread_position_in_grid.y;",
            "uint first_out = output_pair * ROWS_PER_SIMD;",
            "uint tid = lane / 2u;",
            "uint ix = lane % 2u;",
            "uint ip = tid / 8u;",
            "uint il = tid % 8u;",
            "uint l0 = 4u * il;",
            "uint scale_offset = 8u * ip + l0 / 16u;",
            "uint input_offset = 128u * ip + l0;",
            "uint ql_offset = 64u * ip + l0;",
            "uint qh_offset = 32u * ip + l0;",
            "float sums[ROWS_PER_SIMD];",
            "for (uint row = 0; row < ROWS_PER_SIMD; ++row) sums[row] = 0.0f;",
            "for (uint block = ix; block < BLOCKS; block += 2u) {",
            " uint input_base = block * 256u + input_offset;",
            " float yl[16];",
            " for (uint i = 0; i < 4u; ++i) {",
            "  yl[4u*i] = float(input[input_base + i]);",
            "  yl[4u*i + 1u] = float(input[input_base + 32u + i]);",
            "  yl[4u*i + 2u] = float(input[input_base + 64u + i]);",
            "  yl[4u*i + 3u] = float(input[input_base + 96u + i]);",
            " }",
            " for (uint row = 0; row < ROWS_PER_SIMD; ++row) {",
            "  uint out_col = first_out + row;",
            "  if (out_col >= OUT_DIM) continue;",
            "  uint physical_row = ROW_START + out_col;",
            "  uint base = (physical_row * BLOCKS + block) * 210u;",
            "  const device uchar* q1 = weight + base + ql_offset;",
            "  const device uchar* q2 = q1 + 32u;",
            "  const device uchar* qh = weight + base + 128u + qh_offset;",
            "  const device char* scales = (const device char*)(weight + base + 192u + scale_offset);",
            "  float4 acc = float4(0.0f);",
            "  for (uint i = 0; i < 4u; ++i) {",
            "   acc[0] += yl[4u*i] * float(int((q1[i] & 0x0fu) | ((qh[i] & 0x03u) << 4u)) - 32);",
            "   acc[1] += yl[4u*i + 1u] * float(int((q2[i] & 0x0fu) | ((qh[i] & 0x0cu) << 2u)) - 32);",
            "   acc[2] += yl[4u*i + 2u] * float(int((q1[i] >> 4u) | (qh[i] & 0x30u)) - 32);",
            "   acc[3] += yl[4u*i + 3u] * float(int((q2[i] >> 4u) | ((qh[i] & 0xc0u) >> 2u)) - 32);",
            "  }",
            "  float d = float(*(const device half*)(weight + base + 208u));",
            "  sums[row] += d * (acc[0] * float(scales[0]) + acc[1] * float(scales[2]) +",
            "                    acc[2] * float(scales[4]) + acc[3] * float(scales[6]));",
            " }",
            "}",
            "for (uint row = 0; row < ROWS_PER_SIMD; ++row) {",
            " float total = simd_sum(sums[row]);",
            " uint out_col = first_out + row;",
            " if (lane == 0u && out_col < OUT_DIM) out[out_col] = T(total);",
            "}"
        ),
        "",
        true,
        false,
    )
}

pub(super) fn qk_matmul_kernel(format: NativeQuantizationFormat) -> Result<MetalKernel, Exception> {
    let name = match format {
        NativeQuantizationFormat::GgufQ4K => "native_q4k_matmul_64x32",
        NativeQuantizationFormat::GgufQ5K => "native_q5k_matmul_64x32",
        NativeQuantizationFormat::GgufQ6K => "native_q6k_matmul_64x32",
        _ => {
            return Err(Exception::custom(format!(
                "no tiled K-quant kernel for {format:?}"
            )))
        }
    };
    // This is the architecture-neutral 64x32 GGML Metal tile: four SIMD
    // groups cooperatively dequantize a 64x32 weight tile and load a 32x32
    // activation tile, then accumulate eight 8x8 output fragments per SIMD.
    // The layout follows llama.cpp's MIT-licensed matrix-matrix kernel while
    // retaining safemlx's raw-byte storage and activation/output dtypes.
    MetalKernel::new(
        name,
        ["input", "weight"],
        ["out"],
        concat!(
            "constexpr uint NR0 = 64u;",
            "constexpr uint NR1 = 32u;",
            "constexpr uint NK = 32u;",
            "uint input_size = 1u;",
            "for (int dim = 0; dim < input_ndim; ++dim) input_size *= uint(input_shape[dim]);",
            "uint in_dim = uint(input_shape[input_ndim - 1]);",
            "uint rows = input_size / in_dim;",
            "uint blocks = in_dim / 256u;",
            "uint weight_size = 1u;",
            "for (int dim = 0; dim < weight_ndim; ++dim) weight_size *= uint(weight_shape[dim]);",
            "uint out_dim = weight_size / (blocks * QK_BLOCK_BYTES);",
            "uint tid = uint(thread_index_in_threadgroup);",
            "uint simd = uint(simdgroup_index_in_threadgroup);",
            "uint output_tile = uint(threadgroup_position_in_grid.y);",
            "uint activation_tile = uint(threadgroup_position_in_grid.z);",
            "uint r0 = output_tile * NR0;",
            "uint r1 = activation_tile * NR1;",
            "uint nr0 = min(NR0, out_dim - r0);",
            "uint nr1 = min(NR1, rows - r1);",
            "uint il0 = tid & 1u;",
            "uint lr0 = min(tid >> 1u, nr0 - 1u);",
            "uint lr1 = min(tid >> 2u, nr1 - 1u);",
            "uint iy = 8u * (tid & 3u);",
            "threadgroup half sa[2048];",
            "threadgroup half sb[1024];",
            "simdgroup_half8x8 ma[4];",
            "simdgroup_half8x8 mb[2];",
            "simdgroup_float8x8 mc[8];",
            "for (uint i = 0u; i < 8u; ++i) {",
            " mc[i].thread_elements()[0] = 0.0f;",
            " mc[i].thread_elements()[1] = 0.0f;",
            "}",
            "uint row_base = (r0 + lr0) * blocks * QK_BLOCK_BYTES;",
            "for (uint loop_k = 0u; loop_k < in_dim; loop_k += NK) {",
            " half values[16];",
            " qk_dequantize_16(weight, row_base, (loop_k >> 4u) + il0, values);",
            " threadgroup_barrier(mem_flags::mem_threadgroup);",
            " for (uint i = 0u; i < 16u; ++i) {",
            "  uint sx = 2u * il0 + i / 8u;",
            "  uint sy = (tid >> 1u) / 8u;",
            "  uint lx = (tid >> 1u) & 7u;",
            "  uint ly = i & 7u;",
            "  uint ib = 8u * sx + sy;",
            "  sa[64u * ib + 8u * ly + lx] = values[i];",
            " }",
            " uint input_base = (r1 + lr1) * in_dim + loop_k + iy;",
            " for (uint i = 0u; i < 8u; ++i) {",
            "  uint sx = tid & 3u;",
            "  uint sy = (tid >> 2u) / 8u;",
            "  uint ly = (tid >> 2u) & 7u;",
            "  uint ib = 4u * sx + sy;",
            "  sb[64u * ib + 8u * ly + i] = half(input[input_base + i]);",
            " }",
            " threadgroup_barrier(mem_flags::mem_threadgroup);",
            " threadgroup const half* lsma = sa + 4u * 64u * (simd & 1u);",
            " threadgroup const half* lsmb = sb + 2u * 64u * (simd >> 1u);",
            " for (uint ik = 0u; ik < 4u; ++ik) {",
            "  simdgroup_barrier(mem_flags::mem_none);",
            "  for (uint i = 0u; i < 4u; ++i) simdgroup_load(ma[i], lsma + 64u * i, 8u, 0u, false);",
            "  simdgroup_barrier(mem_flags::mem_none);",
            "  for (uint i = 0u; i < 2u; ++i) simdgroup_load(mb[i], lsmb + 64u * i, 8u, 0u, false);",
            "  simdgroup_barrier(mem_flags::mem_none);",
            "  for (uint i = 0u; i < 8u; ++i)",
            "   simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i & 3u], mc[i]);",
            "  lsma += 8u * 64u;",
            "  lsmb += 4u * 64u;",
            " }",
            "}",
            "uint fragment_r0 = 32u * (simd & 1u);",
            "uint fragment_r1 = 16u * (simd >> 1u);",
            " threadgroup float result[2048];",
            " threadgroup float* tile = result + fragment_r0 + fragment_r1 * NR0;",
            " for (uint i = 0u; i < 8u; ++i)",
            "  simdgroup_store(mc[i], tile + 8u * (i & 3u) + 8u * NR0 * (i / 4u), NR0, 0u, false);",
            " threadgroup_barrier(mem_flags::mem_threadgroup);",
            " if (simd == 0u) {",
            "  for (uint index = tid; index < nr0 * nr1; index += 32u) {",
            "   uint row = index / nr0;",
            "   uint col = index - row * nr0;",
            "   out[(r1 + row) * out_dim + r0 + col] = T(result[row * NR0 + col]);",
            "  }",
            "}"
        ),
        qk_matmul_header(format)?,
        true,
        false,
    )
}

pub(super) fn qk_matmul_header(format: NativeQuantizationFormat) -> Result<String, Exception> {
    let (block_bytes, body) = match format {
        NativeQuantizationFormat::GgufQ4K => (
            Q4_K_BLOCK_BYTES,
            concat!(
                "uint group = chunk_in_block >> 1u;",
                "uint scale; uint minv;",
                "if (group < 4u) {",
                " scale = uint(weight[base + 4u + group]) & 63u;",
                " minv = uint(weight[base + 8u + group]) & 63u;",
                "} else {",
                " scale = (uint(weight[base + 8u + group]) & 15u) | ((uint(weight[base + group]) >> 6u) << 4u);",
                " minv = (uint(weight[base + 8u + group]) >> 4u) | ((uint(weight[base + 4u + group]) >> 6u) << 4u);",
                "}",
                "float d = qk_half(weight, base); float dm = qk_half(weight, base + 2u);",
                "for (uint j = 0u; j < 16u; ++j) {",
                " uint i = (chunk_in_block * 16u + j) & 31u;",
                " uint packed = uint(weight[base + 16u + (group >> 1u) * 32u + i]);",
                " uint q = (group & 1u) == 0u ? (packed & 15u) : (packed >> 4u);",
                " values[j] = half(d * float(scale) * float(q) - dm * float(minv));",
                "}"
            ),
        ),
        NativeQuantizationFormat::GgufQ5K => (
            Q5_K_BLOCK_BYTES,
            concat!(
                "uint group = chunk_in_block >> 1u;",
                "uint scale; uint minv;",
                "if (group < 4u) {",
                " scale = uint(weight[base + 4u + group]) & 63u;",
                " minv = uint(weight[base + 8u + group]) & 63u;",
                "} else {",
                " scale = (uint(weight[base + 8u + group]) & 15u) | ((uint(weight[base + group]) >> 6u) << 4u);",
                " minv = (uint(weight[base + 8u + group]) >> 4u) | ((uint(weight[base + 4u + group]) >> 6u) << 4u);",
                "}",
                "float d = qk_half(weight, base); float dm = qk_half(weight, base + 2u);",
                "for (uint j = 0u; j < 16u; ++j) {",
                " uint i = (chunk_in_block * 16u + j) & 31u;",
                " uint packed = uint(weight[base + 48u + (group >> 1u) * 32u + i]);",
                " uint q = (group & 1u) == 0u ? (packed & 15u) : (packed >> 4u);",
                " q |= ((uint(weight[base + 16u + i]) >> group) & 1u) << 4u;",
                " values[j] = half(d * float(scale) * float(q) - dm * float(minv));",
                "}"
            ),
        ),
        NativeQuantizationFormat::GgufQ6K => (
            Q6_K_BLOCK_BYTES,
            concat!(
                "uint section = chunk_in_block >> 3u;",
                "uint scale_group = chunk_in_block & 7u;",
                "int scale = int(as_type<char>(weight[base + 192u + section * 8u + scale_group]));",
                "float d = qk_half(weight, base + 208u);",
                "for (uint j = 0u; j < 16u; ++j) {",
                " uint x = chunk_in_block * 16u + j; uint z = x & 127u;",
                " uint quarter = z >> 5u; uint i = z & 31u;",
                " uint ql = base + section * 64u; uint high = uint(weight[base + 128u + section * 32u + i]);",
                " uint q;",
                " if (quarter == 0u) q = (uint(weight[ql + i]) & 15u) | ((high & 3u) << 4u);",
                " else if (quarter == 1u) q = (uint(weight[ql + 32u + i]) & 15u) | (((high >> 2u) & 3u) << 4u);",
                " else if (quarter == 2u) q = (uint(weight[ql + i]) >> 4u) | (((high >> 4u) & 3u) << 4u);",
                " else q = (uint(weight[ql + 32u + i]) >> 4u) | (((high >> 6u) & 3u) << 4u);",
                " values[j] = half(d * float(scale) * float(int(q) - 32));",
                "}"
            ),
        ),
        _ => {
            return Err(Exception::custom(format!(
                "no tiled K-quant dequantizer for {format:?}"
            )))
        }
    };
    Ok(format!(
        concat!(
            "#include <metal_simdgroup_matrix>\n",
            "constant uint QK_BLOCK_BYTES = {block_bytes}u;\n",
            "float qk_half(const device uint8_t* weight, uint offset) {{",
            " uint bits = uint(weight[offset]) | (uint(weight[offset + 1u]) << 8u);",
            " return float(as_type<half>(ushort(bits)));",
            "}}\n",
            "void qk_dequantize_16(const device uint8_t* weight, uint row_base, uint chunk, thread half* values) {{",
            " uint block = chunk >> 4u; uint chunk_in_block = chunk & 15u;",
            " uint base = row_base + block * QK_BLOCK_BYTES;",
            " {body}",
            "}}\n"
        ),
        block_bytes = block_bytes,
        body = body,
    ))
}

pub(super) fn q5k_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5k_small_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint simd_group = thread_position_in_grid.y & 1u;",
            "uint output_group = thread_position_in_grid.y >> 1u;",
            "uint tx = lane & 7u;",
            "uint ty = lane >> 3u;",
            "uint out_col = output_group * 8u + simd_group * 4u + ty;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc[BATCH_TILE];",
            "for (uint row = 0; row < BATCH_TILE; ++row) acc[row] = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint row_base = physical_row * BLOCKS * 176u;",
            " for (uint chunk = tx; chunk < BLOCKS * 16u; chunk += 8u) {",
            "  uint block = chunk >> 4u;",
            "  uint chunk_in_block = chunk & 15u;",
            "  uint base = row_base + block * 176u;",
            "  float d = float(*(const device half*)(weight + base));",
            "  float dm = float(*(const device half*)(weight + base + 2u));",
            "  uint group = chunk_in_block >> 1u;",
            "  uint scale; uint minv;",
            "  if (group < 4u) {",
            "   scale = uint(weight[base + 4u + group]) & 63u;",
            "   minv = uint(weight[base + 8u + group]) & 63u;",
            "  } else {",
            "   scale = (uint(weight[base + 8u + group]) & 15u) |",
            "           ((uint(weight[base + group]) >> 6u) << 4u);",
            "   minv = (uint(weight[base + 8u + group]) >> 4u) |",
            "          ((uint(weight[base + 4u + group]) >> 6u) << 4u);",
            "  }",
            "  uint col_base = chunk * 16u;",
            "  for (uint j = 0; j < 16u; ++j) {",
            "   uint i = (chunk_in_block * 16u + j) & 31u;",
            "   uint packed = uint(weight[base + 48u + (group >> 1u) * 32u + i]);",
            "   uint q = (group & 1u) == 0u ? (packed & 15u) : (packed >> 4u);",
            "   q |= ((uint(weight[base + 16u + i]) >> group) & 1u) << 4u;",
            "   float value = d * float(scale) * float(q) - dm * float(minv);",
            "   for (uint local_row = 0; local_row < BATCH_TILE; ++local_row) {",
            "    uint row = first_row + local_row;",
            "    if (row < ROWS) acc[local_row] += float(input[row * IN_DIM + col_base + j]) * value;",
            "   }",
            "  }",
            " }",
            "}",
            "for (uint local_row = 0; local_row < BATCH_TILE; ++local_row) {",
            " float total = acc[local_row];",
            " total += simd_shuffle_down(total, 4u);",
            " total += simd_shuffle_down(total, 2u);",
            " total += simd_shuffle_down(total, 1u);",
            " uint row = first_row + local_row;",
            " if (tx == 0u && row < ROWS && out_col < OUT_DIM)",
            "  out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        "",
        true,
        false,
    )
}

pub(super) fn q6k_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q6k_small_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint simd_group = thread_position_in_grid.y & 1u;",
            "uint output_group = thread_position_in_grid.y >> 1u;",
            "uint tx = lane & 7u;",
            "uint ty = lane >> 3u;",
            "uint out_col = output_group * 8u + simd_group * 4u + ty;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc[BATCH_TILE];",
            "for (uint row = 0; row < BATCH_TILE; ++row) acc[row] = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint row_base = physical_row * BLOCKS * 210u;",
            " for (uint chunk = tx; chunk < BLOCKS * 16u; chunk += 8u) {",
            "  uint block = chunk >> 4u;",
            "  uint chunk_in_block = chunk & 15u;",
            "  uint base = row_base + block * 210u;",
            "  float d = float(*(const device half*)(weight + base + 208u));",
            "  uint col_base = chunk * 16u;",
            "  uint scale_group = chunk_in_block & 7u;",
            "  uint section = chunk_in_block >> 3u;",
            "  int scale = int(as_type<char>(weight[base + 192u + section * 8u + scale_group]));",
            "  for (uint j = 0; j < 16u; ++j) {",
            "   uint x = chunk_in_block * 16u + j;",
            "   uint section = x / 128u; uint z = x % 128u;",
            "   uint quarter = z / 32u; uint i = z % 32u;",
            "   uint ql = base + section * 64u;",
            "   uint high = uint(weight[base + 128u + section * 32u + i]);",
            "   uint q;",
            "   if (quarter == 0u) q = (uint(weight[ql + i]) & 15u) | ((high & 3u) << 4u);",
            "   else if (quarter == 1u) q = (uint(weight[ql + 32u + i]) & 15u) | (((high >> 2u) & 3u) << 4u);",
            "   else if (quarter == 2u) q = (uint(weight[ql + i]) >> 4u) | (((high >> 4u) & 3u) << 4u);",
            "   else q = (uint(weight[ql + 32u + i]) >> 4u) | (((high >> 6u) & 3u) << 4u);",
            "   float value = d * float(scale) * float(int(q) - 32);",
            "   for (uint local_row = 0; local_row < BATCH_TILE; ++local_row) {",
            "    uint row = first_row + local_row;",
            "    if (row < ROWS) acc[local_row] += float(input[row * IN_DIM + col_base + j]) * value;",
            "   }",
            "  }",
            " }",
            "}",
            "for (uint local_row = 0; local_row < BATCH_TILE; ++local_row) {",
            " float total = acc[local_row];",
            " total += simd_shuffle_down(total, 4u);",
            " total += simd_shuffle_down(total, 2u);",
            " total += simd_shuffle_down(total, 1u);",
            " uint row = first_row + local_row;",
            " if (tx == 0u && row < ROWS && out_col < OUT_DIM)",
            "  out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        "",
        true,
        false,
    )
}

pub(super) fn q4k_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc[BATCH_TILE];",
            "for (uint r = 0; r < BATCH_TILE; ++r) acc[r] = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 144;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 144;",
            "  uint input_block = block * 256;",
            "  for (uint g = 0; g < 8; ++g) {",
            "   float w = q4k_value(weight, base, g, lane);",
            "   uint col = input_block + g * 32 + lane;",
            "   for (uint r = 0; r < BATCH_TILE; ++r) {",
            "    uint row = first_row + r;",
            "    if (row < ROWS) acc[r] += float(input[row * IN_DIM + col]) * w;",
            "   }",
            "  }",
            " }",
            "}",
            "for (uint r = 0; r < BATCH_TILE; ++r) {",
            " float total = simd_sum(acc[r]);",
            " uint row = first_row + r;",
            " if (lane == 0 && row < ROWS && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q5_1_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_linear",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint row = thread_position_in_grid.y / OUT_GRID;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 24;",
            " for (uint block = lane; block < BLOCKS; block += REDUCTION_TILE) {",
            "  uint base = matrix_base + block * 24;",
            "  uint input_block = row * IN_DIM + block * 32;",
            "  for (uint i = 0; i < 32; ++i) {",
            "   acc += float(input[input_block + i]) * q5_1_value(weight, base, i);",
            "  }",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);"
        ),
        Q5_1_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q5_1_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc[BATCH_TILE];",
            "for (uint r = 0; r < BATCH_TILE; ++r) acc[r] = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 24;",
            " for (uint block = lane; block < BLOCKS; block += REDUCTION_TILE) {",
            "  uint base = matrix_base + block * 24;",
            "  for (uint i = 0; i < 32; ++i) {",
            "   float w = q5_1_value(weight, base, i);",
            "   uint col = block * 32 + i;",
            "   for (uint r = 0; r < BATCH_TILE; ++r) {",
            "    uint row = first_row + r;",
            "    if (row < ROWS) acc[r] += float(input[row * IN_DIM + col]) * w;",
            "   }",
            "  }",
            " }",
            "}",
            "for (uint r = 0; r < BATCH_TILE; ++r) {",
            " float total = simd_sum(acc[r]);",
            " uint row = first_row + r;",
            " if (lane == 0 && row < ROWS && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        Q5_1_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q4k_grouped_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_grouped",
        ["input", "weight", "group_ids"],
        ["out"],
        [
            Q4K_TILED_PROLOGUE,
            "uint group = uint(group_ids[row]);",
            "uint physical_row = group * PHYSICAL_ROWS + ROW_START + out_col;",
            "uint matrix_base = physical_row * BLOCKS * 144;",
            Q4K_ACCUMULATE,
            Q4K_TILED_EPILOGUE,
        ]
        .concat(),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q5_1_grouped_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_grouped",
        ["input", "weight", "group_ids"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint row = thread_position_in_grid.y / OUT_GRID;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint group = uint(group_ids[row]);",
            " uint physical_row = group * PHYSICAL_ROWS + ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 24;",
            " for (uint block = lane; block < BLOCKS; block += REDUCTION_TILE) {",
            "  uint base = matrix_base + block * 24;",
            "  uint input_block = row * IN_DIM + block * 32;",
            "  for (uint i = 0; i < 32; ++i) {",
            "   acc += float(input[input_block + i]) * q5_1_value(weight, base, i);",
            "  }",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);"
        ),
        Q5_1_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q5_1_embedding_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q5_1_embedding",
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint block = col / 32;",
            "uint within = col % 32;",
            "uint base = (physical_row * BLOCKS + block) * 24;",
            "out[elem] = q5_1_value(weight, base, within);"
        ),
        Q5_1_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q4k_embedding_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q4k_embedding",
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint block = col / 256;",
            "uint within = col % 256;",
            "uint g = within / 32;",
            "uint i = within % 32;",
            "uint base = (physical_row * BLOCKS + block) * 144;",
            "out[elem] = q4k_value(weight, base, g, i);"
        ),
        Q4K_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q8_0_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_linear",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 34;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 34;",
            "  float d = q8_0_scale(weight, base);",
            "  int q = q8_0_quant(weight, base + 2 + lane);",
            "  acc += float(input[block * 32 + lane]) * d * float(q);",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[out_col] = T(total);"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q8_0_batch_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_batch",
        ["input", "weight"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint out_col = thread_position_in_grid.y;",
            "uint first_row = thread_position_in_grid.z * BATCH_TILE;",
            "float acc[BATCH_TILE];",
            "for (uint r = 0; r < BATCH_TILE; ++r) acc[r] = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint physical_row = ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 34;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 34;",
            "  float w = q8_0_scale(weight, base) * float(q8_0_quant(weight, base + 2 + lane));",
            "  uint input_col = block * 32 + lane;",
            "  for (uint r = 0; r < BATCH_TILE; ++r) {",
            "   uint row = first_row + r;",
            "   if (row < ROWS) acc[r] += float(input[row * IN_DIM + input_col]) * w;",
            "  }",
            " }",
            "}",
            "for (uint r = 0; r < BATCH_TILE; ++r) {",
            " float total = simd_sum(acc[r]);",
            " uint row = first_row + r;",
            " if (lane == 0 && row < ROWS && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);",
            "}"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q8_0_grouped_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_grouped",
        ["input", "weight", "group_ids"],
        ["out"],
        concat!(
            "uint lane = thread_position_in_grid.x;",
            "uint row = thread_position_in_grid.y / OUT_GRID;",
            "uint out_col = thread_position_in_grid.y % OUT_GRID;",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " uint group = uint(group_ids[row]);",
            " uint physical_row = group * PHYSICAL_ROWS + ROW_START + out_col;",
            " uint matrix_base = physical_row * BLOCKS * 34;",
            " for (uint block = 0; block < BLOCKS; ++block) {",
            "  uint base = matrix_base + block * 34;",
            "  float d = q8_0_scale(weight, base);",
            "  int q = q8_0_quant(weight, base + 2 + lane);",
            "  acc += float(input[row * IN_DIM + block * 32 + lane]) * d * float(q);",
            " }",
            "}",
            "float total = simd_sum(acc);",
            "if (lane == 0 && out_col < OUT_DIM) out[row * OUT_DIM + out_col] = T(total);"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

pub(super) fn q8_0_embedding_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "native_q8_0_embedding",
        ["weight", "indices"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "uint col = elem % IN_DIM;",
            "uint output_row = elem / IN_DIM;",
            "uint row = uint(indices[output_row]);",
            "if (row >= ROWS) { out[elem] = 0.0f; return; }",
            "uint physical_row = ROW_START + row;",
            "uint block = col / 32;",
            "uint lane = col % 32;",
            "uint base = (physical_row * BLOCKS + block) * 34;",
            "out[elem] = q8_0_scale(weight, base) * float(q8_0_quant(weight, base + 2 + lane));"
        ),
        Q8_0_METAL_HEADER,
        true,
        false,
    )
}

const Q4K_TILED_PROLOGUE: &str = concat!(
    "uint lane = thread_position_in_grid.x;",
    "uint row = thread_position_in_grid.y / OUT_GRID;",
    "uint out_col = thread_position_in_grid.y % OUT_GRID;",
    "uint local_col = out_col % OUT_TILE;",
    "float acc = 0.0f;",
    "if (out_col < OUT_DIM) {"
);

const Q4K_ACCUMULATE: &str = concat!(
    " for (uint block = 0; block < BLOCKS; ++block) {",
    "  uint base = matrix_base + block * 144;",
    "  uint input_block = row * IN_DIM + block * 256;",
    "  for (uint g = 0; g < 8; ++g) {",
    "   for (uint i = lane; i < 32; i += REDUCTION_TILE) {",
    "    acc += float(input[input_block + g * 32 + i]) * q4k_value(weight, base, g, i);",
    "   }",
    "  }",
    " }",
    "}"
);

const Q4K_TILED_EPILOGUE: &str = concat!(
    "float total = simd_sum(acc);",
    "if (lane == 0 && out_col < OUT_DIM) {",
    " out[row * OUT_DIM + out_col] = T(total);",
    "}"
);

// Q4_K layout and scale unpacking follow llama.cpp's ggml-quants reference
// implementation (MIT) and MLX's GGUF converter (MIT).
const Q4K_METAL_HEADER: &str = concat!(
    "float q4k_value(const device uint8_t* weight, uint base, uint g, uint i) {",
    " uint d_bits = uint(weight[base]) | (uint(weight[base + 1]) << 8);",
    " uint dm_bits = uint(weight[base + 2]) | (uint(weight[base + 3]) << 8);",
    " float d = float(as_type<half>(ushort(d_bits)));",
    " float dm = float(as_type<half>(ushort(dm_bits)));",
    " uint sc;",
    " uint m;",
    " if (g < 4) {",
    "  sc = uint(weight[base + 4 + g]) & 63u;",
    "  m = uint(weight[base + 8 + g]) & 63u;",
    " } else {",
    "  sc = (uint(weight[base + 8 + g]) & 15u) | ((uint(weight[base + g]) >> 6) << 4);",
    "  m = (uint(weight[base + 8 + g]) >> 4) | ((uint(weight[base + 4 + g]) >> 6) << 4);",
    " }",
    " uint packed = uint(weight[base + 16 + (g / 2) * 32 + i]);",
    " uint q = (g & 1u) == 0u ? (packed & 15u) : (packed >> 4);",
    " return d * float(sc) * float(q) - dm * float(m);",
    "}\n"
);

// Q5_1 layout follows llama.cpp's ggml-quants reference implementation
// (MIT) and MLX's GGUF converter (MIT).
const Q5_1_METAL_HEADER: &str = concat!(
    "float q5_1_value(const device uint8_t* weight, uint base, uint i) {",
    " uint d_bits = uint(weight[base]) | (uint(weight[base + 1]) << 8);",
    " uint m_bits = uint(weight[base + 2]) | (uint(weight[base + 3]) << 8);",
    " float d = float(as_type<half>(ushort(d_bits)));",
    " float m = float(as_type<half>(ushort(m_bits)));",
    " uint qh = uint(weight[base + 4]) | (uint(weight[base + 5]) << 8) |",
    "           (uint(weight[base + 6]) << 16) | (uint(weight[base + 7]) << 24);",
    " uint packed = uint(weight[base + 8 + (i & 15u)]);",
    " uint low = i < 16 ? (packed & 15u) : (packed >> 4);",
    " uint q = low | (((qh >> i) & 1u) << 4);",
    " return d * float(q) + m;",
    "}\n"
);

// Q8_0 layout follows llama.cpp's ggml-quants reference implementation
// (MIT) and MLX's GGUF converter (MIT).
const Q8_0_METAL_HEADER: &str = concat!(
    "float q8_0_scale(const device uint8_t* weight, uint base) {",
    " uint d_bits = uint(weight[base]) | (uint(weight[base + 1]) << 8);",
    " return float(as_type<half>(ushort(d_bits)));",
    "}\n",
    "int q8_0_quant(const device uint8_t* weight, uint offset) {",
    " return int(as_type<char>(weight[offset]));",
    "}\n"
);

pub(super) fn iq_metal_array<T: std::fmt::LowerHex>(
    output: &mut String,
    metal_type: &str,
    name: &str,
    values: &[T],
) {
    let _ = write!(output, "constant {metal_type} {name}[{}]={{", values.len());
    for value in values {
        let _ = write!(output, "0x{value:x},");
    }
    output.push_str("};\n");
}

pub(super) fn iq_codebook(format: NativeQuantizationFormat) -> IQuantCodebook {
    format
        .ggml_type()
        .and_then(IQuantCodebook::for_type)
        .expect("IQ native format must have a canonical GGUF codebook")
}

pub(super) fn iq_metal_header(format: NativeQuantizationFormat, big_endian: bool) -> String {
    let mut header = format!(
        "constant bool BIG_ENDIAN = {};\n",
        if big_endian { "true" } else { "false" }
    );
    header.push_str(concat!(
        "ushort iq_u16(const device uint8_t* w,uint p){",
        "return BIG_ENDIAN ? ushort((uint(w[p])<<8)|uint(w[p+1])) : ushort(uint(w[p])|(uint(w[p+1])<<8));}\n",
        "uint iq_u32(const device uint8_t* w,uint p){",
        "return BIG_ENDIAN ? ((uint(w[p])<<24)|(uint(w[p+1])<<16)|(uint(w[p+2])<<8)|uint(w[p+3]))",
        ": (uint(w[p])|(uint(w[p+1])<<8)|(uint(w[p+2])<<16)|(uint(w[p+3])<<24));}\n",
        "float iq_half(const device uint8_t* w,uint p){return float(as_type<half>(iq_u16(w,p)));}\n",
        "uint iq_grid8(const constant ulong* t,uint i,uint j){return uint((t[i]>>(8*j))&255ul);}\n",
        "uint iq_grid4(const constant uint* t,uint i,uint j){return (t[i]>>(8*j))&255u;}\n",
        "float iq_sign(float v,uint signs,uint j){return ((signs>>j)&1u)!=0u ? -v:v;}\n"
    ));

    match format {
        NativeQuantizationFormat::GgufQ5K => {
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint i=x%32u;uint p=r+b*176u;",
                "uint s;uint m;if(g<4u){s=uint(w[p+4u+g])&63u;m=uint(w[p+8u+g])&63u;}",
                "else{s=(uint(w[p+8u+g])&15u)|((uint(w[p+g])>>6u)<<4u);",
                "m=(uint(w[p+8u+g])>>4u)|((uint(w[p+4u+g])>>6u)<<4u);}",
                "uint packed=uint(w[p+48u+(g/2u)*32u+i]);",
                "uint q=(g&1u)==0u?(packed&15u):(packed>>4u);",
                "q|=((uint(w[p+16u+i])>>g)&1u)<<4u;",
                "return iq_half(w,p)*float(s)*float(q)-iq_half(w,p+2u)*float(m);}\n"
            ));
        }
        NativeQuantizationFormat::GgufQ6K => {
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint section=x/128u;uint z=x%128u;",
                "uint quarter=z/32u;uint i=z%32u;uint p=r+b*210u;",
                "uint ql=p+section*64u;uint h=uint(w[p+128u+section*32u+i]);uint q;",
                "if(quarter==0u)q=(uint(w[ql+i])&15u)|((h&3u)<<4u);",
                "else if(quarter==1u)q=(uint(w[ql+32u+i])&15u)|(((h>>2u)&3u)<<4u);",
                "else if(quarter==2u)q=(uint(w[ql+i])>>4u)|(((h>>4u)&3u)<<4u);",
                "else q=(uint(w[ql+32u+i])>>4u)|(((h>>6u)&3u)<<4u);",
                "uint group=z/16u;int sc=int(as_type<char>(w[p+192u+section*8u+group]));",
                "return iq_half(w,p+208u)*float(sc)*float(int(q)-32);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ2XXS => {
            let codebook = iq_codebook(format);
            iq_metal_array(
                &mut header,
                "uchar",
                "IQ_SIGNS",
                codebook.signs().expect("IQ2_XXS sign codebook"),
            );
            iq_metal_array(
                &mut header,
                "ulong",
                "IQ_GRID",
                codebook.u64_values().expect("IQ2_XXS grid codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*66u;float d=iq_half(w,p);uint q=p+2u+8u*g;",
                "uint w0=uint(iq_u16(w,q+2u*(l/2u)));",
                "uint aux=uint(iq_u16(w,q+4u))|(uint(iq_u16(w,q+6u))<<16);",
                "uint index=(l&1u)==0u?(w0&255u):(w0>>8);",
                "float db=d*(0.5f+float(aux>>28))*0.25f;",
                "return iq_sign(db*float(iq_grid8(IQ_GRID,index,j)),uint(IQ_SIGNS[(aux>>(7u*l))&127u]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ2XS => {
            let codebook = iq_codebook(format);
            iq_metal_array(
                &mut header,
                "uchar",
                "IQ_SIGNS",
                codebook.signs().expect("IQ2_XS sign codebook"),
            );
            iq_metal_array(
                &mut header,
                "ulong",
                "IQ_GRID",
                codebook.u64_values().expect("IQ2_XS grid codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*74u;float d=iq_half(w,p);uint q=uint(iq_u16(w,p+2u+2u*(4u*g+l)));",
                "uint s=uint(w[p+66u+g]);uint nib=(l/2u)==0u?(s&15u):(s>>4);",
                "float db=d*(0.5f+float(nib))*0.25f;",
                "return iq_sign(db*float(iq_grid8(IQ_GRID,q&511u,j)),uint(IQ_SIGNS[q>>9]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ2S => {
            iq_metal_array(
                &mut header,
                "ulong",
                "IQ_GRID",
                iq_codebook(format)
                    .u64_values()
                    .expect("IQ2_S grid codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*82u;float d=iq_half(w,p);uint qh=uint(w[p+66u+g]);",
                "uint index=uint(w[p+2u+4u*g+l])|((qh<<(8u-2u*l))&0x300u);",
                "uint s=uint(w[p+74u+g]);uint nib=(l/2u)==0u?(s&15u):(s>>4);",
                "float db=d*(0.5f+float(nib))*0.25f;",
                "return iq_sign(db*float(iq_grid8(IQ_GRID,index,j)),uint(w[p+34u+4u*g+l]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ3XXS => {
            let codebook = iq_codebook(format);
            iq_metal_array(
                &mut header,
                "uchar",
                "IQ_SIGNS",
                codebook.signs().expect("IQ3_XXS sign codebook"),
            );
            iq_metal_array(
                &mut header,
                "uint",
                "IQ_GRID",
                codebook.u32_values().expect("IQ3_XXS grid codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*98u;float d=iq_half(w,p);uint aux=iq_u32(w,p+66u+4u*g);",
                "uint qi=uint(w[p+2u+8u*g+2u*l+(j/4u)]);",
                "float db=d*(0.5f+float(aux>>28))*0.5f;",
                "return iq_sign(db*float(iq_grid4(IQ_GRID,qi,j%4u)),uint(IQ_SIGNS[(aux>>(7u*l))&127u]),j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ3S => {
            iq_metal_array(
                &mut header,
                "uint",
                "IQ_GRID",
                iq_codebook(format)
                    .u32_values()
                    .expect("IQ3_S grid codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint pair=x/64u;uint side=(x%64u)/32u;",
                "uint l=(x%32u)/8u;uint j=x%8u;uint p=r+b*110u;float d=iq_half(w,p);",
                "uint scale=uint(w[p+106u+pair]);uint nib=side==0u?(scale&15u):(scale>>4);",
                "float db=d*float(1u+2u*nib);uint high=uint(w[p+66u+2u*pair+side]);",
                "uint qoff=p+2u+16u*pair+8u*side+2u*l+(j/4u);",
                "uint index=uint(w[qoff])|((high<<(8u-2u*l-(j/4u)))&256u);",
                "uint signs=uint(w[p+74u+8u*pair+4u*side+l]);",
                "return iq_sign(db*float(iq_grid4(IQ_GRID,index,j%4u)),signs,j);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ1S => {
            iq_metal_array(
                &mut header,
                "ulong",
                "IQ_GRID",
                iq_codebook(format)
                    .u64_values()
                    .expect("IQ1_S grid codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*50u;float d=iq_half(w,p);uint hi=uint(iq_u16(w,p+34u+2u*g));",
                "float dl=d*float(2u*((hi>>12)&7u)+1u);float delta=(hi&0x8000u)!=0u?-0.125f:0.125f;",
                "uint index=uint(w[p+2u+4u*g+l])|(((hi>>(3u*l))&7u)<<8);",
                "int q=int(as_type<char>(uchar(iq_grid8(IQ_GRID,index,j))));return dl*(float(q)+delta);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ1M => {
            iq_metal_array(
                &mut header,
                "ulong",
                "IQ_GRID",
                iq_codebook(format)
                    .u64_values()
                    .expect("IQ1_M grid codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint l=(x%32u)/8u;uint j=x%8u;",
                "uint p=r+b*56u;uint s0=uint(iq_u16(w,p+48u));uint s1=uint(iq_u16(w,p+50u));",
                "uint s2=uint(iq_u16(w,p+52u));uint s3=uint(iq_u16(w,p+54u));",
                "uint dbits=(s0>>12)|((s1>>8)&0xf0u)|((s2>>4)&0xf00u)|(s3&0xf000u);",
                "float d=float(as_type<half>(ushort(dbits)));uint sw=g<2u?s0:(g<4u?s1:(g<6u?s2:s3));",
                "uint shift=6u*(g&1u)+3u*(l/2u);float dl=d*float(2u*((sw>>shift)&7u)+1u);",
                "uint h=uint(w[p+32u+2u*g+l/2u]);uint index=uint(w[p+4u*g+l])|",
                "(((h>>((l&1u)*4u))&7u)<<8);uint negbit=(l&1u)==0u?8u:128u;",
                "float delta=(h&negbit)!=0u?-0.125f:0.125f;",
                "int q=int(as_type<char>(uchar(iq_grid8(IQ_GRID,index,j))));return dl*(float(q)+delta);}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ4NL => {
            iq_metal_array(
                &mut header,
                "uchar",
                "IQ4",
                iq_codebook(format)
                    .i8_values()
                    .expect("IQ4_NL value codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/32u;uint x=c%32u;uint p=r+b*18u;uint q=uint(w[p+2u+(x%16u)]);",
                "uint code=x<16u?(q&15u):(q>>4);return iq_half(w,p)*float(as_type<char>(IQ4[code]));}\n"
            ));
        }
        NativeQuantizationFormat::GgufIQ4XS => {
            iq_metal_array(
                &mut header,
                "uchar",
                "IQ4",
                iq_codebook(format)
                    .i8_values()
                    .expect("IQ4_XS value codebook"),
            );
            header.push_str(concat!(
                "float iq_value(const device uint8_t* w,uint r,uint c){",
                "uint b=c/256u;uint x=c%256u;uint g=x/32u;uint z=x%32u;uint p=r+b*136u;",
                "uint sh=uint(iq_u16(w,p+2u));uint sl=uint(w[p+4u+g/2u]);",
                "uint low=(sl>>(4u*(g&1u)))&15u;uint high=(sh>>(2u*g))&3u;",
                "float dl=iq_half(w,p)*float(int(low|(high<<4))-32);",
                "uint q=uint(w[p+8u+16u*g+(z%16u)]);uint code=z<16u?(q&15u):(q>>4);",
                "return dl*float(as_type<char>(IQ4[code]));}\n"
            ));
        }
        _ => unreachable!("IQ Metal header requested for non-IQ format"),
    }
    header
}
