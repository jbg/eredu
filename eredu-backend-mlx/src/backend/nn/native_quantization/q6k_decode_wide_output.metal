uint lane = thread_position_in_grid.x;
uint output_pair = thread_position_in_grid.y;
uint first_out = output_pair * ROWS_PER_SIMD;
uint tid = lane % 16u;
uint ix = lane / 16u;
uint ip = tid / 8u;
uint il = tid % 8u;
uint l0 = 4u * il;
uint input_offset = 128u * ip + l0;
uint ql_offset = 64u * ip + l0;
uint qh_offset = 32u * ip + l0;
float sums[ROWS_PER_SIMD];
for (uint row = 0; row < ROWS_PER_SIMD; ++row) sums[row] = 0.0f;
for (uint block = ix; block < BLOCKS; block += 2u) {
 uint input_base = block * 256u + input_offset;
 float yl[16];
 float4 x0 = float4(*(const device packed_float4*)(input + input_base + 0u));
 float4 x1 = float4(*(const device packed_float4*)(input + input_base + 32u));
 float4 x2 = float4(*(const device packed_float4*)(input + input_base + 64u));
 float4 x3 = float4(*(const device packed_float4*)(input + input_base + 96u));
 #pragma unroll
 for (uint i = 0; i < 4u; ++i) {
  yl[4u*i] = x0[i];
  yl[4u*i + 1u] = x1[i];
  yl[4u*i + 2u] = x2[i];
  yl[4u*i + 3u] = x3[i];
 }
 for (uint row = 0; row < ROWS_PER_SIMD; ++row) {
  uint out_col = first_out + row;
  if (out_col >= OUT_DIM) continue;
  uint physical_row = ROW_START + out_col;
  uint base = (physical_row * BLOCKS + block) * 210u;
  const device ushort* q1_u16 = (const device ushort*)(weight + base + ql_offset);
  uchar4 q1 = as_type<uchar4>(ushort2(q1_u16[0], q1_u16[1]));
  const device ushort* q2_u16 = (const device ushort*)(weight + base + ql_offset + 32u);
  uchar4 q2 = as_type<uchar4>(ushort2(q2_u16[0], q2_u16[1]));
  const device ushort* qh_u16 = (const device ushort*)(weight + base + 128u + qh_offset);
  uchar4 qh = as_type<uchar4>(ushort2(qh_u16[0], qh_u16[1]));
  const device packed_ushort4* all_scales = (const device packed_ushort4*)(weight + base + 192u + 8u * ip);
  ushort4 scale_words = ushort4(*all_scales);
  uint shift = (il / 4u) * 8u;
  char4 selected = as_type<char4>(uchar4(scale_words >> shift));
  float4 acc = float4(0.0f);
  #pragma unroll
  for (uint i = 0; i < 4u; ++i) {
   acc[0] += yl[4u*i] * float(int((q1[i] & 0x0fu) | ((qh[i] & 0x03u) << 4u)) - 32);
   acc[1] += yl[4u*i + 1u] * float(int((q2[i] & 0x0fu) | ((qh[i] & 0x0cu) << 2u)) - 32);
   acc[2] += yl[4u*i + 2u] * float(int((q1[i] >> 4u) | (qh[i] & 0x30u)) - 32);
   acc[3] += yl[4u*i + 3u] * float(int((q2[i] >> 4u) | ((qh[i] & 0xc0u) >> 2u)) - 32);
  }
  float d = float(*(const device half*)(weight + base + 208u));
  sums[row] += d * (acc[0] * float(selected[0]) + acc[1] * float(selected[1]) +
                    acc[2] * float(selected[2]) + acc[3] * float(selected[3]));
 }
}
for (uint row = 0; row < ROWS_PER_SIMD; ++row) {
 // Restore the reference lane order after coalescing packed loads.
 float total = simd_sum(simd_shuffle(sums[row], (lane % 2u) * 16u + lane / 2u));
 uint out_col = first_out + row;
 if (lane == 0u && out_col < OUT_DIM) out[out_col] = T(total);
}
