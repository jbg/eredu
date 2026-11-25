uint lane = thread_position_in_grid.x;
uint output_pair = thread_position_in_grid.y;
uint first_out = output_pair * ROWS_PER_SIMD;
uint tid = lane / 2u;
uint ix = lane % 2u;
uint ip = tid / 8u;
uint il = tid % 8u;
uint l0 = 4u * il;
uint scale_offset = 8u * ip + l0 / 16u;
uint input_offset = 128u * ip + l0;
uint ql_offset = 64u * ip + l0;
uint qh_offset = 32u * ip + l0;
float sums[ROWS_PER_SIMD][TOKENS] = {};
for (uint block = ix; block < BLOCKS; block += 2u) {
 for (uint row = 0; row < ROWS_PER_SIMD; ++row) {
  uint out_col = first_out + row;
  if (out_col >= OUT_DIM) continue;
  uint physical_row = ROW_START + out_col;
  uint base = (physical_row * BLOCKS + block) * 210u;
  uchar4 q1 = uchar4(*(const device packed_uchar4*)(weight + base + ql_offset));
  uchar4 q2 = uchar4(*(const device packed_uchar4*)(weight + base + ql_offset + 32u));
  uchar4 qh = uchar4(*(const device packed_uchar4*)(weight + base + 128u + qh_offset));
  const device char* scales = (const device char*)(weight + base + 192u + scale_offset);
  float d = float(*(const device half*)(weight + base + 208u));

  #pragma unroll
  for (uint token = 0u; token < TOKENS; ++token) {
 uint input_base = (token * BLOCKS + block) * 256u + input_offset;
 float yl[16];
 for (uint i = 0; i < 4u; ++i) {
  yl[4u*i] = float(input[input_base + i]);
  yl[4u*i + 1u] = float(input[input_base + 32u + i]);
  yl[4u*i + 2u] = float(input[input_base + 64u + i]);
  yl[4u*i + 3u] = float(input[input_base + 96u + i]);
 }
  float4 acc = float4(0.0f);
  for (uint i = 0; i < 4u; ++i) {
   acc[0] += yl[4u*i] * float(int((q1[i] & 0x0fu) | ((qh[i] & 0x03u) << 4u)) - 32);
   acc[1] += yl[4u*i + 1u] * float(int((q2[i] & 0x0fu) | ((qh[i] & 0x0cu) << 2u)) - 32);
   acc[2] += yl[4u*i + 2u] * float(int((q1[i] >> 4u) | (qh[i] & 0x30u)) - 32);
   acc[3] += yl[4u*i + 3u] * float(int((q2[i] >> 4u) | ((qh[i] & 0xc0u) >> 2u)) - 32);
  }
  sums[row][token] += d * (acc[0] * float(scales[0]) + acc[1] * float(scales[2]) +
                    acc[2] * float(scales[4]) + acc[3] * float(scales[6]));
  }
 }
}
for (uint row = 0; row < ROWS_PER_SIMD; ++row) {
 uint out_col = first_out + row;
 for (uint token = 0; token < TOKENS; ++token) {
  float total = simd_sum(sums[row][token]);
  if (lane == 0u && out_col < OUT_DIM) out[token * OUT_DIM + out_col] = T(total);
 }
}
