uint first_out = output_pair * ROWS_PER_SIMD;
uint ix = lane / 8u;
uint it = lane % 8u;
uint iq = it / 4u;
uint ir = it % 4u;
float sums[ROWS_PER_SIMD];
for (uint row = 0; row < ROWS_PER_SIMD; ++row) sums[row] = 0.0f;
for (uint block = ix; block < BLOCKS; block += 4u) {
 uint input_base = route * BLOCKS * 256u + block * 256u + 64u * iq + 8u * ir;
 float yl[16];
 float yh[16];
 float4 sumy = float4(0.0f);
 for (uint i = 0; i < 8u; ++i) {
  yl[i] = float(input[input_base + i]);
  yl[i + 8u] = float(input[input_base + 32u + i]);
  yh[i] = float(input[input_base + 128u + i]);
  yh[i + 8u] = float(input[input_base + 160u + i]);
  sumy[0] += yl[i];
  sumy[1] += yl[i + 8u];
  sumy[2] += yh[i];
  sumy[3] += yh[i + 8u];
 }
 for (uint row = 0; row < ROWS_PER_SIMD; ++row) {
  uint out_col = first_out + row;
  if (out_col >= OUT_DIM) continue;
  uint physical_row = group * PHYSICAL_ROWS + ROW_START + out_col;
  uint base = (physical_row * BLOCKS + block) * 144u;
  const device ushort* sc = (const device ushort*)(weight + base + 4u) + iq;
  const device ushort* q1 = (const device ushort*)(weight + base + 16u) + 16u * iq + 4u * ir;
  const device ushort* q2 = q1 + 32u;
  ushort sc16[4];
  thread uchar* sc8 = (thread uchar*)sc16;
  sc16[0] = sc[0] & 0x3f3fu;
  sc16[1] = sc[2] & 0x3f3fu;
  sc16[2] = ((sc[4] >> 0u) & 0x0f0fu) | ((sc[0] & 0xc0c0u) >> 2u);
  sc16[3] = ((sc[4] >> 4u) & 0x0f0fu) | ((sc[2] & 0xc0c0u) >> 2u);
  float4 acc1 = float4(0.0f);
  float4 acc2 = float4(0.0f);
  for (uint i = 0; i < 4u; ++i) {
   acc1[0] += yl[2u*i] * float(q1[i] & 0x000fu);
   acc1[1] += yl[2u*i + 1u] * float(q1[i] & 0x0f00u);
   acc1[2] += yl[2u*i + 8u] * float(q1[i] & 0x00f0u);
   acc1[3] += yl[2u*i + 9u] * float(q1[i] & 0xf000u);
   acc2[0] += yh[2u*i] * float(q2[i] & 0x000fu);
   acc2[1] += yh[2u*i + 1u] * float(q2[i] & 0x0f00u);
   acc2[2] += yh[2u*i + 8u] * float(q2[i] & 0x00f0u);
   acc2[3] += yh[2u*i + 9u] * float(q2[i] & 0xf000u);
  }
  float d = float(*(const device half*)(weight + base));
  float dm = float(*(const device half*)(weight + base + 2u));
  sums[row] += d * (
      (acc1[0] + acc1[1] * (1.0f/256.0f)) * float(sc8[0]) +
      (acc1[2] + acc1[3] * (1.0f/256.0f)) * float(sc8[1]) * (1.0f/16.0f) +
      (acc2[0] + acc2[1] * (1.0f/256.0f)) * float(sc8[4]) +
      (acc2[2] + acc2[3] * (1.0f/256.0f)) * float(sc8[5]) * (1.0f/16.0f)) -
      dm * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
            sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
 }
}
