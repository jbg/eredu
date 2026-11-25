for (uint row = 0; row < ROWS_PER_SIMD; ++row) {
 float total = simd_sum(sums[row]);
 if (lane == 0u) products[local_group * ROWS_PER_SIMD + row] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Each adjacent SIMD pair produces the two projections for the same output rows.
if (lane == 0u && projection == 0u) {
#pragma clang fp contract(off)
 for (uint row = 0; row < ROWS_PER_SIMD; ++row) {
  uint out_col = first_out + row;
  if (out_col >= OUT_DIM) continue;
  float gate = products[local_group * ROWS_PER_SIMD + row];
  float up = products[(local_group + 1u) * ROWS_PER_SIMD + row];
  float e = out_col < (OUT_DIM / 4u) * 4u ? vector_exp_f32(-gate) : stable_exp_f32(-gate);
  float activated = metal::precise::divide(gate, 1.0f + e);
  out[out_col] = activated * up;
 }
}
