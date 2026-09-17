uint KD = query_shape[3];
uint VD = value_shape[3];
uint elem = thread_position_in_grid.x;
uint vd = elem % VD;
uint group = elem / VD;
uint state_base = group * KD * VD;
uint vec_base = group * KD;
uint value_base = group * VD;
float kv_mem = 0.0f;
for (uint kd = 0; kd < KD; ++kd) {
  uint state_idx = state_base + kd * VD + vd;
  float gate = metal::exp(g[group * KD + kd]);
  kv_mem += float(state[state_idx]) * gate * float(key[vec_base + kd]);
}
float delta = (float(value[value_base + vd]) - kv_mem) * float(beta[group]);
float acc = 0.0f;
for (uint kd = 0; kd < KD; ++kd) {
  uint state_idx = state_base + kd * VD + vd;
  float gate = metal::exp(g[group * KD + kd]);
  float updated = float(state[state_idx]) * gate + float(key[vec_base + kd]) * delta;
  state_out[state_idx] = updated;
  acc += updated * float(query[vec_base + kd]);
}
out[value_base + vd] = acc;
