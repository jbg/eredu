uint KD = query_shape[3];
uint VD = value_shape[3];
uint L = query_shape[1];
uint H = query_shape[2];
uint elem = thread_position_in_grid.x;
uint vd = elem % VD;
uint group = elem / VD;
uint h = group % H;
uint b = group / H;
uint state_base = group * KD * VD;
for (uint t = 0; t < L; ++t) {
  uint gh_idx = (b * L + t) * H + h;
  uint vec_base = gh_idx * KD;
  uint value_base = gh_idx * VD;
  float kv_mem = 0.0f;
  for (uint kd = 0; kd < KD; ++kd) {
    uint state_idx = state_base + kd * VD + vd;
    float prev = (t == 0) ? float(state[state_idx]) : float(state_out[state_idx]);
    float gate = metal::exp(g[gh_idx * KD + kd]);
    kv_mem += prev * gate * float(key[vec_base + kd]);
  }
  float delta = (float(value[value_base + vd]) - kv_mem) * float(beta[gh_idx]);
  float acc = 0.0f;
  for (uint kd = 0; kd < KD; ++kd) {
    uint state_idx = state_base + kd * VD + vd;
    float prev = (t == 0) ? float(state[state_idx]) : float(state_out[state_idx]);
    float gate = metal::exp(g[gh_idx * KD + kd]);
    float updated = prev * gate + float(key[vec_base + kd]) * delta;
    state_out[state_idx] = updated;
    acc += updated * float(query[vec_base + kd]);
  }
  out[value_base + vd] = acc;
}
