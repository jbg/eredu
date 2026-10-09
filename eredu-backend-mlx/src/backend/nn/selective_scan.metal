uint group = thread_position_in_grid.y;
uint head = group % HEADS;
// One timestep/decay calculation per tile instead of per state element.
threadgroup float parameters[2];
if (thread_position_in_threadgroup.x == 0) {
    float raw_dt = time_step[group] + bias[head];
    // Match MLX LogAddExp's stable softplus and NaN/Inf behavior. log1p
    // comes from MLX's standard custom-kernel utility header.
    float maxval = metal::max(raw_dt, 0.0f);
    float minval = metal::min(raw_dt, 0.0f);
    float softplus = (minval == -INFINITY || maxval == INFINITY)
        ? maxval : maxval + log1p(metal::exp(minval - maxval));
    if (metal::isnan(raw_dt)) softplus = raw_dt;
    float dt = metal::isnan(softplus) ? softplus
        : (softplus > floor ? softplus : floor);
    float transition = -metal::precise::exp(transition_log[head]);
    parameters[0] = dt;
    parameters[1] = metal::precise::exp(dt * transition);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint column = thread_position_in_grid.x;
if (column >= WIDTH * STATE_WIDTH) return;
uint i = group * WIDTH * STATE_WIDTH + column;
uint row = i / STATE_WIDTH;
uint component = i % STATE_WIDTH;
float dt = parameters[0];
float decay = parameters[1];
float discretized = dt * input_vector[group * STATE_WIDTH + component];
float update = values[row] * discretized;
float decayed = initial[i] * decay;
float next = decayed + update;
state[i] = next;
weighted[i] = next * output_vector[group * STATE_WIDTH + component];
if (component == 0) direct[row] = values[row] * skip[head];
