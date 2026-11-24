uint row = STAGED ? threadgroup_position_in_grid.y : thread_position_in_grid.x;
uint lane = STAGED ? thread_position_in_threadgroup.x : 0u;
uint lanes = STAGED ? 32u : 1u;
size_t offset = size_t(row) * WIDTH;
threadgroup float squares[STAGED ? WIDTH : 1];

if (STAGED) {
    for (uint i = lane; i < WIDTH; i += lanes) {
        float x = input[offset + i];
        squares[i] = x * x;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

float mean = -1.0f;
if (PARALLEL) mean = certified_rms_mean(squares, WIDTH, lane);
float inverse = 0.0f;
if (lane == 0) {
    if (mean < 0.0f) {
        // The compensated sum consumes the rounded squares in input order.
        float high = 0.0f;
        float low = 0.0f;
        for (uint i = 0; i < WIDTH; ++i) {
            float y;
            if (STAGED) {
                y = squares[i];
            } else {
                float x = input[offset + i];
                y = x * x;
            }
            float s = high + y;
            float v = s - high;
            float e = (high - (s - v)) + (y - v) + low;
            high = s + e;
            low = e - (high - s);
        }
        mean = metal::precise::divide(high, float(WIDTH));
        mean += metal::precise::divide(fma(-mean, float(WIDTH), high) + low, float(WIDTH));
    }
    inverse = metal::precise::divide(1.0f, metal::precise::sqrt(mean + epsilon[0]));
}
if (STAGED) inverse = simd_shuffle(inverse, 0);
for (uint i = lane; i < WIDTH; i += lanes) {
    out[offset + i] = (input[offset + i] * inverse) * gain[i];
}
