// Reciprocal of the four interleaved BF16 softmax sums.
inline float bf16_softmax_inverse(float s0, float s1, float s2, float s3) {
#pragma clang fp reassociate(off)
    return metal::precise::divide(1.0f, (s0 + s2) + (s1 + s3));
}
