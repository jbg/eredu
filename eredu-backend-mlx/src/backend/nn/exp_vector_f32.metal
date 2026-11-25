
#ifndef EREDU_LOCAL_EXP_CONTRACTION
#pragma clang fp contract(off)
#endif
// Port of the FP32 vector exponential used by the CPU mechanism.
// See THIRD-PARTY-NOTICES for Arm Optimized Routines and ggml attribution.
inline float vector_exp_f32(float x) {
#ifdef EREDU_LOCAL_EXP_CONTRACTION
#pragma clang fp contract(off)
#endif
    const float biased = fma(x, 0x1.715476p0f, 12582912.f);
    const float n = biased - 12582912.f;
    if (!isfinite(x) || abs(n) > 126.f) return stable_exp_f32(x);
    const float b = fma(-n, as_type<float>(0x35bfbe8eu),
                        fma(-n, as_type<float>(0x3f317200u), x));
    const float scale = as_type<float>((as_type<uint>(biased) << 23) + 0x3f800000u);
    const float b2 = b * b;
    const float lo = fma(b, as_type<float>(0x3e2aaf33u), as_type<float>(0x3efffedbu));
    const float hi = fma(b, as_type<float>(0x3c072010u), as_type<float>(0x3d2b9f17u));
    const float adjustment = fma(fma(hi, b2, lo), b2, as_type<float>(0x3f7ffff6u) * b);
    return fma(scale, adjustment, scale);
}
