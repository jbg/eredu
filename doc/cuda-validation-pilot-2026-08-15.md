# CUDA validation pilot: 2026-08-15

This pilot exercised five pinned checkpoint families on Hugging Face Jobs
using one NVIDIA L4 per case. The immutable validation image was built from
commit `d066ba8` and published as:

```text
ghcr.io/jbg/safemlx-validation@sha256:3a2cd1dff93aa3dac986b1b21392fe4518f0cc6d674aa8d57a934b71097cdff0
```

The image publication and manifest verification completed in
[GitHub Actions run 31850822038](https://github.com/jbg/safemlx/actions/runs/31850822038).

## Results

| Case | Correctness result | 128+128 synchronized wall performance | Peak MLX active memory |
| --- | --- | ---: | ---: |
| Qwen3 0.6B | Blocked before inference: structural admission now accepts the checkpoint's redundant tied `lm_head.weight`, but the separate layerwise loader still rejects it as unexpected. | Not measured | Not measured |
| Qwen3-VL 2B | Failed cached decode: prefill passed; decode matched 8/16 argmaxes. This run covered the text decoder only, not image or video inputs. | 1,563.4 prompt tok/s; 14.3 decode tok/s | 6.19 GiB |
| GPT-OSS 20B MXFP4 | Blocked in SafeMLX's MXFP4 CUDA JIT: NVRTC could not resolve `cute/numeric/numeric_types.hpp`. The image build verifies that CUTLASS, CuTe, CCCL, and CUDA headers are present, so the remaining defect is runtime include-path discovery. | Not measured | Not measured |
| LFM2.5 1.2B | Failed numeric thresholds, while preserving all 17 tested argmaxes and top-5 overlap. Minimum cosine was 0.998829 against 0.999; maximum relative L2 was 0.04871 against 0.02. | 6,628.4 prompt tok/s; 99.2 decode tok/s | 2.23 GiB |
| Nemotron-H 4B | Failed only relative L2 after backporting Transformers' missing MLP cache placeholder. All 17 argmaxes matched. Localization found that SafeMLX promotes the BF16 residual stream to FP32 in its first Mamba block; an FP32 Transformers control reduced prefill relative L2 from 0.02411 to 0.00280 and maximum decode relative L2 from 0.03381 to 0.02263. | Standard 128+128 run not reached. Short 11+16 SafeMLX correctness probe: 62.5 prompt tok/s; 6.8 decode tok/s. | 10.60 GiB on the short SafeMLX probe |

Performance rates use synchronized wall time. The standard smoke profile is a
single warmup followed by one measured 128-token prefill and 128-token decode.
Device-event rates remain in the raw JSON but are not used in this summary.

## Evidence

- [Qwen3 job](https://huggingface.co/jobs/jbg/6a7fb304c97db76cbdf31abe)
- [Qwen3-VL job](https://huggingface.co/jobs/jbg/6a7f9afdc97db76cbdf31952)
- [GPT-OSS job](https://huggingface.co/jobs/jbg/6a7fb5acc97db76cbdf31ae1)
- [LFM2.5 job](https://huggingface.co/jobs/jbg/6a7f9afdc97db76cbdf31950)
- [Nemotron-H job](https://huggingface.co/jobs/jbg/6a7fb5ac1f5885ae605b9d01)
- [Nemotron-H patched reference job](https://huggingface.co/jobs/jbg/6a8059ce1f5885ae605bab8f)
- [Nemotron-H tokenwise BF16 reference job](https://huggingface.co/jobs/jbg/6a805af2c97db76cbdf321a4)
- [Nemotron-H FP32 reference job](https://huggingface.co/jobs/jbg/6a805ce7c97db76cbdf321b1)
- [Nemotron-H tokenwise FP32 reference job](https://huggingface.co/jobs/jbg/6a805da8c97db76cbdf321b9)

Complete comparison and 128+128 artifacts for Qwen3-VL and LFM2.5 are under
`hf://buckets/jbg/jobs-artifacts/safemlx-cuda-pilot/20260815-8054df2/`.
The final GPT-OSS and Nemotron-H partial artifacts are under
`hf://buckets/jbg/safemlx-pilot-results/safemlx-cuda-pilot/20260815-d066ba8-r3/`.
The Qwen3 partial summary is under
`hf://buckets/jbg/jobs-artifacts/safemlx-cuda-pilot/20260815-d066ba8/`.

## Product findings

1. Unify structural admission and layerwise consumption of redundant tied
   output heads. Qwen3 currently passes one exact-catalog check and fails the
   next one on the same tensor.
2. Make MLX CUDA JIT include discovery relocatable for statically linked
   executables, and add an image smoke that compiles an MXFP4 kernel rather
   than checking header presence only.
3. Investigate Qwen3-VL cache-state evolution: prefill agrees with
   Transformers, while teacher-forced decode rapidly diverges.
4. Investigate LFM2.5's small systematic numeric drift before changing the
   tolerance profile; token decisions all agree, but the current BF16 numeric
   contract does not.
5. Keep the scoped Nemotron-H cache-registry backport while Transformers 5.14.1
   is pinned. This revision restores BF16 activation semantics in SafeMLX's Mamba path:
   initialize convolution padding in the projected input dtype, perform gated
   normalization intermediates in FP32, and cast the normalized scan output
   back to the block input dtype before `out_proj`. SafeMLX currently leaves
   the scan output in FP32, which promotes the residual stream after the first
   Mamba block. The FP32 Transformers control nearly eliminates prefill drift
   (0.00280 relative L2), while BF16-versus-FP32 Transformers itself differs by
   about 0.0224 relative L2. The BF16 CUDA oracle still needs to be rerun on
   the corrected image before considering any tolerance change.

The pilot intentionally records failed comparisons and blocked phases as
results. It does not classify a case as passing merely because it loads or
produces matching greedy tokens.
