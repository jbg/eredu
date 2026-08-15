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
| Nemotron-H 4B | Superseded by the follow-up below. The initial corrected run narrowly missed the generic BF16 relative-L2 threshold; the five-prompt recurrent-cache validation now passes. | 263.5 prompt tok/s; 6.88 decode tok/s | 10.67 GiB |

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
- [Nemotron-H corrected BF16 job](https://huggingface.co/jobs/jbg/6a80a85e1f5885ae605bb24b)
- [Corrected CUDA image publication](https://github.com/jbg/safemlx/actions/runs/31895441504)
- [Nemotron-H FP32 recurrent-scan image publication](https://github.com/jbg/safemlx/actions/runs/31906292669)
- [Nemotron-H full-prefill long-context comparison](https://huggingface.co/jobs/jbg/6a80dbef1f5885ae605bba90)
- [Nemotron-H tokenwise-oracle diagnostic](https://huggingface.co/jobs/jbg/6a80dea41f5885ae605bbaeb)
- [Nemotron-H five-prompt tokenwise matrix](https://huggingface.co/jobs/jbg/6a80e0d5c97db76cbdf32789)
- [Nemotron-H policy-aware image publication](https://github.com/jbg/safemlx/actions/runs/31910824221)

Complete comparison and 128+128 artifacts for Qwen3-VL and LFM2.5 are under
`hf://buckets/jbg/jobs-artifacts/safemlx-cuda-pilot/20260815-8054df2/`.
The final GPT-OSS and Nemotron-H partial artifacts are under
`hf://buckets/jbg/safemlx-pilot-results/safemlx-cuda-pilot/20260815-d066ba8-r3/`.
The Qwen3 partial summary is under
`hf://buckets/jbg/jobs-artifacts/safemlx-cuda-pilot/20260815-d066ba8/`.
The corrected Nemotron-H comparison and 128+128 performance artifacts are under
`hf://buckets/jbg/safemlx-pilot-results/safemlx-cuda-pilot/20260815-a3abee9-nemotron-bf16-fix/nemotron_h_dense/`.
The final five-prompt Nemotron-H matrix is under
`hf://buckets/jbg/safemlx-pilot-results/safemlx-cuda-pilot/20260815-41b6c99-nemotron-tokenwise-matrix/`.

## Nemotron-H follow-up

Commit `85917d9` aligns the SafeMLX recurrent scan with the pinned Transformers
slow path: it computes the transition parameter and prefill recurrence in
FP32, applies `time_step_min`, preserves the FP32 recurrent cache, and retains
the reference's BF16 decode output behavior. The corrected binary was published
as:

```text
ghcr.io/jbg/safemlx-validation@sha256:f58d8e35bb2d4ed7bcfd64b2abad9fa301890bad5766faafecba82a4702120e0
```

The initial long-context rerun exposed a separate oracle problem. At 769
repeated tokens, Transformers 5.14.1's naive parallel full-prefill path produced
1.06097 relative L2, 0.65771 cosine similarity, and a different argmax from
SafeMLX. Running the same pinned Transformers model one token at a time through
its recurrent cache instead agreed with SafeMLX: prefill relative L2 was
0.00861, cosine similarity was 0.999969, top-5 overlap was 5/5, and the argmax
matched. This establishes the recurrent-cache path as the appropriate oracle
for this hybrid recurrent architecture.

Commit `41b6c99` therefore scopes Nemotron-H to the tokenwise Transformers
prefill oracle and a family-specific BF16 profile: relative L2 at most 0.025,
cosine similarity at least 0.999, top-5 overlap at least 4, and required argmax
agreement when the reference top-1 margin exceeds 0.25. All five deterministic
prompts pass across 11, 25, 41, 193, and 769 tokens. The matrix-wide worst
relative L2 is 0.02109, minimum cosine similarity is 0.999805, minimum top-5
overlap is 4/5, and every required argmax agrees.

The final image containing both the corrected probe and the scoped validation
policy was published and manifest-verified as:

```text
ghcr.io/jbg/safemlx-validation@sha256:58c485fed65c8fe3de22ae0c175b44a843cb908bb52da7dd1cde3c66710196cb
```

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
   is pinned. The correction restores BF16 activation semantics in SafeMLX's Mamba path:
   initialize convolution padding in the projected input dtype, perform gated
   normalization intermediates in FP32, and cast the normalized scan output
   back to the block input dtype before `out_proj`. It also constructs all
   unquantized parameter targets from the checkpoint's configured weight dtype;
   the pinned checkpoint stores all 263 tensors as BF16. The earlier FP32
   Transformers control nearly eliminated prefill drift
   (0.00280 relative L2), while BF16-versus-FP32 Transformers itself differs by
   about 0.0224 relative L2. The additional deterministic matrix established
   that Transformers' parallel prefill is not a stable long-context oracle for
   this checkpoint. Use its tokenwise recurrent-cache path and the scoped
   Nemotron-H BF16 profile recorded above.

The pilot intentionally records failed comparisons and blocked phases as
results. It does not classify a case as passing merely because it loads or
produces matching greedy tokens.
