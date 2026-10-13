# Vendored native sources

`mlx-v0.32.3.tar.gz` contains MLX v0.32.3 at commit
`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`. Its upstream URL and SHA-256 are:

```text
https://github.com/ml-explore/mlx/archive/refs/tags/v0.32.3.tar.gz
4129039ddcb36cb860982b616c975a5b9b8e6d8fa97efa86335b4427c29c4b4b
```

CMake verifies the digest before extracting the archive and applies patches
from `src/mlx-c/patches` to the build-tree copy.

The patch stack retains persistent completion diagnostics, accepted-work
frontiers, and physical resource ownership through partial submission failure.
CPU scheduling and GPU receipts use these contracts together; a polling error
alone cannot release submitted resources. Fence dependencies retain MLX's
per-update sequence values. CUDA uses native events for GPU-only fences and
tracked signal work for CPU/GPU dependencies. Global CUDA encoders have process
lifetime, and compiled wrappers retain upstream weak cache identities.

CPU cascade sums, BF16 row reductions, softmax exponential arithmetic, and equal-key
partition ordering retain Eredu's numerical contracts. Metal group-size-16 affine
quantization and mixed-storage GEMM use the pinned operation signatures.

Common MLX build dependencies are vendored from the upstream release inputs:

| File | Upstream URL | SHA-256 |
| --- | --- | --- |
| `json-v3.11.3.tar.xz` | `https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz` | `d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d` |
| `fmt-12.1.0.tar.gz` | `https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz` | `ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea` |
| `metal-cpp-26.zip` | `https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip` | `4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4` |

Each archive retains its upstream license and attribution files.

`mlx-cpu-softmax-reduction.patch` adapts the Float32 exponential
from SLEEF `xexpf` at revision `5a1d179df9cf652951b59010a2d2075372d67f68`.
The patch retains the upstream copyright and complete Boost Software License.
Its explicit fused operations make BF16 softmax and sigmoid rounding independent of the
host's scalar/vector exponential implementations.

The same pinned SLEEF exponential polynomial and Boost license are retained in
`eredu-backend-mlx/src/backend/nn/exp_f32.metal` for the native FP32 activation
mechanism. CPU sigmoid uses the same exponential through `exp_f32.h` supplied by
the softmax/reduction patch.

Validation for this pin covers the 52 carried native patch tests in CPU-only and
Metal-enabled builds, foreign-TLS and cross-thread compile-cache cleanup, affine
and MXFP4 gather-QMM, repeated GPU waits, Metal stream handoff, mixed-storage
split-K equivalence, reduction/softmax rounding, attention, and multi-buffer
timing. Mixed-storage numerical checks use `MLX_ENABLE_TF32=0`; allocator-accounting
checks run with `--test-threads=1`. Native CUDA compilation and execution remain
unvalidated on the macOS validation host.
