# Kernel-name producer controls

The pinned MLX v0.32.0 `concatenate` helper copied its remaining argument pack
at every recursive call. The longest selected call has 18 arguments, giving
171 parameter slots across those frames. The shared helper now keeps one
by-value entry pack and appends each value by borrow with a fold. Keeping the
entry values preserves the original snapshot behavior when the destination
string is also supplied as an argument.

The change applies to the same ordinary and admitted selector worker. The
portable helper lives in the patched native dependency's
`mlx/backend/common/kernel_name.h`; it does not depend on a device or runtime.
The exact pinned source/archive identity is retained in the
[evidence record](validation/kernel-name-fold-2026-09-18.json).

The actual selector layout query, compiled without linking MLX or Metal, reports:

| Bytes per lookup | Recursive helper | Fold helper |
| --- | ---: | ---: |
| Name allocations | 2,424 | 2,424 |
| Fixed controls | 6,223 | 2,455 |
| Total | 8,647 | 4,879 |

The 3,768-byte reduction follows from the qualified 24-byte string and 8-byte
pointer layouts: 171 parameter slots become 18 entry values plus one numeric
conversion temporary; 18 accumulator references become three entry/helper
references. The name bound remains 184 bytes, with seven function constants.
Zero, one, and one million lookup queries and overflow refusal were checked.
Kernel-attempt counts, heap-growth bounds, cache ownership and concurrency
assumptions are unchanged.

The permanent host tests cover scalar and character formatting, Unicode,
destination aliases, and the complete attention and quantized argument packs.
All four cases and five assertions pass. The retained
[standalone oracle](validation/kernel-name-parity.cpp) compares the new helper
with the exact archived upstream helper: **200,006 comparisons pass**, including
numeric extremes and 100,000 generated name/alias pairs. Five optimized runs of
400,000 calls each took 0.0545–0.0587 seconds upstream and 0.0544–0.0595 seconds
with the fold. These timings establish no material isolated regression; they
do not establish whole-planner speedup.

To repeat the standalone oracle, point `MLX_PATCHED_SOURCE` at the source tree
produced by the registered CMake patch stack:

```sh
clang++ -std=c++20 -O2 -I "$MLX_PATCHED_SOURCE" \
  doc/validation/kernel-name-parity.cpp -o /tmp/kernel-name-parity
/tmp/kernel-name-parity
```

Validation used Apple clang 21.0.0, C++20 and MacOSX26.5.sdk. Syntax checks also
passed for the actual selector-controls and quantized Metal translation units.
The standalone formatter and layout executables link only libc++ and libSystem;
these checks perform no device work.

The historical released Required48 refusal of 685,353,706,402 bytes came from an
older debug executable and older graph/lookup populations. It is neither a
current quote nor a measurement of successful allocation usage. Later
[released text and image tool validation](bounded-followup-released-tools.md)
passes the concrete sensor request under the unchanged 64 GiB limit, with one
complete tool call for both Required and Auto. Those complete runs include other
producer and lifetime corrections; their success and elapsed times cannot be
attributed to this helper alone. The separate generic-action Required request
reaches `MaxTokens` without a funding failure but does not produce a tool call.

The helper measurements remain isolated formatter/layout evidence. The
[native guide](bounded-followup-native.md) records current CPU/distributed
acceptance separately; this guide makes no claim that pending reruns pass.
