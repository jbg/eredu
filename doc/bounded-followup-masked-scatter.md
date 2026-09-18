# Masked row scatter

The public prepared image/tool test reaches row scatter during a 17-position
prefill chunk. `WorkspaceTensor::masked_scatter` incorrectly required trailing
broadcast compatibility between `[1,17]` and `[1,17,16]`. The pinned MLX worker
instead requires the mask to match the input prefix exactly, appends singleton
axes, and broadcasts across the unmasked row dimensions. The portable numerical
fixture already used prefix rows.

The canonical geometry now validates that prefix and the source row suffix
without allocating. Source rank may include a leading row count; the trace cannot
inspect the number of true mask entries. Numerical callers must supply enough
source rows. No logical shape fact supplies a native storage grant.

Native producer audit and implementation:

- `ops.cpp::masked_scatter` casts the source, expands/broadcasts the mask and
  source, adds a leading batch dimension, invokes `MaskedScatter`, and squeezes.
- Metal copies the destination, flattens/compacts the mask, allocates U32 prefix
  offsets, performs an exclusive sum scan, then calls the existing masked assign
  kernel. Scan has no extra numerical scratch allocation beyond its output.
- The generic native pointwise, host, and resident graph tables lacked this
  operation. Its generated-kernel path also entered a library constructor that
  explicitly refuses original bounded execution. These are implementation gaps.
- The native implementation retains the same primitive and kernel equations,
  declares their actual populations, and provides the existing fourteen kernel
  instantiations (seven dtypes, contiguous/strided sources) in the embedded
  library. Ordinary generated libraries and admitted embedded libraries select
  the same upstream kernel template. The actual `MaskedScatter` shared-control
  size joins the existing graph catalog; selector names join its existing
  compiler-qualified layout.

For `[1,17,16]` F32 input and `[4,16]` source, a 4096-byte-page allocator
reports 2175 bytes for the destination, 511 for a possible source cast, 543
for one mask compaction, and 2175 for U32 offsets: 5404 bytes. Each value
includes the pinned allocator's possible oversized reuse. A copied flatten is
row-contiguous; only an aliasing strided flatten enters the explicit compaction,
so there is one mask backing across those alternatives. Extra source rows still
count in the original source-cast extent. The host numerical payload is zero;
graph, selector, and dispatch controls remain separately declared.

The lazy constructor has at most ten nodes/twelve edges: source cast, mask
reshape/broadcast, source reshape/broadcast, three leading-axis expansions,
ternary scatter, and squeeze. Eval-only flatten/compaction/offset descriptors
join the temporary population. The maximum intermediate rank is one above the
largest actual input rank (including scalar-mask sources with an explicit row
axis). The scan writes its supplied offsets buffer. Domains exceeding the
unchanged Metal kernel's U32 flat indexing stay unknown instead of wrapping.

Validation completed so far:

- Three portable `row_scatter` tests pass, including exact leading geometry,
  source broadcasting, invalid trailing masks and source widths, zero/scalar
  shapes, dtype preservation, and refusing to infer an unknown native bound.
  Command: `cargo test --offline -p eredu-nn --lib row_scatter` with Rust 1.98.0
  and incremental compilation disabled. Log:
  `/private/tmp/eredu-row-scatter-nn-tests-final.log`.
- The new patch applies cleanly to the current pinned native source. All three
  changed C++ translation units pass syntax checks using their actual build
  commands. All fourteen Metal template instantiations compile with the actual
  selected Metal toolchain. Logs:
  `/private/tmp/eredu-masked-scatter-{cxx,metal}-check.log`.
- The actual Metal/image backend library test binary builds in 4m49s with
  Rust 1.98 and incremental compilation disabled, without `DOCS_RS`:
  `cargo test -p eredu-backend-mlx --lib --features metal,image masked_scatter --no-run`.
  The two physical-bound/geometry tests and numerical test pass (3/3 in 1.37s).
  The numerical fixture runs both CPU and Metal against explicit expected
  values, covering cross-batch prefix masks, strided source rows, integer source
  conversion, row-width broadcasting, and scalar element-mask source, with
  absolute tolerance `1e-5`. Run the binary with
  `masked_scatter --include-ignored --nocapture --test-threads=1`.
  Logs: `/private/tmp/eredu-masked-scatter-native-build.log` and
  `/private/tmp/eredu-masked-scatter-native-tests-device.log`.
- The 2.7 GiB debug executable was fully stripped to a 663 MiB copy for the
  macOS loader. Actual device execution needs native access outside the shell
  sandbox; sandbox execution produced empty fixture arrays before the primitive.
  The unchanged binary passes with device access. The retained runnable artifact
  is `/private/tmp/eredu-masked-scatter-tests`.
- The public image/tool run now passes the former broadcast failure. Its full
  reset/replay validation remains separate and is not inferred from the focused
  primitive tests.

