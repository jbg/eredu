//! Disjoint managed host payloads for the audited Metal realizations. Native
//! tensor constructors, CPU compaction and GPU temporaries share the Metal
//! allocator and are already counted in the tensor domain. Rust-owned numerical
//! vectors are separate. The explicit attention input producer also names its
//! fixed controls; unrelated command, shape and handle metadata are excluded.

use super::*;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{AttentionArithmetic, RotaryAlgorithm, RotaryArithmetic};

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    mechanisms: &MlxMetalWorkspaceMechanisms,
) -> Result<Option<WorkspaceHostBound>, Error> {
    facts::ordinary_host_with(
        |sink| emit(operation.as_view(), mechanisms, sink),
        |error| ordinary_error(operation, error),
    )
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    mechanisms: &MlxMetalWorkspaceMechanisms,
    sink: &mut facts::HostEmitter<'_>,
) -> facts::FactResult<Option<WorkspaceHostFacts>> {
    if let Some(child) = parallel_lookup::embedding(operation)? {
        return emit(child, mechanisms, sink);
    }
    use WorkspaceOperationKindView as K;
    use facts::mul;
    if matches!(operation.kind, K::GroupSelection { .. }) {
        return routing::emit_selector_host(operation, mechanisms.allocation, sink);
    }
    if matches!(operation.kind, K::Grouped { .. }) {
        let Some((_, bytes)) = grouped::emit(
            operation,
            mechanisms.allocation,
            &mut facts::Emitter::count(),
        )?
        else {
            return Ok(None);
        };
        return sink.finish(bytes, format_args!("selected MLX grouped execution: GPU sorting/gathers/projections/reductions use shared native storage; unsigned E8M0 scale decoding borrows the admitted static 256-element lookup table; shape, handle and command metadata excluded")).map(Some);
    }
    // Share the exact descriptor validation and supported execution envelope.
    // A tensor fact is necessary here, but does not itself supply a host fact.
    if mechanisms
        .emit(operation, &mut facts::Emitter::count())?
        .is_none()
    {
        return Ok(None);
    }
    if matches!(operation.kind, K::Sampling(_)) {
        return super::sampling::emit_host(operation, sink);
    }
    let (bytes, derivation) = match &operation.kind {
        K::IndexedElementSelect | K::IndexedElementUpdate => (
            0,
            "sparse element workers borrow retained native I32 indices; the original producer separately pays host lowering and ingress",
        ),
        K::IndexedRowAdd => (
            0,
            "ScatterAxis Sum reads retained native integer/update sources directly; host index construction belongs to its independently funded ingress",
        ),
        K::ValueCompletion | K::ValueRetention => (
            0,
            "borrowed value retention/completion has no host numerical payload; native traversal controls are separately priced",
        ),
        K::SelectiveStateSpaceScan(..) => (
            0,
            "selective scan keeps its numerical payloads in shared native storage; its exact Array output vector and caller controls are separately included in the shared original scan receipt",
        ),
        K::CandidateExtraction { .. } => (
            0,
            "Metal extraction reads K U32 IDs and K F32 scores directly from shared native buffers into the separately priced original candidate host slots; no extra numerical staging; CPU stable_sort is not this selected mechanism",
        ),
        K::ProjectionPrepare(_) => (
            0,
            "observed FP8 preparation borrows the static Ue8m0 table; native scale roots and lookup buffers remain held through callback/finish and are counted in the tensor domain",
        ),
        K::ProjectionFinish(_) => (
            0,
            "observed FP8 finish consumes the actual prepared compact roots and retained decoded weight scales; projection/copies/bias have no disjoint host numerical vector",
        ),
        K::BlockFp8ActivationDecode => (
            0,
            "fixed compact E4M3 Uint8 to F32 ConvertFP8 uses the Metal unary kernel directly; no host conversion vector",
        ),
        K::ParameterPlaceholder => (
            0,
            "the stack scalar is copied directly into the already-priced shared native allocation; lazy fill/shape descriptors have no disjoint host numerical payload",
        ),
        K::GeneratedF32Initialization => {
            let plan = basic::generated_f32_plan_view(operation)?;
            (
                plan.host_buffer_bytes(),
                "shared fixed F32 initializer owns exactly one preallocated Rust vector, fills without growth, and keeps it through host-to-native construction; the architecture scalar expression is allocation-free; native source and copy storage are separately priced",
            )
        }
        K::HostTransferFloating(_) | K::HostLoadStoredFloating(..) | K::HostStoreFloating(..) => {
            if basic::host_transfer_dtype(operation).is_none() {
                return Ok(None);
            }
            (
                0,
                "Host payload and source registration belong to the exact accepted source bank; the native transfer workers create no additional host numerical staging vector",
            )
        }
        K::Elementwise(_) if super::zero_fill::dtype(operation).is_some() =>
            (0,"typed eager zero is copied directly; optional native Broadcast and Full have no Rust numerical payload vector"),
        K::Elementwise("scalar_u8") => {
            if !basic::is_scalar_u8(operation) { return Ok(None); }
            (0, "one borrowed U8 scalar is copied directly; no Rust numerical payload vector")
        }
        K::Elementwise("scalar_f32") => {
            if !basic::is_scalar_f32(operation) { return Ok(None); }
            (0, "one borrowed F32 scalar is copied directly; no Rust numerical payload vector")
        }
        K::Elementwise("prepared_token_input") => {
            if !basic::is_prepared_token_input(operation) { return Ok(None); }
            (0, "prepared token payload belongs to the existing input producer; the model borrows its completed integer matrix")
        }
        K::Initialize | K::InitializeFloating(_) | K::CastFloating(_) => (
            0,
            "borrowed host input is copied directly into shared native storage; scalar fills and copies use the same allocator; the caller owns and separately prices the input payload",
        ),
        K::View(name) if super::byte_view::selected(name).is_some() => {
            if super::byte_view::inspect(operation).is_none(){return Ok(None);}
            (0,"byte reinterpretation uses shared storage or the same GPU general-copy worker; no numerical host vector")
        }
        K::View("broadcast" | "transpose" | "squeeze" | "expand_dims" | "reshape")
        | K::Transpose(_)
        | K::Index { .. }
        | K::StaticSlice { .. }
        | K::SliceUpdate { .. }
        | K::StaticSliceUpdate { .. }
        | K::Contiguous
        | K::Concatenate => (
            0,
            "static indices and shape transformations have no host data vectors; copies, casts and concatenation write shared native storage",
        ),
        K::DeepCopy => (
            0,
            "row-major host data is copied directly between shared native buffers; strided data first compacts on the default CPU stream into the same shared allocator; no intermediate host vector",
        ),
        K::Elementwise("capture_cast_f32" | "cast_f32")
            if basic::is_selected_capture_f32_cast_view(operation) =>
        {
            (
                0,
                "closed nonempty static capture selection covers native F16/BF16 conversion with copy_gpu between shared native buffers and a future F32 no-op; no half scalar conversion or numerical host staging vector",
            )
        }
        K::Elementwise("cast_u32") if basic::is_isolated_text_u32_cast_view(operation) => (
            0,
            "the closed isolated text-matrix I32-to-U32 AsType uses copy_gpu directly between shared native buffers; shape/stride/kernel descriptors contain no numerical host staging",
        ),
        K::Elementwise(
            "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "logical_or"
            | "greater"
            | "less"
            | "greater_equal"
            | "less_equal"
            | "logical_and"
            | "logical_not"
            | "is_finite"
            | "is_nan"
            | "is_positive_infinity"
            | "is_negative_infinity"
            | "bool_to_u32"
            | "square"
            | "tanh"
            | "exp"
            | "log"
            | "multiply_scalar"
            | "maximum_scalar"
            | "maximum_i32"
            | "equal_i32"
            | "clip"
            | "where"
            | "silu"
            | "sigmoid"
            | "softplus"
            | "gelu"
            | "gelu_approximate"
            | "elu",
        )
        | K::GatedProduct(_) => (
            0,
            "pointwise/custom kernels, promotion copies and scalar operands use shared native storage; no disjoint host payload staging",
        ),
        K::Reduction("sum" | "sum_all" | "mean" | "min" | "max" | "softmax", ..)
        | K::Normalization(
            "rms"
            | "l2"
            | "layer_norm"
            | "constructed_rms"
            | "gated_group_rms_norm"
            | "silu_gated_group_rms_norm",
            _,
        )
        | K::ConstructedNormalization(_) => (
            0,
            "normalization/reduction intermediates, partial reductions, learned-scale casts and scalars use shared native storage; local kernel reductions require no host payload",
        ),
        K::Matmul | K::DenseLinear => (
            0,
            "dense Metal products use shared native casts, layout copies, output and split-K buffers; no CPU operand pack or staging vector",
        ),
        K::Projection(format) if format.encoding() == LinearFormat::Dense => (
            0,
            "dense projection and bias arithmetic use shared native tensor buffers; parameter residency is separate",
        ),
        K::Gather { .. } => (
            0,
            "index validation, safe-index selection and row gathering operate on native arrays; completion reads a scalar validation result directly without a host copy vector",
        ),
        K::Embedding(format, _) if format.encoding() == LinearFormat::Dense => (
            0,
            "dense embedding uses native validation and safe-index arrays with direct strided table reads; scalar validation maps the already-counted native result",
        ),
        K::Projection(format) | K::Embedding(format, _)
            if matches!(
                format.encoding(),
                LinearFormat::Affine(_) | LinearFormat::MxFp4 | LinearFormat::GgufIQuant { .. }
            ) =>
        {
            (
                0,
                "selected packed Metal projection/embedding consumes native packed storage directly; compaction, quantized partials and selected-row dequantization use shared native arrays; GGML codebooks are compiled kernel constants, not per-invocation host payload vectors",
            )
        }
        K::Projection(format) if matches!(format.encoding(), LinearFormat::E4M3BlockFp8(_)) => (
            0,
            "FP8 activation quantization uses native buffers; Ue8m0 decoding borrows the admitted static table, with native lookup and gather buffers separately counted",
        ),
        K::BlockwiseAttention { .. } => {
            if super::attention::blockwise::descriptor::decode(operation)?.is_none() {
                return Ok(None);
            }
            (
                0,
                "paged blockwise recurrence uses source-owned arrays, native integer mask coordinates and bounded per-page native buffers; no disjoint host numerical payload",
            )
        }
        K::CausalMask(_) | K::PoolingMask(_) => (
            0,
            "causal/window and pooling masks use native coordinate aranges, scalar quotient and comparisons; no host mask vector or lengths input",
        ),
        K::PooledAttention { .. }
        | K::IndexedAttention { .. }
        | K::PooledPositions { .. }
        | K::GatherPooledMask => (
            0,
            "pooled/indexed attention uses native gathers, masks, two-operand batched contractions, reductions and Metal partition/sort buffers; einsum shape/path vectors contain metadata only, with no disjoint host numerical payload",
        ),
        K::RelativeAttention { .. } => (
            0,
            "relative attention builds integer coordinates, gathers profiles and computes scaling/products in shared native buffers; no disjoint host numerical vectors",
        ),
        K::MultiAxisRotary(spec) => (
            positional::rotary_host_bytes_fixed(*spec)?,
            "one exact-sized Rust frequency vector at a time, copied into owned native storage before the next iteration; maximum axis half-width, or the complete round-robin half-width; native copies are separately priced",
        ),
        K::PreparedMultiAxisRotary(_) => (
            0,
            "immutable prepared host frequencies are a separately retained source input; this operation constructs no Rust frequency vector; native frequency source and stream-copy storage remain unchanged and priced in the tensor bound; this metadata fact grants no original source-account credit",
        ),
        K::MaskedOutputProjection { .. } => (
            0,
            "centroid partition, token and weight gathers, selected products, per-position minimum and vocabulary scatter use shared native buffers; no disjoint host payload",
        ),
        K::HyperCollapse(..) | K::HyperExpand | K::HyperHeadCoefficients(_) | K::HyperHeadSum => (
            0,
            "multi-stream residual mixing uses shared native casts, RMS reductions, dense products, coefficients, Sinkhorn passes and weighted sums; einsum paths and shape vectors contain metadata only; no disjoint host numerical payload",
        ),
        K::JointGroupSelection(_) => (
            0,
            "joint selection uses native projection, sigmoid, Metal partition, direct gather, concatenation, log-sigmoid/softmax and scalar scaling; shared coefficient views retain one allocation; no disjoint host numerical payload",
        ),
        K::Convolution { .. } => (
            0,
            "casts, compaction, unfolding, Winograd transforms and fill scalars use shared native storage; no disjoint host payload staging",
        ),
        K::Pad(_) => (
            0,
            "constant fill/copy and edge slice-update padding use shared native arrays; padding widths and slices are shape metadata, with no host payload vector",
        ),
        #[cfg(not(feature = "cuda"))]
        K::GatedDeltaScan => (
            0,
            "gated-delta chunk slices, casts, recurrent states and concatenation use shared native arrays; the output handle list contains no copied numerical payload",
        ),
        K::TensorRotary(..) | K::RotaryFrequencies(..) => (
            0,
            "native rotary consumes existing frequency arrays and native scalar offsets; no disjoint host frequency vector",
        ),
        K::Rotary(spec, _) => {
            // Include construction even on first use. Range-map collections
            // have exact cardinality; proportional construction reserves that
            // cardinality before pushing. YaRN constructs two vectors for its
            // native denominator and one more for InputProducts. Conservatively
            // retain their sum, even though the constructors run sequentially.
            let vectors = match spec.algorithm {
                RotaryAlgorithm::Yarn { .. } => {
                    2 + u64::from(spec.arithmetic == RotaryArithmetic::InputProducts)
                }
                RotaryAlgorithm::Proportional { .. } => 1,
                RotaryAlgorithm::Default | RotaryAlgorithm::Linear { .. }
                    if spec.arithmetic == RotaryArithmetic::InputProducts =>
                {
                    1
                }
                _ => 0,
            };
            (
                mul(mul(spec.dimensions as u64 / 2, 4)?, vectors)?,
                "first-use rotary frequency construction includes every Rust F32 vector: two native YaRN vectors plus its optional InputProducts vector, one proportional vector, or one default/linear InputProducts vector; native copies are separately counted",
            )
        }
        K::Attention {
            arithmetic,
            softcap,
            ..
        } => {
            // Only the large-row InputScores path constructs a host mask. It
            // processes one query and at most INPUT_SCORE_KEY_TILE keys per
            // completed accumulator call; the local Vec<bool> is dropped before
            // the next call. Sliding query tiling can reduce the key extent, so
            // using the original extent remains conservative for every cut.
            let keys = operation.inputs.get(1).unwrap().shape()[2];
            let bytes = if *arithmetic == AttentionArithmetic::InputScores
                && keys > crate::backend::nn::attention::INPUT_SCORE_ROW_BUDGET
            {
                crate::backend::nn::attention::INPUT_SCORE_KEY_TILE as u64
            } else {
                0
            };
            let input_controls = if *arithmetic != AttentionArithmetic::Fused || *softcap {
                let Some(controls) = super::super::matrix::batched_input_control_bytes() else {
                    return Ok(None);
                };
                u64::try_from(controls)?
            } else {
                0
            };
            (
                facts::add(bytes, input_controls)?,
                "attention group IDs are written directly into their single native I32 source; explicit attention prices one fixed synchronous ID-producer frame, reused across sequential products/tiles; InputScores key rows above the shared threshold additionally allocate one host boolean mask for one query and at most the shared key-tile width, released before the next completed block; native ID/mask storage is counted separately",
            )
        }
        // In particular, new equation kinds or names must not silently inherit
        // a zero host bound when their tensor implementation is added.
        _ => return Ok(None),
    };
    sink.finish(bytes, format_args!("selected MLX Metal managed host payload: {derivation}; other native shape/command metadata, JIT programs, allocator bookkeeping and unrelated process memory are outside this domain")).map(Some)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod f32_initialization_tests;
