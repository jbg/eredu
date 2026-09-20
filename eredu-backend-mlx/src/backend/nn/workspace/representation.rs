//! Exact scalar propagation for selected native workers; all other cases forget it.
//! This refines no allocation by itself and never infers source dtype from a
//! checkpoint format, family, model state dtype, or a logical Float32 layout.
use super::*;
use WorkspaceFloatingType as F;
use WorkspaceOperationKindView as K;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{GatedProductActivation, NormalizationScale};

#[path = "representation/affine.rs"]
mod affine;
#[path = "representation/selector.rs"]
mod selector;
#[path = "representation/layer_norm.rs"]
mod layer_norm;

/// Actual F32 operands of the selected affine router; no storage or worker grant.
pub(super) fn affine_selector_source(operation: WorkspaceOperationView<'_>) -> Option<()> {
    selector::affine_source(operation)
}

fn dtype(operation: WorkspaceOperationView<'_>, index: usize) -> Option<F> {
    operation
        .inputs
        .get(index)?
        .representation()
        .map(WorkspaceRepresentation::dtype)
}
pub(super) fn promote(a: F, b: F) -> F {
    // Native mlx::result_type calls promote_types: equal reduced precision
    // stays reduced; mixed F16/BF16 and either operand F32 produce F32.
    if a == b { a } else { F::Float32 }
}
fn all_floating(operation: WorkspaceOperationView<'_>) -> Option<F> {
    let mut result = None;
    for input in operation.inputs.iter() {
        let value = input.representation()?.dtype();
        result = Some(result.map_or(value, |prior| promote(prior, value)));
    }
    result
}

fn exact_f32_operands(operation: WorkspaceOperationView<'_>) -> Option<F> {
    let mut floating = false;
    for input in operation.inputs.iter() {
        if input.dtype() == WorkspaceDtype::Float32 {
            if input.representation()?.dtype() != F::Float32 { return None; }
            floating = true;
        }
    }
    floating.then_some(F::Float32)
}

fn dense_grouped(bank: &WorkspaceGroupedBank) -> bool {
    match bank {
        WorkspaceGroupedBank::Linear(spec) =>
            spec.projection().format().encoding() == LinearFormat::Dense,
        WorkspaceGroupedBank::GatedProduct(spec) => match spec.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } =>
                gate_up.format().encoding() == LinearFormat::Dense
                    && down.format().encoding() == LinearFormat::Dense,
            _ => false,
        },
        WorkspaceGroupedBank::Relu2(spec) =>
            spec.up().format().encoding() == LinearFormat::Dense
                && spec.down().format().encoding() == LinearFormat::Dense,
    }
}

pub(super) fn output(
    operation: WorkspaceOperationView<'_>,
    index: usize,
) -> Option<WorkspaceRepresentation> {
    if operation.outputs.get(index)?.dtype() != WorkspaceDtype::Float32 {
        return None;
    }
    // The native one-input concatenate returns that exact array before any
    // cast or copy. Keep all retained physical facts, including real strides;
    // an unknown/noncontiguous alias must not acquire canonical layout facts.
    if matches!(operation.kind, K::Concatenate) && operation.inputs.len() == 1 {
        let input = operation.inputs.get(0)?;
        let output = operation.outputs.get(index)?;
        return (input.shape() == output.shape() && input.dtype() == output.dtype())
            .then(|| input.representation()).flatten();
    }
    let value = match operation.kind {
        K::Embedding(format, _) | K::VocabularyParallelLookup { format, .. }
            if format.encoding() == LinearFormat::Dense =>
        {
            dtype(operation, 1)?
        }
        // Selected packed rows are explicitly restored to F32 by the same
        // native dequantizing embedding worker before the ownership mask.
        K::Embedding(format, _) | K::VocabularyParallelLookup { format, .. }
            if matches!(
                format.encoding(),
                LinearFormat::Affine(_) | LinearFormat::MxFp4
            ) =>
        {
            F::Float32
        }
        K::Projection(format) if format.encoding() == LinearFormat::Dense => {
            all_floating(operation)?
        }
        K::Projection(format) if matches!(format.encoding(), LinearFormat::Affine(_)) => {
            affine::projection(operation, format)?
        }
        K::DenseLinear | K::Matmul | K::Convolution { .. } => all_floating(operation)?,
        // Native pad_axes casts the fill value to a.dtype; edge copies retain
        // that same scalar representation. This is also the first-history
        // padding in the shared causal convolution before its dense product.
        K::Pad(_) => dtype(operation, 0)?,

        // Dense grouped matmuls, their bias/activation arithmetic and final
        // weighted reduction preserve F32 when every real floating operand is
        // F32. Route IDs are integral operands, not scalar-type evidence.
        // Other formats keep their own native representation requirement.
        K::Grouped { bank, .. } if dense_grouped(bank) => exact_f32_operands(operation)?,
        // The explicit-index MXFP4 GatherQMM worker preserves its F32 input;
        // gathered biases, bounded activation and weighted TP bias/reduction
        // likewise remain F32. Check the same exact packed parameter roles and
        // geometry as its native source, plus each real floating operand. The
        // packed integer bank and scale bytes are not floating dtype evidence.
        K::Grouped { .. } if resident_recipe::mxfp4_sources(operation).is_some() =>
            exact_f32_operands(operation)?,
        // Affine QMM promotes actual floating input, scales and biases. The
        // F32 invariant also includes activation, route weighting and TP bias;
        // absent physical facts remain absent despite logical F32 geometry.
        K::Grouped { .. } if resident_recipe::affine_sources(operation).is_some() =>
            exact_f32_operands(operation)?,
        // The same dense selector projection, optional normalization, score
        // transform, gathered weights and precision controls all retain F32
        // when the actual input/parameter values are F32. Supplied route IDs
        // are integral. This also covers original/effective output pairs.
        K::GroupSelection { spec, .. } if spec.format().encoding() == LinearFormat::Dense =>
            exact_f32_operands(operation)?,
        K::GroupSelection { .. } if affine_selector_source(operation).is_some() => F::Float32,



        // The selected relative-attention worker scales queries with a real
        // F32 scalar, gathers profile bias, applies its causal softmax, then
        // multiplies values. Its four actual F32 sources yield F32 throughout.
        // Joint selection similarly keeps its dense projection, log-sigmoid,
        // softmax and scaled coefficient slices F32. The integer ID output was
        // excluded above; neither arm invents missing input scalar evidence.
        K::RelativeAttention { .. } | K::JointGroupSelection(_)
            if operation.inputs.len() == 4
                && (0..4).all(|input| dtype(operation, input) == Some(F::Float32)) =>
        {
            F::Float32
        }

        // The existing scalar mask, threshold, sort/gather and scatter workers
        // preserve a known F32 score input, including their input-alias branch.
        // This supplies no reduced-precision or stride claim and no worker grant.
        K::Sampling(WorkspaceSamplingOperation::TokenFilter
            | WorkspaceSamplingOperation::OptionalTokenFilter
            | WorkspaceSamplingOperation::TopK { .. }
            | WorkspaceSamplingOperation::TopP | WorkspaceSamplingOperation::MinP)
            if operation.inputs.len()==1 && operation.outputs.len()==1
                && operation.inputs.get(0)?.shape()==operation.outputs.get(0)?.shape()
                && dtype(operation,0)==Some(F::Float32) => F::Float32,

        // These exact native transforms preserve scalar representation. Layout
        // evidence is deliberately discarded, even for an alias or reshape.
        K::View(name) if super::byte_view::selected(name).is_some() => {
            super::byte_view::selected(name)?.floating()?
        }
        K::View("reshape" | "squeeze" | "expand_dims" | "broadcast")
        | K::Transpose(_)
        | K::Index { .. }
        | K::StaticSlice { .. }
        | K::Gather { .. }
        | K::GatherPooledMask
        | K::Contiguous
        | K::DeepCopy => dtype(operation, 0)?,
        // Native fast RoPE and explicit input-product rounding return the
        // selected input scalar. Native YaRN first multiplies a real F32 array;
        // the nontraditional frequency-scaled worker also uses F32 sin/cos.
        K::Rotary(spec, Some(_)) => match (spec.arithmetic, spec.algorithm) {
            (eredu_nn::RotaryArithmetic::InputProducts, _) => dtype(operation, 0)?,
            (_, eredu_nn::RotaryAlgorithm::Yarn { .. }) => F::Float32,
            (_, eredu_nn::RotaryAlgorithm::Llama3 { .. }) if !spec.traditional => F::Float32,
            _ => dtype(operation, 0)?,
        },
        // The existing supplied-table adapter's negative half uses try_from_f32.
        K::Rotary(_, None) => F::Float32,
        K::TensorRotary(..) => dtype(operation, 0)?,
        // Explicit frequencies are cast to F32 internally, but both native
        // RoPE and its fallback return the input scalar type.
        K::RotaryFrequencies(..) => dtype(operation, 0)?,
        K::MultiAxisRotary(_) | K::PreparedMultiAxisRotary(_) => {
            return super::positional::rotary_representation(operation, index);
        }
        // Vendored SDPA selects result_type(q,k,v) before either native/fallback
        // execution. Masks/sinks are validated against that selected type.
        K::Attention {
            softcap: false,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
            ..
        } => promote(
            promote(dtype(operation, 0)?, dtype(operation, 1)?),
            dtype(operation, 2)?,
        ),
        // The direct explicit worker casts probabilities to query precision,
        // then its final matmul promotes them with the value precision. Key,
        // mask and sink types affect scores but not that final product type.
        K::Attention { window: None, arithmetic, .. }
            if arithmetic == eredu_nn::AttentionArithmetic::Fused
                || operation.inputs.get(0)?.shape().get(2).zip(operation.inputs.get(1)?.shape().get(2))
                    .is_some_and(|(&q,&k)| i64::from(q)*i64::from(k)
                        <= i64::from(crate::backend::nn::attention::INPUT_SCORE_ROW_BUDGET)) =>
        {
            promote(dtype(operation, 0)?, dtype(operation, 2)?)
        }
        // Explicit score policies have multiple rounded/bounded product paths;
        // their F32 Q/K/V case is invariant across every selected path.
        K::Attention { .. } if (0..3).all(|input| dtype(operation, input) == Some(F::Float32)) => {
            F::Float32
        }
        // The shared paged worker casts queries/keys/values to F32 for both
        // recurrence passes. Its output cast uses the original query type,
        // retained by the source-derived policy before Begin overwrites it.
        K::BlockwiseAttention { policy, stage } => match stage {
            eredu_nn::workspace::WorkspaceBlockwiseStage::Begin => {
                if index == 0 {
                    F::Float32
                } else {
                    dtype(operation, 1)?
                }
            }
            eredu_nn::workspace::WorkspaceBlockwiseStage::Accumulate { .. } => F::Float32,
            eredu_nn::workspace::WorkspaceBlockwiseStage::Finish => policy.output_type?,
        },
        K::HyperCollapse(..) => {
            if index == 0 {
                dtype(operation, 0)?
            } else {
                F::Float32
            }
        }
        K::HyperExpand | K::HyperHeadSum => dtype(operation, 0)?,
        K::HyperHeadCoefficients(_) => F::Float32,
        K::Concatenate => all_floating(operation)?,
        K::InitializeFloating(dtype)
        | K::CastFloating(dtype)
        | K::HostTransferFloating(dtype)
        | K::HostLoadStoredFloating(_, dtype) => dtype,
        K::SliceUpdate { .. } | K::StaticSliceUpdate { .. } | K::IndexedRowAdd
        | K::IndexedElementSelect | K::IndexedElementUpdate => dtype(operation, 0)?,
        // The pinned masked row-scatter casts its complete source to the
        // destination dtype before broadcasting and retains that dtype in
        // the MaskedScatter result. Source precision does not promote it.
        K::Elementwise("masked_scatter") => {
            super::masked_scatter::geometry(operation).ok()??;
            dtype(operation, 0)?
        }
        K::Elementwise("capture_cast_f32" | "cast_f32") => F::Float32,
        K::Elementwise(_) if super::zero_fill::dtype(operation).is_some() => super::zero_fill::dtype(operation)?.1?,
        K::Elementwise("scalar_f32") if basic::is_scalar_f32(operation) => F::Float32,
        K::Initialize | K::GeneratedF32Initialization => {
            if operation.inputs.is_empty() {
                F::Float32
            } else {
                dtype(operation, 0)?
            }
        }
        // Array::try_from_f32 is a real F32 array, not a weak scalar. Native
        // binary promotion therefore widens reduced-precision activations.
        K::Elementwise("multiply_scalar" | "maximum_scalar") => F::Float32,
        // Native promotion with an actual signed integer scalar preserves a
        // floating input's physical precision; integral outputs have no F fact.
        K::Elementwise("maximum_i32") => dtype(operation, 0)?,
        // The existing exact and tanh GELU workers construct real F32 arrays.
        // Their binary operations therefore widen each supported floating
        // input, including the exact worker's final division by F32 two.
        // Require the input's retained scalar fact before describing the result.
        K::Elementwise("gelu" | "gelu_approximate") => {
            dtype(operation, 0)?;
            F::Float32
        }
        K::Elementwise("square" | "tanh" | "exp" | "log" | "silu" | "sigmoid" | "softplus") => {
            dtype(operation, 0)?
        }
        K::Elementwise("subtract" | "multiply" | "divide" | "clip") => all_floating(operation)?,
        K::Elementwise("add") => {
            let value = all_floating(operation)?;
            // The portable default add_residual currently shares this same
            // operation and omits its explicit FP32 flag. Its F32 result is
            // proved in either case; reduced-precision additions stay unknown.
            if value != F::Float32 {
                return None;
            }
            value
        }
        K::Elementwise("where") => promote(dtype(operation, 1)?, dtype(operation, 2)?),
        K::ConstructedNormalization(spec) => {
            if spec.groups.is_some() {
                dtype(operation, 0)?
            } else {
                match &spec.scale {
                    NormalizationScale::Unit => dtype(operation, 0)?,
                    NormalizationScale::Learned(_) => all_floating(operation)?,
                    NormalizationScale::LearnedOffset { .. } => F::Float32,
                }
            }
        }
        // Both shared RMS adapters cast their result back to the input dtype.
        K::Normalization("rms" | "silu_gated_group_rms_norm", _) => dtype(operation, 0)?,
        K::Normalization("gated_group_rms_norm", _) => {
            promote(dtype(operation, 0)?, dtype(operation, 2)?)
        }
        // L2 has an explicit F32 epsilon array followed by division/multiply.
        K::LayerNorm { .. } => layer_norm::scalar(operation)?,
        K::Normalization("l2", _) => F::Float32,
        K::GatedProduct(policy) if policy.activation() == GatedProductActivation::GeluApproximate => {
            dtype(operation,0)?; dtype(operation,1)?;
            F::Float32
        }
        K::GatedProduct(policy)
            if policy.activation() == GatedProductActivation::Silu
                && policy.sigmoid_multiplier() == 1.0
                && policy.gate_upper_bound().is_none()
                && policy.up_absolute_bound().is_none()
                && policy.up_offset() == 0.0 =>
        {
            all_floating(operation)?
        }
        // Native recurrent_step/metal_scan accumulates state in F32; the
        // sequence result is explicitly cast to query.dtype at the boundary.
        K::GatedDeltaScan | K::SelectiveStateSpaceScan(..) => match index {
            0 => F::Float32,
            1 => dtype(operation, 0)?,
            _ => return None,
        },
        _ => return None,
    };
    // Both Host-copy workers and the exact eager rank-zero F32 constructor
    // produce row-contiguous storage. Multi-input concatenate allocates its
    // canonical output before copying each input (including zero-sized output).
    // Its singleton alias was handled above without manufacturing layout facts.
    let contiguous = matches!(
        operation.kind,
        K::HostTransferFloating(_) | K::HostLoadStoredFloating(..)
    ) || matches!(operation.kind, K::Concatenate) && operation.inputs.len() > 1
        || matches!(operation.kind, K::Elementwise("scalar_f32"))
        && basic::is_scalar_f32(operation) || super::zero_fill::dtype(operation).is_some();
    Some(WorkspaceRepresentation::new(value, contiguous))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[path = "representation/blockwise_tests.rs"]
mod blockwise_tests;

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[path = "representation/concatenate_tests.rs"]
mod concatenate_tests;
