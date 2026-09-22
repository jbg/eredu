//! Ordinary safe-call metadata for the actual selected CPU producer.
use super::*;
use crate::backend::nn::workspace::OrdinaryNativeControls;
use safemlx::{Array, Stream, ops::OrdinaryRecipeCall};
mod blockwise;
mod common;
mod metal;
mod pooled;
mod sampling;

/// Caller metadata and observed argument-vector storage are disjoint from the
/// native primitive/descriptor/dispatch census and from tensor backing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OrdinaryCallControls {
    pub(crate) metadata_bytes: u64,
    pub(crate) observed: OrdinaryNativeControls,
}
impl OrdinaryCallControls {
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            metadata_bytes: self.metadata_bytes.max(other.metadata_bytes),
            observed: self.observed.union(other.observed),
        }
    }
    pub(crate) fn append(self, other: Self) -> Option<Self> {
        Some(Self {
            metadata_bytes: self.metadata_bytes.checked_add(other.metadata_bytes)?,
            observed: self.observed.append(other.observed)?,
        })
    }
    pub(crate) fn repeat(self, count: usize) -> Option<Self> {
        Some(Self {
            metadata_bytes: self
                .metadata_bytes
                .checked_mul(u64::try_from(count).ok()?)?,
            observed: self.observed.repeat(count)?,
        })
    }
    pub(in crate::backend::nn::workspace) fn call(call: OrdinaryRecipeCall) -> Option<Self> {
        let quoted = call.control_bytes()?;
        let mut observed = OrdinaryNativeControls::default();
        if let Some(source) = quoted.observed_controls() {
            observed.include(source)?;
        }
        Some(Self {
            metadata_bytes: u64::try_from(quoted.metadata_bytes()).ok()?,
            observed,
        })
    }
    pub(in crate::backend::nn::workspace) fn metadata(mut self, bytes: usize) -> Option<Self> {
        self.metadata_bytes = self
            .metadata_bytes
            .checked_add(u64::try_from(bytes).ok()?)?;
        Some(self)
    }
}

impl MlxCpuWorkspaceMechanisms {
    /// Quote the recorded safe calls and actual handle clones without adding a
    /// submission: completion belongs to the enclosing recorded worker.
    pub(crate) fn ordinary_report_call_controls(
        self,
        report: &eredu_nn::workspace::WorkspaceTraceReport,
    ) -> facts::FactResult<Option<OrdinaryCallControls>> {
        let mut controls = OrdinaryCallControls::default();
        for operation in &report.operations {
            let Some(calls) = self.ordinary_call_controls(operation.as_view())? else {
                return Ok(None);
            };
            controls = controls
                .append(calls)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        }
        let Some(clone_bytes) = Array::ordinary_clone_control_bytes() else {
            return Ok(None);
        };
        let Some(clones) = report.tensor_handle_clones else {
            return Ok(None);
        };
        let bytes = clone_bytes
            .checked_mul(clones)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        Ok(Some(
            controls
                .metadata(bytes)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        ))
    }
    /// The native plan must select its genuine CPU worker first. A qualified
    /// numerical primitive does not by itself identify its safe wrapper calls;
    /// composite producers without a caller census remain absent here.
    pub(crate) fn ordinary_call_controls(
        self,
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<OrdinaryCallControls>> {
        if self.plan(operation)?.is_none() {
            return Ok(None);
        }
        selected_call_controls(operation, false)
    }
}
impl super::super::MlxMetalWorkspaceMechanisms {
    /// Same concrete backend callers, selected only after genuine Metal facts.
    /// Device-specific compound workers have separate invocation sources.
    pub(crate) fn ordinary_call_controls(
        self,
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<OrdinaryCallControls>> {
        if self
            .emit(operation, &mut facts::Emitter::count())?
            .is_none()
        {
            return Ok(None);
        }
        selected_call_controls(operation, true)
    }
}
fn selected_call_controls(
    operation: WorkspaceOperationView<'_>,
    metal: bool,
) -> facts::FactResult<Option<OrdinaryCallControls>> {
    let call = OrdinaryCallControls::call;
    let controls = match operation.kind {
        // Numerical selection has already authenticated the transfer layout.
        // Canonical source/destination custody is required independently by
        // the paged companion, whose own query adds publication controls.
        WorkspaceOperationKindView::HostStoreFloating(..) => host_transfer_calls(true),
        WorkspaceOperationKindView::HostLoadStoredFloating(..) => host_transfer_calls(false),
        WorkspaceOperationKindView::Elementwise(name @ ("host_array_i32" | "host_array_u8")) => {
            let output = operation
                .outputs
                .get(0)
                .expect("selected eager integer output");
            let elements = usize::try_from(output.elements()?)
                .map_err(|_| MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            call(if name == "host_array_u8" {
                OrdinaryRecipeCall::SliceU8 {
                    elements,
                    rank: output.shape().len(),
                }
            } else {
                OrdinaryRecipeCall::SliceI32 {
                    elements,
                    rank: output.shape().len(),
                }
            })
            .and_then(|calls| calls.metadata(super::super::host_array::control_bytes()?))
        }
        WorkspaceOperationKindView::GeneratedF32Initialization => {
            let source = basic::generated_f32_plan_view(operation)?;
            selected_call_controls(
                WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::Elementwise("from_f32_slice"),
                    ..operation
                },
                metal,
            )?
            .and_then(|calls| calls.metadata(source.worker_control_bytes::<crate::MlxTensor>()?))
        }
        WorkspaceOperationKindView::Elementwise(name @ ("from_i32_slice" | "from_f32_slice")) => {
            let output = operation
                .outputs
                .get(0)
                .expect("selected typed slice output");
            let elements = usize::try_from(output.elements()?)
                .map_err(|_| MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            let rank = output.shape().len();
            call(if name == "from_i32_slice" {
                OrdinaryRecipeCall::SliceI32 { elements, rank }
            } else {
                OrdinaryRecipeCall::SliceF32 { elements, rank }
            })
            .and_then(|seed| seed.append(call(OrdinaryRecipeCall::Unary)?))
            .and_then(|calls| calls.metadata(super::super::host_array::slice_control_bytes()?))
        }
        WorkspaceOperationKindView::Elementwise("scalar_u8") => call(OrdinaryRecipeCall::ScalarU8),
        WorkspaceOperationKindView::Elementwise("full_f32") => call(OrdinaryRecipeCall::ScalarF32)
            .and_then(|scalar| {
                scalar.append(call(OrdinaryRecipeCall::Full {
                    rank: operation.outputs.get(0)?.shape().len(),
                })?)
            }),
        WorkspaceOperationKindView::Elementwise(name @ ("full_i32" | "full_u32")) => {
            call(if name == "full_i32" {
                OrdinaryRecipeCall::ScalarI32
            } else {
                OrdinaryRecipeCall::ScalarU32
            })
            .and_then(|scalar| {
                scalar.append(call(OrdinaryRecipeCall::Full {
                    rank: operation.outputs.get(0)?.shape().len(),
                })?)
            })
        }
        WorkspaceOperationKindView::Elementwise(
            "zeros_f32" | "zeros_f16" | "zeros_bf16" | "zeros_i32" | "zeros_u32" | "zeros_u8"
            | "zeros_bool",
        ) => call(OrdinaryRecipeCall::Fill {
            rank: operation
                .outputs
                .get(0)
                .expect("selected typed zero output")
                .shape()
                .len(),
        })
        .and_then(|calls| calls.metadata(super::super::zero_fill::control_bytes()?)),
        // Tensor's binary methods borrow both native arrays. The owned
        // Array and borrowed Array transports have the same pointer layout;
        // no RHS conversion or temporary tensor is constructed here.
        WorkspaceOperationKindView::Elementwise("add" | "subtract" | "multiply" | "divide") => {
            call(OrdinaryRecipeCall::Binary)
        }
        WorkspaceOperationKindView::Elementwise("square" | "tanh" | "exp" | "log") => {
            call(OrdinaryRecipeCall::Unary)
        }
        WorkspaceOperationKindView::Elementwise("multiply_scalar" | "maximum_scalar") => {
            call(OrdinaryRecipeCall::ScalarF32)
                .and_then(|scalar| scalar.append(call(OrdinaryRecipeCall::Binary)?))
        }
        WorkspaceOperationKindView::Elementwise("equal_i32" | "maximum_i32") => {
            call(OrdinaryRecipeCall::ScalarI32)
                .and_then(|scalar| scalar.append(call(OrdinaryRecipeCall::Binary)?))
        }
        WorkspaceOperationKindView::CastFloating(_)
        | WorkspaceOperationKindView::Elementwise("capture_cast_f32") => {
            call(OrdinaryRecipeCall::Cast)
        }
        WorkspaceOperationKindView::View("reshape") => call(OrdinaryRecipeCall::Reshape {
            rank: operation
                .outputs
                .get(0)
                .expect("selected reshape output")
                .shape()
                .len(),
        }),
        WorkspaceOperationKindView::View(name)
            if super::super::byte_view::selected(name).is_some() =>
        {
            call(OrdinaryRecipeCall::View)
        }
        WorkspaceOperationKindView::Gather { .. } => common::gather(operation),
        WorkspaceOperationKindView::IndexedRowAdd => call(OrdinaryRecipeCall::ScatterAddAxis),
        WorkspaceOperationKindView::Index { .. } => {
            let input = operation
                .inputs
                .get(0)
                .expect("selected static index input");
            let output = operation
                .outputs
                .get(0)
                .expect("selected static index output");
            call(OrdinaryRecipeCall::BasicIndex {
                input_rank: input.shape().len(),
                output_rank: output.shape().len(),
                operations: input.shape().len(),
                reshape: input.shape().len() != output.shape().len(),
            })
        }
        WorkspaceOperationKindView::View("broadcast") => call(OrdinaryRecipeCall::Broadcast {
            rank: operation
                .outputs
                .get(0)
                .expect("selected broadcast output")
                .shape()
                .len(),
        }),
        WorkspaceOperationKindView::View("expand_dims") => call(OrdinaryRecipeCall::ExpandDims),
        WorkspaceOperationKindView::View("squeeze") => {
            let axes = operation
                .inputs
                .get(0)
                .expect("selected squeeze input")
                .shape()
                .len()
                .checked_sub(
                    operation
                        .outputs
                        .get(0)
                        .expect("selected squeeze output")
                        .shape()
                        .len(),
                )
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            call(OrdinaryRecipeCall::SqueezeAxes { axes })
        }
        WorkspaceOperationKindView::Reduction(
            "sum" | "mean" | "min" | "max" | "argmin" | "softmax",
            _,
            _,
        ) => {
            if metal {
                metal::reduction(operation)
            } else {
                call(OrdinaryRecipeCall::ReduceAxis)
            }
        }
        WorkspaceOperationKindView::Transpose(axes) => {
            call(OrdinaryRecipeCall::Transpose { rank: axes.len() })
        }
        WorkspaceOperationKindView::StaticSlice { starts, .. } => {
            call(OrdinaryRecipeCall::StaticSlice { rank: starts.len() }).and_then(|controls| {
                controls.metadata(crate::tensor::narrow::ordinary_control_bytes(starts.len())?)
            })
        }
        WorkspaceOperationKindView::StaticSliceUpdate { starts, .. } => {
            call(OrdinaryRecipeCall::StaticSliceUpdate { rank: starts.len() })
        }
        WorkspaceOperationKindView::Contiguous => call(OrdinaryRecipeCall::Contiguous),
        WorkspaceOperationKindView::Concatenate => call(OrdinaryRecipeCall::Join {
            inputs: operation.inputs.len(),
            stack: operation
                .inputs
                .get(0)
                .zip(operation.outputs.get(0))
                .is_some_and(|(input, output)| {
                    input.shape().len().checked_add(1) == Some(output.shape().len())
                }),
        }),
        // Tensor::matmul calls the one safe matmul(a,b,stream) wrapper.
        // Native flattening, promotion and batch work stays in its CPU plan.
        WorkspaceOperationKindView::Matmul => call(OrdinaryRecipeCall::Binary),
        WorkspaceOperationKindView::CausalMask(geometry) => {
            causal_mask_calls(geometry.max_past().is_some())
        }
        WorkspaceOperationKindView::HyperCollapse(..)
        | WorkspaceOperationKindView::HyperExpand
        | WorkspaceOperationKindView::HyperHeadCoefficients(_)
        | WorkspaceOperationKindView::HyperHeadSum => {
            super::super::hyper::ordinary_call_controls(operation)?
        }
        WorkspaceOperationKindView::Embedding(..) => common::embedding(operation),
        WorkspaceOperationKindView::VocabularyParallelLookup { .. } => {
            common::parallel_embedding(operation)
        }
        WorkspaceOperationKindView::Attention { .. } => common::attention(operation),
        WorkspaceOperationKindView::BlockwiseAttention { .. } => blockwise::calls(operation),
        WorkspaceOperationKindView::PooledAttention { .. } if metal => pooled::attention(operation),
        WorkspaceOperationKindView::PooledPositions { .. } if metal => pooled::positions(operation),
        WorkspaceOperationKindView::Rotary(..)
        | WorkspaceOperationKindView::RotaryFrequencies(..) => common::rotary(operation),
        WorkspaceOperationKindView::DenseLinear => call(OrdinaryRecipeCall::TransposeDefault)
            .and_then(|transpose| {
                transpose.append(call(if operation.inputs.len() == 3 {
                    OrdinaryRecipeCall::AddMm
                } else {
                    OrdinaryRecipeCall::Binary
                })?)
            }),
        WorkspaceOperationKindView::Projection(format)
            if format.encoding() == eredu_checkpoint::LinearFormat::Dense =>
        {
            dense_projection_calls(operation)
        }
        WorkspaceOperationKindView::Elementwise("silu" | "sigmoid") => {
            if metal {
                metal::activation(operation)
            } else {
                activation_calls(operation)
            }
        }
        WorkspaceOperationKindView::Elementwise(
            "gelu" | "gelu_approximate" | "relu2" | "softplus",
        ) => compound_calls(operation.kind),
        WorkspaceOperationKindView::Grouped { .. } => {
            if metal {
                super::grouped::ordinary_metal_call_controls(operation)
            } else {
                super::grouped::ordinary_call_controls(operation)
            }
        }
        WorkspaceOperationKindView::GroupSelection { .. } if metal => {
            super::router::ordinary_metal_call_controls(operation)?
        }
        WorkspaceOperationKindView::GroupSelection { .. }
        | WorkspaceOperationKindView::JointGroupSelection(_)
            if !metal =>
        {
            super::router::ordinary_call_controls(operation)?
        }
        WorkspaceOperationKindView::GatedProduct(policy) => gated_calls(operation, policy, metal),
        WorkspaceOperationKindView::ConstructedNormalization(spec) if spec.groups.is_some() => {
            grouped_rms_calls(operation, metal)
        }
        WorkspaceOperationKindView::Normalization("rms", None)
        | WorkspaceOperationKindView::ConstructedNormalization(_) => {
            if metal {
                metal::rms(operation)
            } else {
                rms_calls(operation)
            }
        }
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::CreateRandomKey,
        ) => sampling::key(),
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::SelectRandomKey { .. },
        ) => call(OrdinaryRecipeCall::BasicIndex {
            input_rank: 2,
            output_rank: 1,
            operations: 1,
            reshape: true,
        }),
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::SplitRandomKey,
        ) => sampling::split(metal),
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::Categorical,
        ) => sampling::categorical(metal),
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::ValidateToken { .. },
        ) => common::token_validation(operation),
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::TopK { .. }
            | eredu_nn::workspace::WorkspaceSamplingOperation::TopP
            | eredu_nn::workspace::WorkspaceSamplingOperation::MinP,
        ) => sampling::filter(operation),
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::TokenFilter
            | eredu_nn::workspace::WorkspaceSamplingOperation::OptionalTokenFilter,
        ) => sampling::token_mask(operation),
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::Greedy,
        ) => {
            // sample_raw and sample_processed both call the same final-axis
            // argmax wrapper with keepdims=false.
            call(OrdinaryRecipeCall::ReduceAxis)
        }
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::ReadToken,
        ) => call(OrdinaryRecipeCall::U32ScalarRead)
            .and_then(|value| value.metadata(Array::ordinary_clone_control_bytes()?)),
        _ => return Ok(None),
    };
    // Actual Tensor/NeuralBackend caller result conversion plus borrowed
    // inputs and the optional free-function forwarding frame. Rank/index
    // Vecs allocated by an enclosing module remain that module's source.
    let frames = [
        size_of::<(&crate::MlxTensor, &crate::MlxTensor, &Stream)>(),
        size_of::<(Array, &Stream)>(),
        size_of::<(&Array, &Array, &Stream)>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
        size_of::<crate::MlxTensor>(),
        size_of::<f32>(),
        size_of::<&[i32]>(),
    ];
    let frames = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    controls
        .map(|controls| {
            controls
                .metadata(frames)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        })
        .transpose()
}

// core_backend::gated_product retains the optional scalar/clamp calls and the
// selected activation separately, then performs the actual final product.
fn gated_calls(
    operation: WorkspaceOperationView<'_>,
    policy: eredu_nn::GatedProductPolicy,
    metal: bool,
) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let call = OrdinaryCallControls::call;
    let mut controls = OrdinaryCallControls::default();
    if policy.gate_upper_bound().is_some() {
        controls = controls
            .append(call(C::ScalarF32)?)?
            .append(call(C::Binary)?)?;
    }
    if policy.up_absolute_bound().is_some() {
        controls = controls.append(call(C::Clip {
            minimum: true,
            maximum: true,
        })?)?;
    }
    if policy.up_offset() != 0.0 {
        controls = controls
            .append(call(C::ScalarF32)?)?
            .append(call(C::Binary)?)?;
    }
    let activation = match policy.activation() {
        eredu_nn::GatedProductActivation::Silu if policy.sigmoid_multiplier() == 1.0 => {
            let view = WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Elementwise("silu"),
                inputs: operation.inputs.slice(0..1)?,
                outputs: operation.outputs,
            };
            if metal {
                metal::activation(view)?
            } else {
                activation_calls(view)?
            }
        }
        eredu_nn::GatedProductActivation::Silu => {
            // This branch calls the ordinary native unary sigmoid on the scaled
            // gate, then multiplies its probability by the original gate.
            call(C::ScalarF32)?
                .append(call(C::Binary)?.repeat(2)?)?
                .append(call(C::Unary)?)?
        }
        eredu_nn::GatedProductActivation::GeluApproximate => {
            compound_calls(WorkspaceOperationKindView::Elementwise("gelu_approximate"))?
        }
        _ => return None,
    };
    controls.append(activation)?.append(call(C::Binary)?)
}

// The selected CPU activation plan authenticates the same layers.rs fallback:
// optional input widening, the CPU-declined pointwise probe, actual safe calls,
// then the unconditional output dtype call. Native implicit casts/broadcasts
// inside those calls do not add C wrappers.
pub(super) fn activation_calls(
    operation: WorkspaceOperationView<'_>,
) -> Option<OrdinaryCallControls> {
    let call = OrdinaryCallControls::call;
    let dtype = operation.inputs.get(0)?.representation()?.dtype();
    let mut result = OrdinaryCallControls::default();
    if matches!(
        dtype,
        WorkspaceFloatingType::Float16 | WorkspaceFloatingType::Bfloat16
    ) {
        result = result.append(call(OrdinaryRecipeCall::Cast)?)?;
    }
    match operation.kind {
        WorkspaceOperationKindView::Elementwise("silu") => {
            result = result
                .append(call(OrdinaryRecipeCall::Unary)?.repeat(2)?)?
                .append(call(OrdinaryRecipeCall::ScalarF32)?)?
                .append(call(OrdinaryRecipeCall::Binary)?.repeat(2)?)?;
        }
        WorkspaceOperationKindView::Elementwise("sigmoid") => {
            result = result.append(call(OrdinaryRecipeCall::Unary)?)?;
        }
        _ => return None,
    }
    let frames = [
        crate::backend::nn::arithmetic::cpu_pointwise_probe_control_bytes()?,
        size_of::<(Array, &Stream)>(),
        size_of::<safemlx::Dtype>(),
        size_of::<Array>() * 2,
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<Result<Option<Array>, safemlx::error::Exception>>(),
    ];
    result.append(call(OrdinaryRecipeCall::Cast)?)?.metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
}

pub(super) fn compound_calls(kind: WorkspaceOperationKindView<'_>) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let call = OrdinaryCallControls::call;
    let counts: &[(C, usize)] = match kind {
        WorkspaceOperationKindView::Elementwise("gelu") => &[
            (C::ScalarI32, 1),
            (C::ScalarF32, 2),
            (C::Unary, 1),
            (C::Binary, 4),
        ],
        WorkspaceOperationKindView::Elementwise("gelu_approximate") => &[
            (C::ScalarF32, 4),
            (C::ScalarI32, 1),
            (C::Unary, 2),
            (C::Binary, 7),
        ],
        WorkspaceOperationKindView::Elementwise("relu2") => {
            &[(C::ScalarF32, 1), (C::Binary, 1), (C::Unary, 1)]
        }
        WorkspaceOperationKindView::Elementwise("softplus") => &[
            (C::Cast, 2),
            (C::ScalarF32, 3),
            (C::Binary, 3),
            (C::Unary, 2),
            (C::Select, 1),
        ],
        _ => return None,
    };
    counts
        .iter()
        .try_fold(OrdinaryCallControls::default(), |total, &(kind, count)| {
            total.append(call(kind)?.repeat(count)?)
        })?
        .metadata(
            size_of::<(&Array, &Stream)>()
                .checked_add(size_of::<(&crate::MlxTensor, f32, &Stream)>())?
                .checked_add(size_of::<Result<Array, safemlx::error::Exception>>())?,
        )
}

fn dense_projection_calls(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let call = OrdinaryCallControls::call;
    let mut result =
        call(OrdinaryRecipeCall::TransposeDefault)?.append(call(OrdinaryRecipeCall::Binary)?)?;
    if operation.inputs.len() == 3 {
        result = result.append(call(OrdinaryRecipeCall::Binary)?)?;
    }
    result.metadata(crate::backend::nn::matrix::row_projection_probe_control_bytes()?)
}

fn rms_calls(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let call = OrdinaryCallControls::call;
    let (learned, offset, final_cast) = match operation.kind {
        WorkspaceOperationKindView::Normalization("rms", None) => (
            operation.inputs.len() == 2,
            false,
            operation.inputs.len() == 2,
        ),
        WorkspaceOperationKindView::ConstructedNormalization(spec) if spec.groups.is_none() => {
            match spec.scale {
                eredu_nn::NormalizationScale::Unit => (false, false, false),
                eredu_nn::NormalizationScale::Learned(_) => (true, false, false),
                eredu_nn::NormalizationScale::LearnedOffset { .. } => (true, true, false),
            }
        }
        _ => return None,
    };
    let mut result = OrdinaryCallControls::default();
    if offset {
        result = result
            .append(call(C::ScalarF32)?)?
            .append(call(C::Binary)?)?;
    }
    let mut probe = !learned;
    if learned {
        let input = operation.inputs.get(0)?.representation()?.dtype();
        let gain = if offset {
            WorkspaceFloatingType::Float32
        } else {
            operation.inputs.get(1)?.representation()?.dtype()
        };
        if input == gain && input != WorkspaceFloatingType::Float32 {
            result = result.append(call(C::Cast)?)?;
            probe = true;
        }
        result = result.append(call(C::RmsNorm)?)?;
        if final_cast {
            result = result.append(call(C::Cast)?)?;
        }
    } else {
        result = result.append(weightless_rms_fallback_calls()?)?;
    }
    if probe {
        // The same f32_weightless_rms predicate declines a CPU stream before
        // constructing a shader invocation. Its unused widening is above.
        result = result.metadata(weightless_rms_probe_control_bytes()?)?;
    }
    result.metadata(
        size_of::<(&Array, &Array, f32, &Stream)>()
            .checked_add(size_of::<Result<Array, safemlx::error::Exception>>())?,
    )
}

/// The shared weightless arithmetic fallback: mean's internal scalar belongs
/// to the native reduction source, while epsilon is an actual safe scalar call.
fn weightless_rms_fallback_calls() -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let call = OrdinaryCallControls::call;
    call(C::Unary)?
        .repeat(2)?
        .append(call(C::ReduceAxis)?)?
        .append(call(C::ScalarF32)?)?
        .append(call(C::Binary)?.repeat(2)?)?
        .append(call(C::Cast)?)
}

fn weightless_rms_probe_control_bytes() -> Option<usize> {
    let frames = [
        Stream::device_type_control_bytes()?,
        size_of::<(&Array, f32, &Stream)>(),
        size_of::<Result<Option<Array>, safemlx::error::Exception>>(),
        size_of::<safemlx::Dtype>(),
        size_of::<usize>() * 2,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

/// MlxRmsNorm's grouped branch widens before grouping, so the positive Metal
/// source always reaches the same F32 weightless row worker. The shape Vec is
/// caller-owned; neither reshape wrapper owns that retained destination.
fn grouped_rms_calls(
    operation: WorkspaceOperationView<'_>,
    metal: bool,
) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let WorkspaceOperationKindView::ConstructedNormalization(spec) = operation.kind else {
        return None;
    };
    spec.groups?;
    let input = operation.inputs.get(0)?;
    let rank = input.shape().len();
    let grouped_rank = rank.checked_add(1)?;
    let call = OrdinaryCallControls::call;
    let mut calls = call(C::Cast)?.append(call(C::Reshape { rank: grouped_rank })?)?;
    let normalized = if metal && input.elements().ok()? != 0 {
        metal::row_rms(grouped_rank)?
    } else {
        weightless_rms_fallback_calls()?.metadata(weightless_rms_probe_control_bytes()?)?
    };
    calls = calls
        .append(normalized)?
        .append(call(C::Reshape { rank })?)?;
    if !matches!(spec.scale, eredu_nn::NormalizationScale::Unit) {
        calls = calls.append(call(C::Cast)?)?;
        if matches!(
            spec.scale,
            eredu_nn::NormalizationScale::LearnedOffset { .. }
        ) {
            calls = calls
                .append(call(C::ScalarF32)?)?
                .append(call(C::Binary)?)?;
        }
        calls = calls.append(call(C::Binary)?)?;
    }
    calls = calls.append(call(C::Cast)?)?;
    let frames = [
        size_of::<(
            &mut crate::backend::nn::shared::MlxRmsNorm,
            &crate::MlxTensor,
            &Stream,
        )>(),
        size_of::<(&Array, f32, &Stream)>(),
        size_of::<[Array; 5]>(),
        size_of::<Vec<i32>>(),
        std::alloc::Layout::array::<i32>(grouped_rank).ok()?.size(),
        size_of::<[i32; 2]>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<(usize, i32, Option<f32>, safemlx::Dtype)>(),
        size_of::<Result<crate::MlxTensor, Error>>(),
        size_of::<Result<Array, Error>>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<Result<Option<Array>, safemlx::error::Exception>>(),
    ];
    calls.metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
}

/// Actual ordinary grouped-linear assertion shares the scalar token validator.
pub(super) fn grouped_index_validation_calls() -> Option<OrdinaryCallControls> {
    common::validation_prefix(false, false)
}

fn causal_mask_calls(window: bool) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let call = OrdinaryCallControls::call;
    let mut result = call(C::ArangeI32)?
        .repeat(2)?
        .append(call(C::Reshape { rank: 2 })?.repeat(2)?)?
        .append(call(C::Binary)?)?;
    if window {
        result = result
            .append(call(C::ScalarI32)?)?
            .append(call(C::Binary)?.repeat(3)?)?;
    }
    result.metadata(crate::backend::nn::tensor::ordinary_causal_mask_control_bytes()?)
}

/// F32 compound producers retain their actual fixed Metal invocation source.
pub(super) fn metal_f32_pointwise_calls(rank: usize) -> Option<OrdinaryCallControls> {
    metal::f32_pointwise(rank)
}
pub(super) fn metal_f32_sum_calls(rank: usize) -> Option<OrdinaryCallControls> {
    metal::f32_sum(rank)
}

/// The actual ordinary copy methods each return one Event that the owning
/// transfer destination synchronizes before canonical publication.
fn host_transfer_calls(store: bool) -> Option<OrdinaryCallControls> {
    let wrapper = if store {
        safemlx::HostTransferBuffer::ordinary_detach_wrapper_control_bytes()?
    } else {
        safemlx::ImmutableHostTransferBuffer::ordinary_copy_wrapper_control_bytes()?
    };
    OrdinaryCallControls::default()
        .metadata(wrapper.checked_add(safemlx::Event::ordinary_synchronize_control_bytes()?)?)
}
