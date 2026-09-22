//! Safe caller census of the selected dense lookup and native attention workers.
use super::*;

/// Tensor::take_axis uses the shared signed/unsigned domain validator before
/// its ordinary Take. Validation result storage belongs to the parent batch.
pub(super) fn gather(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let indices = operation.inputs.get(1)?;
    let call = OrdinaryCallControls::call;
    let validation = if indices.elements().ok()? == 0 {
        OrdinaryCallControls::default().metadata(Array::ordinary_clone_control_bytes()?)?
    } else {
        call(OrdinaryRecipeCall::ScalarI32)?
            .repeat(2)?
            .append(call(OrdinaryRecipeCall::Binary)?.repeat(3)?)?
            .append(call(OrdinaryRecipeCall::Unary)?)?
            .append(call(OrdinaryRecipeCall::All)?)?
            .append(call(OrdinaryRecipeCall::ZerosLike)?)?
            .append(call(OrdinaryRecipeCall::Select)?)?
    };
    validation
        .append(call(OrdinaryRecipeCall::Take)?)?
        .metadata(
            size_of::<(&Array, i32, &Stream)>()
                .checked_add(size_of::<Array>().checked_mul(6)?)?
                .checked_add(size_of::<Result<Array, safemlx::error::Exception>>())?
                .checked_add(size_of::<(bool, i32, safemlx::Dtype)>())?,
        )
}

// The actual token normalization helper is shared by embedding and sampled or
// forced-token handoff. Its lazy predicate is read by the enclosing submission.
pub(super) fn validation_prefix(empty: bool, sentinel: bool) -> Option<OrdinaryCallControls> {
    let call = OrdinaryCallControls::call;
    let mut result = call(OrdinaryRecipeCall::Cast)?;
    if !empty {
        result = result
            .append(call(OrdinaryRecipeCall::Cast)?)?
            .append(call(OrdinaryRecipeCall::ScalarI32)?.repeat(2)?)?
            .append(call(OrdinaryRecipeCall::Binary)?.repeat(3)?)?
            .append(call(OrdinaryRecipeCall::Unary)?)?
            .append(call(OrdinaryRecipeCall::All)?)?;
        if sentinel {
            result = result
                .append(call(OrdinaryRecipeCall::ScalarI32)?)?
                .append(call(OrdinaryRecipeCall::Binary)?.repeat(2)?)?;
        }
    }
    let frames = [
        size_of::<(&Array, i32, Option<i32>, &Stream)>(),
        size_of::<Option<i32>>(),
        size_of::<safemlx::Dtype>(),
        size_of::<Array>() * 5,
        size_of::<Result<Array, safemlx::error::Exception>>(),
    ];
    result.metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
}

pub(super) fn token_validation(
    operation: WorkspaceOperationView<'_>,
) -> Option<OrdinaryCallControls> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::ValidateToken { .. }
        )
    ) {
        return None;
    }
    validation_prefix(operation.inputs.get(0)?.elements().ok()? == 0, false)
}

pub(super) fn embedding(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let WorkspaceOperationKindView::Embedding(format, policy) = operation.kind else {
        return None;
    };
    if format.encoding() != eredu_checkpoint::LinearFormat::Dense {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let output = operation.outputs.get(0)?;
    let sentinel = matches!(policy, eredu_nn::EmbeddingLookupPolicy::ZeroSentinel(_));
    let call = OrdinaryCallControls::call;
    let mut result = validation_prefix(input.elements().ok()? == 0, sentinel)?;
    // lookup builds safe indices, then dense PhysicalEmbedding uses the direct
    // borrowed Array index worker, which delegates to take_axis without an Rc.
    result = result
        .append(call(OrdinaryRecipeCall::ScalarI32)?.repeat(2)?)?
        .append(call(OrdinaryRecipeCall::Binary)?.repeat(3)?)?
        .append(call(OrdinaryRecipeCall::Fill {
            rank: input.shape().len(),
        })?)?
        .append(call(OrdinaryRecipeCall::Select)?)?
        .append(call(OrdinaryRecipeCall::Take)?)?;
    if sentinel {
        result = result
            .append(call(OrdinaryRecipeCall::ScalarI32)?)?
            .append(call(OrdinaryRecipeCall::Binary)?)?
            .append(call(OrdinaryRecipeCall::ExpandDims)?)?
            .append(call(OrdinaryRecipeCall::Fill {
                rank: output.shape().len(),
            })?)?
            .append(call(OrdinaryRecipeCall::Select)?)?;
    }
    let frames = [
        size_of::<(&Array, i32, Option<i32>, &Stream)>(),
        size_of::<(&Array, &Stream)>(),
        size_of::<(&Array, &Array, i32, &Stream)>(),
        size_of::<safemlx::ops::indexing::ArrayIndexOp<'static>>(),
        size_of::<Option<i32>>(),
        size_of::<safemlx::Dtype>(),
        size_of::<Array>() * 9,
        size_of::<Result<Array, safemlx::error::Exception>>(),
    ];
    result.metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
}

/// Local calls of the actual vocabulary ownership worker. The trace records
/// its final rank Sum separately, with the retained communication source.
pub(super) fn parallel_embedding(
    operation: WorkspaceOperationView<'_>,
) -> Option<OrdinaryCallControls> {
    let WorkspaceOperationKindView::VocabularyParallelLookup { format, policy, .. } =
        operation.kind
    else {
        return None;
    };
    // Keep the source and local parameter checks shared with numerical
    // selection; a dense row caller does not qualify packed row kernels.
    super::super::super::parallel_lookup::embedding(operation).ok()??;
    if format.encoding() != eredu_checkpoint::LinearFormat::Dense {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let sentinel = matches!(policy, eredu_nn::EmbeddingLookupPolicy::ZeroSentinel(_));
    let call = OrdinaryCallControls::call;
    validation_prefix(input.elements().ok()? == 0, sentinel)?
        // Local start/end and safe-index zero are three genuine eager seeds.
        // ge/lt/and and local subtraction are four distinct binary calls.
        .append(call(OrdinaryRecipeCall::ScalarI32)?.repeat(3)?)?
        .append(call(OrdinaryRecipeCall::Binary)?.repeat(4)?)?
        .append(call(OrdinaryRecipeCall::Select)?.repeat(2)?)?
        .append(call(OrdinaryRecipeCall::Take)?)?
        .append(call(OrdinaryRecipeCall::ExpandDims)?)?
        .append(call(OrdinaryRecipeCall::ZerosLike)?)?
        .metadata(crate::backend::nn::shared::parallel_vocabulary_lookup_control_bytes()?)
}

pub(super) fn attention(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let WorkspaceOperationKindView::Attention {
        causal: false,
        window: None,
        sinks,
        softcap: false,
        arithmetic: eredu_nn::AttentionArithmetic::Fused,
    } = operation.kind
    else {
        return None;
    };
    let mut result =
        OrdinaryCallControls::default().metadata(safemlx::fast::ordinary_sdpa_control_bytes()?)?;
    let without_sink = operation.inputs.len().checked_sub(usize::from(sinks))?;
    if !(3..=4).contains(&without_sink) {
        return None;
    }
    if without_sink == 4 {
        let mask = operation.inputs.get(3)?;
        // The ordinary attention adapter owns its promoted mask. The request
        // adapter borrows the same mask/sink; retain the owned adapter's
        // allowance too when these equal operation layouts share a trace.
        result = if mask.dtype() == WorkspaceDtype::Bool {
            result.metadata(Array::ordinary_clone_control_bytes()?)?
        } else {
            result.append(OrdinaryCallControls::call(OrdinaryRecipeCall::Cast)?)?
        };
    }
    result
        .metadata(
            size_of::<Option<Array>>()
                .checked_add(size_of::<Result<Option<Array>, eredu_nn::Error>>())?
                .checked_add(size_of::<Option<&crate::MlxTensor>>())?,
        )?
        .metadata(
            size_of::<eredu_nn::AttentionRequest<'_, crate::MlxTensor>>().checked_add(
                size_of::<(
                    &Array,
                    &Array,
                    &Array,
                    f32,
                    Option<&Array>,
                    Option<&Array>,
                    Option<f32>,
                    eredu_nn::AttentionArithmetic,
                    &Stream,
                )>(),
            )?,
        )
}

pub(super) fn rotary(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    if let WorkspaceOperationKindView::Rotary(spec, Some(offset)) = operation.kind {
        if spec.arithmetic == eredu_nn::RotaryArithmetic::InputProducts {
            return elementwise_rotary(operation, spec, offset);
        }
    }
    let yarn = match operation.kind {
        // Tensor::rope_with_frequencies borrows the actual frequency array and
        // calls the same scalar-offset batch wrapper directly.
        WorkspaceOperationKindView::RotaryFrequencies(..) => false,
        WorkspaceOperationKindView::Rotary(spec, Some(_))
            if spec.arithmetic == eredu_nn::RotaryArithmetic::Native =>
        {
            match spec.algorithm {
                eredu_nn::RotaryAlgorithm::Default => false,
                eredu_nn::RotaryAlgorithm::Yarn { .. } => true,
                _ => return None,
            }
        }
        _ => return None,
    };
    let input = operation.inputs.get(0)?;
    let mut batches = if input.shape().len() > 2 {
        usize::try_from(*input.shape().first()?).ok()?
    } else {
        1
    };
    if yarn {
        batches = batches.checked_mul(usize::try_from(input.shape()[1]).ok()?)?;
    }
    if batches == 0 {
        return None;
    }
    let rank = if yarn { 3 } else { input.shape().len() };
    let call = OrdinaryCallControls::call;
    let mut result = OrdinaryCallControls::default();
    if yarn {
        result = result
            .append(call(OrdinaryRecipeCall::ScalarF32)?)?
            .append(call(OrdinaryRecipeCall::Binary)?)?
            .append(call(OrdinaryRecipeCall::Reshape { rank: 3 })?)?;
    }
    result = result
        .metadata(safemlx::fast::ordinary_rope_invocation_control_bytes()?.checked_mul(batches)?)?;
    if batches == 1 {
        result = result.metadata(Array::ordinary_clone_control_bytes()?)?;
    } else {
        result = result
            .append(call(OrdinaryRecipeCall::StaticSlice { rank })?.repeat(batches)?)?
            .metadata(
                safemlx::ops::indexing::basic_range_index_control_bytes(rank)?
                    .checked_mul(batches)?,
            )?
            .append(call(OrdinaryRecipeCall::Join {
                inputs: batches,
                stack: false,
            })?)?;
    }
    if yarn {
        result = result.append(call(OrdinaryRecipeCall::Reshape {
            rank: input.shape().len(),
        })?)?;
    }
    let frames = [
        std::alloc::Layout::array::<Array>(batches).ok()?.size(),
        size_of::<Vec<Array>>(),
        size_of::<std::ops::Range<i32>>(),
        size_of::<(&Array, &Stream)>(),
        size_of::<&[i32]>(),
        size_of::<Array>() * 2,
        size_of::<Result<Array, safemlx::error::Exception>>(),
    ];
    result.metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
}

/// Calls made by ElementwiseRotary::forward over its retained inverse-frequency
/// array. The numerical source separately authenticates that array's backing.
fn elementwise_rotary(
    operation: WorkspaceOperationView<'_>,
    spec: eredu_nn::RotarySpec,
    offset: i32,
) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let input = operation.inputs.get(0)?;
    let shape = input.shape();
    if shape.len() < 2
        || spec.dimensions <= 0
        || spec.dimensions % 2 != 0
        || shape[shape.len() - 1] < spec.dimensions
    {
        return None;
    }
    offset.checked_add(shape[shape.len() - 2])?;
    let call = OrdinaryCallControls::call;
    let slice = call(C::BasicIndex {
        input_rank: 3,
        output_rank: 3,
        operations: 3,
        reshape: false,
    })?;
    let mut result = call(C::Reshape { rank: 3 })?
        .append(call(C::ArangeI32)?)?
        .append(call(C::Cast)?.repeat(3)?)?
        .append(call(C::ExpandDims)?)?
        .append(call(C::ScalarF32)?)?
        .append(call(C::Unary)?.repeat(2)?)?
        // Angle product, two amplitude products, four rotary products, two sums.
        .append(call(C::Binary)?.repeat(9)?)?
        .append(slice)?;
    if spec.traditional {
        result = result
            .append(call(C::Reshape { rank: 4 })?)?
            .append(
                call(C::BasicIndex {
                    input_rank: 4,
                    output_rank: 3,
                    operations: 4,
                    reshape: true,
                })?
                .repeat(2)?,
            )?
            .append(call(C::ExpandDims)?.repeat(2)?)?
            .append(call(C::Reshape { rank: 3 })?)?;
    } else {
        result = result.append(slice.repeat(2)?)?;
    }
    result = result.append(call(C::Join {
        inputs: 2,
        stack: false,
    })?)?;
    if shape[shape.len() - 1] != spec.dimensions {
        result = result.append(slice)?.append(call(C::Join {
            inputs: 2,
            stack: false,
        })?)?;
    }
    result
        .append(call(C::Reshape { rank: shape.len() })?)?
        .metadata(crate::backend::nn::rope::elementwise_rotary_control_bytes()?)
}
