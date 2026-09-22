//! Safe calls made by the selected positive, dense F32 routing workers.
//! Geometry comes from the same selector plan; native primitive and completion
//! populations stay with that plan and the enclosing completion source.
use super::*;
use crate::backend::nn::workspace::cpu::ordinary_calls::{self, OrdinaryCallControls};
use safemlx::{Array, Stream, ops::OrdinaryRecipeCall as C};

fn calls<const N: usize>(sources: [(C, usize); N]) -> Option<OrdinaryCallControls> {
    sources
        .into_iter()
        .try_fold(OrdinaryCallControls::default(), |total, (source, count)| {
            total.append(OrdinaryCallControls::call(source)?.repeat(count)?)
        })
}

fn slice() -> C {
    C::BasicIndex {
        input_rank: 2,
        output_rank: 2,
        operations: 2,
        reshape: false,
    }
}

fn selector(
    operation: WorkspaceOperationView<'_>,
    g: Geometry<'_>,
    metal: bool,
) -> Option<OrdinaryCallControls> {
    let spec = g.spec?;
    // transform_input always flattens. These positive rows never enter its
    // empty-reduction shortcut or construct a dynamic host shape.
    let mut result = calls([(C::Reshape { rank: 2 }, 1)])?;
    if let Some(transform) = spec.input_transform() {
        result = result.append(calls([
            (C::Unary, 2),      // square, rsqrt
            (C::ReduceAxis, 1), // mean
            (C::ScalarF32, 1),  // epsilon
            (C::Binary, 3),     // epsilon add, normalize, learned scale
        ])?)?;
        if transform.inverse_sqrt_dimensions() {
            result = result.append(calls([(C::ScalarF32, 1), (C::Binary, 1)])?)?;
        }
    }
    if spec.arithmetic().projection == RoutingPrecision::Float32 {
        result = result.append(calls([(C::Cast, 2)])?)?;
    } else {
        // F32 exits the real BF16 projection probe before any kernel launch.
        result =
            result.metadata(crate::backend::nn::matrix::row_projection_probe_control_bytes()?)?;
    }
    result = result.append(calls([
        (C::TransposeDefault, 1),
        (C::Binary, 1), // matmul
        (C::Cast, 2),   // projection and score precision, even when unchanged
    ])?)?;
    if spec.bias().is_some() {
        result = result.append(calls([(C::Binary, 1)])?)?;
    }
    match spec.selection().scoring() {
        GroupScoring::Softmax => result = result.append(calls([(C::ReduceAxis, 1)])?)?,
        GroupScoring::SelectedSoftmax => {}
        GroupScoring::Sigmoid => {
            let activation = if metal {
                ordinary_calls::metal_f32_pointwise_calls(2)
            } else {
                ordinary_calls::activation_calls(WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::Elementwise("sigmoid"),
                    ..operation
                })
            }?;
            result = result.append(activation)?;
        }
        GroupScoring::SqrtSoftplus => {
            // layers::softplus calls logaddexp(x, scalar_i32(0)); its selected
            // native source owns promotion. The final sqrt is one safe call.
            result = result.append(calls([(C::ScalarI32, 1), (C::Binary, 1), (C::Unary, 1)])?)?;
        }
        _ => return None,
    }
    let mut clones = 1usize; // weights_for_indices' selected_scores
    if g.supplied {
        result = result.append(calls([(C::Reshape { rank: 2 }, 1)])?)?;
    } else {
        clones = clones.checked_add(1)?; // scores_for_choice
        if spec.correction_bias().is_some() {
            result = result.append(calls([(C::Binary, 1)])?)?;
        }
        result = result.append(calls([
            (C::ScalarF32, 1),
            (C::Binary, 1), // descending scores
            (C::ArgPartitionAxis, 1),
            (slice(), 1),
        ])?)?;
        if g.selected < g.groups {
            result = result.metadata(Stream::device_type_control_bytes()?)?;
            if metal {
                // The ordinary GPU worker completes the real global cutoff
                // predicate. Its data-dependent CPU partition uses the already
                // admitted fallback stream and the same descriptor shapes.
                result = result
                    .append(calls([
                        (C::Take, 1),
                        (C::ReduceAxis, 3),
                        (C::Binary, 3),
                        (C::Cast, 2),
                        (C::Any, 1),
                        (C::BoolScalarRead, 1),
                        (C::ArgPartitionAxis, 1),
                        (slice(), 1),
                    ])?)?
                    .metadata(
                        crate::backend::nn::grouped::TopKGroupSelector::ordinary_tie_control_bytes(
                        )?,
                    )?;
            }
        }
    }
    result = result.append(calls([(C::Take, 1)])?)?; // take_along_axis
    let policy = spec.selection();
    if policy.scoring() == GroupScoring::SelectedSoftmax {
        result = result.append(calls([(C::ReduceAxis, 1)])?)?;
    }
    if policy.normalize_selected() {
        clones = clones.checked_add(1)?; // routing_sum_last F32 work
        let sum = if metal {
            ordinary_calls::metal_f32_sum_calls(2)?
        } else {
            OrdinaryCallControls::call(C::ReduceAxis)?.metadata(cpu_sum_probe_controls()?)?
        };
        result = result.append(sum)?.append(calls([
            (C::Cast, 1),   // routing_sum_last's result dtype
            (C::Binary, 1), // normalize the selected coefficients
        ])?)?;
        if policy.normalization_epsilon() != 0.0 {
            result = result.append(calls([(C::ScalarF32, 1), (C::Binary, 1)])?)?;
        }
    }
    if policy.coefficient_scale() != 1.0 {
        result = result.append(calls([(C::ScalarF32, 1), (C::Binary, 1)])?)?;
    }
    if spec.coefficient_scale().is_some() {
        result = result.append(calls([(C::Take, 1), (C::Binary, 1)])?)?;
    }
    result
        .append(calls([(C::Cast, 1)])?)? // final coefficient precision
        .metadata(Array::ordinary_clone_control_bytes()?.checked_mul(clones)?)?
        .metadata(
            crate::backend::nn::grouped::TopKGroupSelector::selection_control_bytes(g.supplied)?,
        )?
        .metadata(selector_adapter_controls(g.supplied)?)
}

fn cpu_sum_probe_controls() -> Option<usize> {
    let frames = [
        size_of::<(&Array, &Stream)>(),
        size_of::<Result<Option<Array>, safemlx::error::Exception>>(),
        Stream::device_type_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

fn selector_adapter_controls(supplied: bool) -> Option<usize> {
    use crate::MlxTensor;
    use crate::backend::nn::grouped::GroupSelectionOutput;
    use crate::backend::nn::shared::MlxTopKGroupSelector;
    use eredu_nn::{Error, GroupSelection};
    let frames = [
        size_of::<(&mut MlxTopKGroupSelector, &MlxTensor, &Stream)>(),
        if supplied { size_of::<&MlxTensor>() } else { 0 },
        size_of::<GroupSelectionOutput>(),
        size_of::<Result<GroupSelectionOutput, safemlx::error::Exception>>(),
        size_of::<Result<GroupSelectionOutput, Error>>(),
        size_of::<GroupSelection<MlxTensor>>(),
        size_of::<Result<GroupSelection<MlxTensor>, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

fn joint() -> Option<OrdinaryCallControls> {
    calls([
        (C::Reshape { rank: 2 }, 1),
        (C::TransposeDefault, 1),
        (C::Binary, 5), // matmul, correction, logaddexp, coefficient scales
        (slice(), 5),   // primary/always-on logits, IDs, two coefficient slices
        (C::Unary, 3),  // native sigmoid, two log-sigmoid negatives
        (C::ArgPartitionAxis, 1),
        (C::Take, 1),
        (
            C::Join {
                inputs: 2,
                stack: false,
            },
            1,
        ),
        (C::ScalarI32, 1), // log-sigmoid's actual softplus zero
        (C::ScalarF32, 1),
        (C::ReduceAxis, 1), // softmax
    ])?
    .metadata(crate::backend::nn::grouped::joint_selection_ordinary_frame_bytes()?)
}

pub(super) fn ordinary_call_controls(
    operation: WorkspaceOperationView<'_>,
    metal: bool,
) -> facts::FactResult<Option<OrdinaryCallControls>> {
    // Intervention has a distinct callback/validation worker and custody.
    if matches!(
        operation.kind,
        WorkspaceOperationKindView::GroupSelection {
            control: Some(_),
            ..
        }
    ) {
        return Ok(None);
    }
    let Some(g) = geometry_with_projection_cast(operation, metal)? else {
        return Ok(None);
    };
    if metal && g.spec.is_none() {
        return Ok(None);
    }
    Ok(if g.spec.is_some() {
        selector(operation, g, metal)
    } else {
        joint()
    })
}
