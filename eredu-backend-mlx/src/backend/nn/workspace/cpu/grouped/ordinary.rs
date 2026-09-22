//! Safe-call metadata for the same selected dense F32 grouped worker.
//! Descriptor and chunk geometry belong to the shared native plan. This census
//! adds its Rust/C callers; native leaves and backing stay with that plan.
use super::*;
use crate::backend::nn::workspace::cpu::ordinary_calls::{self, OrdinaryCallControls};
use safemlx::{ops::OrdinaryRecipeCall as C, Array, Stream};

fn calls<const N: usize>(sources: [(C, usize); N]) -> Option<OrdinaryCallControls> {
    sources
        .into_iter()
        .try_fold(OrdinaryCallControls::default(), |total, (source, count)| {
            total.append(OrdinaryCallControls::call(source)?.repeat(count)?)
        })
}

fn index(input_rank: usize, output_rank: usize, operations: usize) -> C {
    C::BasicIndex {
        input_rank,
        output_rank,
        operations,
        reshape: input_rank != output_rank,
    }
}

fn selection() -> Option<OrdinaryCallControls> {
    // group_by_id: flattened IDs, sorted permutation, selected IDs and tokens.
    calls([
        (C::Reshape { rank: 1 }, 1),
        (C::Cast, 2),
        (C::FlatArgsort, 1),
        (C::FlatTake, 1),
        (C::ScalarI32, 1),
        (C::Binary, 1),
    ])?
    .metadata(size_of::<crate::backend::nn::grouping::GroupedSelectionPlan>())
}

fn projection(p: Projection) -> Option<OrdinaryCallControls> {
    // The supplied bank transpose and inverse BF16 probe are both real calls.
    // F32 exits the probe and executes the ordinary optional-index gather_mm.
    let mut result = calls([
        (C::SwapAxes, 2),
        (C::Reshape { rank: 3 }, 1),
        (C::GatherMm, 1),
        (C::Reshape { rank: 2 }, 1),
    ])?
    .metadata(crate::backend::nn::matrix::row_projection_probe_control_bytes()?)?
    .metadata(Stream::device_type_control_bytes()?)?;
    if p.bias {
        result = result.append(calls([(C::Take, 1), (C::Binary, 1)])?)?;
    }
    result.metadata(size_of::<(&Array, &Array, &Array, bool, &Stream)>())
}

fn activation(
    d: Descriptor,
    operation: WorkspaceOperationView<'_>,
    metal: bool,
) -> Option<OrdinaryCallControls> {
    let selected = |name| {
        // The selected projection produces a positive F32 [selections, width]
        // array on either device; Metal selects its actual fixed pointwise source.
        if metal {
            return ordinary_calls::metal_f32_pointwise_calls(2);
        }
        ordinary_calls::activation_calls(WorkspaceOperationView {
            kind: WorkspaceOperationKindView::Elementwise(name),
            ..operation
        })
    };
    match d.activation {
        Activation::Linear(eredu_nn::GroupedLinearActivation::Identity) => {
            Some(OrdinaryCallControls::default())
        }
        Activation::Linear(eredu_nn::GroupedLinearActivation::Silu) => selected("silu"),
        Activation::Relu2 => {
            ordinary_calls::compound_calls(WorkspaceOperationKindView::Elementwise("relu2"))
        }
        Activation::Gated(policy) => {
            let mut result = calls([(index(2, 2, 2), 2)])?;
            if policy.gate_upper_bound().is_some() {
                result = result.append(OrdinaryCallControls::call(C::Clip {
                    minimum: false,
                    maximum: true,
                })?)?;
            }
            if policy.up_absolute_bound().is_some() {
                result = result.append(OrdinaryCallControls::call(C::Clip {
                    minimum: true,
                    maximum: true,
                })?)?;
            }
            if policy.up_offset() != 0.0 {
                result = result.append(calls([(C::ScalarF32, 1), (C::Binary, 1)])?)?;
            }
            let activated = match policy.activation() {
                eredu_nn::GatedProductActivation::Silu if policy.sigmoid_multiplier() == 1.0 => {
                    selected("silu")?
                }
                eredu_nn::GatedProductActivation::Silu => {
                    calls([(C::ScalarF32, 1), (C::Binary, 2)])?.append(selected("sigmoid")?)?
                }
                eredu_nn::GatedProductActivation::GeluApproximate => {
                    ordinary_calls::compound_calls(WorkspaceOperationKindView::Elementwise(
                        "gelu_approximate",
                    ))?
                }
                _ => return None,
            };
            result
                .append(activated)?
                .append(OrdinaryCallControls::call(C::Binary)?)
        }
    }
}

fn weighted(d: Descriptor) -> Option<OrdinaryCallControls> {
    // Positive token geometry is established by Descriptor. Restore the unique
    // selection order before either sum or deterministic group-order addition.
    let mut result = calls([
        (C::Reshape { rank: 1 }, 1),
        (C::FlatTake, 1),
        (index(1, 2, 2), 1),
        (C::Binary, 1),
        (C::Fill { rank: 2 }, 1),
        (C::Reshape { rank: 3 }, 2),
        (C::ScatterSingle, 1),
    ])?;
    if d.reduction == eredu_nn::GroupReduction::Sum {
        return result.append(OrdinaryCallControls::call(C::ReduceAxis)?);
    }
    result = result.append(calls([
        (C::Fill { rank: 1 }, 1),
        (C::Reshape { rank: 2 }, 2),
        (C::ScatterSingle, 1),
        (C::UnaryAxis, 1),
        (C::ExpandDims, 1),
        (C::Broadcast { rank: 3 }, 1),
        (C::Take, 1),
        (C::Fill { rank: 2 }, 1),
    ])?)?;
    result.append(calls([(index(3, 2, 3), 1), (C::Binary, 1), (C::Cast, 1)])?.repeat(d.routes)?)
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    metal: bool,
) -> Option<OrdinaryCallControls> {
    let d = if metal {
        Descriptor::inspect_metal_callers(operation)
    } else {
        Descriptor::inspect(operation)
    }
    .ok()??;
    let before = d.phase != WorkspaceGroupedPhase::Finish;
    let after = d.phase != WorkspaceGroupedPhase::Units;
    let linear = matches!(d.activation, Activation::Linear(_));
    let chunks = d.chunks();
    let chunk_count = chunks
        .into_iter()
        .try_fold(0usize, |n, (_, k)| n.checked_add(k))?;
    let mut total = OrdinaryCallControls::default();
    if before && linear {
        // The ordinary grouped-linear adapter uses the shared lazy token-domain
        // assertion. Packed Gated/Relu2 do not enter the Original mask branch.
        total = total.append(ordinary_calls::grouped_index_validation_calls()?)?;
    }
    if before && !linear {
        total = total.append(OrdinaryCallControls::call(C::Reshape { rank: 2 })?)?;
    }
    for (_, count) in chunks {
        if count == 0 {
            continue;
        }
        let mut chunk = OrdinaryCallControls::default();
        if before {
            if chunk_count > 1 {
                chunk = chunk.append(calls([(index(2, 2, 2), 3)])?)?;
            }
            chunk = chunk
                .append(selection()?)?
                .append(OrdinaryCallControls::call(C::Take)?)?
                .append(projection(d.first)?)?
                .append(activation(d, operation, metal)?)?;
            if d.phase == WorkspaceGroupedPhase::Units {
                chunk = chunk.metadata(
                    crate::backend::nn::shared::MlxGroupedGatedProduct::ordinary_unit_observation_control_bytes(d.input_rank)?,
                )?;
            }
        }
        if after {
            if let Some(down) = d.down {
                chunk = chunk.append(projection(down)?)?;
            }
            chunk = chunk.append(weighted(d)?)?;
        }
        total = total.append(chunk.repeat(count)?)?;
    }
    if after {
        if chunk_count > 1 {
            total = total
                .append(OrdinaryCallControls::call(C::Join {
                    inputs: chunk_count,
                    stack: false,
                })?)?
                .metadata(
                    crate::backend::nn::tensor::GroupedChunkOutputs::ordinary_control_bytes(
                        chunk_count,
                    )?,
                )?;
        }
        if d.separate_bias {
            total = total
                .append(selection()?)?
                .append(OrdinaryCallControls::call(C::Take)?)?
                .append(weighted(d)?)?
                .append(OrdinaryCallControls::call(C::Binary)?)?;
        }
        if !linear {
            total = total.append(calls([(
                C::Reshape { rank: d.input_rank },
                operation.outputs.len(),
            )])?)?;
        }
    }
    if matches!(
        operation.kind,
        WorkspaceOperationKindView::Grouped {
            partitions: Some(_),
            ..
        }
    ) {
        total = total.metadata(
            crate::backend::nn::shared::MlxGroupedGatedProduct::ordinary_tensor_parallel_control_bytes()?,
        )?;
    }
    let frames = [
        size_of::<Descriptor>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<OrdinaryCallControls>() * 3,
        size_of::<[(usize, usize); 2]>(),
        size_of::<std::array::IntoIter<(usize, usize), 2>>(),
        size_of::<crate::backend::nn::grouping::GroupedSelectionPlan>(),
        size_of::<(&Array, &Array, &Array, &Stream)>(),
        size_of::<(&Array, &Array, &Array, usize, usize, &Stream)>(),
        size_of::<Array>() * 12,
        size_of::<Option<Array>>(),
        size_of::<[i32; 3]>() * 3,
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
        size_of::<usize>() * 8,
    ];
    total.metadata(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?,
    )
}
