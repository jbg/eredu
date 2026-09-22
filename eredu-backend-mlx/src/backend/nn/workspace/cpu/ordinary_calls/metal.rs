//! Caller signatures for selected Metal compound producers.
use super::*;

fn fixed(inputs: usize, metadata: usize) -> Option<OrdinaryCallControls> {
    let mut value = OrdinaryCallControls::default()
        .metadata(metadata)?
        // apply_fixed_impl publishes one ordinary C array through mlx_array_set_.
        // The concrete array object is identical to an inspection clone shell;
        // no clone operation or Original result slot is being inferred here.
        .metadata(Array::inspection_clone_handle_bytes())?;
    value
        .observed
        .include(safemlx::OperationEvent::ordinary_reserved_array_vector_control_layout(inputs)?)?;
    Some(value)
}

pub(super) fn activation(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let input = operation.inputs.get(0)?;
    if input.elements().ok()? == 0 {
        return super::activation_calls(operation);
    }
    if input.dtype() != WorkspaceDtype::Float32 {
        return None;
    }
    let dtype = input.representation().map(|value| value.dtype());
    // The same worker widens F16/BF16 inputs before its F32 kernel. An absent
    // representation retains the finite caller alternatives without certifying
    // a tensor precision or changing its backing source.
    let widening = OrdinaryCallControls::call(OrdinaryRecipeCall::Cast)?;
    let result = match dtype {
        Some(WorkspaceFloatingType::Float32) => OrdinaryCallControls::default(),
        Some(WorkspaceFloatingType::Float16 | WorkspaceFloatingType::Bfloat16) => widening,
        None => OrdinaryCallControls::default().union(widening),
        _ => return None,
    };
    result
        .append(fixed(
            1,
            crate::backend::nn::arithmetic::f32_pointwise_control_bytes(input.shape().len())?,
        )?)?
        .append(OrdinaryCallControls::call(OrdinaryRecipeCall::Cast)?)?
        .metadata(size_of::<(Array, &Stream, safemlx::Dtype, Option<Array>)>())
}

pub(super) fn reduction(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    // Tensor's direct reductions call the ordinary MLX wrappers on both devices.
    // The separately selected compound row kernels are not these operations.
    OrdinaryCallControls::call(OrdinaryRecipeCall::ReduceAxis)
}

pub(super) fn row_rms(rank: usize) -> Option<OrdinaryCallControls> {
    OrdinaryCallControls::call(OrdinaryRecipeCall::Unary)?
        .append(OrdinaryCallControls::call(OrdinaryRecipeCall::SliceF32 {
            elements: 1,
            rank: 1,
        })?)?
        .append(fixed(
            3,
            crate::backend::managed_memory::row_kernels::rms_control_bytes(rank)?,
        )?)
}

pub(super) fn rms(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    let call = OrdinaryCallControls::call;
    let input = operation.inputs.get(0)?;
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
    if input.elements().ok()? == 0 {
        return super::rms_calls(operation);
    }
    let dtype = input
        .representation()
        .map(|representation| representation.dtype());
    let mut result = OrdinaryCallControls::default();
    if offset {
        result = result
            .append(call(C::ScalarF32)?)?
            .append(call(C::Binary)?)?;
    }
    if learned {
        let gain = if offset {
            Some(WorkspaceFloatingType::Float32)
        } else {
            operation
                .inputs
                .get(1)?
                .representation()
                .map(|representation| representation.dtype())
        };
        let ordinary = call(C::RmsNorm)?;
        let widened = || {
            call(C::Cast)?
                .append(row_rms(input.shape().len())?)?
                .append(call(C::Cast)?)?
                .append(call(C::Binary)?)
        };
        let selected = match (dtype, gain) {
            (Some(input), Some(weight))
                if input == weight
                    && matches!(
                        input,
                        WorkspaceFloatingType::Float16 | WorkspaceFloatingType::Bfloat16
                    ) =>
            {
                widened()?
            }
            (Some(_), Some(_)) => ordinary,
            // The selected input_precision_rms worker has two finite caller
            // branches. Missing scalar precision retains their allowance union;
            // it does not establish a dtype or a different allocation source.
            _ => ordinary.union(widened()?),
        };
        result = result.append(selected)?;
        if final_cast {
            result = result.append(call(C::Cast)?)?;
        }
    } else if dtype == Some(WorkspaceFloatingType::Float32) {
        result = result.append(row_rms(input.shape().len())?)?;
    } else if dtype.is_none() {
        result =
            result.append(row_rms(input.shape().len())?.union(super::rms_calls(operation)?))?;
    } else {
        return super::rms_calls(operation);
    }
    result.metadata(size_of::<(
        &Array,
        Option<&Array>,
        f32,
        &Stream,
        Result<Array, safemlx::error::Exception>,
    )>())
}

/// Selected F32 pointwise worker, including its final (possibly aliasing) cast.
pub(super) fn f32_pointwise(rank: usize) -> Option<OrdinaryCallControls> {
    fixed(
        1,
        crate::backend::nn::arithmetic::f32_pointwise_control_bytes(rank)?,
    )?
    .append(OrdinaryCallControls::call(OrdinaryRecipeCall::Cast)?)?
    .metadata(size_of::<(Array, &Stream, safemlx::Dtype, Option<Array>)>())
}
/// routing_sum_last selects this source for positive F32 Metal coefficients.
pub(super) fn f32_sum(rank: usize) -> Option<OrdinaryCallControls> {
    fixed(
        1,
        crate::backend::managed_memory::row_kernels::sum_control_bytes(rank)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_metal_gated_activation_preserves_finite_precision_branches() {
        let mechanism =
            crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let make = |dtype: Option<WorkspaceFloatingType>| {
            let input = WorkspaceLayout::new(&[2, 16], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(dtype.map(|dtype| WorkspaceRepresentation::new(dtype, true)));
            WorkspaceOperation {
                kind: WorkspaceOperationKind::GatedProduct(
                    eredu_nn::GatedProductPolicy::ordinary_silu(),
                ),
                inputs: vec![input.clone(), input.clone()],
                outputs: vec![input],
            }
        };
        let unknown = make(None);
        for policy in [
            eredu_nn::GatedProductPolicy::ordinary_silu(),
            eredu_nn::GatedProductPolicy::ordinary_gelu_approximate(),
            eredu_nn::GatedProductPolicy::bounded_silu(1.7).unwrap(),
            eredu_nn::GatedProductPolicy::new(
                eredu_nn::GatedProductActivation::Silu,
                Some(1.3),
                Some(2.1),
                1.7,
                0.2,
            )
            .unwrap(),
        ] {
            let mut absent = make(None);
            absent.kind = WorkspaceOperationKind::GatedProduct(policy);
            let bound = mechanism
                .ordinary_call_controls(absent.as_view())
                .unwrap()
                .unwrap();
            for dtype in [
                WorkspaceFloatingType::Float32,
                WorkspaceFloatingType::Float16,
                WorkspaceFloatingType::Bfloat16,
            ] {
                let mut actual = make(Some(dtype));
                actual.kind = WorkspaceOperationKind::GatedProduct(policy);
                let selected = mechanism
                    .ordinary_call_controls(actual.as_view())
                    .unwrap()
                    .unwrap();
                assert!(selected.metadata_bytes <= bound.metadata_bytes);
                assert!(
                    selected.observed.observed_host_bytes <= bound.observed.observed_host_bytes
                );
                assert!(
                    selected.observed.control_allocations <= bound.observed.control_allocations
                );
            }
            assert!(
                absent
                    .inputs
                    .iter()
                    .all(|value| value.representation().is_none())
            );
        }
        let mut integer = unknown;
        integer.inputs[0] = WorkspaceLayout::new(&[2, 16], WorkspaceDtype::Int32).unwrap();
        assert!(activation(integer.as_view()).is_none());
    }
}
