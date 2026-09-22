//! The actual packed grouped worker: explicit row indices and MXFP4 bank.
use super::*;

pub(super) fn uses_source(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(operation.kind, WorkspaceOperationKindView::Grouped {
        bank: WorkspaceGroupedBank::GatedProduct(spec), ..
    } if matches!(spec.layout(), eredu_nn::GatedProductGroupLayout::Packed { gate_up, down }
        if [gate_up, down].iter().any(|p| p.format().encoding() == eredu_checkpoint::LinearFormat::MxFp4)))
}

/// Complete actual parameter roles, including mixed affine/MXFP4 banks.
pub(super) fn packed_sources(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    if !uses_source(operation) {
        return None;
    }
    Some(super::sources::counts(operation)?[0])
}

pub(in super::super) fn control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    packed_sources(operation)?
        .checked_mul(crate::backend::nn::grouped::mxfp4_projection_control_bytes()?)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::{
        GatedProductGroupLayout, GatedProductPolicy, GroupSelection, GroupedGatedProductOperator,
        GroupedGatedProductSpec, GroupedNeuralBackend, GroupedProjectionSpec, GroupedUnitBatch,
        GroupedUnitObserver, LinearFormatSpec, ParameterSpec, Tensor,
        TensorParallelGroupedGatedProductOperator,
    };
    struct Observe;
    impl GroupedUnitObserver<WorkspaceTensor> for Observe {
        fn observe(&mut self, _: &GroupedUnitBatch<'_, WorkspaceTensor>) -> Result<(), Error> {
            Ok(())
        }
    }
    fn spec() -> GroupedGatedProductSpec {
        let projection = |name: &str| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(format!("{name}.weight")).unwrap(),
                Some(ParameterSpec::trainable(format!("{name}.bias")).unwrap()),
                LinearFormatSpec::scaled(
                    eredu_checkpoint::LinearFormat::MxFp4,
                    ParameterSpec::trainable(format!("{name}.scales")).unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
        };
        GroupedGatedProductSpec::new(
            2,
            32,
            32,
            32,
            GatedProductPolicy::new(
                eredu_nn::GatedProductActivation::Silu,
                Some(7.0),
                Some(7.0),
                1.702,
                1.0,
            )
            .unwrap(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("read"),
                down: projection("write"),
            },
        )
        .unwrap()
    }
    #[test]
    fn packed_mxfp4_tp_and_observed_sources_keep_exact_companions_and_projection_population() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for (positions, observed) in [(2, false), (65, false), (2, true)] {
            let context = WorkspaceContext::new(mechanism);
            let mut module = WorkspaceBackend::grouped_gated_product(spec(), &context).unwrap();
            let input = WorkspaceTensor::unloaded_f32(&[1, positions, 32], &context).unwrap();
            let ids = WorkspaceTensor::existing(
                WorkspaceLayout::new(&[positions, 2], WorkspaceDtype::Uint32).unwrap(),
                &context,
            )
            .unwrap();
            let coefficients = WorkspaceTensor::unloaded_f32(&[positions, 2], &context).unwrap();
            let selection = GroupSelection::new(ids, coefficients.clone(), coefficients);
            context.begin_span();
            let outputs = if observed {
                vec![module
                    .forward_grouped_with_unit_observer(
                        &input,
                        &selection,
                        &context,
                        Some(&mut Observe),
                    )
                    .unwrap()]
            } else {
                let (value, bias) = module
                    .forward_grouped_tensor_parallel(&input, &selection, 2, &context)
                    .unwrap()
                    .into_parts();
                vec![value, bias.expect("separate actual TP bias")]
            };
            let report = context.report(&outputs).unwrap();
            for operation in report
                .operations
                .iter()
                .filter(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
            {
                let source = operation.as_view();
                assert!(uses_source(source));
                assert!(packed_sources(source).unwrap() > 0);
                let lower = super::super::lower(source).unwrap();
                assert!(lower.maximum_operands >= 5);
                assert!(control_bytes(source).unwrap() > 0);
                // Changed declared scale dtype or packed row width must not be
                // admitted by a general QuantizedMatmul allowance.
                let mut inputs = operation.inputs.clone();
                inputs[5] =
                    WorkspaceLayout::new(inputs[5].shape(), WorkspaceDtype::Float32).unwrap();
                assert!(packed_sources(WorkspaceOperationView {
                    inputs: WorkspaceLayoutList::Owned(&inputs),
                    ..source
                })
                .is_none());
                let mut inputs = operation.inputs.clone();
                inputs[4] = WorkspaceLayout::new(&[2, 64, 5], WorkspaceDtype::Uint32).unwrap();
                assert!(packed_sources(WorkspaceOperationView {
                    inputs: WorkspaceLayoutList::Owned(&inputs),
                    ..source
                })
                .is_none());
            }
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: positions as u64,
                    max_output_tokens: 1,
                    prefill_chunk_positions: positions as u64,
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let recipe = recorder
                .reduce_trace(&report, None, 0, outputs.len())
                .unwrap();
            assert_eq!(recipe.first_missing_operation, None);
            assert!(
                recipe.graph.is_some()
                    && recipe.dispatch.is_some()
                    && recipe.mutable_storage.is_some()
            );
        }
    }
    #[test]
    fn packed_mxfp4_tp_scalar_source_preserves_both_outputs_and_requires_real_floating_inputs() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanism);
        let mut module = WorkspaceBackend::grouped_gated_product(spec(), &context).unwrap();
        let input = WorkspaceTensor::unloaded_f32(&[1, 2, 32], &context).unwrap();
        let ids = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 2], WorkspaceDtype::Uint32).unwrap(),
            &context,
        )
        .unwrap();
        let coefficients = WorkspaceTensor::unloaded_f32(&[2, 2], &context).unwrap();
        let selection = GroupSelection::new(ids, coefficients.clone(), coefficients);
        context.begin_span();
        let (value, bias) = module
            .forward_grouped_tensor_parallel(&input, &selection, 2, &context)
            .unwrap()
            .into_parts();
        let report = context.report(&[value, bias.unwrap()]).unwrap();
        let operation = report
            .operations
            .iter()
            .find(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
            .unwrap();
        let mut inputs = operation.inputs.clone();
        for input in &mut inputs {
            if input.dtype() == WorkspaceDtype::Float32 {
                *input = input
                    .clone()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        false,
                    )));
            }
        }
        fn scalar(
            operation: WorkspaceOperationView<'_>,
            inputs: &[WorkspaceLayout],
            output: usize,
        ) -> Option<WorkspaceRepresentation> {
            crate::backend::nn::workspace::representation::output(
                WorkspaceOperationView {
                    inputs: WorkspaceLayoutList::Owned(inputs),
                    ..operation
                },
                output,
            )
        }
        let expected = Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            false,
        ));
        assert_eq!(operation.outputs.len(), 2);
        for output in 0..2 {
            assert_eq!(scalar(operation.as_view(), &inputs, output), expected);
        }
        // The packed U32/U8 operands deliberately have no floating fact.
        assert!(inputs[4].representation().is_none());
        assert!(inputs[5].representation().is_none());
        // A missing or reduced-precision real operand cannot be admitted from
        // the logical F32 layout or the quantized format name alone.
        for slot in [0, 2, 3, 6, 9] {
            for dtype in [None, Some(WorkspaceFloatingType::Float16)] {
                let mut changed = inputs.clone();
                changed[slot] = changed[slot].clone().with_representation(
                    dtype.map(|dtype| WorkspaceRepresentation::new(dtype, false)),
                );
                assert_eq!(scalar(operation.as_view(), &changed, 0), None);
                assert_eq!(scalar(operation.as_view(), &changed, 1), None);
            }
        }
        let mut changed = inputs.clone();
        changed[5] = WorkspaceLayout::new(&[2, 64, 2], WorkspaceDtype::Uint8).unwrap();
        assert_eq!(scalar(operation.as_view(), &changed, 0), None);
        assert_eq!(scalar(operation.as_view(), &changed, 1), None);
    }
}
