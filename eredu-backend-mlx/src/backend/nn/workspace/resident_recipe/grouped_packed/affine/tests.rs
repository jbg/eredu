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
fn specification(group_size: i32) -> GroupedGatedProductSpec {
    let projection = |name: &str| {
        GroupedProjectionSpec::new(
            ParameterSpec::trainable(format!("{name}.weight")).unwrap(),
            Some(ParameterSpec::trainable(format!("{name}.bias")).unwrap()),
            LinearFormatSpec::affine(
                eredu_checkpoint::LinearFormat::Affine(
                    eredu_checkpoint::AffineQuantization::new(group_size, 4).unwrap(),
                ),
                ParameterSpec::trainable(format!("{name}.scales")).unwrap(),
                ParameterSpec::trainable(format!("{name}.biases")).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    };
    GroupedGatedProductSpec::new(
        2,
        64,
        64,
        64,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: projection("read"),
            down: projection("write"),
        },
    )
    .unwrap()
}
#[test]
fn affine_grouped_sources_keep_companions_across_selected_gather_and_observed_chunks() {
    let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for group_size in [16, 64] {
        for (positions, observed) in [(2, false), (65, false), (2, true), (65, true)] {
            let context = WorkspaceContext::new(mechanism);
            let mut module =
                WorkspaceBackend::grouped_gated_product(specification(group_size), &context)
                    .unwrap();
            let input = WorkspaceTensor::unloaded_f32(&[1, positions, 64], &context).unwrap();
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
                vec![value, bias.expect("actual separate TP output bias")]
            };
            let report = context.report(&outputs).unwrap();
            let operations = report.operations.iter().filter(|operation| {
                matches!(operation.kind, WorkspaceOperationKind::Grouped { .. })
            });
            let mut occurrences = 0;
            for operation in operations {
                occurrences += 1;
                let view = operation.as_view();
                let counts = super::super::sources::counts(view).unwrap();
                assert_eq!(counts[0], 0);
                assert_eq!(counts[if group_size == 16 { 1 } else { 2 }], 0);
                assert!(counts[if group_size == 16 { 2 } else { 1 }] > 0);
                let lower = super::super::lower(view).unwrap();
                assert!(lower.maximum_operands >= if group_size == 16 { 4 } else { 6 });
                assert!(control_bytes(view).unwrap() > 0);
                // An ordinary projection bias is not the affine companion.
                // A missing/mis-sized or nonfloating affine bias is rejected.
                for changed in [
                    WorkspaceLayout::new(&[2, 128, 1], WorkspaceDtype::Uint32).unwrap(),
                    WorkspaceLayout::new(&[2, 128, 7], WorkspaceDtype::Float32).unwrap(),
                ] {
                    let mut inputs = operation.inputs.clone();
                    inputs[6] = changed;
                    assert!(super::super::sources::counts(WorkspaceOperationView {
                        inputs: WorkspaceLayoutList::Owned(&inputs),
                        ..view
                    })
                    .is_none());
                }
            }
            assert_eq!(occurrences, if observed { 2 } else { 1 });
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
            let reduced = recorder
                .reduce_trace(&report, None, 0, outputs.len())
                .unwrap();
            assert_eq!(reduced.first_missing_operation, None);
            assert!(
                reduced.graph.is_some()
                    && reduced.dispatch.is_some()
                    && reduced.mutable_storage.is_some()
            );
        }
    }
}
