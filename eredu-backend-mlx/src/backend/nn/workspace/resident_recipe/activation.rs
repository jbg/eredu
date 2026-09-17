//! Receipts for the actual shared GELU equations in backend/nn/layers.rs.
//! Physical and host facts remain in the existing basic/host emitters.
use super::Lowering;

pub(super) fn gelu() -> Lowering {
    // Divide, Add, Multiply, Divide each permit two casts, two broadcasts and
    // one result (five primitives, six input edges). Erf adds a cast and its
    // unary result. The three actual eager scalar descriptors are distinct.
    Lowering::plain(4 * 5 + 2, 4 * 6 + 2, 3)
}

pub(super) fn gelu_approximate() -> Lowering {
    // Seven binary calls (including Power) plus Sqrt and Tanh, each with its
    // possible cast. Five actual scalar constructors feed this same equation.
    // No tensor compaction or custom source/cache is introduced by these unary
    // and binary Metal workers. Their complete output/cast populations remain
    // in the existing physical emitter; no fusion or donation is assumed.
    Lowering::plain(7 * 5 + 2 * 2, 7 * 6 + 2 * 2, 5)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::super::*;
    use eredu_nn::{GatedProductPolicy, NeuralBackend, Tensor};

    #[test]
    fn standalone_and_gated_gelu_use_complete_native_receipts() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let qualified = safemlx::OperationEvent::resident_graph_layout(1, 0, 4).is_some()
            && safemlx::OperationEvent::eval_record_layout(1, 1, 1).is_some();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_some() {
            assert!(
                qualified,
                "selected native Record/Graph layouts must qualify"
            );
        }
        for (approximate, gated) in [(false, false), (true, false), (true, true)] {
            for dtype in [
                WorkspaceFloatingType::Float32,
                WorkspaceFloatingType::Bfloat16,
                WorkspaceFloatingType::Float16,
            ] {
                let context = WorkspaceContext::new(mechanism);
                let input = WorkspaceTensor::existing(
                    context
                        .layout(&[2, 7], WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(dtype, false))),
                    &context,
                )
                .unwrap();
                let up = input.clone();
                context.begin_span();
                let output = if gated {
                    WorkspaceBackend::gated_product(
                        input,
                        up,
                        GatedProductPolicy::ordinary_gelu_approximate(),
                        &context,
                    )
                } else if approximate {
                    WorkspaceBackend::gelu_approximate(input, &context)
                } else {
                    WorkspaceTensor::gelu(&input, &context)
                }
                .unwrap();
                let report = context.report(&[output]).unwrap();
                assert_eq!(report.operations.len(), 1);
                let operation = &report.operations[0];
                let bound = mechanism.operation_bound(operation).unwrap().unwrap();
                let outputs = bound
                    .outputs
                    .iter()
                    .map(|effect| match effect {
                        WorkspaceOutputStorage::Allocate(bytes)
                        | WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, .. } => *bytes,
                        other => panic!("GELU needs its existing output capacity: {other:?}"),
                    })
                    .sum::<u64>();
                let recorder = ResidentRecipeRecorder::new(
                    InferenceGeometry {
                        batch_size: 1,
                        cached_positions: 0,
                        input_positions: 2,
                        max_output_tokens: 1,
                        prefill_chunk_positions: 2,
                        output: eredu_core::OutputDemand::Sequence,
                    },
                    mechanism,
                );
                let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
                assert_eq!(reduced.first_missing_operation, None);
                assert_eq!(reduced.unqualified_kernel_owner, None);
                assert_eq!(reduced.validation_roots, 0);
                assert_eq!(
                    reduced.mutable_storage.unwrap().mutable_bytes(),
                    outputs + bound.scratch_bytes
                );
                assert!(reduced.mutable_storage.unwrap().maximum_births() > 1);
                if qualified {
                    assert!(reduced.graph.is_some() && reduced.traversal.is_some());
                    assert!(reduced.dispatch.is_some() && reduced.query_controls.is_some());
                }
                let mut unknown = report;
                unknown.operations[0].kind =
                    WorkspaceOperationKind::Elementwise("unrecognized_activation");
                let refused = recorder.reduce_trace(&unknown, None, 0, 1).unwrap();
                assert_eq!(refused.first_missing_operation, Some(0));
                assert!(refused.mutable_storage.is_none() && refused.dispatch.is_none());
            }
        }
    }
}
