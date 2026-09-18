use super::*;
use crate::{
    EmbeddingOperator, EmbeddingSpec, Index, LinearFormat, NeuralBackend, PadMode, ParameterSpec, Tensor,
};

mod blockwise;
mod domains;
mod grouped;
mod hyper;
mod output_alias;
mod zeros_like;
mod parallel;
mod storage;
mod masked_scatter;

/// Deliberately simple mechanism with exact documented allocation behavior.
/// These facts are for the test mechanism, never substituted for native facts.
#[derive(Debug)]
struct AllocatingMechanism;
impl WorkspaceMechanisms for AllocatingMechanism {
    fn grouped_observation_schedule(
        &self,
        _: &WorkspaceGroupedBank,
        _: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        Ok(Some(WorkspaceGroupedObservationSchedule::WholeBatch))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "test mechanism has no disjoint host workspace".into(),
        }))
    }

    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        let aliases = matches!(
            operation.kind,
            WorkspaceOperationKind::View(_) | WorkspaceOperationKind::Transpose(_) | WorkspaceOperationKind::Index { .. } | WorkspaceOperationKind::StaticSlice { .. }
        );
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| {
                    if aliases {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        layout.bytes().map(WorkspaceOutputStorage::Allocate)
                    }
                })
                .collect::<Result<_, _>>()?,
            scratch_bytes: if aliases { 0 } else { 7 },
            assumptions:
                "test mechanism: exact logical output, seven scratch bytes, metadata-only views"
                    .into(),
        }))
    }
}
// Equation fixtures describe already resident inputs. Constructor costs are
// exercised separately by placeholder_tests and by layerwise construction.
pub(super) fn existing_f32(
    shape: &[i32],
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    WorkspaceTensor::existing(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32)?,
        context,
    )
}
fn existing_i32(shape: &[i32], context: &WorkspaceContext) -> Result<WorkspaceTensor, Error> {
    WorkspaceTensor::existing(WorkspaceLayout::new(shape, WorkspaceDtype::Int32)?, context)
}
fn context() -> WorkspaceContext {
    WorkspaceContext::new(AllocatingMechanism)
}

#[test]
fn gather_preserves_axis_and_unsigned_reduction_indices() {
    let context = context();
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[3, 5, 7], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    let indices = WorkspaceTensor::argmin_axis(&input, 2, false, &context).unwrap();
    assert_eq!(indices.layout().dtype(), WorkspaceDtype::Uint32);
    let output = input.take_axis(&indices, -1, &context).unwrap();
    assert_eq!(output.shape(), [3, 5, 3, 5]);
    let report = context.report(&[output]).unwrap();
    assert!(matches!(
        report.operations[1].kind,
        WorkspaceOperationKind::Gather { axis: 2 }
    ));
    let floating = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    assert!(input.take_axis(&floating, 0, &context).is_err());
}

#[test]
fn storage_identity_survives_clones_slices_and_retained_state_roots() {
    let context = context();
    let old = existing_f32(&[2, 3, 4], &context).unwrap();
    let old_view = old.reshape(&[6, 4], &context).unwrap();
    assert_eq!(context.report(&[old_view]).unwrap().total_bytes, Some(0));
    let new = old.square(&context).unwrap(); // 96 bytes and seven scratch
    let tail = new
        .index(&[Index::Full, Index::Range(2, 3)], &context)
        .unwrap();
    let retained = context.report(&[new.clone(), tail.clone(), tail]).unwrap();
    assert_eq!(retained.total_bytes, Some(103));
    assert_eq!(retained.retained_bytes, Some(96));
    assert_eq!(retained.transient_bytes, Some(7));
    let _ = new.tanh(&context).unwrap(); // another distinct backing allocation
    assert_eq!(context.report(&[new]).unwrap().transient_bytes, Some(110));
}

#[test]
fn a_new_span_treats_previous_retained_storage_as_existing_residency() {
    let context = context();
    let old = WorkspaceTensor::full_f32(0.7, &[3, 5], &context).unwrap();
    assert_eq!(context.report(&[]).unwrap().total_bytes, Some(67));
    context.begin_span();
    let alias = old.reshape(&[15], &context).unwrap();
    assert_eq!(context.report(&[alias]).unwrap().total_bytes, Some(0));
    let _ = old.square(&context).unwrap();
    assert_eq!(context.report(&[]).unwrap().total_bytes, Some(67));
}

#[test]
fn inference_state_replacement_keeps_displaced_backing_until_completion() {
    let context = context();
    let old = existing_f32(&[2, 3, 4], &context).unwrap();
    let tail = old
        .index(&[Index::Full, Index::Range(2, 3)], &context)
        .unwrap();
    // A narrow view still retains the complete 96-byte allocation.
    context.begin_state_span([&old, &tail, &tail]).unwrap();
    let replacement = tail.square(&context).unwrap();
    let report = context.report(&[replacement.clone()]).unwrap();
    assert_eq!(report.transient_bytes, Some(7));
    let state = report.state.as_ref().unwrap();
    assert_eq!(state.retained_bytes, Some(32));
    assert_eq!(state.displaced_bytes, Some(96));
    assert_eq!(report.inference_transient_bytes(), Some(103));
    // The next completed span has one 32-byte root, not a retained history of
    // all old generations of this state. Cloned handles never add a charge.
    context
        .begin_state_span([&replacement, &replacement])
        .unwrap();
    let next = replacement.square(&context).unwrap();
    let report = context.report(&[next]).unwrap();
    assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(32));
    assert_eq!(report.inference_transient_bytes(), Some(39));
}

#[test]
fn inference_state_aliases_and_unchanged_components_are_not_displaced_or_double_counted() {
    let context = context();
    let fixed = existing_f32(&[5], &context).unwrap();
    let history = existing_f32(&[2, 3, 4], &context).unwrap();
    context.begin_state_span([&fixed, &history]).unwrap();
    let view = history
        .index(&[Index::Full, Index::Range(2, 3)], &context)
        .unwrap();
    let report = context
        .report(&[fixed.clone(), view.clone(), view])
        .unwrap();
    assert_eq!(report.retained_bytes, Some(0));
    assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(116));
    assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
    assert_eq!(report.inference_transient_bytes(), Some(0));
    assert_eq!(
        report.total_bytes,
        Some(0),
        "primitive diagnostics exclude existing state"
    );
}

#[test]
fn inference_state_requires_explicit_seeding_and_keeps_unknown_backing_unknown() {
    #[derive(Debug)]
    struct Missing;
    impl WorkspaceMechanisms for Missing {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(None)
        }
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "no fixture host payload".into(),
            }))
        }
    }
    let context = WorkspaceContext::new(Missing);
    assert_eq!(
        context.report(&[]).unwrap().inference_transient_bytes(),
        None
    );
    context.begin_state_span([]).unwrap();
    assert_eq!(
        context.report(&[]).unwrap().inference_transient_bytes(),
        Some(0)
    );
    let unknown = WorkspaceTensor::full_f32(1.0, &[7], &context).unwrap();
    context.begin_state_span([&unknown]).unwrap();
    let retained = context.report(&[unknown.clone()]).unwrap();
    assert_eq!(retained.transient_bytes, Some(0));
    assert_eq!(retained.state.as_ref().unwrap().retained_bytes, None);
    assert_eq!(retained.inference_transient_bytes(), None);
    let displaced = context.report(&[]).unwrap();
    assert_eq!(displaced.state.as_ref().unwrap().retained_bytes, Some(0));
    assert_eq!(displaced.state.as_ref().unwrap().displaced_bytes, None);
    assert_eq!(displaced.inference_transient_bytes(), None);
}

#[test]
fn foreign_state_cannot_clear_a_live_span_or_its_opening_state() {
    let context = context();
    let old = existing_f32(&[7], &context).unwrap();
    context.begin_state_span([&old]).unwrap();
    let new = old.square(&context).unwrap();
    let other = WorkspaceContext::new(AllocatingMechanism);
    let foreign = existing_f32(&[7], &other).unwrap();
    assert!(context.begin_state_span([&foreign]).is_err());
    let report = context.report(&[new]).unwrap();
    assert_eq!(report.operations.len(), 1);
    assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(28));
    assert_eq!(report.inference_transient_bytes(), Some(35));
}

#[test]
fn incomplete_native_facts_propagate_through_aliases_and_state() {
    #[derive(Debug)]
    struct Missing;
    impl WorkspaceMechanisms for Missing {
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "test mechanism has no disjoint host workspace".into(),
            }))
        }

        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            if matches!(
                operation.kind,
                WorkspaceOperationKind::Elementwise("square")
            ) {
                Ok(None)
            } else {
                AllocatingMechanism.operation_bound(operation)
            }
        }
    }
    let context = WorkspaceContext::new(Missing);
    let old = existing_f32(&[2, 4], &context).unwrap();
    let new = old.square(&context).unwrap();
    let alias = new.reshape(&[8], &context).unwrap();
    let report = context.report(&[alias]).unwrap();
    assert_eq!(report.unpriced_operations, [0]);
    assert_eq!(
        (
            report.total_bytes,
            report.retained_bytes,
            report.transient_bytes
        ),
        (None, None, None)
    );
}

#[test]
fn contexts_cannot_reuse_storage_from_a_different_selected_mechanism() {
    let a = context();
    let b = context();
    let value = existing_f32(&[2], &a).unwrap();
    assert!(value.square(&b).is_err());
    assert!(b.report(&[value]).is_err());
    assert_eq!(b.report(&[]).unwrap().operations.len(), 0);
}

#[test]
fn byte_and_shape_overflows_are_rejected_without_native_or_host_values() {
    assert!(WorkspaceLayout::new(&[-1], WorkspaceDtype::Float32).is_err());
    assert!(WorkspaceLayout::new(&[i32::MAX; 3], WorkspaceDtype::Float32).is_err());
    assert_eq!(
        WorkspaceLayout::new(&[i32::MAX, i32::MAX, 0], WorkspaceDtype::Float32)
            .unwrap()
            .bytes()
            .unwrap(),
        0
    );
    let context = context();
    let input = existing_f32(&[2, 3, 4], &context).unwrap();
    assert_eq!(input.reshape(&[-1, 4], &context).unwrap().shape(), [6, 4]);
    assert!(input.reshape(&[5, -1], &context).is_err());
    assert!(input.transpose_axes(&[0, 0, 2], &context).is_err());
    assert!(input.to_f32_vec(&context).is_err());
}

#[test]
fn native_output_declarations_must_cover_the_represented_storage() {
    #[derive(Debug)]
    struct Underpriced;
    impl WorkspaceMechanisms for Underpriced {
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "test mechanism has no disjoint host workspace".into(),
            }))
        }

        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(Some(WorkspaceOperationBound {
                outputs: vec![WorkspaceOutputStorage::Allocate(1)],
                scratch_bytes: 0,
                assumptions: "incorrect native bound".into(),
            }))
        }
    }
    let context = WorkspaceContext::new(Underpriced);
    assert!(WorkspaceTensor::full_f32(1.0, &[2], &context).is_err());
}

#[test]
fn scratch_overflow_is_an_error_not_a_small_or_unknown_quote() {
    #[derive(Debug)]
    struct Huge;
    impl WorkspaceMechanisms for Huge {
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "test mechanism has no disjoint host workspace".into(),
            }))
        }

        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            let mut bound = AllocatingMechanism.operation_bound(operation)?.unwrap();
            bound.scratch_bytes = u64::MAX;
            Ok(Some(bound))
        }
    }
    let context = WorkspaceContext::new(Huge);
    let value = WorkspaceTensor::full_f32(1.0, &[1], &context).unwrap();
    assert!(context.report(&[]).is_err());
    assert!(value.square(&context).is_err());
}

#[test]
fn packed_tied_readout_retains_encoding_companions_and_selected_position_count() {
    let context = context();
    let quantization = eredu_checkpoint::AffineQuantization::new(32, 3).unwrap();
    let format = LinearFormatSpec::affine(
        LinearFormat::Affine(quantization),
        ParameterSpec::trainable("embed.scales").unwrap(),
        ParameterSpec::trainable("embed.biases").unwrap(),
    )
    .unwrap();
    let mut embedding = WorkspaceBackend::embedding(
        EmbeddingSpec {
            vocabulary: 35,
            dimensions: 64,
            weight: ParameterSpec::trainable("embed.weight").unwrap(),
            format: format.clone(),
        },
        &context,
    )
    .unwrap();
    let topology = crate::validate_parameter_topology(&embedding).unwrap();
    assert_eq!(topology.len(), 3);
    assert_eq!(
        topology[1].linear_companion_of,
        Some(topology[0].id.clone())
    );
    let ids = WorkspaceTensor::from_i32_slice(&[2, 5, 8, 11, 14], &[1, 5], &context).unwrap();
    let hidden = embedding.forward(&ids, &context).unwrap();
    assert_eq!(hidden.shape(), [1, 5, 64]);
    let tail = hidden
        .index(&[Index::Full, Index::Range(4, 5)], &context)
        .unwrap();
    let scores = embedding.as_linear(&tail, &context).unwrap();
    assert_eq!(scores.shape(), [1, 1, 35]);
    let report = context.report(&[]).unwrap();
    let projection = report.operations.last().unwrap();
    assert!(
        matches!(&projection.kind, WorkspaceOperationKind::Projection(actual) if *actual == format)
    );
    assert_eq!(
        projection
            .inputs
            .iter()
            .map(|layout| layout.shape())
            .collect::<Vec<_>>(),
        [&[1, 1, 64][..], &[35, 6], &[35, 2], &[35, 2]]
    );
    assert_eq!(projection.inputs[1].dtype(), WorkspaceDtype::Uint32);
}

#[test]
fn checked_tensor_shapes_cover_broadcast_batched_matmul_and_convolution() {
    let context = context();
    let a = existing_f32(&[2, 1, 3, 4], &context).unwrap();
    let b = existing_f32(&[1, 5, 4, 7], &context).unwrap();
    assert_eq!(
        WorkspaceTensor::matmul(&a, &b, &context).unwrap().shape(),
        [2, 5, 3, 7]
    );
    let input = existing_f32(&[2, 11, 4], &context).unwrap();
    let kernel = existing_f32(&[6, 3, 2], &context).unwrap();
    assert_eq!(
        WorkspaceTensor::conv1d(&input, &kernel, 2, 1, 1, 2, &context)
            .unwrap()
            .shape(),
        [2, 6, 6]
    );
    let empty = existing_f32(&[0, 4], &context).unwrap();
    let row = existing_f32(&[1, 4], &context).unwrap();
    assert_eq!(empty.add(&row, &context).unwrap().shape(), [0, 4]);
}

#[test]
fn selective_scan_quotes_new_state_and_sequence_without_recharging_existing_state() {
    let context = context();
    let values = existing_f32(&[2, 3, 4, 5], &context).unwrap();
    let vectors = existing_f32(&[2, 3, 4, 7], &context).unwrap();
    let times = existing_f32(&[2, 3, 4], &context).unwrap();
    let heads = existing_f32(&[4], &context).unwrap();
    let old = existing_f32(&[2, 4, 5, 7], &context).unwrap();
    let input = crate::SelectiveStateSpaceScanInput {
        values: &values,
        input_state: &vectors,
        output_state: &vectors,
        time_step: &times,
        time_step_bias: &heads,
        transition_log: &heads,
        skip: &heads,
        initial_state: Some(&old),
        time_step_floor: 0.01,
        chunk_size: 2,
    };
    let output = WorkspaceBackend::selective_state_space_scan(input.clone(), &context).unwrap();
    assert_eq!(output.state.shape(), old.shape());
    assert_eq!(output.output.shape(), values.shape());
    let report = context.report(&[output.state]).unwrap();
    assert_eq!(report.retained_bytes, Some(2 * 4 * 5 * 7 * 4));
    assert_eq!(report.transient_bytes, Some(2 * 3 * 4 * 5 * 4 + 7));
    let mut invalid = input;
    invalid.initial_state = Some(&values);
    assert!(WorkspaceBackend::selective_state_space_scan(invalid, &context).is_err());
}

#[test]
fn integer_control_values_cannot_underprice_widening_float_and_reduction_outputs() {
    let context = context();
    let ids = existing_i32(&[2, 3], &context).unwrap();
    let boolean = ids.equal_i32(3, &context).unwrap();
    let count = WorkspaceTensor::sum_axis(&boolean, 1, false, &context).unwrap();
    assert_eq!(count.layout().bytes().unwrap(), 8);
    for value in [
        boolean.divide(&boolean, &context).unwrap(),
        boolean.multiply_scalar(0.5, &context).unwrap(),
        boolean.tanh(&context).unwrap(),
    ] {
        assert_eq!(value.layout().dtype(), WorkspaceDtype::Float32);
        assert_eq!(value.layout().bytes().unwrap(), 24);
    }
}

#[test]
fn copy_or_view_state_preserves_every_possible_backing_owner_once() {
    #[derive(Debug)]
    struct StridedMechanism;
    impl WorkspaceMechanisms for StridedMechanism {
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "test mechanism has no disjoint host workspace".into(),
            }))
        }

        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            let mut bound = AllocatingMechanism.operation_bound(operation)?.unwrap();
            if matches!(operation.kind, WorkspaceOperationKind::View(_) | WorkspaceOperationKind::Transpose(_)) {
                bound.outputs = operation
                    .outputs
                    .iter()
                    .map(|layout| {
                        Ok(WorkspaceOutputStorage::AllocateOrAliasInputs {
                            bytes: layout.bytes()?,
                            inputs: vec![0],
                        })
                    })
                    .collect::<Result<_, Error>>()?;
            }
            Ok(Some(bound))
        }
    }
    let context = WorkspaceContext::new(StridedMechanism);
    let full = WorkspaceTensor::full_f32(0.125, &[2, 3, 4], &context).unwrap();
    let tail = full
        .index(&[Index::Full, Index::Range(2, 3)], &context)
        .unwrap();
    let possible_copy = tail.reshape(&[2, 4], &context).unwrap();
    let alias = possible_copy
        .index(&[Index::Range(0, 1)], &context)
        .unwrap();
    let report = context.report(&[alias, possible_copy]).unwrap();
    assert_eq!(report.total_bytes, Some(96 + 32 + 7));
    assert_eq!(report.retained_bytes, Some(96 + 32));
    assert_eq!(report.transient_bytes, Some(7));
}

#[test]
fn sliding_attention_joins_heads_and_checks_retained_position_geometry() {
    let context = context();
    let tensor = |shape: &[i32]| existing_f32(shape, &context).unwrap();
    let q = tensor(&[2, 4, 3, 8]);
    let k = tensor(&[2, 2, 7, 8]);
    let v = tensor(&[2, 2, 7, 5]);
    let plain =
        WorkspaceBackend::attention(q.clone(), k.clone(), v.clone(), 0.5, None, &context).unwrap();
    assert_eq!(plain.shape(), [2, 4, 3, 5]);
    let sliding = WorkspaceBackend::sliding_window_attention(
        q.clone(),
        k.clone(),
        v.clone(),
        0.5,
        3,
        4,
        &context,
    )
    .unwrap();
    assert_eq!(sliding.shape(), [2, 3, 20]);
    for (window, offset) in [(0, 4), (3, -1), (3, 3), (3, i32::MAX)] {
        assert!(
            WorkspaceBackend::sliding_window_attention(
                q.clone(),
                k.clone(),
                v.clone(),
                0.5,
                window,
                offset,
                &context
            )
            .is_err()
        );
    }
}

#[test]
fn tensor_clone_population_tracks_aliases_and_preserves_failed_span() {
    let context = context();
    let value = existing_f32(&[4], &context).unwrap();
    context.begin_state_span([&value]).unwrap();
    let alias = {
        let _loan = context.trace.borrow_mut();
        value.clone()
    };
    let first = context.report(std::slice::from_ref(&alias)).unwrap();
    assert_eq!(first.tensor_handle_clones, Some(1));
    assert_eq!(first.tensor_buffers.total_bytes, Some(0));
    assert_eq!(first.clone().tensor_handle_clones, Some(1));
    let foreign = WorkspaceContext::new(AllocatingMechanism);
    let wrong = existing_f32(&[4], &foreign).unwrap();
    assert!(context.begin_state_span([&wrong]).is_err());
    assert_eq!(context.report(&[]).unwrap().tensor_handle_clones, Some(1));
    context.begin_span();
    assert_eq!(context.report(&[]).unwrap().tensor_handle_clones, Some(0));
    context.identity.tensor_handle_clones.set(Some(usize::MAX));
    drop(alias.clone());
    assert_eq!(context.report(&[]).unwrap().tensor_handle_clones, None);
    context.begin_state_span([&value]).unwrap();
    assert_eq!(context.report(&[]).unwrap().tensor_handle_clones, Some(0));
}

#[test]
fn transpose_retains_normalized_axes_and_shared_storage_for_repeated_dimensions() {
    let context=context();let input=existing_f32(&[2,3,3,4],&context).unwrap();
    context.begin_span();
    let output=input.transpose_axes(&[0,-2,1,-1],&context).unwrap();
    assert_eq!(output.shape(),input.shape());
    assert!(Rc::ptr_eq(&input.storage,&output.storage));
    let report=context.report(&[output]).unwrap();
    let WorkspaceOperationKind::Transpose(axes)=&report.operations[0].kind else {panic!("exact transpose descriptor");};
    assert_eq!(axes,&[0,2,1,3]);
    let WorkspaceOperationKindView::Transpose(borrowed)=report.operations[0].kind.as_view() else {unreachable!()};
    assert!(std::ptr::eq(axes.as_ptr(),borrowed.as_ptr()));
    // The transposed result aliases an existing source; this span creates no backing.
    assert_eq!(report.tensor_buffers.retained_bytes,Some(0));
    assert_eq!(report.tensor_buffers.total_bytes,Some(0));
    let unknown=WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false);
    assert!(!unknown.last_axis_contiguous());
    assert!(!unknown.with_last_axis_contiguous(true).row_contiguous());
}

#[test]
fn integer_coordinate_clamp_preserves_gather_indices_through_padding_and_views() {
    let context = context();
    let coordinates = existing_i32(&[1, 2, 2], &context).unwrap();
    let sanitized = WorkspaceTensor::pad(
        &coordinates, &[(0, 0), (0, 1), (0, 0)], PadMode::Constant, &context,
    ).unwrap().maximum_i32(0, &context).unwrap();
    let batch = WorkspaceTensor::concatenate(&[sanitized], 0, &context).unwrap();
    let x = batch.index(&[Index::Full, Index::Full, Index::At(0)], &context).unwrap();
    assert_eq!(x.layout().dtype(), WorkspaceDtype::Int32);
    let table = existing_f32(&[4, 7], &context).unwrap();
    let positions = table.take_axis(&x, 0, &context).unwrap();
    assert_eq!(positions.shape(), [1, 3, 7]);
    assert_eq!(positions.layout().dtype(), WorkspaceDtype::Float32);
    assert!(table.take_axis(&x.maximum_scalar(0.0, &context).unwrap(), 0, &context).is_err());
}


#[test]
fn prepared_parameter_representations_extend_static_source_before_construction_only() {
    use crate::ParameterId;
    let context=WorkspaceContext::new(AllocatingMechanism);
    let row=|name:&str,shape:&[i32],dtype:WorkspaceFloatingType| {
        WorkspaceParameterRepresentation::new(ParameterId::new(name).unwrap(),
            context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,false))))
    };
    context.install_parameter_representations(vec![
        row("static",&[2,3],WorkspaceFloatingType::Bfloat16),
        row("conflict",&[2,3],WorkspaceFloatingType::Float32),
    ]).unwrap();
    context.extend_parameter_representations(vec![
        row("unit",&[2,3],WorkspaceFloatingType::Float16),
        row("conflict",&[2,3],WorkspaceFloatingType::Bfloat16),
    ]).unwrap();
    let get=|name:&str,shape:&[i32]|WorkspaceTensor::unloaded_parameter_f32(
        &ParameterSpec::trainable(name).unwrap(),shape,&context).unwrap();
    assert_eq!(get("static",&[2,3]).layout().representation().unwrap().dtype(),WorkspaceFloatingType::Bfloat16);
    assert_eq!(get("unit",&[2,3]).layout().representation().unwrap().dtype(),WorkspaceFloatingType::Float16);
    assert_eq!(get("conflict",&[2,3]).layout().representation(),None);
    assert_eq!(get("absent",&[2,3]).layout().representation(),None);
    assert_eq!(get("unit",&[3,2]).layout().representation(),None);
    assert!(context.extend_parameter_representations(vec![row("late",&[2,3],WorkspaceFloatingType::Float32)]).is_err());
    context.begin_span();
    assert!(context.extend_parameter_representations(Vec::new()).is_err());
}

#[test]
fn layer_norm_trace_keeps_exact_optional_affine_roles() {
    for weight in [false, true] {
        for bias in [false, true] {
            let context = context();
            let input = existing_f32(&[2, 8], &context).unwrap();
            let scale = existing_f32(&[8], &context).unwrap();
            let offset = existing_f32(&[8], &context).unwrap();
            let output = WorkspaceTensor::layer_norm(
                &input, weight.then_some(&scale), bias.then_some(&offset), 1e-6, &context,
            ).unwrap();
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let operation = &report.operations[0];
            assert!(matches!(operation.kind,
                WorkspaceOperationKind::LayerNorm { weight: actual_weight, bias: actual_bias }
                if actual_weight == weight && actual_bias == bias));
            assert_eq!(operation.inputs.len(), 1 + usize::from(weight) + usize::from(bias));
            assert_eq!(operation.outputs[0].shape(), [2, 8]);
            assert!(matches!(operation.as_view().kind,
                WorkspaceOperationKindView::LayerNorm { weight: actual_weight, bias: actual_bias }
                if actual_weight == weight && actual_bias == bias));
        }
    }
}

#[test]
fn pure_range_index_retains_normalized_coordinates_without_removing_axes() {
    let context = context();
    let input = existing_f32(&[2, 3, 7, 4], &context).unwrap();
    let selected = input.index(&[Index::Full, Index::Range(1, 3), Index::Range(-5, -1)], &context).unwrap();
    assert_eq!(selected.shape(), [2, 2, 4, 4]);
    let empty = input.index(&[Index::Range(2, 2)], &context).unwrap();
    assert_eq!(empty.shape(), [0, 3, 7, 4]);
    let removed = input.index(&[Index::Full, Index::At(-1), Index::Range(2, 6)], &context).unwrap();
    assert_eq!(removed.shape(), [2, 4, 4]);
    let report = context.report(&[selected, empty, removed]).unwrap();
    for (operation, starts, ends) in [
        (&report.operations[0], [0, 1, 2, 0], [2, 3, 6, 4]),
        (&report.operations[1], [2, 0, 0, 0], [2, 3, 7, 4]),
    ] {
        let WorkspaceOperationKind::StaticSlice { starts: actual_start, ends: actual_end, strides } = &operation.kind else {
            panic!("pure ranges must retain their actual rectangle");
        };
        assert_eq!(actual_start, &starts); assert_eq!(actual_end, &ends);
        assert_eq!(strides, &[1, 1, 1, 1]);
    }
    assert!(matches!(report.operations[2].kind, WorkspaceOperationKind::Index { selected_axes: 1 }));
    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
    context.begin_span();
    for indexes in [vec![Index::Range(1, 0)], vec![Index::Range(-3, 2)], vec![Index::Range(0, 3)],
        vec![Index::Full; 5]] {
        assert!(input.index(&indexes, &context).is_err());
    }
    assert!(context.report(&[]).unwrap().operations.is_empty());
}
