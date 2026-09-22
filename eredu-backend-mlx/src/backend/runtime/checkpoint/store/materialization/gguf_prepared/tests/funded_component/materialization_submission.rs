use super::*;
use crate::backend::runtime::checkpoint::store::{
    MaterializationPayloadShape, PreparedWeightMaterialization, WeightMaterialization,
};
use safemlx::{OperationEvalTraversalLimits, OperationEvent};

fn with_materialization(
    outputs: usize,
    operation: impl FnOnce(WeightMaterialization, &OriginalScopeObserver),
) {
    let shape = MaterializationPayloadShape {
        inputs: 0,
        outputs,
        pending_sources: 0,
    };
    // This component prepares one payload/observer node, without an operation
    // bank transport. Native support is priced by the enclosing fixture.
    let bytes = PreparedWeightMaterialization::bank_layout_with_payload::<(), ()>(1, shape)
        .unwrap()
        .prepared_slot_control_bytes;
    component_destinations_with_retained_account_and_controls(
        0,
        0,
        0,
        None,
        0,
        |_| 0,
        0,
        bytes,
        |controls, observer, _, host| {
            let ready = PreparedWeightMaterialization::try_new(controls.clone(), shape)
                .unwrap_or_else(|_| panic!("prepared output storage"));
            let owner = WeightMaterialization::prepare_retained_impl(
                Vec::new(),
                Vec::new(),
                Some((ready, observer.clone())),
                None,
            )
            .unwrap();
            operation(owner, observer);
            drop(host);
        },
    );
}

fn traversal(roots: usize) -> Option<safemlx::OperationEvalTraversalLayout> {
    let layout = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots,
        arrays: roots + 1,
        tape_entries: 1,
        input_edges: roots,
        output_slots: 1,
        streams: 1,
        captures: 16,
    });
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(layout.is_some());
    }
    layout
}

#[test]
fn prepared_materialization_capacity_refuses_before_production_and_keeps_roots() {
    let input = Array::from_slice(&[7.0f32, -2.0], &[2]);
    input.evaluated().unwrap();
    with_materialization(1, |mut owner, observer| {
        owner.prepare_output_capacity(1).unwrap();
        let pointer = owner.outputs().as_ptr();
        assert!(matches!(
            owner.prepare_output_capacity(2),
            Err(CheckpointMaterializationError::OriginalPayloadCapacity {
                family: "materialization outputs",
                required: 2,
                capacity: 1,
            })
        ));
        owner.retain_output(input.clone()).unwrap();
        assert!(matches!(
            owner.retain_output(input.clone()),
            Err(CheckpointMaterializationError::OriginalPayloadCapacity { .. })
        ));
        assert_eq!(owner.outputs().as_ptr(), pointer);
        assert_eq!(owner.outputs().len(), 1);
        assert_eq!(
            owner.outputs()[0]
                .completed_in_original_scope(observer)
                .unwrap()
                .as_slice::<f32>(),
            &[7.0, -2.0]
        );
        owner.finish().unwrap();
    });
}

#[test]
fn prepared_materialization_submits_finite_roots_and_keeps_completion_order() {
    let Some(layout) = traversal(2) else { return };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let first = Array::from_slice(&[1.25f32, -3.5], &[2]);
    let second = Array::from_slice(&[19u32, 42], &[2]);
    first.evaluated().unwrap();
    second.evaluated().unwrap();
    with_materialization(2, |mut owner, _| {
        owner.prepare_output_capacity(2).unwrap();
        owner.retain_output(first.clone()).unwrap();
        owner.retain_output(second.clone()).unwrap();
        let pointer = owner.outputs().as_ptr();
        let owner = owner
            .submit_prepared_outputs_with_traversal(&stream, &layout)
            .unwrap();
        owner.wait().unwrap();
        assert_eq!(owner.outputs().as_ptr(), pointer);
        let mut outputs = owner.completed_outputs();
        assert_eq!(
            outputs.next().unwrap().unwrap().as_slice::<f32>(),
            &[1.25, -3.5]
        );
        assert_eq!(
            outputs.next().unwrap().unwrap().as_slice::<u32>(),
            &[19, 42]
        );
        assert!(outputs.next().is_none());
        drop(outputs);
        owner.finish().unwrap();
    });
}

#[test]
fn ordinary_materialization_cannot_claim_a_prepared_original_traversal() {
    let Some(layout) = traversal(1) else { return };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut owner = WeightMaterialization::prepare_retained(Vec::new(), Vec::new()).unwrap();
    owner
        .retain_output(Array::from_slice(&[5i32], &[1]))
        .unwrap();
    assert!(matches!(
        owner.submit_prepared_outputs_with_traversal(&stream, &layout),
        Err(CheckpointMaterializationError::OriginalOperationDomain)
    ));
}
