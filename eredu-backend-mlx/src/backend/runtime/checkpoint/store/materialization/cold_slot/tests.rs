use super::*;
use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::{
    bounded_quantization::{submit_original_affine_tile, BoundedQuantizationTarget},
    store::{CheckpointMaterializationError, PreparedEncodedInputPlan, WeightMaterialization},
};
use crate::backend::submission_recovery::native_role::{cold, NativeRoleCapacity};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::{MemoryWeightStore, TensorSelection},
    AffineQuantization,
};
use safemlx::{
    CpuAffineQuantizeSubmissionLayout, Device, DeviceType, Dtype, OperationEvent,
    OriginalScopeObserver, PrefillRootsRuntime, PreparedInputRuntime, PreparedOriginalBufferBudget,
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    PreparedSubmissionScopeOwner, Stream, SubmissionRetirement, SubmissionScope,
};

fn shape(inputs: usize) -> MaterializationPayloadShape {
    MaterializationPayloadShape {
        inputs,
        outputs: 3,
        pending_sources: 0,
    }
}

#[test]
fn cold_slot_comparison_checkout_and_custody_precede_any_request() {
    let shape = shape(1);
    let bytes = ColdMaterializationSlot::required_bytes(shape).unwrap();
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let error = ColdMaterializationSlot::prepare(&short, shape).unwrap_err();
    assert!(
        matches!(error.0.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == bytes && *available_bytes == bytes - 1)
    );
    assert!(error.0.rejected_plan().is_some());
    assert!(error.0.constructor_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), 0);
    drop(error);

    let exact = WorkingMemoryPool::new(bytes, 0).unwrap();
    let mut slot = ColdMaterializationSlot::prepare(&exact, shape).unwrap();
    assert!(matches!(
        slot.take(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let ready = slot.take(&exact).unwrap();
    assert!(matches!(
        slot.take(&exact),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(slot);
    assert_eq!(exact.used_bytes().unwrap(), bytes);
    drop(ready);
    assert_eq!(exact.used_bytes().unwrap(), bytes);
    crate::backend::submission_recovery::wait_for_retirement(|| exact.used_bytes() == Ok(0));
    assert_eq!(exact.used_bytes().unwrap(), 0);
    assert!(matches!(
        ColdMaterializationSlot::required_bytes(shape_with_overflow()),
        Err(WorkingMemoryError::Overflow)
    ));
}

fn shape_with_overflow() -> MaterializationPayloadShape {
    MaterializationPayloadShape {
        inputs: usize::MAX,
        outputs: 3,
        pending_sources: 0,
    }
}

fn layout() -> Option<CpuAffineQuantizeSubmissionLayout> {
    let layout = OperationEvent::cpu_affine_quantize_submission_layout(
        Dtype::Float32,
        Dtype::Float16,
        2,
        2,
        64,
        32,
        4,
    );
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(layout.is_some());
    }
    layout
}

fn with_scope(
    layout: CpuAffineQuantizeSubmissionLayout,
    runtime: &PreparedInputRuntime,
    operation: impl FnOnce(&OriginalScopeObserver),
) {
    // These actual native capacities are component fixtures. No text request
    // guard or inferred model admission is used to construct the cold slot.
    let budget = PreparedOriginalBufferBudget::try_new(
        runtime,
        layout.physical_capacity(runtime).unwrap(),
        (),
    )
    .unwrap()
    .try_allocate()
    .unwrap();
    let graph = PreparedSubmissionGraphQuota::try_new(layout.graph_capacity(), ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(layout.record_capacity(), ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let failure = PreparedPrefillFailure::try_new(())
        .unwrap()
        .try_allocate()
        .unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_graph_quota(graph.clone())
            .with_record_quota(records.clone()),
    )
    .unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    let observer = OriginalScopeObserver::require_current().unwrap();
    operation(&observer);
    scope.seal();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let (_, status) = observer.progress().unwrap();
        assert!(!status.failed() && !status.blocked());
        if status.is_settled()
            && observer.retire_completed_records().unwrap()
                == SubmissionRetirement::CompleteSnapshot
        {
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop((observer, scope, failure));
    safemlx::reclaim_allocation_owners();
    assert_eq!(budget.occupied_bytes(), 0);
    assert_eq!(graph.occupied_bytes(), 0);
    assert_eq!(records.occupied_bytes(), 0);
}

#[test]
fn cold_slot_runs_encoded_affine_conversion_without_a_text_request() {
    let Some(layout) = layout() else { return };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let bytes = (0..128)
        .flat_map(|index| ((index % 16) as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let source = MemoryWeightStore::from_safetensors([(
        "weight".into(),
        safetensors::tensor::Dtype::F32,
        vec![2, 64],
        bytes,
    )])
    .unwrap();
    let read = DerivedWeightRecipe::source("weight", TensorSelection::Full)
        .prepare_encoded_read(&source)
        .unwrap()
        .unwrap();
    let input_plan =
        PreparedEncodedInputPlan::new(&read, &runtime, &[2, 64], Dtype::Float32).unwrap();
    let input_bytes = input_plan.required_bytes().unwrap();
    let slot_bytes = ColdMaterializationSlot::required_bytes(shape(1)).unwrap();
    let pool_cell = std::cell::OnceCell::<WorkingMemoryPool>::new();
    let role_bytes = std::cell::Cell::new(0);
    let target = BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
        .unwrap()
        .with_affine_companion_dtype(RecipeDtype::F16)
        .unwrap();
    let capacity = NativeRoleCapacity {
        graph: layout.graph_capacity(),
        records: layout.record_capacity(),
        backing: layout.physical_capacity(&runtime).unwrap(),
    };
    let plan = cold::Plan::new(&runtime, capacity, None, (), |_, context| {
        let pool = pool_cell.get().unwrap();
        let observer = context.observer();
        let mut slot = ColdMaterializationSlot::prepare(pool, shape(1)).unwrap();
        let ready = slot.take(&pool).unwrap();
        drop(slot);
        let mut owner = WeightMaterialization::prepare_original_slot(ready, observer).unwrap();
        owner.prepare_input_capacity(1).unwrap();
        let pointer = owner.inputs().as_ptr();
        let input = input_plan.prepare(&pool).unwrap();
        owner
            .retain_input(input.output().try_prepared_source_array().unwrap())
            .unwrap();
        drop(input);
        assert_eq!(
            pool.used_bytes().unwrap(),
            role_bytes.get() + input_bytes + slot_bytes
        );
        assert_eq!(owner.inputs().as_ptr(), pointer);
        let owner = submit_original_affine_tile(
            owner,
            AffineQuantization::new(32, 4).unwrap(),
            &target,
            &stream,
            layout,
        )
        .unwrap();
        owner.wait().unwrap();
        let mut outputs = owner.completed_outputs();
        let weights = outputs.next().unwrap().unwrap();
        assert_eq!(weights.as_slice::<u32>().len(), 16);
        for (index, &word) in weights.as_slice::<u32>().iter().enumerate() {
            assert_eq!(
                word,
                if index % 2 == 0 {
                    0x89abcdef
                } else {
                    0x01234567
                }
            );
        }
        for expected in [-1.0, 15.0] {
            let view = outputs.next().unwrap().unwrap();
            assert_eq!(
                view.as_slice::<half::f16>(),
                &[half::f16::from_f32(expected); 4]
            );
        }
        assert!(outputs.next().is_none());
        drop((outputs, weights));
        Ok(Ok::<_, Error>(owner))
    });
    role_bytes.set(plan.required_bytes().unwrap());
    pool_cell
        .set(WorkingMemoryPool::new(role_bytes.get() + input_bytes + slot_bytes, 0).unwrap())
        .unwrap();
    let pool = pool_cell.get().unwrap();
    let owner = plan.execute(pool).unwrap().unwrap();
    // Completed outputs escape the role constructor. Their native owners and
    // input aliases still retain both source accounts and the physical budget.
    assert_eq!(
        pool.used_bytes().unwrap(),
        role_bytes.get() + input_bytes + slot_bytes
    );
    owner.finish().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.used_bytes() == Ok(0)
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn cold_slot_refuses_input_capacity_before_a_source_is_created() {
    let Some(layout) = layout() else { return };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let bytes = ColdMaterializationSlot::required_bytes(shape(0)).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let mut slot = ColdMaterializationSlot::prepare(&pool, shape(0)).unwrap();
    with_scope(layout, &runtime, |observer| {
        let ready = slot.take(&pool).unwrap();
        drop(slot);
        let mut owner = WeightMaterialization::prepare_original_slot(ready, observer).unwrap();
        assert!(matches!(
            owner.prepare_input_capacity(1),
            Err(CheckpointMaterializationError::OriginalPayloadCapacity {
                family: "materialization inputs",
                required: 1,
                capacity: 0
            })
        ));
        assert!(owner.inputs().is_empty());
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        owner.finish().unwrap();
    });
    crate::backend::submission_recovery::wait_for_retirement(|| pool.used_bytes() == Ok(0));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
