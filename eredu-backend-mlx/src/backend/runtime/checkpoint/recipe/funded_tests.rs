use super::*;
use crate::backend::{
    managed_memory,
    nn::workspace::{
        MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms,
        ResidentExecutionMechanisms,
    },
    runtime::residency::storage::StorageIdentity,
    submission_recovery::{PreparedRecovery, Retention, Status},
};
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    HostPreparationAuthority, InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand,
    StateMemoryLayout, WorkspaceBound,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceFloatingType,
    WorkspaceRepresentation, WorkspaceTensor,
};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, MemoryLedger, StorageMetadataFunding, WorkingMemoryFundingRun,
    WorkingMemoryFundingScope, WorkingMemoryReservation,
};
use safemlx::{Device, DeviceType, OperationEvent};
use std::num::NonZeroU8;

struct Resources {
    inputs: [Array; 2],
    output: Option<Array>,
    completion: Option<OperationEvent>,
}
impl Retention for Resources {
    fn observe(&self, _: Status) {}
}

fn admission(bytes: u64) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 2,
        max_output_tokens: 0,
        prefill_chunk_positions: 2,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "prepared recipe ordinary sources");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(2),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        },
    ))
    .unwrap();
    crate::memory_fixture::host_admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 2,
        state,
        incremental_required_bytes: Some(bytes),
    })
}
fn observer_bytes() -> u64 {
    u64::try_from(
        StorageMetadataFunding::host_owner_bytes(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
    .checked_add(MemoryLedger::storage_metadata_control_bytes().unwrap())
    .unwrap()
    .checked_add(WorkingMemoryFundingScope::allocation_funding_control_bytes().unwrap())
    .unwrap()
}
fn fund(pool: &MemoryLedger, bytes: u64) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &admission(bytes),
        pool.configured_limits().clone(),
    )
    .unwrap()
    .into_funding()
    .unwrap()
}
fn reclaim(stream: &Stream) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::reap();
    safemlx::try_retire_completed_submissions().unwrap();
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    crate::backend::ordinary_retirement::reclaim_all();
}
fn charge(pool: &MemoryLedger) -> u64 {
    pool.snapshot()
        .unwrap()
        .domains
        .iter()
        .map(|domain| domain.current_charge_bytes)
        .sum()
}

#[test]
fn funded_prepared_recipe_uses_quoted_wrappers_and_retires_after_completion() {
    if !crate::tests::support::native_process::enter("funded-prepared-recipe") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let old_cache = safemlx::memory::set_cache_limit(0).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let materialization = MlxParameterMaterializationContext::new(&stream, &stream);
    let native = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(
        native.allocation(),
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
    )
    .ordinary_storage();
    let context = WorkspaceContext::new(cpu);
    let source = |key| DerivedWeightRecipe::source(key, TensorSelection::Full);
    let recipe = DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![
            DerivedWeightRecipe::NegLog {
                input: Box::new(source("rates")),
            },
            DerivedWeightRecipe::SubtractOne {
                input: Box::new(source("offsets")),
            },
        ],
    };
    let sources = [0, 1].map(|_| {
        let storage = WorkspaceExistingStorage::try_new(
            Some(native.allocation().buffer_capacity(8).unwrap()),
            &context,
        )
        .unwrap();
        WorkspaceTensor::existing_with_storage(
            context
                .layout(&[2], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            &storage,
            &context,
        )
        .unwrap()
    });
    context.begin_state_span(sources.iter()).unwrap();
    let mut ordinal = 0;
    let trace = trace_ordinary_recipe(&recipe, &context, |key, selection, actual| {
        assert_eq!(key, ["rates", "offsets"][ordinal]);
        assert_eq!(selection, &TensorSelection::Full);
        assert!(context.shares_trace(actual));
        let result = sources[ordinal].clone();
        ordinal += 1;
        Ok(result)
    })
    .unwrap();
    assert_eq!(ordinal, 2);
    let (wrapper_metadata, wrapper_observed) = trace.ordinary_wrapper_population();
    let numerical = trace
        .finish_native_population(
            ResidentExecutionMechanisms::Cpu {
                ordinary: native,
                cpu,
            },
            &context,
        )
        .unwrap();
    let mut observed = numerical
        .ordinary_cpu_controls()
        .unwrap()
        .append(wrapper_observed)
        .unwrap();
    // This worker submits the one final output through the C ArrayVector.
    // Scalar validation reads use core Eval directly and have no such caller.
    observed
        .include(OperationEvent::ordinary_array_vector_control_layout(1).unwrap())
        .unwrap();
    let metadata = [
        usize::try_from(wrapper_metadata).unwrap(),
        prepared_recipe_entry_control_bytes().unwrap(),
        WeightMaterialization::prepared_validation_control_bytes()
            .unwrap()
            .checked_mul(trace.validations.len())
            .unwrap(),
        usize::try_from(
            PreparedRecovery::<Resources, HostPreparationAuthority>::control_bytes().unwrap(),
        )
        .unwrap(),
        OperationEvent::ordinary_submission_wrapper_control_bytes().unwrap(),
        Array::ordinary_clone_control_bytes()
            .unwrap()
            .checked_mul(3)
            .unwrap(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .unwrap();
    let paid_host =
        u64::try_from(StorageMetadataFunding::host_owner_bytes(metadata).unwrap()).unwrap();
    let backing_controls = u64::try_from(safemlx::physical_backing_control_bytes())
        .unwrap()
        .checked_add(managed_memory::ordinary_root_metadata_bytes().unwrap())
        .unwrap()
        .checked_mul(u64::try_from(numerical.storage.maximum_births()).unwrap())
        .unwrap();
    let allowance = [
        observer_bytes(),
        paid_host,
        numerical.storage.mutable_bytes(),
        backing_controls,
        observed.host_allowance_with_ledger_metadata().unwrap(),
    ]
    .into_iter()
    .try_fold(0u64, u64::checked_add)
    .unwrap();

    // These actual prepared leaves belong to the caller before admission.
    let inputs = [
        Array::try_from_slice(&[-1.0f32, -4.0], &[2]).unwrap(),
        Array::try_from_slice(&[4.0f32, 9.0], &[2]).unwrap(),
    ];
    let retained_inputs = inputs
        .each_ref()
        .map(|value| value.try_clone_handle().unwrap());
    reclaim(&stream);
    let baseline = charge(&pool);
    let accounts = pool.snapshot().unwrap().funding_accounts;
    let (reservation, run) = fund(&pool, allowance);
    let mut scope = run.scope().unwrap();
    let funding = scope.prepare_storage_metadata().unwrap();
    let observer_host = funding
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    let payer = managed_memory::prepare_scoped_observer(&mut scope, observer_host).unwrap();
    let host = funding.prepare_host_owner(metadata).unwrap();
    let resources = Resources {
        inputs: retained_inputs,
        output: None,
        completion: None,
    };
    let prepared = PreparedRecovery::new(resources, host.clone())
        .unwrap_or_else(|error| panic!("prepared recipe recovery: {:?}", error.cause));
    let mut recovery = prepared
        .try_begin()
        .unwrap_or_else(|error| panic!("begin recipe recovery: {:?}", error.cause));
    recovery
        .configure_scope(|native| native.bind_physical_observer(&payer))
        .unwrap();
    assert!(OriginalScopeObserver::try_current().unwrap().is_none());
    let resources = recovery.retention_mut();
    let mut calls = 0;
    let pending = {
        let mut leaves = |key: &str, selection: &TensorSelection, actual: &Stream| {
            assert_eq!(key, ["rates", "offsets"][calls]);
            assert_eq!(selection, &TensorSelection::Full);
            assert_eq!(actual, &stream);
            let value = resources.inputs[calls].try_clone_handle()?;
            calls += 1;
            Ok(value)
        };
        prepare_ordinary_recipe_from_funded_leaves(&recipe, &materialization, &mut leaves, &host)
            .unwrap()
    };
    assert_eq!(calls, 2);
    let (output, pending_sources) = pending.into_parts();
    assert!(pending_sources.is_empty());
    resources.output = Some(output);
    resources.completion = Some(
        safemlx::transforms::async_eval_with_operation_event(resources.output.iter()).unwrap(),
    );
    resources
        .completion
        .as_ref()
        .unwrap()
        .synchronize()
        .unwrap();
    drop(resources.output.as_ref().unwrap().evaluated().unwrap());
    let alias = resources
        .output
        .as_ref()
        .unwrap()
        .try_clone_handle()
        .unwrap();
    let identity = StorageIdentity::Native(alias.allocation_info().unwrap().unwrap().identity());
    recovery.seal();
    let status = recovery.finish().unwrap();
    assert!(status.settled && !status.failed && !status.blocked);
    drop((payer, funding));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert!(pool.registered_allocation(&identity).unwrap().is_some());
    assert!(charge(&pool) > baseline);
    for (actual, expected) in
        alias
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .zip([0.0, 4.0f32.ln(), 3.0, 8.0])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    // The caller retains the prepaid C-wrapper custody through the last alias.
    drop((alias, host));
    reclaim(&stream);
    assert!(pool.registered_allocation(&identity).unwrap().is_none());
    assert_eq!(charge(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);

    // One byte short for the exact host producer refuses before any leaf or
    // recipe worker can run, despite spare global ledger capacity.
    let account_controls = MemoryLedger::storage_metadata_control_bytes().unwrap();
    let (reservation, run) = fund(&pool, paid_host.checked_add(account_controls).unwrap() - 1);
    let scope = run.scope().unwrap();
    let funding = scope.prepare_storage_metadata().unwrap();
    assert!(matches!(
        funding.prepare_host_owner(metadata),
        Err(eredu_core::HostMetadataFundingError::DomainAllowance { .. })
    ));
    drop(funding);
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert_eq!(charge(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
    safemlx::memory::set_cache_limit(old_cache).unwrap();
}
