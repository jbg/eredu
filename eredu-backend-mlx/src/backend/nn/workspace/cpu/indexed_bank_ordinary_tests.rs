use super::*;
use crate::backend::managed_memory;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_nn::Tensor;
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, MemoryLedger, StorageMetadataFunding, WorkingMemoryFundingRun,
    WorkingMemoryFundingScope, WorkingMemoryReservation,
};
use safemlx::{Stream, SubmissionScope};
use std::num::NonZeroU8;

#[path = "ordinary_communication_tests.rs"]
mod ordinary_communication_tests;

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
    let bound = |bytes| WorkspaceBound::bounded(bytes, "ordinary indexed CPU allocation sources");
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
fn funding(
    pool: &MemoryLedger,
    native: u64,
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    let quote = admission(native.checked_add(observer_bytes()).unwrap());
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &quote,
        pool.configured_limits().clone(),
    )
    .unwrap()
    .into_funding()
    .unwrap()
}
fn observer(scope: &mut WorkingMemoryFundingScope) -> safemlx::ScopedPhysicalBackingObserver {
    let metadata = scope.prepare_storage_metadata().unwrap();
    let host = metadata
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    managed_memory::prepare_scoped_observer(scope, host).unwrap()
}
fn reclaim(stream: &Stream) {
    stream.synchronize().unwrap();
    safemlx::try_retire_completed_submissions().unwrap();
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    crate::backend::ordinary_retirement::reclaim_all();
}
fn current(pool: &MemoryLedger) -> u64 {
    pool.snapshot()
        .unwrap()
        .domains
        .iter()
        .map(|d| d.current_charge_bytes)
        .sum()
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(value) = error.downcast_ref::<T>() {
            return Some(value);
        }
        error = error.source()?;
    }
}

#[test]
fn ordinary_indexed_registration_refusal_retains_prepaid_host_custody() {
    if !crate::tests::support::native_process::enter("ordinary-indexed-registration-custody") {
        return;
    }
    use crate::backend::{
        nn::shared::{
            current_ordinary_execution_owner, OrdinaryExecutionOwner, OrdinaryExecutionRegistration,
        },
        runtime::residency::parameter_bank::{
            OrdinaryIndexedOccurrence, OrdinaryIndexedRequestOwner, OrdinaryIndexedRequestProgram,
        },
    };
    struct EmptyProgram;
    impl OrdinaryIndexedRequestProgram for EmptyProgram {
        fn len(&self) -> usize {
            0
        }
        fn occurrence(&self, _: usize) -> Option<OrdinaryIndexedOccurrence<'_>> {
            None
        }
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    reclaim(&stream);
    let baseline = current(&pool);
    let accounts = pool.snapshot().unwrap().funding_accounts;
    // An empty genuine request exercises registration without adding a bank
    // or numerical producer to this host-owner retirement fixture.
    let program = std::rc::Rc::new(EmptyProgram);
    let wrappers = OrdinaryExecutionRegistration::control_bytes().unwrap();
    let paid = StorageMetadataFunding::host_owner_bytes(wrappers)
        .unwrap()
        .checked_add(OrdinaryIndexedRequestOwner::runtime_control_bytes(&*program).unwrap())
        .unwrap();
    let (reservation, run) = funding(&pool, u64::try_from(paid).unwrap());
    let mut scope = run.scope().unwrap();
    let metadata = scope.prepare_storage_metadata().unwrap();
    let host = metadata.prepare_host_owner(wrappers).unwrap();
    let observer_host = metadata
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    let payer = managed_memory::prepare_scoped_observer(&mut scope, observer_host).unwrap();
    let mut indexed = OrdinaryIndexedRequestOwner::new(program, metadata.funding()).unwrap();
    let mut native = SubmissionScope::begin().unwrap();
    native.bind_physical_observer(&payer).unwrap();
    let registration = OrdinaryExecutionRegistration::new(OrdinaryExecutionOwner::new(
        host.clone(),
        payer.clone(),
    ))
    .unwrap();
    registration.bind_indexed(indexed.local_source()).unwrap();
    assert!(current_ordinary_execution_owner()
        .unwrap()
        .unwrap()
        .indexed_local()
        .is_some());
    let failure = registration
        .bind_indexed(indexed.local_source())
        .unwrap_err();
    assert!(std::error::Error::source(&failure).is_some());
    native.seal();
    assert!(native.progress().is_settled());
    indexed.finish().unwrap();
    drop((indexed, registration, native, payer, host, metadata));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert!(
        current(&pool) > baseline,
        "escaped refusal retains its actual prepaid owner"
    );
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts + 1);
    assert!(failure.to_string().contains("duplicate execution owners"));
    drop(failure);
    reclaim(&stream);
    assert_eq!(current(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
}

#[test]
fn ordinary_validation_error_retains_prepaid_host_custody() {
    if !crate::tests::support::native_process::enter("ordinary-validation-error-custody") {
        return;
    }
    use crate::backend::nn::{
        shared::{OrdinaryExecutionOwner, OrdinaryExecutionRegistration},
        tensor::{
            active_token_validation_arrays, ordinary_validation_completion_controls,
            validate_active_token_validations, validate_token_domain, TokenValidationScope,
        },
    };
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let old_cache = safemlx::memory::set_cache_limit(0).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    // This fixture isolates the completed predicate reader. Its genuine
    // producer and native reduction complete before the funded reader begins.
    let validations = TokenValidationScope::begin().unwrap();
    let tokens = Array::try_from_slice(&[3i32, 6], &[2]).unwrap();
    let normalized = validate_token_domain(&tokens, 4, None, &stream).unwrap();
    let roots = active_token_validation_arrays();
    safemlx::transforms::async_eval_with_event(roots.iter())
        .unwrap()
        .synchronize()
        .unwrap();
    for root in &roots {
        root.evaluated().unwrap();
    }
    assert_eq!(roots.len(), 1);
    reclaim(&stream);
    let baseline = current(&pool);
    let accounts = pool.snapshot().unwrap().funding_accounts;
    let wrappers = ordinary_validation_completion_controls(roots.len())
        .unwrap()
        .checked_add(OrdinaryExecutionRegistration::control_bytes().unwrap())
        .unwrap();
    let paid = u64::try_from(StorageMetadataFunding::host_owner_bytes(wrappers).unwrap()).unwrap();
    let (reservation, run) = funding(&pool, paid);
    let mut scope = run.scope().unwrap();
    let metadata = scope.prepare_storage_metadata().unwrap();
    let host = metadata.prepare_host_owner(wrappers).unwrap();
    let observer_host = metadata
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    let payer = managed_memory::prepare_scoped_observer(&mut scope, observer_host).unwrap();
    let mut native = SubmissionScope::begin().unwrap();
    native.bind_physical_observer(&payer).unwrap();
    let registration = OrdinaryExecutionRegistration::new(OrdinaryExecutionOwner::new(
        host.clone(),
        payer.clone(),
    ))
    .unwrap();
    let failure = validate_active_token_validations().unwrap_err();
    assert!(std::error::Error::source(&failure).is_some());
    native.seal();
    assert!(native.progress().is_settled());
    drop(registration);
    drop((native, payer, host, metadata));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    let validations = validations.finish();
    reclaim(&stream);
    assert!(
        current(&pool) > baseline,
        "escaped failure retains its prepaid host owner"
    );
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts + 1);
    assert!(failure.to_string().contains("token ID is outside 0..4"));
    drop(failure);
    reclaim(&stream);
    assert_eq!(current(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
    drop((validations, roots, tokens, normalized));
    reclaim(&stream);
    safemlx::memory::set_cache_limit(old_cache).unwrap();
}

#[test]
fn ordinary_grouped_chunk_table_refusal_retains_prepaid_host_custody() {
    if !crate::tests::support::native_process::enter("ordinary-grouped-table-custody") {
        return;
    }
    use crate::backend::nn::{
        shared::{OrdinaryExecutionOwner, OrdinaryExecutionRegistration},
        tensor::GroupedChunkOutputs,
    };
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let old_cache = safemlx::memory::set_cache_limit(0).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let input = Array::try_from_slice(&[1.25f32, -0.5], &[1, 2]).unwrap();
    input.evaluated().unwrap();
    reclaim(&stream);
    let baseline = current(&pool);
    let accounts = pool.snapshot().unwrap().funding_accounts;
    let wrappers = GroupedChunkOutputs::ordinary_control_bytes(3)
        .unwrap()
        .checked_add(OrdinaryExecutionRegistration::control_bytes().unwrap())
        .unwrap()
        .checked_add(
            Array::ordinary_clone_control_bytes()
                .unwrap()
                .checked_mul(4)
                .unwrap(),
        )
        .unwrap();
    let paid = u64::try_from(StorageMetadataFunding::host_owner_bytes(wrappers).unwrap()).unwrap();
    let (reservation, run) = funding(&pool, paid);
    let mut scope = run.scope().unwrap();
    let metadata = scope.prepare_storage_metadata().unwrap();
    let host = metadata.prepare_host_owner(wrappers).unwrap();
    let observer_host = metadata
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    let payer = managed_memory::prepare_scoped_observer(&mut scope, observer_host).unwrap();
    let mut native = SubmissionScope::begin().unwrap();
    native.bind_physical_observer(&payer).unwrap();
    let registration = OrdinaryExecutionRegistration::new(OrdinaryExecutionOwner::new(
        host.clone(),
        payer.clone(),
    ))
    .unwrap();
    let mut table = GroupedChunkOutputs::prepare(3).unwrap();
    for _ in 0..3 {
        table.push(input.clone()).unwrap();
    }
    let failure = table.push(input.clone()).unwrap_err();
    assert_eq!(table.as_slice().len(), 3);
    assert!(failure
        .to_string()
        .contains("grouped output table exhausted"));
    assert!(std::error::Error::source(&failure).is_some());
    native.seal();
    assert!(native.progress().is_settled());
    drop(registration);
    drop((native, payer, host, metadata));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert!(current(&pool) > baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts + 1);
    assert_eq!(
        table.as_slice()[0].evaluated().unwrap().as_slice::<f32>(),
        &[1.25, -0.5]
    );
    drop(table);
    reclaim(&stream);
    assert!(
        current(&pool) > baseline,
        "refusal keeps its prepaid host owner"
    );
    drop(failure);
    reclaim(&stream);
    assert_eq!(current(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
    drop(input);
    reclaim(&stream);
    safemlx::memory::set_cache_limit(old_cache).unwrap();
}

#[test]
fn ordinary_indexed_lazy_routing_uses_combined_allowance_and_retires_final_alias() {
    if !crate::tests::support::native_process::enter("ordinary-indexed-lazy-routing") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let old_cache = safemlx::memory::set_cache_limit(0).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(
        ordinary.allocation(),
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
    )
    .ordinary_storage();
    let context = WorkspaceContext::new(cpu);
    let left = WorkspaceTensor::existing(
        context.layout(&[2, 2], WorkspaceDtype::Int32).unwrap(),
        &context,
    )
    .unwrap();
    let right = WorkspaceTensor::existing(
        context.layout(&[2, 2], WorkspaceDtype::Int32).unwrap(),
        &context,
    )
    .unwrap();
    context.begin_state_span([&left, &right]).unwrap();
    let routing = left.add(&right, &context).unwrap();
    let report = context.finish_report(&[routing]).unwrap();
    let recipe =
        SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context)
            .unwrap();
    let plan = eredu_runtime::expert::AddressableChunkPlan::new(
        2,
        2,
        4,
        eredu_runtime::ParameterBankAccess::Bulk,
        Some(4),
        64,
    )
    .unwrap();
    let first = eredu_runtime::expert::AddressableChunkCensus::new(
        0,
        0,
        plan,
        0,
        eredu_runtime::ParameterBankAccess::Bulk,
    )
    .unwrap();
    let indexed = cpu
        .ordinary_indexed_numerical_facts(first, WorkspaceDtype::Int32)
        .unwrap();
    let calls = cpu.ordinary_report_call_controls(&report).unwrap().unwrap();
    use crate::backend::nn::shared::{
        current_ordinary_execution_owner, MlxNeuralBackend, OrdinaryExecutionOwner,
        OrdinaryExecutionRegistration,
    };
    use eredu_core::Completion;
    use eredu_runtime::SubmissionBackend;
    let submission = MlxNeuralBackend::ordinary_submission_call_controls(1).unwrap();
    let consumer = MlxNeuralBackend::ordinary_consumer_call_controls().unwrap();
    let mut wait = OrdinaryNativeControls::default();
    wait.include(safemlx::OperationEvent::ordinary_cpu_wait_control_layout().unwrap())
        .unwrap();
    let controls = recipe
        .ordinary_cpu_controls_with(
            indexed.raw_population,
            [(2, indexed.discovery_completions), (1, 1)],
        )
        .unwrap()
        .append(indexed.caller_native_controls)
        .unwrap()
        .append(calls.observed)
        .unwrap()
        .append(submission.observed)
        .unwrap()
        .append(wait)
        .unwrap();
    use safemlx::ops::OrdinaryRecipeCall as Call;
    let wrapper = |call: Call| call.control_bytes().unwrap().metadata_bytes();
    let wrapper_bytes = OrdinaryExecutionRegistration::control_bytes()
        .unwrap()
        .checked_add(usize::try_from(submission.metadata_bytes).unwrap())
        .unwrap()
        .checked_add(usize::try_from(consumer.metadata_bytes).unwrap())
        .unwrap()
        .checked_add(usize::try_from(indexed.fixed_host_controls).unwrap())
        .unwrap()
        .checked_add(usize::try_from(calls.metadata_bytes).unwrap())
        .unwrap()
        .checked_add(wrapper(Call::Evaluate { inputs: 2 }))
        .unwrap()
        .checked_add(wrapper(Call::Evaluate { inputs: 1 }))
        .unwrap()
        .checked_add(wrapper(Call::BorrowedEvaluation).checked_mul(5).unwrap())
        .unwrap()
        .checked_add(
            safemlx::EvaluatedArray::iteration_control_bytes::<i32>()
                .unwrap()
                .checked_mul(5)
                .unwrap(),
        )
        .unwrap()
        .checked_add(Array::ordinary_clone_control_bytes().unwrap())
        .unwrap();
    let paid_wrapper =
        u64::try_from(StorageMetadataFunding::host_owner_bytes(wrapper_bytes).unwrap()).unwrap();
    let births = recipe
        .storage
        .maximum_births()
        .checked_add(indexed.backing_births)
        .unwrap();
    let payload = recipe
        .storage
        .mutable_bytes()
        .checked_add(indexed.backing_bytes)
        .unwrap();
    let payload_controls = u64::try_from(safemlx::physical_backing_control_bytes())
        .unwrap()
        .checked_add(managed_memory::ordinary_root_metadata_bytes().unwrap())
        .unwrap()
        .checked_mul(u64::try_from(births).unwrap())
        .unwrap();
    let allowance = controls
        .host_allowance_with_ledger_metadata()
        .unwrap()
        .checked_add(payload)
        .unwrap()
        .checked_add(payload_controls)
        .unwrap()
        .checked_add(paid_wrapper)
        .unwrap();

    // Inputs and the stream belong to the caller before admission. The actual
    // routing Add stays lazy until discovery evaluates its two result roots.
    let left = Array::try_from_slice(&[0i32, 1, 0, 1], &[2, 2]).unwrap();
    let right = Array::try_from_slice(&[2i32, -1, 2, -1], &[2, 2]).unwrap();
    reclaim(&stream);
    let baseline = current(&pool);
    let accounts = pool.snapshot().unwrap().funding_accounts;
    let (reservation, run) = funding(&pool, allowance);
    let mut scope = run.scope().unwrap();
    let metadata = scope.prepare_storage_metadata().unwrap();
    let host = metadata.prepare_host_owner(wrapper_bytes).unwrap();
    let observer_host = metadata
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    let payer = managed_memory::prepare_scoped_observer(&mut scope, observer_host).unwrap();
    let mut native = SubmissionScope::begin().unwrap();
    native.bind_physical_observer(&payer).unwrap();
    assert!(
        current_ordinary_execution_owner().is_err(),
        "a missing matching source cannot use legacy recovery"
    );
    let registration = OrdinaryExecutionRegistration::new(OrdinaryExecutionOwner::new(
        host.clone(),
        payer.clone(),
    ))
    .unwrap();
    assert!(current_ordinary_execution_owner()
        .unwrap()
        .unwrap()
        .observer()
        .is_current());
    let routing = left.add(&right, &stream).unwrap();
    assert!(
        routing.allocation_info().unwrap().is_none(),
        "routing must still be lazy"
    );
    let discovered =
        indexed_numerical::discover(&mut indexed_numerical::Native(&stream), &routing, 4).unwrap();
    safemlx::transforms::eval([&discovered.histogram, &discovered.invalid]).unwrap();
    assert_eq!(
        discovered.histogram.evaluated().unwrap().as_slice::<i32>(),
        &[2, 0, 2, 0]
    );
    assert_eq!(
        discovered.invalid.evaluated().unwrap().as_slice::<i32>(),
        &[0]
    );
    // A scheduled intermediate is not an available host-read descriptor until
    // its own completed reader observes it. This read waits the completed
    // routing node; it does not submit a second upstream graph.
    assert_eq!(
        routing.evaluated().unwrap().as_slice::<i32>(),
        &[2, 0, 2, 0]
    );
    let output = indexed_numerical::remap(
        &mut indexed_numerical::Native(&stream),
        &routing,
        &[1, -1, 0, -1][..],
    )
    .unwrap();
    safemlx::transforms::eval([&output]).unwrap();
    assert_eq!(output.evaluated().unwrap().as_slice::<i32>(), &[0, 1, 0, 1]);
    let output = crate::MlxTensor::from_array(output);
    let completion = MlxNeuralBackend::submit(&stream, [&output]).unwrap();
    completion.wait_on(&stream).unwrap();
    completion.wait().unwrap();
    let alias = output.as_array().try_clone_handle().unwrap();
    let key = StorageIdentity::Native(alias.allocation_info().unwrap().unwrap().identity());
    assert!(pool.registered_allocation(&key).unwrap().is_some());
    drop(completion);
    native.seal();
    assert!(
        current_ordinary_execution_owner().unwrap().is_none(),
        "a retained registration cannot shadow a later native source"
    );
    assert!(native.progress().is_settled());
    drop(registration);
    drop((native, payer, discovered, routing, output, metadata));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert!(pool.registered_allocation(&key).unwrap().is_some());
    assert!(current(&pool) > baseline);
    assert_eq!(alias.evaluated().unwrap().as_slice::<i32>(), &[0, 1, 0, 1]);
    drop((alias, host));
    reclaim(&stream);
    assert!(pool.registered_allocation(&key).unwrap().is_none());
    assert_eq!(current(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);

    // A genuine admitted account with no native remainder refuses at the same
    // allocator callback despite the process ledger having spare capacity.
    let (reservation, run) = funding(&pool, 0);
    let mut scope = run.scope().unwrap();
    let payer = observer(&mut scope);
    let mut native = SubmissionScope::begin().unwrap();
    native.bind_physical_observer(&payer).unwrap();
    let refusal = left.add(&right, &stream).unwrap_err();
    assert!(
        matches!(
            cause::<eredu_core::HostMetadataFundingError>(&refusal),
            Some(eredu_core::HostMetadataFundingError::DomainAllowance { available: 0, .. })
        ),
        "{refusal:?}"
    );
    native.seal();
    assert!(native.status().is_settled());
    drop((native, payer, refusal));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert_eq!(current(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
    // The identical operation succeeds after leaving the refused source.
    assert_eq!(
        left.add(&right, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[2, 0, 2, 0]
    );
    safemlx::memory::set_cache_limit(old_cache).unwrap();
}

#[test]
fn ordinary_grouped_observer_error_retains_prepaid_host_custody() {
    if !crate::tests::support::native_process::enter("ordinary-grouped-observer-custody") {
        return;
    }
    use crate::backend::nn::shared::{
        MlxGroupedGatedProduct, OrdinaryExecutionOwner, OrdinaryExecutionRegistration,
    };
    use eredu_nn::{GroupedUnitBatch, GroupedUnitObserver};
    struct Replacement(Option<MlxTensor>);
    impl GroupedUnitObserver<MlxTensor> for Replacement {
        fn observe(&mut self, _: &GroupedUnitBatch<'_, MlxTensor>) -> Result<(), eredu_nn::Error> {
            Ok(())
        }
        fn intervene(
            &mut self,
            _: &GroupedUnitBatch<'_, MlxTensor>,
        ) -> Result<Option<MlxTensor>, eredu_nn::Error> {
            Ok(self.0.take())
        }
        fn observe_effective(
            &mut self,
            _: &GroupedUnitBatch<'_, MlxTensor>,
        ) -> Result<(), eredu_nn::Error> {
            panic!("invalid replacement must stop before the effective observation")
        }
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let old_cache = safemlx::memory::set_cache_limit(0).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let input = Array::try_from_slice(&[1.25f32, -0.5], &[1, 2]).unwrap();
    let ids = Array::try_from_slice(&[0i32], &[1]).unwrap();
    let weights = Array::try_from_slice(&[1.0f32], &[1, 1]).unwrap();
    let invalid =
        MlxTensor::from_array(Array::try_from_slice(&[2.5f32, 3.0], &[1, 1, 1, 1, 2]).unwrap());
    for value in [&input, &ids, &weights, invalid.as_array()] {
        value.evaluated().unwrap();
    }
    let invalid_alias = invalid.clone();
    reclaim(&stream);
    let baseline = current(&pool);
    let accounts = pool.snapshot().unwrap().funding_accounts;
    let wrappers = MlxGroupedGatedProduct::ordinary_unit_observation_control_bytes(2)
        .unwrap()
        .checked_add(OrdinaryExecutionRegistration::control_bytes().unwrap())
        .unwrap();
    let paid = u64::try_from(StorageMetadataFunding::host_owner_bytes(wrappers).unwrap()).unwrap();
    let (reservation, run) = funding(&pool, paid);
    let mut scope = run.scope().unwrap();
    let metadata = scope.prepare_storage_metadata().unwrap();
    let host = metadata.prepare_host_owner(wrappers).unwrap();
    let observer_host = metadata
        .prepare_host_owner(
            usize::try_from(managed_memory::scoped_observer_bytes().unwrap()).unwrap(),
        )
        .unwrap();
    let payer = managed_memory::prepare_scoped_observer(&mut scope, observer_host).unwrap();
    let mut native = SubmissionScope::begin().unwrap();
    native.bind_physical_observer(&payer).unwrap();
    let registration = OrdinaryExecutionRegistration::new(OrdinaryExecutionOwner::new(
        host.clone(),
        payer.clone(),
    ))
    .unwrap();
    let batch = GroupedUnitBatch {
        values: &input,
        group_indices: &ids,
        selection_indices: &ids,
        token_indices: &ids,
        coefficients: &weights,
        token_offset: 0,
        total_token_count: 1,
        group_count: 1,
    };
    let failure = MlxGroupedGatedProduct::test_ordinary_unit_observer(
        &mut Replacement(Some(invalid)),
        &batch,
    )
    .unwrap_err();
    assert!(std::error::Error::source(&failure).is_some());
    assert!(failure.to_string().contains("first 4 axes of rank 5"));
    native.seal();
    assert!(native.progress().is_settled());
    drop(registration);
    drop((native, payer, host, metadata));
    scope.certify().unwrap();
    run.close().unwrap();
    drop(reservation);
    reclaim(&stream);
    assert!(
        current(&pool) > baseline,
        "escaped observer failure retains the prepaid host account"
    );
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts + 1);
    drop(failure);
    reclaim(&stream);
    assert_eq!(current(&pool), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
    drop((input, ids, weights, invalid_alias));
    reclaim(&stream);
    safemlx::memory::set_cache_limit(old_cache).unwrap();
}
