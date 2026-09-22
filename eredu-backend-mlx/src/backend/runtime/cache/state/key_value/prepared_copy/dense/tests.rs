//! Copy-only fresh prompt preparation: actual dense P plus the isolated native
//! program N. These fixtures grant no model-forward or resumed-text permission.

use super::super::tests::{
    controls,
    funded::{admit, finish_native, metal, publish_source, sampler, settle},
    operands, state, values, FAIL_AFTER_LAYER,
};
use super::*;
use crate::backend::{
    managed_memory::NativeMemoryOwner,
    nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms},
    runtime::{cache::kv::ConcatKeyValueCache, residency::storage::RetainedStorage},
};
use crate::memory_fixture::LedgerFixture;
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig,
    SharedStorageAttachmentError, StateMemoryLayout, TextGenerationConfig, WorkspaceBound,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceIsolatedCopyPlan};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, InferenceTextPreparation,
    RegisteredWorkspaceCopy, RegisteredWorkspaceStorage, WorkingMemoryFundingRun,
    WorkingMemoryFundingScope, WorkingMemoryStorage,
};
use std::convert::Infallible;

mod binding;

struct CopyQuote<'a> {
    host: eredu_runtime::working_memory::RegisteredDenseDecoderInitialization<
        'a,
        MlxKeyValueLayerState,
        MlxKeyValueLayerState,
        StorageIdentity,
    >,
    bytes: u64,
    native_bytes: u64,
    complete: WorkingMemoryStorage<StorageIdentity>,
}

// The source descriptors come from the actual settled operand handles. The
// returned pins include shared layout, while dense_host_copy separately binds
// the actual table. This standard concat fixture has no independent compressed
// store backing hidden behind its logical operands.
fn quote<'a>(plan: &PreparedResidentKvCopy<'a>, pool: &MemoryLedger) -> CopyQuote<'a> {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::new(&context);
    let inputs = operands(plan)
        .into_iter()
        .map(|array| projection.project(array).unwrap())
        .collect::<Vec<_>>();
    let native = projection.into_storage();
    assert!(native.is_complete());
    let source = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        native
            .iter()
            .map(|(id, _, root)| crate::backend::nn::workspace::registered_storage_row(id, root)),
    )
    .unwrap();
    let program =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &inputs).unwrap();
    let native_bytes = program
        .incremental_bytes()
        .expect("complete Metal isolated-copy facts");
    let association = RegisteredWorkspaceCopy::bind(program, source).unwrap();
    let complete = pool
        .pin_registered_storage(
            native
                .iter()
                .map(|(id, bytes, _)| (StorageIdentity::Native(id), bytes))
                .chain(std::iter::once((
                    StorageIdentity::HostMetadata(
                        plan.shared_layout().identity().registry_key().clone(),
                    ),
                    plan.shared_layout().capacity_bytes().unwrap(),
                ))),
        )
        .unwrap();
    drop(association);
    CopyQuote {
        host: plan.dense_host_copy(pool).unwrap(),
        bytes: plan
            .dense_initialization()
            .unwrap()
            .initialization_peak_bytes()
            .checked_add(native_bytes)
            .and_then(|n| {
                n.checked_add(crate::memory_fixture::publication_control_bytes(
                    inputs.len().checked_mul(2).unwrap(),
                ))
            })
            .unwrap(),
        native_bytes,
        complete,
    }
}

// Actual fresh, zero-output prompt authority. Every requested byte comes from
// the concrete P+N program above, not from a synthetic forward/state estimate.
fn fresh_admission(bytes: u64) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound =
        |n| WorkspaceBound::bounded(n, "actual dense initialization and isolated native copy");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    crate::memory_fixture::admission(Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(bytes),
        memory_limits: Default::default(),
        additional_headroom: Default::default(),
    })
}

fn fresh_requirements(pool: &MemoryLedger, bytes: u64) -> u64 {
    crate::memory_fixture::host_total(
        &pool
            .reservation_requirements(&fresh_admission(bytes), None)
            .unwrap(),
    )
}

fn fresh(
    pool: &MemoryLedger,
    bytes: u64,
    capacity: u64,
) -> Result<(InferenceTextPreparation, WorkingMemoryFundingRun), WorkingMemoryError> {
    let execution = InferenceExecutionIdentity::default();
    let admission = fresh_admission(bytes);
    let geometry = admission
        .state
        .execution_workspace
        .as_ref()
        .unwrap()
        .geometry;
    let reservation = pool.reserve_with_capacity(
        &execution,
        &admission,
        crate::memory_fixture::physical_host_limits(pool, capacity),
    )?;
    let (reservation, run) = reservation.into_funding()?;
    // The low-level preparation contract accepts zero outputs; no application
    // generation configuration or forward loop is being constructed here.
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: false,
        temperature: 0.0,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(0),
    });
    let preparation =
        InferenceRequest::from(reservation).prepare_text(&execution, geometry, config)?;
    Ok((preparation, run))
}

fn usage(pool: &MemoryLedger) -> (u64, u64) {
    (
        pool.fixture_funded_charge().unwrap(),
        pool.fixture_host_peak().unwrap(),
    )
}

fn publish_arrays(
    completed: &PreparedDenseResidentKvState<'_>,
    scope: &WorkingMemoryFundingScope,
    roots: &RefCell<Vec<Array>>,
) {
    for array in roots.borrow().iter() {
        array.evaluated().unwrap();
    }
    let mut inventory = RetainedStorage::default();
    completed.visit_operands(&mut |array| {
        array.evaluated().unwrap();
        inventory.include_array(array).unwrap();
    });
    drop(inventory.publish_funded(scope).unwrap());
}

fn assert_prompt_claimed(preparation: &InferenceTextPreparation) {
    assert!(matches!(
        preparation.claim_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    assert!(matches!(
        preparation.bind_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
}

#[test]
fn dense_prompt_exact_native_bound_preserves_source_and_transfers_actual_table() {
    // Padded/full + stateless + key-only; repeated source backing; and an empty
    // cache all traverse the same real layer worker and fixed dense destination.
    for variant in 0..3 {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = metal();
        let mut source = state(&stream);
        match variant {
            1 => {
                let alias = source.layers.slots()[0].clone();
                source.layers.slots_mut()[2] = alias;
            }
            2 => {
                source.layers.slots_mut()[2] =
                    MlxKeyValueLayerState::Device(ConcatKeyValueCache::new())
            }
            _ => {}
        }
        source.paged_transaction_branch = true;
        publish_source(&source, &loading);
        drop(loading);
        settle(&pool, pool.fixture_funded_charge().unwrap());
        let (old_sampler, old_preparation, old_run) = sampler(&pool);
        source.inference_retention.admit(old_preparation.request());
        let old_revision = source.inference_retention.revision().clone();
        let old_frontier = source.inference_retention.admission().unwrap().position();
        assert_eq!(source.inference_retention.requests().len(), 1);
        let plan = source.prepare_resident_copy().unwrap();
        let expected = values(&plan);
        let expected_controls = controls(&plan);
        let old_ids = operands(&plan)
            .iter()
            .map(|a| a.allocation_info().unwrap().unwrap().identity())
            .collect::<Vec<_>>();
        let cold = usage(&pool);
        let quoted = quote(&plan, &pool);
        assert_eq!(
            usage(&pool).0,
            cold.0,
            "cold source association allocates no numerical payload"
        );
        let cold = usage(&pool);
        assert!(quoted.native_bytes > 0);
        let capacity = pool
            .fixture_host_current()
            .unwrap()
            .checked_add(fresh_requirements(&pool, quoted.bytes))
            .unwrap();
        let short = fresh(&pool, quoted.bytes, capacity - 1);
        assert!(
            matches!(short, Err(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))
            if required_bytes == fresh_requirements(&pool, quoted.bytes) && limit_bytes.checked_sub(existing_bytes).unwrap() == required_bytes - 1)
        );
        assert_eq!(usage(&pool), cold);
        assert_eq!(values(&plan), expected);
        let (preparation, run) = fresh(&pool, quoted.bytes, capacity).unwrap();
        let (slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(quoted.host, &run, quoted.complete)
            .unwrap();
        let d = slots.retained_bytes();
        let p = slots.protected_bytes();
        assert!(p > d);
        let roots = RefCell::new(Vec::with_capacity(old_ids.len() * 2));
        assert_prompt_claimed(&preparation);
        let completed = plan.copy_dense_retained(slots, &stream, &roots).unwrap();
        assert_eq!(roots.borrow().len(), old_ids.len() * 2);
        assert_eq!(completed.retained_slot_bytes(), d);
        assert_eq!(completed.protected_slot_bytes(), p);
        assert!(completed.shared_layout().same_storage(&source.layout));
        assert_eq!(completed.global_layer_start(), 13);
        let pointer = completed.layers.get(0).unwrap() as *const MlxKeyValueLayerState;
        let table_token = completed.slot_metadata().clone();
        publish_arrays(&completed, &native, &roots);
        let before_transfer = usage(&pool);
        let (destination, completion) = completed.publish().unwrap();
        assert_eq!(
            usage(&pool),
            before_transfer,
            "P to D transfers accounting without a second charge"
        );
        assert_eq!(destination.layers.slots().as_ptr(), pointer);
        assert!(destination.layer_slot_metadata().same_storage(&table_token));
        assert_prompt_claimed(&preparation);
        roots.borrow_mut().clear();
        native.certify().unwrap();
        completion.finish().unwrap();
        preparation.bind_prompt().unwrap();
        let copied = destination.prepare_resident_copy().unwrap();
        assert_eq!(values(&copied), expected);
        assert_eq!(controls(&copied), expected_controls);
        let new_ids = operands(&copied)
            .iter()
            .map(|a| a.allocation_info().unwrap().unwrap().identity())
            .collect::<Vec<_>>();
        assert!(new_ids.iter().all(|id| !old_ids.contains(id)));
        if variant == 1 {
            assert_eq!(old_ids[0], old_ids[2]);
            assert_ne!(
                new_ids[0], new_ids[2],
                "source aliases become independent destination operands"
            );
        }
        assert!(destination.inference_retention.admission().is_none());
        assert_eq!(destination.inference_retention.requests().len(), 0);
        assert_ne!(destination.inference_retention.revision(), &old_revision);
        assert!(!destination.paged_transaction_branch);
        assert_eq!(source.inference_retention.revision(), &old_revision);
        assert_eq!(source.inference_retention.requests().len(), 1);
        let old_admission = source.inference_retention.admission().unwrap();
        assert_eq!(old_admission.position(), old_frontier);
        old_admission
            .request()
            .validate_same_request(old_preparation.request())
            .unwrap();
        assert!(source.paged_transaction_branch);
        assert_eq!(values(&source.prepare_resident_copy().unwrap()), expected);
        let raw = Array::clone(operands(&copied)[0]);
        let raw_bytes = {
            let info = raw.allocation_info().unwrap().unwrap();
            u64::try_from(info.bytes())
                .unwrap()
                .checked_add(info.host_control_bytes() as u64)
                .unwrap()
        };
        drop(copied);
        let account_controls = crate::memory_fixture::request_control_bytes(preparation.request());
        drop((
            source,
            old_sampler,
            old_preparation,
            old_run,
            preparation,
            run,
        ));
        assert_eq!(
            values(&destination.prepare_resident_copy().unwrap()),
            expected
        );
        drop(destination);
        settle(&pool, d + raw_bytes + account_controls);
        assert!(matches!(
            table_token.try_attach(pool.shared_storage_accounting_id(), || Ok::<_, Infallible>(
                Box::new(())
            )),
            Err(HostSlotAttachmentError::Retired)
        ));
        drop(table_token);
        settle(&pool, raw_bytes + account_controls);
        assert!(!raw
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap()
            .is_empty());
        drop(raw);
        settle(&pool, 0);
    }
}

#[test]
fn saved_dense_source_survives_original_request_then_builds_fresh_live_table() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = metal();
    let mut source = state(&stream);
    publish_source(&source, &loading);
    drop(loading);
    settle(&pool, pool.fixture_funded_charge().unwrap());
    let (source_sampler, source_preparation, source_run) = sampler(&pool);
    source
        .inference_retention
        .admit(source_preparation.request());
    let plan = source.prepare_resident_copy().unwrap();
    let expected = values(&plan);
    let expected_controls = controls(&plan);
    let (saved_sampler, slots, native) = admit(&plan, source_sampler.borrow_funded(), &pool);
    let (saved_custody, native) = native.into_parts();
    let roots = RefCell::new(Vec::new());
    let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
    finish_native(&saved, native, &roots);
    drop((source, source_sampler, source_preparation, source_run));
    settle(
        &pool,
        saved_custody
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap()
            .checked_sub(crate::memory_fixture::publication_control_bytes(
                operands(&saved.prepare_copy().unwrap())
                    .len()
                    .checked_mul(2)
                    .unwrap(),
            ))
            .unwrap()
            + saved.shared_layout().capacity_bytes().unwrap(),
    );
    let plan = saved.prepare_copy().unwrap();
    assert_eq!(values(&plan), expected);
    let quoted = quote(&plan, &pool);
    let (preparation, run) = fresh(
        &pool,
        quoted.bytes,
        pool.fixture_host_current()
            .unwrap()
            .checked_add(fresh_requirements(&pool, quoted.bytes))
            .unwrap(),
    )
    .unwrap();
    let (slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(quoted.host, &run, quoted.complete)
        .unwrap();
    let completed = plan.copy_dense_retained(slots, &stream, &roots).unwrap();
    publish_arrays(&completed, &native, &roots);
    let (destination, completion) = completed.publish().unwrap();
    roots.borrow_mut().clear();
    native.certify().unwrap();
    completion.finish().unwrap();
    preparation.bind_prompt().unwrap();
    drop((saved, saved_sampler, saved_custody, preparation, run));
    let plan = destination.prepare_resident_copy().unwrap();
    assert_eq!(values(&plan), expected);
    assert_eq!(controls(&plan), expected_controls);
    assert!(destination.inference_retention.admission().is_none());
    assert_eq!(destination.inference_retention.requests().len(), 0);
    assert_eq!(destination.global_layer_start, 13);
    drop(plan);
    drop(destination);
    settle(&pool, 0);
}

#[test]
fn dense_wrong_source_stops_before_native_and_late_failure_keeps_recovery_roots() {
    for fail_late in [false, true] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = metal();
        let source = state(&stream);
        let other = state(&stream);
        publish_source(&source, &loading);
        publish_source(&other, &loading);
        drop(loading);
        settle(&pool, pool.fixture_funded_charge().unwrap());
        let plan = source.prepare_resident_copy().unwrap();
        let expected = values(&plan);
        let source_bytes = {
            let mut inventory = RetainedStorage::default();
            for array in operands(&plan) {
                inventory.include_array(array).unwrap();
            }
            inventory
                .include_slot_metadata(source.layer_slot_metadata().clone())
                .unwrap();
            inventory
                .include_metadata(eredu_runtime::SharedHostMetadata::Layout(
                    source.layout.clone(),
                ))
                .unwrap();
            inventory.byte_bound().unwrap().unwrap()
        };
        let quoted = quote(&plan, &pool);
        let required = quoted.bytes;
        let (preparation, run) = fresh(
            &pool,
            required,
            pool.fixture_host_current()
                .unwrap()
                .checked_add(fresh_requirements(&pool, required))
                .unwrap(),
        )
        .unwrap();
        let (slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(quoted.host, &run, quoted.complete)
            .unwrap();
        let roots = RefCell::new(Vec::new());
        if fail_late {
            FAIL_AFTER_LAYER.with(|f| f.set(Some(0)));
            let error = plan
                .copy_dense_retained(slots, &stream, &roots)
                .unwrap_err();
            assert!(matches!(&error, ResidentKvCopyError::Native(_)));
            assert!(error
                .to_string()
                .contains("injected resident decoder layer-copy failure"));
            assert_eq!(FAIL_AFTER_LAYER.with(|f| f.get()), None);
            assert_eq!(
                roots.borrow().len(),
                4,
                "first layer's contiguous intermediates and independent outputs survive"
            );
            assert_prompt_claimed(&preparation);
            assert_eq!(values(&source.prepare_resident_copy().unwrap()), expected);
            // Observe the retained numerical prefix without treating a failed
            // publication as certified completion. Abandonment retains demand
            // and complete source pins even after all wrappers are dropped.
            for array in roots.borrow().iter() {
                array.evaluated().unwrap();
            }
            let before = usage(&pool);
            drop(native);
            assert_eq!(usage(&pool), before);
            roots.borrow_mut().clear();
            let account_controls =
                crate::memory_fixture::request_control_bytes(preparation.request());
            drop((source, other, preparation, run));
            settle(&pool, source_bytes + required + account_controls);
        } else {
            drop(plan);
            let error = other
                .prepare_resident_copy()
                .unwrap()
                .copy_dense_retained(slots, &stream, &roots)
                .unwrap_err();
            assert!(matches!(
                error,
                ResidentKvCopyError::Memory(WorkingMemoryError::IdentityMismatch)
            ));
            assert!(roots.borrow().is_empty());
            assert_prompt_claimed(&preparation);
            assert_eq!(values(&source.prepare_resident_copy().unwrap()), expected);
            native.certify().unwrap(); // No native operation was entered.
            drop((source, other, preparation, run));
            settle(&pool, 0);
        }
    }
}

#[test]
fn dense_publication_error_retains_complete_native_owner_and_original_prompt_claim() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = metal();
    let source = state(&stream);
    publish_source(&source, &loading);
    drop(loading);
    settle(&pool, pool.fixture_funded_charge().unwrap());
    let plan = source.prepare_resident_copy().unwrap();
    let expected = values(&plan);
    let quoted = quote(&plan, &pool);
    let (preparation, run) = fresh(
        &pool,
        quoted.bytes,
        pool.fixture_host_current()
            .unwrap()
            .checked_add(fresh_requirements(&pool, quoted.bytes))
            .unwrap(),
    )
    .unwrap();
    let (slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(quoted.host, &run, quoted.complete)
        .unwrap();
    let roots = RefCell::new(Vec::new());
    let completed = plan.copy_dense_retained(slots, &stream, &roots).unwrap();
    let pointer = completed.layers.get(0).unwrap() as *const MlxKeyValueLayerState;
    let token = completed.slot_metadata().clone();
    assert!(token
        .try_attach(pool.shared_storage_accounting_id(), || Ok::<_, Infallible>(
            Box::new(())
        ))
        .unwrap());
    for array in roots.borrow().iter() {
        array.evaluated().unwrap();
    }
    let before = usage(&pool);
    let error = completed.publish().unwrap_err();
    assert_eq!(usage(&pool), before);
    assert_prompt_claimed(&preparation);
    let (completed, cause) = error.into_parts();
    assert!(matches!(
        cause,
        HostSlotAttachmentError::Attachment(SharedStorageAttachmentError::Provider(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(
        completed.layers.get(0).unwrap() as *const MlxKeyValueLayerState,
        pointer
    );
    assert!(completed.slot_metadata().same_storage(&token));
    let mut recovered_values = Vec::new();
    completed.visit_operands(&mut |a| {
        recovered_values.push(a.evaluated().unwrap().try_to_vec::<f32>().unwrap())
    });
    assert_eq!(recovered_values, expected);
    drop(completed);
    assert!(matches!(
        token.try_attach(pool.shared_storage_accounting_id(), || Ok::<_, Infallible>(
            Box::new(())
        )),
        Err(HostSlotAttachmentError::Retired)
    ));
    drop(token);
    roots.borrow_mut().clear();
    // All outputs/intermediates were evaluated and discarded; there is no
    // destination to publish. This explicit cleanup does not finish the prompt.
    native.certify().unwrap();
    assert_prompt_claimed(&preparation);
    drop((source, preparation, run));
    settle(&pool, 0);
}

#[test]
fn all_stateless_dense_prompt_copies_real_slots_without_native_payload() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = metal();
    let layout = eredu_runtime::StateLayout::new(
        LayerSchedule::new(3, vec![LayerCachePolicy::NoState; 3]).unwrap(),
    )
    .unwrap();
    let source = MlxKeyValueState::device_with_global_layer_start(layout, 29).unwrap();
    publish_source(&source, &loading);
    drop(loading);
    settle(&pool, pool.fixture_funded_charge().unwrap());
    let plan = source.prepare_resident_copy().unwrap();
    assert!(operands(&plan).is_empty());
    let quoted = quote(&plan, &pool);
    assert_eq!(quoted.native_bytes, 0);
    assert_eq!(
        quoted.bytes,
        plan.dense_initialization()
            .unwrap()
            .initialization_peak_bytes()
            .checked_add(crate::memory_fixture::publication_control_bytes(0))
            .unwrap()
    );
    let (preparation, run) = fresh(
        &pool,
        quoted.bytes,
        pool.fixture_host_current()
            .unwrap()
            .checked_add(fresh_requirements(&pool, quoted.bytes))
            .unwrap(),
    )
    .unwrap();
    let (slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(quoted.host, &run, quoted.complete)
        .unwrap();
    assert_eq!(slots.len(), 3);
    let roots = RefCell::new(Vec::new());
    let completed = plan.copy_dense_retained(slots, &stream, &roots).unwrap();
    assert!(roots.borrow().is_empty());
    let address = completed.layers.get(0).unwrap() as *const MlxKeyValueLayerState;
    let (destination, completion) = completed.publish().unwrap();
    assert_eq!(destination.layers.slots().as_ptr(), address);
    native.certify().unwrap();
    completion.finish().unwrap();
    preparation.bind_prompt().unwrap();
    assert_eq!(destination.layers.slots().len(), 3);
    assert!(destination
        .layers
        .slots()
        .iter()
        .all(|layer| matches!(layer, MlxKeyValueLayerState::Stateless)));
    assert_eq!(destination.global_layer_start, 29);
    assert!(destination.inference_retention.admission().is_none());
    assert!(operands(&destination.prepare_resident_copy().unwrap()).is_empty());
    let token = destination.layer_slot_metadata().clone();
    let d = token.capacity_bytes().unwrap();
    assert!(d > 0);
    let account_controls = crate::memory_fixture::request_control_bytes(preparation.request());
    drop((source, destination, preparation, run));
    settle(&pool, d + account_controls);
    drop(token);
    settle(&pool, 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
