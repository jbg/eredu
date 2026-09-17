use super::*;
use crate::working_memory::{
    BorrowedFundedSampler, InferenceRequest, InferenceTextPreparation, RegisteredWorkspaceCopy,
    RegisteredWorkspaceStorage, RunOwnedTextSampler, WorkingMemoryFundingRun,
};
use crate::{ConfiguredTextSampler, DenseHostSlotInitialization, HostSlotTable};
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig,
    StateMemoryLayout, TextGenerationConfig, WorkspaceBound,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceHostBound,
    WorkspaceIsolatedCopyPlan, WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation,
    WorkspaceOperationBound, WorkspaceOperationKind, WorkspaceOutputStorage, WorkspaceTensor,
};
use std::{cell::Cell, mem::size_of, num::NonZeroU8};

#[path = "../fixtures.rs"]
mod fixtures;
use fixtures::{grow, history, source, Facts};

const CAPACITY: u64 = 1_048_576;
const NATIVE: u64 = 48;
thread_local! {static INITIALIZATIONS:Cell<usize>=const {Cell::new(0)};}
pub(super) fn before_initialize() {
    INITIALIZATIONS.set(INITIALIZATIONS.get() + 1);
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Host(HostMetadataKey),
    Array(u32),
}
impl HostSlotStorageKey for Key {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        match self {
            Self::Host(id) => Some(id),
            _ => None,
        }
    }
}
fn key<T>(table: &HostSlotTable<T>) -> Key {
    Key::Host(table.metadata().identity().registry_key().clone())
}
fn empty(pool: &WorkingMemoryPool) -> WorkingMemoryStorage<Key> {
    pool.pin_registered_storage(std::iter::empty::<(Key, u64)>())
        .unwrap()
}
fn usage(pool: &WorkingMemoryPool) -> (u64, u64, u64, usize, u64) {
    let u = pool.0.usage.lock().unwrap();
    (
        u.reserved,
        u.registered,
        u.peak,
        u.funding.values().map(|s| s.scopes).sum(),
        u.funding.values().map(|s| s.host_held).sum(),
    )
}
fn array_plan(pool: &WorkingMemoryPool) -> RegisteredWorkspaceCopy<Key> {
    let ctx = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &ctx);
    let source =
        RegisteredWorkspaceStorage::bind(pool, &ctx, [(Key::Array(1), root.clone())]).unwrap();
    let a = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
        &root,
        &ctx,
    )
    .unwrap();
    let plan = WorkspaceIsolatedCopyPlan::prepare(&ctx, source.borrowed_storage(), &[a]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(NATIVE));
    RegisteredWorkspaceCopy::bind(plan, source).unwrap()
}
#[derive(Debug)]
struct Live {
    fixed: HostSlotTable<Option<u32>>,
    tag: u64,
}
#[derive(Debug)]
struct Saved {
    fixed: FundedDecoderSlots<Option<u64>>,
    tag: u64,
}
#[derive(Debug)]
struct Fresh {
    fixed: HostSlotTable<Option<u64>>,
    tag: u64,
}
fn live() -> HostSlotTable<Live> {
    HostSlotTable::new(
        vec![
            Live {
                fixed: HostSlotTable::new(Vec::new().into_boxed_slice()),
                tag: 11,
            },
            Live {
                fixed: HostSlotTable::new(Box::new([None])),
                tag: 23,
            },
            Live {
                fixed: HostSlotTable::new(Box::new([Some(7), Some(19)])),
                tag: 47,
            },
        ]
        .into_boxed_slice(),
    )
}
fn register_tree(
    pool: &WorkingMemoryPool,
    tree: &HostSlotTable<Live>,
) -> WorkingMemoryStorage<Key> {
    pool.register_storage(
        std::iter::once((key(tree), tree.metadata().capacity_bytes().unwrap())).chain(
            tree.slots()
                .iter()
                .map(|s| (key(&s.fixed), s.fixed.metadata().capacity_bytes().unwrap())),
        ),
    )
    .unwrap()
}
fn frozen_plan<'a>(
    pool: &WorkingMemoryPool,
    tree: &'a HostSlotTable<Live>,
) -> RegisteredDecoderTableGroup<'a, Live, Saved, Option<u32>, Option<u64>, Key> {
    let outer =
        RegisteredDecoderHostCopy::bind(pool, tree.prepare_copy_slots().unwrap(), key(tree))
            .unwrap()
            .for_destination::<Saved>()
            .unwrap();
    let children = tree
        .slots()
        .iter()
        .map(|s| {
            RegisteredDecoderHostCopy::bind(
                pool,
                s.fixed.prepare_copy_slots().unwrap(),
                key(&s.fixed),
            )
            .unwrap()
            .for_destination::<Option<u64>>()
            .unwrap()
        })
        .collect();
    RegisteredDecoderTableGroup::new(outer, children).unwrap()
}
fn dense_plan<'a>(
    pool: &WorkingMemoryPool,
    tree: &'a HostSlotTable<Live>,
) -> RegisteredDenseDecoderTableGroup<'a, Live, Fresh, Option<u32>, Option<u64>, Key> {
    let outer =
        RegisteredDecoderHostCopy::bind(pool, tree.prepare_copy_slots().unwrap(), key(tree))
            .unwrap()
            .for_dense_destination::<Fresh>()
            .unwrap();
    let children = tree
        .slots()
        .iter()
        .map(|s| {
            RegisteredDecoderHostCopy::bind(
                pool,
                s.fixed.prepare_copy_slots().unwrap(),
                key(&s.fixed),
            )
            .unwrap()
            .for_dense_destination::<Option<u64>>()
            .unwrap()
        })
        .collect();
    RegisteredDenseDecoderTableGroup::new(outer, children).unwrap()
}
fn joint<'a>(
    pool: &WorkingMemoryPool,
    sampler: BorrowedFundedSampler<'a>,
    tree: &'a HostSlotTable<Live>,
    complete: WorkingMemoryStorage<Key>,
) -> RegisteredTextComponentsGroupCopy<'a, Live, Saved, Option<u32>, Option<u64>, Key> {
    RegisteredSamplingCopy::prepare(sampler, array_plan(pool))
        .unwrap()
        .with_decoder_group(frozen_plan(pool, tree), complete)
        .unwrap()
}
fn fill_frozen(
    mut slots: InitializedDecoderTableGroup<Saved, Option<u64>>,
    tree: &HostSlotTable<Live>,
) -> FundedDecoderSlots<Saved> {
    slots
        .validate_source(
            &tree
                .prepare_copy_slots()
                .unwrap()
                .for_destination::<Saved>()
                .unwrap(),
        )
        .unwrap();
    for (i, row) in tree.slots().iter().enumerate() {
        let mut child = slots.take_child(i).unwrap();
        assert!(matches!(
            slots.take_child(i),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        child
            .validate_source(
                &row.fixed
                    .prepare_copy_slots()
                    .unwrap()
                    .for_destination::<Option<u64>>()
                    .unwrap(),
            )
            .unwrap();
        for value in row.fixed.slots() {
            child.push(value.map(u64::from)).unwrap();
        }
        slots
            .push(Saved {
                fixed: child.finish().unwrap(),
                tag: row.tag,
            })
            .unwrap();
    }
    slots.finish().unwrap()
}

fn fresh(
    pool: &WorkingMemoryPool,
    bytes: u64,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    let execution = InferenceExecutionIdentity::default();
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
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "closed scalar host preparation fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let reservation = pool
        .reserve_with_capacity(
            &execution,
            &Admission {
                requested_positions: 1,
                state,
                incremental_required_bytes: bytes,
                available_memory_bytes: None,
            },
            CAPACITY,
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let request = InferenceRequest::from(reservation);
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
    (
        request.prepare_text(&execution, geometry, config).unwrap(),
        run,
        config,
    )
}
fn sampler(
    preparation: &InferenceTextPreparation,
    run: &WorkingMemoryFundingRun,
    config: TextGenerationConfig,
) -> RunOwnedTextSampler {
    let (sampler, complete) = preparation
        .claim_sampling(config)
        .unwrap()
        .construct_sampler(run.sampler_scope().unwrap())
        .unwrap();
    complete.finish().unwrap();
    sampler
}

#[test]
fn grouped_frozen_exact_and_one_short_include_absent_roles_and_all_sources() {
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let tree = live();
    let tables = register_tree(&pool, &tree);
    let arrays = pool
        .register_storage([(Key::Array(1), 64), (Key::Array(2), 80)])
        .unwrap();
    let (mut source_sampler, source_prep, source_run) = source(&pool, CAPACITY, 8, true);
    grow(&mut source_sampler, &[3, 11, 7, 19]);
    let required =
        joint(&pool, source_sampler.borrow_funded(), &tree, arrays.clone()).required_bytes();
    let p = frozen_plan(&pool, &tree).initialization_peak_bytes();
    assert_eq!(
        required,
        p + NATIVE
            + source_sampler
                .as_sampler()
                .prepare_copy()
                .unwrap()
                .retained_bytes()
    );
    for app in [true, false] {
        let before = (usage(&pool), INITIALIZATIONS.get());
        let mut limits = WorkspaceCopyLimits::new(if app {
            CAPACITY
        } else {
            pool.used_bytes().unwrap() + required - 1
        });
        limits.application_memory_budget_bytes = app.then_some(required - 1);
        let error = pool
            .copy_text_components_group(
                joint(&pool, source_sampler.borrow_funded(), &tree, arrays.clone()),
                limits,
            )
            .err()
            .unwrap();
        if app {
            assert!(
                matches!(error, DecoderCopyAdmissionError::ApplicationBudgetExceeded { required_bytes, budget_bytes }
                if required_bytes == required && budget_bytes == required - 1)
            );
        } else {
            assert!(
                matches!(error, DecoderCopyAdmissionError::Memory(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
                if required_bytes == required && available_bytes == required - 1)
            );
        }
        assert_eq!((usage(&pool), INITIALIZATIONS.get()), before);
    }
    let before = usage(&pool);
    let (saved_sampler, slots, native) = pool
        .copy_text_components_group(
            joint(&pool, source_sampler.borrow_funded(), &tree, arrays.clone()),
            WorkspaceCopyLimits::new(pool.used_bytes().unwrap() + required),
        )
        .unwrap();
    assert_eq!(
        usage(&pool).3,
        before.3 + 6,
        "sampler + outer + three children + native"
    );
    // There are six scopes: sampler, outer, three children, and native.
    assert_eq!(usage(&pool).4, before.4 + p + saved_sampler.bytes());
    assert_eq!(history(saved_sampler.as_sampler()), &[3, 11, 7, 19]);
    let saved = fill_frozen(slots, &tree);
    assert_eq!(saved.get(0).unwrap().fixed.len(), 0);
    assert_eq!(saved.get(1).unwrap().fixed.len(), 1);
    assert_eq!(*saved.get(1).unwrap().fixed.get(0).unwrap(), None);
    assert_eq!(*saved.get(2).unwrap().fixed.get(1).unwrap(), Some(19));
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((
        custody,
        saved,
        saved_sampler,
        source_sampler,
        source_prep,
        source_run,
        arrays,
        tables,
        tree,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn dense_group_claim_waits_for_every_child_publication_and_original_input_binding() {
    for short in [true, false] {
        let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
        let tree = live();
        let tables = register_tree(&pool, &tree);
        let plan = dense_plan(&pool, &tree);
        let p = plan.initialization_peak_bytes();
        let d = plan.retained_bytes();
        let h = size_of::<ConfiguredTextSampler>() as u64;
        let (prep, run, config) = fresh(&pool, h + p - u64::from(short));
        let sampler = sampler(&prep, &run, config);
        let before = (usage(&pool), INITIALIZATIONS.get());
        let result =
            prep.claim_prompt()
                .unwrap()
                .construct_dense_decoder_group(plan, &run, empty(&pool));
        if short {
            assert!(
                matches!(result,Err(DecoderCopyAdmissionError::Memory(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes})) if required_bytes==p&&available_bytes==p-1)
            );
            assert_eq!((usage(&pool), INITIALIZATIONS.get()), before);
            assert!(matches!(
                prep.claim_prompt(),
                Err(WorkingMemoryError::PreparationAlreadyStarted)
            ));
            drop((sampler, prep, run, tables, tree));
            assert_eq!(pool.used_bytes().unwrap(), 0);
            continue;
        }
        let (mut group, native) = result.unwrap();
        assert_eq!(usage(&pool).3, before.0 .3 + 5);
        assert_eq!(usage(&pool).4, h + p);
        assert!(matches!(
            native.adopt_storage_individually([(Key::Array(99), 1)]),
            Err(WorkingMemoryError::BudgetExceeded {
                available_bytes: 0,
                ..
            })
        ));
        group
            .validate_source(
                &tree
                    .prepare_copy_slots()
                    .unwrap()
                    .for_dense_destination::<Fresh>()
                    .unwrap(),
            )
            .unwrap();
        let foreign = live();
        assert!(matches!(
            group.validate_source(
                &foreign
                    .prepare_copy_slots()
                    .unwrap()
                    .for_dense_destination::<Fresh>()
                    .unwrap()
            ),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        group = group.finish().unwrap_err().into_owner();
        let mut tokens = Vec::new();
        for (i, row) in tree.slots().iter().enumerate() {
            let mut child = group.take_child(i).unwrap();
            assert!(matches!(
                group.take_child(i),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            child
                .validate_source(
                    &row.fixed
                        .prepare_copy_slots()
                        .unwrap()
                        .for_dense_destination::<Option<u64>>()
                        .unwrap(),
                )
                .unwrap();
            if i == 1 {
                let unrelated = HostSlotTable::new(Box::new([None::<u32>]));
                assert!(matches!(
                    child.validate_source(
                        &unrelated
                            .prepare_copy_slots()
                            .unwrap()
                            .for_dense_destination::<Option<u64>>()
                            .unwrap()
                    ),
                    Err(WorkingMemoryError::IdentityMismatch)
                ));
            }
            if !child.is_empty() {
                child = child.finish().unwrap_err().into_owner();
            }
            for value in row.fixed.slots() {
                child.push(value.map(u64::from)).unwrap();
            }
            let completed = child.finish().unwrap();
            let token = completed.metadata().clone();
            let before_failure = usage(&pool);
            let failed = completed.publish(Key::Array(999)).unwrap_err();
            assert!(matches!(
                failed.error(),
                crate::HostSlotAttachmentError::Attachment(
                    eredu_core::SharedStorageAttachmentError::Provider(
                        WorkingMemoryError::IdentityMismatch
                    )
                )
            ));
            assert_eq!(usage(&pool), before_failure);
            let (completed, _) = failed.into_parts();
            let address = completed.get(0).map(|x| x as *const Option<u64>);
            let fixed = completed
                .publish(Key::Host(token.identity().registry_key().clone()))
                .unwrap();
            assert_eq!(
                fixed.slots().first().map(|x| x as *const Option<u64>),
                address
            );
            group
                .push(Fresh {
                    fixed,
                    tag: row.tag,
                })
                .unwrap();
            tokens.push(token);
            assert!(matches!(
                prep.bind_prompt(),
                Err(WorkingMemoryError::PreparationAlreadyStarted)
            ));
        }
        let completed = group.finish().unwrap();
        let outer_token = completed.metadata().clone();
        let pointer = completed.get(0).unwrap() as *const Fresh;
        let (table, completion) = completed
            .publish(Key::Host(outer_token.identity().registry_key().clone()))
            .unwrap();
        assert_eq!(table.slots().as_ptr(), pointer);
        assert_eq!(usage(&pool).4, h);
        assert_eq!(table.slots()[1].fixed.slots(), &[None]);
        assert_eq!(table.slots()[2].fixed.slots(), &[Some(7), Some(19)]);
        assert_eq!(table.slots()[2].tag, 47);
        completion.finish().unwrap();
        let request = prep.request();
        let execution = &request.memory_reservation().unwrap().0.execution;
        assert!(matches!(
            request.begin_prefill(execution, request.geometry()),
            Err(WorkingMemoryError::PreparationNotReady)
        ));
        prep.bind_prompt().unwrap();
        request
            .begin_prefill(execution, request.geometry())
            .unwrap();
        assert!(matches!(
            prep.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        native.certify().unwrap();
        drop((sampler, prep, run, tables, tree, table));
        assert_eq!(
            pool.used_bytes().unwrap(),
            d,
            "escaped metadata retains actual registered group payload charges"
        );
        drop((tokens, outer_token));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn published_child_and_partial_parent_retire_independently_without_early_refund() {
    for child_first in [true, false] {
        let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
        let tree = live();
        let tables = register_tree(&pool, &tree);
        let plan = dense_plan(&pool, &tree);
        let p = plan.initialization_peak_bytes();
        let (prep, run, _) = fresh(&pool, p);
        let (mut group, native) = prep
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder_group(plan, &run, empty(&pool))
            .unwrap();
        let mut child = group.take_child(2).unwrap();
        for value in tree.slots()[2].fixed.slots() {
            child.push(value.map(u64::from)).unwrap();
        }
        let completed = child.finish().unwrap();
        let token = completed.metadata().clone();
        let bytes = completed.retained_bytes();
        let child = completed
            .publish(Key::Host(token.identity().registry_key().clone()))
            .unwrap();
        let mut missing = group.take_child(1).unwrap();
        missing.push(None).unwrap();
        let failure = group.finish().unwrap_err();
        assert!(matches!(
            prep.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        if child_first {
            drop(child);
            drop(failure);
        } else {
            drop(failure);
            drop(child);
        }
        drop(missing);
        native.certify().unwrap();
        drop((prep, run, tables, tree));
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        assert!(matches!(
            token.try_attach(pool.shared_storage_domain(), || Ok::<
                Box<dyn Send + Sync>,
                WorkingMemoryError,
            >(Box::new(()))),
            Err(crate::HostSlotAttachmentError::Retired)
        ));
        drop(token);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn native_scope_alone_retains_complete_uncopied_roots_after_host_results_retire() {
    for certify in [true, false] {
        let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
        let tree = live();
        let tables = register_tree(&pool, &tree);
        let arrays = pool
            .register_storage([(Key::Array(1), 64), (Key::Array(2), 80)])
            .unwrap();
        let source_bytes = tables.bytes() + arrays.bytes();
        let (sampler, prep, run) = source(&pool, CAPACITY, 4, false);
        let (copy, group, native) = pool
            .copy_text_components_group(
                joint(&pool, sampler.borrow_funded(), &tree, arrays.clone()),
                WorkspaceCopyLimits::new(CAPACITY),
            )
            .unwrap();
        let (custody, scope) = native.into_parts();
        drop((
            copy, group, sampler, prep, run, arrays, tables, tree, custody,
        ));
        assert!(pool.used_bytes().unwrap() >= source_bytes);
        assert!(pool.pin_registered_storage([(Key::Array(2), 80)]).is_ok());
        if certify {
            scope.certify().unwrap();
            assert_eq!(pool.used_bytes().unwrap(), 0);
        } else {
            drop(scope);
            assert!(pool.used_bytes().unwrap() >= source_bytes);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        }
    }
}

fn saved_fixture(
    pool: &WorkingMemoryPool,
) -> (
    FundedSamplerCopy,
    FundedDecoderSlots<Saved>,
    crate::working_memory::WorkspaceCopyCustody,
    crate::working_memory::WorkingMemoryFundingScope,
) {
    let tree = live();
    let tables = register_tree(pool, &tree);
    let arrays = pool
        .register_storage([(Key::Array(1), 64), (Key::Array(2), 80)])
        .unwrap();
    let (sampler, prep, run) = source(pool, CAPACITY, 4, false);
    let (copied, group, native) = pool
        .copy_text_components_group(
            joint(pool, sampler.borrow_funded(), &tree, arrays.clone()),
            WorkspaceCopyLimits::new(CAPACITY),
        )
        .unwrap();
    let saved = fill_frozen(group, &tree);
    let (custody, scope) = native.into_parts();
    drop((sampler, prep, run, arrays, tables, tree));
    (copied, saved, custody, scope)
}
fn saved_dense_plan(
    saved: &FundedDecoderSlots<Saved>,
) -> RegisteredDenseDecoderTableGroup<'_, Saved, Fresh, Option<u64>, Option<u64>, Key> {
    let outer = saved
        .prepare_copy::<Key>()
        .unwrap()
        .for_dense_destination::<Fresh>()
        .unwrap();
    let children = saved
        .iter()
        .map(|s| {
            s.fixed
                .prepare_copy::<Key>()
                .unwrap()
                .for_dense_destination::<Option<u64>>()
                .unwrap()
        })
        .collect();
    RegisteredDenseDecoderTableGroup::new(outer, children).unwrap()
}

#[test]
fn actual_funded_saved_children_support_typechanging_fresh_without_original_registrations() {
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (sampler, saved, old_custody, old_scope) = saved_fixture(&pool);
    old_scope.certify().unwrap();
    drop(old_custody);
    let plan = saved_dense_plan(&saved);
    let p = plan.initialization_peak_bytes();
    let (prep, run, _) = fresh(&pool, p);
    let (mut group, native) = prep
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder_group(plan, &run, empty(&pool))
        .unwrap();
    group
        .validate_source(
            &saved
                .prepare_copy_slots()
                .unwrap()
                .for_dense_destination::<Fresh>()
                .unwrap(),
        )
        .unwrap();
    for (i, row) in saved.iter().enumerate() {
        let mut child = group.take_child(i).unwrap();
        child
            .validate_source(
                &row.fixed
                    .prepare_copy_slots()
                    .unwrap()
                    .for_dense_destination::<Option<u64>>()
                    .unwrap(),
            )
            .unwrap();
        for &value in row.fixed.iter() {
            child.push(value).unwrap();
        }
        let completed = child.finish().unwrap();
        let identity = completed.metadata().identity().registry_key().clone();
        let fixed = completed.publish(Key::Host(identity)).unwrap();
        group
            .push(Fresh {
                fixed,
                tag: row.tag,
            })
            .unwrap();
    }
    let completed = group.finish().unwrap();
    let identity = completed.metadata().identity().registry_key().clone();
    let (fresh_table, completion) = completed.publish(Key::Host(identity)).unwrap();
    completion.finish().unwrap();
    native.certify().unwrap();
    drop((sampler, saved, prep, run));
    assert_eq!(fresh_table.slots()[2].fixed.slots(), &[Some(7), Some(19)]);
    assert_eq!(fresh_table.slots()[1].fixed.slots(), &[None]);
    assert_eq!(fresh_table.slots()[0].tag, 11);
    let d = fresh_table.metadata().capacity_bytes().unwrap()
        + fresh_table
            .slots()
            .iter()
            .map(|r| r.fixed.metadata().capacity_bytes().unwrap())
            .sum::<u64>();
    assert_eq!(pool.used_bytes().unwrap(), d);
    drop(fresh_table);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn quarantined_saved_child_origin_and_foreign_registered_child_reject_before_any_initializer() {
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (sampler, saved, old_custody, old_scope) = saved_fixture(&pool);
    let plan = saved_dense_plan(&saved);
    let p = plan.initialization_peak_bytes();
    let (prep, run, _) = fresh(&pool, p);
    // The source plan was sound when prepared. Its actual origin changes before
    // construction; the same-lock group check must reject the entire operation.
    drop(old_scope);
    let before = (usage(&pool), INITIALIZATIONS.get());
    assert!(matches!(
        prep.claim_prompt()
            .unwrap()
            .construct_dense_decoder_group(plan, &run, empty(&pool)),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), INITIALIZATIONS.get()), before);
    drop((sampler, saved, old_custody, prep, run));

    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let foreign_pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let tree = live();
    let tables = register_tree(&pool, &tree);
    let foreign = HostSlotTable::new(Box::new([None::<u32>]));
    let foreign_charge = foreign_pool
        .register_storage([(key(&foreign), foreign.metadata().capacity_bytes().unwrap())])
        .unwrap();
    let outer =
        RegisteredDecoderHostCopy::bind(&pool, tree.prepare_copy_slots().unwrap(), key(&tree))
            .unwrap()
            .for_dense_destination::<Fresh>()
            .unwrap();
    let children = tree
        .slots()
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let (pool, table) = if i == 1 {
                (&foreign_pool, &foreign)
            } else {
                (&pool, &row.fixed)
            };
            RegisteredDecoderHostCopy::bind(pool, table.prepare_copy_slots().unwrap(), key(table))
                .unwrap()
                .for_dense_destination::<Option<u64>>()
                .unwrap()
        })
        .collect();
    let group = RegisteredDenseDecoderTableGroup::new(outer, children).unwrap();
    let p = group.initialization_peak_bytes();
    let (prep, run, _) = fresh(&pool, p);
    let before = (usage(&pool), usage(&foreign_pool), INITIALIZATIONS.get());
    assert!(matches!(
        prep.claim_prompt()
            .unwrap()
            .construct_dense_decoder_group(group, &run, empty(&pool)),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(
        (usage(&pool), usage(&foreign_pool), INITIALIZATIONS.get()),
        before
    );
    drop((prep, run, tables, tree, foreign_charge, foreign));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
}

#[test]
fn exact_group_count_zero_outer_and_checked_typechanging_overflow_preserve_legacy_path() {
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let tree = live();
    let tables = register_tree(&pool, &tree);
    let outer =
        RegisteredDecoderHostCopy::bind(&pool, tree.prepare_copy_slots().unwrap(), key(&tree))
            .unwrap()
            .for_destination::<Saved>()
            .unwrap();
    assert!(matches!(
        RegisteredDecoderTableGroup::new(
            outer,
            Vec::<RegisteredDecoderHostCopy<'_, Option<u32>, Key, Option<u64>>>::new()
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let enormous = HostSlotTable::new(vec![(); isize::MAX as usize / 2].into_boxed_slice());
    let charge = pool.register_storage([(key(&enormous), 0)]).unwrap();
    let source = RegisteredDecoderHostCopy::bind(
        &pool,
        enormous.prepare_copy_slots().unwrap(),
        key(&enormous),
    )
    .unwrap();
    assert!(matches!(
        source.for_destination::<u64>(),
        Err(DecoderCopyAdmissionError::Initialization(
            HostSlotInitializationError::Overflow { .. }
        ))
    ));
    drop((charge, enormous));
    let empty_outer = HostSlotTable::<Live>::new(Vec::new().into_boxed_slice());
    let empty_charge = register_tree(&pool, &empty_outer);
    let outer = RegisteredDecoderHostCopy::bind(
        &pool,
        empty_outer.prepare_copy_slots().unwrap(),
        key(&empty_outer),
    )
    .unwrap()
    .for_dense_destination::<Fresh>()
    .unwrap();
    let group = RegisteredDenseDecoderTableGroup::new(
        outer,
        Vec::<RegisteredDenseDecoderInitialization<'_, Option<u32>, Option<u64>, Key>>::new(),
    )
    .unwrap();
    assert_eq!(
        (group.retained_bytes(), group.initialization_peak_bytes()),
        (0, 0)
    );
    let (prep, run, _) = fresh(&pool, 0);
    let (group, native) = prep
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder_group(group, &run, empty(&pool))
        .unwrap();
    let completed = group.finish().unwrap();
    let id = completed.metadata().identity().registry_key().clone();
    let (table, completion) = completed.publish(Key::Host(id)).unwrap();
    assert!(table.is_empty());
    completion.finish().unwrap();
    native.certify().unwrap();
    drop((table, prep, run, empty_charge, empty_outer, tables, tree));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

fn empty_array_plan(pool: &WorkingMemoryPool) -> RegisteredWorkspaceCopy<Key> {
    let context = WorkspaceContext::new(Facts::default());
    let registration = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        std::iter::empty::<(Key, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, registration.borrowed_storage(), &[]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(0));
    RegisteredWorkspaceCopy::bind(plan, registration).unwrap()
}

#[test]
fn saved_to_saved_group_reuses_actual_held_origins_without_option_nesting_or_old_registry() {
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (sampler, saved, custody, scope) = saved_fixture(&pool);
    scope.certify().unwrap();
    drop(custody);
    let outer = saved
        .prepare_copy::<Key>()
        .unwrap()
        .for_destination::<Saved>()
        .unwrap();
    let children = saved
        .iter()
        .map(|row| {
            row.fixed
                .prepare_copy::<Key>()
                .unwrap()
                .for_destination::<Option<u64>>()
                .unwrap()
        })
        .collect();
    let group = RegisteredDecoderTableGroup::new(outer, children).unwrap();
    let p = group.initialization_peak_bytes();
    assert_eq!(
        p,
        saved.protected_bytes()
            + saved
                .iter()
                .map(|row| row.fixed.protected_bytes())
                .sum::<u64>()
    );
    let joined = RegisteredSamplingCopy::prepare(sampler.borrow_funded(), empty_array_plan(&pool))
        .unwrap()
        .with_decoder_group(group, empty(&pool))
        .unwrap();
    let (new_sampler, mut group, native) = pool
        .copy_text_components_group(joined, WorkspaceCopyLimits::new(CAPACITY))
        .unwrap();
    group
        .validate_source(
            &saved
                .prepare_copy_slots()
                .unwrap()
                .for_destination::<Saved>()
                .unwrap(),
        )
        .unwrap();
    for (i, row) in saved.iter().enumerate() {
        let mut child = group.take_child(i).unwrap();
        child
            .validate_source(
                &row.fixed
                    .prepare_copy_slots()
                    .unwrap()
                    .for_destination::<Option<u64>>()
                    .unwrap(),
            )
            .unwrap();
        for &value in row.fixed.iter() {
            child.push(value).unwrap();
        }
        group
            .push(Saved {
                fixed: child.finish().unwrap(),
                tag: row.tag,
            })
            .unwrap();
    }
    let copied = group.finish().unwrap();
    assert_eq!(copied.retained_bytes(), saved.retained_bytes());
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((custody, sampler, saved));
    assert_eq!(*copied.get(1).unwrap().fixed.get(0).unwrap(), None);
    assert_eq!(*copied.get(2).unwrap().fixed.get(0).unwrap(), Some(7));
    drop((copied, new_sampler));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unrelated_complete_source_quarantine_is_checked_with_all_fresh_group_holds() {
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let tree = live();
    let tables = register_tree(&pool, &tree);
    let (old_prep, old_run, _) = fresh(&pool, 16);
    let old_scope = old_run.scope().unwrap();
    let extra = old_scope
        .adopt_storage_individually([(Key::Array(500), 16)])
        .unwrap()
        .into_values()
        .next()
        .unwrap();
    let group = dense_plan(&pool, &tree);
    let (prep, run, _) = fresh(&pool, group.initialization_peak_bytes());
    drop(old_scope);
    let before = (usage(&pool), INITIALIZATIONS.get());
    assert!(matches!(
        prep.claim_prompt()
            .unwrap()
            .construct_dense_decoder_group(group, &run, extra.clone()),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), INITIALIZATIONS.get()), before);
    assert!(matches!(
        prep.claim_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    drop((extra, old_prep, old_run, prep, run, tables, tree));
}
