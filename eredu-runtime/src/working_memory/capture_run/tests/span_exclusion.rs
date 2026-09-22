//! Portable accounting conformance, not native opening/completion evidence.
//! Receipt: actual cold equations plus strict core admission. Context/ticket:
//! existing internal retention fixture; canonical issuance is tested separately
//! through SessionPrefill. No production activation or native work occurs here.
use super::*;
use crate::{
    HostMetadataKey, HostSlotTable,
    inspection::PrefillChunkRetentionContext,
    prefill::PrefillChunk,
    working_memory::{
        funding::DecoderCopySource, residual::RegisteredStoragePin,
        storage::publish_dense_host_slots,
    },
};
use eredu_nn::{Error as NnError, Tensor, workspace::*};

#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, NnError> {
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|v| v.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "portable fixed F32 allocation".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, NnError> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "portable no-host operation".into(),
        }))
    }
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn chunk() -> PrefillChunk {
    PrefillChunk {
        input: 0..1,
        position: 2,
        output: OutputDemand::StateOnly,
    }
}
fn context<'a>(
    r: &'a InferenceRequest,
    c: &'a PrefillChunk,
    n: u64,
) -> PrefillChunkRetentionContext<'a> {
    PrefillChunkRetentionContext::new(r, c, DistributedCommitEpoch::new(n).unwrap())
}
fn quote(pool: &MemoryLedger, h: u64) -> IncrementalInferenceQuote {
    let g = geometry();
    let ctx = WorkspaceContext::new(Facts);
    let registered = RegisteredWorkspaceStorage::<u32>::bind(
        pool,
        &ctx,
        std::iter::empty::<(u32, eredu_nn::workspace::WorkspaceExistingStorage)>(),
    )
    .unwrap();
    let report = quote_inference_workspace(g, |_| {
        ctx.begin_state_span(std::iter::empty::<&WorkspaceTensor>())?;
        let value = WorkspaceTensor::full_f32(0.75, &[4], &ctx)?;
        ctx.report(&[value])
    })
    .unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(5),
        4,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let b = |n| WorkspaceBound::bounded(n, "portable exact fixture contribution");
    ResidualInferenceQuote::compose(
        &report,
        state,
        ExecutionWorkspaceEstimate {
            geometry: g,
            activations: b(0),
            attention: b(0),
            vocabulary: b(0),
            state_update: b(0),
            materialization: b(0),
            retained: b(h + 256),
            physical_domains: Some({
                let mut domains = crate::working_memory::memory_fixture::host_workspace(pool, g, 0);
                domains.retained =
                    crate::working_memory::memory_fixture::host_requirements(pool, h + 256);
                domains
            }),
        },
        &registered,
    )
    .unwrap()
    .into_incremental()
}
fn reserve(
    pool: &MemoryLedger,
    q: IncrementalInferenceQuote,
) -> (
    InferenceRequest,
    WorkingMemoryFundingRun,
    IncrementalInferenceQuote,
) {
    reserve_sealed(pool, q.with_span_workspace().unwrap())
}
fn reserve_sealed(
    pool: &MemoryLedger,
    q: IncrementalInferenceQuote,
) -> (
    InferenceRequest,
    WorkingMemoryFundingRun,
    IncrementalInferenceQuote,
) {
    let g = geometry();
    let cap = ModelCapabilities {
        effective_model_type: "portable active-span fixture".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let (r, q) = plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &cap,
        AdmissionRequest {
            input: InputTokenCount::text(5),
            max_output_tokens: 4,
            batch_size: 1,
            additional_headroom: Default::default(),
            memory_limits: Default::default(),
        },
        g,
        pool.configured_limits().clone(),
        |_| Ok(q.clone()),
    )
    .unwrap();
    let (r, run) = r.into_funding().unwrap();
    (r.into(), run, q)
}
fn headroom(pool: &MemoryLedger, r: &InferenceRequest) -> u64 {
    let u = pool.0.usage.lock().unwrap();
    let s = &u.funding[&r.memory_reservation().0.funding.unwrap()];
    s.remaining - s.host_held
}
fn bank(
    run: &WorkingMemoryFundingRun,
    r: &InferenceRequest,
    source: &SharedCapturePlan,
) -> PreparedCaptureRun {
    run.prepare_capture_run(r.memory_reservation(), plan(source))
        .unwrap()
}
fn ticket(
    reg: crate::inspection::PreparedPrefillChunkRetention,
    cx: &PrefillChunkRetentionContext<'_>,
) -> crate::inspection::SettledPrefillChunkRetention {
    reg.validate_commit(cx, Some(DistributedCommitOutcome::Committed(cx.epoch())))
        .unwrap();
    reg.into_settled()
}
fn fenced<T>(r: Result<T, WorkingMemoryError>) {
    assert!(matches!(r, Err(WorkingMemoryError::ExecutionFenced)));
}

#[test]
fn accepted_span_exact_headroom_and_last_alias_retirement_are_atomic() {
    for short in [false, true] {
        let pool = capture_test_ledger(4_000_000, 0).unwrap();
        let source = source();
        let (r, run, q) = reserve(
            &pool,
            quote(&pool, plan(&source).initialization_peak_bytes()),
        );
        let bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let c = chunk();
        let cx = context(&r, &c, 7);
        let (segment, reg) = bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &cx)
            .unwrap();
        let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
        let n = receipt
            .span_bytes(&InferenceWorkspaceSpan::Prefill(c.clone()))
            .unwrap();
        assert_eq!(n, 16);
        let earlier = headroom(&pool, &r) - n + u64::from(short);
        let mut storage = native
            .adopt_capture_host_storage([(41u32, earlier)])
            .unwrap();
        let alias = storage.remove(&41).unwrap();
        let escaped = alias.clone();
        let before = ledger(&pool);
        let result = segment.activate_reserved_span(&mut native, &receipt, &cx);
        if short {
            assert!(matches!(
                result,
                Err(WorkingMemoryError::DomainAllowanceExceeded {
                    required_bytes: 16,
                    available_bytes: 15,
                    ..
                })
            ));
            assert_eq!(ledger(&pool), before);
            drop(alias);
            assert_eq!(headroom(&pool, &r), 15);
            assert!(
                segment
                    .activate_reserved_span(&mut native, &receipt, &cx)
                    .is_err()
            );
            drop(escaped);
            assert_eq!(headroom(&pool, &r), 15 + earlier);
            segment
                .activate_reserved_span(&mut native, &receipt, &cx)
                .unwrap();
        } else {
            result.unwrap();
            assert_eq!(ledger(&pool), before);
            drop((alias, escaped));
        }
        let allocated = native
            .adopt_capture_host_storage([(42u32, 16), (43, 0)])
            .unwrap();
        fenced(run.scope());
        assert!(
            segment
                .activate_reserved_span(&mut native, &receipt, &cx)
                .is_err()
        );
        let t = ticket(reg, &cx);
        let before = ledger(&pool);
        let parcel = segment.take_settled_sources(&mut native, &t).unwrap();
        assert_eq!(ledger(&pool), before);
        assert!(segment.take_settled_sources(&mut native, &t).is_err());
        run.scope().unwrap().certify().unwrap();
        drop((parcel, t, segment, allocated, storage));
        native.certify().unwrap();
        drop((bank, r, run, q, source));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn sibling_scopes_new_host_copy_holds_and_zero_publications_cannot_spend() {
    let pool = capture_test_ledger(4_000_000, 0).unwrap();
    let source = source();
    let (r, run, q) = reserve(
        &pool,
        quote(&pool, plan(&source).initialization_peak_bytes()),
    );
    let bank = bank(&run, &r, &source);
    let mut native = run.scope().unwrap();
    let sibling = run.scope().unwrap();
    let mut sampler = run.sampler_scope().unwrap();
    let c = chunk();
    let cx = context(&r, &c, 3);
    let (segment, reg) = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
    segment
        .activate_reserved_span(&mut native, &receipt, &cx)
        .unwrap();
    let own_storage = native.adopt_capture_host_storage([(20u32, 8)]).unwrap();
    let before = ledger(&pool);
    for entries in [vec![], vec![(8u32, 0)], vec![(9, 1)], vec![(20, 8)]] {
        fenced(sibling.adopt_capture_host_storage(entries));
    }
    fenced(run.scope());
    fenced(run.sampler_scope());
    fenced(sampler.hold_sampler_payload(1));
    let rsv = r.memory_reservation();
    fenced(run.open_pending_input_scope(rsv, 1));
    assert!(matches!(
        run.prepare_capture_run(rsv, plan(&source)),
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    let empty = pool.register_storage::<u32>([]).unwrap();
    fenced(run.open_no_decoder_prompt_scope(rsv, empty.clone()));
    fenced(run.open_dense_prompt_scopes(
        rsv,
        DecoderCopySource::Registered(&empty),
        &empty,
        RegisteredStoragePin::new(empty.clone()),
        1,
    ));
    fenced(run.open_grouped_dense_prompt_scopes(
        rsv,
        &[DecoderCopySource::Registered(&empty)],
        &[1],
        &empty,
        RegisteredStoragePin::new(empty.clone()),
    ));
    assert_eq!(ledger(&pool), before);
    // Reading a source into a separately admitted account remains independent.
    let (copy_run, copy_scope) = pool
        .open_workspace_copy_account(
            &own_storage[&20],
            RegisteredStoragePin::new(own_storage[&20].clone()),
            &InferenceExecutionIdentity::default(),
            &crate::working_memory::memory_fixture::host_requirements(&pool, 8),
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 4_000_000),
        )
        .unwrap();
    let values = copy_scope.adopt_capture_host_storage([(7u64, 8)]).unwrap();
    copy_scope.certify().unwrap();
    drop((values, copy_run));
    assert_eq!(ledger(&pool).0, before.0);
    let t = ticket(reg, &cx);
    drop(segment.take_settled_sources(&mut native, &t).unwrap());
    drop((t, segment, empty, sampler, own_storage));
    sibling.certify().unwrap();
    native.certify().unwrap();
    drop((bank, r, run, q, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn wrong_receipt_scope_chunk_epoch_and_unstamped_channel_never_activate() {
    let pool = capture_test_ledger(4_000_000, 0).unwrap();
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let (r, run, q) = reserve(&pool, quote(&pool, h));
    let (other, other_run, other_q) = reserve(&pool, quote(&pool, h));
    let bank_a = bank(&run, &r, &source);
    let bank_b = bank(&other_run, &other, &source);
    let mut native = run.scope().unwrap();
    let mut sibling = run.scope().unwrap();
    let mut other_native = other_run.scope().unwrap();
    let c = chunk();
    let cx = context(&r, &c, 9);
    let (mut segment, reg) = bank_a
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let (foreign, foreign_reg) = bank_b
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut other_native, &context(&other, &c, 9))
        .unwrap();
    let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
    let wrong = other_q
        .reserved_span_workspace(other.memory_reservation())
        .unwrap();
    let before = ledger(&pool);
    assert!(
        segment
            .activate_reserved_span(&mut native, &wrong, &cx)
            .is_err()
    );
    assert!(
        segment
            .activate_reserved_span(&mut sibling, &receipt, &cx)
            .is_err()
    );
    assert!(
        segment
            .activate_reserved_span(&mut other_native, &receipt, &cx)
            .is_err()
    );
    assert!(
        segment
            .activate_reserved_span(&mut native, &receipt, &context(&r, &c, 10))
            .is_err()
    );
    let bad = PrefillChunk {
        position: 3,
        ..c.clone()
    };
    assert!(
        segment
            .activate_reserved_span(&mut native, &receipt, &context(&r, &bad, 9))
            .is_err()
    );
    let unstamped = bank_a.custody.begin_source_segment(&mut sibling).unwrap();
    assert!(
        unstamped
            .activate_reserved_span(&mut sibling, &receipt, &cx)
            .is_err()
    );
    assert_eq!(ledger(&pool), before);
    segment
        .activate_reserved_span(&mut native, &receipt, &cx)
        .unwrap();
    let foreign_t = ticket(foreign_reg, &context(&other, &c, 9));
    assert!(
        segment
            .take_settled_sources(&mut native, &foreign_t)
            .is_err()
    );
    fenced(segment.retire_after_settled_boundary(&mut native));
    fenced(run.scope());
    let t = ticket(reg, &cx);
    drop(segment.take_settled_sources(&mut native, &t).unwrap());
    drop(
        foreign
            .take_settled_sources(&mut other_native, &foreign_t)
            .unwrap(),
    );
    drop((t, foreign_t, foreign, segment, unstamped));
    native.certify().unwrap();
    sibling.certify().unwrap();
    other_native.certify().unwrap();
    drop((bank_a, bank_b, r, run, q, other, other_run, other_q, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HostKey(HostMetadataKey);
impl HostSlotStorageKey for HostKey {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        Some(&self.0)
    }
}
#[test]
fn previously_held_host_conversion_preserves_unheld_headroom_during_exclusion() {
    for count in [0usize, 3] {
        let pool = capture_test_ledger(4_000_000, 0).unwrap();
        let source = source();
        let original = HostSlotTable::new(
            (0..count)
                .map(|i| 11 + i as u32 * 7)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let original_key = HostKey(original.metadata().identity().registry_key().clone());
        let registered = pool
            .register_host_storage([(original_key, original.metadata().capacity_bytes().unwrap())])
            .unwrap();
        let host_plan = original
            .prepare_copy_slots()
            .unwrap()
            .for_dense_destination::<u32>()
            .unwrap();
        let (retained, protected) = (
            host_plan.retained_bytes(),
            host_plan.initialization_peak_bytes(),
        );
        let controls =
            crate::working_memory::storage::ordinary_dense_preparation_bytes::<u32, u32, HostKey>()
                .unwrap();
        let (r, run, q) = reserve(
            &pool,
            quote(&pool, plan(&source).initialization_peak_bytes() + controls),
        );
        let (execution, mut host, extra_native) = run
            .open_dense_prompt_scopes(
                r.memory_reservation(),
                DecoderCopySource::Registered(&registered),
                &registered,
                RegisteredStoragePin::new(registered.clone()),
                protected,
            )
            .unwrap();
        extra_native.certify().unwrap();
        let preparation =
            crate::working_memory::storage::prepare_dense_host_metadata::<u32, u32, HostKey>(
                &host, &execution,
            )
            .unwrap();
        let identity = crate::HostMetadataIdentity::prepared_host(&preparation).unwrap();
        let mut builder = host_plan.initialize();
        for value in original.slots() {
            builder.push(*value).unwrap();
        }
        let completed = builder
            .finish_with_preparation(Some((identity, &preparation)))
            .unwrap();
        let bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let c = chunk();
        let cx = context(&r, &c, 4);
        let (segment, reg) = bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &cx)
            .unwrap();
        let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
        segment
            .activate_reserved_span(&mut native, &receipt, &cx)
            .unwrap();
        let before = (ledger(&pool), headroom(&pool, &r));
        let key = HostKey(completed.metadata().identity().registry_key().clone());
        publish_dense_host_slots(
            &completed,
            &mut host,
            &execution,
            retained,
            protected,
            key.clone(),
            Some(&preparation),
        )
        .unwrap();
        drop(preparation);
        assert_eq!(headroom(&pool, &r), before.1);
        assert_eq!(ledger(&pool).0, before.0.0);
        assert_eq!(ledger(&pool).1, before.0.1 - retained);
        let pin = pool.pin_registered_storage([(key, retained)]).unwrap();
        assert_eq!(
            completed.iter().copied().collect::<Vec<_>>(),
            original.slots()
        );
        fenced(run.scope());
        let t = ticket(reg, &cx);
        drop(segment.take_settled_sources(&mut native, &t).unwrap());
        drop((t, segment, completed, pin, host));
        native.certify().unwrap();
        drop((bank, r, run, q, source, registered, original));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn every_joined_or_segment_source_origin_is_rechecked_at_activation_and_ticket_take() {
    for mode in 0..3 {
        let pool = capture_test_ledger(4_000_000, 0).unwrap();
        let source = source();
        let (origin_r, origin_run) = fresh(&pool, 8);
        let origin_native = origin_run.scope().unwrap();
        let poison = origin_run.scope().unwrap();
        let mut registered = origin_native
            .adopt_capture_host_storage([(71u32, 8), (72, 0)])
            .unwrap();
        origin_native.certify().unwrap();
        let charged = registered.remove(&71).unwrap();
        let zero = registered.remove(&72).unwrap();
        let healthy = pool.register_host_storage([(75u64, 0)]).unwrap();
        let mut q = quote(&pool, plan(&source).initialization_peak_bytes());
        if mode == 0 {
            q = q
                .with_registered_sources(zero.clone())
                .unwrap()
                .with_registered_sources(charged.clone())
                .unwrap()
                .with_registered_sources(healthy.clone())
                .unwrap();
        }
        let (r, run, q) = reserve(&pool, q);
        let bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let c = chunk();
        let cx = context(&r, &c, 8);
        let (segment, reg) = bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &cx)
            .unwrap();
        if mode != 0 {
            bank.custody
                .bind_segment_source(&mut native, &segment, &zero)
                .unwrap()
                .commit();
            bank.custody
                .bind_segment_source(&mut native, &segment, &charged)
                .unwrap()
                .commit();
            bank.custody
                .bind_segment_source(&mut native, &segment, &healthy)
                .unwrap()
                .commit();
        }
        let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
        let t = ticket(reg, &cx);
        if mode == 2 {
            segment
                .activate_reserved_span(&mut native, &receipt, &cx)
                .unwrap();
        }
        drop(poison);
        let before = ledger(&pool);
        if mode == 2 {
            fenced(segment.take_settled_sources(&mut native, &t));
            fenced(run.scope());
        } else {
            fenced(segment.activate_reserved_span(&mut native, &receipt, &cx));
        }
        assert_eq!(ledger(&pool), before);
        if mode == 2 {
            fenced(native.certify());
        } else {
            native.certify().unwrap();
        }
        drop((
            t, segment, bank, r, run, q, source, charged, zero, registered, healthy, origin_r,
            origin_run,
        ));
        assert!(
            pool.payload_used_bytes().unwrap() >= 8,
            "quarantined origin is never refunded"
        );
    }
}

#[test]
fn abandoned_or_failed_active_scope_keeps_exclusion_and_recovery_publication() {
    for explicit_certify in [false, true] {
        let pool = capture_test_ledger(4_000_000, 0).unwrap();
        let source = source();
        let (r, run, q) = reserve(
            &pool,
            quote(&pool, plan(&source).initialization_peak_bytes()),
        );
        let mut bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let sibling = run.scope().unwrap();
        let failed = run.scope().unwrap();
        let recovery_preparation = StoragePublicationLayout::<u32>::new(2)
            .unwrap()
            .fund(&pool)
            .unwrap();
        let c = chunk();
        let cx = context(&r, &c, 12);
        let (segment, reg) = bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &cx)
            .unwrap();
        let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
        segment
            .activate_reserved_span(&mut native, &receipt, &cx)
            .unwrap();
        // Already held original H still builds and writes the scheduled target.
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let mut target = step.take_tensor(0).unwrap().prepare().unwrap();
        while target.initialized_count() < target.len() {
            target.push_f32(0.75).unwrap();
        }
        drop(target.finish().unwrap());
        drop(step);
        let before = ledger(&pool).0;
        // Failed/Busy backend retirement does not invoke take. Dropping its
        // ephemeral association also cannot clear the native slot/marker.
        drop((segment, reg));
        drop(failed);
        let recovery = recovery_preparation
            .adopt_storage_individually(
                &native,
                [(91u32, 8), (92, 0)].map(|(key, bytes)| {
                    (
                        key,
                        StorageAllocation::new(bytes, pool.host_placement_handle()),
                    )
                }),
            )
            .unwrap();
        assert_eq!(
            ledger(&pool).0,
            before,
            "exact active recovery publication shifts existing coverage"
        );
        fenced(run.scope());
        fenced(sibling.adopt_capture_host_storage([(1u32, 0)]));
        if explicit_certify {
            fenced(native.certify());
        } else {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    drop(native);
                    panic!("active native failure");
                }))
                .is_err()
            );
        }
        fenced(run.scope());
        // Sibling recovery cannot consume this active channel's remaining funds.
        fenced(sibling.adopt_capture_host_storage([(2u32, 1)]));
        sibling.certify().unwrap();
        drop((recovery, bank, r, run, q, source));
        assert_eq!(pool.payload_used_bytes().unwrap(), before);
    }
}

#[test]
fn active_span_blocks_new_plan_host_hold_without_losing_quote_then_exact_ticket_allows_it() {
    let pool = capture_test_ledger(4_000_000, 0).unwrap();
    let source = source();
    let (r, run, q) = reserve(
        &pool,
        quote(&pool, plan(&source).initialization_peak_bytes()),
    );
    let bank = bank(&run, &r, &source);
    let mut native = run.scope().unwrap();
    let c = chunk();
    let cx = context(&r, &c, 71);
    let (segment, reg) = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
    segment
        .activate_reserved_span(&mut native, &receipt, &cx)
        .unwrap();
    drop(receipt);
    let before = ledger(&pool);
    let original = q.span_workspace().plan().clone();
    let error = q
        .into_funded_span_workspace(&run, r.memory_reservation())
        .unwrap_err();
    assert!(matches!(error.cause(), WorkingMemoryError::ExecutionFenced));
    assert_eq!(ledger(&pool), before);
    let (q, _) = error.into_parts();
    assert!(q.span_workspace().plan().same_plan(&original));
    let t = ticket(reg, &cx);
    let parcel = segment.take_settled_sources(&mut native, &t).unwrap();
    let old = headroom(&pool, &r);
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    let (owner, witness) = q
        .into_funded_span_workspace(&run, r.memory_reservation())
        .unwrap();
    assert!(witness.is_none());
    assert_eq!(headroom(&pool, &r), old - p);
    native.certify().unwrap();
    drop((owner, original, parcel, t, segment, bank, r, run, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

mod text_controls;

#[test]
fn native_partition_alias_publication_obeys_the_actual_active_span() {
    use crate::working_memory::{
        funding::native_partition::test_receipt,
        storage::native_publication::{
            PreparedNativePublication, test_existing_native, test_native,
        },
    };
    let pool = capture_test_ledger(4_000_000, 0).unwrap();
    let source = source();
    let (r, run, q) = reserve(
        &pool,
        quote(&pool, plan(&source).initialization_peak_bytes()),
    );
    let partition = run.take_native_partition(test_receipt(&run, 32)).unwrap();
    let namespace = pool.register_host_storage([(0u32, 0)]).unwrap();
    let bank = bank(&run, &r, &source);
    let mut native = run.scope().unwrap();
    let sibling = run.scope().unwrap();
    let mut first = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_native(77u32, 16, &partition)],
    );
    first.publish(&native).unwrap();
    let first_owner = first.take(0).unwrap();
    let c = chunk();
    let cx = context(&r, &c, 31);
    let (segment, reg) = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let receipt = q.reserved_span_workspace(r.memory_reservation()).unwrap();
    segment
        .activate_reserved_span(&mut native, &receipt, &cx)
        .unwrap();
    let before = ledger(&pool);
    let mut wrong = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_native(77u32, 16, &partition)],
    );
    fenced(wrong.publish(&sibling));
    assert!(wrong.take(0).is_none());
    assert_eq!(ledger(&pool), before);
    let mut matching = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_native(77u32, 16, &partition)],
    );
    matching.publish(&native).unwrap();
    let alias = matching.take(0).unwrap();
    assert_eq!(ledger(&pool).0, before.0);
    // The canonical-only route derives the same origin, but it still needs
    // this exact active span rather than another scope of the same account.
    let mut wrong_existing = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_existing_native(77u32, 16)],
    );
    let before_existing = ledger(&pool);
    fenced(wrong_existing.publish(&sibling));
    assert!(wrong_existing.take(0).is_none());
    assert_eq!(ledger(&pool), before_existing);
    let mut matching_existing = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_existing_native(77u32, 16)],
    );
    matching_existing.publish(&native).unwrap();
    let existing_alias = matching_existing.take(0).unwrap();
    assert_eq!(ledger(&pool).0, before_existing.0);
    let t = ticket(reg, &cx);
    drop(segment.take_settled_sources(&mut native, &t).unwrap());
    drop((
        t,
        segment,
        first,
        wrong,
        matching,
        first_owner,
        alias,
        wrong_existing,
        matching_existing,
        existing_alias,
    ));
    sibling.certify().unwrap();
    native.certify().unwrap();
    drop((bank, r, run, q, source, namespace));
    assert_eq!(pool.payload_used_bytes().unwrap(), 32);
    drop(partition);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn projected_host_binding_keeps_distinct_paid_owners_and_rejects_foreign_sources() {
    let pool = capture_test_ledger(4_000_000, 0).unwrap();
    let source = source();
    let equal_source = super::source();
    assert!(!source.same_storage(&equal_source));
    let h = plan(&source).initialization_peak_bytes();
    let (request, run, quote) = reserve(&pool, quote(&pool, h * 2));
    let original = bank(&run, &request, &source);
    let destination = bank(&run, &request, &source);
    assert!(!original.custody.same_schedule(&destination.custody));
    let (foreign_request, foreign_run, foreign_quote) = reserve(&pool, self::quote(&pool, h));
    let foreign = bank(&foreign_run, &foreign_request, &source);
    let mut native = run.scope().unwrap();
    let input = chunk();
    let cx = context(&request, &input, 41);
    let (mut segment, registration) = original
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let registered = pool.register_host_storage([(501u32, 16)]).unwrap();
    let before = ledger(&pool);
    assert!(matches!(
        destination
            .custody
            .bind_segment_source(&mut native, &segment, &registered),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        destination.custody.bind_projected_segment_source(
            &mut native,
            &segment,
            &registered,
            equal_source.admission()
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        foreign.custody.bind_projected_segment_source(
            &mut native,
            &segment,
            &registered,
            source.admission()
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        ledger(&pool),
        before,
        "rejection changes no original hold or spend"
    );
    destination
        .custody
        .bind_projected_segment_source(&mut native, &segment, &registered, source.admission())
        .unwrap()
        .commit();
    assert_eq!(
        ledger(&pool),
        before,
        "binding supplies no new destination or native grant"
    );
    segment.validate_native_scope(&native).unwrap();
    drop(registered);
    let settled = ticket(registration, &cx);
    let parcel = segment.take_settled_sources(&mut native, &settled).unwrap();
    native.certify().unwrap();
    drop((
        segment,
        settled,
        original,
        foreign,
        foreign_quote,
        foreign_request,
        foreign_run,
    ));
    drop((quote, request, run));
    assert!(
        pool.payload_used_bytes().unwrap() >= h + 16,
        "independent Host and source parcel retain their owners"
    );
    drop(destination);
    assert!(pool.payload_used_bytes().unwrap() >= 16);
    drop(parcel);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
