use super::*;
use crate::working_memory::*;
use eredu_core::{cache::LayerCachePolicy, *};
use std::{
    cell::Cell,
    num::NonZeroU8,
    panic::{AssertUnwindSafe, catch_unwind},
};

mod candidates;
mod funded_session;
mod host_summary;
mod ingress;
mod partition;
mod pending;
mod prefill;
mod retention;
mod routed;
mod segments;
mod span_exclusion;

thread_local! {
    static CLAIM_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static TRANSFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static PANIC_CLAIMS: Cell<bool> = const { Cell::new(false) };
    static PANIC_TRANSFER: Cell<bool> = const { Cell::new(false) };
}
pub(in crate::working_memory) fn before_claim_allocation() {
    CLAIM_ALLOCATIONS.set(CLAIM_ALLOCATIONS.get() + 1);
    assert!(!PANIC_CLAIMS.replace(false), "injected before claim buffer");
}
pub(in crate::working_memory) fn before_transfer_allocation() {
    TRANSFER_ALLOCATIONS.set(TRANSFER_ALLOCATIONS.get() + 1);
    assert!(
        !PANIC_TRANSFER.replace(false),
        "injected after source pin commit"
    );
}
fn unlimited() -> CaptureUsage {
    CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    }
}
fn point() -> ObservationPoint {
    ObservationPoint {
        path: "block.output".into(),
        node_id: "block-α".into(),
        meaning: "actual context values".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "context".into(),
                dimension: SymbolicDimension::Context,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    }
}
fn raw() -> CapturePlan {
    CapturePlan {
        schema_version: 1,
        selections: (0..3)
            .map(|index| CaptureSelection {
                id: format!("selected-{index}-é"),
                path: "block.output".into(),
                schedule: if index == 1 {
                    CaptureSchedule {
                        prefill: false,
                        first_prediction: 1,
                        every: 2,
                        ..Default::default()
                    }
                } else {
                    CaptureSchedule::default()
                },
                slices: vec![],
                transform: if index == 2 {
                    CaptureTransform::Preview { max_elements: 3 }
                } else {
                    CaptureTransform::FullTensor
                },
            })
            .collect(),
        limits: CaptureLimits {
            per_step: unlimited(),
            cumulative: unlimited(),
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
}
fn admit(
    raw: CapturePlan,
    point: ObservationPoint,
    maximum: u64,
    explicit: bool,
) -> SharedCapturePlan {
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let caps = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::FullTensor,
            CaptureTransformKind::Slice,
            CaptureTransformKind::Preview,
            CaptureTransformKind::Summary,
            CaptureTransformKind::RoutedUnits,
        ],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    SharedCapturePlan::new(
        if explicit {
            raw.admit_invocations(
                &catalog,
                &support,
                &caps,
                CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 3,
                    max_context: Some(12),
                    max_predictions: maximum,
                },
            )
        } else {
            raw.admit_with_text_origin(
                &catalog,
                &support,
                &caps,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 3,
                    max_predictions: maximum,
                },
                CaptureTextOrigin {
                    cached_positions: 2,
                },
            )
        }
        .unwrap(),
    )
}
fn source() -> SharedCapturePlan {
    admit(raw(), point(), 4, false)
}
fn plan(source: &SharedCapturePlan) -> CaptureRunHostPlan<'_> {
    CaptureRunHostPlan::prepare(source).unwrap()
}
pub(in crate::working_memory) fn ledger(pool: &MemoryLedger) -> (u64, u64, usize, usize, usize) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.registered + usage.reserved
            - usage.domains[0].registry_metadata
            - usage.funding.values().map(|s| s.control_floor).sum::<u64>(),
        usage
            .funding
            .values()
            .map(|s| s.host_held - s.control_floor)
            .sum(),
        usage.funding.values().map(|s| s.scopes).sum(),
        usage.reservations,
        usage.funding.len(),
    )
}
fn admission(pool: &MemoryLedger, bytes: u64) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 3,
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
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(3),
        4,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "portable parent-account fixture");
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
    crate::working_memory::memory_fixture::attribute_host_admission(
        &pool,
        Admission {
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
            state,
            requested_positions: 7,
            incremental_required_bytes: Some(bytes),
        },
    )
}
fn constructor_bytes() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
        crate::working_memory::memory_fixture::reservation_bytes(&pool, &admission(&pool, 0))
    })
}
pub(in crate::working_memory) fn capture_test_ledger(
    capacity: u64,
    existing: u64,
) -> Result<MemoryLedger, WorkingMemoryError> {
    // These capture fixtures admit at most three live request accounts and
    // sixteen separately prepared source/pin registries of at most four roots.
    // The requested payload remains the exact capture/native envelope.
    let metadata = 3u64
        .checked_mul(constructor_bytes().max(retention::constructor_bytes()))
        .unwrap()
        + 16 * (crate::working_memory::StoragePublicationLayout::<[usize; 8]>::new(4)
            .unwrap()
            .requested_bytes()
            + MemoryLedger::storage_metadata_control_bytes().unwrap());
    crate::working_memory::memory_fixture::host_ledger(
        if capacity == u64::MAX {
            capacity
        } else {
            capacity
                .checked_add(metadata)
                .ok_or(WorkingMemoryError::Overflow)?
        },
        existing,
    )
}
pub(in crate::working_memory) fn fresh(
    pool: &MemoryLedger,
    bytes: u64,
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &admission(pool, bytes),
        crate::working_memory::memory_fixture::resolved_host_limits(
            pool,
            pool.payload_effective_capacity().unwrap(),
        ),
    )
    .unwrap()
    .into_funding()
    .unwrap()
}

pub(in crate::working_memory) trait CaptureFundingFixture {
    fn adopt_capture_host_storage<K: Clone + Ord + Send + 'static>(
        &self,
        entries: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError>;
}
impl CaptureFundingFixture for WorkingMemoryFundingScope {
    fn adopt_capture_host_storage<K: Clone + Ord + Send + 'static>(
        &self,
        entries: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError> {
        assert!(std::mem::size_of::<K>() <= std::mem::size_of::<[usize; 8]>());
        let prepared =
            crate::working_memory::StoragePublicationLayout::<K>::new(4)?.fund(self.pool())?;
        let placement = self.pool().host_placement_handle();
        prepared.adopt_storage_individually(
            self,
            entries.into_iter().map(|(key, bytes)| {
                (
                    key,
                    crate::working_memory::StorageAllocation::new(bytes, placement.clone()),
                )
            }),
        )
    }
}

fn complete(mut tensor: ScheduledCaptureTensor<'_, '_>, value: f32) -> ClaimedCaptureTensor {
    while tensor.initialized_count() < tensor.len() {
        tensor.push_f32(value).unwrap();
    }
    tensor.finish().unwrap()
}
fn finish(frame: ScheduledCaptureStep<'_>) -> SharedCapturedStep {
    frame
        .finish(CaptureStepOutcome::Committed, unlimited(), unlimited(), 0.0)
        .unwrap()
}

#[test]
fn cumulative_plan_matches_actual_frames_context_growth_and_preview_plateau() {
    let source = source();
    let planned = plan(&source);
    assert_eq!(planned.claim_slots(), 16);
    let mut frames = 0;
    let mut tensors = 0;
    let mut full_sizes = Vec::new();
    let mut preview_sizes = Vec::new();
    for p in 0..4 {
        let phase = plan::phase(p);
        let frame =
            CaptureStepHostPlan::prepare(source.admission(), phase, p as u64, None).unwrap();
        frames += frame.initialization_peak_bytes();
        for i in 0..3 {
            if let Some(g) = frame.geometry(i).unwrap() {
                let tensor = CaptureTensorHostPlan::prepare(g).unwrap();
                tensors += tensor.initialization_peak_bytes();
                if i == 0 {
                    assert_eq!(tensor.geometry().shape(), &[5 + p, 2]);
                    full_sizes.push(tensor.initialization_peak_bytes());
                }
                if i == 2 {
                    preview_sizes.push(tensor.initialization_peak_bytes());
                }
            }
        }
    }
    assert_eq!(planned.frame_peak_bytes(), frames);
    assert_eq!(planned.tensor_peak_bytes(), tensors);
    assert_eq!(
        planned.initialization_peak_bytes(),
        frames + tensors + planned.control_peak_bytes()
    );
    assert!(full_sizes.windows(2).all(|p| p[1] == p[0] + 8));
    assert!(preview_sizes.windows(2).all(|p| p[1] == p[0]));
    assert!(planned.control_peak_bytes() >= 16 + size_of::<PreparedCaptureRun>() as u64);
}

#[test]
fn exact_and_one_short_commit_only_original_hold_before_claim_allocation() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    for bytes in [h - 1, h] {
        let pool = capture_test_ledger(h, 0).unwrap();
        let (r, run) = fresh(&pool, bytes);
        let before = ledger(&pool);
        let allocations = CLAIM_ALLOCATIONS.get();
        let result = run.prepare_capture_run(&r, plan(&source));
        if bytes < h {
            assert!(
                matches!(result, Err(CaptureRunHostError::Memory(WorkingMemoryError::DomainAllowanceExceeded { required_bytes, available_bytes, .. })) if required_bytes == h && available_bytes == bytes)
            );
            assert_eq!(ledger(&pool), before);
            assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
        } else {
            let bank = result.unwrap();
            assert_eq!(bank.claims.len(), 16);
            assert_eq!(bank.protected_bytes(), h);
            assert!(bank.source().same_storage(&source));
            assert_eq!(ledger(&pool), (h, h, 1, 1, 1));
            assert_eq!(CLAIM_ALLOCATIONS.get(), allocations + 1);
            drop(bank);
            assert_eq!(ledger(&pool), before);
        }
        drop((r, run));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn whole_future_hold_blocks_native_adoption_before_any_frame_and_no_child_double_charge() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 37, 0).unwrap();
    let (r, run) = fresh(&pool, h + 37);
    let native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let before = ledger(&pool);
    assert!(matches!(
        native.adopt_capture_host_storage([(1u32, 38)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded { .. })
    ));
    assert_eq!(ledger(&pool), before);
    let storage = native.adopt_capture_host_storage([(1u32, 37)]).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let tensor = complete(step.take_tensor(0).unwrap().prepare().unwrap(), 2.25);
    assert_eq!(ledger(&pool), before);
    step.record_tensor(tensor, TensorDtype::Bf16, CaptureUsage::default())
        .unwrap();
    let frame = finish(step);
    assert_eq!(ledger(&pool), before);
    drop((frame, bank, r, run));
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 37);
    drop(storage);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn concurrent_finished_outputs_keep_total_hold_until_last_tensor_alias() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut frames = Vec::new();
    let mut aliases = Vec::new();
    for p in 0..4 {
        let mut step = bank
            .begin_step(plan::phase(p), p as u64)
            .unwrap()
            .prepare()
            .unwrap();
        let tensor = complete(
            step.take_tensor(0).unwrap().prepare().unwrap(),
            p as f32 + 0.5,
        );
        aliases.push(tensor.observation().clone());
        step.record_tensor(tensor, TensorDtype::F32, CaptureUsage::default())
            .unwrap();
        frames.push(finish(step));
        assert_eq!(ledger(&pool).1, h);
    }
    drop((bank, source, r, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(frames);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    let last = aliases.pop().unwrap();
    drop(aliases);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    assert_eq!(last.shape(), &[8, 2]);
    let TensorObservationData::F32(values) = last.data() else {
        unreachable!()
    };
    assert!(values.iter().all(|v| *v == 3.5));
    drop(last);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn schedule_gaps_and_decode_only_still_emit_prefill_metadata_frame() {
    let mut raw = raw();
    raw.selections = vec![raw.selections.remove(1)];
    let source = admit(raw, point(), 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    for p in 0..4 {
        let mut step = bank
            .begin_step(plan::phase(p), p as u64)
            .unwrap()
            .prepare()
            .unwrap();
        assert_eq!(step.records().len(), 1);
        if p % 2 == 0 {
            assert!(matches!(
                step.records()[0].outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            ));
            assert!(matches!(
                step.take_tensor(0),
                Err(CaptureRunHostError::ClaimUnavailable { index: 0 })
            ));
        } else {
            let value = complete(step.take_tensor(0).unwrap().prepare().unwrap(), 1.);
            step.record_tensor(value, TensorDtype::F16, CaptureUsage::default())
                .unwrap();
        }
        let frame = finish(step);
        assert_eq!(frame.prediction_index(), p as u64);
        drop(frame);
    }
    drop((bank, r, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn spent_failed_dropped_and_terminal_claims_never_reissue() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    assert!(matches!(
        bank.begin_step(CapturePhase::Decode, 0),
        Err(CaptureRunHostError::Coordinate)
    ));
    assert_eq!(bank.spent_steps(), 0);
    drop(bank.begin_step(CapturePhase::Prefill, 0).unwrap());
    assert!(matches!(
        bank.begin_step(CapturePhase::Prefill, 0),
        Err(CaptureRunHostError::Coordinate)
    ));
    let mut step = bank
        .begin_step(CapturePhase::Decode, 1)
        .unwrap()
        .prepare()
        .unwrap();
    drop(step.take_tensor(0).unwrap());
    assert!(matches!(
        step.take_tensor(0),
        Err(CaptureRunHostError::ClaimUnavailable { .. })
    ));
    step.record_skip(
        1,
        CaptureSkipReason::Schedule,
        None,
        CaptureUsage::default(),
    )
    .unwrap();
    assert!(step.take_tensor(1).is_err());
    let mut partial = step.take_tensor(2).unwrap().prepare().unwrap();
    partial.push_f32(5.).unwrap();
    let mut partial = partial.finish().unwrap_err().into_builder().unwrap();
    assert_eq!(partial.initialized_count(), 1);
    partial.push_f32(6.).unwrap();
    partial.push_f32(7.).unwrap();
    let receipt = partial.finish().unwrap();
    assert!(step.take_tensor(2).is_err());
    step.record_tensor(receipt, TensorDtype::F32, CaptureUsage::default())
        .unwrap();
    let frame = finish(step);
    drop(frame);
    for p in 2..4 {
        drop(bank.begin_step(CapturePhase::Decode, p).unwrap());
    }
    assert!(bank.begin_step(CapturePhase::Decode, 4).is_err());
    assert_eq!(bank.spent_steps(), 4);
    assert_eq!(ledger(&pool).1, h);
}

#[test]
fn receipt_is_bound_to_exact_bank_and_coordinate_not_equal_plan_values() {
    let source = source();
    let equal = SharedCapturePlan::new(source.admission().clone());
    assert!(!equal.same_storage(&source));
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(3 * h, 0).unwrap();
    let (r, run) = fresh(&pool, 3 * h);
    let mut a = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut b = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut c = run.prepare_capture_run(&r, plan(&equal)).unwrap();
    let mut sa = a
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut sb = b
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut sc = c
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    for (index, target) in [(0, &mut sb), (2, &mut sc)] {
        let receipt = complete(sa.take_tensor(index).unwrap().prepare().unwrap(), 4.);
        assert!(matches!(
            target.record_tensor(receipt, TensorDtype::F32, CaptureUsage::default()),
            Err(CaptureRunHostError::ReceiptMismatch)
        ));
        assert!(matches!(
            target.records()[index].outcome,
            CaptureOutcome::Missing
        ));
    }
    drop((sb, sc, sa));
    let mut sa = a
        .begin_step(CapturePhase::Decode, 1)
        .unwrap()
        .prepare()
        .unwrap();
    let receipt = complete(sa.take_tensor(0).unwrap().prepare().unwrap(), 8.);
    drop(sa);
    let mut later = a
        .begin_step(CapturePhase::Decode, 2)
        .unwrap()
        .prepare()
        .unwrap();
    assert!(matches!(
        later.record_tensor(receipt, TensorDtype::F32, CaptureUsage::default()),
        Err(CaptureRunHostError::ReceiptMismatch)
    ));
}

#[test]
fn closed_or_quarantined_parent_blocks_fill_finish_and_future_claims() {
    for quarantine in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut run = Some(run);
        let mut bank = run
            .as_ref()
            .unwrap()
            .prepare_capture_run(&r, plan(&source))
            .unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let mut tensor = step.take_tensor(0).unwrap().prepare().unwrap();
        tensor.push_f32(3.).unwrap();
        if quarantine {
            drop(run.as_ref().unwrap().scope().unwrap());
        } else {
            drop(run.take());
        }
        assert!(matches!(
            tensor.push_f32(4.),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        let error = tensor.finish().unwrap_err();
        assert!(error.into_builder().is_err());
        let error = step
            .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
            .unwrap_err();
        assert!(matches!(
            error.error(),
            CaptureStepError::Memory(WorkingMemoryError::ExecutionFenced)
        ));
        drop(error);
        assert!(bank.begin_step(CapturePhase::Decode, 1).is_err());
        drop((bank, r, run));
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            if quarantine { h } else { 0 }
        );
    }
}

#[test]
fn claim_buffer_and_partial_payload_unwind_preserve_prior_outputs_and_native_scope() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let native = run.scope().unwrap();
    let before = ledger(&pool);
    PANIC_CLAIMS.set(true);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            drop(run.prepare_capture_run(&r, plan(&source)));
        }))
        .is_err()
    );
    assert_eq!(ledger(&pool), before);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let first = finish(
        bank.begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap(),
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let mut step = bank
                .begin_step(CapturePhase::Decode, 1)
                .unwrap()
                .prepare()
                .unwrap();
            let mut partial = step.take_tensor(0).unwrap().prepare().unwrap();
            partial.push_f32(7.25).unwrap();
            panic!("after real partial payload");
        }))
        .is_err()
    );
    assert_eq!(bank.spent_steps(), 2);
    assert_eq!(ledger(&pool).1, h);
    drop((bank, r, run));
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn source_domain_and_native_account_mismatch_reject_before_allocation_without_refund() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(2 * h + 17, 0).unwrap();
    let foreign = capture_test_ledger(17, 0).unwrap();
    let storage = pool.register_host_storage([(1u32, 17)]).unwrap();
    let foreign_storage = foreign.register_host_storage([(1u32, 17)]).unwrap();
    let (r, run) = fresh(&pool, h);
    let (other_r, other) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let mut wrong = other.scope().unwrap();
    let allocations = CLAIM_ALLOCATIONS.get();
    assert!(matches!(
        run.prepare_capture_run(&other_r, plan(&source)),
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let before = ledger(&pool);
    let allocations = TRANSFER_ALLOCATIONS.get();
    assert!(
        step.take_tensor(0)
            .unwrap()
            .prepare_with_source(&mut wrong, storage.clone())
            .is_err()
    );
    assert!(
        step.take_tensor(2)
            .unwrap()
            .prepare_with_source(&mut native, foreign_storage.clone())
            .is_err()
    );
    assert_eq!(TRANSFER_ALLOCATIONS.get(), allocations);
    assert_eq!(ledger(&pool), before);
    assert!(step.take_tensor(0).is_err());
    assert!(step.take_tensor(2).is_err());
    drop(step);
    native.certify().unwrap();
    wrong.certify().unwrap();
    drop((bank, r, run, other_r, other, storage, foreign_storage));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
}

#[test]
fn exclusive_exact_scope_keeps_pins_after_sibling_certification_and_host_retirement() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 19, 0).unwrap();
    let storage = pool.register_host_storage([(1u32, 19)]).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let sibling = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut tensor = step
        .take_tensor(0)
        .unwrap()
        .prepare_with_source(&mut native, storage)
        .unwrap();
    sibling.certify().unwrap(); // `native` is inaccessible until the partial owner retires.
    tensor.validate().unwrap();
    tensor.push_f32(-2.5).unwrap();
    let mut tensor = tensor.finish().unwrap_err().into_builder().unwrap();
    assert_eq!(tensor.initialized_count(), 1);
    while tensor.initialized_count() < tensor.len() {
        tensor.push_f32(1.25).unwrap();
    }
    let receipt = tensor.finish().unwrap();
    let alias = receipt.observation().clone();
    step.record_tensor(receipt, TensorDtype::F32, CaptureUsage::default())
        .unwrap();
    drop(finish(step));
    drop((bank, alias, r, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), h + 19);
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn constructor_rollback_restores_prior_pins_but_execution_failure_keeps_enlarged_bundle() {
    for fail_constructor in [true, false] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h + 30, 0).unwrap();
        let a = pool.register_host_storage([(1u32, 11)]).unwrap();
        let b = pool.register_host_storage([(2u32, 19)]).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut native = run.scope().unwrap();
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        drop(
            step.take_tensor(0)
                .unwrap()
                .prepare_with_source(&mut native, a.clone())
                .unwrap(),
        );
        if fail_constructor {
            PANIC_TRANSFER.set(true);
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    drop(
                        step.take_tensor(2)
                            .unwrap()
                            .prepare_with_source(&mut native, b.clone()),
                    );
                }))
                .is_err()
            );
        } else {
            let mut partial = step
                .take_tensor(2)
                .unwrap()
                .prepare_with_source(&mut native, b.clone())
                .unwrap();
            partial.push_f32(6.).unwrap();
            drop(partial); // Entered worker: complete B pins must remain.
        }
        assert!(step.take_tensor(2).is_err());
        drop(step);
        drop((bank, a, b));
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            h + if fail_constructor { 11 } else { 30 }
        );
        drop((r, run));
        if fail_constructor {
            native.certify().unwrap();
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        } else {
            drop(native); // Quarantine retains both source inventories.
            assert_eq!(pool.payload_used_bytes().unwrap(), h + 30);
        }
    }
}

#[test]
fn later_scope_full_pins_survive_earlier_partial_inventory_quarantine() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 30, 0).unwrap();
    let a = pool.register_host_storage([(1u32, 11)]).unwrap();
    let b = pool.register_host_storage([(2u32, 19)]).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut early = run.scope().unwrap();
    let mut later = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    drop(
        step.take_tensor(0)
            .unwrap()
            .prepare_with_source(&mut early, a.clone())
            .unwrap(),
    );
    drop(
        step.take_tensor(2)
            .unwrap()
            .prepare_with_source(&mut later, a.clone())
            .unwrap(),
    );
    drop(step);
    let mut step = bank
        .begin_step(CapturePhase::Decode, 1)
        .unwrap()
        .prepare()
        .unwrap();
    let mut partial = step
        .take_tensor(0)
        .unwrap()
        .prepare_with_source(&mut later, b.clone())
        .unwrap();
    partial.push_f32(2.).unwrap();
    drop(early);
    assert!(matches!(
        partial.validate(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    drop(partial);
    drop(step);
    drop((bank, a, b, later, r, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), h + 30);
    assert!(pool.acquire_unquoted().is_err());
}

#[test]
fn complete_source_origin_health_is_rechecked_after_host_constructor() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 31, 0).unwrap();
    let (source_r, source_run) = fresh(&pool, 31);
    let source_native = source_run.scope().unwrap();
    let storage = source_native
        .adopt_capture_host_storage([(1u32, 31)])
        .unwrap()
        .into_values()
        .next()
        .unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut tensor = step
        .take_tensor(0)
        .unwrap()
        .prepare_with_source(&mut native, storage)
        .unwrap();
    tensor.push_f32(3.).unwrap();
    drop(source_native);
    assert!(matches!(
        tensor.push_f32(4.),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(tensor.finish().unwrap_err().into_builder().is_err());
    drop(step);
    drop((bank, r, run, source_r, source_run, native));
    assert_eq!(pool.payload_used_bytes().unwrap(), h + 31);
}

#[test]
fn zero_empty_unknown_overflow_and_explicit_invocation_are_not_fabricated_schedules() {
    let explicit = admit(raw(), point(), 4, true);
    assert!(matches!(
        CaptureRunHostPlan::prepare(&explicit),
        Err(CaptureRunHostError::ExplicitInvocation)
    ));
    let mut unknown = point();
    unknown.axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Unknown;
    let unknown = admit(raw(), unknown, 4, false);
    assert!(matches!(
        CaptureRunHostPlan::prepare(&unknown),
        Err(CaptureRunHostError::Step(CaptureStepError::Geometry(
            CaptureTensorGeometryError::UnknownShape
        )))
    ));
    let mut empty = raw();
    empty.selections.clear();
    for source in [admit(empty, point(), 4, false)] {
        let p = plan(&source);
        assert_eq!(
            (p.frame_peak_bytes(), p.tensor_peak_bytes(), p.claim_slots()),
            (0, 0, 0)
        );
        let h = p.initialization_peak_bytes(); // Actual control owner remains priced.
        assert_eq!(h, p.control_peak_bytes());
        let pool = capture_test_ledger(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&r, p).unwrap();
        assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
        assert!(bank.claims.is_empty());
        drop((bank, r, run));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    let mut zero = point();
    zero.axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(0);
    let zero = admit(raw(), zero, 4, false);
    let h = plan(&zero).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&zero)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let tensor = step.take_tensor(0).unwrap().prepare().unwrap();
    assert!(tensor.is_empty());
    let receipt = tensor.finish().unwrap();
    assert_eq!(receipt.observation().shape(), &[5, 0]);
    step.record_tensor(receipt, TensorDtype::F32, CaptureUsage::default())
        .unwrap();
    drop(finish(step));
    // Overflow is rejected from the actual finite extent before enumeration.
    let mut raw = raw();
    for s in &mut raw.selections {
        s.schedule.end_prediction = Some(2);
    }
    let mut scalar = point();
    scalar.axes = Some(vec![]);
    let huge = admit(raw, scalar, u64::MAX - 8, false);
    assert!(matches!(
        CaptureRunHostPlan::prepare(&huge),
        Err(CaptureRunHostError::Memory(WorkingMemoryError::Overflow))
    ));
}

mod delivery;

mod token_scores;

mod intervention_schedule;
