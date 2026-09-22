use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use std::cell::Cell;
thread_local! {
    static EVALUATIONS: Cell<usize> = const { Cell::new(0) };
    static FAIL_EVALUATION: Cell<bool> = const { Cell::new(false) };
    static FAIL_PUBLICATION: Cell<bool> = const { Cell::new(false) };
    static PANIC_PUBLICATION: Cell<bool> = const { Cell::new(false) };
}
#[derive(Debug, thiserror::Error)]
#[error("injected active capture ingress failure")]
struct InjectedIngress;
pub(super) fn before_evaluation() -> Result<(), Error> {
    EVALUATIONS.set(EVALUATIONS.get() + 1);
    if FAIL_EVALUATION.replace(false) {
        Err(Error::Other(Box::new(InjectedIngress)))
    } else {
        Ok(())
    }
}
pub(super) fn after_source_publication() -> Result<(), Error> {
    assert!(
        !PANIC_PUBLICATION.replace(false),
        "injected after capture source publication"
    );
    if FAIL_PUBLICATION.replace(false) {
        Err(Error::Other(Box::new(InjectedIngress)))
    } else {
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod metal {
    mod carrier;
    mod scheduled;
    use super::*;
    use crate::backend::{
        array_copy::CaptureTensorSelection,
        managed_memory::NativeMemoryOwner,
        nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms},
    };
    use eredu_core::{cache::LayerCachePolicy, capture::*, *};
    use eredu_nn::{workspace::WorkspaceContext, Tensor};
    use eredu_runtime::working_memory::*;
    use safemlx::{Device, DeviceType, Dtype};
    use std::{
        num::NonZeroU8,
        panic::{catch_unwind, AssertUnwindSafe},
    };

    fn admitted(
        axes: Vec<SymbolicDimension>,
        transform: CaptureTransform,
        slices: Vec<CaptureSlice>,
    ) -> AdmittedCapturePlan {
        let point = ObservationPoint {
            path: "block.output".into(),
            node_id: "block".into(),
            meaning: "actual activation".into(),
            value_type: ObservationValueType::Tensor,
            dtype: ObservationDtype::Floating,
            axes: Some(
                axes.into_iter()
                    .enumerate()
                    .map(|(index, dimension)| TensorAxis {
                        name: format!("axis{index}"),
                        dimension,
                    })
                    .collect(),
            ),
            prefill: true,
            decode: true,
            requirements: vec![ObservationRequirement::ActivationHooks],
            position: ObservationPosition::BeforeIntervention,
            retained_bytes: None,
            host_bytes: None,
        };
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
        let usage = CaptureUsage {
            captures: 10,
            retained_bytes: u64::MAX,
            host_bytes: u64::MAX,
            encoded_bytes: u64::MAX,
        };
        CapturePlan {
            schema_version: 1,
            selections: vec![CaptureSelection {
                id: "tensor".into(),
                path: "block.output".into(),
                schedule: CaptureSchedule::default(),
                slices,
                transform,
            }],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &catalog,
            &support,
            &CaptureCapabilities {
                transformations: vec![
                    CaptureTransformKind::Preview,
                    CaptureTransformKind::Slice,
                    CaptureTransformKind::FullTensor,
                    CaptureTransformKind::Summary,
                ],
                max_histogram_bins: 0,
                conditions: vec![],
            },
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
        )
        .unwrap()
    }
    fn fresh(
        pool: &MemoryLedger,
        bytes: u64,
    ) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
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
        pool.reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &crate::memory_fixture::admission(Admission {
                memory_limits: Default::default(),
                additional_headroom: Default::default(),
                state,
                requested_positions: 7,
                incremental_required_bytes: Some(bytes),
            }),
            crate::memory_fixture::resolved_limits(pool.fixture_host_limit().unwrap()),
        )
        .unwrap()
        .into_funding()
        .unwrap()
    }

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Gpu, 0))
    }
    fn shared() -> SharedCapturePlan {
        SharedCapturePlan::new(admitted(
            vec![SymbolicDimension::Known(8)],
            CaptureTransform::FullTensor,
            vec![],
        ))
    }
    fn quote(input: &Array, plan: &SharedCapturePlan) -> (u64, usize) {
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let mut projection = ExistingArrayProjection::new(&context);
        let input = projection.project(input).unwrap();
        context
            .begin_state_span(std::slice::from_ref(&input))
            .unwrap();
        let source = input.multiply(&input, &context).unwrap();
        let other = input.add(&input, &context).unwrap();
        let geometry =
            CaptureTensorGeometry::prepare(plan.admission(), 0, CapturePhase::Prefill, 0, None)
                .unwrap();
        let selected = CaptureTensorSelection::from_geometry(&geometry)
            .unwrap()
            .trace_within(&source, &context)
            .unwrap();
        let report = context.report(&[input, source, other, selected]).unwrap();
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        (
            report.state.unwrap().transient_bytes.unwrap() + report.host_workspace_bytes.unwrap(),
            report.closing_storage.maximum_allocations,
        )
    }
    // One observed source is published and pinned, then the complete finite
    // root inventory is published at completion. Each constructor keeps its own
    // host account until its last attached backing retires.
    fn capture_publication_allowance(captures: usize) -> u64 {
        ordinary_publication_control_bytes()
            .unwrap()
            .checked_mul(u64::try_from(captures).unwrap())
            .unwrap()
    }
    struct Fixture {
        pool: MemoryLedger,
        input: Array,
        plan: SharedCapturePlan,
        reservation: WorkingMemoryReservation,
        run: WorkingMemoryFundingRun,
        work: FundedWorkOwner,
        bank: PreparedCaptureRun,
        h: u64,
        n: u64,
        _sources: RetainedStoragePublication,
    }
    fn fixture(dtype: Dtype) -> Fixture {
        fixture_in_pool(dtype, shared(), None)
    }
    fn fixture_in_pool(
        dtype: Dtype,
        plan: SharedCapturePlan,
        existing_pool: Option<MemoryLedger>,
    ) -> Fixture {
        fixture_with_publication_allowance(dtype, plan, existing_pool, 0)
    }
    fn fixture_with_publication_allowance(
        dtype: Dtype,
        plan: SharedCapturePlan,
        existing_pool: Option<MemoryLedger>,
        opening_publication_rows: usize,
    ) -> Fixture {
        // Bootstrap owns input/source construction. The request includes its
        // closed host program, inspected span and finite publication population.
        let bootstrap = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let bootstrap_owner = NativeMemoryOwner::acquire(&bootstrap).unwrap();
        let stream = stream();
        let input = Array::from_slice(&[1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[8])
            .as_dtype(dtype, &stream)
            .unwrap();
        input.evaluated().unwrap();
        let host = CaptureRunHostPlan::prepare(&plan).unwrap();
        let h = host.initialization_peak_bytes();
        let (native, closing) = quote(&input, &plan);
        let (ordinary, ordinary_controls) = OrdinaryPublicationPlan::fixture(
            closing
                .checked_mul(2)
                .unwrap()
                .checked_add(1)
                .unwrap()
                .checked_add(opening_publication_rows)
                .unwrap(),
        );
        let n = native
            .checked_add(ordinary_controls)
            .unwrap()
            .checked_add(capture_publication_allowance(1))
            .unwrap();
        let mut initial = RetainedStorage::default();
        initial.include_array(&input).unwrap();
        initial.include_capture_plan(plan.clone()).unwrap();
        let pool =
            existing_pool.unwrap_or_else(|| crate::memory_fixture::ledger(u64::MAX, 0).unwrap());
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let sources = initial.publish_unquoted(&owner).unwrap();
        drop((owner, bootstrap_owner));
        let (reservation, run) = fresh(&pool, h + n);
        let bank = run.prepare_capture_run(&reservation, host).unwrap();
        let work = FundedWork::new_ordinary(run.scope().unwrap(), ordinary).unwrap();
        Fixture {
            pool,
            input,
            plan,
            reservation,
            run,
            work,
            bank,
            h,
            n,
            _sources: sources,
        }
    }
    fn retire_native(work: &FundedWork) {
        for root in work.roots.borrow().iter() {
            root.evaluated().unwrap();
        }
        if !work.published.get() {
            work.publish(work.prepare_inventory().unwrap()).unwrap();
        }
        work.certify().unwrap();
        assert!(work.scope.borrow().is_none());
    }
    fn reclaim(pool: &MemoryLedger) -> u64 {
        // Cached native backing remains physical storage until cache eviction.
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
        crate::backend::ordinary_retirement::reclaim_all();
        pool.fixture_host_charge().unwrap()
    }
    fn settled_bytes(pool: &MemoryLedger, expected: u64) {
        // Native allocation owners may queue their ordinary-host destruction
        // after the final graph handle drops. Observe retirement through the
        // same bounded helper as the other native session fixtures.
        crate::backend::submission_recovery::wait_for_retirement(|| reclaim(pool) == expected);
        assert_eq!(reclaim(pool), expected);
    }
    fn caused_by<T: std::error::Error + 'static>(
        mut error: &(dyn std::error::Error + 'static),
    ) -> bool {
        loop {
            if error.is::<T>() {
                return true;
            }
            match error.source() {
                Some(source) => error = source,
                None => return false,
            }
        }
    }

    #[test]
    fn lazy_capture_uses_original_scope_and_preserves_unrelated_pending_roots() {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let mut f = fixture(dtype);
            let stream = stream();
            let source = f.input.multiply(&f.input, &stream).unwrap();
            let unrelated = f.input.add(&f.input, &stream).unwrap();
            assert!(source
                .try_metadata_snapshot()
                .unwrap()
                .allocation()
                .is_none());
            f.work.retain(&unrelated);
            let mut step = f
                .bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare()
                .unwrap();
            let before = EVALUATIONS.get();
            let collectors = (
                f.work.roots.borrow().capacity(),
                f.work.publications.borrow().capacity(),
                f.work.publications.borrow().len(),
            );
            let tensor = f
                .work
                .capture_tensor(&source, step.take_tensor(0).unwrap(), &stream)
                .unwrap();
            assert_eq!(EVALUATIONS.get(), before + 1);
            assert_eq!((f.work.roots.borrow().capacity(), f.work.publications.borrow().capacity(), f.work.publications.borrow().len()), collectors,
                "ordinary capture uses its paid lexical source collector without growing Work destinations");
            assert_eq!(
                tensor.observation().data(),
                &TensorObservationData::F32(vec![1., 4., 9., 16., 25., 36., 49., 64.])
            );
            assert!(unrelated
                .try_metadata_snapshot()
                .unwrap()
                .allocation()
                .is_none());
            assert!(!f.work.published.get());
            assert!(f.work.roots.borrow().len() >= 3);
            f.work.certify().unwrap();
            assert!(
                f.work.scope.borrow().is_some(),
                "capture alone never certifies full work"
            );
            let alias = tensor.observation().clone();
            step.record_tensor(
                tensor,
                match dtype {
                    Dtype::Float16 => eredu_core::checkpoint::TensorDtype::F16,
                    Dtype::Bfloat16 => eredu_core::checkpoint::TensorDtype::Bf16,
                    _ => eredu_core::checkpoint::TensorDtype::F32,
                },
                CaptureUsage::default(),
            )
            .unwrap();
            drop(step);
            retire_native(&f.work);
            assert!(f.pool.fixture_host_peak().unwrap() <= f.pool.fixture_host_limit().unwrap());
            let pool = f.pool.clone();
            let protected = f.h;
            drop((source, unrelated));
            drop(f);
            // Completed native/source owners retire independently. After run
            // closure, this host-only alias retains exactly its original H.
            settled_bytes(&pool, protected);
            assert_eq!(alias.shape(), &[8]);
            drop(alias);
            settled_bytes(&pool, 0);
        }
    }

    #[test]
    fn invalid_lazy_geometry_and_borrowed_scope_reject_before_evaluation() {
        for busy in [false, true] {
            let mut f = fixture(Dtype::Float32);
            let stream = stream();
            let source = if busy {
                f.input.multiply(&f.input, &stream).unwrap()
            } else {
                f.input
                    .reshape(&[2, 4], &stream)
                    .unwrap()
                    .multiply(f.input.reshape(&[2, 4], &stream).unwrap(), &stream)
                    .unwrap()
            };
            let mut step = f
                .bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare()
                .unwrap();
            let before = EVALUATIONS.get();
            let lock = busy.then(|| f.work.scope.borrow_mut());
            let error = f
                .work
                .capture_tensor(&source, step.take_tensor(0).unwrap(), &stream)
                .unwrap_err();
            assert_eq!(EVALUATIONS.get(), before);
            assert!(source
                .try_metadata_snapshot()
                .unwrap()
                .allocation()
                .is_none());
            assert!(f.work.roots.borrow().is_empty());
            assert!(step.take_tensor(0).is_err());
            drop(lock);
            drop(error);
            drop(step);
            retire_native(&f.work);
            let pool = f.pool.clone();
            drop(source);
            drop(f);
            settled_bytes(&pool, 0);
        }
    }

    #[test]
    fn failures_and_unwind_keep_source_roots_and_uncertified_original_envelope() {
        for mode in 0..3 {
            let mut f = fixture(Dtype::Float32);
            let stream = stream();
            let source = f.input.multiply(&f.input, &stream).unwrap();
            let mut step = f
                .bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare()
                .unwrap();
            match mode {
                0 => FAIL_EVALUATION.set(true),
                1 => FAIL_PUBLICATION.set(true),
                _ => PANIC_PUBLICATION.set(true),
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                f.work
                    .capture_tensor(&source, step.take_tensor(0).unwrap(), &stream)
            }));
            if mode == 2 {
                assert!(result.is_err());
            } else {
                assert!(caused_by::<InjectedIngress>(&result.unwrap().unwrap_err()));
            }
            assert!(!f.work.roots.borrow().is_empty());
            assert!(!f.work.published.get());
            f.work.certify().unwrap();
            assert!(f.work.scope.borrow().is_some());
            assert_eq!(
                source
                    .try_metadata_snapshot()
                    .unwrap()
                    .allocation()
                    .is_some(),
                mode != 0
            );
            assert!(step.take_tensor(0).is_err());
            drop(step);
            let pool = f.pool.clone();
            let protected = f.h + f.n;
            drop(source);
            drop(f);
            settled_bytes(&pool, protected);
        }
    }

    #[test]
    fn foreign_claim_is_rejected_before_retaining_or_settling_source() {
        let f = fixture(Dtype::Float32);
        let mut other = fixture(Dtype::Float32);
        let stream = stream();
        let source = f.input.multiply(&f.input, &stream).unwrap();
        let mut step = other
            .bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let before = EVALUATIONS.get();
        let error = f
            .work
            .capture_tensor(&source, step.take_tensor(0).unwrap(), &stream)
            .unwrap_err();
        assert!(caused_by::<WorkingMemoryError>(&error));
        assert_eq!(EVALUATIONS.get(), before);
        assert!(source
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .is_none());
        assert!(f.work.roots.borrow().is_empty());
        drop(step);
        retire_native(&f.work);
        retire_native(&other.work);
        let a = f.pool.clone();
        let b = other.pool.clone();
        drop(source);
        drop((f, other));
        settled_bytes(&a, 0);
        settled_bytes(&b, 0);
    }
}
