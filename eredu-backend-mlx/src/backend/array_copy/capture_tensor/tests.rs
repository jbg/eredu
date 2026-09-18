use super::*;
use std::cell::Cell;
thread_local! {
    static BEFORE_FINISH: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
    static FAIL_AFTER_SLICE: Cell<bool> = const { Cell::new(false) };
    static FAIL_AFTER_CAST: Cell<bool> = const { Cell::new(false) };
    static PANIC_AFTER_CAST: Cell<bool> = const { Cell::new(false) };
    static FAIL_AFTER_EVAL: Cell<bool> = const { Cell::new(false) };
    static PANIC_AFTER_SLICE: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn before_finish() {
    let callback = BEFORE_FINISH.with(|slot| slot.borrow_mut().take());
    if let Some(callback) = callback {
        callback();
    }
}
#[derive(Debug, thiserror::Error)]
#[error("injected capture failure")]
struct InjectedFailure;
pub(super) fn after_slice() -> Result<(), Exception> {
    assert!(!PANIC_AFTER_SLICE.replace(false), "capture worker unwind");
    if FAIL_AFTER_SLICE.replace(false) {
        Err(Exception::from_source(InjectedFailure))
    } else {
        Ok(())
    }
}
pub(super) fn after_cast() -> Result<(), Exception> {
    assert!(!PANIC_AFTER_CAST.replace(false), "capture cast unwind");
    if FAIL_AFTER_CAST.replace(false) {
        Err(Exception::from_source(InjectedFailure))
    } else {
        Ok(())
    }
}
pub(super) fn after_evaluation() -> Result<(), Exception> {
    if FAIL_AFTER_EVAL.replace(false) {
        Err(Exception::from_source(InjectedFailure))
    } else {
        Ok(())
    }
}
#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod metal {
    mod floating;
    mod fragments;
    mod prepared_fragments;
    mod scheduled;
    use super::*;
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
    use eredu_core::{cache::LayerCachePolicy, capture::*, *};
    use eredu_nn::workspace::{
        WorkspaceContext, WorkspaceHostBound, WorkspaceMechanisms, WorkspaceOperation,
        WorkspaceOperationBound,
    };
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
    use safemlx::{
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Device, DeviceType,
    };
    use std::{
        collections::BTreeMap,
        error::Error as _,
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
                physical_native_bytes: None,
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
                physical_native_limit: false,
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
        pool: &WorkingMemoryPool,
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
        pool.reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &Admission {
                state,
                requested_positions: 7,
                incremental_required_bytes: bytes,
                available_memory_bytes: None,
            },
            pool.effective_capacity().unwrap(),
        )
        .unwrap()
        .into_funding()
        .unwrap()
    }

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Gpu, 0))
    }
    fn host(source: &AdmittedCapturePlan) -> CaptureTensorHostPlan<'_> {
        CaptureTensorHostPlan::prepare(
            CaptureTensorGeometry::prepare(source, 0, CapturePhase::Prefill, 0, None).unwrap(),
        )
        .unwrap()
    }
    fn admission(
        shape: &[i32],
        transform: CaptureTransform,
        slices: Vec<CaptureSlice>,
    ) -> AdmittedCapturePlan {
        admitted(
            shape
                .iter()
                .map(|&n| SymbolicDimension::Known(n as usize))
                .collect(),
            transform,
            slices,
        )
    }
    fn register(
        pool: &WorkingMemoryPool,
        source: &Array,
    ) -> eredu_runtime::working_memory::WorkingMemoryStorage<StorageIdentity> {
        let info = source
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .unwrap();
        pool.register_storage(
            [(
                StorageIdentity::Native(info.identity()),
                info.bytes() as u64,
            )]
            .into_iter()
            .filter(|(_, n)| *n != 0),
        )
        .unwrap()
    }
    fn settle(roots: &RefCell<Vec<Array>>) {
        for root in roots.borrow().iter() {
            root.evaluated().unwrap();
        }
        roots.borrow_mut().clear();
    }
    fn cause(error: &(dyn std::error::Error + 'static)) -> bool {
        if error.downcast_ref::<InjectedFailure>().is_some() {
            true
        } else {
            error.source().is_some_and(cause)
        }
    }

    #[test]
    fn full_slice_preview_scalar_zero_and_strided_values_match_actual_selection() {
        let stream = stream();
        let root = Array::from_slice(
            &(0..4096).map(|n| n as f32 + 0.25).collect::<Vec<_>>(),
            &[64, 64],
        );
        let view = root
            .try_index_device((2..5, (4..12).stride_by(2)), &stream)
            .unwrap();
        view.evaluated().unwrap();
        assert!(view.allocation_info().unwrap().unwrap().bytes() > view.nbytes());
        for (transform, slices, shape, expected) in [
            (
                CaptureTransform::FullTensor,
                vec![],
                vec![3, 4],
                (2..5)
                    .flat_map(|r| (4..12).step_by(2).map(move |c| (64 * r + c) as f32 + 0.25))
                    .collect::<Vec<_>>(),
            ),
            (
                CaptureTransform::Slice,
                vec![
                    CaptureSlice {
                        axis: "axis0".into(),
                        start: 1,
                        end: 3,
                        stride: 1,
                    },
                    CaptureSlice {
                        axis: "axis1".into(),
                        start: 0,
                        end: 4,
                        stride: 2,
                    },
                ],
                vec![2, 2],
                vec![196.25, 200.25, 260.25, 264.25],
            ),
            (
                CaptureTransform::Preview { max_elements: 5 },
                vec![],
                vec![5],
                vec![132.25, 134.25, 136.25, 138.25, 196.25],
            ),
            (
                CaptureTransform::Slice,
                vec![CaptureSlice {
                    axis: "axis0".into(),
                    start: 0,
                    end: 0,
                    stride: 1,
                }],
                vec![0, 4],
                vec![],
            ),
        ] {
            let admitted = admission(&[3, 4], transform, slices);
            assert_eq!(host(&admitted).geometry().source_shape(), &[3, 4]);
            let plan = PreparedCaptureTensor::new(&view, host(&admitted)).unwrap();
            let p = plan.host_peak_bytes();
            let context =
                WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
            let mut projection = ExistingArrayProjection::new(&context);
            let input = projection.project(&view).unwrap();
            context.begin_state_span(&[input.clone()]).unwrap();
            let output = plan.trace(&mut projection).unwrap();
            let report = context.report(&[input, output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            assert_eq!(report.host_workspace_bytes, Some(0));
            let root_bytes = view.allocation_info().unwrap().unwrap().bytes() as u64;
            let n = report.total_bytes.unwrap();
            assert!(report.state.as_ref().unwrap().retained_bytes.unwrap() >= root_bytes);
            let pool = WorkingMemoryPool::new(root_bytes + p + n, 0).unwrap();
            let registered = register(&pool, &view);
            let (reservation, run) = fresh(&pool, p + n);
            let mut native = run.scope().unwrap();
            let roots = RefCell::new(Vec::with_capacity(plan.recovery_descriptors()));
            let capacity = roots.borrow().capacity();
            let result = plan
                .transfer(&run, &reservation, &mut native, &stream, &roots)
                .unwrap();
            assert_eq!(result.shape(), shape);
            let TensorObservationData::F32(data) = result.data() else {
                unreachable!()
            };
            assert_eq!(data, &expected);
            assert_eq!(roots.borrow().capacity(), capacity);
            let native_bytes = roots
                .borrow()
                .iter()
                .map(|a| {
                    a.evaluated().unwrap();
                    let x = a.allocation_info().unwrap().unwrap();
                    (x.identity(), x.bytes() as u64)
                })
                .collect::<BTreeMap<_, _>>()
                .values()
                .sum::<u64>();
            assert!(native_bytes <= root_bytes + n);
            // Before settlement, source plus host and native workspace stay charged.
            assert_eq!(pool.used_bytes().unwrap(), root_bytes + p + n);
            settle(&roots);
            native.certify().unwrap();
            drop((run, reservation, registered));
            // Only the finished host observation survives; native scratch is free.
            assert_eq!(pool.used_bytes().unwrap(), p);
            drop(result);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
        for (source, shape, expected) in [
            (Array::from_slice(&[7.25f32], &[]), vec![], vec![7.25]),
            (
                Array::from_slice(&[] as &[f32], &[0, 3]),
                vec![0, 3],
                vec![],
            ),
            (
                Array::from_slice(&[1f32, 2., 3., 4.], &[4])
                    .try_index_device((..).stride_by(-1), &stream)
                    .unwrap(),
                vec![4],
                vec![4., 3., 2., 1.],
            ),
        ] {
            source.evaluated().unwrap();
            let admitted = admission(&shape, CaptureTransform::FullTensor, vec![]);
            let plan = PreparedCaptureTensor::new(&source, host(&admitted)).unwrap();
            let p = plan.host_peak_bytes();
            let bytes = source.allocation_info().unwrap().unwrap().bytes() as u64;
            let pool = WorkingMemoryPool::new(bytes + p, 0).unwrap();
            let registered = register(&pool, &source);
            let (reservation, run) = fresh(&pool, p);
            let mut native = run.scope().unwrap();
            let roots = RefCell::new(vec![]);
            let result = plan
                .transfer(&run, &reservation, &mut native, &stream, &roots)
                .unwrap();
            assert_eq!(
                result.shape(),
                shape.iter().map(|&n| n as usize).collect::<Vec<_>>()
            );
            let TensorObservationData::F32(data) = result.data() else {
                unreachable!()
            };
            assert_eq!(data, &expected);
            settle(&roots);
            native.certify().unwrap();
            drop((registered, run, reservation, result));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }

    thread_local! {static HOUSEKEEPING:Cell<usize>=const{Cell::new(0)};}
    fn housekeeping() {
        HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
    }
    struct Cold;
    impl Cold {
        fn new() -> Self {
            safemlx::register_thread_runtime_housekeeping(housekeeping);
            HOUSEKEEPING.set(0);
            Self
        }
    }
    impl Drop for Cold {
        fn drop(&mut self) {
            safemlx::unregister_thread_runtime_housekeeping(housekeeping);
        }
    }
    #[test]
    fn cold_trace_preserves_source_and_exact_program_and_unknown_facts() {
        let stream = stream();
        let source = Array::from_slice(&(0..24).map(|n| n as f32).collect::<Vec<_>>(), &[4, 6]);
        source.evaluated().unwrap();
        let admitted = admission(
            &[4, 6],
            CaptureTransform::Preview { max_elements: 3 },
            vec![CaptureSlice {
                axis: "axis1".into(),
                start: 1,
                end: 6,
                stride: 2,
            }],
        );
        let snapshot = source.try_metadata_snapshot().unwrap();
        {
            let _cold = Cold::new();
            let plan = PreparedCaptureTensor::new(&source, host(&admitted)).unwrap();
            let context =
                WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
            let mut projection = ExistingArrayProjection::new(&context);
            let output = plan.trace(&mut projection).unwrap();
            let report = context.report(&[output]).unwrap();
            assert!(matches!(
                report
                    .operations
                    .iter()
                    .map(|o| &o.kind)
                    .collect::<Vec<_>>()
                    .as_slice(),
                [
                    WorkspaceOperationKind::StaticSlice { .. },
                    WorkspaceOperationKind::View("reshape"),
                    WorkspaceOperationKind::StaticSlice { .. }
                ]
            ));
            assert_eq!(source.try_metadata_snapshot().unwrap(), snapshot);
            drop(plan);
            assert_eq!(HOUSEKEEPING.get(), 0);
        }
        #[derive(Debug)]
        struct Missing;
        impl WorkspaceMechanisms for Missing {
            fn operation_bound(
                &self,
                _: &WorkspaceOperation,
            ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
                Ok(None)
            }
            fn host_workspace_bound(
                &self,
                _: &WorkspaceOperation,
            ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
                Ok(None)
            }
        }
        let context = WorkspaceContext::new(Missing);
        let mut projection = ExistingArrayProjection::new(&context);
        let output = PreparedCaptureTensor::new(&source, host(&admitted))
            .unwrap()
            .trace(&mut projection)
            .unwrap();
        let report = context.report(&[output]).unwrap();
        assert_eq!(report.total_bytes, None);
        assert!(!report.unpriced_operations.is_empty());
        assert!(!report.unpriced_host_operations.is_empty());
    }

    #[test]
    fn lazy_dtype_geometry_and_short_parent_fail_before_native_or_host_fill() {
        let stream = stream();
        let source = Array::from_slice(&[1f32, 2., 3., 4.], &[2, 2]);
        source.evaluated().unwrap();
        let wrong = admission(&[4], CaptureTransform::FullTensor, vec![]);
        assert!(matches!(
            PreparedCaptureTensor::new(&source, host(&wrong)),
            Err(CaptureTensorNativeError::ShapeMismatch)
        ));
        let admitted = admission(&[2, 2], CaptureTransform::FullTensor, vec![]);
        let overflow = admission(
            &[2, 2],
            CaptureTransform::Slice,
            vec![CaptureSlice {
                axis: "axis0".into(),
                start: 0,
                end: 2,
                stride: i32::MAX as u64,
            }],
        );
        assert!(matches!(
            PreparedCaptureTensor::new(&source, host(&overflow)),
            Err(CaptureTensorNativeError::GeometryOverflow)
        ));
        let lazy = source.reshape(&[4], &stream).unwrap();
        assert!(matches!(
            PreparedCaptureTensor::new(&lazy, host(&wrong)),
            Err(CaptureTensorNativeError::UnsettledSource)
        ));
        let integers = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        integers.evaluated().unwrap();
        assert!(matches!(
            PreparedCaptureTensor::new(&integers, host(&admitted)),
            Err(CaptureTensorNativeError::UnsupportedDtype(Dtype::Int32))
        ));
        let p = host(&admitted).initialization_peak_bytes();
        let bytes = source.allocation_info().unwrap().unwrap().bytes() as u64;
        let pool = WorkingMemoryPool::new(bytes + p, 0).unwrap();
        let registration = register(&pool, &source);
        let (reservation, run) = fresh(&pool, p - 1);
        let mut native = run.scope().unwrap();
        let roots = RefCell::new(vec![]);
        let before = pool.used_bytes().unwrap();
        let error = PreparedCaptureTensor::new(&source, host(&admitted))
            .unwrap()
            .transfer(&run, &reservation, &mut native, &stream, &roots)
            .unwrap_err();
        assert!(matches!(
            error,
            CaptureTensorExecutionError::Mechanism(CaptureTensorNativeError::Host(
                CaptureTensorConstructionError::Memory(WorkingMemoryError::BudgetExceeded { .. })
            ))
        ));
        assert!(roots.borrow().is_empty());
        assert_eq!(pool.used_bytes().unwrap(), before);
        drop(error);
        native.certify().unwrap();
        drop((registration, reservation, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }

    #[test]
    fn original_typed_failures_and_unwind_keep_native_recovery_and_source_pins() {
        let stream = stream();
        for phase in 0..3 {
            let source = Array::from_slice(&(0..32).map(|n| n as f32).collect::<Vec<_>>(), &[4, 8]);
            source.evaluated().unwrap();
            let admitted = admission(
                &[4, 8],
                CaptureTransform::Preview { max_elements: 5 },
                vec![],
            );
            let p = host(&admitted).initialization_peak_bytes();
            let bytes = source.allocation_info().unwrap().unwrap().bytes() as u64;
            let pool = WorkingMemoryPool::new(bytes + p + 4096, 0).unwrap();
            let registration = register(&pool, &source);
            let (reservation, run) = fresh(&pool, p + 4096);
            let mut native = run.scope().unwrap();
            let roots = RefCell::new(vec![]);
            match phase {
                0 => FAIL_AFTER_SLICE.set(true),
                1 => FAIL_AFTER_EVAL.set(true),
                _ => PANIC_AFTER_SLICE.set(true),
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                let transfer = PreparedCaptureTensor::new(&source, host(&admitted))
                    .unwrap()
                    .transfer(&run, &reservation, &mut native, &stream, &roots);
                if phase < 2 {
                    let error = transfer.unwrap_err();
                    assert!(cause(error.source().unwrap()));
                    drop(error);
                } else {
                    drop(transfer);
                }
            }));
            if phase < 2 {
                result.unwrap();
            } else {
                assert!(result.is_err());
            }
            assert!(roots.borrow().len() >= 2);
            drop(registration);
            assert_eq!(pool.used_bytes().unwrap(), bytes + p + 4096);
            // Settling actual retained payload is explicit and separate from the
            // failed host result; safe recovery permits certification of this scope.
            settle(&roots);
            native.certify().unwrap();
            drop((run, reservation));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }

    #[test]
    fn borrowed_collector_and_foreign_native_scope_do_not_start_work() {
        let stream = stream();
        let source = Array::from_slice(&[1f32, 2.], &[2]);
        source.evaluated().unwrap();
        let admitted = admission(&[2], CaptureTransform::FullTensor, vec![]);
        let p = host(&admitted).initialization_peak_bytes();
        let bytes = source.allocation_info().unwrap().unwrap().bytes() as u64;
        let pool = WorkingMemoryPool::new(bytes + 2 * p, 0).unwrap();
        let registered = register(&pool, &source);
        let (reservation, run) = fresh(&pool, p);
        let (other, other_run) = fresh(&pool, p);
        let mut native = run.scope().unwrap();
        let mut wrong = other_run.scope().unwrap();
        let roots = RefCell::new(vec![]);
        let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let before = pool.used_bytes().unwrap();
        let error = PreparedCaptureTensor::new(&source, host(&admitted))
            .unwrap()
            .transfer(&run, &reservation, &mut native, &cpu, &roots)
            .unwrap_err();
        assert!(matches!(
            error,
            CaptureTensorExecutionError::Mechanism(CaptureTensorNativeError::UnsupportedStream(
                DeviceType::Cpu
            ))
        ));
        drop(error);
        assert!(roots.borrow().is_empty());
        assert_eq!(pool.used_bytes().unwrap(), before);
        {
            let _borrow = roots.borrow_mut();
            let error = PreparedCaptureTensor::new(&source, host(&admitted))
                .unwrap()
                .transfer(&run, &reservation, &mut native, &stream, &roots)
                .unwrap_err();
            assert!(matches!(
                error,
                CaptureTensorExecutionError::Mechanism(CaptureTensorNativeError::CollectorBusy)
            ));
        }
        let error = PreparedCaptureTensor::new(&source, host(&admitted))
            .unwrap()
            .transfer(&run, &reservation, &mut wrong, &stream, &roots)
            .unwrap_err();
        assert!(matches!(
            error,
            CaptureTensorExecutionError::Mechanism(CaptureTensorNativeError::Host(
                CaptureTensorConstructionError::Memory(WorkingMemoryError::IdentityMismatch)
            ))
        ));
        assert!(roots.borrow().is_empty());
        drop(error);
        native.certify().unwrap();
        wrong.certify().unwrap();
        drop((registered, reservation, run, other, other_run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
