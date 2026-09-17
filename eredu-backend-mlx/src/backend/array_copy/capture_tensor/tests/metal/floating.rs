use super::*;
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceLayout, WorkspaceOutputStorage};
use half::{bf16, f16};

const F16_BITS: [u16; 16] = [
    0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3555, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
    0x7c01, 0x7e01, 0xfc01, 0xffff,
];
const BF16_BITS: [u16; 16] = [
    0x0000, 0x8000, 0x0001, 0x007f, 0x0080, 0x3eab, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
    0x7f81, 0x7fc1, 0xff81, 0xffff,
];
fn source(bfloat: bool, shape: &[i32]) -> Array {
    let n = shape.iter().map(|&n| n as usize).product::<usize>();
    if bfloat {
        Array::from_slice(
            &(0..n)
                .map(|i| bf16::from_bits(BF16_BITS[i % 16]))
                .collect::<Vec<_>>(),
            shape,
        )
    } else {
        Array::from_slice(
            &(0..n)
                .map(|i| f16::from_bits(F16_BITS[i % 16]))
                .collect::<Vec<_>>(),
            shape,
        )
    }
}
fn reference(selected: &Array, stream: &Stream) -> Vec<u32> {
    // This is the existing observation conversion, including its native cast
    // and compaction. Empty observe_tensor returns before conversion.
    if selected.size() == 0 {
        return vec![];
    }
    crate::MlxTensor::from_array(selected.clone())
        .to_f32_vec(stream)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}
fn run_case(
    source: &Array,
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
    reference: &Array,
    stream: &Stream,
) {
    source.evaluated().unwrap();
    let expected = self::reference(reference, stream);
    let admitted = admission(source.shape(), transform, slices);
    let snapshot = source.try_metadata_snapshot().unwrap();
    let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let (plan, report, cast_bytes) = {
        let _cold = Cold::new();
        let plan = PreparedCaptureTensor::new(source, host(&admitted)).unwrap();
        let context = WorkspaceContext::new(facts);
        let mut projection = ExistingArrayProjection::new(&context);
        let input = projection.project(source).unwrap();
        context.begin_state_span(&[input.clone()]).unwrap();
        let output = plan.trace(&mut projection).unwrap();
        let report = context.report(&[input, output]).unwrap();
        let casts = report
            .operations
            .iter()
            .filter(|o| {
                matches!(
                    o.kind,
                    WorkspaceOperationKind::Elementwise("capture_cast_f32")
                )
            })
            .collect::<Vec<_>>();
        let cast_bytes = if expected.is_empty() {
            assert!(casts.is_empty());
            None
        } else {
            assert_eq!(casts.len(), 1);
            let cast = casts[0];
            assert_eq!(cast.inputs[0].dtype(), WorkspaceDtype::Float32);
            assert_eq!(cast.outputs[0].dtype(), WorkspaceDtype::Float32);
            assert_eq!(cast.inputs[0].shape(), cast.outputs[0].shape());
            let bound = facts.operation_bound(cast).unwrap().unwrap();
            assert_eq!(bound.scratch_bytes, 0);
            let [WorkspaceOutputStorage::Allocate(bytes)] = bound.outputs.as_slice() else {
                panic!("actual half cast must allocate despite identical metadata dtype")
            };
            assert_eq!(facts.host_workspace_bound(cast).unwrap().unwrap().bytes, 0);
            Some(*bytes)
        };
        assert_eq!(report.host_workspace_bytes, Some(0));
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        assert_eq!(source.try_metadata_snapshot().unwrap(), snapshot);
        assert_eq!(HOUSEKEEPING.get(), 0);
        (plan, report, cast_bytes)
    };
    let h = plan.host_peak_bytes();
    let n = report.total_bytes.unwrap();
    let source_info = snapshot.allocation().unwrap();
    let bytes = source_info.bytes() as u64;
    let pool = WorkingMemoryPool::new(bytes + h + n, 0).unwrap();
    let registration = register(&pool, source);
    let (reservation, run) = fresh(&pool, h + n);
    let mut native = run.scope().unwrap();
    let roots = RefCell::new(Vec::with_capacity(plan.recovery_descriptors()));
    let capacity = roots.borrow().capacity();
    let value = plan
        .transfer(&run, &reservation, &mut native, stream, &roots)
        .unwrap();
    assert_eq!(roots.borrow().capacity(), capacity);
    assert_eq!(value.shape(), host(&admitted).geometry().shape());
    let TensorObservationData::F32(data) = value.data() else {
        unreachable!()
    };
    assert_eq!(
        data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected
    );
    if let Some(bound) = cast_bytes {
        let roots = roots.borrow();
        let cast = roots.last().unwrap();
        let info = cast.allocation_info().unwrap().unwrap();
        assert_eq!(cast.dtype(), Dtype::Float32);
        assert_ne!(info.identity(), source_info.identity());
        assert!(info.bytes() as u64 <= bound, "{} > {bound}", info.bytes());
    } else {
        assert!(roots.borrow().iter().all(|a| a.dtype() == source.dtype()));
    }
    // Publish actual completed roots from the same account while host H stays
    // protected. No made-up byte inventory or allocation callback is used.
    let inventory = roots
        .borrow()
        .iter()
        .map(|a| {
            a.evaluated().unwrap();
            let info = a.allocation_info().unwrap().unwrap();
            (
                StorageIdentity::Native(info.identity()),
                info.bytes() as u64,
            )
        })
        .filter(|(_, bytes)| *bytes != 0)
        .collect::<BTreeMap<_, _>>();
    assert!(inventory.values().sum::<u64>() <= bytes + n);
    let before_publication = pool.used_bytes().unwrap();
    let publication = pool.adopt_storage_individually(&native, inventory).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), before_publication);
    let alias = value.clone();
    settle(&roots);
    native.certify().unwrap();
    drop((publication, registration, run, reservation, value));
    assert!(pool.used_bytes().unwrap() >= h);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(source.try_metadata_snapshot().unwrap(), snapshot);
}

#[test]
fn raw_half_patterns_and_scalar_match_native_cast_bits() {
    let stream = stream();
    for bfloat in [false, true] {
        let source = source(bfloat, &[16]);
        let expected = reference(&source, &stream);
        assert_eq!(expected[1], (-0.0f32).to_bits());
        assert_eq!(expected[10], f32::INFINITY.to_bits());
        assert_eq!(expected[11], f32::NEG_INFINITY.to_bits());
        assert!(expected[12..]
            .iter()
            .all(|&bits| f32::from_bits(bits).is_nan()));
        run_case(
            &source,
            CaptureTransform::FullTensor,
            vec![],
            &source,
            &stream,
        );
        let scalar = if bfloat {
            Array::from_slice(&[bf16::from_bits(0x8000)], &[])
        } else {
            Array::from_slice(&[f16::from_bits(0x8000)], &[])
        };
        run_case(
            &scalar,
            CaptureTransform::FullTensor,
            vec![],
            &scalar,
            &stream,
        );
    }
}

#[test]
fn nonempty_narrow_padded_reversed_broadcast_split_and_preview_preserve_values_and_bound() {
    let stream = stream();
    for bfloat in [false, true] {
        let root = source(bfloat, &[32, 16]);
        let narrow = root
            .try_index_device((3..6, (1..16).stride_by(2)), &stream)
            .unwrap();
        narrow.evaluated().unwrap();
        assert!(narrow.allocation_info().unwrap().unwrap().bytes() > narrow.nbytes());
        run_case(
            &narrow,
            CaptureTransform::FullTensor,
            vec![],
            &narrow,
            &stream,
        );
        let slice = narrow
            .try_index_device((1..3, (0..8).stride_by(3)), &stream)
            .unwrap();
        run_case(
            &narrow,
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
                    end: 8,
                    stride: 3,
                },
            ],
            &slice,
            &stream,
        );
        let flat = narrow.reshape(&[24], &stream).unwrap();
        let prefix = flat.try_index_device(0..5, &stream).unwrap();
        run_case(
            &narrow,
            CaptureTransform::Preview { max_elements: 5 },
            vec![],
            &prefix,
            &stream,
        );
        let transposed = narrow.transpose(&stream).unwrap();
        run_case(
            &transposed,
            CaptureTransform::FullTensor,
            vec![],
            &transposed,
            &stream,
        );
        let reversed = root
            .try_index_device(((..).stride_by(-1), (..).stride_by(-1)), &stream)
            .unwrap();
        run_case(
            &reversed,
            CaptureTransform::FullTensor,
            vec![],
            &reversed,
            &stream,
        );
        let row = root.try_index_device((0..1, ..), &stream).unwrap();
        let broadcast = safemlx::ops::broadcast_to(&row, &[4, 16], &stream).unwrap();
        run_case(
            &broadcast,
            CaptureTransform::FullTensor,
            vec![],
            &broadcast,
            &stream,
        );
        let pieces = broadcast.split_axis(&[1, 3], Some(0), &stream).unwrap();
        run_case(
            &pieces[1],
            CaptureTransform::FullTensor,
            vec![],
            &pieces[1],
            &stream,
        );
        // The selected native Split repair is a prerequisite for this case.
        // Probe destination geometry before any host read: old MLX incorrectly
        // marks a negative-stride fast Split contiguous and casts one element.
        let six = self::source(bfloat, &[6]);
        let reverse = six.try_index_device((..).stride_by(-1), &stream).unwrap();
        let negative_pieces = reverse.split_axis(&[3], Some(0), &stream).unwrap();
        for piece in &negative_pieces {
            let probe = piece.as_type::<f32>(&stream).unwrap();
            probe.evaluated().unwrap();
            assert_eq!(probe.shape(), &[3]);
            assert_eq!(probe.strides(), &[1]);
            assert!(probe.allocation_info().unwrap().unwrap().bytes() >= 3 * 4);
            run_case(piece, CaptureTransform::FullTensor, vec![], piece, &stream);
        }
        // Shared Slice needs the same signed-layout invariant as fast Split.
        // Overlap makes the old span equal its positive-stride product even
        // though the negative axis is meaningful. Guard the cast before reads.
        let five = self::source(bfloat, &[5]);
        let overlap = five
            .as_strided(&[3, 2, 2][..], &[-1, 1, 1][..], 2, &stream)
            .unwrap();
        let selected = overlap.try_index_device((1..3, .., ..), &stream).unwrap();
        let probe = selected.as_type::<f32>(&stream).unwrap();
        probe.evaluated().unwrap();
        assert_eq!(probe.shape(), &[2, 2, 2]);
        assert_eq!(probe.signed_strides(), &[4, 2, 1]);
        assert!(probe.allocation_info().unwrap().unwrap().bytes() >= 8 * 4);
        run_case(
            &overlap,
            CaptureTransform::Slice,
            vec![CaptureSlice {
                axis: "axis0".into(),
                start: 1,
                end: 3,
                stride: 1,
            }],
            &selected,
            &stream,
        );
        let padded = root
            .as_strided(&[2, 3][..], &[31, 2][..], 2, &stream)
            .unwrap();
        run_case(
            &padded,
            CaptureTransform::FullTensor,
            vec![],
            &padded,
            &stream,
        );
    }
}

#[test]
fn all_empty_routes_skip_half_cast_and_f32_iteration_including_identity_split() {
    let stream = stream();
    for bfloat in [false, true] {
        let empty = source(bfloat, &[0, 16]);
        run_case(
            &empty,
            CaptureTransform::FullTensor,
            vec![],
            &empty,
            &stream,
        );
        let root = source(bfloat, &[1, 16]);
        let broadcast = safemlx::ops::broadcast_to(&root, &[4, 16], &stream).unwrap();
        let split = broadcast.split_axis(&[1, 1], Some(0), &stream).unwrap();
        assert_eq!(split[1].shape(), &[0, 16]);
        // Works both before and after the independent native empty-Split fix.
        // Full's identity slice is not assumed to normalize old data_size.
        run_case(
            &split[1],
            CaptureTransform::FullTensor,
            vec![],
            &split[1],
            &stream,
        );
        run_case(
            &split[1],
            CaptureTransform::Preview { max_elements: 8 },
            vec![],
            &split[1],
            &stream,
        );
        let empty_prefix = root.try_index_device((0..0, ..), &stream).unwrap();
        run_case(
            &root,
            CaptureTransform::Slice,
            vec![CaptureSlice {
                axis: "axis0".into(),
                start: 0,
                end: 0,
                stride: 1,
            }],
            &empty_prefix,
            &stream,
        );
        run_case(
            &root,
            CaptureTransform::Preview { max_elements: 0 },
            vec![],
            &empty_prefix,
            &stream,
        );
    }
}

#[test]
fn half_trace_requires_explicit_nonempty_cast_fact_and_rejects_wrong_descriptors() {
    let stream = stream();
    let source = source(true, &[4, 16]);
    source.evaluated().unwrap();
    let admitted = admission(
        &[4, 16],
        CaptureTransform::Preview { max_elements: 3 },
        vec![],
    );
    let plan = PreparedCaptureTensor::new(&source, host(&admitted)).unwrap();
    let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let context = WorkspaceContext::new(facts);
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
            WorkspaceOperationKind::Index { selected_axes: 0 },
            WorkspaceOperationKind::View("reshape"),
            WorkspaceOperationKind::Index { selected_axes: 0 },
            WorkspaceOperationKind::Elementwise("capture_cast_f32"),
        ]
    ));
    let cast = report.operations.last().unwrap().clone();
    let mut invalid = vec![];
    for (inputs, outputs) in [(0, 1), (2, 1), (1, 0), (1, 2)] {
        invalid.push(WorkspaceOperation {
            kind: cast.kind.clone(),
            inputs: vec![cast.inputs[0].clone(); inputs],
            outputs: vec![cast.outputs[0].clone(); outputs],
        });
    }
    for dtype in [
        WorkspaceDtype::Int32,
        WorkspaceDtype::Uint32,
        WorkspaceDtype::Bool,
    ] {
        let mut op = cast.clone();
        op.inputs[0] = WorkspaceLayout::new(&[3], dtype).unwrap();
        invalid.push(op);
        let mut op = cast.clone();
        op.outputs[0] = WorkspaceLayout::new(&[3], dtype).unwrap();
        invalid.push(op);
    }
    let mut mismatch = cast.clone();
    mismatch.outputs[0] = WorkspaceLayout::new(&[1, 3], WorkspaceDtype::Float32).unwrap();
    invalid.push(mismatch);
    for shape in [&[0, 3][..], &[1; 33][..]] {
        let layout = WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
        invalid.push(WorkspaceOperation {
            kind: cast.kind.clone(),
            inputs: vec![layout.clone()],
            outputs: vec![layout],
        });
    }
    for op in invalid {
        assert!(facts.operation_bound(&op).unwrap().is_none());
        assert!(facts.host_workspace_bound(&op).unwrap().is_none());
    }
    #[derive(Debug)]
    struct MissingCast(MlxMetalWorkspaceMechanisms);
    impl WorkspaceMechanisms for MissingCast {
        fn operation_bound(
            &self,
            op: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
            if matches!(
                op.kind,
                WorkspaceOperationKind::Elementwise("capture_cast_f32")
            ) {
                Ok(None)
            } else {
                self.0.operation_bound(op)
            }
        }
        fn host_workspace_bound(
            &self,
            op: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
            if matches!(
                op.kind,
                WorkspaceOperationKind::Elementwise("capture_cast_f32")
            ) {
                Ok(None)
            } else {
                self.0.host_workspace_bound(op)
            }
        }
    }
    let context = WorkspaceContext::new(MissingCast(facts));
    let mut projection = ExistingArrayProjection::new(&context);
    let output = plan.trace(&mut projection).unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(report.total_bytes, None);
    assert!(!report.unpriced_operations.is_empty());
    assert!(!report.unpriced_host_operations.is_empty());
    let lazy = source.reshape(&[64], &stream).unwrap();
    let flat = admission(&[64], CaptureTransform::FullTensor, vec![]);
    assert!(matches!(
        PreparedCaptureTensor::new(&lazy, host(&flat)),
        Err(CaptureTensorNativeError::UnsettledSource)
    ));
    let f64_source = Array::from_slice_f64(&[1.25f64, -0., f64::INFINITY], &[3]);
    f64_source.evaluated().unwrap();
    let admitted = admission(&[3], CaptureTransform::FullTensor, vec![]);
    assert!(matches!(
        PreparedCaptureTensor::new(&f64_source, host(&admitted)),
        Err(CaptureTensorNativeError::UnsupportedDtype(Dtype::Float64))
    ));
}

#[test]
fn half_short_parent_rejects_before_work_and_cast_failures_keep_actual_recovery() {
    let stream = stream();
    for bfloat in [false, true] {
        let source = source(bfloat, &[16]);
        source.evaluated().unwrap();
        let admitted = admission(&[16], CaptureTransform::FullTensor, vec![]);
        let h = host(&admitted).initialization_peak_bytes();
        let bytes = source.allocation_info().unwrap().unwrap().bytes() as u64;
        let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(facts);
        let mut projection = ExistingArrayProjection::new(&context);
        let plan = PreparedCaptureTensor::new(&source, host(&admitted)).unwrap();
        let output = plan.trace(&mut projection).unwrap();
        let n = context.report(&[output]).unwrap().total_bytes.unwrap();
        let pool = WorkingMemoryPool::new(bytes + h + n, 0).unwrap();
        let registration = register(&pool, &source);
        let (reservation, run) = fresh(&pool, h - 1);
        let mut native = run.scope().unwrap();
        let roots = RefCell::new(vec![]);
        let before = pool.used_bytes().unwrap();
        let error = plan
            .transfer(&run, &reservation, &mut native, &stream, &roots)
            .unwrap_err();
        assert!(matches!(
            error,
            CaptureTensorExecutionError::Mechanism(CaptureTensorNativeError::Host(
                CaptureTensorConstructionError::Memory(WorkingMemoryError::BudgetExceeded { .. })
            ))
        ));
        drop(error);
        assert!(roots.borrow().is_empty());
        assert_eq!(pool.used_bytes().unwrap(), before);
        native.certify().unwrap();
        drop((run, reservation, registration));
        assert_eq!(pool.used_bytes().unwrap(), 0);
        for panic in [false, true] {
            let pool = WorkingMemoryPool::new(bytes + h + n, 0).unwrap();
            let registration = register(&pool, &source);
            let (reservation, run) = fresh(&pool, h + n);
            let mut native = run.scope().unwrap();
            let roots = RefCell::new(vec![]);
            if panic {
                PANIC_AFTER_CAST.set(true);
            } else {
                FAIL_AFTER_CAST.set(true);
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                let transfer = PreparedCaptureTensor::new(&source, host(&admitted))
                    .unwrap()
                    .transfer(&run, &reservation, &mut native, &stream, &roots);
                if panic {
                    drop(transfer);
                } else {
                    let error = transfer.unwrap_err();
                    assert!(cause(error.source().unwrap()));
                    drop(error);
                }
            }));
            if panic {
                assert!(result.is_err());
            } else {
                result.unwrap();
            }
            assert_eq!(roots.borrow().len(), 3);
            assert_eq!(roots.borrow().last().unwrap().dtype(), Dtype::Float32);
            drop(registration);
            assert_eq!(pool.used_bytes().unwrap(), bytes + h + n);
            settle(&roots);
            native.certify().unwrap();
            drop((run, reservation));
            assert_eq!(pool.used_bytes().unwrap(), 0);
            assert_eq!(reference(&source, &stream).len(), 16);
        }
    }
}
