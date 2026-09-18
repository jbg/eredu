use super::*;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceHostBound, WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation,
    WorkspaceOperationBound, WorkspaceOutputStorage,
};
use safemlx::{Device, DeviceType, ops::indexing::TryIndexOp};
use std::{
    cell::Cell,
    error::Error as _,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

thread_local! {
    static AFTER_ISOLATION: RefCell<Option<Exception>> = const { RefCell::new(None) };
    static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) };
}
pub(super) fn after_isolation() -> Result<(), Exception> {
    AFTER_ISOLATION.with(|fault| fault.borrow_mut().take().map_or(Ok(()), Err))
}
struct Fault;
impl Fault {
    fn new(error: Exception) -> Self {
        AFTER_ISOLATION.with(|fault| assert!(fault.borrow_mut().replace(error).is_none()));
        Self
    }
    fn assert_consumed(&self) {
        AFTER_ISOLATION.with(|fault| assert!(fault.borrow().is_none()));
    }
}
impl Drop for Fault {
    fn drop(&mut self) {
        AFTER_ISOLATION.with(|fault| drop(fault.borrow_mut().take()));
    }
}
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct ColdGuard;
impl ColdGuard {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
    fn assert_cold(&self) {
        assert_eq!(HOUSEKEEPING.get(), 0);
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}
fn scalar_view(signed: bool, stream: &Stream) -> Array {
    let root = if signed {
        Array::from_slice(&(700..1724).collect::<Vec<i32>>(), &[1024])
    } else {
        Array::from_slice(&(700..1724).collect::<Vec<u32>>(), &[1024])
    };
    let view = root.try_index_device(23..24, stream).unwrap();
    view.evaluated().unwrap();
    assert!(view.allocation_info().unwrap().unwrap().bytes() > view.nbytes());
    view
}
fn read_scalar(array: &Array) -> u32 {
    let evaluated = array.evaluated().unwrap();
    match array.dtype() {
        Dtype::Int32 => evaluated.try_to_vec::<i32>().unwrap()[0] as u32,
        Dtype::Uint32 => evaluated.try_to_vec::<u32>().unwrap()[0],
        other => panic!("unexpected scalar dtype {other:?}"),
    }
}
#[derive(Debug)]
struct LogicalFacts {
    missing: bool,
}
impl WorkspaceMechanisms for LogicalFacts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        if self.missing {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "test fact allocates each output at logical extent".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        Ok((!self.missing).then(|| WorkspaceHostBound {
            bytes: 0,
            assumptions: "test fact has no host staging".into(),
        }))
    }
}
fn check_order(operations: &[WorkspaceOperation], signed: bool, source: &[i32]) {
    assert_eq!(operations.len(), if signed { 4 } else { 3 });
    assert!(matches!(
        operations[0].kind,
        WorkspaceOperationKind::Contiguous
    ));
    assert!(matches!(
        operations[1].kind,
        WorkspaceOperationKind::DeepCopy
    ));
    assert!(matches!(
        operations[2].kind,
        WorkspaceOperationKind::View("reshape")
    ));
    assert_eq!(operations[0].inputs[0].shape(), source);
    assert_eq!(operations[1].outputs[0].shape(), source);
    assert_eq!(operations[2].outputs[0].shape(), [1, 1]);
    if signed {
        assert!(matches!(
            operations[3].kind,
            WorkspaceOperationKind::Elementwise("cast_u32")
        ));
        assert_eq!(operations[3].inputs[0].dtype(), WorkspaceDtype::Int32);
        assert_eq!(operations[3].inputs[0].shape(), [1, 1]);
        assert_eq!(operations[3].outputs[0].dtype(), WorkspaceDtype::Uint32);
        assert_eq!(operations[3].outputs[0].shape(), [1, 1]);
    }
}

#[test]
fn scalar_and_backing_views_follow_the_same_copy_program_without_collector_growth() {
    let stream = stream();
    let cases = [
        (Array::from_slice(&[73_u32], &[]), 73),
        (Array::from_slice(&[101_u32], &[1]), 101),
        (Array::from_slice(&[97_i32], &[1, 1, 1]), 97),
        (scalar_view(false, &stream), 723),
        (scalar_view(true, &stream), 723),
    ];
    for (source, expected) in cases {
        source.evaluated().unwrap();
        let original = source.try_metadata_snapshot().unwrap();
        let signed = source.dtype() == Dtype::Int32;
        let context = WorkspaceContext::new(LogicalFacts { missing: false });
        let guard = ColdGuard::new();
        let plan = PreparedPendingTokenInput::new(&source).unwrap();
        let count = plan.retained_descriptor_count();
        assert_eq!(count, if signed { 4 } else { 3 });
        let mut projection = ExistingArrayProjection::new(&context);
        let projected_source = projection.project(&source).unwrap();
        context
            .begin_state_span(&[projected_source.clone()])
            .unwrap();
        let projected = plan.trace(&mut projection).unwrap();
        assert_eq!(projected.layout().dtype(), WorkspaceDtype::Uint32);
        assert_eq!(projected.shape(), [1, 1]);
        let report = context.report(&[projected_source, projected]).unwrap();
        check_order(&report.operations, signed, original.shape());
        assert!(projection.is_complete());
        assert!(report.total_bytes.is_some());
        guard.assert_cold();
        drop(guard);
        drop(projection);
        let sentinel = Array::from_slice(&[991_u32], &[1]);
        let roots = RefCell::new(Vec::with_capacity(count + 7));
        roots.borrow_mut().push(sentinel);
        let capacity = roots.borrow().capacity();
        let pointer = roots.borrow().as_ptr();
        let prepared = plan.copy_retained(&stream, &roots).unwrap();
        assert_eq!(roots.borrow().len(), count + 1);
        assert_eq!(roots.borrow().capacity(), capacity);
        assert_eq!(roots.borrow().as_ptr(), pointer);
        assert_eq!(prepared.array().shape(), [1, 1]);
        assert_eq!(prepared.array().dtype(), Dtype::Uint32);
        assert_eq!(read_scalar(prepared.array()), expected);
        let destination = prepared.array().allocation_info().unwrap().unwrap();
        assert_ne!(
            destination.identity(),
            original.allocation().unwrap().identity()
        );
        assert_eq!(source.try_metadata_snapshot().unwrap(), original);
        assert_eq!(read_scalar(&source), expected);
        assert_eq!(read_scalar(&roots.borrow()[0]), 991);
        assert_eq!(
            roots
                .borrow()
                .last()
                .unwrap()
                .allocation_info()
                .unwrap()
                .unwrap()
                .identity(),
            destination.identity()
        );
        let array = prepared.into_array();
        assert_eq!(array.allocation_info().unwrap().unwrap(), destination);
        drop(source);
        drop(roots);
        assert_eq!(read_scalar(&array), expected);
    }
}

#[test]
fn invalid_scalar_shape_dtype_and_lazy_source_reject_without_native_housekeeping() {
    let stream = stream();
    let empty = Array::from_slice::<u32>(&[], &[0]);
    let pair = Array::from_slice(&[13_u32, 17], &[1, 2]);
    let float = Array::from_slice(&[19_f32], &[]);
    let wide = Array::from_slice(&[23_i64], &[1]);
    let boolean = Array::from_slice(&[true], &[1, 1]);
    let valid = Array::from_slice(&[29_u32], &[1]);
    let lazy = valid.square(&stream).unwrap();
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    let guard = ColdGuard::new();
    for source in [&empty, &pair] {
        assert!(matches!(
            PreparedPendingTokenInput::new(source),
            Err(PendingTokenInputError::InvalidShape)
        ));
    }
    for (source, expected) in [
        (&float, Dtype::Float32),
        (&wide, Dtype::Int64),
        (&boolean, Dtype::Bool),
    ] {
        assert!(
            matches!(PreparedPendingTokenInput::new(source), Err(PendingTokenInputError::InvalidDtype { actual }) if actual == expected)
        );
    }
    assert!(matches!(
        PreparedPendingTokenInput::new(&lazy),
        Err(PendingTokenInputError::UnsettledSource)
    ));
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    guard.assert_cold();
    drop(guard);
    lazy.evaluated().unwrap();
    assert!(PreparedPendingTokenInput::new(&lazy).is_ok());
    assert_eq!(read_scalar(&lazy), 29 * 29);
}

#[test]
fn missing_copy_facts_remain_unknown_despite_known_scalar_storage() {
    let stream = stream();
    let source = scalar_view(true, &stream);
    let context = WorkspaceContext::new(LogicalFacts { missing: true });
    let guard = ColdGuard::new();
    let plan = PreparedPendingTokenInput::new(&source).unwrap();
    let mut projection = ExistingArrayProjection::new(&context);
    let input = projection.project(&source).unwrap();
    context.begin_state_span(&[input.clone()]).unwrap();
    let output = plan.trace(&mut projection).unwrap();
    let report = context.report(&[input, output]).unwrap();
    assert!(projection.is_complete());
    assert!(report.total_bytes.is_none());
    assert!(report.tensor_buffers.total_bytes.is_none());
    assert!(report.host_workspace_bytes.is_none());
    assert_eq!(report.unpriced_operations, vec![0, 1, 2, 3]);
    assert_eq!(report.unpriced_host_operations, vec![0, 1, 2, 3]);
    check_order(&report.operations, true, &[1]);
    guard.assert_cold();
}

#[test]
fn busy_recovery_collector_rejects_before_native_work_and_remains_reusable() {
    let stream = stream();
    let source = Array::from_slice(&[31_i32], &[1]);
    source.evaluated().unwrap();
    let original = source.try_metadata_snapshot().unwrap();
    let roots = RefCell::new(Vec::with_capacity(4));
    let held = roots.borrow();
    let plan = PreparedPendingTokenInput::new(&source).unwrap();
    let guard = ColdGuard::new();
    let error = plan
        .copy_retained(&stream, &roots)
        .err()
        .expect("borrow conflict");
    assert!(matches!(error, PendingTokenInputError::CollectorBusy));
    assert!(held.is_empty());
    guard.assert_cold();
    drop(guard);
    drop(held);
    assert_eq!(source.try_metadata_snapshot().unwrap(), original);
    let output = PreparedPendingTokenInput::new(&source)
        .unwrap()
        .copy_retained(&stream, &roots)
        .unwrap();
    assert_eq!(roots.borrow().len(), 4);
    assert_eq!(read_scalar(output.array()), 31);
}

#[derive(Debug)]
struct LateFailure {
    id: usize,
    dropped: Arc<AtomicUsize>,
}
impl std::fmt::Display for LateFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pending input isolation sentinel {}", self.id)
    }
}
impl std::error::Error for LateFailure {}
impl Drop for LateFailure {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn late_failure_preserves_the_original_cause_and_both_isolated_recovery_roots() {
    let stream = stream();
    for signed in [false, true] {
        let source = scalar_view(signed, &stream);
        let original = source.try_metadata_snapshot().unwrap();
        let roots = RefCell::new(Vec::with_capacity(4));
        let capacity = roots.borrow().capacity();
        let pointer = roots.borrow().as_ptr();
        let dropped = Arc::new(AtomicUsize::new(0));
        let fault = Fault::new(Exception::from_source(LateFailure {
            id: 123,
            dropped: dropped.clone(),
        }));
        let error = PreparedPendingTokenInput::new(&source)
            .unwrap()
            .copy_retained(&stream, &roots)
            .err()
            .expect("injected failure");
        fault.assert_consumed();
        let native = error.source().unwrap().downcast_ref::<Exception>().unwrap();
        let cause = native
            .source()
            .unwrap()
            .downcast_ref::<LateFailure>()
            .unwrap();
        assert_eq!(cause.id, 123);
        assert!(Arc::ptr_eq(&cause.dropped, &dropped));
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        assert_eq!(roots.borrow().len(), 2);
        assert_eq!(roots.borrow().capacity(), capacity);
        assert_eq!(roots.borrow().as_ptr(), pointer);
        assert!(
            roots
                .borrow()
                .iter()
                .all(|root| root.shape() == original.shape() && root.dtype() == original.dtype())
        );
        for root in roots.borrow().iter() {
            assert_eq!(read_scalar(root), 723);
        }
        assert_ne!(
            roots.borrow()[1]
                .allocation_info()
                .unwrap()
                .unwrap()
                .identity(),
            original.allocation().unwrap().identity()
        );
        assert_eq!(source.try_metadata_snapshot().unwrap(), original);
        drop((fault, error, source));
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        // The collector remains the caller's recovery custody after the worker
        // and original source leave scope. No completion is inferred here.
        for root in roots.borrow().iter() {
            assert_eq!(read_scalar(root), 723);
        }
    }
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn metal_pending_input_bounds_cover_actual_overlap_and_only_the_closed_cast_geometry() {
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
    use std::collections::BTreeMap;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for signed in [false, true] {
        let source = scalar_view(signed, &stream);
        let original = source
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .unwrap();
        let context = WorkspaceContext::new(facts);
        let plan = PreparedPendingTokenInput::new(&source).unwrap();
        let mut projection = ExistingArrayProjection::new(&context);
        let input = projection.project(&source).unwrap();
        context.begin_state_span(&[input.clone()]).unwrap();
        let projected = plan.trace(&mut projection).unwrap();
        let report = context.report(&[input, projected]).unwrap();
        check_order(&report.operations, signed, &[1]);
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        assert_eq!(report.host_workspace_bytes, Some(0));
        let roots = RefCell::new(Vec::with_capacity(plan.retained_descriptor_count()));
        let output = plan.copy_retained(&stream, &roots).unwrap();
        assert_eq!(read_scalar(output.array()), 723);
        let actual = roots
            .borrow()
            .iter()
            .map(|array| {
                array.evaluated().unwrap();
                let allocation = array.allocation_info().unwrap().unwrap();
                (allocation.identity(), allocation.bytes() as u64)
            })
            .collect::<BTreeMap<_, _>>();
        let new_bytes: u64 = actual
            .iter()
            .filter(|(id, _)| **id != original.identity())
            .map(|(_, bytes)| *bytes)
            .sum();
        assert!(report.tensor_buffers.total_bytes.unwrap() >= new_bytes);
        let destination = output.array().allocation_info().unwrap().unwrap();
        assert_ne!(destination.identity(), original.identity());
        assert!(
            report.state.as_ref().unwrap().retained_bytes.unwrap()
                >= original.bytes() as u64 + destination.bytes() as u64
        );
    }
    let scalar = |shape: &[i32], dtype| WorkspaceLayout::new(shape, dtype).unwrap();
    let valid = WorkspaceOperation {
        kind: WorkspaceOperationKind::Elementwise("cast_u32"),
        inputs: vec![scalar(&[1, 1], WorkspaceDtype::Int32)],
        outputs: vec![scalar(&[1, 1], WorkspaceDtype::Uint32)],
    };
    let bound = facts.operation_bound(&valid).unwrap().unwrap();
    assert_eq!(bound.scratch_bytes, 0);
    assert!(
        matches!(&bound.outputs[..], [WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, inputs }] if *bytes >= 4 && inputs == &[0])
    );
    assert_eq!(
        facts.host_workspace_bound(&valid).unwrap().unwrap().bytes,
        0
    );
    for (inputs, outputs) in [(0, 1), (2, 1), (1, 0), (1, 2)] {
        let unsupported = WorkspaceOperation {
            kind: WorkspaceOperationKind::Elementwise("cast_u32"),
            inputs: vec![valid.inputs[0].clone(); inputs],
            outputs: vec![valid.outputs[0].clone(); outputs],
        };
        assert!(facts.operation_bound(&unsupported).unwrap().is_none());
        assert!(facts.host_workspace_bound(&unsupported).unwrap().is_none());
    }
    for (shape, input, output) in [
        (&[1][..], WorkspaceDtype::Int32, WorkspaceDtype::Uint32),
        (&[2, 1][..], WorkspaceDtype::Int32, WorkspaceDtype::Uint32),
        (&[1, 1][..], WorkspaceDtype::Float32, WorkspaceDtype::Uint32),
        (&[1, 1][..], WorkspaceDtype::Uint32, WorkspaceDtype::Uint32),
        (&[1, 1][..], WorkspaceDtype::Int32, WorkspaceDtype::Int32),
    ] {
        let unsupported = WorkspaceOperation {
            kind: WorkspaceOperationKind::Elementwise("cast_u32"),
            inputs: vec![scalar(shape, input)],
            outputs: vec![scalar(shape, output)],
        };
        assert!(facts.operation_bound(&unsupported).unwrap().is_none());
        assert!(facts.host_workspace_bound(&unsupported).unwrap().is_none());
    }
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn pending_prefill_matrix_uses_complete_source_and_shared_copy_program() {
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let expected = [7_u32, 19, 23, 41, 53];
    for signed in [false, true] {
        let source = if signed {
            Array::from_slice(&expected.map(|n| n as i32), &[1, 5])
        } else {
            Array::from_slice(&expected, &[1, 5])
        };
        source.evaluated().unwrap();
        let source_id = source.allocation_info().unwrap().unwrap().identity();
        let guard = ColdGuard::new();
        assert!(matches!(
            PreparedPendingTokenInput::new_prefill_fixed(
                &source,
                std::num::NonZeroU64::new(4).unwrap()
            ),
            Err(PendingTokenSourceCause::InvalidShape)
        ));
        assert!(PreparedPendingTokenInput::new_fixed(&source).is_err());
        let plan = PreparedPendingTokenInput::new_prefill_fixed(
            &source,
            std::num::NonZeroU64::new(5).unwrap(),
        )
        .unwrap();
        let population = plan.original_population().unwrap();
        assert_eq!(
            population.logical_backing_bytes,
            if signed { 60 } else { 40 }
        );
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let mut projection = ExistingArrayProjection::new(&context);
        let input = projection.project(&source).unwrap();
        context.begin_state_span(&[input.clone()]).unwrap();
        let output = plan.trace(&mut projection).unwrap();
        assert_eq!(output.shape(), [1, 5]);
        let report = context.report(&[input, output]).unwrap();
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        guard.assert_cold();
        drop(guard);
        let roots = RefCell::new(Vec::with_capacity(plan.retained_descriptor_count()));
        let capacity = roots.borrow().capacity();
        let output = plan.copy_retained(&stream, &roots).unwrap().into_array();
        assert_eq!(roots.borrow().capacity(), capacity);
        assert_eq!(output.shape(), [1, 5]);
        output.evaluated().unwrap();
        assert_ne!(
            output.allocation_info().unwrap().unwrap().identity(),
            source_id
        );
        let actual = roots
            .borrow()
            .iter()
            .map(|array| {
                array.evaluated().unwrap();
                let allocation = array.allocation_info().unwrap().unwrap();
                (allocation.identity(), allocation.bytes() as u64)
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let bytes = actual
            .iter()
            .filter(|(id, _)| **id != source_id)
            .map(|(_, b)| *b)
            .sum::<u64>();
        assert!(report.tensor_buffers.total_bytes.unwrap() >= bytes);
        drop(source);
        drop(roots);
        assert_eq!(
            output.evaluated().unwrap().try_to_vec::<u32>().unwrap(),
            expected
        );
    }
}

#[cfg(target_vendor = "apple")]
#[test]
fn cpu_pending_trace_prices_integer_scalar_and_complete_matrix_worker() {
    use crate::backend::nn::workspace::{
        MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, MetalAllocationFacts,
    };
    let native = MetalAllocationFacts::current_host().unwrap();
    let selected = MlxCpuMatmulMechanism::select(
        eredu_nn::CpuMatmulImplementation::Float32Tiles,
    ).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(native, selected);
    let stream = stream();
    let cases = [
        (Array::from_slice(&[73_u32], &[]), false),
        (scalar_view(false, &stream), false),
        (scalar_view(true, &stream), false),
        (Array::from_slice(&[97_i32], &[1, 1, 1, 1, 1]), false),
        (Array::from_slice(&[7_u32, 19, 23, 41, 53], &[1, 5]), true),
        (Array::from_slice(&[7_i32, 19, 23, 41, 53], &[1, 5]), true),
    ];
    for (source, matrix) in cases {
        source.evaluated().unwrap();
        let signed = source.dtype() == Dtype::Int32;
        let plan = if matrix {
            PreparedPendingTokenInput::new_prefill_fixed(
                &source, std::num::NonZeroU64::new(5).unwrap(),
            )
        } else {
            PreparedPendingTokenInput::new_fixed(&source)
        }.unwrap();
        let guard = ColdGuard::new();
        let context = WorkspaceContext::new(cpu);
        let mut projection = ExistingArrayProjection::new(&context);
        let input = projection.project(&source).unwrap();
        context.begin_state_span(std::slice::from_ref(&input)).unwrap();
        let output = plan.trace(&mut projection).unwrap();
        assert_eq!(output.shape(), [1, if matrix { 5 } else { 1 }]);
        assert_eq!(output.layout().dtype(), WorkspaceDtype::Uint32);
        assert!(output.layout().representation().is_none());
        let report = context.report(&[input, output]).unwrap();
        assert!(report.unpriced_operations.is_empty(), "{:?}", report.unpriced_operations);
        assert!(report.unpriced_host_operations.is_empty());
        assert!(report.tensor_buffers.total_bytes.is_some());
        assert_eq!(report.host_workspace_bytes, Some(0));
        assert_eq!(report.operations.iter().filter(|operation|
            matches!(operation.kind, WorkspaceOperationKind::Elementwise("cast_u32"))).count(),
            usize::from(signed));
        assert!(report.state.as_ref().unwrap().retained_bytes.is_some());
        guard.assert_cold();
        drop(guard);
    }
    // Integer shapes alone cannot prove that a nontrivial reshape aliases.
    // A cast also requires the exact packed single text matrix and signed input.
    for (kind, input_shape, output_shape, input_dtype, output_dtype) in [
        (WorkspaceOperationKind::View("reshape"), &[2, 3][..], &[6][..], WorkspaceDtype::Int32, WorkspaceDtype::Int32),
        (WorkspaceOperationKind::Elementwise("cast_u32"), &[2, 3][..], &[2, 3][..], WorkspaceDtype::Int32, WorkspaceDtype::Uint32),
        (WorkspaceOperationKind::Elementwise("cast_u32"), &[1, 3][..], &[1, 3][..], WorkspaceDtype::Uint32, WorkspaceDtype::Uint32),
    ] {
        let operation = WorkspaceOperation {
            kind,
            inputs: vec![WorkspaceLayout::new(input_shape, input_dtype).unwrap()],
            outputs: vec![WorkspaceLayout::new(output_shape, output_dtype).unwrap()],
        };
        assert!(cpu.operation_bound(&operation).unwrap().is_none());
        assert!(cpu.host_workspace_bound(&operation).unwrap().is_none());
    }
}
