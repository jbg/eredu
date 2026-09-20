use super::*;
use eredu_nn::workspace::{
    WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound, WorkspaceOperationKind,
};
use safemlx::{ops::indexing::TryIndexOp, Device, DeviceType};
use std::cell::Cell;

#[derive(Debug)]
struct Unpriced;
impl WorkspaceMechanisms for Unpriced {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct HousekeepingGuard;
impl Drop for HousekeepingGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn borrowed_copy_and_projection_keep_lazy_sources_unknown_without_housekeeping() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[1_f32, 2., 3., 4., 5., 6.], &[2, 3]);
    let lazy = root.transpose_axes(&[1, 0], &stream).unwrap();
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    let context = WorkspaceContext::new(Unpriced);
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = HousekeepingGuard;
    HOUSEKEEPING.set(0);
    let plan = IsolatedArrayCopy::new(&lazy);
    let mut projection = ExistingArrayProjection::new(&context);
    let source = projection.project(&lazy).unwrap();
    context.begin_state_span(&[source.clone()]).unwrap();
    let destination = plan.trace(&mut projection).unwrap();
    let report = context.report(&[source, destination]).unwrap();
    assert!(!projection.is_complete());
    assert!(report.total_bytes.is_none());
    assert!(report.state.unwrap().retained_bytes.is_none());
    assert!(matches!(
        report.operations[0].kind,
        WorkspaceOperationKind::Contiguous
    ));
    assert!(matches!(
        report.operations[1].kind,
        WorkspaceOperationKind::DeepCopy
    ));
    assert_eq!(report.operations.len(), 2);
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    assert_eq!(HOUSEKEEPING.get(), 0);
    drop(guard);
    let copied = plan.copy(&stream).unwrap();
    assert_eq!(copied.shape(), [3, 2]);
    assert_eq!(
        copied.evaluated().unwrap().as_slice::<f32>(),
        [1., 4., 2., 5., 3., 6.]
    );
}

#[test]
fn shared_copy_program_preserves_strided_broadcast_and_row_values_independently() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[1_f32, 2., 3., 4., 5., 6., 7., 8.], &[2, 4]);
    let padded = root.try_index_device((.., 1..3), &stream).unwrap();
    let transposed = root.transpose_axes(&[1, 0], &stream).unwrap();
    let row = root.try_index_device((0..1, ..), &stream).unwrap();
    let broadcast = safemlx::ops::broadcast_to(&row, &[3, 4], &stream).unwrap();
    for source in [&root, &padded, &transposed, &broadcast] {
        let expected = source.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        let original = source.allocation_info().unwrap().unwrap();
        let copied = IsolatedArrayCopy::new(source).copy(&stream).unwrap();
        assert_eq!(copied.shape(), source.shape());
        assert_eq!(copied.dtype(), source.dtype());
        assert_eq!(copied.evaluated().unwrap().as_slice::<f32>(), expected);
        assert_ne!(
            copied.allocation_info().unwrap().unwrap().identity(),
            original.identity()
        );
        assert_eq!(source.allocation_info().unwrap().unwrap(), original);
    }
}

#[test]
fn retained_copy_preserves_compaction_and_destination_until_collector_retirement() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[1_f32, 2., 3., 4., 5., 6., 7., 8.], &[2, 4]);
    let view = source.try_index_device((.., 1..3), &stream).unwrap();
    let roots = RefCell::new(Vec::new());
    let copy = IsolatedArrayCopy::new(&view)
        .copy_retained(&stream, &roots)
        .unwrap();
    let identity = copy.allocation_info().unwrap().unwrap().identity();
    assert_ne!(
        identity,
        source.allocation_info().unwrap().unwrap().identity()
    );
    assert_eq!(roots.borrow().len(), 2);
    assert_eq!(
        roots.borrow()[1]
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity(),
        identity
    );
    drop(copy);
    drop(view);
    drop(source);
    for root in roots.borrow().iter() {
        assert_eq!(
            root.evaluated().unwrap().as_slice::<f32>(),
            [2., 3., 6., 7.]
        );
    }
    roots.borrow_mut().clear();
}

#[test]
fn unsupported_workspace_dtype_does_not_evaluate_or_disable_the_native_copy() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[11_i64, -7, 23, 41], &[2, 2]);
    let lazy = root.transpose_axes(&[1, 0], &stream).unwrap();
    let context = WorkspaceContext::new(Unpriced);
    let plan = IsolatedArrayCopy::new(&lazy);
    let mut projection = ExistingArrayProjection::new(&context);
    assert!(plan.trace(&mut projection).is_err());
    assert!(context.report(&[]).unwrap().operations.is_empty());
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    let copied = plan.copy(&stream).unwrap();
    assert_eq!(copied.dtype(), safemlx::Dtype::Int64);
    assert_eq!(
        copied.evaluated().unwrap().as_slice::<i64>(),
        [11, 23, -7, 41]
    );
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn metal_copy_program_report_covers_source_destination_and_compaction_overlap() {
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let root = Array::from_slice(&(1..=96).map(|n| n as f32).collect::<Vec<_>>(), &[2, 16, 3]);
    let view = root.try_index_device((.., 4..9, ..), &stream).unwrap();
    let expected = view.evaluated().unwrap().try_to_vec::<f32>().unwrap();
    let backing = view.allocation_info().unwrap().unwrap();
    assert!(backing.bytes() > view.nbytes());
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let plan = IsolatedArrayCopy::new(&view);
    let mut projection = ExistingArrayProjection::new(&context);
    let source = projection.project(&view).unwrap();
    context.begin_state_span(&[source.clone()]).unwrap();
    let destination = plan.trace(&mut projection).unwrap();
    let report = context.report(&[source, destination]).unwrap();
    assert!(projection.is_complete());
    assert_eq!(report.operations.len(), 2);
    assert_eq!(report.host_workspace_bytes, Some(0));
    assert!(report.unpriced_operations.is_empty());
    assert!(report.unpriced_host_operations.is_empty());
    let copied = plan.copy(&stream).unwrap();
    assert_eq!(copied.evaluated().unwrap().as_slice::<f32>(), expected);
    let destination = copied.allocation_info().unwrap().unwrap();
    assert_ne!(destination.identity(), backing.identity());
    let state = report.state.unwrap();
    assert!(state.retained_bytes.unwrap() >= (backing.bytes() + destination.bytes()) as u64);
    assert!(report.retained_bytes.unwrap() >= destination.bytes() as u64);
    assert!(report.transient_bytes.unwrap() > 0);
    assert_eq!(state.displaced_bytes, Some(0));
}

#[cfg(all(target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn cpu_saved_copy_trace_preserves_floating_source_and_independent_destination() {
    use crate::backend::nn::workspace::{MlxMetalWorkspaceMechanisms,MlxCpuWorkspaceMechanisms,MlxCpuMatmulMechanism};
    let stream=Stream::new_with_device(&Device::new(DeviceType::Cpu,0));
    let native=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(native.allocation(),selected);
    let root=Array::from_slice(&(1..=40).map(|n|n as f32-17.).collect::<Vec<_>>(),&[1,1,5,8]);
    let padded=root.try_slice(&[0,0,0,0],&[1,1,5,4],&[1,1,1,1],&stream).unwrap();
    for source in [&root,&padded] {
        let expected=source.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        let before=source.allocation_info().unwrap().unwrap();
        let context=WorkspaceContext::new(cpu);
        let mut projection=ExistingArrayProjection::new(&context);
        let opening=projection.project(source).unwrap();
        context.begin_state_span([&opening]).unwrap();
        let worker=IsolatedArrayCopy::new(source);
        let traced=worker.trace(&mut projection).unwrap();
        assert_eq!(traced.layout().representation(),Some(eredu_nn::workspace::WorkspaceRepresentation::new(
            eredu_nn::workspace::WorkspaceFloatingType::Float32,true)));
        let report=context.report(&[opening,traced]).unwrap();
        assert!(report.unpriced_operations.is_empty()&&report.unpriced_host_operations.is_empty());
        assert_eq!(report.operations.len(),2);
        let copied=worker.copy(&stream).unwrap();
        let after=copied.allocation_info().unwrap().unwrap();
        assert_ne!(before.identity(),after.identity());
        assert_eq!(copied.evaluated().unwrap().as_slice::<f32>(),expected);
        assert!(report.state.as_ref().unwrap().retained_bytes.unwrap()>=(before.bytes()+after.bytes())as u64);
        assert_eq!(report.state.unwrap().displaced_bytes,Some(0));
        drop(copied);
        assert_eq!(source.evaluated().unwrap().try_to_vec::<f32>().unwrap(),expected);
    }
    // A logical F32 descriptor without physical source evidence stays unknown.
    let context=WorkspaceContext::new(cpu);
    let source=eredu_nn::workspace::WorkspaceTensor::existing(context.layout(&[1,1,5,4],
        eredu_nn::workspace::WorkspaceDtype::Float32).unwrap(),&context).unwrap();
    let traced=eredu_nn::isolated_copy(eredu_nn::WorkspaceIsolatedCopy::new(source,&context)).unwrap();
    assert!(traced.layout().representation().is_none());
    assert!(!context.report(&[traced]).unwrap().unpriced_operations.is_empty());
}
