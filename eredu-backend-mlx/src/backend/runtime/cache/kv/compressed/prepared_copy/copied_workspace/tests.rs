use super::*;
use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
use eredu_nn::{
    CompressedAttentionCache, CompressedAttentionState, Tensor,
    workspace::{WorkspaceContext, WorkspaceOperationKind, WorkspaceTensor},
};
use safemlx::{Device, DeviceType, Stream};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

fn nz(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}
fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Gpu, 0))
}
fn context() -> WorkspaceContext {
    WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap())
}
fn payload(start: i32, count: i32, width: i32) -> Vec<f32> {
    (0..2)
        .flat_map(|batch| {
            (start..start + count).flat_map(move |position| {
                (0..width).map(move |channel| (batch * 1000 + position * 10 + channel + 1) as f32)
            })
        })
        .collect()
}
fn append_native(
    cache: &mut super::super::super::CompressedLatentCache,
    count: i32,
    stream: &Stream,
) {
    let position = cache.offset();
    cache
        .update_and_fetch(
            Array::from_slice(&payload(position, count, 8), &[2, count, 8]),
            Array::from_slice(&payload(position, count, 4), &[2, count, 4]),
            stream,
        )
        .unwrap();
    for array in cache.retained_arrays() {
        array.evaluated().unwrap();
    }
}
fn values(array: &Array, stream: &Stream) -> Vec<f32> {
    array
        .contiguous(false, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
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
    fn check(&self) {
        assert_eq!(HOUSEKEEPING.get(), 0);
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn detached_four_root_source_projects_two_compact_destinations_and_matches_cached_growth() {
    let stream = stream();
    let mut original = super::super::super::CompressedLatentCache::new();
    append_native(&mut original, 3, &stream);
    let source = original.deep_clone_state().unwrap();
    for array in source.retained_arrays() {
        array.evaluated().unwrap();
    }
    let source_facts: BTreeMap<_, _> = source
        .retained_arrays()
        .iter()
        .map(|array| {
            let info = array.allocation_info().unwrap().unwrap();
            (info.identity(), info.bytes() as u64)
        })
        .collect();
    assert_eq!(source_facts.len(), 4);
    let plan = source.prepare_isolated_copy().unwrap();
    let mut operands = 0;
    plan.visit_operands(&mut |_| operands += 1);
    assert_eq!(operands, 2);
    let context = context();
    let cold = ColdGuard::new();
    let mut projection = ExistingArrayProjection::new(&context);
    let mut roots = Vec::new();
    plan.visit_retained_arrays(&mut |array| roots.push(projection.project(array).unwrap()));
    assert_eq!(roots.len(), 4);
    context.begin_state_span(&roots).unwrap();
    let mut copied = plan
        .project_copied_workspace(nz(2), nz(8), nz(4), &mut projection)
        .unwrap();
    assert_eq!(
        (copied.offset(), copied.capacity(), copied.capacity_step()),
        (3, 3, source.step)
    );
    roots.extend(copied.retained_arrays().cloned());
    let report = context.report(&roots).unwrap();
    assert_eq!(report.operations.len(), 4);
    for pair in report.operations.chunks_exact(2) {
        assert!(matches!(pair[0].kind, WorkspaceOperationKind::Contiguous));
        assert!(matches!(pair[1].kind, WorkspaceOperationKind::DeepCopy));
    }
    assert!(report.total_bytes.is_some());
    let storage = projection.try_into_storage().unwrap();
    assert_eq!(
        storage
            .iter()
            .map(|(id, n, _)| (id, n))
            .collect::<BTreeMap<_, _>>(),
        source_facts
    );
    cold.check();
    drop(cold);
    let retained = RefCell::new(Vec::new());
    let mut native = plan.copy_retained(&stream, &retained).unwrap();
    for array in native.retained_arrays() {
        array.evaluated().unwrap();
    }
    assert_eq!(retained.borrow().len(), 4);
    let native_facts: BTreeMap<_, _> = native
        .retained_arrays()
        .iter()
        .map(|array| {
            let info = array.allocation_info().unwrap().unwrap();
            (info.identity(), info.bytes())
        })
        .collect();
    assert_eq!(native_facts.len(), 2);
    assert!(native_facts.keys().all(|id| !source_facts.contains_key(id)));
    // Witnesses belong to admission only; they must not pin old native buffers
    // while we measure subsequent growth and displaced state.
    drop((storage, roots, retained));
    for count in [1, 252, 2] {
        let position = native.offset();
        context.begin_state_span(copied.retained_arrays()).unwrap();
        copied
            .append(
                CompressedAttentionState {
                    latent: WorkspaceTensor::from_f32_slice(
                        &payload(position, count, 8),
                        &[2, count, 8],
                        &context,
                    )
                    .unwrap(),
                    rotary: WorkspaceTensor::from_f32_slice(
                        &payload(position, count, 4),
                        &[2, count, 4],
                        &context,
                    )
                    .unwrap(),
                },
                &context,
            )
            .unwrap();
        append_native(&mut native, count, &stream);
        assert_eq!(
            (copied.offset(), copied.capacity()),
            (native.offset(), native.capacity())
        );
        assert!(
            context
                .report(&copied.retained_arrays().cloned().collect::<Vec<_>>())
                .unwrap()
                .total_bytes
                .is_some()
        );
        let (latent, rotary) = native.arrays().unwrap();
        assert_eq!(values(latent, &stream), payload(0, native.offset(), 8));
        assert_eq!(values(rotary, &stream), payload(0, native.offset(), 4));
    }
    assert_eq!(
        (source.offset(), source.capacity()),
        (3, original.capacity())
    );
    let (latent, rotary) = source.arrays().unwrap();
    assert_eq!(values(latent, &stream), payload(0, 3, 8));
    assert_eq!(values(rotary, &stream), payload(0, 3, 4));
}

#[test]
fn actual_empty_and_present_zero_length_sources_keep_distinct_geometry_and_grow() {
    let stream = stream();
    for present in [false, true] {
        let mut source = super::super::super::CompressedLatentCache::new();
        if present {
            append_native(&mut source, 0, &stream);
        }
        let plan = source.prepare_isolated_copy().unwrap();
        let context = context();
        let mut projection = ExistingArrayProjection::new(&context);
        let mut roots = Vec::new();
        plan.visit_retained_arrays(&mut |array| roots.push(projection.project(array).unwrap()));
        context.begin_state_span(&roots).unwrap();
        let mut copy = plan
            .project_copied_workspace(nz(2), nz(8), nz(4), &mut projection)
            .unwrap();
        assert_eq!(copy.retained_arrays().count(), if present { 4 } else { 0 });
        let mut native = plan.copy(&stream).unwrap();
        context.begin_state_span(copy.retained_arrays()).unwrap();
        copy.append(
            CompressedAttentionState {
                latent: WorkspaceTensor::from_f32_slice(&payload(0, 3, 8), &[2, 3, 8], &context)
                    .unwrap(),
                rotary: WorkspaceTensor::from_f32_slice(&payload(0, 3, 4), &[2, 3, 4], &context)
                    .unwrap(),
            },
            &context,
        )
        .unwrap();
        append_native(&mut native, 3, &stream);
        assert_eq!(
            (copy.offset(), copy.capacity()),
            (native.offset(), native.capacity())
        );
        let (latent, rotary) = native.arrays().unwrap();
        assert_eq!(values(latent, &stream), payload(0, 3, 8));
        assert_eq!(values(rotary, &stream), payload(0, 3, 4));
        assert_eq!(source.offset(), 0);
    }
}

#[test]
fn unknown_source_stays_unknown_and_wrong_geometry_rejects_without_native_work() {
    let stream = stream();
    let mut source = super::super::super::CompressedLatentCache::new();
    source.latent_storage = Some(
        Array::from_slice(&payload(0, 3, 8), &[2, 3, 8])
            .square(&stream)
            .unwrap(),
    );
    source.rotary_key_storage = Some(
        Array::from_slice(&payload(0, 3, 4), &[2, 3, 4])
            .square(&stream)
            .unwrap(),
    );
    assert!(
        source
            .latent_storage
            .as_ref()
            .unwrap()
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .is_none()
    );
    source.latent = source.latent_storage.clone();
    source.rotary_key = source.rotary_key_storage.clone();
    source.offset = 3;
    source.length = 3;
    source.capacity = 3;
    let plan = source.prepare_isolated_copy().unwrap();
    let context = context();
    let cold = ColdGuard::new();
    let mut projection = ExistingArrayProjection::new(&context);
    let mut roots = Vec::new();
    plan.visit_retained_arrays(&mut |array| roots.push(projection.project(array).unwrap()));
    context.begin_state_span(&roots).unwrap();
    assert!(
        plan.project_copied_workspace(nz(1), nz(8), nz(4), &mut projection)
            .is_err()
    );
    assert!(context.report(&roots).unwrap().operations.is_empty());
    let copy = plan
        .project_copied_workspace(nz(2), nz(8), nz(4), &mut projection)
        .unwrap();
    roots.extend(copy.retained_arrays().cloned());
    let report = context.report(&roots).unwrap();
    assert_eq!(report.state.as_ref().unwrap().retained_bytes, None);
    assert_eq!(report.inference_transient_bytes(), None);
    assert!(!projection.is_complete());
    cold.check();
}
