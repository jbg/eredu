use super::*;
use crate::backend::runtime::cache::kv::CompressedLatentCache;
use eredu_nn::{CompressedAttentionCache, CompressedAttentionState, Tensor};
use eredu_runtime::{working_memory::WorkspaceCompressedCache, RuntimeLayerState};
use safemlx::{Array, Device, DeviceType, Stream};
use std::num::NonZeroU32;

fn payload(start: i32, count: i32, width: i32) -> Vec<f32> {
    (0..2)
        .flat_map(|batch| {
            (start..start + count).flat_map(move |position| {
                (0..width).map(move |channel| {
                    (10000 * batch + 100 * position + channel + 1) as f32 * 0.001
                })
            })
        })
        .collect()
}
fn verify(native: &CompressedLatentCache, metadata: &WorkspaceCompressedCache, stream: &Stream) {
    assert_eq!(native.offset(), metadata.offset());
    assert_eq!(native.capacity(), metadata.capacity());
    let (latent, rotary) = native.arrays().unwrap();
    for (array, width) in [(latent, 8), (rotary, 4)] {
        assert_eq!(array.shape(), [2, native.offset(), width]);
        let contiguous = array.contiguous(false, stream).unwrap();
        safemlx::transforms::eval([&contiguous]).unwrap();
        assert_eq!(
            contiguous.evaluated().unwrap().as_slice::<f32>(),
            payload(0, native.offset(), width)
        );
    }
}
fn append(
    native: &mut CompressedLatentCache,
    metadata: &mut WorkspaceCompressedCache,
    count: i32,
    stream: &Stream,
    context: &WorkspaceContext,
) {
    let start = native.offset();
    let latent = payload(start, count, 8);
    let rotary = payload(start, count, 4);
    native
        .update_and_fetch(
            Array::from_slice(&latent, &[2, count, 8]),
            Array::from_slice(&rotary, &[2, count, 4]),
            stream,
        )
        .unwrap();
    metadata
        .append(
            CompressedAttentionState {
                latent: WorkspaceTensor::from_f32_slice(&latent, &[2, count, 8], context).unwrap(),
                rotary: WorkspaceTensor::from_f32_slice(&rotary, &[2, count, 4], context).unwrap(),
            },
            context,
        )
        .unwrap();
    verify(native, metadata, stream);
}

#[test]
#[ignore = "requires MLX Metal execution"]
fn compressed_metadata_matches_native_capacity_updates_snapshot_and_restore() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let stream = &stream;
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut native = CompressedLatentCache::new();
    let mut metadata = WorkspaceCompressedCache::new(
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(8).unwrap(),
        NonZeroU32::new(4).unwrap(),
        CompressedLatentCache::default_resident_capacity_step(),
        &context,
    )
    .unwrap();
    for count in [2, 253, 2, 7] {
        context.begin_span();
        append(&mut native, &mut metadata, count, stream, &context);
        let report = context
            .report(&metadata.retained_values().cloned().collect::<Vec<_>>())
            .unwrap();
        assert!(report.total_bytes.is_some());
        assert_eq!(report.total_bytes, report.tensor_buffers.total_bytes);
        assert_eq!(report.host_workspace_bytes, Some(0));
        assert!(report.unpriced_host_operations.is_empty());
    }
    let saved_native = native.clone();
    let saved_metadata = metadata.checkpoint();
    append(&mut native, &mut metadata, 3, stream, &context);
    native.restore_checkpoint(&saved_native, stream).unwrap();
    metadata.restore(&saved_metadata, &context).unwrap();
    verify(&native, &metadata, stream);
    append(&mut native, &mut metadata, 1, stream, &context);
    let mut native_snapshot = native.isolated_snapshot(stream).unwrap();
    let mut metadata_snapshot = metadata.isolated_snapshot(&context).unwrap();
    verify(&native_snapshot, &metadata_snapshot, stream);
    append(
        &mut native_snapshot,
        &mut metadata_snapshot,
        2,
        stream,
        &context,
    );
    verify(&native, &metadata, stream);
    native.clear().unwrap();
    metadata.clear().unwrap();
    append(&mut native, &mut metadata, 256, stream, &context);
    println!("native compressed cache conformance: 8 appends, restore, compact snapshot and continuation; batch=2, widths=8/4, step=256, exact nonzero F32 equality");
}
