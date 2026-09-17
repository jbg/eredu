//! Fast native Split shares nonempty pieces but normalizes empty pieces before
//! later unary/copy kernels can use a stale broadcast backing extent.

use super::*;
use crate::MlxTensor;
use eredu_nn::Tensor;
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use std::cell::Cell;

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|calls| calls.set(calls.get() + 1));
}
struct ColdProbe;
impl Drop for ColdProbe {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn cold<T>(f: impl FnOnce() -> T) -> T {
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let _probe = ColdProbe;
    HOUSEKEEPING.with(|calls| calls.set(0));
    let result = f();
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    result
}

fn source(stream: &Stream, dtype: Dtype, axis: usize) -> (Array, Vec<f32>) {
    let values = (0..1024)
        .map(|i| 1.0 + (i % 7) as f32 * 0.5)
        .collect::<Vec<_>>();
    let shape = if axis == 0 { [1, 1024] } else { [1024, 1] };
    let source = Array::from_slice(&values, &shape)
        .as_dtype(dtype, stream)
        .unwrap();
    safemlx::transforms::eval([&source]).unwrap();
    stream.synchronize().unwrap();
    (source, values)
}
fn pieces(source: &Array, axis: usize, stream: &Stream) -> Vec<Array> {
    let shape = if axis == 0 { [4, 1024] } else { [1024, 4] };
    safemlx::ops::broadcast_to(source, &shape, stream)
        .unwrap()
        .split_axis(&[1, 1], Some(axis as i32), stream)
        .unwrap()
}
fn values(array: &Array, stream: &Stream) -> Vec<f32> {
    array
        .as_dtype(Dtype::Float32, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

#[test]
fn duplicate_interior_split_normalizes_empty_storage_and_keeps_nonempty_aliases() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for axis in [0, 1] {
        let (source, expected) = source(&stream, Dtype::Float32, axis);
        let original = source
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .unwrap();
        assert!(original.bytes() >= 4096);
        let pieces = pieces(&source, axis, &stream);
        cold(|| {
            assert_eq!(
                pieces[1].try_metadata_snapshot().unwrap().allocation(),
                None
            );
        });
        safemlx::transforms::eval(&pieces).unwrap();
        stream.synchronize().unwrap();
        cold(|| {
            assert_eq!(
                pieces[0].try_metadata_snapshot().unwrap().allocation(),
                Some(original)
            );
            assert_eq!(
                pieces[2].try_metadata_snapshot().unwrap().allocation(),
                Some(original)
            );
            let empty = pieces[1].try_metadata_snapshot().unwrap();
            assert_eq!(empty.nbytes(), 0);
            assert_eq!(empty.allocation().unwrap().bytes(), 0);
        });
        // The nonempty pieces retain the real source after its original handle
        // retires. The empty middle is separately normalized, not a dense copy.
        drop(source);
        assert_eq!(values(&pieces[0], &stream), expected);
        assert!(values(&pieces[1], &stream).is_empty());
        let expected_tail = if axis == 0 {
            expected.repeat(3)
        } else {
            expected.iter().flat_map(|&value| [value; 3]).collect()
        };
        assert_eq!(values(&pieces[2], &stream), expected_tail);
        assert_eq!(
            pieces[2].try_metadata_snapshot().unwrap().allocation(),
            Some(original)
        );
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_duplicate_interior_split_unary_and_cast_fit_actual_cold_bound() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
        for axis in [0, 1] {
            let (source, expected) = source(&stream, dtype, axis);
            let pieces = pieces(&source, axis, &stream);
            safemlx::transforms::eval(&pieces).unwrap();
            stream.synchronize().unwrap();
            let source_info = source
                .try_metadata_snapshot()
                .unwrap()
                .allocation()
                .unwrap();
            // Keep the source and all three pieces live throughout measurement:
            // donation cannot conceal a separate positive output allocation.
            for index in [1, 0] {
                let input = &pieces[index];
                let context = WorkspaceContext::new(selected);
                let allowed = cold(|| {
                    let mut projection = ExistingArrayProjection::new(&context);
                    let metadata = projection.project(input).unwrap();
                    assert!(projection.is_complete());
                    let result = metadata.square(&context).unwrap();
                    assert_eq!(result.shape(), input.shape());
                    context
                        .report(&[result])
                        .unwrap()
                        .tensor_buffers
                        .total_bytes
                        .unwrap()
                });
                // Empty shape cannot hide a large byte allowance in the fact.
                if index == 1 {
                    assert!(allowed > 0 && allowed < 1024, "bound={allowed}");
                }
                let before = safemlx::memory::active_memory().unwrap();
                safemlx::memory::reset_peak_memory().unwrap();
                let squared = MlxTensor::from_array(input.clone())
                    .square(&stream)
                    .unwrap();
                safemlx::transforms::eval([squared.as_array()]).unwrap();
                stream.synchronize().unwrap();
                let observed = safemlx::memory::peak_memory()
                    .unwrap()
                    .saturating_sub(before) as u64;
                assert!(
                    observed <= allowed,
                    "{dtype:?} axis={axis} piece={index}: {observed}>{allowed}"
                );
                if index == 1 {
                    assert_eq!(observed, 0);
                    assert_eq!(safemlx::memory::active_memory().unwrap(), before);
                    assert_eq!(
                        squared
                            .as_array()
                            .try_metadata_snapshot()
                            .unwrap()
                            .allocation()
                            .unwrap()
                            .bytes(),
                        0
                    );
                } else {
                    assert!(observed > 0, "nonempty control must really allocate");
                    let output_info = squared
                        .as_array()
                        .try_metadata_snapshot()
                        .unwrap()
                        .allocation()
                        .unwrap();
                    assert_ne!(output_info.identity(), source_info.identity());
                }
                let actual = squared.to_f32_vec(&stream).unwrap();
                if index == 1 {
                    assert!(actual.is_empty());
                } else {
                    assert_eq!(actual, expected.iter().map(|v| v * v).collect::<Vec<_>>());
                }
                eprintln!(
                    "split-empty {dtype:?} axis={axis} piece={index}: observed={observed} cold={allowed}"
                );
            }
            // AsType's vector copy shares the same stale-extent failure. There
            // is no new generic cast fact here: measure the exact empty cast.
            let before = safemlx::memory::active_memory().unwrap();
            safemlx::memory::reset_peak_memory().unwrap();
            let cast = pieces[1].as_dtype(Dtype::Float32, &stream).unwrap();
            safemlx::transforms::eval([&cast]).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(
                safemlx::memory::peak_memory()
                    .unwrap()
                    .saturating_sub(before),
                0
            );
            assert_eq!(safemlx::memory::active_memory().unwrap(), before);
            assert_eq!(
                cast.try_metadata_snapshot()
                    .unwrap()
                    .allocation()
                    .unwrap()
                    .bytes(),
                0
            );
            assert_eq!(
                pieces[0].try_metadata_snapshot().unwrap().allocation(),
                Some(source_info)
            );
            assert_eq!(
                pieces[2].try_metadata_snapshot().unwrap().allocation(),
                Some(source_info)
            );
            assert!(
                cast.evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .is_empty()
            );
        }
    }
}

// The old descriptors must never be sent to a host read. These tests first
// evaluate the corrected native conversion, then prove its new layout/capacity.
// They are intended to run only after the revised native patch is built.
fn assert_dense_output(array: &Array, shape: &[i32], source: safemlx::AllocationIdentity) {
    assert_eq!(array.shape(), shape);
    assert!(array.signed_strides().iter().all(|&stride| stride >= 0));
    let info = array.try_metadata_snapshot().unwrap().allocation().unwrap();
    assert!(info.bytes() >= array.nbytes());
    assert_ne!(info.identity(), source);
}

fn negative_descriptor_case(
    input: &Array,
    expected: &[f32],
    stream: &Stream,
    selected: Option<MlxMetalWorkspaceMechanisms>,
) {
    input.evaluated().unwrap();
    stream.synchronize().unwrap();
    let info = input.try_metadata_snapshot().unwrap().allocation().unwrap();
    assert!(input.signed_strides().iter().any(|&stride| stride < 0));
    let quote = selected.map(|selected| {
        cold(|| {
            let context = WorkspaceContext::new(selected);
            let mut projection = ExistingArrayProjection::new(&context);
            let meta = projection.project(input).unwrap();
            assert!(projection.is_complete());
            let output = meta.square(&context).unwrap();
            (
                context
                    .report(&[output])
                    .unwrap()
                    .tensor_buffers
                    .total_bytes
                    .unwrap(),
                selected
                    .allocation()
                    .buffer_capacity(expected.len() as u64 * 4)
                    .unwrap(),
            )
        })
    });
    let before_cast = quote.map(|_| {
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        before
    });
    let cast = input.as_dtype(Dtype::Float32, stream).unwrap();
    safemlx::transforms::eval([&cast]).unwrap();
    stream.synchronize().unwrap();
    assert_dense_output(&cast, input.shape(), info.identity());
    if let Some(((_, bound), before)) = quote.zip(before_cast) {
        let observed = safemlx::memory::peak_memory()
            .unwrap()
            .saturating_sub(before) as u64;
        assert!(
            observed > 0 && observed <= bound,
            "cast {observed} > {bound}"
        );
        assert!(
            cast.try_metadata_snapshot()
                .unwrap()
                .allocation()
                .unwrap()
                .bytes() as u64
                <= bound
        );
    }
    // Only after the guards above may the test read the newly allocated cast.
    assert_eq!(
        cast.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        expected
    );
    let before_square = quote.map(|_| {
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        before
    });
    let square = MlxTensor::from_array(input.clone()).square(stream).unwrap();
    safemlx::transforms::eval([square.as_array()]).unwrap();
    stream.synchronize().unwrap();
    assert_dense_output(square.as_array(), input.shape(), info.identity());
    if let Some(((bound, _), before)) = quote.zip(before_square) {
        let observed = safemlx::memory::peak_memory()
            .unwrap()
            .saturating_sub(before) as u64;
        assert!(
            observed > 0 && observed <= bound,
            "square {observed} > {bound}"
        );
    }
    assert_eq!(
        values(square.as_array(), stream),
        expected.iter().map(|x| x * x).collect::<Vec<_>>()
    );
    assert_eq!(
        input.try_metadata_snapshot().unwrap().allocation(),
        Some(info)
    );
}

fn negative_descriptor_cases(stream: &Stream, selected: Option<MlxMetalWorkspaceMechanisms>) {
    use safemlx::ops::indexing::{TryIndexMutOp, TryIndexOp};
    for dtype in [Dtype::Float16, Dtype::Bfloat16] {
        let root = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
            .as_dtype(dtype, stream)
            .unwrap();
        let reversed = root.as_strided(&[6][..], &[-1][..], 5, stream).unwrap();
        let pieces = reversed.split_axis(&[3], Some(0), stream).unwrap();
        safemlx::transforms::eval(&pieces).unwrap();
        stream.synchronize().unwrap();
        let original = root.try_metadata_snapshot().unwrap().allocation().unwrap();
        for piece in &pieces {
            assert_eq!(
                piece.try_metadata_snapshot().unwrap().allocation(),
                Some(original)
            );
        }
        // Both offsets remain valid after the original handles retire. Holding
        // the sibling pieces also rules out donation hiding a new allocation.
        drop((root, reversed));
        negative_descriptor_case(&pieces[0], &[6.0, 5.0, 4.0], stream, selected);
        negative_descriptor_case(&pieces[1], &[3.0, 2.0, 1.0], stream, selected);
        // SliceUpdate consults data_size==1 independently of contiguous. A
        // flag-only repair would incorrectly replicate the first value here.
        let mut updated = pieces[0].clone();
        let replacement = Array::from_slice(&[9.0_f32], &[1])
            .as_dtype(dtype, stream)
            .unwrap();
        updated
            .try_index_mut_device(1..2, &replacement, stream)
            .unwrap();
        updated.evaluated().unwrap();
        assert_dense_output(&updated, &[3], original.identity());
        assert_eq!(values(&updated, stream), [6.0, 9.0, 4.0]);
        assert_eq!(
            pieces[1].try_metadata_snapshot().unwrap().allocation(),
            Some(original)
        );

        // Safe overlapping source: all touched coordinates stay within five
        // initialized elements. The selected span and positive-stride product
        // both equal four, despite the meaningful negative first-axis stride.
        let root = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0], &[5])
            .as_dtype(dtype, stream)
            .unwrap();
        let overlapping = root
            .as_strided(&[3, 2, 2][..], &[-1, 1, 1][..], 2, stream)
            .unwrap();
        let selected_slice = overlapping
            .try_index_device((1..3, .., ..), stream)
            .unwrap();
        selected_slice.evaluated().unwrap();
        let original = root.try_metadata_snapshot().unwrap().allocation().unwrap();
        assert_eq!(
            selected_slice.try_metadata_snapshot().unwrap().allocation(),
            Some(original)
        );
        drop((root, overlapping));
        negative_descriptor_case(
            &selected_slice,
            &[2.0, 3.0, 3.0, 4.0, 1.0, 2.0, 2.0, 3.0],
            stream,
            selected,
        );
    }
}

#[test]
fn corrected_negative_split_and_overlapping_slice_preserve_cpu_values_and_span() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    negative_descriptor_cases(&stream, None);
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_corrected_negative_split_and_overlapping_slice_fit_cold_cast_and_square_bounds() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    negative_descriptor_cases(
        &stream,
        Some(MlxMetalWorkspaceMechanisms::current_host().unwrap()),
    );
}
