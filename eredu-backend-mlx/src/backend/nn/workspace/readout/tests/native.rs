use super::*;
use crate::MlxTensor;
use eredu_nn::multimodal::reference_masked_output_projection;
use safemlx::{
    ops::indexing::{IntoStrideBy, TryIndexOp},
    Array, Device, DeviceType, Dtype, Stream,
};

fn floating(
    layout: &WorkspaceLayout,
    slot: usize,
    dtype: Dtype,
    strided: bool,
    stream: &Stream,
) -> Array {
    let width = *layout.shape().last().unwrap() as usize;
    let data = (0..layout.elements().unwrap() as usize)
        .map(|i| {
            if slot == 2 {
                // Distinct F32 centroid scores with a different maximum per row.
                ((i % width + (i / width) * 7) % width) as f32
            } else {
                ((i * 11 + slot * 5) % 37) as f32 / 32. - 0.5
            }
        })
        .collect::<Vec<_>>();
    let data = if strided {
        data.iter().flat_map(|v| [*v, 17.]).collect()
    } else {
        data
    };
    let mut array = Array::from_slice(&data, &[data.len() as i32])
        .as_dtype(dtype, stream)
        .unwrap();
    if strided {
        array = array.try_index_device((..).stride_by(2), stream).unwrap();
    }
    array.reshape(layout.shape(), stream).unwrap()
}
fn values(a: &MlxTensor, stream: &Stream) -> eredu_core::HostTensorBuffer<f32> {
    a.to_f32_vec(stream).unwrap()
}
fn execute(x: &[MlxTensor], top: i32, stream: &Stream) -> MlxTensor {
    MlxTensor::masked_output_projection(
        MaskedOutputProjectionInput {
            hidden: &x[0],
            output_weight: &x[1],
            centroid_logits: &x[2],
            token_ordering: &x[3],
            top_centroids: top,
            mask_margin: 1.,
        },
        stream,
    )
    .unwrap()
}
fn compare(actual: &[f32], expected: &[f32], dtype: Dtype) {
    assert_eq!(actual.len(), expected.len());
    let (atol, rtol) = if dtype == Dtype::Float32 {
        (1e-5, 1e-5)
    } else {
        (0.005, 0.01)
    };
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= atol + rtol * e.abs(),
            "{dtype:?} element {i}: {a} != {e}"
        );
    }
}

#[test]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_masked_readout_matches_reference_chunks_and_allocation_bound() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let m = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let mut cases = 0;
    for dims in [
        [1, 1, 2, 4, 2, 1],
        [2, 7, 65, 24, 4, 2],
        [2, 3, 7, 24, 4, 4],
        [1, 1, 1025, 16, 4, 1],
        [1, 2, 3, 4097, 4097, 3],
        [1, 1, 7, 4097, 1, 1],
        [2, 0, 7, 8, 2, 2],
        [0, 3, 7, 8, 2, 1],
    ] {
        let [b, s, h, v, c, top] = dims;
        for (dtype, weight_dtype) in [
            (Dtype::Float32, Dtype::Float32),
            (Dtype::Float16, Dtype::Float16),
            (Dtype::Bfloat16, Dtype::Bfloat16),
            (Dtype::Float16, Dtype::Float32),
            (Dtype::Bfloat16, Dtype::Float32),
        ] {
            for strided in [false, true] {
                for unsigned in [false, true] {
                    let op = operation(dims, unsigned);
                    let mut input = op.inputs[..3]
                        .iter()
                        .enumerate()
                        .map(|(slot, layout)| {
                            floating(
                                layout,
                                slot,
                                match slot {
                                    0 => dtype,
                                    1 => weight_dtype,
                                    _ => Dtype::Float32,
                                },
                                strided,
                                &stream,
                            )
                            .into()
                        })
                        .collect::<Vec<MlxTensor>>();
                    let ordering = (0..v).rev().collect::<Vec<_>>();
                    let data = if strided {
                        ordering.iter().flat_map(|i| [*i, 0]).collect()
                    } else {
                        ordering.clone()
                    };
                    let mut ids = Array::from_slice(&data, &[data.len() as i32]);
                    if strided {
                        ids = ids.try_index_device((..).stride_by(2), &stream).unwrap();
                    }
                    if unsigned {
                        ids = ids.as_dtype(Dtype::Uint32, &stream).unwrap();
                    }
                    input.push(ids.into());
                    safemlx::transforms::eval(input.iter().map(MlxTensor::as_array)).unwrap();
                    stream.synchronize().unwrap();
                    let before = safemlx::memory::active_memory().unwrap();
                    safemlx::memory::reset_peak_memory().unwrap();
                    let actual = execute(&input, top, &stream);
                    safemlx::transforms::eval([actual.as_array()]).unwrap();
                    stream.synchronize().unwrap();
                    let observed = safemlx::memory::peak_memory()
                        .unwrap()
                        .saturating_sub(before) as u64;
                    let allowed = total(&op, m);
                    assert!(observed <= allowed, "{dims:?} {dtype:?} strided={strided} unsigned={unsigned}: {observed}>{allowed}");
                    assert_eq!(actual.shape(), op.outputs[0].shape());
                    assert_eq!(actual.as_array().dtype(), Dtype::Float32);
                    if let Some(info) = actual.as_array().allocation_info().unwrap() {
                        assert!(
                            info.bytes() as u64
                                <= capacity(m.allocation(), op.outputs[0].elements().unwrap())
                                    .unwrap()
                        );
                    }
                    let reference = reference_masked_output_projection(
                        &values(&input[0], &stream),
                        (b * s) as usize,
                        h as usize,
                        &values(&input[1], &stream),
                        v as usize,
                        &values(&input[2], &stream),
                        c as usize,
                        &ordering.iter().map(|i| *i as usize).collect::<Vec<_>>(),
                        top as usize,
                        1.,
                    )
                    .unwrap();
                    let result = values(&actual, &stream);
                    compare(&result, &reference, dtype);
                    if s == 7 {
                        for chunk_size in [1, 3, 4] {
                            let mut chunks = vec![];
                            for start in (0..s).step_by(chunk_size) {
                                let end = (start + chunk_size as i32).min(s);
                                let mut chunk = input.clone();
                                for slot in [0, 2] {
                                    chunk[slot] = input[slot]
                                        .as_array()
                                        .try_index_device((.., start..end, ..), &stream)
                                        .unwrap()
                                        .into();
                                }
                                chunks.push(execute(&chunk, top, &stream));
                            }
                            let joined = safemlx::ops::concatenate_axis(
                                &chunks.iter().map(MlxTensor::as_array).collect::<Vec<_>>(),
                                1,
                                &stream,
                            )
                            .unwrap();
                            compare(&values(&joined.into(), &stream), &result, dtype);
                        }
                    }
                    eprintln!("MASKED_READOUT_CASE={cases} dims={dims:?} dtype={dtype:?} weight_dtype={weight_dtype:?} strided={strided} unsigned={unsigned} peak={observed} bound={allowed}");
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 160);
}

#[path = "native/selected.rs"]
mod selected;
