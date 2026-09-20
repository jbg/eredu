use super::*;
use eredu_nn::{
    EmbeddingOperator, EmbeddingSpec, LinearFormatSpec, NeuralBackend, ParameterMetadata,
    ParameterSpec, ParameterVisitorMut, Parameterized, Tensor,
};

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts { page_size: 16384, cpu_header: false },
        sdpa_blocks: None,
    }
}
fn meta(shape: &[i32], dtype: WorkspaceDtype, context: &WorkspaceContext) -> WorkspaceTensor {
    WorkspaceTensor::existing(WorkspaceLayout::new(shape, dtype).unwrap(), context).unwrap()
}
struct Bind<'a, T>(&'a T);
impl<'a, T: Tensor + 'a> ParameterVisitorMut<'a, T> for Bind<'_, T> {
    fn visit_mut(&mut self, _: eredu_nn::ParameterMetadataView<'_>, value: &'a mut T) {
        *value = self.0.clone();
    }
}
fn embedding<B: NeuralBackend>(
    weight: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
) -> B::Embedding {
    let mut value = B::embedding(
        EmbeddingSpec {
            vocabulary: weight.shape()[0],
            dimensions: weight.shape()[1],
            weight: ParameterSpec::trainable("table.weight").unwrap(),
            format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        },
        context,
    )
    .unwrap();
    value.visit_parameters_mut(&mut Bind(weight));
    value
}

#[test]
fn dense_lookup_charges_selected_rows_and_validation_not_the_whole_table() {
    for shape in [vec![], vec![7], vec![2, 3], vec![0], vec![4097]] {
        for dtype in [WorkspaceDtype::Int32, WorkspaceDtype::Uint32] {
            for policy in [
                EmbeddingLookupPolicy::Strict,
                EmbeddingLookupPolicy::ZeroSentinel(-7),
            ] {
                let quote = |vocabulary| {
                    let context = WorkspaceContext::new(mechanisms());
                    let ids = meta(&shape, dtype, &context);
                    let weight = meta(&[vocabulary, 13], WorkspaceDtype::Float32, &context);
                    let output = embedding::<WorkspaceBackend>(&weight, &context)
                        .lookup(&ids, policy, &context)
                        .unwrap();
                    let mut expected = shape.clone();
                    expected.push(13);
                    assert_eq!(output.shape(), expected);
                    let report = context.report(&[output]).unwrap();
                    assert!(report.tensor_buffers.transient_bytes.unwrap() > 0);
                    report.tensor_buffers.total_bytes.unwrap()
                };
                assert_eq!(quote(37), quote(1_000_003));
            }
        }
    }
}

#[test]
fn gather_retains_its_exact_axis_and_rejects_bad_domains_before_pricing() {
    for axis in 0..3 {
        for shape in [vec![], vec![2, 4], vec![0]] {
            let context = WorkspaceContext::new(mechanisms());
            let source = meta(&[3, 5, 7], WorkspaceDtype::Float32, &context);
            let ids = meta(&shape, WorkspaceDtype::Uint32, &context);
            let output = source.take_axis(&ids, axis as i32 - 3, &context).unwrap();
            let report = context.report(&[output]).unwrap();
            assert!(report.tensor_buffers.total_bytes.is_some());
            assert!(
                matches!(report.operations[0].kind,WorkspaceOperationKind::Gather {axis: selected} if selected == axis)
            );
        }
    }
    let context = WorkspaceContext::new(mechanisms());
    let empty = meta(&[0, 7], WorkspaceDtype::Float32, &context);
    let ids = meta(&[2], WorkspaceDtype::Int32, &context);
    assert!(empty.take_axis(&ids, 0, &context).is_err());
    let floating = meta(&[2], WorkspaceDtype::Float32, &context);
    assert!(empty.take_axis(&floating, 0, &context).is_err());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::{
        backend::nn::{shared::MlxNeuralBackend, tensor::TokenValidationScope},
        MlxTensor,
    };
    use safemlx::{
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };

    fn count(shape: &[i32]) -> usize {
        shape.iter().map(|n| *n as usize).product()
    }
    fn floats(
        values: &[f32],
        shape: &[i32],
        dtype: Dtype,
        strided: bool,
        stream: &Stream,
    ) -> MlxTensor {
        let storage = if strided {
            values.iter().flat_map(|v| [*v, 0.75]).collect()
        } else {
            values.to_vec()
        };
        let mut value = Array::from_slice(&storage, &[storage.len() as i32])
            .as_dtype(dtype, stream)
            .unwrap();
        if strided {
            value = value.try_index_device((..).stride_by(2), stream).unwrap();
        }
        MlxTensor::from_array(value.reshape(shape, stream).unwrap())
    }
    fn indices(
        values: &[i32],
        shape: &[i32],
        dtype: Dtype,
        strided: bool,
        stream: &Stream,
    ) -> MlxTensor {
        let storage = if strided {
            values.iter().flat_map(|v| [*v, 0]).collect()
        } else {
            values.to_vec()
        };
        let mut value = Array::from_slice(&storage, &[storage.len() as i32])
            .as_dtype(dtype, stream)
            .unwrap();
        if strided {
            value = value.try_index_device((..).stride_by(2), stream).unwrap();
        }
        MlxTensor::from_array(value.reshape(shape, stream).unwrap())
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_dense_embedding_retains_validation_and_fits_cold_bounds() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let vocabulary = 37;
        let width = 13;
        let weights = (0..vocabulary * width)
            .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
            .collect::<Vec<_>>();
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for unsigned in [false, true] {
                for strided in [false, true] {
                    for shape in [vec![], vec![7], vec![2, 3], vec![0], vec![4097]] {
                        for sentinel in [false, true] {
                            for invalid in [false, true] {
                                if invalid && count(&shape) == 0 {
                                    continue;
                                }
                                let policy = if sentinel {
                                    EmbeddingLookupPolicy::ZeroSentinel(-7)
                                } else {
                                    EmbeddingLookupPolicy::Strict
                                };
                                let mut values = (0..count(&shape))
                                    .map(|i| {
                                        if sentinel && i % 3 == 0 {
                                            -7
                                        } else {
                                            (i * 5 % vocabulary as usize) as i32
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                if invalid {
                                    values[0] = vocabulary + 1;
                                }
                                let context = WorkspaceContext::new(selected);
                                let ids_meta = meta(
                                    &shape,
                                    if unsigned {
                                        WorkspaceDtype::Uint32
                                    } else {
                                        WorkspaceDtype::Int32
                                    },
                                    &context,
                                );
                                let weight_meta =
                                    meta(&[vocabulary, width], WorkspaceDtype::Float32, &context);
                                let declared =
                                    embedding::<WorkspaceBackend>(&weight_meta, &context)
                                        .lookup(&ids_meta, policy, &context)
                                        .unwrap();
                                let allowed = context
                                    .report(&[])
                                    .unwrap()
                                    .tensor_buffers
                                    .total_bytes
                                    .unwrap();
                                let weight =
                                    floats(&weights, &[vocabulary, width], dtype, strided, &stream);
                                let ids = indices(
                                    &values,
                                    &shape,
                                    if unsigned {
                                        Dtype::Uint32
                                    } else {
                                        Dtype::Int32
                                    },
                                    strided,
                                    &stream,
                                );
                                let mut embedding = embedding::<MlxNeuralBackend>(&weight, &stream);
                                safemlx::transforms::eval([weight.as_array(), ids.as_array()])
                                    .unwrap();
                                stream.synchronize().unwrap();
                                let before = safemlx::memory::active_memory().unwrap();
                                safemlx::memory::reset_peak_memory().unwrap();
                                let scope = TokenValidationScope::begin().unwrap();
                                let output = embedding.lookup(&ids, policy, &stream).unwrap();
                                let validation = scope.finish();
                                let arrays = std::iter::once(output.as_array())
                                    .chain(validation.arrays())
                                    .collect::<Vec<_>>();
                                safemlx::transforms::eval(arrays).unwrap();
                                stream.synchronize().unwrap();
                                let observed = safemlx::memory::peak_memory()
                                    .unwrap()
                                    .saturating_sub(before)
                                    as u64;
                                assert!(
                                    observed <= allowed,
                                    "embedding peak {observed} exceeds {allowed}"
                                );
                                assert_eq!(validation.validate_completed().is_err(), invalid);
                                assert_eq!(output.shape(), declared.shape());
                                let table = &weights;
                                let expected = values
                                    .iter()
                                    .flat_map(|&id| {
                                        (0..width as usize).map(move |col| {
                                            if sentinel && id == -7 {
                                                0.0
                                            } else {
                                                table[(if (0..vocabulary).contains(&id) {
                                                    id
                                                } else {
                                                    0
                                                })
                                                    as usize
                                                    * width as usize
                                                    + col]
                                            }
                                        })
                                    })
                                    .collect::<Vec<_>>();
                                assert_eq!(output.to_f32_vec(&stream).unwrap(), expected);
                                eprintln!("embedding dtype={dtype:?} unsigned={unsigned} strided={strided} shape={shape:?} sentinel={sentinel} invalid={invalid} observed={observed} bound={allowed}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_gather_validates_signed_unsigned_and_strided_indices_with_bounded_storage() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for (index_dtype, meta_dtype) in [
                (Dtype::Int32, WorkspaceDtype::Int32),
                (Dtype::Uint32, WorkspaceDtype::Uint32),
                (Dtype::Uint8, WorkspaceDtype::Uint8),
            ] {
                for strided in [false, true] {
                    for axis in 0..3 {
                        for shape in [vec![], vec![2, 3], vec![0], vec![4097]] {
                            for invalid in [false, true] {
                                if invalid && count(&shape) == 0 {
                                    continue;
                                }
                                let source_shape = [3, 5, 7];
                                let extent = source_shape[axis];
                                let mut values = (0..count(&shape))
                                    .map(|i| {
                                        let id = (i % extent as usize) as i32;
                                        if index_dtype == Dtype::Int32 && i % 2 == 0 {
                                            id - extent
                                        } else {
                                            id
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                if invalid {
                                    values[0] = if index_dtype == Dtype::Uint32 {
                                        -1
                                    } else {
                                        extent
                                    };
                                }
                                let context = WorkspaceContext::new(selected);
                                let a_meta = meta(&source_shape, WorkspaceDtype::Float32, &context);
                                let i_meta = meta(&shape, meta_dtype, &context);
                                let declared =
                                    a_meta.take_axis(&i_meta, axis as i32, &context).unwrap();
                                let allowed = context
                                    .report(&[])
                                    .unwrap()
                                    .tensor_buffers
                                    .total_bytes
                                    .unwrap();
                                let data = (0..count(&source_shape))
                                    .map(|i| ((i % 19) as f32 - 9.0) / 16.0)
                                    .collect::<Vec<_>>();
                                let a = floats(&data, &source_shape, dtype, strided, &stream);
                                let ids = indices(&values, &shape, index_dtype, strided, &stream);
                                safemlx::transforms::eval([a.as_array(), ids.as_array()]).unwrap();
                                stream.synchronize().unwrap();
                                let before = safemlx::memory::active_memory().unwrap();
                                safemlx::memory::reset_peak_memory().unwrap();
                                let scope = TokenValidationScope::begin().unwrap();
                                let output = a.take_axis(&ids, axis as i32, &stream).unwrap();
                                let validation = scope.finish();
                                safemlx::transforms::eval(
                                    std::iter::once(output.as_array())
                                        .chain(validation.arrays())
                                        .collect::<Vec<_>>(),
                                )
                                .unwrap();
                                stream.synchronize().unwrap();
                                let observed = safemlx::memory::peak_memory()
                                    .unwrap()
                                    .saturating_sub(before)
                                    as u64;
                                assert!(
                                    observed <= allowed,
                                    "gather peak {observed} exceeds {allowed}"
                                );
                                assert_eq!(validation.validate_completed().is_err(), invalid);
                                assert_eq!(output.shape(), declared.shape());
                                let prefix = count(&source_shape[..axis]);
                                let suffix = count(&source_shape[axis + 1..]);
                                let mut expected = Vec::new();
                                for p in 0..prefix {
                                    for (i, &id) in values.iter().enumerate() {
                                        let selected = if invalid && i == 0 {
                                            0
                                        } else if id < 0 {
                                            id + extent
                                        } else {
                                            id
                                        };
                                        for s in 0..suffix {
                                            expected.push(
                                                data[(p * extent as usize + selected as usize)
                                                    * suffix
                                                    + s],
                                            );
                                        }
                                    }
                                }
                                assert_eq!(output.to_f32_vec(&stream).unwrap(), expected);
                                eprintln!("gather dtype={dtype:?} indices={index_dtype:?} strided={strided} axis={axis} shape={shape:?} invalid={invalid} observed={observed} bound={allowed}");
                            }
                        }
                    }
                }
            }
        }
    }
}
