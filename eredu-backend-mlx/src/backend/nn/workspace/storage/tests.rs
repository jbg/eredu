use super::*;
use eredu_nn::{Index, Tensor};

fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 16384,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    }
}

#[test]
fn storage_bounds_retain_capacity_and_count_shared_host_storage_once() {
    let context = WorkspaceContext::new(selected());
    let storage =
        WorkspaceTensor::from_f32_slice(&vec![1.0; 2 * 256 * 8], &[2, 256, 8], &context).unwrap();
    let update = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2, 3, 8], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    let updated = storage
        .update_slice(&update, &[0, 251, 0], &context)
        .unwrap();
    let logical = updated
        .index(&[Index::Full, Index::Range(0, 254), Index::Full], &context)
        .unwrap();
    let compact = logical.contiguous(&context).unwrap();
    let isolated = logical.deep_copy(&context).unwrap();
    let report = context.report(&[compact, isolated]).unwrap();
    assert!(report.tensor_buffers.total_bytes.is_some());
    assert!(report.retained_bytes.unwrap() >= 2 * 256 * 8 * 4);
    assert!(report.unpriced_operations.is_empty());
    assert!(report.unpriced_host_operations.is_empty());
    assert_eq!(
        report.total_bytes,
        report.tensor_buffers.total_bytes.map(
            |n| n + crate::backend::nn::workspace::test_backing_controls(&selected(), &report)
        )
    );
    assert_eq!(
        report.transient_bytes,
        report.tensor_buffers.transient_bytes.map(|n| n
            + crate::backend::nn::workspace::test_backing_controls(&selected(), &report)
            - (report.retained_bytes.unwrap() - report.tensor_buffers.retained_bytes.unwrap()))
    );
    assert_eq!(report.host_workspace_bytes, Some(0));
    let independent = selected()
        .operation_bound(report.operations.last().unwrap())
        .unwrap()
        .unwrap();
    assert!(matches!(
        independent.outputs[0],
        WorkspaceOutputStorage::Allocate(_)
    ));
}

#[test]
fn malformed_storage_descriptors_are_rejected_before_a_quote() {
    let source = WorkspaceLayout::new(&[2, 256, 8], WorkspaceDtype::Float32).unwrap();
    let update = WorkspaceLayout::new(&[2, 3, 8], WorkspaceDtype::Float32).unwrap();
    for starts in [
        vec![],
        vec![0, -1, 0],
        vec![0, 254, 0],
        vec![0, i32::MAX, 0],
    ] {
        let operation = WorkspaceOperation {
            kind: WorkspaceOperationKind::SliceUpdate { starts },
            inputs: vec![source.clone(), update.clone()],
            outputs: vec![source.clone()],
        };
        assert!(selected().operation_bound(&operation).is_err());
    }
    for kind in [
        WorkspaceOperationKind::Contiguous,
        WorkspaceOperationKind::DeepCopy,
        WorkspaceOperationKind::SliceUpdate {
            starts: vec![0, 0, 0],
        },
    ] {
        let operation = WorkspaceOperation {
            kind,
            inputs: vec![source.clone(), update.clone()],
            outputs: vec![update.clone()],
        };
        assert!(selected().operation_bound(&operation).is_err());
    }
    let operation = WorkspaceOperation {
        kind: WorkspaceOperationKind::SliceUpdate {
            starts: vec![0, 0, 0],
        },
        inputs: vec![
            source.clone(),
            WorkspaceLayout::new(&[2, 3, 8], WorkspaceDtype::Int32).unwrap(),
        ],
        outputs: vec![source],
    };
    assert!(selected().operation_bound(&operation).is_err());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use safemlx::{
        ops::indexing::{TryIndexMutOp, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };

    #[derive(Clone, Copy, Debug)]
    enum Layout {
        Row,
        Padded,
        Transposed,
        Broadcast,
    }
    const LAYOUTS: [Layout; 4] = [
        Layout::Row,
        Layout::Padded,
        Layout::Transposed,
        Layout::Broadcast,
    ];
    const DTYPES: [Dtype; 7] = [
        Dtype::Float32,
        Dtype::Float16,
        Dtype::Bfloat16,
        Dtype::Int32,
        Dtype::Uint32,
        Dtype::Uint8,
        Dtype::Bool,
    ];

    fn meta_dtype(dtype: Dtype) -> WorkspaceDtype {
        match dtype {
            Dtype::Int32 => WorkspaceDtype::Int32,
            Dtype::Uint32 => WorkspaceDtype::Uint32,
            Dtype::Uint8 => WorkspaceDtype::Uint8,
            Dtype::Bool => WorkspaceDtype::Bool,
            _ => WorkspaceDtype::Float32,
        }
    }
    fn array(
        shape: [i32; 3],
        dtype: Dtype,
        layout: Layout,
        seed: i32,
        stream: &Stream,
    ) -> (Array, Vec<f32>) {
        let [batch, positions, width] = shape;
        let value = |b: i32, p: i32, c: i32| ((seed + b * 17 + p * 3 + c) % 61) as f32;
        let value = |b, p, c| {
            let v = if matches!(layout, Layout::Broadcast) {
                value(0, 0, c)
            } else {
                value(b, p, c)
            };
            if dtype == Dtype::Bool {
                f32::from(v != 0.0)
            } else {
                v
            }
        };
        let expected = (0..batch)
            .flat_map(|b| (0..positions).flat_map(move |p| (0..width).map(move |c| value(b, p, c))))
            .collect::<Vec<_>>();
        let (storage_shape, values) = match layout {
            Layout::Row => (shape, expected.clone()),
            Layout::Padded => (
                [batch, positions + 5, width],
                (0..batch)
                    .flat_map(|b| {
                        (0..positions + 5)
                            .flat_map(move |p| (0..width).map(move |c| value(b, p, c)))
                    })
                    .collect(),
            ),
            Layout::Transposed => (
                [width, positions, batch],
                (0..width)
                    .flat_map(|c| {
                        (0..positions).flat_map(move |p| (0..batch).map(move |b| value(b, p, c)))
                    })
                    .collect(),
            ),
            Layout::Broadcast => ([1, 1, width], (0..width).map(|c| value(0, 0, c)).collect()),
        };
        let source = Array::from_slice(&values, &storage_shape)
            .as_dtype(dtype, stream)
            .unwrap();
        let source = match layout {
            Layout::Row => source,
            Layout::Padded => source
                .try_index_device((.., ..positions, ..), stream)
                .unwrap(),
            Layout::Transposed => source.transpose_axes(&[2, 1, 0], stream).unwrap(),
            Layout::Broadcast => safemlx::ops::broadcast_to(&source, &shape, stream).unwrap(),
        };
        safemlx::transforms::eval([&source]).unwrap();
        (source, expected)
    }
    fn values(array: &Array, stream: &Stream) -> Vec<f32> {
        let packed = array
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .contiguous(false, stream)
            .unwrap();
        packed.evaluated().unwrap().try_to_vec::<f32>().unwrap()
    }
    fn allowed(
        kind: WorkspaceOperationKind,
        inputs: &[&Array],
        selected: MlxMetalWorkspaceMechanisms,
    ) -> u64 {
        let source =
            WorkspaceLayout::new(inputs[0].shape(), meta_dtype(inputs[0].dtype())).unwrap();
        let operation = WorkspaceOperation {
            kind,
            inputs: inputs
                .iter()
                .map(|a| WorkspaceLayout::new(a.shape(), meta_dtype(a.dtype())).unwrap())
                .collect(),
            outputs: vec![source],
        };
        let bound = selected.operation_bound(&operation).unwrap().unwrap();
        let result = match bound.outputs[0] {
            WorkspaceOutputStorage::Allocate(bytes)
            | WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, .. } => bytes,
            _ => panic!("storage bound must cover possible independent output"),
        };
        result + bound.scratch_bytes
    }
    fn measure(stream: &Stream, run: impl FnOnce() -> Array) -> (Array, u64) {
        stream.synchronize().unwrap();
        let before = safemlx::memory::active_memory().unwrap();
        safemlx::memory::reset_peak_memory().unwrap();
        let result = run();
        safemlx::transforms::eval([&result]).unwrap();
        stream.synchronize().unwrap();
        let observed = safemlx::memory::peak_memory()
            .unwrap()
            .saturating_sub(before) as u64;
        (result, observed)
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_storage_copy_peaks_fit_bounds_and_preserve_logical_values() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut cases = 0;
        for dtype in DTYPES {
            for shape in [[1, 1, 1], [2, 7, 3], [2, 257, 32], [2, 513, 8], [2, 0, 3]] {
                for layout in LAYOUTS {
                    let (source, expected) = array(shape, dtype, layout, 3, &stream);
                    for deep in [false, true] {
                        let kind = if deep {
                            WorkspaceOperationKind::DeepCopy
                        } else {
                            WorkspaceOperationKind::Contiguous
                        };
                        let bound = allowed(kind, &[&source], selected);
                        let (output, observed) = measure(&stream, || {
                            if deep {
                                source.clone().deep_clone().unwrap()
                            } else {
                                source.contiguous(false, &stream).unwrap()
                            }
                        });
                        assert!(
                            observed <= bound,
                            "copy dtype={dtype:?} shape={shape:?} layout={layout:?} deep={deep}: {observed} exceeds {bound}"
                        );
                        assert_eq!(output.shape(), shape);
                        assert_eq!(values(&output, &stream), expected);
                        eprintln!(
                            "storage_copy dtype={dtype:?} shape={shape:?} layout={layout:?} deep={deep} observed={observed} bound={bound}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 280);
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_storage_update_peaks_fit_bounds_and_preserve_rectangles() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut cases = 0;
        for source_dtype in DTYPES {
            let updates = if matches!(
                source_dtype,
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            ) {
                vec![Dtype::Float32, Dtype::Float16, Dtype::Bfloat16]
            } else {
                vec![source_dtype]
            };
            for update_dtype in updates {
                for shape in [[1, 7, 1], [2, 7, 3], [2, 257, 32], [2, 513, 8], [2, 0, 3]] {
                    let [batch, capacity, width] = shape;
                    let intervals = if capacity == 0 {
                        vec![(0, 0)]
                    } else {
                        vec![(0, capacity), (0, 0), (capacity - 1, 1), (1, 3)]
                    };
                    for (start, count) in intervals {
                        for (source_layout, update_layout) in [
                            (Layout::Row, Layout::Row),
                            (Layout::Padded, Layout::Transposed),
                            (Layout::Transposed, Layout::Padded),
                            (Layout::Broadcast, Layout::Broadcast),
                        ] {
                            let (source, mut expected) =
                                array(shape, source_dtype, source_layout, 1, &stream);
                            let (update, update_values) = array(
                                [batch, count, width],
                                update_dtype,
                                update_layout,
                                19,
                                &stream,
                            );
                            for b in 0..batch {
                                for p in 0..count {
                                    for c in 0..width {
                                        expected
                                            [((b * capacity + start + p) * width + c) as usize] =
                                            update_values[((b * count + p) * width + c) as usize];
                                    }
                                }
                            }
                            let bound = allowed(
                                WorkspaceOperationKind::SliceUpdate {
                                    starts: vec![0, start, 0],
                                },
                                &[&source, &update],
                                selected,
                            );
                            let (output, observed) = measure(&stream, || {
                                let mut output = source.clone();
                                output
                                    .try_index_mut_device(
                                        (.., start..start + count, ..),
                                        &update,
                                        &stream,
                                    )
                                    .unwrap();
                                output
                            });
                            assert!(
                                observed <= bound,
                                "update {source_dtype:?}/{update_dtype:?} {shape:?} {source_layout:?}/{update_layout:?} {start}+{count}: {observed} exceeds {bound}"
                            );
                            assert_eq!(output.shape(), shape);
                            assert_eq!(values(&output, &stream), expected);
                            eprintln!(
                                "storage_update dtypes={source_dtype:?}/{update_dtype:?} shape={shape:?} layouts={source_layout:?}/{update_layout:?} start={start} count={count} observed={observed} bound={bound}"
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 884);
    }
}
