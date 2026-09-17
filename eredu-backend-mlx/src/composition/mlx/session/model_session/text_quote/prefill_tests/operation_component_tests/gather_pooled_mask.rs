//! Query-specific eligibility uses the actual shared native gather and source.
use super::*;
use eredu_nn::workspace::WorkspaceOperationKind;

#[test]
fn original_pooled_mask_gather_preserves_strided_queries_and_signed_positions() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let one = MlxTensor::from_array(Array::from_slice(&[1.0f32], &[]));
    let zero = MlxTensor::from_array(Array::from_slice(&[0.0f32], &[]));
    let (batch, queries, pooled, selected) = (2usize, 3usize, 5usize, 3usize);
    let logical: Vec<f32> = (0..queries * pooled)
        .map(|i| (i as f32 - 6.0) / 4.0)
        .collect();
    let eligibility: Vec<bool> = (0..queries * pooled)
        .map(|i| (i + i / pooled) % 3 != 0)
        .collect();
    let positions: Vec<usize> = (0..batch * queries * selected)
        .map(|i| (i * 3 + i / selected) % pooled)
        .collect();
    for (dtype, signed, strided_mask, strided_positions) in [
        (Dtype::Bool, false, false, false),
        (Dtype::Bool, true, true, true),
        (Dtype::Float32, true, true, false),
        (Dtype::Float16, true, false, true),
        (Dtype::Bfloat16, true, true, true),
    ] {
        // Transposed backing keeps logical values identical while exercising
        // actual noncontiguous source strides, including the batch broadcast.
        let mask_order: Vec<usize> = if strided_mask {
            (0..pooled)
                .flat_map(|p| (0..queries).map(move |q| q * pooled + p))
                .collect()
        } else {
            (0..queries * pooled).collect()
        };
        let shape = if strided_mask {
            [pooled as i32, queries as i32]
        } else {
            [queries as i32, pooled as i32]
        };
        let mask = if dtype == Dtype::Bool {
            Array::from_slice(
                &mask_order
                    .iter()
                    .map(|&i| eligibility[i])
                    .collect::<Vec<_>>(),
                &shape,
            )
        } else {
            Array::from_slice(
                &mask_order.iter().map(|&i| logical[i]).collect::<Vec<_>>(),
                &shape,
            )
            .as_dtype(dtype, &stream)
            .unwrap()
        };
        let mask = if strided_mask {
            mask.transpose_axes(&[1, 0], &stream).unwrap()
        } else {
            mask
        };
        let index_order: Vec<usize> = if strided_positions {
            (0..batch)
                .flat_map(|b| {
                    (0..selected).flat_map(move |s| {
                        (0..queries).map(move |q| (b * queries + q) * selected + s)
                    })
                })
                .collect()
        } else {
            (0..positions.len()).collect()
        };
        let shape = if strided_positions {
            [batch as i32, selected as i32, queries as i32]
        } else {
            [batch as i32, queries as i32, selected as i32]
        };
        let indices = if signed {
            Array::from_slice(
                &index_order
                    .iter()
                    .map(|&i| {
                        let p = positions[i] as i32;
                        if i % 2 == 0 { p - pooled as i32 } else { p }
                    })
                    .collect::<Vec<_>>(),
                &shape,
            )
        } else {
            Array::from_slice(
                &index_order
                    .iter()
                    .map(|&i| positions[i] as u32)
                    .collect::<Vec<_>>(),
                &shape,
            )
        };
        let indices = if strided_positions {
            indices.transpose_axes(&[0, 2, 1], &stream).unwrap()
        } else {
            indices
        };
        let mask = MlxTensor::from_array(mask);
        let indices = MlxTensor::from_array(indices);
        let expected: Vec<f32> = positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let q = i / selected % queries;
                if dtype == Dtype::Bool {
                    if eligibility[q * pooled + p] {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    logical[q * pooled + p]
                }
            })
            .collect();
        let invoke = || {
            let output = MlxNeuralBackend::gather_pooled_mask(&mask, &indices, &stream).unwrap();
            if dtype == Dtype::Bool {
                MlxTensor::where_condition(&output, &one, &zero, &stream)
                    .unwrap()
                    .into_array()
            } else {
                output
                    .into_array()
                    .as_dtype(Dtype::Float32, &stream)
                    .unwrap()
            }
        };
        assert_eq!(
            invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            expected
        );
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let mut layout = context
            .layout(
                &[queries as i32, pooled as i32],
                if dtype == Dtype::Bool {
                    WorkspaceDtype::Bool
                } else {
                    WorkspaceDtype::Float32
                },
            )
            .unwrap();
        let representation = match dtype {
            Dtype::Float16 => Some(WorkspaceFloatingType::Float16),
            Dtype::Bfloat16 => Some(WorkspaceFloatingType::Bfloat16),
            Dtype::Float32 => Some(WorkspaceFloatingType::Float32),
            _ => None,
        };
        layout = layout.with_representation(
            representation.map(|dtype| WorkspaceRepresentation::new(dtype, !strided_mask)),
        );
        let source = WorkspaceTensor::existing(layout, &context).unwrap();
        let positions_metadata = WorkspaceTensor::existing(
            context
                .layout(
                    &[batch as i32, queries as i32, selected as i32],
                    if signed {
                        WorkspaceDtype::Int32
                    } else {
                        WorkspaceDtype::Uint32
                    },
                )
                .unwrap(),
            &context,
        )
        .unwrap();
        let yes = WorkspaceTensor::existing(
            context.layout(&[], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let no = WorkspaceTensor::existing(
            context.layout(&[], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        let output =
            WorkspaceBackend::gather_pooled_mask(&source, &positions_metadata, &context).unwrap();
        let output = if dtype == Dtype::Bool {
            WorkspaceTensor::where_condition(&output, &yes, &no, &context).unwrap()
        } else {
            context
                .execute(
                    WorkspaceOperationKind::Elementwise("cast_f32"),
                    &[&output],
                    vec![
                        context
                            .layout(
                                &[batch as i32, 1, queries as i32, selected as i32],
                                WorkspaceDtype::Float32,
                            )
                            .unwrap(),
                    ],
                )
                .unwrap()
                .remove(0)
        };
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        exercise_component(
            &mut prepared,
            &plan,
            &[
                mask.as_array(),
                indices.as_array(),
                one.as_array(),
                zero.as_array(),
            ],
            &expected,
            invoke,
        );
    }
    drop((one, zero, stream));
    prepared.finish();
}
