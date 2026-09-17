//! Actual pooled ranking retains the old einsum path as an independent oracle.
use super::*;
use eredu_nn::PooledPositionInput;
use safemlx::{Stream, ops::indexing::TryIndexOp};

// The original pre-factoring worker, including native einsum path construction
// and tuple indexing. This deliberately does not call the new fixed contraction.
fn legacy(input: PooledPositionInput<'_, MlxTensor>, stream: &Stream) -> Array {
    let queries = input
        .queries
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .unwrap();
    let pooled = input
        .pooled_keys
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .unwrap();
    let scores = ops::einsum("bhld,bpd->bhlp", [&queries, &pooled], stream).unwrap();
    let scores = ops::maximum(scores, Array::from(0.0f32), stream).unwrap();
    let scores = scores.multiply(Array::from(input.scale), stream).unwrap();
    let weights = input
        .head_weights
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .unwrap();
    let weights = weights
        .multiply(Array::from(input.head_scale), stream)
        .unwrap();
    let weights = weights
        .transpose_axes(&[0, 2, 1], stream)
        .unwrap()
        .expand_dims(-1, stream)
        .unwrap();
    let mut scores = scores
        .multiply(weights, stream)
        .unwrap()
        .sum_axis(1, false, stream)
        .unwrap();
    if let Some(mask) = input.mask {
        scores = ops::r#where(
            mask.as_array(),
            scores,
            Array::from(f32::NEG_INFINITY),
            stream,
        )
        .unwrap();
    }
    let count = input.top_k.min(input.pooled_keys.shape()[1]);
    let indices = ops::argpartition_axis(&scores, -count, -1, stream).unwrap();
    let start = indices.dim(-1) - count;
    indices.try_index_device((.., .., start..), stream).unwrap()
}

#[test]
fn original_pooled_positions_preserve_masked_ranking_and_multiblock_tail() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    for (batch, queries, pooled, selected, masked, strided, dtype) in [
        (2usize, 3usize, 7usize, 3usize, false, false, Dtype::Float32),
        (2, 3, 11, 4, true, true, Dtype::Bfloat16),
        (1, 2, 2051, 5, true, false, Dtype::Float32),
    ] {
        let heads = 2usize;
        let dimensions = 2usize;
        let query_shape = [
            batch as i32,
            heads as i32,
            queries as i32,
            dimensions as i32,
        ];
        let pool_shape = [batch as i32, pooled as i32, dimensions as i32];
        let weight_shape = [batch as i32, queries as i32, heads as i32];
        let qvalues: Vec<f32> = (0..batch * heads * queries * dimensions)
            .map(|i| {
                let d = i % dimensions;
                let q = i / dimensions % queries;
                let h = i / dimensions / queries % heads;
                if d == 1 {
                    if h == 0 { -0.25 } else { 0.25 }
                } else if h == 0 {
                    if q % 2 == 0 { 0.5 } else { -0.5 }
                } else {
                    (q + 1) as f32 * 0.25
                }
            })
            .collect();
        let pvalues: Vec<f32> = (0..batch * pooled * dimensions)
            .map(|i| {
                let p = i / dimensions % pooled;
                if i % dimensions == 0 {
                    ((p * 37 + 11) % pooled + 1) as f32 * 0.25
                } else {
                    1.0
                }
            })
            .collect();
        let wvalues: Vec<f32> = (0..batch * queries * heads)
            .map(|i| {
                let b = i / heads / queries;
                let q = i / heads % queries;
                let h = i % heads;
                (b + q + h + 1) as f32 * 0.25
            })
            .collect();
        let eligibility: Vec<bool> = (0..queries * pooled)
            .map(|i| i % pooled % 3 != (i / pooled + 1) % 3)
            .collect();
        let query = if strided {
            let order: Vec<f32> = (0..batch)
                .flat_map(|b| {
                    (0..heads).flat_map(move |h| {
                        (0..dimensions).flat_map(move |d| {
                            (0..queries)
                                .map(move |q| ((b * heads + h) * queries + q) * dimensions + d)
                        })
                    })
                })
                .map(|i| qvalues[i])
                .collect();
            Array::from_slice(
                &order,
                &[
                    batch as i32,
                    heads as i32,
                    dimensions as i32,
                    queries as i32,
                ],
            )
            .as_dtype(dtype, &stream)
            .unwrap()
            .transpose_axes(&[0, 1, 3, 2], &stream)
            .unwrap()
        } else {
            Array::from_slice(&qvalues, &query_shape)
                .as_dtype(dtype, &stream)
                .unwrap()
        };
        let pool = if strided {
            let order: Vec<f32> = (0..batch)
                .flat_map(|b| {
                    (0..dimensions).flat_map(move |d| {
                        (0..pooled).map(move |p| (b * pooled + p) * dimensions + d)
                    })
                })
                .map(|i| pvalues[i])
                .collect();
            Array::from_slice(&order, &[batch as i32, dimensions as i32, pooled as i32])
                .as_dtype(dtype, &stream)
                .unwrap()
                .transpose_axes(&[0, 2, 1], &stream)
                .unwrap()
        } else {
            Array::from_slice(&pvalues, &pool_shape)
                .as_dtype(dtype, &stream)
                .unwrap()
        };
        let query = MlxTensor::from_array(query);
        let pool = MlxTensor::from_array(pool);
        let weights = MlxTensor::from_array(
            Array::from_slice(&wvalues, &weight_shape)
                .as_dtype(dtype, &stream)
                .unwrap(),
        );
        let mask = MlxTensor::from_array(Array::from_slice(
            &eligibility,
            &[queries as i32, pooled as i32],
        ));
        let labels: Vec<f32> = (0..queries * pooled).map(|i| (i % pooled) as f32).collect();
        let labels =
            MlxTensor::from_array(Array::from_slice(&labels, &[queries as i32, pooled as i32]));
        let input = || PooledPositionInput {
            queries: &query,
            pooled_keys: &pool,
            head_weights: &weights,
            mask: masked.then_some(&mask),
            top_k: selected as i32,
            scale: 0.5,
            head_scale: 0.25,
        };
        let reference = legacy(input(), &stream);
        let reference_ids = reference.evaluated().unwrap().try_to_vec::<u32>().unwrap();
        for b in 0..batch {
            for q in 0..queries {
                let mut ranked: Vec<(f32, u32)> = (0..pooled)
                    .filter(|&p| !masked || eligibility[q * pooled + p])
                    .map(|p| {
                        let score: f32 = (0..heads)
                            .map(|h| {
                                let dot: f32 = (0..dimensions)
                                    .map(|d| {
                                        qvalues[((b * heads + h) * queries + q) * dimensions + d]
                                            * pvalues[(b * pooled + p) * dimensions + d]
                                    })
                                    .sum();
                                dot.max(0.0) * 0.5 * (wvalues[(b * queries + q) * heads + h] * 0.25)
                            })
                            .sum();
                        (score, p as u32)
                    })
                    .collect();
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
                assert!(
                    ranked[selected - 1].0 > ranked[selected].0,
                    "unambiguous cutoff"
                );
                let mut host: Vec<u32> = ranked[..selected].iter().map(|row| row.1).collect();
                host.sort_unstable();
                let start = (b * queries + q) * selected;
                let mut actual = reference_ids[start..start + selected].to_vec();
                actual.sort_unstable();
                assert_eq!(actual, host, "independent head-weighted score oracle");
            }
        }
        let expected: Vec<f32> = reference_ids.into_iter().map(|i| i as f32).collect();
        drop(reference);
        let invoke = || {
            let positions = MlxNeuralBackend::select_pooled_positions(input(), &stream).unwrap();
            assert_eq!(positions.as_array().dtype(), Dtype::Uint32);
            // Use the real next pooled-chain worker to expose U32 positions as
            // labels. No synthetic floating cast is added to the receipt.
            MlxNeuralBackend::gather_pooled_mask(&labels, &positions, &stream)
                .unwrap()
                .into_array()
        };
        assert_eq!(
            invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            expected
        );
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let floating = |shape: &[i32], contiguous| {
            WorkspaceTensor::existing(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        if dtype == Dtype::Bfloat16 {
                            WorkspaceFloatingType::Bfloat16
                        } else {
                            WorkspaceFloatingType::Float32
                        },
                        contiguous,
                    ))),
                &context,
            )
            .unwrap()
        };
        let qmeta = floating(&query_shape, !strided);
        let pmeta = floating(&pool_shape, !strided);
        let wmeta = floating(&weight_shape, true);
        let maskmeta = WorkspaceTensor::existing(
            context
                .layout(&[queries as i32, pooled as i32], WorkspaceDtype::Bool)
                .unwrap(),
            &context,
        )
        .unwrap();
        let labelsmeta = WorkspaceTensor::existing(
            context
                .layout(&[queries as i32, pooled as i32], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            &context,
        )
        .unwrap();
        context.begin_span();
        let positions = WorkspaceBackend::select_pooled_positions(
            PooledPositionInput {
                queries: &qmeta,
                pooled_keys: &pmeta,
                head_weights: &wmeta,
                mask: masked.then_some(&maskmeta),
                top_k: selected as i32,
                scale: 0.5,
                head_scale: 0.25,
            },
            &context,
        )
        .unwrap();
        let output =
            WorkspaceBackend::gather_pooled_mask(&labelsmeta, &positions, &context).unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        let mut leaves = vec![
            query.as_array(),
            pool.as_array(),
            weights.as_array(),
            labels.as_array(),
        ];
        if masked {
            leaves.push(mask.as_array());
        }
        exercise_component(&mut prepared, &plan, &leaves, &expected, invoke);
    }
    drop(stream);
    prepared.finish();
}
