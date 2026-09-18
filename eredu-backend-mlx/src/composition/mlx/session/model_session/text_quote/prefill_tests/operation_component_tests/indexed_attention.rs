//! Fixed indexed attention compared with an einsum worker and host oracle.
use super::*;
use eredu_nn::IndexedAttentionInput;
use safemlx::{
    Stream,
    error::Exception,
    ops::{
        broadcast_to, concatenate_axis, einsum,
        indexing::{NewAxis, TryIndexOp, take_along_axis},
        softmax_axis, r#where,
    },
};

#[allow(clippy::too_many_arguments)]
fn legacy(
    queries: &Array,
    local_keys: &Array,
    local_values: &Array,
    pooled_keys: &Array,
    pooled_values: &Array,
    pooled_indices: &Array,
    scale: f32,
    local_mask: Option<&Array>,
    pooled_mask: Option<&Array>,
    sinks: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    if queries.ndim() != 4
        || local_keys.ndim() != 3
        || local_values.ndim() != 3
        || pooled_keys.ndim() != 3
        || pooled_values.ndim() != 3
        || pooled_indices.ndim() != 3
        || queries.dim(0) != local_keys.dim(0)
        || queries.dim(0) != local_values.dim(0)
        || queries.dim(0) != pooled_keys.dim(0)
        || queries.dim(0) != pooled_values.dim(0)
        || queries.dim(0) != pooled_indices.dim(0)
        || queries.dim(2) != pooled_indices.dim(1)
        || queries.dim(3) != local_keys.dim(2)
        || queries.dim(3) != pooled_keys.dim(2)
        || local_keys.dim(1) != local_values.dim(1)
        || pooled_keys.dim(1) != pooled_values.dim(1)
        || local_values.dim(2) != pooled_values.dim(2)
    {
        return Err(Exception::custom(format!(
            "indexed sparse attention received incompatible shapes q={:?}, local_keys={:?}, local_values={:?}, pooled_keys={:?}, pooled_values={:?}, indices={:?}",
            queries.shape(),
            local_keys.shape(),
            local_values.shape(),
            pooled_keys.shape(),
            pooled_values.shape(),
            pooled_indices.shape()
        )));
    }
    let batch = queries.dim(0);
    let heads = queries.dim(1);
    let query_tokens = queries.dim(2);
    let key_dim = queries.dim(3);
    let value_dim = local_values.dim(2);
    let selected = pooled_indices.dim(2);
    let pooled_tokens = pooled_keys.dim(1);
    if selected <= 0 || pooled_tokens <= 0 {
        return Err(Exception::custom(
            "indexed sparse attention requires at least one pooled token and selected index",
        ));
    }

    let expanded_pooled_keys = broadcast_to(
        &pooled_keys.try_index_device((.., NewAxis, .., ..), stream)?,
        &[batch, query_tokens, pooled_tokens, key_dim],
        stream,
    )?;
    let expanded_key_indices = broadcast_to(
        &pooled_indices.try_index_device((.., .., .., NewAxis), stream)?,
        &[batch, query_tokens, selected, key_dim],
        stream,
    )?;
    let selected_pooled_keys =
        take_along_axis(expanded_pooled_keys, &expanded_key_indices, 2, stream)?;
    let expanded_pooled_values = broadcast_to(
        &pooled_values.try_index_device((.., NewAxis, .., ..), stream)?,
        &[batch, query_tokens, pooled_tokens, value_dim],
        stream,
    )?;
    let expanded_value_indices = broadcast_to(
        &pooled_indices.try_index_device((.., .., .., NewAxis), stream)?,
        &[batch, query_tokens, selected, value_dim],
        stream,
    )?;
    let selected_pooled_values =
        take_along_axis(expanded_pooled_values, &expanded_value_indices, 2, stream)?;

    let scaled_queries = queries.multiply(Array::try_from_f32(scale)?, stream)?;
    let mut local_scores = einsum("bhld,btd->bhlt", [&scaled_queries, local_keys], stream)?;
    let mut pooled_scores = einsum(
        "bhld,blkd->bhlk",
        [&scaled_queries, &selected_pooled_keys],
        stream,
    )?;
    apply_score_mask(&mut local_scores, local_mask, stream)?;
    apply_score_mask(&mut pooled_scores, pooled_mask, stream)?;

    let mut score_parts = vec![local_scores, pooled_scores];
    if let Some(sinks) = sinks {
        if sinks.shape() != [heads] {
            return Err(Exception::custom(format!(
                "attention sinks require shape [{heads}], got {:?}",
                sinks.shape()
            )));
        }
        score_parts.push(broadcast_to(
            &sinks
                .as_dtype(score_parts[0].dtype(), stream)?
                .reshape(&[1, heads, 1, 1], stream)?,
            &[batch, heads, query_tokens, 1],
            stream,
        )?);
    }
    let scores = concatenate_axis(&score_parts, -1, stream)?;
    let weights = softmax_axis(scores, -1, true, stream)?;
    let local_tokens = local_keys.dim(1);
    let local_weights = weights.try_index_device((.., .., .., ..local_tokens), stream)?;
    let pooled_weights =
        weights.try_index_device((.., .., .., local_tokens..local_tokens + selected), stream)?;
    let local_context = einsum("bhlt,btv->bhlv", [&local_weights, local_values], stream)?;
    let pooled_context = einsum(
        "bhlk,blkv->bhlv",
        [&pooled_weights, &selected_pooled_values],
        stream,
    )?;
    local_context.add(pooled_context, stream)
}

fn apply_score_mask(
    scores: &mut Array,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<(), Exception> {
    let Some(mask) = mask else {
        return Ok(());
    };
    *scores = if mask.dtype() == Dtype::Bool {
        r#where(
            mask,
            &*scores,
            Array::try_from_f32(scores.dtype().finfo_min()? as f32)?,
            stream,
        )?
    } else {
        scores.add(mask.as_dtype(scores.dtype(), stream)?, stream)?
    };
    Ok(())
}

fn bank(
    values: &[f32],
    shape: [i32; 3],
    dtype: Dtype,
    strided: bool,
    stream: &Stream,
) -> MlxTensor {
    let [b, t, d] = shape;
    let array = if strided {
        let order: Vec<f32> = (0..b as usize)
            .flat_map(|b| {
                (0..d as usize).flat_map(move |d| {
                    (0..t as usize)
                        .map(move |t| (b * shape[1] as usize + t) * shape[2] as usize + d)
                })
            })
            .map(|i| values[i])
            .collect();
        Array::from_slice(&order, &[b, d, t])
            .as_dtype(dtype, stream)
            .unwrap()
            .transpose_axes(&[0, 2, 1], stream)
            .unwrap()
    } else {
        Array::from_slice(values, &shape)
            .as_dtype(dtype, stream)
            .unwrap()
    };
    MlxTensor::from_array(array)
}
#[test]
fn original_indexed_attention_preserves_four_contractions_masks_and_sink() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    for (batch, queries, local, pooled, selected, strided, dtype, mode) in [
        (
            2usize,
            3usize,
            2usize,
            5usize,
            3usize,
            false,
            Dtype::Float32,
            0usize,
        ),
        (1, 3, 3, 7, 3, true, Dtype::Bfloat16, 1),
        (1, 1, 0, 5, 2, false, Dtype::Float32, 2),
    ] {
        let heads = 2usize;
        let dimensions = 3usize;
        let width = 2usize;
        let qshape = [
            batch as i32,
            heads as i32,
            queries as i32,
            dimensions as i32,
        ];
        let lkshape = [batch as i32, local as i32, dimensions as i32];
        let lvshape = [batch as i32, local as i32, width as i32];
        let pkshape = [batch as i32, pooled as i32, dimensions as i32];
        let pvshape = [batch as i32, pooled as i32, width as i32];
        let qvalues: Vec<f32> = (0..batch * heads * queries * dimensions)
            .map(|i| (i as i32 % 9 - 4) as f32 / 8.0)
            .collect();
        let lkvalues: Vec<f32> = (0..batch * local * dimensions)
            .map(|i| ((i * 3) as i32 % 11 - 5) as f32 / 8.0)
            .collect();
        let lvvalues: Vec<f32> = (0..batch * local * width)
            .map(|i| ((i * 7) as i32 % 13 - 6) as f32 / 8.0)
            .collect();
        let pkvalues: Vec<f32> = (0..batch * pooled * dimensions)
            .map(|i| ((i * 5) as i32 % 13 - 6) as f32 / 8.0)
            .collect();
        let pvvalues: Vec<f32> = (0..batch * pooled * width)
            .map(|i| ((i * 3) as i32 % 11 - 5) as f32 / 8.0)
            .collect();
        let q = if strided {
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
            Array::from_slice(&qvalues, &qshape)
                .as_dtype(dtype, &stream)
                .unwrap()
        };
        let q = MlxTensor::from_array(q);
        let lk = bank(&lkvalues, lkshape, dtype, strided, &stream);
        let lv = bank(&lvvalues, lvshape, dtype, strided, &stream);
        let pk = bank(&pkvalues, pkshape, dtype, strided, &stream);
        let pv = bank(&pvvalues, pvshape, dtype, strided, &stream);
        let positions: Vec<usize> = (0..batch * queries * selected)
            .map(|i| ((i % selected) * 2 + i / selected) % pooled)
            .collect();
        let ishape = [batch as i32, queries as i32, selected as i32];
        let indices = if mode == 1 {
            Array::from_slice(
                &positions
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| {
                        if i % 2 == 0 {
                            p as i32 - pooled as i32
                        } else {
                            p as i32
                        }
                    })
                    .collect::<Vec<_>>(),
                &ishape,
            )
        } else {
            Array::from_slice(
                &positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
                &ishape,
            )
        };
        let indices = MlxTensor::from_array(indices);
        let lm: Vec<bool> = (0..queries * local)
            .map(|i| (i + i / local) % 3 != 0)
            .collect();
        let pm: Vec<bool> = (0..queries * selected)
            .map(|i| (i + i / selected) % 3 != 0)
            .collect();
        let additive: Vec<f32> = (0..queries * selected)
            .map(|i| -(i as i32 % 3) as f32 * 0.25)
            .collect();
        let lm = MlxTensor::from_array(Array::from_slice(&lm, &[queries as i32, local as i32]));
        let pm = MlxTensor::from_array(if mode == 1 {
            Array::from_slice(&additive, &[queries as i32, selected as i32])
        } else {
            Array::from_slice(&pm, &[queries as i32, selected as i32])
        });
        let sink_values = [0.0f32, 0.25];
        let sinks = MlxTensor::from_array(Array::from_slice(&sink_values, &[heads as i32]));
        let input = || IndexedAttentionInput {
            queries: &q,
            local_keys: &lk,
            local_values: &lv,
            pooled_keys: &pk,
            pooled_values: &pv,
            selected_positions: &indices,
            scale: 0.5,
            local_mask: (mode == 1).then_some(&lm),
            pooled_mask: (mode != 0).then_some(&pm),
            sinks: (mode != 0).then_some(&sinks),
        };
        let expected = legacy(
            q.as_array(),
            lk.as_array(),
            lv.as_array(),
            pk.as_array(),
            pv.as_array(),
            indices.as_array(),
            0.5,
            (mode == 1).then_some(lm.as_array()),
            (mode != 0).then_some(pm.as_array()),
            (mode != 0).then_some(sinks.as_array()),
            &stream,
        )
        .unwrap()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
        let mut host = Vec::new();
        for b in 0..batch {
            for h in 0..heads {
                for query in 0..queries {
                    let mut logits = Vec::new();
                    let mut values = Vec::new();
                    for l in 0..local {
                        let mut score: f32 = (0..dimensions)
                            .map(|d| {
                                qvalues[((b * heads + h) * queries + query) * dimensions + d]
                                    * 0.5
                                    * lkvalues[(b * local + l) * dimensions + d]
                            })
                            .sum();
                        if mode == 1 && (query * local + l + query) % 3 == 0 {
                            score = f32::MIN;
                        }
                        logits.push(score);
                        values
                            .push(&lvvalues[(b * local + l) * width..(b * local + l + 1) * width]);
                    }
                    for k in 0..selected {
                        let p = positions[(b * queries + query) * selected + k];
                        let mut score: f32 = (0..dimensions)
                            .map(|d| {
                                qvalues[((b * heads + h) * queries + query) * dimensions + d]
                                    * 0.5
                                    * pkvalues[(b * pooled + p) * dimensions + d]
                            })
                            .sum();
                        if mode == 1 {
                            score += additive[query * selected + k];
                        }
                        if mode == 2 && (query * selected + k + query) % 3 == 0 {
                            score = f32::MIN;
                        }
                        logits.push(score);
                        values.push(
                            &pvvalues[(b * pooled + p) * width..(b * pooled + p + 1) * width],
                        );
                    }
                    let mut maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if mode != 0 {
                        maximum = maximum.max(sink_values[h]);
                    }
                    let weights: Vec<f32> = logits.iter().map(|l| (*l - maximum).exp()).collect();
                    let denominator = weights.iter().sum::<f32>()
                        + if mode != 0 {
                            (sink_values[h] - maximum).exp()
                        } else {
                            0.0
                        };
                    for v in 0..width {
                        host.push(
                            weights
                                .iter()
                                .zip(&values)
                                .map(|(w, value)| w * value[v])
                                .sum::<f32>()
                                / denominator,
                        );
                    }
                }
            }
        }
        for (actual, host) in expected.iter().zip(host) {
            assert!(
                (*actual - host).abs() < 2e-5,
                "legacy {actual}, independent indexed attention {host}"
            );
        }
        let invoke = || {
            MlxNeuralBackend::indexed_attention(input(), &stream)
                .unwrap()
                .into_array()
        };
        let ordinary = invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap();
        for (actual, expected) in ordinary.iter().zip(&expected) {
            assert!((*actual - *expected).abs() < 2e-5);
        }
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let source = |shape: &[i32], kind, contiguous| {
            WorkspaceTensor::existing(
                context.layout(shape, kind).unwrap().with_representation(
                    (kind == WorkspaceDtype::Float32).then_some(WorkspaceRepresentation::new(
                        if dtype == Dtype::Bfloat16 {
                            WorkspaceFloatingType::Bfloat16
                        } else {
                            WorkspaceFloatingType::Float32
                        },
                        contiguous,
                    )),
                ),
                &context,
            )
            .unwrap()
        };
        let qm = source(&qshape, WorkspaceDtype::Float32, !strided);
        let lkm = source(&lkshape, WorkspaceDtype::Float32, !strided);
        let lvm = source(&lvshape, WorkspaceDtype::Float32, !strided);
        let pkm = source(&pkshape, WorkspaceDtype::Float32, !strided);
        let pvm = source(&pvshape, WorkspaceDtype::Float32, !strided);
        let im = source(
            &ishape,
            if mode == 1 {
                WorkspaceDtype::Int32
            } else {
                WorkspaceDtype::Uint32
            },
            true,
        );
        let lmm = source(&[queries as i32, local as i32], WorkspaceDtype::Bool, true);
        let pmm = WorkspaceTensor::existing(
            context
                .layout(
                    &[queries as i32, selected as i32],
                    if mode == 1 {
                        WorkspaceDtype::Float32
                    } else {
                        WorkspaceDtype::Bool
                    },
                )
                .unwrap(),
            &context,
        )
        .unwrap();
        let sm = WorkspaceTensor::existing(
            context
                .layout(&[heads as i32], WorkspaceDtype::Float32)
                .unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        let output = WorkspaceBackend::indexed_attention(
            IndexedAttentionInput {
                queries: &qm,
                local_keys: &lkm,
                local_values: &lvm,
                pooled_keys: &pkm,
                pooled_values: &pvm,
                selected_positions: &im,
                scale: 0.5,
                local_mask: (mode == 1).then_some(&lmm),
                pooled_mask: (mode != 0).then_some(&pmm),
                sinks: (mode != 0).then_some(&sm),
            },
            &context,
        )
        .unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        let mut leaves = vec![
            q.as_array(),
            lk.as_array(),
            lv.as_array(),
            pk.as_array(),
            pv.as_array(),
            indices.as_array(),
        ];
        if mode == 1 {
            leaves.push(lm.as_array());
        }
        if mode != 0 {
            leaves.extend([pm.as_array(), sinks.as_array()]);
        }
        exercise_component(&mut prepared, &plan, &leaves, &expected, invoke);
    }
    drop(stream);
    prepared.finish();
}
