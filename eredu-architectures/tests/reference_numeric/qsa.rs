use super::*;
use eredu_architectures::qwen4_exp::qsa::*;
use eredu_nn::{RotaryArithmetic, TensorElementType};
use eredu_runtime::{
    AppendOnlyStream, AppendStreamError, AppendStreamLimits, AppendStreamSpec, ResidentAppendStream,
};
fn selection(tile: i32) -> QsaSelectionSpec {
    QsaSelectionSpec {
        heads: 2,
        dimensions: 2,
        ratio: 3,
        token_budget: 6,
        tile_blocks: tile,
        workspace_bytes: 16384,
    }
}
fn stream(
    slot: u32,
    width: i32,
    element: TensorElementType,
    read: usize,
) -> ResidentAppendStream<NumericTensor> {
    ResidentAppendStream::new(
        AppendStreamSpec {
            slot,
            width,
            element,
        },
        AppendStreamLimits {
            entries: 64,
            page_entries: 2,
            read_entries: read,
        },
        65536,
        65536,
        65536,
    )
    .unwrap()
}
fn indexer(context: &NumericContext) -> QsaIndexer<NumericBackend> {
    let p = |id| ParameterSpec::trainable(id).unwrap();
    let n = |id| NormalizationConstructionSpec {
        dimensions: 2,
        epsilon: 1e-5,
        groups: None,
        scale: NormalizationScale::LearnedOffset {
            weight: p(id),
            offset: 1.,
        },
    };
    let mut indexer = QsaIndexer::new(
        QsaIndexerSpec {
            selection: selection(2),
            query_key: LinearSpec {
                input: 2,
                output: 6,
                weight: p("qk"),
                bias: None,
                format: dense_linear_format(),
            },
            query_norm: n("qn"),
            key_norm: n("kn"),
            rotary: RotarySpec {
                dimensions: 2,
                base: 10000.,
                traditional: false,
                algorithm: eredu_nn::RotaryAlgorithm::Default,
                arithmetic: RotaryArithmetic::InputProducts,
            },
        },
        context,
    )
    .unwrap();
    struct Load;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
        fn visit_mut(&mut self, m: ParameterMetadata, v: &'a mut NumericTensor) {
            v.data = match m.id.as_str() {
                "qk" => vec![
                    0.1, 0.2, 0.3, -0.1, -0.2, 0.4, 0.5, 0.2, 0.4, -0.3, -0.1, 0.2,
                ],
                "qn" => vec![0.2, -0.1],
                "kn" => vec![-0.1, 0.3],
                _ => panic!(),
            };
        }
    }
    indexer.visit_parameters_mut(&mut Load);
    indexer
}
fn scalar_summary(raw: &[[f32; 2]], position: i32) -> [f32; 2] {
    let pooled: [f64; 2] =
        std::array::from_fn(|d| raw.iter().map(|k| k[d] as f64).sum::<f64>() / 3.);
    let rms = ((pooled[0] * pooled[0] + pooled[1] * pooled[1]) / 2. + 1e-5).sqrt();
    let a = pooled[0] / rms * 0.9;
    let b = pooled[1] / rms * 1.3;
    let angle = position as f64;
    [
        (a * angle.cos() - b * angle.sin()) as f32,
        (b * angle.cos() + a * angle.sin()) as f32,
    ]
}
fn scalar_selection(
    query: &[f32],
    keys: &[[f32; 2]],
    positions: &[Vec<i32>],
    tail: &[i32],
) -> Vec<i32> {
    let mut scores: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let score = query
                .chunks_exact(2)
                .map(|q| (q[0] as f64 * key[0] as f64 + q[1] as f64 * key[1] as f64).max(0.))
                .sum::<f64>()
                / 2f64.sqrt();
            (index, score)
        })
        .collect();
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut selected: Vec<i32> = scores
        .into_iter()
        .take(2)
        .flat_map(|(index, _)| positions[index].clone())
        .chain(tail.iter().copied())
        .collect();
    selected.sort_unstable();
    selected.resize(8, -1);
    selected
}

#[test]
fn qsa_owned_state_restores_ragged_media_tails_and_selected_streams_across_chunks() {
    use eredu_core::cache::{AppendStreamPolicy, StateTensorDtype};
    use eredu_core::AttentionPolicy;
    use eredu_runtime::RuntimeAppendStreams;
    let context = NumericContext::default();
    let spec = QsaPartialStateSpec::new(
        3,
        2,
        2,
        TensorElementType::F32,
        TensorElementType::F32,
        7,
        10,
    )
    .unwrap();
    let workspace = spec.required_workspace::<NumericTensor>(2).unwrap();
    let policy = LayerCachePolicy::key_value_with_state(
        AttentionPolicy::Full,
        1,
        2,
        spec.policies(),
        vec![
            AppendStreamPolicy::new(0, 2, StateTensorDtype::Float32, 3).unwrap(),
            AppendStreamPolicy::new(1, 3, StateTensorDtype::Int32, 3).unwrap(),
        ],
    )
    .unwrap();
    let raw: Vec<[f32; 2]> = (0..23)
        .map(|i| {
            [
                ((i * 7 % 17) as f32 - 8.) / 5.,
                ((i * 13 % 19) as f32 - 9.) / 7.,
            ]
        })
        .collect();
    let query = NumericTensor::new([2, 2], vec![0.37, -0.7, 0.6, -0.2]);
    let visible = |lane, i| {
        if lane == 0 {
            ![0, 4, 11, 17].contains(&i)
        } else {
            ![1, 2, 4, 5, 10, 14, 22].contains(&i)
        }
    };
    let run = |chunks: &[usize]| {
        let mut state = NumericHybridLayerState::new(&policy);
        for lane in 0..2 {
            state
                .streams
                .push((0, lane, stream(0, 2, TensorElementType::F32, 2)));
            state
                .streams
                .push((1, lane, stream(1, 3, TensorElementType::I32, 1)));
        }
        let mut indexer = indexer(&context);
        let mut start = 0;
        let mut result = vec![];
        for &length in chunks {
            let mut partial = spec
                .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
                .unwrap();
            for i in start..start + length {
                for lane in 0..2 {
                    let angle = (i + lane * 3) as f32;
                    let cosine = NumericTensor::new([1, 1, 1, 2], vec![angle.cos(); 2]);
                    let sine = NumericTensor::new([1, 1, 1, 2], vec![angle.sin(); 2]);
                    let rotary = if lane == 0 {
                        RotaryPosition::Offset(i as i32)
                    } else {
                        RotaryPosition::Embeddings {
                            cosine: &cosine,
                            sine: &sine,
                        }
                    };
                    let block = partial[lane]
                        .push(
                            &NumericTensor::new([1, 2], raw[i].to_vec()),
                            i as i32,
                            visible(lane, i),
                            rotary,
                            &context,
                        )
                        .unwrap();
                    let (keys, positions) = state.append_stream_pair(0, 1, lane as u32).unwrap();
                    if let Some(block) = block {
                        let summary = indexer
                            .summarize(
                                &block.raw_keys,
                                block.first_position.as_position(),
                                &context,
                            )
                            .unwrap();
                        keys.append(keys.len(), summary, &context).unwrap();
                        positions
                            .append(
                                positions.len(),
                                NumericTensor::from_i32_slice(&block.positions, &[1, 3], &context)
                                    .unwrap(),
                                &context,
                            )
                            .unwrap();
                    }
                    let selected = select_positions(
                        selection(2),
                        &query,
                        keys,
                        positions,
                        partial[lane].tail(),
                        &context,
                    )
                    .unwrap();
                    let seen = (0..=i)
                        .filter(|&p| visible(lane, p))
                        .map(|p| p as i32)
                        .collect::<Vec<_>>();
                    let blocks = seen.chunks_exact(3).map(|v| v.to_vec()).collect::<Vec<_>>();
                    let summaries = blocks
                        .iter()
                        .map(|p| {
                            scalar_summary(
                                &p.iter().map(|&j| raw[j as usize]).collect::<Vec<_>>(),
                                p[0] + lane as i32 * 3,
                            )
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        selected,
                        scalar_selection(
                            &query.data,
                            &summaries,
                            &blocks,
                            &seen[blocks.len() * 3..]
                        )
                    );
                    result.push(selected);
                }
            }
            spec.write::<NumericBackend, _>(&mut state, &partial, workspace, &context)
                .unwrap();
            // Only declared fixed components and named streams survive each boundary.
            drop(partial);
            state = state.clone();
            start += length;
        }
        let checkpoint = state.clone();
        let mut partial = spec
            .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
            .unwrap();
        partial[0]
            .push(
                &NumericTensor::new([1, 2], vec![0.8, -0.3]),
                23,
                true,
                RotaryPosition::Offset(23),
                &context,
            )
            .unwrap();
        spec.write::<NumericBackend, _>(&mut state, &partial, workspace, &context)
            .unwrap();
        assert_eq!(
            spec.read::<NumericBackend, _>(&mut state, 2, workspace, &context)
                .unwrap()[0]
                .last_seen(),
            Some(23)
        );
        state = checkpoint;
        let restored = spec
            .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
            .unwrap();
        assert_eq!(restored[0].last_seen(), Some(22));
        assert_eq!(restored[1].last_seen(), Some(22));
        result
    };
    assert_eq!(run(&[23]), run(&[1, 2, 4, 3, 1, 5, 7]));
}

#[test]
fn qsa_partial_publication_preserves_exact_controls_and_rejects_malformed_companions() {
    let context = NumericContext::default();
    let spec = QsaPartialStateSpec::new(
        3,
        2,
        2,
        TensorElementType::F32,
        TensorElementType::F32,
        7,
        10,
    )
    .unwrap();
    let workspace = spec.required_workspace::<NumericTensor>(2).unwrap();
    let policy = LayerCachePolicy::fixed_only(spec.policies()).unwrap();
    let mut state = NumericHybridLayerState::new(&policy);
    let mut lanes = spec
        .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
        .unwrap();
    let key = NumericTensor::new([1, 2], vec![0.7, -0.3]);
    lanes[0]
        .push(
            &key,
            16_777_217,
            true,
            RotaryPosition::Offset(i32::MAX),
            &context,
        )
        .unwrap();
    lanes[1]
        .push(&key, 16_777_218, false, RotaryPosition::Offset(0), &context)
        .unwrap();
    spec.write::<NumericBackend, _>(&mut state, &lanes, workspace, &context)
        .unwrap();
    let saved = state.clone();
    let restored = spec
        .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
        .unwrap();
    assert_eq!(restored[0].tail(), [16_777_217]);
    assert_eq!(restored[1].last_seen(), Some(16_777_218));
    assert!(matches!(
        spec.write::<NumericBackend, _>(&mut state, &lanes, workspace - 1, &context),
        Err(QsaError::Workspace { .. })
    ));
    for missing in spec.roles() {
        let mut malformed = saved.clone();
        *malformed.fixed_component(missing).unwrap() = None;
        assert!(matches!(
            spec.read::<NumericBackend, _>(&mut malformed, 2, workspace, &context),
            Err(QsaError::Geometry)
        ));
    }
    let controls = saved.fixed[&spec.roles()[0]]
        .as_ref()
        .unwrap()
        .to_i32_vec(&context)
        .unwrap();
    assert_eq!(&controls[..5], &[16_777_217, 1, i32::MAX, 16_777_217, -1]);
    for (index, value) in [(0, -2), (1, 3), (2, -2), (3, -1), (4, 9)] {
        let mut malformed = saved.clone();
        let mut changed = controls.clone();
        changed[index] = value;
        *malformed.fixed_component(spec.roles()[0]).unwrap() =
            Some(NumericTensor::from_i32_slice(&changed, &[2, 5], &context).unwrap());
        assert!(spec
            .read::<NumericBackend, _>(&mut malformed, 2, workspace, &context)
            .is_err());
    }
    let before = state.fixed[&spec.roles()[0]].clone();
    state.fixed.remove(&spec.roles()[3]);
    assert!(spec
        .write::<NumericBackend, _>(&mut state, &lanes, workspace, &context)
        .is_err());
    assert_eq!(
        state.fixed[&spec.roles()[0]]
            .as_ref()
            .unwrap()
            .to_i32_vec(&context)
            .unwrap(),
        before.as_ref().unwrap().to_i32_vec(&context).unwrap()
    );
}
#[test]
fn qsa_empty_and_single_token_blocks_preserve_precision_and_reject_bad_buffers() {
    let context = NumericContext::default();
    for ratio in [1, 3] {
        let spec = QsaPartialStateSpec::new(
            ratio,
            2,
            2,
            TensorElementType::Bf16,
            TensorElementType::F32,
            7,
            10,
        )
        .unwrap();
        let workspace = spec.required_workspace::<NumericTensor>(2).unwrap();
        let mut state =
            NumericHybridLayerState::new(&LayerCachePolicy::fixed_only(spec.policies()).unwrap());
        let mut lanes = spec
            .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
            .unwrap();
        spec.write::<NumericBackend, _>(&mut state, &lanes, workspace, &context)
            .unwrap();
        assert!(spec
            .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
            .unwrap()
            .iter()
            .all(|lane| lane.tail().is_empty() && lane.last_seen().is_none()));
        let key = NumericTensor::new([1, 2], vec![0.75, -0.25])
            .cast_float(TensorElementType::Bf16, &context)
            .unwrap();
        let cosine = NumericTensor::new([1, 1, 1, 2], vec![0.6, 0.8]);
        let sine = NumericTensor::new([1, 1, 1, 2], vec![0.8, -0.6]);
        let complete = lanes[0]
            .push(
                &key,
                16_777_217,
                true,
                RotaryPosition::Embeddings {
                    cosine: &cosine,
                    sine: &sine,
                },
                &context,
            )
            .unwrap();
        assert_eq!(complete.is_some(), ratio == 1);
        lanes[1]
            .push(&key, 16_777_218, false, RotaryPosition::Offset(0), &context)
            .unwrap();
        spec.write::<NumericBackend, _>(&mut state, &lanes, workspace, &context)
            .unwrap();
        let restored = spec
            .read::<NumericBackend, _>(&mut state, 2, workspace, &context)
            .unwrap();
        assert_eq!(restored[0].last_seen(), Some(16_777_217));
        assert_eq!(restored[0].tail().len(), usize::from(ratio != 1));
        assert_eq!(restored[1].last_seen(), Some(16_777_218));
        for role in &spec.roles()[1..] {
            let mut invalid = state.clone();
            let value = invalid.fixed_component(*role).unwrap().as_mut().unwrap();
            *value = value.cast_float(TensorElementType::F16, &context).unwrap();
            assert!(matches!(
                spec.read::<NumericBackend, _>(&mut invalid, 2, workspace, &context),
                Err(QsaError::Geometry)
            ));
        }
        let mut invalid = state.clone();
        let sine = invalid
            .fixed_component(spec.roles()[3])
            .unwrap()
            .as_mut()
            .unwrap();
        *sine = sine.reshape(&[1, 4], &context).unwrap();
        assert!(spec
            .read::<NumericBackend, _>(&mut invalid, 2, workspace, &context)
            .is_err());
    }
}

#[test]
fn qsa_visible_blocks_cross_budget_with_bounded_reads_and_arbitrary_chunks() {
    let context = NumericContext::default();
    let mut indexer = indexer(&context);
    let raw: Vec<[f32; 2]> = (0..23)
        .map(|i| {
            [
                ((i * 7 % 17) as f32 - 8.) / 5.,
                ((i * 13 % 19) as f32 - 9.) / 7.,
            ]
        })
        .collect();
    let visible: Vec<bool> = (0..23).map(|i| ![0, 4, 11, 17].contains(&i)).collect();
    let queries: Vec<Vec<f32>> = (0..23)
        .map(|i| vec![0.1 + i as f32 * 0.09, -0.7, 0.6, -0.02 * i as f32])
        .collect();
    let run = |chunks: &[usize], indexer: &mut QsaIndexer<NumericBackend>| {
        let mut partial = QsaPartialBlock::new(3, 2).unwrap();
        let mut keys = stream(0, 2, TensorElementType::F32, 2);
        let mut positions = stream(1, 3, TensorElementType::I32, 1);
        let mut result = vec![];
        let mut start = 0;
        for &length in chunks {
            for i in start..start + length {
                let key = NumericTensor::new([1, 2], raw[i].to_vec());
                if let Some(block) = partial
                    .push(
                        &key,
                        i as i32,
                        visible[i],
                        RotaryPosition::Offset(i as i32),
                        &context,
                    )
                    .unwrap()
                {
                    let summary = indexer
                        .summarize(
                            &block.raw_keys,
                            block.first_position.as_position(),
                            &context,
                        )
                        .unwrap();
                    keys.append(keys.len(), summary, &context).unwrap();
                    positions
                        .append(
                            positions.len(),
                            NumericTensor::from_i32_slice(&block.positions, &[1, 3], &context)
                                .unwrap(),
                            &context,
                        )
                        .unwrap();
                }
                result.push(
                    select_positions(
                        selection(2),
                        &NumericTensor::new([2, 2], queries[i].clone()),
                        &mut keys,
                        &mut positions,
                        partial.tail(),
                        &context,
                    )
                    .unwrap(),
                );
                assert!(partial.retained_values().iter().all(|t| t.dim(0) < 3));
            }
            start += length;
        }
        (result, partial, keys, positions)
    };
    let (whole, partial, mut keys, mut positions) = run(&[23], &mut indexer);
    let (chunks, _, _, _) = run(&[1, 2, 4, 3, 1, 5, 7], &mut indexer);
    assert_eq!(whole, chunks);
    for i in 0..23 {
        let visible_positions: Vec<i32> =
            (0..=i).filter(|&j| visible[j]).map(|j| j as i32).collect();
        let blocks: Vec<Vec<i32>> = visible_positions
            .chunks_exact(3)
            .map(|v| v.to_vec())
            .collect();
        let summaries: Vec<[f32; 2]> = blocks
            .iter()
            .map(|p| {
                scalar_summary(
                    &p.iter().map(|&j| raw[j as usize]).collect::<Vec<_>>(),
                    p[0],
                )
            })
            .collect();
        let expected = scalar_selection(
            &queries[i],
            &summaries,
            &blocks,
            &visible_positions[blocks.len() * 3..],
        );
        assert_eq!(whole[i], expected, "query {i}");
    }
    let checkpoint = partial.clone();
    let mut fork = partial;
    fork.push(
        &NumericTensor::new([1, 2], vec![0.2, 0.3]),
        23,
        true,
        RotaryPosition::Offset(23),
        &context,
    )
    .unwrap();
    assert_eq!(checkpoint.last_seen(), Some(22));
    assert_eq!(fork.last_seen(), Some(23));
    let mut denied = selection(2);
    denied.workspace_bytes = 0;
    assert!(matches!(
        select_positions(
            denied,
            &NumericTensor::new([2, 2], queries[22].clone()),
            &mut keys,
            &mut positions,
            checkpoint.tail(),
            &context
        ),
        Err(QsaError::Workspace { .. })
    ));
    assert!(keys.read(0..3, &context).is_err());
}
#[test]
fn append_stream_bounds_integer_rows_and_snapshot_are_independent() {
    let context = NumericContext::default();
    let mut state = stream(9, 2, TensorElementType::I32, 3);
    let values = [16777215, -16777215, 1023, -1023, 3, 4, 5, 6];
    state
        .append(
            0,
            NumericTensor::from_i32_slice(&values, &[4, 2], &context).unwrap(),
            &context,
        )
        .unwrap();
    let mut saved = state.clone();
    state
        .append(
            4,
            NumericTensor::from_i32_slice(&[7, 8], &[1, 2], &context).unwrap(),
            &context,
        )
        .unwrap();
    assert_eq!(saved.len(), 4);
    assert_eq!(state.len(), 5);
    assert_eq!(
        saved
            .read(0..3, &context)
            .unwrap()
            .to_i32_vec(&context)
            .unwrap(),
        values[..6]
    );
    assert!(matches!(
        state.append(
            0,
            NumericTensor::from_i32_slice(&[1, 2], &[1, 2], &context).unwrap(),
            &context
        ),
        Err(AppendStreamError::Frontier { .. })
    ));
    assert_eq!(state.len(), 5);
    assert!(state
        .append(5, NumericTensor::new([1, 2], vec![1., 2.]), &context)
        .is_err());
    assert!(state.read(0..4, &context).is_err());
    assert!(ResidentAppendStream::<NumericTensor>::new(
        AppendStreamSpec {
            slot: 0,
            width: 2,
            element: TensorElementType::I32
        },
        AppendStreamLimits {
            entries: 4,
            page_entries: 2,
            read_entries: 2
        },
        31,
        100,
        1000
    )
    .is_err());
}
