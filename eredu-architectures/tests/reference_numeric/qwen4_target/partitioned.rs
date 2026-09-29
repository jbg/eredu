//! Exact local state ownership across every lexical and mixer pipeline cut.
use super::*;
use eredu_architectures::partitioned_execution::TextPartitionArchitecture;
use eredu_runtime::*;

fn slice_state(global: &State, range: std::ops::Range<usize>) -> State {
    DeviceState::create(global.layout().slice(range.clone()).unwrap(), |index, _| {
        Ok::<_, Error>(global.as_ref()[range.start + index].clone())
    })
    .unwrap()
}

struct Input {
    ids: Vec<u64>,
    visible: Option<Vec<bool>>,
    embeddings: Option<NumericTensor>,
    cosine: Option<NumericTensor>,
    sine: Option<NumericTensor>,
    tokens: i32,
}
impl Input {
    fn new(start: usize, end: usize) -> Self {
        let (ids, visible, embeddings) = input_values();
        if start >= 19 {
            return Self {
                ids: vec![(start - 16) as u64, (start - 8) as u64],
                visible: None,
                embeddings: None,
                cosine: None,
                sine: None,
                tokens: 1,
            };
        }
        let tokens = (end - start) as i32;
        Self {
            ids: (0..2)
                .flat_map(|lane| ids[lane * 19 + start..lane * 19 + end].iter().copied())
                .collect(),
            visible: Some(
                (0..2)
                    .flat_map(|lane| visible[lane * 19 + start..lane * 19 + end].iter().copied())
                    .collect(),
            ),
            embeddings: Some(embeddings.axis_slice(1, start, end)),
            cosine: Some(NumericTensor::new(
                [2, tokens, 2],
                vec![0.8; 4 * tokens as usize],
            )),
            sine: Some(NumericTensor::new(
                [2, tokens, 2],
                vec![0.6; 4 * tokens as usize],
            )),
            tokens,
        }
    }
    fn input(&self) -> TargetInput<'_, NumericTensor> {
        TargetInput {
            ids: Some(OriginalTokenIds::Host(&self.ids)),
            batch: 2,
            tokens: self.tokens,
            embeddings: self.embeddings.as_ref(),
            visible: self.visible.as_deref().map(TokenVisibility::Host),
            rotary: self
                .cosine
                .as_ref()
                .zip(self.sine.as_ref())
                .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            position_delta: Some(4),
        }
    }
}

fn pipeline(
    spec: &TargetSpec,
    parameters: &Parameters,
    cut: usize,
    rank: usize,
    ranks: usize,
    groups: &[Arc<NumericParallelGroup>],
) -> (Vec<NumericTensor>, State, Vec<String>) {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let tensor = (ranks > 1).then(|| spec.tensor_partition(rank, ranks).unwrap());
    let local = tensor
        .as_ref()
        .map_or(spec, |partition| partition.local_spec());
    let ranges = [0..cut, cut..spec.units.len()];
    let empty = state(local);
    let mut states: Vec<_> = ranges
        .iter()
        .map(|range| slice_state(&empty, range.clone()))
        .collect();
    let mut models: Vec<_> = ranges.iter().enumerate().map(|(stage, range)| {
        let bound = bind_spec(spec.clone());
        let mut model = match &tensor {
            Some(plan) => TargetModel::<NumericBackend>::new_tensor_parallel(bound, plan.clone(), &ctx),
            None => TargetModel::<NumericBackend>::new(bound, &ctx),
        }.unwrap();
        <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model)
            .visit_parameters_mut(&mut tensor_parallel::Bind { parameters, partition: tensor.as_ref() });
        let selected = PartitionState::new(states[stage].layout().clone(), range.start).unwrap();
        model.set_partition_state(&selected).unwrap();
        let identity = model.state_identity(&selected, Default::default()).unwrap();
        let identity_again = model.clone().state_identity(&selected, Default::default()).unwrap();
        assert_eq!(identity, identity_again, "pipeline cloning preserves global state identity");
        model
    }).collect();
    let mut units: Vec<Vec<_>> = models
        .iter()
        .zip(&ranges)
        .map(|(model, range)| {
            range
                .clone()
                .map(|index| {
                    let mut unit = model.construct_unit(index, &ctx).unwrap();
                    unit.visit_parameters_mut(&mut tensor_parallel::Bind {
                        parameters,
                        partition: tensor.as_ref(),
                    });
                    unit
                })
                .collect()
        })
        .collect();
    let parallels: Vec<_> = groups
        .iter()
        .map(|group| NumericParallelContext::new(rank, group.clone()))
        .collect();
    let UnitSpec::Lexical { spec: lexical, .. } = &spec.units[1] else {
        panic!()
    };
    let mut outputs = Vec::new();
    let mut observed = Observe::default();
    for (start, end) in [(0, 3), (3, 8), (8, 19), (19, 20), (20, 21), (21, 22)] {
        let input = Input::new(start, end);
        let before = states.clone();
        let mut expected_replay = None;
        // Replay every cached decode from a fork of the exact local snapshots.
        for replay in 0..if start >= 19 { 2 } else { 1 } {
            if replay == 1 {
                states = before.clone();
            }
            let mut incoming = None;
            let mut replay_observer = Observe::default();
            let observer = if replay == 0 {
                &mut observed
            } else {
                &mut replay_observer
            };
            for stage in 0..2 {
                let model = &mut models[stage];
                let state = &mut states[stage];
                let layout = state.layout().clone();
                let parallel = (ranks > 1).then_some(&parallels[stage]);
                if stage == 1 {
                    let saved = state.clone();
                    let mut wrong = input.input();
                    wrong.tokens += 1;
                    let error = match model.begin_text_partition(
                        wrong,
                        incoming.clone(),
                        state,
                        &layout,
                        0,
                        parallel,
                        &ctx,
                        &mut NoopObserver,
                    ) {
                        Err(error) => error,
                        Ok(_) => {
                            panic!("inbound boundary must match the retained request dimensions")
                        }
                    };
                    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
                    let mut boundary_error = false;
                    while let Some(error) = cause {
                        boundary_error |= matches!(
                            error.downcast_ref::<RequestError>(),
                            Some(RequestError::Boundary)
                        );
                        cause = error.source();
                    }
                    assert!(
                        boundary_error,
                        "invalid inbound dimensions retain their typed cause"
                    );
                    session::assert_checkpoint(state, &saved);
                }
                let mut forward = model
                    .begin_text_partition(
                        input.input(),
                        incoming.take(),
                        state,
                        &layout,
                        0,
                        parallel,
                        &ctx,
                        observer,
                    )
                    .unwrap();
                assert_eq!(forward.context.request.ids(), input.ids);
                assert_eq!(forward.context.request.position_delta(), 4);
                if let Some(cosine) = &input.cosine {
                    assert_tensor_exact(
                        &forward.context.request.boundary().cosine,
                        cosine,
                        "PP media products",
                    );
                }
                forward.hidden = model
                    .enter_partition_group(
                        0,
                        &forward.hidden,
                        state,
                        &mut forward.context,
                        parallel,
                        &ctx,
                    )
                    .unwrap();
                let rows = TensorParallelRowLookup::<NumericBackend, Rows, _>::new(
                    lexical.embedding.lookup_spec().clone(),
                    128,
                    (rank == 0).then(Rows::default),
                    &parallels[stage],
                    &parallels[stage],
                    rank,
                    0,
                )
                .unwrap();
                let mut provider = ParameterProviders {
                    grouped: ResidentExpertProvider,
                    rows,
                };
                let mut plain = ParameterProviders {
                    grouped: ResidentExpertProvider,
                    rows: Rows::default(),
                };
                for (local_index, index) in ranges[stage].clone().enumerate() {
                    forward.hidden = match parallel {
                        Some(parallel) => model.forward_unit_parallel_observed_with_provider(
                            0,
                            index,
                            &mut units[stage][local_index],
                            &forward.hidden,
                            state,
                            &mut forward.context,
                            ExpertPass::Prefill,
                            &mut provider,
                            parallel,
                            &ctx,
                            observer,
                        ),
                        None => model.forward_unit_observed_with_provider(
                            0,
                            index,
                            &mut units[stage][local_index],
                            &forward.hidden,
                            state,
                            &mut forward.context,
                            ExpertPass::Prefill,
                            &mut plain,
                            &ctx,
                            observer,
                        ),
                    }
                    .unwrap();
                }
                forward.hidden = model
                    .leave_partition_group(
                        0,
                        &forward.hidden,
                        state,
                        &mut forward.context,
                        parallel,
                        &ctx,
                    )
                    .unwrap();
                match model
                    .finish_partition_observed(
                        &forward.hidden,
                        state,
                        &forward.context,
                        stage == 1,
                        parallel,
                        &ctx,
                        observer,
                    )
                    .unwrap()
                {
                    LayeredPartitionOutput::Boundary { hidden, auxiliary } => {
                        let schema = spec.boundary_schema().unwrap();
                        let wire = schema.encode(auxiliary).unwrap();
                        let restored = schema
                            .decode(wire.into_iter().map(|value| value.into_parts().1).collect())
                            .unwrap();
                        incoming = Some((hidden, restored));
                    }
                    LayeredPartitionOutput::Final { output, retained } => {
                        assert_tensor_exact(
                            &retained.unwrap(),
                            &forward.hidden,
                            "PP retains pre-collapse residuals",
                        );
                        if let Some(expected) = &expected_replay {
                            assert_tensor_exact(&output, expected, "partition snapshot replay");
                        } else {
                            expected_replay = Some(output.clone());
                            outputs.push(output);
                        }
                    }
                }
            }
        }
    }
    let merged = DeviceState::create(local.state_layout().unwrap(), |index, _| {
        let stage = usize::from(index >= cut);
        Ok::<_, Error>(states[stage].as_ref()[index - ranges[stage].start].clone())
    })
    .unwrap();
    (outputs, merged, observed.paths)
}

#[test]
fn qwen4_pp_cuts_and_tp2_pp2_preserve_cached_state_media_and_replay() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut config = configuration();
    config.attention.heads = 4;
    config.attention.kv_heads = 2;
    config.recurrent.key_heads = 4;
    config.recurrent.value_heads = 8;
    config.experts.intermediate = 4;
    let spec = specification_for(config);
    let mut parameters = Parameters::default();
    let mut full = TargetModel::<NumericBackend>::new(bind_spec(spec.clone()), &ctx).unwrap();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut full).visit_parameters_mut(&mut parameters);
    for index in 0..spec.units.len() {
        full.construct_unit(index, &ctx)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    let mut ordinary = tensor_parallel::runtime(bind_spec(spec.clone()), &parameters, None, &ctx);
    let mut state = state(&spec);
    let mut observer = Observe::default();
    let mut provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: Rows::default(),
    };
    let expected: Vec<_> = [(0, 3), (3, 8), (8, 19), (19, 20), (20, 21), (21, 22)]
        .into_iter()
        .map(|(start, end)| {
            ordinary
                .forward_with_provider_and_observer(
                    Input::new(start, end).input(),
                    &mut state,
                    ExpertPass::Prefill,
                    &mut provider,
                    &ctx,
                    &mut observer,
                )
                .unwrap()
        })
        .collect();
    // The direct partition adapter emits internal hooks. Unit envelopes belong
    // to the shared traversal driver, exercised by the pipeline-input fixture.
    let unit_envelopes: std::collections::BTreeSet<_> = spec
        .units
        .iter()
        .flat_map(|unit| {
            [
                eredu_core::UnitObservation::Input.path(&unit.path()),
                eredu_core::UnitObservation::Output.path(&unit.path()),
            ]
        })
        .collect();
    let expected_paths: Vec<_> = observer
        .paths
        .iter()
        .filter(|path| !unit_envelopes.contains(*path))
        .cloned()
        .collect();
    for ranks in [1, 2] {
        for cut in 1..spec.units.len() {
            let groups: Vec<_> = (0..2).map(|_| NumericParallelGroup::new(ranks)).collect();
            let results = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..ranks)
                    .map(|rank| {
                        let (spec, parameters, groups) = (&spec, &parameters, &groups);
                        scope.spawn(move || pipeline(spec, parameters, cut, rank, ranks, groups))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for (rank, (actual, actual_state, paths)) in results.into_iter().enumerate() {
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_tensor_close(actual, expected, "TP/PP cached logits");
                }
                tensor_parallel::compare_state(&actual_state, &state, &spec, rank, ranks, &ctx);
                assert_eq!(
                    paths, expected_paths,
                    "global observation identities across PP cuts"
                );
            }
        }
    }
}
