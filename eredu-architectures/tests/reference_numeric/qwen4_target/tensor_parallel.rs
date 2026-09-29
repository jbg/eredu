//! Whole-target TP with independently authored nonzero weights and compact table transport.
use super::*;
use eredu_architectures::qwen4_exp::target::{MixerSpec, TargetTensorPartition};
use eredu_checkpoint::store::TensorSelection;
use eredu_nn::DistributedNeuralBackend;
use eredu_runtime::*;
use std::sync::atomic::{AtomicUsize, Ordering};

type Runtime = LayerwiseRuntime<
    TargetModel<NumericBackend>,
    NumericBackend,
    State,
    ResidentUnitWindow<Unit<NumericBackend>>,
>;

pub(super) struct Bind<'a> {
    pub(super) parameters: &'a Parameters,
    pub(super) partition: Option<&'a TargetTensorPartition>,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
        let full = &self.parameters.0[name];
        let expected_shape = value.shape.clone();
        let selected = self
            .partition
            .and_then(|partition| partition.parameter_selections(name));
        *value = match selected {
            None => full.clone(),
            Some(selections) => {
                let mut axis = None;
                let pieces: Vec<_> = selections
                    .iter()
                    .map(|selection| {
                        let TensorSelection::Range {
                            axis: current,
                            start,
                            end,
                        } = selection
                        else {
                            panic!("unexpected TP fixture selection {selection:?}");
                        };
                        if let Some(axis) = axis {
                            assert_eq!(axis, *current);
                        }
                        axis = Some(*current);
                        full.axis_slice(*current, *start, *end)
                    })
                    .collect();
                NumericTensor::concatenate(
                    &pieces,
                    axis.unwrap() as i32,
                    &NumericContext::default(),
                )
                .unwrap()
            }
        };
        assert_eq!(
            value.shape, expected_shape,
            "local parameter geometry: {name}"
        );
    }
}

pub(super) fn runtime(
    bound: BoundTargetSpec,
    parameters: &Parameters,
    partition: Option<&TargetTensorPartition>,
    ctx: &NumericContext,
) -> Runtime {
    let count = bound.geometry().units.len();
    let mut architecture = match partition {
        Some(partition) => {
            TargetModel::<NumericBackend>::new_tensor_parallel(bound, partition.clone(), ctx)
        }
        None => TargetModel::<NumericBackend>::new(bound, ctx),
    }
    .unwrap();
    let mut bind = Bind {
        parameters,
        partition,
    };
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut architecture).visit_parameters_mut(&mut bind);
    let units = (0..count)
        .map(|ordinal| {
            let mut unit = architecture.construct_unit(ordinal, ctx).unwrap();
            unit.visit_parameters_mut(&mut bind);
            unit
        })
        .collect();
    LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units))
}

struct CountingRows {
    calls: Arc<AtomicUsize>,
    inner: Rows,
}
impl RowLookupProvider<NumericBackend> for CountingRows {
    fn has_row_parameter(&self, parameter: &eredu_nn::ParameterId) -> bool {
        self.inner.has_row_parameter(parameter)
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        ctx: &NumericContext,
    ) -> Result<NumericTensor, RowLookupError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.lookup_rows(spec, rows, access, ctx)
    }
}

pub(super) fn compare_state(
    actual: &State,
    global: &State,
    spec: &TargetSpec,
    rank: usize,
    ranks: usize,
    ctx: &NumericContext,
) {
    let mut expected = global.clone();
    for (ordinal, unit) in spec.units.iter().enumerate() {
        let layer = &mut expected.as_mut()[ordinal];
        match unit {
            UnitSpec::Decoder {
                mixer: MixerSpec::Recurrent(recurrent),
                ..
            } => {
                let value_heads = recurrent.mixer.value_heads as usize;
                let matrix = layer.fixed[&StateTensorRole::Recurrent].as_ref().unwrap();
                layer.fixed.insert(
                    StateTensorRole::Recurrent,
                    Some(matrix.axis_slice(
                        1,
                        rank * value_heads / ranks,
                        (rank + 1) * value_heads / ranks,
                    )),
                );
                let key_width = (recurrent.mixer.key_heads * recurrent.mixer.key_head_dim) as usize;
                let value_width =
                    (recurrent.mixer.value_heads * recurrent.mixer.value_head_dim) as usize;
                let history = layer.fixed[&StateTensorRole::Convolution { slot: 0 }]
                    .as_ref()
                    .unwrap();
                let pieces: Vec<_> = [
                    (0, key_width),
                    (key_width, key_width),
                    (2 * key_width, value_width),
                ]
                .into_iter()
                .map(|(start, width)| {
                    history.axis_slice(
                        2,
                        start + rank * width / ranks,
                        start + (rank + 1) * width / ranks,
                    )
                })
                .collect();
                layer.fixed.insert(
                    StateTensorRole::Convolution { slot: 0 },
                    Some(NumericTensor::concatenate(&pieces, 2, ctx).unwrap()),
                );
            }
            UnitSpec::Decoder {
                mixer: MixerSpec::Indexed(attention),
                ..
            } => {
                let heads = attention.kv_heads as usize;
                let width = heads.div_ceil(ranks);
                let first = if ranks > heads {
                    rank / (ranks / heads)
                } else {
                    rank * width
                };
                let cache = layer.attention.as_mut().unwrap();
                cache.keys = Some(
                    cache
                        .keys
                        .as_ref()
                        .unwrap()
                        .axis_slice(1, first, first + width),
                );
                cache.values = Some(cache.values.as_ref().unwrap().axis_slice(
                    1,
                    first,
                    first + width,
                ));
            }
            UnitSpec::Lexical { .. } => {}
        }
    }
    for (ordinal, (actual, expected)) in actual.as_ref().iter().zip(expected.as_ref()).enumerate() {
        assert_eq!(actual.position(), expected.position());
        assert_eq!(
            actual.fixed.keys().collect::<Vec<_>>(),
            expected.fixed.keys().collect::<Vec<_>>()
        );
        for (role, value) in &expected.fixed {
            let actual = actual.fixed[role].as_ref().unwrap();
            let expected = value.as_ref().unwrap();
            assert_tensor_close(actual, expected, "whole-target TP fixed state");
            assert_eq!(
                actual.exact_i32, expected.exact_i32,
                "retained integer history"
            );
        }
        if let Some(cache) = &expected.attention {
            let actual = actual.attention.as_ref().unwrap();
            assert_tensor_close(
                actual.keys.as_ref().unwrap(),
                cache.keys.as_ref().unwrap(),
                &format!("whole-target TP unit{ordinal} K history"),
            );
            assert_tensor_close(
                actual.values.as_ref().unwrap(),
                cache.values.as_ref().unwrap(),
                "whole-target TP V history",
            );
        }
        assert_eq!(actual.streams.len(), expected.streams.len());
        for ((slot, lane, a), (other_slot, other_lane, b)) in
            actual.streams.iter().zip(&expected.streams)
        {
            assert_eq!((slot, lane, a.len()), (other_slot, other_lane, b.len()));
            for entry in 0..a.len() {
                assert_tensor_close(
                    &a.clone().read(entry..entry + 1, ctx).unwrap(),
                    &b.clone().read(entry..entry + 1, ctx).unwrap(),
                    "replicated QSA summaries and positions",
                );
            }
        }
    }
}

#[test]
fn qwen4_whole_target_tp2_tp4_matches_cached_state_replay_and_owner_only_rows() {
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
    let bound = bind_spec(spec.clone());
    let mut architecture = TargetModel::<NumericBackend>::new(bound.clone(), &ctx).unwrap();
    let mut parameters = Parameters::default();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut architecture).visit_parameters_mut(&mut parameters);
    for ordinal in 0..spec.units.len() {
        architecture
            .construct_unit(ordinal, &ctx)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    let (ids, visible, embeddings) = input_values();
    let mut ordinary = runtime(bound.clone(), &parameters, None, &ctx);
    let mut full_state = state(&spec);
    let mut provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: Rows::default(),
    };
    let missing_plan_state = full_state.clone();
    let isolated = NumericParallelContext::new(0, NumericParallelGroup::new(2));
    assert!(ordinary
        .forward_parallel_with_provider_and_observer(
            TargetInput {
                ids: Some(OriginalTokenIds::Host(&[3, 4])),
                batch: 2,
                tokens: 1,
                embeddings: None,
                visible: None,
                rotary: None,
                position_delta: None
            },
            &mut full_state,
            ExpertPass::Prefill,
            &mut provider,
            &isolated,
            &ctx,
            &mut Observe::default(),
        )
        .is_err());
    session::assert_checkpoint(&full_state, &missing_plan_state);
    assert_eq!(
        isolated.next_sequence.load(Ordering::Relaxed),
        0,
        "missing partition rejects before communication"
    );
    let mut baseline_observer = Observe::default();
    let expected = ordinary
        .forward_with_provider_and_observer(
            TargetInput {
                ids: Some(OriginalTokenIds::Host(&ids)),
                batch: 2,
                tokens: 19,
                embeddings: Some(&embeddings),
                visible: Some(TokenVisibility::Host(&visible)),
                rotary: None,
                position_delta: None,
            },
            &mut full_state,
            ExpertPass::Prefill,
            &mut provider,
            &ctx,
            &mut baseline_observer,
        )
        .unwrap();
    assert!(expected.data.iter().any(|value| value.abs() > 0.01));
    let prefill_state = full_state.clone();
    let mut chunk_runtime = runtime(bound.clone(), &parameters, None, &ctx);
    let mut chunk_state = state(&spec);
    let mut chunk_provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: Rows::default(),
    };
    let mut chunk_output = Vec::new();
    let mut chunk_observer = Observe::default();
    for (start, end) in [(0, 3), (3, 8), (8, 19)] {
        let tokens: Vec<_> = (0..2)
            .flat_map(|lane| ids[lane * 19 + start..lane * 19 + end].iter().copied())
            .collect();
        let mask: Vec<_> = (0..2)
            .flat_map(|lane| visible[lane * 19 + start..lane * 19 + end].iter().copied())
            .collect();
        chunk_output.push(
            chunk_runtime
                .forward_with_provider_and_observer(
                    TargetInput {
                        ids: Some(OriginalTokenIds::Host(&tokens)),
                        batch: 2,
                        tokens: (end - start) as i32,
                        embeddings: Some(&embeddings.axis_slice(1, start, end)),
                        visible: Some(TokenVisibility::Host(&mask)),
                        rotary: None,
                        position_delta: None,
                    },
                    &mut chunk_state,
                    ExpertPass::Prefill,
                    &mut chunk_provider,
                    &ctx,
                    &mut chunk_observer,
                )
                .unwrap(),
        );
    }
    assert_tensor_close(
        &NumericTensor::concatenate(&chunk_output, 1, &ctx).unwrap(),
        &expected,
        "ordinary chunked fixture prefill",
    );
    compare_state(&chunk_state, &prefill_state, &spec, 0, 1, &ctx);
    let decoded: Vec<_> = (0..4)
        .map(|step| {
            let tokens = [3 + step, 11 + step];
            ordinary
                .forward_with_provider_and_observer(
                    TargetInput {
                        ids: Some(OriginalTokenIds::Host(&tokens)),
                        batch: 2,
                        tokens: 1,
                        embeddings: None,
                        visible: None,
                        rotary: None,
                        position_delta: None,
                    },
                    &mut full_state,
                    ExpertPass::Decode,
                    &mut provider,
                    &ctx,
                    &mut Observe::default(),
                )
                .unwrap()
        })
        .collect();

    for ranks in [2, 4] {
        let fingerprints: std::collections::BTreeSet<_> = (0..ranks)
            .map(|rank| {
                bound
                    .tensor_partition(&spec.tensor_partition(rank, ranks).unwrap())
                    .unwrap()
                    .state_fingerprint()
                    .to_owned()
            })
            .collect();
        assert_eq!(
            fingerprints.len(),
            ranks,
            "rank-local state identities must differ"
        );
        let group = NumericParallelGroup::new(ranks);
        let calls: Vec<_> = (0..ranks).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        std::thread::scope(|scope| {
            let baseline_paths: std::collections::BTreeSet<_> =
                baseline_observer.paths.iter().cloned().collect();
            let handles: Vec<_> = (0..ranks)
                .map(|rank| {
                    let group = group.clone();
                    let calls = calls[rank].clone();
                    let baseline_paths = baseline_paths.clone();
                    let (
                        spec,
                        bound,
                        parameters,
                        ids,
                        visible,
                        embeddings,
                        expected,
                        prefill_state,
                        decoded,
                        full_state,
                    ) = (
                        &spec,
                        &bound,
                        &parameters,
                        &ids,
                        &visible,
                        &embeddings,
                        &expected,
                        &prefill_state,
                        &decoded,
                        &full_state,
                    );
                    scope.spawn(move || {
                        let ctx = NumericContext {
                            bind_checkpoint_values: true,
                            ..Default::default()
                        };
                        let parallel = NumericParallelContext::new(rank, group);
                        let partition = spec.tensor_partition(rank, ranks).unwrap();
                        let mut runtime =
                            runtime(bound.clone(), parameters, Some(&partition), &ctx);
                        let mut local_state = state(partition.local_spec());
                        let UnitSpec::Lexical { spec: lexical, .. } = &spec.units[1] else {
                            panic!()
                        };
                        let rows = TensorParallelRowLookup::<NumericBackend, CountingRows, _>::new(
                            lexical.embedding.lookup_spec().clone(),
                            128,
                            (rank == 0).then(|| CountingRows {
                                calls,
                                inner: Rows::default(),
                            }),
                            &parallel,
                            &parallel,
                            rank,
                            0,
                        )
                        .unwrap();
                        let mut provider = ParameterProviders {
                            grouped: ResidentExpertProvider,
                            rows,
                        };
                        for wrong_group in [
                            NumericParallelContext::new(rank, NumericParallelGroup::new(ranks + 1)),
                            NumericParallelContext::new(
                                (rank + 1) % ranks,
                                NumericParallelGroup::new(ranks),
                            ),
                        ] {
                            let before = local_state.clone();
                            assert!(runtime
                                .forward_parallel_with_provider_and_observer(
                                    TargetInput {
                                        ids: Some(OriginalTokenIds::Host(&[3, 4])),
                                        batch: 2,
                                        tokens: 1,
                                        embeddings: None,
                                        visible: None,
                                        rotary: None,
                                        position_delta: None
                                    },
                                    &mut local_state,
                                    ExpertPass::Prefill,
                                    &mut provider,
                                    &wrong_group,
                                    &ctx,
                                    &mut Observe::default(),
                                )
                                .is_err());
                            session::assert_checkpoint(&local_state, &before);
                            assert_eq!(
                                wrong_group.next_sequence.load(Ordering::Relaxed),
                                0,
                                "wrong group rejects before communication"
                            );
                        }
                        let mut output = Vec::new();
                        let mut observed = Observe::default();
                        for (start, end) in [(0, 3), (3, 8), (8, 19)] {
                            let tokens: Vec<_> = (0..2)
                                .flat_map(|lane| {
                                    ids[lane * 19 + start..lane * 19 + end].iter().copied()
                                })
                                .collect();
                            let mask: Vec<_> = (0..2)
                                .flat_map(|lane| {
                                    visible[lane * 19 + start..lane * 19 + end].iter().copied()
                                })
                                .collect();
                            output.push(
                                runtime
                                    .forward_parallel_with_provider_and_observer(
                                        TargetInput {
                                            ids: Some(OriginalTokenIds::Host(&tokens)),
                                            batch: 2,
                                            tokens: (end - start) as i32,
                                            embeddings: Some(&embeddings.axis_slice(1, start, end)),
                                            visible: Some(TokenVisibility::Host(&mask)),
                                            rotary: None,
                                            position_delta: None,
                                        },
                                        &mut local_state,
                                        ExpertPass::Prefill,
                                        &mut provider,
                                        &parallel,
                                        &ctx,
                                        &mut observed,
                                    )
                                    .unwrap(),
                            );
                        }
                        assert_tensor_close(
                            &NumericTensor::concatenate(&output, 1, &ctx).unwrap(),
                            expected,
                            "whole-target TP chunked prefill",
                        );
                        assert_eq!(
                            observed
                                .paths
                                .iter()
                                .cloned()
                                .collect::<std::collections::BTreeSet<_>>(),
                            baseline_paths,
                            "TP logical observation identities"
                        );
                        compare_state(&local_state, prefill_state, spec, rank, ranks, &ctx);
                        for (step, expected) in decoded.iter().enumerate() {
                            let tokens = [3 + step as u64, 11 + step as u64];
                            let checkpoint = local_state.clone();
                            let mut invoke = |state: &mut State| {
                                runtime
                                    .forward_parallel_with_provider_and_observer(
                                        TargetInput {
                                            ids: Some(OriginalTokenIds::Host(&tokens)),
                                            batch: 2,
                                            tokens: 1,
                                            embeddings: None,
                                            visible: None,
                                            rotary: None,
                                            position_delta: None,
                                        },
                                        state,
                                        ExpertPass::Decode,
                                        &mut provider,
                                        &parallel,
                                        &ctx,
                                        &mut Observe::default(),
                                    )
                                    .unwrap()
                            };
                            let value = invoke(&mut local_state);
                            assert_tensor_close(&value, expected, "whole-target TP cached decode");
                            let committed = local_state.clone();
                            local_state = checkpoint;
                            let replay = invoke(&mut local_state);
                            assert_tensor_exact(&value, &replay, "whole-target TP decode replay");
                            session::assert_checkpoint(&local_state, &committed);
                        }
                        compare_state(&local_state, full_state, spec, rank, ranks, &ctx);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert_eq!(
            calls[0].load(Ordering::SeqCst),
            11,
            "one table lookup per target invocation on its owner"
        );
        assert!(
            calls[1..]
                .iter()
                .all(|calls| calls.load(Ordering::SeqCst) == 0),
            "peer ranks must not acquire table rows"
        );
    }
}

#[test]
fn tensor_parallel_row_owner_failure_agrees_before_values_and_allows_retry() {
    struct FailOnce {
        failed: bool,
        inner: Rows,
    }
    impl RowLookupProvider<NumericBackend> for FailOnce {
        fn has_row_parameter(&self, id: &eredu_nn::ParameterId) -> bool {
            self.inner.has_row_parameter(id)
        }
        fn lookup_rows(
            &mut self,
            spec: &RowLookupSpec,
            rows: &[u64],
            access: ParameterBankAccess,
            ctx: &NumericContext,
        ) -> Result<NumericTensor, RowLookupError> {
            if !self.failed {
                self.failed = true;
                return Err(RowLookupError::OutOfRange { row: 99, rows: 24 });
            }
            self.inner.lookup_rows(spec, rows, access, ctx)
        }
    }
    let spec = specification();
    let UnitSpec::Lexical { spec: lexical, .. } = &spec.units[1] else {
        panic!()
    };
    let lookup = lexical.embedding.lookup_spec().clone();
    let group = NumericParallelGroup::new(2);
    let invalid = NumericParallelContext::new(0, group.clone());
    for (rank, owner, maximum) in [
        (0, 0, 128),
        (2, 1, 128),
        (0, 2, 128),
        (0, 1, 0),
        (1, 0, 128),
    ] {
        assert!(TensorParallelRowLookup::<NumericBackend, Rows, _>::new(
            lookup.clone(),
            maximum,
            None,
            &invalid,
            &invalid,
            rank,
            owner
        )
        .is_err());
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let lookup = lookup.clone();
                let group = group.clone();
                scope.spawn(move || {
                    let ctx = NumericContext::default();
                    let parallel = NumericParallelContext::new(rank, group);
                    let mut rows = TensorParallelRowLookup::<NumericBackend, FailOnce, _>::new(
                        lookup.clone(),
                        128,
                        (rank == 1).then(|| FailOnce {
                            failed: false,
                            inner: Rows::default(),
                        }),
                        &parallel,
                        &parallel,
                        rank,
                        1,
                    )
                    .unwrap();
                    let error = rows
                        .lookup_rows(&lookup, &[2, 2, 5], ParameterBankAccess::Bulk, &ctx)
                        .unwrap_err();
                    if rank == 1 {
                        assert!(matches!(error, RowLookupError::OutOfRange { row: 99, .. }));
                    } else {
                        assert!(matches!(error, RowLookupError::ParallelPeerFailure));
                    }
                    assert_eq!(
                        parallel.next_sequence.load(Ordering::Relaxed),
                        1,
                        "failure must stop before row-value collective"
                    );
                    let actual = rows
                        .lookup_rows(&lookup, &[2, 2, 5], ParameterBankAccess::Bulk, &ctx)
                        .unwrap();
                    let expected = Rows::default()
                        .lookup_rows(&lookup, &[2, 2, 5], ParameterBankAccess::Bulk, &ctx)
                        .unwrap();
                    assert_tensor_exact(
                        &actual,
                        &expected,
                        "retry preserves duplicate lookup order",
                    );
                    assert_eq!(
                        parallel.next_sequence.load(Ordering::Relaxed),
                        3,
                        "successful retry exchanges status then values"
                    );
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
    // Idle pipeline stages join the wider status wave but never the row-data group.
    let data = NumericParallelGroup::new(2);
    let status = NumericParallelGroup::new(4);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|rank| {
                let data = data.clone();
                let status = status.clone();
                let lookup = lookup.clone();
                scope.spawn(move || {
                    let ctx = NumericContext::default();
                    let status = NumericParallelContext::new(rank, status);
                    if rank < 2 {
                        let data = NumericParallelContext::new(rank, data);
                        let mut rows = TensorParallelRowLookup::<NumericBackend, FailOnce, _>::new(
                            lookup.clone(),
                            128,
                            (rank == 0).then(|| FailOnce {
                                failed: false,
                                inner: Rows::default(),
                            }),
                            &data,
                            &status,
                            rank,
                            0,
                        )
                        .unwrap();
                        let error = rows
                            .lookup_rows(&lookup, &[2, 2, 5], ParameterBankAccess::Bulk, &ctx)
                            .unwrap_err();
                        if rank == 0 {
                            assert!(matches!(error, RowLookupError::OutOfRange { row: 99, .. }));
                        } else {
                            assert!(matches!(error, RowLookupError::ParallelPeerFailure));
                        }
                        assert_eq!(data.next_sequence.load(Ordering::Relaxed), 0);
                        let actual = rows
                            .lookup_rows(&lookup, &[2, 2, 5], ParameterBankAccess::Bulk, &ctx)
                            .unwrap();
                        let expected = Rows::default()
                            .lookup_rows(&lookup, &[2, 2, 5], ParameterBankAccess::Bulk, &ctx)
                            .unwrap();
                        assert_tensor_exact(&actual, &expected, "separate wave status retry");
                        assert_eq!(data.next_sequence.load(Ordering::Relaxed), 1);
                    } else {
                        for failed in [1, 0] {
                            let local = NumericTensor::full_i32(0, &[1], &ctx).unwrap();
                            let actual =
                                NumericBackend::sum_parallel(local, &status, &ctx).unwrap();
                            assert_eq!(actual.to_i32_vec(&ctx).unwrap(), [failed]);
                        }
                    }
                    assert_eq!(status.next_sequence.load(Ordering::Relaxed), 2);
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}
