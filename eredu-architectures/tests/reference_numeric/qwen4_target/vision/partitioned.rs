//! Direct conditional TP equations against independently composed vision/target execution.
use super::*;
use eredu_architectures::qwen4_exp::conditional::ConditionalUnit;

fn parameter_values(
    target: &PreparedTarget,
    ingress: &MediaIngress,
) -> BTreeMap<String, NumericTensor> {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut values = BTreeMap::new();
    let mut add = |prepared: &PreparedParameters| {
        for (name, recipe) in prepared.recipes() {
            values.insert(
                name.clone(),
                payload::recipe_value(recipe, prepared.source().as_ref(), &context).unwrap(),
            );
        }
    };
    add(target.static_parameters());
    for index in 0..target.spec().units.len() {
        add(target.unit(index).unwrap());
    }
    add(ingress.vision().static_parameters());
    for index in 0..ingress.vision().config().layer_count() {
        add(ingress.vision().block(index).unwrap());
    }
    for index in 0..target.spec().units.len() {
        if matches!(target.spec().units[index], UnitSpec::Decoder { .. }) {
            let experts: Vec<_> = (0..target.spec().configuration().experts.count as usize)
                .map(|expert| target.expert(index, expert).unwrap())
                .collect();
            for name in experts[0].recipes().keys() {
                let pieces: Vec<_> = experts
                    .iter()
                    .map(|expert| {
                        payload::recipe_value(
                            &expert.recipes()[name],
                            expert.source().as_ref(),
                            &context,
                        )
                        .unwrap()
                    })
                    .collect();
                values.insert(
                    name.clone(),
                    NumericTensor::concatenate(&pieces, 0, &context).unwrap(),
                );
            }
        }
    }
    values
}

struct LocalBind<'a> {
    values: &'a BTreeMap<String, NumericTensor>,
    layout: &'a LocalModelLayout,
    context: &'a NumericContext,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for LocalBind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
        let mut data = self.values[name].clone();
        if name.ends_with("conv1d.weight") && data.shape.len() == 3 && value.shape.len() == 2 {
            data = data
                .reshape(&[data.shape[0], data.shape[2]], self.context)
                .unwrap();
        }
        let layout = self.layout.tensor(name).unwrap();
        for selection in layout.additional_placements() {
            data = select_parameter(&data, selection).unwrap();
        }
        data = select_parameter(&data, layout.placement()).unwrap();
        assert_eq!(data.shape, value.shape, "{name}");
        *value = data;
    }
}

type ConditionalRuntime = LayerwiseRuntime<
    ConditionalModel<NumericBackend>,
    NumericBackend,
    State,
    ResidentUnitWindow<ConditionalUnit<NumericBackend>>,
>;

#[allow(clippy::too_many_arguments)]
fn conditional_forward<P: TensorParallelParameterProvider<NumericBackend>>(
    runtime: &mut ConditionalRuntime,
    input: ConditionalInput<'_, NumericTensor>,
    state: &mut State,
    pass: ExpertPass,
    provider: &mut P,
    parallel: &NumericParallelContext,
    context: &NumericContext,
    observer: &mut Observe,
) -> Result<NumericTensor, Error>
where
    P::Error: std::fmt::Display,
{
    if parallel.group.size == 1 {
        runtime.forward_with_provider_and_observer(input, state, pass, provider, context, observer)
    } else {
        runtime.forward_parallel_with_provider_and_observer(
            input, state, pass, provider, parallel, context, observer,
        )
    }
    .map_err(|error| Error::backend(error.to_string()))
}

fn run_tensor_rank(
    rank: usize,
    group: Arc<NumericParallelGroup>,
    target: &PreparedTarget,
    ingress: &MediaIngress,
    values: &BTreeMap<String, NumericTensor>,
    parameters: &ArchitectureParameterDescription,
    expected: &[NumericTensor],
) {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let ranks = group.size;
    let rank_topology =
        ParallelRankTopology::new(ParallelTopology::new(ranks, 1, 1, 1).unwrap(), rank).unwrap();
    let tensor = target.tensor_partition(rank, ranks).unwrap();
    let global_target =
        TargetModel::<NumericBackend>::new(tensor.source_bound_spec().clone(), &context).unwrap();
    let local_target = TargetModel::<NumericBackend>::new_tensor_parallel(
        tensor.source_bound_spec().clone(),
        tensor.partition().clone(),
        &context,
    )
    .unwrap();
    let target_layout = tensor
        .partition()
        .local_layout(
            &global_target.parameter_description(&context).unwrap(),
            &local_target.parameter_description(&context).unwrap(),
        )
        .unwrap();
    let mut layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        parameters,
        rank_topology,
    )
    .unwrap();
    for (name, tensor) in target_layout.tensors() {
        layout.insert(name.to_owned(), tensor.clone());
    }
    let partition =
        PartitionState::new(tensor.partition().local_spec().state_layout().unwrap(), 0).unwrap();
    let mut model = if ranks == 1 {
        ConditionalModel::<NumericBackend>::new(
            target.bound_spec().unwrap(),
            ingress.clone(),
            ingress.vision().config().clone(),
            &context,
        )
        .unwrap()
    } else {
        ConditionalModel::<NumericBackend>::new_partitioned(
            tensor.source_bound_spec().clone(),
            tensor.partition().clone(),
            ingress.clone(),
            ingress.vision().config().clone(),
            rank_topology,
            &layout,
            &partition,
            &context,
        )
        .unwrap()
    };
    let mut binding = LocalBind {
        values,
        layout: &layout,
        context: &context,
    };
    <ConditionalModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut binding);
    let mut units = Vec::<ConditionalUnit<NumericBackend>>::new();
    for group in 0..2 {
        let count = <ConditionalModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            State,
        >>::group_unit_count(&model, group)
        .unwrap();
        for index in 0..count {
            let mut unit = <ConditionalModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                State,
            >>::build_unit(&model, group, index, &context)
            .unwrap();
            unit.visit_parameters_mut(&mut binding);
            units.push(unit);
        }
    }
    let mut runtime =
        LayerwiseRuntime::<_, NumericBackend, State, _>::new(model, ResidentUnitWindow::new(units));
    let mut state = gguf::state_from_layout(partition.layout()).unwrap();
    let parallel = NumericParallelContext::new(rank, group);
    let rows = tensor
        .row_lookups(
            RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 65536,
                output_bytes: 32768,
            },
            ResidencyPolicy::Cacheable,
        )
        .unwrap();
    let banks = if rank == 0 {
        rows.entries()
            .iter()
            .map(|(id, row)| {
                (
                    id.clone(),
                    super::super::super::row_bank::SourceRows::new(row),
                )
            })
            .collect()
    } else {
        BTreeMap::new()
    };
    let mut provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: rows
            .bind_tensor_parallel::<NumericBackend, _, _>(banks, &parallel, &parallel, rank, 0)
            .unwrap(),
    };
    let inspector = Inspector(0.into());
    let source = prompt(&inspector);
    let prepared = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap()
        .prepare(&context)
        .unwrap();
    let mut observer = Observe::default();
    let output = conditional_forward(
        &mut runtime,
        ConditionalInput::Media(&prepared),
        &mut state,
        ExpertPass::Prefill,
        &mut provider,
        &parallel,
        &context,
        &mut observer,
    )
    .unwrap();
    let length = output.dim(1);
    assert_tensor_close(
        &output.axis_slice(1, length as usize - 1, length as usize),
        &expected[0],
        "conditional TP2 media prefill",
    );
    let saved = state.clone();
    for step in 0..16 {
        let ids = [(step % 12) as u64];
        let output = conditional_forward(
            &mut runtime,
            ConditionalInput::Target(TargetInput {
                ids: Some(OriginalTokenIds::Host(&ids)),
                batch: 1,
                tokens: 1,
                embeddings: None,
                visible: None,
                rotary: None,
                position_delta: None,
            }),
            &mut state,
            ExpertPass::Decode,
            &mut provider,
            &parallel,
            &context,
            &mut observer,
        )
        .unwrap();
        assert_tensor_close(
            &output,
            &expected[step + 1],
            "conditional TP2 cached decode",
        );
    }
    state = saved;
    let ids = [0u64];
    let replay = conditional_forward(
        &mut runtime,
        ConditionalInput::Target(TargetInput {
            ids: Some(OriginalTokenIds::Host(&ids)),
            batch: 1,
            tokens: 1,
            embeddings: None,
            visible: None,
            rotary: None,
            position_delta: None,
        }),
        &mut state,
        ExpertPass::Decode,
        &mut provider,
        &parallel,
        &context,
        &mut observer,
    )
    .unwrap();
    assert_tensor_close(&replay, &expected[1], "conditional TP2 restored decode");
}

#[test]
fn conditional_tp2_image_video_encoder_and_cached_target_match_separate_equations() {
    let (_directory, target, ingress) = setup();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let inspector = Inspector(0.into());
    let source = prompt(&inspector);
    let prepared = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap()
        .prepare(&context)
        .unwrap();
    let (expected, _) = reference(&target, &ingress, &prepared, &context);
    let values = parameter_values(&target, &ingress);
    let global = ConditionalModel::<NumericBackend>::new(
        target.bound_spec().unwrap(),
        ingress.clone(),
        ingress.vision().config().clone(),
        &context,
    )
    .unwrap();
    let parameters = global.parameter_description(&context).unwrap();
    // First compare unpartitioned full prefill with the independently composed,
    // chunked reference. TP differences cannot be hidden by that chunk boundary.
    run_tensor_rank(
        0,
        NumericParallelGroup::new(1),
        &target,
        &ingress,
        &values,
        &parameters,
        &expected,
    );
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for rank in 0..2 {
            let group = Arc::clone(&group);
            let target = &target;
            let ingress = &ingress;
            let values = &values;
            let parameters = &parameters;
            let expected = &expected;
            handles.push(scope.spawn(move || {
                run_tensor_rank(rank, group, target, ingress, values, parameters, expected)
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

#[test]
fn conditional_pp_decoder_boundaries_preserve_media_ids_and_cached_text_after_injection() {
    use eredu_architectures::composite_execution::CompositeArchitecture;
    let (_directory, target, ingress) = setup();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let inspector = Inspector(0.into());
    let source = prompt(&inspector);
    let prepared = ingress
        .admission_config()
        .admit(&source, &inspector)
        .unwrap()
        .prepare(&context)
        .unwrap();
    let (expected, _) = reference(&target, &ingress, &prepared, &context);
    let values = parameter_values(&target, &ingress);
    let tensor = target.tensor_partition(0, 1).unwrap();
    let global = ConditionalModel::<NumericBackend>::new(
        target.bound_spec().unwrap(),
        ingress.clone(),
        ingress.vision().config().clone(),
        &context,
    )
    .unwrap();
    let parameters = global.parameter_description(&context).unwrap();
    let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
        &parameters,
        ParallelRankTopology::new(ParallelTopology::new(1, 2, 1, 1).unwrap(), 0).unwrap(),
    )
    .unwrap();
    let full_state = tensor.partition().local_spec().state_layout().unwrap();
    // Every internal cut includes immediately after lexical injection and before
    // subsequent indexed/recurrent mixers. Vision executes on the first owner;
    // this test isolates the new typed conditional decoder boundary callbacks.
    for cut in 1..target.spec().units.len() {
        let ranges = [0..cut, cut..target.spec().units.len()];
        let mut models = vec![];
        let mut states = vec![];
        let mut units = vec![];
        for (stage, range) in ranges.iter().enumerate() {
            let partition =
                PartitionState::new(full_state.slice(range.clone()).unwrap(), range.start).unwrap();
            let mut model = ConditionalModel::<NumericBackend>::new_partitioned(
                tensor.source_bound_spec().clone(),
                tensor.partition().clone(),
                ingress.clone(),
                ingress.vision().config().clone(),
                ParallelRankTopology::new(ParallelTopology::new(1, 2, 1, 1).unwrap(), stage)
                    .unwrap(),
                &layout,
                &partition,
                &context,
            )
            .unwrap();
            let mut binding = LocalBind {
                values: &values,
                layout: &layout,
                context: &context,
            };
            <ConditionalModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut binding);
            let mut local_units = vec![];
            for index in range.clone() {
                let mut unit = <ConditionalModel<NumericBackend> as LayeredArchitecture<
                    NumericBackend,
                    State,
                >>::build_unit(&model, 0, index, &context)
                .unwrap();
                unit.visit_parameters_mut(&mut binding);
                local_units.push(unit);
            }
            models.push(model);
            states.push(gguf::state_from_layout(partition.layout()).unwrap());
            units.push(local_units);
        }
        let mut vision_units = vec![];
        for index in 0..ingress.vision().config().layer_count() {
            let mut unit = <ConditionalModel<NumericBackend> as LayeredArchitecture<
                NumericBackend,
                State,
            >>::build_unit(&models[0], 1, index, &context)
            .unwrap();
            unit.visit_parameters_mut(&mut LocalBind {
                values: &values,
                layout: &layout,
                context: &context,
            });
            vision_units.push(unit);
        }
        let rows = tensor
            .row_lookups(
                RowLookupLimits {
                    requests: 128,
                    rows_per_acquisition: 2,
                    acquisition_bytes: 128,
                    host_bytes: 65536,
                    output_bytes: 32768,
                },
                ResidencyPolicy::Cacheable,
            )
            .unwrap();
        let mut provider = ParameterProviders {
            grouped: ResidentExpertProvider,
            rows: rows
                .bind(
                    rows.entries()
                        .iter()
                        .map(|(id, row)| {
                            (
                                id.clone(),
                                super::super::super::row_bank::SourceRows::new(row),
                            )
                        })
                        .collect(),
                )
                .unwrap(),
        };
        let mut saved = None;
        for step in 0usize..5 {
            // Final iteration repeats the first cached token after restoring the
            // two stage-local checkpoints captured immediately after media.
            if step == 4 {
                states = saved.clone().unwrap();
            }
            let token = if step == 4 { 0 } else { step.saturating_sub(1) } as u64;
            let ids = [token];
            let input = || {
                if step == 0 {
                    ConditionalInput::Media(&prepared)
                } else {
                    ConditionalInput::Target(TargetInput {
                        ids: Some(OriginalTokenIds::Host(&ids)),
                        batch: 1,
                        tokens: 1,
                        embeddings: None,
                        visible: None,
                        rotary: None,
                        position_delta: None,
                    })
                }
            };
            let mut first = models[0]
                .begin_forward(input(), &mut states[0], &context)
                .unwrap();
            if step == 0 {
                for (index, unit) in vision_units.iter_mut().enumerate() {
                    first.hidden = models[0]
                        .forward_unit(
                            1,
                            index,
                            unit,
                            &first.hidden,
                            &mut states[0],
                            &mut first.context,
                            &context,
                        )
                        .unwrap();
                }
                let projected = models[0]
                    .complete_execution_group(
                        1,
                        &first.hidden,
                        &mut states[0],
                        &mut first.context,
                        &context,
                    )
                    .unwrap();
                first.hidden = models[0]
                    .begin_execution_group(
                        0,
                        &first.hidden,
                        &[&projected],
                        &mut states[0],
                        &mut first.context,
                        &context,
                    )
                    .unwrap();
            }
            let pass = if step == 0 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            for (local, index) in ranges[0].clone().enumerate() {
                first.hidden = models[0]
                    .forward_unit_with_provider(
                        0,
                        index,
                        &mut units[0][local],
                        &first.hidden,
                        &mut states[0],
                        &mut first.context,
                        pass,
                        &mut provider,
                        &context,
                    )
                    .unwrap();
            }
            let schema = target
                .spec()
                .boundary_schema()
                .unwrap()
                .wire_schema()
                .unwrap()
                .resolve(
                    1,
                    if step == 0 {
                        prepared.token_ids().dim(1)
                    } else {
                        1
                    },
                )
                .unwrap();
            let wire = <ConditionalModel<NumericBackend> as CompositeArchitecture<
                NumericBackend,
                State,
            >>::partition_boundary_values(
                &models[0], 0, 0, &schema, &first.hidden, &first.context
            )
            .unwrap()
            .unwrap();
            let before_embeddings = context.embedding_forward_calls.get();
            let mut second = models[1]
                .begin_forward(input(), &mut states[1], &context)
                .unwrap();
            assert_eq!(
                context.embedding_forward_calls.get(),
                before_embeddings,
                "later stage must await media activation and retained original-ID boundary"
            );
            let received = <ConditionalModel<NumericBackend> as CompositeArchitecture<
                NumericBackend,
                State,
            >>::accept_partition_boundary(
                &mut models[1],
                0,
                0,
                &schema,
                wire.into_iter().map(|value| value.into_parts().1).collect(),
                &mut second.context,
            )
            .unwrap()
            .unwrap();
            second.hidden = models[1]
                .enter_partition_group(
                    0,
                    &received,
                    &mut states[1],
                    &mut second.context,
                    None,
                    &context,
                )
                .unwrap();
            for (local, index) in ranges[1].clone().enumerate() {
                second.hidden = models[1]
                    .forward_unit_with_provider(
                        0,
                        index,
                        &mut units[1][local],
                        &second.hidden,
                        &mut states[1],
                        &mut second.context,
                        pass,
                        &mut provider,
                        &context,
                    )
                    .unwrap();
            }
            let output = models[1]
                .finish_partition(
                    &second.hidden,
                    &mut states[1],
                    &second.context,
                    true,
                    None,
                    &context,
                )
                .unwrap();
            let LayeredPartitionOutput::Final { output, .. } = output else {
                panic!("final owner did not publish");
            };
            let length = output.dim(1);
            let expected_step = if step == 4 { 1 } else { step };
            assert_tensor_close(
                &output.axis_slice(1, length as usize - 1, length as usize),
                &expected[expected_step],
                "conditional PP media/cached/replay",
            );
            if step == 0 {
                saved = Some(states.clone());
            }
        }
    }
}
