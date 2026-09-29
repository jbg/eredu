//! Prepared TP equations and explicit EP owner sums; this does not simulate exchange.
use super::*;
use eredu_nn::{GroupSelection, GroupedGatedProductOperator, GroupedNeuralBackend};
use eredu_runtime::{
    ParameterProvider, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
    TensorParallelParameterProvider,
};

use super::super::prepared_tensor_parallel::Bind as RecipeBind;

struct Owners(Vec<(Vec<usize>, NumericExpertBank)>);
impl ParameterProvider<NumericBackend> for Owners {
    type Error = Error;
    fn forward_grouped(
        &mut self,
        _: &mut NumericExpertBank,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        assert!(request.unit_observer.is_none());
        let mut sum = NumericTensor::zeros(request.input.shape.clone());
        for (ids, bank) in &mut self.0 {
            let mut indices = request.routes.group_indices().clone();
            let mut coefficients = request.routes.coefficients().clone();
            for (index, coefficient) in indices.data.iter_mut().zip(&mut coefficients.data) {
                match ids.iter().position(|id| *id == *index as usize) {
                    Some(local) => *index = local as f32,
                    None => {
                        *index = 0.;
                        *coefficient = 0.;
                    }
                }
            }
            let routes = GroupSelection::new(
                indices,
                request.routes.selected_scores().clone(),
                coefficients,
            );
            sum = sum.add(
                &bank.forward_grouped(request.input, &routes, context)?,
                context,
            )?;
        }
        Ok(sum)
    }
    fn forward_linear_routed(
        &mut self,
        _: &mut <NumericBackend as GroupedNeuralBackend>::LinearGroups,
        _: RoutedExpertRequest<'_, '_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        panic!("prediction uses gated experts")
    }
    fn forward_relu2_routed(
        &mut self,
        _: &mut <NumericBackend as GroupedNeuralBackend>::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, NumericTensor>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        panic!("prediction uses gated experts")
    }
}
impl TensorParallelParameterProvider<NumericBackend> for Owners {
    fn forward_grouped_tensor_parallel(
        &mut self,
        bank: &mut NumericExpertBank,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        _: usize,
        context: &NumericContext,
    ) -> Result<RoutedExpertTensorParallelOutput<NumericTensor>, Error> {
        Ok(RoutedExpertTensorParallelOutput::Partial(
            eredu_nn::TensorParallelGroupedOutput::new(
                self.forward_grouped(bank, request, context)?,
                None,
            ),
        ))
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut <NumericBackend as GroupedNeuralBackend>::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, NumericTensor>,
        _: usize,
        _: &NumericContext,
    ) -> Result<RoutedExpertTensorParallelOutput<NumericTensor>, Error> {
        panic!("prediction uses gated experts")
    }
}

#[test]
fn qwen4_prepared_prediction_tp2_ep2_equations_cached_restore_and_shared_logits() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
fn run() {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut config = configuration();
    config.recurrent.key_heads = 2;
    config.recurrent.value_heads = 4;
    config.prediction = Some(PredictionGeometry {
        layers: eredu_core::LayerSchedule::new(2, vec![LayerKind::Indexed, LayerKind::Recurrent])
            .unwrap(),
        rope_theta: 7777.,
    });
    let target_spec = specification_for(config.clone());
    let prediction_spec = spec(&config);
    let mut parameters = Parameters::default();
    let mut target =
        TargetModel::<NumericBackend>::new(bind_spec(target_spec.clone()), &context).unwrap();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut target).visit_parameters_mut(&mut parameters);
    for index in 0..target_spec.units.len() {
        target
            .construct_unit(index, &context)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    PredictionShared::<NumericBackend>::new(&prediction_spec, &context)
        .unwrap()
        .visit_parameters_mut(&mut parameters);
    for unit in &prediction_spec.units {
        PredictionUnit::<NumericBackend>::new(unit.clone(), &context)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    let mut values = table_values();
    for (name, value) in parameters.0 {
        let mut shape: Vec<_> = value.shape.iter().map(|n| *n as usize).collect();
        if name.ends_with(".conv1d.weight") && shape.len() == 2 {
            shape.insert(1, 1);
        }
        values.push((
            name,
            safetensors::Dtype::F32,
            shape,
            value.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    }
    let (_directory, source) = super::super::transforms::fixture(&values);
    let prepared_target = PreparedTarget::safetensors(
        source.clone(),
        config,
        SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap(),
        target_spec.limits,
    )
    .unwrap();
    let prepared = prepared_target.prediction(limits()).unwrap();
    let modules = <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut target);
    let embedding = modules
        .embeddings
        .forward(
            &NumericTensor::token_ids(&(0..23).map(|i| i % 31 + 1).collect::<Vec<_>>()),
            &context,
        )
        .unwrap();
    let residual = NumericTensor::new(
        [1, 23, 2, 2],
        (0..92)
            .map(|i| ((i * 13 % 41) as f32 - 20.) / 17.)
            .collect(),
    );
    let visible: Vec<_> = (0..23).map(|i| i != 0 && i != 5).collect();
    let ranges: Vec<_> = [0..3, 3..7]
        .into_iter()
        .chain((7..23).map(|i| i..i + 1))
        .collect();
    for depth in 0..2 {
        let mut full_state = prediction_state(prepared.spec());
        let mut full_unit = unit(&prepared, depth, &context);
        let mut full_shared = shared(&prepared, &context);
        let expected: Vec<_> = ranges
            .iter()
            .map(|range| {
                let embedded = embedding.axis_slice(1, range.start, range.end);
                let residual = residual.axis_slice(1, range.start, range.end);
                full_unit
                    .forward(
                        &mut full_shared,
                        PredictionInput {
                            embeddings: &embedded,
                            residual: &residual,
                            visible: Some(&visible[range.clone()]),
                            rotary: None,
                        },
                        full_state.layer(depth).unwrap(),
                        &mut ResidentExpertProvider,
                        &context,
                        &mut ComponentInstrumentation::disabled(),
                    )
                    .unwrap()
            })
            .collect();
        assert!(expected
            .iter()
            .any(|o| o.hidden.data.iter().any(|v| v.abs() > 1e-4)));
        for (tensor_ranks, expert_ranks) in [(2, 1), (1, 2), (2, 2)] {
            let before = source.source_diagnostics().unwrap().physical_reads;
            let partitions = (0..tensor_ranks)
                .map(|rank| prepared.tensor_partition(rank, tensor_ranks).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                before,
                source.source_diagnostics().unwrap().physical_reads,
                "partition preparation reads no payload"
            );
            let group = NumericParallelGroup::new(tensor_ranks);
            let world = Arc::new(NumericPartitionWorld::default());
            std::thread::scope(|scope| {
                let handles = partitions
                    .iter()
                    .enumerate()
                    .map(|(rank, partition)| {
                        let group = group.clone();
                        let world = world.clone();
                        let (embedding, residual, visible, ranges, expected) =
                            (&embedding, &residual, &visible, &ranges, &expected);
                        let mut head = modules.lm_head.as_ref().unwrap().clone();
                        let vocabulary = prepared.vocabulary().recipes();
                        let artifact = &source;
                        std::thread::Builder::new()
                            .stack_size(32 * 1024 * 1024)
                            .spawn_scoped(scope, move || {
                                let context = NumericContext {
                                    bind_checkpoint_values: true,
                                    ..Default::default()
                                };
                                let parallel = NumericParallelContext::new_partitioned(
                                    rank, rank, group, world,
                                );
                                let topology =
                                    eredu_core::ParallelTopology::new(1, 1, expert_ranks, 1)
                                        .unwrap();
                                let realization = partition
                                    .spec()
                                    .expert_realization(
                                        eredu_core::ParallelRankTopology::new(topology, 0).unwrap(),
                                    )
                                    .unwrap();
                                let mut local = partition
                                    .partition()
                                    .construct_unit_with_experts::<NumericBackend>(
                                        depth,
                                        &realization,
                                        &context,
                                    )
                                    .unwrap();
                                local.visit_parameters_mut(&mut RecipeBind {
                                    artifact,
                                    ctx: &context,
                                    ordinary: partition.unit(depth).unwrap(),
                                    experts: realization
                                        .local_global_group_indices()
                                        .iter()
                                        .map(|id| partition.expert(depth, *id).unwrap())
                                        .collect(),
                                });
                                let mut owners = Owners(
                                    (0..expert_ranks)
                                        .map(|owner| {
                                            let realization = partition
                                                .spec()
                                                .expert_realization(
                                                    eredu_core::ParallelRankTopology::new(
                                                        topology, owner,
                                                    )
                                                    .unwrap(),
                                                )
                                                .unwrap();
                                            let groups =
                                                realization.local_global_group_indices().to_vec();
                                            let mut bank = NumericBackend::grouped_gated_product(
                                                realization
                                                    .unit_spec("prediction", depth)
                                                    .unwrap()
                                                    .clone(),
                                                &context,
                                            )
                                            .unwrap();
                                            bank.visit_parameters_mut(&mut RecipeBind {
                                                artifact,
                                                ctx: &context,
                                                ordinary: partition.unit(depth).unwrap(),
                                                experts: groups
                                                    .iter()
                                                    .map(|id| partition.expert(depth, *id).unwrap())
                                                    .collect(),
                                            });
                                            (groups, bank)
                                        })
                                        .collect(),
                                );
                                let mut shared = PredictionShared::<NumericBackend>::new(
                                    partition.spec(),
                                    &context,
                                )
                                .unwrap();
                                shared.visit_parameters_mut(&mut RecipeBind {
                                    artifact,
                                    ctx: &context,
                                    ordinary: partition.shared(),
                                    experts: vec![],
                                });
                                assert_eq!(partition.vocabulary().recipes(), vocabulary);
                                let mut state = prediction_state(partition.spec());
                                if tensor_ranks > 1 {
                                    let token = embedding.axis_slice(1, 0, 1);
                                    let capture = residual.axis_slice(1, 0, 1);
                                    assert!(local
                                        .forward(
                                            &mut shared,
                                            PredictionInput {
                                                embeddings: &token,
                                                residual: &capture,
                                                visible: Some(&visible[..1]),
                                                rotary: None,
                                            },
                                            state.layer(depth).unwrap(),
                                            &mut owners,
                                            &context,
                                            &mut ComponentInstrumentation::disabled()
                                        )
                                        .is_err());
                                    let wrong = NumericParallelContext::new(
                                        0,
                                        NumericParallelGroup::new(1),
                                    );
                                    assert!(local
                                        .forward_parallel(
                                            &mut shared,
                                            PredictionInput {
                                                embeddings: &token,
                                                residual: &capture,
                                                visible: Some(&visible[..1]),
                                                rotary: None,
                                            },
                                            state.layer(depth).unwrap(),
                                            &mut owners,
                                            &wrong,
                                            &context,
                                            &mut ComponentInstrumentation::disabled()
                                        )
                                        .is_err());
                                    assert_eq!(
                                        RuntimeStateComponents::<NumericBackend>::position(
                                            state.layer(depth).unwrap()
                                        ),
                                        0
                                    );
                                }
                                for (range, expected) in ranges.iter().zip(expected) {
                                    let embedding = embedding.axis_slice(1, range.start, range.end);
                                    let residual = residual.axis_slice(1, range.start, range.end);
                                    let mut snapshot = state.clone();
                                    for actual_state in [&mut state, &mut snapshot] {
                                        let output = local
                                            .forward_parallel(
                                                &mut shared,
                                                PredictionInput {
                                                    embeddings: &embedding,
                                                    residual: &residual,
                                                    visible: Some(&visible[range.clone()]),
                                                    rotary: None,
                                                },
                                                actual_state.layer(depth).unwrap(),
                                                &mut owners,
                                                &parallel,
                                                &context,
                                                &mut ComponentInstrumentation::disabled(),
                                            )
                                            .unwrap();
                                        assert_tensor_close(
                                            &output.hidden,
                                            &expected.hidden,
                                            "prediction TP/EP collapse and restored state",
                                        );
                                        assert_tensor_close(
                                            &output.capture,
                                            &expected.capture,
                                            "prediction TP/EP residual capture",
                                        );
                                        assert_tensor_close(
                                            &head.forward(&output.hidden, &context).unwrap(),
                                            &head.forward(&expected.hidden, &context).unwrap(),
                                            "prediction shared vocabulary logits",
                                        );
                                    }
                                }
                                assert_eq!(
                                    RuntimeStateComponents::<NumericBackend>::position(
                                        state.layer(depth).unwrap()
                                    ),
                                    23
                                );
                            })
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        }
    }
}
