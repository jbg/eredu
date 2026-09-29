//! Ordinary retained distributed target/MTP preparation and cached prediction replay.
use super::executor::{Materializer, SourceBinding};
use super::*;
use eredu_architectures::{
    partitioned_execution::*, prediction_extension::*, prepared_execution::*,
    speculative_execution::SpeculativeActivationExecution,
};
use eredu_runtime::*;

type PredictionResult = (Vec<NumericTensor>, SpeculativeActivationExecution);

struct ProjectionBudget;
impl eredu_core::capture::CaptureReservation for ProjectionBudget {
    fn reserve(
        &mut self,
        _: eredu_core::capture::CaptureUsage,
    ) -> Result<Option<eredu_core::capture::CaptureSkipReason>, eredu_core::capture::CaptureError>
    {
        Ok(None)
    }
}

type Session<A, D> = ReplicatedTextSession<A, NumericBackend, NumericReplicatedMechanisms, D>;
trait Driver<A>:
    ReplicatedTextExecutionStrategy<
    A,
    NumericBackend,
    State,
    NumericReplicatedPolicy<A::Unit>,
    NumericReplicatedPolicy<A::Unit>,
>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
{
}
impl<A, D> Driver<A> for D
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    D: ReplicatedTextExecutionStrategy<
        A,
        NumericBackend,
        State,
        NumericReplicatedPolicy<A::Unit>,
        NumericReplicatedPolicy<A::Unit>,
    >,
{
}
struct Invoker<'a, A, D>
where
    A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>,
    D: Driver<A>,
{
    session: &'a mut Session<A, D>,
    context: &'a NumericContext,
}
impl<A, D> PredictionOperationInvoker<A, NumericBackend, State> for Invoker<'_, A, D>
where
    A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error>,
    D: Driver<A>,
{
    type Error = Error;
    fn invoke<O: PredictionTargetOperation<A, NumericBackend, State>>(
        &mut self,
        operation: O,
    ) -> Result<O::Output, Error> {
        self.session
            .apply_prediction_target_operation(operation, self.context)
            .map_err(Error::backend_source)
    }
    fn invalid(message: String) -> Error {
        Error::backend(message)
    }
}
struct Consumer<P> {
    prediction: P,
    provider: PartitionBankProviders<NumericBackend>,
    residency: LayerWeightResidency,
    selected: SelectedSpeculativeRealization,
}
impl<A, P> partitioned_adapter::SessionConsumer<A> for Consumer<P>
where
    A: ReplicatedTextArchitecture<NumericBackend, State, Error = Error> + 'static,
    P: PreparedRoutedPrediction<NumericBackend, A>,
{
    type Output = PredictionResult;
    fn finish<D>(
        self,
        mut session: Session<A, D>,
        context: NumericContext,
        _: eredu_core::cache::PromptCacheModelIdentity,
    ) -> Result<Self::Output, Error>
    where
        D: Driver<A> + 'static,
    {
        let mut source = SourceBinding {
            source: self.prediction.source().clone(),
            context: &context,
            residency: self.residency,
            roles: vec![],
        };
        let extension = self
            .prediction
            .materialize::<Materializer>(&mut source, |_, state| super::installed::realize(state))
            .map_err(|e| Error::backend(e.to_string()))?;
        let mut extension = P::with_provider::<Materializer, _>(extension, self.provider);
        let execution = extension
            .activation_execution(&self.selected)
            .expect("materialized prediction declares its actual observation hooks");
        let mut lane = extension.new_state();
        let prompt = NumericTensor::token_ids(&[
            3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        ]);
        let (logits, capture) = session
            .prefill_prediction_target(&prompt, None, &context)
            .map_err(Error::backend_source)?;
        assert_eq!(capture.shape, [1, 19, 2, 32]);
        let mut outputs = vec![logits.axis_slice(1, 18, 19)];
        let mut capture = capture.axis_slice(1, 18, 19);
        for id in 0..16 {
            let token = NumericTensor::token_ids(&[id]);
            let target = session
                .checkpoint(&context)
                .map_err(Error::backend_source)?;
            let saved_lane = extension
                .snapshot(&lane, &context)
                .map_err(Error::backend_source)?
                .unwrap();
            let draft = extension.logits::<State, _>(
                &mut Invoker {
                    session: &mut session,
                    context: &context,
                },
                &capture,
                &token,
                0,
                &mut lane,
            )?;
            assert!(draft.0.data.iter().all(|v| v.is_finite()));
            assert!(draft.0.data.iter().any(|v| v.abs() > 1e-5));
            super::super::session::assert_checkpoint(
                &target,
                &session
                    .checkpoint(&context)
                    .map_err(Error::backend_source)?,
            );
            lane = saved_lane;
            let replay = extension.logits::<State, _>(
                &mut Invoker {
                    session: &mut session,
                    context: &context,
                },
                &capture,
                &token,
                0,
                &mut lane,
            )?;
            assert_tensor_exact(
                &draft.0,
                &replay.0,
                "partition prediction lane rollback logits",
            );
            assert_tensor_exact(
                &draft.1,
                &replay.1,
                "partition prediction lane rollback capture",
            );
            outputs.push(draft.0);
            let next = session
                .decode_prediction_target(&token, &context)
                .map_err(Error::backend_source)?;
            outputs.push(next.0);
            capture = next.1;
        }
        Ok((outputs, execution))
    }
}
struct Binding {
    context: NumericContext,
    world: Arc<NumericPartitionWorld>,
    residency: LayerWeightResidency,
    selected: SelectedSpeculativeRealization,
}
impl RoutedPartitionedProductionVisitor<NumericBackend, State> for Binding {
    type Output = PredictionResult;
    type Error = Error;
    fn visit<A, G>(
        self,
        _: NumericPreparedRoutedPartition<A, G>,
        _: SharedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: TextPartitionArchitecture<NumericBackend, State>
            + ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + ParallelRoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        Err(Error::backend(
            "explicit partitioned prediction request reached target-only binding",
        ))
    }
    fn visit_prediction<A, G, P>(
        self,
        prepared: NumericPreparedRoutedPartition<A, G>,
        prediction: P,
        target_source: SharedCheckpointSource,
        provider_source: SharedCheckpointSource,
        _: PredictionBinding,
    ) -> Result<Self::Output, DenseDecoderPartitionedDispatchError<Error>>
    where
        A: TextPartitionArchitecture<NumericBackend, State>
            + ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + ParallelRoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
        P: PreparedRoutedPrediction<NumericBackend, A>,
    {
        let binding_context = self.context.clone();
        let banks = prediction.banks();
        construct_selected_partition_providers(
            prepared,
            |prepared, options| {
                let mut context = binding_context.clone();
                context.local_layout = Some(Arc::new(prepared.provider_layout().clone()));
                partition_banks::bind(
                    prepared.banks(),
                    options,
                    provider_source.as_ref(),
                    &context,
                )
            },
            (self, prediction),
            |(this, prediction), prepared, mut providers, _| {
                let predictor = providers
                    .split_off(&banks)?
                    .ok_or_else(|| Error::backend("partition prediction provider missing"))?;
                let target = CountingPartitionedRoutedProvider {
                    inner: providers,
                    calls: Arc::new(AtomicUsize::new(0)),
                };
                let consumer = Consumer {
                    prediction,
                    provider: predictor,
                    residency: this.residency,
                    selected: this.selected,
                };
                prepared.dispatch_execution(
                    (this.world, this.context, target, target_source, consumer),
                    |prepared, (world, context, provider, source, consumer)| {
                        bind_numeric_routed_direct(
                            prepared, world, context, provider, source, consumer,
                        )
                    },
                    |prepared, (world, context, provider, source, consumer)| {
                        bind_numeric_routed_pipeline_impl(
                            prepared, world, context, provider, None, source, consumer,
                        )
                    },
                )
            },
        )
        .map_err(|e| DenseDecoderPartitionedDispatchError::Visitor(Error::backend(e.to_string())))
    }
}

fn exact_layout(
    target: &PreparedTarget,
    rank: ParallelRankTopology,
) -> (ArchitectureParameterDescription, LocalModelLayout) {
    let context = NumericContext::default();
    let tensor = target
        .tensor_partition(rank.tensor_parallel_rank(), rank.tensor_parallel_size())
        .unwrap();
    let global =
        TargetModel::<NumericBackend>::new(tensor.source_bound_spec().clone(), &context).unwrap();
    let mut local = TargetModel::<NumericBackend>::new_tensor_parallel(
        tensor.source_bound_spec().clone(),
        tensor.partition().clone(),
        &context,
    )
    .unwrap();
    let tensor_parameters = local.parameter_description(&context).unwrap();
    local
        .set_expert_realization(
            &tensor
                .partition()
                .local_spec()
                .expert_realization(rank)
                .unwrap(),
        )
        .unwrap();
    let parameters = global.parameter_description(&context).unwrap();
    let layout = tensor
        .partition()
        .local_expert_layout(
            &parameters,
            &tensor_parameters,
            &local.parameter_description(&context).unwrap(),
            rank.expert_parallel_rank(),
            rank.expert_parallel_size(),
        )
        .unwrap();
    (
        parameters
            .with_partition_layout(rank, layout.clone())
            .unwrap(),
        layout,
    )
}
fn run(
    path: &std::path::Path,
    target: &PreparedTarget,
    companion: Option<&std::path::Path>,
    topology: ParallelTopology,
    residency: LayerWeightResidency,
) -> Vec<Vec<NumericTensor>> {
    let inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    std::thread::scope(|scope| {
        (0..topology.world_size())
            .map(|rank| {
                let inspection = inspection.clone();
                let world = world.clone();
                let residency = residency.clone();
                std::thread::Builder::new()
                    .name(format!("qwen4-mtp-partition-{rank}"))
                    .stack_size(32 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        let mut request = super::super::partition_selection::request(
                            topology.tensor(),
                            topology.pipeline(),
                            topology.expert(),
                            rank,
                        )
                        .with_weight_residency(WeightResidency::with_layers(residency.clone()))
                        .with_drafting(DraftingLoadRequest::embedded(1).unwrap());
                        if let Some(path) = companion {
                            request = request.with_prediction_source(path.to_path_buf());
                        }
                        let mechanisms = super::super::registry::Mechanisms {
                            prediction: true,
                            ..Default::default()
                        };
                        let selected = eredu_architectures::select_preparation(
                            &inspection,
                            &request,
                            &mechanisms,
                        )
                        .unwrap();
                        assert!(selected.prediction_realization().is_some());
                        let admission = eredu_core::ModelPreparationPlan::from_retained_admission(
                            inspection,
                            selected.admission(),
                        )
                        .unwrap();
                        let sources = eredu_architectures::prepared_sources::prepare_model_sources(
                            admission, selected,
                        )
                        .unwrap();
                        let prediction_selected = sources.selected().prediction_realization().unwrap().clone();
                        let (parameters, layout) = exact_layout(target, ParallelRankTopology::new(topology, rank).unwrap());
                        let tasks = sources.selected().text_realization().auxiliary_materialization_tasks().to_vec();
                        let discovery = sources.prepare_discovery(Default::default(), Default::default())
                            .bind_partition_parameters(Some(Arc::new(parameters))).unwrap();
                        assert!(discovery.prediction_placement().is_none(), "cold selection cannot publish materialized prediction placement");
                        let mut context = NumericContext::with_partition(layout, rank, world);
                        context.bind_checkpoint_values = true;
                        let communication =
                            partitioned_adapter::NumericPreparedCommunication::realize(
                                &sources, &context,
                            )
                            .unwrap();
                        let visitor = |resources: PreparedPartitionResources<
                            partitioned_adapter::NumericPreparedCommunication,
                        >| {
                            let communication = resources.into_communication();
                            Binding {
                                context: context.clone(),
                                world: communication.world,
                                residency: residency.clone(),
                                selected: prediction_selected.clone(),
                            }
                        };
                        let routes = PreparedExecutionRoutes::new()
                            .with_partitioned_routed(PartitionedRoutedRoute::<
                            NumericBackend,
                            State,
                            State,
                            State,
                            _,
                            _,
                            _,
                        >::new(
                            &context, &context, visitor, visitor, visitor,
                        ));
                        let (result, execution) = construct_prepared_execution(
                            sources,
                            Some(communication),
                            routes,
                            partitioned_adapter::PartitionAssembler {
                                context: &context,
                                executable: std::marker::PhantomData,
                            },
                        )
                        .unwrap();
                        let placement = discovery
                            .prediction_placement()
                            .expect("successful prediction materialization binds discovery");
                        assert_eq!(
                            placement.topology(),
                            ParallelRankTopology::new(topology, rank).unwrap()
                        );
                        assert!(!placement.modules().is_empty());
                        assert_eq!(placement.state().len(), 1);
                        let layouts = discovery.speculative_component_partition_layouts(&execution, topology.world_size()).unwrap().unwrap();
                        let activations = discovery.speculative_activations_with_partition_support(
                            &execution, &Default::default(), "numeric-partition-mtp", None, &layouts,
                            |_| eredu_core::ObservationSupportStatus::Supported,
                        ).unwrap();
                        for (path, axis, sharded) in [
                            ("mtp.layers.0.prediction.fusion", "hidden", false),
                            ("mtp.layers.0.prediction.capture", "hidden", false),
                            ("mtp.layers.0.prediction.attention.channels", "attention_channel", true),
                        ] {
                            let point = activations.captures.catalog.points.iter().find(|point| point.path == path)
                                .unwrap_or_else(|| panic!("retained prediction catalog is missing {path}"));
                            assert!(activations.bindings.iter().any(|binding| {
                                binding.node_id == point.node_id
                                    && binding.scope == eredu_core::speculative::SpeculativeCaptureScope::Prediction { depth: 0 }
                            }), "prediction invocation binding for {path}");
                            assert_eq!(layouts.capture_hook_members(path).unwrap(), (0..topology.world_size()).collect::<Vec<_>>(), "PP/EP replicas execute {path}");
                            for peer in 0..topology.world_size() {
                                let peer_rank = ParallelRankTopology::new(topology, peer).unwrap();
                                let point = layouts.rank(peer).unwrap().observation(path).unwrap();
                                assert_eq!(point.axis(), axis);
                                assert!(point.exports(), "prediction peer {peer} can export {path}");
                                let coordinates = point.coordinates().unwrap();
                                assert_eq!(coordinates.global_count(), 32);
                                let local_width = if sharded { 32 / topology.tensor() } else { 32 };
                                let start = if sharded { peer_rank.tensor_parallel_rank() * local_width } else { 0 };
                                assert_eq!(coordinates.local_count(), local_width);
                                assert_eq!(coordinates.contiguous_range(), Some(start..start + local_width), "exact prediction coordinates {path} peer {peer}");
                            }
                        }
                        let own = placement.local_layout().unwrap();
                        let own_query = own.tensor("mtp.layers.0.self_attn.q_proj.weight").unwrap();
                        let own_experts = own.tensor("mtp.layers.0.mlp.experts.down_proj").unwrap();
                        assert_eq!(
                            own_experts.local_shape()[0],
                            own_experts.global_shape()[0],
                            "prediction experts replicate over target EP"
                        );
                        assert!(own_experts.additional_placements().is_empty());
                        for peer in 0..topology.world_size() {
                            let peer_rank = ParallelRankTopology::new(topology, peer).unwrap();
                            let peer_layout = placement.layout_for_rank(peer).unwrap().unwrap();
                            let query = peer_layout
                                .tensor("mtp.layers.0.self_attn.q_proj.weight")
                                .unwrap();
                            let experts = peer_layout
                                .tensor("mtp.layers.0.mlp.experts.down_proj")
                                .unwrap();
                            assert_eq!(experts.local_shape()[0], own_experts.global_shape()[0]);
                            assert!(experts.additional_placements().is_empty());
                            assert_eq!(query.local_shape(), own_query.local_shape());
                            if peer_rank.tensor_parallel_rank()
                                == placement.topology().tensor_parallel_rank()
                            {
                                assert_eq!(
                                    &*peer_layout, own,
                                    "PP/EP replicas share exact prediction tensor geometry"
                                );
                            } else {
                                assert_ne!(
                                    query.placement(),
                                    own_query.placement(),
                                    "peer TP query coordinates must use authored ranges"
                                );
                            }
                            for (name, tensor) in [
                                ("mtp.layers.0.self_attn.q_proj.weight", query),
                                ("mtp.layers.0.mlp.experts.down_proj", experts),
                            ] {
                                let task = tasks.iter().find(|task| task.name() == name || task.aliases().iter().any(|alias| alias == name)).unwrap();
                                let expected = eredu_architectures::parameter_partition::derive_parameter_coordinates(task, tensor, peer, &mut ProjectionBudget).unwrap();
                                let actual = discovery.parameter_partition_layout_for_rank(name, peer, &mut ProjectionBudget).unwrap().unwrap();
                                assert_eq!(actual.coordinates(), expected.as_ref(), "prediction parameter coordinates {name} peer {peer}");
                            }
                            if topology.tensor() == 1 {
                                assert_eq!(query.placement(), &TensorPlacement::Replicated);
                            } else {
                                let width = query.global_shape()[0] / topology.tensor();
                                assert_eq!(
                                    query.placement(),
                                    &TensorPlacement::Range {
                                        axis: 0,
                                        start: peer_rank.tensor_parallel_rank() * width,
                                        end: (peer_rank.tensor_parallel_rank() + 1) * width,
                                    }
                                );
                            }
                        }
                        result
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    })
}
#[test]
fn qwen4_ordinary_partitioned_prediction_sources_residencies_cached_replay() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let expected = super::ordinary::run(
        &fixtures.safetensors_path,
        None,
        LayerWeightResidency::FullyResident,
    )
    .0;
    for (path, target, companion) in [
        (&fixtures.safetensors_path, &fixtures.safetensors, None),
        (
            &fixtures.gguf_path,
            &fixtures.gguf,
            Some(fixtures.safetensors_path.as_path()),
        ),
    ] {
        // The released-format miniature has six recurrent value heads, so its
        // exact recurrent projection cannot be split four ways. TP4/GQA remains
        // covered by the separately authored compatible indexed-attention fixtures.
        for (tp, pp, ep) in [
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 2, 1),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
        ] {
            for residency in [
                LayerWeightResidency::FullyResident,
                LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1)
                        .unwrap(),
                )),
                LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
                ),
            ] {
                eprintln!(
                    "ordinary MTP {} TP{tp} PP{pp} EP{ep} {residency:?}",
                    path.display()
                );
                for actual in run(
                    path,
                    target,
                    companion,
                    ParallelTopology::new(tp, pp, ep, 1).unwrap(),
                    residency,
                ) {
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(&expected) {
                        assert_tensor_close(
                            actual,
                            expected,
                            "ordinary partitioned target/prediction trajectory",
                        );
                    }
                }
            }
        }
    }
}
