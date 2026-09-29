//! Common prediction executor: owners, completion, capture and lane rollback.
use super::*;
use eredu_architectures::prediction_extension::*;
use eredu_runtime::{ParameterProvider, RoutedExpertRequest};
use std::rc::Rc;

#[derive(Clone)]
pub(super) struct Lane(pub(super) State);
impl RuntimeState<NumericBackend> for Lane {
    type RetainedValues<'a> = <State as RuntimeState<NumericBackend>>::RetainedValues<'a>;
    fn layout(&self) -> &eredu_runtime::StateLayout {
        self.0.layout()
    }
    fn retained_values(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        self.0.retained_values(ordinal, address)
    }
}
impl PredictionModelState<NumericBackend> for Lane {
    type LayerState = NumericHybridLayerState;
    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        self.0.as_mut()
    }
}
pub(super) struct Module<T>(pub(super) T);
impl<T> AsMut<T> for Module<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
#[derive(Default)]
struct Probe {
    active: usize,
    completions: Vec<(usize, Vec<NumericTensor>)>,
    fail: bool,
}
thread_local! { static PROBE: RefCell<Probe> = RefCell::new(Probe::default()); }
pub(super) struct Materializer;
pub(super) struct SourceBinding<'a> {
    pub source: SharedCheckpointSource,
    pub context: &'a NumericContext,
    pub residency: eredu_runtime::LayerWeightResidency,
    pub roles: Vec<PredictionModuleRole>,
}
impl PredictionExtensionMaterializer<NumericBackend> for Materializer {
    type Error = String;
    type Module<T> = Module<T>;
    type PoolingState = NumericPoolingCache;
    type SequentialState = NumericCompressedCache;
    type ModelState = Lane;
    type Context<'a> = SourceBinding<'a>;
    fn invoke_module<U, O>(
        module: &mut Module<U>,
        context: &NumericContext,
        operation: impl FnOnce(&mut U) -> PredictionInvocation<NumericTensor, O>,
    ) -> Result<O, Error>
    where
        U: Parameterized<NumericTensor>,
    {
        PROBE.with(|p| p.borrow_mut().active += 1);
        let result = operation(module.as_mut());
        let completed = Self::complete_prediction_values(result.retained_values(), context);
        PROBE.with(|p| p.borrow_mut().active -= 1);
        completed.map_err(Error::backend_source)?;
        result.into_outcome()
    }
    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a NumericTensor>,
        _: &NumericContext,
    ) -> Result<(), eredu_core::BackendFailure> {
        PROBE.with(|p| {
            let mut p = p.borrow_mut();
            let active = p.active;
            p.completions
                .push((active, values.into_iter().cloned().collect()));
            if p.fail {
                Err(eredu_core::BackendFailure::from_error(
                    std::io::Error::other("completion failure"),
                ))
            } else {
                Ok(())
            }
        })
    }
    fn materialize_module<T>(
        binding: &mut SourceBinding<'_>,
        prepared: PreparedPredictionUnit<T>,
        layout: Option<&LocalModelLayout>,
    ) -> Result<Module<T>, String>
    where
        T: Parameterized<NumericTensor>,
    {
        assert_eq!(prepared.residency(), binding.residency);
        binding.roles.push(prepared.role());
        let (_, mut module, tasks) = prepared.into_parts();
        let mut context = binding.context.clone();
        if let Some(layout) = layout {
            context.local_layout = Some(Arc::new(layout.clone()));
            context.expanded_weight_layout = None;
        }
        payload::bind_module(&mut module, &tasks, binding.source.as_ref(), &context)
            .map_err(|e| e.to_string())?;
        Ok(Module(module))
    }
    fn pooling_state(
        _: &mut SourceBinding<'_>,
        _: usize,
        _: LayerCachePolicy,
    ) -> Result<NumericPoolingCache, String> {
        Err("unused pooling".into())
    }
    fn model_state(
        _: &mut SourceBinding<'_>,
        _: eredu_runtime::StateLayout,
    ) -> Result<Lane, String> {
        Err("fixture supplies exact declared state".into())
    }
    fn model_snapshot_estimate(
        state: &Lane,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::snapshots::state(&state.0)
    }
    fn model_snapshot(
        state: &Lane,
        _: &NumericContext,
    ) -> Result<Option<Lane>, eredu_core::BackendFailure> {
        super::snapshots::copied();
        if super::snapshots::FAIL_PREDICTION.with(Cell::get) {
            return Err(eredu_core::BackendFailure::from_error(
                std::io::Error::other("injected prediction copy failure"),
            ));
        }
        Ok(Some(state.clone()))
    }
    fn sequential_state() -> NumericCompressedCache {
        NumericCompressedCache::resident()
    }
}
struct Provider(Rc<Cell<usize>>);
impl ParameterProvider<NumericBackend> for Provider {
    type Error = Error;
    fn forward_grouped(
        &mut self,
        bank: &mut <NumericBackend as GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        ctx: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.0.set(self.0.get() + 1);
        PROBE.with(|p| {
            assert_eq!(
                p.borrow().active,
                2,
                "both prediction weight owners must be live"
            )
        });
        <ResidentExpertProvider as ParameterProvider<NumericBackend>>::forward_grouped(
            &mut ResidentExpertProvider,
            bank,
            request,
            ctx,
        )
    }
    fn forward_linear_routed(
        &mut self,
        bank: &mut <NumericBackend as GroupedNeuralBackend>::LinearGroups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        ctx: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        <ResidentExpertProvider as ParameterProvider<NumericBackend>>::forward_linear_routed(
            &mut ResidentExpertProvider,
            bank,
            request,
            ctx,
        )
    }
    fn forward_relu2_routed(
        &mut self,
        bank: &mut <NumericBackend as GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        ctx: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        <ResidentExpertProvider as ParameterProvider<NumericBackend>>::forward_relu2_routed(
            &mut ResidentExpertProvider,
            bank,
            request,
            ctx,
        )
    }
}
impl eredu_runtime::TensorParallelParameterProvider<NumericBackend> for Provider {
    fn forward_grouped_tensor_parallel(
        &mut self,
        bank: &mut <NumericBackend as GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        partitions: usize,
        context: &NumericContext,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<NumericTensor>, Error> {
        self.0.set(self.0.get() + 1);
        PROBE.with(|p| {
            assert_eq!(
                p.borrow().active,
                2,
                "both prediction weight owners must be live"
            )
        });
        <ResidentExpertProvider as eredu_runtime::TensorParallelParameterProvider<NumericBackend>>::forward_grouped_tensor_parallel(&mut ResidentExpertProvider, bank, request, partitions, context)
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        bank: &mut <NumericBackend as GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, NumericTensor>,
        partitions: usize,
        context: &NumericContext,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<NumericTensor>, Error> {
        <ResidentExpertProvider as eredu_runtime::TensorParallelParameterProvider<NumericBackend>>::forward_relu2_routed_tensor_parallel(&mut ResidentExpertProvider, bank, request, partitions, context)
    }
}
pub(super) struct Invoker<'a> {
    pub target: &'a mut TargetModel<NumericBackend>,
    pub state: &'a mut State,
    pub ctx: &'a NumericContext,
}
impl PredictionOperationInvoker<TargetModel<NumericBackend>, NumericBackend, State>
    for Invoker<'_>
{
    type Error = Error;
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Error>
    where
        O: eredu_runtime::PredictionTargetOperation<
            TargetModel<NumericBackend>,
            NumericBackend,
            State,
        >,
    {
        operation.apply(self.target, self.state, None, self.ctx)
    }
    fn invalid(message: String) -> Error {
        Error::backend(message)
    }
}
type Executor = MaterializedQwen4Prediction<NumericBackend, Materializer, Provider>;
type Target = TargetModel<NumericBackend>;
// Fix the target type in otherwise family-generic lifecycle calls.
fn new_lane(executor: &Executor) -> Lane {
    <Executor as MaterializedPredictionExecutor<Target, NumericBackend, Materializer>>::new_state(
        executor,
    )
}
fn initialized<T: Parameterized<NumericTensor>>(mut value: T) -> T {
    value.visit_parameters_mut(&mut Parameters::default());
    value
}
#[derive(Default)]
struct Observe {
    paths: Vec<String>,
    fail_capture: bool,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observe {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.paths.push(path.to_owned());
        if path.ends_with("prediction.capture") {
            assert_eq!(value.shape.len(), 4);
            if self.fail_capture {
                return Err(Error::backend("capture observer failed after state update"));
            }
        }
        Ok(())
    }
}

#[test]
fn qwen4_shared_prediction_executor_preserves_capture_completion_and_rollback() {
    PROBE.with(|p| *p.borrow_mut() = Probe::default());
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    for tied in [false, true] {
        let mut config = configuration();
        config.tied_embeddings = tied;
        config.prediction = Some(PredictionGeometry {
            layers: eredu_core::LayerSchedule::new(1, vec![LayerKind::Indexed]).unwrap(),
            rope_theta: 7777.,
        });
        let target_spec = specification_for(config.clone());
        let spec = spec(&config);
        let mut target = Target::new(bind_spec(target_spec.clone()), &ctx).unwrap();
        <Target as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut target)
            .visit_parameters_mut(&mut Parameters::default());
        let mut target_state = state(&target_spec);
        let untouched = target_state.clone();
        let charged = Rc::new(Cell::new(0));
        let mut executor = Executor::new(
            vec![Module(initialized(
                PredictionUnit::new(spec.units[0].clone(), &ctx).unwrap(),
            ))],
            Module(initialized(PredictionShared::new(&spec, &ctx).unwrap())),
            Lane(prediction_state(&spec)),
            Provider(charged.clone()),
        )
        .unwrap();
        let mut lane = new_lane(&executor);
        let mut replay_lane = new_lane(&executor);
        let mut expected_state = prediction_state(&spec);
        let mut expected_unit = initialized(
            PredictionUnit::<NumericBackend>::new(spec.units[0].clone(), &ctx).unwrap(),
        );
        let mut expected_shared =
            initialized(PredictionShared::<NumericBackend>::new(&spec, &ctx).unwrap());
        let tokens = NumericTensor::token_ids(&(0..35).map(|i| i % 31 + 1).collect::<Vec<_>>());
        let captures = NumericTensor::new(
            [1, 35, 2, 2],
            (0..140)
                .map(|i| ((i * 13 % 41) as f32 - 20.) / 17.)
                .collect(),
        );
        assert!(<Executor as MaterializedPredictionExecutor<
            Target,
            NumericBackend,
            Materializer,
        >>::logical_capture_shapes(&executor, &[1, 35, 2])
        .is_err());
        assert_eq!(<Executor as MaterializedPredictionExecutor<Target, NumericBackend, Materializer>>::logical_capture_shapes(&executor, &captures.shape).unwrap(), vec![vec![1,35,2,2]]);
        let mut observed = Observe::default();
        for range in [0..3, 3..8, 8..19]
            .into_iter()
            .chain((19..35).map(|i| i..i + 1))
        {
            let hidden = captures.axis_slice(1, range.start, range.end);
            let token = tokens.axis_slice(1, range.start, range.end);
            let embedding = target.prediction_embedding(&token, &ctx).unwrap();
            let expected = expected_unit
                .forward(
                    &mut expected_shared,
                    PredictionInput {
                        embeddings: &embedding,
                        residual: &hidden,
                        visible: None,
                        rotary: None,
                    },
                    expected_state.layer(0).unwrap(),
                    &mut ResidentExpertProvider,
                    &ctx,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap();
            let expected_logits = target
                .prediction_logits(
                    &expected.hidden,
                    &ctx,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap();
            let mut invoker = Invoker {
                target: &mut target,
                state: &mut target_state,
                ctx: &ctx,
            };
            if range.start == 0 {
                executor
                    .prefill_observed::<State, _>(
                        &mut invoker,
                        &hidden,
                        &hidden,
                        &token,
                        &mut replay_lane,
                        None,
                    )
                    .unwrap();
            } else {
                executor
                    .advance_observed::<State, _>(
                        &mut invoker,
                        &hidden,
                        &token,
                        &mut replay_lane,
                        None,
                    )
                    .unwrap();
            }
            let (logits, capture) = executor
                .logits_observed::<State, _>(
                    &mut invoker,
                    &hidden,
                    &token,
                    0,
                    &mut lane,
                    Some(&mut observed),
                )
                .unwrap();
            assert_tensor_exact(
                &capture,
                &expected.capture,
                "shared executor pre-collapse capture",
            );
            assert_tensor_exact(
                &logits,
                &expected_logits,
                "shared executor uses target vocabulary once",
            );
        }
        assert_state_exact(
            &lane.0,
            &replay_lane.0,
            1,
            "prefill and committed replay match proposals",
        );
        assert_state_exact(
            &lane.0,
            &expected_state,
            1,
            "executor retains the module state",
        );
        assert!(observed
            .paths
            .iter()
            .any(|p| p.ends_with("prediction.readout.linear")));
        let saved = lane.clone();
        let before = charged.get();
        let hidden = captures.axis_slice(1, 34, 35);
        let token = tokens.axis_slice(1, 34, 35);
        let mut invoker = Invoker {
            target: &mut target,
            state: &mut target_state,
            ctx: &ctx,
        };
        assert!(executor
            .logits_observed::<State, _>(
                &mut invoker,
                &hidden,
                &token,
                0,
                &mut lane,
                Some(&mut Observe {
                    fail_capture: true,
                    ..Default::default()
                })
            )
            .is_err());
        assert!(charged.get() > before);
        let charge = charged.get();
        assert!(lane.0.layer(0).unwrap().position() > saved.0.clone().layer(0).unwrap().position());
        PROBE.with(|p| {
            let p = p.borrow();
            let roots = &p.completions[p.completions.len() - 2..];
            assert_eq!((roots[0].0, roots[1].0), (2, 1));
            assert!(!roots[0].1.is_empty());
            for (a, b) in roots[0].1.iter().zip(&roots[1].1) {
                assert_tensor_exact(a, b, "shared completion dependencies");
            }
            assert_eq!(p.active, 0);
        });
        lane = saved.clone();
        assert_eq!(charged.get(), charge);
        let proposal = executor
            .logits::<State, _>(&mut invoker, &hidden, &token, 0, &mut lane)
            .unwrap();
        lane = saved.clone();
        let replay = executor
            .logits::<State, _>(&mut invoker, &hidden, &token, 0, &mut lane)
            .unwrap();
        assert_tensor_exact(&proposal.0, &replay.0, "proposal rollback logits");
        assert_tensor_exact(&proposal.1, &replay.1, "proposal rollback capture");
        // An out-of-schedule depth is rejected without dispatch or state mutation.
        let charged_before = charged.get();
        assert!(executor
            .logits::<State, _>(&mut invoker, &hidden, &token, 1, &mut lane)
            .is_err());
        assert_eq!(charged.get(), charged_before);
        // A native completion error prevents successful publication.
        PROBE.with(|p| p.borrow_mut().fail = true);
        assert!(executor
            .logits::<State, _>(&mut invoker, &hidden, &token, 0, &mut lane)
            .unwrap_err()
            .to_string()
            .contains("completion failure"));
        PROBE.with(|p| {
            let mut p = p.borrow_mut();
            p.fail = false;
            assert_eq!(p.active, 0);
        });
        assert_state_exact(
            &target_state,
            &untouched,
            target_spec.units.len(),
            "prediction leaves target state untouched",
        );
    }
}
