use super::*;
use eredu_runtime::{ActivationObserver, PreparedLayeredObservationError as PreparedError};

type State = DeviceState<FakeBackend, FakeLayerState>;
struct PathsFixture {
    inner: GroupedFixture,
    declarations: Rc<Cell<usize>>,
    generated: Rc<Cell<usize>>,
    namespace: &'static str,
    owns_boundaries: bool,
    changed_graph: bool,
}
impl PathsFixture {
    fn new() -> Self {
        Self {
            inner: GroupedFixture {
                static_modules: FakeOperator,
                trace: Vec::new(),
            },
            declarations: Rc::new(Cell::new(0)),
            generated: Rc::new(Cell::new(0)),
            namespace: "group",
            owns_boundaries: false,
            changed_graph: false,
        }
    }
}
impl ArchitectureParameters<FakeBackend> for PathsFixture {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        self.inner.state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.inner.state_identity(state, topology)
    }
    fn parameter_description(
        &self,
        context: &(),
    ) -> Result<ArchitectureParameterDescription, Error> {
        self.inner.parameter_description(context)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<FakeBackend>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.inner.visit_static_parameters(visitor)
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<FakeBackend>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.inner.visit_static_parameters_mut(visitor)
    }
}
impl LayeredArchitecture<FakeBackend, State> for PathsFixture {
    type Input<'a> = Option<&'a PreparedModelInput<FakeTensor>>;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = GroupedForwardContext;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;
    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Error> {
        GroupedFixture::inference_input_shape(input)
    }
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        self.inner.group_transport(group)
    }
    fn primary_execution_group(&self) -> &str {
        self.inner.primary_execution_group()
    }
    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.inner.state_partition_plan(layout)
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        if self.changed_graph {
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("replacement")], "replacement")
                .map_err(Error::backend)
        } else {
            self.inner.execution_graph()
        }
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.inner.group_unit_count(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        self.declarations.set(self.declarations.get() + 1);
        Ok(format!("{}.{group}.unit.{index}", self.namespace))
    }
    fn observes_unit_boundaries(&self, _: usize, _: usize) -> bool {
        self.owns_boundaries
    }
    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Error> {
        self.inner.group_input_observation_path(group)
    }
    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Error> {
        self.inner.group_output_observation_path(group)
    }
    fn static_modules(&self) -> &FakeOperator {
        self.inner.static_modules()
    }
    fn static_modules_mut(&mut self) -> &mut FakeOperator {
        self.inner.static_modules_mut()
    }
    fn build_unit(&self, group: usize, index: usize, context: &()) -> Result<FakeUnit, Error> {
        self.inner.build_unit(group, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut State,
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, GroupedForwardContext>, Error> {
        self.inner.begin_forward(input, state, context)
    }
    fn begin_forward_observed<'a, O: ActivationObserver<FakeTensor, Error> + ?Sized>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut State,
        context: &(),
        observer: &mut O,
    ) -> Result<LayeredForwardState<FakeTensor, GroupedForwardContext>, Error> {
        let forward = self.begin_forward(input, state, context)?;
        observer.observe("embedding.internal", &forward.hidden)?;
        Ok(forward)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &FakeTensor,
        dependencies: &[&FakeTensor],
        state: &mut State,
        forward: &mut GroupedForwardContext,
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.inner
            .begin_execution_group(group, initial, dependencies, state, forward, context)
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut State,
        forward: &mut GroupedForwardContext,
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.inner
            .forward_unit(group, index, unit, hidden, state, forward, context)
    }
    fn forward_unit_observed<O: ActivationObserver<FakeTensor, Error> + ?Sized>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut State,
        forward: &mut GroupedForwardContext,
        context: &(),
        observer: &mut O,
    ) -> Result<FakeTensor, Error> {
        let input = if self.owns_boundaries {
            eredu_runtime::observe_and_intervene(
                observer,
                &eredu_core::UnitObservation::Input.path(&self.unit_path(group, index)?),
                hidden,
            )?
        } else {
            hidden.clone()
        };
        let mut output = self.forward_unit(group, index, unit, &input, state, forward, context)?;
        let source = eredu_core::capture::GeneratedCaptureSource {
            creation_bytes: 64,
            source_dtype: Some(TensorDtype::I32),
        };
        observer.observe_generated("unit.generated", &output, &source, &mut || {
            self.generated.set(self.generated.get() + 1);
            Ok(FakeTensor(output.0.iter().map(|value| value * 2).collect()))
        })?;
        if self.owns_boundaries {
            output = eredu_runtime::observe_and_intervene(
                observer,
                &eredu_core::UnitObservation::Output.path(&self.unit_path(group, index)?),
                &output,
            )?;
        }
        Ok(output)
    }
    fn select_readout_positions(
        &self,
        hidden: &FakeTensor,
        forward: &GroupedForwardContext,
        demand: eredu_core::OutputDemand,
        context: &(),
    ) -> Result<Option<FakeTensor>, Error> {
        self.inner
            .select_readout_positions(hidden, forward, demand, context)
    }
    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        state: &mut State,
        forward: &GroupedForwardContext,
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.inner.finish_forward(hidden, state, forward, context)
    }
    fn finish_forward_observed<O: ActivationObserver<FakeTensor, Error> + ?Sized>(
        &mut self,
        hidden: &FakeTensor,
        state: &mut State,
        forward: &GroupedForwardContext,
        context: &(),
        observer: &mut O,
    ) -> Result<FakeTensor, Error> {
        let output = self.finish_forward(hidden, state, forward, context)?;
        observer.observe("head.internal", &output)?;
        Ok(output)
    }
    fn retained_context_values<'a>(
        &'a self,
        _: &'a GroupedForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
fn layerwise() -> LayerwiseRuntime<PathsFixture, FakeBackend, State, RecordingPolicy> {
    LayerwiseRuntime::new(
        PathsFixture::new(),
        RecordingPolicy::new(
            [0, 10, 20, 21]
                .into_iter()
                .map(|marker| FakeUnit { marker })
                .collect(),
        ),
    )
}
#[derive(Default)]
struct Observer {
    values: Vec<(String, FakeTensor)>,
    pointers: Vec<usize>,
    generated: bool,
    replace_vision: bool,
    final_row: bool,
    fail: Option<&'static str>,
}
impl ActivationObserver<FakeTensor, Error> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        !self.final_row
    }
    fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Error> {
        self.values.push((path.to_owned(), value.clone()));
        self.pointers.push(path.as_ptr() as usize);
        if self.fail == Some(path) {
            return Err(Error::backend_source(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "prepared observer sentinel",
            )));
        }
        Ok(())
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &FakeTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<FakeTensor, Error>,
    ) -> Result<(), Error> {
        if self.generated {
            self.observe(path, &generate()?)?;
        }
        Ok(())
    }
    fn intervene(&mut self, path: &str, _: &FakeTensor) -> Result<Option<FakeTensor>, Error> {
        Ok(
            (self.replace_vision && path == eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH)
                .then(|| FakeTensor(vec![30, 0])),
        )
    }
}
fn has_io_source(error: &(dyn std::error::Error + 'static)) -> bool {
    if error.downcast_ref::<std::io::Error>().is_some() {
        return true;
    }
    error.source().is_some_and(has_io_source)
}

#[test]
fn prepared_resident_and_layerwise_reuse_paths_and_ordinary_observed_equations() {
    let input = prepared_composite_input(3);
    for replace in [false, true] {
        let mut resident =
            ResidentRuntime::<_, FakeBackend, State>::new(PathsFixture::new(), &()).unwrap();
        let prepared = resident.prepare_observation_paths().unwrap();
        let source = prepared.source().clone();
        let mut bounded = layerwise();
        let rebound = bounded.bind_observation_paths(&source).unwrap();
        assert!(rebound.source().same_storage(&source));
        let resident_calls = resident.architecture().declarations.get();
        let bounded_calls = bounded.architecture().declarations.get();
        let mut old = layerwise();
        let mut old_observer = Observer {
            replace_vision: replace,
            ..Default::default()
        };
        let legacy = old
            .forward_with_observer_and_context_with_readout(
                Some(&input),
                &mut fixture_state(),
                &(),
                &mut old_observer,
                eredu_core::OutputDemand::Sequence,
            )
            .unwrap();
        let mut resident_state = fixture_state();
        let mut bounded_state = fixture_state();
        for _ in 0..2 {
            let mut resident_observer = Observer {
                replace_vision: replace,
                ..Default::default()
            };
            let mut bounded_observer = Observer {
                replace_vision: replace,
                ..Default::default()
            };
            let left = resident
                .forward_with_prepared_observer_and_context_with_readout(
                    Some(&input),
                    &mut resident_state,
                    &(),
                    &mut resident_observer,
                    &prepared,
                    eredu_core::OutputDemand::Sequence,
                )
                .unwrap();
            let right = bounded
                .forward_with_prepared_observer_and_context_with_readout(
                    Some(&input),
                    &mut bounded_state,
                    &(),
                    &mut bounded_observer,
                    &rebound,
                    eredu_core::OutputDemand::Sequence,
                )
                .unwrap();
            assert_eq!(left.0, right.0);
            assert_eq!(left.0, legacy.0);
            assert_eq!(
                left.0,
                Some(FakeTensor(vec![if replace { 42 } else { 15 }, 20, 21]))
            );
            assert_eq!(resident_observer.values, bounded_observer.values);
            assert_eq!(resident_observer.values, old_observer.values);
            assert!(!resident_observer
                .values
                .iter()
                .any(|(path, _)| path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH));
            for group in 0..source.group_count() {
                for index in 0..source.unit_count(group).unwrap() {
                    for path in [
                        source.unit_paths(group, index).unwrap().0,
                        source.unit_paths(group, index).unwrap().1,
                        source.unit_effective_paths(group, index).unwrap().0,
                        source.unit_effective_paths(group, index).unwrap().1,
                    ] {
                        let slot = resident_observer
                            .values
                            .iter()
                            .position(|(candidate, _)| candidate == path)
                            .unwrap();
                        assert_eq!(resident_observer.pointers[slot], path.as_ptr() as usize);
                        assert_eq!(bounded_observer.pointers[slot], path.as_ptr() as usize);
                    }
                }
            }
        }
        assert_eq!(resident.architecture().declarations.get(), resident_calls);
        assert_eq!(bounded.architecture().declarations.get(), bounded_calls);
        assert_eq!(resident.architecture().generated.get(), 0);
        assert_eq!(bounded.architecture().generated.get(), 0);
        assert_eq!(
            resident_state
                .as_ref()
                .iter()
                .map(|layer| layer.0)
                .collect::<Vec<_>>(),
            bounded_state
                .as_ref()
                .iter()
                .map(|layer| layer.0)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn foreign_and_invalidated_bindings_reject_before_observer_or_state_work() {
    let mut first = layerwise();
    let prepared = first.prepare_observation_paths().unwrap();
    let source = prepared.source().clone();
    let mut other = layerwise();
    let mut state = fixture_state();
    let mut observer = Observer::default();
    assert!(matches!(
        other.forward_with_prepared_observer_and_context_with_readout(
            None,
            &mut state,
            &(),
            &mut observer,
            &prepared,
            eredu_core::OutputDemand::Sequence
        ),
        Err(PreparedError::BindingMismatch)
    ));
    assert!(observer.values.is_empty());
    assert!(other.policy().addresses.is_empty());
    first.architecture_mut().namespace = "changed";
    assert!(matches!(
        first.forward_with_prepared_observer_and_context_with_readout(
            None,
            &mut state,
            &(),
            &mut observer,
            &prepared,
            eredu_core::OutputDemand::Sequence
        ),
        Err(PreparedError::BindingMismatch)
    ));
    assert!(matches!(
        first.bind_observation_paths(&source),
        Err(PreparedError::SemanticMismatch)
    ));
    assert!(first.policy().addresses.is_empty());
    assert!(state.as_ref().iter().all(|layer| layer.0 == 0));
    first.architecture_mut().namespace = "group";
    let fresh = first.bind_observation_paths(&source).unwrap();
    first
        .forward_with_prepared_observer_and_context_with_readout(
            None,
            &mut state,
            &(),
            &mut observer,
            &fresh,
            eredu_core::OutputDemand::Sequence,
        )
        .unwrap();
    assert!(fresh.source().same_storage(&source));
    // Custom unit execution exposes &mut Architecture and invalidates the stamp.
    first
        .forward_with_unit_executor(
            None,
            &mut fixture_state(),
            &(),
            |architecture, group, index, unit, hidden, state, forward, context| {
                architecture.forward_unit(group, index, unit, hidden, state, forward, context)
            },
        )
        .unwrap();
    assert!(matches!(
        first.forward_with_prepared_observer_and_context_with_readout(
            None,
            &mut state,
            &(),
            &mut observer,
            &fresh,
            eredu_core::OutputDemand::Sequence
        ),
        Err(PreparedError::BindingMismatch)
    ));
}

#[test]
fn resident_cached_graph_cannot_be_rebound_after_semantic_mutation() {
    let mut runtime =
        ResidentRuntime::<_, FakeBackend, State>::new(PathsFixture::new(), &()).unwrap();
    let prepared = runtime.prepare_observation_paths().unwrap();
    runtime.architecture_mut().changed_graph = true;
    assert!(matches!(
        runtime.bind_observation_paths(prepared.source()),
        Err(PreparedError::SemanticMismatch)
    ));
    assert!(matches!(
        runtime.prepare_observation_paths(),
        Err(PreparedError::SemanticMismatch)
    ));
    assert!(runtime.architecture().inner.trace.is_empty());
}

#[test]
fn readout_demand_rejects_before_work_and_failure_returns_active_unit_with_cause() {
    let mut runtime = layerwise();
    let prepared = runtime.prepare_observation_paths().unwrap();
    let mut observer = Observer::default();
    let mut state = fixture_state();
    for demand in [
        eredu_core::OutputDemand::LastPosition,
        eredu_core::OutputDemand::StateOnly,
    ] {
        assert!(matches!(
            runtime.forward_with_prepared_observer_and_context_with_readout(
                None,
                &mut state,
                &(),
                &mut observer,
                &prepared,
                demand
            ),
            Err(PreparedError::ReadoutDemand)
        ));
    }
    assert!(runtime.policy().addresses.is_empty());
    assert!(observer.values.is_empty());
    assert!(state.as_ref().iter().all(|layer| layer.0 == 0));
    observer.fail = Some("group.2.unit.0.output");
    let error = runtime
        .forward_with_prepared_observer_and_context_with_readout(
            None,
            &mut state,
            &(),
            &mut observer,
            &prepared,
            eredu_core::OutputDemand::Sequence,
        )
        .unwrap_err();
    assert!(has_io_source(&error));
    assert_eq!(runtime.policy().aborts, 1);
    assert!(!runtime.policy().forward_active);
    assert!(runtime.policy().units.iter().all(Option::is_some));
    observer.fail = None;
    observer.final_row = true;
    let result = runtime
        .forward_with_prepared_observer_and_context_with_readout(
            None,
            &mut state,
            &(),
            &mut observer,
            &prepared,
            eredu_core::OutputDemand::LastPosition,
        )
        .unwrap();
    assert_eq!(result.0, Some(FakeTensor(vec![30, 20, 21])));
}

#[test]
fn selected_generated_hooks_and_owned_boundaries_preserve_exact_internal_dispatch() {
    for owned in [false, true] {
        let mut architecture = PathsFixture::new();
        architecture.owns_boundaries = owned;
        let mut runtime = ResidentRuntime::<_, FakeBackend, State>::new(architecture, &()).unwrap();
        let prepared = runtime.prepare_observation_paths().unwrap();
        let mut observer = Observer {
            generated: true,
            ..Default::default()
        };
        let output = runtime
            .forward_with_prepared_observer_and_context_with_readout(
                None,
                &mut fixture_state(),
                &(),
                &mut observer,
                &prepared,
                eredu_core::OutputDemand::Sequence,
            )
            .unwrap()
            .0
            .unwrap();
        assert_eq!(output, FakeTensor(vec![30, 20, 21]));
        assert_eq!(runtime.architecture().generated.get(), 4);
        assert_eq!(
            observer
                .values
                .iter()
                .filter(|(path, _)| path == "unit.generated")
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>(),
            [
                FakeTensor(vec![20, 0]),
                FakeTensor(vec![40, 20]),
                FakeTensor(vec![60, 40]),
                FakeTensor(vec![60, 40, 42])
            ]
        );
        for group in 0..3 {
            for index in 0..prepared.source().unit_count(group).unwrap() {
                for path in [
                    prepared.source().unit_paths(group, index).unwrap().0,
                    prepared.source().unit_paths(group, index).unwrap().1,
                ] {
                    assert_eq!(
                        observer
                            .values
                            .iter()
                            .filter(|(candidate, _)| candidate == path)
                            .count(),
                        1
                    );
                }
            }
        }
        assert_eq!(
            observer
                .values
                .iter()
                .filter(|(path, _)| path == "embedding.internal")
                .count(),
            1
        );
        assert_eq!(
            observer
                .values
                .iter()
                .filter(|(path, _)| path == "head.internal")
                .count(),
            1
        );
        let mut other = PathsFixture::new();
        other.owns_boundaries = owned;
        other.namespace = "other";
        let other = ResidentRuntime::<_, FakeBackend, State>::new(other, &()).unwrap();
        assert!(matches!(
            other.bind_observation_paths(prepared.source()),
            Err(PreparedError::SemanticMismatch)
        ));
    }
}

#[test]
fn outer_effective_hooks_report_the_consumed_replacement_in_both_traversals() {
    struct Replace(Observer);
    impl ActivationObserver<FakeTensor, Error> for Replace {
        fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Error> {
            self.0.observe(path, value)
        }
        fn intervene(&mut self, path: &str, _: &FakeTensor) -> Result<Option<FakeTensor>, Error> {
            Ok((path == "group.2.unit.0.input").then(|| FakeTensor(vec![23, -7])))
        }
    }
    let mut observations = Vec::new();
    for prepared in [false, true] {
        let mut runtime = layerwise();
        let paths = runtime.prepare_observation_paths().unwrap();
        let mut state = fixture_state();
        let mut observer = Replace(Observer::default());
        let output = if prepared {
            runtime
                .forward_with_prepared_observer_and_context_with_readout(
                    None,
                    &mut state,
                    &(),
                    &mut observer,
                    &paths,
                    eredu_core::OutputDemand::Sequence,
                )
                .unwrap()
                .0
        } else {
            runtime
                .forward_with_observer_and_context_with_readout(
                    None,
                    &mut state,
                    &(),
                    &mut observer,
                    eredu_core::OutputDemand::Sequence,
                )
                .unwrap()
                .0
        };
        assert_eq!(output, Some(FakeTensor(vec![23, -7, 20, 21])));
        let values = &observer.0.values;
        let original = values
            .iter()
            .position(|(p, _)| p == "group.2.unit.0.input")
            .unwrap();
        assert_eq!(
            values[original + 1],
            (
                "group.2.unit.0.input.effective".into(),
                FakeTensor(vec![23, -7])
            )
        );
        assert_ne!(values[original].1, values[original + 1].1);
        for group in 0..paths.source().group_count() {
            for index in 0..paths.source().unit_count(group).unwrap() {
                let pair = paths.source().unit_paths(group, index).unwrap();
                let effective = paths.source().unit_effective_paths(group, index).unwrap();
                for (path, after) in [(pair.0, effective.0), (pair.1, effective.1)] {
                    assert_eq!(values.iter().filter(|(p, _)| p == path).count(), 1);
                    assert_eq!(values.iter().filter(|(p, _)| p == after).count(), 1);
                    let i = values.iter().position(|(p, _)| p == path).unwrap();
                    assert_eq!(values[i + 1].0, after);
                    if path != "group.2.unit.0.input" {
                        assert_eq!(values[i].1, values[i + 1].1);
                    }
                }
            }
        }
        observations.push((
            output,
            observer.0.values,
            state.as_ref().iter().map(|s| s.0).collect::<Vec<_>>(),
        ));
    }
    assert_eq!(observations[0], observations[1]);
}

#[test]
fn failed_effective_input_hook_stops_before_unit_state_and_releases_its_loan() {
    for prepared in [false, true] {
        let mut runtime = layerwise();
        let paths = runtime.prepare_observation_paths().unwrap();
        let mut state = fixture_state();
        let mut observer = Observer {
            fail: Some("group.0.unit.0.input.effective"),
            ..Default::default()
        };
        let error: Box<dyn std::error::Error> = if prepared {
            Box::new(
                runtime
                    .forward_with_prepared_observer_and_context_with_readout(
                        None,
                        &mut state,
                        &(),
                        &mut observer,
                        &paths,
                        eredu_core::OutputDemand::Sequence,
                    )
                    .unwrap_err(),
            )
        } else {
            Box::new(
                runtime
                    .forward_with_observer_and_context_with_readout(
                        None,
                        &mut state,
                        &(),
                        &mut observer,
                        eredu_core::OutputDemand::Sequence,
                    )
                    .unwrap_err(),
            )
        };
        assert!(has_io_source(error.as_ref()));
        assert!(state.as_ref().iter().all(|layer| layer.0 == 0));
        assert_eq!(runtime.policy().aborts, 1);
        assert!(!runtime.policy().forward_active);
        assert!(runtime.policy().units.iter().all(Option::is_some));
        assert_eq!(
            observer.values.last().unwrap().0,
            "group.0.unit.0.input.effective"
        );
        assert!(!observer
            .values
            .iter()
            .any(|(path, _)| path == "group.0.unit.0.output"));
    }
}

impl ParallelLayeredArchitecture<FakeBackend, State> for PathsFixture {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut State,
        _: &(),
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, GroupedForwardContext>, Error> {
        self.begin_forward(input, state, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut State,
        forward: &mut GroupedForwardContext,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.forward_unit(group, index, unit, hidden, state, forward, context)
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &FakeTensor,
        state: &mut State,
        forward: &GroupedForwardContext,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.finish_forward(hidden, state, forward, context)
    }
}

#[test]
fn custom_unit_executor_adapters_emit_each_original_and_effective_boundary_once() {
    let mut results = Vec::new();
    for parallel in [false, true] {
        for owned in [false, true] {
            let mut runtime = layerwise();
            runtime.architecture_mut().owns_boundaries = owned;
            let paths = runtime.prepare_observation_paths().unwrap();
            let mut observer = Observer::default();
            let mut state = fixture_state();
            let result = if parallel {
                runtime
                    .forward_parallel_with_unit_executor_and_observer(
                        None,
                        &mut state,
                        &(),
                        &(),
                        |a, g, i, u, h, s, f, _, c| a.forward_unit(g, i, u, h, s, f, c),
                        &mut observer,
                    )
                    .unwrap()
            } else {
                runtime
                    .forward_with_unit_executor_and_observer_and_context(
                        None,
                        &mut state,
                        &(),
                        |a, g, i, u, h, s, f, c| a.forward_unit(g, i, u, h, s, f, c),
                        &mut observer,
                    )
                    .unwrap()
                    .0
            };
            for group in 0..paths.source().group_count() {
                for index in 0..paths.source().unit_count(group).unwrap() {
                    let original = paths.source().unit_paths(group, index).unwrap();
                    let effective = paths.source().unit_effective_paths(group, index).unwrap();
                    for (before, after) in [(original.0, effective.0), (original.1, effective.1)] {
                        let values = &observer.values;
                        assert_eq!(values.iter().filter(|(p, _)| p == before).count(), 1);
                        assert_eq!(values.iter().filter(|(p, _)| p == after).count(), 1);
                        let i = values.iter().position(|(p, _)| p == before).unwrap();
                        assert_eq!(values[i + 1].0, after);
                        assert_eq!(values[i].1, values[i + 1].1);
                        assert!(values[i].1 .0.iter().any(|v| *v != 0));
                    }
                }
            }
            results.push((
                result,
                state.as_ref().iter().map(|s| s.0).collect::<Vec<_>>(),
            ));
        }
    }
    assert!(results.iter().all(|r| r == &results[0]));
}

#[path = "observation_paths/prediction_operations.rs"]
mod prediction_operations;
