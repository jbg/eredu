use super::*;
use eredu_runtime::LayerRuntimeState;
use eredu_runtime::{
    ArchitectureStatePartitionPlan, ComponentAdapter, ComponentDefinition,
    ComponentDefinitionError, ComponentEquations, ComponentKernel, ComponentUnit,
    StatelessComponentState,
};

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error(transparent)]
    Definition(#[from] ComponentDefinitionError),
    #[error("injected equation failure")]
    Equation,
}

struct Kernel {
    fail: bool,
    pinned: Option<FakeModule>,
}
impl ComponentKernel<FakeBackend> for Kernel {
    type Input<'a> = (&'a FakeTensor, bool);
    type Unit = FakeUnit;
    type ForwardContext = bool;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Failure;

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Failure> {
        Ok(eredu_runtime::ModelStateIdentity::new(
            "component-fixture",
            "component",
            "fixed",
            1,
            state.global_layer_offset(),
            0,
            topology,
        )
        .unwrap())
    }
    fn build_unit(&self, address: ExecutionUnitAddress, _: &()) -> Result<FakeUnit, Failure> {
        Ok(FakeUnit {
            marker: (address.group() * 10 + address.index()) as i32,
        })
    }

    fn executes_group(&self, group: usize, extended: &bool) -> bool {
        group == 0 || *extended
    }

    fn retained_values<'a>(
        &'a self,
        _: &'a bool,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl eredu_runtime::ComponentPinnedResources<FakeBackend> for Kernel {
    type Modules = Option<FakeModule>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.pinned
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.pinned
    }
    fn visit_pinned<V: StaticParameterVisitor<FakeBackend>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(module) = &self.pinned {
            visitor.visit("readout", module)?;
        }
        Ok(())
    }
    fn visit_pinned_mut<V: StaticParameterVisitorMut<FakeBackend>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(module) = &mut self.pinned {
            visitor.visit_mut("readout", module)?;
        }
        Ok(())
    }
}
impl ComponentEquations<FakeBackend, StatelessComponentState> for Kernel {
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut StatelessComponentState,
        context: &(),
        observer: &mut O,
    ) -> Result<LayeredForwardState<FakeTensor, bool>, Failure>
    where
        O: eredu_runtime::ActivationObserver<FakeTensor, Failure> + ?Sized,
    {
        let mut forward = self.begin(input, state, context)?;
        forward.hidden =
            eredu_runtime::observe_and_intervene(observer, "fixture.input", &forward.hidden)?;
        Ok(forward)
    }
    fn execute_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut StatelessComponentState,
        forward: &mut bool,
        context: &(),
        observer: &mut O,
    ) -> Result<FakeTensor, Failure>
    where
        O: eredu_runtime::ActivationObserver<FakeTensor, Failure> + ?Sized,
    {
        let output = self.execute(address, unit, hidden, state, forward, context)?;
        eredu_runtime::observe_and_intervene(observer, "fixture.internal", &output)
    }
    fn finish_observed<O>(
        &mut self,
        hidden: &FakeTensor,
        state: &mut StatelessComponentState,
        forward: &bool,
        context: &(),
        observer: &mut O,
    ) -> Result<FakeTensor, Failure>
    where
        O: eredu_runtime::ActivationObserver<FakeTensor, Failure> + ?Sized,
    {
        let output = self.finish(hidden, state, forward, context)?;
        eredu_runtime::observe_and_intervene(observer, "fixture.readout", &output)
    }
    fn begin<'a>(
        &mut self,
        (input, extended): Self::Input<'a>,
        _: &mut StatelessComponentState,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, bool>, Failure> {
        Ok(LayeredForwardState {
            hidden: input.clone(),
            context: extended,
        })
    }
    fn begin_group(
        &mut self,
        _: usize,
        initial: &FakeTensor,
        dependencies: &[&FakeTensor],
        _: &mut StatelessComponentState,
        _: &mut bool,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        Ok(dependencies.first().copied().unwrap_or(initial).clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        _: &mut StatelessComponentState,
        _: &mut bool,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        if self.fail && address.group() == 1 {
            return Err(Failure::Equation);
        }
        let mut output = hidden.clone();
        output.0.push(unit.marker);
        Ok(output)
    }
    fn finish(
        &mut self,
        hidden: &FakeTensor,
        _: &mut StatelessComponentState,
        _: &bool,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        let mut output = hidden.clone();
        if let Some(module) = &self.pinned {
            output.0.extend_from_slice(&module.weight.0);
        }
        Ok(output)
    }
}

fn definition(pinned: bool) -> ComponentDefinition {
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root("first"),
            ExecutionGroupSpec::with_dependencies("second", ["first"]),
        ],
        "second",
    )
    .unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [1, 1]).unwrap();
    let groups = if pinned {
        vec![eredu_runtime::ParameterGroupSpec::new(
            "readout",
            eredu_runtime::ParameterRole::Replicated,
            [eredu_runtime::ParameterMemberSpec::new(
                "weight",
                [1],
                eredu_runtime::MemberSharding::Replicated,
            )],
        )
        .unwrap()]
    } else {
        vec![]
    };
    let owned = groups.iter().cloned().map(|group| {
        eredu_runtime::OwnedParameterGroupSpec::new(
            eredu_runtime::ParameterGroupOwner::static_any_of(["readout", "embedding"]),
            group,
        )
    });
    let parameters =
        ArchitectureParameterDescription::new(&graph, &layout, groups.clone(), owned).unwrap();
    let transport = ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::Projector,
        first_owner_static_roles: vec![],
        last_owner_static_roles: vec![],
        merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
        parallel_subgroup: None,
        request_optional: false,
    };
    let definition = ComponentDefinition::new(
        "fixture",
        parameters,
        vec![
            ComponentUnit {
                path: "first".into(),
                state: 0..0,
            },
            ComponentUnit {
                path: "second".into(),
                state: 0..0,
            },
        ],
        vec![transport.clone(), transport],
        "first",
        vec![],
        StatelessComponentState::state_layout().unwrap(),
        ArchitectureStatePartitionPlan::new([]),
    )
    .unwrap();
    definition
}
fn adapter(fail: bool) -> ComponentAdapter<ComponentDefinition, Kernel> {
    ComponentAdapter::new(definition(false), Kernel { fail, pinned: None }).unwrap()
}

#[test]
fn component_adapter_uses_ordinary_residency_and_operation_activity() {
    let mut resident = ResidentRuntime::new(adapter(false), &()).unwrap();
    let mut bounded = LayerwiseRuntime::new(adapter(false), RecordingPolicy::bounded(2));
    let input = FakeTensor(vec![7]);
    let mut state = StatelessComponentState::new().unwrap();
    for extended in [false, true, false] {
        let actual = bounded
            .forward((&input, extended), &mut state, &())
            .unwrap();
        let expected = resident
            .forward((&input, extended), &mut state, &())
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.0, if extended { vec![7, 0, 10] } else { vec![7, 0] });
    }
    assert_eq!(
        bounded.policy().addresses,
        [(0, 0, 0), (0, 0, 0), (1, 1, 0), (0, 0, 0)]
    );
}

#[test]
fn component_equation_failure_aborts_the_same_policy_and_allows_retry() {
    let mut runtime = LayerwiseRuntime::new(adapter(true), RecordingPolicy::bounded(2));
    let input = FakeTensor(vec![7]);
    let mut state = StatelessComponentState::new().unwrap();
    assert!(runtime.forward((&input, true), &mut state, &()).is_err());
    assert_eq!(runtime.policy().aborts, 1);
    assert!(!runtime.policy().forward_active);
    runtime.architecture_mut().kernel_mut().fail = false;
    assert_eq!(
        runtime.forward((&input, true), &mut state, &()).unwrap().0,
        [7, 0, 10]
    );
}

#[test]
fn component_observation_failure_aborts_and_intervention_reaches_downstream_equations() {
    struct Observer {
        fail_at: Option<&'static str>,
        seen: Vec<String>,
    }
    impl eredu_runtime::ActivationObserver<FakeTensor, Failure> for Observer {
        fn observe(&mut self, path: &str, _: &FakeTensor) -> Result<(), Failure> {
            self.seen.push(path.into());
            if self.fail_at == Some(path) {
                return Err(Failure::Equation);
            }
            Ok(())
        }
        fn intervene(
            &mut self,
            path: &str,
            value: &FakeTensor,
        ) -> Result<Option<FakeTensor>, Failure> {
            if path != "fixture.internal" {
                return Ok(None);
            }
            let mut value = value.clone();
            value.0.push(99);
            Ok(Some(value))
        }
    }
    let mut runtime = LayerwiseRuntime::new(adapter(false), RecordingPolicy::bounded(2));
    let input = FakeTensor(vec![7]);
    let mut state = StatelessComponentState::new().unwrap();
    for path in ["fixture.input", "fixture.internal", "fixture.readout"] {
        let mut observer = Observer {
            fail_at: Some(path),
            seen: vec![],
        };
        assert!(runtime
            .forward_with_observer((&input, true), &mut state, &(), &mut observer)
            .is_err());
        assert!(!runtime.policy().forward_active);
        assert_eq!(observer.seen.last().unwrap(), path);
    }
    // Input admission fails before a residency pass starts.
    assert_eq!(runtime.policy().aborts, 2);
    let mut observer = Observer {
        fail_at: None,
        seen: vec![],
    };
    let output = runtime
        .forward_with_observer((&input, true), &mut state, &(), &mut observer)
        .unwrap();
    assert_eq!(output.0, [7, 0, 99, 10, 99]);
    let internal: Vec<_> = observer
        .seen
        .iter()
        .filter(|p| p.starts_with("fixture."))
        .map(String::as_str)
        .collect();
    assert_eq!(
        internal,
        [
            "fixture.input",
            "fixture.internal",
            "fixture.internal",
            "fixture.readout"
        ]
    );
    assert_eq!(
        runtime.forward((&input, true), &mut state, &()).unwrap().0,
        [7, 0, 10]
    );
}

#[test]
fn component_pinned_resources_follow_canonical_roles_and_parameter_replacement() {
    struct Replace;
    impl StaticParameterVisitorMut<FakeBackend> for Replace {
        type Error = Infallible;
        fn visit_mut<M: Parameterized<FakeTensor>>(
            &mut self,
            role: &str,
            module: &mut M,
        ) -> Result<(), Infallible> {
            assert_eq!(role, "readout");
            struct Slot;
            impl<'a> ParameterVisitorMut<'a, FakeTensor> for Slot {
                fn visit_mut(&mut self, _: eredu_nn::ParameterMetadata, value: &'a mut FakeTensor) {
                    *value = FakeTensor(vec![42]);
                }
            }
            module.visit_parameters_mut(&mut Slot);
            Ok(())
        }
    }
    let declaration = definition(true);
    let kernel = || Kernel {
        fail: false,
        pinned: Some(FakeModule {
            weight: FakeTensor(vec![5]),
        }),
    };
    assert!(ComponentAdapter::new(declaration.clone(), kernel()).is_err());
    assert!(ComponentAdapter::with_pinned_resources::<FakeBackend>(
        declaration.clone(),
        Kernel {
            fail: false,
            pinned: None
        }
    )
    .is_err());
    let mut component =
        ComponentAdapter::with_pinned_resources::<FakeBackend>(declaration.clone(), kernel())
            .unwrap();
    component.visit_static_parameters_mut(&mut Replace).unwrap();
    let input = FakeTensor(vec![7]);
    let mut state = StatelessComponentState::new().unwrap();
    let mut bounded = LayerwiseRuntime::new(component, RecordingPolicy::bounded(2));
    assert_eq!(
        bounded.forward((&input, true), &mut state, &()).unwrap().0,
        [7, 0, 10, 42]
    );
    let partition =
        ArchitecturePartition::from_architecture::<FakeBackend, StatelessComponentState, _, _>(
            bounded.architecture(),
            [("first", 0..1)],
            PartitionOwnership::new(true, false, std::iter::empty::<String>()).unwrap(),
            (),
            NoAuxiliaryBoundarySchema::new(1),
            declaration.parameters(),
        )
        .unwrap();
    // A cut cannot silently retain a readout replica that it does not own.
    assert!(
        ComponentAdapter::with_pinned_resources::<FakeBackend>(declaration.clone(), kernel())
            .unwrap()
            .select_partition(&partition)
            .is_err()
    );
    ComponentAdapter::with_partition_resources::<FakeBackend, _, _>(
        declaration.clone(),
        Kernel {
            fail: false,
            pinned: None,
        },
        &partition,
    )
    .unwrap();
    // A tied output consumer still binds its replica through the canonical readout role.
    let tied =
        ArchitecturePartition::from_architecture::<FakeBackend, StatelessComponentState, _, _>(
            bounded.architecture(),
            [("second", 0..1)],
            PartitionOwnership::new(false, true, ["embedding"]).unwrap(),
            (),
            NoAuxiliaryBoundarySchema::new(1),
            declaration.parameters(),
        )
        .unwrap();
    ComponentAdapter::with_partition_resources::<FakeBackend, _, _>(declaration, kernel(), &tied)
        .unwrap();
}

// A mutable fixture uses global invocation addresses and independently sliced carry.
struct CarryKernel {
    offset: usize,
    has_ingress: bool,
}
type CarryState = DeviceState<FakeBackend, RetainedOrdinalLayerState>;
impl ComponentKernel<FakeBackend> for CarryKernel {
    type Input<'a> = &'a FakeTensor;
    type Unit = FakeUnit;
    type ForwardContext = ();
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Failure;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Failure> {
        Kernel {
            fail: false,
            pinned: None,
        }
        .state_identity(state, topology)
    }
    fn build_unit(&self, address: ExecutionUnitAddress, _: &()) -> Result<FakeUnit, Failure> {
        Ok(FakeUnit {
            marker: address.index() as i32,
        })
    }

    fn executes_group(&self, _: usize, _: &()) -> bool {
        true
    }

    fn retained_values<'a>(
        &'a self,
        _: &'a (),
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl ComponentEquations<FakeBackend, CarryState> for CarryKernel {
    fn begin<'a>(
        &mut self,
        input: &'a FakeTensor,
        _: &mut CarryState,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, ()>, Failure> {
        Ok(LayeredForwardState {
            hidden: input.clone(),
            context: (),
        })
    }
    fn begin_group(
        &mut self,
        _: usize,
        input: &FakeTensor,
        dependencies: &[&FakeTensor],
        _: &mut CarryState,
        _: &mut (),
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        Ok(dependencies.first().copied().unwrap_or(input).clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        _: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut CarryState,
        _: &mut (),
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        if self.has_ingress && address.group() == 0 {
            return Ok(hidden.clone());
        }
        let carry = state.layer(address.index() - self.offset).unwrap();
        carry.0 .0[0] += 1;
        let mut output = hidden.clone();
        output.0.push(carry.0 .0[0]);
        Ok(output)
    }
    fn finish(
        &mut self,
        hidden: &FakeTensor,
        _: &mut CarryState,
        _: &(),
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        Ok(hidden.clone())
    }
}

impl eredu_runtime::ParallelComponentKernel<FakeBackend, CarryState> for CarryKernel {
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut CarryState,
        _: &(),
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, ()>, Failure> {
        self.begin(input, state, context)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut CarryState,
        forward: &mut (),
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Failure> {
        self.execute(address, unit, hidden, state, forward, context)
    }
    fn finish_parallel(
        &mut self,
        hidden: &FakeTensor,
        state: &mut CarryState,
        forward: &(),
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Failure> {
        self.finish(hidden, state, forward, context)
    }
}
impl eredu_runtime::PartitionComponentKernel<FakeBackend, CarryState> for CarryKernel {
    type Boundary = NoAuxiliaryBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Failure> {
        Ok(NoAuxiliaryBoundarySchema::new(1))
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, FakeTensor, NoAuxiliaryBoundary>,
        _: Option<&FakeTensor>,
        state: &mut CarryState,
        selected: eredu_runtime::ComponentPartitionContext<'_, FakeBackend>,
    ) -> Result<LayeredForwardState<FakeTensor, ()>, Failure> {
        let hidden = match input {
            LayeredPartitionInput::Tokens(input) => input.clone(),
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin(&hidden, state, selected.native)
    }
    fn finish_partition(
        &mut self,
        hidden: &FakeTensor,
        _: &mut CarryState,
        _: &(),
        owns_output: bool,
        _: Option<&()>,
        _: &(),
    ) -> Result<LayeredPartitionOutput<FakeTensor, NoAuxiliaryBoundary>, Failure> {
        Ok(if owns_output {
            LayeredPartitionOutput::Final {
                output: hidden.clone(),
                retained: None,
            }
        } else {
            LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: NoAuxiliaryBoundary,
            }
        })
    }
}

fn carry_definition() -> ComponentDefinition {
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("carry")], "carry").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [3]).unwrap();
    let parameters = ArchitectureParameterDescription::new(&graph, &units, [], []).unwrap();
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap();
    let state = StateLayout::new(LayerSchedule::new(3, vec![policy; 3]).unwrap()).unwrap();
    ComponentDefinition::new(
        "carry",
        parameters,
        (0..3)
            .map(|index| ComponentUnit {
                path: format!("carry.{index}"),
                state: index..index + 1,
            })
            .collect(),
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Recurrent,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        "carry",
        vec![],
        state,
        ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..3),
        ]),
    )
    .unwrap()
}
fn carry_state(layout: StateLayout) -> CarryState {
    DeviceState::create(layout, |_, _| {
        Ok::<_, Infallible>(RetainedOrdinalLayerState(FakeTensor(vec![0])))
    })
    .unwrap()
}

#[test]
fn component_partitions_preserve_repeated_advancement_and_local_carry_retention() {
    let full = ComponentAdapter::new(
        carry_definition(),
        CarryKernel {
            offset: 0,
            has_ingress: false,
        },
    )
    .unwrap();
    let mut resident = ResidentRuntime::new(full, &()).unwrap();
    let mut fullstate = carry_state(carry_definition().state_layout().clone());
    let mut stages = [0..1, 1..3].map(|range| {
        let declaration = carry_definition();
        let full = ComponentAdapter::new(
            declaration.clone(),
            CarryKernel {
                offset: range.start,
                has_ingress: false,
            },
        )
        .unwrap();
        let partition = ArchitecturePartition::from_architecture::<FakeBackend, CarryState, _, _>(
            &full,
            [("carry", range.clone())],
            PartitionOwnership::new(
                range.start == 0,
                range.end == 3,
                std::iter::empty::<String>(),
            )
            .unwrap(),
            (),
            NoAuxiliaryBoundarySchema::new(1),
            declaration.parameters(),
        )
        .unwrap();
        let state = carry_state(partition.state().unwrap().layout().clone());
        let driver = LayeredPartitionDriver::new(&partition, 0, range).unwrap();
        let adapter = full.select_partition(&partition).unwrap();
        (adapter, driver, state)
    });
    for step in 1..=3 {
        let input = FakeTensor(vec![7]);
        let expected = resident.forward(&input, &mut fullstate, &()).unwrap();
        let mut hidden = input.clone();
        for (adapter, driver, state) in &mut stages {
            let ingress = if driver.owns_input() {
                LayeredPartitionInput::Tokens(&input)
            } else {
                LayeredPartitionInput::Hidden {
                    hidden,
                    auxiliary: NoAuxiliaryBoundary,
                }
            };
            let mut forward = driver
                .begin::<FakeBackend, _, _>(adapter, ingress, None, state, None, &())
                .unwrap();
            for index in driver.range() {
                let mut unit = adapter.build_unit(0, index, &()).unwrap();
                forward.hidden = adapter
                    .forward_unit(
                        0,
                        index,
                        &mut unit,
                        &forward.hidden,
                        state,
                        &mut forward.context,
                        &(),
                    )
                    .unwrap();
                let address = carry_definition()
                    .parameters()
                    .unit_layout()
                    .address(index)
                    .unwrap();
                let roots = adapter
                    .retained_state_ordinals(0, index, index)
                    .flat_map(|local| {
                        RuntimeState::<FakeBackend>::retained_values(state, local, address).unwrap()
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                assert_eq!(roots, [FakeTensor(vec![step])]);
            }
            hidden = match driver
                .finish::<FakeBackend, _, _>(
                    adapter,
                    &forward.hidden,
                    state,
                    &mut forward.context,
                    None,
                    &(),
                )
                .unwrap()
            {
                LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
                LayeredPartitionOutput::Final { output, .. } => output,
            };
        }
        assert_eq!(hidden, expected);
        assert_eq!(hidden, FakeTensor(vec![7, step, step, step]));
    }
    let (last, _, _) = &stages[1];
    assert!(last.build_unit(0, 0, &()).is_err());
}

impl eredu_runtime::PipelineComponentKernel<FakeBackend, CarryState> for CarryKernel {
    fn input_dimensions(&self, _: &Self::Input<'_>) -> Result<(i32, i32), Failure> {
        Ok((1, 1))
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(FakeTensor, NoAuxiliaryBoundary)>,
        state: &mut CarryState,
        selected: eredu_runtime::ComponentPartitionContext<'_, FakeBackend>,
        _: &mut O,
    ) -> Result<LayeredForwardState<FakeTensor, ()>, Failure>
    where
        O: eredu_runtime::ActivationObserver<FakeTensor, Failure> + ?Sized,
    {
        let input = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input),
        };
        eredu_runtime::PartitionComponentKernel::begin_partition(self, input, None, state, selected)
    }
    fn output_width(&self) -> i32 {
        1
    }
    fn tensor_vocabulary_sharded(&self) -> bool {
        false
    }
}

#[test]
fn component_stage_accepts_stateless_ingress_and_retains_later_group_carry() {
    let base = carry_definition();
    let source = ComponentAdapter::new(
        base.clone(),
        CarryKernel {
            offset: 0,
            has_ingress: false,
        },
    )
    .unwrap();
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root("ingress"),
            ExecutionGroupSpec::with_dependencies("carry", ["ingress"]),
        ],
        "carry",
    )
    .unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [1, 3]).unwrap();
    let parameters = ArchitectureParameterDescription::new(&graph, &layout, [], []).unwrap();
    let mut units = vec![ComponentUnit {
        path: "ingress".into(),
        state: 0..0,
    }];
    units.extend((0..3).map(|index| base.unit(0, index).unwrap().clone()));
    let definition = ComponentDefinition::new(
        "staged-carry",
        parameters,
        units,
        vec![source.group_transport(0), source.group_transport(0)],
        "carry",
        vec![],
        base.state_layout().clone(),
        ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(1, 0..3),
        ]),
    )
    .unwrap();
    let adapter = ComponentAdapter::new(
        definition.clone(),
        CarryKernel {
            offset: 0,
            has_ingress: true,
        },
    )
    .unwrap();
    let partition = ArchitecturePartition::from_architecture::<FakeBackend, CarryState, _, _>(
        &adapter,
        [("ingress", 0..1), ("carry", 0..3)],
        PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        definition.parameters(),
    )
    .unwrap();
    let mut state = carry_state(partition.state().unwrap().layout().clone());
    let mut runtime = adapter
        .into_partition_runtime(RecordingPolicy::bounded(4), &partition)
        .unwrap();
    let incorrect_entry = RuntimeState::<FakeBackend>::layout(&state).clone();
    let input = FakeTensor(vec![9]);
    assert!(runtime
        .forward_with_preparation(&mut state, None, &(), |adapter, state, _| {
            eredu_runtime::PipelinePartitionArchitecture::begin_pipeline_partition(
                adapter,
                &input,
                None,
                state,
                Some(&incorrect_entry),
                0,
                None,
                &(),
                &mut eredu_runtime::inspection::NoopObserver,
            )
        })
        .is_err());
    for step in 1..=3 {
        let input = FakeTensor(vec![9]);
        let output = runtime
            .forward_with_preparation(&mut state, None, &(), |adapter, state, expected| {
                eredu_runtime::PipelinePartitionArchitecture::begin_pipeline_partition(
                    adapter,
                    &input,
                    None,
                    state,
                    expected,
                    0,
                    None,
                    &(),
                    &mut eredu_runtime::inspection::NoopObserver,
                )
            })
            .unwrap();
        let LayeredPartitionOutput::Final { output, .. } = output else {
            panic!("stage owns output")
        };
        assert_eq!(output, FakeTensor(vec![9, step, step, step]));
    }
}
