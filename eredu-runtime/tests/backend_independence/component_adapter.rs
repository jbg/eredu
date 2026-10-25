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
    Neural(#[from] eredu_nn::Error),
    #[error(transparent)]
    Preparation(#[from] eredu_runtime::ComponentPreparationError),
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

impl eredu_runtime::ParallelComponentKernel<FakeBackend, StatelessComponentState> for Kernel {
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut StatelessComponentState,
        _: &(),
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, bool>, Failure> {
        self.begin(input, state, context)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut StatelessComponentState,
        forward: &mut bool,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Failure> {
        self.execute(address, unit, hidden, state, forward, context)
    }
    fn finish_parallel(
        &mut self,
        hidden: &FakeTensor,
        state: &mut StatelessComponentState,
        forward: &bool,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Failure> {
        self.finish(hidden, state, forward, context)
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
    let prototype = definition(false);
    let graph = prototype.parameters().graph();
    let shared_layout = ExecutionUnitLayout::new(graph, [1, 2]).unwrap();
    let weights = eredu_runtime::ParameterGroupSpec::new(
        "shared",
        eredu_runtime::ParameterRole::Replicated,
        [eredu_runtime::ParameterMemberSpec::new(
            "shared.weight",
            [1, 64],
            eredu_runtime::MemberSharding::Replicated,
        )],
    )
    .unwrap();
    let parameters = ArchitectureParameterDescription::new(
        graph,
        &shared_layout,
        [weights.clone()],
        [eredu_runtime::OwnedParameterGroupSpec::new(
            eredu_runtime::ParameterGroupOwner::execution_unit_consumers(
                shared_layout.group_id(1).unwrap().clone(),
                1,
                [0, 1],
            ),
            weights,
        )],
    )
    .unwrap();
    let shared_definition = ComponentDefinition::new(
        "shared-fixture",
        parameters,
        ["first", "second.0", "second.1"]
            .map(|path| ComponentUnit {
                path: path.into(),
                state: 0..0,
            })
            .to_vec(),
        vec![
            prototype.group_transport(0).unwrap().clone(),
            prototype.group_transport(1).unwrap().clone(),
        ],
        "first",
        vec![],
        prototype.state_layout().clone(),
        ArchitectureStatePartitionPlan::new([]),
    )
    .unwrap();
    let uses = [0, 1].map(|index| {
        (
            format!("second.{index}.weight"),
            eredu_runtime::ComponentParameterUse {
                invocation: shared_layout
                    .address(shared_layout.ordinal(1, index).unwrap())
                    .unwrap(),
                target: "shared.weight".into(),
            },
        )
    });
    let mut wrong_consumer = uses[0].clone();
    wrong_consumer.1.invocation = shared_layout.address(0).unwrap();
    assert!(shared_definition
        .clone()
        .with_parameter_uses([wrong_consumer])
        .is_err());
    let mut dangling = uses[0].clone();
    dangling.1.target = "absent.weight".into();
    assert!(shared_definition
        .clone()
        .with_parameter_uses([dangling])
        .is_err());
    assert!(shared_definition
        .clone()
        .with_parameter_uses([uses[0].clone(), uses[0].clone()])
        .is_err());
    let shared_definition = shared_definition
        .with_parameter_uses(uses)
        .unwrap()
        .map_unit_observation_boundaries(|address, original| {
            if address.group() != 1 {
                return original.clone();
            }
            if address.index() == 0 {
                eredu_runtime::UnitObservationBoundaries::new(Some("shared.input".into()), None)
            } else {
                eredu_runtime::UnitObservationBoundaries::new(None, Some("shared.output".into()))
            }
        })
        .unwrap();
    let source_group = shared_definition.parameters().groups()[0].clone();
    let selected_group = eredu_runtime::expand_linear_format_parameter_groups(
        vec![source_group.group().clone()],
        |_| {
            Ok(Some(
                eredu_nn::LinearFormatSpec::scaled(
                    eredu_checkpoint::LinearFormat::MxFp4,
                    eredu_nn::ParameterSpec::trainable("shared.scale").unwrap(),
                )
                .unwrap(),
            ))
        },
    )
    .unwrap()
    .remove(0);
    let projected_parameters = ArchitectureParameterDescription::new(
        graph,
        &shared_layout,
        [selected_group.clone()],
        [eredu_runtime::OwnedParameterGroupSpec::new(
            source_group.owner().clone(),
            selected_group,
        )],
    )
    .unwrap();
    let projected_uses = [0, 1]
        .into_iter()
        .flat_map(|index| {
            let invocation = shared_layout
                .address(shared_layout.ordinal(1, index).unwrap())
                .unwrap();
            ["weight", "scale"].map(move |suffix| {
                (
                    format!("second.{index}.{suffix}"),
                    eredu_runtime::ComponentParameterUse {
                        invocation,
                        target: format!("shared.{suffix}"),
                    },
                )
            })
        })
        .collect::<Vec<_>>();
    let renamed = eredu_runtime::ParameterGroupSpec::new(
        "shared",
        eredu_runtime::ParameterRole::Replicated,
        [eredu_runtime::ParameterMemberSpec::new(
            "neighbor.weight",
            [1, 64],
            eredu_runtime::MemberSharding::Replicated,
        )],
    )
    .unwrap();
    let renamed = ArchitectureParameterDescription::new(
        graph,
        &shared_layout,
        [renamed.clone()],
        [eredu_runtime::OwnedParameterGroupSpec::new(
            source_group.owner().clone(),
            renamed,
        )],
    )
    .unwrap();
    assert!(shared_definition
        .clone()
        .with_parameters(renamed, projected_uses.clone())
        .is_err());
    let packed = projected_parameters.groups()[0].group();
    let detached = [
        eredu_runtime::OwnedParameterGroupSpec::new(
            source_group.owner().clone(),
            eredu_runtime::ParameterGroupSpec::new(
                "shared",
                packed.role(),
                packed
                    .members()
                    .iter()
                    .filter(|member| member.linear_companion().is_none())
                    .cloned(),
            )
            .unwrap(),
        ),
        eredu_runtime::OwnedParameterGroupSpec::new(
            eredu_runtime::ParameterGroupOwner::static_role("detached"),
            eredu_runtime::ParameterGroupSpec::new(
                "detached",
                packed.role(),
                packed
                    .members()
                    .iter()
                    .filter(|member| member.linear_companion().is_some())
                    .cloned(),
            )
            .unwrap(),
        ),
    ];
    let detached = ArchitectureParameterDescription::new(
        graph,
        &shared_layout,
        detached.iter().map(|group| group.group().clone()),
        detached.clone(),
    )
    .unwrap();
    assert!(shared_definition
        .clone()
        .with_parameters(detached, projected_uses.clone())
        .is_err());
    let shared_definition = shared_definition
        .with_parameters(projected_parameters, projected_uses)
        .unwrap();
    struct SharedObservation(Vec<String>);
    impl eredu_runtime::ActivationObserver<FakeTensor, Failure> for SharedObservation {
        fn observe(&mut self, path: &str, _: &FakeTensor) -> Result<(), Failure> {
            self.0.push(path.into());
            Ok(())
        }
    }
    let mut observed = LayerwiseRuntime::new(
        ComponentAdapter::new(
            shared_definition.clone(),
            Kernel {
                fail: false,
                pinned: None,
            },
        )
        .unwrap(),
        RecordingPolicy::bounded(shared_layout.len()),
    );
    let mut observer = SharedObservation(vec![]);
    assert_eq!(
        observed
            .forward_with_observer((&input, true), &mut state, &(), &mut observer)
            .unwrap()
            .0,
        [7, 0, 11, 11]
    );
    assert_eq!(
        observer
            .0
            .iter()
            .filter(|path| path.starts_with("shared."))
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["shared.input", "shared.output"]
    );
    let mut shared = ResidentRuntime::new(
        ComponentAdapter::new(
            shared_definition,
            Kernel {
                fail: false,
                pinned: None,
            },
        )
        .unwrap(),
        &(),
    )
    .unwrap();
    for extended in [true, false, true] {
        let output = shared.forward((&input, extended), &mut state, &()).unwrap();
        assert_eq!(
            output.0,
            if extended {
                vec![7, 0, 11, 11]
            } else {
                vec![7, 0]
            }
        );
    }
    assert_eq!(shared.units().iter().map(Vec::len).sum::<usize>(), 2);
    shared.units_mut()[1][0].marker = 29;
    assert_eq!(
        shared.forward((&input, true), &mut state, &()).unwrap().0,
        [7, 0, 29, 29]
    );
    let declaration = shared.architecture().definition().clone();
    let partition = eredu_runtime::ArchitecturePartition::from_description(
        declaration.parameters(),
        [("second", 0..1)],
        eredu_runtime::PartitionOwnership::new(false, true, std::iter::empty::<String>()).unwrap(),
        declaration.state_layout(),
        declaration.state_partition_plan(),
        (),
        eredu_runtime::NoAuxiliaryBoundarySchema::new(2),
    )
    .unwrap();
    let other_storage = declaration
        .parameters()
        .invocations()
        .storage_location(0)
        .unwrap();
    let selected = ComponentAdapter::new(
        declaration,
        Kernel {
            fail: false,
            pinned: None,
        },
    )
    .unwrap()
    .select_partition(&partition)
    .unwrap();
    assert!(selected.build_storage_unit(other_storage, &()).is_err());
    assert_eq!(
        selected
            .build_storage_unit(partition.invocations().storage_location(0).unwrap(), &())
            .unwrap()
            .marker,
        11
    );
    assert_eq!(selected.build_unit(1, 0, &()).unwrap().marker, 11);
    assert!(selected.build_unit(1, 1, &()).is_err());
    // Topology admission and low-level construction retain sparse global IDs.
    let prototype = definition(false);
    let graph = prototype.parameters().graph();
    let complete = ExecutionUnitLayout::new(graph, [4, 8]).unwrap();
    let addresses = [complete.address(3).unwrap(), complete.address(11).unwrap()];
    let sparse = complete.select(addresses).unwrap();
    let parameters = ArchitectureParameterDescription::new(graph, &sparse, [], []).unwrap();
    let declaration = ComponentDefinition::new(
        "sparse-fixture",
        parameters,
        vec![
            prototype.unit(0, 0).unwrap().clone(),
            prototype.unit(1, 0).unwrap().clone(),
        ],
        vec![
            prototype.group_transport(0).unwrap().clone(),
            prototype.group_transport(1).unwrap().clone(),
        ],
        "first",
        vec![],
        prototype.state_layout().clone(),
        ArchitectureStatePartitionPlan::new([]),
    )
    .unwrap();
    let sparse_parameters = declaration.parameters().clone();
    let mut component = ComponentAdapter::new(
        declaration.clone(),
        Kernel {
            fail: false,
            pinned: None,
        },
    )
    .unwrap();
    sparse_parameters
        .validate_architecture::<FakeBackend, StatelessComponentState, _>(&component)
        .unwrap();
    let prepare = || {
        eredu_runtime::PreparedComponent::prepare::<
            Failure,
            eredu_checkpoint::recipe::DerivedWeightRecipe,
        >(
            "sparse-fixture",
            sparse_parameters.clone(),
            Arc::new(counting_source().0),
            |_| unreachable!("fixture has no parameters"),
        )
        .unwrap()
    };
    prepare()
        .with_storage_preferences::<FakeBackend, StatelessComponentState, _>(&component, &())
        .unwrap();
    assert!(matches!(
        prepare().with_storage_preferences::<FakeBackend, StatelessComponentState, _>(
            &adapter(false),
            &()
        ),
        Err(Failure::Preparation(
            eredu_runtime::ComponentPreparationError::ArchitectureMismatch(_)
        ))
    ));
    let selected =
        <_ as LayeredArchitecture<FakeBackend, StatelessComponentState>>::execution_unit_layout(
            &component,
        )
        .unwrap()
        .select(addresses)
        .unwrap();
    let mut hidden = input.clone();
    let mut forward = true;
    for ordinal in 0..selected.len() {
        let address = selected.address(ordinal).unwrap();
        let mut unit = component
            .build_unit(address.group(), address.index(), &())
            .unwrap();
        hidden = component
            .forward_unit(
                address.group(),
                address.index(),
                &mut unit,
                &hidden,
                &mut state,
                &mut forward,
                &(),
            )
            .unwrap();
    }
    assert_eq!(hidden.0, [7, 3, 17]);
    assert!(component.build_unit(0, 0, &()).is_err());
    let adopted_units = addresses
        .into_iter()
        .map(|address| {
            vec![component
                .build_unit(address.group(), address.index(), &())
                .unwrap()]
        })
        .collect();
    let mut adopted = ResidentRuntime::from_units(component, adopted_units).unwrap();
    let mut resident = ResidentRuntime::new(
        ComponentAdapter::new(
            declaration.clone(),
            Kernel {
                fail: false,
                pinned: None,
            },
        )
        .unwrap(),
        &(),
    )
    .unwrap();
    let mut sparse_runtime = LayerwiseRuntime::new(
        ComponentAdapter::new(
            declaration,
            Kernel {
                fail: false,
                pinned: None,
            },
        )
        .unwrap(),
        RecordingPolicy::bounded(2),
    );
    for _ in 0..2 {
        assert_eq!(
            resident.forward((&input, true), &mut state, &()).unwrap(),
            hidden
        );
        assert_eq!(
            adopted.forward((&input, true), &mut state, &()).unwrap(),
            hidden
        );
        assert_eq!(
            sparse_runtime
                .forward((&input, true), &mut state, &())
                .unwrap(),
            hidden
        );
        assert_eq!(
            sparse_runtime
                .forward_parallel((&input, true), &mut state, &(), &())
                .unwrap(),
            hidden
        );
    }
    struct Paths(Vec<String>);
    impl eredu_runtime::ActivationObserver<FakeTensor, Failure> for Paths {
        fn observe(&mut self, path: &str, _: &FakeTensor) -> Result<(), Failure> {
            self.0.push(path.to_owned());
            Ok(())
        }
    }
    let mut paths = Paths(vec![]);
    assert_eq!(
        sparse_runtime
            .forward_with_observer((&input, true), &mut state, &(), &mut paths)
            .unwrap(),
        hidden
    );
    for path in [
        "first.input",
        "first.output",
        "second.input",
        "second.output",
    ] {
        assert_eq!(
            paths.0.iter().filter(|seen| seen.as_str() == path).count(),
            1
        );
    }
    assert!(sparse_runtime
        .policy()
        .addresses
        .chunks_exact(2)
        .all(|pair| pair == [(0, 0, 3), (1, 1, 7)]));
}

#[test]
fn component_logical_observation_boundaries_span_independently_leased_operations() {
    use eredu_runtime::UnitObservationBoundaries;
    struct Observer {
        seen: Vec<(String, FakeTensor)>,
        fail: Option<&'static str>,
    }
    impl eredu_runtime::ActivationObserver<FakeTensor, Failure> for Observer {
        fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Failure> {
            self.seen.push((path.into(), value.clone()));
            if self.fail == Some(path) {
                Err(Failure::Equation)
            } else {
                Ok(())
            }
        }
        fn intervene(&mut self, path: &str, _: &FakeTensor) -> Result<Option<FakeTensor>, Failure> {
            Ok(match path {
                "logical.input" => Some(FakeTensor(vec![9])),
                "logical.output" => Some(FakeTensor(vec![99])),
                _ => None,
            })
        }
    }
    let split = |kernel_owned| {
        definition(false)
            .map_unit_observation_boundaries(|address, _| {
                let paths = if address.group() == 0 {
                    UnitObservationBoundaries::new(Some("logical.input".into()), None)
                } else {
                    UnitObservationBoundaries::new(None, Some("logical.output".into()))
                };
                if kernel_owned {
                    paths.with_kernel_ownership()
                } else {
                    paths
                }
            })
            .and_then(|definition| {
                let state = definition.state_layout().clone();
                definition.with_state_layout(state)
            })
            .unwrap()
    };
    assert!(definition(false)
        .map_unit_observation_boundaries(|_, _| {
            UnitObservationBoundaries::new(Some("ambiguous".into()), None)
        })
        .is_err());
    assert!(definition(false)
        .map_unit_observation_boundaries(|_, _| {
            UnitObservationBoundaries::new(Some(" ".into()), None)
        })
        .is_err());
    let check = |seen: &[(String, FakeTensor)]| {
        let logical = seen
            .iter()
            .filter(|(path, _)| path.starts_with("logical."))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            logical,
            [
                ("logical.input".into(), FakeTensor(vec![1])),
                ("logical.output".into(), FakeTensor(vec![9, 0, 10]))
            ]
        );
        assert!(!seen.iter().any(|(path, _)| matches!(
            path.as_str(),
            "first.input" | "first.output" | "second.input" | "second.output"
        )));
    };
    let component = |kernel_owned| {
        ComponentAdapter::new(
            split(kernel_owned),
            Kernel {
                fail: false,
                pinned: None,
            },
        )
        .unwrap()
    };
    let architecture = component(false);
    let units = [(0, 0), (1, 0)]
        .into_iter()
        .map(|(group, index)| architecture.build_unit(group, index, &()).unwrap())
        .collect();
    let mut resident =
        LayerwiseRuntime::new(architecture, eredu_runtime::ResidentUnitWindow::new(units));
    let mut state = StatelessComponentState::new().unwrap();
    let mut observer = Observer {
        seen: vec![],
        fail: None,
    };
    assert_eq!(
        resident
            .forward_with_observer((&FakeTensor(vec![1]), true), &mut state, &(), &mut observer)
            .unwrap(),
        FakeTensor(vec![99])
    );
    check(&observer.seen);
    for mode in 0..4 {
        for kernel_owned in [false, true] {
            // Raw custom executors borrow their declared seams from the driver;
            // ordinary observed equations emit kernel-owned seams themselves.
            if kernel_owned && mode < 2 {
                continue;
            }
            let mut runtime =
                LayerwiseRuntime::new(component(kernel_owned), RecordingPolicy::bounded(2));
            let mut observer = Observer {
                seen: vec![],
                fail: None,
            };
            let input = FakeTensor(vec![1]);
            let output = match mode {
                0 => runtime.forward_with_observer((&input, true), &mut state, &(), &mut observer),
                1 => runtime.forward_parallel_with_observer(
                    (&input, true),
                    &mut state,
                    &(),
                    &(),
                    &mut observer,
                ),
                2 => runtime.forward_with_unit_executor_and_observer(
                    (&input, true),
                    &mut state,
                    &(),
                    |a, g, i, u, h, s, f, c| a.forward_unit(g, i, u, h, s, f, c),
                    &mut observer,
                ),
                _ => runtime.forward_parallel_with_unit_executor_and_observer(
                    (&input, true),
                    &mut state,
                    &(),
                    &(),
                    |a, g, i, u, h, s, f, p, c| a.forward_unit_parallel(g, i, u, h, s, f, p, c),
                    &mut observer,
                ),
            }
            .unwrap();
            assert_eq!(output, FakeTensor(vec![99]));
            check(&observer.seen);
            assert!(!runtime.policy().forward_active);
        }
    }
    let mut runtime = LayerwiseRuntime::new(component(false), RecordingPolicy::bounded(2));
    for path in ["logical.input", "logical.output"] {
        let mut observer = Observer {
            seen: vec![],
            fail: Some(path),
        };
        assert!(runtime
            .forward_with_observer((&FakeTensor(vec![1]), true), &mut state, &(), &mut observer)
            .is_err());
        assert!(!runtime.policy().forward_active);
    }
    assert_eq!(runtime.policy().aborts, 2);
    let mut observer = Observer {
        seen: vec![],
        fail: None,
    };
    assert_eq!(
        runtime
            .forward_with_observer((&FakeTensor(vec![1]), true), &mut state, &(), &mut observer)
            .unwrap(),
        FakeTensor(vec![99])
    );
    check(&observer.seen);
    let owned = definition(false)
        .map_unit_observation_boundaries(|address, boundaries| {
            if address.group() == 0 {
                UnitObservationBoundaries::new(None, Some("fixture.internal".into()))
                    .with_kernel_ownership()
            } else {
                boundaries.clone()
            }
        })
        .unwrap();
    let mut runtime = LayerwiseRuntime::new(
        ComponentAdapter::new(
            owned,
            Kernel {
                fail: false,
                pinned: None,
            },
        )
        .unwrap(),
        RecordingPolicy::bounded(2),
    );
    let mut observer = Observer {
        seen: vec![],
        fail: None,
    };
    runtime
        .forward_with_observer((&FakeTensor(vec![1]), true), &mut state, &(), &mut observer)
        .unwrap();
    assert_eq!(
        observer
            .seen
            .iter()
            .filter(|(path, _)| path == "fixture.internal")
            .count(),
        2
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
    let groups = declaration.parameters().groups();
    let roles = declaration
        .select_complete_pinned_roles(groups, &["readout", "embedding"])
        .unwrap();
    assert_eq!(
        roles.into_iter().collect::<Vec<_>>(),
        ["embedding", "readout"]
    );
    assert!(declaration
        .select_complete_pinned_roles(&[], &["readout", "embedding"])
        .unwrap()
        .is_empty());
    assert!(declaration
        .select_complete_pinned_roles(&[groups[0].clone(), groups[0].clone()], &["embedding"])
        .is_err());
    // A pinned module may have separate canonical groups that must be admitted together.
    let companion = eredu_runtime::OwnedParameterGroupSpec::new(
        groups[0].owner().clone(),
        eredu_runtime::ParameterGroupSpec::new(
            "readout.bias",
            eredu_runtime::ParameterRole::Replicated,
            [eredu_runtime::ParameterMemberSpec::new(
                "bias",
                [1],
                eredu_runtime::MemberSharding::Replicated,
            )],
        )
        .unwrap(),
    );
    let complete_groups = vec![groups[0].clone(), companion.clone()];
    let complete = ComponentDefinition::new(
        "coupled-pinned-fixture",
        ArchitectureParameterDescription::new(
            declaration.parameters().graph(),
            declaration.parameters().unit_layout(),
            complete_groups
                .iter()
                .map(|binding| binding.group().clone()),
            complete_groups.clone(),
        )
        .unwrap(),
        vec![
            declaration.unit(0, 0).unwrap().clone(),
            declaration.unit(1, 0).unwrap().clone(),
        ],
        vec![
            declaration.group_transport(0).unwrap().clone(),
            declaration.group_transport(1).unwrap().clone(),
        ],
        "first",
        vec![],
        declaration.state_layout().clone(),
        declaration.state_partition_plan().clone(),
    )
    .unwrap();
    assert!(complete
        .select_complete_pinned_roles(groups, &["embedding"])
        .is_err());
    // Independent resources within a role retain their own canonical group admission.
    assert_eq!(
        complete
            .select_pinned_groups(groups, &["embedding"])
            .unwrap(),
        vec![&complete_groups[0]],
    );
    assert!(complete
        .select_pinned_groups(
            &[complete_groups[0].clone(), complete_groups[0].clone()],
            &["embedding"]
        )
        .is_err());
    assert!(declaration
        .select_complete_pinned_roles(&[companion], &["embedding"])
        .is_err());
    assert_eq!(
        complete
            .select_complete_pinned_roles(&complete_groups, &["embedding"])
            .unwrap()
            .len(),
        1
    );
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
    slots: Option<std::collections::BTreeMap<usize, Option<usize>>>,
}
impl CarryKernel {
    fn new(offset: usize, has_ingress: bool) -> Self {
        Self {
            offset,
            has_ingress,
            slots: None,
        }
    }
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
    fn prediction_target_placeholder_shape(&self, _: &()) -> Result<Option<Vec<i32>>, Failure> {
        Ok(Some(vec![1]))
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
        let slot = self
            .slots
            .as_ref()
            .map_or(Some(address.index()), |slots| slots[&address.index()]);
        let Some(slot) = slot else {
            return Ok(hidden.clone());
        };
        let carry = state.layer(slot - self.offset).unwrap();
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
    fn begin_group_parallel(
        &mut self,
        group: usize,
        input: &FakeTensor,
        dependencies: &[&FakeTensor],
        state: &mut CarryState,
        forward: &mut (),
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Failure> {
        // A collective-wave ingress advances carry only after graph validation.
        state.layer(0).unwrap().0 .0[0] += 1;
        self.begin_group(group, input, dependencies, state, forward, context)
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

fn carry_definition(indices: &[usize]) -> ComponentDefinition {
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("carry")], "carry").unwrap();
    let complete = ExecutionUnitLayout::new(&graph, [indices.last().unwrap() + 1]).unwrap();
    let units = complete
        .select(
            indices
                .iter()
                .map(|&index| complete.address(index).unwrap()),
        )
        .unwrap();
    let parameters = ArchitectureParameterDescription::new(&graph, &units, [], []).unwrap();
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap();
    let state =
        StateLayout::new(LayerSchedule::new(indices.len(), vec![policy; indices.len()]).unwrap())
            .unwrap();
    ComponentDefinition::new(
        "carry",
        parameters,
        indices
            .iter()
            .enumerate()
            .map(|(position, index)| ComponentUnit {
                path: format!("carry.{index}"),
                state: position..position + 1,
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
            eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..indices.len()),
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

type CarryStage = (
    ComponentAdapter<ComponentDefinition, CarryKernel>,
    LayeredPartitionDriver,
    CarryState,
    Vec<ExecutionUnitAddress>,
);

fn advance_carry_stages(stages: &mut [CarryStage], step: i32) -> FakeTensor {
    let input = FakeTensor(vec![7]);
    let mut hidden = input.clone();
    for (adapter, driver, state, addresses) in stages {
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
        for address in addresses.iter().copied() {
            let mut unit = adapter
                .build_unit(address.group(), address.index(), &())
                .unwrap();
            forward.hidden = adapter
                .forward_unit(
                    address.group(),
                    address.index(),
                    &mut unit,
                    &forward.hidden,
                    state,
                    &mut forward.context,
                    &(),
                )
                .unwrap();
            let roots = adapter
                .retained_state_ordinals(address.group(), address.index(), 0)
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
    hidden
}

#[test]
fn component_partitions_preserve_repeated_advancement_and_local_carry_retention() {
    let full = ComponentAdapter::new(
        carry_definition(&[0, 1, 2]),
        CarryKernel::new(0, false),
    )
    .unwrap();
    let mut resident = ResidentRuntime::new(full, &()).unwrap();
    let mut fullstate = carry_state(carry_definition(&[0, 1, 2]).state_layout().clone());
    let mut stages = [0..1, 1..3].map(|range| {
        let declaration = carry_definition(&[0, 1, 2]);
        let full = ComponentAdapter::new(
            declaration.clone(),
            CarryKernel::new(range.start, false),
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
        let addresses = partition.units().collect();
        let state = carry_state(partition.state().unwrap().layout().clone());
        let driver = LayeredPartitionDriver::new(&partition, 0, range).unwrap();
        let adapter = full.select_partition(&partition).unwrap();
        (adapter, driver, state, addresses)
    });
    for step in 1..=3 {
        let input = FakeTensor(vec![7]);
        let expected = resident.forward(&input, &mut fullstate, &()).unwrap();
        let hidden = advance_carry_stages(&mut stages, step);
        assert_eq!(hidden, expected);
        assert_eq!(hidden, FakeTensor(vec![7, step, step, step]));
    }
    let (last, _, _, _) = &stages[1];
    assert!(last.build_unit(0, 0, &()).is_err());
    // Logical units 3/7 own independent dense state slots 0/1.
    let declaration = carry_definition(&[3, 7]);
    let mut sparse_stages = [3, 7]
        .into_iter()
        .enumerate()
        .map(|(position, index)| {
            let full = ComponentAdapter::new(
                declaration.clone(),
                CarryKernel::new(index, false),
            )
            .unwrap();
            for invalid in [0..1, 3..8] {
                assert!(
                    ArchitecturePartition::from_architecture::<FakeBackend, CarryState, _, _>(
                        &full,
                        [("carry", invalid)],
                        PartitionOwnership::new(false, false, std::iter::empty::<String>())
                            .unwrap(),
                        (),
                        NoAuxiliaryBoundarySchema::new(1),
                        declaration.parameters(),
                    )
                    .is_err()
                );
            }
            let partition =
                ArchitecturePartition::from_architecture::<FakeBackend, CarryState, _, _>(
                    &full,
                    [("carry", index..index + 1)],
                    PartitionOwnership::new(
                        position == 0,
                        position == 1,
                        std::iter::empty::<String>(),
                    )
                    .unwrap(),
                    (),
                    NoAuxiliaryBoundarySchema::new(1),
                    declaration.parameters(),
                )
                .unwrap();
            let addresses = partition.units().collect::<Vec<_>>();
            let state = carry_state(partition.state().unwrap().layout().clone());
            let driver = LayeredPartitionDriver::new_with_state_range(
                &partition,
                0,
                index..index + 1,
                Some(position..position + 1),
            )
            .unwrap();
            (
                full.select_partition(&partition).unwrap(),
                driver,
                state,
                addresses,
            )
        })
        .collect::<Vec<_>>();
    for step in 1..=3 {
        assert_eq!(
            advance_carry_stages(&mut sparse_stages, step),
            FakeTensor(vec![7, step, step])
        );
    }
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

impl SumReductionBackend for FakeBackend {
    fn all_reduce_sum(
        value: FakeTensor,
        _: &(),
        _: &(),
    ) -> Result<Submission<FakeTensor, CommunicationDone>, FakeCommunicationError> {
        Ok(Submission {
            output: value,
            completion: CommunicationDone,
        })
    }
}

struct CaptureAllocator;
impl eredu_runtime::PartitionTensorAllocator<FakeBackend> for CaptureAllocator {
    fn tensor_placeholder(
        &mut self,
        shape: &[i32],
        _: eredu_runtime::BoundaryTensorDtype,
        _: eredu_runtime::PipelineActivationDtype,
        _: &(),
    ) -> Result<FakeTensor, eredu_nn::Error> {
        assert_eq!(shape, [1]);
        Ok(FakeTensor(vec![-1]))
    }
}

#[test]
fn component_invocation_state_ownership_skips_interleaved_stateless_units() {
    for offset in [0, 7] {
        let base = carry_definition(&[0, 1]);
        let graph = base.parameters().graph();
        let complete = ExecutionUnitLayout::new(graph, [offset + 7]).unwrap();
        let layout = complete
            .select((offset..offset + 7).map(|index| complete.address(index).unwrap()))
            .unwrap();
        let units = (0..7)
            .map(|index| ComponentUnit {
                path: format!("operation.{}", offset + index),
                state: match index {
                    1 => 0..1,
                    4 => 1..2,
                    _ => 0..0,
                },
            })
            .collect::<Vec<_>>();
        let make = |units: Vec<ComponentUnit>, state: StateLayout, plan| {
            ComponentDefinition::new(
                "interleaved",
                ArchitectureParameterDescription::new(graph, &layout, [], []).unwrap(),
                units,
                vec![base.group_transport(0).unwrap().clone()],
                "carry",
                vec![],
                state,
                plan,
            )
        };
        let owners = |shared: bool| {
            ArchitectureStatePartitionPlan::new(
                std::iter::once(eredu_runtime::ArchitectureStatePartitionRule::invocation(
                    layout.address(1).unwrap(),
                    0..1,
                ))
                .chain((!shared).then(|| {
                    eredu_runtime::ArchitectureStatePartitionRule::invocation(
                        layout.address(4).unwrap(),
                        1..2,
                    )
                })),
            )
        };
        let definition = make(units.clone(), base.state_layout().clone(), owners(false)).unwrap();
        // Cold placement consumes the complete declaration; selected execution
        // below retains the nonzero interval without inventing prefix state.
        let cold_units = (0..offset)
            .map(|index| ComponentUnit {
                path: format!("operation.{index}"),
                state: 0..0,
            })
            .chain(units.iter().cloned())
            .collect();
        let cold_definition = ComponentDefinition::new(
            "interleaved",
            ArchitectureParameterDescription::new(graph, &complete, [], []).unwrap(),
            cold_units,
            vec![base.group_transport(0).unwrap().clone()],
            "carry",
            vec![],
            definition.state_layout().clone(),
            owners(false),
        )
        .unwrap();
        let requirements = eredu_runtime::ReplicatedTextRequirements::new(
            "interleaved",
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            complete.clone(),
            vec![base.group_transport(0).unwrap().clone()],
            definition.state_layout().clone(),
            eredu_runtime::ReplicatedTextStateAccess::KeyValue,
            vec![],
        )
        .unwrap();
        let incompatible = ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::output_owner(0..2),
        ]);
        let cold = requirements
            .clone()
            .with_component_definition(cold_definition.clone())
            .unwrap();
        let changed_seams = cold_definition
            .clone()
            .map_unit_observation_boundaries(|address, boundaries| {
                if address.index() == 0 {
                    eredu_runtime::UnitObservationBoundaries::new(Some("changed.input".into()), None)
                } else {
                    boundaries.clone()
                }
            })
            .unwrap();
        assert!(cold
            .clone()
            .with_component_definition(changed_seams)
            .is_err());
        let cold = cold
            .with_component_definition(cold_definition.clone())
            .unwrap();
        assert!(cold
            .clone()
            .with_state_partition_plan(incompatible.clone())
            .is_err());
        assert!(requirements
            .with_state_partition_plan(incompatible)
            .unwrap()
            .with_component_definition(cold_definition.clone())
            .is_err());
        for rank in 0..3 {
            let projection = eredu_runtime::PipelineRankProjection::new(
                &cold,
                Some(definition.state_layout()),
                eredu_core::ParallelRankTopology::new(
                    eredu_core::ParallelTopology::new(1, 3, 1, 1).unwrap(),
                    rank,
                )
                .unwrap(),
                false,
            )
            .unwrap();
            let expected = [offset + 1, offset + 4]
                .into_iter()
                .enumerate()
                .filter(|(_, invocation)| {
                    projection
                        .groups()
                        .iter()
                        .any(|group| group.units().contains(invocation))
                })
                .map(|(slot, _)| slot)
                .collect::<Vec<_>>();
            assert_eq!(
                projection.state().map(|state| state.global_layers()),
                expected
                    .first()
                    .zip(expected.last())
                    .map(|(first, last)| *first..last + 1)
            );
        }
        let kernel = |start, shared| {
            let mut kernel = CarryKernel::new(start, false);
            kernel.slots = Some(
                (0..7)
                    .map(|index| {
                        (
                            offset + index,
                            match index {
                                1 => Some(0),
                                4 => Some(if shared { 0 } else { 1 }),
                                _ => None,
                            },
                        )
                    })
                    .collect(),
            );
            kernel
        };
        let adapter = || ComponentAdapter::new(definition.clone(), kernel(0, false)).unwrap();
        let mut resident = ResidentRuntime::new(adapter(), &()).unwrap();
        let mut full_state = carry_state(definition.state_layout().clone());
        let mut stages = [
            offset..offset + 3,
            offset + 3..offset + 6,
            offset + 6..offset + 7,
        ]
        .map(|range| {
            let partition = ArchitecturePartition::from_description(
                definition.parameters(),
                [("carry", range.clone())],
                PartitionOwnership::new(
                    range.start == offset,
                    range.end == offset + 7,
                    std::iter::empty::<String>(),
                )
                .unwrap(),
                definition.state_layout(),
                definition.state_partition_plan(),
                (),
                NoAuxiliaryBoundarySchema::new(1),
            )
            .unwrap();
            let start = partition
                .state()
                .map_or(0, |state| state.global_layer_offset());
            let state = partition
                .state()
                .map_or_else(CarryState::stateless, |state| {
                    carry_state(state.layout().clone())
                });
            if range.start == offset + 6 {
                assert!(state.optional_layout().is_none());
            }
            let component =
                ComponentAdapter::new(definition.clone(), kernel(start, false)).unwrap();
            let policy = eredu_runtime::ResidentUnitWindow::new(
                partition
                    .units()
                    .map(|address| {
                        component
                            .build_unit(address.group(), address.index(), &())
                            .unwrap()
                    })
                    .collect(),
            );
            let runtime = component
                .into_partition_runtime::<FakeBackend, CarryState, _, _, _>(policy, &partition)
                .unwrap();
            (runtime, state, range.start == offset)
        });
        for step in 1..=6 {
            let input = FakeTensor(vec![9]);
            let expected = resident.forward(&input, &mut full_state, &()).unwrap();
            let mut hidden = input.clone();
            for (runtime, state, input_owner) in &mut stages {
                let ingress = if *input_owner {
                    LayeredPartitionInput::Tokens(&input)
                } else {
                    LayeredPartitionInput::Hidden {
                        hidden,
                        auxiliary: NoAuxiliaryBoundary,
                    }
                };
                let parallel = (step % 2 == 0).then_some(&());
                let output = runtime
                    .forward_with_preparation(state, parallel, &(), |adapter, state, expected| {
                        adapter.begin_prepared_partition(
                            ingress,
                            None,
                            state,
                            expected,
                            parallel,
                            &(),
                        )
                    })
                    .unwrap();
                hidden = match output {
                    LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
                    LayeredPartitionOutput::Final { output, .. } => output,
                };
            }
            assert_eq!(hidden, expected);
            assert_eq!(hidden, FakeTensor(vec![9, step, step]));
        }
        let mut shared_units = units.clone();
        shared_units[4].state = 0..1;
        let shared = make(
            shared_units,
            base.state_layout().slice(0..1).unwrap(),
            owners(true),
        )
        .unwrap();
        let mut runtime = ResidentRuntime::new(
            ComponentAdapter::new(shared.clone(), kernel(0, true)).unwrap(),
            &(),
        )
        .unwrap();
        let mut state = carry_state(shared.state_layout().clone());
        for step in 1..=3 {
            assert_eq!(
                runtime
                    .forward(&FakeTensor(vec![9]), &mut state, &())
                    .unwrap(),
                FakeTensor(vec![9, step * 2 - 1, step * 2])
            );
        }
        let other_consumer = ArchitecturePartition::from_description(
            shared.parameters(),
            [("carry", offset + 3..offset + 6)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            shared.state_layout(),
            shared.state_partition_plan(),
            (),
            NoAuxiliaryBoundarySchema::new(1),
        )
        .unwrap();
        assert!(shared.select_partition(&other_consumer).is_err());
        let unknown = ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::invocation(
                layout.address(1).unwrap().with_index(offset + 99),
                0..2,
            ),
        ]);
        assert!(matches!(
            make(units.clone(), base.state_layout().clone(), unknown),
            Err(ComponentDefinitionError::StatePartition {
                source: eredu_runtime::ArchitectureStatePartitionError::UnknownInvocation { .. },
                ..
            })
        ));
        let wrong_owner = ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::invocation(
                layout.address(0).unwrap(),
                0..1,
            ),
            eredu_runtime::ArchitectureStatePartitionRule::invocation(
                layout.address(4).unwrap(),
                1..2,
            ),
        ]);
        assert!(make(units.clone(), base.state_layout().clone(), wrong_owner).is_err());
        let mut gap_units = units.clone();
        gap_units[4].state = 2..3;
        gap_units[6].state = 1..2;
        let gap_plan = ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::invocation(
                layout.address(1).unwrap(),
                0..1,
            ),
            eredu_runtime::ArchitectureStatePartitionRule::invocation(
                layout.address(6).unwrap(),
                1..2,
            ),
            eredu_runtime::ArchitectureStatePartitionRule::invocation(
                layout.address(4).unwrap(),
                2..3,
            ),
        ])
        .with_packing(eredu_runtime::StatePartitionPacking::PreserveGlobalOrdinals);
        let gap_state = carry_definition(&[0, 1, 2]).state_layout().clone();
        let gap = make(gap_units.clone(), gap_state.clone(), gap_plan.clone()).unwrap();
        let partition = ArchitecturePartition::from_description(
            gap.parameters(),
            [("carry", offset..offset + 6)],
            PartitionOwnership::new(true, false, std::iter::empty::<String>()).unwrap(),
            gap.state_layout(),
            gap.state_partition_plan(),
            (),
            NoAuxiliaryBoundarySchema::new(1),
        )
        .unwrap();
        assert_eq!(
            partition.state().unwrap().selected_global_layers(),
            [0..1, 2..3]
        );
        assert!(matches!(
            partition.state().unwrap().layout().layers().get(1),
            Some(LayerCachePolicy::NoState)
        ));
        let span = gap
            .state_partition_plan()
            .group_state_range(&layout, 0, offset..offset + 6)
            .unwrap();
        assert_eq!(span, Some(0..3));
        assert!(gap.select_partition(&partition).is_ok());
        assert!(LayeredPartitionDriver::new_with_state_range(
            &partition,
            0,
            offset..offset + 6,
            span
        )
        .is_ok());
        gap_units[0].state = 1..2;
        let unowned = make(gap_units, gap_state, gap_plan).unwrap();
        assert!(unowned.select_partition(&partition).is_err());
    }
}

#[test]
fn component_stage_accepts_stateless_ingress_and_retains_later_group_carry() {
    for offset in [0, 7] {
        check_stage_carry(offset);
    }
}

fn check_stage_carry(offset: usize) {
    let base = carry_definition(&[0, 1, 2]);
    let source = ComponentAdapter::new(
        base.clone(),
        CarryKernel::new(0, false),
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
    let complete = ExecutionUnitLayout::new(&graph, [offset + 1, offset + 3]).unwrap();
    let layout = complete
        .select(
            std::iter::once(complete.address(offset).unwrap())
                .chain((0..3).map(|index| complete.address(offset + 1 + offset + index).unwrap())),
        )
        .unwrap();
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
        CarryKernel::new(offset, true),
    )
    .unwrap();
    let partition = ArchitecturePartition::from_architecture::<FakeBackend, CarryState, _, _>(
        &adapter,
        [
            ("ingress", offset..offset + 1),
            ("carry", offset..offset + 3),
        ],
        PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        definition.parameters(),
    )
    .unwrap();
    for (ordinals, valid) in [
        (&[0, 1][..], true),
        (&[0, 1, 2, 3][..], true),
        (&[0, 2, 3][..], false),
        (&[0, 1, 3][..], false),
    ] {
        let adapter = ComponentAdapter::new(
            definition.clone(),
            CarryKernel::new(offset, true),
        )
        .unwrap();
        let addresses: Vec<_> = ordinals
            .iter()
            .map(|&ordinal| layout.address(ordinal).unwrap())
            .collect();
        let policy = eredu_runtime::ResidentUnitWindow::new(
            addresses
                .iter()
                .map(|address| {
                    adapter
                        .build_unit(address.group(), address.index(), &())
                        .unwrap()
                })
                .collect(),
        );
        let result = eredu_runtime::PipelinePartitionExecutor::new(
            adapter,
            policy,
            addresses,
            None,
            CaptureAllocator,
            eredu_runtime::PipelineActivationDtype::Float32,
        );
        if !valid {
            assert!(result.is_err());
            continue;
        }
        let mut executor = result.unwrap();
        let capture = <_ as eredu_runtime::PartitionedGroupExecutor<
            _,
            FakeBackend,
            CarryState,
            (),
            (),
            FakeTensorMetadata,
        >>::prediction_target_capture(&mut executor, &(), &())
        .unwrap();
        assert_eq!(
            capture,
            if ordinals.last() == Some(&3) {
                None
            } else {
                Some(FakeTensor(vec![-1]))
            }
        );
    }
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
    let incorrect_resume = incorrect_entry.slice(0..1).unwrap();
    assert!(runtime
        .forward_with_preparation(&mut state, None, &(), |adapter, state, _| {
            eredu_runtime::PartitionedLayeredArchitecture::begin_partition(
                adapter,
                LayeredPartitionInput::Hidden {
                    hidden: input.clone(),
                    auxiliary: NoAuxiliaryBoundary,
                },
                None,
                state,
                &incorrect_resume,
                0,
                &(),
            )
        })
        .is_err());
    for step in 1..=6 {
        let input = FakeTensor(vec![9]);
        let output = runtime
            .forward_with_preparation(&mut state, None, &(), |adapter, state, expected| {
                if step % 2 == 0 {
                    // A boundary resumes the stateful group even when this rank
                    // also owns an earlier stateless ingress group.
                    let local = RuntimeState::<FakeBackend>::layout(state).clone();
                    eredu_runtime::PartitionedLayeredArchitecture::begin_partition(
                        adapter,
                        LayeredPartitionInput::Hidden {
                            hidden: input.clone(),
                            auxiliary: NoAuxiliaryBoundary,
                        },
                        None,
                        state,
                        &local,
                        0,
                        &(),
                    )
                } else {
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
                }
            })
            .unwrap();
        let LayeredPartitionOutput::Final { output, .. } = output else {
            panic!("stage owns output")
        };
        assert_eq!(output, FakeTensor(vec![9, step, step, step]));
    }
}

// Group closure contributes to dependencies while its captured target remains a
// separate retained value for a later readout. A failed closure aborts the pass.
struct CaptureKernel {
    fail_completion: bool,
}
impl ComponentKernel<FakeBackend> for CaptureKernel {
    type Input<'a> = &'a FakeTensor;
    type Unit = FakeUnit;
    type ForwardContext = Option<FakeTensor>;
    type RetainedContextValues<'a> = std::option::Iter<'a, FakeTensor>;
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
            marker: (address.group() * 10 + address.index()) as i32,
        })
    }
    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&FakeTensor> {
        forward.as_ref()
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.iter()
    }
}
impl ComponentEquations<FakeBackend, StatelessComponentState> for CaptureKernel {
    fn begin<'a>(
        &mut self,
        input: &'a FakeTensor,
        _: &mut StatelessComponentState,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Failure> {
        Ok(LayeredForwardState {
            hidden: input.clone(),
            context: None,
        })
    }
    fn begin_group(
        &mut self,
        _: usize,
        initial: &FakeTensor,
        dependencies: &[&FakeTensor],
        _: &mut StatelessComponentState,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        Ok(dependencies.first().copied().unwrap_or(initial).clone())
    }
    fn execute(
        &mut self,
        _: ExecutionUnitAddress,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        _: &mut StatelessComponentState,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        let mut output = hidden.clone();
        output.0.push(unit.marker);
        Ok(output)
    }
    fn complete_group(
        &mut self,
        group: usize,
        hidden: &FakeTensor,
        _: &mut StatelessComponentState,
        forward: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        if group == 0 {
            *forward = Some(hidden.clone());
        }
        if self.fail_completion && group == 1 {
            return Err(Failure::Equation);
        }
        let mut output = hidden.clone();
        output.0.push(100 + group as i32);
        Ok(output)
    }
    fn finish(
        &mut self,
        hidden: &FakeTensor,
        _: &mut StatelessComponentState,
        forward: &Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        let mut output = hidden.clone();
        output.0.extend_from_slice(
            &Self::prediction_target_capture(forward)
                .ok_or(Failure::Equation)?
                .0,
        );
        Ok(output)
    }
}

#[test]
fn component_group_completion_preserves_dependencies_capture_and_retry() {
    let component = |fail_completion| {
        ComponentAdapter::new(definition(false), CaptureKernel { fail_completion }).unwrap()
    };
    let input = FakeTensor(vec![7]);
    let mut state = StatelessComponentState::new().unwrap();
    let mut resident = ResidentRuntime::new(component(false), &()).unwrap();
    let mut bounded = LayerwiseRuntime::new(component(true), RecordingPolicy::bounded(2));
    assert!(bounded.forward(&input, &mut state, &()).is_err());
    assert_eq!(bounded.policy().aborts, 1);
    assert!(!bounded.policy().forward_active);
    bounded.architecture_mut().kernel_mut().fail_completion = false;
    let expected = resident.forward(&input, &mut state, &()).unwrap();
    assert_eq!(expected.0, [7, 0, 100, 10, 101, 7, 0]);
    assert_eq!(bounded.forward(&input, &mut state, &()).unwrap(), expected);
}

#[test]
fn component_parallel_group_ingress_validates_before_advancing_state() {
    let declaration = carry_definition(&[0, 1, 2]);
    let mut state = carry_state(declaration.state_layout().clone());
    let mut adapter = ComponentAdapter::new(
        declaration,
        CarryKernel::new(0, false),
    )
    .unwrap();
    let input = FakeTensor(vec![7]);
    let before = state.layer(0).unwrap().0 .0[0];
    for (group, dependencies) in [(0, vec![&input]), (1, vec![])] {
        assert!(adapter
            .begin_execution_group_parallel(
                group,
                &input,
                &dependencies,
                &mut state,
                &mut (),
                &(),
                &(),
            )
            .is_err());
        assert_eq!(state.layer(0).unwrap().0 .0[0], before);
    }
    let result = adapter
        .begin_execution_group_parallel(0, &input, &[], &mut state, &mut (), &(), &())
        .unwrap();
    assert_eq!(result.0, input.0);
    assert_eq!(state.layer(0).unwrap().0 .0[0], before + 1);
}
