use super::*;
use eredu_runtime::{
    ArchitectureStatePartitionPlan, ComponentAdapter, ComponentDefinition,
    ComponentDefinitionError, ComponentKernel, ComponentUnit, StatelessComponentState,
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
}
impl ComponentKernel<FakeBackend> for Kernel {
    type State = StatelessComponentState;
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
    fn begin<'a>(
        &mut self,
        (input, extended): Self::Input<'a>,
        _: &mut Self::State,
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
        _: &mut Self::State,
        _: &mut bool,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        Ok(dependencies.first().copied().unwrap_or(initial).clone())
    }
    fn executes_group(&self, group: usize, extended: &bool) -> bool {
        group == 0 || *extended
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        _: &mut Self::State,
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
        _: &mut Self::State,
        _: &bool,
        _: &(),
    ) -> Result<FakeTensor, Failure> {
        Ok(hidden.clone())
    }
    fn retained_values<'a>(
        &'a self,
        _: &'a bool,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}

fn adapter(fail: bool) -> ComponentAdapter<ComponentDefinition, Kernel> {
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root("first"),
            ExecutionGroupSpec::with_dependencies("second", ["first"]),
        ],
        "second",
    )
    .unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [1, 1]).unwrap();
    let parameters = ArchitectureParameterDescription::new(&graph, &layout, [], []).unwrap();
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
    ComponentAdapter::new(definition, Kernel { fail }).unwrap()
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
