//! Ordinary input and media-control lifetimes must not constrain one another.
use super::*;

type State = DeviceState<FakeBackend, FakeLayerState>;

struct BorrowedState<'state>(&'state mut State);

impl RuntimeState<FakeBackend> for BorrowedState<'_> {
    type RetainedValues<'a>
        = <State as RuntimeState<FakeBackend>>::RetainedValues<'a>
    where
        Self: 'a;

    fn layout(&self) -> &StateLayout {
        self.0.layout()
    }
    fn visit_all_retained_values(
        &self,
        visitor: &mut dyn FnMut(&FakeTensor),
    ) -> Result<(), StateError> {
        self.0.visit_all_retained_values(visitor)
    }
    fn retained_values(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        self.0.retained_values(ordinal, address)
    }
}

// This adapter delegates the existing fixture equations and selected partition
// traversal. Only the state representation is borrowed for this regression.
struct BorrowedArchitecture(OrdinaryTextFixture);

impl ArchitectureParameters<FakeBackend> for BorrowedArchitecture {
    type DefinitionError = Error;
    fn state_layout(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<StateLayout, Error> {
        self.0.state_layout(metadata)
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.0.state_identity(state, topology, metadata)
    }
    fn parameter_description(
        &self,
        context: &(),
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Error> {
        self.0.parameter_description(context)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<FakeBackend>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.0.visit_static_parameters(visitor)
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<FakeBackend>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.0.visit_static_parameters_mut(visitor)
    }
}

impl<'state> LayeredArchitecture<FakeBackend, BorrowedState<'state>> for BorrowedArchitecture {
    type Input<'a> = &'a FakeTensor;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = ();
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Error> {
        OrdinaryTextFixture::inference_input_shape(input)
    }
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        self.0.group_transport(group)
    }
    fn primary_execution_group(&self) -> &str {
        self.0.primary_execution_group()
    }
    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.0.state_partition_plan(layout)
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        self.0.execution_graph()
    }
    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Error> {
        self.0.group_unit_count(group, metadata_context)
    }
    fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Error> {
        self.0.unit_path(group, index, metadata_context)
    }
    fn static_modules(&self) -> &FakeOperator {
        self.0.static_modules()
    }
    fn static_modules_mut(&mut self) -> &mut FakeOperator {
        self.0.static_modules_mut()
    }
    fn build_unit(&self, group: usize, index: usize, context: &()) -> Result<FakeUnit, Error> {
        self.0.build_unit(group, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut BorrowedState<'state>,
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, ()>, Error> {
        self.0.begin_forward(input, state.0, context)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &FakeTensor,
        dependencies: &[&FakeTensor],
        state: &mut BorrowedState<'state>,
        forward: &mut (),
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.0
            .begin_execution_group(group, initial, dependencies, state.0, forward, context)
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut BorrowedState<'state>,
        forward: &mut (),
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.0
            .forward_unit(group, index, unit, hidden, state.0, forward, context)
    }
    fn select_readout_positions(
        &self,
        hidden: &FakeTensor,
        forward: &(),
        demand: eredu_core::OutputDemand,
        context: &(),
    ) -> Result<Option<FakeTensor>, Error> {
        self.0
            .select_readout_positions(hidden, forward, demand, context)
    }
    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        state: &mut BorrowedState<'state>,
        forward: &(),
        context: &(),
    ) -> Result<FakeTensor, Error> {
        self.0.finish_forward(hidden, state.0, forward, context)
    }
    fn retained_context_values<'a>(
        &'a self,
        _: &'a (),
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}

struct BorrowedExecutor(ReferencePartitionExecutor);

struct BorrowedPass<'input, 'control, 'state: 'control> {
    input: &'input FakeTensor,
    inner: ReferencePartitionPass,
    control: std::marker::PhantomData<&'control BorrowedState<'state>>,
}

impl<'state>
    PartitionedGroupExecutor<
        BorrowedArchitecture,
        FakeBackend,
        BorrowedState<'state>,
        (),
        (),
        FakeTensorMetadata,
    > for BorrowedExecutor
{
    fn group_submission_mechanism(&self) -> eredu_runtime::GroupSubmissionMechanism {
        self.0.group_submission_mechanism()
    }

    type Pass<'input, 'control>
        = BorrowedPass<'input, 'control, 'state>
    where
        BorrowedState<'state>: 'control;

    fn begin<'input, 'control>(
        &mut self,
        input: <BorrowedArchitecture as LayeredArchitecture<FakeBackend, BorrowedState<'state>>>::Input<'input>,
        state: &mut BorrowedState<'state>,
        pass: eredu_runtime::ExpertPass,
        demand: eredu_core::OutputDemand,
        context: &(),
    ) -> Result<Self::Pass<'input, 'control>, Error>
    where
        BorrowedState<'state>: 'control,
    {
        Ok(BorrowedPass {
            input,
            inner: self.0.begin(input, state.0, pass, demand, context)?,
            control: std::marker::PhantomData,
        })
    }
    fn request_group_active(&self, pass: &Self::Pass<'_, '_>, group: usize) -> Result<bool, Error> {
        self.0.request_group_active(&pass.inner, group)
    }
    fn execute_group<'control, O: eredu_runtime::ActivationObserver<FakeTensor, Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_, 'control>,
        driver: &LayeredPartitionDriver,
        state: &mut BorrowedState<'state>,
        communication: &PartitionCommunication<FakeBackend, (), (), FakeTensorMetadata>,
        executor: &(),
        context: &(),
        observer: &mut O,
    ) -> Result<(), Error>
    where BorrowedState<'state>: 'control {
        self.0.execute_group(
            &mut pass.inner,
            driver,
            state.0,
            communication,
            executor,
            context,
            observer,
        )
    }
    fn boundary_values(
        &mut self,
        pass: &mut Self::Pass<'_, '_>,
        route: &PartitionBoundaryRoute,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        source: bool,
        context: &(),
    ) -> Result<Vec<eredu_runtime::ArchitectureBoundaryValue<FakeTensor>>, Error> {
        self.0
            .boundary_values(&mut pass.inner, route, schema, source, context)
    }
    fn boundary_schema(
        &self,
        pass: &Self::Pass<'_, '_>,
        route: &PartitionBoundaryRoute,
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, Error> {
        self.0.boundary_schema(&pass.inner, route)
    }
    fn accept_boundary(
        &mut self,
        pass: &mut Self::Pass<'_, '_>,
        route: &PartitionBoundaryRoute,
        values: Vec<FakeTensor>,
    ) -> Result<(), Error> {
        self.0.accept_boundary(&mut pass.inner, route, values)
    }
    fn finish(
        &mut self,
        pass: Self::Pass<'_, '_>,
        state: &mut BorrowedState<'state>,
        context: &(),
    ) -> Result<(Option<FakeTensor>, ()), Error> {
        self.0.finish(pass.inner, state.0, context)
    }
}

// The explicit static input in the return type prevents this regression from
// silently shortening the input lifetime to the lifetime of the borrowed state.
fn begin_static<'state, 'control>(
    executor: &mut BorrowedExecutor,
    state: &mut BorrowedState<'state>,
    input: &'static FakeTensor,
) -> BorrowedPass<'static, 'control, 'state>
where
    'state: 'control,
{
    executor
        .begin(
            input,
            state,
            eredu_runtime::ExpertPass::Prefill,
            eredu_core::OutputDemand::LastPosition,
            &(),
        )
        .unwrap()
}

#[test]
fn static_partition_input_allows_borrowed_state_and_traversal_reborrow() {
    static INPUT: std::sync::OnceLock<FakeTensor> = std::sync::OnceLock::new();
    let input = INPUT.get_or_init(|| FakeTensor(vec![7, 11]));
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: ReplicatedSessionCounters::default(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let parameters = architecture.parameter_description(&()).unwrap();
    let partition = ArchitecturePartition::<(), NoAuxiliaryBoundarySchema>::from_architecture::<
        FakeBackend,
        State,
        _,
        _,
    >(
        &architecture,
        [("decoder", 0..1)],
        PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &parameters,
    )
    .unwrap();
    let driver = LayeredPartitionDriver::new(&partition, 0, 0..1).unwrap();
    let layout = architecture.state_layout(None).unwrap();
    let mut state = DeviceState::create(layout, |_, _| {
        Ok::<_, Error>(FakeLayerState(0, Some(FakeTensor(vec![13]))))
    })
    .unwrap();
    let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
        CommunicationManifest::new(1, 0, Vec::new(), Vec::new()).unwrap(),
        Vec::new(),
        Vec::new(),
        FakeTensorMetadata,
    )
    .unwrap();
    let mut executor = BorrowedExecutor(ReferencePartitionExecutor {
        architecture,
        policy: RecordingPolicy::new(vec![FakeUnit { marker: 5 }]),
        expose_policy: false,
        fail_after_state: Rc::new(Cell::new(false)),
    });
    {
        let mut borrowed = BorrowedState(&mut state);
        let mut pass = begin_static(&mut executor, &mut borrowed, input);
        assert!(std::ptr::eq(pass.input, input));
        assert_eq!(borrowed.0.as_ref()[0].0, 0, "begin must not execute a unit");
        executor
            .execute_group(
                &mut pass,
                &driver,
                &mut borrowed,
                &communication,
                &(),
                &(),
                &mut eredu_runtime::NoopObserver,
            )
            .unwrap();
        let (output, ()) = executor.finish(pass, &mut borrowed, &()).unwrap();
        assert_eq!(output, Some(FakeTensor(vec![7, 11, 5])));
        assert_eq!(borrowed.0.as_ref()[0].0, 1);
        assert_eq!(borrowed.0.as_ref()[0].1, Some(FakeTensor(vec![14])));
    }
    assert_eq!(
        state.as_ref()[0].0,
        1,
        "the caller regains the same mutated state"
    );
    assert_eq!(executor.0.architecture.trace, ["unit", "output"]);
}
