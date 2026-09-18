//! The existing uneven scalar equations with an actual retained numerical helper.
use super::*;
// Deliberately not Clone: the source hook only borrows this physical owner.
pub(super) struct Chunked {
    pub(super) inner: OrdinaryTextFixture,
    pub(super) helper: Helper,
}
pub(super) struct Helper {
    pub(super) value: FakeTensor,
    pub(super) visits: Rc<Cell<usize>>,
    pub(super) complete: bool,
}
impl Parameterized<FakeTensor> for Helper {
    fn visit_parameter_sources<'a,V:eredu_nn::ParameterSourceVisitor<'a,FakeTensor>>(&'a self, visitor:&mut V)->Result<(),eredu_nn::ParameterSourceError>{
        self.visits.set(self.visits.get()+1);
        visitor.retained(&self.value);
        if self.complete { Ok(()) } else { Err(eredu_nn::ParameterSourceError::UnclassifiedRetainedField) }
    }
    fn visit_parameters_mut<'a,V:ParameterVisitorMut<'a,FakeTensor>>(&'a mut self,_:&mut V){}
    fn set_trainable(&mut self,_:bool){}
}

impl ArchitectureParameters<FakeBackend> for Chunked {
    type DefinitionError = Error;
    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&FakeTensor)) -> bool {
        // The delegated static aggregate has no tensor parameters or helpers.
        self.helper.visit_retained_values(visitor)
    }

    fn state_layout(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<StateLayout, Error> {
        self.inner.state_layout(metadata)
    }
    fn state_identity(
        &self,
        s: &eredu_runtime::PartitionState,
        t: eredu_core::cache::PromptCacheTopology,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.inner.state_identity(s, t, metadata)
    }
    fn parameter_description(&self, c: &()) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Error> {
        self.inner.parameter_description(c)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<FakeBackend>>(
        &self,
        v: &mut V,
    ) -> Result<(), V::Error> {
        self.inner.visit_static_parameters(v)
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<FakeBackend>>(
        &mut self,
        v: &mut V,
    ) -> Result<(), V::Error> {
        self.inner.visit_static_parameters_mut(v)
    }
}
impl LayeredArchitecture<FakeBackend, State> for Chunked {
    type Input<'a> = &'a FakeTensor;
    type StaticModules = Helper;
    type Unit = FakeUnit;
    type ForwardContext = usize;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;
    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Error> {
        Ok(Some([1, input.0.len() as u64]))
    }
    fn group_transport(&self, g: usize) -> ArchitectureGroupTransport {
        self.inner.group_transport(g)
    }
    fn primary_execution_group(&self) -> &str {
        self.inner.primary_execution_group()
    }
    fn state_partition_plan(
        &self,
        s: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.inner.state_partition_plan(s)
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        self.inner.execution_graph()
    }
    fn group_unit_count(&self, g: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Error> {
        self.inner.group_unit_count(g, metadata_context)
    }
    fn unit_path(&self, g: usize, i: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Error> {
        self.inner.unit_path(g, i, metadata_context)
    }
    fn static_modules(&self) -> &Helper {
        &self.helper
    }
    fn static_modules_mut(&mut self) -> &mut Helper {
        &mut self.helper
    }
    fn build_unit(&self, g: usize, i: usize, c: &()) -> Result<FakeUnit, Error> {
        self.inner.build_unit(g, i, c)
    }
    fn begin_forward<'a>(
        &mut self,
        input: &'a FakeTensor,
        state: &mut State,
        c: &(),
    ) -> Result<LayeredForwardState<FakeTensor, usize>, Error> {
        let first = self.inner.begin_forward(input, state, c)?;
        Ok(LayeredForwardState {
            hidden: first.hidden,
            context: input.0.len(),
        })
    }
    fn begin_execution_group(
        &mut self,
        g: usize,
        input: &FakeTensor,
        deps: &[&FakeTensor],
        state: &mut State,
        _: &mut usize,
        c: &(),
    ) -> Result<FakeTensor, Error> {
        self.inner
            .begin_execution_group(g, input, deps, state, &mut (), c)
    }
    fn forward_unit(
        &mut self,
        g: usize,
        i: usize,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut State,
        width: &mut usize,
        c: &(),
    ) -> Result<FakeTensor, Error> {
        assert!(*width > 0);
        let output = self
            .inner
            .forward_unit(g, i, unit, hidden, state, &mut (), c)?;
        let remaining = i32::try_from(*width - 1).unwrap();
        state.as_mut()[0].0 += remaining;
        if let Some(value) = &mut state.as_mut()[0].1 {
            value.0[0] += remaining;
        }
        self.helper.value.0[0] += i32::try_from(*width).unwrap();
        Ok(output)
    }
    fn select_readout_positions(
        &self,
        hidden: &FakeTensor,
        _: &usize,
        demand: OutputDemand,
        c: &(),
    ) -> Result<Option<FakeTensor>, Error> {
        self.inner.select_readout_positions(hidden, &(), demand, c)
    }
    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        state: &mut State,
        _: &usize,
        c: &(),
    ) -> Result<FakeTensor, Error> {
        self.inner.finish_forward(hidden, state, &(), c)
    }
    fn retained_context_values<'a>(
        &'a self,
        _: &'a usize,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl ReplicatedTextArchitecture<FakeBackend, State> for Chunked {
    fn text_input<'a>(tokens: &'a FakeTensor, _: Option<&'a FakeTensor>) -> &'a FakeTensor {
        tokens
    }
}
