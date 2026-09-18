//! Geometry-correct variable-width adapter over the existing scalar fixture.
//! Construction, state, policy, completion and SessionPrefill remain unchanged.
use super::*;
struct Chunked(OrdinaryTextFixture);
impl ArchitectureParameters<FakeBackend> for Chunked {
    type DefinitionError = Error;
    fn state_layout(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<StateLayout, Error> {
        self.0.state_layout(metadata)
    }
    fn state_identity(
        &self,
        s: &eredu_runtime::PartitionState,
        t: eredu_core::cache::PromptCacheTopology,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.0.state_identity(s, t, metadata)
    }
    fn parameter_description(&self, c: &()) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Error> {
        self.0.parameter_description(c)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<FakeBackend>>(
        &self,
        v: &mut V,
    ) -> Result<(), V::Error> {
        self.0.visit_static_parameters(v)
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<FakeBackend>>(
        &mut self,
        v: &mut V,
    ) -> Result<(), V::Error> {
        self.0.visit_static_parameters_mut(v)
    }
}
impl LayeredArchitecture<FakeBackend, State> for Chunked {
    type Input<'a> = &'a FakeTensor;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = usize;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;
    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Error> {
        Ok(Some([1, input.0.len() as u64]))
    }
    fn group_transport(&self, g: usize) -> ArchitectureGroupTransport {
        self.0.group_transport(g)
    }
    fn primary_execution_group(&self) -> &str {
        self.0.primary_execution_group()
    }
    fn state_partition_plan(
        &self,
        s: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.0.state_partition_plan(s)
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        self.0.execution_graph()
    }
    fn group_unit_count(&self, g: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Error> {
        self.0.group_unit_count(g, metadata_context)
    }
    fn unit_path(&self, g: usize, i: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Error> {
        self.0.unit_path(g, i, metadata_context)
    }
    fn static_modules(&self) -> &FakeOperator {
        self.0.static_modules()
    }
    fn static_modules_mut(&mut self) -> &mut FakeOperator {
        self.0.static_modules_mut()
    }
    fn build_unit(&self, g: usize, i: usize, c: &()) -> Result<FakeUnit, Error> {
        self.0.build_unit(g, i, c)
    }
    fn begin_forward<'a>(
        &mut self,
        input: &'a FakeTensor,
        state: &mut State,
        c: &(),
    ) -> Result<LayeredForwardState<FakeTensor, usize>, Error> {
        let first = self.0.begin_forward(input, state, c)?;
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
        self.0
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
        let output = self.0.forward_unit(g, i, unit, hidden, state, &mut (), c)?;
        let remaining = i32::try_from(*width - 1).unwrap();
        state.as_mut()[0].0 += remaining;
        if let Some(value) = &mut state.as_mut()[0].1 {
            value.0[0] += remaining;
        }
        Ok(output)
    }
    fn select_readout_positions(
        &self,
        hidden: &FakeTensor,
        _: &usize,
        demand: OutputDemand,
        c: &(),
    ) -> Result<Option<FakeTensor>, Error> {
        self.0.select_readout_positions(hidden, &(), demand, c)
    }
    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        state: &mut State,
        _: &usize,
        c: &(),
    ) -> Result<FakeTensor, Error> {
        self.0.finish_forward(hidden, state, &(), c)
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
struct UnevenInput {
    geometry: InferenceGeometry,
    record: Rc<RefCell<Record>>,
}
impl PreparedPrefillSource<Chunked, FakeBackend, State> for UnevenInput {
    type Chunk = FakeTensor;
    fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(&self, chunk: &PrefillChunk, _: &()) -> Result<FakeTensor, Error> {
        guarded();
        let mut r = self.record.borrow_mut();
        assert!(matches!(r.events.last(),Some(Log::Open(start,_)) if *start==chunk.input.start));
        r.prepared += 1;
        r.events.push(Log::Prepare(chunk.input.start));
        Ok(FakeTensor(vec![
            7;
            usize::try_from(
                chunk.input.end - chunk.input.start
            )
            .unwrap()
        ]))
    }
    fn input<'a>(&'a self, v: &'a FakeTensor) -> &'a FakeTensor {
        v
    }
}
#[test]
fn opening_state_tracks_real_uneven_chunks_and_nonzero_cached_frontier() {
    let scopes = ScopeTrace::new();
    let counters = ReplicatedSessionCounters::default();
    let architecture = Chunked(OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: vec![],
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    });
    let selected = selected_reference_text(&architecture.0, LayerWeightResidency::FullyResident);
    let identity = selected.requirements().architecture_identity().to_owned();
    let contract = prepare_replicated_text_contract::<_, FakeBackend, State>(
        &architecture,
        None,
        selected,
        &identity,
        &(),
    )
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Default::default(),
        completions: Default::default(),
        counters: counters.clone(),
        fail_completion: Default::default(),
        fail_checkpoint: false,
        fail_construction_report: false,
        prepared_partition: None,
        prompt_cache: None,
    };
    let mut session = construct_replicated_text_session::<_, FakeBackend, _>(
        architecture,
        None,
        contract,
        mechanisms,
        &(),
    )
    .unwrap();
    let layout = session
        .inspect_runtime_state(|s| Ok(s.layout().clone()))
        .unwrap();
    let mut state = State::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState(4, Some(FakeTensor(vec![17, 29]))))
    })
    .unwrap();
    let pointer = state.as_ref()[0].1.as_ref().unwrap().0.as_ptr();
    session
        .exchange_prediction_target_state(&mut state, &())
        .unwrap();
    drop(state);
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 4,
        input_positions: 5,
        max_output_tokens: 0,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request: InferenceRequest = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(geometry),
        )
        .unwrap()
        .into();
    let record = Rc::new(RefCell::new(Record::default()));
    let mut observer = Opening::new(record.clone(), pointer);
    assert!(matches!(
        session
            .try_prefill_source_cancellable(
                Some(&request),
                Some([1, 5]),
                std::num::NonZeroU64::new(2),
                |geometry| Ok(Some(UnevenInput {
                    geometry,
                    record: record.clone()
                })),
                &GenerationCancellationToken::new(),
                &(),
                &mut observer
            )
            .unwrap(),
        PrefillSourceOutcome::Complete(_)
    ));
    assert_eq!(
        record.borrow().events,
        [
            Log::Begin(0..2),
            Log::Open(0, vec![17, 29]),
            Log::Prepare(0),
            Log::Commit,
            Log::Begin(2..4),
            Log::Open(2, vec![19, 29]),
            Log::Prepare(2),
            Log::Commit,
            Log::Begin(4..5),
            Log::Open(4, vec![21, 29]),
            Log::Prepare(4),
            Log::Commit,
            Log::Finish(true)
        ]
    );
    assert_eq!(session.report().unwrap().state_report(), &[9]);
    assert_eq!(counters.snapshot().forward_calls, 3);
    scopes.settled();
    drop((observer, request, session));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
