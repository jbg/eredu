//! Exact integer causal fixture: cumulative prefix sum, one retained accumulator,
//! an independent retained sentinel and a position. Test vectors are not native
//! allocation facts; actual runtime admission/driver/collector contracts are used.
use super::*;
pub(super) struct Rows(OrdinaryTextFixture);
impl ArchitectureParameters<FakeBackend> for Rows {
    type DefinitionError = Error;
    fn state_layout(
        &self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Error> {
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
    fn parameter_description(
        &self,
        c: &(),
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Error> {
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
impl LayeredArchitecture<FakeBackend, State> for Rows {
    type Input<'a> = &'a FakeTensor;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = usize;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;
    fn prefill_observation_declarations(
        &self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Vec<PrefillObservationDeclaration>, Error> {
        let mut rows = match metadata {
            Some(context) => context.metadata_vec(PATHS.len())?,
            None => Vec::with_capacity(PATHS.len()),
        };
        for (i, &path) in PATHS.iter().enumerate() {
            rows.push(PrefillObservationDeclaration::causal_ordinary_text(
                architecture_metadata::text(format_args!("{path}"), metadata)?,
                0,
                if i < 4 {
                    PrefillReadoutStage::BeforeReadout
                } else {
                    PrefillReadoutStage::VocabularyScores
                },
            ));
        }
        Ok(rows)
    }
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
    fn group_unit_count(
        &self,
        g: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Error> {
        self.0.group_unit_count(g, metadata_context)
    }
    fn unit_path(
        &self,
        g: usize,
        i: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<String, Error> {
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
        _: usize,
        _: usize,
        unit: &mut FakeUnit,
        hidden: &FakeTensor,
        state: &mut State,
        _: &mut usize,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        check_inference_scope();
        self.0.counters.update(|counts| counts.forward_calls += 1);
        let state = &mut state.as_mut()[0];
        let accumulator = &mut state.1.as_mut().expect("real seeded state").0[0];
        let output = hidden
            .0
            .iter()
            .map(|&value| {
                *accumulator += value;
                *accumulator + unit.marker
            })
            .collect();
        state.0 += i32::try_from(hidden.0.len()).unwrap();
        Ok(FakeTensor(output))
    }
    fn select_readout_positions(
        &self,
        hidden: &FakeTensor,
        _: &usize,
        demand: OutputDemand,
        _: &(),
    ) -> Result<Option<FakeTensor>, Error> {
        Ok(match demand {
            OutputDemand::StateOnly => None,
            OutputDemand::Sequence => Some(hidden.clone()),
            OutputDemand::LastPosition => Some(FakeTensor(vec![*hidden.0.last().unwrap()])),
        })
    }
    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        _: &mut State,
        _: &usize,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        Ok(FakeTensor(hidden.0.iter().map(|v| 2 * v + 1).collect()))
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
impl ReplicatedTextArchitecture<FakeBackend, State> for Rows {
    fn text_input<'a>(tokens: &'a FakeTensor, _: Option<&'a FakeTensor>) -> &'a FakeTensor {
        tokens
    }
}

pub(super) fn session() -> Session {
    let counters = ReplicatedSessionCounters::default();
    let architecture = Rows(OrdinaryTextFixture {
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
        counters,
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
        Ok::<_, Infallible>(FakeLayerState(0, Some(FakeTensor(vec![0, 29]))))
    })
    .unwrap();
    session
        .exchange_prediction_target_state(&mut state, &())
        .unwrap();
    struct Prefix(Vec<String>);
    impl ActivationObserver<FakeTensor, Error> for Prefix {
        fn requires_prepared_traversal(&self) -> bool {
            true
        }
        fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Error> {
            assert!(value.0.iter().all(|value| *value != 0));
            self.0.push(path.into());
            Ok(())
        }
    }
    let mut prefix = Prefix(vec![]);
    let output = session
        .prefill_with_observer(&FakeTensor(vec![8, 9]), None, &(), &mut prefix)
        .unwrap();
    assert_eq!(output.0, vec![45]);
    assert_eq!(self::state(&session), (2, vec![17, 29]));
    for path in PATHS {
        assert_eq!(
            prefix
                .0
                .iter()
                .filter(|value| value.as_str() == path)
                .count(),
            1
        );
    }
    session
}
pub(super) fn state(session: &Session) -> (i32, Vec<i32>) {
    session
        .inspect_runtime_state(|s| {
            Ok((s.as_ref()[0].0, s.as_ref()[0].1.as_ref().unwrap().0.clone()))
        })
        .unwrap()
}
pub(super) fn source(g: InferenceGeometry, body: bool) -> SharedCapturePlan {
    let names = if body { &PATHS[..4] } else { &PATHS[..] };
    let points = names
        .iter()
        .map(|&path| ObservationPoint {
            path: path.into(),
            node_id: "fixture".into(),
            meaning: "actual fixture hook".into(),
            value_type: ObservationValueType::Tensor,
            dtype: ObservationDtype::Floating,
            axes: Some(vec![TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            }]),
            prefill: true,
            decode: true,
            requirements: vec![],
            position: if path.ends_with(".effective") {
                ObservationPosition::AfterIntervention
            } else {
                ObservationPosition::BeforeIntervention
            },
            retained_bytes: None,
            host_bytes: None,
        })
        .collect();
    let catalog = ObservationCatalog {
        schema_version: 1,
        points,
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: names
            .iter()
            .map(|&path| ObservationSupport {
                path: path.into(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    let mut plan = CapturePlan::none();
    plan.selections = names
        .iter()
        .map(|&path| CaptureSelection {
            id: path.into(),
            path: path.into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        })
        .collect();
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    plan.limits.per_step = unlimited;
    plan.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &catalog,
            &support,
            &CaptureCapabilities {
                transformations: vec![CaptureTransformKind::FullTensor],
                ..Default::default()
            },
            CaptureRequestShape {
                batch: g.batch_size,
                prompt_tokens: g.input_positions,
                max_predictions: g.max_output_tokens,
            },
            CaptureTextOrigin {
                cached_positions: g.cached_positions,
            },
        )
        .unwrap(),
    )
}
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory::topology_ref())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory::placement_ref())
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::memory::placement_ref())
    }

    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|l| l.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "exact scalar equation fixture outputs".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no disjoint scalar workspace".into(),
        }))
    }
}
pub(super) fn quote(
    pool: &MemoryLedger,
    source: &SharedCapturePlan,
    g: InferenceGeometry,
) -> IncrementalInferenceQuote {
    let context = WorkspaceContext::new(Facts);
    let backing = WorkspaceExistingStorage::try_new_placed(
        Some(64),
        crate::memory::placement_ref(),
        &context,
    )
    .unwrap();
    let tensor = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(),
        &backing,
        &context,
    )
    .unwrap();
    let storage = RegisteredWorkspaceStorage::bind(pool, &context, [(1u32, backing)]).unwrap();
    let report = quote_inference_workspace(g, |_| {
        context.begin_state_span([&tensor])?;
        let output = tensor.add(&tensor, &context)?;
        context.report(&[output])
    })
    .unwrap();
    let h = CaptureRunHostPlan::prepare(source)
        .unwrap()
        .initialization_peak_bytes();
    let b = |n| WorkspaceBound::bounded(n, "neutral scalar fixture enclosing owner");
    let outside = crate::memory::workspace(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry: g,
        activations: b(4096),
        attention: b(0),
        vocabulary: b(0),
        state_update: b(0),
        materialization: b(0),
        retained: b(h),
    });
    ResidualInferenceQuote::compose(
        &report,
        mock_inference_admission(g).state,
        outside,
        &storage,
    )
    .unwrap()
    .into_incremental()
}
/// Each planner candidate receives its own complete metadata traversal and
/// original seal, with the same physical source/path owners and join policy.
/// This is still cold quotation, before any successful reservation or native work.
pub(super) fn candidate_quote(
    pool: &MemoryLedger,
    source: &SharedCapturePlan,
    selection: &PreparedCaptureSelection,
    g: InferenceGeometry,
    seal: bool,
    explicit_source: Option<&WorkingMemoryStorage<u32>>,
) -> IncrementalInferenceQuote {
    let q = quote(pool, source, g);
    let q = if let Some(root) = explicit_source {
        q.with_registered_sources(root.clone()).unwrap()
    } else {
        q
    };
    let mut c = controls(&q, source);
    if seal {
        c = c
            .with_prefill_capture_selection(selection.bind_geometry(g).unwrap())
            .unwrap();
    }
    let before = q.incremental_bytes().unwrap();
    let q = q.with_span_workspace_and_text_controls(c).unwrap();
    assert_eq!(q.geometry(), g);
    assert_eq!(
        q.incremental_bytes().unwrap(),
        before + q.span_workspace().retention_peak_bytes().unwrap()
    );
    q
}

pub(super) fn accept(
    session: &Session,
    pool: &MemoryLedger,
    g: InferenceGeometry,
    capacity: u64,
    mut candidate_quote: impl FnMut(InferenceGeometry) -> IncrementalInferenceQuote,
) -> Result<(WorkingMemoryReservation, IncrementalInferenceQuote), PrefillPlanningError> {
    let caps = ModelCapabilities {
        effective_model_type: "ordinary-text-fixture".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    plan_prefill_incremental_with_capacity(
        session.inference_execution_identity(),
        pool,
        &caps,
        AdmissionRequest {
            input: InputTokenCount::text(g.cached_positions + g.input_positions),
            max_output_tokens: g.max_output_tokens,
            batch_size: 1,
            additional_headroom: Default::default(),
            memory_limits: Default::default(),
        },
        g,
        crate::memory::resolved_limits(capacity),
        |candidate| Ok(candidate_quote(candidate)),
    )
}
pub(super) fn controls(
    q: &IncrementalInferenceQuote,
    source: &SharedCapturePlan,
) -> PreparedTextControlWorkspace {
    PreparedTextControlWorkspace::prepare(
        source,
        q.geometry(),
        q.span_workspace().plan(),
        TextHostControlFacts::new(Some(0), Some(0), Some(0)),
    )
    .unwrap()
}
