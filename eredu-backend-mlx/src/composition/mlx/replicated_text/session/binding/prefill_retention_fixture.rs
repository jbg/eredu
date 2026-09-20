//! Test-only typed session retained through the ordinary construction driver.
//! No production bank/context access or manufactured runtime success ticket.
use super::*;
use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
use crate::backend::runtime::residency::storage::RetainedStorage;
use eredu_architectures::prepared_execution::{
    construct_prepared_execution, PreparedExecutableAssembler, PreparedExecutableParts,
    PreparedExecutionRoutes, PreparedInferenceBlueprint, ReplicatedRoute,
};
use eredu_core::{
    capture::SharedCapturePlan, ExecutionWorkspaceEstimate, GenerationCancellationToken,
    InferenceGeometry, InputTokenCount, RuntimeStateEstimate, WorkspaceBound,
};
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::{
    prefill::{PrefillDriver, PrefillOutcome},
    replicated_session::SessionPrefill,
    working_memory::{
        quote_text_prompt_workspace, CaptureRunHostPlan, InferenceExecutionIdentity,
        InferenceRequest, InferenceWorkspaceReport, WorkingMemoryError,
    },
    ActivationObserver,
};
use std::{num::NonZeroU32, path::Path};

pub(crate) struct PrefillFixtureResult {
    pub(crate) outcome: PrefillOutcome,
    pub(crate) scores: Option<Array>,
}

trait TypedSession {
    fn row_proposal(
        &self,
        bound: eredu_runtime::layered::BoundCaptureSelection<'_>,
    ) -> Result<crate::composition::mlx::replicated_text::NativeOpeningRowsPlan, Error>;
    fn install_rows(
        &self,
        rows: &crate::composition::mlx::replicated_text::NativeOpeningRowsOwner,
    ) -> Result<(), Error>;
    fn numeric_state(&self) -> Result<Vec<(Vec<i32>, Vec<f32>)>, Error>;
    fn state_aliases(&self) -> Result<Vec<Array>, Error>;
    fn captured_quote(
        &self,
        blueprint: &PreparedInferenceBlueprint,
        bound: eredu_runtime::layered::BoundCaptureSelection<'_>,
        context: &WorkspaceContext,
    ) -> Result<InferenceWorkspaceReport, Error>;

    fn validate_opening_paths_after_forward(&self) -> Result<(), Error>;
    fn rebind_after_parameter_access(&mut self) -> Result<(), Error>;
    fn prepare_pin_snapshot(
        &self,
        source: &SharedCapturePlan,
        quote: eredu_runtime::working_memory::IncrementalInferenceQuote,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        capacity: Option<u64>,
    ) -> Result<super::super::mechanisms::OpeningPinSetup, Error>;
    fn opening_paths(&self) -> &eredu_runtime::SharedLayeredObservationPaths;
    fn inspect_fixed_opening_pair(
        &self,
        expected: Option<&eredu_runtime::SharedLayeredObservationPaths>,
    ) -> Result<(usize, usize, u64), Error>;
    fn identity(&self) -> &InferenceExecutionIdentity;
    fn inventory(&self) -> Result<RetainedStorage, Error>;
    fn settle_loaded_roots(&self) -> Result<(), Error>;
    fn state_estimate(
        &self,
        geometry: InferenceGeometry,
        dtype: std::num::NonZeroU8,
    ) -> Result<RuntimeStateEstimate, Error>;
    fn capabilities(&self) -> &eredu_core::ModelCapabilities;
    fn quote(
        &self,
        blueprint: &PreparedInferenceBlueprint,
        geometry: InferenceGeometry,
        context: &WorkspaceContext,
    ) -> Result<InferenceWorkspaceReport, Error>;
    fn validate_frontier(&self, expected: u64) -> Result<(), Error>;
    fn run(
        &mut self,
        request: InferenceRequest,
        tokens: Arc<[i32]>,
        cancellation: GenerationCancellationToken,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<PrefillFixtureResult, Error>;
}

impl<A, S> TypedSession for CompletedReplicatedText<A, S>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn row_proposal(
        &self,
        bound: eredu_runtime::layered::BoundCaptureSelection<'_>,
    ) -> Result<crate::composition::mlx::replicated_text::NativeOpeningRowsPlan, Error> {
        ErasedReplicatedTextExecutable::prepare_opening_rows(self, bound)
    }
    fn install_rows(
        &self,
        rows: &crate::composition::mlx::replicated_text::NativeOpeningRowsOwner,
    ) -> Result<(), Error> {
        ErasedReplicatedTextExecutable::install_opening_rows(self, rows)
    }
    fn state_aliases(&self) -> Result<Vec<Array>, Error> {
        self.session
            .inspect_runtime_state(|state| {
                Ok(state
                    .retained_arrays()
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>())
            })
            .map_err(|e| Error::Other(Box::new(e)))
    }
    fn numeric_state(&self) -> Result<Vec<(Vec<i32>, Vec<f32>)>, Error> {
        // Snapshot actual handles at the quiescent boundary, then read settled
        // values after that immutable inspection has ended.
        let arrays = self
            .session
            .inspect_runtime_state(|state| {
                Ok(state
                    .retained_arrays()
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>())
            })
            .map_err(|e| Error::Other(Box::new(e)))?;
        arrays
            .into_iter()
            .map(|a| {
                Ok((
                    a.shape().to_vec(),
                    a.evaluated()?
                        .try_to_vec::<f32>()
                        .map_err(|e| Error::Other(Box::new(e)))?,
                ))
            })
            .collect()
    }
    fn captured_quote(
        &self,
        blueprint: &PreparedInferenceBlueprint,
        bound: eredu_runtime::layered::BoundCaptureSelection<'_>,
        context: &WorkspaceContext,
    ) -> Result<InferenceWorkspaceReport, Error> {
        let geometry = bound.geometry();
        let state = self.project_resident_workspace_with_storage(
            NonZeroU32::new(geometry.batch_size.try_into().unwrap()).unwrap(),
            context,
        )?;
        let (mut observer, _) = crate::composition::mlx::session::capture_workspace::CaptureWorkspaceObserver::with_prefill(bound, context)
            .map_err(|e| Error::Other(Box::new(e)))?;
        let paths = self.session.shared_observation_paths().unwrap();
        let quote = match blueprint.selected().text_realization().residency() {
            eredu_runtime::LayerWeightResidency::FullyResident => blueprint
                .quote_replicated_resident_text_observed(
                    geometry,
                    &state.state,
                    context,
                    paths,
                    &mut observer,
                ),
            eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
            | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                let facts = MlxMetalWorkspaceMechanisms::current_host()?;
                let parameters = self.layerwise_workspace(facts.allocation())?;
                blueprint.quote_replicated_layerwise_text_observed(
                    geometry,
                    &state.state,
                    context,
                    &crate::composition::mlx::model::NativeLayerwiseParameters(&parameters),
                    paths,
                    &mut observer,
                )
            }
            _ => return Err(Error::Other(Box::new(WorkingMemoryError::UnknownBound))),
        };
        quote.map_err(|e| Error::Other(Box::new(e)))
    }
    fn validate_opening_paths_after_forward(&self) -> Result<(), Error> {
        let paths = self.session.shared_observation_paths().unwrap().clone();
        self.session
            .validate_prepared_observation_paths(&paths)
            .map_err(|error| Error::Other(Box::new(error)))?;
        assert!(self
            .session
            .shared_observation_paths()
            .unwrap()
            .same_storage(&paths));
        Ok(())
    }
    fn rebind_after_parameter_access(&mut self) -> Result<(), Error> {
        struct Slots;
        impl<T: eredu_nn::Tensor> eredu_nn::ParameterSlotVisitor<T> for Slots {
            fn visit_slot(&mut self, _: eredu_nn::ParameterMetadataView<'_>, _: &mut T) {}
        }
        let source = self.session.shared_observation_paths().unwrap().clone();
        let _ = self.session.visit_loaded_parameters(&mut Slots);
        assert!(matches!(
            self.session.validate_prepared_observation_paths(&source),
            Err(eredu_runtime::ReplicatedTextSessionError::PreparedObservation(
                eredu_runtime::PreparedSessionObservationError::BindingMismatch
            ))
        ));
        self.session.rebind_observation_paths()
            .map_err(|error| Error::Other(Box::new(error)))?;
        self.session.validate_prepared_observation_paths(&source)
            .map_err(|error| Error::Other(Box::new(error)))
    }
    fn opening_paths(&self) -> &eredu_runtime::SharedLayeredObservationPaths {
        self.session.shared_observation_paths().unwrap()
    }
    fn inspect_fixed_opening_pair(
        &self,
        expected: Option<&eredu_runtime::SharedLayeredObservationPaths>,
    ) -> Result<(usize, usize, u64), Error> {
        MlxReplicatedTextMechanisms::check_fixed_opening_pair_for_test(&self.session, expected)
    }
    fn prepare_pin_snapshot(
        &self,
        source: &SharedCapturePlan,
        quote: eredu_runtime::working_memory::IncrementalInferenceQuote,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        capacity: Option<u64>,
    ) -> Result<super::super::mechanisms::OpeningPinSetup, Error> {
        MlxReplicatedTextMechanisms::prepare_opening_pins_for_test(
            &self.session,
            source,
            quote,
            pool,
            self.capability_estimate().capabilities(),
            capacity,
        )
    }
    fn identity(&self) -> &InferenceExecutionIdentity {
        self.session.inference_execution_identity()
    }
    fn inventory(&self) -> Result<RetainedStorage, Error> {
        let mut storage = self.retained_target_storage()?;
        storage.merge(self.retained_idle_auxiliary_storage()?)?;
        Ok(storage)
    }
    fn settle_loaded_roots(&self) -> Result<(), Error> {
        // Same actual retained roots used for loading finalization. No editable
        // parameter visitor, state mutation or synthetic completion proof.
        let roots = self
            .inventory()?
            .into_retained_arrays()
            .map_err(|(cause, _storage)| cause)?
            .collect::<Vec<_>>();
        async_eval_with_event(roots.iter())?.synchronize()?;
        self.stream.synchronize()?;
        Ok(())
    }
    fn state_estimate(
        &self,
        geometry: InferenceGeometry,
        dtype: std::num::NonZeroU8,
    ) -> Result<RuntimeStateEstimate, Error> {
        eredu_core::estimate_runtime_state(
            self.capability_estimate().state_layout(),
            InputTokenCount::text(geometry.cached_positions + geometry.input_positions),
            geometry.max_output_tokens,
            geometry.batch_size,
            dtype,
        )
        .map_err(|error| Error::Other(Box::new(error)))
    }
    fn capabilities(&self) -> &eredu_core::ModelCapabilities {
        self.capability_estimate().capabilities()
    }
    fn quote(
        &self,
        blueprint: &PreparedInferenceBlueprint,
        geometry: InferenceGeometry,
        context: &WorkspaceContext,
    ) -> Result<InferenceWorkspaceReport, Error> {
        let projected = self.project_resident_workspace_with_storage(
            NonZeroU32::new(
                u32::try_from(geometry.batch_size).map_err(|e| Error::Other(Box::new(e)))?,
            )
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)))?,
            context,
        )?;
        blueprint
            .quote_replicated_resident_text(geometry, &projected.state, context)
            .map_err(|error| Error::Other(Box::new(error)))
    }
    fn validate_frontier(&self, expected: u64) -> Result<(), Error> {
        self.validate_text_frontier(expected)
    }
    fn run(
        &mut self,
        request: InferenceRequest,
        tokens: Arc<[i32]>,
        cancellation: GenerationCancellationToken,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<PrefillFixtureResult, Error> {
        let geometry = request.geometry();
        let source =
            eredu_architectures::prefill::PreparedTextPrefill::<MlxTensor>::from_token_ids(
                tokens, geometry,
            )
            .map_err(|error| Error::Other(Box::new(error)))?;
        let mut driver =
            PrefillDriver::new(self.identity(), request.clone(), geometry, cancellation)
                .map_err(|error| Error::Other(Box::new(error)))?;
        let mut terminal = Terminal {
            observer,
            finished: false,
        };
        let mut neutral =
            crate::composition::NeutralActivationObserver::new(&mut *terminal.observer);
        let mut executor = SessionPrefill::new(
            &mut self.session,
            source,
            request,
            &self.stream,
            &mut neutral,
        )
        .map_err(|error| Error::Other(Box::new(error)))?;
        let mut scores = None;
        let result = driver.run(&mut executor, |_, output| {
            if let Some(output) = output {
                assert!(
                    scores.is_none(),
                    "one full Sequence chunk supplies one output"
                );
                scores = Some(output.into_array());
            }
        });
        drop(executor);
        drop(neutral);
        match result {
            Ok(outcome) => {
                terminal.finish(outcome == PrefillOutcome::Complete);
                Ok(PrefillFixtureResult { outcome, scores })
            }
            Err(eredu_runtime::prefill::PrefillError::Submission(error)) => {
                Err(Error::Other(Box::new(error)))
            }
            Err(eredu_runtime::prefill::PrefillError::Completion(never)) => match never {},
            Err(error) => Err(Error::Other(Box::new(error))),
        }
    }
}

struct Terminal<'a> {
    observer: &'a mut dyn ActivationObserver<Array, Error>,
    finished: bool,
}
impl Terminal<'_> {
    fn finish(&mut self, committed: bool) {
        self.finished = true;
        self.observer.finish_prefill(committed);
    }
}
impl Drop for Terminal<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.observer.finish_prefill(false);
        }
    }
}

#[derive(Clone, Copy)]
struct Visitor<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
}
impl<S> ReplicatedTextArchitectureVisitor<MlxNeuralBackend, S> for Visitor<'_>
where
    S: MlxStateMechanisms + 'static,
{
    type Output = Box<dyn TypedSession>;
    type Error = Error;
    fn construction_started(&mut self) {
        crate::tests::support::path_instrumentation::architecture_construction();
    }
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        CompletedReplicatedText::<A, S>::new(prepared, store, self.stream, self.weights_stream)
            .map(|session| Box::new(session) as Box<dyn TypedSession>)
    }
}
struct Assembler {
    discovery: eredu_core::capture::CaptureDiscovery,
}
impl PreparedExecutableAssembler<()> for Assembler {
    type Executable = Box<dyn TypedSession>;
    type Output = PrefillRetentionFixture;
    type Error = Error;
    fn floating_state_dtype(
        &mut self,
        source: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Error> {
        floating_state_storage_dtype(source.dtype())
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::UnknownBound)))
    }
    fn validate_communication(
        &mut self,
        _: &eredu_runtime::CommunicationManifest,
        _: &(),
    ) -> Result<(), Error> {
        Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)))
    }
    fn finish(
        self,
        parts: PreparedExecutableParts<Self::Executable, ()>,
    ) -> Result<Self::Output, Error> {
        let blueprint = parts.inference_blueprint().clone();
        let dtype = parts.floating_state_bytes();
        Ok(PrefillRetentionFixture {
            session: parts.into_executable(),
            blueprint,
            dtype,
            discovery: self.discovery,
        })
    }
}

pub(crate) struct PrefillRetentionFixture {
    session: Box<dyn TypedSession>,
    blueprint: PreparedInferenceBlueprint,
    dtype: std::num::NonZeroU8,
    discovery: eredu_core::capture::CaptureDiscovery,
}
impl PrefillRetentionFixture {
    pub(crate) fn opening_source(&self, geometry: InferenceGeometry) -> SharedCapturePlan {
        opening_pins::source(self, geometry)
    }
    pub(crate) fn opening_selection(
        &self,
        source: &SharedCapturePlan,
    ) -> Result<eredu_runtime::layered::PreparedCaptureSelection, Error> {
        self.session
            .opening_paths()
            .prepare_capture_selection(source)
            .map_err(|e| Error::Other(Box::new(e)))
    }
    pub(crate) fn state_aliases(&self) -> Result<Vec<Array>, Error> {
        self.session.state_aliases()
    }
    pub(crate) fn inactive_opening_source(
        &self,
        geometry: InferenceGeometry,
        mode: usize,
    ) -> SharedCapturePlan {
        use eredu_core::capture::*;
        let original = self.opening_source(geometry);
        let mut plan = original.admission().plan().clone();
        match mode {
            0 => plan.selections.clear(),
            1 => plan.selections[0].schedule.prefill = false,
            _ => plan.selections[0].schedule.first_prediction = 1,
        }
        SharedCapturePlan::new(
            plan.admit_with_text_origin(
                &self.discovery.catalog,
                &self.discovery.support,
                &self.discovery.support.capture,
                CaptureRequestShape {
                    batch: geometry.batch_size,
                    prompt_tokens: geometry.input_positions,
                    max_predictions: geometry.max_output_tokens,
                },
                CaptureTextOrigin {
                    cached_positions: geometry.cached_positions,
                },
            )
            .unwrap(),
        )
    }
    pub(crate) fn opening_rows_proposal(
        &self,
        bound: eredu_runtime::layered::BoundCaptureSelection<'_>,
    ) -> Result<crate::composition::mlx::replicated_text::NativeOpeningRowsPlan, Error> {
        self.session.row_proposal(bound)
    }
    pub(crate) fn install_opening_rows(
        &self,
        rows: &crate::composition::mlx::replicated_text::NativeOpeningRowsOwner,
    ) -> Result<(), Error> {
        self.session.install_rows(rows)
    }
    pub(crate) fn numeric_state(&self) -> Result<Vec<(Vec<i32>, Vec<f32>)>, Error> {
        self.session.numeric_state()
    }
    pub(crate) fn rebind_after_parameter_access(&mut self) -> Result<(), Error> {
        self.session.validate_opening_paths_after_forward()?;
        self.session.rebind_after_parameter_access()
    }
    pub(crate) fn opening_rows_quote(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        geometry: InferenceGeometry,
        source: &SharedCapturePlan,
        tokens: &Arc<[i32]>,
    ) -> Result<eredu_runtime::working_memory::IncrementalInferenceQuote, Error> {
        use eredu_runtime::working_memory::{RegisteredWorkspaceStorage, ResidualInferenceQuote};
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host()?);
        // Bind before any projection/equation can start the trace.
        let storage = RegisteredWorkspaceStorage::<u32>::bind(pool, &context, [])
            .map_err(|e| Error::Other(Box::new(e)))?;
        let selected = self.opening_selection(source)?;
        let bound = selected
            .bind_geometry(geometry)
            .map_err(|e| Error::Other(Box::new(e)))?;
        let equations = self
            .session
            .captured_quote(&self.blueprint, bound, &context)?;
        let state = self.session.state_estimate(geometry, self.dtype)?;
        let state = equations
            .refine_state_backing(state)
            .map_err(|e| Error::Other(Box::new(e)))?;
        let zero = || {
            WorkspaceBound::bounded(0, "actual selected equation and native capture trace includes work; fixture has no sampling")
        };
        let outside = ExecutionWorkspaceEstimate {
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: WorkspaceBound::bounded(
                CaptureRunHostPlan::prepare(source)
                    .map_err(|e| Error::Other(Box::new(e)))?
                    .initialization_peak_bytes(),
                "actual original scheduled H",
            ),
        };
        let outside = quote_text_prompt_workspace(
            geometry,
            Some(std::mem::size_of_val(tokens.as_ref()) as u64),
            &context,
        )
        .map_err(|e| Error::Other(Box::new(e)))?
        .compose(outside)
        .map_err(|e| Error::Other(Box::new(e)))?;
        ResidualInferenceQuote::compose(&equations, state, outside, &storage)
            .map(|q| q.into_incremental())
            .map_err(|e| Error::Other(Box::new(e)))
    }
    pub(crate) fn load(artifact: &Path, stream: &Stream) -> Result<Self, Error> {
        Self::load_with_options(artifact, stream, crate::MlxLoadRequest::default())
    }
    pub(crate) fn load_with_options(
        artifact: &Path,
        stream: &Stream,
        options: crate::MlxLoadRequest,
    ) -> Result<Self, Error> {
        let inspection = eredu_architectures::configuration::inspect_artifact(artifact)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let policy = options
            .normalized()
            .preparation_policy()
            .map_err(|error| Error::Other(Box::new(error)))?;
        let selected = crate::composition::mlx::loading::select_preparation(&inspection, options)?;
        let plan =
            eredu_core::plan_model_preparation(inspection, policy, selected.session_capabilities())
                .map_err(|error| Error::Other(Box::new(error)))?;
        let (sources, rank) =
            crate::composition::mlx::loading::prepare_selected_sources(plan, selected, None)?;
        assert!(rank.is_none(), "fixture is ordinary resident construction");
        let weights_stream =
            Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let routes = PreparedExecutionRoutes::new().with_replicated(ReplicatedRoute::<
            MlxNeuralBackend,
            _,
        >::new(
            stream,
            &weights_stream,
            eredu_architectures::replicated_text::SharedReplicatedTextVisitor::<
                MlxReplicatedStateProfiles,
                _,
            >::new(Visitor {
                stream,
                weights_stream: &weights_stream,
            }),
        ));
        let discovery = sources
            .capture_discovery(
                eredu_core::ObservationMechanisms {
                    activation_tensors: true,
                    floating_to_f32: true,
                    ..Default::default()
                },
                eredu_core::capture::CaptureCapabilities {
                    transformations: vec![eredu_core::capture::CaptureTransformKind::FullTensor],
                    ..Default::default()
                },
            )
            .map_err(|error| Error::Other(Box::new(error)))?;
        let fixture =
            construct_prepared_execution(sources, None::<()>, routes, Assembler { discovery })
                .map_err(|error| Error::Other(Box::new(error)))?;
        fixture.session.settle_loaded_roots()?;
        Ok(fixture)
    }
    pub(crate) fn identity(&self) -> &InferenceExecutionIdentity {
        self.session.identity()
    }
    pub(crate) fn inventory(&self) -> Result<RetainedStorage, Error> {
        self.session.inventory()
    }
    pub(crate) fn validate_frontier(&self, expected: u64) -> Result<(), Error> {
        self.session.validate_frontier(expected)
    }
    /// Full conservative equation + prompt + actual scheduled-H composition.
    /// Existing registered sources remain charged separately; no residual credit.
    /// This fixture performs no sampling and only copies an already-settled F32
    /// external observation into its existing host destination, with no transforms.
    pub(crate) fn quote(
        &self,
        geometry: InferenceGeometry,
        capture: &SharedCapturePlan,
        tokens: &Arc<[i32]>,
    ) -> Result<RuntimeStateEstimate, Error> {
        geometry
            .validate()
            .map_err(|error| Error::Other(Box::new(error)))?;
        assert_eq!(geometry.input_positions, geometry.prefill_chunk_positions);
        assert_eq!(
            tokens.len() as u64,
            geometry.batch_size * geometry.input_positions
        );
        assert_eq!(capture.admission().request().batch, geometry.batch_size);
        assert_eq!(
            capture.admission().request().prompt_tokens,
            geometry.input_positions
        );
        assert_eq!(
            capture.admission().request().max_predictions,
            geometry.max_output_tokens
        );
        assert_eq!(
            capture.admission().text_origin().unwrap().cached_positions,
            geometry.cached_positions
        );
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host()?);
        let equation = self.session.quote(&self.blueprint, geometry, &context)?;
        let state = self.session.state_estimate(geometry, self.dtype)?;
        let state = equation
            .refine_state_backing(state)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let h = CaptureRunHostPlan::prepare(capture)
            .map_err(|error| Error::Other(Box::new(error)))?
            .initialization_peak_bytes();
        let zero = || {
            WorkspaceBound::bounded(0, "covered by actual resident equation trace; no materialization, sampling or native capture transformation in this fixture")
        };
        let outside = ExecutionWorkspaceEstimate {
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: WorkspaceBound::bounded(
                h,
                "actual original-bank cumulative capture host plan",
            ),
        };
        let prompt = quote_text_prompt_workspace(
            geometry,
            Some(std::mem::size_of_val(tokens.as_ref()) as u64),
            &context,
        )
        .map_err(|error| Error::Other(Box::new(error)))?;
        let outside = prompt
            .compose(outside)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let state = equation
            .compose(state, outside)
            .map_err(|error| Error::Other(Box::new(error)))?;
        Ok(state)
    }
    pub(crate) fn capabilities(&self) -> &eredu_core::ModelCapabilities {
        self.session.capabilities()
    }
    pub(crate) fn run(
        &mut self,
        request: InferenceRequest,
        tokens: Arc<[i32]>,
        cancellation: GenerationCancellationToken,
        observer: &mut dyn ActivationObserver<Array, Error>,
    ) -> Result<PrefillFixtureResult, Error> {
        self.session.run(request, tokens, cancellation, observer)
    }
}

#[test]
fn actual_typed_native_session_lends_its_own_fixed_opening_pair() {
    use crate::backend::managed_memory::NativeMemoryOwner;
    use eredu_runtime::working_memory::WorkingMemoryPool;
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading_owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", false);
    let fixture = PrefillRetentionFixture::load(artifact.path(), &stream).unwrap();
    let before = pool.used_bytes().unwrap();
    let (retained, slots, controls) = fixture.session.inspect_fixed_opening_pair(None).unwrap();
    assert!(retained > 0);
    assert!(slots >= retained);
    assert!(controls > 0);
    assert_eq!(
        pool.used_bytes().unwrap(),
        before,
        "cold pairing creates no grant or hold"
    );
    fixture.validate_frontier(0).unwrap();
    let foreign = PrefillRetentionFixture::load(artifact.path(), &stream).unwrap();
    assert!(!fixture
        .session
        .opening_paths()
        .same_storage(foreign.session.opening_paths()));
    let before = pool.used_bytes().unwrap();
    assert!(fixture
        .session
        .inspect_fixed_opening_pair(Some(foreign.session.opening_paths()))
        .is_err());
    assert_eq!(pool.used_bytes().unwrap(), before);
    fixture.validate_frontier(0).unwrap();
    drop(foreign);
    drop(fixture);
    drop(loading_owner);
}

#[path = "prefill_retention_fixture/opening_pins.rs"]
mod opening_pins;
