//! Actual runtime text admission for tensor-free conformance models.
use super::original_sources::{Environment, Prompt};
use eredu_core::*;
use eredu_nn::workspace::*;
use eredu_runtime::working_memory::*;
use std::{alloc::Layout, cell::RefCell, mem::size_of, num::NonZeroU8, rc::Rc};

pub(super) struct Preparation {
    controller: Option<(ControllerStorageContract, TextControllerContract)>,
    pub request: InferenceTextPreparation,
    first_prediction: u64,
    owner: RefCell<Option<OwnedTextSpanWorkspace>>,
    _resumed: Option<OwnedInferenceSpanWorkspace>,
    sequence: RefCell<Option<OriginalGenerationSequenceBank>>,
    capture: RefCell<Option<PreparedCaptureRun>>,
    input: RefCell<Option<OwnedPromptTokenIds>>,
    _run: WorkingMemoryFundingRun,
    _planning: Option<HostMetadataFunding>,
}
/// The final shared shell retires before its request payer.
pub(super) struct PreparationOwner(Option<Rc<Preparation>>);
impl Clone for PreparationOwner {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl std::ops::Deref for PreparationOwner {
    type Target = Preparation;
    fn deref(&self) -> &Preparation {
        self.0.as_ref().unwrap()
    }
}
impl Drop for PreparationOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
#[derive(Debug)]
struct NoOperations;
impl WorkspaceMechanisms for NoOperations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        unreachable!("neutral fixture has no tensor operations")
    }
}
impl WorkspaceFactMechanisms for NoOperations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        unreachable!("no tensor equation")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        unreachable!("no tensor equation")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        unreachable!("no tensor equation")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        unreachable!("no tensor equation")
    }
}
#[path = "admitted_text/resume.rs"]
mod resume;
fn memory(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct FundedError<E: std::error::Error + Send + Sync + 'static> {
    #[source]
    cause: E,
    _funding: HostMetadataFunding,
}
pub(super) fn funded_error<E: std::error::Error + Send + Sync + 'static>(
    error: E,
    funding: &HostMetadataFunding,
) -> BackendFailure {
    let Some(bytes) = BackendFailure::source_retention_peak_bytes::<FundedError<E>>() else {
        return HostMetadataFundingError::Overflow.into();
    };
    if let Err(refusal) = funding.reserve_metadata(bytes) {
        return refusal.into();
    }
    BackendFailure::from_error(FundedError {
        cause: error,
        _funding: funding.clone(),
    })
}
fn optional_error<E: std::error::Error + Send + Sync + 'static>(
    error: E,
    funding: Option<&HostMetadataFunding>,
) -> BackendFailure {
    match funding {
        Some(funding) => funded_error(error, funding),
        None => memory(error),
    }
}
fn quote(
    env: &Environment,
    geometry: InferenceGeometry,
    mask_bytes: u64,
    capture_bytes: u64,
    funding: Option<&HostMetadataFunding>,
) -> Result<IncrementalInferenceQuote, BackendFailure> {
    let context = match funding {
        Some(funding) => WorkspaceContext::new_with_metadata_funding(NoOperations, funding.clone())
            .map_err(|e| funded_error(e, funding))?,
        None => WorkspaceContext::new(NoOperations),
    };
    let storage = match funding {
        Some(funding) => {
            let layout = RegisteredWorkspaceStorageLayout::<u32>::new(0)
                .map_err(|e| funded_error(e, funding))?;
            funding.reserve_metadata(layout.requested_bytes())?;
            layout
                .construct(&env.pool, &context, std::iter::empty())
                .map_err(|e| funded_error(e, funding))?
        }
        None => RegisteredWorkspaceStorage::bind(
            &env.pool,
            &context,
            std::iter::empty::<(u32, WorkspaceExistingStorage)>(),
        )
        .map_err(memory)?,
    };
    let report = quote_inference_workspace_with_context(geometry, &context, |_| {
        context.begin_state_span([])?;
        context.finish_report(&[])
    })
    .map_err(|e| optional_error(e, funding))?;
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .map_err(|e| optional_error(e, funding))?;
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(
            geometry
                .cached_positions
                .checked_add(geometry.input_positions)
                .ok_or_else(|| optional_error(WorkingMemoryError::Overflow, funding))?,
        ),
        geometry.max_output_tokens,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .map_err(|e| optional_error(e, funding))?;
    if let Some(funding) = funding {
        funding.reserve_metadata(
            4 * "the fixture has no neural storage or tensor operation".len()
                + "actual finite capture host schedule".len()
                + "actual controller filter; fixture logits are retained caller input".len()
                + size_of::<ExecutionWorkspaceEstimate>(),
        )?;
    }
    let zero =
        || WorkspaceBound::bounded(0, "the fixture has no neural storage or tensor operation");
    let outside = ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: WorkspaceBound::bounded(
            mask_bytes,
            "actual controller filter; fixture logits are retained caller input",
        ),
        state_update: zero(),
        materialization: zero(),
        retained: WorkspaceBound::bounded(capture_bytes, "actual finite capture host schedule"),
    };
    ResidualInferenceQuote::compose_metadata(
        &report,
        state,
        outside,
        &storage,
        funding.map_or_else(
            WorkspaceReportMetadata::ordinary,
            WorkspaceReportMetadata::with_funding,
        ),
    )
    .map(ResidualInferenceQuote::into_incremental)
    .map_err(|e| optional_error(e, funding))
}
impl Preparation {
    pub fn admit<C: TokenFilterController>(
        env: &Environment,
        config: TextGenerationConfig,
        controller_input: &C,
        claim: &GenerationSequencePreparation<'_, '_>,
        output_width: usize,
        capture: Option<CaptureRunHostPlan<'_>>,
    ) -> Result<PreparationOwner, BackendFailure> {
        let capacity = config
            .inference_policy()
            .managed_memory_capacity_bytes
            .ok_or_else(|| TokenInputRejection::Unsupported.into_backend_failure())?;
        let planning = env
            .pool
            .prepare_workspace_metadata(&env.execution, capacity)?;
        planning.reserve_metadata(
            size_of::<Self>()
                + size_of::<IncrementalInferenceQuote>()
                + size_of::<PreparedTextControlWorkspace>()
                + size_of::<ModelCapabilities>()
                + size_of::<Result<PreparationOwner, BackendFailure>>()
                + size_of::<CaptureRunHostPlan<'_>>()
                + size_of::<Option<PreparedCaptureRun>>()
                + size_of::<HostMetadataFunding>(),
        )?;
        let input = claim
            .request()
            .token_input()
            .ok_or_else(|| TokenInputRejection::Unsupported.into_backend_failure())?;
        let controller = if let Some(workspace) =
            controller_input.inference_workspace(claim.request().max_new_tokens() as u64)
        {
            Some((
                ControllerStorageContract::inspect_original_sequence(
                    controller_input,
                    workspace,
                    &env.pool,
                    &env.execution,
                    claim,
                )?,
                TextControllerContract::from_workspace(workspace, output_width).map_err(memory)?,
            ))
        } else {
            None
        };
        let mut decoder = OriginalGenerationDecoderSource::take_original(claim, &env.pool)?;
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: input.tokens().len() as u64,
            max_output_tokens: claim.request().max_new_tokens() as u64,
            prefill_chunk_positions: input.tokens().len().max(1) as u64,
            output: OutputDemand::LastPosition,
        };
        if let Some(capture) = &capture {
            let request = capture.source().admission().request();
            if request.batch != geometry.batch_size
                || request.prompt_tokens != geometry.input_positions
                || request.max_predictions != geometry.max_output_tokens
            {
                return Err(TokenInputRejection::IdentityMismatch.into_backend_failure());
            }
        }
        let quote_candidate = |geometry| -> Result<IncrementalInferenceQuote, BackendFailure> {
            let base = quote(
                env,
                geometry,
                controller
                    .as_ref()
                    .map_or(0, |(_, c)| c.filter_capacity_bytes()),
                capture
                    .as_ref()
                    .map_or(0, CaptureRunHostPlan::initialization_peak_bytes),
                Some(&planning),
            )?;
            let shell = Layout::new::<[usize; 2]>()
                .extend(Layout::new::<Self>())
                .unwrap()
                .0
                .pad_to_align()
                .size();
            let controls = PreparedTextControlWorkspace::prepare_sequence(
                claim,
                geometry,
                base.span_workspace().plan(),
                TextHostControlFacts::new(
                    Some(
                        (shell
                            + size_of::<Self>()
                            + size_of::<Result<PreparationOwner, BackendFailure>>())
                            as u64,
                    ),
                    Some(0),
                    Some((Prompt::construction_bytes() + size_of::<Step>()) as u64),
                ),
            )
            .map_err(memory)?;
            let quoted = base
                .with_span_workspace_and_text_controls(controls)
                .map_err(memory)?;
            Ok(quoted)
        };
        let capacity = config
            .inference_policy()
            .managed_memory_capacity_bytes
            .ok_or_else(|| TokenInputRejection::Unsupported.into_backend_failure())?;
        planning.reserve_metadata(
            "neutral public conformance fixture".len() + 2 * "no model context storage".len(),
        )?;
        let caps = ModelCapabilities {
            effective_model_type: "neutral public conformance fixture".into(),
            native_max_context: Observed::exact(u64::MAX, "no model context storage"),
            effective_max_context: Observed::exact(u64::MAX, "no model context storage"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        let admission = AdmissionRequest {
            input: InputTokenCount::text(geometry.input_positions),
            max_output_tokens: geometry.max_output_tokens,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        planning.reserve_metadata(
            size_of::<Option<BackendFailure>>()
                + size_of::<Result<IncrementalInferenceQuote, BackendFailure>>(),
        )?;
        let mut quote_failure = None;
        let planned = plan_prefill_incremental_with_capacity(
            &env.execution,
            &env.pool,
            &caps,
            admission,
            geometry,
            capacity,
            |candidate| match quote_candidate(candidate) {
                Ok(quote) => Ok(quote),
                Err(error) => {
                    quote_failure = Some(error);
                    Err(PrefillPlanningError::Reservation(
                        WorkingMemoryError::UnknownBound,
                    ))
                }
            },
        );
        if let Some(error) = quote_failure {
            return Err(error);
        }
        let (reservation, accepted) = planned.map_err(|e| funded_error(e, &planning))?;
        let (reservation, run) = reservation.into_funding().map_err(memory)?;
        let (mut owner, witness) = accepted
            .into_funded_text_span_workspace(&run, &reservation)
            .map_err(memory)?;
        drop(witness);
        let mut sequence = owner.take_generation_sequence_bank().unwrap();
        if decoder.is_some() {
            sequence = sequence.with_decoder_source(&mut decoder)?;
        }
        // Installation consumes the same original grammar/receipt authenticated
        // above; it does not register or copy a cold controller source.
        if let Some((storage, _)) = &controller {
            storage
                .prepare_original_source(controller_input, &run, &reservation)
                .map_err(memory)?;
        }
        let capture = capture
            .map(|plan| run.prepare_capture_run(&reservation, plan))
            .transpose()
            .map_err(memory)?;
        let request = InferenceRequest::from(&reservation)
            .prepare_text(&env.execution, geometry, config)
            .map_err(memory)?;
        Ok(PreparationOwner(Some(Rc::new(Self {
            controller,
            request,
            first_prediction: 0,
            owner: RefCell::new(Some(owner)),
            _resumed: None,
            sequence: RefCell::new(Some(sequence)),
            capture: RefCell::new(capture),
            input: RefCell::new(None),
            _run: run,
            _planning: Some(planning),
        }))))
    }
    pub fn funding(&self) -> &HostMetadataFunding {
        self._planning.as_ref().expect("original producer funding")
    }
    pub fn capture(
        &self,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<eredu_runtime::capture::FundedCaptureSession, BackendFailure> {
        self.capture_bank(source)?
            .into_capture_session()
            .map_err(|e| funded_error(e, self.funding()))
    }
    pub fn capture_bank(
        &self,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<PreparedCaptureRun, BackendFailure> {
        let bank = self
            .capture
            .borrow_mut()
            .take()
            .ok_or_else(|| TokenInputRejection::IdentityMismatch.into_backend_failure())?;
        if !bank.source().same_storage(source) {
            return Err(TokenInputRejection::IdentityMismatch.into_backend_failure());
        }
        Ok(bank)
    }
    pub fn bind<C: TokenFilterController>(
        &self,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        if let Some((source, _)) = &self.controller {
            source.validate(controller).map_err(memory)?;
        }
        self.request.bind_run(context).map_err(memory)
    }
    pub fn validate_ready<C: TokenFilterController>(
        &self,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        if let Some((source, _)) = &self.controller {
            source
                .validate(controller)
                .map_err(|e| funded_error(e, self.funding()))?;
        }
        self.request
            .validate_initial_ready_context(context)
            .map_err(|e| funded_error(e, self.funding()))
    }
    pub fn sequence(
        &self,
        claim: GenerationSequencePreparation<'_, '_>,
    ) -> Result<RetainedGenerationSequence, BackendFailure> {
        let input = self
            .owner
            .borrow_mut()
            .as_mut()
            .unwrap()
            .take_token_input_bank()
            .unwrap();
        *self.input.borrow_mut() = Some(input.construct(&self.request, &claim)?);
        self.sequence
            .borrow_mut()
            .take()
            .unwrap()
            .prepare(&self.request, claim)
    }
    pub fn prompt(&self) -> Result<Prompt, BackendFailure> {
        let input = self.input.borrow_mut().take().unwrap();
        input.validate(&self.request).map_err(memory)?;
        self.request
            .claim_prompt()
            .map_err(memory)?
            .finish()
            .map_err(memory)?;
        Ok(Prompt::original(
            input,
            self.owner
                .borrow()
                .as_ref()
                .unwrap()
                .control_guard()
                .metadata_custody(),
        ))
    }
    pub fn step<C: TokenFilterController>(
        &self,
        controller: &C,
        receipt: Option<&InferenceTextStepReceipt>,
        context: &TextStepContext,
    ) -> Result<Step, WorkingMemoryError> {
        if let Some((storage, _)) = &self.controller {
            storage
                .validate(controller)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        }
        let input = match receipt {
            None => PendingTextInput::Prefill(()),
            Some(receipt) => PendingTextInput::Decode(receipt),
        };
        Ok(Step {
            controller: self.controller.clone(),
            step: self.request.claim_step(context, input)?,
        })
    }
}
pub(super) struct Step {
    controller: Option<(ControllerStorageContract, TextControllerContract)>,
    step: InferenceTextStep,
}
impl Step {
    pub fn validate(
        &self,
        decision: &TokenSamplingDecision<'_>,
        env: &Environment,
    ) -> Result<(), WorkingMemoryError> {
        if let Some((storage, contract)) = &self.controller {
            storage
                .validate_sampling_decision(contract, decision, &env.pool)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        }
        Ok(())
    }
    pub fn receipt(&self) -> InferenceTextStepReceipt {
        self.step.receipt()
    }
    pub fn finish(self) -> Result<(), WorkingMemoryError> {
        self.step.finish()
    }
}
