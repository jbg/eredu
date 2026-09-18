//! Fresh request admission for the same tensor-free equations after a saved copy.
use super::*;

impl Preparation {
    pub fn resumed<C: TokenFilterController>(
        env: &Environment,
        config: TextGenerationConfig,
        controller_input: &C,
        context: &TextStepContext,
        geometry: InferenceGeometry,
        first_prediction: u64,
        funding: HostMetadataFunding,
        capture: Option<CaptureRunHostPlan<'_>>,
    ) -> Result<PreparationOwner, BackendFailure> {
        let maximum = geometry.max_output_tokens;
        let controller = controller_input
            .inference_workspace(maximum)
            .map(|workspace| {
                Ok::<_, BackendFailure>((
                    ControllerStorageContract::inspect_original_retained(
                        controller_input,
                        workspace,
                        &env.pool,
                        &env.execution,
                        maximum,
                    )?,
                    TextControllerContract::from_workspace(workspace, env.output_width.get())
                        .map_err(|e| funded_error(e, &funding))?,
                ))
            })
            .transpose()?;
        let quote_candidate = |geometry| -> Result<IncrementalInferenceQuote, BackendFailure> {
            let quote = quote(
                env,
                geometry,
                controller
                    .as_ref()
                    .map_or(0, |(_, c)| c.filter_capacity_bytes()),
                capture
                    .as_ref()
                    .map_or(0, CaptureRunHostPlan::initialization_peak_bytes),
                Some(&funding),
            )?;
            if maximum == 0 {
                Ok(quote)
            } else {
                quote
                    .with_span_workspace()
                    .map_err(|e| funded_error(e, &funding))
            }
        };
        let capacity = config
            .inference_policy()
            .managed_memory_capacity_bytes
            .ok_or_else(|| TokenInputRejection::Unsupported.into_backend_failure())?;
        let kind = "neutral public conformance fixture";
        let context_reason = "no model context storage";
        funding.reserve_metadata(
            kind.len() + 2 * context_reason.len() + size_of::<ModelCapabilities>(),
        )?;
        let capabilities = ModelCapabilities {
            effective_model_type: kind.into(),
            native_max_context: Observed::exact(u64::MAX, context_reason),
            effective_max_context: Observed::exact(u64::MAX, context_reason),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        let admission = AdmissionRequest {
            input: InputTokenCount::text(
                geometry
                    .cached_positions
                    .checked_add(geometry.input_positions)
                    .ok_or_else(|| funded_error(WorkingMemoryError::Overflow, &funding))?,
            ),
            max_output_tokens: maximum,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        funding.reserve_metadata(
            size_of::<Option<BackendFailure>>()
                + size_of::<Result<IncrementalInferenceQuote, BackendFailure>>(),
        )?;
        let mut quote_failure = None;
        let planned = plan_prefill_incremental_with_capacity(
            &env.execution,
            &env.pool,
            &capabilities,
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
        let (reservation, accepted) = planned.map_err(|e| funded_error(e, &funding))?;
        let (reservation, run) = reservation
            .into_funding()
            .map_err(|e| funded_error(e, &funding))?;
        let workspace = if maximum == 0 {
            drop(accepted);
            None
        } else {
            let (workspace, witness) = accepted
                .into_funded_span_workspace(&run, &reservation)
                .map_err(|e| funded_error(e, &funding))?;
            drop(witness);
            Some(workspace)
        };
        if let Some((source, _)) = &controller {
            source
                .prepare_original_source(controller_input, &run, &reservation)
                .map_err(|e| funded_error(e, &funding))?;
        }
        let capture = capture
            .map(|plan| run.prepare_capture_run(&reservation, plan))
            .transpose()
            .map_err(|e| funded_error(e, &funding))?;
        let request = InferenceRequest::from(&reservation)
            .prepare_text(&env.execution, geometry, config)
            .map_err(|e| funded_error(e, &funding))?;
        request
            .bind_run(context)
            .map_err(|e| funded_error(e, &funding))?;
        let shell = Layout::new::<[usize; 2]>()
            .extend(Layout::new::<Self>())
            .unwrap()
            .0
            .pad_to_align()
            .size();
        funding.reserve_metadata(
            shell + size_of::<Self>() + size_of::<Result<PreparationOwner, BackendFailure>>(),
        )?;
        Ok(PreparationOwner(Some(Rc::new(Self {
            controller,
            request,
            owner: RefCell::new(None),
            _resumed: workspace,
            sequence: RefCell::new(None),
            capture: RefCell::new(capture),
            input: RefCell::new(None),
            first_prediction,
            _run: run,
            _planning: Some(funding),
        }))))
    }
    pub fn completed_positions(&self, next_prediction: u64) -> Result<u64, WorkingMemoryError> {
        let geometry = self.request.request().geometry();
        let committed = next_prediction
            .checked_sub(self.first_prediction)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if committed == 0 {
            return Ok(geometry.cached_positions);
        }
        geometry
            .cached_positions
            .checked_add(geometry.input_positions)
            .and_then(|n| n.checked_add(committed - 1))
            .ok_or(WorkingMemoryError::Overflow)
    }
}
