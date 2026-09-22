//! Original saved source -> newly admitted request, using the same model equations.
use super::*;
use eredu_core::{
    HostMetadataFunding, HostPreparationAuthority, OriginalTextResumeOptions, TextResumeBackend,
    TextStepContext,
};
use eredu_core::{InferenceGeometry, OutputDemand};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::size_of;

pub(crate) struct ResumePreparation {
    source: original::Source,
    original: Option<admitted_text::PreparationOwner>,
    config: TextGenerationConfig,
    sampling: Option<eredu_core::SamplingOverride>,
    revised_capture: Option<eredu_runtime::capture::FundedCaptureCheckpoint>,
    kind: eredu_core::OriginalTextResumeKind,
    native: Option<Native>,
    phase: u8,
    _host: HostPreparationAuthority,
    funding: HostMetadataFunding,
}
fn mismatch() -> MockError {
    WorkingMemoryError::IdentityMismatch.into()
}
fn map(error: MockError) -> eredu_core::BackendFailure {
    MockBackend::into_backend_failure(error)
}

fn admit<C: eredu_core::TokenFilterController>(
    runtime: &ModelRuntime<MockBackend>,
    saved: &SavedComponents,
    config: TextGenerationConfig,
    controller: &C,
    context: &TextStepContext,
    host: &HostPreparationAuthority,
    options: &OriginalTextResumeOptions<'_>,
) -> Result<ResumePreparation, eredu_core::BackendFailure> {
    let source = saved.original.as_ref().ok_or_else(|| map(mismatch()))?;
    source.validate(runtime).map_err(map)?;
    if options.kind == eredu_core::OriginalTextResumeKind::Branch {
        super::super::provider_errors::check("growth").map_err(map)?;
    }
    if source.native.0 != runtime.session().intervention_identity {
        return Err(map(mismatch()));
    }
    let env = MockBackend::source_environment(runtime);
    let maximum = config
        .sampling()
        .max_new_tokens
        .ok_or_else(|| map(WorkingMemoryError::UnknownBound.into()))? as u64;
    let input = if options.terminal {
        0
    } else {
        match source.sampling.pending.as_ref() {
            Some(PendingTextInput::Prefill(prompt)) => prompt.len() as u64,
            Some(PendingTextInput::Decode(_)) => 1,
            None => return Err(map(mismatch())),
        }
    };
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: source.frontier,
        input_positions: input,
        max_output_tokens: if options.terminal { 0 } else { maximum },
        prefill_chunk_positions: input,
        output: if options.terminal {
            OutputDemand::StateOnly
        } else {
            OutputDemand::LastPosition
        },
    };
    let capacity = config
        .inference_policy()
        .memory_limits
        .resolve(env.pool.topology())
        .map_err(|error| map(WorkingMemoryError::from(error).into()))?;
    let funding = env
        .pool
        .prepare_workspace_metadata(&env.execution, capacity.clone())?;
    funding.reserve_metadata(
        size_of::<ResumePreparation>()
            + size_of::<Result<ResumePreparation, eredu_core::BackendFailure>>()
            + size_of::<(
                OriginalTextResumeOptions<'_>,
                InferenceGeometry,
                Option<PendingTextInput<&Prompt, &MockToken>>,
            )>(),
    )?;
    let revised_capture = revise_capture(runtime, source, options, &funding)?;
    let checkpoint = revised_capture.as_ref().or(source.capture.as_ref());
    let capture = checkpoint
        .map(|c| c.continuation_host_plan_for(geometry))
        .transpose()
        .map_err(|e| admitted_text::funded_error(e, &funding))?;
    let original = admitted_text::Preparation::resumed(
        env,
        config.clone(),
        controller,
        context,
        geometry,
        source.sampling.sampling.prediction,
        funding.clone(),
        capture,
    )?;
    Ok(ResumePreparation {
        source: source.clone(),
        original: Some(original),
        config,
        sampling: options.sampling,
        revised_capture,
        kind: options.kind,
        native: None,
        phase: 0,
        _host: host.clone(),
        funding,
    })
}
fn revise_capture(
    runtime: &ModelRuntime<MockBackend>,
    source: &original::Source,
    options: &OriginalTextResumeOptions<'_>,
    funding: &HostMetadataFunding,
) -> Result<Option<eredu_runtime::capture::FundedCaptureCheckpoint>, eredu_core::BackendFailure> {
    let inherited = source
        .capture
        .as_ref()
        .and_then(|c| c.intervention_source());
    let edits = options.intervention.or_else(|| {
        options
            .session_id
            .and_then(|_| inherited.map(|s| s.plan().admission().plan()))
    });
    if options.capture_limits.is_none() && edits.is_none() {
        return Ok(None);
    }
    if options.kind != eredu_core::OriginalTextResumeKind::Branch {
        return Err(map(mismatch()));
    }
    let saved = source.capture.as_ref().ok_or_else(|| map(mismatch()))?;
    let pool = &MockBackend::source_environment(runtime).pool;
    funding.reserve_metadata(
        eredu_core::capture::PreparedCapturePlanCopy::inspection_control_bytes()
            .and_then(|n| {
                n.checked_add(
                    size_of::<Option<eredu_runtime::capture::FundedCaptureCheckpoint>>() * 3,
                )
            })
            .ok_or(eredu_core::HostMetadataFundingError::Overflow)?,
    )?;
    let mut revised = if let Some(limits) = options.capture_limits {
        let copy = eredu_core::capture::PreparedCapturePlanCopy::inspect_limit_revision(
            saved.source(),
            limits.clone(),
        )
        .map_err(|e| admitted_text::funded_error(e, funding))?;
        let declaration = pool
            .compile_capture_source(copy)
            .map_err(|e| admitted_text::funded_error(e, funding))?;
        Some(
            saved
                .with_branch_capture_source(&declaration, pool, funding)
                .map_err(|e| admitted_text::funded_error(e, funding))?,
        )
    } else {
        None
    };
    if let Some(edits) = edits {
        let previous = revised.as_ref().unwrap_or(saved);
        let session = options
            .session_id
            .or_else(|| inherited.map(|s| s.plan().admission().session_id()))
            .ok_or_else(|| map(mismatch()))?;
        let declaration = original_sources::compile_intervention::<MockBackend>(
            runtime,
            edits,
            previous.source(),
            session,
            funding,
        )?;
        revised = Some(
            previous
                .with_branch_intervention_source(&declaration, pool, funding)
                .map_err(|e| admitted_text::funded_error(e, funding))?,
        );
    }
    Ok(revised)
}
impl TextResumeBackend for MockBackend {
    type ResumeSource = SavedComponents;
    type ResumePreparation = ResumePreparation;
    type DisplacedState = Native;
    fn saved_text_resume_facts(
        saved: &SavedComponents,
    ) -> Option<eredu_core::TextResumeSourceFacts> {
        saved.original.as_ref()?;
        let sampling = &saved.sampling().sampling;
        Some(eredu_core::TextResumeSourceFacts {
            inherited_capture_usage: saved
                .original
                .as_ref()?
                .capture
                .as_ref()
                .map_or(Default::default(), |c| c.inherited_usage()),
            sampling_before: eredu_core::SamplingStateFacts {
                temperature: sampling.temperature,
                has_rng: sampling.seed.is_some(),
                requires_positive_temperature: false,
            },
        })
    }
    fn text_resume_capture_source(
        state: &observed_mock::State,
    ) -> Option<&eredu_core::capture::SharedCapturePlan> {
        state.funded.as_ref().map(|c| c.source())
    }
    fn text_resume_intervention_source(
        state: &observed_mock::State,
    ) -> Option<&eredu_core::intervention::SharedInterventionPlan> {
        state
            .funded
            .as_ref()
            .and_then(|c| c.intervention_source())
            .map(|s| s.plan())
    }
    fn text_resume_facts(state: &observed_mock::State) -> eredu_core::TextResumeFacts<'_> {
        eredu_core::TextResumeFacts {
            sampling_after:
                <Self as eredu_core::TextSamplingControlBackend>::sampling_control_facts(state),
            capture_plan_id: state.funded.as_ref().map(|c| c.source().identity()),
            intervention_plan_id: state
                .funded
                .as_ref()
                .and_then(|c| c.intervention_source())
                .map(|s| s.plan().admission().identity()),
            has_interventions: state
                .funded
                .as_ref()
                .and_then(|c| c.intervention_source())
                .is_some_and(|s| !s.plan().admission().plan().operations.is_empty()),
        }
    }
    fn admit_text_resume<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &SavedComponents,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<ResumePreparation, eredu_core::BackendFailure> {
        // The same original source is required even for a low-level caller;
        // ordinary saved data never becomes admission authority.
        admit(
            runtime,
            saved,
            config,
            controller,
            context,
            &HostPreparationAuthority::unmanaged(),
            &OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Restore),
        )
    }
    fn admit_original_text_resume<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &SavedComponents,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
        host: &HostPreparationAuthority,
        options: &OriginalTextResumeOptions<'_>,
    ) -> Result<ResumePreparation, eredu_core::BackendFailure> {
        if host.is_unmanaged() {
            return Err(map(mismatch()));
        }
        admit(runtime, saved, config, controller, context, host, options)
    }
    fn prepare_text_resume_prompt(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut ResumePreparation,
    ) -> Result<Prompt, MockError> {
        if prepared.phase != 0 {
            return Err(mismatch());
        }
        prepared.source.validate(runtime)?;
        super::super::provider_errors::check("copy")?;
        let request = &prepared.original.as_ref().unwrap().request;
        let stage = request.claim_prompt()?;
        let prompt = match prepared.source.sampling.pending.as_ref() {
            Some(PendingTextInput::Prefill(prompt)) => prompt.clone(),
            Some(PendingTextInput::Decode(token)) => Prompt::copy(&[token.0], &prepared.funding)
                .map_err(MockError::ProviderRetained)?
                .with_resume_token(token.0),
            None => Prompt::copy(&[], &prepared.funding).map_err(MockError::ProviderRetained)?,
        };
        let identity = &prepared.source.native.0;
        prepared.funding.reserve_metadata(identity.len())?;
        let mut copy = String::new();
        copy.try_reserve_exact(identity.len())
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        copy.push_str(identity);
        prepared.native = Some(Native(copy, Some(prepared.funding.clone())));
        stage.finish()?;
        request.bind_prompt()?;
        prepared.phase = 1;
        Ok(prompt)
    }
    fn prepare_text_resume_sampling(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut ResumePreparation,
    ) -> Result<observed_mock::State, MockError> {
        if prepared.phase != 1 {
            return Err(mismatch());
        }
        prepared.source.validate(runtime)?;
        let original = prepared.original.as_ref().unwrap();
        let stage = original.request.claim_sampling(prepared.config.clone())?;
        let mut state = observed_mock::State {
            sampling: prepared.source.sampling.sampling.clone(),
            ..Default::default()
        };
        if let Some(request) = prepared.sampling {
            let request = eredu_runtime::execution_control::prepare_sampling_override::<Self>(
                runtime, &state, request,
            )
            .map_err(|_| mismatch())?;
            state.sampling.temperature = request.temperature();
            if let Some(seed) = request.reseed() {
                state.sampling.seed = Some(seed);
            }
        }
        if let Some(checkpoint) = prepared
            .revised_capture
            .as_ref()
            .or(prepared.source.capture.as_ref())
        {
            let bank = original
                .capture_bank(checkpoint.source())
                .map_err(MockError::ProviderRetained)?;
            let capture = match prepared.kind {
                eredu_core::OriginalTextResumeKind::Restore => checkpoint.into_restoration(bank),
                eredu_core::OriginalTextResumeKind::Branch => checkpoint.into_continuation(bank),
            }
            .map_err(|e| {
                MockError::ProviderRetained(admitted_text::funded_error(e, &prepared.funding))
            })?;
            state.install_funded(capture, &prepared.funding)?;
        }
        state.original = Some(original.clone());
        stage.finish()?;
        prepared.phase = 2;
        Ok(state)
    }
    fn install_text_resume<C: eredu_core::TokenFilterController>(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut ResumePreparation,
        _: &mut observed_mock::State,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), MockError> {
        if prepared.phase != 2 {
            return Err(mismatch());
        }
        prepared.source.validate(runtime)?;
        if prepared.native.as_ref().unwrap().0 != runtime.session().intervention_identity {
            return Err(mismatch());
        }
        prepared
            .original
            .as_ref()
            .unwrap()
            .validate_ready(controller, context)
            .map_err(MockError::ProviderRetained)?;
        // There is no mutable native model state in this fixture. Sampler and
        // pending input install only through the core machine's atomic move.
        prepared.phase = 3;
        Ok(())
    }
    fn finish_text_resume(prepared: &mut ResumePreparation) -> (preparation::Admission, Native) {
        assert_eq!(prepared.phase, 3);
        prepared.phase = 4;
        (
            preparation::Admission {
                probe: None,
                original: prepared.original.take(),
            },
            prepared.native.take().unwrap(),
        )
    }
}
