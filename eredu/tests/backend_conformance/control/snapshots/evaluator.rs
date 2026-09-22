//! Conformance adapter over the actual retained sequence and shared snapshot workers.
use super::*;
use eredu_core::{
    ControlledTextGeneration, HostMetadataFunding, MemoryLimits, ModelRuntime,
    OriginalTextResumeOptions, RetainedGenerationSequence, TextGenerationConfig,
    TextSamplingStrategy, TextSnapshotSource,
};
use eredu_evaluation::execution_control::ContinuationSnapshotProvider;
use eredu_runtime::{
    execution_control::{
        SnapshotBudget, SnapshotTokenController, TextContinuationSnapshot, TextSnapshotBackend,
        TextSnapshotError,
    },
    working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkspaceCopyLimits},
};
type Backend = MockBackend;
type Error = MockError;

struct Provider {
    sequence: RetainedGenerationSequence,
    semantic: eredu_core::SemanticStateOwner,
    config: TextGenerationConfig,
    pool: MemoryLedger,
    choice: HostMetadataFunding,
}
impl Provider {
    fn new(
        sequence: RetainedGenerationSequence,
        semantic: eredu_core::SemanticStateOwner,
        config: TextGenerationConfig,
        pool: MemoryLedger,
    ) -> Self {
        let choice = pool
            .prepare_workspace_metadata(
                &InferenceExecutionIdentity::default(),
                MemoryLimits::unlimited(pool.topology()),
            )
            .unwrap();
        Self {
            sequence,
            semantic,
            config,
            pool,
            choice,
        }
    }
    fn observe_token(&mut self, token: u32) {
        assert!(!self.semantic.push_token(token).unwrap());
        self.semantic.publish_events(&mut |_| {});
        self.sequence
            .commit(token, eredu_core::TokenTerminalSignals::default())
            .unwrap();
    }
    fn exchange_host(
        &mut self,
        host: &mut (
            RetainedGenerationSequence,
            eredu_core::SemanticStateOwner,
            (),
        ),
    ) {
        std::mem::swap(&mut self.sequence, &mut host.0);
        std::mem::swap(&mut self.semantic, &mut host.1);
    }
    fn resumed_config(&self, remaining: Option<usize>) -> TextGenerationConfig {
        let mut sampling = self.config.sampling();
        sampling.max_new_tokens = remaining;
        let config = TextGenerationConfig::new(sampling)
            .with_seed(self.config.seed())
            .with_inference_policy(self.config.inference_policy().clone());
        match self.config.strategy() {
            TextSamplingStrategy::Standard => config,
            TextSamplingStrategy::MirostatV2 { tau, eta } => {
                config.with_mirostat_v2(tau, eta).unwrap()
            }
        }
    }
}
impl<C: SnapshotTokenController + 'static> ContinuationSnapshotProvider<Backend, C> for Provider {
    type Host = (
        RetainedGenerationSequence,
        eredu_core::SemanticStateOwner,
        (),
    );
    fn capture(
        &mut self,
        source: &mut TextSnapshotSource<'_, Backend, C>,
        budget: &SnapshotBudget,
        host_bytes: Option<u64>,
    ) -> Result<(TextContinuationSnapshot<Backend, C>, Self::Host), TextSnapshotError<Error>> {
        host_bytes.ok_or(eredu_core::execution_control::ExecutionControlError::UnknownEstimate)?;
        let (runtime, state, pending) = source.parts();
        let preparation =
            Backend::original_saved_generation_preparation_bytes(runtime, state, pending)
                .map_err(TextSnapshotError::HostAdmission)?
                .ok_or(TextSnapshotError::Unsupported(
                    "native snapshot preparation",
                ))?;
        let host = self
            .pool
            .prepare_semantic_generation_snapshot_host_copy::<Provider, Error, Backend, C, _>(
                &self.sequence,
                &self.semantic,
                MemoryLimits::unlimited(self.pool.topology()),
                preparation,
                (),
            )
            .map_err(TextSnapshotError::HostAdmission)?;
        TextContinuationSnapshot::capture_original_host(
            source,
            budget,
            host,
            WorkspaceCopyLimits::new(Default::default()),
        )
    }
    fn resume<'a>(
        &mut self,
        runtime: &'a mut ModelRuntime<Backend>,
        saved: &TextContinuationSnapshot<Backend, C>,
        source: &Self::Host,
        options: &OriginalTextResumeOptions<'_>,
    ) -> Result<
        Option<(
            ControlledTextGeneration<'a, Backend, C>,
            <Backend as eredu_core::execution_control::NativeTextStateBackend>::NativeTextState,
            Self::Host,
        )>,
        TextSnapshotError<Error>,
    > {
        let config = self.resumed_config(saved.remaining_tokens());
        let preparation =
            saved.original_resume_preparation_bytes(runtime, config.clone(), options)?;
        let host = self
            .pool
            .prepare_semantic_generation_resume_host_copy::<Provider, Error, Backend, C, _>(
                &source.0,
                &source.1,
                MemoryLimits::unlimited(self.pool.topology()),
                preparation,
                (),
            )
            .map_err(TextSnapshotError::HostAdmission)?;
        saved.resume_original_host_with_displaced(
            runtime,
            config,
            host,
            &Default::default(),
            options,
        )
    }
    fn capture_usage(
        &self,
        source: &TextSnapshotSource<'_, Backend, C>,
    ) -> eredu_core::capture::CaptureUsage {
        Backend::capture_usage(source.parts().1)
    }
    fn observe_token(&mut self, token: u32) {
        Provider::observe_token(self, token);
    }
    fn exchange_host(&mut self, host: &mut Self::Host) {
        Provider::exchange_host(self, host);
    }
    fn choice_funding(&self) -> HostMetadataFunding {
        self.choice.clone()
    }
}

fn plain_controller(
    runtime: &ModelRuntime<Backend>,
) -> (
    eredu_evaluation::execution_control::fixture::PlainController,
    eredu_core::SemanticStateOwner,
) {
    let env = Backend::source_environment(runtime);
    let json = unicode_tokenizer(None, 64).to_string(false).unwrap();
    let tokenizer = env
        .pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(json.as_bytes())
                .unwrap()
                .with_generation_domain()
                .unwrap(),
        )
        .unwrap();
    let validity = env
        .pool
        .prepare_shared_token_filter(|| tokenizer.generation_domain().unwrap().clone())
        .unwrap();
    let source = eredu_runtime::working_memory::PreparedSemanticSource::new(
        &tokenizer,
        &env.execution,
        MemoryLimits::unlimited(env.pool.topology()),
    )
    .unwrap();
    let controller = eredu_evaluation::execution_control::fixture::PlainController::from_prepared(
        &source, validity, 20,
    );
    let stops = env
        .pool
        .compile_stop_source(eredu_text::stop_storage::StopCompilePlan::prepare_refs(&[]).unwrap())
        .unwrap();
    let semantic = source
        .prepare(&stops, 20, std::num::NonZeroUsize::new(32).unwrap(), false)
        .unwrap();
    (controller, semantic)
}

fn capture_options(
    runtime: &ModelRuntime<Backend>,
    mode: u8,
) -> Option<eredu_core::TextPreparationOptions> {
    if mode == 0 {
        return None;
    }
    let mut plan = observed_mock::plan();
    if mode & 1 == 0 {
        plan.selections.clear();
    }
    let discovery = &runtime.session().capture_discovery;
    let admitted = plan
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            eredu_core::capture::CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 20,
            },
            Default::default(),
        )
        .unwrap();
    let capture = eredu_core::capture::SharedCapturePlan::new(admitted);
    let interventions = (mode & 2 != 0).then(|| {
        let env = Backend::source_environment(runtime);
        let funding = env
            .pool
            .prepare_workspace_metadata(
                &env.execution,
                MemoryLimits::unlimited(env.pool.topology()),
            )
            .unwrap();
        original_sources::compile_intervention::<Backend>(
            runtime,
            &observed_mock::intervention_plan(1.0),
            &capture,
            "fixture",
            &funding,
        )
        .unwrap()
        .plan()
        .clone()
    });
    Some(eredu_core::TextPreparationOptions {
        capture: Some(capture),
        interventions,
    })
}
fn fixture_limits(mode: u8) -> eredu_evaluation::execution_control::ContinuationFixtureLimits {
    eredu_evaluation::execution_control::ContinuationFixtureLimits {
        host_bytes: 64_000_000,
        growth_bytes: 64_000_000,
        max_predictions: 20,
        capture: (mode != 0).then(|| observed_mock::plan().limits),
    }
}

#[test]
fn reusable_evaluator_conforms_through_canonical_sources_for_every_record_mode() {
    use eredu_evaluation::execution_control::{
        continuation_conformance, forced_choice_conformance, sampling_override_conformance,
    };
    use eredu_runtime::execution_control::TokenChoiceController;
    for mode in 0..4 {
        let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
        let env = Backend::source_environment(&runtime);
        env.output_width.set(64);
        let pool = env.pool.clone();
        let probe = crate::host_authority::Guard::new(&pool);
        let (controller, semantic) = plain_controller(&runtime);
        let controller =
            TokenChoiceController::new(controller, eredu_runtime::TokenDomain::new(64));
        let options = capture_options(&runtime, mode);
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                GenerationConfigOverrides {
                    max_new_tokens: Some(20),
                    temperature: Some(0.7),
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .with_seed(17);
        let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
            Provider,
            MockError,
            MockError,
        >()
        .unwrap();
        let mut state = ControlledTextGeneration::from_token_ids_with_sequence(
            &mut runtime,
            eredu_core::TokenIdsInputPlan::new(&[11, 7, 3]).unwrap(),
            config.clone(),
            controller,
            options,
            eredu_core::GenerationSequenceRequest::new(20, &[])
                .with_consumer(&consumer)
                .with_semantic_state(&semantic),
        )
        .unwrap();
        let sequence = state
            .take_prepared_sequence()
            .unwrap()
            .prepare_storage()
            .unwrap();
        let mut provider = Provider::new(sequence, semantic, config, pool.clone());
        forced_choice_conformance(&mut state, &mut provider, &fixture_limits(mode), 64, || {
            probe.update(|p| p.steps.len())
        });
        sampling_override_conformance(&mut state, &mut provider, &fixture_limits(mode), || {
            probe.update(|p| p.steps.len())
        });
        continuation_conformance(state, provider, fixture_limits(mode), || {
            probe.update(|p| p.steps.len())
        })
        .finish();
        drop(runtime);
        drop(probe);
        assert_eq!(pool.live_charge_bytes().unwrap(), 0, "mode {mode}");
    }
}
