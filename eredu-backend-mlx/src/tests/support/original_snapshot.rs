//! Conformance adapter over the actual retained sequence and shared snapshot workers.
use crate::composition::mlx::MlxBackend;
use eredu_core::{
    ControlledTextGeneration, HostMetadataFunding, MemoryLimits, ModelRuntime,
    OriginalTextResumeOptions, RetainedGenerationSequence, TextGenerationConfig,
    TextSamplingControlBackend, TextSamplingStrategy, TextSnapshotSource,
};
use eredu_evaluation::execution_control::ContinuationSnapshotProvider;
use eredu_runtime::{
    execution_control::{
        SnapshotBudget, SnapshotTokenController, TextContinuationSnapshot, TextSnapshotBackend,
        TextSnapshotError,
    },
    working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkspaceCopyLimits},
};
type Backend<'model> = MlxBackend<'model>;
type Error = crate::backend::error::Error;

pub(crate) struct NativeSnapshotProvider {
    sequence: RetainedGenerationSequence,
    config: TextGenerationConfig,
    pool: MemoryLedger,
    choice: HostMetadataFunding,
}
impl NativeSnapshotProvider {
    pub(crate) fn new(
        sequence: RetainedGenerationSequence,
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
            config,
            pool,
            choice,
        }
    }
    pub(crate) fn observe_token(&mut self, token: u32) {
        self.sequence
            .commit(token, eredu_core::TokenTerminalSignals::default())
            .unwrap();
    }
    pub(crate) fn exchange_host(&mut self, host: &mut (RetainedGenerationSequence, f32)) {
        std::mem::swap(&mut self.sequence, &mut host.0);
    }
    fn resumed_config(&self, remaining: Option<usize>, temperature: f32) -> TextGenerationConfig {
        let mut sampling = self.config.sampling();
        sampling.max_new_tokens = remaining;
        sampling.temperature = temperature;
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
impl<'model, C: SnapshotTokenController + 'static> ContinuationSnapshotProvider<Backend<'model>, C>
    for NativeSnapshotProvider
{
    type Host = (RetainedGenerationSequence, f32);
    fn capture(
        &mut self,
        source: &mut TextSnapshotSource<'_, Backend<'model>, C>,
        budget: &SnapshotBudget,
        host_bytes: Option<u64>,
    ) -> Result<(TextContinuationSnapshot<Backend<'model>, C>, Self::Host), TextSnapshotError<Error>>
    {
        host_bytes.ok_or(eredu_core::execution_control::ExecutionControlError::UnknownEstimate)?;
        let (runtime, state, pending) = source.parts();
        let temperature = Backend::sampling_control_facts(state).temperature;
        let preparation =
            Backend::original_saved_generation_preparation_bytes(runtime, state, pending)
                .map_err(TextSnapshotError::HostAdmission)?
                .ok_or(TextSnapshotError::Unsupported(
                    "native snapshot preparation",
                ))?;
        let host = self
            .pool
            .prepare_generation_snapshot_host_copy::<Self, Error, Backend<'model>, C>(
                &self.sequence,
                MemoryLimits::unlimited(self.pool.topology()),
                preparation,
            )
            .map_err(TextSnapshotError::HostAdmission)?;
        TextContinuationSnapshot::capture_original_host(
            source,
            budget,
            host,
            WorkspaceCopyLimits::new(Default::default()),
        )
        .map(|(snapshot, host)| (snapshot, (host, temperature)))
    }
    fn resume<'a>(&mut self, runtime: &'a mut ModelRuntime<Backend<'model>>, saved: &TextContinuationSnapshot<Backend<'model>,C>,
        source: &Self::Host, options: &OriginalTextResumeOptions<'_>)
    -> Result<Option<(ControlledTextGeneration<'a,Backend<'model>,C>, <Backend<'model> as eredu_core::execution_control::NativeTextStateBackend>::NativeTextState,Self::Host)>,TextSnapshotError<Error>>{
        let config = self.resumed_config(saved.remaining_tokens(), source.1);
        let preparation =
            saved.original_resume_preparation_bytes(runtime, config.clone(), options)?;
        let host = self
            .pool
            .prepare_generation_resume_host_copy::<Self, Error, Backend<'model>, C>(
                &source.0,
                MemoryLimits::unlimited(self.pool.topology()),
                preparation,
            )
            .map_err(TextSnapshotError::HostAdmission)?;
        saved
            .resume_original_host_with_displaced(
                runtime,
                config,
                host,
                &Default::default(),
                options,
            )
            .map(|result| {
                result
                    .map(|(generation, displaced, host)| (generation, displaced, (host, source.1)))
            })
    }
    fn capture_usage(
        &self,
        source: &TextSnapshotSource<'_, Backend<'model>, C>,
    ) -> eredu_core::capture::CaptureUsage {
        Backend::capture_usage(source.parts().1)
    }
    fn observe_token(&mut self, token: u32) {
        NativeSnapshotProvider::observe_token(self, token);
    }
    fn exchange_host(&mut self, host: &mut Self::Host) {
        NativeSnapshotProvider::exchange_host(self, host);
    }
    fn settle_retirement(complete: impl FnMut() -> bool) {
        crate::backend::submission_recovery::wait_for_retirement(complete);
    }
    fn choice_funding(&self) -> HostMetadataFunding {
        self.choice.clone()
    }
}

pub(crate) struct NativeSemanticSnapshotProvider {
    sequence: RetainedGenerationSequence,
    semantic: eredu_core::SemanticStateOwner,
    config: TextGenerationConfig,
    pool: MemoryLedger,
    choice: HostMetadataFunding,
}
impl NativeSemanticSnapshotProvider {
    pub(crate) fn new(
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
    fn resumed_config(&self, remaining: Option<usize>, temperature: f32) -> TextGenerationConfig {
        let mut sampling = self.config.sampling();
        sampling.max_new_tokens = remaining;
        sampling.temperature = temperature;
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
impl<'model, C: SnapshotTokenController + 'static> ContinuationSnapshotProvider<Backend<'model>, C>
    for NativeSemanticSnapshotProvider
{
    type Host = (
        (
            RetainedGenerationSequence,
            eredu_core::SemanticStateOwner,
            (),
        ),
        f32,
    );
    fn capture(
        &mut self,
        source: &mut TextSnapshotSource<'_, Backend<'model>, C>,
        budget: &SnapshotBudget,
        host_bytes: Option<u64>,
    ) -> Result<(TextContinuationSnapshot<Backend<'model>, C>, Self::Host), TextSnapshotError<Error>>
    {
        host_bytes.ok_or(eredu_core::execution_control::ExecutionControlError::UnknownEstimate)?;
        let (runtime, state, pending) = source.parts();
        let temperature = Backend::sampling_control_facts(state).temperature;
        let preparation =
            Backend::original_saved_generation_preparation_bytes(runtime, state, pending)
                .map_err(TextSnapshotError::HostAdmission)?
                .ok_or(TextSnapshotError::Unsupported(
                    "native snapshot preparation",
                ))?;
        let host = self
            .pool
            .prepare_semantic_generation_snapshot_host_copy::<Self, Error, Backend<'model>, C, _>(
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
        .map(|(snapshot, host)| (snapshot, (host, temperature)))
    }
    fn resume<'a>(&mut self, runtime: &'a mut ModelRuntime<Backend<'model>>, saved: &TextContinuationSnapshot<Backend<'model>,C>,
        source: &Self::Host, options: &OriginalTextResumeOptions<'_>)
    -> Result<Option<(ControlledTextGeneration<'a,Backend<'model>,C>, <Backend<'model> as eredu_core::execution_control::NativeTextStateBackend>::NativeTextState,Self::Host)>,TextSnapshotError<Error>>{
        let config = self.resumed_config(saved.remaining_tokens(), source.1);
        let preparation =
            saved.original_resume_preparation_bytes(runtime, config.clone(), options)?;
        let host = self
            .pool
            .prepare_semantic_generation_resume_host_copy::<Self, Error, Backend<'model>, C, _>(
                &source.0 .0,
                &source.0 .1,
                MemoryLimits::unlimited(self.pool.topology()),
                preparation,
                (),
            )
            .map_err(TextSnapshotError::HostAdmission)?;
        saved
            .resume_original_host_with_displaced(
                runtime,
                config,
                host,
                &Default::default(),
                options,
            )
            .map(|result| {
                result
                    .map(|(generation, displaced, host)| (generation, displaced, (host, source.1)))
            })
    }
    fn capture_usage(
        &self,
        source: &TextSnapshotSource<'_, Backend<'model>, C>,
    ) -> eredu_core::capture::CaptureUsage {
        Backend::capture_usage(source.parts().1)
    }
    fn observe_token(&mut self, token: u32) {
        NativeSemanticSnapshotProvider::observe_token(self, token);
    }
    fn exchange_host(&mut self, host: &mut Self::Host) {
        NativeSemanticSnapshotProvider::exchange_host(self, &mut host.0);
    }
    fn settle_retirement(complete: impl FnMut() -> bool) {
        crate::backend::submission_recovery::wait_for_retirement(complete);
    }
    fn choice_funding(&self) -> HostMetadataFunding {
        self.choice.clone()
    }
}
