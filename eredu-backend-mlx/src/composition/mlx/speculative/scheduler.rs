//! MLX resource adaptation for neutral speculative scheduling.

use std::{cell::Cell, marker::PhantomData, rc::Rc, time::Duration};

#[cfg(test)]
use eredu_core::generation::SpeculativeRequestStatus;
#[cfg(test)]
use eredu_core::BoundedCompletion;
#[cfg(test)]
use eredu_core::{
    resolve_optimistic_branch, SpeculativeDraftBlock, SpeculativeExecutionTopology,
    SpeculativeOptimisticBranch, SpeculativeProposal,
};
#[cfg(test)]
use eredu_core::{
    SpeculativeCallbackPublisher, SpeculativeDriverError, SpeculativeOutputError,
    SpeculativeOutputRuntime, SpeculativeSampling, SpeculativeSemanticConstraint,
    SpeculativeSemanticState,
};
use eredu_core::{SpeculativeExecutor, SpeculativeStats, SpeculativeTelemetry};
#[cfg(test)]
use eredu_core::{SpeculativeRequestTable, SpeculativeSchedulerStats};
use safemlx::{error::Exception, Array};
#[cfg(test)]
use safemlx::{ops::indexing::TryIndexOp, transforms::async_eval_with_event, Stream};

use crate::composition::mlx::{
    speculative::{MlxSpeculativeCompletion, SpeculativeExecutionStreams},
    MlxModelInput,
};
#[cfg(test)]
use crate::{
    backend::runtime::generation::MlxSamplingBackend,
    backend::runtime::media::input::{InputPayload, ModelInput},
    composition::mlx::speculative::MlxSpeculativeSampling,
    MlxTensor,
};
#[cfg(test)]
use eredu_core::generation::{
    FinishReason, GenerationCancellationToken, GenerationSequence, SemanticEvent,
    SpeculativeConfig, SpeculativeSchedulerOptions,
};
#[cfg(test)]
use eredu_core::{generation::SpeculativeRequestId, InputModality};
#[cfg(test)]
use eredu_runtime::SpeculativeSampler;

/// Component timings accumulated by an architecture-specific speculative backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeculativeComponentTimings {
    /// Committed-context encoding and assistant K/V projection.
    pub draft_context: Duration,
    /// Assistant proposal-block execution.
    pub draft_assistant: Duration,
    /// Raw vocabulary-head projection.
    pub draft_head: Duration,
    /// Target verification execution.
    pub target_verification: Duration,
}

thread_local! {
    static COMPONENT_TIMING_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Scoped opt-in for device-timeline speculative component profiling.
///
/// Schedulers created while this guard is alive collect architecture-level
/// timings. Timestamp boundaries add profiling overhead, so normal generation
/// leaves them disabled. The guard is intentionally bound to its creating
/// thread and restores the previous setting when dropped.
pub struct SpeculativeComponentTimingGuard {
    previous: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl SpeculativeComponentTimingGuard {
    /// Enables component profiling for schedulers created on this thread.
    pub fn enable() -> Self {
        let previous = COMPONENT_TIMING_ENABLED.replace(true);
        Self {
            previous,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for SpeculativeComponentTimingGuard {
    fn drop(&mut self) {
        COMPONENT_TIMING_ENABLED.set(self.previous);
    }
}

pub(crate) fn component_timing_enabled() -> bool {
    COMPONENT_TIMING_ENABLED.get()
}

impl SpeculativeComponentTimings {
    fn add_to(self, stats: &mut SpeculativeStats) {
        stats.add_component_timings(
            self.draft_context,
            self.draft_assistant,
            self.draft_head,
            self.target_verification,
        );
    }
}

impl SpeculativeTelemetry for SpeculativeComponentTimings {
    fn record(self, stats: &mut SpeculativeStats) {
        self.add_to(stats);
    }
}

/// MLX-associated types required by the facade's sampling loop.
///
/// Model execution is expressed by the portable core contract; only logits
/// sampling and stream placement remain MLX-specific in this module.
pub trait MlxSpeculativeRuntime<'a>:
    SpeculativeExecutor<
        Input = MlxModelInput,
        Logits = Array,
        Context<'a> = SpeculativeExecutionStreams<'a>,
        Completion = MlxSpeculativeCompletion,
        Telemetry = SpeculativeComponentTimings,
        Error = Exception,
    > + 'a
{
}

impl<'a, T> MlxSpeculativeRuntime<'a> for T where
    T: SpeculativeExecutor<
            Input = MlxModelInput,
            Logits = Array,
            Context<'a> = SpeculativeExecutionStreams<'a>,
            Completion = MlxSpeculativeCompletion,
            Telemetry = SpeculativeComponentTimings,
            Error = Exception,
        > + 'a
{
}

#[cfg(test)]
type CommittedOutputRuntime<'a, S> = SpeculativeOutputRuntime<
    MlxSpeculativeSampling<S>,
    SpeculativeSemanticConstraint,
    SpeculativeCallbackPublisher<'a>,
>;

#[cfg(test)]
fn plain_runtime<'a, S, F>(
    sampler: S,
    config: &SpeculativeConfig,
    mut on_token: F,
) -> CommittedOutputRuntime<'a, S>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
    F: FnMut(&[u32]) -> Result<(), Exception> + 'a,
{
    SpeculativeOutputRuntime::new(
        MlxSpeculativeSampling::new(sampler),
        GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied()),
        SpeculativeSemanticConstraint::plain(),
        SpeculativeCallbackPublisher::tokens(move |tokens| {
            on_token(tokens).map_err(|error| SpeculativeOutputError::publication(error.to_string()))
        }),
        GenerationCancellationToken::new(),
    )
}

#[cfg(test)]
fn semantic_runtime<'a, S, F>(
    sampler: S,
    config: &SpeculativeConfig,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: F,
) -> CommittedOutputRuntime<'a, S>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
    F: FnMut(SemanticEvent) + 'a,
{
    SpeculativeOutputRuntime::new(
        MlxSpeculativeSampling::new(sampler),
        GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied()),
        SpeculativeSemanticConstraint::semantic(semantic),
        SpeculativeCallbackPublisher::semantic(on_event),
        cancellation,
    )
}

#[cfg(test)]
type MlxRequestTable<'a, B, S> = SpeculativeRequestTable<
    'a,
    B,
    MlxSpeculativeSampling<S>,
    SpeculativeSemanticConstraint,
    SpeculativeCallbackPublisher<'a>,
>;

#[cfg(test)]
pub struct SpeculativeRequestOutput<S> {
    pub token_ids: Vec<u32>,
    pub stats: SpeculativeStats,
    pub sampler: S,
    pub finish_reason: Option<FinishReason>,
    #[cfg(test)]
    pub cancelled: bool,
}

#[cfg(test)]
pub struct SpeculativeScheduleOutput<S> {
    pub requests: Vec<SpeculativeRequestOutput<S>>,
    pub scheduler: SpeculativeSchedulerStats,
}

/// Test harness for exercising the neutral fair speculative request table with MLX values.
///
/// MLX streams already provide asynchronous device queues. The scheduler stays
/// deliberately single-threaded: it submits lazy target graphs, performs CPU
/// draft work, and synchronizes only when a verification result is resolved.
/// Model parameters are shared through the backend; every request owns its
/// cache, target state, sampler, PRNG substreams, output, and statistics.
#[cfg(test)]
struct MlxSpeculativeScheduler<'a, B, S>
where
    B: MlxSpeculativeRuntime<'a>,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'a,
{
    backend: &'a mut B,
    streams: SpeculativeExecutionStreams<'a>,
    component_timing: bool,
    requests: MlxRequestTable<'a, B, S>,
}

#[cfg(test)]
impl<'a, B, S> MlxSpeculativeScheduler<'a, B, S>
where
    B: MlxSpeculativeRuntime<'a>,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'a,
{
    /// Creates a scheduler over shared model parameters and explicit streams.
    pub fn new(
        backend: &'a mut B,
        streams: SpeculativeExecutionStreams<'a>,
        options: SpeculativeSchedulerOptions,
    ) -> Result<Self, Exception> {
        let wait = options
            .completion_wait()
            .map_err(|error| Exception::custom(error.to_string()))?;
        if !MlxSpeculativeCompletion::supports_cancellation(wait.cancellation()) {
            return Err(Exception::custom(format!(
                "MLX speculative completion does not support {:?}",
                wait.cancellation()
            )));
        }
        let requests = SpeculativeRequestTable::new(options, streams.topology())
            .map_err(|error| Exception::custom(error.to_string()))?;
        let component_timing = component_timing_enabled() && backend.supports_telemetry();
        backend.set_telemetry_enabled(component_timing);
        Ok(Self {
            backend,
            streams,
            component_timing,
            requests,
        })
    }

    /// Submits one independent request.
    #[allow(clippy::too_many_arguments)]
    pub fn submit<F>(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: SpeculativeConfig,
        prng_key: Option<Array>,
        sampler: S,
        on_token: F,
    ) -> Result<SpeculativeRequestId, Exception>
    where
        F: FnMut(&[u32]) -> Result<(), Exception> + 'a,
    {
        let runtime = plain_runtime(sampler, &config, on_token);
        self.submit_runtime(cache, input, config, prng_key, runtime)
    }

    /// Submits one request with transactional decoded semantic output.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn submit_with_semantics<F>(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: SpeculativeConfig,
        prng_key: Option<Array>,
        sampler: S,
        semantic: Box<dyn SpeculativeSemanticState>,
        on_event: F,
    ) -> Result<SpeculativeRequestId, Exception>
    where
        F: FnMut(SemanticEvent) + 'a,
    {
        self.submit_with_semantics_cancellable(
            cache,
            input,
            config,
            prng_key,
            sampler,
            semantic,
            GenerationCancellationToken::new(),
            on_event,
        )
    }

    /// Submits one semantic request controlled by a public cancellation token.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_with_semantics_cancellable<F>(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: SpeculativeConfig,
        prng_key: Option<Array>,
        sampler: S,
        semantic: Box<dyn SpeculativeSemanticState>,
        cancellation: GenerationCancellationToken,
        on_event: F,
    ) -> Result<SpeculativeRequestId, Exception>
    where
        F: FnMut(SemanticEvent) + 'a,
    {
        let runtime = semantic_runtime(sampler, &config, semantic, cancellation, on_event);
        self.submit_runtime(cache, input, config, prng_key, runtime)
    }

    fn submit_runtime(
        &mut self,
        cache: &'a mut B::Cache,
        input: ModelInput<'_>,
        config: SpeculativeConfig,
        prng_key: Option<Array>,
        runtime: CommittedOutputRuntime<'a, S>,
    ) -> Result<SpeculativeRequestId, Exception> {
        config
            .validate()
            .map_err(|error| Exception::custom(error.to_string()))?;
        validate_input(input)?;
        let input = MlxModelInput::from(input);
        let randomness = <MlxSpeculativeSampling<S> as SpeculativeSampling>::initialize_randomness(
            prng_key,
            config.temperature,
            self.streams,
        )?;
        self.requests
            .submit(
                self.backend,
                cache,
                input,
                config,
                runtime,
                randomness,
                self.component_timing,
                self.streams,
            )
            .map_err(speculative_driver_error)
    }

    /// Returns the current status for a submitted request.
    #[cfg(test)]
    pub fn status(&self, id: SpeculativeRequestId) -> Option<SpeculativeRequestStatus> {
        self.requests.status(id)
    }

    /// Requests independent cancellation.
    ///
    /// An in-flight target transaction is resolved to a safe cache boundary
    /// before the request enters `Cancelled`; other requests continue.
    #[cfg(test)]
    pub fn cancel(&mut self, id: SpeculativeRequestId) -> Result<(), Exception> {
        self.requests.cancel(id).map_err(speculative_driver_error)
    }

    /// Returns whether every request is completed or cancelled.
    #[cfg(test)]
    pub fn is_finished(&self) -> bool {
        self.requests.is_finished()
    }

    /// Performs one fair scheduler operation.
    ///
    /// Returns `false` when every request is terminal.
    pub fn step(&mut self) -> Result<bool, Exception> {
        self.requests
            .step(self.backend, self.streams.is_split(), self.streams)
            .map_err(speculative_driver_error)
    }

    /// Drives all submitted requests to completion.
    pub fn run(&mut self) -> Result<(), Exception> {
        while self.step()? {}
        Ok(())
    }

    /// Consumes a finished scheduler and returns results in submission order.
    pub fn finish(self) -> Result<SpeculativeScheduleOutput<S>, Exception> {
        let mut output = self.requests.finish().map_err(speculative_driver_error)?;
        Ok(SpeculativeScheduleOutput {
            requests: output
                .take_requests()
                .into_iter()
                .map(|request| {
                    let mut request = request.into_artifact();
                    let token_ids = request.take_token_ids();
                    let stats = request.take_stats();
                    let finish_reason = request.finish_reason();
                    #[cfg(test)]
                    let cancelled = request.status() == SpeculativeRequestStatus::Cancelled;
                    let sampler = request.into_sampler().into_inner();
                    SpeculativeRequestOutput {
                        token_ids,
                        stats,
                        sampler,
                        finish_reason,
                        #[cfg(test)]
                        cancelled,
                    }
                })
                .collect(),
            scheduler: output.take_scheduler(),
        })
    }
}

#[cfg(test)]
fn speculative_driver_error(error: SpeculativeDriverError<Exception>) -> Exception {
    Exception::custom(error.to_string())
}

#[cfg(test)]
fn generate<'runtime, B, S>(
    backend: &'runtime mut B,
    cache: &'runtime mut B::Cache,
    input: ModelInput<'_>,
    config: &SpeculativeConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &'runtime Stream,
) -> Result<(Vec<u32>, SpeculativeStats), Exception>
where
    B: MlxSpeculativeRuntime<'runtime>,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'runtime,
{
    generate_with_streams(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        SpeculativeExecutionStreams::single(stream),
    )
}

#[cfg(test)]
fn generate_with_streams<'runtime, B, S>(
    backend: &'runtime mut B,
    cache: &'runtime mut B::Cache,
    input: ModelInput<'_>,
    config: &SpeculativeConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    streams: SpeculativeExecutionStreams<'runtime>,
) -> Result<(Vec<u32>, SpeculativeStats), Exception>
where
    B: MlxSpeculativeRuntime<'runtime>,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'runtime,
{
    generate_tokens(
        backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        streams,
        SpeculativeSchedulerOptions::default(),
        |_| Ok(()),
    )
}

/// Runs one scheduled request with explicit streams, scheduler options, and a
/// commit callback.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn generate_tokens<'runtime, B, S, F>(
    backend: &'runtime mut B,
    cache: &'runtime mut B::Cache,
    input: ModelInput<'_>,
    config: &SpeculativeConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    streams: SpeculativeExecutionStreams<'runtime>,
    options: SpeculativeSchedulerOptions,
    on_token: F,
) -> Result<(Vec<u32>, SpeculativeStats), Exception>
where
    B: MlxSpeculativeRuntime<'runtime>,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'runtime,
    F: FnMut(&[u32]) -> Result<(), Exception> + 'runtime,
{
    validate_input(input)?;
    let randomness = <MlxSpeculativeSampling<S> as SpeculativeSampling>::initialize_randomness(
        prng_key,
        config.temperature,
        streams,
    )?;
    let component_timings_collected = component_timing_enabled() && backend.supports_telemetry();
    let mut scheduler = eredu_runtime::SpeculativeScheduler::new(
        backend,
        options,
        streams.topology(),
        streams.is_split(),
        component_timings_collected,
        streams,
    )
    .map_err(speculative_driver_error)?;
    scheduler
        .submit(eredu_core::PreparedSpeculativeLane::new(
            cache,
            MlxModelInput::from(input),
            config.clone(),
            plain_runtime(sampler.clone(), config, on_token),
            randomness,
        ))
        .map_err(speculative_driver_error)?;
    scheduler.run().map_err(speculative_driver_error)?;
    let mut output = scheduler.finish().map_err(speculative_driver_error)?;
    let mut request = output
        .take_requests()
        .pop()
        .expect("one request was submitted")
        .into_artifact();
    let token_ids = request.take_token_ids();
    let stats = request.take_stats();
    *sampler = request.into_sampler().into_inner();
    Ok((token_ids, stats))
}

#[cfg(test)]
fn validate_input(input: ModelInput<'_>) -> Result<(), Exception> {
    if input.parts.is_empty() {
        return Err(Exception::custom(
            "speculative input must contain at least one part",
        ));
    }
    if input
        .parts
        .iter()
        .all(|part| part.modality() == InputModality::Text)
    {
        let mut tokens = 0i32;
        for part in input.parts {
            let InputPayload::TokenIds(ids) = part.payload() else {
                return Err(Exception::custom(
                    "speculative text input must contain token-id payloads",
                ));
            };
            if ids.ndim() != 2 {
                return Err(Exception::custom(format!(
                    "speculative text token ids must have rank 2, got {:?}",
                    ids.shape()
                )));
            }
            tokens = tokens.saturating_add(ids.dim(1));
        }
        if tokens == 0 {
            return Err(Exception::custom(
                "speculative text input contains no tokens",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    include!("scheduler/tests/support.rs");
    include!("scheduler/tests/semantic.rs");
    include!("scheduler/tests/sampling.rs");
    include!("scheduler/tests/transactions.rs");
    include!("scheduler/tests/streams.rs");
    include!("scheduler/tests/fairness.rs");
    include!("scheduler/tests/stats.rs");
}
