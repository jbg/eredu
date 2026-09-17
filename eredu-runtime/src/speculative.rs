//! Backend-neutral speculative request lifecycle and fair scheduling.

use eredu_core::{
    BoundedCompletion, CompletedSpeculativeSchedule, GenerationError, GenerationTiming,
    PreparedSpeculativeLane, SpeculativeConstraint, SpeculativeDriverError, SpeculativeExecutor,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationOutput, SpeculativeGenerationVisitor,
    SpeculativePublisher, SpeculativeRequestTable, SpeculativeSampling, SpeculativeBuffer,
};

mod control;
pub use control::*;
pub mod autoregressive;
pub mod numerical;
pub mod embedded_occurrence;
pub mod external_occurrence;
mod attempts;

/// Neutral owner of speculative request registration and fair scheduling.
pub struct SpeculativeScheduler<'a, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    executor: &'a mut E,
    context: E::Context<'a>,
    optimistic_execution_available: bool,
    component_timings_collected: bool,
    requests: SpeculativeRequestTable<'a, E, S, C, P>,
}

impl<'a, E, S, C, P> SpeculativeScheduler<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Creates a scheduler for one prepared executor and placement.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        executor: &'a mut E,
        options: eredu_core::generation::SpeculativeSchedulerOptions,
        topology: eredu_core::SpeculativeExecutionTopology,
        optimistic_execution_available: bool,
        component_timings_collected: bool,
        context: E::Context<'a>,
    ) -> Result<Self, SpeculativeDriverError<E::Error>> {
        let prepared = (|| {
            let completion_wait = options
                .completion_wait()
                .map_err(SpeculativeDriverError::Generation)?;
            if !E::Completion::supports_cancellation(completion_wait.cancellation()) {
                return Err(SpeculativeDriverError::UnsupportedCompletionCancellation {
                    cancellation: completion_wait.cancellation(),
                });
            }
            SpeculativeRequestTable::new(options, topology)
                .map_err(SpeculativeDriverError::Generation)
        })();
        let stage = eredu_core::run_preparation::TextPreparationStage::Request;
        let requests = eredu_core::run_preparation::finish_preparation(
            stage,
            prepared,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        executor.set_telemetry_enabled(component_timings_collected);
        Ok(Self {
            executor,
            context,
            optimistic_execution_available,
            component_timings_collected,
            requests,
        })
    }

    /// Prepares the actual known lane destinations before the first prefill.
    pub fn reserve_requests(&mut self, count: usize) -> Result<(), SpeculativeDriverError<E::Error>> {
        let prepared = self.requests.reserve_requests(count, self.executor, self.context);
        let stage = eredu_core::run_preparation::TextPreparationStage::Request;
        eredu_core::run_preparation::finish_preparation(stage, prepared,
            |status| self.executor.agree_text_preparation(stage, status, self.context),
            SpeculativeDriverError::Preparation)
    }

    /// Registers and prefills one independently progressing lane.
    pub fn submit(
        &mut self,
        mut lane: PreparedSpeculativeLane<'a, E, S, C, P>,
    ) -> Result<eredu_core::generation::SpeculativeRequestId, SpeculativeDriverError<E::Error>>
    {
        self.requests.submit(
            self.executor,
            lane.take_cache(),
            lane.take_input(),
            lane.take_config(),
            lane.take_runtime(),
            lane.take_randomness(),
            self.component_timings_collected,
            self.context,
        )
    }

    /// Performs one fairly selected lifecycle action.
    pub fn step(&mut self) -> Result<bool, SpeculativeDriverError<E::Error>> {
        self.requests.step(
            self.executor,
            self.optimistic_execution_available,
            self.context,
        )
    }

    /// Drives all registered lanes to terminal states.
    pub fn run(&mut self) -> Result<(), SpeculativeDriverError<E::Error>> {
        while self.step()? {}
        Ok(())
    }

    /// Returns the current portable status for one lane.
    pub fn status(
        &self,
        id: eredu_core::generation::SpeculativeRequestId,
    ) -> Option<eredu_core::generation::SpeculativeRequestStatus> {
        self.requests.status(id)
    }

    /// Requests cancellation at the next exact safe boundary.
    pub fn cancel(
        &mut self,
        id: eredu_core::generation::SpeculativeRequestId,
    ) -> Result<(), SpeculativeDriverError<E::Error>> {
        self.requests.signal_cancellation(id)
    }

    /// Whether all registered lanes are terminal.
    pub fn is_finished(&self) -> bool {
        self.requests.is_finished()
    }

    /// Consumes a terminal scheduler into stable ordered results.
    pub fn finish(
        self,
    ) -> Result<CompletedSpeculativeSchedule<S>, SpeculativeDriverError<E::Error>> {
        self.requests.finish()
    }
}

/// Facade-selected speculative generation driver.
///
/// Backends lend prepared native resources through
/// [`SpeculativeGenerationVisitor`]. This driver alone registers lanes, runs
/// the fair schedule, observes exact completions, and constructs public
/// terminal outputs.
#[derive(Debug, Clone, Copy)]
pub struct RunSpeculativeGeneration {
    options: eredu_core::generation::SpeculativeSchedulerOptions,
    started: std::time::Instant,
}

impl Default for RunSpeculativeGeneration {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl RunSpeculativeGeneration {
    /// Creates a driver with facade-selected scheduling and lookahead controls.
    /// Starts the shared request clock; construct before preparing backend inputs
    /// and resources so terminal TTFT includes that preparation.
    pub fn new(options: eredu_core::generation::SpeculativeSchedulerOptions) -> Self {
        Self {
            options,
            started: std::time::Instant::now(),
        }
    }
}

impl SpeculativeGenerationVisitor for RunSpeculativeGeneration {
    fn scheduler_options(&self) -> Option<eredu_core::generation::SpeculativeSchedulerOptions> {
        Some(self.options)
    }

    fn run<'a, E, S, C, P>(
        self,
        executor: &'a mut E,
        lanes: impl Into<SpeculativeBuffer<PreparedSpeculativeLane<'a, E, S, C, P>>>,
        topology: eredu_core::SpeculativeExecutionTopology,
        optimistic_execution_available: bool,
        component_timings_collected: bool,
        context: E::Context<'a>,
    ) -> Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E::Error>>
    where
        E: SpeculativeExecutor + 'a,
        S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>>
            + 'a,
        C: SpeculativeConstraint,
        P: SpeculativePublisher<C>,
    {
        let lanes = lanes.into();
        let mut scheduler = SpeculativeScheduler::new(
            executor,
            self.options,
            topology,
            optimistic_execution_available,
            component_timings_collected,
            context,
        )?;
        scheduler.reserve_requests(lanes.len())?;
        let prepared = (|| {
            Ok::<_, SpeculativeDriverError<E::Error>>((scheduler.executor.driver_buffer(lanes.len(), context)?,
                scheduler.executor.driver_buffer(lanes.len(), context)?))
        })();
        let stage = eredu_core::run_preparation::TextPreparationStage::Request;
        let (outputs, mut preparation_elapsed) = eredu_core::run_preparation::finish_preparation(
            stage, prepared,
            |status| scheduler.executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        for lane in lanes {
            preparation_elapsed.try_push(self.started.elapsed()).map_err(SpeculativeDriverError::Generation)?;
            scheduler.submit(lane)?;
        }
        scheduler.run()?;
        let completed = scheduler.finish()?;
        completed_output(completed, outputs, |request| {
            GenerationTiming::new(
                request
                    .stats()
                    .submission_to_first_token()
                    .map(|elapsed| preparation_elapsed[request.id().index()] + elapsed),
            )
        })
    }
}

fn completed_output<S, E: std::error::Error + 'static>(
    mut completed: CompletedSpeculativeSchedule<S>,
    mut requests: SpeculativeBuffer<SpeculativeGenerationOutput>,
    timing: impl Fn(&eredu_core::CompletedSpeculativeRequest<S>) -> GenerationTiming,
) -> Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E>> {
    for request in completed.take_requests() {
        let reason = request.finish_reason().ok_or_else(|| {
            SpeculativeDriverError::Generation(GenerationError::MissingSpeculativeFinishReason {
                index: request.id().index(),
            })
        })?;
        let timing = timing(&request);
        let mut request = request.into_artifact();
        requests.try_push(SpeculativeGenerationOutput::from_speculative(
            request.take_token_ids(), reason, timing, request.take_stats(),
        )).map_err(SpeculativeDriverError::Generation)?;
    }
    Ok(SpeculativeGenerationBatchOutput::new(requests, completed.take_scheduler()))
}
