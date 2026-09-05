//! Backend-neutral speculative request lifecycle and fair scheduling.

use eredu_core::{
    BoundedCompletion, CompletedSpeculativeSchedule, PreparedSpeculativeLane,
    SpeculativeConstraint, SpeculativeDriverError, SpeculativeExecutor,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationOutput, SpeculativeGenerationVisitor,
    SpeculativePublisher, SpeculativeRequestTable, SpeculativeSampling,
};

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
        let completion_wait = options
            .completion_wait()
            .map_err(SpeculativeDriverError::Generation)?;
        if !E::Completion::supports_cancellation(completion_wait.cancellation()) {
            return Err(SpeculativeDriverError::UnsupportedCompletionCancellation {
                cancellation: completion_wait.cancellation(),
            });
        }
        executor.set_telemetry_enabled(component_timings_collected);
        Ok(Self {
            executor,
            context,
            optimistic_execution_available,
            component_timings_collected,
            requests: SpeculativeRequestTable::new(options, topology)
                .map_err(SpeculativeDriverError::Generation)?,
        })
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
        self.requests.cancel(id)
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
#[derive(Debug, Clone, Copy, Default)]
pub struct RunSpeculativeGeneration {
    options: eredu_core::generation::SpeculativeSchedulerOptions,
}

impl RunSpeculativeGeneration {
    /// Creates a driver with facade-selected scheduling and lookahead controls.
    pub const fn new(options: eredu_core::generation::SpeculativeSchedulerOptions) -> Self {
        Self { options }
    }
}

impl SpeculativeGenerationVisitor for RunSpeculativeGeneration {
    fn run<'a, E, S, C, P>(
        self,
        executor: &'a mut E,
        lanes: Vec<PreparedSpeculativeLane<'a, E, S, C, P>>,
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
        let mut scheduler = SpeculativeScheduler::new(
            executor,
            self.options,
            topology,
            optimistic_execution_available,
            component_timings_collected,
            context,
        )?;
        for lane in lanes {
            scheduler.submit(lane)?;
        }
        scheduler.run()?;
        let mut completed = scheduler.finish()?;
        let requests = completed
            .take_requests()
            .into_iter()
            .map(|request| -> Result<_, SpeculativeDriverError<E::Error>> {
                let finish_reason = request.finish_reason().ok_or_else(|| {
                    SpeculativeDriverError::Generation(
                        eredu_core::generation::GenerationError::MissingSpeculativeFinishReason {
                            index: request.id().index(),
                        },
                    )
                })?;
                Ok(SpeculativeGenerationOutput::new(
                    request.token_ids().to_vec(),
                    finish_reason,
                    request.stats().clone(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SpeculativeGenerationBatchOutput::new(
            requests,
            completed.take_scheduler(),
        ))
    }
}
