//! Backend-neutral speculative request lifecycle and fair scheduling.

use eredu_core::{
    CompletedSpeculativeSchedule, PreparedSpeculativeLane, SpeculativeConstraint,
    SpeculativeDriverError, SpeculativeExecutor, SpeculativeGenerationBatchOutput,
    SpeculativeGenerationOutput, SpeculativeGenerationVisitor, SpeculativePublisher,
    SpeculativeRequestTable, SpeculativeSampling,
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
        options: eredu_core::generation::MtpSchedulerOptions,
        topology: eredu_core::SpeculativeExecutionTopology,
        optimistic_execution_available: bool,
        component_timings_collected: bool,
        context: E::Context<'a>,
    ) -> Result<Self, SpeculativeDriverError<E::Error>> {
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
        lane: PreparedSpeculativeLane<'a, E, S, C, P>,
    ) -> Result<eredu_core::generation::MtpRequestId, SpeculativeDriverError<E::Error>> {
        self.requests.submit(
            self.executor,
            lane.cache,
            lane.input,
            lane.config,
            lane.runtime,
            lane.randomness,
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

    /// Returns the current portable phase for one lane.
    pub fn phase(
        &self,
        id: eredu_core::generation::MtpRequestId,
    ) -> Option<eredu_core::generation::MtpRequestPhase> {
        self.requests.phase(id)
    }

    /// Requests cancellation at the next exact safe boundary.
    pub fn cancel(
        &mut self,
        id: eredu_core::generation::MtpRequestId,
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
    options: eredu_core::generation::MtpSchedulerOptions,
}

impl RunSpeculativeGeneration {
    /// Creates a driver with facade-selected scheduling and lookahead controls.
    pub const fn new(options: eredu_core::generation::MtpSchedulerOptions) -> Self {
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
        let completed = scheduler.finish()?;
        let requests = completed
            .requests
            .into_iter()
            .map(|request| -> Result<_, SpeculativeDriverError<E::Error>> {
                let finish_reason = request.finish_reason.ok_or_else(|| {
                    SpeculativeDriverError::Generation(
                        eredu_core::generation::GenerationError::MissingMtpFinishReason {
                            index: request.id.index(),
                        },
                    )
                })?;
                Ok(SpeculativeGenerationOutput {
                    token_ids: request.token_ids,
                    finish_reason,
                    stats: request.stats,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SpeculativeGenerationBatchOutput {
            requests,
            scheduler: completed.scheduler,
        })
    }
}
