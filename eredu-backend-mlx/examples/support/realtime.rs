#![allow(dead_code)]

use std::time::{Duration, Instant};

use eredu_architectures::moshi::MoshiRealtimeExecution;
use eredu_backend_mlx::native::{
    MlxRealtimeCompletion, MlxRealtimeExecution, MlxRealtimeExecutionContext,
    MlxRealtimeHostObserver, RandomState,
};
use eredu_core::{
    scheduler::{RequestId, SchedulerLimits},
    RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
};
use eredu_runtime::{
    GenerationSampler, RealtimeGenerationState, RealtimePayloadState, RealtimeSessionScheduler,
};
use safemlx::Array;

pub type PreparedRealtime = MoshiRealtimeExecution<MlxRealtimeExecution>;

#[derive(Debug)]
pub struct DriverError(String);

impl std::fmt::Display for DriverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DriverError {}

fn driver_error(error: impl std::fmt::Display) -> DriverError {
    DriverError(error.to_string())
}

pub type Scheduler = RealtimeSessionScheduler<
    eredu_runtime::RealtimePayloadState<
        eredu_backend_mlx::backend::runtime::cache::state::MlxKeyValueState,
        eredu_backend_mlx::MlxTensor,
    >,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
    eredu_runtime::PrepublicationRealtimeFrame<
        eredu_backend_mlx::MlxTensor,
        MlxRealtimeCompletion,
        MlxRealtimeHostObserver,
    >,
>;

pub fn load(
    backend: &MlxRealtimeExecutionContext,
    preparation: eredu_architectures::moshi::RealtimePreparationPlan,
    options: eredu_backend_mlx::MlxLoadRequest,
) -> Result<PreparedRealtime, DriverError> {
    let selected =
        MlxRealtimeExecutionContext::select_realtime_execution(preparation, &options, false)
            .map_err(driver_error)?;
    backend
        .materialize_realtime_execution(selected, options)
        .map_err(driver_error)
}

pub fn scheduler(
    backend: &MlxRealtimeExecutionContext,
    model: &PreparedRealtime,
    request: RequestId,
    sampling: RealtimeSampling,
) -> Result<Scheduler, DriverError> {
    let mut scheduler = RealtimeSessionScheduler::new(
        eredu_runtime::RealtimeModelSessionIdentity::from_selected(model.selected()),
        SchedulerLimits::new(1, 1).map_err(driver_error)?,
    )
    .map_err(driver_error)?;
    let schedule = model.execution_config().frame_schedule().clone();
    let samplers = eredu_architectures::moshi::realtime_generation_samplers(&schedule, sampling)
        .map_err(driver_error)?;
    let model_state = RealtimePayloadState::fresh(
        backend
            .new_realtime_model_state(model)
            .map_err(driver_error)?,
        schedule.clone(),
    );
    let random = backend
        .realize_random_state(sampling.is_stochastic().then_some(sampling.seed()))
        .map_err(driver_error)?;
    scheduler
        .register(
            request,
            RealtimeGenerationState::new(model_state, schedule, sampling, samplers, random)
                .map_err(driver_error)?,
        )
        .map_err(driver_error)?;
    Ok(scheduler)
}

pub fn frame(array: &Array) -> Result<RealtimeInputFrame, DriverError> {
    let batch = usize::try_from(array.dim(0)).map_err(driver_error)?;
    Ok(RealtimeInputFrame::new(
        batch,
        array
            .evaluated()
            .map_err(driver_error)?
            .as_slice::<i32>()
            .to_vec(),
    ))
}

pub fn run_frame(
    scheduler: &mut Scheduler,
    backend: &MlxRealtimeExecutionContext,
    model: &mut PreparedRealtime,
    request: RequestId,
    frame: RealtimeInputFrame,
) -> Result<RealtimeOutputFrame, DriverError> {
    scheduler.enqueue(request, frame).map_err(driver_error)?;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let mut progress = scheduler
            .run_local_turn(Instant::now(), |_, frame, branch| {
                backend.submit_realtime_frame(model, frame, branch)
            })
            .map_err(driver_error)?;
        if let Some((_, _, output)) = progress.committed.pop() {
            return output.into_host_output().map_err(driver_error);
        }
        if let Some((_, error)) = progress.failed.pop() {
            return Err(driver_error(error));
        }
        if Instant::now() >= deadline {
            return Err(DriverError("realtime frame exceeded 60 seconds".into()));
        }
        std::thread::yield_now();
    }
}

pub struct SelectedRealtimeDriver {
    backend: MlxRealtimeExecutionContext,
    model: PreparedRealtime,
    scheduler: Option<Scheduler>,
    request: RequestId,
}

impl SelectedRealtimeDriver {
    pub fn new(backend: MlxRealtimeExecutionContext, model: PreparedRealtime) -> Self {
        Self {
            backend,
            model,
            scheduler: None,
            request: RequestId::new(1),
        }
    }

    pub fn model(&self) -> &PreparedRealtime {
        &self.model
    }
}

impl eredu_evaluation::RealtimeEvaluationDriver for SelectedRealtimeDriver {
    type Error = DriverError;

    fn speech_config(&self) -> &eredu_core::RealtimeSpeechConfig {
        self.model.execution_config().frame_schedule()
    }

    fn start_trace(&mut self, sampling: RealtimeSampling) -> Result<(), Self::Error> {
        if self.scheduler.is_some() {
            return Err(DriverError("realtime trace is already active".into()));
        }
        self.scheduler = Some(
            scheduler(&self.backend, &self.model, self.request, sampling).map_err(driver_error)?,
        );
        Ok(())
    }

    fn evaluate_frame(
        &mut self,
        frame: RealtimeInputFrame,
    ) -> Result<RealtimeOutputFrame, Self::Error> {
        run_frame(
            self.scheduler
                .as_mut()
                .ok_or_else(|| DriverError("realtime trace is not active".into()))?,
            &self.backend,
            &mut self.model,
            self.request,
            frame,
        )
        .map_err(driver_error)
    }

    fn finish_trace(&mut self) -> Result<(), Self::Error> {
        let mut scheduler = self
            .scheduler
            .take()
            .ok_or_else(|| DriverError("realtime trace is not active".into()))?;
        scheduler.finish(self.request).map_err(driver_error)
    }
}
