#![allow(dead_code)]

use std::time::{Duration, Instant};

use eredu_architectures::moshi::MoshiRealtimeExecution;
use eredu_backend_mlx::native::{
    MlxPrepublicationFrame, MlxRealtimeCompletion, MlxRealtimeExecution,
    MlxRealtimeExecutionContext, RandomState,
};
use eredu_core::{
    scheduler::{RequestId, SchedulerLimits},
    RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
};
use eredu_runtime::{GenerationSampler, RealtimeSessionScheduler};
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
    RealtimeInputFrame,
    eredu_runtime::RealtimePayloadState<
        eredu_backend_mlx::backend::runtime::cache::state::MlxKeyValueState,
        eredu_backend_mlx::MlxTensor,
    >,
    eredu_core::RealtimeFrameScheduleState,
    RealtimeSampling,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
    MlxPrepublicationFrame<eredu_architectures::moshi::MoshiRealtimeOutputProjection>,
>;

pub fn load(
    backend: &MlxRealtimeExecutionContext,
    preparation: eredu_architectures::moshi::MoshiPreparationPlan,
    options: eredu_backend_mlx::MlxLoadRequest,
) -> Result<PreparedRealtime, DriverError> {
    let selected = eredu_architectures::moshi::prepare_moshi_realtime_with_support(
        preparation,
        options.normalized(),
        &eredu_backend_mlx::native::MlxRealtimeSupport::new(false),
    )
    .map_err(driver_error)?;
    backend.bind_realtime(selected, None).map_err(driver_error)
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
    backend
        .validate_execution(model.executor())
        .map_err(driver_error)?;
    let generation = model
        .initialize_generation(backend, sampling)
        .map_err(driver_error)?;
    scheduler
        .register(request, generation)
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
                let plan = model.prepare_frame().map_err(driver_error)?;
                backend
                    .submit_realtime(model.executor(), plan, frame, branch)
                    .map_err(driver_error)
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
