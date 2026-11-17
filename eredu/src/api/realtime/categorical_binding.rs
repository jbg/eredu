//! Portable construction and scheduling of categorical realtime execution.

use std::{rc::Rc, time::Instant};

use eredu_architectures::moshi::{
    MoshiRealtimeExecution, MoshiRealtimeOutputProjection, PreparedMoshiRealtimeSource,
};
use eredu_core::{
    scheduler::{SchedulerError, WorkId},
    BackendFailure,
};
use eredu_runtime::{
    CategoricalRealtimeMechanisms, GenerationSampler, RealtimePayloadState,
    RealtimeStartupMechanisms,
};

use super::*;

type NativeTensor<M> =
    <<M as CategoricalRealtimeMechanisms>::Neural as eredu_nn::NeuralBackend>::Tensor;

type CategoricalScheduler<M> = RealtimeSessionScheduler<
    RealtimeInputFrame,
    RealtimePayloadState<<M as CategoricalRealtimeMechanisms>::State, NativeTensor<M>>,
    RealtimeFrameScheduleState,
    RealtimeSampling,
    GenerationSampler,
    <M as RealtimeStartupMechanisms>::Random,
    <M as RealtimeStartupMechanisms>::Completion,
    <M as CategoricalRealtimeMechanisms>::Frame<MoshiRealtimeOutputProjection>,
>;

pub(super) fn bind_categorical<M: CategoricalRealtimeMechanisms>(
    selected: PreparedMoshiRealtimeSource,
    mechanisms: M,
) -> Result<Box<dyn CategoricalRealtimeModel>, BackendFailure> {
    let model = mechanisms.bind(selected)?;
    Ok(Box::new(Rc::new(CategoricalModel { mechanisms, model })))
}

struct CategoricalModel<M: CategoricalRealtimeMechanisms> {
    mechanisms: M,
    model: MoshiRealtimeExecution<M::Execution>,
}

impl<M: CategoricalRealtimeMechanisms> CategoricalRealtimeModel for Rc<CategoricalModel<M>> {
    fn frame_schedule(&self) -> &RealtimeSpeechConfig {
        self.model.execution_config().frame_schedule()
    }

    fn start(
        &self,
        request: RequestId,
        sampling: RealtimeSampling,
        limits: SchedulerLimits,
    ) -> Result<Box<dyn CategoricalRealtimeSession>, CategoricalRealtimeStartError> {
        let mut scheduler = RealtimeSessionScheduler::new(
            RealtimeModelSessionIdentity::from_selected(self.model.selected()),
            limits,
        )?;
        self.mechanisms.validate_execution(self.model.executor())?;
        let state = self
            .model
            .initialize_generation(&self.mechanisms, sampling)
            .map_err(BackendFailure::from_error)?;
        scheduler.register(request, state)?;
        Ok(Box::new(CategoricalSession {
            model: Rc::clone(self),
            scheduler,
            request,
        }))
    }
}

struct CategoricalSession<M: CategoricalRealtimeMechanisms> {
    model: Rc<CategoricalModel<M>>,
    scheduler: CategoricalScheduler<M>,
    request: RequestId,
}

impl<M: CategoricalRealtimeMechanisms> CategoricalRealtimeSession for CategoricalSession<M> {
    fn enqueue(&mut self, frame: RealtimeInputFrame) -> Result<WorkId, SchedulerError> {
        self.scheduler.enqueue(self.request, frame)
    }

    fn advance(&mut self, now: Instant) -> Result<CategoricalRealtimeProgress, SchedulerError> {
        let model = &self.model.model;
        let progress = self.scheduler.run_local_turn(now, |_, input, branch| {
            let plan = model
                .prepare_frame()
                .map_err(std::io::Error::other)
                .map_err(BackendFailure::from_error)?;
            self.model
                .mechanisms
                .submit(model.executor(), plan, input, branch)
        })?;
        Ok(CategoricalRealtimeProgress {
            newly_submitted: progress.newly_submitted,
            committed: progress
                .committed
                .into_iter()
                .map(|(work, input, frame)| (work, input, M::into_host_output(frame)))
                .collect(),
            failed: progress.failed,
        })
    }

    fn status(&self) -> Option<RequestStatus> {
        self.scheduler.request_status(self.request)
    }
    fn finish(&mut self) -> Result<(), SchedulerError> {
        self.scheduler.finish(self.request)
    }
    fn cancel(&mut self) -> Result<(), SchedulerError> {
        self.scheduler.cancel(self.request)
    }
}
