//! Portable construction and scheduling of categorical realtime execution.

use eredu_core::realtime_mechanisms::CategoricalRealtimeNative;
use eredu_core::realtime_mechanisms::RealtimeStartupMechanisms;
use std::{rc::Rc, time::Instant};

use eredu_architectures::moshi::{
    MoshiRealtimeExecution, MoshiRealtimeOutputProjection, PreparedMoshiRealtimeSource,
};
use eredu_core::{
    scheduler::{SchedulerError, WorkId},
    BackendFailure,
};
use eredu_runtime::{
    CategoricalRealtimeExecution, CategoricalRealtimeStateFor, GenerationSampler,
    RealtimePayloadState,
};

use super::*;

type NativeTensor<M> =
    <<M as eredu_core::realtime_mechanisms::CategoricalRealtimeNative>::Neural as eredu_nn::NeuralBackend>::Tensor;

type CategoricalScheduler<M> = RealtimeSessionScheduler<
    RealtimeInputFrame,
    RealtimePayloadState<CategoricalRealtimeStateFor<M>, NativeTensor<M>>,
    RealtimeFrameScheduleState,
    RealtimeSampling,
    GenerationSampler,
    <M as RealtimeStartupMechanisms>::Random,
    <M as RealtimeStartupMechanisms>::Completion,
    eredu_runtime::CategoricalRealtimeFrame<M, MoshiRealtimeOutputProjection>,
>;

pub(super) fn bind_categorical<M: CategoricalRealtimeNative>(
    selected: PreparedMoshiRealtimeSource,
    mechanisms: M,
) -> Result<Box<dyn CategoricalRealtimeModel>, BackendFailure> {
    let model = eredu_runtime::bind_prepared_realtime(&mechanisms, selected, None)?;
    Ok(Box::new(Rc::new(CategoricalModel { mechanisms, model })))
}

struct CategoricalModel<M: CategoricalRealtimeNative> {
    mechanisms: M,
    model: MoshiRealtimeExecution<CategoricalRealtimeExecution<M>>,
}

impl<M: CategoricalRealtimeNative> CategoricalRealtimeModel for Rc<CategoricalModel<M>> {
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
        self.model.executor().require_healthy()?;
        self.mechanisms
            .validate_execution(self.model.executor().queue())?;
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

struct CategoricalSession<M: CategoricalRealtimeNative> {
    model: Rc<CategoricalModel<M>>,
    scheduler: CategoricalScheduler<M>,
    request: RequestId,
}

impl<M: CategoricalRealtimeNative> CategoricalRealtimeSession for CategoricalSession<M> {
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
            eredu_runtime::submit_categorical_realtime_frame(
                &self.model.mechanisms,
                model.executor(),
                plan,
                input,
                branch,
            )
        })?;
        Ok(CategoricalRealtimeProgress {
            newly_submitted: progress.newly_submitted,
            committed: progress
                .committed
                .into_iter()
                .map(|(work, input, frame)| {
                    (
                        work,
                        input,
                        frame.into_host_output().map_err(BackendFailure::from_error),
                    )
                })
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
