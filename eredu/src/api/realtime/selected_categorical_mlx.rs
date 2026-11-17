//! Selected native realization of the categorical application protocol.

use std::{rc::Rc, time::Instant};

use eredu_architectures::moshi::{MoshiRealtimeExecution, MoshiRealtimeOutputProjection};
use eredu_backend_mlx::{
    backend::runtime::cache::state::MlxKeyValueState,
    native::{
        MlxPrepublicationFrame, MlxRealtimeCompletion, MlxRealtimeExecution,
        MlxRealtimeExecutionContext, RandomState,
    },
    MlxTensor,
};
use eredu_core::scheduler::{SchedulerError, WorkId};
use eredu_core::BackendFailure;
use eredu_runtime::{GenerationSampler, RealtimePayloadState};

use super::*;

type CategoricalScheduler = RealtimeSessionScheduler<
    RealtimeInputFrame,
    RealtimePayloadState<MlxKeyValueState, MlxTensor>,
    RealtimeFrameScheduleState,
    RealtimeSampling,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
    MlxPrepublicationFrame<MoshiRealtimeOutputProjection>,
>;

pub(super) struct MlxCategoricalModel {
    context: MlxRealtimeExecutionContext,
    model: MoshiRealtimeExecution<MlxRealtimeExecution>,
}

impl MlxCategoricalModel {
    pub(super) fn new(
        context: MlxRealtimeExecutionContext,
        model: MoshiRealtimeExecution<MlxRealtimeExecution>,
    ) -> Self {
        Self { context, model }
    }
}

impl CategoricalRealtimeModel for Rc<MlxCategoricalModel> {
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
        self.context.validate_execution(self.model.executor())?;
        let state = self
            .model
            .initialize_generation(&self.context, sampling)
            .map_err(BackendFailure::from_error)?;
        scheduler.register(request, state)?;
        Ok(Box::new(MlxCategoricalSession {
            model: Rc::clone(self),
            scheduler,
            request,
        }))
    }
}

struct MlxCategoricalSession {
    model: Rc<MlxCategoricalModel>,
    scheduler: CategoricalScheduler,
    request: RequestId,
}

impl CategoricalRealtimeSession for MlxCategoricalSession {
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
                .context
                .submit_realtime(model.executor(), plan, input, branch)
                .map_err(BackendFailure::from_error)
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
