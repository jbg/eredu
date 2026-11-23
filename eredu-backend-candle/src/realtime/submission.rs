//! Frame submission and exact host readback on a synchronous Candle context.

use std::{rc::Rc, sync::Arc};

use eredu_core::{
    eager_completion::EagerCompletion,
    realtime_mechanisms::{
        RealtimeCompletionCreationError, RealtimeQueue, RealtimeSubmissionOwner,
        RealtimeTensorReadback,
    },
    BackendFailure, Completion,
};
use eredu_nn::Tensor;

use crate::{device::CandleSettlement, CandleContext, CandleTensor};

#[doc(hidden)]
#[derive(Clone)]
pub struct CandleRealtimeCompletion(
    Result<EagerCompletion<BackendFailure, Rc<dyn RealtimeSubmissionOwner>>, Arc<BackendFailure>>,
);

impl Completion for CandleRealtimeCompletion {
    type Error = BackendFailure;

    fn is_complete(&self) -> Result<bool, BackendFailure> {
        self.wait().map(|()| true)
    }

    fn wait(&self) -> Result<(), BackendFailure> {
        match &self.0 {
            Ok(completion) => completion.wait(),
            Err(error) => Err(BackendFailure::new(error.kind(), Arc::clone(error))
                .with_operation(error.operation())
                .with_model_state_preserved(error.model_state_preserved())),
        }
    }
}

struct PendingFrame {
    owner: Rc<dyn RealtimeSubmissionOwner>,
    roots: Vec<CandleTensor>,
}

#[doc(hidden)]
pub struct CandleRealtimeScope(Option<CandleSettlement<PendingFrame>>);

impl CandleRealtimeScope {
    fn complete(mut self, roots: Vec<CandleTensor>, abandoned: bool) -> CandleRealtimeCompletion {
        let mut settlement = self.0.take().expect("live Candle realtime scope");
        let owner = Rc::clone(&settlement.resources().owner);
        settlement.resources_mut().roots.extend(roots);
        let result = settlement
            .finish()
            .map(|frame| EagerCompletion::completed_retaining(frame.owner));
        if abandoned || result.is_err() {
            owner.poison();
        }
        CandleRealtimeCompletion(result.map_err(Arc::new))
    }
}

impl Drop for CandleRealtimeScope {
    fn drop(&mut self) {
        if let Some(settlement) = &self.0 {
            settlement.resources().owner.poison();
        }
    }
}

impl RealtimeQueue for CandleContext {
    type Tensor = CandleTensor;
    type Readback = Self;
    type Completion = CandleRealtimeCompletion;
    type Scope = CandleRealtimeScope;
    type Retired<T: 'static> = Box<T>;

    fn context(&self) -> &Self {
        self
    }
    fn readback(&self) -> Self {
        self.clone()
    }
    fn retire<T: 'static>(value: T) -> Box<T> {
        Box::new(value)
    }
    fn reclaim() {
        Self::reclaim_retired();
    }
    fn release_retired() {
        Self::reclaim_retired();
    }

    fn begin(
        &self,
        owner: impl RealtimeSubmissionOwner + 'static,
        initial: &[CandleTensor],
    ) -> Result<Self::Scope, BackendFailure> {
        self.check_healthy()?;
        for root in initial {
            self.validate_tensor_device(root)?;
        }
        Ok(CandleRealtimeScope(Some(CandleSettlement::new(
            self,
            PendingFrame {
                owner: Rc::new(owner),
                roots: initial.to_vec(),
            },
        ))))
    }

    fn finish(
        scope: Self::Scope,
        roots: Vec<CandleTensor>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, BackendFailure>>
    {
        let completion = scope.complete(roots, false);
        match completion.wait() {
            Ok(()) => Ok(completion),
            Err(error) => Err(RealtimeCompletionCreationError::after_submission(
                error, completion,
            )),
        }
    }

    fn retained_resources(_: &Self::Completion) -> usize {
        1
    }

    fn abandon(scope: Self::Scope, roots: Vec<CandleTensor>) -> Self::Completion {
        scope.complete(roots, true)
    }
    fn finish_startup() -> Result<(), BackendFailure> {
        Ok(())
    }
}

impl RealtimeTensorReadback for CandleContext {
    type Tensor = CandleTensor;

    fn read_i32(&self, value: &CandleTensor) -> Result<Vec<i32>, BackendFailure> {
        use candle_core::DType;

        self.validate_tensor_device(value)?;
        match value.value.dtype() {
            DType::I32 => value.to_i32_vec(self).map_err(BackendFailure::from_error),
            DType::U32 => value
                .value
                .flatten_all()
                .and_then(|values| values.to_vec1::<u32>())
                .map_err(BackendFailure::from_error)?
                .into_iter()
                .map(|id| i32::try_from(id).map_err(BackendFailure::from_error))
                .collect(),
            _ => Err(crate::text::invalid(
                "realtime integer readback requires exact 32-bit integer storage",
            )),
        }
    }

    fn read_f32(&self, value: &CandleTensor) -> Result<Vec<f32>, BackendFailure> {
        self.validate_tensor_device(value)?;
        value.to_f32_vec(self).map_err(BackendFailure::from_error)
    }
}
