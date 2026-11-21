//! Native completion and retirement for enrolled tensor copies.

use eredu_core::{
    cache::mechanisms::{CacheCopyScope, CacheCopySubmission},
    BackendFailure,
};
use safemlx::{error::Exception, transforms::async_eval_with_event, Array, Event, Stream};

use super::MlxCacheBackend;
use crate::backend::{
    ordinary_retirement::{self, OrdinaryRetirement},
    submission_recovery::{self, Recovery, Retention, Status},
};

struct CopyResources<R> {
    owners: R,
    event: Option<Event>,
    _queue: Stream,
}

struct CopyRetention<R: 'static>(OrdinaryRetirement<CopyResources<R>>);

impl<R: 'static> Retention for CopyRetention<R> {
    fn observe(&self, _: Status) {}
}

/// Native scope that keeps opaque copy owners live until retirement.
#[doc(hidden)]
pub struct MlxCacheCopyScope<R: 'static>(Recovery<CopyRetention<R>>);

impl<R: 'static> CacheCopyScope<Array, R> for MlxCacheCopyScope<R> {
    fn resources(&self) -> &R {
        &self.0.retention().0.owners
    }

    fn resources_mut(&mut self) -> &mut R {
        &mut self.0.retention_mut().0.owners
    }

    fn settle(&mut self, roots: &[Array]) -> Result<(), BackendFailure> {
        let event = async_eval_with_event(roots).map_err(BackendFailure::from_error)?;
        self.0.retention_mut().0.event = Some(event);
        self.0.seal();
        let status = self.0.wait();
        if status.failed || status.blocked {
            return Err(BackendFailure::from_error(Exception::custom(
                "state copy failed; unresolved resources remain retained",
            )));
        }
        self.0
            .retention()
            .0
            .event
            .as_ref()
            .expect("copy event")
            .synchronize()
            .map_err(BackendFailure::from_error)
    }

    fn finish(self) {
        self.0.finish();
        ordinary_retirement::reclaim();
    }
}

impl CacheCopySubmission for MlxCacheBackend {
    type CopyScope<R: 'static> = MlxCacheCopyScope<R>;

    fn begin_copy<R: 'static>(
        queue: &Stream,
        owners: R,
    ) -> Result<Self::CopyScope<R>, BackendFailure> {
        Recovery::begin(CopyRetention(OrdinaryRetirement::new(CopyResources {
            owners,
            event: None,
            _queue: queue.clone(),
        })))
        .map(MlxCacheCopyScope)
        .map_err(BackendFailure::from_error)
    }

    fn reclaim_copies() {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
    }
}
