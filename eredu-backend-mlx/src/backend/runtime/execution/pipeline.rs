//! Native storage, communication, and completion for component pipelines.

use crate::{
    backend::{
        distributed::MlxDistributedSession,
        nn::shared::MlxNeuralBackend,
        runtime::distributed::{completion::MlxCommunicationCompletion, Group},
        submission_recovery::MlxComponentPreparationScope,
    },
    MlxTensor,
};
use eredu_core::communication::pipeline::ComponentPipelineMechanisms;
use eredu_core::communication::CommunicationAuthority;
use eredu_core::communication::CommunicationOperation;
use eredu_core::{
    BackendFailure, BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait,
};
use safemlx::Stream;
/// Native lifetime resources used by the portable pipeline driver.
pub struct MlxPipelineMechanisms {
    pub(crate) groups: Vec<Group>,
}

impl ComponentPipelineMechanisms<MlxNeuralBackend> for MlxPipelineMechanisms {
    type Scope = MlxComponentPreparationScope;

    fn begin(&self, roots: Vec<MlxTensor>) -> Result<Self::Scope, BackendFailure> {
        MlxComponentPreparationScope::begin_with_roots(
            false,
            roots
                .into_iter()
                .map(|tensor| tensor.as_array().clone())
                .collect(),
        )
    }

    fn complete_outputs(
        &self,
        roots: Vec<MlxTensor>,
        authority: &CommunicationAuthority,
        wait: BoundedCompletionWait,
        stream: &Stream,
    ) -> Result<BoundedCompletionOutcome, BackendFailure> {
        let arrays = roots
            .into_iter()
            .map(|tensor| tensor.as_array().clone())
            .collect::<Vec<_>>();
        MlxCommunicationCompletion::submit(
            arrays.iter(),
            arrays.clone(),
            vec![],
            self.groups.clone(),
            vec![],
            vec![stream.clone()],
        )
        .map(|completion| {
            completion.with_authority(authority.clone(), CommunicationOperation::FailureAgreement)
        })
        .and_then(|completion| completion.wait_bounded(wait))
        .map_err(|error| {
            authority.completion_failure(error, CommunicationOperation::FailureAgreement, None)
        })
        .map_err(BackendFailure::from_error)
    }

    fn settle_values<'a>(
        values: impl Iterator<Item = &'a MlxTensor>,
    ) -> Result<(), BackendFailure> {
        safemlx::transforms::eval(values.map(MlxTensor::as_array))
            .map_err(BackendFailure::from_error)
    }
}

pub(crate) trait PipelineCommunication {
    fn world(&self) -> Option<&safemlx::distributed::Group>;
    fn session(
        &self,
        prepared: Option<
            eredu_core::communication::WorldCommunicationBinding<
                crate::native::NativeWorldManifestTransport<'_>,
            >,
        >,
    ) -> Result<MlxDistributedSession, BackendFailure>;
}

impl PipelineCommunication for safemlx::distributed::Group {
    fn world(&self) -> Option<&safemlx::distributed::Group> {
        Some(self)
    }

    fn session(
        &self,
        prepared: Option<
            eredu_core::communication::WorldCommunicationBinding<
                crate::native::NativeWorldManifestTransport<'_>,
            >,
        >,
    ) -> Result<MlxDistributedSession, BackendFailure> {
        let prepared = prepared.ok_or_else(|| {
            BackendFailure::from_error(std::io::Error::other(
                "component pipeline requires an agreed world setup",
            ))
        })?;
        MlxDistributedSession::from_prepared(prepared).map_err(BackendFailure::from_error)
    }
}

impl PipelineCommunication for MlxDistributedSession {
    fn world(&self) -> Option<&safemlx::distributed::Group> {
        None
    }

    fn session(
        &self,
        prepared: Option<
            eredu_core::communication::WorldCommunicationBinding<
                crate::native::NativeWorldManifestTransport<'_>,
            >,
        >,
    ) -> Result<MlxDistributedSession, BackendFailure> {
        if prepared.is_some() {
            return Err(BackendFailure::from_error(std::io::Error::other(
                "component pipeline already owns communication",
            )));
        }
        Ok(self.clone())
    }
}
