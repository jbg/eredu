//! Parameter coordination borrows the same paid portable-word native transport.
use super::*;
use eredu_core::{
    capture::CaptureUsage,
    consensus::{BoundedConsensusTransport, ConsensusTransport},
    parameters::{ParameterCoordinationError, ParameterError},
    BackendFailure, MemoryLimits,
};
use eredu_runtime::{
    parameter_operations::ParameterOperationTransport,
    working_memory::{InferenceExecutionIdentity, MemoryLedger},
    CommunicationSessionIdentity,
};
use std::mem::size_of;

pub(crate) struct PreparedParameterTransport<'a> {
    session: &'a MlxDistributedSession,
    transport: PreparedConsensusTransport,
}
impl<'a> PreparedParameterTransport<'a> {
    pub(crate) fn prepare(
        session: &'a MlxDistributedSession,
        ledger: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        limits: MemoryLimits,
    ) -> Result<Self, Error> {
        let transport = PreparedConsensusTransport::prepare(session, ledger, execution, limits)?;
        transport
            .funding()
            .reserve_metadata(
                size_of::<Self>()
                    .checked_add(size_of::<Result<Self, Error>>())
                    .ok_or(Error::WorkspacePlanning(
                        eredu_core::HostMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        Ok(Self { session, transport })
    }
    pub(crate) fn funding(&self) -> &eredu_core::HostMetadataFunding {
        self.transport.funding()
    }
}
impl ConsensusTransport for PreparedParameterTransport<'_> {
    type Error = Error;
    fn participant_count(&self) -> usize {
        self.transport.participant_count()
    }
    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Error> {
        self.transport.all_gather_words(local)
    }
}
impl BoundedConsensusTransport for PreparedParameterTransport<'_> {
    type Completion = <PreparedConsensusTransport as BoundedConsensusTransport>::Completion;
    type GatherOutput = <PreparedConsensusTransport as BoundedConsensusTransport>::GatherOutput;
    fn metadata_funding(&self) -> Option<&eredu_core::HostMetadataFunding> {
        Some(self.funding())
    }
    fn submit_all_gather_words(
        &self,
        local: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Error> {
        self.transport.submit_all_gather_words(local)
    }
    fn resolve_all_gather_words(&self, output: Self::GatherOutput) -> Result<Vec<u32>, Error> {
        self.transport.resolve_all_gather_words(output)
    }
    fn with_resolved_all_gather_words<T, E, F>(
        &self,
        output: Self::GatherOutput,
        validate: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(&[u32]) -> Result<T, E>,
    {
        self.transport
            .with_resolved_all_gather_words(output, validate)
    }
}
impl ParameterOperationTransport for PreparedParameterTransport<'_> {
    fn parameter_rank(&self) -> usize {
        self.session.parameter_rank()
    }
    fn parameter_setup(&self) -> CommunicationSessionIdentity {
        self.session.parameter_setup()
    }
    fn parameter_wait(&self) -> Result<eredu_core::BoundedCompletionWait, ParameterError> {
        self.session.parameter_wait()
    }
    fn estimate_parameter_gather(&self, words: usize) -> Result<CaptureUsage, ParameterError> {
        self.session.estimate_parameter_gather(words)
    }
    fn ensure_parameter_active(&self) -> Result<(), BackendFailure> {
        self.session.ensure_parameter_active()
    }
    fn fail_parameter_operation(&self, error: &ParameterCoordinationError) {
        self.session.fail_parameter_operation(error)
    }
}
