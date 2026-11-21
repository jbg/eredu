//! Native rank, submission, and completion resources for component pipelines.

use super::{PartitionTensorAllocator, RankCommunication};
use crate::{checkpoint_mechanisms::unit_binding::NativeParameterUnitBinding, BackendFailure};

/// Native lifetime and completion mechanisms used by the component driver.
pub trait ComponentPipelineMechanisms<B: crate::communication::CommunicationBackend> {
    /// Native scope that retains pending work after the driver returns.
    type Scope: crate::checkpoint_mechanisms::unit_binding::ComponentPreparationScope;

    /// Retains the driver-selected tensor roots before work starts.
    fn begin(&self, roots: Vec<B::Tensor>) -> Result<Self::Scope, BackendFailure>;

    /// Settles driver-selected output and state roots within the admitted bound.
    fn complete_outputs(
        &self,
        roots: Vec<B::Tensor>,
        authority: &crate::communication::CommunicationAuthority,
        wait: crate::BoundedCompletionWait,
        context: &B::Executor,
    ) -> Result<crate::BoundedCompletionOutcome, BackendFailure>;

    /// Settles tensor values retained by a caller-owned cursor.
    fn settle_values<'a>(values: impl Iterator<Item = &'a B::Tensor>) -> Result<(), BackendFailure>
    where
        B::Tensor: 'a;
}

/// Communicators, allocation, and settlement for one selected component rank.
pub trait NativeComponentRank<B: eredu_nn::NeuralBackend, C>:
    NativeParameterUnitBinding<B>
{
    /// Reports whether the exact cohort can agree without inactive ranks.
    fn supports_independent_agreement(&self, world_size: usize, members: &[usize]) -> bool;

    /// Exact native communication for one selected component rank.
    type PipelineRank: RankCommunication<
        Backend: crate::communication::CommunicationBackend<Tensor = B::Tensor>,
    >;

    /// Provides the context used to construct resident units.
    fn pipeline_context(
        &self,
    ) -> &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context;

    /// Failure exchanging portable setup frames over the native world.
    type PipelineSetupError: std::error::Error + Send + Sync + 'static;

    /// Native control plane used when the rank has no retained communication.
    type PipelineSetup<'a>: crate::communication::CommunicationSetupTransport<
        Error = Self::PipelineSetupError,
    >
    where
        Self: 'a,
        C: 'a;

    /// Borrows the native world if this route needs a fresh setup.
    fn open_pipeline_setup<'a>(&'a self, communication: &'a C) -> Option<Self::PipelineSetup<'a>>;

    /// Realizes the admitted communication handles before parameter work begins.
    fn realize_pipeline_rank<'a>(
        &'a self,
        communication: &'a C,
        prepared: Option<crate::communication::WorldCommunicationBinding<Self::PipelineSetup<'a>>>,
    ) -> Result<Self::PipelineRank, crate::BackendFailure>;

    /// Allocates the exact boundary buffers requested by portable traversal.
    type PipelineAllocator: PartitionTensorAllocator<
        <Self::PipelineRank as RankCommunication>::Backend,
    >;
    /// Native submission scopes and output settlement.
    type PipelineExecution: ComponentPipelineMechanisms<
        <Self::PipelineRank as RankCommunication>::Backend,
    >;

    /// Retains only the already selected native groups.
    fn pipeline_mechanisms(
        &self,
        groups: Vec<<Self::PipelineRank as RankCommunication>::Group>,
    ) -> (Self::PipelineAllocator, Self::PipelineExecution);
}

/// Opens the native world for a selected pipeline route.
pub trait NativePipelineWorld {
    /// Exact native world retained during agreement.
    type Transport;

    /// Opens native transport without constructing an execution rank.
    fn open_world(&self) -> Result<Self::Transport, BackendFailure>;

    /// Returns native world size and this participant's global rank.
    fn identity(world: &Self::Transport) -> (usize, usize);
}
