//! MLX implementation of the optional distributed-session contract.

use eredu_core::checkpoint::TensorDtype;
use eredu_core::{
    BackendError, CollectiveGroupId, CollectiveScope, DistributedBackend, DistributedCapabilities,
    DistributedSession, DistributedSessionDescriptor, Submission, ValueDescriptor,
};
use safemlx::{
    distributed::Group as NativeGroup,
    error::Exception,
    ops::{concatenate_axis, indexing::TryIndexOp, zeros_dtype},
    Array, Dtype, Stream,
};

use crate::{
    backend::error::Error,
    backend::runtime::distributed::{
        self,
        completion::{DistributedCompletion, MlxCommunicationCompletion},
        topology::ParallelCommunicators,
        Group,
    },
};

use super::MlxBackend;
use crate::MlxTensor;

mod axis;
mod consensus;
mod session;

pub use axis::{all_gather_axis, all_gather_uneven_axis, all_to_all_v_axis};
pub use consensus::MlxRealtimeConsensusTransport;
pub use session::MlxDistributedSession;

#[cfg(test)]
mod tests;

/// Initializes the Ring world from the process rank and hostfile.
pub fn initialize_ring_world() -> Result<safemlx::distributed::Group, eredu_core::BackendFailure> {
    safemlx::distributed::Group::init(true, safemlx::distributed::Backend::Ring)
        .map_err(eredu_core::BackendFailure::from_error)
}

/// Opens an MLX Ring pipeline from the process rank and hostfile.
pub struct MlxRingTransport;

impl MlxRingTransport {
    /// Whether this process can use Ring transport.
    pub fn is_available() -> bool {
        safemlx::distributed::is_available(safemlx::distributed::Backend::Ring)
    }
}

impl eredu_runtime::PipelineTransportProvider for MlxRingTransport {
    type Transport = NativeGroup;

    fn open(
        &self,
    ) -> Result<(eredu_runtime::PlacementRank, Self::Transport), eredu_core::BackendFailure> {
        let world = initialize_ring_world()?;
        let rank = eredu_runtime::PlacementRank::new(world.size(), world.rank())
            .map_err(eredu_core::BackendFailure::from_error)?;
        Ok((rank, world))
    }
}
