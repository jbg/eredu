//! MLX implementation of the optional distributed-session contract.

use eredu_core::checkpoint::TensorDtype;
use eredu_core::{
    BackendError, CollectiveGroupId, CollectiveScope, DistributedCapabilities, DistributedSession,
    DistributedSessionDescriptor, Submission, ValueDescriptor,
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

use crate::MlxTensor;

mod axis;
pub(crate) use axis::validate_uneven_axis;
mod consensus;
mod session;

pub use axis::{all_gather_axis, all_gather_uneven_axis, all_to_all_v_axis};
pub use consensus::MlxRealtimeConsensusTransport;
pub use session::MlxDistributedSession;

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

impl eredu_core::communication::pipeline::NativePipelineWorld for MlxRingTransport {
    type Transport = NativeGroup;

    fn open_world(&self) -> Result<Self::Transport, eredu_core::BackendFailure> {
        initialize_ring_world()
    }

    fn identity(world: &Self::Transport) -> (usize, usize) {
        (world.size(), world.rank())
    }
}
