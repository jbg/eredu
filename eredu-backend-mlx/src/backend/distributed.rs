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
