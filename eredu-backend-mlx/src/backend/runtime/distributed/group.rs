//! Backend-owned logical groups layered over native MLX communication groups.

use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId};
use eredu_runtime::{
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationOperation, CommunicationOperationRequirement,
};
use safemlx::{
    distributed as native,
    error::{Exception, Result},
    ops::{
        concatenate_axis, indexing::TryIndexMutOp, indexing::TryIndexOp, stack_axis, zeros_dtype,
    },
    Array, Dtype, Stream,
};

mod collectives;
mod handle;
mod point_to_point;

pub use collectives::{all_gather, all_sum, all_to_all_v};
pub(crate) use collectives::{
    all_gather_for, all_gather_unchecked, all_sum_for, payload_free_all_sum_for,
};
pub use handle::Group;
pub(crate) use point_to_point::recv_like;
pub use point_to_point::{recv, send};

#[cfg(test)]
pub(crate) use handle::{
    contracted_collective_submissions, native_collective_submissions,
    reset_native_collective_submissions,
};
