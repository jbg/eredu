//! MLX implementation of backend-neutral neural operators.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Arc,
};

use eredu_checkpoint::LinearFormat;
use eredu_core::{checkpoint::TensorDtype, Completion, Submission};
use eredu_nn::{
    validate_parameter_topology, AttentionCache, AttentionRequest, BlockwiseAttentionBackend,
    BlockwiseAttentionSpec, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    Error as ComputeError, GatedDeltaScanInput, GatedDeltaScanOutput, GatedProductGroupLayout,
    GatedProductPolicy, GroupScoring, GroupSelection, GroupSelectionOperator,
    GroupedGatedProductOperator, GroupedGatedProductSpec, GroupedNeuralBackend,
    GroupedRelu2Operator, GroupedRelu2Spec, HyperConnectionOperator, HyperConnectionSpec,
    HyperConnectionState, HyperHeadOperator, HyperHeadSpec, HyperNeuralBackend,
    IndexedAttentionInput, JointGroupSelection, JointGroupSelectionInput, LinearFormatSpec,
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, ParameterMetadata, ParameterSpec, ParameterVisitor,
    ParameterVisitorMut, Parameterized, PooledAttentionInput, PooledPositionInput,
    RelativeAttentionInput, RotaryOperator, RotaryPosition, RotarySpec, SegmentedAttentionInput,
    SelectiveStateSpaceScanInput, SelectiveStateSpaceScanOutput, Tensor,
    TensorParallelGroupedGatedProductOperator, TensorParallelGroupedOutput,
    TensorParallelGroupedRelu2Operator, TopKGroupSelectorSpec, VocabularyParallelRange,
};
use eredu_runtime::{
    BarrierBackend, BroadcastBackend, CommunicationBackend, CommunicationPeerCounts,
    EvenGatherBackend, FailureAgreementBackend, ParameterBackend, PointToPointBackend,
    RoleExactBoundaryValue, SubmissionBackend, SumReductionBackend, TransferBackend,
    UnevenGatherBackend, VariableAllToAllBackend,
};
use ref_cast::RefCast;
use safemlx::ops::{
    arange, argpartition_axis, broadcast_to, clip, concatenate_axis, einsum,
    indexing::{take_along_axis, NewAxis, TryIndexOp},
    matmul, maximum, r#where, sigmoid, softmax_axis, zeros_dtype,
};
use safemlx::{
    fast::ScaledDotProductAttentionMask, Array, Dtype, Event, HostTransferBuffer,
    HostTransferPolicy, ImmutableHostTransferBuffer, Stream,
};

use crate::backend::runtime::distributed::{
    completion::{MlxCommunicationCompletion, MlxFailureAgreement},
    recv_like, send,
    topology::CommunicationRouteRealization,
    Group,
};
use crate::backend::{
    nn::{
        self as common,
        rope::{self, RopeVariant},
        tensor::validate_token_domain,
    },
    runtime::cache::kv::{
        BlockwiseAttentionAccumulator, ConcatKeyValueCache, KeyValueAttentionBlock, KeyValueCache,
        PagedKeyValueCache,
    },
};
#[cfg(test)]
use crate::module::ModuleParamMut;
use crate::MlxTensor;
use crate::{
    module::{Module, ModuleParamRef, PhysicalParameters},
    nested::NestedValue,
    nn,
};

#[cfg(test)]
fn trace_partition_collective(operation: &str, input: &Array, group: &Group, detail: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    if std::env::var_os("EREDU_TEST_PARTITION_COLLECTIVE_TRACE").is_some() {
        static ORDINAL: AtomicUsize = AtomicUsize::new(0);
        eprintln!(
            "partition-schedule rank={} ordinal={} operation={} subgroup={}/{} shape={:?} {}",
            std::env::var("MLX_RANK").unwrap_or_else(|_| "?".into()),
            ORDINAL.fetch_add(1, Ordering::Relaxed),
            operation,
            group.rank(),
            group.size(),
            input.shape(),
            detail,
        );
    }
}

mod core_backend;
mod extensions;
mod operators;
mod selected_linear;
pub use selected_linear::MlxGroupedLinear;
mod parameters;
mod submission;

pub use core_backend::MlxParameterError;
pub use operators::*;
pub use parameters::*;
pub use submission::*;

/// Zero-sized MLX backend selector. All calls are statically dispatched.
#[derive(Debug, Clone, Copy)]
pub struct MlxNeuralBackend;

#[cfg(test)]
mod tests;
