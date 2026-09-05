use super::*;
use eredu_architectures::decoder::{MultiTableEmbedding, NamedEmbeddingSpec};
use eredu_architectures::operator_requirements;
use eredu_checkpoint::{AffineQuantization, LinearFormat, WeightQuantization};
use eredu_nn::{
    reference_expand_heads, reference_segmented_attention, EmbeddingLookupPolicy,
    EmbeddingOperator, EmbeddingSpec, FusedProjectionLayout, FusedProjectionSegment,
    GatedProductGroupLayout, GroupedGatedProductSpec, GroupedNeuralBackend, GroupedProjectionSpec,
    GroupedRelu2Spec, HeadExpansion, JointGroupSelectionInput, JointGroupSelectionSpec,
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec, NormalizationScale,
    ParameterSpec, RelativeAttentionInput, SegmentedAttentionInput, Tensor,
};
use eredu_runtime::{
    BarrierBackend, BroadcastBackend, CommunicationBackend, CommunicationPeerCounts,
    EvenGatherBackend, SumReductionBackend, UnevenGatherBackend, VariableAllToAllBackend,
};
use safemlx::{
    ops::{quantize_with_mode, QuantizationMode},
    transforms::async_eval_with_event,
    Array, Device, DeviceType, Dtype,
};

use crate::backend::{
    nn::{linear::PhysicalEmbedding, tensor::TokenValidationScope},
    ExecutionContext,
};

use super::{MlxEmbedding, MlxLinear, MlxNeuralBackend, MlxTensor};

include!("tests/communication.rs");
include!("tests/backend_extension.rs");
include!("tests/parameters.rs");
include!("tests/operators.rs");
