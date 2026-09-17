use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
};

use eredu_architectures::{
    agree_expert_route_counts, decoder, deepseek, exchange_expert_rows,
    execute_routed_gated_product, gemma4, gpt_oss, inkling, kimi_linear, lfm2, llama, moshi,
    muse_glimmer, nemotron_h, qwen,
    replicated_text::{
        dispatch_replicated_text_architecture, CompositeTextArchitectureVisitor,
        PreparedCompositeTextArchitecture, PreparedReplicatedTextArchitecture,
        ReplicatedTextArchitectureVisitor, ReplicatedTextStateProfiles,
        SharedReplicatedTextVisitor,
    },
    ExpertParameterRecipe, ExpertParameterRole, ExpertRealizationPlan, ExpertResidencyCatalog,
    ExpertResidencyDistribution, ExpertResidencyUnit, ExpertRouteCountPlan,
    ExpertRouteExchangeDirection, ExpertRoutePackingPlan, PlannedAddressableGatedProduct,
    PlannedAddressableRelu2, PlannedResidentGatedProduct, PreparedRoutedTextArchitecture,
    RoutedTextArchitectureVisitor, RoutedTextExecutionError,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{
        EncodedTensorLease, ReadPolicy, RetainedCheckpointSource, TensorReadRequest,
        TensorSelection,
    },
};
use eredu_core::cache::{LayerCachePolicy, PromptCacheTopology, StateTensorRole};
use eredu_core::{
    CollectiveGroupId, Completion, CompletionCancellationMode, DistributedCommitOutcome,
    LayerSchedule, ModelConfigurationResolver, ParallelRankTopology, ParallelTopology, TokenFilter,
};
use eredu_nn::{
    reference_gated_delta_scan, reference_selective_state_space_scan, validate_parameter_topology,
    AttentionCache, AttentionMask, AttentionRequest, BlockwiseAttentionBackend,
    BlockwiseAttentionSpec, CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec,
    CompressedAttentionBlock, CompressedAttentionCache, CompressedAttentionScan,
    CompressedAttentionState, CompressedAttentionView, ConvolutionActivation,
    EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, FusedProjectionLayout,
    FusedProjectionSegment, GatedDeltaScanInput, GatedDeltaScanOutput, GatedProductGroupLayout,
    GatedProductPolicy, GatedShortConvolution, GatedShortConvolutionSpec, GroupSelection,
    GroupSelectionOperator, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedLinearOperator, GroupedNeuralBackend, GroupedRelu2Operator, GroupedRelu2Spec,
    HyperConnection, HyperConnectionOperator, HyperConnectionSpec, HyperConnectionState, HyperHead,
    HyperHeadOperator, HyperHeadSpec, HyperNeuralBackend, Index, IndexedAttentionInput,
    JointGroupSelection, JointGroupSelectionInput, LinearOperator, LinearSpec, LowRankProjection,
    LowRankProjectionSpec, NeuralBackend, NormalizationConstructionSpec, NormalizationOperator,
    NormalizationScale, PadMode, ParameterMetadata, ParameterSpec, ParameterVisitor,
    ParameterVisitorMut, Parameterized, PooledAttentionInput, PooledPositionInput,
    PoolingAttentionCache, PoolingOverlap, PoolingWindows, RelativeAttentionInput, RotaryOperator,
    RotaryPosition, RotarySpec, RotarySubspace, SelectiveStateSpaceScanInput,
    SelectiveStateSpaceScanOutput, Tensor, TensorParallelGroupedGatedProductOperator,
    TensorParallelGroupedOutput, TensorParallelGroupedRelu2Operator, TopKGroupSelectionSpec,
    TopKGroupSelectorSpec, VocabularyParallelRange,
};
use eredu_runtime::{
    AddressableExpertRouteProvider, AddressableExpertRouteRequest, AddressableGatedProductBank,
    AddressableGroupedBank, ArchitectureParameters, ArchitecturePartition,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, CollectiveBackend,
    CommunicationBackend, CommunicationCompletionPolicy, CommunicationGroupDescriptor,
    CommunicationGroupRequirements, CommunicationManifest, CommunicationOperation,
    CommunicationOperationRequirement, CommunicationTensorLimits, CommunicationTensorMetadata,
    CompositeLayeredTraversalHook, DeviceState, EvenGatherBackend, ExecutionGroupId,
    ExecutionUnitAddress, ExpertPass, ExpertRouteExchange, ExpertRouteTensorMovement,
    IndexedMovement, LayerRuntimeState, LayerWeightResidency, LayeredArchitecture,
    LayeredPartitionDriver, LayeredPartitionInput, LayeredPartitionOutput, LayeredTraversalHook,
    LayerwiseAcquireError, LayerwisePolicy, LayerwiseRuntime, LocalModelLayout, LocalTensorLayout,
    MemberSharding, NoAuxiliaryBoundary, NoAuxiliaryBoundarySchema, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, ParameterBankAcquisition, ParameterBankKey,
    ParameterBankLoadOptions, ParameterGroupSpec, ParameterRole, PartitionCommunication,
    PartitionOwnership, PenaltyConfig, PredictionDirective, RealizedCommunicationGroup,
    RealizedCommunicationRoute, ReplicatedTextMaterializationTask, ReplicatedTextSessionMechanisms,
    ResettableRuntimeLayerState, ResidentRuntime, ResidentUnitWindow, RoutedExpertProvider,
    RoutedExpertRequest, RoutedExpertTensorParallelOutput, RoutedLayeredArchitecture,
    RuntimeLayerState, RuntimeState, RuntimeStateComponents, Sampler, SamplingBackend,
    SequentialDecisionDriver, SequentialDecisionPlan, SequentialDecisionSource,
    SequentialDecisionTraversal, StateError, SubmissionBackend, TensorParallelRoutedExpertProvider,
    TensorPlacement, TokenDomain, VariableAllToAllBackend,
};
use safetensors::tensor::Dtype;
