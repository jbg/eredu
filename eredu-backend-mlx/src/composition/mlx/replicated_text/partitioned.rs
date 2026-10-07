use super::*;

pub(super) type MlxDirectPartitionExecutor<A> =
    eredu_architectures::partitioned_execution::DirectPartitionExecutor<
        A,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<A, MlxHybridState>,
    >;

pub(super) type MlxDirectPartitionStrategy<A> = eredu_runtime::PartitionedTextExecution<
    MlxDirectPartitionExecutor<A>,
    crate::backend::runtime::distributed::Group,
    crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    crate::backend::nn::shared::MlxCommunicationTensorMetadata,
    eredu_runtime::NoBoundaryTransport,
    eredu_runtime::OpaqueOutputPublisher,
    eredu_runtime::OpaqueFailureAgreement,
>;

pub(super) type MlxSharedAddressableBank =
    crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank;
pub(super) type MlxEmbeddedPredictionObservers =
    eredu_architectures::speculative_execution::EmbeddedPredictionObservers<
        MlxTensor,
        Array,
        Exception,
    >;
pub(super) use crate::backend::runtime::distributed::boundary::MlxPartitionTensorAllocator;

pub(super) type MlxPipelinePartitionExecutor<A, S> = eredu_runtime::PipelinePartitionExecutor<
    A,
    MlxNeuralBackend,
    S,
    MlxArchitectureLayerwisePolicy<A, S>,
    MlxPartitionTensorAllocator,
>;

pub(super) type MlxPipelinePartitionStrategy<A, S> = eredu_runtime::PartitionedTextExecution<
    MlxPipelinePartitionExecutor<A, S>,
    crate::backend::runtime::distributed::Group,
    crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    crate::backend::nn::shared::MlxCommunicationTensorMetadata,
    eredu_runtime::OpaqueBoundaryTransport,
    eredu_runtime::OpaqueOutputPublisher,
    eredu_runtime::OpaqueFailureAgreement,
>;

pub(super) use crate::backend::runtime::execution::generic::MlxSelectedLayerwisePolicy;

pub(super) type MlxArchitectureLayerwisePolicy<A, S> = MlxSelectedLayerwisePolicy<
    <A as LayeredArchitecture<MlxNeuralBackend, S>>::Unit,
    MlxSelectiveUnitPopulator,
>;
