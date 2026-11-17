//! Backend-neutral Moshi-family architecture policy.

mod artifact;
mod block;
mod checkpoint;
mod config;
mod construction;
mod declarations;
mod depth;
mod model;
mod parallel;
pub mod personaplex_prompt;
mod realtime;
mod realtime_output;

pub use realtime_output::{MoshiRealtimeOutputError, MoshiRealtimeOutputProjection};

pub(crate) use artifact::prepare_artifact;

pub use artifact::{prepare_moshi_from_catalog, MoshiPreparationError, MoshiPreparationPlan};
pub use checkpoint::{canonical_recipes, safetensors_plan};
pub use config::{
    ArtifactProfile, CheckpointLayout, EffectiveModelType, MoshiConfig, MoshiConfigError,
    MoshiIdentity, MoshiPolicy, MoshiTransformerConfig, ParameterSharing, PositionalEncoding,
    MOSHI_FAMILY, PERSONAPLEX_VERSION,
};
pub use depth::DepthSlice;
pub use model::{
    component_definition, observation_points, state_layout, DecisionBoundary, ForwardContext,
    Input, LayeredModel, MoshiKernel, ObservationPoint, StaticModules, Unit, DEPTH_STATE_SEGMENT,
    TEMPORAL_STATE_SEGMENT,
};
pub use parallel::{
    collective_count, forward_temporal_block_parallel, local_geometry, parameter_contract,
    parameter_description, select_parallel_execution, LocalGeometry, LocalTransformerGeometry,
    MoshiCollectiveCount, MoshiMatrixContract, MoshiParallelSelection, MoshiParameterContract,
};
pub use realtime::{
    inspect_moshi_realtime, moshi_realtime_request_from_normalized,
    prepare_moshi_realtime_with_support, realtime_frame_contract, select_inspected_moshi_realtime,
    select_moshi_realtime_with_support, InspectedMoshiRealtime, MoshiRealtimeDispatchError,
    MoshiRealtimeExecution, MoshiRealtimeExecutionDescriptor, MoshiRealtimeGeneration,
    MoshiRealtimePreparationError, MoshiRealtimeRequest, MoshiRealtimeSamplingError,
    MoshiRealtimeSelectionError, MoshiRealtimeSourceError, MoshiRealtimeStartupError,
    PreparedMoshiRealtime, PreparedMoshiRealtimeSource,
};
