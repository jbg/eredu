//! Backend-neutral Moshi-family architecture policy.

mod artifact;
mod block;
mod checkpoint;
mod config;
mod depth;
mod model;
mod parallel;
pub mod personaplex_prompt;
mod realtime;

pub use artifact::{
    prepare_realtime_model, prepare_realtime_model_from_catalog, RealtimePreparationError,
    RealtimePreparationPlan,
};
pub use checkpoint::{canonical_recipes, safetensors_plan};
pub use config::{
    ArtifactProfile, CheckpointLayout, EffectiveModelType, MoshiConfig, MoshiConfigError,
    MoshiIdentity, MoshiTransformerConfig, ParameterSharing, PositionalEncoding, MOSHI_FAMILY,
    PERSONAPLEX_VERSION,
};
pub use depth::DepthSlice;
pub use model::{
    observation_points, state_layout, DecisionBoundary, ForwardContext, Input, LayeredModel,
    ObservationPoint, StaticModules, Unit, DEPTH_STATE_SEGMENT, TEMPORAL_STATE_SEGMENT,
};
pub use parallel::{
    collective_count, forward_temporal_block_parallel, local_geometry, parameter_contract,
    parameter_description, select_parallel_execution, static_parameter_groups,
    unit_parameter_groups, LocalGeometry, LocalTransformerGeometry, MoshiCollectiveCount,
    MoshiMatrixContract, MoshiParallelSelection, MoshiParameterContract,
};
pub use realtime::{
    execute_detached_partitioned_moshi_realtime, execute_detached_replicated_moshi_realtime,
    execute_replicated_moshi_realtime, inspect_moshi_realtime, realtime_decision_execution,
    realtime_generation_samplers, realtime_ingress_contract, select_inspected_moshi_realtime,
    select_moshi_realtime, visit_selected_moshi_realtime_architecture, InspectedMoshiRealtime,
    MoshiPreparedRealtimeFrameExecutor, MoshiRealtimeArchitectureVisitor,
    MoshiRealtimeDispatchError, MoshiRealtimeExecution, MoshiRealtimeExecutionArchitecture,
    MoshiRealtimeExecutionDescriptor, MoshiRealtimeExecutionError, MoshiRealtimeRequest,
    MoshiRealtimeSamplingError, MoshiRealtimeSelectionError, PreparedMoshiRealtime,
    PreparedMoshiRealtimeArchitecture,
};
