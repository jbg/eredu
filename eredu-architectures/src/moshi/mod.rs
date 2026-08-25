//! Backend-neutral Moshi-family architecture policy.

mod artifact;
mod block;
mod checkpoint;
mod config;
mod depth;
mod model;
mod parallel;
pub mod personaplex_prompt;

pub(crate) use artifact::admit_checkpoint;
pub use artifact::{prepare_realtime_model, RealtimePreparationError, RealtimePreparationPlan};
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
    collective_count, forward_temporal_block_parallel, local_geometry, static_parameter_groups,
    unit_parameter_groups, LocalGeometry, LocalTransformerGeometry, MoshiCollectiveCount,
};
