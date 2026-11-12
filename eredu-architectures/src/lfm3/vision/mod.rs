//! Portable LFM3 gated-convolution and axial-attention vision components.
//!
//! The released VL artifact retains its complete source. Vision units are
//! independently constructible and the shared decoder accepts projected media.

mod artifact;
mod component;
mod config;
mod declarations;
mod execution;
pub(crate) mod gguf;
mod kernel;
mod model;
mod policy;
mod prefill;
mod preparation;
pub(crate) use preparation::validate_rank_state;
mod prompt;

pub(crate) use artifact::DECODER_ROLE;
pub use artifact::{
    MediaTokens, PreparedVisionLanguage, PreparedVisionWeights, PreparedVlDecoder,
    PreparedVlTensorDecoder, VisionLanguageDeclaration,
};
pub use config::{AxialRope, ProjectorConfig, VisionConfig, VisionLayerKind};
pub use declarations::{Declaration, VisionUnit};
pub use execution::{VisionAdvance, VisionExecution, VisionModule};
pub use gguf::resolve_gguf_projector;
pub use kernel::{VlInput, VlKernel, VlLayeredModel, VlUnit};
pub use model::{VisionBlock, VisionInput, VisionPatchifier, VisionProjector, VisionState};

pub use policy::{
    PolicyVision, PolicyVisionProgress, PolicyVisionSession, PreparedVisionExecution,
};
pub use prefill::{VlPrefill, VlPrefillProgress};
pub use preparation::{
    PreparedVlGraph, PreparedVlRankGraph, SelectedVlGraph, SelectedVlRankGraph,
    VL_SESSION_CAPABILITIES,
};
pub use prompt::{
    ProjectedVisionPrompt, VisionPrompt, VisionPromptAdmission, VisionPromptPreparation,
    VlInputPartPlan,
};

mod selection;
pub use selection::{
    PreparedVlRankWeights, PreparedVlWeights, SelectedVlRankPlacement, SelectedVlRankWeights,
    SelectedVlWeights, VlDeviceWeightBytes, VlWeightPolicies,
};
