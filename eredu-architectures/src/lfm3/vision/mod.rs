//! Portable LFM3 gated-convolution and axial-attention vision components.
//!
//! Components are independently constructible. Their existence does not admit a
//! conditional-generation task through the application facade.

mod config;
mod declarations;
mod model;

pub use config::{AxialRope, ProjectorConfig, VisionConfig, VisionLayerKind};
pub use declarations::{Declaration, VisionUnit};
pub use model::{VisionBlock, VisionInput, VisionPatchifier, VisionProjector, VisionState};
