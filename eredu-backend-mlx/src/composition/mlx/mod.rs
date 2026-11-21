//! Cold-path neutral execution selection and MLX mechanism composition.

pub mod automatic;
mod capability;
pub(crate) mod component_binding;
pub mod distributed;
mod inspection;
pub mod loading;
mod prepared_speculative;
mod processor;
pub mod realtime;
pub(crate) mod replicated_text;
mod session;
pub mod speculative;
pub mod structural;

pub use inspection::{inspection_mechanisms, inspection_media_features};
#[cfg(any(feature = "image", feature = "audio"))]
pub use processor::MlxMediaInputLowering;
pub use session::MlxSessionMechanisms;
pub use session::{MlxModelOutput, MlxSessionCompletion};

pub(crate) use crate::backend::{error::Error, MlxBackend, MlxCompletion, MlxDistributedSession};
