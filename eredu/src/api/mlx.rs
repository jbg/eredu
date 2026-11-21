//! Type wiring for the selected MLX mechanisms.

use eredu_backend_mlx::{backend::MlxBackend, native::MlxSessionMechanisms};
use eredu_runtime::{executable_binding::BoundModelFor, text_session_driver::PreparedTextDriver};

use super::TextSessionComposition;

impl<Dispatch> TextSessionComposition for MlxBackend<'_, Dispatch> {
    type Model = BoundModelFor<Self>;
    type Installed = PreparedTextDriver<Self, MlxSessionMechanisms>;
    type Installer = Self::Installed;
}
