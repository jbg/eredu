//! Embedded capture binds its genuine active role to the shared funded worker.
use crate::backend::error::Error;
use crate::composition::mlx::session::bounded_capture::funded_model;
use crate::composition::mlx::speculative::embedded_native::ActiveEmbeddedNativeInvocation;
use eredu_runtime::working_memory::OriginalEmbeddedSpeculativeRole;
use safemlx::{OriginalScopeObserver, Stream};

pub(super) type Owner = funded_model::Owner<OriginalEmbeddedSpeculativeRole>;
pub(super) type Capture = funded_model::Capture<OriginalEmbeddedSpeculativeRole>;

impl funded_model::CaptureExecution<OriginalEmbeddedSpeculativeRole>
    for ActiveEmbeddedNativeInvocation<'_>
{
    fn validate_equation_scope(&self, stream: &Stream) -> Result<(), Error> {
        self.validate_equation_scope(stream)
    }
    fn role(&self) -> &OriginalEmbeddedSpeculativeRole {
        self.role()
    }
    fn observer(&self) -> &OriginalScopeObserver {
        self.observer()
    }
}
