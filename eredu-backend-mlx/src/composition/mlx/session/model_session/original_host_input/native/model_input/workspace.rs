//! Actual selected native consumer of the original B source's metadata projection.
use super::*;
mod recipe;
use crate::composition::mlx::model::NativeLayerwiseParameters;
use eredu_architectures::prepared_execution::{
    OriginalMediaWorkspaceInput, OriginalMediaWorkspaceInputError, OriginalMediaWorkspaceReport,
    OriginalMediaWorkspaceTraceError,
};
use eredu_nn::workspace::WorkspaceContext;

#[derive(Debug)]
enum Cause {
    Boundary(WorkingMemoryError),
    Native(Error),
    Projection(OriginalMediaWorkspaceInputError),
    Trace(OriginalMediaWorkspaceTraceError),
}
/// Every allocating error remains under genuine ordinary custody. The outer
/// result is inline; no unpriced owning box is added after custody retirement.
#[derive(Debug)]
pub(crate) struct MlxOriginalMediaWorkspaceError {
    cause: Cause,
    _ordinary: Option<NativeMemoryOwner>,
}
impl std::fmt::Display for MlxOriginalMediaWorkspaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            Cause::Boundary(e) => std::fmt::Display::fmt(e, f),
            Cause::Native(e) => std::fmt::Display::fmt(e, f),
            Cause::Projection(e) => std::fmt::Display::fmt(e, f),
            Cause::Trace(e) => std::fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for MlxOriginalMediaWorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Boundary(e) => e,
            Cause::Native(e) => e,
            Cause::Projection(e) => e,
            Cause::Trace(e) => e,
        })
    }
}
impl MlxModelInput {
    /// Ordinary diagnostic only: actual current session, original source slots,
    /// retained blueprint, native facts and selected parameter/state projection.
    /// No native execution, implicit synchronization, source registration or grant.
    pub(crate) fn quote_original_media_workspace(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<OriginalMediaWorkspaceReport, MlxOriginalMediaWorkspaceError> {
        let ordinary = NativeMemoryOwner::acquire_typed(runtime.backend().memory_ledger())
            .map_err(|e| MlxOriginalMediaWorkspaceError {
                cause: Cause::Boundary(e),
                _ordinary: None,
            })?;
        let result = (|| {
            let Some(input::OriginalMediaPacket::Original(packet)) = self.original_media.as_ref()
            else {
                return Err(Cause::Boundary(WorkingMemoryError::UnknownBound));
            };
            packet
                .body
                .source()
                .validate_pool(runtime.backend().memory_ledger())
                .map_err(Cause::Boundary)?;
            let session = runtime.session();
            if session.poison.get() || session.authority.borrow().require_idle().is_err() {
                return Err(Cause::Boundary(WorkingMemoryError::ExecutionFenced));
            }
            if !runtime
                .backend()
                .matches_prepared_target(&session.payload.target)
            {
                return Err(Cause::Boundary(WorkingMemoryError::IdentityMismatch));
            }
            let executable = &session.payload.model;
            let current = executable
                .erased()
                .current_media_semantic_binding()
                .map_err(|e| Cause::Native(Error::Other(Box::new(e))))?;
            if !packet.semantics.binding().matches(&current) {
                return Err(Cause::Boundary(WorkingMemoryError::IdentityMismatch));
            }
            let blueprint = executable
                .inference_blueprint()
                .ok_or(Cause::Boundary(WorkingMemoryError::UnknownBound))?;
            let facts = executable
                .resident_workspace_mechanisms()
                .ok_or(Cause::Boundary(WorkingMemoryError::UnknownBound))?;
            let context = WorkspaceContext::new(facts);
            let parameters = executable.layerwise_workspace().map_err(Cause::Native)?;
            let _parameters = executable
                .install_parameter_source(&context, parameters.as_ref())
                .map_err(|cause| Cause::Native(Error::Other(Box::new(cause))))?;
            let lease = ordinary.unquoted_lease().map_err(Cause::Native)?;
            let input = OriginalMediaWorkspaceInput::project(
                packet.body.prepared().expect("complete B prepared"),
                packet.semantics.clone(),
                &context,
                &lease,
            )
            .map_err(Cause::Projection)?;
            let batch = std::num::NonZeroU32::new(
                u32::try_from(geometry.batch_size)
                    .map_err(|_| Cause::Boundary(WorkingMemoryError::Overflow))?,
            )
            .ok_or(Cause::Boundary(WorkingMemoryError::IdentityMismatch))?;
            let state = executable
                .erased()
                .project_resident_workspace(batch, &context)
                .map_err(Cause::Native)?;

            match parameters.as_ref() {
                Some(parameters) => blueprint.quote_original_media_ordinary(
                    input,
                    &current,
                    geometry,
                    &state,
                    &context,
                    Some(&NativeLayerwiseParameters(parameters)),
                ),
                None => blueprint.quote_original_media_ordinary(
                    input, &current, geometry, &state, &context, None,
                ),
            }
            .map_err(Cause::Trace)
        })();
        match result {
            Ok(report) => Ok(report),
            Err(cause) => Err(MlxOriginalMediaWorkspaceError {
                cause,
                _ordinary: Some(ordinary),
            }),
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
