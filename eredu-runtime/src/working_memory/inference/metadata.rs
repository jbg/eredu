//! Owning preparation controls for the shared inference quote scheduler.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::{alloc::Layout, fmt, mem::size_of};

#[derive(Clone, Copy)]
pub(super) struct Metadata<'a>(Option<&'a WorkspaceContext>);
impl<'a> Metadata<'a> {
    pub(super) fn ordinary() -> Self {
        Self(None)
    }
    pub(super) fn from_context(context: &'a WorkspaceContext) -> Self {
        Self(context.uses_checked_metadata().then_some(context))
    }
    pub(super) fn context(self) -> Option<&'a WorkspaceContext> {
        self.0
    }
    pub(super) fn validate_geometry<E>(
        self,
        geometry: InferenceGeometry,
    ) -> Result<(), InferenceWorkspaceError<E>> {
        if self.0.is_some() {
            geometry.validate_fixed()?;
        } else {
            geometry.validate()?;
        }
        Ok(())
    }
    pub(super) fn text(self, arguments: fmt::Arguments<'_>) -> Result<String, Error> {
        match self.0 {
            Some(context) => context.metadata_string(arguments),
            None => Ok(arguments.to_string()),
        }
    }
    pub(super) fn invalid<E>(self, detail: &'static str) -> InferenceWorkspaceError<E> {
        match self.0 {
            Some(context) => {
                InferenceWorkspaceError::Metadata(context.metadata_error(format_args!("{detail}")))
            }
            None => super::invalid(detail).into(),
        }
    }
    pub(super) fn prefill_error<E>(
        self,
        error: super::super::WorkingMemoryError,
    ) -> InferenceWorkspaceError<E> {
        match self.0 {
            Some(context) => InferenceWorkspaceError::Metadata(context.metadata_source(error)),
            None => super::invalid(&error.to_string()).into(),
        }
    }
    pub(super) fn reserve<T>(self, values: &mut Vec<T>, additional: usize) -> Result<(), Error> {
        if let Some(context) = self.0 {
            context.reserve_metadata_vec(values, additional)?;
        }
        Ok(())
    }
    pub(super) fn admit_schedule<F, E, R>(self) -> Result<(), Error> {
        let Some(context) = self.0 else {
            return Ok(());
        };
        let identity = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(Layout::new::<()>())
            .ok()
            .map(|value| value.0.pad_to_align().size())
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let request = super::super::InferenceRequest::unbudgeted_control_bytes()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let parts = [
            identity,
            usize::try_from(request).map_err(|_| WorkspaceMetadataError::Overflow)?,
            GenerationCancellationToken::construction_bytes()
                .ok_or(WorkspaceMetadataError::Overflow)?,
            super::super::control_mutex::operation_control_bytes::<
                super::super::text_preparation::RequestStart,
            >(),
            size_of::<InferenceExecutionIdentity>(),
            size_of::<InferenceRequest>(),
            size_of::<PrefillDriver<(), ColdCompletion>>(),
            size_of::<ColdCompletion>(),
            size_of::<Inspection<'_, F>>(),
            size_of::<F>(),
            size_of::<R>(),
            size_of::<Result<R, E>>(),
            size_of::<&WorkspaceTraceReport>(),
            size_of::<Metadata<'_>>(),
            size_of::<InferenceWorkspaceReport>(),
            size_of::<Option<WorkspaceBorrowedStorage>>(),
            size_of::<std::cell::Ref<'static, Option<WorkspaceBorrowedStorage>>>(),
            size_of::<InferenceWorkspaceError<E>>(),
            size_of::<Result<InferenceWorkspaceReport, InferenceWorkspaceError<E>>>(),
            size_of::<Result<PrefillOutcome, PrefillError<InferenceWorkspaceError<E>, Infallible>>>(
            ),
            size_of::<Result<PrefillDriver<(), ColdCompletion>, super::super::WorkingMemoryError>>(
            ),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(bytes)?;
        Ok(())
    }
}

pub(super) struct Assumptions<'a>(pub &'a [String]);
impl fmt::Display for Assumptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, text) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str("; ")?;
            }
            f.write_str(text)?;
        }
        Ok(())
    }
}
