//! Owning preparation controls for the shared inference quote scheduler.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::{fmt, mem::size_of};

#[derive(Clone, Copy)]
pub(super) struct Metadata<'a>(Option<&'a WorkspaceContext>);
impl<'a> Metadata<'a> {
    pub(super) fn ordinary() -> Self {
        Self(None)
    }
    pub(super) fn from_context(context: &'a WorkspaceContext) -> Self {
        Self(Some(context))
    }
    pub(super) fn context(self) -> Option<&'a WorkspaceContext> {
        self.0
    }
    pub(super) fn report(self) -> super::super::WorkspaceReportMetadata<'a> {
        self.0.map_or_else(
            super::super::WorkspaceReportMetadata::ordinary,
            super::super::WorkspaceReportMetadata::new,
        )
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
        let parts = [
            eredu_core::GenerationCancellationToken::construction_bytes()
                .ok_or(WorkspaceMetadataError::Overflow)?,
            size_of::<PrefillDriver<(), ColdCompletion, ()>>(),
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
            size_of::<
                Result<PrefillDriver<(), ColdCompletion, ()>, eredu_core::AdmissionPolicyError>,
            >(),
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
