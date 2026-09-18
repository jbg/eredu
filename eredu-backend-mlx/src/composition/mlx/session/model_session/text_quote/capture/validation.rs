//! Borrowed source comparison and its prospective typed rejection destination.
use eredu_core::{
    capture::{AdmittedCapturePlan, CapturePhase, CaptureRevalidationError},
    ObservationCatalog, ObservationPoint, ObservationSupportReport, ObservationSupportStatus,
};
use eredu_runtime::working_memory::WorkspaceReportMetadata;
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct RetainedFailure<E> {
    cause: E,
    // The closed NN source allocation retires before this payload/account.
    _funding: Option<eredu_core::HostMetadataFunding>,
}
impl<E: std::fmt::Display> std::fmt::Display for RetainedFailure<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for RetainedFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(&self.cause) }
}
pub(super) fn source<E: std::error::Error + Send + Sync + 'static>(
    metadata: WorkspaceReportMetadata<'_>, cause: E,
) -> eredu_nn::Error {
    metadata.source(RetainedFailure { cause, _funding: metadata.funding() })
}

pub(super) fn validate_parts(
    admission: &AdmittedCapturePlan,
    catalog: &ObservationCatalog,
    support: &ObservationSupportReport,
    metadata: WorkspaceReportMetadata<'_>,
) -> Result<(), eredu_nn::Error> {
    let phase = |point: &ObservationPoint, phase| {
        support.points.iter().find(|entry| entry.path == point.path)
            .is_some_and(|entry| matches!(match phase {
                CapturePhase::Prefill => &entry.prefill,
                CapturePhase::Decode => &entry.decode,
            }, ObservationSupportStatus::Supported | ObservationSupportStatus::Conditional(_)))
    };
    let bytes = AdmittedCapturePlan::borrowed_declaration_validation_control_bytes()
        .and_then(|bytes| bytes.checked_add(size_of_val(&phase)))
        .and_then(|bytes| bytes.checked_add(size_of::<(
            &AdmittedCapturePlan, &ObservationCatalog, &ObservationSupportReport,
            WorkspaceReportMetadata<'_>, Result<(), CaptureRevalidationError>,
            Result<(), eredu_nn::Error>,
            std::slice::Iter<'static, eredu_core::ObservationSupport>,
            &eredu_core::ObservationSupport, &ObservationPoint, CapturePhase,
            std::slice::Iter<'static, eredu_core::capture::CaptureSelection>,
        )>()));
    metadata.charge(bytes.ok_or_else(|| source(
        metadata, eredu_core::HostMetadataFundingError::Overflow,
    ))?).map_err(|cause| metadata.error(cause))?;
    admission.revalidate_borrowed_declarations(
        catalog, support.schema_version, &support.capture, phase,
    ).map_err(|cause| source(metadata, cause))?;
    // Ordinary revalidation requires a selected support row even if this
    // selection enables neither phase. Preserve that check independently of
    // the enabled-phase predicate supplied to the shared fixed validator.
    if admission.plan().selections.iter().any(|selection|
        !support.points.iter().any(|entry| entry.path == selection.path))
    {
        return Err(source(metadata, CaptureRevalidationError::Phase));
    }
    Ok(())
}

#[cfg(test)]
#[path = "validation/tests.rs"]
mod tests;
