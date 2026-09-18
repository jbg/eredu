//! Revalidation borrows an existing immutable proof and current discovery.
use super::*;

/// Allocation-free mismatch against a retained capture declaration or capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CaptureRevalidationError {
    /// Schema versions differ from the supported contracts.
    #[error("unsupported capture schema version")]
    Schema,
    /// Selected declarations differ from this immutable admission.
    #[error("capture declaration differs from the selected source")]
    Declaration,
    /// A requested transform or physical limit lacks current support.
    #[error("selected capture capability is unavailable")]
    Capability,
    /// Histogram edges exceed the current selected capability.
    #[error("capture histogram edges exceed the selected capability")]
    Histogram,
    /// The selected execution cannot emit an enabled phase.
    #[error("selected capture phase is unavailable")]
    Phase,
}

impl AdmittedCapturePlan {
    /// Validates this exact immutable admission against current selected facts.
    ///
    /// Selected declarations must be unchanged. Request, origin, invocation,
    /// schedule and geometry invariants come from the existing opaque admission;
    /// current schemas, transformation limits and enabled-phase support use the
    /// same checks as initial admission. Successful validation borrows all data:
    /// it neither clones plan buffers nor reconstructs its digest. Error messages
    /// can allocate. Artifact/session identity and execution authority remain
    /// obligations of the enclosing session, as with `readmit`.
    ///
    /// To admit changed catalog semantics, use `readmit` and explicitly accept
    /// the resulting new identity instead of validating this existing proof.
    pub fn revalidate(&self, discovery: &CaptureDiscovery) -> Result<(), CaptureError> {
        self.revalidate_parts(&discovery.catalog, &discovery.support)
    }

    /// Revalidates against borrowed catalog and selected collector facts.
    ///
    /// This is the same semantic check as [`Self::revalidate`], without building
    /// a discovery DTO or resolving an artifact identity. The enclosing session
    /// must authenticate the selected facts, source, origin and execution binding.
    /// Success neither clones payloads nor computes a new admission digest.
    pub fn revalidate_parts(
        &self,
        catalog: &ObservationCatalog,
        support: &ObservationSupportReport,
    ) -> Result<(), CaptureError> {
        let capabilities = &support.capture;
        validate_schema(&self.plan, catalog, support, capabilities)?;
        if self.plan.selections.len() != self.points.len() {
            return Err(CaptureError::Invalid(
                "capture admission point count mismatch".into(),
            ));
        }
        for (selection, retained) in self.plan.selections.iter().zip(&self.points) {
            let selected = catalog
                .get(&selection.path)
                .ok_or_else(|| CaptureError::MissingPath(selection.path.clone()))?;
            if selected != retained {
                return Err(CaptureError::Invalid(
                    "capture admission does not match this session's catalog".into(),
                ));
            }
            validate_transform(selection, capabilities)?;
            validate_phase_support(selection, support)?;
            validate_histogram(selection, capabilities)?;
        }
        Ok(())
    }
}

impl AdmittedCapturePlan {
    /// Revalidates source declarations and current capabilities without allocating
    /// diagnostics. The supplied borrowed predicate must describe the actual
    /// selected hook for this point and phase; it grants no native authority.
    pub fn revalidate_borrowed_declarations(
        &self,
        catalog: &ObservationCatalog,
        support_schema: u32,
        capabilities: &CaptureCapabilities,
        mut phase: impl FnMut(&ObservationPoint, CapturePhase) -> bool,
    ) -> Result<(), CaptureRevalidationError> {
        check_schema(
            &self.plan,
            catalog.schema_version,
            support_schema,
            capabilities,
        )?;
        if self.plan.selections.len() != self.points.len() {
            return Err(CaptureRevalidationError::Declaration);
        }
        for (selection, retained) in self.plan.selections.iter().zip(&self.points) {
            let selected = catalog
                .get(&selection.path)
                .ok_or(CaptureRevalidationError::Declaration)?;
            if selected != retained {
                return Err(CaptureRevalidationError::Declaration);
            }
            check_transform(selection, capabilities)?;
            for (enabled, current) in [
                (selection.schedule.prefill, CapturePhase::Prefill),
                (selection.schedule.decode, CapturePhase::Decode),
            ] {
                if enabled && !phase(selected, current) {
                    return Err(CaptureRevalidationError::Phase);
                }
            }
            check_histogram(selection, capabilities)?;
        }
        Ok(())
    }

    /// Fixed comparison/iterator frames; callers additionally price their actual
    /// phase callback and its borrowed source controls. No payload is copied.
    pub fn borrowed_declaration_validation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<(&Self, &ObservationCatalog, u32, &CaptureCapabilities)>(),
            size_of::<
                std::iter::Zip<
                    std::slice::Iter<'static, CaptureSelection>,
                    std::slice::Iter<'static, ObservationPoint>,
                >,
            >(),
            size_of::<(&CaptureSelection, &ObservationPoint, &ObservationPoint)>(),
            size_of::<std::slice::Iter<'static, ObservationPoint>>(),
            size_of::<(&ObservationCatalog, &str)>(),
            size_of::<[(bool, CapturePhase); 2]>(),
            size_of::<std::array::IntoIter<(bool, CapturePhase), 2>>(),
            size_of::<(&CapturePlan, u32, u32, &CaptureCapabilities)>(),
            size_of::<(&CaptureSelection, &CaptureCapabilities)>(),
            size_of::<CaptureTransformKind>(),
            size_of::<std::slice::Iter<'static, CaptureTransformKind>>(),
            size_of::<std::slice::Iter<'static, f32>>(),
            size_of::<std::slice::Windows<'static, f32>>(),
            size_of::<&[f32]>(),
            size_of::<(bool, CapturePhase)>(),
            size_of::<Result<(), CaptureRevalidationError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn check_schema(
    plan: &CapturePlan,
    catalog_schema: u32,
    support_schema: u32,
    capabilities: &CaptureCapabilities,
) -> Result<(), CaptureRevalidationError> {
    if plan.schema_version != CAPTURE_SCHEMA_VERSION
        || catalog_schema != crate::DISCOVERY_SCHEMA_VERSION
        || support_schema != crate::DISCOVERY_SCHEMA_VERSION
    {
        return Err(CaptureRevalidationError::Schema);
    }
    if plan.limits.physical_native_bytes.is_some() && !capabilities.physical_native_limit {
        return Err(CaptureRevalidationError::Capability);
    }
    Ok(())
}
fn check_transform(
    selection: &CaptureSelection,
    capabilities: &CaptureCapabilities,
) -> Result<(), CaptureRevalidationError> {
    if !capabilities
        .transformations
        .contains(&selection.transform.kind())
    {
        return Err(CaptureRevalidationError::Capability);
    }
    Ok(())
}
fn check_histogram(
    selection: &CaptureSelection,
    capabilities: &CaptureCapabilities,
) -> Result<(), CaptureRevalidationError> {
    if let CaptureTransform::Histogram { edges } = &selection.transform {
        if edges.len() < 2
            || (edges.len() - 1) as u64 > capabilities.max_histogram_bins
            || edges.iter().any(|edge| !edge.is_finite())
            || edges.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(CaptureRevalidationError::Histogram);
        }
    }
    Ok(())
}

pub(super) fn validate_schema(plan: &CapturePlan, catalog: &ObservationCatalog,
    support: &ObservationSupportReport, capabilities: &CaptureCapabilities) -> Result<(), CaptureError> {
    validate_schema_with(plan,catalog,support,capabilities,super::admission::allocation::Allocation(None))
}
pub(super) fn validate_schema_with(plan: &CapturePlan, catalog: &ObservationCatalog,
    support: &ObservationSupportReport, capabilities: &CaptureCapabilities,
    allocation: super::admission::allocation::Allocation<'_>) -> Result<(), CaptureError> {
    match check_schema(plan,catalog.schema_version,support.schema_version,capabilities) {
        Ok(()) => Ok(()),
        Err(CaptureRevalidationError::Schema) => Err(CaptureError::Invalid(allocation.text("unsupported schema version")?)),
        Err(_) => Err(CaptureError::Unsupported(allocation.text("physical native allocator/workspace bound")?)),
    }
}
pub(super) fn validate_transform(selection: &CaptureSelection, capabilities: &CaptureCapabilities) -> Result<(), CaptureError> {
    validate_transform_with(selection,capabilities,super::admission::allocation::Allocation(None))
}
pub(super) fn validate_transform_with(selection: &CaptureSelection, capabilities: &CaptureCapabilities,
    allocation: super::admission::allocation::Allocation<'_>) -> Result<(), CaptureError> {
    if check_transform(selection,capabilities).is_err() {
        return Err(CaptureError::Unsupported(allocation.format(format_args!("{:?}",selection.transform.kind()))?));
    }
    Ok(())
}
pub(super) fn validate_phase_support(selection: &CaptureSelection, support: &ObservationSupportReport) -> Result<(), CaptureError> {
    validate_phase_support_with(selection,support,super::admission::allocation::Allocation(None))
}
pub(super) fn validate_phase_support_with(selection: &CaptureSelection, support: &ObservationSupportReport,
    allocation: super::admission::allocation::Allocation<'_>) -> Result<(), CaptureError> {
    let phase_support=match support.points.iter().find(|p|p.path==selection.path) {
        Some(value)=>value,
        None=>return Err(CaptureError::Unsupported(allocation.format(format_args!("no selected support for {}",selection.path))?)),
    };
    for(enabled,status) in [(selection.schedule.prefill,&phase_support.prefill),(selection.schedule.decode,&phase_support.decode)] {
        if enabled && !matches!(status,ObservationSupportStatus::Supported|ObservationSupportStatus::Conditional(_)) {
            return Err(CaptureError::Unsupported(allocation.format(format_args!("{}: {status:?}",selection.path))?));
        }
    }
    Ok(())
}
pub(super) fn validate_histogram(selection: &CaptureSelection, capabilities: &CaptureCapabilities) -> Result<(), CaptureError> {
    validate_histogram_with(selection,capabilities,super::admission::allocation::Allocation(None))
}
pub(super) fn validate_histogram_with(selection: &CaptureSelection, capabilities: &CaptureCapabilities,
    allocation: super::admission::allocation::Allocation<'_>) -> Result<(), CaptureError> {
    if check_histogram(selection,capabilities).is_err() {
        return Err(CaptureError::Invalid(allocation.text("histogram edges must be finite, increasing, and within the bin limit")?));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
