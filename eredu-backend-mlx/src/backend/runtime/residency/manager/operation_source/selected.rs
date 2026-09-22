//! Exact source loans for independently selected residency windows.
use super::*;

/// The retained selection remains the manager's actual target or supplementary
/// source. A subset loan changes neither membership nor constructor ordinals.
#[derive(Clone)]
pub(crate) enum SelectedResidencySource {
    Supplementary(SupplementaryResidencySource),
    Target {
        source: OriginalResidencySource,
        host: Option<HostCopyWorkspace>,
    },
}

impl SelectedResidencySource {
    pub(crate) fn source(&self) -> &OriginalResidencySource {
        match self {
            Self::Supplementary(value) => value.source(),
            Self::Target { source, .. } => source,
        }
    }
    pub(crate) fn foreground(&self) -> Option<&ForegroundDiskIdentity> {
        self.source().foreground_identity()
    }
    fn host(&self) -> Option<&HostCopyWorkspace> {
        match self {
            Self::Supplementary(value) => value.host(),
            Self::Target { host, .. } => host.as_ref(),
        }
    }
    fn contains(&self, id: &OffloadUnitId) -> bool {
        match self {
            Self::Supplementary(value) => value.ids().contains(id),
            Self::Target { source, .. } => match source.selection.as_ref() {
                Some(OperationSelection::Host(value)) => {
                    value.sources().iter().any(|(unit, _)| unit.id() == id)
                }
                Some(OperationSelection::Foreground(value)) => {
                    (0..value.layout().len()).any(|ordinal| {
                        value
                            .requested_unit(ordinal)
                            .and_then(|index| value.unit(index))
                            .is_some_and(|unit| unit.id() == id)
                    })
                }
                None => false,
            },
        }
    }
    pub(crate) fn projection_control_bytes(&self) -> Option<usize> {
        let fixed = [
            size_of::<Self>(),
            size_of::<Self>(),
            size_of::<(&ResidencyManager, &Self)>(),
            size_of::<Result<(), OperationSourceFailure>>(),
            size_of::<Result<(), HostCopyWorkspaceError>>(),
        ];
        let bytes = fixed
            .into_iter()
            .try_fold(std::mem::size_of_val(&fixed), usize::checked_add)?;
        bytes.checked_add(match self {
            Self::Supplementary(value) => value.projection_control_bytes()?,
            Self::Target {
                host: Some(value), ..
            } => value.source_validation_control_bytes()?,
            Self::Target { host: None, .. } => ForegroundDiskIdentity::qualifier_control_bytes()?,
        })
    }
}

impl ResidencyManager {
    /// Borrows the exact source already selected by the installed target policy.
    /// No source snapshot or replacement selection is constructed here.
    pub(crate) fn selected_target_source(
        &self,
        source: &OriginalResidencySource,
    ) -> Result<SelectedResidencySource, OperationSourceFailure> {
        let actual = self
            .inner
            .original_operation_source
            .get()
            .ok_or(OperationSourceFailure::Layout)?;
        if !self.owns_source_windows(actual, &source.windows) {
            return Err(OperationSourceFailure::Layout);
        }
        let host = match source.selection.as_ref() {
            Some(OperationSelection::Host(_)) => Some(
                self.inner
                    .host_workspace
                    .get()
                    .ok_or(OperationSourceFailure::Layout)?
                    .clone(),
            ),
            Some(OperationSelection::Foreground(_)) => None,
            None => return Err(OperationSourceFailure::Layout),
        };
        Ok(SelectedResidencySource::Target {
            source: source.clone(),
            host,
        })
    }
    pub(crate) fn validate_selected_source(
        &self,
        source: &SelectedResidencySource,
    ) -> Result<(), OperationSourceFailure> {
        match source {
            SelectedResidencySource::Supplementary(value) => {
                self.validate_supplementary_source(value)
            }
            SelectedResidencySource::Target { source, .. } => {
                let actual = self
                    .inner
                    .original_operation_source
                    .get()
                    .ok_or(OperationSourceFailure::Layout)?;
                if self.owns_source_windows(actual, &source.windows) {
                    Ok(())
                } else {
                    Err(OperationSourceFailure::Layout)
                }
            }
        }
    }
    pub(crate) fn validate_selected_host(
        &self,
        source: &SelectedResidencySource,
    ) -> Result<(), HostCopyWorkspaceError> {
        source
            .host()
            .ok_or_else(|| {
                HostCopyWorkspaceError::Storage(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            })?
            .validate_original_snapshot(self)
    }
    pub(crate) fn selected_source_population(
        &self,
        source: &SelectedResidencySource,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<WindowPopulation, OperationSourceFailure> {
        self.validate_selected_source(source)?;
        if roots.is_empty()
            || roots
                .iter()
                .enumerate()
                .any(|(index, id)| !source.contains(id) || roots[..index].contains(id))
        {
            return Err(OperationSourceFailure::Layout);
        }
        self.selected_root_population(roots, scratch)
    }
    pub(in crate::backend::runtime::residency::manager) fn selected_root_population(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<WindowPopulation, OperationSourceFailure> {
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => OperationSourceFailure::Busy,
            std::sync::TryLockError::Poisoned(_) => OperationSourceFailure::Poisoned,
        })?;
        let mut population = WindowPopulation::collect(&state.control, roots, scratch)?;
        population.request_start = 0;
        population.request_end = roots.len();
        Ok(population)
    }
}
