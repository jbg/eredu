//! Fixed reads of the actual immutable host/disk snapshot owned by the workspace.
use super::*;
use eredu_runtime::working_memory::{
    WorkspaceParameterLifetime as Lifetime, WorkspaceParameterOwner as Owner,
    WorkspaceParameterRow as Row, WorkspaceParameterRows as Rows,
    WorkspaceParameterSourceError as Failure, WorkspaceParameterSourceLoan,
    WorkspaceParameterUnit as Unit,
};

fn representation(
    value: safemlx::Dtype,
    output: Option<safemlx::HostTransferArrayLayout>,
) -> Option<eredu_nn::workspace::WorkspaceRepresentation> {
    use eredu_nn::workspace::{WorkspaceFloatingType as F, WorkspaceRepresentation};
    let dtype = match value {
        safemlx::Dtype::Float32 => F::Float32,
        safemlx::Dtype::Float16 => F::Float16,
        safemlx::Dtype::Bfloat16 => F::Bfloat16,
        _ => return None,
    };
    // Only the retained canonical Host-transfer producer supplies this fact.
    // A logical shape or dtype alone still cannot establish contiguity.
    Some(WorkspaceRepresentation::new(
        dtype,
        output.is_some_and(safemlx::HostTransferArrayLayout::row_contiguous),
    ))
}

fn dtype(value: safemlx::Dtype, unit: usize, row: usize) -> Result<WorkspaceDtype, Failure> {
    match value {
        safemlx::Dtype::Float32 | safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16 => {
            Ok(WorkspaceDtype::Float32)
        }
        safemlx::Dtype::Int32 => Ok(WorkspaceDtype::Int32),
        safemlx::Dtype::Uint32 => Ok(WorkspaceDtype::Uint32),
        safemlx::Dtype::Uint8 => Ok(WorkspaceDtype::Uint8),
        safemlx::Dtype::Bool => Ok(WorkspaceDtype::Bool),
        _ => Err(Failure::UnsupportedDtype { unit, row }),
    }
}
impl LayerwiseWorkspace {
    /// Same retained source manager; descriptive matching creates no operation grant.
    pub(crate) fn parameter_source_matches_manager(&self, manager: &crate::backend::runtime::residency::manager::ResidencyManager) -> bool {
        manager.same_source_manager(&self.manager)
    }

    /// The loan borrows this existing owner, including its selected identity,
    /// immutable layout, snapshots and receipt. It creates no temporary provider.
    pub(crate) fn parameter_source(&self) -> WorkspaceParameterSourceLoan<'_> {
        WorkspaceParameterSourceLoan::new(self)
    }
}

impl Rows for LayerwiseWorkspace {
    fn layout(&self) -> &ExecutionUnitLayout {
        LayerwiseWorkspace::layout(self)
    }
    fn execution_address(&self, ordinal: usize) -> Option<ExecutionUnitAddress> {
        LayerwiseWorkspace::execution_address(self, ordinal)
    }
    fn unit_count(&self) -> usize {
        match &self.copies {
            LayerwiseCopies::Host(copies) => copies.units().len(),
            LayerwiseCopies::Foreground(identity) => identity.unit_count(),
            LayerwiseCopies::Disk { copies, .. } => copies.units().len(),
        }
    }
    fn unit(&self, index: usize) -> Result<Unit<'_>, Failure> {
        match &self.copies {
            LayerwiseCopies::Host(copies) => {
                let unit = copies.units().get(index).ok_or(Failure::MissingIndex)?;
                Ok(Unit {
                    id: unit.id(),
                    rows: copies.copies(unit).len(),
                    fresh_capacity_bytes: unit.fresh_capacity_bytes(),
                })
            }
            LayerwiseCopies::Foreground(identity) => {
                let unit = identity.unit(index).ok_or(Failure::MissingIndex)?;
                Ok(Unit {
                    id: unit.id(),
                    rows: unit.bindings().len(),
                    fresh_capacity_bytes: identity
                        .unit_capacity(index)
                        .ok_or(Failure::MissingIndex)?,
                })
            }
            LayerwiseCopies::Disk { copies, .. } => {
                let unit = copies.units().get(index).ok_or(Failure::MissingIndex)?;
                Ok(Unit {
                    id: unit.id(),
                    rows: unit.bindings().len(),
                    fresh_capacity_bytes: unit.fresh_capacity_bytes(),
                })
            }
        }
    }
    fn row(&self, unit: usize, row: usize) -> Result<Row<'_>, Failure> {
        match &self.copies {
            LayerwiseCopies::Host(copies) => {
                let copy = copies
                    .units()
                    .get(unit)
                    .and_then(|u| copies.copies(u).get(row))
                    .ok_or(Failure::MissingIndex)?;
                Ok(Row {
                    binding: copy.binding(),
                    shape: copy.shape(),
                    dtype: dtype(copy.dtype(), unit, row)?,
                    representation: representation(copy.dtype(), Some(copy.copy_output_layout())),
                    physical_bytes: copy.logical_bytes(),
                    capacity_bytes: copy.output_capacity_bytes(),
                    // Every actual dispatch can independently allocate, even
                    // when two names currently borrow the same host allocation.
                    owner: Owner {
                        unit,
                        row,
                        lifetime: Lifetime::Invocation,
                    },
                })
            }
            LayerwiseCopies::Foreground(identity) => {
                let (binding, shape, native, output_layout, physical_bytes, capacity_bytes, owner) =
                    identity.row(unit, row).ok_or(Failure::MissingIndex)?;
                Ok(Row {
                    binding,
                    shape,
                    dtype: dtype(native, unit, row)?,
                    representation: representation(native, Some(output_layout)),
                    physical_bytes,
                    capacity_bytes,
                    owner,
                })
            }
            LayerwiseCopies::Disk { copies, receipt } => {
                let copy = copies
                    .units()
                    .get(unit)
                    .and_then(|u| u.bindings().get(row))
                    .ok_or(Failure::MissingIndex)?;
                let owner = copy.owner();
                let owner_unit = copies
                    .units()
                    .iter()
                    .position(|u| u.id() == owner.unit())
                    .ok_or(Failure::MissingIndex)?;
                let owner_row = copies.units()[owner_unit]
                    .bindings()
                    .iter()
                    .position(|r| r.name() == owner.name())
                    .ok_or(Failure::MissingIndex)?;
                let lifetime = if receipt.0.parameter_owner_is_persistent(owner.unit()) {
                    Lifetime::Trace
                } else {
                    Lifetime::Invocation
                };
                Ok(Row {
                    binding: copy.binding(),
                    shape: copy.shape(),
                    dtype: dtype(copy.dtype(), unit, row)?,
                    representation: representation(copy.dtype(), None),
                    physical_bytes: copy.logical_bytes(),
                    capacity_bytes: copy.output_capacity_bytes(),
                    owner: Owner {
                        unit: owner_unit,
                        row: owner_row,
                        lifetime,
                    },
                })
            }
        }
    }
    fn requested_unit(&self, ordinal: usize) -> Result<usize, Failure> {
        self.layout()
            .address(ordinal)
            .ok_or(Failure::MissingIndex)?;
        match &self.copies {
            LayerwiseCopies::Host(copies) => copies
                .units()
                .get(ordinal)
                .map(|_| ordinal)
                .ok_or(Failure::MissingIndex),
            LayerwiseCopies::Foreground(identity) => identity
                .requested_unit(ordinal)
                .ok_or(Failure::MissingIndex),
            LayerwiseCopies::Disk { copies, .. } => {
                let id = copies
                    .requested_units()
                    .get(ordinal)
                    .ok_or(Failure::MissingIndex)?;
                copies
                    .units()
                    .iter()
                    .position(|u| u.id() == id)
                    .ok_or(Failure::MissingIndex)
            }
        }
    }
    fn window_len(&self, ordinal: usize) -> Result<usize, Failure> {
        match &self.copies {
            LayerwiseCopies::Host(_) | LayerwiseCopies::Foreground(_) => self
                .layout()
                .window_range(
                    ordinal,
                    std::num::NonZeroUsize::new(self.identity.depth())
                        .ok_or(Failure::SourceMismatch)?,
                )
                .map(|range| range.len())
                .ok_or(Failure::MissingIndex),
            LayerwiseCopies::Disk { receipt, .. } => receipt
                .0
                .parameter_window(ordinal)
                .map(<[_]>::len)
                .ok_or(Failure::MissingIndex),
        }
    }
    fn window_unit(&self, ordinal: usize, member: usize) -> Result<usize, Failure> {
        match &self.copies {
            LayerwiseCopies::Host(_) | LayerwiseCopies::Foreground(_) => {
                let range = self
                    .layout()
                    .window_range(
                        ordinal,
                        std::num::NonZeroUsize::new(self.identity.depth())
                            .ok_or(Failure::SourceMismatch)?,
                    )
                    .ok_or(Failure::MissingIndex)?;
                if member >= range.len() {
                    return Err(Failure::MissingIndex);
                }
                self.requested_unit(range.start + member)
            }
            LayerwiseCopies::Disk { copies, receipt } => {
                let id = receipt
                    .0
                    .parameter_window(ordinal)
                    .and_then(|w| w.get(member))
                    .ok_or(Failure::MissingIndex)?;
                copies
                    .units()
                    .iter()
                    .position(|u| u.id() == id)
                    .ok_or(Failure::MissingIndex)
            }
        }
    }
}

#[cfg(test)]
mod tests;
