//! Independent empty-state construction from an exact canonical source.
use super::*;
use eredu_core::HostPreparationAuthority;
use std::collections::TryReserveError;

/// Fixed geometry refusal or actual table allocation failure. The enclosing
/// admitted caller retains host custody through this error's transport.
#[derive(Debug, thiserror::Error)]
pub enum ResidentEmptyStateError {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error("could not allocate the prepared empty resident table: {0}")]
    Allocation(#[from] TryReserveError),
}

/// One borrowed empty source and its exact outer/child table construction.
/// This proves no account, native completion, byte grant or populated-state copy.
#[must_use]
pub struct PreparedResidentEmptyState<'a, S: ResidentTableResetState> {
    source: ResidentResetSource<'a, S>,
    bytes: u64,
}
impl<'a, S: ResidentTableResetState> PreparedResidentEmptyState<'a, S> {
    /// Performs the same selected geometry/placement checks as resident reset,
    /// then requires the actual source to equal its empty inline state worker.
    /// Populated canonical caches are never discarded by this constructor.
    pub fn inspect(source: ResidentResetSource<'a, S>) -> Result<Self, WorkingMemoryError> {
        let mut child_bytes = 0u64;
        validate_source_geometry(
            &source,
            |cause| cause,
            |child| {
                child_bytes = child_bytes
                    .checked_add(table_bytes(child)?)
                    .ok_or(WorkingMemoryError::Overflow)?;
                Ok(())
            },
        )?;
        if source.state.resident_reset_plan()?.1 != 0 || !source.state.resident_fork_is_empty() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let controls = [
            source_geometry_control_bytes::<S, WorkingMemoryError>(size_of::<&mut u64>())
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<ResidentResetSource<'_, S>>(),
            size_of::<S>(),
            size_of::<Result<S, ResidentEmptyStateError>>(),
            size_of::<ResidentEmptyStateError>(),
            size_of::<WorkingMemoryError>(),
            size_of::<TryReserveError>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<SharedStateLayout>(),
            size_of::<S::ResetContext>(),
            size_of::<&mut S::ResetContext>(),
            size_of::<Option<HostSlotTable<S::Child>>>(),
            6 * size_of::<usize>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let bytes = table_bytes(source.state.resident_reset_layers())?
            .checked_add(child_bytes)
            .and_then(|n| n.checked_add(controls))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { source, bytes })
    }
    /// Exact typed host construction envelope. Shared source layout/storage and
    /// the caller's custody/erased result/error transports are independently held.
    pub fn required_bytes(&self) -> u64 {
        self.bytes
    }

    /// Consumes the inspected source after its exact host account was accepted.
    /// Every finished table retains that same host authority, including aliases.
    /// The caller keeps authority across this whole call, error transport and
    /// any enclosing result allocation. No native array or reset is performed.
    pub fn construct(self, host: &HostPreparationAuthority) -> Result<S, ResidentEmptyStateError> {
        let source = self.source.state;
        // State has interior-mutability implementations: recheck the exact
        // empty predicate before the first allocation, never infer from offset.
        if !source.resident_fork_is_empty() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let layout = source.resident_reset_layout();
        let plan = source
            .resident_reset_layers()
            .prepare_copy_slots()
            .and_then(|p| p.for_dense_destination::<S::Layer>())
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let (_, mut builder) = plan.try_initialize_retaining_source()?;
        for (layer, policy) in source
            .resident_reset_layers()
            .slots()
            .iter()
            .zip(layout.layout().layers().iter())
        {
            let child = S::resident_reset_child(layer)
                .map(|child| {
                    let plan = child
                        .prepare_copy_slots()
                        .and_then(|p| p.for_dense_destination::<S::Child>())
                        .map_err(|_| WorkingMemoryError::Overflow)?;
                    let (_, mut builder) = plan.try_initialize_retaining_source()?;
                    for value in child.slots() {
                        assert!(
                            builder.push(S::empty_resident_reset_child(value)).is_ok(),
                            "exact child fill"
                        );
                    }
                    finish(builder, host)
                })
                .transpose()?;
            assert!(
                builder
                    .push(S::empty_resident_reset_layer_with_child(policy, child))
                    .is_ok(),
                "exact empty layer fill"
            );
        }
        let layers = finish(builder, host)?;
        Ok(S::from_resident_reset(
            &mut S::ResetContext::default(),
            layout.clone(),
            source.resident_reset_global_start(),
            layers,
        ))
    }
}
fn table_bytes<T>(source: &HostSlotTable<T>) -> Result<u64, WorkingMemoryError> {
    let plan = source
        .prepare_copy_slots()
        .and_then(|p| p.for_dense_destination::<T>())
        .map_err(|_| WorkingMemoryError::Overflow)?;
    let controls = DenseHostSlotInitialization::<T>::preparation_control_bytes()
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::UnknownBound)?;
    plan.initialization_peak_bytes()
        .checked_add(controls)
        .ok_or(WorkingMemoryError::Overflow)
}
fn finish<T>(
    builder: crate::DenseHostSlotInitializationBuilder<T>,
    host: &HostPreparationAuthority,
) -> Result<HostSlotTable<T>, ResidentEmptyStateError> {
    let identity = crate::HostMetadataIdentity::prepared_host(host)?;
    Ok(builder
        .finish_with_preparation(Some((identity, host)))
        .ok()
        .expect("fixed complete empty table")
        .into_table())
}
