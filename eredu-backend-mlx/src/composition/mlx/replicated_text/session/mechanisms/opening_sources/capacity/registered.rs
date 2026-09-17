//! One-use existing-only pins for an inspected snapshot, never current-state evidence.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::{
    inspection::PrefillChunkRetentionContext,
    working_memory::{
        BoundedPinAttempt, BoundedPinError, BoundedRegisteredStorage, CaptureSourceSegment,
        FailedBoundedPinAttempt, InferenceRequest, OriginalPrefillStoragePinSlots,
        OriginalTextControlGuard, WorkingMemoryFundingScope,
    },
    SharedLayeredObservationPaths,
};

pub(super) type Key = StorageIdentity;

pub(in super::super) fn descriptor_count(n: OpeningSlots) -> Result<usize, OpeningError> {
    [n.arrays, n.hosts, n.bytes, n.sources, n.layouts, n.tables]
        .into_iter()
        .try_fold(0usize, |sum, n| {
            sum.checked_add(n).ok_or(OpeningError::Overflow)
        })
}
pub(super) fn wrapper_control_bytes() -> usize {
    // The caller includes construction/move overlap. The typed original S plan
    // separately measures each finite attempt's allocated keys/ordinals/core.
    size_of::<PreparedOpeningPins>()
        + size_of::<OpeningPinFailure>()
        + size_of::<PinCause>()
        + size_of::<eredu_runtime::working_memory::HostSlotSource>()
        + size_of::<
            Result<
                eredu_runtime::working_memory::HostSlotSource,
                eredu_runtime::working_memory::WorkingMemoryError,
            >,
        >()
}

/// Does not carry the session's non-Clone prepared runtime token. That token was
/// validated while collecting. This retained source alias cannot revalidate a
/// subsequently mutated runtime or authenticate a later current inventory.
/// Payloads and keys precede original controls on every teardown path.
pub(in crate::composition::mlx::replicated_text) struct PreparedOpeningPins {
    owners: FixedOpeningOwners,
    attempt: Option<BoundedPinAttempt<Key>>,
    rejected_key: Option<Key>,
    registered: Option<BoundedRegisteredStorage<Key>>,
    bank: OriginalPrefillStoragePinSlots<Key>,
    paths: SharedLayeredObservationPaths,
    controls: PreparedTextControlWorkspace,
    request: InferenceRequest,
    attempted: bool,
    original_bytes: u64,
    custody: OriginalTextControlGuard,
}

pub(super) struct OpeningPinPreparationFailure<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> {
    pub(super) cause: OpeningError,
    pub(super) inventory: PreparedOpeningInventory<'a, S, E>,
}
impl<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> PreparedOpeningInventory<'a, S, E> {
    /// Consume this complete capsule into its one original typed pin bank. All
    /// lexical source loans end here; only inspected owners are retained. No
    /// current-pair claim, native attachment, publication or activation follows.
    pub(super) fn prepare_registered_pins(
        self,
        accepted: &mut OwnedTextSpanWorkspace,
    ) -> Result<PreparedOpeningPins, OpeningPinPreparationFailure<'a, S, E>> {
        let prepare = || {
            if !self.plan.opening_pins {
                return Err(OpeningError::Identity);
            }
            self.owners.visit(&mut |_| Ok::<_, OpeningError>(()))?;
            let actual = accepted
                .workspace()
                .text_controls()
                .ok_or(OpeningError::Identity)?;
            if !self.plan.controls.same_binding(actual) {
                return Err(OpeningError::Identity);
            }
            self.custody.validate_reservation(accepted.reservation())?;
            self.plan
                .plan
                .execution
                .validate_paths(self.plan.plan.paths)?;
            Ok(())
        };
        if let Err(cause) = prepare() {
            return Err(OpeningPinPreparationFailure {
                cause,
                inventory: self,
            });
        }
        let bank = match accepted.take_prefill_storage_pins::<Key>() {
            Ok(bank) => bank,
            Err(error) => {
                return Err(OpeningPinPreparationFailure {
                    cause: error.into(),
                    inventory: self,
                });
            }
        };
        // From bank extraction through handoff: only existing Arc clones/moves.
        let Self {
            owners,
            plan,
            reservation,
            custody,
        } = self;
        let paths = plan.plan.paths.source().clone();
        let request = InferenceRequest::from(&reservation);
        Ok(PreparedOpeningPins {
            owners,
            attempt: None,
            rejected_key: None,
            registered: None,
            bank,
            paths,
            controls: plan.controls,
            request,
            attempted: false,
            original_bytes: 0,
            custody,
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum PinCause {
    #[error(transparent)]
    Opening(#[from] OpeningError),
    #[error(transparent)]
    Accounting(#[from] BoundedPinError),
    #[error(transparent)]
    Commit(#[from] FailedBoundedPinAttempt<Key>),
}
/// The snapshot stays with its caller, including unwinding. An escaped error
/// retains original control custody; a failed commit also owns all staged keys.
#[derive(Debug)]
pub(in crate::composition::mlx::replicated_text) struct OpeningPinFailure {
    cause: PinCause,
    custody: OriginalTextControlGuard,
}
impl std::fmt::Display for OpeningPinFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("inspected native opening could not pin existing storage")
    }
}
impl std::error::Error for OpeningPinFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl PreparedOpeningPins {
    /// This one-use operation pins the captured snapshot. The first canonical
    /// chunk proves original request/scope association, not snapshot freshness.
    /// Missing keys fail; there is no adoption, attachment, or fallback path.
    pub(in crate::composition::mlx::replicated_text) fn pin_registered(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<(), OpeningPinFailure> {
        self.pin_inner(context, native, segment)
            .map_err(|cause| OpeningPinFailure {
                cause,
                custody: self.custody.clone(),
            })
    }
    fn pin_inner(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
        native: &WorkingMemoryFundingScope,
        segment: &CaptureSourceSegment,
    ) -> Result<(), PinCause> {
        if std::mem::replace(&mut self.attempted, true) {
            return Err(OpeningError::Used.into());
        }
        self.request
            .validate_same_request(context.request())
            .map_err(OpeningError::from)?;
        if context.geometry() != self.controls.geometry() || context.chunk().input.start != 0 {
            return Err(OpeningError::Identity.into());
        }
        self.custody
            .validate_native_scope(native)
            .map_err(OpeningError::from)?;
        // Install the issued attempt before deriving any keys. All owner roots
        // are external to this fallible call and remain installed on unwind.
        self.attempt = Some(self.bank.begin(context, native, segment)?);
        let attempt = self.attempt.as_mut().expect("issued original pin attempt");
        let rejected = &mut self.rejected_key;
        let mut ordinal = 0usize;
        let mut original_bytes = 0u64;
        self.owners.visit(&mut |entry| {
            let table = match &entry {
                OpeningEntry::Table(owner, bytes) => Some((*owner, *bytes)),
                _ => None,
            };
            if let Some((key, bytes)) = storage_entry_in(native.pool(), entry)? {
                if let Err(key) = attempt.push_owned(key, bytes) {
                    *rejected = Some(key);
                    return Err(OpeningError::Capacity);
                }
            } else if let Some((table, bytes)) = table {
                // The exact table token already lives in fixed admitted slots.
                // Deduplicate their actual identities without another container.
                let mut seen = false;
                for previous in 0..ordinal {
                    if let OpeningEntry::Table(owner, _) = self.owners.entry_at(previous)? {
                        seen |= owner.same_storage(table);
                    }
                }
                if !seen {
                    original_bytes = original_bytes
                        .checked_add(bytes)
                        .ok_or(OpeningError::Overflow)?;
                }
            }
            ordinal += 1;
            Ok(())
        })?;
        let attempt = self.attempt.take().expect("installed original pin attempt");
        self.registered = Some(attempt.pin_registered(native, segment)?);
        self.original_bytes = original_bytes;
        Ok(())
    }
    /// Unique existing physical bytes, not new capacity or current completeness.
    pub(in crate::composition::mlx::replicated_text) fn registered_bytes(&self) -> Option<u64> {
        self.registered
            .as_ref()
            .and_then(|registered| registered.bytes().checked_add(self.original_bytes))
    }
}

// FixedOpeningOwners retains the actual metadata token through every attempted
// prefix. Classification adds only prepriced inline result/borrow controls.
fn storage_entry_in(
    pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    entry: OpeningEntry<'_>,
) -> Result<Option<(Key, u64)>, OpeningError> {
    if let OpeningEntry::Table(owner, bytes) = entry {
        let source = pool.classify_host_slot_source(owner)?;
        return match source.registered() {
            Some((key, actual)) if actual == bytes => {
                Ok(Some((Key::HostMetadata(key.clone()), bytes)))
            }
            Some(_) => Err(OpeningError::Identity),
            None if owner.capacity_bytes() == Some(bytes) => Ok(None),
            None => Err(OpeningError::Identity),
        };
    }
    storage_entry(entry)
}

pub(in super::super) fn storage_entry(
    entry: OpeningEntry<'_>,
) -> Result<Option<(Key, u64)>, OpeningError> {
    let pair = match entry {
        OpeningEntry::Array(_, info) | OpeningEntry::Host(_, info) => {
            // Match the canonical native inventory: allocation-free empty values
            // retain their handle/control here but have no backing registry key.
            if info.bytes() == 0 {
                return Ok(None);
            }
            (
                Key::Native(info.identity()),
                u64::try_from(info.bytes()).map_err(|_| OpeningError::Overflow)?,
            )
        }
        OpeningEntry::Bytes(owner, bytes) => (Key::from_bytes(owner), bytes),
        // Real zero-byte source/metadata origins still participate in validation.
        OpeningEntry::Source(source) => (Key::Source(source.identity()), source.bytes()),
        OpeningEntry::Layout(owner, bytes) => (
            Key::HostMetadata(owner.identity().registry_key().clone()),
            bytes,
        ),
        OpeningEntry::Table(owner, bytes) => (
            Key::HostMetadata(owner.identity().registry_key().clone()),
            bytes,
        ),
    };
    Ok(Some(pair))
}

#[cfg(test)]
pub(super) mod tests;
