//! Finite source binding and table metadata under the existing preparation grant.
use super::*;
use crate::working_memory::{qualified_storage, ExistingStoragePinLayout, OriginalStorageSourcesLayout};
use eredu_core::HostPreparationAuthority;
use std::{alloc::Layout, mem::size_of};

/// Fixed, allocation-free refusal from the table preparation layout query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DecoderHostPreparationError {
    /// The pinned host allocation producer is not qualified on this target.
    #[error("decoder host preparation layout is unqualified")]
    UnknownBound,
    /// A counted layout or sum cannot be represented.
    #[error("decoder host preparation layout overflow")]
    Overflow,
}

pub(super) fn memory<T>(
    value: Result<T, WorkingMemoryError>,
) -> Result<T, DecoderHostPreparationError> {
    value.map_err(|e| match e {
        WorkingMemoryError::Overflow => DecoderHostPreparationError::Overflow,
        _ => DecoderHostPreparationError::UnknownBound,
    })
}
pub(super) fn add(
    values: impl IntoIterator<Item = usize>,
) -> Result<usize, DecoderHostPreparationError> {
    values
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(DecoderHostPreparationError::Overflow)
}
pub(super) fn vector_bytes<T>(count: usize) -> Result<usize, DecoderHostPreparationError> {
    add([
        Layout::array::<T>(count)
            .map_err(|_| DecoderHostPreparationError::Overflow)?
            .size(),
        usize::try_from(memory(qualified_storage::vector_control_bytes::<T>())?)
            .map_err(|_| DecoderHostPreparationError::Overflow)?,
    ])
}

impl<'a, S, K: HostSlotStorageKey> RegisteredDecoderHostCopy<'a, S, K> {
    /// Existing-source binding under the enclosing finite host preparation.
    /// Original tables use their authenticated source carrier; a rejected original
    /// source is never retried as an ordinary registration. No payload is charged.
    pub fn bind_prepared(
        pool: &WorkingMemoryPool,
        plan: HostSlotInitialization<'a, S>,
        key: K,
        authority: &HostPreparationAuthority,
    ) -> Result<Self, DecoderCopyAdmissionError> {
        if key.host_slot_identity() != Some(plan.source_metadata().identity().registry_key()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        // Classification authenticates the actual original source before the first
        // carrier/vector allocation. Both branches retain the same preparation.
        let class = pool.classify_host_slot_source(plan.source_metadata())?;
        let ordinary = class.registered().is_some();
        let capacity = plan
            .source_metadata()
            .capacity_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let mut carrier = OriginalStorageSourcesLayout::new(usize::from(!ordinary))
            .ok_or(WorkingMemoryError::Overflow)?
            .construct(pool, authority)?;
        let mut inputs = qualified_storage::vector(usize::from(ordinary), true)?;
        if ordinary {
            inputs.push((key, capacity));
        } else {
            carrier.retain_table(plan.source_metadata())?;
        }
        let source = ExistingStoragePinLayout::<K>::new(inputs.len())?
            .construct(pool, inputs)?
            .with_retained_original_sources(&mut carrier)?;
        Ok(Self {
            plan,
            source: DecoderSource::Registered(source),
            destination_identity: None,
        })
    }

    /// Shared ordinary/prepared adapter; only a supplied accepted preparation
    /// selects the finite source constructor. No failed branch is retried.
    pub fn bind_with_preparation(
        pool: &WorkingMemoryPool,
        plan: HostSlotInitialization<'a, S>,
        key: K,
        authority: Option<&HostPreparationAuthority>,
    ) -> Result<Self, DecoderCopyAdmissionError> {
        match authority {
            Some(authority) => Self::bind_prepared(pool, plan, key, authority),
            None => Self::bind(pool, plan, key),
        }
    }
}

impl<S, D, K: HostSlotStorageKey> RegisteredDecoderHostCopy<'_, S, K, D> {
    pub(super) fn binding_control_bytes(
        registered: bool,
    ) -> Result<usize, DecoderHostPreparationError> {
        if registered {
            add([
                memory(ExistingStoragePinLayout::<K>::new(1))?.requested_bytes(),
                OriginalStorageSourcesLayout::new(1)
                    .ok_or(DecoderHostPreparationError::Overflow)?
                    .requested_bytes(),
                vector_bytes::<(K, u64)>(1)?,
                size_of::<Result<Self, DecoderCopyAdmissionError>>(),
                size_of::<DecoderCopyAdmissionError>(),
                size_of::<crate::working_memory::HostSlotSource>(),
                size_of::<Result<crate::working_memory::HostSlotSource, WorkingMemoryError>>(),
            ])
        } else {
            Ok(0)
        }
    }
    /// Finite table binding and initialization controls, excluding the Option<D>
    /// payload already covered by initialization_peak_bytes. `registered` is the
    /// actual Registered/Funded source branch of the enclosing closed worker.
    /// One registered table can select either the one-row existing pin or one
    /// original source carrier; their sum bounds both without assuming provenance.
    pub fn preparation_control_bytes(
        registered: bool,
    ) -> Result<usize, DecoderHostPreparationError> {
        let metadata = HostSlotInitialization::<S, D>::preparation_control_bytes()
            .ok_or(DecoderHostPreparationError::UnknownBound)?;
        let binding = Self::binding_control_bytes(registered)?;
        add([
            metadata,
            binding,
            size_of::<Self>(),
            size_of::<InitializedDecoderSlots<D>>(),
            size_of::<Result<InitializedDecoderSlots<D>, WorkingMemoryError>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Option<HostPreparationAuthority>>(),
            size_of::<Result<HostMetadataIdentity, WorkingMemoryError>>(),
            size_of::<Result<Option<HostMetadataIdentity>, WorkingMemoryError>>(),
        ])
    }
}

impl<OS, OD, CS, CD, K: HostSlotStorageKey> RegisteredDecoderTableGroup<'_, OS, OD, CS, CD, K> {
    /// Exact group bookkeeping before/during destination admission. Per-table
    /// binding/metadata and slot payloads are queried separately; this measures
    /// the source plan children, initialized children, progress and scope arrays.
    pub fn preparation_control_bytes(
        children: usize,
    ) -> Result<usize, DecoderHostPreparationError> {
        let tables = children
            .checked_add(1)
            .ok_or(DecoderHostPreparationError::Overflow)?;
        add([
            vector_bytes::<RegisteredDecoderHostCopy<'_, CS, K, CD>>(children)?,
            vector_bytes::<Option<InitializedDecoderSlots<CD>>>(children)?,
            vector_bytes::<std::sync::atomic::AtomicBool>(children)?,
            vector_bytes::<u64>(tables)?,
            vector_bytes::<DecoderCopySource<'_, K>>(tables)?,
            vector_bytes::<WorkingMemoryDecoderHostScope>(tables)?,
            usize::try_from(memory(qualified_storage::shared_bytes::<
                group::GroupProgress,
            >())?)
            .map_err(|_| DecoderHostPreparationError::Overflow)?,
            size_of::<group::GroupProgress>(),
            size_of::<group::GroupProgressOwner>(),
            size_of::<Option<group::GroupProgress>>(),
            size_of::<Result<group::GroupProgressOwner, WorkingMemoryError>>(),
            size_of::<Self>(),
            size_of::<InitializedDecoderTableGroup<OD, CD>>(),
            size_of::<std::vec::IntoIter<WorkingMemoryDecoderHostScope>>(),
            size_of::<std::vec::IntoIter<RegisteredDecoderHostCopy<'_, CS, K, CD>>>(),
            size_of::<Result<Self, DecoderCopyAdmissionError>>(),
        ])
    }
}

impl<S, D, K: Ord + Send + 'static> RegisteredDecoderHostCopy<'_, S, K, D> {
    pub(super) fn prepare_destination_identity(
        &mut self,
        preparation: Option<&HostPreparationAuthority>,
    ) -> Result<(), WorkingMemoryError> {
        if self.destination_identity.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // All fallible identity work precedes the copy-account commit. Exhaustion
        // therefore retires only preparation/source owners, with no native scope.
        self.destination_identity = preparation
            .map(HostMetadataIdentity::prepared_host)
            .transpose()?;
        Ok(())
    }
}
