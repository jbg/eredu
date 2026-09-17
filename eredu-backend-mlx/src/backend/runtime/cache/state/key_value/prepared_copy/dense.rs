//! Closed resident KV construction into the actual installable dense table.

use super::*;
use crate::backend::error::Error;
use eredu_runtime::{
    DenseHostSlotInitialization, HostSlotAttachmentError, HostSlotMetadata,
    working_memory::{
        FundedDenseHostSlots, InferencePromptCompletion, InferenceRetention,
        InitializedDenseDecoderSlots, RegisteredDenseDecoderInitialization,
    },
};
use std::fmt;

type DenseSlots<'a> =
    FundedDenseHostSlots<'a, MlxKeyValueLayerState, MlxKeyValueLayerState, StorageIdentity>;

impl<'a> PreparedResidentKvCopy<'a> {
    /// Inspect the actual source's dense destination geometry without cloning a
    /// table/array, allocating a destination or acquiring construction authority.
    /// Nested native copies and shared layout custody remain separate.
    pub(crate) fn dense_initialization(
        &self,
    ) -> Result<
        DenseHostSlotInitialization<'a, MlxKeyValueLayerState, MlxKeyValueLayerState>,
        ResidentKvCopyError,
    > {
        Ok(self.slot_initialization()?.for_dense_destination()?)
    }

    /// Bind the actual registered live table or funded saved table to the same
    /// dense program. A fresh prompt stage must consume the result before fill.
    /// The caller supplies complete source pins and separately priced native
    /// work; this method does not establish their semantic completeness.
    pub(crate) fn dense_initialization_peak_bytes_fixed(
        &self,
    ) -> Result<u64, crate::backend::runtime::cache::state::ResidentDecoderPreparationError> {
        Ok(self
            .slot_initialization_fixed()?
            .for_dense_destination::<MlxKeyValueLayerState>()?
            .initialization_peak_bytes())
    }

    pub(crate) fn dense_host_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        let nested =
            eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes().ok_or(E::UnknownBound)?;
        RegisteredDenseDecoderInitialization::<
            MlxKeyValueLayerState,
            MlxKeyValueLayerState,
            StorageIdentity,
        >::preparation_control_bytes(matches!(self.source, KvCopySource::Live(_)), nested)
    }

    pub(crate) fn dense_host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<
        RegisteredDenseDecoderInitialization<
            'a,
            MlxKeyValueLayerState,
            MlxKeyValueLayerState,
            StorageIdentity,
        >,
        ResidentKvCopyError,
    > {
        self.dense_host_copy_with_preparation(pool, None)
    }

    pub(crate) fn dense_host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<
        RegisteredDenseDecoderInitialization<
            'a,
            MlxKeyValueLayerState,
            MlxKeyValueLayerState,
            StorageIdentity,
        >,
        ResidentKvCopyError,
    > {
        Ok(self
            .host_copy_with_preparation(pool, preparation)?
            .for_dense_destination()?)
    }

    /// Same isolated leaf copies, with an independently funded host destination.
    /// Numerical settlement and publication stay with the enclosing copy owner.
    pub(crate) fn copy_dense_with_preparation(
        self,
        host: &eredu_core::HostPreparationAuthority,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<MlxKeyValueState, Error> {
        let source = self
            .slot_initialization_fixed()
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::UnknownBound))?
            .for_dense_destination::<MlxKeyValueLayerState>()
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let layers = super::super::super::resident_copy::host_copy::copy_slots(
            source,
            host,
            funding,
            |_, layer| {
                super::copy_resident_kv_layer_retained(layer, stream, roots)
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))
            },
        )?;
        Ok(MlxKeyValueState {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
            paged_transaction_branch: false,
            inference_retention: InferenceRetention::new(),
        })
    }

    /// Fill the one admitted dense buffer with the existing isolated layer
    /// program. Exact source identity/geometry and an empty destination are
    /// checked before any native work. Every array intermediate/output is kept
    /// in the caller's recovery collector before later fallible work.
    ///
    /// This performs no native settlement or array publication. The caller must
    /// retain the settled source independently and keep the collector plus the
    /// separate native scope through success, failure and recovery. The closed
    /// host bound excludes that collector and nested numerical allocations.
    pub(crate) fn copy_dense_retained(
        self,
        mut destination: InitializedDenseDecoderSlots<
            'a,
            MlxKeyValueLayerState,
            MlxKeyValueLayerState,
            StorageIdentity,
        >,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<PreparedDenseResidentKvState<'a>, ResidentKvCopyError> {
        destination.validate_source(&self.dense_initialization()?)?;
        for index in 0..self.len() {
            let copied = self.copy_layer_retained(index, stream, roots)?;
            destination
                .push(copied)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
            #[cfg(test)]
            super::tests::after_layer(index)?;
        }
        let layers = destination
            .finish()
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        Ok(PreparedDenseResidentKvState {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
        })
    }
}

/// Complete dense cells with the original fresh prompt's protected host hold.
/// No raw table, mutable layer, inference retention or completion authority is
/// exposed. A failed publication keeps this whole owner available for recovery.
/// The same actual buffer becomes the live state only after exact host charge
/// attachment; native completion/publication remain the enclosing caller's duty.
pub(crate) struct PreparedDenseResidentKvState<'a> {
    layers: DenseSlots<'a>,
    layout: SharedStateLayout,
    global_layer_start: usize,
}

impl<'a> PreparedDenseResidentKvState<'a> {
    // Only the source-qualified paged worker can supply this completed table.
    // Numerical completion and publication remain the enclosing operation's.
    pub(super) fn from_paged_parts(
        layers: DenseSlots<'a>,
        layout: SharedStateLayout,
        global_layer_start: usize,
    ) -> Self {
        Self {
            layers,
            layout,
            global_layer_start,
        }
    }
}

impl PreparedDenseResidentKvState<'_> {
    pub(crate) fn visit_paged_operands(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), crate::backend::runtime::cache::state::SnapshotProjectionCause> {
        use crate::backend::runtime::cache::state::SnapshotProjectionCause;
        for index in 0..self.layers.len() {
            match self.layers.get(index) {
                Some(MlxKeyValueLayerState::Paged(cache)) => {
                    cache.visit_snapshot_arrays(&mut |array| {
                        visitor(array);
                        Ok(())
                    })?
                }
                Some(MlxKeyValueLayerState::Stateless) => {}
                _ => return Err(SnapshotProjectionCause::Changed),
            }
        }
        Ok(())
    }

    pub(crate) fn shared_layout(&self) -> &SharedStateLayout {
        &self.layout
    }

    pub(crate) fn global_layer_start(&self) -> usize {
        self.global_layer_start
    }

    pub(crate) fn slot_metadata(&self) -> &HostSlotMetadata {
        self.layers.metadata()
    }

    pub(crate) fn retained_slot_bytes(&self) -> u64 {
        self.layers.retained_bytes()
    }

    pub(crate) fn protected_slot_bytes(&self) -> u64 {
        self.layers.protected_bytes()
    }

    /// Final destination array roots only, in the same layer/key/value order.
    /// The separate recovery collector also contains copy intermediates.
    pub(crate) fn visit_operands<'s>(&'s self, visitor: &mut dyn FnMut(&'s Array)) {
        for index in 0..self.layers.len() {
            if let Some(MlxKeyValueLayerState::Device(cache)) = self.layers.get(index) {
                cache.prepare_isolated_copy().visit_operands(visitor);
            }
        }
    }
}

/// Exact dense table after its protected host hold has been transferred to the
/// table's registered charge. Construction is confined to `publish_for_control`.
/// The contained state starts with fresh inference retention. This owner proves
/// neither numerical completion/publication nor source provenance or admission.
/// Keep native recovery outside it until all copied arrays have safely settled.
#[derive(Debug)]
pub(crate) struct PublishedDenseResidentKvState {
    state: MlxKeyValueState,
}

impl PublishedDenseResidentKvState {
    /// Moves the already published table without copying it. Extraction adds no
    /// completion or execution authority; only the typed KV adapter uses it.
    pub(crate) fn into_state(self) -> MlxKeyValueState {
        self.state
    }
}

impl<'a> PreparedDenseResidentKvState<'a> {
    /// Publishes the host table and keeps the resulting fresh state in the
    /// closed owner accepted by final native control binding. Settle and publish
    /// numerical copies under the caller's recovery scope before binding or
    /// exchanging this owner; the marker does not establish those facts.
    pub(crate) fn publish_for_control(
        self,
    ) -> Result<
        (PublishedDenseResidentKvState, InferencePromptCompletion),
        DenseResidentKvPublishError<'a>,
    > {
        self.publish()
            .map(|(state, completion)| (PublishedDenseResidentKvState { state }, completion))
    }

    /// Attach the actual dense table's independent registered charge, then move
    /// that same table into a fresh state. No table clone or second allocation
    /// occurs. Historical requests, branch revisions and transaction flags are
    /// never inherited. The finish-only prompt remainder is not a text-step or
    /// native-completion grant; finish it only after all preparation succeeds.
    pub(crate) fn publish(
        self,
    ) -> Result<(MlxKeyValueState, InferencePromptCompletion), DenseResidentKvPublishError<'a>>
    {
        let Self {
            layers,
            layout,
            global_layer_start,
        } = self;
        let key =
            StorageIdentity::HostMetadata(layers.metadata().identity().registry_key().clone());
        match layers.publish(key) {
            Ok((layers, completion)) => Ok((
                MlxKeyValueState {
                    layers,
                    layout,
                    global_layer_start,
                    paged_transaction_branch: false,
                    inference_retention: InferenceRetention::new(),
                },
                completion,
            )),
            Err(error) => {
                let (layers, error) = error.into_parts();
                Err(DenseResidentKvPublishError {
                    owner: Self {
                        layers,
                        layout,
                        global_layer_start,
                    },
                    error,
                })
            }
        }
    }
}

impl fmt::Debug for PreparedDenseResidentKvState<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedDenseResidentKvState")
            .field("slot_count", &self.layers.len())
            .field("retained_slot_bytes", &self.retained_slot_bytes())
            .field("global_layer_start", &self.global_layer_start)
            .finish_non_exhaustive()
    }
}

/// Typed failed host publication, retaining the completed table and its hold.
#[derive(Debug)]
pub(crate) struct DenseResidentKvPublishError<'a> {
    owner: PreparedDenseResidentKvState<'a>,
    error: HostSlotAttachmentError<WorkingMemoryError>,
}

impl<'a> DenseResidentKvPublishError<'a> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        PreparedDenseResidentKvState<'a>,
        HostSlotAttachmentError<WorkingMemoryError>,
    ) {
        (self.owner, self.error)
    }
}

impl fmt::Display for DenseResidentKvPublishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}

impl std::error::Error for DenseResidentKvPublishError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
