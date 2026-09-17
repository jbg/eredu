//! Closed dense decoder construction in a fresh request's original prompt stage.

use super::*;
use crate::working_memory::{
    InferencePreparationStage, InferencePromptCompletion, WorkingMemoryFundingRun,
    WorkingMemoryFundingScope, WorkingMemoryReservation,
};
use crate::{
    DenseHostSlotFinishError, DenseHostSlotInitialization, DenseHostSlotInitializationBuilder,
    HostSlotAttachmentError, HostSlotMetadata, HostSlotTable, InitializedDenseHostSlots,
};

/// Actual source values and their original custody with a checked dense
/// destination extent. Neither a raw count nor a metadata token creates this
/// proof. Source interior mutability still requires the provider's settled
/// access contract; nested resources remain separate from this inline bound.
#[must_use = "consume this source plan only through a fresh prompt stage"]
pub struct RegisteredDenseDecoderInitialization<'a, S, D, K: HostSlotStorageKey> {
    plan: DenseHostSlotInitialization<'a, S, D>,
    source: DecoderSource<'a, K>,
    destination_identity: Option<HostMetadataIdentity>,
    // Retains constructor storage; this never replaces the fresh prompt grant.
    preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl<'a, S, PreviousD, K: HostSlotStorageKey> RegisteredDecoderHostCopy<'a, S, K, PreviousD> {
    /// Selects dense destination cells while retaining this exact registered
    /// or funded source borrow. No values are allocated, cloned or converted.
    pub fn for_dense_destination<D>(
        self,
    ) -> Result<RegisteredDenseDecoderInitialization<'a, S, D, K>, DecoderCopyAdmissionError> {
        Ok(RegisteredDenseDecoderInitialization {
            plan: self.plan.for_dense_destination()?,
            source: self.source,
            destination_identity: None,
            preparation: None,
        })
    }
}

impl<'a, S, D, K: HostSlotStorageKey> RegisteredDenseDecoderInitialization<'a, S, D, K> {
    /// Fixed destination count from the actual source.
    pub fn len(&self) -> usize {
        self.plan.len()
    }
    /// Whether the actual source has no slots.
    pub fn is_empty(&self) -> bool {
        self.plan.is_empty()
    }
    /// Original borrowed source values, with no numerical copy.
    pub fn source_at(&self, index: usize) -> Option<&S> {
        self.plan.source_at(index)
    }
    /// Actual source metadata, not the future destination's identity.
    pub fn source_metadata(&self) -> &HostSlotMetadata {
        self.plan.source_metadata()
    }
    /// Exact requested dense destination payload.
    pub fn retained_bytes(&self) -> u64 {
        self.plan.retained_bytes()
    }
    /// Dense payload plus the closed worker's explicit value temporaries.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.plan.initialization_peak_bytes()
    }

    pub(super) fn source(&self) -> DecoderCopySource<'_, K> {
        match &self.source {
            DecoderSource::Registered(source) => DecoderCopySource::Registered(source),
            DecoderSource::Funded { scope, execution } => {
                DecoderCopySource::Funded { scope, execution }
            }
        }
    }

    pub(super) fn prepare_destination(
        &mut self,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<(), WorkingMemoryError> {
        if self.destination_identity.is_some() || self.preparation.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.destination_identity = host.map(HostMetadataIdentity::prepared_host).transpose()?;
        self.preparation = host.cloned();
        Ok(())
    }

    /// Owning cold constructor facts for one exact dense table. The provider
    /// supplies the actual source branch and key clone payload extent.
    pub fn preparation_control_bytes(
        registered: bool,
        nested_key_bytes: u64,
    ) -> Result<usize, preparation::DecoderHostPreparationError> {
        use std::mem::size_of;
        preparation::add([
            RegisteredDecoderHostCopy::<S, K>::binding_control_bytes(registered)?,
            DenseHostSlotInitialization::<S, D>::preparation_control_bytes()
                .ok_or(preparation::DecoderHostPreparationError::UnknownBound)?,
            preparation::memory(
                crate::working_memory::storage::dense_host_transfer_control_bytes::<K>(
                    nested_key_bytes,
                ),
            )?,
            size_of::<Self>(),
            size_of::<InitializedDenseDecoderSlots<'_, S, D, K>>(),
            size_of::<FundedDenseHostSlots<'_, S, D, K>>(),
            size_of::<
                Result<
                    (
                        InitializedDenseDecoderSlots<'_, S, D, K>,
                        WorkingMemoryFundingScope,
                    ),
                    DecoderCopyAdmissionError,
                >,
            >(),
            size_of::<
                Result<
                    (HostSlotTable<D>, InferencePromptCompletion),
                    DenseDecoderHandoffError<'_, S, D, K>,
                >,
            >(),
            size_of::<Option<HostMetadataIdentity>>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
        ])
    }

    pub(in crate::working_memory) fn construct(
        mut self,
        stage: InferencePreparationStage,
        reservation: &WorkingMemoryReservation,
        funding: &WorkingMemoryFundingRun,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<
        (
            InitializedDenseDecoderSlots<'a, S, D, K>,
            WorkingMemoryFundingScope,
        ),
        DecoderCopyAdmissionError,
    > {
        let host = complete_source.source_preparation().cloned();
        self.prepare_destination(host.as_ref())?;
        let registered = matches!(&self.source, DecoderSource::Registered(_));
        let registered_source = self.registered_pin();
        let source_pins = [
            registered_source,
            Some(RegisteredStoragePin::new(complete_source.clone())),
        ]
        .into_iter()
        .flatten();
        let pins = if host.is_some() {
            RegisteredStoragePin::aggregate_counted(source_pins, 1 + usize::from(registered))?
        } else {
            RegisteredStoragePin::aggregate(source_pins)
        };
        let (execution, custody, native) = funding.open_dense_prompt_scopes(
            reservation,
            self.source(),
            &complete_source,
            pins,
            self.initialization_peak_bytes(),
        )?;
        #[cfg(test)]
        tests::before_initialize();
        Ok((
            self.initialize_funded(Some(stage), execution, custody),
            native,
        ))
    }

    pub(super) fn registered_pin(&self) -> Option<RegisteredStoragePin> {
        match &self.source {
            DecoderSource::Registered(source) => Some(RegisteredStoragePin::new(source.clone())),
            DecoderSource::Funded { .. } => None,
        }
    }

    pub(super) fn initialize_funded(
        self,
        stage: Option<InferencePreparationStage>,
        execution: InferenceExecutionIdentity,
        custody: WorkingMemoryDecoderHostScope,
    ) -> InitializedDenseDecoderSlots<'a, S, D, K> {
        let Self {
            plan,
            source,
            destination_identity,
            preparation,
        } = self;
        let (plan, slots) = plan.initialize_retaining_source();
        InitializedDenseDecoderSlots {
            slots,
            source: Self {
                plan,
                source,
                destination_identity,
                preparation,
            },
            stage,
            execution,
            custody,
        }
    }
}

impl<S, D, K: HostSlotStorageKey> fmt::Debug for RegisteredDenseDecoderInitialization<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.plan, f)
    }
}

/// One partial dense destination, its actual source borrow and consumed prompt
/// claim. Payload precedes the final private host custody on every drop path.
/// No primitive builder, mutable slice, host scope or raw table can escape.
#[must_use = "retain partial values with their host custody through recovery"]
pub struct InitializedDenseDecoderSlots<'a, S, D, K: HostSlotStorageKey> {
    slots: DenseHostSlotInitializationBuilder<D>,
    source: RegisteredDenseDecoderInitialization<'a, S, D, K>,
    stage: Option<InferencePreparationStage>,
    execution: InferenceExecutionIdentity,
    custody: WorkingMemoryDecoderHostScope,
}

impl<'a, S, D, K: HostSlotStorageKey> InitializedDenseDecoderSlots<'a, S, D, K> {
    /// Fixed destination count.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Whether no destination values are required.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Number of consecutive initialized values.
    pub fn initialized_count(&self) -> usize {
        self.slots.initialized_count()
    }
    /// Actual source value; creating nested destination resources still needs
    /// the separately admitted native program and its completion custody.
    pub fn source_at(&self, index: usize) -> Option<&S> {
        self.source.source_at(index)
    }
    /// Exact retained dense destination payload.
    pub fn retained_bytes(&self) -> u64 {
        self.source.retained_bytes()
    }
    /// Protected initialization/fill envelope.
    pub fn protected_bytes(&self) -> u64 {
        self.source.initialization_peak_bytes()
    }

    /// Matches a separately retained actual native source plan before filling.
    /// The destination must still be empty; equal-sized unrelated sources fail.
    pub fn validate_source(
        &self,
        source: &DenseHostSlotInitialization<'_, S, D>,
    ) -> Result<(), WorkingMemoryError> {
        if self.initialized_count() != 0
            || source.source_metadata().identity() != self.source.source_metadata().identity()
            || source.source_metadata().capacity_bytes()
                != self.source.source_metadata().capacity_bytes()
            || source.len() != self.len()
            || source.retained_bytes() != self.retained_bytes()
            || source.initialization_peak_bytes() != self.protected_bytes()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }

    /// Moves an already-created value without growth. Full rejection preserves
    /// the incoming value; its independent resources remain the caller's duty.
    pub fn push(&mut self, value: D) -> Result<(), HostSlotPushError<D>> {
        self.slots.push(value)
    }

    /// Finishes only after every value exists. An incomplete error retains the
    /// partial payload, actual source, consumed prompt claim and full host hold.
    pub fn finish(
        self,
    ) -> Result<FundedDenseHostSlots<'a, S, D, K>, DenseDecoderFinishError<'a, S, D, K>> {
        let Self {
            slots,
            mut source,
            stage,
            execution,
            custody,
        } = self;
        // An incomplete finish keeps the prepared identity with the recoverable
        // builder; a later exact finish must retain the same prepared route.
        let prepared = if slots.initialized_count() == slots.len() {
            source
                .destination_identity
                .take()
                .zip(source.preparation.as_ref())
        } else {
            None
        };
        match slots.finish_with_preparation(prepared) {
            Ok(slots) => Ok(FundedDenseHostSlots {
                slots,
                source,
                stage,
                execution,
                custody,
            }),
            Err(error) => Err(DenseDecoderFinishError {
                error,
                source,
                stage,
                execution,
                custody,
            }),
        }
    }
}

impl<S, D, K: HostSlotStorageKey> fmt::Debug for InitializedDenseDecoderSlots<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedDenseDecoderSlots")
            .field("slots", &self.slots)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// Incomplete fill with all payload/source/claim custody retained for recovery.
#[must_use = "retire or recover the partial destination with its host hold"]
pub struct DenseDecoderFinishError<'a, S, D, K: HostSlotStorageKey> {
    error: DenseHostSlotFinishError<D>,
    source: RegisteredDenseDecoderInitialization<'a, S, D, K>,
    stage: Option<InferencePreparationStage>,
    execution: InferenceExecutionIdentity,
    custody: WorkingMemoryDecoderHostScope,
}
impl<'a, S, D, K: HostSlotStorageKey> DenseDecoderFinishError<'a, S, D, K> {
    /// Fixed required destination count.
    pub fn expected(&self) -> usize {
        self.error.expected()
    }
    /// Values still retained in the partial buffer.
    pub fn initialized(&self) -> usize {
        self.error.initialized()
    }
    /// Recovers the same funded partial owner, never the unfunded inner Vec.
    pub fn into_builder(self) -> InitializedDenseDecoderSlots<'a, S, D, K> {
        let Self {
            error,
            source,
            stage,
            execution,
            custody,
        } = self;
        InitializedDenseDecoderSlots {
            slots: error.into_builder(),
            source,
            stage,
            execution,
            custody,
        }
    }
}
impl<S, D, K: HostSlotStorageKey> fmt::Debug for DenseDecoderFinishError<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.error, f)
    }
}
impl<S, D, K: HostSlotStorageKey> fmt::Display for DenseDecoderFinishError<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl<S, D: 'static, K: HostSlotStorageKey> std::error::Error
    for DenseDecoderFinishError<'_, S, D, K>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Complete actual dense table held by its original prompt construction. Only
/// successful exact attachment may release a live table to native installation.
/// It carries no old request grant or permission for new nested allocations.
#[must_use = "publish the actual table or retire it with its protected host hold"]
pub struct FundedDenseHostSlots<'a, S, D, K: HostSlotStorageKey> {
    slots: InitializedDenseHostSlots<D>,
    source: RegisteredDenseDecoderInitialization<'a, S, D, K>,
    stage: Option<InferencePreparationStage>,
    execution: InferenceExecutionIdentity,
    custody: WorkingMemoryDecoderHostScope,
}
impl<S, D, K: HostSlotStorageKey> fmt::Debug for FundedDenseHostSlots<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FundedDenseHostSlots")
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}
impl<'a, S, D, K: HostSlotStorageKey> FundedDenseHostSlots<'a, S, D, K> {
    /// Actual destination metadata, suitable for HostSlotStorageKey projection.
    /// Cloned tokens retain no table payload and grant no allocation authority.
    pub fn metadata(&self) -> &HostSlotMetadata {
        self.slots.metadata()
    }
    /// Fixed completed count.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Whether the actual destination is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Borrows an actual completed value without mutable or owning extraction.
    pub fn get(&self, index: usize) -> Option<&D> {
        self.slots.get(index)
    }
    /// Exact requested dense payload, before transfer to registered custody.
    pub fn retained_bytes(&self) -> u64 {
        self.source.retained_bytes()
    }
    /// Original initialization/fill host envelope.
    pub fn protected_bytes(&self) -> u64 {
        self.source.initialization_peak_bytes()
    }

    /// Transfers only the actual table's retained charge, attaching it before
    /// the table is exposed. Failure preserves the entire completed owner and
    /// typed cause. Success returns the original prompt's finish-only remainder;
    /// all other native/input preparation must succeed before it is finished.
    /// This handoff certifies no general native work or copied nested resource.
    /// The key must obey HostSlotStorageKey's independent-identity contract;
    /// retaining this table's metadata in it would create an ownership cycle.
    pub fn publish(
        self,
        key: K,
    ) -> Result<(HostSlotTable<D>, InferencePromptCompletion), DenseDecoderHandoffError<'a, S, D, K>>
    {
        let (table, stage) = self.publish_table(key)?;
        Ok((
            table,
            InferencePromptCompletion::new(stage.expect("standalone prompt claim")),
        ))
    }

    // Group children have no prompt claim; only the closed group wrapper may
    // invoke this shared handoff for them. It never manufactures completion.
    pub(super) fn publish_table(
        mut self,
        key: K,
    ) -> Result<
        (HostSlotTable<D>, Option<InferencePreparationStage>),
        DenseDecoderHandoffError<'a, S, D, K>,
    > {
        let result = crate::working_memory::storage::publish_dense_host_slots_prepared(
            &self.slots,
            &mut self.custody,
            &self.execution,
            self.source.retained_bytes(),
            self.source.initialization_peak_bytes(),
            key,
            self.source.preparation.as_ref(),
        );
        if let Err(error) = result {
            return Err(DenseDecoderHandoffError { owner: self, error });
        }
        let Self {
            slots,
            source,
            stage,
            execution,
            custody,
        } = self;
        let table = slots.into_table();
        // The exact payload now owns its registered charge. Only the remaining
        // closed host temporaries retire here; native settlement is independent.
        drop(source);
        drop(execution);
        drop(custody);
        Ok((table, stage))
    }
}

/// Failed attachment retains the actual completed table and its prompt claim.
#[must_use = "retire or recover the completed owner without separating custody"]
pub struct DenseDecoderHandoffError<'a, S, D, K: HostSlotStorageKey> {
    owner: FundedDenseHostSlots<'a, S, D, K>,
    error: HostSlotAttachmentError<WorkingMemoryError>,
}
impl<'a, S, D, K: HostSlotStorageKey> DenseDecoderHandoffError<'a, S, D, K> {
    /// Original typed storage/attachment cause.
    pub fn error(&self) -> &HostSlotAttachmentError<WorkingMemoryError> {
        &self.error
    }
    /// Recovers the whole completed owner and cause, with no raw table export.
    pub fn into_parts(
        self,
    ) -> (
        FundedDenseHostSlots<'a, S, D, K>,
        HostSlotAttachmentError<WorkingMemoryError>,
    ) {
        (self.owner, self.error)
    }
}
impl<S, D, K: HostSlotStorageKey> fmt::Debug for DenseDecoderHandoffError<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseDecoderHandoffError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}
impl<S, D, K: HostSlotStorageKey> fmt::Display for DenseDecoderHandoffError<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl<S, D, K: HostSlotStorageKey> std::error::Error for DenseDecoderHandoffError<'_, S, D, K> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests;
