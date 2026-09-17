//! Closed inline decoder slots joined to sampler and numerical copy custody.

use super::{
    funding::{DecoderCopySource, WorkingMemoryDecoderHostScope},
    residual::RegisteredStoragePin,
    AdmittedWorkspaceCopy, FundedSamplerCopy, InferenceExecutionIdentity, RegisteredSamplingCopy,
    SamplingCopyAdmissionError, WorkingMemoryError, WorkingMemoryPool, WorkingMemoryStorage,
    WorkspaceCopyLimits,
};
use crate::{
    HostMetadataIdentity, HostMetadataKey, HostSlotFinishError, HostSlotInitialization,
    HostSlotInitializationBuilder, HostSlotInitializationError, HostSlotPushError,
    InitializedHostSlots,
};
use std::fmt;

/// Provider mapping into the existing registered-storage key namespace.
/// Implementations must project the actual host table identity represented by
/// this key; numerical keys return None. The runtime checks that identity and
/// exact capacity against the actual borrowed table before existing-only pinning.
/// This is an accounting identity contract, not a payload or execution grant.
/// Keys must retain only independent, payload-free identity data. In particular,
/// never retain the table, its HostSlotMetadata token or its funded owner: a
/// completed table may own an attached registration containing this key.
pub trait HostSlotStorageKey: Clone + Ord + Send + Sync + 'static {
    /// Optional inline projection for an actual nested host-table identity.
    /// Runs outside the accounting lock; it may only move the supplied key.
    /// Absence preserves existing flat-table implementations and refuses nested
    /// source pinning. The registry still authenticates exact capacity/origin.
    fn from_host_slot_identity(_identity: crate::HostMetadataKey) -> Option<Self> {
        None
    }

    /// Original table identity, when this is a host-slot storage key.
    fn host_slot_identity(&self) -> Option<&HostMetadataKey>;
}

enum DecoderSource<'a, K: Ord + Send + 'static> {
    Registered(WorkingMemoryStorage<K>),
    Funded {
        scope: &'a WorkingMemoryDecoderHostScope,
        execution: &'a InferenceExecutionIdentity,
    },
}

/// A closed initialization plan borrowing actual original or saved slot values,
/// with exact existing source custody. No arbitrary byte amount, raw metadata
/// token, execution identity, or general funding scope constructs this proof.
/// It describes inline slots only; nested resources require the enclosing native
/// program and its complete source inventory.
#[must_use = "source binding does not itself allocate or authorize a copy"]
pub struct RegisteredDecoderHostCopy<'a, T, K: Ord + Send + 'static, D = T> {
    plan: HostSlotInitialization<'a, T, D>,
    source: DecoderSource<'a, K>,
    destination_identity: Option<HostMetadataIdentity>,
}

impl<'a, T, K: HostSlotStorageKey> RegisteredDecoderHostCopy<'a, T, K> {
    /// Pins an already registered actual table, without registering new bytes.
    /// The provider's key projection must name this exact table, not another
    /// same-sized owner. Callers retain settled source access through copying.
    pub fn bind(
        pool: &WorkingMemoryPool,
        plan: HostSlotInitialization<'a, T>,
        key: K,
    ) -> Result<Self, DecoderCopyAdmissionError> {
        if key.host_slot_identity() != Some(plan.source_metadata().identity().registry_key()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let capacity = plan
            .source_metadata()
            .capacity_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let source = pool.pin_registered_storage([(key, capacity)])?;
        Ok(Self {
            plan,
            source: DecoderSource::Registered(source),
            destination_identity: None,
        })
    }
}

impl<T, D, K: Ord + Send + 'static> RegisteredDecoderHostCopy<'_, T, K, D> {
    /// Exact fixed destination slot payload, excluding nested allocations.
    pub fn retained_bytes(&self) -> u64 {
        self.plan.retained_bytes()
    }
    /// Protected closed initialization/fill envelope, retained for owner lifetime.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.plan.initialization_peak_bytes()
    }
    /// Number of values required by the actual source table.
    pub fn len(&self) -> usize {
        self.plan.len()
    }
    /// Whether the actual source table is empty.
    pub fn is_empty(&self) -> bool {
        self.plan.is_empty()
    }
    /// Original value borrow; copying nested payload needs separate authority.
    pub fn source_at(&self, index: usize) -> Option<&T> {
        self.plan.source_at(index)
    }

    fn source(&self) -> DecoderCopySource<'_, K> {
        match &self.source {
            DecoderSource::Registered(source) => DecoderCopySource::Registered(source),
            DecoderSource::Funded { scope, execution } => {
                DecoderCopySource::Funded { scope, execution }
            }
        }
    }
}

impl<T, D, K: Ord + Send + 'static> fmt::Debug for RegisteredDecoderHostCopy<'_, T, K, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredDecoderHostCopy")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

/// A sampler, fixed-slot initializer and numerical program sharing one admission.
/// `complete_source` pins exactly the supplied registered origins; the native
/// closed decoder projection must supply its complete actual nested inventory.
/// A generic registration handle alone is not semantic completeness evidence.
#[must_use = "the joined plans require one destination admission before copying"]
pub struct RegisteredTextComponentsCopy<'a, T, K: Ord + Send + 'static> {
    sampling: RegisteredSamplingCopy<'a, K>,
    decoder: RegisteredDecoderHostCopy<'a, T, K>,
    complete_source: WorkingMemoryStorage<K>,
    bytes: u64,
}

impl<'a, K: Clone + Ord + Send + Sync + 'static> RegisteredSamplingCopy<'a, K> {
    /// Joins actual slot initialization and mandatory complete native source
    /// custody. Uncopied roots belong in complete_source, not in the destination
    /// operand trace. All origins are revalidated atomically at admission.
    pub fn with_decoder_slots<T>(
        self,
        decoder: RegisteredDecoderHostCopy<'a, T, K>,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<RegisteredTextComponentsCopy<'a, T, K>, DecoderCopyAdmissionError> {
        let bytes = self
            .required_bytes()
            .checked_add(decoder.initialization_peak_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(RegisteredTextComponentsCopy {
            sampling: self,
            decoder,
            complete_source,
            bytes,
        })
    }
}

impl<T, K: Ord + Send + 'static> RegisteredTextComponentsCopy<'_, T, K> {
    /// Combined host holds plus numerical demand, before a safety reserve.
    pub fn required_bytes(&self) -> u64 {
        self.bytes
    }
}
impl<T, K: Ord + Send + 'static> fmt::Debug for RegisteredTextComponentsCopy<'_, T, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredTextComponentsCopy")
            .field("decoder", &self.decoder)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

/// Rejection of a closed decoder component before destination publication.
#[derive(Debug, thiserror::Error)]
pub enum DecoderCopyAdmissionError {
    /// Actual source or destination slot geometry is not representable.
    #[error("{0}")]
    Initialization(#[from] HostSlotInitializationError),
    /// A finite original source carrier retained its failed construction prefix.
    #[error("{0}")]
    OriginalSources(#[from] super::OriginalStorageSourcesError),
    /// The actual sampler component could not be prepared.
    #[error("{0}")]
    Sampling(#[from] SamplingCopyAdmissionError),
    /// Source custody, arithmetic or shared-domain admission failed.
    #[error("{0}")]
    Memory(#[from] WorkingMemoryError),
    /// The complete incremental operation exceeds its application allowance.
    #[error(
        "text component copy needs {required_bytes} bytes; application limit is {budget_bytes}"
    )]
    ApplicationBudgetExceeded {
        /// Both host holds, numerical demand and safety reserve.
        required_bytes: u64,
        /// Explicit application allowance.
        budget_bytes: u64,
    },
}

/// Admitted fixed destination whose values are filled in source order.
/// There is no mutable slice, resize, owning export or future allocation grant.
/// Nested values must be created under the separately admitted native program.
#[must_use = "retain partial payloads and their host custody through failure"]
pub struct InitializedDecoderSlots<T> {
    slots: HostSlotInitializationBuilder<T>,
    source_identity: HostMetadataIdentity,
    source_capacity: u64,
    execution: InferenceExecutionIdentity,
    retained: u64,
    protected: u64,
    // Actual slot values and the fixed box retire before host certification.
    custody: WorkingMemoryDecoderHostScope,
}
impl<T> InitializedDecoderSlots<T> {
    /// Checks a separately retained actual source plan before native filling.
    /// The builder must still be empty and must match the original table and
    /// closed destination geometry. This check grants no access or submission
    /// authority; the native worker retains its actual settled source borrow.
    pub fn validate_source<S>(
        &self,
        source: &HostSlotInitialization<'_, S, T>,
    ) -> Result<(), WorkingMemoryError> {
        if self.initialized_count() != 0
            || source.source_metadata().identity() != &self.source_identity
            || source.source_metadata().capacity_bytes() != Some(self.source_capacity)
            || source.len() != self.len()
            || source.retained_bytes() != self.retained
            || source.initialization_peak_bytes() != self.protected
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }

    /// Total fixed number of destination values.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Whether no values are required.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Count already installed without allocation.
    pub fn initialized_count(&self) -> usize {
        self.slots.initialized_count()
    }
    /// Exact retained inline slot payload.
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }
    /// Initialization/fill envelope protected for this owner's lifetime.
    pub fn protected_bytes(&self) -> u64 {
        self.protected
    }
    /// Moves one already-created value. Rejection returns the incoming value;
    /// the caller retains its separate nested-resource/failure custody.
    pub fn push(&mut self, value: T) -> Result<(), HostSlotPushError<T>> {
        self.slots.push(value)
    }
    /// Publishes an immutable slot owner only when every slot is initialized.
    /// Failure retains both the partial payload and its host funding together.
    pub fn finish(self) -> Result<FundedDecoderSlots<T>, DecoderSlotsFinishError<T>> {
        let Self {
            slots,
            source_identity,
            source_capacity,
            execution,
            retained,
            protected,
            custody,
        } = self;
        match slots.finish() {
            Ok(slots) => Ok(FundedDecoderSlots {
                slots,
                execution,
                retained,
                protected,
                custody,
            }),
            Err(source) => Err(DecoderSlotsFinishError {
                source,
                source_identity,
                source_capacity,
                execution,
                retained,
                protected,
                custody,
            }),
        }
    }
}
impl<T> fmt::Debug for InitializedDecoderSlots<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedDecoderSlots")
            .field("slots", &self.slots)
            .field("retained", &self.retained)
            .field("protected", &self.protected)
            .finish_non_exhaustive()
    }
}

/// Incomplete finish keeps the partial table before its final host custody.
pub struct DecoderSlotsFinishError<T> {
    source: HostSlotFinishError<T>,
    source_identity: HostMetadataIdentity,
    source_capacity: u64,
    execution: InferenceExecutionIdentity,
    retained: u64,
    protected: u64,
    custody: WorkingMemoryDecoderHostScope,
}
impl<T> DecoderSlotsFinishError<T> {
    /// Required fixed value count.
    pub fn expected(&self) -> usize {
        self.source.expected()
    }
    /// Values already installed in the partial table.
    pub fn initialized(&self) -> usize {
        self.source.initialized()
    }
    /// Recovers the same admitted destination, preserving custody and contents.
    pub fn into_slots(self) -> InitializedDecoderSlots<T> {
        let Self {
            source,
            source_identity,
            source_capacity,
            execution,
            retained,
            protected,
            custody,
        } = self;
        InitializedDecoderSlots {
            slots: source.into_builder(),
            source_identity,
            source_capacity,
            execution,
            retained,
            protected,
            custody,
        }
    }
}
impl<T> fmt::Debug for DecoderSlotsFinishError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.source, f)
    }
}
impl<T> fmt::Display for DecoderSlotsFinishError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}
impl<T: 'static> std::error::Error for DecoderSlotsFinishError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Immutable fixed decoder slots with closed host funding, but no runnable
/// state, native grant, allocating Clone, or owning/mutable payload export.
/// T may contain interior mutability: provider source-exclusion and transitive
/// numerical inventory obligations remain necessary for an actual copy.
#[must_use = "retain this owner for the lifetime of its inline slot payload"]
pub struct FundedDecoderSlots<T> {
    slots: InitializedHostSlots<T>,
    execution: InferenceExecutionIdentity,
    retained: u64,
    protected: u64,
    custody: WorkingMemoryDecoderHostScope,
}
impl<T> FundedDecoderSlots<T> {
    /// Number of initialized values.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Whether this owner is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Borrow one value without transferring its host custody.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index)
    }
    /// Borrow all values in original slot order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        self.slots.iter()
    }
    /// Exact retained inline payload; nested resources are separately accounted.
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }
    /// Host envelope held independently of the native work/custody lifetime.
    pub fn protected_bytes(&self) -> u64 {
        self.protected
    }
    /// Borrows a diagnostic initialization plan from this actual saved table.
    /// The allocating worker remains crate-private. Use this only to match a
    /// separately admitted builder to the native worker's actual source borrow.
    pub fn prepare_copy_slots(
        &self,
    ) -> Result<HostSlotInitialization<'_, T>, HostSlotInitializationError> {
        self.slots.prepare_copy_slots()
    }

    /// Prepares another same-representation copy from this actual funded owner.
    /// It uses private host custody, never the original live table registration
    /// or request. Admission rechecks current origin health under the pool lock.
    pub fn prepare_copy<K: Ord + Send + 'static>(
        &self,
    ) -> Result<RegisteredDecoderHostCopy<'_, T, K>, DecoderCopyAdmissionError> {
        Ok(RegisteredDecoderHostCopy {
            plan: self.prepare_copy_slots()?,
            source: DecoderSource::Funded {
                scope: &self.custody,
                execution: &self.execution,
            },
            destination_identity: None,
        })
    }
}
impl<T> fmt::Debug for FundedDecoderSlots<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FundedDecoderSlots")
            .field("slots", &self.slots)
            .field("retained", &self.retained)
            .field("protected", &self.protected)
            .finish_non_exhaustive()
    }
}

impl WorkingMemoryPool {
    /// Commits one destination account after checking every source under the
    /// same pool lock, then copies the sampler and initializes fixed vacant slots.
    /// Both host scopes protect their own envelopes from native publication.
    /// The returned native scope alone retains the complete source pin bundle.
    /// No host cleanup certifies native work; partial/uncertain native copying
    /// must use existing recovery and conservative quarantine.
    ///
    /// Complete nested source inventory and the actual native program remain
    /// provider obligations. This is not whole-snapshot or runnable resume proof.
    /// Aggregate `AdmittedWorkspaceCopy::bytes` includes both host envelopes;
    /// do not add the overlapping component diagnostics to it again.
    pub fn copy_text_components<T, K: Clone + Ord + Send + Sync + 'static>(
        &self,
        mut copy: RegisteredTextComponentsCopy<'_, T, K>,
        limits: WorkspaceCopyLimits,
    ) -> Result<
        (
            FundedSamplerCopy,
            InitializedDecoderSlots<T>,
            AdmittedWorkspaceCopy,
        ),
        DecoderCopyAdmissionError,
    > {
        let bytes = copy
            .bytes
            .checked_add(limits.safety_reserve_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        if let Some(budget_bytes) = limits.application_memory_budget_bytes {
            if bytes > budget_bytes {
                return Err(DecoderCopyAdmissionError::ApplicationBudgetExceeded {
                    required_bytes: bytes,
                    budget_bytes,
                });
            }
        }
        let preparation = copy.complete_source.source_preparation().cloned();
        copy.decoder
            .prepare_destination_identity(preparation.as_ref())?;
        let execution = match preparation.as_ref() {
            Some(preparation) => {
                super::WorkspaceCopyAccountLayout::decoder_table()?.execution(preparation)
            }
            None => InferenceExecutionIdentity::default(),
        };
        let operands = copy.sampling.arrays.source().registration();
        // Stage allocation/destruction of the entire accounting-only bundle
        // outside the lock. Quarantine owns this single complete pin bundle.
        let table_pin = match &copy.decoder.source {
            DecoderSource::Registered(source) => Some(RegisteredStoragePin::new(source.clone())),
            DecoderSource::Funded { .. } => None,
        };
        let count = 2 + usize::from(table_pin.is_some());
        let pins = [
            Some(RegisteredStoragePin::new(operands.clone())),
            Some(RegisteredStoragePin::new(copy.complete_source.clone())),
            table_pin,
        ]
        .into_iter()
        .flatten();
        let pin = if copy.complete_source.has_source_preparation() {
            RegisteredStoragePin::aggregate_counted(pins, count)?
        } else {
            RegisteredStoragePin::aggregate(pins)
        };
        let source_identity = copy.decoder.plan.source_metadata().identity().clone();
        let source_capacity = copy
            .decoder
            .plan
            .source_metadata()
            .capacity_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let retained = copy.decoder.retained_bytes();
        let protected = copy.decoder.initialization_peak_bytes();
        let host = copy.sampling.host_bytes;
        let (funding, host_scope, decoder_scope, scope) = self.open_text_components_copy_account(
            copy.sampling.sampler.source(),
            copy.sampling.sampler.execution(),
            copy.decoder.source(),
            operands,
            &copy.complete_source,
            pin,
            &execution,
            bytes,
            host,
            protected,
            limits.capacity_bytes,
        )?;
        #[cfg(test)]
        tests::before_copy();
        let sampler = copy.sampling.sampler_plan.copy();
        let sampler =
            FundedSamplerCopy::from_shared_account(sampler, execution.clone(), host, host_scope);
        let slots = match &preparation {
            Some(authority) => copy.decoder.plan.initialize_prepared(
                copy.decoder
                    .destination_identity
                    .expect("identity prepared before admission"),
                authority,
            ),
            None => copy.decoder.plan.initialize(),
        };
        let slots = InitializedDecoderSlots {
            slots,
            source_identity,
            source_capacity,
            execution: execution.clone(),
            retained,
            protected,
            custody: decoder_scope,
        };
        let native = AdmittedWorkspaceCopy::from_account(execution, bytes, funding, scope);
        Ok((sampler, slots, native))
    }
}

#[cfg(test)]
mod tests;

mod dense_prompt;
pub use dense_prompt::{
    DenseDecoderFinishError, DenseDecoderHandoffError, FundedDenseHostSlots,
    InitializedDenseDecoderSlots, RegisteredDenseDecoderInitialization,
};

mod group;
pub use group::*;

mod preparation;
pub use preparation::DecoderHostPreparationError;
