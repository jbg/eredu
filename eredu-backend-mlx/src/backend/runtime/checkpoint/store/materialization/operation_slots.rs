//! Complete typed slots for explicitly supplied original-operation banks.
//! The closed selected caller supplies counts and original custody before work.
//! Final empty payload controls may be allocated here; no source lookup, native
//! observer, stream, numerical value, or lease is constructed.
use super::*;
use crate::backend::submission_recovery::observed::{
    bank::{BankLayout, PreparedOperationBank},
    operation::OperationRecovery,
    PreparedObservedRecovery,
};
use eredu_runtime::working_memory::OriginalTextControlGuard;
use safemlx::{error::Exception, OriginalScopeObserver};
use std::alloc::Layout;
/// Capacities from the closed selected source plan; no source or authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MaterializationPayloadShape {
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) pending_sources: usize,
}
impl MaterializationPayloadShape {
    /// Exact requested Vec element layouts, excluding allocator bookkeeping.
    pub(crate) fn requested_bytes(self) -> Option<usize> {
        Layout::array::<Array>(self.inputs)
            .ok()?
            .size()
            .checked_add(Layout::array::<Array>(self.outputs).ok()?.size())?
            .checked_add(
                Layout::array::<PendingWeightMaterialization>(self.pending_sources)
                    .ok()?
                    .size(),
            )
    }
}

fn rc_bytes<T>() -> Option<u64> {
    // Pinned Rust 1.98 repr(C, align(2)) RcInner, not a stable-Rust ABI.
    let layout = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align()
        .extend(Layout::new::<T>())
        .ok()?
        .0
        .pad_to_align();
    u64::try_from(layout.size()).ok()
}

/// One final node; its payload remains absent until the selected caller activates it.
pub(crate) struct PreparedPendingWeight {
    ready: PreparedObservedRecovery<PendingResources, OriginalTextControlGuard>,
    gguf_custody: OriginalTextControlGuard,
    gguf_host: super::gguf_host::PreparedGgufHostCopy,
    acquisition_metadata: Option<eredu_runtime::working_memory::OriginalHostMetadataCustody>,
}
impl PreparedPendingWeight {
    pub(in crate::backend::runtime::checkpoint::store) fn retain_acquisition_metadata(
        &mut self,
        custody: Option<eredu_runtime::working_memory::OriginalHostMetadataCustody>,
    ) {
        self.acquisition_metadata = custody;
    }
    pub(super) fn take_acquisition_metadata(
        &mut self,
    ) -> Option<eredu_runtime::working_memory::OriginalHostMetadataCustody> {
        self.acquisition_metadata.take()
    }
    /// Complete neutral supplied schedule plus the backend's existing scalar
    /// transformations and current-miss key buffers. Move outputs reuse their
    /// actual supplied destination.
    pub(crate) fn host_destination_requests(
        plan: &eredu_checkpoint::gguf_store::GgufConversionPlan,
    ) -> Result<eredu_gguf::StorageRequestBound, super::gguf_host::GgufHostCopyCause> {
        use super::gguf_host::GgufHostCopyCause;
        use crate::backend::runtime::execution::generic::gguf_host_typed::{
            TransformKind, TypedOutputRequest,
        };
        let mut bound = plan
            .supplied_storage_bound()
            .map_err(GgufHostCopyCause::SourcePlan)?;
        for ordinal in 0..plan.conversion().outputs().len() {
            let request = TypedOutputRequest::for_output(plan.conversion(), ordinal)
                .ok_or(GgufHostCopyCause::TypedBinding)?;
            if request.kind != TransformKind::Move {
                let layout = request
                    .requested_layout()
                    .ok_or(GgufHostCopyCause::Layout)?;
                bound = bound
                    .checked_add(eredu_gguf::StorageRequestBound::from_layout(layout))
                    .ok_or(GgufHostCopyCause::Layout)?;
            }
        }
        bound
            .checked_add(
                plan.identity()
                    .cache_storage_requests()
                    .ok_or(GgufHostCopyCause::Layout)?,
            )
            .ok_or(GgufHostCopyCause::Layout)
    }
    /// Current-miss cache allocation/transport controls. Key buffers are in the
    /// separate destination requests, shared existing context/source shells are not.
    pub(crate) fn cache_metadata_control_bytes(
    ) -> Result<u64, eredu_runtime::working_memory::WorkingMemoryError> {
        super::super::cache::control_bytes()
    }
    /// Read-only source controls; excludes the immutable payload backing.
    pub(crate) fn source_copy_control_bytes(
        plan: &eredu_checkpoint::gguf_store::GgufConversionPlan,
        runtime: &safemlx::PreparedInputRuntime,
    ) -> Result<(u64, usize), super::gguf_host::GgufHostCopyCause> {
        super::gguf_host::source_copy_control_bytes(plan, runtime)
    }

    /// Source layouts, immutable copied backing and one cache-control debit.
    /// Key buffers are in host_destination_requests; shared existing owners are
    /// separate. The source bank includes both controls and owned payloads.
    pub(crate) fn source_copy_storage_bytes(
        plan: &eredu_checkpoint::gguf_store::GgufConversionPlan,
        runtime: &safemlx::PreparedInputRuntime,
    ) -> Result<(u64, usize), super::gguf_host::GgufHostCopyCause> {
        let (bytes, scratch) = super::gguf_host::source_copy_storage_bytes(plan, runtime)?;
        let cache = super::super::cache::control_bytes()
            .map_err(super::gguf_host::GgufHostCopyCause::SourcePublication)?;
        Ok((
            bytes
                .checked_add(cache)
                .ok_or(super::gguf_host::GgufHostCopyCause::Layout)?,
            scratch,
        ))
    }

    /// Called only during the closed caller's prepaid bank construction.
    /// Uses the existing Box abort-on-OOM contract; custody remains in the node.
    pub(crate) fn new(custody: OriginalTextControlGuard) -> Self {
        Self::with_host_runtime(custody, None)
    }

    /// An actual runtime prepared during native model construction, never here.
    pub(crate) fn new_with_gguf_host_copy(
        custody: OriginalTextControlGuard,
        runtime: Rc<safemlx::PreparedInputRuntime>,
    ) -> Self {
        Self::with_host_runtime(custody, Some(runtime))
    }

    /// Concrete admitted G1/typed lane. It consumes the actual separately priced
    /// bank; refusal returns it unchanged before any pending-node allocation.
    /// A supplied source component also funds cache metadata. The explicit
    /// Vec-only component form leaves metadata ordinary and does not close the
    /// aggregate original-path fit. A present source bank never falls back.
    pub(crate) fn new_with_gguf_host_destinations(
        custody: OriginalTextControlGuard,
        runtime: Rc<safemlx::PreparedInputRuntime>,
        mut bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
    ) -> Result<
        Self,
        (
            eredu_runtime::working_memory::OriginalHostDestinationBank,
            eredu_runtime::working_memory::WorkingMemoryError,
        ),
    > {
        if !bank.belongs_to(&custody) {
            return Err((
                bank,
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let failure = PreparedGgufAdmittedFailure::prepare(custody.clone());
        let mut value = Self::new_with_gguf_host_copy(custody, runtime);
        value.gguf_host.raw_failure = Some(failure);
        value.gguf_host.source_constructions = bank.take_source_constructions();
        value.gguf_host.host_destinations = Some(bank);
        Ok(value)
    }

    /// Install the actual accepted source component. A foreign bank is returned
    /// untouched before the pending recovery/source owners are constructed.
    pub(crate) fn new_with_gguf_source_constructions(
        custody: OriginalTextControlGuard,
        runtime: Rc<safemlx::PreparedInputRuntime>,
        bank: eredu_runtime::working_memory::OriginalHostSourceBank,
    ) -> Result<
        Self,
        (
            eredu_runtime::working_memory::OriginalHostSourceBank,
            eredu_runtime::working_memory::WorkingMemoryError,
        ),
    > {
        if !bank.belongs_to(&custody) {
            return Err((
                bank,
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let mut value = Self::new_with_gguf_host_copy(custody, runtime);
        value.gguf_host.source_constructions = Some(bank);
        Ok(value)
    }

    fn with_host_runtime(
        custody: OriginalTextControlGuard,
        runtime: Option<Rc<safemlx::PreparedInputRuntime>>,
    ) -> Self {
        Self {
            ready: PreparedObservedRecovery::new(custody.clone()),
            gguf_host: super::gguf_host::PreparedGgufHostCopy::new(runtime, custody.clone()),
            gguf_custody: custody,
            acquisition_metadata: None,
        }
    }

    /// Owning per-slot controls shared by direct construction and bank sizing.
    /// Enclosing slot/constructor transports are priced by their actual caller.
    pub(crate) fn slot_control_bytes() -> Option<u64> {
        let node =
            PreparedObservedRecovery::<PendingResources, OriginalTextControlGuard>::control_bytes::<
                Exception,
            >()?;
        let dispatch =
            OperationRecovery::<PendingResources, OriginalTextControlGuard>::control_bytes()?;
        let native = u64::try_from(OriginalScopeObserver::control_bytes()?).ok()?;
        node.checked_add(dispatch)?
            .checked_add(native)?
            .checked_add(PreparedGgufMaterializationFailure::control_bytes()?)?
            .checked_add(PreparedGgufAdmittedFailure::control_bytes()?)?
            .checked_add(super::gguf_host::PreparedGgufHostCopy::control_bytes()?)
    }
    /// F/E must be the actual factory and its owning failure type. This prices
    /// named controls, not dynamic payloads or a proof of the supplied count.
    pub(crate) fn bank_layout<F, E>(count: usize) -> Option<BankLayout> {
        PreparedOperationBank::<Self>::layout::<F, E>(count, Self::slot_control_bytes()?)
    }

    /// The caller authenticated the current innermost role before any producer.
    /// Moving the final prepared node never allocates or creates a child Scope.
    pub(super) fn activate(
        self,
        mut value: PendingResources,
        observer: OriginalScopeObserver,
    ) -> OperationRecovery<PendingResources, OriginalTextControlGuard> {
        value.acquisition_metadata = self.acquisition_metadata;
        value.gguf_custody = Some(self.gguf_custody);
        value.gguf_host = Some(self.gguf_host);
        OperationRecovery::original(self.ready, value, observer)
    }
}

/// Original account retained by the shared materialization completion node.
pub(super) enum WeightMaterializationCustody {
    Text {
        _guard: OriginalTextControlGuard,
    },
    Source {
        _guard: eredu_runtime::working_memory::SharedNativeInitializationCustody,
    },
}

/// Final empty recovery and shared payload controls, prepared before activation.
pub(crate) struct PreparedWeightMaterialization {
    value: Rc<MaterializationResources>,
    ready: PreparedObservedRecovery<Rc<MaterializationResources>, WeightMaterializationCustody>,
}
impl PreparedWeightMaterialization {
    /// Called only during the closed caller's prepaid bank construction.
    /// Uses the existing Box abort-on-OOM contract; custody remains in the node.
    pub(crate) fn new(custody: OriginalTextControlGuard) -> Self {
        Self::with_custody(WeightMaterializationCustody::Text { _guard: custody })
    }

    fn with_custody(custody: WeightMaterializationCustody) -> Self {
        Self {
            value: Rc::new(MaterializationResources {
                inputs: Vec::new(),
                outputs: Vec::new(),
                _sources: Vec::new(),
                event: None,
                children: Cell::new(0),
                failed: Cell::new(false),
            }),
            ready: PreparedObservedRecovery::new(custody),
        }
    }

    /// Prepare the final three Vec allocations before issuing this slot.
    /// Failure retains the exact slot, successful prefix and original custody.
    pub(crate) fn try_new(
        custody: OriginalTextControlGuard,
        shape: MaterializationPayloadShape,
    ) -> Result<Self, (Self, std::collections::TryReserveError)> {
        Self::reserve(Self::new(custody), shape)
    }

    pub(super) fn try_new_source(
        custody: eredu_runtime::working_memory::SharedNativeInitializationCustody,
        shape: MaterializationPayloadShape,
    ) -> Result<Self, (Self, std::collections::TryReserveError)> {
        Self::reserve(
            Self::with_custody(WeightMaterializationCustody::Source { _guard: custody }),
            shape,
        )
    }

    fn reserve(
        mut ready: Self,
        shape: MaterializationPayloadShape,
    ) -> Result<Self, (Self, std::collections::TryReserveError)> {
        let value = Rc::get_mut(&mut ready.value).expect("unissued prepared payload");
        let reserved = (|| {
            value.inputs.try_reserve_exact(shape.inputs)?;
            value.outputs.try_reserve_exact(shape.outputs)?;
            value._sources.try_reserve_exact(shape.pending_sources)
        })();
        match reserved {
            Ok(()) => Ok(ready),
            Err(error) => Err((ready, error)),
        }
    }

    pub(crate) fn bank_layout_with_payload<F, E>(
        count: usize,
        shape: MaterializationPayloadShape,
    ) -> Option<BankLayout> {
        let mut layout = Self::bank_layout::<F, E>(count)?;
        let bytes = u64::try_from(shape.requested_bytes()?)
            .ok()?
            .checked_mul(u64::try_from(count).ok()?)?;
        layout.prepared_slot_control_bytes =
            layout.prepared_slot_control_bytes.checked_add(bytes)?;
        layout.total_control_bytes = layout.total_control_bytes.checked_add(bytes)?;
        Some(layout)
    }

    /// F/E must be the actual factory and its owning failure type. This prices
    /// named controls, not dynamic payloads or a proof of the supplied count.
    pub(crate) fn bank_layout<F, E>(count: usize) -> Option<BankLayout> {
        let node = PreparedObservedRecovery::<
            Rc<MaterializationResources>,
            WeightMaterializationCustody,
        >::control_bytes::<Exception>()?;
        let dispatch = OperationRecovery::<Rc<MaterializationResources>, WeightMaterializationCustody>::control_bytes()?;
        let native = u64::try_from(OriginalScopeObserver::control_bytes()?).ok()?;
        let slot = node
            .checked_add(dispatch)?
            .checked_add(native)?
            .checked_add(rc_bytes::<MaterializationResources>()?)?;
        PreparedOperationBank::<Self>::layout::<F, E>(count, slot)
    }

    /// The caller authenticated the current innermost role before any producer.
    /// Moving the final prepared node never allocates or creates a child Scope.
    pub(super) fn activate(
        mut self,
        mut inputs: Vec<Array>,
        mut sources: Vec<PendingWeightMaterialization>,
        observer: OriginalScopeObserver,
    ) -> Result<
        OperationRecovery<Rc<MaterializationResources>, WeightMaterializationCustody>,
        CheckpointMaterializationError,
    > {
        let value = Rc::get_mut(&mut self.value).expect("unissued prepared payload");
        require_capacity(&value.inputs, inputs.len(), "materialization inputs")?;
        require_capacity(&value._sources, sources.len(), "materialization sources")?;
        value.inputs.append(&mut inputs);
        value._sources.append(&mut sources);
        Ok(OperationRecovery::original(
            self.ready, self.value, observer,
        ))
    }

    /// Take the source population while returning the already reserved empty
    /// source vector to its enclosing transfer. Both actual arrays remain owned.
    pub(super) fn activate_detaching(
        mut self,
        outputs: &[Array],
        sources: &mut Vec<PendingWeightMaterialization>,
        observer: OriginalScopeObserver,
    ) -> Result<
        OperationRecovery<Rc<MaterializationResources>, WeightMaterializationCustody>,
        CheckpointMaterializationError,
    > {
        let value = Rc::get_mut(&mut self.value).expect("unissued prepared payload");
        require_capacity(&value.outputs, outputs.len(), "materialization outputs")?;
        require_capacity(
            &value._sources,
            sources.capacity(),
            "replacement transfer sources",
        )?;
        // Validate every destination before cloning handles or changing source
        // ownership. No Vec allocation or original-source fallback follows.
        value.outputs.extend(outputs.iter().cloned());
        std::mem::swap(&mut value._sources, sources);
        Ok(OperationRecovery::original(
            self.ready, self.value, observer,
        ))
    }
}

pub(super) fn require_capacity<T>(
    destination: &Vec<T>,
    required: usize,
    family: &'static str,
) -> Result<(), CheckpointMaterializationError> {
    if required > destination.capacity() {
        return Err(CheckpointMaterializationError::OriginalPayloadCapacity {
            family,
            required,
            capacity: destination.capacity(),
        });
    }
    Ok(())
}

/// One final node; its payload remains absent until the selected caller activates it.
pub(crate) struct PreparedMaterializationObservation {
    ready: PreparedObservedRecovery<MaterializationObservation, OriginalTextControlGuard>,
}
impl PreparedMaterializationObservation {
    /// Called only during the closed caller's prepaid bank construction.
    /// Uses the existing Box abort-on-OOM contract; custody remains in the node.
    pub(crate) fn new(custody: OriginalTextControlGuard) -> Self {
        Self {
            ready: PreparedObservedRecovery::new(custody),
        }
    }

    /// F/E must be the actual factory and its owning failure type. This prices
    /// named controls, not dynamic payloads or a proof of the supplied count.
    pub(crate) fn bank_layout<F, E>(count: usize) -> Option<BankLayout> {
        let node = PreparedObservedRecovery::<MaterializationObservation, OriginalTextControlGuard>
            ::control_bytes::<Exception>()?;
        let dispatch = OperationRecovery::<MaterializationObservation, OriginalTextControlGuard>::control_bytes()?;
        let native = u64::try_from(OriginalScopeObserver::control_bytes()?).ok()?;
        let slot = node.checked_add(dispatch)?.checked_add(native)?;
        PreparedOperationBank::<Self>::layout::<F, E>(count, slot)
    }

    /// The caller authenticated the current innermost role before any producer.
    /// Moving the final prepared node never allocates or creates a child Scope.
    pub(super) fn activate(
        self,
        value: MaterializationObservation,
        observer: OriginalScopeObserver,
    ) -> OperationRecovery<MaterializationObservation, OriginalTextControlGuard> {
        OperationRecovery::original(self.ready, value, observer)
    }
}

/// Explicit mutable storage loan issued after the selected request/role check.
/// The projection itself is not permission and cannot refill or replace a bank.
pub(crate) struct OriginalMaterializationSlots<'a> {
    pub(crate) acquisitions: &'a mut super::super::PreparedSourceAcquisitions,
    pub(crate) pending_weights: &'a mut PreparedOperationBank<PreparedPendingWeight>,
    pub(crate) weight_materializations:
        &'a mut PreparedOperationBank<PreparedWeightMaterialization>,
    pub(crate) observations: &'a mut PreparedOperationBank<PreparedMaterializationObservation>,
}
impl OriginalMaterializationSlots<'_> {
    pub(crate) fn acquire_lease(
        &mut self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        key: &str,
        selection: &eredu_checkpoint::store::TensorSelection,
        observer: &OriginalScopeObserver,
    ) -> Result<super::super::acquisition::AcquiredSourceLease, CheckpointMaterializationError>
    {
        super::validate_operation(observer)?;
        self.acquisitions
            .acquire(source, key, selection)
            .map_err(CheckpointMaterializationError::from)
    }

    pub(crate) fn reborrow(&mut self) -> OriginalMaterializationSlots<'_> {
        OriginalMaterializationSlots {
            acquisitions: self.acquisitions,
            pending_weights: self.pending_weights,
            weight_materializations: self.weight_materializations,
            observations: self.observations,
        }
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        size_of::<Self>()
            .checked_add(size_of::<&mut Self>())?
            .checked_add(size_of::<Option<Self>>())
    }
}
