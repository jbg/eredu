use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use eredu_checkpoint::{recipe::EncodedRecipeRead, store::CheckpointSource};
use eredu_core::{
    checkpoint_mechanisms::{
        bounded_quantization::BoundedQuantizationBackend,
        unit_binding::{
            ComponentPreparationScope, ImmutableUnitStorage, NativeParameterUnitBinding,
            ParameterStorageBinding,
        },
        ExactParameterMaterializer, ParameterReadInput, ParameterReadScope,
    },
    residency::{
        conversion_claim::{ConversionRetentionBudget, ConversionRetentionError},
        observations::{DenseResidencyMechanisms, NativeResidencyObservations},
        parameter_binding::{physical_binding_bytes, WeightBinding},
        MemoryTier, NativeBoundedUnits, NativeResidentUnits, OffloadUnitId,
        ParameterConversionRetentionTrimReport, RealizedResidencyCopy, ResidencyCopyRequest,
        ResidencyCopyScope, ResidencyLease, ResidencyLeaseStorage, ResidencyObservationBindings,
        ResidencyStorage, ResidencyStorageMechanisms, ResidencyTransferTicket,
        ResidentBindingAlias, TransferDirection, UnitRetirement,
    },
    resources::{
        detached::{DetachedResourceLimits, DetachedResourceObservation, DetachedResourceScan},
        ResourceCoverage, ResourceDescription, ResourceIdentity,
        RESOURCE_DESCRIPTION_SCHEMA_VERSION,
    },
    BackendFailure, BackendFailureKind,
};
use eredu_nn::{ParameterStorageLayout, Tensor};

use crate::{CandleContext, CandleNeuralBackend, CandleTensor, ParameterReadError};

type Bindings = BTreeMap<String, CandleTensor>;
pub struct CandleWeightTransfer(Vec<ResidencyLease<CandleWeightCopy>>);
type Transfer = CandleWeightTransfer;

impl CandleWeightTransfer {
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl eredu_core::residency::ResidencyTransferOrdering<CandleContext> for CandleWeightTransfer {
    type Error = ParameterReadError;

    fn order_after(&self, _: &CandleContext) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn invalid(message: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidInput,
        std::io::Error::other(message),
    )
}

#[derive(Clone)]
pub struct CandleWeightCopy {
    tier: MemoryTier,
    weights: Arc<Bindings>,
}

impl CandleWeightCopy {
    fn lookup(&self, tier: MemoryTier, name: &str) -> Result<&CandleTensor, ParameterReadError> {
        if self.tier != tier {
            return Err(invalid("parameter residency tier mismatch").into());
        }
        self.weights
            .get(name)
            .ok_or_else(|| invalid("parameter residency binding is missing").into())
    }
}

impl ResidencyLeaseStorage for CandleWeightCopy {
    type DeviceValue = CandleTensor;
    type HostValue = CandleTensor;
    type Error = ParameterReadError;
    type BindingNames<'a> = std::iter::Map<
        std::collections::btree_map::Keys<'a, String, CandleTensor>,
        fn(&String) -> &str,
    >;

    fn device_value<'a>(
        &'a self,
        _: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a CandleTensor, Self::Error> {
        self.lookup(MemoryTier::Device, name)
    }

    fn host_value<'a>(
        &'a self,
        _: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a CandleTensor, Self::Error> {
        self.lookup(MemoryTier::Host, name)
    }

    fn binding_names(&self) -> Self::BindingNames<'_> {
        self.weights.keys().map(String::as_str)
    }
}

pub struct CandleWeightStorage {
    copies: BTreeMap<OffloadUnitId, [Option<CandleWeightCopy>; 2]>,
    context: CandleContext,
}

fn slot(tier: MemoryTier) -> Result<usize, ParameterReadError> {
    match tier {
        MemoryTier::Device => Ok(0),
        MemoryTier::Host => Ok(1),
        MemoryTier::Disk => Err(invalid("disk parameter tier has no native allocation").into()),
    }
}

impl ResidencyStorage for CandleWeightStorage {
    type Backend = CandleNeuralBackend;
    type Copy = CandleWeightCopy;
    type Transfer = std::convert::Infallible;
    type Error = ParameterReadError;

    fn prepare_storage(&mut self, id: &OffloadUnitId) {
        self.copies.entry(id.clone()).or_default();
    }

    fn retire_storage(&mut self, id: &OffloadUnitId) {
        self.copies.remove(id);
    }

    fn capacity(
        &self,
        _: &[WeightBinding],
        logical_bytes: u64,
        tier: MemoryTier,
    ) -> Result<u64, Self::Error> {
        slot(tier)?;
        Ok(logical_bytes)
    }

    fn resident_storage(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<Self::Copy, Self::Error> {
        let index = slot(tier)?;
        self.copies
            .get(id)
            .and_then(|unit| unit[index].clone())
            .ok_or_else(|| invalid("parameter copy is missing").into())
    }

    fn release_copy(&mut self, id: &OffloadUnitId, tier: MemoryTier) -> Result<(), Self::Error> {
        let index = slot(tier)?;
        self.copies
            .get_mut(id)
            .and_then(|unit| unit[index].take())
            .ok_or_else(|| invalid("parameter copy is missing"))?;
        Ok(())
    }

    type Realization<'a> = CandleWeightRead<'a>;

    fn begin_realization(&self, tier: MemoryTier) -> Result<Self::Realization<'_>, Self::Error> {
        slot(tier)?;
        Ok(CandleWeightRead {
            storage: self,
            context: self.context.clone(),
        })
    }

    fn retain_settlement(
        transfer: &Self::Transfer,
        _: Box<dyn ResidencyTransferTicket<Error = Self::Error>>,
    ) {
        match *transfer {}
    }

    fn install_copy(&mut self, id: OffloadUnitId, storage: Self::Copy) {
        let index = match storage.tier {
            MemoryTier::Device => 0,
            MemoryTier::Host => 1,
            MemoryTier::Disk => unreachable!("native copies are created in memory"),
        };
        self.copies.entry(id).or_default()[index] = Some(storage);
    }
}

pub struct CandleWeightRead<'a> {
    storage: &'a CandleWeightStorage,
    context: CandleContext,
}

impl ParameterReadScope for CandleWeightRead<'_> {
    type DirectRead = EncodedRecipeRead;
    type Value = CandleTensor;
    type Pending = Vec<CandleTensor>;
    type Error = ParameterReadError;

    fn slots(&self) -> NonZeroUsize {
        self.context.slots()
    }
    fn direct_read(
        &self,
        source: &dyn CheckpointSource,
        binding: &WeightBinding,
    ) -> Result<Option<Self::DirectRead>, Self::Error> {
        self.context.direct_read(source, binding)
    }
    fn submit_direct(
        &mut self,
        reads: Vec<(Self::DirectRead, ParameterStorageLayout)>,
    ) -> Result<Self::Pending, Self::Error> {
        self.context.submit_direct(reads)
    }
    fn submit_ordinary(
        &mut self,
        name: &str,
        input: ParameterReadInput<'_>,
    ) -> Result<Self::Pending, Self::Error> {
        self.context.submit_ordinary(name, input)
    }
    fn take_values(&mut self, pending: Self::Pending) -> Result<Vec<Self::Value>, Self::Error> {
        self.context.take_values(pending)
    }
    fn checkpoint_capacity_exhausted(error: &Self::Error) -> bool {
        CandleContext::checkpoint_capacity_exhausted(error)
    }
}

fn realized_copy(
    request: &ResidencyCopyRequest<'_>,
    tensors: Bindings,
    tier: MemoryTier,
    direction: TransferDirection,
) -> Result<RealizedResidencyCopy<CandleWeightCopy>, ParameterReadError> {
    let bytes = request
        .bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |bytes, binding| -> Result<_, ParameterReadError> {
            let tensor = tensors
                .get(binding.name())
                .ok_or_else(|| invalid("realized parameter is missing"))?;
            let size =
                tensor.value.elem_count() as u64 * tensor.value.dtype().size_in_bytes() as u64;
            if size != binding.expected_bytes() {
                return Err(invalid("realized parameter extent differs from the binding").into());
            }
            bytes
                .checked_add(size)
                .ok_or_else(|| invalid("parameter copy size overflow").into())
        })?;
    if bytes != request.logical_bytes || bytes > request.reserved_capacity {
        return Err(invalid("realized parameter copy exceeds its admitted extent").into());
    }
    Ok(RealizedResidencyCopy {
        id: request.id.clone(),
        storage: CandleWeightCopy {
            tier,
            weights: Arc::new(tensors),
        },
        capacity: bytes,
        bytes,
        direction,
    })
}

impl ResidencyCopyScope for CandleWeightRead<'_> {
    type HostValue = CandleTensor;
    type Copy = CandleWeightCopy;
    type Transfer = std::convert::Infallible;

    fn has_host_copy(&self, id: &OffloadUnitId) -> bool {
        self.storage
            .copies
            .get(id)
            .is_some_and(|unit| unit[1].is_some())
    }

    fn host_alias(&self, alias: &ResidentBindingAlias) -> Result<CandleTensor, Self::Error> {
        self.storage
            .resident_storage(&alias.owner, MemoryTier::Host)?
            .host_value(&alias.owner, &alias.binding)
            .cloned()
    }

    fn device_alias(&mut self, alias: &ResidentBindingAlias) -> Result<CandleTensor, Self::Error> {
        self.storage
            .resident_storage(&alias.owner, MemoryTier::Device)?
            .device_value(&alias.owner, &alias.binding)
            .cloned()
    }

    fn read_host(
        &mut self,
        _: &OffloadUnitId,
        binding: &WeightBinding,
        input: ParameterReadInput<'_>,
    ) -> Result<CandleTensor, Self::Error> {
        let mut host = CandleContext::cpu();
        host.submit_ordinary(binding.name(), input)?
            .pop()
            .ok_or_else(|| invalid("host parameter read produced no tensor").into())
    }

    fn copy_from_host(
        &mut self,
        id: &OffloadUnitId,
        names: &[&str],
    ) -> Result<Bindings, Self::Error> {
        let source = self.storage.resident_storage(id, MemoryTier::Host)?;
        names
            .iter()
            .map(|&name| {
                let host = source.host_value(id, name)?;
                let copy = if self.context.placement().is_cpu() {
                    host.compact(&self.context)?
                } else if let Some(packed) =
                    crate::quantized::upload_matrix(host, &self.context.device)?
                {
                    packed
                } else {
                    let mut copy = CandleTensor::wrap(host.value.to_device(&self.context.device))?;
                    copy.boolean = host.boolean;
                    copy
                };
                Ok((name.to_owned(), copy))
            })
            .collect()
    }

    fn host_owner(
        &self,
        request: &ResidencyCopyRequest<'_>,
        buffers: Bindings,
    ) -> Result<RealizedResidencyCopy<Self::Copy>, Self::Error> {
        realized_copy(
            request,
            buffers,
            MemoryTier::Host,
            TransferDirection::DiskToHost,
        )
    }

    fn device_owner(
        &self,
        request: &ResidencyCopyRequest<'_>,
        arrays: Bindings,
        direction: TransferDirection,
    ) -> Result<RealizedResidencyCopy<Self::Copy>, Self::Error> {
        realized_copy(request, arrays, MemoryTier::Device, direction)
    }

    fn finish_device(
        self,
        copies: &[RealizedResidencyCopy<Self::Copy>],
        _: bool,
    ) -> Result<Option<Self::Transfer>, Self::Error> {
        if let Err(error) = self.context.synchronize() {
            let storage = copies
                .iter()
                .map(|copy| copy.storage.clone())
                .collect::<Vec<_>>();
            self.context.quarantine(storage);
            return Err(error.into());
        }
        Ok(None)
    }
}

#[derive(Clone, Copy)]
pub struct CandleWeightResidency;

impl ResidencyStorageMechanisms for CandleWeightResidency {
    type Context = CandleContext;
    type Storage = CandleWeightStorage;
    type Transfer = Transfer;
    type Error = ParameterReadError;

    fn register_conversion_resources(
        _: &mut Self::Storage,
        _: &ConversionRetentionBudget,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn trim_conversion_resources(
        budget: &ConversionRetentionBudget,
    ) -> Result<ParameterConversionRetentionTrimReport, ConversionRetentionError> {
        budget.release_claims(&mut Vec::new())
    }
    fn open_storage(context: CandleContext) -> Result<(Self::Storage, Self), Self::Error> {
        Ok((
            CandleWeightStorage {
                copies: BTreeMap::new(),
                context,
            },
            Self,
        ))
    }
    fn progress(&self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn transfer(
        leases: Vec<ResidencyLease<CandleWeightCopy>>,
        submitted: Option<std::convert::Infallible>,
    ) -> Self::Transfer {
        if let Some(submission) = submitted {
            match submission {}
        }
        CandleWeightTransfer(leases)
    }
}

impl DenseResidencyMechanisms for CandleWeightResidency {
    fn transfer_stream_index(_: &Self::Storage) -> Result<i32, Self::Error> {
        Ok(0)
    }
    fn settle_transfer(_: &mut Transfer) -> Result<(), Self::Error> {
        Ok(())
    }
    fn allocator_memory(
        &self,
        storage: &Self::Storage,
    ) -> Result<eredu_core::AllocatorTelemetry, Self::Error> {
        Ok(crate::memory::telemetry(crate::memory::context_facts(
            &storage.context,
        )))
    }

    fn observe_residency(
        &self,
        _: &Self::Storage,
        _: ResidencyObservationBindings<'_>,
    ) -> Result<NativeResidencyObservations, Self::Error> {
        Ok(NativeResidencyObservations {
            host_storage: ResourceDescription {
                schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
                scope: ResourceIdentity {
                    scope: "candle".into(),
                    key: "host-parameters".into(),
                },
                context: Default::default(),
                horizon: Default::default(),
                coverage: ResourceCoverage::Partial {
                    reasons: vec!["Candle does not report backing allocations".into()],
                },
                allocations: Vec::new(),
            },
            conversions: Vec::new(),
            projection_storage: BTreeMap::new(),
            f32_normalization_gains: BTreeSet::new(),
        })
    }

    fn observe_detached(
        &self,
        _: DetachedResourceLimits,
    ) -> Result<DetachedResourceObservation, Self::Error> {
        Ok(DetachedResourceObservation {
            source: "candle".into(),
            inspected_nodes: 0,
            undescribed_nodes: 0,
            scan: DetachedResourceScan::Complete,
            descriptions: Vec::new(),
        })
    }
}

pub struct CandleWeightLease<U> {
    unit: U,
    transfer: Option<Transfer>,
}

impl<U> CandleWeightLease<U> {
    fn transfer(&self) -> Result<&Transfer, BackendFailure> {
        self.transfer
            .as_ref()
            .ok_or_else(|| invalid("parameter unit lease is pending native retirement"))
    }
}

impl<U> Deref for CandleWeightLease<U> {
    type Target = U;
    fn deref(&self) -> &U {
        &self.unit
    }
}

impl<U> DerefMut for CandleWeightLease<U> {
    fn deref_mut(&mut self) -> &mut U {
        &mut self.unit
    }
}

pub(crate) fn retained_weights(
    transfer: &Transfer,
    unit: usize,
) -> Result<Vec<(String, CandleTensor)>, ParameterReadError> {
    let lease = transfer
        .0
        .get(unit)
        .ok_or_else(|| invalid("parameter unit lease is missing"))?;
    lease
        .binding_names()
        .map(|name| Ok((name.to_owned(), lease.device_value(name)?.clone())))
        .collect()
}

pub struct CandleUnitOperations(CandleContext);

impl UnitRetirement for CandleUnitOperations {
    type Error = BackendFailure;
    type RetainedResource<T: 'static> = Box<T>;
    fn retain_resource<T: 'static>(resource: T) -> Self::RetainedResource<T> {
        Box::new(resource)
    }
}

impl NativeBoundedUnits for CandleUnitOperations {
    type Tensor = CandleTensor;
    type Context = CandleContext;
    type Parameters = CandleNeuralBackend;
    type Lease<U: 'static> = CandleWeightLease<U>;
    type Transfer = Transfer;

    fn reclaim(&mut self, _: bool) {
        CandleContext::reclaim_retired();
    }
    fn reclaim_retired(&mut self) {
        CandleContext::reclaim_retired();
    }
    fn token_width(&self, input: &CandleTensor) -> usize {
        input.shape().get(1).copied().unwrap_or_default() as usize
    }
    fn is_complete<U: 'static>(&self, lease: &Self::Lease<U>) -> Result<bool, Self::Error> {
        Ok(lease.transfer.is_some())
    }
    fn retire<U: 'static>(&mut self, _: Self::Lease<U>) -> Result<(), Self::Error> {
        Ok(())
    }
    fn bind_lease<U: 'static>(
        &mut self,
        unit: U,
        transfer: Transfer,
        _: &CandleContext,
    ) -> Result<Self::Lease<U>, Self::Error> {
        Ok(CandleWeightLease {
            unit,
            transfer: Some(transfer),
        })
    }
    fn unit_weights<U: 'static>(
        lease: &mut Self::Lease<U>,
    ) -> Result<Vec<(String, CandleTensor)>, Self::Error> {
        retained_weights(lease.transfer()?, 0).map_err(BackendFailure::from)
    }
    fn submit<'a, U: 'static>(
        &mut self,
        lease: &mut Self::Lease<U>,
        values: impl Iterator<Item = &'a CandleTensor>,
    ) -> Result<(), Self::Error> {
        if let Err(error) = self.0.synchronize() {
            self.0
                .quarantine((lease.transfer.take(), values.cloned().collect::<Vec<_>>()));
            return Err(error);
        }
        Ok(())
    }
    fn submit_inspection<U: 'static>(
        &mut self,
        lease: &mut Self::Lease<U>,
        _: &CandleContext,
    ) -> Result<(), Self::Error> {
        self.submit(lease, std::iter::empty())
    }

    fn finish_output(
        &mut self,
        value: &CandleTensor,
        queue: &CandleContext,
    ) -> Result<(), Self::Error> {
        if let Err(error) = queue.synchronize() {
            queue.quarantine(value.clone());
            return Err(error);
        }
        Ok(())
    }
}

impl NativeResidentUnits<CandleNeuralBackend, Transfer> for CandleUnitOperations {
    type Module<U: 'static> = Box<U>;
    type Parameters = CandleNeuralBackend;
    fn order_resident_transfer(
        transfer: &Transfer,
        _: &CandleContext,
    ) -> Result<usize, Self::Error> {
        Ok(transfer.0.len())
    }
    fn resident_weights(
        transfer: &Transfer,
        ordinal: usize,
    ) -> Result<Vec<(String, CandleTensor)>, Self::Error> {
        retained_weights(transfer, ordinal).map_err(BackendFailure::from)
    }
    fn retain_unit<U: 'static>(unit: U) -> Self::Module<U> {
        Box::new(unit)
    }
}

pub struct CandleResidentParameters;

impl ImmutableUnitStorage<CandleContext> for CandleResidentParameters {
    fn identity(&self, _: &CandleContext) -> Option<&ResourceIdentity> {
        None
    }
}

pub struct CandleUnitScope(CandleContext);

impl ComponentPreparationScope for CandleUnitScope {
    fn finish<T, E>(self, result: Result<T, E>) -> Result<T, E>
    where
        E: From<BackendFailure>,
    {
        self.0.synchronize().map_err(E::from)?;
        result
    }
}

impl ParameterStorageBinding<CandleNeuralBackend> for CandleContext {
    type UnitOperations = CandleUnitOperations;
    type ImmutableParameters = CandleResidentParameters;
    type ResidencyMechanisms = CandleWeightResidency;
    type Parameters = CandleNeuralBackend;
    fn unit_quantizer(&self) -> Result<Box<dyn BoundedQuantizationBackend + '_>, BackendFailure> {
        CandleNeuralBackend::quantizer(self)
    }
    fn unit_host_capacity(&self, bindings: &[WeightBinding]) -> Result<u64, BackendFailure> {
        physical_binding_bytes(bindings).map_err(BackendFailure::from_error)
    }
    fn unit_residency_context(&self, _: bool) -> Result<CandleContext, BackendFailure> {
        Ok(self.clone())
    }
    fn retain_parameter(value: &CandleTensor) -> CandleTensor {
        value.clone()
    }
    fn unit_operations(&self) -> CandleUnitOperations {
        CandleUnitOperations(self.clone())
    }
}

impl NativeParameterUnitBinding<CandleNeuralBackend> for CandleContext {
    type Scope = CandleUnitScope;
    type UnitError = BackendFailure;
    fn begin_unit_binding(&self) -> Result<Self::Scope, BackendFailure> {
        self.check_healthy()?;
        Ok(CandleUnitScope(self.clone()))
    }
    fn unit_executor(&self) -> &CandleContext {
        self
    }
    fn resident_parameter_certificate(&self) -> Result<CandleResidentParameters, BackendFailure> {
        Ok(CandleResidentParameters)
    }
}

impl From<ParameterReadError> for BackendFailure {
    fn from(error: ParameterReadError) -> Self {
        Self::from_error(error)
    }
}

impl eredu_core::residency::NativeBankMechanisms for CandleWeightResidency {
    type Backend = CandleNeuralBackend;
    type BankError = BackendFailure;
    type Loan = CandleParameterLoan;

    fn host_capacity_exhausted(error: &ParameterReadError) -> bool {
        matches!(
            error,
            ParameterReadError::Residency(
                eredu_core::residency::ResidencyLedgerError::BudgetExhausted {
                    tier: MemoryTier::Host,
                    ..
                }
            )
        )
    }

    fn member(loan: &Self::Loan, index: usize, name: &str) -> Result<CandleTensor, BackendFailure> {
        let lease = loan
            .transfer()
            .0
            .get(index)
            .ok_or_else(|| invalid("missing parameter bank lease"))?;
        lease
            .device_value(name)
            .cloned()
            .map_err(BackendFailure::from_error)
    }

    fn replacement(
        value: &CandleTensor,
        row: usize,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let row = i32::try_from(row).map_err(BackendFailure::from_error)?;
        let end = row
            .checked_add(1)
            .ok_or_else(|| invalid("parameter bank row exceeds i32"))?;
        value
            .index(&[eredu_nn::Index::Range(row, end)], context)
            .map_err(BackendFailure::from_error)
    }

    fn concatenate(
        values: &[CandleTensor],
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        CandleTensor::concatenate(values, 0, context).map_err(BackendFailure::from_error)
    }

    fn begin_loan(
        transfer: Transfer,
        context: &CandleContext,
    ) -> Result<Self::Loan, BackendFailure> {
        context.check_healthy()?;
        Ok(CandleParameterLoan(crate::device::CandleSettlement::new(
            context,
            (transfer, Vec::new()),
        )))
    }

    fn assemble_members(
        values: Vec<CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        Self::concatenate(&values, context)
    }
}

/// Owns ordered parameter leases and native roots through consumer settlement.
pub struct CandleParameterLoan(crate::device::CandleSettlement<(Transfer, Vec<CandleTensor>)>);

impl CandleParameterLoan {
    pub(crate) fn transfer(&self) -> &Transfer {
        &self.0.resources().0
    }

    pub(crate) fn check_queue(&self, context: &CandleContext) -> Result<(), BackendFailure> {
        if !self.0.queue.same_context(context) {
            return Err(invalid(
                "parameter loan belongs to a different Candle queue",
            ));
        }
        context.check_healthy()
    }
}

impl eredu_core::residency::ParameterLoan<CandleTensor, CandleContext> for CandleParameterLoan {
    type Error = BackendFailure;

    fn copy_binding(
        &mut self,
        name: &str,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        use eredu_core::residency::NativeBankMechanisms;
        self.check_queue(context)?;
        let value = CandleWeightResidency::member(self, 0, name)?
            .compact(context)
            .map_err(BackendFailure::from_error)?;
        self.retain(&value);
        Ok(value)
    }

    fn retain(&mut self, tensor: &CandleTensor) {
        self.0.resources_mut().1.push(tensor.clone());
    }

    fn finish(mut self, output: &CandleTensor) -> Result<(), BackendFailure> {
        self.retain(output);
        self.0.finish().map(drop)
    }
}

impl eredu_core::checkpoint_mechanisms::unit_binding::LocalComponentBinding<CandleNeuralBackend>
    for CandleContext
{
    fn local_executor(&self) -> CandleContext {
        self.clone()
    }
}

impl eredu_core::backend::LocalComponentStorage for CandleNeuralBackend {
    type CacheStorage = crate::CandleCacheStorage;

    fn settle_local_values<'a>(
        values: impl Iterator<Item = &'a CandleTensor>,
    ) -> Result<(), BackendFailure> {
        use eredu_core::backend::NativeTextValues as _;
        let mut settled = Ok(());
        for value in values {
            // Every borrowed value must settle or enter native retirement.
            settled = settled.and(crate::CandleTextValues::settle_input(value));
        }
        settled
    }
}

impl eredu_core::checkpoint_mechanisms::unit_binding::ComponentPreparation for CandleContext {
    type Context = CandleContext;
    type Scope = CandleUnitScope;

    fn component_context(&self) -> &CandleContext {
        self
    }

    fn begin_component_preparation(&self) -> Result<Self::Scope, BackendFailure> {
        self.begin_unit_binding()
    }
}
