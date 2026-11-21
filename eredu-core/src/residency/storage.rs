//! Exact inputs and terminal evidence for native residency copy mechanisms.

use super::{
    parameter_binding::WeightBinding, MemoryTier, OffloadUnitId, ResidencyLeaseStorage,
    ResidencyTransferTicket, TransferDirection,
};
use crate::checkpoint_mechanisms::CheckpointParameterBackend;

/// One prepared native copy and its measured storage, awaiting publication.
pub struct RealizedResidencyCopy<C> {
    /// Exact unit named by the reservation.
    pub id: OffloadUnitId,
    /// Native owner to install after publishing the reservation.
    pub storage: C,
    /// Actual physical storage capacity.
    pub capacity: u64,
    /// Logical bytes transferred.
    pub bytes: u64,
    /// Observed physical transfer direction.
    pub direction: TransferDirection,
}

/// Realization evidence for all missing copies in one batch.
pub struct RealizedResidency<C, T> {
    /// Copies produced in request order.
    pub copies: Vec<RealizedResidencyCopy<C>>,
    /// Pending exact native completion, if any.
    pub transfer: Option<T>,
}

/// One selected native copy with its exact source and allocation bound.
pub struct ResidencyCopyRequest<'a> {
    /// Unit receiving storage.
    pub id: &'a OffloadUnitId,
    /// Checkpoint data accessible to this unit.
    pub source: &'a dyn eredu_checkpoint::store::CheckpointSource,
    /// Recipes for the logical bindings.
    pub bindings: Vec<WeightBinding>,
    /// Canonical owner locations for logical aliases. External owners are already resident.
    pub aliases: Vec<ResidentBindingAlias>,
    /// Logical byte size established by admission.
    pub logical_bytes: u64,
    /// Physical storage capacity reserved for this copy.
    pub reserved_capacity: u64,
}

/// One selected binding borrowed from another resident unit.
pub struct ResidentBindingAlias {
    /// Local binding receiving the borrowed value.
    pub target: String,
    /// Unit holding the already resident owner.
    pub owner: OffloadUnitId,
    /// Owner's local binding name.
    pub binding: String,
}

/// Orders a consumer queue after the exact transfer that owns its source leases.
pub trait ResidencyTransferOrdering<Q: ?Sized> {
    /// Native dependency or completion failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Establishes a dependency on the consumer queue and retains it until safe.
    fn order_after(&self, queue: &Q) -> Result<(), Self::Error>;
}

/// Native storage operations for an admitted residency acquisition.
pub trait ResidencyStorage: Send + 'static {
    /// Native parameter materialization and cold recipe support.
    type Backend: CheckpointParameterBackend;
    /// One native storage owner.
    type Copy: ResidencyLeaseStorage<Error = Self::Error>;
    /// A submitted transfer retained through native completion.
    type Transfer;
    /// Native storage failure.
    type Error: std::error::Error
        + Send
        + Sync
        + From<crate::BackendFailure>
        + From<eredu_checkpoint::store::StoreError>
        + 'static;

    /// Ensures the native slots exist for a prepared unit.
    fn prepare_storage(&mut self, id: &OffloadUnitId);

    /// Removes native slots after both tiers release a transient row.
    fn retire_storage(&mut self, id: &OffloadUnitId);

    /// Returns the physical upper bound for one selected native copy.
    fn capacity(
        &self,
        bindings: &[WeightBinding],
        logical_bytes: u64,
        tier: MemoryTier,
    ) -> Result<u64, Self::Error>;

    /// Retains the native storage for one selected resident copy.
    fn resident_storage(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<Self::Copy, Self::Error>;

    /// Releases native storage for a copy already invalidated by the ledger.
    fn release_copy(&mut self, id: &OffloadUnitId, tier: MemoryTier) -> Result<(), Self::Error>;

    /// Native queue, conversion and recovery resources borrowed for one batch.
    type Realization<'a>: ResidencyCopyScope<
        Copy = Self::Copy,
        Transfer = Self::Transfer,
        Error = Self::Error,
    >
    where
        Self: 'a;

    /// Opens native resources for the selected storage tier.
    fn begin_realization(&self, tier: MemoryTier) -> Result<Self::Realization<'_>, Self::Error>;

    /// Retains the move-only settlement authority alongside the submitted resources.
    fn retain_settlement(
        transfer: &Self::Transfer,
        settlement: Box<dyn ResidencyTransferTicket<Error = Self::Error>>,
    );

    /// Installs the native owner once its ledger entry is published.
    fn install_copy(&mut self, id: OffloadUnitId, storage: Self::Copy);
}

/// Creates and retains native storage through transfers and recovery.
pub trait ResidencyStorageMechanisms: Sized {
    /// Exact checkpoint and transfer queue placement.
    type Context;
    /// Native storage and transfer primitives.
    type Storage: ResidencyStorage<Error = Self::Error>;
    /// Native transfer retaining the portable lease until settlement.
    type Transfer;
    /// Native storage, submission, or completion failure.
    type Error: std::error::Error
        + Send
        + Sync
        + 'static
        + From<crate::BackendFailure>
        + From<super::conversion_claim::ConversionRetentionError>
        + From<super::ResidencyLedgerError>;

    /// Registers permanent native parameter allocations with the chosen budget.
    fn register_conversion_resources(
        storage: &mut Self::Storage,
        budget: &super::conversion_claim::ConversionRetentionBudget,
    ) -> Result<(), Self::Error>;
    /// Detaches native conversion allocations at a settled boundary.
    fn trim_conversion_resources(
        budget: &super::conversion_claim::ConversionRetentionBudget,
    ) -> Result<
        super::ParameterConversionRetentionTrimReport,
        super::conversion_claim::ConversionRetentionError,
    >;
    /// Opens empty native storage and returns its native recovery resources.
    fn open_storage(context: Self::Context) -> Result<(Self::Storage, Self), Self::Error>;
    /// Progresses native work and checks whether further transfers are safe.
    fn progress(&self) -> Result<(), Self::Error>;
    /// Retains leases and exactly submitted native work together.
    fn transfer(
        leases: Vec<super::ResidencyLease<<Self::Storage as ResidencyStorage>::Copy>>,
        submitted: Option<<Self::Storage as ResidencyStorage>::Transfer>,
    ) -> Self::Transfer;
}

/// Native resources retained across the realization of one admitted residency batch.
///
/// A scope owns the native resources staged for aggregate completion, including
/// when preparation fails before the batch is handed to the residency ledger.
pub trait ResidencyCopyScope:
    crate::checkpoint_mechanisms::ParameterReadScope<Value: Clone>
{
    /// Exact host-buffer owner.
    type HostValue: Clone;
    /// Native storage assigned to the unit at publication.
    type Copy;
    /// Aggregate device completion and its retained resources.
    type Transfer;

    /// Reports whether an exact unit already has a host copy.
    fn has_host_copy(&self, id: &OffloadUnitId) -> bool;

    /// Retains one already installed host binding.
    fn host_alias(&self, alias: &ResidentBindingAlias) -> Result<Self::HostValue, Self::Error>;

    /// Retains one already installed device binding.
    fn device_alias(&mut self, alias: &ResidentBindingAlias) -> Result<Self::Value, Self::Error>;

    /// Converts the exact source input into one completed native host buffer.
    fn read_host(
        &mut self,
        id: &OffloadUnitId,
        binding: &WeightBinding,
        input: crate::checkpoint_mechanisms::ParameterReadInput<'_>,
    ) -> Result<Self::HostValue, Self::Error>;

    /// Submits the selected physical bindings of a retained host copy.
    fn copy_from_host(
        &mut self,
        id: &OffloadUnitId,
        names: &[&str],
    ) -> Result<std::collections::BTreeMap<String, Self::Value>, Self::Error>;

    /// Creates an immutable native owner from the completed host buffers.
    fn host_owner(
        &self,
        request: &ResidencyCopyRequest<'_>,
        buffers: std::collections::BTreeMap<String, Self::HostValue>,
    ) -> Result<RealizedResidencyCopy<Self::Copy>, Self::Error>;

    /// Creates a native owner from the selected device allocation roots.
    fn device_owner(
        &self,
        request: &ResidencyCopyRequest<'_>,
        arrays: std::collections::BTreeMap<String, Self::Value>,
        direction: TransferDirection,
    ) -> Result<RealizedResidencyCopy<Self::Copy>, Self::Error>;

    /// Submits aggregate completion, settling before return when requested.
    fn finish_device(
        self,
        copies: &[RealizedResidencyCopy<Self::Copy>],
        submit: bool,
    ) -> Result<Option<Self::Transfer>, Self::Error>;
}
