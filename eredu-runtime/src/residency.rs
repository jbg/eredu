//! Backend-neutral immutable-weight residency declarations and control state.

mod acquisition;
mod driver;
mod materialization;
pub use driver::ResidencyDriver;
mod conversion_retention;
mod prepared;
pub use acquisition::{
    ResidencyAccess, ResidencyOwner, ResidencySnapshot, ResidencyState, ResidencyStatePoisoned,
    ResidentLease,
};
mod rows;
pub use prepared::{
    PreparedResidency, PreparedResidencyError, ResidencyInstallation, ResidencySources,
};
pub use rows::RowResidencyRange;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Weak,
    time::Duration,
};

use eredu_checkpoint::{
    recipe::{RecipeCatalog, RecipeError},
    store::WeightStoreDiagnostics,
};
use eredu_core::residency::{
    conversion_claim::ConversionRetentionBudget, EvictedResidencyCopy, MemoryTier, OffloadPlan,
    OffloadReport, OffloadUnitId, PrefetchOutcome, ResidencyLedger, ResidencyLedgerError,
    UnitResidencyReport,
};

use eredu_checkpoint::quantization::WeightMaterializationReport;

use eredu_core::residency::parameter_binding::{
    physical_binding_bytes, OffloadUnit, ResidencyDeclarationError, WeightBinding,
};
pub use eredu_core::residency::{ResidentParameterConversion, ResidentParameterConversionBinding};

pub(crate) fn conversion_payload_bytes(
    conversions: &[ResidentParameterConversion],
) -> Result<u64, eredu_core::resources::ResourceDescriptionError> {
    use eredu_core::resources::{ResourceDescriptionError, ResourceSize};
    let invalid = |reason: &str| ResourceDescriptionError::Invalid(reason.into());
    let mut identities = BTreeSet::new();
    let mut bytes = 0u64;
    for conversion in conversions {
        conversion.allocation.validate()?;
        if conversion.bindings.is_empty() {
            return Err(invalid("cached conversion has no retaining bindings"));
        }
        for binding in &conversion.bindings {
            if binding.name.trim().is_empty()
                || binding
                    .logical_target
                    .as_ref()
                    .is_some_and(|name| name.trim().is_empty())
            {
                return Err(invalid("cached conversion has an empty binding identity"));
            }
            if !conversion.allocation.uses.iter().any(|usage| {
                usage.owner == binding.owner
                    && usage.role == eredu_core::resources::ResourceRole::Parameters
            }) {
                return Err(invalid(
                    "cached conversion binding has no matching parameter owner",
                ));
            }
        }
        if conversion.allocation.uses.iter().any(|usage| {
            usage.role != eredu_core::resources::ResourceRole::Parameters
                || !conversion
                    .bindings
                    .iter()
                    .any(|binding| binding.owner == usage.owner)
        }) {
            return Err(invalid(
                "cached conversion owner has no retaining parameter binding",
            ));
        }
        if !identities.insert(&conversion.allocation.identity) {
            return Err(invalid("cached conversion repeats a backing identity"));
        }
        let ResourceSize::Fixed { extent } = &conversion.allocation.size else {
            return Err(invalid(
                "cached conversion requires a fixed current payload",
            ));
        };
        if extent.payload.upper_bytes != Some(extent.payload.lower_bytes)
            || !matches!(
                extent.payload.kind,
                eredu_core::ObservationKind::Exact | eredu_core::ObservationKind::Observational
            )
        {
            return Err(invalid(
                "cached conversion requires an observed exact payload",
            ));
        }
        bytes = bytes
            .checked_add(extent.payload.lower_bytes)
            .ok_or_else(|| invalid("cached conversion payload total overflows"))?;
    }
    Ok(bytes)
}

/// Immutable residency-control and checkpoint-storage telemetry snapshot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyReport {
    initialized: bool,
    offload: OffloadReport,
    units: Vec<UnitResidencyReport>,
    active_window: Vec<OffloadUnitId>,
    weight_store: WeightStoreDiagnostics,
    unit_sources: BTreeMap<OffloadUnitId, WeightStoreDiagnostics>,
    materialization: Option<WeightMaterializationReport>,
    device_parameter_conversion_bytes: u64,
    device_parameter_conversions: Option<Vec<ResidentParameterConversion>>,
    host_storage_resources: Option<eredu_core::resources::ResourceDescription>,
    detached_resources: Option<crate::detached_resources::DetachedResourceInventory>,
    projection_storage: BTreeMap<String, eredu_core::residency::projection::ProjectionStorageFacts>,
    f32_rms_normalization_gains: std::collections::BTreeSet<String>,
}

impl ResidencyReport {
    /// Creates a report from one coherent controller and storage snapshot.
    pub fn new(
        initialized: bool,
        offload: OffloadReport,
        units: Vec<UnitResidencyReport>,
        active_window: Vec<OffloadUnitId>,
        weight_store: WeightStoreDiagnostics,
    ) -> Self {
        Self {
            initialized,
            offload,
            units,
            active_window,
            weight_store,
            unit_sources: BTreeMap::new(),
            materialization: None,
            device_parameter_conversion_bytes: 0,
            device_parameter_conversions: None,
            host_storage_resources: None,
            detached_resources: None,
            projection_storage: BTreeMap::new(),
            f32_rms_normalization_gains: Default::default(),
        }
    }

    /// Detached-work observation in its declared producer scope, which may span
    /// several managers. Missing data is not evidence of reclaimed storage.
    pub fn detached_resources(
        &self,
    ) -> Option<&crate::detached_resources::DetachedResourceInventory> {
        self.detached_resources.as_ref()
    }

    /// Retains an already validated, bounded detached-work observation without
    /// changing residency credits, completion state or the offload ledger.
    pub fn with_detached_resources(
        mut self,
        resources: crate::detached_resources::DetachedResourceInventory,
    ) -> Self {
        self.detached_resources = Some(resources);
        self
    }

    /// Point-in-time native host allocations retained by the residency cache.
    /// Missing observations are not zero storage. This does not include detached
    /// transfers, device arrays, source-store caches or materialization scratch.
    pub fn host_storage_resources(&self) -> Option<&eredu_core::resources::ResourceDescription> {
        self.host_storage_resources.as_ref()
    }

    /// Attaches native allocation identities and capacities without changing the
    /// logical offload ledger. The supplied scope and coverage must remain explicit.
    pub fn with_host_storage_resources(
        mut self,
        resources: eredu_core::resources::ResourceDescription,
    ) -> Result<Self, eredu_core::resources::ResourceDescriptionError> {
        resources.validate()?;
        self.host_storage_resources = Some(resources);
        Ok(self)
    }

    /// Storage coverage established by current permanent native parameter bindings.
    pub fn projection_storage(
        &self,
    ) -> &BTreeMap<String, eredu_core::residency::projection::ProjectionStorageFacts> {
        &self.projection_storage
    }
    /// Attaches allocation-free backend facts. Missing bindings remain uncovered.
    pub fn with_projection_storage(
        mut self,
        facts: BTreeMap<String, eredu_core::residency::projection::ProjectionStorageFacts>,
    ) -> Self {
        self.projection_storage = facts;
        self
    }

    /// Gains whose current binding guarantees F32 ungrouped learned RMS output.
    pub fn f32_rms_normalization_gains(&self) -> &std::collections::BTreeSet<String> {
        &self.f32_rms_normalization_gains
    }
    /// Attaches backend normalization dtype facts without native handles.
    pub fn with_f32_rms_normalization_gains(
        mut self,
        gains: std::collections::BTreeSet<String>,
    ) -> Self {
        self.f32_rms_normalization_gains = gains;
        self
    }

    /// Additional device payload retained as reusable parameter conversions.
    /// This is separate from the offload ledger's original parameter bytes.
    pub const fn device_parameter_conversion_bytes(&self) -> u64 {
        self.device_parameter_conversion_bytes
    }

    /// Attaches current reusable conversion residency without changing logical
    /// checkpoint sizes or the original parameter admission ledger.
    pub fn with_device_parameter_conversion_bytes(mut self, bytes: u64) -> Self {
        self.device_parameter_conversion_bytes = bytes;
        self.device_parameter_conversions = None;
        self
    }

    /// Point-in-time conversion allocations, or `None` when the producer only
    /// supplies aggregate bytes. An observed empty list means no cache is live.
    pub fn device_parameter_conversions(&self) -> Option<&[ResidentParameterConversion]> {
        self.device_parameter_conversions.as_deref()
    }

    /// Attaches observed allocations and derives their exact aggregate payload.
    /// Rejects repeated backing identities, unknown or varying current payload,
    /// and arithmetic overflow rather than turning lower bounds into totals.
    pub fn with_device_parameter_conversions(
        mut self,
        conversions: Vec<ResidentParameterConversion>,
    ) -> Result<Self, eredu_core::resources::ResourceDescriptionError> {
        self.device_parameter_conversion_bytes = conversion_payload_bytes(&conversions)?;
        self.device_parameter_conversions = Some(conversions);
        Ok(self)
    }

    /// Returns whether explicit initialization completed successfully.
    pub const fn initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the immutable offload telemetry snapshot.
    pub const fn offload(&self) -> &OffloadReport {
        &self.offload
    }

    /// Returns unit states in identifier order.
    pub fn units(&self) -> &[UnitResidencyReport] {
        &self.units
    }

    /// Returns the protected execution window in identifier order.
    pub fn active_window(&self) -> &[OffloadUnitId] {
        &self.active_window
    }

    /// Returns storage diagnostics, distinct from logical residency telemetry.
    pub const fn weight_store(&self) -> &WeightStoreDiagnostics {
        &self.weight_store
    }

    /// Diagnostics for units backed by distinct exact source views. Views may
    /// share physical storage or counters, so these reports are not additive.
    pub fn unit_sources(&self) -> &BTreeMap<OffloadUnitId, WeightStoreDiagnostics> {
        &self.unit_sources
    }

    /// Attaches source-view diagnostics without changing the shared residency ledger.
    pub fn with_unit_sources(
        mut self,
        sources: BTreeMap<OffloadUnitId, WeightStoreDiagnostics>,
    ) -> Self {
        self.unit_sources = sources;
        self
    }

    /// Returns bounded load-time materialization telemetry for these units.
    pub const fn materialization(&self) -> Option<&WeightMaterializationReport> {
        self.materialization.as_ref()
    }

    /// Attaches bounded load-time materialization telemetry.
    pub fn with_materialization(
        mut self,
        materialization: Option<WeightMaterializationReport>,
    ) -> Self {
        self.materialization = materialization;
        self
    }
}

/// Validated backend-neutral declarations paired with residency control state.
///
/// Concrete backends own their native values separately and mirror every
/// publication or eviction through this controller's ledger. Keeping the
/// declarations here ensures checkpoint shape policy and plan identity are
/// validated before any backend allocation begins.
#[derive(Debug)]
pub struct ResidencyController {
    conversion_retention: Option<ConversionRetentionBudget>,
    ledger: ResidencyLedger,
    units: BTreeMap<OffloadUnitId, OffloadUnit>,
    row_ranges: Vec<RowResidencyRange>,
    alias_owners: BTreeMap<(OffloadUnitId, String), (OffloadUnitId, String)>,
    alias_owner_pins: BTreeSet<(OffloadUnitId, MemoryTier)>,
}

/// One validated immutable-weight acquisition batch and its initial hit/miss state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyAcquisition {
    ids: Vec<OffloadUnitId>,
    missing: Vec<bool>,
}

/// Backend hook used by the neutral exact-transfer ownership lifecycle.
pub trait ResidencyTransferOwner<C, R>: Sized {
    /// Backend executor on which a dependent submission can be ordered.
    type Executor: ?Sized;
    /// Concrete completion or publication failure.
    type Error;

    /// Orders an executor after one exact transfer without blocking the host.
    fn order_after(
        completion: &C,
        executor: &Self::Executor,
        id: &OffloadUnitId,
    ) -> Result<(), Self::Error>;

    /// Observes exact completion without blocking.
    fn is_complete(completion: &C, id: &OffloadUnitId) -> Result<bool, Self::Error>;

    /// Waits for this exact transfer only.
    fn wait(completion: &C, id: &OffloadUnitId) -> Result<(), Self::Error>;

    /// Releases retained backend resources according to transfer success.
    fn finish_resources(resources: R, succeeded: bool);

    /// Publishes or rolls back the exact transfer generation.
    fn resolve_transfer(
        &self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
        generation: u64,
        succeeded: bool,
    ) -> Result<(), Self::Error>;
}

/// Caller-owned exact transfer and source-lifetime guard.
pub struct ResidencyTransfer<L, C, R, O>
where
    O: ResidencyTransferOwner<C, R>,
{
    leases: Vec<L>,
    completion: Option<C>,
    resources: Option<R>,
    owner: Weak<O>,
    ids: Vec<OffloadUnitId>,
    tier: MemoryTier,
    generation: u64,
}

impl<L, C, R, O> ResidencyTransfer<L, C, R, O>
where
    O: ResidencyTransferOwner<C, R>,
{
    /// Creates an already-complete transfer containing only resident leases.
    pub fn immediate(leases: Vec<L>, tier: MemoryTier) -> Self {
        Self {
            leases,
            completion: None,
            resources: None,
            owner: Weak::new(),
            ids: Vec::new(),
            tier,
            generation: 0,
        }
    }

    /// Creates an in-flight transfer owning its exact completion and retained resources.
    pub fn submitted(
        leases: Vec<L>,
        completion: C,
        resources: R,
        owner: Weak<O>,
        ids: Vec<OffloadUnitId>,
        tier: MemoryTier,
        generation: u64,
    ) -> Self {
        assert!(
            !ids.is_empty(),
            "an in-flight residency transfer must contain at least one unit"
        );
        Self {
            leases,
            completion: Some(completion),
            resources: Some(resources),
            owner,
            ids,
            tier,
            generation,
        }
    }

    /// Returns resident unit leases protected by this transfer.
    pub fn leases(&self) -> &[L] {
        &self.leases
    }

    /// Returns whether the transfer contains no resident units.
    pub fn is_empty(&self) -> bool {
        self.leases.is_empty()
    }

    /// Orders dependent backend work after this exact transfer.
    pub fn order_after(&self, executor: &O::Executor) -> Result<(), O::Error> {
        match &self.completion {
            Some(completion) => O::order_after(completion, executor, self.primary_id()),
            None => Ok(()),
        }
    }

    /// Returns whether this exact transfer completed without blocking.
    pub fn is_complete(&self) -> Result<bool, O::Error> {
        match &self.completion {
            Some(completion) => O::is_complete(completion, self.primary_id()),
            None => Ok(true),
        }
    }

    /// Waits for exact completion and publishes or rolls back the transfer.
    pub fn synchronize(&mut self) -> Result<(), O::Error> {
        self.finish(true)
    }

    fn primary_id(&self) -> &OffloadUnitId {
        self.ids
            .first()
            .expect("an in-flight residency transfer has at least one unit")
    }

    fn finish(&mut self, report_error: bool) -> Result<(), O::Error> {
        let result = match &self.completion {
            Some(completion) => O::wait(completion, self.primary_id()),
            None => Ok(()),
        };
        let succeeded = result.is_ok();
        if let Some(resources) = self.resources.take() {
            O::finish_resources(resources, succeeded);
        }
        if succeeded {
            self.completion = None;
        }
        self.publish(succeeded)?;
        match result {
            Ok(()) => Ok(()),
            Err(error) if report_error => Err(error),
            Err(_) => Ok(()),
        }
    }

    fn publish(&mut self, succeeded: bool) -> Result<(), O::Error> {
        if self.generation == 0 {
            return Ok(());
        }
        let generation = std::mem::take(&mut self.generation);
        if let Some(owner) = self.owner.upgrade() {
            owner.resolve_transfer(&self.ids, self.tier, generation, succeeded)?;
        }
        Ok(())
    }
}

impl<L, C, R, O> Drop for ResidencyTransfer<L, C, R, O>
where
    O: ResidencyTransferOwner<C, R>,
{
    fn drop(&mut self) {
        let _ = self.finish(false);
    }
}

impl ResidencyAcquisition {
    /// Returns requested units in caller order.
    pub fn ids(&self) -> &[OffloadUnitId] {
        &self.ids
    }

    /// Returns one flag per requested unit; `true` requires backend realization.
    pub fn missing(&self) -> &[bool] {
        &self.missing
    }

    /// Returns whether every requested copy was already resident.
    pub fn is_hit(&self) -> bool {
        self.missing.iter().all(|missing| !missing)
    }

    /// Returns missing units in caller order.
    pub fn missing_ids(&self) -> impl Iterator<Item = &OffloadUnitId> {
        self.ids
            .iter()
            .zip(&self.missing)
            .filter_map(|(id, &missing)| missing.then_some(id))
    }

    fn temporary_protection(&self) -> BTreeSet<OffloadUnitId> {
        self.ids.iter().cloned().collect()
    }
}

impl ResidencyController {
    /// Attaches the selected execution's shared conversion budget before use.
    /// Every partition and embedded owner must receive a clone of this same
    /// budget; named layer windows never create an independent allowance.
    /// Existing controllers remain unwired unless composition supplies a budget.
    pub fn with_conversion_retention(mut self, budget: ConversionRetentionBudget) -> Self {
        self.conversion_retention = Some(budget);
        self
    }

    /// Selected execution's conversion admission authority, when installed.
    pub fn conversion_retention(&self) -> Option<&ConversionRetentionBudget> {
        self.conversion_retention.as_ref()
    }

    /// Validates declarations against checkpoint metadata and an explicit plan.
    pub fn new<C: RecipeCatalog + ?Sized>(
        catalog: &C,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
    ) -> Result<Self, ResidencyControllerError> {
        Self::new_with_catalogs(|_| catalog, plan, units)
    }

    /// Validates each physical unit against its own retained source catalog,
    /// while all units consume one residency plan and reservation ledger.
    /// Identical source keys in different catalogs need not name the same value.
    pub fn new_with_catalogs<'a, C: RecipeCatalog + ?Sized + 'a>(
        catalog_for_unit: impl Fn(&OffloadUnitId) -> &'a C,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
    ) -> Result<Self, ResidencyControllerError> {
        Self::new_with_row_ranges(catalog_for_unit, plan, units, Vec::new())
    }

    /// Combines explicit units with compact row namespaces over retained sources.
    pub fn new_with_row_ranges<'a, C: RecipeCatalog + ?Sized + 'a>(
        catalog_for_unit: impl Fn(&OffloadUnitId) -> &'a C,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        mut row_ranges: Vec<RowResidencyRange>,
    ) -> Result<Self, ResidencyControllerError> {
        row_ranges.sort_by(|a, b| {
            (a.range().prefix(), a.range().start()).cmp(&(b.range().prefix(), b.range().start()))
        });
        if row_ranges
            .iter()
            .map(RowResidencyRange::range)
            .ne(plan.ranges().iter())
        {
            return Err(ResidencyControllerError::RangeCatalogMismatch);
        }
        let mut definitions = BTreeMap::new();
        for unit in units {
            let id = unit.id().clone();
            if definitions.insert(id.clone(), unit).is_some() {
                return Err(ResidencyControllerError::DuplicateUnitDefinition { id });
            }
        }
        for spec in plan.units() {
            if !definitions.contains_key(spec.id()) {
                return Err(ResidencyControllerError::MissingUnitDefinition {
                    id: spec.id().clone(),
                });
            }
        }
        if let Some(id) = definitions
            .keys()
            .find(|id| plan.unit(id).is_none())
            .cloned()
        {
            return Err(ResidencyControllerError::UnexpectedUnitDefinition { id });
        }

        let alias_owners = validate_global_binding_aliases(&definitions)?;

        for spec in plan.units() {
            let unit = definitions
                .get(spec.id())
                .expect("definition identity validated above");
            let catalog = catalog_for_unit(unit.id());
            let total = physical_binding_bytes(unit.bindings())?;
            for binding in unit.bindings().iter().filter(|binding| !binding.is_alias()) {
                let actual = binding
                    .recipe_view()
                    .infer(catalog)
                    .map_err(|source| ResidencyControllerError::Recipe {
                        binding: binding.name().to_owned(),
                        source,
                    })?
                    .byte_len();
                if actual != binding.expected_bytes() {
                    return Err(ResidencyControllerError::BindingByteMismatch {
                        id: unit.id().clone(),
                        binding: binding.name().to_owned(),
                        expected_bytes: binding.expected_bytes(),
                        actual_bytes: actual,
                    });
                }
            }
            if total != spec.bytes() {
                return Err(ResidencyControllerError::UnitByteMismatch {
                    id: unit.id().clone(),
                    planned_bytes: spec.bytes(),
                    actual_bytes: total,
                });
            }
        }

        Ok(Self {
            conversion_retention: None,
            ledger: ResidencyLedger::new(plan),
            units: definitions,
            row_ranges,
            alias_owners,
            alias_owner_pins: BTreeSet::new(),
        })
    }

    /// Returns the validated declaration for one planned unit.
    pub fn unit(&self, id: &OffloadUnitId) -> Option<&OffloadUnit> {
        self.units.get(id)
    }

    /// Prepares only demanded declarations before a native acquisition. Failed
    /// planning publishes none of the new bindings and performs no payload I/O.
    pub fn prepare_units(&mut self, ids: &[OffloadUnitId]) -> Result<(), ResidencyControllerError> {
        let mut prepared = BTreeMap::new();
        for id in ids {
            if self.units.contains_key(id) || prepared.contains_key(id) {
                continue;
            }
            let range = self
                .row_ranges
                .iter()
                .find(|range| range.range().member(id).is_some())
                .ok_or_else(|| ResidencyControllerError::MissingUnitDefinition {
                    id: id.clone(),
                })?;
            let unit =
                range
                    .unit(id)?
                    .ok_or_else(|| ResidencyControllerError::MissingUnitDefinition {
                        id: id.clone(),
                    })?;
            prepared.insert(id.clone(), unit);
        }
        self.units.extend(prepared);
        Ok(())
    }

    /// Releases transient row declarations once neither tier has a live copy.
    /// Explicit model units keep their original immutable catalog entries.
    pub fn retire_unit_if_unused(
        &mut self,
        id: &OffloadUnitId,
    ) -> Result<bool, ResidencyLedgerError> {
        if !self
            .row_ranges
            .iter()
            .any(|range| range.range().member(id).is_some())
        {
            return Ok(false);
        }
        if !self.ledger.has_allocation(id, MemoryTier::Host)?
            && !self.ledger.has_allocation(id, MemoryTier::Device)?
        {
            self.units.remove(id);
            return Ok(true);
        }
        Ok(false)
    }

    /// Returns declarations in stable unit-identifier order.
    pub fn units(&self) -> impl ExactSizeIterator<Item = &OffloadUnit> {
        self.units.values()
    }

    /// Supplies the exact prepared declaration and retention identities for native observation.
    pub(crate) fn observation_bindings(
        &self,
    ) -> eredu_core::residency::ResidencyObservationBindings<'_> {
        let retention = self.conversion_retention();
        eredu_core::residency::ResidencyObservationBindings {
            units: &self.units,
            retention_group: retention.map(|budget| budget.group()),
            projection_storage_eligible: retention.is_some_and(|budget| {
                budget.policy().eligibility
                    == eredu_core::residency::ParameterConversionRetentionEligibility::Eligible
            }),
        }
    }

    /// Resolves a logical alias to its canonical owner unit and binding.
    pub fn binding_owner(
        &self,
        unit: &OffloadUnitId,
        binding: &WeightBinding,
    ) -> Option<(&OffloadUnitId, &WeightBinding)> {
        let (owner_unit, owner_name) = self
            .alias_owners
            .get(&(unit.clone(), binding.name().to_owned()))?;
        let owner = self.units.get(owner_unit)?;
        let binding = owner
            .bindings()
            .iter()
            .find(|binding| binding.name() == owner_name)?;
        Some((owner_unit, binding))
    }

    /// Returns the canonical owner location when a binding participates in a
    /// shared alias family, including the canonical owner itself.
    pub fn shared_binding_owner(
        &self,
        unit: &OffloadUnitId,
        binding: &WeightBinding,
    ) -> Option<(&OffloadUnitId, &WeightBinding)> {
        if binding.is_alias() {
            return self.binding_owner(unit, binding);
        }
        let location = (unit.clone(), binding.name().to_owned());
        if !self.alias_owners.values().any(|owner| owner == &location) {
            return None;
        }
        let (owner_unit, owner) = self.units.get_key_value(unit)?;
        let owner = owner
            .bindings()
            .iter()
            .find(|candidate| candidate.name() == binding.name())?;
        Some((owner_unit, owner))
    }

    /// Returns immutable ownership, capacity, and telemetry state.
    pub const fn ledger(&self) -> &ResidencyLedger {
        &self.ledger
    }

    /// Returns mutable ownership, capacity, and telemetry state.
    pub fn ledger_mut(&mut self) -> &mut ResidencyLedger {
        &mut self.ledger
    }

    /// Validates one batch and snapshots which requested copies need realization.
    pub fn plan_acquisition(
        &mut self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<ResidencyAcquisition, ResidencyLedgerError> {
        self.ledger.require_initialized()?;
        self.plan_acquisition_inner(ids, tier)
    }

    /// Validates a batch while the manager is realizing its initial planned tiers.
    pub fn plan_initialization_acquisition(
        &mut self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<ResidencyAcquisition, ResidencyLedgerError> {
        self.plan_acquisition_inner(ids, tier)
    }

    fn plan_acquisition_inner(
        &mut self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<ResidencyAcquisition, ResidencyLedgerError> {
        self.ledger.validate_batch(ids, tier)?;
        let missing = ids
            .iter()
            .map(|id| self.ledger.is_resident(id, tier).map(|resident| !resident))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ResidencyAcquisition {
            ids: ids.to_vec(),
            missing,
        })
    }

    /// Reserves backend-supplied physical capacities while protecting the complete batch.
    pub fn reserve_acquisition(
        &mut self,
        acquisition: &ResidencyAcquisition,
        reservations: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Vec<EvictedResidencyCopy>, ResidencyLedgerError> {
        self.ledger
            .reserve_copies(reservations, tier, &acquisition.temporary_protection())
    }

    /// Updates recency for copies which were hits when the batch began.
    pub fn touch_acquisition_hits(
        &mut self,
        acquisition: &ResidencyAcquisition,
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        for (id, &missing) in acquisition.ids.iter().zip(&acquisition.missing) {
            if !missing {
                self.ledger.touch(id, tier)?;
            }
        }
        Ok(())
    }

    /// Rolls back every missing copy which remains an unpublished reservation.
    pub fn rollback_acquisition(
        &mut self,
        acquisition: &ResidencyAcquisition,
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        for id in acquisition.missing_ids() {
            self.ledger.rollback_reserved(id, tier)?;
        }
        Ok(())
    }

    /// Publishes one realized copy and records its backend transfer observation.
    #[allow(clippy::too_many_arguments)]
    pub fn publish_acquisition_copy(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        actual_bytes: u64,
        transferred_bytes: u64,
        transfer_generation: Option<u64>,
        direction: eredu_core::residency::TransferDirection,
        duration: Duration,
    ) -> Result<(), ResidencyLedgerError> {
        self.ledger
            .publish_reserved(id, tier, actual_bytes, transfer_generation)?;
        self.ledger
            .record_transfer(direction, transferred_bytes, duration);
        Ok(())
    }

    /// Records a prefetch hit or miss before backend realization begins.
    pub fn begin_prefetch(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<PrefetchOutcome, ResidencyLedgerError> {
        self.ledger.require_initialized()?;
        let outcome = if self.ledger.is_resident(id, tier)? {
            PrefetchOutcome::Hit
        } else {
            PrefetchOutcome::Miss
        };
        self.ledger.record_prefetch(tier, outcome);
        Ok(outcome)
    }

    /// Resolves one exact transfer and returns backend copies invalidated by failure.
    pub fn resolve_transfer(
        &mut self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
        generation: u64,
        succeeded: bool,
    ) -> Result<Vec<EvictedResidencyCopy>, ResidencyLedgerError> {
        self.ledger
            .resolve_transfer(ids, tier, generation, succeeded)
    }

    /// Replaces one protected window and selects unique bounded lookahead in caller order.
    ///
    /// A concrete backend calls this after any in-flight copies touching the requested
    /// units have reached a stable state.
    pub fn commit_group_window(
        &mut self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<OffloadUnitId>, eredu_core::residency::ResidencyLedgerError> {
        self.ledger.require_initialized()?;
        for id in active.iter().chain(upcoming) {
            self.ledger.spec(id)?;
        }
        self.ledger.set_group_window(group, active, tier)?;
        let depth = self.ledger.plan().config().prefetch_depth();
        let mut seen = BTreeSet::new();
        Ok(upcoming
            .iter()
            .filter(|id| seen.insert((*id).clone()))
            .take(depth)
            .cloned()
            .collect())
    }

    /// Replaces one protected window without selecting or materializing lookahead.
    pub fn protect_group_window(
        &mut self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), eredu_core::residency::ResidencyLedgerError> {
        self.commit_group_window(group, active, &[], tier)
            .map(|_| ())
    }
}

type BindingLocation = (OffloadUnitId, String);
type BindingAliasMap = BTreeMap<BindingLocation, BindingLocation>;

fn validate_global_binding_aliases(
    units: &BTreeMap<OffloadUnitId, OffloadUnit>,
) -> Result<BindingAliasMap, ResidencyControllerError> {
    let mut identities = BTreeMap::<String, Vec<BindingLocation>>::new();
    let mut aliases = BTreeMap::<BindingLocation, String>::new();
    let mut bytes = BTreeMap::<BindingLocation, u64>::new();
    for (unit_id, unit) in units {
        for binding in unit.bindings() {
            let location = (unit_id.clone(), binding.name().to_owned());
            let identity = binding
                .logical_target()
                .unwrap_or(binding.name())
                .to_owned();
            identities
                .entry(identity)
                .or_default()
                .push(location.clone());
            bytes.insert(location.clone(), binding.expected_bytes());
            if let Some(owner) = binding.alias_of() {
                aliases.insert(location, owner.to_owned());
            }
        }
    }

    fn resolve(
        location: &BindingLocation,
        identities: &BTreeMap<String, Vec<BindingLocation>>,
        aliases: &BTreeMap<BindingLocation, String>,
        visiting: &mut BTreeSet<BindingLocation>,
    ) -> Result<BindingLocation, ResidencyControllerError> {
        let Some(destination) = aliases.get(location) else {
            return Ok(location.clone());
        };
        if !visiting.insert(location.clone()) {
            return Err(ResidencyControllerError::Declaration(
                ResidencyDeclarationError::BindingAliasCycle {
                    name: location.1.clone(),
                },
            ));
        }
        let candidates = identities.get(destination).ok_or_else(|| {
            ResidencyControllerError::Declaration(
                ResidencyDeclarationError::UnknownBindingAliasOwner {
                    alias: location.1.clone(),
                    owner: destination.clone(),
                },
            )
        })?;
        if candidates.len() != 1 {
            return Err(ResidencyControllerError::Declaration(
                ResidencyDeclarationError::AmbiguousBindingAliasOwner {
                    alias: location.1.clone(),
                    owner: destination.clone(),
                },
            ));
        }
        let owner = resolve(&candidates[0], identities, aliases, visiting)?;
        visiting.remove(location);
        Ok(owner)
    }

    let mut resolved = BTreeMap::new();
    for alias in aliases.keys() {
        let owner = resolve(alias, &identities, &aliases, &mut BTreeSet::new())?;
        let alias_bytes = bytes[alias];
        let owner_bytes = bytes[&owner];
        if alias_bytes != owner_bytes {
            return Err(ResidencyControllerError::Declaration(
                ResidencyDeclarationError::BindingAliasByteMismatch {
                    alias: alias.1.clone(),
                    owner: owner.1.clone(),
                    alias_bytes,
                    owner_bytes,
                },
            ));
        }
        resolved.insert(alias.clone(), owner);
    }
    Ok(resolved)
}

/// Failure while validating a residency control plane.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyControllerError {
    /// Compact declarations differ from the admitted offload plan.
    #[error("row residency catalog differs from admitted compact ranges")]
    RangeCatalogMismatch,
    /// Matrix extent or encoded per-row bytes differ from the declared range.
    #[error("row residency geometry differs for {prefix}")]
    RowRangeGeometry {
        /// Invalid namespace.
        prefix: OffloadUnitId,
    },
    /// Bounded row planning failed before native materialization.
    #[error(transparent)]
    RowRead(#[from] eredu_checkpoint::rows::RowReadError),
    /// Binding declarations, alias topology or physical byte accounting are invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
    /// More than one definition used the same plan identifier.
    #[error("duplicate residency unit definition: {id}")]
    DuplicateUnitDefinition {
        /// Duplicated identifier.
        id: OffloadUnitId,
    },
    /// The plan had no matching unit definition.
    #[error("offload plan unit {id} has no residency unit definition")]
    MissingUnitDefinition {
        /// Missing identifier.
        id: OffloadUnitId,
    },
    /// A definition had no matching plan entry.
    #[error("residency unit {id} is absent from the offload plan")]
    UnexpectedUnitDefinition {
        /// Unexpected identifier.
        id: OffloadUnitId,
    },
    /// Binding sizes did not sum to the plan's unit size.
    #[error(
        "residency unit {id} defines {actual_bytes} bytes but its plan reserves {planned_bytes}"
    )]
    UnitByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Bytes reserved by the plan.
        planned_bytes: u64,
        /// Sum of binding sizes.
        actual_bytes: u64,
    },
    /// A binding's selected checkpoint size contradicted its declaration.
    #[error(
        "binding {binding:?} in unit {id} selects {actual_bytes} bytes but declares {expected_bytes}"
    )]
    BindingByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Binding name.
        binding: String,
        /// Declared size.
        expected_bytes: u64,
        /// Catalog-validated size.
        actual_bytes: u64,
    },
    /// A derived-weight recipe was invalid.
    #[error("derived-weight recipe for binding {binding:?} failed: {source}")]
    Recipe {
        /// Local binding name.
        binding: String,
        /// Invalid recipe.
        #[source]
        source: RecipeError,
    },
}

/// Backend-independent operations required by ordered residency windows.
pub trait ResidencyWindowManager {
    /// Manager-specific failure including neutral window validation failures.
    type Error: std::error::Error + From<eredu_core::BackendFailure>;

    /// Replaces the default protected window and prepares bounded lookahead.
    fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error>;

    /// Replaces one named protected window and prepares bounded lookahead.
    fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error>;

    /// Removes one concrete resident copy if present.
    fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, Self::Error>;

    /// Returns logical unit state in stable identifier order.
    fn unit_reports(&self) -> Result<Vec<UnitResidencyReport>, Self::Error>;
}

/// Deterministic controller for a bounded ordered device-layer window.
#[derive(Debug, Clone)]
pub struct DeviceLayerWindow {
    units: Vec<OffloadUnitId>,
    depth: usize,
}

impl DeviceLayerWindow {
    /// Creates a controller for a non-empty, duplicate-free unit sequence.
    pub fn new(
        units: impl IntoIterator<Item = OffloadUnitId>,
        depth: usize,
    ) -> Result<Self, ResidencyWindowError> {
        let units = units.into_iter().collect::<Vec<_>>();
        if units.is_empty() {
            return Err(ResidencyWindowError::EmptyLayerWindow);
        }
        if depth == 0 || depth > units.len() {
            return Err(ResidencyWindowError::OversizedLayerWindow {
                depth,
                layer_count: units.len(),
            });
        }
        let unique = units.iter().collect::<BTreeSet<_>>();
        if unique.len() != units.len() {
            return Err(ResidencyWindowError::DuplicateLayerWindowUnit {
                id: units
                    .iter()
                    .find(|id| units.iter().filter(|candidate| *candidate == *id).count() > 1)
                    .expect("duplicate exists")
                    .clone(),
            });
        }
        Ok(Self { units, depth })
    }

    /// Returns the maximum number of ordered units kept on the device.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns units in execution order.
    pub fn units(&self) -> &[OffloadUnitId] {
        &self.units
    }

    /// Returns the desired window beginning at `current`.
    pub fn desired(&self, current: usize) -> Result<&[OffloadUnitId], ResidencyWindowError> {
        if current >= self.units.len() {
            return Err(ResidencyWindowError::InvalidLayerIndex {
                index: current,
                layer_count: self.units.len(),
            });
        }
        let end = current.saturating_add(self.depth).min(self.units.len());
        Ok(&self.units[current..end])
    }

    /// Prepares and trims the default window beginning at `current`.
    pub fn prepare<M: ResidencyWindowManager>(
        &self,
        manager: &M,
        current: usize,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, M::Error> {
        let desired = self
            .desired(current)
            .map_err(eredu_core::BackendFailure::from_error)?;
        let outcomes = manager.prepare_window(desired, desired, MemoryTier::Device)?;
        self.trim_to(manager, desired)?;
        Ok(outcomes)
    }

    /// Explicitly evicts every managed device copy outside `desired`.
    pub fn trim_to<M: ResidencyWindowManager>(
        &self,
        manager: &M,
        desired: &[OffloadUnitId],
    ) -> Result<(), M::Error> {
        let desired = desired.iter().collect::<BTreeSet<_>>();
        for id in &self.units {
            if !desired.contains(id) {
                manager.evict(id, MemoryTier::Device)?;
            }
        }
        Ok(())
    }

    /// Clears protection and removes every managed device copy.
    pub fn clear<M: ResidencyWindowManager>(&self, manager: &M) -> Result<(), M::Error> {
        manager.prepare_window(&[], &[], MemoryTier::Device)?;
        self.trim_to(manager, &[])
    }
}

/// A named sequential execution stack with an independent device window.
#[derive(Debug, Clone)]
pub struct ResidentLayerGroup {
    id: String,
    window: DeviceLayerWindow,
}

impl ResidentLayerGroup {
    /// Creates a named group over ordered residency units.
    pub fn new(
        id: impl Into<String>,
        units: impl IntoIterator<Item = OffloadUnitId>,
        depth: usize,
    ) -> Result<Self, ResidencyWindowError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ResidencyWindowError::InvalidGroupId);
        }
        Ok(Self {
            id,
            window: DeviceLayerWindow::new(units, depth)?,
        })
    }

    /// Returns the stable group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns ordered units in this group.
    pub fn units(&self) -> &[OffloadUnitId] {
        self.window.units()
    }

    /// Returns the configured device-unit bound.
    pub const fn depth(&self) -> usize {
        self.window.depth()
    }

    /// Prepares this group's window without replacing another group's window.
    pub fn prepare<M: ResidencyWindowManager>(
        &self,
        manager: &M,
        current: usize,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, M::Error> {
        let desired = self
            .window
            .desired(current)
            .map_err(eredu_core::BackendFailure::from_error)?;
        let outcomes =
            manager.prepare_group_window(&self.id, desired, desired, MemoryTier::Device)?;
        self.window.trim_to(manager, desired)?;
        Ok(outcomes)
    }

    /// Trims this group to the desired window.
    pub fn trim_to<M: ResidencyWindowManager>(
        &self,
        manager: &M,
        desired: &[OffloadUnitId],
    ) -> Result<(), M::Error> {
        self.window.trim_to(manager, desired)
    }

    /// Clears only this group's protection and device copies.
    pub fn clear<M: ResidencyWindowManager>(&self, manager: &M) -> Result<(), M::Error> {
        manager.prepare_group_window(&self.id, &[], &[], MemoryTier::Device)?;
        self.window.trim_to(manager, &[])
    }

    /// Returns current logical residency attributed to this group's units.
    pub fn report<M: ResidencyWindowManager>(
        &self,
        manager: &M,
    ) -> Result<ResidentLayerGroupReport, M::Error> {
        let ids = self.units().iter().collect::<BTreeSet<_>>();
        let mut host_bytes = 0u64;
        let mut device_bytes = 0u64;
        let mut device_units = 0usize;
        for unit in manager
            .unit_reports()?
            .iter()
            .filter(|unit| ids.contains(unit.id()))
        {
            if unit.host_resident() {
                host_bytes = host_bytes
                    .checked_add(unit.host_allocated_bytes())
                    .ok_or(ResidencyWindowError::ArithmeticOverflow {
                        context: "execution group host bytes",
                    })
                    .map_err(eredu_core::BackendFailure::from_error)?;
            }
            if unit.device_resident() {
                device_bytes = device_bytes
                    .checked_add(unit.device_allocated_bytes())
                    .ok_or(ResidencyWindowError::ArithmeticOverflow {
                        context: "execution group device bytes",
                    })
                    .map_err(eredu_core::BackendFailure::from_error)?;
                device_units += 1;
            }
        }
        Ok(ResidentLayerGroupReport {
            id: self.id.clone(),
            ordered_units: self.units().len(),
            window_depth: self.depth(),
            host_bytes,
            device_bytes,
            device_units,
        })
    }
}

/// Logical residency attributed to one named execution group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidentLayerGroupReport {
    id: String,
    ordered_units: usize,
    window_depth: usize,
    host_bytes: u64,
    device_bytes: u64,
    device_units: usize,
}

impl ResidentLayerGroupReport {
    /// Returns the group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns the number of ordered units.
    pub const fn ordered_units(&self) -> usize {
        self.ordered_units
    }
    /// Returns the configured maximum device-unit count.
    pub const fn window_depth(&self) -> usize {
        self.window_depth
    }
    /// Returns current physical host allocation capacity for group units.
    pub const fn host_bytes(&self) -> u64 {
        self.host_bytes
    }
    /// Returns current device-resident bytes for group units.
    pub const fn device_bytes(&self) -> u64 {
        self.device_bytes
    }
    /// Returns current device-resident group units.
    pub const fn device_units(&self) -> usize {
        self.device_units
    }
}

/// Invalid ordered residency-window configuration or accounting.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ResidencyWindowError {
    /// A layer window had no units.
    #[error("device layer window requires at least one ordered unit")]
    EmptyLayerWindow,
    /// A device window depth was zero or exceeded its unit count.
    #[error("device layer window depth {depth} exceeds {layer_count} ordered units")]
    OversizedLayerWindow {
        /// Requested resident-unit bound.
        depth: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A requested current unit was outside the sequence.
    #[error("device layer index {index} is outside {layer_count} ordered units")]
    InvalidLayerIndex {
        /// Requested index.
        index: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// An ordered layer window repeated a unit identifier.
    #[error("device layer window contains duplicate unit {id}")]
    DuplicateLayerWindowUnit {
        /// Duplicated identifier.
        id: OffloadUnitId,
    },
    /// A named execution group had an empty identifier.
    #[error("residency window group identifiers must not be empty")]
    InvalidGroupId,
    /// Checked group accounting overflowed.
    #[error("residency window arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        sync::{Arc, Mutex},
    };

    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        store::{TensorMetadata, TensorSelection},
        StoredDtype,
    };
    use eredu_core::residency::{MemoryTier, OffloadConfig, OffloadUnitSpec, ResidencyPolicy};

    use super::*;

    #[test]
    fn conversion_payload_rejects_inexact_duplicate_and_overflowing_observations() {
        use eredu_core::resources::{
            ResourceAllocation, ResourceByteBounds, ResourceExtent, ResourceIdentity, ResourceRole,
            ResourceSize, ResourceUse,
        };
        let conversion = |key: &str, bytes| ResidentParameterConversion {
            allocation: ResourceAllocation {
                identity: ResourceIdentity {
                    scope: "test".into(),
                    key: key.into(),
                },
                uses: vec![ResourceUse {
                    owner: ResourceIdentity {
                        scope: "test".into(),
                        key: "owner".into(),
                    },
                    role: ResourceRole::Parameters,
                }],
                placement: eredu_core::Observed::Unavailable {
                    reason: "unknown pool".into(),
                },
                size: ResourceSize::Fixed {
                    extent: ResourceExtent {
                        payload: ResourceByteBounds::exact(bytes),
                        capacity: ResourceByteBounds::unknown(bytes, "unknown capacity"),
                    },
                },
            },
            bindings: vec![ResidentParameterConversionBinding {
                retention_group: None,
                owner: ResourceIdentity {
                    scope: "test".into(),
                    key: "owner".into(),
                },
                unit: OffloadUnitId::new("unit").unwrap(),
                name: "weight".into(),
                logical_target: Some("module.weight".into()),
            }],
        };
        let first = conversion("first", 16);
        let second = conversion("second", 32);
        assert_eq!(conversion_payload_bytes(&[]).unwrap(), 0);
        assert_eq!(
            conversion_payload_bytes(&[first.clone(), second]).unwrap(),
            48
        );
        assert!(conversion_payload_bytes(&[first.clone(), first.clone()]).is_err());
        assert!(conversion_payload_bytes(&[conversion("max", u64::MAX), first.clone()]).is_err());
        let mut missing_bindings = first.clone();
        missing_bindings.bindings.clear();
        assert!(conversion_payload_bytes(&[missing_bindings]).is_err());
        let mut unrelated_owner = first.clone();
        unrelated_owner.bindings[0].owner.key = "unrelated".into();
        assert!(conversion_payload_bytes(&[unrelated_owner]).is_err());
        let mut empty_binding = first.clone();
        empty_binding.bindings[0].logical_target = Some(" ".into());
        assert!(conversion_payload_bytes(&[empty_binding]).is_err());
        let mut uncertain = first;
        let ResourceSize::Fixed { extent } = &mut uncertain.allocation.size else {
            unreachable!()
        };
        extent.payload = ResourceByteBounds::unknown(16, "partial observation");
        assert!(conversion_payload_bytes(&[uncertain]).is_err());
    }

    #[test]
    fn conversion_resources_merge_backing_owners_and_reject_conflicts_atomically() {
        use crate::prediction_resources::include_resident_parameter_conversions;
        use eredu_core::resources::{
            ResourceAllocation, ResourceByteBounds, ResourceContext, ResourceCoverage,
            ResourceDescription, ResourceExtent, ResourceIdentity, ResourceRole, ResourceSize,
            ResourceUse, RESOURCE_DESCRIPTION_SCHEMA_VERSION,
        };

        let identity = |key: &str| ResourceIdentity {
            scope: "test".into(),
            key: key.into(),
        };
        let allocation = |key: &str, owner: &str, bytes, pool: &str| ResourceAllocation {
            identity: identity(key),
            uses: vec![ResourceUse {
                owner: identity(owner),
                role: ResourceRole::Parameters,
            }],
            placement: eredu_core::Observed::exact(identity(pool), "test physical pool"),
            size: ResourceSize::Fixed {
                extent: ResourceExtent {
                    payload: ResourceByteBounds::exact(bytes),
                    capacity: ResourceByteBounds::unknown(bytes, "native capacity unavailable"),
                },
            },
        };
        let conversion = |allocation: ResourceAllocation| ResidentParameterConversion {
            bindings: allocation
                .uses
                .iter()
                .map(|usage| ResidentParameterConversionBinding {
                    retention_group: None,
                    owner: usage.owner.clone(),
                    unit: OffloadUnitId::new(usage.owner.key.clone()).unwrap(),
                    name: "weight".into(),
                    logical_target: Some(format!("{}.weight", usage.owner.key)),
                })
                .collect(),
            allocation,
        };
        let report = ResidencyReport::new(
            true,
            eredu_core::residency::OffloadTelemetry::default().snapshot(),
            Vec::new(),
            Vec::new(),
            WeightStoreDiagnostics {
                backend: eredu_checkpoint::store::WeightStoreBackend::Memory,
                cache_hits: 0,
                cache_misses: 0,
                evictions: 0,
                currently_cached_shards: 0,
                touched_shard_paths: Vec::new(),
                payload_shard_paths: Vec::new(),
                physical_reads: 0,
                physical_read_bytes: 0,
                coalesced_group_hits: 0,
            },
        );
        let baseline = ResourceDescription {
            schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
            scope: identity("execution"),
            context: ResourceContext::default(),
            horizon: ResourceContext::default(),
            coverage: ResourceCoverage::Complete,
            allocations: vec![allocation("shared", "target", 16, "pool")],
        };
        baseline.validate().unwrap();
        assert!(report.host_storage_resources().is_none());
        let host_report = report
            .clone()
            .with_host_storage_resources(baseline.clone())
            .unwrap();
        assert_eq!(host_report.host_storage_resources(), Some(&baseline));
        assert_eq!(host_report.offload(), report.offload());
        let mut malformed_host = baseline.clone();
        malformed_host
            .allocations
            .push(malformed_host.allocations[0].clone());
        assert!(report
            .clone()
            .with_host_storage_resources(malformed_host)
            .is_err());
        let observed = report
            .clone()
            .with_device_parameter_conversions(vec![conversion(allocation(
                "shared",
                "prediction",
                16,
                "pool",
            ))])
            .unwrap();
        let mut merged = baseline.clone();
        include_resident_parameter_conversions(&mut merged, &observed).unwrap();
        assert_eq!(merged.allocations.len(), 1);
        assert_eq!(merged.allocations[0].size, baseline.allocations[0].size);
        assert_eq!(merged.allocations[0].uses.len(), 2);
        assert!(merged.allocations[0]
            .uses
            .iter()
            .any(|usage| usage.owner == identity("target")));
        assert!(merged.allocations[0]
            .uses
            .iter()
            .any(|usage| usage.owner == identity("prediction")));
        let once = merged.clone();
        include_resident_parameter_conversions(&mut merged, &observed).unwrap();
        assert_eq!(
            merged, once,
            "repeated observations must not add owners or bytes"
        );

        let empty = report
            .clone()
            .with_device_parameter_conversions(Vec::new())
            .unwrap();
        let mut observed_empty = baseline.clone();
        include_resident_parameter_conversions(&mut observed_empty, &empty).unwrap();
        assert_eq!(observed_empty, baseline, "observed empty is complete");
        let mut missing = baseline.clone();
        include_resident_parameter_conversions(&mut missing, &report).unwrap();
        assert_eq!(missing.allocations, baseline.allocations);
        let ResourceCoverage::Partial { reasons } = missing.coverage else {
            panic!("missing conversion identities must preserve incomplete coverage")
        };
        assert!(reasons
            .iter()
            .any(|reason| reason.contains("conversion backing identities")));

        for conflicting in [
            allocation("shared", "prediction", 32, "pool"),
            allocation("shared", "prediction", 16, "different_pool"),
        ] {
            // The first entry could be added before the conflicting backing is found.
            // Failure must leave the entire input unchanged, including that addition.
            let conflict = report
                .clone()
                .with_device_parameter_conversions(vec![
                    conversion(allocation("new", "prediction", 8, "pool")),
                    conversion(conflicting),
                ])
                .unwrap();
            let mut unchanged = baseline.clone();
            assert!(include_resident_parameter_conversions(&mut unchanged, &conflict).is_err());
            assert_eq!(unchanged, baseline);
        }
        let mut duplicates = baseline.clone();
        duplicates
            .allocations
            .push(duplicates.allocations[0].clone());
        let invalid_input = duplicates.clone();
        assert!(include_resident_parameter_conversions(&mut duplicates, &empty).is_err());
        assert_eq!(
            duplicates, invalid_input,
            "duplicate input must not be silently coalesced"
        );
    }

    struct Catalog(BTreeMap<String, TensorMetadata>);

    struct TestLeaseStorage(BTreeMap<String, u32>);

    impl eredu_core::residency::ResidencyLeaseStorage for TestLeaseStorage {
        type DeviceValue = u32;
        type HostValue = u32;
        type Error = &'static str;
        type BindingNames<'a> = std::iter::Map<
            std::collections::btree_map::Keys<'a, String, u32>,
            fn(&'a String) -> &'a str,
        >;

        fn device_value<'a>(
            &'a self,
            _: &OffloadUnitId,
            name: &str,
        ) -> Result<&'a Self::DeviceValue, Self::Error> {
            self.0.get(name).ok_or("unknown binding")
        }

        fn host_value<'a>(
            &'a self,
            _: &OffloadUnitId,
            name: &str,
        ) -> Result<&'a Self::HostValue, Self::Error> {
            self.0.get(name).ok_or("unknown binding")
        }

        fn binding_names(&self) -> Self::BindingNames<'_> {
            self.0.keys().map(String::as_str)
        }
    }

    #[derive(Default)]
    struct TestLeaseOwner(Mutex<Vec<(OffloadUnitId, MemoryTier)>>);

    impl eredu_core::residency::ResidencyLeaseOwner for TestLeaseOwner {
        fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier) {
            self.0.lock().unwrap().push((id.clone(), tier));
        }
    }

    struct TestTransferCompletion {
        succeeds: bool,
        waits: Arc<Mutex<usize>>,
    }

    struct TestTransferResources(Arc<Mutex<Vec<bool>>>);

    type TransferCompletionRecord = (Vec<OffloadUnitId>, MemoryTier, u64, bool);

    #[derive(Default)]
    struct TestTransferOwner(Mutex<Vec<TransferCompletionRecord>>);

    impl ResidencyTransferOwner<TestTransferCompletion, TestTransferResources> for TestTransferOwner {
        type Executor = Mutex<usize>;
        type Error = &'static str;

        fn order_after(
            _: &TestTransferCompletion,
            executor: &Self::Executor,
            _: &OffloadUnitId,
        ) -> Result<(), Self::Error> {
            *executor.lock().unwrap() += 1;
            Ok(())
        }

        fn is_complete(
            completion: &TestTransferCompletion,
            _: &OffloadUnitId,
        ) -> Result<bool, Self::Error> {
            Ok(completion.succeeds)
        }

        fn wait(completion: &TestTransferCompletion, _: &OffloadUnitId) -> Result<(), Self::Error> {
            *completion.waits.lock().unwrap() += 1;
            completion.succeeds.then_some(()).ok_or("transfer failed")
        }

        fn finish_resources(resources: TestTransferResources, succeeded: bool) {
            resources.0.lock().unwrap().push(succeeded);
        }

        fn resolve_transfer(
            &self,
            ids: &[OffloadUnitId],
            tier: MemoryTier,
            generation: u64,
            succeeded: bool,
        ) -> Result<(), Self::Error> {
            self.0
                .lock()
                .unwrap()
                .push((ids.to_vec(), tier, generation, succeeded));
            Ok(())
        }
    }

    #[derive(Default)]
    struct WindowManager {
        prepared: RefCell<Vec<(String, Vec<OffloadUnitId>)>>,
        evicted: RefCell<Vec<OffloadUnitId>>,
    }

    impl ResidencyWindowManager for WindowManager {
        type Error = eredu_core::BackendFailure;

        fn prepare_window(
            &self,
            active: &[OffloadUnitId],
            _: &[OffloadUnitId],
            _: MemoryTier,
        ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
            self.prepared
                .borrow_mut()
                .push(("default".into(), active.to_vec()));
            Ok(active
                .iter()
                .cloned()
                .map(|id| (id, PrefetchOutcome::Hit))
                .collect())
        }

        fn prepare_group_window(
            &self,
            group: &str,
            active: &[OffloadUnitId],
            _: &[OffloadUnitId],
            _: MemoryTier,
        ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
            self.prepared
                .borrow_mut()
                .push((group.to_owned(), active.to_vec()));
            Ok(active
                .iter()
                .cloned()
                .map(|id| (id, PrefetchOutcome::Miss))
                .collect())
        }

        fn evict(&self, id: &OffloadUnitId, _: MemoryTier) -> Result<bool, Self::Error> {
            self.evicted.borrow_mut().push(id.clone());
            Ok(true)
        }

        fn unit_reports(&self) -> Result<Vec<UnitResidencyReport>, Self::Error> {
            Ok(Vec::new())
        }
    }

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(
            &self,
            key: &str,
        ) -> Result<TensorMetadata, eredu_checkpoint::store::StoreError> {
            self.0.get(key).cloned().ok_or_else(|| {
                eredu_checkpoint::store::StoreError::UnknownTensor {
                    key: key.to_owned(),
                }
            })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            name: name.to_owned(),
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: StoredDtype::F32,
            encoded_byte_len: 0,
            backing_shard: None,
        }
    }

    #[test]
    fn declarations_are_validated_and_deterministic() {
        let b = WeightBinding::new("b", "b.weight", TensorSelection::Full, 4).unwrap();
        let a = WeightBinding::new("a", "a.weight", TensorSelection::Full, 8).unwrap();
        let id = OffloadUnitId::new("layer.0").unwrap();
        let unit = OffloadUnit::new(id, [b, a]).unwrap();
        assert_eq!(unit.bindings()[0].name(), "a");
        assert_eq!(unit.bindings()[1].name(), "b");
    }

    #[test]
    fn neutral_lease_exposes_native_storage_and_releases_exact_pin() {
        let owner = Arc::new(TestLeaseOwner::default());
        let id = OffloadUnitId::new("layer.0").unwrap();
        let pin_owner: Arc<dyn eredu_core::residency::ResidencyLeaseOwner> = owner.clone();
        let lease = eredu_core::residency::ResidencyLease::new(
            id.clone(),
            MemoryTier::Device,
            TestLeaseStorage(BTreeMap::from([("weight".into(), 7)])),
            Arc::downgrade(&pin_owner),
        );
        assert_eq!(lease.device_value("weight"), Ok(&7));
        assert_eq!(lease.binding_names().collect::<Vec<_>>(), vec!["weight"]);
        drop(lease);
        assert_eq!(*owner.0.lock().unwrap(), vec![(id, MemoryTier::Device)]);
    }

    #[test]
    fn neutral_transfer_orders_publishes_and_releases_resources() {
        let owner = Arc::new(TestTransferOwner::default());
        let waits = Arc::new(Mutex::new(0));
        let resources = Arc::new(Mutex::new(Vec::new()));
        let executor = Mutex::new(0);
        let id = OffloadUnitId::new("layer.0").unwrap();
        let mut transfer = ResidencyTransfer::submitted(
            vec![7],
            TestTransferCompletion {
                succeeds: true,
                waits: Arc::clone(&waits),
            },
            TestTransferResources(Arc::clone(&resources)),
            Arc::downgrade(&owner),
            vec![id.clone()],
            MemoryTier::Device,
            11,
        );

        assert_eq!(transfer.leases(), &[7]);
        transfer.order_after(&executor).unwrap();
        assert_eq!(*executor.lock().unwrap(), 1);
        transfer.synchronize().unwrap();
        assert!(transfer.is_complete().unwrap());
        assert_eq!(*waits.lock().unwrap(), 1);
        assert_eq!(*resources.lock().unwrap(), vec![true]);
        assert_eq!(
            *owner.0.lock().unwrap(),
            vec![(vec![id], MemoryTier::Device, 11, true)]
        );
    }

    #[test]
    fn neutral_transfer_failure_remains_observable_and_resolves_once() {
        let owner = Arc::new(TestTransferOwner::default());
        let waits = Arc::new(Mutex::new(0));
        let resources = Arc::new(Mutex::new(Vec::new()));
        let id = OffloadUnitId::new("layer.0").unwrap();
        let mut transfer = ResidencyTransfer::submitted(
            Vec::<u8>::new(),
            TestTransferCompletion {
                succeeds: false,
                waits: Arc::clone(&waits),
            },
            TestTransferResources(Arc::clone(&resources)),
            Arc::downgrade(&owner),
            vec![id.clone()],
            MemoryTier::Host,
            4,
        );

        assert_eq!(transfer.synchronize(), Err("transfer failed"));
        assert_eq!(transfer.synchronize(), Err("transfer failed"));
        assert_eq!(*resources.lock().unwrap(), vec![false]);
        assert_eq!(
            *owner.0.lock().unwrap(),
            vec![(vec![id], MemoryTier::Host, 4, false)]
        );
        drop(transfer);
        assert_eq!(*waits.lock().unwrap(), 3);
    }

    #[test]
    fn binding_selection_rewrites_sources_and_exact_bytes_neutrally() {
        let catalog = Catalog(BTreeMap::from([(
            "weight".into(),
            metadata("weight", vec![2, 2]),
        )]));
        let binding = WeightBinding::new("weight", "weight", TensorSelection::Full, 16)
            .unwrap()
            .select_bounded_output(
                &catalog,
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
            )
            .unwrap();

        assert_eq!(binding.expected_bytes(), 8);
        assert!(matches!(
            binding.source_recipe(),
            DerivedWeightRecipe::Source {
                selection: TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
                ..
            }
        ));
    }

    #[test]
    fn controller_validates_catalog_bytes_before_allocating_backend_storage() {
        let catalog = Catalog(BTreeMap::from([
            ("a.weight".into(), metadata("a.weight", vec![2])),
            ("b.weight".into(), metadata("b.weight", vec![1])),
        ]));
        let id = OffloadUnitId::new("layer.0").unwrap();
        let unit = OffloadUnit::new(
            id.clone(),
            [
                WeightBinding::new("a", "a.weight", TensorSelection::Full, 8).unwrap(),
                WeightBinding::new("b", "b.weight", TensorSelection::Full, 4).unwrap(),
            ],
        )
        .unwrap();
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [
                OffloadUnitSpec::new(id.clone(), 12, ResidencyPolicy::Windowed, MemoryTier::Disk)
                    .unwrap(),
            ],
        )
        .unwrap();

        let controller = ResidencyController::new(&catalog, plan, [unit]).unwrap();
        assert_eq!(controller.units().len(), 1);
        assert_eq!(controller.unit(&id).unwrap().bindings().len(), 2);
        assert!(!controller.ledger().initialized());
    }

    #[test]
    fn controller_resolves_aliases_across_independent_units() {
        let catalog = Catalog(BTreeMap::from([
            ("physical.owner".into(), metadata("physical.owner", vec![1])),
            ("slice.local".into(), metadata("slice.local", vec![1])),
        ]));
        let owner_id = OffloadUnitId::new("slice.0").unwrap();
        let alias_id = OffloadUnitId::new("slice.1").unwrap();
        let owner_binding =
            WeightBinding::new("weight", "physical.owner", TensorSelection::Full, 4)
                .unwrap()
                .with_logical_target("shared.owner")
                .unwrap();
        let alias_binding = WeightBinding::alias("weight", "shared.owner", 4)
            .unwrap()
            .with_logical_target("slice.1.weight")
            .unwrap();
        let local_binding =
            WeightBinding::new("local", "slice.local", TensorSelection::Full, 4).unwrap();
        let units = [
            OffloadUnit::new(owner_id.clone(), [owner_binding]).unwrap(),
            OffloadUnit::new(alias_id.clone(), [alias_binding, local_binding]).unwrap(),
        ];
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [
                OffloadUnitSpec::new(
                    owner_id.clone(),
                    4,
                    ResidencyPolicy::Windowed,
                    MemoryTier::Disk,
                )
                .unwrap(),
                OffloadUnitSpec::new(
                    alias_id.clone(),
                    4,
                    ResidencyPolicy::Windowed,
                    MemoryTier::Disk,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let controller = ResidencyController::new(&catalog, plan, units).unwrap();
        let alias = controller
            .unit(&alias_id)
            .unwrap()
            .bindings()
            .iter()
            .find(|binding| binding.is_alias())
            .unwrap();
        let (resolved_unit, resolved) = controller.binding_owner(&alias_id, alias).unwrap();
        assert_eq!(resolved_unit, &owner_id);
        assert_eq!(resolved.logical_target(), Some("shared.owner"));
    }

    #[test]
    fn controller_uses_exact_unit_catalogs_with_one_shared_budget() {
        let ids = ["target", "extension"].map(|name| OffloadUnitId::new(name).unwrap());
        let first = Catalog(BTreeMap::from([(
            "weight".into(),
            metadata("weight", vec![1]),
        )]));
        let second = Catalog(BTreeMap::from([(
            "weight".into(),
            metadata("weight", vec![2]),
        )]));
        let units = ids
            .iter()
            .zip([4, 8])
            .map(|(id, bytes)| {
                OffloadUnit::new(
                    id.clone(),
                    [
                        WeightBinding::new("weight", "weight", TensorSelection::Full, bytes)
                            .unwrap(),
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(8), None, 1).unwrap(),
            ids.iter().zip([4, 8]).map(|(id, bytes)| {
                OffloadUnitSpec::new(
                    id.clone(),
                    bytes,
                    ResidencyPolicy::Cacheable,
                    MemoryTier::Disk,
                )
                .unwrap()
            }),
        )
        .unwrap();
        assert!(matches!(
            ResidencyController::new(&first, plan.clone(), units.clone()),
            Err(ResidencyControllerError::BindingByteMismatch { .. })
        ));
        let mut controller = ResidencyController::new_with_catalogs(
            |id| if id == &ids[0] { &first } else { &second },
            plan,
            units,
        )
        .unwrap();
        controller.ledger_mut().mark_initialized();
        let acquisition = controller
            .plan_acquisition(&ids, MemoryTier::Device)
            .unwrap();
        assert!(controller
            .reserve_acquisition(
                &acquisition,
                &[(ids[0].clone(), 4), (ids[1].clone(), 8)],
                MemoryTier::Device
            )
            .is_err());
        assert_eq!(
            controller
                .ledger()
                .telemetry()
                .resident_bytes()
                .get(MemoryTier::Device),
            0
        );
    }

    #[test]
    fn controller_owns_named_window_and_unique_lookahead_selection() {
        let ids = ["a", "b", "c"].map(|name| OffloadUnitId::new(format!("layer.{name}")).unwrap());
        let catalog = Catalog(BTreeMap::from([
            ("a".into(), metadata("a", vec![1])),
            ("b".into(), metadata("b", vec![1])),
            ("c".into(), metadata("c", vec![1])),
        ]));
        let units = ids.iter().zip(["a", "b", "c"]).map(|(id, key)| {
            OffloadUnit::new(
                id.clone(),
                [WeightBinding::new("weight", key, TensorSelection::Full, 4).unwrap()],
            )
            .unwrap()
        });
        let plan = OffloadPlan::new(
            OffloadConfig::new(None, None, 2).unwrap(),
            ids.iter().map(|id| {
                OffloadUnitSpec::new(id.clone(), 4, ResidencyPolicy::Windowed, MemoryTier::Disk)
                    .unwrap()
            }),
        )
        .unwrap();
        let mut controller = ResidencyController::new(&catalog, plan, units).unwrap();
        controller.ledger_mut().mark_initialized();
        let selected = controller
            .commit_group_window(
                "decoder",
                &[ids[0].clone()],
                &[ids[1].clone(), ids[1].clone(), ids[2].clone()],
                MemoryTier::Device,
            )
            .unwrap();
        assert_eq!(selected, vec![ids[1].clone(), ids[2].clone()]);
        assert_eq!(
            controller.ledger().active_window(),
            BTreeSet::from([ids[0].clone()])
        );
    }

    #[test]
    fn controller_owns_acquisition_reservation_and_rollback() {
        let ids = ["a", "b"].map(|name| OffloadUnitId::new(format!("layer.{name}")).unwrap());
        let catalog = Catalog(BTreeMap::from([
            ("a".into(), metadata("a", vec![1])),
            ("b".into(), metadata("b", vec![1])),
        ]));
        let units = ids.iter().zip(["a", "b"]).map(|(id, key)| {
            OffloadUnit::new(
                id.clone(),
                [WeightBinding::new("weight", key, TensorSelection::Full, 4).unwrap()],
            )
            .unwrap()
        });
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            ids.iter().map(|id| {
                OffloadUnitSpec::new(id.clone(), 4, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                    .unwrap()
            }),
        )
        .unwrap();
        let mut controller = ResidencyController::new(&catalog, plan, units).unwrap();
        controller.ledger_mut().mark_initialized();

        let acquisition = controller
            .plan_acquisition(&ids, MemoryTier::Device)
            .unwrap();
        assert_eq!(acquisition.missing(), &[true, true]);
        assert!(controller
            .reserve_acquisition(
                &acquisition,
                &[(ids[0].clone(), 4), (ids[1].clone(), 4)],
                MemoryTier::Device,
            )
            .unwrap()
            .is_empty());
        controller
            .rollback_acquisition(&acquisition, MemoryTier::Device)
            .unwrap();
        assert!(!controller
            .ledger()
            .is_resident(&ids[0], MemoryTier::Device)
            .unwrap());
        assert!(!controller
            .ledger()
            .is_resident(&ids[1], MemoryTier::Device)
            .unwrap());

        let acquisition = controller
            .plan_acquisition(&[ids[0].clone()], MemoryTier::Device)
            .unwrap();
        controller
            .reserve_acquisition(&acquisition, &[(ids[0].clone(), 4)], MemoryTier::Device)
            .unwrap();
        controller
            .publish_acquisition_copy(
                &ids[0],
                MemoryTier::Device,
                4,
                4,
                None,
                eredu_core::residency::TransferDirection::DiskToDevice,
                Duration::from_millis(2),
            )
            .unwrap();
        assert!(controller
            .ledger()
            .is_resident(&ids[0], MemoryTier::Device)
            .unwrap());
        assert_eq!(
            controller
                .ledger()
                .telemetry()
                .transfer(eredu_core::residency::TransferDirection::DiskToDevice)
                .bytes(),
            4
        );
    }

    #[test]
    fn controller_rejects_binding_and_plan_byte_mismatches() {
        let catalog = Catalog(BTreeMap::from([(
            "weight".into(),
            metadata("weight", vec![2]),
        )]));
        let id = OffloadUnitId::new("layer.0").unwrap();
        let plan = |bytes| {
            OffloadPlan::new(
                OffloadConfig::default(),
                [OffloadUnitSpec::new(
                    id.clone(),
                    bytes,
                    ResidencyPolicy::Windowed,
                    MemoryTier::Disk,
                )
                .unwrap()],
            )
            .unwrap()
        };

        let wrong_binding = OffloadUnit::new(
            id.clone(),
            [WeightBinding::new("weight", "weight", TensorSelection::Full, 4).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            ResidencyController::new(&catalog, plan(4), [wrong_binding]),
            Err(ResidencyControllerError::BindingByteMismatch { .. })
        ));

        let wrong_plan = OffloadUnit::new(
            id.clone(),
            [WeightBinding::new("weight", "weight", TensorSelection::Full, 8).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            ResidencyController::new(&catalog, plan(16), [wrong_plan]),
            Err(ResidencyControllerError::UnitByteMismatch { .. })
        ));
    }

    #[test]
    fn named_windows_prepare_and_trim_without_backend_types() {
        let ids = ["layer.0", "layer.1", "layer.2"].map(|id| OffloadUnitId::new(id).unwrap());
        let group = ResidentLayerGroup::new("decoder", ids.clone(), 2).unwrap();
        let manager = WindowManager::default();

        assert_eq!(group.prepare(&manager, 1).unwrap().len(), 2);
        assert_eq!(
            manager.prepared.borrow()[0],
            ("decoder".into(), vec![ids[1].clone(), ids[2].clone()])
        );
        assert_eq!(manager.evicted.borrow().as_slice(), &[ids[0].clone()]);
        assert_eq!(group.report(&manager).unwrap().device_units(), 0);
    }
}
