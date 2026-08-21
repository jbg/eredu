//! Backend-neutral immutable-weight residency declarations and control state.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Weak,
    time::Duration,
};

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeError},
    store::{TensorSelection, WeightStoreDiagnostics},
};
use eredu_core::residency::{
    EvictedResidencyCopy, MemoryTier, OffloadPlan, OffloadReport, OffloadUnitId, PrefetchOutcome,
    ResidencyLedger, ResidencyLedgerError, UnitResidencyReport,
};

/// Deterministic telemetry from one bounded weight-materialization pass.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct WeightMaterializationReport {
    /// Planner ceiling for simultaneously live conversion data.
    pub admitted_working_set_bytes: u64,
    /// Number of dense semantic matrices transformed.
    pub transformed_weights: usize,
    /// Number of independently evaluated source row tiles.
    pub source_tiles: usize,
    /// Largest number of submitted tile completions retained simultaneously.
    pub peak_in_flight_tiles: usize,
    /// Total logical dense bytes selected from the source store.
    pub source_bytes_read: u64,
    /// Total encoded output bytes written to persistent storage.
    pub output_bytes: u64,
    /// Largest conservative conversion working set admitted for one tile.
    pub peak_planned_working_set_bytes: u64,
    /// Largest source recipe output tile.
    pub largest_source_tile_bytes: u64,
    /// Largest encoded output tile written together.
    pub largest_output_tile_bytes: u64,
}

/// Immutable residency-control and checkpoint-storage telemetry snapshot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyReport {
    initialized: bool,
    offload: OffloadReport,
    units: Vec<UnitResidencyReport>,
    active_window: Vec<OffloadUnitId>,
    weight_store: WeightStoreDiagnostics,
    materialization: Option<WeightMaterializationReport>,
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
            materialization: None,
        }
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

/// One named checkpoint selection within an atomic resident unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightBinding {
    name: String,
    alias_of: Option<String>,
    logical_target: Option<String>,
    checkpoint_key: String,
    selection: TensorSelection,
    recipe: Option<DerivedWeightRecipe>,
    expected_bytes: u64,
}

impl WeightBinding {
    /// Creates a direct binding with a stable local name and selected size.
    pub fn new(
        name: impl Into<String>,
        checkpoint_key: impl Into<String>,
        selection: TensorSelection,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        let name = validate_name(name.into())?;
        let checkpoint_key = checkpoint_key.into();
        if checkpoint_key.trim().is_empty() {
            return Err(ResidencyDeclarationError::InvalidCheckpointKey { name });
        }
        validate_size(&name, expected_bytes)?;
        Ok(Self {
            name,
            alias_of: None,
            logical_target: None,
            checkpoint_key,
            selection,
            recipe: None,
            expected_bytes,
        })
    }

    /// Creates a binding backed by a composable derived-weight recipe.
    pub fn from_recipe(
        name: impl Into<String>,
        recipe: DerivedWeightRecipe,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        let name = validate_name(name.into())?;
        validate_size(&name, expected_bytes)?;
        let checkpoint_key = first_source(&name, &recipe)?;
        Ok(Self {
            name,
            alias_of: None,
            logical_target: None,
            checkpoint_key,
            selection: TensorSelection::Full,
            recipe: Some(recipe),
            expected_bytes,
        })
    }

    /// Creates one logical destination that shares an already materialized
    /// owner binding in the same atomic unit.
    pub fn alias(
        name: impl Into<String>,
        owner: impl Into<String>,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        let name = validate_name(name.into())?;
        let owner = validate_name(owner.into())?;
        validate_size(&name, expected_bytes)?;
        Ok(Self {
            name,
            alias_of: Some(owner),
            logical_target: None,
            checkpoint_key: String::new(),
            selection: TensorSelection::Full,
            recipe: None,
            expected_bytes,
        })
    }

    /// Returns the stable name used to look up a resident value.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the logical owner when this binding is an alias.
    pub fn alias_of(&self) -> Option<&str> {
        self.alias_of.as_deref()
    }

    /// Returns whether this binding reuses another logical binding's native
    /// materialization.
    pub const fn is_alias(&self) -> bool {
        self.alias_of.is_some()
    }

    /// Replaces the stable name used to address this value inside its resident unit.
    pub fn with_name(mut self, name: impl Into<String>) -> Result<Self, ResidencyDeclarationError> {
        self.name = validate_name(name.into())?;
        Ok(self)
    }

    /// Returns the architecture-logical parameter destination.
    pub fn logical_target(&self) -> Option<&str> {
        self.logical_target.as_deref()
    }

    /// Attaches the architecture-logical parameter destination.
    pub fn with_logical_target(
        mut self,
        target: impl Into<String>,
    ) -> Result<Self, ResidencyDeclarationError> {
        self.logical_target = Some(validate_name(target.into())?);
        Ok(self)
    }

    /// Returns the first physical checkpoint source.
    pub fn checkpoint_key(&self) -> &str {
        &self.checkpoint_key
    }

    /// Returns the direct checkpoint selection.
    pub fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Returns the derived recipe when this is not a direct binding.
    pub const fn recipe(&self) -> Option<&DerivedWeightRecipe> {
        self.recipe.as_ref()
    }

    /// Returns the complete source recipe represented by this binding.
    pub fn source_recipe(&self) -> DerivedWeightRecipe {
        assert!(
            self.alias_of.is_none(),
            "logical aliases have no physical recipe"
        );
        self.recipe.clone().unwrap_or_else(|| {
            DerivedWeightRecipe::source(self.checkpoint_key.clone(), self.selection.clone())
        })
    }

    /// Returns every checkpoint key consumed by this binding.
    pub fn checkpoint_keys(&self) -> Vec<&str> {
        if self.alias_of.is_some() {
            return Vec::new();
        }
        self.recipe.as_ref().map_or_else(
            || vec![self.checkpoint_key.as_str()],
            DerivedWeightRecipe::source_keys,
        )
    }

    /// Returns the exact logical materialized byte length.
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }

    /// Replaces the physical source with an equivalent validated recipe.
    pub fn with_source_recipe(
        mut self,
        recipe: DerivedWeightRecipe,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyDeclarationError> {
        if self.alias_of.is_some() {
            return Err(ResidencyDeclarationError::AliasHasPhysicalSource { name: self.name });
        }
        validate_size(&self.name, expected_bytes)?;
        self.checkpoint_key = first_source(&self.name, &recipe)?;
        self.selection = TensorSelection::Full;
        self.recipe = Some(recipe);
        self.expected_bytes = expected_bytes;
        Ok(self)
    }

    /// Rewrites one logical output selection into bounded physical sources.
    pub fn select_bounded_output<C: RecipeCatalog + ?Sized>(
        self,
        catalog: &C,
        selection: TensorSelection,
    ) -> Result<Self, WeightBindingSelectionError> {
        let recipe = self.source_recipe().select_bounded(catalog, selection)?;
        let bytes = recipe.infer(catalog)?.byte_len();
        Ok(self.with_source_recipe(recipe, bytes)?)
    }
}

/// Validated owner/alias partition for one atomic binding unit.
#[derive(Debug)]
pub struct WeightBindingPlan<'a> {
    owners: Vec<&'a WeightBinding>,
    aliases: Vec<(&'a WeightBinding, &'a WeightBinding)>,
}

impl<'a> WeightBindingPlan<'a> {
    /// Validates unique identities, owner existence, cycles, and byte geometry.
    pub fn new(bindings: &'a [WeightBinding]) -> Result<Self, ResidencyDeclarationError> {
        let by_name = bindings
            .iter()
            .map(|binding| (binding.name(), binding))
            .collect::<BTreeMap<_, _>>();
        if by_name.len() != bindings.len() {
            let duplicate = bindings
                .iter()
                .map(WeightBinding::name)
                .find(|name| bindings.iter().filter(|item| item.name() == *name).count() > 1)
                .unwrap_or("<unknown>");
            return Err(ResidencyDeclarationError::DuplicateLogicalBinding {
                name: duplicate.to_owned(),
            });
        }

        fn resolve<'a>(
            binding: &'a WeightBinding,
            by_name: &BTreeMap<&str, &'a WeightBinding>,
            visiting: &mut BTreeSet<String>,
        ) -> Result<&'a WeightBinding, ResidencyDeclarationError> {
            let Some(owner_name) = binding.alias_of() else {
                return Ok(binding);
            };
            if !visiting.insert(binding.name().to_owned()) {
                return Err(ResidencyDeclarationError::BindingAliasCycle {
                    name: binding.name().to_owned(),
                });
            }
            let owner = by_name.get(owner_name).copied().ok_or_else(|| {
                ResidencyDeclarationError::UnknownBindingAliasOwner {
                    alias: binding.name().to_owned(),
                    owner: owner_name.to_owned(),
                }
            })?;
            let resolved = resolve(owner, by_name, visiting)?;
            visiting.remove(binding.name());
            Ok(resolved)
        }

        let owners = bindings
            .iter()
            .filter(|binding| !binding.is_alias())
            .collect::<Vec<_>>();
        let mut aliases = Vec::new();
        for alias in bindings.iter().filter(|binding| binding.is_alias()) {
            let owner = resolve(alias, &by_name, &mut BTreeSet::new())?;
            if alias.expected_bytes() != owner.expected_bytes() {
                return Err(ResidencyDeclarationError::BindingAliasByteMismatch {
                    alias: alias.name().to_owned(),
                    owner: owner.name().to_owned(),
                    alias_bytes: alias.expected_bytes(),
                    owner_bytes: owner.expected_bytes(),
                });
            }
            aliases.push((alias, owner));
        }
        Ok(Self { owners, aliases })
    }

    /// Canonical bindings which require physical materialization.
    pub fn owners(&self) -> impl Iterator<Item = &'a WeightBinding> + '_ {
        self.owners.iter().copied()
    }

    /// Logical aliases paired with their resolved canonical owners.
    pub fn aliases(&self) -> impl Iterator<Item = (&'a WeightBinding, &'a WeightBinding)> + '_ {
        self.aliases.iter().copied()
    }
}

/// Failure while rewriting a binding into bounded physical selections.
#[derive(Debug, thiserror::Error)]
pub enum WeightBindingSelectionError {
    /// Neutral recipe inference or selection pushdown failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// The rewritten binding declaration was invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
}

/// A deterministic group of weight bindings managed as one atomic unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadUnit {
    id: OffloadUnitId,
    bindings: Vec<WeightBinding>,
}

/// Validated backend-neutral declarations paired with residency control state.
///
/// Concrete backends own their native values separately and mirror every
/// publication or eviction through this controller's ledger. Keeping the
/// declarations here ensures checkpoint shape policy and plan identity are
/// validated before any backend allocation begins.
#[derive(Debug)]
pub struct ResidencyController {
    ledger: ResidencyLedger,
    units: BTreeMap<OffloadUnitId, OffloadUnit>,
    alias_owners: BTreeMap<(OffloadUnitId, String), (OffloadUnitId, String)>,
}

/// One validated immutable-weight acquisition batch and its initial hit/miss state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyAcquisition {
    ids: Vec<OffloadUnitId>,
    missing: Vec<bool>,
}

/// Backend-native host/device storage exposed through a neutral residency lease.
pub trait ResidencyLeaseStorage {
    /// Backend-native executable value.
    type DeviceValue;
    /// Backend-native host-resident value.
    type HostValue;
    /// Concrete lookup failure.
    type Error;
    /// Allocation-free or cold-path iterator over stable binding names.
    type BindingNames<'a>: Iterator<Item = &'a str>
    where
        Self: 'a;

    /// Looks up one executable binding.
    fn device_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::DeviceValue, Self::Error>;

    /// Looks up one host-resident binding.
    fn host_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::HostValue, Self::Error>;

    /// Returns binding names in stable order.
    fn binding_names(&self) -> Self::BindingNames<'_>;
}

/// Concrete manager hook used when a residency lease releases its exact pin.
pub trait ResidencyLeaseOwner: Sized {
    /// Releases one tier pin. Drop paths must tolerate an already-destroyed manager.
    fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier);
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

/// Statically dispatched lease retaining one backend-native resident unit.
pub struct ResidencyLease<S, O>
where
    S: ResidencyLeaseStorage,
    O: ResidencyLeaseOwner,
{
    id: OffloadUnitId,
    tier: MemoryTier,
    storage: S,
    owner: Weak<O>,
}

impl<S, O> ResidencyLease<S, O>
where
    S: ResidencyLeaseStorage,
    O: ResidencyLeaseOwner,
{
    /// Creates a lease after the neutral controller has pinned the resident copy.
    pub fn new(id: OffloadUnitId, tier: MemoryTier, storage: S, owner: Weak<O>) -> Self {
        Self {
            id,
            tier,
            storage,
            owner,
        }
    }

    /// Returns the acquired unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the protected resident tier.
    pub const fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Looks up one backend-native executable binding without cloning it.
    pub fn device_value(&self, name: &str) -> Result<&S::DeviceValue, S::Error> {
        self.storage.device_value(&self.id, name)
    }

    /// Looks up one backend-native host binding without cloning it.
    pub fn host_value(&self, name: &str) -> Result<&S::HostValue, S::Error> {
        self.storage.host_value(&self.id, name)
    }

    /// Returns binding names in stable order.
    pub fn binding_names(&self) -> S::BindingNames<'_> {
        self.storage.binding_names()
    }
}

impl<S, O> Drop for ResidencyLease<S, O>
where
    S: ResidencyLeaseStorage,
    O: ResidencyLeaseOwner,
{
    fn drop(&mut self) {
        if let Some(owner) = self.owner.upgrade() {
            owner.release_residency_pin(&self.id, self.tier);
        }
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
    /// Validates declarations against checkpoint metadata and an explicit plan.
    pub fn new<C: RecipeCatalog + ?Sized>(
        catalog: &C,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
    ) -> Result<Self, ResidencyControllerError> {
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
            let mut total = 0u64;
            for binding in unit.bindings().iter().filter(|binding| !binding.is_alias()) {
                total = total.checked_add(binding.expected_bytes()).ok_or(
                    ResidencyControllerError::ArithmeticOverflow {
                        context: "unit binding byte total",
                    },
                )?;
                if !binding.is_alias() {
                    let actual = binding
                        .source_recipe()
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
            ledger: ResidencyLedger::new(plan),
            units: definitions,
            alias_owners,
        })
    }

    /// Returns the validated declaration for one planned unit.
    pub fn unit(&self, id: &OffloadUnitId) -> Option<&OffloadUnit> {
        self.units.get(id)
    }

    /// Returns declarations in stable unit-identifier order.
    pub fn units(&self) -> impl ExactSizeIterator<Item = &OffloadUnit> {
        self.units.values()
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

fn validate_global_binding_aliases(
    units: &BTreeMap<OffloadUnitId, OffloadUnit>,
) -> Result<BTreeMap<(OffloadUnitId, String), (OffloadUnitId, String)>, ResidencyControllerError> {
    type Location = (OffloadUnitId, String);
    let mut identities = BTreeMap::<String, Vec<Location>>::new();
    let mut aliases = BTreeMap::<Location, String>::new();
    let mut bytes = BTreeMap::<Location, u64>::new();
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
        location: &Location,
        identities: &BTreeMap<String, Vec<Location>>,
        aliases: &BTreeMap<Location, String>,
        visiting: &mut BTreeSet<Location>,
    ) -> Result<Location, ResidencyControllerError> {
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
    /// A binding alias graph was invalid.
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
    /// Checked byte arithmetic overflowed.
    #[error("residency arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: &'static str,
    },
}

/// Backend-independent operations required by ordered residency windows.
pub trait ResidencyWindowManager {
    /// Manager-specific failure including neutral window validation failures.
    type Error: std::error::Error + From<ResidencyWindowError>;

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
        let desired = self.desired(current).map_err(M::Error::from)?;
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
        let desired = self.window.desired(current).map_err(M::Error::from)?;
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
                    .map_err(M::Error::from)?;
            }
            if unit.device_resident() {
                device_bytes = device_bytes
                    .checked_add(unit.device_allocated_bytes())
                    .ok_or(ResidencyWindowError::ArithmeticOverflow {
                        context: "execution group device bytes",
                    })
                    .map_err(M::Error::from)?;
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

impl OffloadUnit {
    /// Creates a non-empty unit and sorts bindings by local name.
    pub fn new(
        id: OffloadUnitId,
        bindings: impl IntoIterator<Item = WeightBinding>,
    ) -> Result<Self, ResidencyDeclarationError> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        if bindings.is_empty() {
            return Err(ResidencyDeclarationError::EmptyUnit { id });
        }
        bindings.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = bindings
            .windows(2)
            .find(|pair| pair[0].name == pair[1].name)
        {
            return Err(ResidencyDeclarationError::DuplicateBindingName {
                id,
                name: pair[0].name.clone(),
            });
        }
        Ok(Self { id, bindings })
    }

    /// Returns the plan identifier for this unit.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns bindings in stable local-name order.
    pub fn bindings(&self) -> &[WeightBinding] {
        &self.bindings
    }
}

fn validate_name(name: String) -> Result<String, ResidencyDeclarationError> {
    if name.trim().is_empty() {
        Err(ResidencyDeclarationError::InvalidBindingName)
    } else {
        Ok(name)
    }
}

fn validate_size(name: &str, expected_bytes: u64) -> Result<(), ResidencyDeclarationError> {
    if expected_bytes == 0 {
        Err(ResidencyDeclarationError::ZeroSizedBinding {
            name: name.to_owned(),
        })
    } else {
        Ok(())
    }
}

fn first_source(
    name: &str,
    recipe: &DerivedWeightRecipe,
) -> Result<String, ResidencyDeclarationError> {
    recipe
        .source_keys()
        .first()
        .map(|key| (*key).to_owned())
        .ok_or_else(|| ResidencyDeclarationError::EmptyRecipeSources {
            name: name.to_owned(),
        })
}

/// Invalid backend-neutral residency declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ResidencyDeclarationError {
    /// A binding name was empty.
    #[error("weight binding names must not be empty")]
    InvalidBindingName,
    /// A binding checkpoint key was empty.
    #[error("weight binding {name:?} has an empty checkpoint key")]
    InvalidCheckpointKey {
        /// Invalid local name.
        name: String,
    },
    /// A recipe had no physical source.
    #[error("weight binding {name:?} has no checkpoint recipe source")]
    EmptyRecipeSources {
        /// Invalid local name.
        name: String,
    },
    /// A binding declared no bytes.
    #[error("weight binding {name:?} must contain at least one byte")]
    ZeroSizedBinding {
        /// Invalid local name.
        name: String,
    },
    /// A unit had no bindings.
    #[error("residency unit {id} must contain at least one binding")]
    EmptyUnit {
        /// Unit identifier.
        id: OffloadUnitId,
    },
    /// Two bindings in one unit had the same local name.
    #[error("residency unit {id} has duplicate binding name {name:?}")]
    DuplicateBindingName {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Duplicated local name.
        name: String,
    },
    /// A general binding list repeated one logical identity.
    #[error("duplicate logical weight binding {name:?}")]
    DuplicateLogicalBinding {
        /// Repeated logical identity.
        name: String,
    },
    /// An alias named no binding in its atomic unit.
    #[error("weight binding alias {alias:?} has unknown owner {owner:?}")]
    UnknownBindingAliasOwner {
        /// Alias identity.
        alias: String,
        /// Missing owner identity.
        owner: String,
    },
    /// An alias destination did not identify one unique global owner.
    #[error("weight binding alias {alias:?} has ambiguous owner {owner:?}")]
    AmbiguousBindingAliasOwner {
        /// Alias identity.
        alias: String,
        /// Ambiguous owner identity.
        owner: String,
    },
    /// Alias declarations formed a cycle.
    #[error("weight binding alias cycle contains {name:?}")]
    BindingAliasCycle {
        /// One member of the cycle.
        name: String,
    },
    /// Alias and resolved owner disagreed on materialized byte geometry.
    #[error("weight binding alias {alias:?} declares {alias_bytes} bytes but owner {owner:?} declares {owner_bytes}")]
    BindingAliasByteMismatch {
        /// Alias identity.
        alias: String,
        /// Resolved canonical owner.
        owner: String,
        /// Alias byte declaration.
        alias_bytes: u64,
        /// Owner byte declaration.
        owner_bytes: u64,
    },
    /// An alias was incorrectly rewritten with a physical source.
    #[error("weight binding alias {name:?} cannot own a physical checkpoint source")]
    AliasHasPhysicalSource {
        /// Alias identity.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        sync::{Arc, Mutex},
    };

    use eredu_checkpoint::{store::TensorMetadata, StoredDtype};
    use eredu_core::residency::{MemoryTier, OffloadConfig, OffloadUnitSpec, ResidencyPolicy};

    use super::*;

    struct Catalog(BTreeMap<String, TensorMetadata>);

    struct TestLeaseStorage(BTreeMap<String, u32>);

    impl ResidencyLeaseStorage for TestLeaseStorage {
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

    impl ResidencyLeaseOwner for TestLeaseOwner {
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
        type Error = ResidencyWindowError;

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
        let lease = ResidencyLease::new(
            id.clone(),
            MemoryTier::Device,
            TestLeaseStorage(BTreeMap::from([("weight".into(), 7)])),
            Arc::downgrade(&owner),
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
