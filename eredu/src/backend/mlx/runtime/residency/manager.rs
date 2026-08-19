//! MLX materialization and transfer execution for immutable weight units.
//!
//! A [`crate::backend::mlx::runtime::residency::manager::ResidencyManager`] moves caller-defined groups of
//! checkpoint selections from a [`crate::backend::mlx::runtime::checkpoint::store::WeightStore`] into
//! immutable typed host-transfer buffers or execution-stream arrays. The
//! manager accounts for logical host and device copies independently, even on
//! unified-memory systems.
//! Missing units can be reserved and submitted as one batch. Caller-owned
//! [`crate::backend::mlx::runtime::residency::manager::ResidentTransfer`] values retain
//! source mappings until MLX reports exact completion of the submitted
//! transfer.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Condvar, Mutex, MutexGuard, Weak},
    time::Instant,
};

use safemlx::{
    host_transfer_capacity_upper_bound,
    transforms::{async_eval_with_event, eval},
    Array, DeviceType, Event, HostTransferBuffer, HostTransferPolicy, ImmutableHostTransferBuffer,
    Stream,
};

use crate::{
    backend::mlx::residency::sample_allocator_memory,
    backend::mlx::runtime::checkpoint::recipe::{
        DerivedWeightRecipe, MlxWeightRecipeExt, WeightRecipeError,
    },
    backend::mlx::runtime::checkpoint::store::{
        PendingWeightMaterialization, TensorSelection, WeightReadPolicy, WeightStore,
        WeightStoreDiagnostics, WeightStoreError,
    },
    core::residency::{
        EvictedResidencyCopy, MemoryTier, OffloadPlan, OffloadReport, OffloadUnitId,
        OffloadUnitSpec, PrefetchOutcome, ResidencyLedger, ResidencyLedgerError, TransferDirection,
        UnitResidencyReport,
    },
};

/// One named checkpoint selection within an atomic resident unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightBinding {
    name: String,
    logical_target: Option<String>,
    checkpoint_key: String,
    selection: TensorSelection,
    recipe: Option<DerivedWeightRecipe>,
    expected_bytes: u64,
}

impl WeightBinding {
    /// Creates a binding with a stable local name and expected selected size.
    pub fn new(
        name: impl Into<String>,
        checkpoint_key: impl Into<String>,
        selection: TensorSelection,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ResidencyError::InvalidBindingName);
        }
        let checkpoint_key = checkpoint_key.into();
        if checkpoint_key.trim().is_empty() {
            return Err(ResidencyError::InvalidCheckpointKey { name });
        }
        if expected_bytes == 0 {
            return Err(ResidencyError::ZeroSizedBinding { name });
        }
        Ok(Self {
            name,
            logical_target: None,
            checkpoint_key,
            selection,
            recipe: None,
            expected_bytes,
        })
    }

    /// Creates a binding backed by a composable derived-weight recipe.
    ///
    /// The recipe is validated against checkpoint metadata when the residency
    /// manager is constructed and materialized once on the host during
    /// initialization. Device promotion copies that transformed representation.
    pub fn from_recipe(
        name: impl Into<String>,
        recipe: DerivedWeightRecipe,
        expected_bytes: u64,
    ) -> Result<Self, ResidencyError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ResidencyError::InvalidBindingName);
        }
        let checkpoint_key = recipe
            .source_keys()
            .first()
            .map(|key| (*key).to_string())
            .ok_or_else(|| ResidencyError::Recipe {
                binding: name.clone(),
                source: WeightRecipeError::EmptyInputs,
            })?;
        if expected_bytes == 0 {
            return Err(ResidencyError::ZeroSizedBinding { name });
        }
        Ok(Self {
            name,
            logical_target: None,
            checkpoint_key,
            selection: TensorSelection::Full,
            recipe: Some(recipe),
            expected_bytes,
        })
    }

    /// Returns the stable name used to look up a resident array.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the architecture-logical parameter target, when it differs
    /// from the physical checkpoint source selected by this binding.
    pub(crate) fn logical_target(&self) -> Option<&str> {
        self.logical_target.as_deref()
    }

    /// Attaches the architecture-logical destination used by structural and
    /// distributed placement plans.
    pub(crate) fn with_logical_target(
        mut self,
        target: impl Into<String>,
    ) -> Result<Self, ResidencyError> {
        let target = target.into();
        if target.trim().is_empty() {
            return Err(ResidencyError::InvalidBindingName);
        }
        self.logical_target = Some(target);
        Ok(self)
    }

    /// Returns the source checkpoint key.
    pub fn checkpoint_key(&self) -> &str {
        &self.checkpoint_key
    }

    /// Returns the checkpoint selection delegated to the weight store.
    pub fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Returns the derived recipe when this is not a direct binding.
    pub const fn recipe(&self) -> Option<&DerivedWeightRecipe> {
        self.recipe.as_ref()
    }

    /// Returns the complete semantic source recipe represented by this binding.
    pub(crate) fn source_recipe(&self) -> DerivedWeightRecipe {
        self.recipe.clone().unwrap_or_else(|| {
            DerivedWeightRecipe::source(self.checkpoint_key.clone(), self.selection.clone())
        })
    }

    /// Returns every checkpoint key consumed by this binding.
    pub fn checkpoint_keys(&self) -> Vec<&str> {
        match &self.recipe {
            Some(recipe) => recipe.source_keys(),
            None => vec![self.checkpoint_key.as_str()],
        }
    }

    /// Returns the expected logical and materialized byte length.
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }
    /// Pushes an output selection through this binding's direct source or
    /// derived recipe before residency initialization.
    pub(crate) fn select_bounded_output(
        mut self,
        store: &dyn WeightStore,
        selection: TensorSelection,
    ) -> Result<Self, ResidencyError> {
        let recipe = self.recipe.clone().unwrap_or_else(|| {
            DerivedWeightRecipe::source(self.checkpoint_key.clone(), self.selection.clone())
        });
        let recipe =
            recipe
                .select_bounded(store, selection)
                .map_err(|source| ResidencyError::Recipe {
                    binding: self.name.clone(),
                    source,
                })?;
        let metadata = recipe
            .infer(store)
            .map_err(WeightRecipeError::from)
            .map_err(|source| ResidencyError::Recipe {
                binding: self.name.clone(),
                source,
            })?;
        self.checkpoint_key = recipe
            .source_keys()
            .first()
            .map(|key| (*key).to_string())
            .ok_or_else(|| ResidencyError::Recipe {
                binding: self.name.clone(),
                source: WeightRecipeError::EmptyInputs,
            })?;
        self.selection = TensorSelection::Full;
        self.recipe = Some(recipe);
        self.expected_bytes = metadata.byte_len();
        Ok(self)
    }
}

/// A deterministic group of weight bindings managed as one atomic unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OffloadUnit {
    id: OffloadUnitId,
    bindings: Vec<WeightBinding>,
}

impl OffloadUnit {
    /// Creates a non-empty unit and sorts its bindings by local name.
    pub fn new(
        id: OffloadUnitId,
        bindings: impl IntoIterator<Item = WeightBinding>,
    ) -> Result<Self, ResidencyError> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        if bindings.is_empty() {
            return Err(ResidencyError::EmptyUnit { id });
        }
        bindings.sort_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = bindings
            .windows(2)
            .find(|pair| pair[0].name == pair[1].name)
        {
            return Err(ResidencyError::DuplicateBindingName {
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

/// A resident unit that prevents eviction of one tier until it is dropped.
pub struct ResidentUnitLease {
    id: OffloadUnitId,
    tier: MemoryTier,
    storage: ResidentLeaseStorage,
    manager: Weak<ManagerInner>,
}

enum ResidentLeaseStorage {
    Host(Arc<ResidentHostBuffers>),
    Device(Arc<ResidentArrays>),
}

impl ResidentUnitLease {
    /// Returns the acquired unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the protected resident tier.
    pub const fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Looks up an immutable resident array by stable binding name.
    ///
    /// Consumers should not retain cloned `Array` handles beyond this lease if
    /// residency accounting is expected to remain authoritative. Arbitrary
    /// external array clones cannot be tracked by the manager.
    pub fn array(&self, name: &str) -> Result<&Array, ResidencyError> {
        match &self.storage {
            ResidentLeaseStorage::Device(arrays) => {
                arrays
                    .arrays
                    .get(name)
                    .ok_or_else(|| ResidencyError::UnknownBinding {
                        id: self.id.clone(),
                        name: name.to_string(),
                    })
            }
            ResidentLeaseStorage::Host(_) => Err(ResidencyError::HostBindingIsNotArray {
                id: self.id.clone(),
                name: name.to_string(),
            }),
        }
    }

    /// Looks up an immutable typed host-transfer buffer by stable binding name.
    ///
    /// Host-resident weights are deliberately not MLX arrays. Device leases
    /// reject this accessor so physical storage and execution residency cannot
    /// be confused by callers.
    pub fn host_buffer(&self, name: &str) -> Result<&ImmutableHostTransferBuffer, ResidencyError> {
        match &self.storage {
            ResidentLeaseStorage::Host(buffers) => {
                buffers
                    .buffers
                    .get(name)
                    .ok_or_else(|| ResidencyError::UnknownBinding {
                        id: self.id.clone(),
                        name: name.to_string(),
                    })
            }
            ResidentLeaseStorage::Device(_) => Err(ResidencyError::DeviceBindingIsNotHostBuffer {
                id: self.id.clone(),
                name: name.to_string(),
            }),
        }
    }

    /// Returns binding names in stable order.
    pub fn binding_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        match &self.storage {
            ResidentLeaseStorage::Host(buffers) => {
                Box::new(buffers.buffers.keys().map(String::as_str))
            }
            ResidentLeaseStorage::Device(arrays) => {
                Box::new(arrays.arrays.keys().map(String::as_str))
            }
        }
    }
}

impl Drop for ResidentUnitLease {
    fn drop(&mut self) {
        let Some(manager) = self.manager.upgrade() else {
            return;
        };
        let Ok(mut state) = manager.state.lock() else {
            return;
        };
        state.ledger.unpin(&self.id, self.tier);
    }
}

/// Caller-owned completion and source-lifetime guard for one residency batch.
///
/// Missing copies are submitted together. This value owns both the resident
/// unit leases and the transfer completion, so pins cannot be released before
/// the submitted copy is safe. Its unit leases may be used to construct
/// dependent work after [`Self::wait_on`] orders the consumer stream. The guard
/// retains mmap leases, source arrays, host buffers, and their copy events until
/// the aggregate event completes. Dropping an unfinished guard waits for that exact event; it never
/// synchronizes an entire MLX stream.
///
/// `ResidentTransfer` is intentionally neither `Send` nor `Sync`, matching the
/// underlying [`Event`] contract. The shared [`ResidencyManager`] remains safe
/// to use from multiple host threads; a competing synchronous acquisition
/// waits for the caller which owns the in-flight transfer to publish it.
pub struct ResidentTransfer {
    leases: Vec<ResidentUnitLease>,
    event: Option<Event>,
    retained: Option<ResidentTransferResources>,
    manager: Weak<ManagerInner>,
    ids: Vec<OffloadUnitId>,
    tier: MemoryTier,
    generation: u64,
}

impl ResidentTransfer {
    /// Returns the resident unit leases protected by this transfer.
    pub fn leases(&self) -> &[ResidentUnitLease] {
        &self.leases
    }

    /// Returns whether this transfer contains no resident units.
    pub fn is_empty(&self) -> bool {
        self.leases.is_empty()
    }

    /// Orders subsequently evaluated work on a compatible stream after this transfer.
    ///
    /// MLX operations remain lazy: call this before evaluating the consumer
    /// graph. This inserts a backend dependency and does not block the host.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), ResidencyError> {
        if let Some(event) = &self.event {
            event
                .wait_on(stream)
                .map_err(|source| ResidencyError::Mlx {
                    id: self.ids.first().cloned().unwrap_or_else(internal_id),
                    operation: "resident transfer stream wait",
                    source,
                })?;
        }
        Ok(())
    }

    /// Blocks for this exact transfer and publishes its copies as ready.
    ///
    /// An asynchronous failure rolls the submitted copies back and remains
    /// observable on repeated calls.
    pub fn synchronize(&mut self) -> Result<(), ResidencyError> {
        self.finish(true)
    }

    /// Returns whether the transfer has completed without blocking.
    pub fn is_complete(&self) -> Result<bool, ResidencyError> {
        self.event.as_ref().map_or(Ok(true), |event| {
            event.is_complete().map_err(|source| ResidencyError::Mlx {
                id: self.ids.first().cloned().unwrap_or_else(internal_id),
                operation: "resident transfer query",
                source,
            })
        })
    }

    fn finish(&mut self, report_error: bool) -> Result<(), ResidencyError> {
        let result = self.event.as_ref().map_or(Ok(()), Event::synchronize);
        let succeeded = result.is_ok();
        if let Some(resources) = self.retained.take() {
            if succeeded {
                for source in resources.sources {
                    source.complete();
                }
                drop(resources.retained_arrays);
                drop(resources.retained_host);
                drop(resources.retained_events);
            } else {
                // Preserve PendingWeightMaterialization's conservative failure
                // cleanup: it synchronizes the involved streams and retains a
                // mapping permanently if backend state is unknowable.
                drop(resources);
            }
        }
        // Retain a failed event so repeated synchronization and queries keep
        // reporting the original asynchronous backend error. Successful
        // events can release their backend resource after publication.
        if succeeded {
            self.event = None;
        }
        self.publish(succeeded)?;
        match result {
            Ok(()) => Ok(()),
            Err(source) if report_error => Err(ResidencyError::Mlx {
                id: self.ids.first().cloned().unwrap_or_else(internal_id),
                operation: "resident transfer completion",
                source,
            }),
            Err(_) => Ok(()),
        }
    }

    fn publish(&mut self, succeeded: bool) -> Result<(), ResidencyError> {
        if self.generation == 0 {
            return Ok(());
        }
        let generation = std::mem::take(&mut self.generation);
        let Some(manager) = self.manager.upgrade() else {
            return Ok(());
        };
        let mut state = manager
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        let removed = state
            .ledger
            .resolve_transfer(&self.ids, self.tier, generation, succeeded)?;
        release_backend_copies(&mut state, &removed)?;
        manager.changed.notify_all();
        Ok(())
    }
}

impl Drop for ResidentTransfer {
    fn drop(&mut self) {
        let _ = self.finish(false);
    }
}

/// Structured failures from residency validation and state transitions.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    /// Backend-neutral ownership or capacity transition failed.
    #[error(transparent)]
    Ledger(#[from] ResidencyLedgerError),
    /// An ordered layer window had no units.
    #[error("device layer window requires at least one ordered unit")]
    EmptyLayerWindow,
    /// The configured device layer window exceeded the ordered unit count.
    #[error("device layer window depth {depth} exceeds {layer_count} ordered units")]
    OversizedLayerWindow {
        /// Requested resident-layer bound.
        depth: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A layer index was outside the ordered sequence.
    #[error("device layer index {index} is outside {layer_count} ordered units")]
    InvalidLayerIndex {
        /// Requested index.
        index: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A binding name was empty or whitespace-only.
    #[error("weight binding names must not be empty")]
    InvalidBindingName,
    /// A binding checkpoint key was empty.
    #[error("weight binding {name:?} has an empty checkpoint key")]
    InvalidCheckpointKey {
        /// Invalid local binding name.
        name: String,
    },
    /// A binding declared no bytes.
    #[error("weight binding {name:?} must contain at least one byte")]
    ZeroSizedBinding {
        /// Invalid local binding name.
        name: String,
    },
    /// A unit had no bindings.
    #[error("residency unit {id} must contain at least one binding")]
    EmptyUnit {
        /// Invalid unit identifier.
        id: OffloadUnitId,
    },
    /// Two bindings in one unit had the same local name.
    #[error("residency unit {id} has duplicate binding name {name:?}")]
    DuplicateBindingName {
        /// Invalid unit identifier.
        id: OffloadUnitId,
        /// Duplicated local name.
        name: String,
    },
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
    /// A binding's selected checkpoint size contradicted its definition.
    #[error("binding {binding:?} in unit {id} selects {actual_bytes} bytes but declares {expected_bytes}")]
    BindingByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Binding name.
        binding: String,
        /// Declared size.
        expected_bytes: u64,
        /// Store-validated size.
        actual_bytes: u64,
    },
    /// A derived-weight recipe was invalid or could not be materialized.
    #[error("derived-weight recipe for binding {binding:?} failed: {source}")]
    Recipe {
        /// Local binding name.
        binding: String,
        /// Recipe failure.
        #[source]
        source: WeightRecipeError,
    },
    /// The configured source stream was not a CPU stream.
    #[error("the residency source stream must target the CPU")]
    InvalidSourceStream,
    /// A binding lookup failed on a valid resident unit.
    #[error("residency unit {id} has no binding named {name:?}")]
    UnknownBinding {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Unknown local name.
        name: String,
    },
    /// A caller requested an executable array from typed host storage.
    #[error(
        "host-resident binding {name:?} in unit {id} is a typed transfer buffer, not an MLX array"
    )]
    HostBindingIsNotArray {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Requested binding name.
        name: String,
    },
    /// A caller requested typed host storage from a device-resident copy.
    #[error(
        "device-resident binding {name:?} in unit {id} is an MLX array, not a host-transfer buffer"
    )]
    DeviceBindingIsNotHostBuffer {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Requested binding name.
        name: String,
    },
    /// A backend allocated beyond its advertised pre-allocation capacity bound.
    #[error(
        "host-transfer allocation for residency unit {id} used {actual_bytes} bytes, exceeding reserved upper bound {reserved_bytes}"
    )]
    HostCapacityBoundExceeded {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Capacity reserved before materialization.
        reserved_bytes: u64,
        /// Exact allocated capacity.
        actual_bytes: u64,
    },
    /// Checked byte or recency arithmetic overflowed.
    #[error("residency arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: &'static str,
    },
    /// Persistent store validation or materialization failed.
    #[error(transparent)]
    WeightStore(#[from] WeightStoreError),
    /// An MLX copy or evaluation failed.
    #[error("MLX {operation} failed for residency unit {id}: {source}")]
    Mlx {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Failed operation.
        operation: &'static str,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// Serialized manager state was poisoned by a prior panic.
    #[error("residency manager state is poisoned")]
    StatePoisoned,
}

/// Immutable manager, telemetry, and store diagnostic snapshot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyReport {
    initialized: bool,
    offload: OffloadReport,
    units: Vec<UnitResidencyReport>,
    active_window: Vec<OffloadUnitId>,
    weight_store: WeightStoreDiagnostics,
    materialization: Option<
        crate::backend::mlx::runtime::checkpoint::bounded_quantization::BoundedQuantizationReport,
    >,
}

impl ResidencyReport {
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
    pub const fn materialization(
        &self,
    ) -> Option<
        &crate::backend::mlx::runtime::checkpoint::bounded_quantization::BoundedQuantizationReport,
    > {
        self.materialization.as_ref()
    }

    pub(crate) fn with_materialization(
        mut self,
        materialization: Option<
            crate::backend::mlx::runtime::checkpoint::bounded_quantization::BoundedQuantizationReport,
        >,
    ) -> Self {
        self.materialization = materialization;
        self
    }
}

/// Serialized, shareable manager for immutable checkpoint weight residency.
#[derive(Clone)]
pub struct ResidencyManager {
    inner: Arc<ManagerInner>,
}

/// Deterministic controller for a bounded ordered device-layer window.
///
/// The current layer counts toward `depth`. Preparation is synchronous and
/// explicit trimming is performed even when the manager has an unlimited
/// device budget, so stale decoder copies cannot accumulate.
#[derive(Debug, Clone)]
pub struct DeviceLayerWindow {
    units: Vec<OffloadUnitId>,
    depth: usize,
}

/// A named sequential execution stack with an independent device window.
///
/// Models with text, vision, audio, temporal, or depth-transformer stacks can
/// use one group per ordered stack without imposing a checkpoint naming scheme
/// on the residency core.
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
    ) -> Result<Self, ResidencyError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ResidencyLedgerError::InvalidGroupId.into());
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

    /// Synchronously prepares this group's window without replacing another group's window.
    pub fn prepare(
        &self,
        manager: &ResidencyManager,
        current: usize,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        let desired = self.window.desired(current)?;
        let outcomes =
            manager.prepare_group_window(&self.id, desired, desired, MemoryTier::Device)?;
        self.window.trim_to(manager, desired)?;
        Ok(outcomes)
    }

    /// Trims this group to the desired window.
    pub fn trim_to(
        &self,
        manager: &ResidencyManager,
        desired: &[OffloadUnitId],
    ) -> Result<(), ResidencyError> {
        self.window.trim_to(manager, desired)
    }

    /// Clears only this group's protection and device copies.
    pub fn clear(&self, manager: &ResidencyManager) -> Result<(), ResidencyError> {
        manager.prepare_group_window(&self.id, &[], &[], MemoryTier::Device)?;
        self.window.trim_to(manager, &[])
    }

    /// Returns current logical residency attributed to this group's units.
    pub fn report(
        &self,
        manager: &ResidencyManager,
    ) -> Result<ResidentLayerGroupReport, ResidencyError> {
        let report = manager.report()?;
        let ids = self.units().iter().collect::<BTreeSet<_>>();
        let mut host_bytes = 0u64;
        let mut device_bytes = 0u64;
        let mut device_units = 0usize;
        for unit in report.units().iter().filter(|unit| ids.contains(unit.id())) {
            if unit.host_resident() {
                host_bytes = host_bytes.checked_add(unit.host_allocated_bytes()).ok_or(
                    ResidencyError::ArithmeticOverflow {
                        context: "execution group host bytes",
                    },
                )?;
            }
            if unit.device_resident() {
                device_bytes = device_bytes
                    .checked_add(unit.device_allocated_bytes())
                    .ok_or(ResidencyError::ArithmeticOverflow {
                        context: "execution group device bytes",
                    })?;
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

impl DeviceLayerWindow {
    /// Creates a controller for a non-empty ordered unit sequence.
    pub fn new(
        units: impl IntoIterator<Item = OffloadUnitId>,
        depth: usize,
    ) -> Result<Self, ResidencyError> {
        let units = units.into_iter().collect::<Vec<_>>();
        if units.is_empty() {
            return Err(ResidencyError::EmptyLayerWindow);
        }
        if depth == 0 || depth > units.len() {
            return Err(ResidencyError::OversizedLayerWindow {
                depth,
                layer_count: units.len(),
            });
        }
        let unique = units.iter().collect::<BTreeSet<_>>();
        if unique.len() != units.len() {
            return Err(ResidencyError::DuplicateUnitDefinition {
                id: units
                    .iter()
                    .find(|id| units.iter().filter(|candidate| *candidate == *id).count() > 1)
                    .expect("duplicate exists")
                    .clone(),
            });
        }
        Ok(Self { units, depth })
    }

    /// Returns the maximum number of decoder units kept on the device.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns decoder units in execution order.
    pub fn units(&self) -> &[OffloadUnitId] {
        &self.units
    }

    /// Returns the desired window beginning at `current`.
    pub fn desired(&self, current: usize) -> Result<&[OffloadUnitId], ResidencyError> {
        if current >= self.units.len() {
            return Err(ResidencyError::InvalidLayerIndex {
                index: current,
                layer_count: self.units.len(),
            });
        }
        let end = current.saturating_add(self.depth).min(self.units.len());
        Ok(&self.units[current..end])
    }

    /// Synchronously prepares and trims the window beginning at `current`.
    pub fn prepare(
        &self,
        manager: &ResidencyManager,
        current: usize,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        let desired = self.desired(current)?;
        let outcomes = manager.prepare_window(desired, desired, MemoryTier::Device)?;
        self.trim_to(manager, desired)?;
        Ok(outcomes)
    }

    /// Explicitly evicts every managed device copy outside `desired`.
    pub fn trim_to(
        &self,
        manager: &ResidencyManager,
        desired: &[OffloadUnitId],
    ) -> Result<(), ResidencyError> {
        let desired = desired.iter().collect::<BTreeSet<_>>();
        for id in &self.units {
            if !desired.contains(id) {
                manager.evict(id, MemoryTier::Device)?;
            }
        }
        Ok(())
    }

    /// Clears protection and removes every managed device-layer copy.
    pub fn clear(&self, manager: &ResidencyManager) -> Result<(), ResidencyError> {
        manager.prepare_window(&[], &[], MemoryTier::Device)?;
        self.trim_to(manager, &[])
    }
}

impl ResidencyManager {
    /// Returns the MLX stream index used for device residency transfers.
    pub(crate) fn device_stream_index(&self) -> Result<i32, ResidencyError> {
        self.lock()?
            .device_stream
            .get_index()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "device residency stream index",
                source,
            })
    }

    /// Validates plan/unit identity, binding sizes, selections, and streams.
    ///
    /// Construction does not create MLX arrays. Call [`Self::initialize`] to
    /// materialize units assigned to host or device by the plan.
    pub fn new<S>(
        store: Arc<S>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError>
    where
        S: WeightStore + Send + Sync + 'static,
    {
        let store: Arc<dyn WeightStore + Send + Sync> = store;
        Self::new_shared(store, plan, units, source_stream, device_stream)
    }

    /// Creates a manager from an already type-erased checkpoint store.
    pub fn new_shared(
        store: Arc<dyn WeightStore + Send + Sync>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError> {
        let source_device = source_stream
            .get_device()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source stream inspection",
                source,
            })?;
        if source_device
            .get_type()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source device inspection",
                source,
            })?
            != DeviceType::Cpu
        {
            return Err(ResidencyError::InvalidSourceStream);
        }

        let mut definitions = BTreeMap::new();
        for unit in units {
            let id = unit.id.clone();
            if definitions.insert(id.clone(), unit).is_some() {
                return Err(ResidencyError::DuplicateUnitDefinition { id });
            }
        }
        for spec in plan.units() {
            if !definitions.contains_key(spec.id()) {
                return Err(ResidencyError::MissingUnitDefinition {
                    id: spec.id().clone(),
                });
            }
        }
        if let Some(id) = definitions
            .keys()
            .find(|id| plan.unit(id).is_none())
            .cloned()
        {
            return Err(ResidencyError::UnexpectedUnitDefinition { id });
        }

        let mut records = BTreeMap::new();
        for spec in plan.units() {
            let definition = definitions.remove(spec.id()).expect("validated above");
            validate_unit_bytes(store.as_ref(), spec, &definition)?;
            records.insert(
                spec.id().clone(),
                UnitRecord {
                    definition,
                    host: None,
                    device: None,
                },
            );
        }

        let ledger = ResidencyLedger::new(plan);
        Ok(Self {
            inner: Arc::new(ManagerInner {
                store,
                state: Mutex::new(ManagerState {
                    ledger,
                    units: records,
                    source_stream,
                    device_stream,
                }),
                changed: Condvar::new(),
            }),
        })
    }

    /// Materializes all planned host and device units in identifier order.
    ///
    /// Disk units remain array-free. A failure never publishes a partial unit;
    /// units completed earlier remain resident and fully accounted, allowing a
    /// caller to inspect the report and retry initialization.
    pub fn initialize(&self) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if state.ledger.initialized() {
            return Ok(());
        }
        let assignments = state
            .ledger
            .plan()
            .units()
            .iter()
            .map(|unit| (unit.id().clone(), unit.tier()))
            .collect::<Vec<_>>();
        for (id, tier) in assignments {
            if tier != MemoryTier::Disk {
                ensure_resident(&mut state, self.inner.store.as_ref(), &id, tier)?;
            }
        }
        state.ledger.mark_initialized();
        Ok(())
    }

    /// Synchronously prepares one host or device copy and records hit/miss telemetry.
    ///
    /// This provides caller-directed lookahead but does not overlap transfer
    /// with computation.
    pub fn prefetch(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<PrefetchOutcome, ResidencyError> {
        validate_target(tier, "prefetch")?;
        let mut state = self.lock()?;
        loop {
            state.ledger.require_initialized()?;
            let copy = state.ledger.copy_status(id, tier)?;
            if !copy.is_some_and(|copy| copy.in_flight().is_some()) {
                break;
            }
            state = self
                .inner
                .changed
                .wait(state)
                .map_err(|_| ResidencyError::StatePoisoned)?;
        }
        prefetch_locked(&mut state, self.inner.store.as_ref(), id, tier)
    }

    /// Ensures residency and returns an RAII lease protecting the requested copy.
    pub fn acquire(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_with_demand(id, tier, 1)
    }

    /// Ensures residency and records route-weighted demand for eviction policy.
    ///
    /// `demand` may be larger than one when duplicate routed-expert requests
    /// share a single acquisition. Frequency counters saturate on overflow.
    pub fn acquire_with_demand(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        demand: u64,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_many_with_demand(&[(id.clone(), demand)], tier)?
            .pop()
            .ok_or(ResidencyError::StatePoisoned)
    }

    /// Acquires a deterministic expert set with one batched residency transition.
    ///
    /// Missing copies reserve capacity before any materialization starts. All
    /// requested units are protected from eviction, all lazy outputs are
    /// evaluated together, and leases are published only after the batch is
    /// complete.
    pub fn acquire_many_with_demand(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Vec<ResidentUnitLease>, ResidencyError> {
        self.acquire_many_with_mode(requests, tier, false)
            .map(|(leases, _)| leases)
    }

    /// Submits one residency batch and returns its owning completion lease.
    ///
    /// Missing copies are submitted to MLX for asynchronous evaluation, but
    /// this method does not block the host for their completion. Call
    /// [`ResidentTransfer::wait_on`] before evaluating work on another
    /// compatible stream. The transfer guard owns every source dependency
    /// until it is synchronized or dropped.
    pub fn acquire_many_with_transfer(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<ResidentTransfer, ResidencyError> {
        let (leases, submitted) = self.acquire_many_with_mode(requests, tier, true)?;
        let transfer = match submitted {
            None => ResidentTransfer {
                leases,
                event: None,
                retained: None,
                manager: Weak::new(),
                ids: Vec::new(),
                tier,
                generation: 0,
            },
            Some(submitted) => ResidentTransfer {
                leases,
                event: Some(submitted.event),
                retained: Some(submitted.retained),
                manager: Arc::downgrade(&self.inner),
                ids: submitted.ids,
                tier,
                generation: submitted.generation,
            },
        };
        Ok(transfer)
    }

    fn acquire_many_with_mode(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        return_transfer: bool,
    ) -> Result<(Vec<ResidentUnitLease>, Option<SubmittedResidentTransfer>), ResidencyError> {
        validate_target(tier, "acquire")?;
        let mut state = self.lock()?;
        let ids = requests
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        state.ledger.validate_batch(&ids, tier)?;
        loop {
            state.ledger.require_initialized()?;
            for (id, _) in requests {
                state.ledger.spec(id)?;
            }
            let waiting = requests.iter().any(|(id, _)| {
                state
                    .ledger
                    .copy_status(id, tier)
                    .ok()
                    .flatten()
                    .is_some_and(|copy| copy.in_flight().is_some())
            });
            if !waiting {
                break;
            }
            state = self
                .inner
                .changed
                .wait(state)
                .map_err(|_| ResidencyError::StatePoisoned)?;
        }
        let missing = requests
            .iter()
            .filter(|(id, _)| !state.ledger.is_resident(id, tier).unwrap_or(false))
            .count();
        let started = Instant::now();
        let residency = ensure_many_resident(
            &mut state,
            self.inner.store.as_ref(),
            &ids,
            tier,
            return_transfer,
        );
        if missing > 0 {
            state.ledger.record_prefetch_stall(started.elapsed());
        }
        let (_, submitted) = residency?;
        let leases = requests
            .iter()
            .map(|(id, demand)| {
                state.ledger.pin(id, tier, *demand)?;
                let unit = state.units.get(id).ok_or(ResidencyError::StatePoisoned)?;
                let storage = match tier {
                    MemoryTier::Host => ResidentLeaseStorage::Host(Arc::clone(
                        unit.host.as_ref().ok_or(ResidencyError::StatePoisoned)?,
                    )),
                    MemoryTier::Device => ResidentLeaseStorage::Device(Arc::clone(
                        unit.device.as_ref().ok_or(ResidencyError::StatePoisoned)?,
                    )),
                    MemoryTier::Disk => unreachable!("validated above"),
                };
                Ok(ResidentUnitLease {
                    id: id.clone(),
                    tier,
                    storage,
                    manager: Arc::downgrade(&self.inner),
                })
            })
            .collect::<Result<Vec<_>, ResidencyError>>()?;
        Ok((leases, submitted))
    }

    /// Returns whether a logical copy currently resides in a memory tier.
    pub fn is_resident(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<bool, ResidencyError> {
        validate_target(tier, "is_resident")?;
        let state = self.lock()?;
        Ok(state.ledger.is_resident(id, tier)?)
    }

    /// Replaces the protected window and synchronously prepares bounded lookahead.
    ///
    /// `active` units are protected from automatic eviction. At most the first
    /// configured number of distinct `upcoming` units are prefetched, in caller
    /// order. Repeated and overlapping windows are deterministic.
    pub fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        self.prepare_group_window("default", active, upcoming, tier)
    }

    /// Replaces one named group's protected window and prepares bounded lookahead.
    ///
    /// Protection owned by other groups remains active. This permits independent
    /// text, vision, audio, temporal, and depth stack scheduling.
    pub fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        validate_target(tier, "prepare_group_window")?;
        let mut state = self.lock()?;
        loop {
            state.ledger.require_initialized()?;
            for id in active.iter().chain(upcoming) {
                state.ledger.spec(id)?;
            }
            let waiting = active.iter().chain(upcoming).any(|id| {
                state
                    .ledger
                    .copy_status(id, tier)
                    .ok()
                    .flatten()
                    .is_some_and(|copy| copy.in_flight().is_some())
            });
            if !waiting {
                break;
            }
            state = self
                .inner
                .changed
                .wait(state)
                .map_err(|_| ResidencyError::StatePoisoned)?;
        }
        state.ledger.set_group_window(group, active, tier)?;
        let depth = state.ledger.plan().config().prefetch_depth();
        let mut seen = BTreeSet::new();
        let selected = upcoming
            .iter()
            .filter(|id| seen.insert((*id).clone()))
            .take(depth)
            .cloned()
            .collect::<Vec<_>>();
        selected
            .into_iter()
            .map(|id| {
                prefetch_locked(&mut state, self.inner.store.as_ref(), &id, tier)
                    .map(|outcome| (id, outcome))
            })
            .collect()
    }

    /// Replaces one named protected window without materializing its units.
    ///
    /// This is used by schedulers that submit materialization through a
    /// separate bounded service. Protection owned by other named windows is
    /// preserved.
    pub fn protect_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        state.ledger.require_initialized()?;
        for id in active {
            state.ledger.spec(id)?;
        }
        validate_target(tier, "protect_group_window")?;
        state.ledger.set_group_window(group, active, tier)?;
        Ok(())
    }

    /// Explicitly evicts one host or device copy.
    ///
    /// Evicting an absent copy is an idempotent success returning `false`.
    pub fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, ResidencyError> {
        validate_target(tier, "evict")?;
        let mut state = self.lock()?;
        let Some(evicted) = state.ledger.evict(id, tier)? else {
            return Ok(false);
        };
        if !state
            .units
            .get_mut(id)
            .is_some_and(|unit| unit.remove_storage(tier))
        {
            return Err(ResidencyError::StatePoisoned);
        }
        debug_assert_eq!(evicted.id, *id);
        Ok(true)
    }

    /// Samples optional MLX allocator and process metrics on explicit request.
    pub fn sample_memory(
        &self,
        include_mlx: bool,
        include_process: bool,
    ) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if include_mlx {
            let metrics = sample_allocator_memory().map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "allocator memory sampling",
                source,
            })?;
            state.ledger.record_allocator_memory(metrics);
        }
        if include_process {
            state.ledger.sample_process_metrics();
        }
        Ok(())
    }

    /// Returns an immutable point-in-time residency and storage report.
    pub fn report(&self) -> Result<ResidencyReport, ResidencyError> {
        let (initialized, offload, units, active_window) = self.telemetry_snapshot()?;
        Ok(ResidencyReport {
            initialized,
            offload,
            units,
            active_window,
            weight_store: self.inner.store.diagnostics()?,
            materialization: None,
        })
    }

    pub(crate) fn telemetry_snapshot(
        &self,
    ) -> Result<
        (
            bool,
            OffloadReport,
            Vec<UnitResidencyReport>,
            Vec<OffloadUnitId>,
        ),
        ResidencyError,
    > {
        let state = self.lock()?;
        let active = state.ledger.active_window();
        let units = state.ledger.unit_reports();
        Ok((
            state.ledger.initialized(),
            state.ledger.telemetry(),
            units,
            active.into_iter().collect(),
        ))
    }

    fn lock(&self) -> Result<MutexGuard<'_, ManagerState>, ResidencyError> {
        self.inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)
    }
}

struct ManagerInner {
    store: Arc<dyn WeightStore + Send + Sync>,
    state: Mutex<ManagerState>,
    changed: Condvar,
}

struct ManagerState {
    ledger: ResidencyLedger,
    units: BTreeMap<OffloadUnitId, UnitRecord>,
    source_stream: Stream,
    device_stream: Stream,
}

// SAFETY: every access to the MLX stream handles and resident arrays in this
// state is serialized by `ManagerInner::state`. No stream reference escapes
// the lock, and MLX operations use safemlx's runtime guard internally.
unsafe impl Send for ManagerState {}

struct UnitRecord {
    definition: OffloadUnit,
    host: Option<Arc<ResidentHostBuffers>>,
    device: Option<Arc<ResidentArrays>>,
}

impl UnitRecord {
    fn remove_storage(&mut self, tier: MemoryTier) -> bool {
        match tier {
            MemoryTier::Host => self.host.take().is_some(),
            MemoryTier::Device => self.device.take().is_some(),
            MemoryTier::Disk => false,
        }
    }
}

fn release_backend_copies(
    state: &mut ManagerState,
    copies: &[EvictedResidencyCopy],
) -> Result<(), ResidencyError> {
    for copy in copies {
        let removed = state
            .units
            .get_mut(&copy.id)
            .is_some_and(|unit| unit.remove_storage(copy.tier));
        if !removed {
            return Err(ResidencyError::StatePoisoned);
        }
    }
    Ok(())
}

struct ResidentArrays {
    arrays: BTreeMap<String, Array>,
}

struct ResidentHostBuffers {
    buffers: BTreeMap<String, ImmutableHostTransferBuffer>,
}

struct ResidentTransferResources {
    sources: Vec<PendingWeightMaterialization>,
    retained_arrays: Vec<Array>,
    retained_host: Vec<Arc<ResidentHostBuffers>>,
    retained_events: Vec<Event>,
}

struct SubmittedResidentTransfer {
    event: Event,
    retained: ResidentTransferResources,
    ids: Vec<OffloadUnitId>,
    generation: u64,
}

fn validate_unit_bytes(
    store: &dyn WeightStore,
    spec: &OffloadUnitSpec,
    unit: &OffloadUnit,
) -> Result<(), ResidencyError> {
    let mut total = 0u64;
    for binding in &unit.bindings {
        total = total.checked_add(binding.expected_bytes).ok_or(
            ResidencyError::ArithmeticOverflow {
                context: "unit binding byte total",
            },
        )?;
        let actual = match &binding.recipe {
            Some(recipe) => {
                recipe
                    .preflight_bounded(store)
                    .map_err(|source| ResidencyError::Recipe {
                        binding: binding.name.clone(),
                        source,
                    })?;
                recipe
                    .infer(store)
                    .map_err(WeightRecipeError::from)
                    .map_err(|source| ResidencyError::Recipe {
                        binding: binding.name.clone(),
                        source,
                    })?
                    .byte_len()
            }
            None => {
                let lease = store.acquire_with_policy(
                    &binding.checkpoint_key,
                    binding.selection.clone(),
                    WeightReadPolicy::RequireBounded,
                )?;
                u64::try_from(lease.selected_byte_len()).map_err(|_| {
                    ResidencyError::ArithmeticOverflow {
                        context: "selected binding byte conversion",
                    }
                })?
            }
        };
        if actual != binding.expected_bytes {
            return Err(ResidencyError::BindingByteMismatch {
                id: unit.id.clone(),
                binding: binding.name.clone(),
                expected_bytes: binding.expected_bytes,
                actual_bytes: actual,
            });
        }
    }
    if total != spec.bytes() {
        return Err(ResidencyError::UnitByteMismatch {
            id: unit.id.clone(),
            planned_bytes: spec.bytes(),
            actual_bytes: total,
        });
    }
    Ok(())
}

fn validate_target(tier: MemoryTier, operation: &'static str) -> Result<(), ResidencyError> {
    if tier == MemoryTier::Disk {
        Err(ResidencyLedgerError::InvalidTargetTier { operation }.into())
    } else {
        Ok(())
    }
}

fn internal_id() -> OffloadUnitId {
    OffloadUnitId::new("residency-manager").expect("static identifier is valid")
}

fn prefetch_locked(
    state: &mut ManagerState,
    store: &dyn WeightStore,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<PrefetchOutcome, ResidencyError> {
    let hit = state.ledger.is_resident(id, tier)?;
    let outcome = if hit {
        PrefetchOutcome::Hit
    } else {
        PrefetchOutcome::Miss
    };
    state.ledger.record_prefetch(tier, outcome);
    ensure_resident(state, store, id, tier)?;
    Ok(outcome)
}

fn ensure_resident(
    state: &mut ManagerState,
    store: &dyn WeightStore,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<bool, ResidencyError> {
    ensure_many_resident(state, store, std::slice::from_ref(id), tier, false)
        .map(|(created, _)| created[0])
}

fn ensure_many_resident(
    state: &mut ManagerState,
    store: &dyn WeightStore,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    return_transfer: bool,
) -> Result<(Vec<bool>, Option<SubmittedResidentTransfer>), ResidencyError> {
    validate_target(tier, "residency transition")?;
    if ids.is_empty() {
        return Ok((Vec::new(), None));
    }
    state.ledger.validate_batch(ids, tier)?;
    let created = ids
        .iter()
        .map(|id| state.ledger.is_resident(id, tier).map(|resident| !resident))
        .collect::<Result<Vec<_>, _>>()?;
    if created.iter().all(|value| !value) {
        for id in ids {
            state.ledger.touch(id, tier)?;
        }
        return Ok((created, None));
    }

    let temporary_protection = ids.iter().cloned().collect::<BTreeSet<_>>();
    let started = Instant::now();
    let mut reserved = Vec::new();
    let result = (|| {
        let mut reservations = Vec::new();
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                continue;
            }
            let required = resident_capacity_requirement(
                &state.units[id],
                state.ledger.spec(id)?.bytes(),
                tier,
            )?;
            reservations.push((id.clone(), required));
            reserved.push(id.clone());
        }
        let evicted = state
            .ledger
            .reserve_copies(&reservations, tier, &temporary_protection)?;
        release_backend_copies(state, &evicted)?;

        if tier == MemoryTier::Host {
            let mut prepared = Vec::new();
            for (id, is_missing) in ids.iter().zip(&created) {
                if !is_missing {
                    continue;
                }
                let bindings = state.units[id].definition.bindings.clone();
                let buffers = materialize_host_buffers(id, store, &bindings, &state.source_stream)?;
                let logical = host_buffers_nbytes(&buffers)?;
                let planned = state.ledger.spec(id)?.bytes();
                if logical != planned {
                    return Err(ResidencyError::UnitByteMismatch {
                        id: id.clone(),
                        planned_bytes: planned,
                        actual_bytes: logical,
                    });
                }
                let capacity = host_buffers_capacity(&buffers)?;
                let reserved_capacity = resident_capacity_requirement(
                    &state.units[id],
                    state.ledger.spec(id)?.bytes(),
                    tier,
                )?;
                if capacity > reserved_capacity {
                    return Err(ResidencyError::HostCapacityBoundExceeded {
                        id: id.clone(),
                        reserved_bytes: reserved_capacity,
                        actual_bytes: capacity,
                    });
                }
                prepared.push((id.clone(), buffers, logical, capacity, reserved_capacity));
            }
            for (id, buffers, logical, capacity, _) in prepared {
                state.ledger.publish_reserved(&id, tier, capacity, None)?;
                state
                    .units
                    .get_mut(&id)
                    .ok_or(ResidencyError::StatePoisoned)?
                    .host = Some(Arc::new(buffers));
                state.ledger.record_transfer(
                    TransferDirection::DiskToHost,
                    logical,
                    started.elapsed(),
                );
            }
            for (id, is_missing) in ids.iter().zip(&created) {
                if !is_missing {
                    state.ledger.touch(id, tier)?;
                }
            }
            return Ok((created.clone(), None));
        }

        let mut prepared = Vec::new();
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                continue;
            }
            let bindings = state.units[id].definition.bindings.clone();
            let item = loop {
                let item = match tier {
                    MemoryTier::Device => {
                        if let Some(host) = state.units[id].host.as_ref().map(Arc::clone) {
                            prepare_copy_to_device(id, host, &state.device_stream)
                        } else {
                            prepare_from_disk(
                                store,
                                &bindings,
                                &state.source_stream,
                                &state.device_stream,
                                TransferDirection::DiskToDevice,
                            )
                        }
                    }
                    MemoryTier::Host | MemoryTier::Disk => unreachable!("validated above"),
                };
                match item {
                    Ok(item) => break item,
                    Err(error)
                        if is_mapping_capacity_error(&error)
                            && prepared.iter().any(
                                |(_, item): &(OffloadUnitId, PreparedResidentArrays)| {
                                    !item.pending_sources.is_empty()
                                },
                            ) =>
                    {
                        // Earlier units in this batch can pin the only mapped
                        // shard while a later cross-shard expert is prepared.
                        // Their output arrays are complete evaluation roots, so
                        // detach those mappings and retry the current unit.
                        eval(prepared.iter().flat_map(|(_, item)| item.arrays.values())).map_err(
                            |source| ResidencyError::Mlx {
                                id: internal_id(),
                                operation: "mapping-capacity batch evaluation",
                                source,
                            },
                        )?;
                        for (_, item) in &mut prepared {
                            for source in item.pending_sources.drain(..) {
                                source.complete();
                            }
                            item.retained_arrays.clear();
                            item.retained_host = None;
                            item.retained_events.clear();
                        }
                    }
                    Err(error) => return Err(error),
                }
            };
            prepared.push((id.clone(), item));
        }

        for (id, item) in &prepared {
            let actual = arrays_nbytes(&item.arrays)?;
            let required = state.ledger.spec(id)?.bytes();
            if actual != required {
                return Err(ResidencyError::UnitByteMismatch {
                    id: id.clone(),
                    planned_bytes: required,
                    actual_bytes: actual,
                });
            }
        }

        let event =
            async_eval_with_event(prepared.iter().flat_map(|(_, item)| item.arrays.values()))
                .map_err(|source| ResidencyError::Mlx {
                    id: internal_id(),
                    operation: "batched residency submission",
                    source,
                })?;
        let generation = if return_transfer {
            state.ledger.next_transfer_generation()?
        } else {
            let completion = event.synchronize();
            for (_, item) in &mut prepared {
                for source in item.pending_sources.drain(..) {
                    source.complete();
                }
                item.retained_arrays.clear();
                item.retained_host = None;
                item.retained_events.clear();
            }
            completion.map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "batched residency completion",
                source,
            })?;
            0
        };

        let submitted_ids = prepared
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let mut retained = ResidentTransferResources {
            sources: Vec::new(),
            retained_arrays: Vec::new(),
            retained_host: Vec::new(),
            retained_events: Vec::new(),
        };

        for (id, mut item) in prepared {
            if return_transfer {
                retained.sources.append(&mut item.pending_sources);
                retained.retained_arrays.append(&mut item.retained_arrays);
                if let Some(host) = item.retained_host.take() {
                    retained.retained_host.push(host);
                }
                retained.retained_events.append(&mut item.retained_events);
            }
            let actual = arrays_nbytes(&item.arrays)?;
            state.ledger.publish_reserved(
                &id,
                tier,
                actual,
                return_transfer.then_some(generation),
            )?;
            let unit = state
                .units
                .get_mut(&id)
                .ok_or(ResidencyError::StatePoisoned)?;
            match tier {
                MemoryTier::Device => {
                    unit.device = Some(Arc::new(ResidentArrays {
                        arrays: item.arrays,
                    }))
                }
                MemoryTier::Host | MemoryTier::Disk => unreachable!("validated above"),
            }
            state
                .ledger
                .record_transfer(item.direction, actual, started.elapsed());
        }
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                state.ledger.touch(id, tier)?;
            }
        }
        let submitted = return_transfer.then_some(SubmittedResidentTransfer {
            event,
            retained,
            ids: submitted_ids,
            generation,
        });
        Ok((created.clone(), submitted))
    })();

    if result.is_err() {
        for id in &reserved {
            state.ledger.rollback_reserved(id, tier)?;
        }
    }
    result
}

struct PreparedResidentArrays {
    arrays: BTreeMap<String, Array>,
    pending_sources: Vec<PendingWeightMaterialization>,
    retained_arrays: Vec<Array>,
    retained_host: Option<Arc<ResidentHostBuffers>>,
    retained_events: Vec<Event>,
    direction: TransferDirection,
}

fn materialize_host_buffers(
    id: &OffloadUnitId,
    store: &dyn WeightStore,
    bindings: &[WeightBinding],
    source_stream: &Stream,
) -> Result<ResidentHostBuffers, ResidencyError> {
    let mut buffers = BTreeMap::new();
    for binding in bindings {
        let (array, sources) = match &binding.recipe {
            Some(recipe) => {
                let pending = recipe
                    .prepare_materialization(store, source_stream)
                    .map_err(|source| ResidencyError::Recipe {
                        binding: binding.name.clone(),
                        source,
                    })?;
                pending.into_parts()
            }
            None => {
                let lease = store.acquire_with_policy(
                    &binding.checkpoint_key,
                    binding.selection.clone(),
                    WeightReadPolicy::RequireBounded,
                )?;
                let pending = lease.prepare_materialization(source_stream, source_stream)?;
                (pending.output().clone(), vec![pending])
            }
        };
        let buffer = HostTransferBuffer::copy_from_array(
            &array,
            HostTransferPolicy::Transfer,
            source_stream,
        )
        .map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "weight array-to-host-buffer submission",
            source,
        })?
        .synchronize()
        .map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "weight array-to-host-buffer completion",
            source,
        })?;
        for source in sources {
            source.complete();
        }
        let actual = u64::try_from(buffer.nbytes().map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "host-buffer byte inspection",
            source,
        })?)
        .map_err(|_| ResidencyError::ArithmeticOverflow {
            context: "host-buffer byte conversion",
        })?;
        if actual != binding.expected_bytes {
            return Err(ResidencyError::BindingByteMismatch {
                id: id.clone(),
                binding: binding.name.clone(),
                expected_bytes: binding.expected_bytes,
                actual_bytes: actual,
            });
        }
        buffers.insert(binding.name.clone(), buffer.freeze());
    }
    Ok(ResidentHostBuffers { buffers })
}

fn prepare_from_disk(
    store: &dyn WeightStore,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
    direction: TransferDirection,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = BTreeMap::new();
    let mut pending_sources = Vec::new();
    let mut retained_arrays = Vec::new();
    for binding in bindings {
        let mut retried_after_capacity = false;
        loop {
            let prepared = (|| match &binding.recipe {
                Some(recipe) => {
                    let pending = recipe
                        .prepare_materialization(store, source_stream)
                        .map_err(|source| ResidencyError::Recipe {
                            binding: binding.name.clone(),
                            source,
                        })?;
                    let (host, sources) = pending.into_parts();
                    if execution_stream == source_stream {
                        Ok((host, sources, None))
                    } else {
                        let output = host.copy(execution_stream).map_err(|source| {
                            ResidencyError::Recipe {
                                binding: binding.name.clone(),
                                source: WeightRecipeError::Mlx(source),
                            }
                        })?;
                        Ok((output, sources, Some(host)))
                    }
                }
                None => {
                    let lease = store.acquire_with_policy(
                        &binding.checkpoint_key,
                        binding.selection.clone(),
                        WeightReadPolicy::RequireBounded,
                    )?;
                    let pending = lease.prepare_materialization(source_stream, execution_stream)?;
                    let output = pending.output().clone();
                    Ok((output, vec![pending], None))
                }
            })();
            match prepared {
                Ok((output, sources, retained)) => {
                    pending_sources.extend(sources);
                    retained_arrays.extend(retained);
                    arrays.insert(binding.name.clone(), output);
                    break;
                }
                Err(error)
                    if !retried_after_capacity
                        && !pending_sources.is_empty()
                        && is_mapping_capacity_error(&error) =>
                {
                    eval(arrays.values()).map_err(|source| ResidencyError::Mlx {
                        id: internal_id(),
                        operation: "mapping-capacity residency evaluation",
                        source,
                    })?;
                    for source in pending_sources.drain(..) {
                        source.complete();
                    }
                    retained_arrays.clear();
                    retried_after_capacity = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources,
        retained_arrays,
        retained_host: None,
        retained_events: Vec::new(),
        direction,
    })
}

fn is_mapping_capacity_error(error: &ResidencyError) -> bool {
    matches!(
        error,
        ResidencyError::WeightStore(WeightStoreError::CapacityExhausted { .. })
            | ResidencyError::Recipe {
                source: WeightRecipeError::WeightStore(WeightStoreError::CapacityExhausted { .. }),
                ..
            }
    )
}

fn prepare_copy_to_device(
    id: &OffloadUnitId,
    host: Arc<ResidentHostBuffers>,
    device_stream: &Stream,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = BTreeMap::new();
    let mut retained_events = Vec::new();
    for (name, buffer) in &host.buffers {
        let submitted =
            buffer
                .copy_to_array(device_stream)
                .map_err(|source| ResidencyError::Mlx {
                    id: id.clone(),
                    operation: "host-buffer-to-device copy",
                    source,
                })?;
        let (array, completion) = submitted.into_parts();
        arrays.insert(name.clone(), array);
        retained_events.push(completion);
    }
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources: Vec::new(),
        retained_arrays: Vec::new(),
        retained_host: Some(host),
        retained_events,
        direction: TransferDirection::HostToDevice,
    })
}

fn arrays_nbytes(arrays: &BTreeMap<String, Array>) -> Result<u64, ResidencyError> {
    arrays.values().try_fold(0u64, |total, array| {
        let bytes =
            u64::try_from(array.nbytes()).map_err(|_| ResidencyError::ArithmeticOverflow {
                context: "array byte conversion",
            })?;
        total
            .checked_add(bytes)
            .ok_or(ResidencyError::ArithmeticOverflow {
                context: "resident array byte total",
            })
    })
}

fn resident_capacity_requirement(
    unit: &UnitRecord,
    planned_bytes: u64,
    tier: MemoryTier,
) -> Result<u64, ResidencyError> {
    if tier != MemoryTier::Host {
        return Ok(planned_bytes);
    }
    host_capacity_upper_bound_for_bindings(&unit.definition.bindings)
}

/// Returns the complete charged host-transfer capacity for one atomic unit.
pub(crate) fn host_capacity_upper_bound_for_bindings(
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings.iter().try_fold(0u64, |total, binding| {
        let logical = usize::try_from(binding.expected_bytes).map_err(|_| {
            ResidencyError::ArithmeticOverflow {
                context: "host capacity-bound input conversion",
            }
        })?;
        let capacity = host_transfer_capacity_upper_bound(logical, HostTransferPolicy::Transfer)
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "host-transfer capacity-bound query",
                source,
            })?;
        let capacity = u64::try_from(capacity).map_err(|_| ResidencyError::ArithmeticOverflow {
            context: "host capacity-bound output conversion",
        })?;
        total
            .checked_add(capacity)
            .ok_or(ResidencyError::ArithmeticOverflow {
                context: "host unit capacity bound",
            })
    })
}

fn host_buffers_nbytes(buffers: &ResidentHostBuffers) -> Result<u64, ResidencyError> {
    buffers.buffers.values().try_fold(0u64, |total, buffer| {
        let bytes = u64::try_from(buffer.nbytes().map_err(|source| ResidencyError::Mlx {
            id: internal_id(),
            operation: "host-buffer byte inspection",
            source,
        })?)
        .map_err(|_| ResidencyError::ArithmeticOverflow {
            context: "host-buffer byte conversion",
        })?;
        total
            .checked_add(bytes)
            .ok_or(ResidencyError::ArithmeticOverflow {
                context: "resident host-buffer byte total",
            })
    })
}

fn host_buffers_capacity(buffers: &ResidentHostBuffers) -> Result<u64, ResidencyError> {
    buffers.buffers.values().try_fold(0u64, |total, buffer| {
        let bytes = u64::try_from(buffer.capacity().map_err(|source| ResidencyError::Mlx {
            id: internal_id(),
            operation: "host-buffer capacity inspection",
            source,
        })?)
        .map_err(|_| ResidencyError::ArithmeticOverflow {
            context: "host-buffer capacity conversion",
        })?;
        total
            .checked_add(bytes)
            .ok_or(ResidencyError::ArithmeticOverflow {
                context: "resident host-buffer capacity total",
            })
    })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{mpsc, Arc, Barrier},
        time::Duration,
    };

    use safemlx::{
        host_transfer_capacity_upper_bound, Device, DeviceType, HostTransferPolicy,
        HostTransferStorageKind,
    };
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    use super::*;
    use crate::{
        backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore,
        core::residency::{OffloadConfig, OffloadUnitSpec, ResidencyLedgerError, ResidencyPolicy},
    };

    fn cpu_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn write_fixture(path: &std::path::Path) {
        let a = [1i32, 2]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let b = [3i32, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let c = [5i32, 6]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let matrix = [10i32, 11, 12, 13, 14, 15]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                ("a", TensorView::new(Dtype::I32, vec![2], &a).unwrap()),
                ("b", TensorView::new(Dtype::I32, vec![2], &b).unwrap()),
                ("c", TensorView::new(Dtype::I32, vec![2], &c).unwrap()),
                (
                    "matrix",
                    TensorView::new(Dtype::I32, vec![3, 2], &matrix).unwrap(),
                ),
            ],
            None,
            path,
        )
        .unwrap();
    }

    fn fixture_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(&dir.path().join("model.safetensors"));
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        (dir, store)
    }

    fn cross_shard_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        for (file, key, values) in [
            ("model-00001-of-00002.safetensors", "left", [1i32, 2]),
            ("model-00002-of-00002.safetensors", "right", [3i32, 4]),
        ] {
            let bytes = values
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            serialize_to_file(
                [(key, TensorView::new(Dtype::I32, vec![2], &bytes).unwrap())],
                None,
                &dir.path().join(file),
            )
            .unwrap();
        }
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "left": "model-00001-of-00002.safetensors",
                    "right": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let store =
            Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1).unwrap());
        (dir, store)
    }

    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }

    fn binding(name: &str, key: &str, selection: TensorSelection, bytes: u64) -> WeightBinding {
        WeightBinding::new(name, key, selection, bytes).unwrap()
    }

    fn unit(name: &str, bindings: impl IntoIterator<Item = WeightBinding>) -> OffloadUnit {
        OffloadUnit::new(id(name), bindings).unwrap()
    }

    fn spec(name: &str, bytes: u64, policy: ResidencyPolicy, tier: MemoryTier) -> OffloadUnitSpec {
        OffloadUnitSpec::new(id(name), bytes, policy, tier).unwrap()
    }

    fn manager(
        store: Arc<SafetensorsWeightStore>,
        config: OffloadConfig,
        specs: impl IntoIterator<Item = OffloadUnitSpec>,
        units: impl IntoIterator<Item = OffloadUnit>,
    ) -> ResidencyManager {
        // Most fixtures below use one independently allocated eight-byte
        // binding as their accounting unit. Preserve those unit-count tests
        // while exercising the production contract, which charges the full
        // backing extent of every host-transfer allocation.
        let host_budget = config.host_budget_bytes().map(fixture_physical_host_budget);
        let config = OffloadConfig::new(
            config.device_budget_bytes(),
            host_budget,
            config.prefetch_depth(),
        )
        .unwrap()
        .with_eviction_policy(config.eviction_policy());
        ResidencyManager::new(
            store,
            OffloadPlan::new(config, specs).unwrap(),
            units,
            cpu_stream(),
            cpu_stream(),
        )
        .unwrap()
    }

    fn fixture_physical_host_budget(logical_bytes: u64) -> u64 {
        if logical_bytes == 0 {
            return 0;
        }
        let minimum_capacity =
            host_transfer_capacity_upper_bound(1, HostTransferPolicy::Transfer).unwrap() as u64;
        if logical_bytes >= minimum_capacity {
            return logical_bytes;
        }
        let complete_bindings = logical_bytes / 8;
        let remainder = logical_bytes % 8;
        let binding_capacity = fixture_binding_capacity();
        complete_bindings
            .checked_mul(binding_capacity)
            .and_then(|bytes| {
                (remainder != 0)
                    .then(|| {
                        host_transfer_capacity_upper_bound(
                            remainder as usize,
                            HostTransferPolicy::Transfer,
                        )
                        .unwrap() as u64
                    })
                    .map_or(Some(bytes), |tail| bytes.checked_add(tail))
            })
            .unwrap()
    }

    fn fixture_binding_capacity() -> u64 {
        host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64
    }

    fn fixture_host_capacity(bindings: u64) -> u64 {
        bindings.checked_mul(fixture_binding_capacity()).unwrap()
    }

    fn single(name: &str, key: &str) -> OffloadUnit {
        unit(name, [binding("weight", key, TensorSelection::Full, 8)])
    }

    fn host_i32(lease: &ResidentUnitLease, name: &str) -> Vec<i32> {
        lease
            .host_buffer(name)
            .unwrap()
            .as_bytes()
            .unwrap()
            .chunks_exact(size_of::<i32>())
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn named_execution_groups_keep_independent_windows_and_clear_in_isolation() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, None, 1).unwrap(),
            [
                spec("text.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("text.1", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("vision.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            ],
            [
                single("text.0", "a"),
                single("text.1", "b"),
                single("vision.0", "c"),
            ],
        );
        manager.initialize().unwrap();
        let text = ResidentLayerGroup::new("text", [id("text.0"), id("text.1")], 1).unwrap();
        let vision = ResidentLayerGroup::new("vision", [id("vision.0")], 1).unwrap();

        text.prepare(&manager, 0).unwrap();
        vision.prepare(&manager, 0).unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.active_window(), &[id("text.0"), id("vision.0")]);
        assert!(state(&report, "text.0").device_resident());
        assert!(state(&report, "vision.0").device_resident());

        text.clear(&manager).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "text.0").device_resident());
        assert!(state(&report, "vision.0").device_resident());
        assert_eq!(report.active_window(), &[id("vision.0")]);
        let vision_report = vision.report(&manager).unwrap();
        assert_eq!(vision_report.device_units(), 1);
        assert_eq!(vision_report.device_bytes(), 8);
    }

    fn state<'a>(report: &'a ResidencyReport, name: &str) -> &'a UnitResidencyReport {
        report
            .units()
            .iter()
            .find(|unit| unit.id() == &id(name))
            .unwrap()
    }

    #[test]
    fn failed_batch_reservation_rolls_back_and_cache_remains_usable() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b")],
        );
        manager.initialize().unwrap();
        assert!(matches!(
            manager.acquire_many_with_demand(&[(id("a"), 1), (id("b"), 1)], MemoryTier::Device),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::BudgetExhausted { .. }
            ))
        ));
        let report = manager.report().unwrap();
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 0);
        assert!(!state(&report, "a").device_resident());
        assert!(!state(&report, "b").device_resident());

        let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(lease.array("weight").unwrap().shape(), &[2]);
    }

    #[test]
    fn batched_units_detach_prior_shards_at_mapping_capacity() {
        let (_dir, store) = cross_shard_store();
        let manager = manager(
            Arc::clone(&store),
            OffloadConfig::new(None, None, 1).unwrap(),
            [
                spec("left", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("right", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("left", "left"), single("right", "right")],
        );
        manager.initialize().unwrap();

        let leases = manager
            .acquire_many_with_demand(&[(id("left"), 1), (id("right"), 1)], MemoryTier::Host)
            .unwrap();

        assert_eq!(host_i32(&leases[0], "weight"), [1, 2]);
        assert_eq!(host_i32(&leases[1], "weight"), [3, 4]);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert!(diagnostics.evictions >= 1);
    }

    #[test]
    fn caller_owned_transfer_publishes_only_after_exact_completion() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let mut transfer = manager
            .acquire_many_with_transfer(&[(id("a"), 2)], MemoryTier::Device)
            .unwrap();
        let lease = &transfer.leases()[0];
        {
            let state = manager.lock().unwrap();
            let copy = state
                .ledger
                .copy_status(&id("a"), MemoryTier::Device)
                .unwrap()
                .unwrap();
            assert_eq!(copy.pins(), 1);
            assert!(copy.in_flight().is_some());
        }

        let consumer = cpu_stream();
        transfer.wait_on(&consumer).unwrap();
        let dependent = lease
            .array("weight")
            .unwrap()
            .add(Array::from_int(1), &consumer)
            .unwrap();
        eval([&dependent]).unwrap();
        transfer.synchronize().unwrap();
        {
            let state = manager.lock().unwrap();
            let copy = state
                .ledger
                .copy_status(&id("a"), MemoryTier::Device)
                .unwrap()
                .unwrap();
            assert!(copy.in_flight().is_none());
        }
        let ready = manager
            .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        assert!(ready.is_complete().unwrap());
        drop(ready);
        drop(transfer);
        assert_eq!(state(&manager.report().unwrap(), "a").device_pins(), 0);
    }

    #[test]
    fn empty_caller_owned_transfer_is_immediately_complete() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let mut transfer = manager
            .acquire_many_with_transfer(&[], MemoryTier::Device)
            .unwrap();
        assert!(transfer.is_empty());
        assert!(transfer.is_complete().unwrap());
        transfer.wait_on(&cpu_stream()).unwrap();
        transfer.synchronize().unwrap();
        assert!(!manager.is_resident(&id("a"), MemoryTier::Device).unwrap());
    }

    #[test]
    fn dropping_transfer_retains_queued_consumer_and_publishes_copy() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let transfer = manager
            .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        let consumer = cpu_stream();
        transfer.wait_on(&consumer).unwrap();
        let dependent = transfer.leases()[0]
            .array("weight")
            .unwrap()
            .add(Array::from_int(1), &consumer)
            .unwrap();
        drop(transfer);
        assert!(manager
            .lock()
            .unwrap()
            .ledger
            .copy_status(&id("a"), MemoryTier::Device)
            .unwrap()
            .unwrap()
            .in_flight()
            .is_none());
        assert_eq!(dependent.evaluated().unwrap().as_slice::<i32>(), &[2, 3]);
    }

    #[test]
    fn synchronous_acquisition_waits_for_caller_owned_transfer() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let mut transfer = manager
            .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        let waiting_manager = manager.clone();
        let (sender, receiver) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let result = waiting_manager
                .acquire(&id("a"), MemoryTier::Device)
                .map(|lease| {
                    lease
                        .array("weight")
                        .unwrap()
                        .evaluated()
                        .unwrap()
                        .as_slice::<i32>()
                        .to_vec()
                });
            sender.send(result).unwrap();
        });
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        transfer.synchronize().unwrap();
        assert_eq!(
            receiver
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap(),
            vec![1, 2]
        );
        waiter.join().unwrap();
    }

    #[test]
    fn validates_unit_identity_bindings_sizes_and_targets() {
        let (_dir, store) = fixture_store();
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        )
        .unwrap();
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::MissingUnitDefinition { .. })
        ));
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [single("a", "a"), single("a", "a")],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::DuplicateUnitDefinition { .. })
        ));
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [single("a", "a"), single("b", "b")],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::UnexpectedUnitDefinition { .. })
        ));
        assert!(matches!(
            OffloadUnit::new(id("empty"), []),
            Err(ResidencyError::EmptyUnit { .. })
        ));
        let duplicate = binding("same", "a", TensorSelection::Full, 8);
        assert!(matches!(
            OffloadUnit::new(id("duplicate"), [duplicate.clone(), duplicate]),
            Err(ResidencyError::DuplicateBindingName { .. })
        ));
        let wrong = unit("a", [binding("weight", "a", TensorSelection::Full, 4)]);
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [wrong],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::BindingByteMismatch { .. })
        ));

        let valid =
            ResidencyManager::new(store, plan, [single("a", "a")], cpu_stream(), cpu_stream())
                .unwrap();
        assert!(matches!(
            valid.prefetch(&id("a"), MemoryTier::Disk),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::InvalidTargetTier { .. }
            ))
        ));
    }

    #[test]
    fn detects_unit_total_overflow_before_checkpoint_access() {
        let (_dir, store) = fixture_store();
        let overflowing = unit(
            "a",
            [
                binding("a", "a", TensorSelection::Full, 8),
                binding("z", "missing", TensorSelection::Full, u64::MAX),
            ],
        );
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        )
        .unwrap();
        assert!(matches!(
            ResidencyManager::new(store, plan, [overflowing], cpu_stream(), cpu_stream()),
            Err(ResidencyError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn initialization_honors_planned_tiers_and_pinning() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [
                spec("disk", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("host", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
                spec("device", 8, ResidencyPolicy::Cacheable, MemoryTier::Device),
            ],
            [
                single("disk", "a"),
                single("host", "b"),
                single("device", "c"),
            ],
        );
        assert!(matches!(
            manager.acquire(&id("disk"), MemoryTier::Host),
            Err(ResidencyError::Ledger(ResidencyLedgerError::NotInitialized))
        ));
        manager.initialize().unwrap();
        let report = manager.report().unwrap();
        assert!(report.initialized());
        assert!(!state(&report, "disk").host_resident());
        assert!(state(&report, "host").host_resident());
        assert!(state(&report, "device").device_resident());
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
        assert!(matches!(
            manager.evict(&id("host"), MemoryTier::Host),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::PinnedEviction { .. }
            ))
        ));
    }

    #[test]
    fn host_residency_charges_physical_allocation_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let values = (0..5000u32)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![5000], &values).unwrap(),
            )],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let logical = u64::try_from(values.len()).unwrap();
        let capacity =
            safemlx::host_transfer_capacity_upper_bound(values.len(), HostTransferPolicy::Transfer)
                .map(|capacity| u64::try_from(capacity).unwrap())
                .unwrap();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(capacity), 1).unwrap(),
            [spec(
                "host",
                logical,
                ResidencyPolicy::Pinned,
                MemoryTier::Host,
            )],
            [unit(
                "host",
                [binding("weight", "weight", TensorSelection::Full, logical)],
            )],
        );
        manager.initialize().unwrap();
        let report = manager.report().unwrap();
        let unit = state(&report, "host");
        assert_eq!(unit.expected_bytes(), logical);
        assert_eq!(unit.host_allocated_bytes(), capacity);
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            capacity
        );
    }

    #[test]
    fn partial_initialization_failure_remains_consistent_and_inspectable() {
        let dir = tempfile::tempdir().unwrap();
        let good = [7u8, 8];
        let bad = [9u8, 10];
        serialize_to_file(
            [
                ("good", TensorView::new(Dtype::U8, vec![2], &good).unwrap()),
                (
                    "unsupported",
                    TensorView::new(Dtype::F8_E5M2, vec![2], &bad).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(fixture_host_capacity(3)), 1).unwrap(),
            [
                spec("a-good", 2, ResidencyPolicy::Cacheable, MemoryTier::Host),
                spec("z-bad", 4, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
            [
                unit(
                    "a-good",
                    [binding("weight", "good", TensorSelection::Full, 2)],
                ),
                unit(
                    "z-bad",
                    [
                        binding("a-good-copy", "good", TensorSelection::Full, 2),
                        binding("z-unsupported", "unsupported", TensorSelection::Full, 2),
                    ],
                ),
            ],
        );
        assert!(matches!(
            manager.initialize(),
            Err(ResidencyError::WeightStore(
                WeightStoreError::UnsupportedStoredDtype { .. }
            ))
        ));
        let report = manager.report().unwrap();
        assert!(!report.initialized());
        assert!(state(&report, "a-good").host_resident());
        assert!(!state(&report, "z-bad").host_resident());
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
    }

    #[test]
    fn materializes_promotes_and_publishes_multi_tensor_units_atomically() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(16), Some(16), 1).unwrap(),
            [spec(
                "quantized",
                16,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )],
            [unit(
                "quantized",
                [
                    binding("scales", "b", TensorSelection::Full, 8),
                    binding("weight", "a", TensorSelection::Full, 8),
                ],
            )],
        );
        manager.initialize().unwrap();
        assert_eq!(
            manager
                .prefetch(&id("quantized"), MemoryTier::Host)
                .unwrap(),
            PrefetchOutcome::Miss
        );
        let host = manager.acquire(&id("quantized"), MemoryTier::Host).unwrap();
        assert_eq!(
            host.binding_names().collect::<Vec<_>>(),
            ["scales", "weight"]
        );
        assert_eq!(host_i32(&host, "weight"), [1, 2]);
        assert!(matches!(
            host.array("weight"),
            Err(ResidencyError::HostBindingIsNotArray { .. })
        ));
        assert!(matches!(
            host.host_buffer("unknown"),
            Err(ResidencyError::UnknownBinding { .. })
        ));
        assert!(matches!(
            host.host_buffer("weight").unwrap().storage_kind().unwrap(),
            HostTransferStorageKind::Cpu
                | HostTransferStorageKind::MetalShared
                | HostTransferStorageKind::CudaPinned
        ));
        drop(host);
        assert_eq!(
            manager
                .prefetch(&id("quantized"), MemoryTier::Device)
                .unwrap(),
            PrefetchOutcome::Miss
        );
        let device = manager
            .acquire(&id("quantized"), MemoryTier::Device)
            .unwrap();
        assert_eq!(device.array("scales").unwrap().shape(), &[2]);
        assert!(matches!(
            device.host_buffer("scales"),
            Err(ResidencyError::DeviceBindingIsNotHostBuffer { .. })
        ));
        let report = manager.report().unwrap();
        assert!(state(&report, "quantized").host_resident());
        assert!(state(&report, "quantized").device_resident());
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToHost)
                .bytes(),
            16
        );
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .bytes(),
            16
        );
    }

    #[test]
    fn direct_disk_to_device_does_not_create_a_host_copy() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "a").device_resident());
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToDevice)
                .bytes(),
            8
        );
    }

    #[test]
    fn budgets_use_deterministic_policy_then_lru_eviction() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("cache-a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("window-b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("cache-c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                single("cache-a", "a"),
                single("window-b", "b"),
                single("cache-c", "c"),
            ],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(state(&report, "cache-a").host_resident());
        assert!(!state(&report, "window-b").host_resident());
        assert!(state(&report, "cache-c").host_resident());
        assert_eq!(report.offload().evictions().count(), 1);
        assert_eq!(
            report.offload().evictions().bytes(),
            fixture_binding_capacity()
        );

        manager.evict(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.evict(&id("cache-c"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "cache-a").host_resident());
        assert!(state(&report, "cache-c").host_resident());
        assert!(state(&report, "window-b").host_resident());
        assert!(
            report.offload().resident_bytes().get(MemoryTier::Host) <= fixture_host_capacity(2)
        );
    }

    #[test]
    fn oldest_copy_is_evicted_deterministically() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("c"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "b").host_resident());
        assert!(state(&report, "c").host_resident());
    }

    #[test]
    fn host_and_device_budgets_are_independent() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
        manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "a").device_resident());
        assert!(state(&report, "b").host_resident());
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
    }

    #[test]
    fn leases_block_eviction_and_drop_or_unwind_releases_pins() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
        assert!(matches!(
            manager.evict(&id("a"), MemoryTier::Host),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::InUseEviction { pin_count: 1, .. }
            ))
        ));
        drop(lease);
        assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
        assert!(!manager.evict(&id("a"), MemoryTier::Host).unwrap());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let manager = manager.clone();
            move || {
                let _lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
                panic!("exercise lease unwinding");
            }
        }));
        assert!(result.is_err());
        assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    }

    #[test]
    fn concurrent_acquisition_materializes_once_and_counts_pins() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let manager = manager.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
                    barrier.wait();
                    barrier.wait();
                    drop(lease);
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let report = manager.report().unwrap();
        assert_eq!(state(&report, "a").host_pins(), 2);
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToHost)
                .count(),
            1
        );
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(state(&manager.report().unwrap(), "a").host_pins(), 0);
    }

    #[test]
    fn windows_bound_lookahead_protect_active_units_and_record_hits() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 2).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        let first = manager
            .prepare_window(&[id("a")], &[id("a"), id("b"), id("c")], MemoryTier::Host)
            .unwrap();
        assert_eq!(first.len(), 2);
        assert!(first
            .iter()
            .all(|(_, value)| *value == PrefetchOutcome::Miss));
        let report_before = manager.report().unwrap();
        assert!(state(&report_before, "a").host_resident());
        assert!(state(&report_before, "b").host_resident());
        assert!(!state(&report_before, "c").host_resident());

        let second = manager
            .prepare_window(&[id("b")], &[id("b"), id("c")], MemoryTier::Host)
            .unwrap();
        assert_eq!(second[0].1, PrefetchOutcome::Hit);
        assert_eq!(second[1].1, PrefetchOutcome::Miss);
        let report_after = manager.report().unwrap();
        assert!(!state(&report_after, "a").host_resident());
        assert!(state(&report_after, "b").host_resident());
        assert!(state(&report_after, "c").host_resident());
        assert_eq!(report_after.active_window(), &[id("b")]);
        assert_eq!(report_after.offload().prefetch().requests(), 4);
        assert_eq!(report_after.offload().prefetch().hits(), 1);
        assert_eq!(report_after.offload().prefetch().misses(), 3);
        assert_eq!(report_before.offload().prefetch().requests(), 2);
    }

    #[test]
    fn exhaustion_reports_pinned_in_use_and_active_blockers() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("pinned", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
                spec("leased", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("wanted", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                single("pinned", "a"),
                single("leased", "b"),
                single("wanted", "c"),
            ],
        );
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("leased"), MemoryTier::Host).unwrap();
        let error = manager
            .prefetch(&id("wanted"), MemoryTier::Host)
            .unwrap_err();
        match error {
            ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
                required_bytes,
                budget_bytes,
                blocking_units,
                ..
            }) => {
                assert_eq!(required_bytes, fixture_binding_capacity());
                assert_eq!(budget_bytes, fixture_host_capacity(2));
                assert_eq!(blocking_units.len(), 2);
                assert!(blocking_units.iter().any(|unit| unit.pinned));
                assert!(blocking_units.iter().any(|unit| unit.in_use == 1));
            }
            other => panic!("unexpected error: {other}"),
        }
        drop(lease);
    }

    #[test]
    fn demand_stalls_and_rank_local_selections_are_reported() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("range", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("indices", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                unit(
                    "range",
                    [binding(
                        "weight",
                        "matrix",
                        TensorSelection::Range {
                            axis: 0,
                            start: 1,
                            end: 2,
                        },
                        8,
                    )],
                ),
                unit(
                    "indices",
                    [binding(
                        "weight",
                        "matrix",
                        TensorSelection::Indices {
                            axis: 0,
                            indices: vec![2],
                        },
                        8,
                    )],
                ),
            ],
        );
        manager.initialize().unwrap();
        let range = manager.acquire(&id("range"), MemoryTier::Host).unwrap();
        assert_eq!(host_i32(&range, "weight"), [12, 13]);
        drop(range);
        let indices = manager.acquire(&id("indices"), MemoryTier::Host).unwrap();
        assert_eq!(host_i32(&indices, "weight"), [14, 15]);
        let report = manager.report().unwrap();
        assert_eq!(report.offload().prefetch().stalls(), 2);
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_host_capacity(2)
        );
        assert_eq!(
            report.offload().peak_resident_bytes().get(MemoryTier::Host),
            fixture_host_capacity(2)
        );
        assert!(report.weight_store().mapping_hits > 0);
    }

    #[test]
    fn ordered_device_window_trims_stale_units_with_unlimited_budget() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(24), 2).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        let window = DeviceLayerWindow::new([id("a"), id("b"), id("c")], 2).unwrap();

        window.prepare(&manager, 0).unwrap();
        let first = manager.report().unwrap();
        assert!(state(&first, "a").device_resident());
        assert!(state(&first, "b").device_resident());
        assert!(!state(&first, "c").device_resident());

        let lease = manager.acquire(&id("b"), MemoryTier::Device).unwrap();
        window.prepare(&manager, 1).unwrap();
        let second = manager.report().unwrap();
        assert!(!state(&second, "a").device_resident());
        assert!(state(&second, "b").device_resident());
        assert!(state(&second, "c").device_resident());
        assert_eq!(state(&second, "b").device_pins(), 1);
        drop(lease);

        window.clear(&manager).unwrap();
        assert!(manager
            .report()
            .unwrap()
            .units()
            .iter()
            .all(|unit| !unit.device_resident()));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn promotes_host_buffers_to_a_real_metal_stream() {
        let (_dir, store) = fixture_store();
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(8), Some(fixture_binding_capacity()), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Host)],
        )
        .unwrap();
        let manager = ResidencyManager::new(
            store,
            plan,
            [single("a", "a")],
            cpu_stream(),
            Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)),
        )
        .unwrap();
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(
            lease
                .array("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            manager
                .report()
                .unwrap()
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .count(),
            1
        );
    }
}
