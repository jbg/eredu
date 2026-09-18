//! Cold parameter backing and materialization for completed layerwise windows.

use super::*;
use crate::backend::{
    nn::workspace::MetalAllocationFacts,
    runtime::residency::manager::{
        DiskCopyWorkspace, DiskRouteGuard, DiskRouteReceipt, ForegroundDiskIdentity,
        HostCopyIdentity, HostCopyWorkspace,
    },
};
use eredu_core::WorkspaceBound;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceLayout, WorkspaceTensor,
};
use std::{cell::RefCell, collections::BTreeMap};
mod parameter_sources;
mod parameter_representations;
mod supplementary;
use eredu_runtime::working_memory::{WorkspaceParameterLifetime, WorkspaceParameterRows};

/// Payload-free witness used to reject a different policy or host source graph.
#[derive(Debug)]
pub(crate) struct LayerwiseWorkspaceIdentity {
    policy: Arc<()>,
    geometry: LayerwiseGeometry,
    parameter_locations: Option<Arc<[(ExecutionUnitAddress, ExecutionUnitAddress)]>>,
    excluded: Option<MlxParameterExclusions>,
    manager_unit_constructors: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum LayerwiseGeometry {
    Owned {
        layout: ExecutionUnitLayout,
        depth: usize,
        sources: LayerwiseSourcesIdentity,
    },
    PreparedHost(HostCopyIdentity),
    PreparedForeground(ForegroundDiskIdentity),
}

impl LayerwiseWorkspaceIdentity {
    fn layout(&self) -> &ExecutionUnitLayout {
        match &self.geometry {
            LayerwiseGeometry::Owned { layout, .. } => layout,
            LayerwiseGeometry::PreparedHost(identity) => identity.layout(),
            LayerwiseGeometry::PreparedForeground(identity) => identity.layout(),
        }
    }
    fn depth(&self) -> usize {
        match &self.geometry {
            LayerwiseGeometry::Owned { depth, .. } => *depth,
            LayerwiseGeometry::PreparedHost(identity) => identity.depth(),
            LayerwiseGeometry::PreparedForeground(identity) => identity.depth(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LayerwiseSourcesIdentity {
    Host(Vec<(OffloadUnit, Vec<(String, safemlx::AllocationIdentity, u64)>)>),
    // The policy identity binds its immutable loading-time direct plans. Warm
    // cache identities may change; the retained receipt checks their envelope.
    Disk,
}

#[derive(Debug, Clone)]
pub(crate) struct DiskLayerwiseReceipt(DiskRouteReceipt);
impl DiskLayerwiseReceipt {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        self.0
            .validate()
            .map_err(|error| Error::Other(Box::new(error)))
    }
    pub(crate) fn activate(&self) -> Result<DiskRouteGuard, Error> {
        self.0
            .activate()
            .map_err(|error| Error::Other(Box::new(error)))
    }
}

#[derive(Debug)]
enum LayerwiseCopies {
    Host(HostCopyWorkspace),
    Foreground(ForegroundDiskIdentity),
    Disk {
        copies: DiskCopyWorkspace,
        receipt: DiskLayerwiseReceipt,
    },
}
type ParameterRoots = BTreeMap<(OffloadUnitId, String), WorkspaceExistingStorage>;

impl PartialEq for LayerwiseWorkspaceIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.policy, &other.policy) && self.geometry == other.geometry
            && self.manager_unit_constructors == other.manager_unit_constructors
            && self.parameter_locations == other.parameter_locations
            && match (&self.excluded, &other.excluded) {
                (Some(left), Some(right)) => left == right,
                (None, None) => true,
                _ => false,
            }
    }
}
impl Eq for LayerwiseWorkspaceIdentity {}

/// Exact native sources for cold projection and selected original installation.
/// A quote prices its retained controls without creating admission or source
/// credit. Legacy quotes move only the identity and retire this snapshot.
pub(crate) struct LayerwiseWorkspace {
    manager: ResidencyManager,
    identity: LayerwiseWorkspaceIdentity,
    copies: LayerwiseCopies,
    persistent_roots: RefCell<Option<(WorkspaceContext, ParameterRoots)>>,
    materialization: LayerwiseMaterialization,
    execution_trace: RefCell<Option<Vec<usize>>>,
    speculative_foreground:
        std::cell::OnceCell<super::original_operations::PreparedSpeculativeForegroundSource>,
}

#[derive(Debug)]
enum LayerwiseMaterialization {
    Owned(WorkspaceBound),
    PreparedHost(HostCopyIdentity),
    PreparedForeground(ForegroundDiskIdentity),
}
impl LayerwiseMaterialization {
    fn bound(&self) -> &WorkspaceBound {
        match self {
            Self::Owned(bound) => bound,
            Self::PreparedHost(identity) => identity.materialization(),
            Self::PreparedForeground(identity) => identity.materialization(),
        }
    }
}

impl std::fmt::Debug for LayerwiseWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerwiseWorkspace")
            .field("identity", &self.identity)
            .field("copies", &self.copies)
            .field("materialization", &self.materialization)
            .finish()
    }
}

// These are the existing source-route owners, not a source-format dispatch or
// submission grant. Host leases prevent rematerialization; legacy disk routes
// use DiskRouteGuard, while foreground metadata retains its original source.
pub(crate) enum OriginalLayerwiseSourceCustody {
    Host(crate::backend::runtime::residency::manager::HostCopySourcePins),
    Disk(DiskLayerwiseReceipt),
    Foreground(ForegroundDiskIdentity),
}

impl LayerwiseWorkspace {
    /// The original manager retains the full unloaded-module population,
    /// independently of the smaller set of rows populated by its leases.
    pub(crate) fn parameter_constructors(&self, ordinal: usize) -> Option<super::ParameterConstructors> {
        if self.identity.manager_unit_constructors {
            self.manager.parameter_constructors(ordinal)
        } else {
            None
        }
    }

    pub(crate) fn has_parameter_exclusions(&self) -> bool {
        self.identity.excluded.as_ref().is_some_and(|names| !names.names().is_empty())
    }

    pub(crate) fn excludes_parameter(&self, name: &str) -> bool {
        self.identity.excluded.as_ref().is_some_and(|names| names.contains(name))
    }

    pub(crate) fn destination_device_type(&self) -> safemlx::DeviceType {
        match &self.copies {
            LayerwiseCopies::Host(source) => source.destination_device_type(),
            LayerwiseCopies::Foreground(source) => source.destination_device_type(),
            // This existing legacy source is constructed only after its
            // device snapshot has positively selected the GPU worker.
            LayerwiseCopies::Disk { .. } => safemlx::DeviceType::Gpu,
        }
    }
    /// Enables one exact shared-driver unit visit. The source retains each
    /// logical ordinal separately even when its checkpoint owner is shared.
    pub(crate) fn begin_execution_trace(&self,context:&WorkspaceContext)->Result<(),eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(Self,&WorkspaceContext,
            std::cell::RefMut<'_,Option<Vec<usize>>>,Result<(),eredu_nn::Error>)>())?;
        let mut trace=self.execution_trace.try_borrow_mut().map_err(|cause|context.metadata_source(cause))?;
        if trace.is_some(){return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());}
        *trace=Some(context.metadata_vec(self.layout().len())?);Ok(())
    }
    /// Borrows the recorded invocation if enabled; ordinary whole-forward
    /// source users retain the existing full-layout census.
    pub(crate) fn with_execution_ordinals<T>(&self,visit:impl FnOnce(Option<&[usize]>)->T)->T {
        let trace=self.execution_trace.borrow();visit(trace.as_deref())
    }
    fn observe_execution_acquire(&self,ordinal:usize,address:ExecutionUnitAddress,
        context:&WorkspaceContext)->Result<(),eredu_nn::Error> {
        let mut trace=self.execution_trace.try_borrow_mut().map_err(|cause|context.metadata_source(cause))?;
        let Some(trace)=trace.as_mut()else{return Ok(());};
        context.charge_metadata(std::mem::size_of::<(usize,ExecutionUnitAddress,&WorkspaceContext,
            std::cell::RefMut<'_,Option<Vec<usize>>>,Result<(),eredu_nn::Error>)>())?;
        if self.execution_address(ordinal)!=Some(address) || trace.len()==trace.capacity()
            || trace.last().is_some_and(|previous|*previous>=ordinal) {
            return Err(context.metadata_error(format_args!("layerwise invocation order differs from its retained source")));
        }
        trace.push(ordinal);Ok(())
    }
    pub(crate) fn speculative_foreground(
        &self,
    ) -> Option<&super::original_operations::ForegroundDiskRequestPlan> {
        self.speculative_foreground
            .get()
            .map(|source| source.plan())
    }
    pub(crate) fn prepare_speculative_foreground(
        &self,
        create: impl FnOnce() -> Result<
            super::original_operations::PreparedSpeculativeForegroundSource,
            Error,
        >,
    ) -> Result<&super::original_operations::ForegroundDiskRequestPlan, Error> {
        let unknown = || {
            Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
        };
        if self.speculative_foreground.get().is_some() {
            return Err(unknown());
        }
        let prepared = create()?;
        self.speculative_foreground
            .set(prepared)
            .map_err(|_| unknown())?;
        Ok(self
            .speculative_foreground
            .get()
            .expect("prepared source")
            .plan())
    }

    #[cfg(test)]
    pub(crate) fn detached_physical_read_bytes(&self, source: usize) -> Option<u64> {
        self.manager.detached_physical_read_bytes(source)
    }

    pub(crate) fn visit_native_copy_layouts<E>(&self,
        mut visit:impl FnMut(usize,safemlx::Dtype)->Result<(),E>)
        ->Result<bool,E> {
        match &self.copies {
            LayerwiseCopies::Host(source)=>for unit in source.units() {
                for copy in source.copies(unit) {visit(copy.shape().len(),copy.dtype())?;}
            },
            LayerwiseCopies::Foreground(source)=>for (shape,dtype) in source.source().native_reads() {
                visit(shape.len(),dtype)?;
            },
            LayerwiseCopies::Disk{..}=>return Ok(false),
        }
        Ok(true)
    }

    pub(crate) fn bind_native_host_copies(
        &self,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
    ) -> Result<(), Error> {
        match &self.copies {
            LayerwiseCopies::Host(copies) => recipe.bind_host_copies(copies),
            LayerwiseCopies::Foreground(identity) => {
                if recipe.matches_foreground_disk_source(identity.source(), identity.destination_device_type()) {
                    Ok(())
                } else {
                    Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    ))
                }
            }
            LayerwiseCopies::Disk { .. } => Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )),
        }?;
        recipe.bind_layerwise_parameter_constructors(self)
    }
    /// Known retained snapshot payload, composed once into its enclosing quote.
    /// The inline value is already in the quote owner layout. Cold construction
    /// scratch and Disk copies/cache remain separate obligations. The ready-host
    /// case also includes its one initial original pin and fixed error owner.
    pub(crate) fn known_retained_control_bytes(&self, retain_sources: bool) -> Option<u64> {
        use std::alloc::Layout;
        crate::backend::runtime::residency::storage::native_storage::Bank::shared_borrowed_owner_bytes()?;
        let mut bytes = self.identity.excluded.as_ref()
            .map_or(Some(0), MlxParameterExclusions::unfunded_retained_bytes)?;
        match (&self.identity.geometry, &self.copies) {
            (LayerwiseGeometry::PreparedHost(_), LayerwiseCopies::Host(_)) => {
                // Immutable snapshot and identity backing were constructed once
                // inside the manager source account. Sharing them allocates no
                // additional payload and keeps that account alive.
            }
            (
                LayerwiseGeometry::PreparedForeground(identity),
                LayerwiseCopies::Foreground(copies),
            ) => {
                if identity != copies {
                    return None;
                }
                bytes = bytes.checked_add(ForegroundDiskIdentity::qualifier_control_bytes()?)?;
            }
            (
                LayerwiseGeometry::Owned {
                    layout, sources, ..
                },
                copies,
            ) => {
                bytes = bytes.checked_add(layout.cloned_payload_bytes()?)?;
                if retain_sources {
                    bytes = bytes.checked_add(match self.materialization() {
                        WorkspaceBound::Bounded { assumptions, .. } => assumptions.capacity(),
                        WorkspaceBound::Unknown { reason } => reason.capacity(),
                    })?;
                }
                match (sources, copies) {
                    (LayerwiseSourcesIdentity::Host(sources), LayerwiseCopies::Host(_)) => {
                        bytes = bytes.checked_add(
                            Layout::array::<(
                                OffloadUnit,
                                Vec<(String, safemlx::AllocationIdentity, u64)>,
                            )>(sources.capacity())
                            .ok()?
                            .size(),
                        )?;
                        for (definition, rows) in sources {
                            bytes = bytes
                                .checked_add(HostCopyWorkspace::cloned_definition_payload_bytes(
                                    definition,
                                )?)?
                                .checked_add(
                                    Layout::array::<(String, safemlx::AllocationIdentity, u64)>(
                                        rows.capacity(),
                                    )
                                    .ok()?
                                    .size(),
                                )?;
                            for (name, _, _) in rows {
                                bytes = bytes.checked_add(name.capacity())?;
                            }
                        }
                    }
                    (LayerwiseSourcesIdentity::Disk, LayerwiseCopies::Disk { .. }) => {}
                    _ => return None,
                }
            }
            _ => return None,
        }
        if let (true, LayerwiseCopies::Host(copies)) = (retain_sources, &self.copies) {
            bytes = bytes
                .checked_add(copies.retained_payload_bytes()?)?
                .checked_add(usize::try_from(copies.initial_pin_control_bytes()?).ok()?)?
                .checked_add(std::mem::size_of::<
                    Result<
                        Option<crate::backend::runtime::residency::manager::HostCopySourcePins>,
                        Error,
                    >,
                >())?
                .checked_add(std::mem::size_of::<
                    Option<crate::backend::runtime::residency::manager::HostCopySourcePins>,
                >())?
                .checked_add(std::mem::size_of::<Error>())?;
        }
        u64::try_from(bytes).ok()
    }

    pub(crate) fn validate_original_policy(
        &self,
        manager: &ResidencyManager,
        identity: &Arc<()>,
        units: &[OffloadUnitId],
    ) -> Result<(), Error> {
        if !manager.same_source_manager(&self.manager)
            || !Arc::ptr_eq(identity, &self.identity.policy)
            || units.len() != self.layout().len()
        {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        for (ordinal, id) in units.iter().enumerate() {
            let selected = self
                .requested_unit(ordinal)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            if self
                .unit(selected)
                .map_err(|cause| Error::Other(Box::new(cause)))?
                .id
                != id
            {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn original_pin_control_bytes(&self) -> Option<u64> {
        match &self.copies {
            LayerwiseCopies::Host(copies) => copies.retained_pin_control_bytes(),
            // The retained receipt clone allocates no new shared backing. Its
            // legacy definition is part of the still-unclosed cold snapshot.
            // Foreground identity is already source-owned; sharing adds no payload.
            LayerwiseCopies::Disk { .. } | LayerwiseCopies::Foreground(_) => Some(0),
        }
    }

    /// The ready-host constructor owns the detached catalogs, immutable native
    /// sources, finite manager rows, cached snapshot and selected window rows.
    /// Equal ordinary snapshots cannot supply this producer qualification.
    pub(crate) fn has_original_ready_host_source(
        &self,
        manager: &ResidencyManager,
        windows: &crate::backend::runtime::residency::manager::OperationWindows,
    ) -> bool {
        if !manager.same_source_manager(&self.manager) {
            return false;
        }
        match (&self.identity.geometry, &self.copies) {
            (LayerwiseGeometry::PreparedHost(identity), LayerwiseCopies::Host(copies)) => {
                copies.prepared_identity() == Some(identity)
                    && copies.is_original_source_for(manager, windows)
            }
            _ => false,
        }
    }
    pub(crate) fn has_original_foreground_source(
        &self,
        manager: &ResidencyManager,
        windows: &crate::backend::runtime::residency::manager::OperationWindows,
    ) -> bool {
        if !manager.same_source_manager(&self.manager) {
            return false;
        }
        match (&self.identity.geometry, &self.copies) {
            (
                LayerwiseGeometry::PreparedForeground(identity),
                LayerwiseCopies::Foreground(copies),
            ) => {
                identity == copies && manager.owns_original_foreground_workspace(identity, windows)
            }
            _ => false,
        }
    }
    pub(crate) fn retain_original_sources(
        &self,
        controls: &eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> Result<OriginalLayerwiseSourceCustody, Error> {
        match &self.copies {
            LayerwiseCopies::Host(copies) => copies
                .pin_retained_sources(&self.manager, controls)
                .map(OriginalLayerwiseSourceCustody::Host)
                .map_err(|cause| Error::Other(Box::new(cause))),
            LayerwiseCopies::Foreground(identity) => {
                identity
                    .validate_request_custody(controls)
                    .map_err(Error::PrefillControl)?;
                Ok(OriginalLayerwiseSourceCustody::Foreground(identity.clone()))
            }
            LayerwiseCopies::Disk { receipt, .. } => {
                receipt.validate()?;
                Ok(OriginalLayerwiseSourceCustody::Disk(receipt.clone()))
            }
        }
    }
    /// Same retained source/pin producer with already admitted operation custody.
    /// The consuming bank separately validates manager, selected policy, and role.
    pub(crate) fn retain_ready_host_sources(
        &self,
        custody: eredu_runtime::working_memory::OriginalOperationMetadataCustody,
    ) -> Result<OriginalLayerwiseSourceCustody, Error> {
        match &self.copies {
            LayerwiseCopies::Host(copies) => copies
                .pin_retained_sources_with_custody(&self.manager, custody)
                .map(OriginalLayerwiseSourceCustody::Host)
                .map_err(|cause| Error::Other(Box::new(cause))),
            LayerwiseCopies::Foreground(identity) => {
                identity
                    .validate_operation_custody(&custody)
                    .map_err(Error::PrefillControl)?;
                Ok(OriginalLayerwiseSourceCustody::Foreground(identity.clone()))
            }
            _ => Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )),
        }
    }

    /// Initial ready-host custody for the native original path. No disk guard
    /// is activated and no checkpoint provider is entered by this operation.
    pub(crate) fn pin_initial_original_sources(
        &self,
        controls: &eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> Result<Option<crate::backend::runtime::residency::manager::HostCopySourcePins>, Error>
    {
        match &self.copies {
            LayerwiseCopies::Host(copies) => copies
                .pin_initial_sources(&self.manager, controls)
                .map(Some)
                .map_err(|cause| Error::retained_original(cause, true)),
            LayerwiseCopies::Disk { .. } | LayerwiseCopies::Foreground(_) => Ok(None),
        }
    }

    pub(crate) fn pin_sources(
        &self,
    ) -> Result<Option<crate::backend::runtime::residency::manager::HostCopySourcePins>, Error>
    {
        match &self.copies {
            LayerwiseCopies::Host(copies) => copies
                .pin_sources(&self.manager)
                .map(Some)
                .map_err(|error| Error::Other(Box::new(error))),
            LayerwiseCopies::Disk { .. } | LayerwiseCopies::Foreground(_) => Ok(None),
        }
    }
    pub(crate) fn disk_receipt(&self) -> Option<DiskLayerwiseReceipt> {
        match &self.copies {
            LayerwiseCopies::Disk { receipt, .. } => Some(receipt.clone()),
            LayerwiseCopies::Host(_) | LayerwiseCopies::Foreground(_) => None,
        }
    }
    /// Consume the cold snapshot for legacy callers. No source, manager or disk
    /// trace cache lifetime is extended; the already-owned identity is moved.
    pub(crate) fn into_identity(self) -> LayerwiseWorkspaceIdentity {
        self.identity
    }
    pub(crate) fn identity(&self) -> &LayerwiseWorkspaceIdentity {
        &self.identity
    }
    pub(crate) fn layout(&self) -> &ExecutionUnitLayout {
        self.identity.layout()
    }
    pub(crate) fn execution_address(&self, ordinal: usize) -> Option<ExecutionUnitAddress> {
        match &self.identity.parameter_locations {
            Some(locations) => locations.get(ordinal).map(|&(global, _)| global),
            None => self.layout().address(ordinal),
        }
    }
    /// Exact initial policy mapping, retained without copying names or rows.
    /// Local layout/window semantics remain unchanged; module ids are global.
    pub(crate) fn with_parameter_locations(
        mut self,
        locations: Arc<[(ExecutionUnitAddress, ExecutionUnitAddress)]>,
        context: Option<&WorkspaceContext>,
    ) -> Result<Self, Error> {
        if let Some(context) = context {
            context.charge_metadata(std::mem::size_of::<(
                Self, Arc<[(ExecutionUnitAddress, ExecutionUnitAddress)]>,
                Option<&WorkspaceContext>, usize, ExecutionUnitAddress, ExecutionUnitAddress,
                Result<Self, Error>,
            )>()).map_err(|cause| Error::Neural(cause.into()))?;
        }
        let mismatch = || Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch);
        if self.identity.parameter_locations.is_some() || locations.len() != self.layout().len() {
            return Err(mismatch());
        }
        for (ordinal, &(global, local)) in locations.iter().enumerate() {
            if self.layout().address(ordinal) != Some(local) || global.group() != local.group()
                || locations[..ordinal].iter().any(|&(prior, _)| prior == global)
            { return Err(mismatch()); }
        }
        self.identity.parameter_locations = Some(locations);
        Ok(self)
    }
    pub(crate) fn materialization(&self) -> &WorkspaceBound {
        self.materialization.bound()
    }

    /// Future device destinations can allocate or share their source. Each
    /// separately dispatched name gets a conservative backing envelope. These
    /// are prospective parameter roots, never exact registered-state roots and
    /// never eligible for residual credit. Construction does not emit an
    /// allocation operation: its possible fresh bytes belong to materialization.
    pub(crate) fn parameters(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        context: &WorkspaceContext,
    ) -> Result<BTreeMap<eredu_nn::ParameterId, WorkspaceTensor>, eredu_nn::Error> {
        if self.execution_address(ordinal) != Some(address) {
            return Err(eredu_nn::Error::backend(
                "host parameter workspace address mismatch",
            ));
        }
        let unit = self
            .requested_unit(ordinal)
            .map_err(eredu_nn::Error::backend_retained_source)?;
        let rows = self
            .unit(unit)
            .map_err(eredu_nn::Error::backend_retained_source)?
            .rows;
        match &self.copies {
            LayerwiseCopies::Host(_) => (0..rows)
                .map(|index| {
                    let row = self
                        .row(unit, index)
                        .map_err(eredu_nn::Error::backend_retained_source)?;
                    let layout = WorkspaceLayout::new(row.shape, row.dtype)?;
                    let root = WorkspaceExistingStorage::new(Some(row.capacity_bytes), context);
                    Ok((
                        eredu_nn::ParameterId::new(row.binding.name())
                            .map_err(eredu_nn::Error::backend_retained_source)?,
                        WorkspaceTensor::existing_with_storage(layout, &root, context)?,
                    ))
                })
                .collect(),
            LayerwiseCopies::Disk { .. } | LayerwiseCopies::Foreground(_) => {
                let mut cached = self.persistent_roots.borrow_mut();
                if !cached
                    .as_ref()
                    .is_some_and(|(current, _)| current.shares_trace(context))
                {
                    *cached = Some((context.clone(), BTreeMap::new()));
                }
                let roots = &mut cached.as_mut().expect("initialized context roots").1;
                let mut local = BTreeMap::new();
                (0..rows)
                    .map(|index| {
                        let row = self
                            .row(unit, index)
                            .map_err(eredu_nn::Error::backend_retained_source)?;
                        let owner_unit = self
                            .unit(row.owner.unit)
                            .map_err(eredu_nn::Error::backend_retained_source)?;
                        let owner_row = self
                            .row(row.owner.unit, row.owner.row)
                            .map_err(eredu_nn::Error::backend_retained_source)?;
                        let map = if row.owner.lifetime == WorkspaceParameterLifetime::Trace {
                            &mut *roots
                        } else {
                            &mut local
                        };
                        let key = (owner_unit.id.clone(), owner_row.binding.name().to_owned());
                        let root = map.entry(key).or_insert_with(|| {
                            WorkspaceExistingStorage::new(Some(row.capacity_bytes), context)
                        });
                        if root.capacity_bytes() != Some(row.capacity_bytes) {
                            return Err(eredu_nn::Error::backend(
                                "canonical disk owner capacity differs between aliases",
                            ));
                        }
                        let layout = WorkspaceLayout::new(row.shape, row.dtype)?;
                        Ok((
                            eredu_nn::ParameterId::new(row.binding.name())
                                .map_err(eredu_nn::Error::backend_retained_source)?,
                            WorkspaceTensor::existing_with_storage(layout, root, context)?,
                        ))
                    })
                    .collect()
            }
        }
    }
}

impl<U: 'static, P: MlxUnitPopulator<U>> MlxLayerwisePolicy<U, P> {
    pub(crate) fn layerwise_workspace(
        &self,
        allocation: MetalAllocationFacts,
    ) -> Result<LayerwiseWorkspace, Error> {
        self.layerwise_workspace_impl(allocation, None)
    }
    pub(crate) fn layerwise_workspace_with_metadata(
        &self,
        allocation: MetalAllocationFacts,
        context: &WorkspaceContext,
    ) -> Result<LayerwiseWorkspace, Error> {
        let parts = [
            std::mem::size_of::<LayerwiseWorkspace>(),
            std::mem::size_of::<Result<LayerwiseWorkspace, Error>>(),
            std::mem::size_of::<Option<&WorkspaceContext>>(),
            std::mem::size_of::<(
                &P, Option<&MlxParameterExclusions>, Option<MlxParameterExclusions>,
            )>(),
            std::mem::size_of::<HostCopyWorkspace>(),
            std::mem::size_of::<
                Result<
                    HostCopyWorkspace,
                    crate::backend::runtime::residency::manager::HostCopyWorkspaceError,
                >,
            >(),
            if self.dense.is_some() {
                ForegroundDiskIdentity::qualifier_control_bytes().ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                ))?
            } else {
                0
            },
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| Error::Neural(cause.into()))?;
        self.layerwise_workspace_impl(allocation, Some(context))
    }
    fn layerwise_workspace_impl(
        &self,
        allocation: MetalAllocationFacts,
        context: Option<&WorkspaceContext>,
    ) -> Result<LayerwiseWorkspace, Error> {
        let unknown = || match context {
            Some(_) => Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ),
            None => Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )),
        };
        if !self.pending.is_empty()
            || !self.populator.preserves_prepared_parameter_source()
            || (context.is_some() && self.populator.prepared_parameter_exclusions()
                .is_some_and(|names| !names.is_source_funded()))
        {
            return Err(unknown());
        }
        if let Some(dense) = &self.dense {
            // Original background execution borrows this same exact Device
            // declaration; its separate Host window/worker is retained by the
            // operation plan. An ordinary background controller cannot qualify.
            let prepared_background = dense.controller.original_background_options().is_some();
            if !self.dense_window_matches_layout()
                || (!dense.controller.is_foreground() && !prepared_background)
                || dense.aborted
                || dense.forward.is_some()
                || dense.windows.iter().any(Option::is_some)
                || dense.groups.iter().any(Option::is_some)
            {
                return Err(unknown());
            }
            if let Some(identity) = self.residency.original_foreground_workspace() {
                if !identity.matches_selection(&self.unit_ids, &self.layout, self.window_depth) {
                    return Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    ));
                }
                return Ok(LayerwiseWorkspace {
                    manager: self.residency.clone(),
                    identity: LayerwiseWorkspaceIdentity {
                        policy: Arc::clone(&self.workspace_identity),
                    parameter_locations: None,
                    excluded: self.populator.prepared_parameter_exclusions().cloned(),
                    manager_unit_constructors: true,
                        geometry: LayerwiseGeometry::PreparedForeground(identity.clone()),
                    },
                    copies: LayerwiseCopies::Foreground(identity.clone()),
                    persistent_roots: RefCell::new(None),
                    speculative_foreground: std::cell::OnceCell::new(),
                    execution_trace: RefCell::new(None),
                    materialization: LayerwiseMaterialization::PreparedForeground(identity.clone()),
                });
            }
            if context.is_some() || prepared_background {
                return Err(unknown());
            }
            return self.disk_workspace(allocation);
        }
        let copies = match context {
            Some(context) => self
                .residency
                .prepared_host_copy_workspace(&self.unit_ids, allocation)
                .map_err(|cause| Error::Neural(context.metadata_source(cause))),
            None => self
                .residency
                .host_copy_workspace(&self.unit_ids, allocation)
                .map_err(|error| Error::Other(Box::new(error))),
        }?;
        if let Some(identity) = copies.prepared_identity() {
            if identity.layout() != &self.layout || identity.depth() != self.window_depth {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            }
            return Ok(LayerwiseWorkspace {
                manager: self.residency.clone(),
                identity: LayerwiseWorkspaceIdentity {
                    policy: Arc::clone(&self.workspace_identity),
                    parameter_locations: None,
                    excluded: self.populator.prepared_parameter_exclusions().cloned(),
                    manager_unit_constructors: true,
                    geometry: LayerwiseGeometry::PreparedHost(identity.clone()),
                },
                materialization: LayerwiseMaterialization::PreparedHost(identity.clone()),
                copies: LayerwiseCopies::Host(copies),
                persistent_roots: RefCell::new(None),
                speculative_foreground: std::cell::OnceCell::new(),
                    execution_trace: RefCell::new(None),
            });
        }
        if context.is_some() {
            return Err(unknown());
        }
        let units = copies.units().iter().map(|unit| WorkspaceBound::bounded(
            unit.fresh_capacity_bytes(),
            "every selected host-copy name can allocate one destination; Metal capacity rounding and oversized reuse included; existing host aliases are already charged; no separate numeric staging",
        )).collect::<Vec<_>>();
        let materialization = eredu_runtime::working_memory::quote_completed_layerwise_window(
            &self.layout,
            std::num::NonZeroUsize::new(self.window_depth).expect("validated window depth"),
            &units,
        )
        .map_err(|error| Error::Other(Box::new(error)))?;
        let sources = copies
            .units()
            .iter()
            .map(|unit| {
                (
                    unit.definition().clone(),
                    copies
                        .copies(unit)
                        .iter()
                        .map(|copy| {
                            (
                                copy.name().to_owned(),
                                copy.source_allocation().identity(),
                                copy.source_allocation().bytes() as u64,
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        Ok(LayerwiseWorkspace {
            manager: self.residency.clone(),
            identity: LayerwiseWorkspaceIdentity {
                policy: Arc::clone(&self.workspace_identity),
                    parameter_locations: None,
                    excluded: self.populator.prepared_parameter_exclusions().cloned(),
                    manager_unit_constructors: true,
                geometry: LayerwiseGeometry::Owned {
                    layout: self.layout.clone(),
                    depth: self.window_depth,
                    sources: LayerwiseSourcesIdentity::Host(sources),
                },
            },
            copies: LayerwiseCopies::Host(copies),
            persistent_roots: RefCell::new(None),
            speculative_foreground: std::cell::OnceCell::new(),
                    execution_trace: RefCell::new(None),
            materialization: LayerwiseMaterialization::Owned(materialization),
        })
    }
}

impl<U: 'static, P: MlxUnitPopulator<U>> MlxLayerwisePolicy<U, P> {
    fn disk_workspace(
        &self,
        allocation: MetalAllocationFacts,
    ) -> Result<LayerwiseWorkspace, Error> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        let plans = self
            .disk_reads
            .as_ref()
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::UnknownBound)))?
            .as_ref()
            .map_err(|error| Error::Other(Box::new(Arc::clone(error))))?;
        let copies = self
            .residency
            .disk_copy_workspace(plans, allocation)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let persistent = copies.persistent_units();
        let mut persistent_bytes = 0_u64;
        for unit in copies
            .units()
            .iter()
            .filter(|unit| persistent.contains(unit.id()))
        {
            persistent_bytes = persistent_bytes
                .checked_add(unit.fresh_capacity_bytes())
                .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::Overflow)))?;
        }
        let units = self.unit_ids.iter().map(|id| {
            let unit = copies.units().iter().find(|unit| unit.id() == id)
                .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)))?;
            Ok(WorkspaceBound::bounded(if persistent.contains(id) { 0 } else { unit.fresh_capacity_bytes() },
                "exact direct owner outputs; alias names share canonical backing; no numeric staging"))
        }).collect::<Result<Vec<_>, Error>>()?;
        let depth = std::num::NonZeroUsize::new(self.window_depth).expect("validated depth");
        let window = eredu_runtime::working_memory::quote_completed_layerwise_window(
            &self.layout,
            depth,
            &units,
        )
        .map_err(|error| Error::Other(Box::new(error)))?;
        let bytes = window
            .bytes()
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::UnknownBound)))?
            .checked_add(persistent_bytes)
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::Overflow)))?;
        let materialization = WorkspaceBound::bounded(
            bytes,
            format!(
                "{window:?}; separately retained complete canonical-owner unit closure: {persistent_bytes} bytes"
            ),
        );
        let windows = (0..self.layout.len())
            .map(|ordinal| {
                let range = self
                    .layout
                    .window_range(ordinal, depth)
                    .expect("validated ordinal");
                self.unit_ids[range].to_vec()
            })
            .collect();
        let receipt = DiskLayerwiseReceipt(
            copies
                .receipt(windows)
                .map_err(|error| Error::Other(Box::new(error)))?,
        );
        Ok(LayerwiseWorkspace {
            manager: self.residency.clone(),
            identity: LayerwiseWorkspaceIdentity {
                policy: Arc::clone(&self.workspace_identity),
                    parameter_locations: None,
                    excluded: self.populator.prepared_parameter_exclusions().cloned(),
                    manager_unit_constructors: true,
                geometry: LayerwiseGeometry::Owned {
                    layout: self.layout.clone(),
                    depth: self.window_depth,
                    sources: LayerwiseSourcesIdentity::Disk,
                },
            },
            copies: LayerwiseCopies::Disk { copies, receipt },
            persistent_roots: RefCell::new(None),
            speculative_foreground: std::cell::OnceCell::new(),
                    execution_trace: RefCell::new(None),
            materialization: LayerwiseMaterialization::Owned(materialization),
        })
    }
}


// The source itself implements the portable loan. Composition wrappers need
// not reopen private model modules or copy their parameter/source directory.
impl eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters for LayerwiseWorkspace {
    fn excludes_parameter(&self, name: &str) -> bool {
        LayerwiseWorkspace::excludes_parameter(self, name)
    }

    fn observe_acquire(&self,ordinal:usize,address:ExecutionUnitAddress,context:&WorkspaceContext)
        ->Result<(),eredu_nn::Error> {self.observe_execution_acquire(ordinal,address,context)}

    fn layout(&self)->&eredu_runtime::ExecutionUnitLayout { LayerwiseWorkspace::layout(self) }
    fn execution_address(&self,ordinal:usize)->Option<eredu_runtime::ExecutionUnitAddress> {
        LayerwiseWorkspace::execution_address(self,ordinal)
    }
    fn parameter_source(&self)->Result<eredu_runtime::working_memory::WorkspaceParameterSourceLoan<'_>,
        eredu_runtime::working_memory::WorkspaceParameterSourceError> {
        Ok(LayerwiseWorkspace::parameter_source(self))
    }
    fn parameters(&self,ordinal:usize,address:eredu_runtime::ExecutionUnitAddress,context:&WorkspaceContext)
        ->Result<BTreeMap<eredu_nn::ParameterId,WorkspaceTensor>,eredu_nn::Error> {
        LayerwiseWorkspace::parameters(self,ordinal,address,context)
    }
}
