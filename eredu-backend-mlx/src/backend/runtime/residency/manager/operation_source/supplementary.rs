//! Source-owned supplementary inventory, separate from target traversal policy.
use super::*;

/// Unique physical source rows. Repeated or nested invocations map to these IDs
/// in the consuming bank; this inventory neither deduplicates calls nor grants
/// a role. No strong manager backedge is retained.
#[derive(Clone, Debug)]
pub(crate) struct SupplementaryResidencySource {
    ids: Arc<Vec<OffloadUnitId>>,
    policy: Arc<()>,
    source: OriginalResidencySource,
    host: Option<HostCopyWorkspace>,
    _custody: ManagerCustody,
}
impl SupplementaryResidencySource {
    pub(crate) fn ids(&self) -> &[OffloadUnitId] {
        &self.ids
    }
    pub(crate) fn ordinal(&self, id: &OffloadUnitId) -> Option<usize> {
        self.ids.iter().position(|actual| actual == id)
    }
    pub(crate) fn source(&self) -> &OriginalResidencySource {
        &self.source
    }
    pub(crate) fn policy_identity(&self) -> &Arc<()> {
        &self.policy
    }
    pub(crate) fn host(&self) -> Option<&HostCopyWorkspace> {
        self.host.as_ref()
    }
    pub(crate) fn foreground(&self) -> Option<&ForegroundDiskIdentity> {
        match self.source.selection.as_ref()? {
            OperationSelection::Foreground(identity) => Some(identity),
            _ => None,
        }
    }
    pub(crate) fn projection_control_bytes(&self) -> Option<usize> {
        let controls = [
            size_of::<(&ResidencyManager, &Self)>(),
            size_of::<Option<&Self>>(),
            size_of::<Result<(), OperationSourceFailure>>(),
            size_of::<(&OriginalResidencySource, &OperationWindows)>(),
            size_of::<Option<&ManagerWeak>>(),
            size_of::<Option<&HostCopyWorkspace>>(),
            size_of::<Option<&ForegroundDiskIdentity>>(),
            size_of::<Option<&HostCopyIdentity>>(),
            size_of::<bool>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)?;
        bytes.checked_add(match self.host.as_ref() {
            Some(host) => host.source_validation_control_bytes()?,
            None => ForegroundDiskIdentity::qualifier_control_bytes()?,
        })
    }
    pub(in crate::backend::runtime::residency::manager) fn constructor_bytes(
        ids: &[OffloadUnitId],
    ) -> Option<usize> {
        use eredu_runtime::working_memory::OriginalHostMetadataCustody as C;
        let bytes =
            usize::try_from(C::shared_storage_bytes(Layout::new::<Vec<OffloadUnitId>>()).ok()?)
                .ok()?
                .checked_add(
                    usize::try_from(C::shared_storage_bytes(Layout::new::<()>()).ok()?).ok()?,
                )?
                .checked_add(Layout::array::<OffloadUnitId>(ids.len()).ok()?.size())?;
        let bytes = ids
            .iter()
            .try_fold(bytes, |sum, id| sum.checked_add(id.as_str().len()))?;
        let controls = [
            size_of::<Self>(),
            size_of::<Vec<OffloadUnitId>>(),
            size_of::<Arc<()>>(),
            size_of::<Result<Self, OperationSourceFailure>>(),
            size_of::<Result<(), Self>>(),
            size_of::<std::slice::Iter<'_, OffloadUnitId>>(),
            size_of::<Option<HostCopyWorkspace>>(),
            size_of::<OriginalResidencySource>(),
            size_of::<Result<OriginalResidencySource, OperationSourceFailure>>(),
            size_of::<NonZeroUsize>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<std::iter::Cloned<std::slice::Iter<'_, OffloadUnitId>>>(),
            size_of::<(
                &[OffloadUnitId],
                OriginalResidencySource,
                Option<HostCopyWorkspace>,
                ManagerCustody,
            )>(),
        ];
        controls.into_iter().try_fold(
            bytes.checked_add(std::mem::size_of_val(&controls))?,
            usize::checked_add,
        )
    }
    pub(in crate::backend::runtime::residency::manager) fn construct(
        ids: &[OffloadUnitId],
        source: OriginalResidencySource,
        host: Option<HostCopyWorkspace>,
        custody: ManagerCustody,
    ) -> Result<Self, OperationSourceFailure> {
        // Storage is paid by the same unpublished manager initializer. Clone
        // source IDs into that account; never adopt the cold planner's Vec.
        let mut retained = Vec::new();
        retained
            .try_reserve_exact(ids.len())
            .map_err(OperationSourceFailure::Reserve)?;
        retained.extend(ids.iter().cloned());
        Ok(Self {
            ids: Arc::new(retained),
            policy: Arc::new(()),
            source,
            host,
            _custody: custody,
        })
    }
}
impl ResidencyManager {
    pub(crate) fn supplementary_residency_source(&self) -> Option<&SupplementaryResidencySource> {
        self.inner.supplementary_source.get()
    }
    pub(crate) fn validate_supplementary_source(
        &self,
        source: &SupplementaryResidencySource,
    ) -> Result<(), OperationSourceFailure> {
        let actual = self
            .inner
            .supplementary_source
            .get()
            .ok_or(OperationSourceFailure::Layout)?;
        if !Arc::ptr_eq(&actual.policy, &source.policy)
            || !Arc::ptr_eq(&actual.ids, &source.ids)
            || !self.owns_source_windows(&actual.source, &source.source.windows)
        {
            return Err(OperationSourceFailure::Layout);
        }
        Ok(())
    }
    pub(in crate::backend::runtime::residency::manager) fn owns_original_host_workspace(
        &self,
        copies: &HostCopyWorkspace,
        windows: &OperationWindows,
    ) -> bool {
        let target = self
            .inner
            .original_operation_source
            .get()
            .zip(self.inner.host_workspace.get());
        let supplementary = self
            .inner
            .supplementary_source
            .get()
            .and_then(|s| s.host.as_ref().map(|h| (&s.source, h)));
        target
            .into_iter()
            .chain(supplementary)
            .any(|(source, actual)| {
                self.owns_source_windows(source, windows) && copies.same_snapshot(actual)
            })
    }
    pub(crate) fn validate_supplementary_host(
        &self,
        source: &SupplementaryResidencySource,
    ) -> Result<(), HostCopyWorkspaceError> {
        source
            .host
            .as_ref()
            .ok_or_else(|| {
                HostCopyWorkspaceError::Storage(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            })?
            .validate_original_snapshot(self)
    }
}

impl ResidencyManager {
    /// Exact dynamically selected roots from this retained source domain. The
    /// canonical closure and every binding/recipe extent use the ordinary
    /// collector; no maximum-sized controller attempt is substituted.
    pub(crate) fn selected_operation_population(
        &self, source: &SupplementaryResidencySource, roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<WindowPopulation, OperationSourceFailure> {
        self.validate_supplementary_source(source)?;
        if roots.is_empty() || roots.iter().enumerate().any(|(index, id)|
            !source.ids().contains(id) || roots[..index].contains(id)) {
            return Err(OperationSourceFailure::Layout);
        }
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => OperationSourceFailure::Busy,
            std::sync::TryLockError::Poisoned(_) => OperationSourceFailure::Poisoned,
        })?;
        let mut population = WindowPopulation::collect(&state.control, roots, scratch)?;
        population.request_start = 0;
        population.request_end = roots.len();
        Ok(population)
    }
}

impl ResidencyManager {
    /// Source-derived ceiling for any nonempty subset of this retained member
    /// domain. Work scales with the declaration count, never token populations
    /// or the combinatorial number of possible demand sets. This describes
    /// storage only: acquisition recomputes the exact actual selected closure.
    pub(crate) fn selected_operation_ceiling(
        &self,source:&SupplementaryResidencySource,eligible:&[OffloadUnitId],
        maximum_requested:usize,scratch:&mut [ResidencyClosureSlot],
    )->Result<WindowPopulation,OperationSourceFailure> {
        self.validate_supplementary_source(source)?;
        if maximum_requested==0 || maximum_requested>eligible.len()
            || eligible.iter().enumerate().any(|(index,id)|
                !source.ids().contains(id) || eligible[..index].contains(id)) {
            return Err(OperationSourceFailure::Layout);
        }
        let state=self.inner.state.try_lock().map_err(|cause|match cause {
            std::sync::TryLockError::WouldBlock=>OperationSourceFailure::Busy,
            std::sync::TryLockError::Poisoned(_)=>OperationSourceFailure::Poisoned,
        })?;
        let full=WindowPopulation::collect(&state.control,eligible,scratch)?;
        let mut maximum=WindowPopulation::default();
        for id in eligible {
            let one=WindowPopulation::collect(&state.control,std::slice::from_ref(id),scratch)?;
            maximum.component_maximum(one);
        }
        maximum.subset_ceiling(full,maximum_requested).ok_or(OperationSourceFailure::Overflow)
    }
}
impl WindowPopulation {
    fn component_maximum(&mut self,other:Self) {
        self.requested_id_bytes=self.requested_id_bytes.max(other.requested_id_bytes);
        self.units=self.units.max(other.units);
        self.unit_id_bytes=self.unit_id_bytes.max(other.unit_id_bytes);
        self.bindings=self.bindings.max(other.bindings);
        self.physical_bindings=self.physical_bindings.max(other.physical_bindings);
        self.aliases=self.aliases.max(other.aliases);
        self.local_aliases=self.local_aliases.max(other.local_aliases);
        self.binding_name_bytes=self.binding_name_bytes.max(other.binding_name_bytes);
        self.alias_owner_name_bytes=self.alias_owner_name_bytes.max(other.alias_owner_name_bytes);
        self.physical_bytes=self.physical_bytes.max(other.physical_bytes);
        self.recipe_pending=self.recipe_pending.max(other.recipe_pending);
        self.recipe_materializations=self.recipe_materializations.max(other.recipe_materializations);
        self.recipe_depth=self.recipe_depth.max(other.recipe_depth);
        self.declaration_clone_bytes=self.declaration_clone_bytes.max(other.declaration_clone_bytes);
        self.declaration_clone_allocations=self.declaration_clone_allocations.max(other.declaration_clone_allocations);
        self.named.catalog_requested_bytes=self.named.catalog_requested_bytes.max(other.named.catalog_requested_bytes);
        self.named.catalog_allocations=self.named.catalog_allocations.max(other.named.catalog_allocations);
        self.named.destination_requested_bytes=self.named.destination_requested_bytes.max(other.named.destination_requested_bytes);
        self.named.destination_allocations=self.named.destination_allocations.max(other.named.destination_allocations);
        self.named.units=self.named.units.max(other.named.units);
        self.named.rows=self.named.rows.max(other.named.rows);
        self.named.physical_cells=self.named.physical_cells.max(other.named.physical_cells);
    }
    fn subset_ceiling(self,full:Self,count:usize)->Option<Self> {
        // A union of k singleton closures is bounded both by their k maxima
        // and by the complete eligible closure. All these fields are additive
        // nonnegative sizes/counts; recipe depth is a maximum, not a sum.
        let mut out=full;
        out.request_start=0;
        out.request_end=count;
        out.requested=count;
        out.requested_id_bytes=full.requested_id_bytes.min(self.requested_id_bytes.checked_mul(count)?);
        out.units=full.units.min(self.units.checked_mul(count)?);
        out.unit_id_bytes=full.unit_id_bytes.min(self.unit_id_bytes.checked_mul(count)?);
        out.bindings=full.bindings.min(self.bindings.checked_mul(count)?);
        out.physical_bindings=full.physical_bindings.min(self.physical_bindings.checked_mul(count)?);
        out.aliases=full.aliases.min(self.aliases.checked_mul(count)?);
        out.local_aliases=full.local_aliases.min(self.local_aliases.checked_mul(count)?);
        out.binding_name_bytes=full.binding_name_bytes.min(self.binding_name_bytes.checked_mul(count)?);
        out.alias_owner_name_bytes=full.alias_owner_name_bytes.min(self.alias_owner_name_bytes.checked_mul(count)?);
        out.physical_bytes=full.physical_bytes.min(self.physical_bytes.checked_mul(u64::try_from(count).ok()?)?);
        out.recipe_pending=full.recipe_pending.min(self.recipe_pending.checked_mul(count)?);
        out.recipe_materializations=full.recipe_materializations.min(self.recipe_materializations.checked_mul(count)?);
        out.recipe_depth=self.recipe_depth;
        out.declaration_clone_bytes=full.declaration_clone_bytes.min(self.declaration_clone_bytes.checked_mul(count)?);
        out.declaration_clone_allocations=full.declaration_clone_allocations.min(self.declaration_clone_allocations.checked_mul(count)?);
        out.named.catalog_requested_bytes=full.named.catalog_requested_bytes.min(self.named.catalog_requested_bytes.checked_mul(count)?);
        out.named.catalog_allocations=full.named.catalog_allocations.min(self.named.catalog_allocations.checked_mul(count)?);
        out.named.destination_requested_bytes=full.named.destination_requested_bytes.min(self.named.destination_requested_bytes.checked_mul(count)?);
        out.named.destination_allocations=full.named.destination_allocations.min(self.named.destination_allocations.checked_mul(count)?);
        out.named.units=full.named.units.min(self.named.units.checked_mul(count)?);
        out.named.rows=full.named.rows.min(self.named.rows.checked_mul(count)?);
        out.named.physical_cells=full.named.physical_cells.min(self.named.physical_cells.checked_mul(count)?);
        Some(out)
    }
}
