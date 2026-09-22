//! Original metadata/read descriptors for a selected foreground disk source.
//! No native transfer, prefetch worker, residency publication or fit is granted.
use super::*;
use eredu_checkpoint::{
    recipe::{EncodedRecipeRead, RecipeMetadata},
    store::RetainedCheckpointSource,
};
use source::{OriginalHostCatalogPlan, OriginalReadSources};
use std::sync::Arc;
mod ordinary;
mod read;
pub(crate) use read::{
    ForegroundDiskReadError, ForegroundDiskReadLayout, ForegroundDiskReadPlan,
    ForegroundMaterializationPopulation, PreparedForegroundDiskIo, PreparedForegroundDiskRead,
    ReadForegroundDiskBatch,
};

struct UnitDescriptor {
    definition: OffloadUnit,
    canonical_reads: Vec<usize>,
    own_reads: std::ops::Range<usize>,
    own_leaves: std::ops::Range<usize>,
}
struct NativeRead {
    shape: Vec<i32>,
    dtype: safemlx::Dtype,
    leaves: std::ops::Range<usize>,
    materialized: Option<Arc<read_source_plan::MaterializedReadPlan>>,
}
struct DescriptorData {
    peak_source: eredu_runtime::working_memory::HostSourcePeakSelection,
    units: Vec<UnitDescriptor>,
    source: OriginalReadSources,
    native_reads: Vec<NativeRead>,
}
/// Immutable descriptor aliases retain the same source account through their
/// final Arc release. No owning read-plan, native stream or weak grant escapes.
#[derive(Clone)]
pub(crate) struct ForegroundDiskDescriptors {
    value: Arc<DescriptorData>,
    _custody: ManagerCustody,
}
impl ForegroundDiskDescriptors {
    pub(crate) fn peak_selection(
        &self,
        bytes: u64,
    ) -> eredu_runtime::working_memory::HostSourcePeakSelection {
        self.value.peak_source.with_backing_bytes(bytes)
    }
    /// Exact retained descriptor ownership; equal geometry is not authority.
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
    /// Borrow actual canonical native read geometry without cloning metadata.
    /// Aliases use these same owners through their validated canonical indexes.
    pub(crate) fn native_reads(&self) -> impl Iterator<Item = (&[i32], safemlx::Dtype)> {
        self.value
            .native_reads
            .iter()
            .map(|read| (read.shape.as_slice(), read.dtype))
    }
    pub(super) fn read_source(&self) -> &OriginalReadSources {
        &self.value.source
    }
    /// Native geometry for the existing canonical alias target of one binding.
    pub(crate) fn native_output(
        &self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Option<(&[i32], safemlx::Dtype)> {
        self.native_copy_output(id, name)
            .map(|(shape, dtype, _)| (shape, dtype))
    }
    /// Both encoded and materialized sources finish in the same immutable
    /// PreparedHostTransferPlan destination. The following native copy has
    /// that constructor's layout, independently of the producer equation.
    pub(crate) fn native_copy_output(
        &self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Option<(&[i32], safemlx::Dtype, safemlx::HostTransferArrayLayout)> {
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.definition.id() == id)?;
        let binding = unit
            .definition
            .bindings()
            .binary_search_by(|binding| binding.name().cmp(name))
            .ok()?;
        let read = self
            .value
            .native_reads
            .get(*unit.canonical_reads.get(binding)?)?;
        Some((
            read.shape.as_slice(),
            read.dtype,
            safemlx::PreparedHostTransferPlan::COPY_OUTPUT_LAYOUT,
        ))
    }
    pub(crate) fn units(&self) -> impl Iterator<Item = &OffloadUnit> {
        self.value.units.iter().map(|unit| &unit.definition)
    }
    pub(crate) fn output(&self, id: &OffloadUnitId, name: &str) -> Option<&RecipeMetadata> {
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.definition.id() == id)?;
        let ordinal = unit
            .definition
            .bindings()
            .binary_search_by(|binding| binding.name().cmp(name))
            .ok()?;
        self.read_output(*unit.canonical_reads.get(ordinal)?)
    }
    fn read_output(&self, index: usize) -> Option<&RecipeMetadata> {
        let read = self.value.native_reads.get(index)?;
        match &read.materialized {
            Some(plan) => Some(&plan.output),
            None if read.leaves.len() == 1 => self.value.source.read_output(read.leaves.start),
            None => None,
        }
    }

    pub(crate) fn source(
        &self,
        id: &OffloadUnitId,
    ) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.value.source.catalog(id)
    }
    pub(crate) fn policy_source(&self) -> &RetainedCheckpointSource {
        self.value.source.policy()
    }
    /// Cold adoption authenticates exact declarations and all source catalogs,
    /// then revalidates each immutable admitted file/range recipe. It performs
    /// no payload reads and retains no incoming ordinary catalog/cache owner.
    pub(crate) fn validate(
        &self,
        primary: &RetainedCheckpointSource,
        sources: &BTreeMap<OffloadUnitId, RetainedCheckpointSource>,
        units: &[OffloadUnit],
    ) -> Result<(), ResidencyError> {
        if units.len() != self.value.units.len()
            || !self
                .value
                .source
                .matches_primary_catalog(primary.as_ref())?
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        for (ordinal, (unit, retained)) in units.iter().zip(&self.value.units).enumerate() {
            if unit != &retained.definition {
                return Err(ResidencyError::OriginalOperationDomain);
            }
            let incoming = sources.get(unit.id()).unwrap_or(primary);
            let already = (self.value.source.same_primary_catalog(unit.id())
                && incoming.same_source(primary))
                || units[..ordinal].iter().any(|prior| {
                    self.value.source.same_catalog(prior.id(), unit.id())
                        && incoming.same_source(sources.get(prior.id()).unwrap_or(primary))
                });
            if (!already
                && !self
                    .value
                    .source
                    .matches_catalog(unit.id(), incoming.as_ref())?)
                || !self.value.source.matches_reads(unit, incoming.as_ref())?
            {
                return Err(ResidencyError::OriginalOperationDomain);
            }
        }
        Ok(())
    }
}

struct DescriptorPlan<'a> {
    units: &'a [OffloadUnit],
    reads: &'a [PlannedRead],
    catalogs: OriginalHostCatalogPlan,
    canonical_reads: Vec<Vec<usize>>,
}
impl DescriptorPlan<'_> {
    fn source_plan(&self) -> ReadSourcePlan<'_> {
        ReadSourcePlan {
            units: self.units,
            reads: self.reads,
            catalogs: &self.catalogs,
        }
    }
}
impl SharedNativeInitializer for DescriptorPlan<'_> {
    type Output = ForegroundDiskDescriptors;
    type Error = ConstructionError;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let overflow = || WorkingMemoryError::Overflow;
        let detached = eredu_checkpoint::recipe::EncodedRecipeReadView::prepare_detached(
            self.source_plan()
                .encoded_leaves()
                .ok_or(WorkingMemoryError::Overflow)?,
        )
        .ok_or(WorkingMemoryError::UnknownBound)?;
        let mut bytes = OriginalReadSources::storage_bytes(&self.source_plan())?
            .checked_add(
                detached
                    .required_bytes::<ManagerCustody>()
                    .ok_or_else(overflow)?,
            )
            .and_then(|n| {
                n.checked_add(
                    Layout::array::<UnitDescriptor>(self.units.len())
                        .ok()?
                        .size(),
                )
            })
            .ok_or_else(overflow)?;
        bytes = bytes
            .checked_add(
                Layout::array::<NativeRead>(self.reads.len())
                    .map_err(|_| overflow())?
                    .size(),
            )
            .ok_or_else(overflow)?;
        for row in self.reads {
            bytes = bytes
                .checked_add(
                    Layout::array::<i32>(row.read.shape().len())
                        .map_err(|_| overflow())?
                        .size(),
                )
                .ok_or_else(overflow)?;
        }
        for (unit, reads) in self.units.iter().zip(&self.canonical_reads) {
            bytes = bytes
                .checked_add(
                    super::super::operation_source::unit_clone_payload_bytes(unit)
                        .ok_or_else(overflow)?,
                )
                .and_then(|n| n.checked_add(Layout::array::<usize>(reads.len()).ok()?.size()))
                .ok_or_else(overflow)?;
        }
        for n in [
            usize::try_from(ManagerCustody::storage_bytes()?).map_err(|_| overflow())?,
            usize::try_from(OriginalHostMetadataCustody::shared_storage_bytes(
                Layout::new::<DescriptorData>(),
            )?)
            .map_err(|_| overflow())?,
            size_of::<DescriptorData>(),
            size_of::<ForegroundDiskDescriptors>(),
            size_of::<UnitDescriptor>(),
            size_of::<NativeRead>(),
            size_of::<Vec<NativeRead>>(),
            size_of::<Vec<UnitDescriptor>>(),
            size_of::<ConstructionError>(),
            size_of::<Result<ForegroundDiskDescriptors, ConstructionError>>(),
            size_of::<ForegroundDiskSourceError>(),
        ] {
            bytes = bytes.checked_add(n).ok_or_else(overflow)?;
        }
        Ok(bytes)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        let custody = ManagerCustody::new(custody);
        let result = (|| -> Result<_, ConstructionCause> {
            let detached = eredu_checkpoint::recipe::EncodedRecipeReadView::prepare_detached(
                self.source_plan()
                    .encoded_leaves()
                    .ok_or(ResidencyError::OriginalCache(WorkingMemoryError::Overflow))?,
            )
            .ok_or(ResidencyError::OriginalOperationDomain)?
            .construct(custody.clone())?;
            let source = OriginalReadSources::new(&self.source_plan(), detached, custody.clone())?;
            let mut units = Vec::with_capacity(self.units.len());
            for (ordinal, (unit, reads)) in self.units.iter().zip(&self.canonical_reads).enumerate()
            {
                units.push(UnitDescriptor {
                    definition: unit.clone(),
                    canonical_reads: reads.clone(),
                    own_reads: self.source_plan().read_range(ordinal),
                    own_leaves: self
                        .source_plan()
                        .leaf_range(self.source_plan().read_range(ordinal))
                        .ok_or(ResidencyError::OriginalCache(WorkingMemoryError::Overflow))?,
                });
            }
            Ok(ForegroundDiskDescriptors {
                value: Arc::new(DescriptorData {
                    peak_source: eredu_runtime::working_memory::HostSourcePeakSelection::new(0)
                        .map_err(ConstructionCause::SourceSelection)?,
                    units,
                    source,
                    native_reads: self
                        .reads
                        .iter()
                        .enumerate()
                        .map(|(index, row)| {
                            Ok(NativeRead {
                                shape: row.read.shape().to_vec(),
                                dtype: row.read.dtype(),
                                leaves: self.source_plan().leaf_range(index..index + 1).ok_or(
                                    ResidencyError::OriginalCache(WorkingMemoryError::Overflow),
                                )?,
                                materialized: match &row.read {
                                    read_source_plan::ReadValue::Direct(_) => None,
                                    read_source_plan::ReadValue::Materialized(plan) => {
                                        Some(plan.shared())
                                    }
                                },
                            })
                        })
                        .collect::<Result<Vec<_>, ConstructionCause>>()?,
                }),
                _custody: custody.clone(),
            })
        })();
        result.map_err(|cause| ConstructionError {
            cause,
            _custody: custody,
        })
    }
}
#[derive(Debug, thiserror::Error)]
enum ForegroundDiskFailure {
    #[error(transparent)]
    Store(#[from] eredu_checkpoint::store::StoreError),
    #[error(transparent)]
    Recipe(#[from] WeightRecipeError),
    #[error(transparent)]
    Policy(#[from] WorkingMemoryError),
    #[error(transparent)]
    Layout(#[from] eredu_runtime::residency::ResidencyConstructionError),
    #[error(transparent)]
    Declaration(#[from] ResidencyControllerError),
    #[error("foreground disk descriptor construction: {0}")]
    Constructor(
        #[source]
        eredu_runtime::working_memory::SharedNativeInitializationFailure<
            ForegroundDiskDescriptors,
            ConstructionError,
        >,
    ),
}

#[derive(Debug)]
pub(crate) struct ForegroundDiskSourceError(ForegroundDiskFailure);
impl fmt::Display for ForegroundDiskSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for ForegroundDiskSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
/// Consumes only selected neutral declarations and exact admitted reads. This
/// source constructor can run before native loading and never creates a device,
/// stream, host payload, worker, callback, or completed residency publication.
fn prepare_descriptors(
    primary: &RetainedCheckpointSource,
    sources: &BTreeMap<OffloadUnitId, RetainedCheckpointSource>,
    plan: &OffloadPlan,
    units: &[OffloadUnit],
    groups: &[String],
    pool: &MemoryLedger,
) -> Result<Option<ForegroundDiskDescriptors>, ForegroundDiskFailure> {
    if !plan
        .units()
        .iter()
        .any(|unit| unit.tier() == MemoryTier::Disk)
    {
        return Ok(None);
    }
    let Some(reads) =
        read_source_plan::prepare_reads(units, |id| sources.get(id).unwrap_or(primary).as_ref())?
    else {
        return Ok(None);
    };
    prepare_descriptors_from_reads(primary, sources, plan, units, groups, pool, &reads)
}

fn prepare_descriptors_from_reads(
    primary: &RetainedCheckpointSource,
    sources: &BTreeMap<OffloadUnitId, RetainedCheckpointSource>,
    plan: &OffloadPlan,
    units: &[OffloadUnit],
    groups: &[String],
    pool: &MemoryLedger,
    reads: &[PlannedRead],
) -> Result<Option<ForegroundDiskDescriptors>, ForegroundDiskFailure> {
    let prepared = ResidencyController::prepare_constructor(
        |id| sources.get(id).unwrap_or(primary).as_ref(),
        plan,
        units,
        groups,
    )?;
    prepared.validate_unit_definitions()?;
    let index: BTreeMap<_, _> = reads
        .iter()
        .enumerate()
        .map(|(index, row)| {
            (
                (
                    units[row.unit].id().as_str(),
                    units[row.unit].bindings()[row.binding].name(),
                ),
                index,
            )
        })
        .collect();
    let mut canonical_reads = Vec::with_capacity(units.len());
    for unit in units {
        let mut bindings = Vec::with_capacity(unit.bindings().len());
        for binding in unit.bindings() {
            let (owner, canonical) = if binding.is_alias() {
                prepared
                    .binding_owner(unit.id(), binding)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?
            } else {
                (unit.id(), binding)
            };
            bindings.push(
                *index
                    .get(&(owner.as_str(), canonical.name()))
                    .ok_or(WorkingMemoryError::IdentityMismatch)?,
            );
        }
        canonical_reads.push(bindings);
    }
    drop(index);
    let value = DescriptorPlan {
        units,
        catalogs: OriginalHostCatalogPlan::capture(primary, sources, units)?,
        reads,
        canonical_reads,
    };
    pool.initialize_shared_native(value)
        .map(|ready| Some(ready.output().clone()))
        .map_err(|error| {
            let (uncalled, failure) = error.into_parts();
            drop(uncalled);
            ForegroundDiskFailure::Constructor(failure)
        })
}

pub(crate) fn prepare_foreground_disk_descriptors(
    primary: &RetainedCheckpointSource,
    sources: &BTreeMap<OffloadUnitId, RetainedCheckpointSource>,
    plan: &OffloadPlan,
    units: &[OffloadUnit],
    groups: &[String],
    pool: &MemoryLedger,
) -> Result<Option<ForegroundDiskDescriptors>, ForegroundDiskSourceError> {
    prepare_descriptors(primary, sources, plan, units, groups, pool)
        .map_err(ForegroundDiskSourceError)
}

pub(super) fn prepare_foreground_disk_descriptors_from_reads(
    primary: &RetainedCheckpointSource,
    sources: &BTreeMap<OffloadUnitId, RetainedCheckpointSource>,
    plan: &OffloadPlan,
    units: &[OffloadUnit],
    groups: &[String],
    pool: &MemoryLedger,
    reads: &[PlannedRead],
) -> Result<Option<ForegroundDiskDescriptors>, ForegroundDiskSourceError> {
    prepare_descriptors_from_reads(primary, sources, plan, units, groups, pool, reads)
        .map_err(ForegroundDiskSourceError)
}
