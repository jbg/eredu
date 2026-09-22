//! Exact selected catalogs and detached read owners, without ordinary roots.
use super::*;
use eredu_checkpoint::recipe::RecipeMetadata;
use eredu_checkpoint::store::{
    CheckpointLease, CheckpointSource, DetachedEncodedReads, MetadataCloneLayout,
    PreparedTensorSource, RetainedCheckpointSource, SourceKeyAuthority, SourceMetadataBorrowError,
    SourceMetadataLoan, StoreError, TensorMetadata, TensorReadRequest, TensorSourceProvenance,
    WeightStoreBackend, WeightStoreDiagnostics,
};
use std::{
    ops::Range,
    path::{Path, PathBuf},
};
mod output;
use output::BindingOutput;

#[derive(Clone, PartialEq, Eq)]
struct CatalogEntry {
    key: String,
    source: PreparedTensorSource,
    authoritative: bool,
}
#[derive(Clone, PartialEq, Eq)]
struct CatalogData {
    entries: Vec<CatalogEntry>,
    materialized_keys: Vec<String>,
    materialized_shards: Vec<PathBuf>,
    unclaimed_keys: Vec<String>,
    resolved: bool,
}
impl CatalogData {
    // Owned cold loading inputs. Source construction clones measured fields only
    // after admission, without consulting an ordinary source again.
    fn capture(source: &dyn CheckpointSource) -> Result<Self, StoreError> {
        let keys = source.source_keys();
        let mut entries = Vec::with_capacity(keys.len());
        for key in keys {
            if entries.iter().any(|row: &CatalogEntry| row.key == key) {
                return Err(StoreError::PreparedCatalogMismatch { key });
            }
            let metadata = source.source_metadata(&key)?;
            let provenance = source.source_provenance(&key)?;
            let authoritative = source.is_authoritative_materialized_key(&key);
            entries.push(CatalogEntry {
                key,
                source: PreparedTensorSource {
                    metadata,
                    provenance,
                },
                authoritative,
            });
        }
        Ok(Self {
            entries,
            materialized_keys: source.materialized_source_keys(),
            materialized_shards: source.materialized_source_shards(),
            unclaimed_keys: source.unclaimed_checkpoint_keys(),
            resolved: source.is_checkpoint_contract_resolved(),
        })
    }
    fn entry(&self, key: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|row| row.key == key)
    }
    fn clone_bytes(&self) -> Option<usize> {
        let mut bytes = Layout::array::<CatalogEntry>(self.entries.len())
            .ok()?
            .size();
        for row in &self.entries {
            let p = &row.source.provenance;
            bytes = bytes
                .checked_add(row.key.len())?
                .checked_add(MetadataCloneLayout::of(&row.source.metadata)?.payload_bytes()?)?
                .checked_add(p.catalog_key.len())?
                .checked_add(p.physical_tensor.len())?
                .checked_add(p.output.len())?
                .checked_add(p.backing_shard.as_deref().map_or(0, path_bytes))?
                .checked_add(match &p.source_encoding {
                    eredu_checkpoint::SourceTensorEncoding::Safetensors(dtype)
                    | eredu_checkpoint::SourceTensorEncoding::RecipeOutput(dtype) => match dtype {
                        eredu_checkpoint::StoredDtype::Other(name) => name.len(),
                        _ => 0,
                    },
                    eredu_checkpoint::SourceTensorEncoding::Gguf { .. } => 0,
                    _ => return None,
                })?;
        }
        for names in [&self.materialized_keys, &self.unclaimed_keys] {
            bytes = bytes.checked_add(Layout::array::<String>(names.len()).ok()?.size())?;
            for name in names {
                bytes = bytes.checked_add(name.len())?;
            }
        }
        bytes = bytes.checked_add(
            Layout::array::<PathBuf>(self.materialized_shards.len())
                .ok()?
                .size(),
        )?;
        for path in &self.materialized_shards {
            bytes = bytes.checked_add(path_bytes(path))?;
        }
        Some(bytes)
    }
}
fn path_bytes(path: &Path) -> usize {
    path.as_os_str().as_encoded_bytes().len()
}

/// Owned cold planning data; no ordinary checkpoint/cache/file Arc is retained.
pub(super) struct OriginalHostCatalogPlan {
    catalogs: Vec<CatalogData>,
    units: Vec<usize>,
}
impl OriginalHostCatalogPlan {
    pub(super) fn capture(
        primary: &RetainedCheckpointSource,
        sources: &BTreeMap<OffloadUnitId, RetainedCheckpointSource>,
        units: &[OffloadUnit],
    ) -> Result<Self, StoreError> {
        let mut owners = vec![primary];
        let mut catalogs = vec![CatalogData::capture(primary.as_ref())?];
        let mut mapping = Vec::with_capacity(units.len());
        for unit in units {
            let source = sources.get(unit.id()).unwrap_or(primary);
            let index =
                if let Some(index) = owners.iter().position(|owner| owner.same_source(source)) {
                    index
                } else {
                    let index = owners.len();
                    catalogs.push(CatalogData::capture(source.as_ref())?);
                    owners.push(source);
                    index
                };
            mapping.push(index);
        }
        Ok(Self {
            catalogs,
            units: mapping,
        })
    }
}
#[derive(Clone)]
struct CatalogOwner {
    value: Arc<CatalogData>,
    _custody: ManagerCustody,
}
pub(super) struct DetachedCatalog {
    catalog: Option<CatalogOwner>,
    reads: Arc<DetachedEncodedReads<ManagerCustody>>,
    range: Range<usize>,
    outputs: Vec<BindingOutput>,
}
impl DetachedCatalog {
    fn entry(&self, key: &str) -> Option<&CatalogEntry> {
        self.catalog.as_ref()?.value.entry(key)
    }
    fn output(&self, name: &str) -> Option<&RecipeMetadata> {
        self.outputs
            .iter()
            .find(|row| row.name() == name)?
            .metadata(&self.reads)
    }
}
impl CheckpointSource for DetachedCatalog {
    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        self.entry(key)
            .map(|row| &row.source.metadata)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)
    }
    fn source_key_authority_borrowed(
        &self,
        key: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        Ok(if self.is_authoritative_materialized_key(key) {
            SourceKeyAuthority::Materialized
        } else {
            SourceKeyAuthority::Ordinary
        })
    }
    fn source_keys(&self) -> Vec<String> {
        self.catalog.as_ref().map_or_else(Vec::new, |catalog| {
            catalog
                .value
                .entries
                .iter()
                .map(|row| row.key.clone())
                .collect()
        })
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.entry(key)
            .map(|row| row.source.metadata.clone())
            .ok_or_else(|| StoreError::UnknownTensor {
                key: key.to_owned(),
            })
    }
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        self.entry(key)
            .map(|row| row.source.provenance.clone())
            .ok_or_else(|| StoreError::UnknownTensor {
                key: key.to_owned(),
            })
    }
    fn materialized_source_keys(&self) -> Vec<String> {
        self.catalog
            .as_ref()
            .map_or_else(Vec::new, |catalog| catalog.value.materialized_keys.clone())
    }
    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.catalog.as_ref().map_or_else(Vec::new, |catalog| {
            catalog.value.materialized_shards.clone()
        })
    }
    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.catalog
            .as_ref()
            .map_or_else(Vec::new, |catalog| catalog.value.unclaimed_keys.clone())
    }
    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.entry(key).is_some_and(|row| row.authoritative)
    }
    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.catalog
            .as_ref()
            .is_some_and(|catalog| catalog.value.resolved)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        Err(StoreError::BoundedSelectionUnavailable {
            key: request.key,
            message: "detached catalog payload acquisition uses its prepared manager".into(),
        })
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        Ok(self
            .reads
            .slice(self.range.clone())
            .expect("validated detached read range")
            .diagnostics())
    }
}
pub(in crate::backend::runtime::residency::manager) struct OriginalReadSources {
    primary: CatalogOwner,
    catalogs: rows::Rows<OffloadUnitId, DetachedCatalog>,
    empty: DetachedCatalog,
    reads: Arc<DetachedEncodedReads<ManagerCustody>>,
    policy: RetainedCheckpointSource,
    _custody: ManagerCustody,
}
impl OriginalReadSources {
    pub(in crate::backend::runtime::residency::manager) fn catalog(
        &self,
        id: &OffloadUnitId,
    ) -> &dyn CheckpointSource {
        self.catalogs.get(id).unwrap_or(&self.empty)
    }
    pub(in crate::backend::runtime::residency::manager) fn output(
        &self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Option<&RecipeMetadata> {
        self.catalogs.get(id)?.output(name)
    }
    pub(super) fn read_output(&self, index: usize) -> Option<&RecipeMetadata> {
        self.reads.output(index)
    }
    #[cfg(test)]
    pub(in crate::backend::runtime::residency::manager) fn detached_physical_read_bytes(
        &self,
        source: usize,
    ) -> Option<u64> {
        self.reads.physical_read_bytes(source)
    }
    pub(super) fn read_slice(
        &self,
        range: std::ops::Range<usize>,
    ) -> Option<eredu_checkpoint::store::DetachedEncodedReadSlice<'_, ManagerCustody>> {
        self.reads.slice(range)
    }
    /// Exact retained catalog-owner identity, without cloning its backing.
    pub(super) fn same_catalog(&self, a: &OffloadUnitId, b: &OffloadUnitId) -> bool {
        match (
            self.catalogs.get(a).and_then(|row| row.catalog.as_ref()),
            self.catalogs.get(b).and_then(|row| row.catalog.as_ref()),
        ) {
            (Some(a), Some(b)) => Arc::ptr_eq(&a.value, &b.value),
            _ => false,
        }
    }
    pub(super) fn same_primary_catalog(&self, unit: &OffloadUnitId) -> bool {
        self.catalogs
            .get(unit)
            .and_then(|row| row.catalog.as_ref())
            .is_some_and(|owner| Arc::ptr_eq(&owner.value, &self.primary.value))
    }
    pub(super) fn matches_primary_catalog(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<bool, WeightRecipeError> {
        Ok(*self.primary.value == CatalogData::capture(source)?)
    }
    pub(super) fn matches_catalog(
        &self,
        unit: &OffloadUnitId,
        source: &dyn CheckpointSource,
    ) -> Result<bool, WeightRecipeError> {
        let Some(owner) = self.catalogs.get(unit).and_then(|row| row.catalog.as_ref()) else {
            return Ok(false);
        };
        // Existing cold source-reuse validation, once per distinct source and
        // retained catalog pair before the next source grant.
        Ok(*owner.value == CatalogData::capture(source)?)
    }
    pub(super) fn matches_reads(
        &self,
        unit: &OffloadUnit,
        source: &dyn CheckpointSource,
    ) -> Result<bool, WeightRecipeError> {
        let Some(catalog) = self.catalogs.get(unit.id()) else {
            return Ok(false);
        };
        let bindings = unit.bindings().iter().filter(|binding| !binding.is_alias());
        if bindings.clone().count() != catalog.outputs.len() {
            return Ok(false);
        }
        for (binding, output) in bindings.zip(&catalog.outputs) {
            if !output.matches(binding, source, &self.reads)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(in crate::backend::runtime::residency::manager) fn policy(
        &self,
    ) -> &RetainedCheckpointSource {
        &self.policy
    }
    /// Catalog/read Arc requests, exact clone backing and fixed controls only.
    /// Host map/key/buffer and detached construction/read scratch are separate.
    pub(super) fn storage_bytes(plan: &ReadSourcePlan<'_>) -> Result<usize, WorkingMemoryError> {
        let overflow = || WorkingMemoryError::Overflow;
        let shared = |layout| {
            usize::try_from(OriginalHostMetadataCustody::shared_storage_bytes(layout)?)
                .map_err(|_| overflow())
        };
        let mut bytes = shared(Layout::new::<DetachedEncodedReads<ManagerCustody>>())?
            .checked_add(
                Layout::array::<CatalogOwner>(plan.catalogs.catalogs.len())
                    .map_err(|_| overflow())?
                    .size(),
            )
            .ok_or_else(overflow)?
            .checked_add(
                rows::Rows::<OffloadUnitId, DetachedCatalog>::requested_layout(plan.units.len())
                    .ok_or_else(overflow)?
                    .size(),
            )
            .ok_or_else(overflow)?;
        for catalog in &plan.catalogs.catalogs {
            bytes = bytes
                .checked_add(shared(Layout::new::<CatalogData>())?)
                .ok_or_else(overflow)?
                .checked_add(catalog.clone_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        for (ordinal, unit) in plan.units.iter().enumerate() {
            let count = unit
                .bindings()
                .iter()
                .filter(|binding| !binding.is_alias())
                .count();
            bytes = bytes
                .checked_add(unit.id().as_str().len())
                .ok_or_else(overflow)?
                .checked_add(
                    Layout::array::<BindingOutput>(count)
                        .map_err(|_| overflow())?
                        .size(),
                )
                .ok_or_else(overflow)?;
            for row in &plan.reads[plan.read_range(ordinal)] {
                let binding = &unit.bindings()[row.binding];
                bytes = bytes
                    .checked_add(
                        BindingOutput::payload_bytes(row, binding.name()).ok_or_else(overflow)?,
                    )
                    .ok_or_else(overflow)?;
            }
        }
        let erasure =
            RetainedCheckpointSource::source_storage_request::<DetachedCatalog, ManagerCustody>()
                .ok_or_else(overflow)?;
        for n in [
            shared(erasure.source_body())?,
            shared(erasure.custody_body())?,
            erasure.control_bytes(),
            BindingOutput::control_bytes().ok_or_else(overflow)?,
            size_of::<Self>(),
            size_of::<ReadSourcePlan<'_>>(),
            size_of::<DetachedCatalog>(),
            size_of::<CatalogOwner>(),
            size_of::<CatalogData>(),
            size_of::<CatalogEntry>(),
            size_of::<PreparedTensorSource>(),
            size_of::<TensorSourceProvenance>(),
            size_of::<Vec<CatalogOwner>>(),
            size_of::<ManagerCustody>(),
            size_of::<ConstructionCause>(),
            size_of::<Result<Self, ConstructionCause>>(),
            size_of::<Vec<(OffloadUnitId, DetachedCatalog)>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
        ] {
            bytes = bytes.checked_add(n).ok_or_else(overflow)?;
        }
        Ok(bytes)
    }
    pub(super) fn new(
        plan: &ReadSourcePlan<'_>,
        reads: DetachedEncodedReads<ManagerCustody>,
        custody: ManagerCustody,
    ) -> Result<Self, ConstructionCause> {
        if plan.catalogs.units.len() != plan.units.len()
            || Some(reads.len()) != plan.encoded_leaves().map(|leaves| leaves.len())
        {
            return Err(ResidencyError::StatePoisoned.into());
        }
        let reads = Arc::new(reads);
        let mut owners = Vec::new();
        owners.try_reserve_exact(plan.catalogs.catalogs.len())?;
        for catalog in &plan.catalogs.catalogs {
            owners.push(CatalogOwner {
                value: Arc::new(catalog.clone()),
                _custody: custody.clone(),
            });
        }
        let mut catalogs = Vec::new();
        catalogs.try_reserve_exact(plan.units.len())?;
        for (ordinal, unit) in plan.units.iter().enumerate() {
            let rows = plan.read_range(ordinal);
            let range = plan
                .leaf_range(rows.clone())
                .ok_or(ResidencyError::StatePoisoned)?;
            if rows.len()
                != unit
                    .bindings()
                    .iter()
                    .filter(|binding| !binding.is_alias())
                    .count()
            {
                return Err(ResidencyError::StatePoisoned.into());
            }
            let mut outputs = Vec::new();
            outputs.try_reserve_exact(rows.len())?;
            let mut leaf = range.start;
            for row in &plan.reads[rows] {
                let end = leaf
                    .checked_add(row.read.leaf_count())
                    .ok_or(ResidencyError::StatePoisoned)?;
                outputs.push(BindingOutput::construct(
                    row,
                    unit.bindings()[row.binding].name(),
                    leaf..end,
                )?);
                leaf = end;
            }
            let catalog = owners
                .get(
                    *plan
                        .catalogs
                        .units
                        .get(ordinal)
                        .ok_or(ResidencyError::StatePoisoned)?,
                )
                .ok_or(ResidencyError::StatePoisoned)?
                .clone();
            catalogs.push((
                unit.id().clone(),
                DetachedCatalog {
                    catalog: Some(catalog),
                    reads: reads.clone(),
                    range,
                    outputs,
                },
            ));
        }
        let primary = owners.first().ok_or(ResidencyError::StatePoisoned)?.clone();
        let policy = RetainedCheckpointSource::from_source_with_custody(
            DetachedCatalog {
                catalog: Some(primary.clone()),
                reads: reads.clone(),
                range: 0..reads.len(),
                outputs: Vec::new(),
            },
            custody.clone(),
        );
        Ok(Self {
            primary,
            policy,
            catalogs: catalogs.into_iter().collect(),
            empty: DetachedCatalog {
                catalog: None,
                reads: reads.clone(),
                range: 0..0,
                outputs: Vec::new(),
            },
            reads,
            _custody: custody,
        })
    }
}

/// The foreground handle shares its already admitted descriptor/read/catalog
/// owner; no second catalog, read plan or unaccounted source Arc is created.
pub(super) enum ReadSourceOwner {
    Host(OriginalReadSources),
    Foreground(ForegroundDiskDescriptors),
}
impl std::ops::Deref for ReadSourceOwner {
    type Target = OriginalReadSources;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Host(source) => source,
            Self::Foreground(source) => source.read_source(),
        }
    }
}
/// Host buffers are an additional consumer of the same exact detached read and
/// selected catalog owner used by foreground disk descriptors.
pub(in crate::backend::runtime::residency::manager) struct OriginalHostSources {
    source: ReadSourceOwner,
    hosts: rows::Rows<OffloadUnitId, ResidentHostOwner>,
    _custody: ManagerCustody,
}
impl std::ops::Deref for OriginalHostSources {
    type Target = OriginalReadSources;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}
impl OriginalHostSources {
    pub(in crate::backend::runtime::residency::manager) fn host(
        &self,
        id: &OffloadUnitId,
    ) -> Option<&ResidentHostOwner> {
        self.hosts.get(id)
    }
    pub(in crate::backend::runtime::residency::manager) fn hosts(
        &self,
    ) -> impl Iterator<Item = &ResidentHostOwner> {
        self.hosts.values()
    }

    pub(in crate::backend::runtime::residency::manager) fn foreground(
        &self,
    ) -> Option<&ForegroundDiskDescriptors> {
        match &self.source {
            ReadSourceOwner::Foreground(source) => Some(source),
            ReadSourceOwner::Host(_) => None,
        }
    }
    pub(super) fn storage_bytes(plan: &OriginalManagerPlan) -> Result<usize, WorkingMemoryError> {
        let source_bytes = if plan.foreground.is_some() {
            0
        } else {
            OriginalReadSources::storage_bytes(&plan.read_source_plan())?
        };
        source_bytes
            .checked_add(size_of::<Self>())
            .and_then(|bytes| bytes.checked_add(size_of::<Result<Self, ConstructionCause>>()))
            .ok_or(WorkingMemoryError::Overflow)
    }
    pub(super) fn new(
        plan: &OriginalManagerPlan,
        source: ReadSourceOwner,
        buffers: Vec<RetainedHostBuffer>,
        control: &ResidencyController,
        custody: ManagerCustody,
    ) -> Result<Self, ConstructionCause> {
        let mut hosts = Vec::with_capacity(plan.host_units.len());
        for &unit_index in &plan.host_units {
            let unit = &plan.units[unit_index];
            let mut named = Vec::with_capacity(unit.bindings().len());
            for binding in unit.bindings() {
                let (owner, canonical) = if binding.is_alias() {
                    control
                        .binding_owner_borrowed(unit.id(), binding)
                        .ok_or(ResidencyError::StatePoisoned)?
                } else {
                    (unit.id(), binding)
                };
                let index = plan
                    .read_index(owner, canonical.name())
                    .and_then(|index| plan.host_read_slots.get(index).copied().flatten())
                    .ok_or(ResidencyError::StatePoisoned)?;
                named.push((
                    binding.name().to_owned(),
                    buffers
                        .get(index)
                        .ok_or(ResidencyError::StatePoisoned)?
                        .clone(),
                ));
            }
            hosts.push((
                unit.id().clone(),
                ResidentHostOwner::original(
                    ResidentHostBuffers {
                        buffers: named.into_iter().collect(),
                    },
                    custody.clone(),
                ),
            ));
        }
        Ok(Self {
            source,
            hosts: hosts.into_iter().collect(),
            _custody: custody,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct SelectedSource {
        _owner: Arc<()>,
        shape: [usize; 2],
        output: &'static str,
    }
    impl SelectedSource {
        fn new(owner: Arc<()>) -> Self {
            Self {
                _owner: owner,
                shape: [2, 3],
                output: "selected-view",
            }
        }
    }
    impl CheckpointSource for SelectedSource {
        fn source_keys(&self) -> Vec<String> {
            vec!["renamed".into(), "unused".into()]
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            if !["renamed", "unused"].contains(&key) {
                return Err(StoreError::UnknownTensor { key: key.into() });
            }
            Ok(TensorMetadata {
                name: "physical-leaf".into(),
                logical_shape: self.shape.to_vec(),
                physical_shape: vec![6],
                stored_dtype: eredu_checkpoint::StoredDtype::Other("packed".into()),
                encoded_byte_len: 24,
                backing_shard: Some(PathBuf::from("admitted.shard")),
            })
        }
        fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
            self.source_metadata(key)?;
            Ok(TensorSourceProvenance {
                catalog_key: key.into(),
                physical_tensor: "physical-leaf".into(),
                output: self.output.into(),
                backing_shard: Some(PathBuf::from("admitted.shard")),
                source_encoding: eredu_checkpoint::SourceTensorEncoding::RecipeOutput(
                    eredu_checkpoint::StoredDtype::Other("packed".into()),
                ),
            })
        }
        fn materialized_source_keys(&self) -> Vec<String> {
            vec!["physical-leaf".into()]
        }
        fn materialized_source_shards(&self) -> Vec<PathBuf> {
            vec![PathBuf::from("admitted.shard")]
        }
        fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
            vec!["excluded".into()]
        }
        fn is_authoritative_materialized_key(&self, key: &str) -> bool {
            key == "renamed"
        }
        fn is_checkpoint_contract_resolved(&self) -> bool {
            true
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("catalog capture acquired payload")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            panic!("catalog capture consulted cache")
        }
    }
    #[test]
    fn detached_catalog_preserves_selected_keys_provenance_and_releases_source() {
        let alive = Arc::new(());
        let weak = Arc::downgrade(&alive);
        let source = SelectedSource::new(alive);
        let expected = source.source_provenance("renamed").unwrap();
        let metadata = source.source_metadata("renamed").unwrap();
        let captured = CatalogData::capture(&source).unwrap();
        assert!(captured.clone_bytes().unwrap() > 2 * size_of::<CatalogEntry>());
        let reads =
            eredu_checkpoint::recipe::EncodedRecipeRead::prepare_detached(std::iter::empty())
                .unwrap()
                .construct(ManagerCustody::default())
                .unwrap();
        let catalog = DetachedCatalog {
            catalog: Some(CatalogOwner {
                value: Arc::new(captured),
                _custody: ManagerCustody::default(),
            }),
            reads: Arc::new(reads),
            range: 0..0,
            outputs: Vec::new(),
        };
        drop(source);
        assert!(weak.upgrade().is_none());
        assert_eq!(catalog.source_keys(), ["renamed", "unused"]);
        assert_eq!(catalog.source_metadata("renamed").unwrap(), metadata);
        assert_eq!(
            catalog.source_metadata_borrowed("renamed").unwrap(),
            &metadata
        );
        assert_eq!(catalog.source_provenance("renamed").unwrap(), expected);
        assert!(catalog.source_metadata("physical-leaf").is_err());
        assert_eq!(catalog.materialized_source_keys(), ["physical-leaf"]);
        assert_eq!(catalog.unclaimed_checkpoint_keys(), ["excluded"]);
        assert!(catalog.is_checkpoint_contract_resolved());
        assert_eq!(
            catalog.source_key_authority_borrowed("renamed").unwrap(),
            SourceKeyAuthority::Materialized
        );
        assert_eq!(
            catalog.source_key_authority_borrowed("unused").unwrap(),
            SourceKeyAuthority::Ordinary
        );
    }
    #[test]
    fn primary_catalog_adoption_checks_metadata_and_provenance_with_all_units_overridden() {
        let primary = SelectedSource::new(Arc::new(()));
        let overridden = SelectedSource {
            output: "override-view",
            ..SelectedSource::new(Arc::new(()))
        };
        let primary_owner = CatalogOwner {
            value: Arc::new(CatalogData::capture(&primary).unwrap()),
            _custody: ManagerCustody::default(),
        };
        let override_owner = CatalogOwner {
            value: Arc::new(CatalogData::capture(&overridden).unwrap()),
            _custody: ManagerCustody::default(),
        };
        // Only the exact cold catalog-authentication worker is exercised here;
        // no stream, host-buffer constructor or native/source grant is created.
        let reads = Arc::new(
            eredu_checkpoint::recipe::EncodedRecipeRead::prepare_detached(std::iter::empty())
                .unwrap()
                .construct(ManagerCustody::default())
                .unwrap(),
        );
        let catalog = |owner| DetachedCatalog {
            catalog: owner,
            reads: reads.clone(),
            range: 0..0,
            outputs: Vec::new(),
        };
        let ids = [
            OffloadUnitId::new("first").unwrap(),
            OffloadUnitId::new("second").unwrap(),
        ];
        let source = OriginalReadSources {
            primary: primary_owner.clone(),
            catalogs: ids
                .iter()
                .map(|id| (id.clone(), catalog(Some(override_owner.clone()))))
                .collect(),
            empty: catalog(None),
            policy: RetainedCheckpointSource::from_source_with_custody(
                catalog(Some(primary_owner)),
                ManagerCustody::default(),
            ),
            reads: reads.clone(),
            _custody: ManagerCustody::default(),
        };
        assert!(source.matches_primary_catalog(&primary).unwrap());
        assert!(source.same_catalog(&ids[0], &ids[1]));
        for id in &ids {
            assert!(!source.same_primary_catalog(id));
            assert!(source.matches_catalog(id, &overridden).unwrap());
        }
        let changed_metadata = SelectedSource {
            shape: [3, 2],
            ..SelectedSource::new(Arc::new(()))
        };
        let changed_provenance = SelectedSource {
            output: "changed-primary-view",
            ..SelectedSource::new(Arc::new(()))
        };
        assert_eq!(
            changed_provenance.source_metadata("renamed").unwrap(),
            primary.source_metadata("renamed").unwrap()
        );
        for incoming in [&changed_metadata, &changed_provenance] {
            // All per-unit override checks still succeed. The independently
            // consumed primary check must reject both kinds of policy change.
            for id in &ids {
                assert!(source.matches_catalog(id, &overridden).unwrap());
            }
            assert!(!source.matches_primary_catalog(incoming).unwrap());
        }
        assert_eq!(
            source.policy().source_metadata("renamed").unwrap(),
            primary.source_metadata("renamed").unwrap()
        );
        assert_eq!(
            source.policy().source_provenance("renamed").unwrap(),
            primary.source_provenance("renamed").unwrap()
        );
    }
}
