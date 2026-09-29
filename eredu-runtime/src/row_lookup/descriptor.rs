//! Metadata-only row admission, before a readable source exists.
use super::*;
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeDtype, RecipeMetadata},
    store::{StoreError, TensorMetadata},
};
use eredu_core::residency::OffloadUnitRange;
use std::collections::{BTreeMap, BTreeSet};

/// Source representation used by a scalar recipe, without artifact identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowScaleSource {
    /// Logical and physical tensor geometry. Shard paths are discarded at admission.
    pub metadata: TensorMetadata,
    /// Exact container encoding required by the materializer.
    pub encoding: eredu_checkpoint::SourceTensorEncoding,
}

/// Exact scalar recipe and physical headers; this catalog has no payload methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowScaleDescriptor {
    parameter: ParameterId,
    recipe: DerivedWeightRecipe,
    catalog: BTreeMap<String, RowScaleSource>,
    metadata: RecipeMetadata,
}
impl RecipeCatalog for RowScaleDescriptor {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.catalog
            .get(key)
            .map(|entry| entry.metadata.clone())
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}
impl RowScaleDescriptor {
    /// Admits a complete exact scalar recipe against retained metadata only.
    pub fn new(
        parameter: ParameterId,
        recipe: DerivedWeightRecipe,
        mut catalog: BTreeMap<String, RowScaleSource>,
    ) -> Result<Self, RowLookupError> {
        for source in catalog.values_mut() {
            source.metadata.backing_shard = None;
        }
        let keys: BTreeSet<_> = recipe
            .source_keys()
            .into_iter()
            .map(str::to_owned)
            .collect();
        if keys != catalog.keys().cloned().collect() {
            return Err(RowLookupError::Specification(parameter));
        }
        struct Catalog<'a>(&'a BTreeMap<String, RowScaleSource>);
        impl RecipeCatalog for Catalog<'_> {
            fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
                self.0
                    .get(key)
                    .map(|entry| entry.metadata.clone())
                    .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
            }
        }
        let metadata = recipe
            .infer(&Catalog(&catalog))
            .map_err(RowLookupError::bank)?;
        if !matches!(
            metadata.dtype,
            RecipeDtype::F32 | RecipeDtype::F16 | RecipeDtype::BF16
        ) || !matches!(metadata.shape.as_slice(), [] | [1])
        {
            return Err(RowLookupError::Specification(parameter));
        }
        Ok(Self {
            parameter,
            recipe,
            catalog,
            metadata,
        })
    }
    /// Authoritative companion parameter identity.
    pub fn parameter(&self) -> &ParameterId {
        &self.parameter
    }
    /// Exact materialization equation.
    pub fn recipe(&self) -> &DerivedWeightRecipe {
        &self.recipe
    }
    /// Physical source headers and encodings, without source handles or artifact identity.
    pub fn catalog(&self) -> &BTreeMap<String, RowScaleSource> {
        &self.catalog
    }
    /// Inferred scalar representation and bytes.
    pub fn metadata(&self) -> &RecipeMetadata {
        &self.metadata
    }
}

/// Complete cold row geometry and residency, containing no readable sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowLookupDescriptor {
    range: OffloadUnitRange,
    metadata: RecipeMetadata,
    spec: RowLookupSpec,
    scale: Option<RowScaleDescriptor>,
    limits: RowLookupLimits,
}
impl RowLookupDescriptor {
    /// Checks row storage, scalar companions and finite request bounds.
    pub fn new(
        range: OffloadUnitRange,
        metadata: RecipeMetadata,
        spec: RowLookupSpec,
        scale: Option<RowScaleDescriptor>,
        limits: RowLookupLimits,
    ) -> Result<Self, RowLookupError> {
        spec.validate()?;
        limits.validate()?;
        if range.end() - range.start() != spec.rows
            || metadata.shape.len() != 2
            || metadata.shape[0] as u64 != spec.rows
            || metadata.byte_len != range.total_bytes()
        {
            return Err(RowLookupError::Geometry);
        }
        let floating = |dtype: &RecipeDtype| {
            matches!(
                dtype,
                RecipeDtype::F32 | RecipeDtype::F16 | RecipeDtype::BF16
            )
        };
        match (&spec.encoding, &scale) {
            (RowEncoding::Dense, None)
                if floating(&metadata.dtype) && metadata.shape[1] == spec.dimensions as usize =>
            {
                let bytes = if metadata.dtype == RecipeDtype::F32 {
                    4
                } else {
                    2
                };
                if range.bytes() != spec.dimensions as u64 * bytes {
                    return Err(RowLookupError::Geometry);
                }
            }
            (RowEncoding::ScalarE4M3 { scale: id }, Some(scale))
                if id == scale.parameter()
                    && metadata.dtype == RecipeDtype::F8E4M3
                    && metadata.shape[1] == spec.dimensions as usize
                    && range.bytes() == spec.dimensions as u64 => {}
            (RowEncoding::Gguf { encoding, .. }, None) => {
                let (block, bytes) = encoding
                    .block_and_bytes()
                    .map_err(|_| RowLookupError::Geometry)?;
                if range.bytes() != spec.dimensions as u64 / block * bytes
                    || metadata.dtype != RecipeDtype::U8
                    || metadata.shape[1] as u64 != range.bytes()
                {
                    return Err(RowLookupError::Geometry);
                }
            }
            _ => return Err(RowLookupError::Specification(spec.parameter.clone())),
        }
        admit("acquisition bytes", range.bytes(), limits.acquisition_bytes)?;
        let host = (limits.requests as u64)
            .checked_mul(128)
            .ok_or(RowLookupError::Geometry)?;
        let output = (limits.requests as u64)
            .checked_mul(spec.output_bytes())
            .and_then(|n| n.checked_mul(3))
            .ok_or(RowLookupError::Geometry)?;
        admit("host planning bytes", host, limits.host_bytes)?;
        admit("output bytes", output, limits.output_bytes)?;
        Ok(Self {
            range,
            metadata,
            spec,
            scale,
            limits,
        })
    }
    /// Compact source-backed residency namespace.
    pub fn range(&self) -> &OffloadUnitRange {
        &self.range
    }
    /// Inferred encoded table geometry.
    pub fn metadata(&self) -> &RecipeMetadata {
        &self.metadata
    }
    /// Operator and owner contract.
    pub fn spec(&self) -> &RowLookupSpec {
        &self.spec
    }
    /// Optional scalar recipe contract.
    pub fn scale(&self) -> Option<&RowScaleDescriptor> {
        self.scale.as_ref()
    }
    /// Invocation bounds.
    pub fn limits(&self) -> RowLookupLimits {
        self.limits
    }
    /// Largest distinct-row acquisition allowed by all limits.
    pub fn maximum_acquisition_rows(&self) -> u64 {
        self.spec
            .rows
            .min(self.limits.requests as u64)
            .min(self.limits.rows_per_acquisition as u64)
            .min(self.limits.acquisition_bytes / self.range.bytes())
    }
}

/// One metadata-only row set, with ownership and aggregate budget validation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowLookupDescriptors {
    entries: BTreeMap<ParameterId, RowLookupDescriptor>,
    requirements: RowLookupRequirements,
}
impl RowLookupDescriptors {
    /// Validates owners and computes bounded cold totals without source access.
    pub fn new(
        entries: impl IntoIterator<Item = RowLookupDescriptor>,
        execution_units: usize,
    ) -> Result<Self, RowLookupError> {
        let mut result = Self::default();
        let mut owners = BTreeSet::new();
        let mut prefixes = BTreeSet::new();
        for entry in entries {
            let spec = entry.spec();
            if spec.unit >= execution_units
                || !owners.insert((spec.bank, spec.unit))
                || !prefixes.insert(entry.range().prefix().clone())
                || result.entries.contains_key(&spec.parameter)
            {
                return Err(RowLookupError::Specification(spec.parameter.clone()));
            }
            let facts = &mut result.requirements;
            facts.source_bytes = facts
                .source_bytes
                .checked_add(entry.metadata().byte_len)
                .ok_or(RowLookupError::Geometry)?;
            if let Some(scale) = entry.scale() {
                facts.scalar_bytes = facts
                    .scalar_bytes
                    .checked_add(scale.metadata().byte_len)
                    .ok_or(RowLookupError::Geometry)?;
            }
            facts.acquisition_bytes = facts
                .acquisition_bytes
                .max(entry.maximum_acquisition_rows() * entry.range().bytes());
            facts.host_bytes = facts.host_bytes.max(entry.limits().host_bytes);
            facts.output_bytes = facts.output_bytes.max(entry.limits().output_bytes);
            result.entries.insert(spec.parameter.clone(), entry);
        }
        Ok(result)
    }
    /// Exact cold descriptors, one per parameter rather than per row.
    pub fn entries(&self) -> &BTreeMap<ParameterId, RowLookupDescriptor> {
        &self.entries
    }
    /// Cold storage and invocation allowances.
    pub const fn requirements(&self) -> RowLookupRequirements {
        self.requirements
    }
    /// Rejects overlap with another selected parameter-bank namespace.
    pub fn validate_other_banks(
        &self,
        banks: impl IntoIterator<Item = usize>,
    ) -> Result<(), RowLookupError> {
        let banks: BTreeSet<_> = banks.into_iter().collect();
        if let Some(entry) = self
            .entries
            .values()
            .find(|entry| banks.contains(&entry.spec().bank))
        {
            return Err(RowLookupError::Specification(
                entry.spec().parameter.clone(),
            ));
        }
        Ok(())
    }
    /// Checks the shared pool can hold the largest permitted acquisition.
    pub fn validate_pool_budget(
        &self,
        config: eredu_core::residency::OffloadConfig,
    ) -> Result<(), RowLookupError> {
        if let Some(limit) = config.device_budget_bytes() {
            admit(
                "row acquisition residency",
                self.requirements.acquisition_bytes,
                limit,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        rows::PreparedRowSource,
        store::{MemoryWeightStore, SharedCheckpointSource, TensorSelection},
    };
    use eredu_core::residency::{OffloadConfig, OffloadUnitId, ResidencyPolicy};
    use std::{cell::Cell, sync::Arc};

    struct Support(Cell<usize>);
    impl RowLookupMechanismSupport for Support {
        fn storage(&self) -> Option<crate::AddressableStorageCapabilities> {
            self.0.set(self.0.get() + 1);
            Some(crate::AddressableStorageCapabilities::new(
                true, true, true, 4096,
            ))
        }
        fn workspace(
            &self,
            _: &RowLookupDescriptor,
        ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
            self.0.set(self.0.get() + 1);
            Ok(Some(RowLookupWorkspace {
                decode_bytes: 80,
                scalar_bytes: 0,
            }))
        }
        fn decode_memory(
            &self,
            entry: &RowLookupDescriptor,
        ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, RowLookupError> {
            self.0.set(self.0.get() + 1);
            entry.decode_memory()
        }
    }
    fn descriptor(policy: ResidencyPolicy) -> RowLookupDescriptor {
        RowLookupDescriptor::new(
            OffloadUnitRange::new(OffloadUnitId::new("header.rows").unwrap(), 0, 4, 8, policy)
                .unwrap(),
            RecipeMetadata {
                shape: vec![4, 2],
                dtype: RecipeDtype::F32,
                byte_len: 32,
            },
            RowLookupSpec {
                parameter: ParameterId::new("table").unwrap(),
                bank: 1,
                unit: 0,
                rows: 4,
                dimensions: 2,
                encoding: RowEncoding::Dense,
                output_type: TensorElementType::F32,
            },
            None,
            RowLookupLimits {
                requests: 8,
                rows_per_acquisition: 2,
                acquisition_bytes: 16,
                host_bytes: 1024,
                output_bytes: 192,
            },
        )
        .unwrap()
    }
    fn bindable(descriptor: &RowLookupDescriptor) -> PreparedRowLookups {
        // A readable source first comes into existence after cold admission.
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "data".into(),
                safetensors::Dtype::F32,
                vec![4, 2],
                (1..=8).flat_map(|n| (n as f32).to_le_bytes()).collect(),
            )])
            .unwrap(),
        );
        let rows = PreparedRowSource::new(
            source,
            DerivedWeightRecipe::source("data", TensorSelection::Full),
        )
        .unwrap();
        let range =
            crate::RowResidencyRange::new(descriptor.range().clone(), rows, "table").unwrap();
        PreparedRowLookups::new(
            [
                PreparedRowLookup::new(range, descriptor.spec().clone(), None, descriptor.limits())
                    .unwrap(),
            ],
            1,
        )
        .unwrap()
    }
    #[test]
    fn headers_select_before_sources_and_binding_never_reselects() {
        let descriptor = descriptor(ResidencyPolicy::Cacheable);
        let headers = RowLookupDescriptors::new([descriptor.clone()], 1).unwrap();
        let support = Support(Cell::new(0));
        let options = crate::ParameterBankLoadOptions::new(
            OffloadConfig::new(Some(16), Some(0), 1).unwrap(),
            4096,
            4096,
        )
        .unwrap();
        let plan = SelectedRowLookupPlans::select(headers, options, 0, &support).unwrap();
        assert_eq!(support.0.get(), 3);
        assert_eq!(plan.requirements().rows.source_bytes, 32);
        let prepared = bindable(&descriptor);
        let source = prepared.entries().values().next().unwrap().range().clone();
        let selected = plan.clone().bind(prepared).unwrap();
        assert_eq!(support.0.get(), 3);
        assert!(selected
            .prepared()
            .entries()
            .values()
            .next()
            .unwrap()
            .range()
            .same_binding(&source));
        let independently_bound = plan.clone().bind(bindable(&descriptor)).unwrap();
        assert_ne!(selected, independently_bound);
        let mut altered = descriptor.clone();
        altered.limits.rows_per_acquisition = 1;
        assert!(matches!(
            plan.clone().bind(bindable(&altered)),
            Err(RowLookupSelectionError::DescriptorMismatch)
        ));
        let mut altered = descriptor.clone();
        altered.spec.output_type = TensorElementType::F16;
        assert!(matches!(
            plan.clone().bind(bindable(&altered)),
            Err(RowLookupSelectionError::DescriptorMismatch)
        ));
        assert!(matches!(
            plan.bind(PreparedRowLookups::default()),
            Err(RowLookupSelectionError::DescriptorMismatch)
        ));
        assert_eq!(support.0.get(), 3);
        assert_eq!(
            source.source().source_diagnostics().unwrap().physical_reads,
            0
        );
    }
    #[test]
    fn descriptor_admission_checks_storage_and_scalar_recipe_headers() {
        let original = descriptor(ResidencyPolicy::Cacheable);
        let mut metadata = original.metadata().clone();
        metadata.byte_len += 1;
        assert!(RowLookupDescriptor::new(
            original.range().clone(),
            metadata,
            original.spec().clone(),
            None,
            original.limits()
        )
        .is_err());
        let mut metadata = original.metadata().clone();
        metadata.dtype = RecipeDtype::F16;
        assert!(RowLookupDescriptor::new(
            original.range().clone(),
            metadata,
            original.spec().clone(),
            None,
            original.limits()
        )
        .is_err());
        let scalar_metadata = TensorMetadata {
            name: "scalar".into(),
            logical_shape: vec![1],
            physical_shape: vec![1],
            stored_dtype: eredu_checkpoint::StoredDtype::F32,
            encoded_byte_len: 4,
            backing_shard: Some("a.safetensors".into()),
        };
        let source = RowScaleSource {
            metadata: scalar_metadata,
            encoding: eredu_checkpoint::SourceTensorEncoding::Safetensors(
                eredu_checkpoint::StoredDtype::F32,
            ),
        };
        let recipe = DerivedWeightRecipe::source("scalar", TensorSelection::Full);
        let parameter = ParameterId::new("scale").unwrap();
        let descriptor = RowScaleDescriptor::new(
            parameter.clone(),
            recipe.clone(),
            BTreeMap::from([("scalar".into(), source.clone())]),
        )
        .unwrap();
        let mut other = source.clone();
        other.metadata.backing_shard = Some("b.safetensors".into());
        assert_eq!(
            descriptor,
            RowScaleDescriptor::new(
                parameter.clone(),
                recipe.clone(),
                BTreeMap::from([("scalar".into(), other)])
            )
            .unwrap()
        );
        assert!(
            RowScaleDescriptor::new(parameter.clone(), recipe.clone(), BTreeMap::new()).is_err()
        );
        let mut non_scalar = source;
        non_scalar.metadata.logical_shape = vec![2];
        non_scalar.metadata.physical_shape = vec![2];
        non_scalar.metadata.encoded_byte_len = 8;
        assert!(RowScaleDescriptor::new(
            parameter,
            recipe,
            BTreeMap::from([("scalar".into(), non_scalar)])
        )
        .is_err());
    }
}
