//! Selected entry catalogs and bounded on-load quantization.

use super::*;

pub use eredu_runtime::ParameterBankKey;

/// Workload class used only for bank chunking and mechanism telemetry.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum BankAccessClass {
    /// A multi-row access that may be split to stay under a working-set target.
    Bulk,
    /// A latency-sensitive access that uses the hard working-set limit directly.
    Incremental,
}

/// Backend controls for an independently addressable parameter bank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ParameterBankOptions {
    pub(super) storage: OffloadConfig,
    pub(super) compact_bank_scratch_bytes: u64,
    pub(super) bulk_compact_bank_target_bytes: u64,
}

impl ParameterBankOptions {
    /// Creates validated bank residency controls.
    pub fn new(
        storage: OffloadConfig,
        compact_bank_scratch_bytes: u64,
        bulk_compact_bank_target_bytes: u64,
    ) -> Result<Self, ParameterBankOptionsError> {
        let options = Self {
            storage,
            compact_bank_scratch_bytes,
            bulk_compact_bank_target_bytes,
        };
        options.validate()?;
        Ok(options)
    }

    /// Validates the bank working-set controls.
    pub fn validate(self) -> Result<(), ParameterBankOptionsError> {
        if self.compact_bank_scratch_bytes == 0 {
            return Err(ParameterBankOptionsError::ZeroScratchLimit);
        }
        if self.bulk_compact_bank_target_bytes == 0 {
            return Err(ParameterBankOptionsError::ZeroBulkBankTarget);
        }
        if self.bulk_compact_bank_target_bytes > self.compact_bank_scratch_bytes {
            return Err(ParameterBankOptionsError::BulkBankTargetExceedsScratch {
                target_bytes: self.bulk_compact_bank_target_bytes,
                scratch_bytes: self.compact_bank_scratch_bytes,
            });
        }
        Ok(())
    }
}

impl Default for ParameterBankOptions {
    fn default() -> Self {
        Self {
            storage: OffloadConfig::default(),
            compact_bank_scratch_bytes: u64::MAX,
            bulk_compact_bank_target_bytes: 1 << 30,
        }
    }
}

/// Invalid addressable-bank residency controls.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ParameterBankOptionsError {
    /// The hard temporary-bank limit was zero.
    #[error("parameter-bank scratch limit must be nonzero")]
    ZeroScratchLimit,
    /// The bulk-access working-set target was zero.
    #[error("parameter-bank bulk target must be nonzero")]
    ZeroBulkBankTarget,
    /// The soft bulk target exceeded the hard working-set limit.
    #[error("parameter-bank bulk target {target_bytes} exceeds scratch limit {scratch_bytes}")]
    BulkBankTargetExceedsScratch {
        /// Invalid soft target.
        target_bytes: u64,
        /// Configured hard limit.
        scratch_bytes: u64,
    },
}

/// One atomic entry definition supplied by a caller.
#[derive(Clone)]
pub struct ParameterBankEntry {
    pub(super) identity: ParameterBankKey,
    pub(super) unit: OffloadUnit,
    pub(super) bytes: u64,
}

impl ParameterBankEntry {
    /// Creates one catalog entry and verifies its stable unit identity.
    pub fn new(
        identity: ParameterBankKey,
        unit: OffloadUnit,
        bytes: u64,
    ) -> Result<Self, AddressableParameterBankError> {
        if bytes == 0 {
            return Err(AddressableParameterBankError::ZeroSizedEntry { identity });
        }
        let expected = identity.unit_id();
        if unit.id() != &expected {
            return Err(AddressableParameterBankError::UnitIdentityMismatch {
                identity,
                expected,
                actual: unit.id().clone(),
            });
        }
        Ok(Self {
            identity,
            unit,
            bytes,
        })
    }

    /// Returns the logical identity.
    pub const fn identity(&self) -> ParameterBankKey {
        self.identity
    }

    /// Returns the atomic materialized byte length.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    fn into_parts(self) -> (ParameterBankKey, OffloadUnit) {
        (self.identity, self.unit)
    }
}

/// Lowers generic selected storage members into MLX residency entries.
/// MLX entry bindings paired with exact per-binding transformation selections.
pub struct SelectedAddressableEntries {
    /// Source-backed entry catalog, before selected load-time transformations.
    pub entries: Vec<ParameterBankEntry>,
    /// Packed format keyed by the exact entry and local binding to transform.
    pub transformations: BTreeMap<(ParameterBankKey, String), SelectedBindingTransform>,
    /// Neutral selected byte total for each exact member.
    pub expected_bytes: BTreeMap<ParameterBankKey, u64>,
    /// Exact architecture ownership retained for every selected entry.
    pub placements: BTreeMap<ParameterBankKey, eredu_runtime::AddressableBankMemberPlacement>,
}

/// Exact native lowering values for one selected bank binding.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedBindingTransform {
    /// Packed executable format.
    pub quantization: WeightQuantization,
    /// Selected scale and affine-bias scalar dtype.
    pub companion_dtype: eredu_checkpoint::recipe::RecipeDtype,
}

/// Validates and lowers exact neutral bank tasks without collapsing mechanisms.
pub fn entries_from_selected_members(
    members: &[eredu_runtime::AddressableBankMember],
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<SelectedAddressableEntries, Error> {
    let plans =
        eredu_runtime::plan_addressable_bank_bindings(members, store, |_task, recipe, source| {
            crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, source)
        })
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut entries = Vec::with_capacity(plans.len());
    let mut transformations = BTreeMap::new();
    let mut expected_bytes = BTreeMap::new();
    let mut placements = BTreeMap::new();
    for plan in plans {
        let (key, bindings, transforms, selected_bytes, placement) = plan.into_parts();

        let source_bytes = bindings.iter().try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                Error::ArchitectureModel(format!(
                    "addressable member {key:?} source byte total overflowed"
                ))
            })
        })?;
        for (binding, transform) in transforms {
            transformations.insert(
                (key, binding),
                SelectedBindingTransform {
                    quantization: transform.quantization(),
                    companion_dtype: transform.companion_dtype().clone(),
                },
            );
        }
        entries.push(ParameterBankEntry::new(
            key,
            OffloadUnit::new(key.unit_id(), bindings)?,
            source_bytes,
        )?);
        expected_bytes.insert(key, selected_bytes);
        placements.insert(key, placement);
    }
    Ok(SelectedAddressableEntries {
        entries,
        transformations,
        expected_bytes,
        placements,
    })
}
/// Result of replacing dense entry bindings with a disk-backed packed overlay.
pub(crate) struct QuantizedParameterBankCatalog {
    /// Store supplying synthetic packed bindings and delegating all other keys.
    pub(crate) store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    /// Entry units rebuilt against the packed store.
    pub(crate) entries: Vec<ParameterBankEntry>,
    /// Deterministic bounded-materialisation telemetry.
    pub(crate) report: WeightMaterializationReport,
}

#[cfg(test)]
pub(crate) fn quantize_entry_catalog(
    source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<ParameterBankEntry>,
    quantization: WeightQuantization,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedParameterBankCatalog, Error> {
    let selected = entries
        .iter()
        .flat_map(|entry| {
            entry.unit.bindings().iter().filter_map(move |binding| {
                binding
                    .quantization_companions()
                    .map(|_| (entry.identity, binding.name().to_owned()))
            })
        })
        .collect();
    quantize_selected_entry_catalog_once(
        source,
        entries,
        quantization,
        &eredu_checkpoint::recipe::RecipeDtype::F32,
        &selected,
        max_working_set_bytes,
        source_stream,
    )
}

pub(super) fn selected_transformation_formats(
    transformations: &BTreeMap<(ParameterBankKey, String), SelectedBindingTransform>,
) -> Vec<WeightQuantization> {
    let mut formats = Vec::new();
    for transform in transformations.values() {
        if !formats.contains(&transform.quantization) {
            formats.push(transform.quantization);
        }
    }
    formats
}

pub(super) fn quantize_selected_entry_catalog(
    mut source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    mut entries: Vec<ParameterBankEntry>,
    transformations: BTreeMap<(ParameterBankKey, String), SelectedBindingTransform>,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedParameterBankCatalog, Error> {
    let mut by_format = Vec::<(SelectedBindingTransform, std::collections::BTreeSet<_>)>::new();
    for (binding, transform) in transformations {
        if let Some((_, selected)) = by_format
            .iter_mut()
            .find(|(candidate, _)| *candidate == transform)
        {
            selected.insert(binding);
        } else {
            by_format.push((transform, std::iter::once(binding).collect()));
        }
    }
    let mut report = WeightMaterializationReport::default();
    for (transform, selected) in by_format {
        let transformed = quantize_selected_entry_catalog_once(
            source,
            entries,
            transform.quantization,
            &transform.companion_dtype,
            &selected,
            max_working_set_bytes,
            source_stream,
        )?;
        source = transformed.store;
        entries = transformed.entries;
        report.merge(transformed.report);
    }
    Ok(QuantizedParameterBankCatalog {
        store: source,
        entries,
        report,
    })
}

/// Quantizes every floating entry projection through its authoritative
/// rank-local semantic recipe and rebuilds the catalog against packed keys.
fn quantize_selected_entry_catalog_once(
    source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<ParameterBankEntry>,
    quantization: WeightQuantization,
    companion_dtype: &eredu_checkpoint::recipe::RecipeDtype,
    selected: &std::collections::BTreeSet<(ParameterBankKey, String)>,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedParameterBankCatalog, Error> {
    let mut units = Vec::with_capacity(entries.len());
    let mut targets = Vec::new();
    let mut target_by_binding = BTreeMap::new();
    let mut packed_catalog_bytes = 0u64;
    for entry in entries {
        let (identity, unit) = entry.into_parts();
        for binding in unit.bindings() {
            let Some(companions) = binding.quantization_companions() else {
                continue;
            };
            if !selected.contains(&(identity, binding.name().to_owned())) {
                continue;
            }
            let recipe = binding.source_recipe();
            let metadata = recipe.infer(source.as_ref())?;
            let target_name = format!(
                "__eredu.entry.bank.{:05}.unit.{:05}.global.{:05}.{}.weight",
                identity.bank(),
                identity.unit(),
                identity.member(),
                binding.name()
            );
            let target_prefix = target_name
                .strip_suffix(".weight")
                .expect("synthetic entry target has a weight suffix");
            let target = BoundedQuantizationTarget::from_recipe(
                target_name.clone(),
                format!("{target_prefix}.scales"),
                companions
                    .affine_bias()
                    .map(|_| format!("{target_prefix}.biases")),
                recipe,
            )?
            .with_affine_companion_dtype(companion_dtype.clone())?;
            packed_catalog_bytes = packed_catalog_bytes
                .checked_add(packed_projection_bytes(
                    metadata.shape(),
                    quantization,
                    companion_dtype,
                )?)
                .ok_or_else(|| {
                    Error::Quantization("packed entry catalog size overflowed".into())
                })?;
            target_by_binding.insert(
                (identity, binding.name().to_string()),
                (
                    target.clone(),
                    companions.scale().to_owned(),
                    companions.affine_bias().map(str::to_owned),
                ),
            );
            targets.push(target);
        }
        units.push((identity, unit));
    }
    if targets.is_empty() {
        return Err(Error::Quantization(
            "entry catalog contains no floating projection bindings to quantize".into(),
        ));
    }
    let plan = BoundedQuantizationPlan::new(
        quantization,
        max_working_set_bytes.min(packed_catalog_bytes),
        targets,
    )?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        Arc::clone(&source),
        plan,
        source_stream,
    )?);
    let report = transformed.report().clone();
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = transformed;
    let mut rebuilt = Vec::with_capacity(units.len());
    for (identity, unit) in units {
        let mut bindings = Vec::new();
        for binding in unit.bindings() {
            let Some((target, scales_name, biases_name)) =
                target_by_binding.get(&(identity, binding.name().to_string()))
            else {
                bindings.push(binding.clone());
                continue;
            };
            bindings.push(packed_binding(
                binding.name(),
                target.weight_name(),
                store.as_ref(),
            )?);
            bindings.push(packed_binding(
                scales_name,
                target.scales_name(),
                store.as_ref(),
            )?);
            if let Some(biases_name) = biases_name {
                bindings.push(packed_binding(
                    biases_name,
                    target
                        .biases_name()
                        .expect("affine entry target declared a bias identity"),
                    store.as_ref(),
                )?);
            }
        }
        let bytes = bindings.iter().try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                Error::Quantization("quantized entry catalog byte total overflowed".into())
            })
        })?;
        rebuilt.push(ParameterBankEntry::new(
            identity,
            OffloadUnit::new(identity.unit_id(), bindings)?,
            bytes,
        )?);
    }
    Ok(QuantizedParameterBankCatalog {
        store,
        entries: rebuilt,
        report,
    })
}

pub(super) fn packed_projection_bytes(
    shape: &[usize],
    quantization: WeightQuantization,
    companion_dtype: &eredu_checkpoint::recipe::RecipeDtype,
) -> Result<u64, Error> {
    let (&columns, rows) = shape
        .split_last()
        .ok_or_else(|| Error::Quantization("entry projection has no input dimension".into()))?;
    let rows = rows.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(*dimension as u64)
            .ok_or_else(|| Error::Quantization("entry projection row count overflowed".into()))
    })?;
    let packed = (columns as u64)
        .checked_mul(quantization.bits() as u64)
        .and_then(|bits| bits.checked_div(8))
        .ok_or_else(|| Error::Quantization("packed entry row size overflowed".into()))?;
    let groups = columns
        .checked_div(quantization.group_size() as usize)
        .ok_or_else(|| Error::Quantization("entry group geometry is invalid".into()))?
        as u64;
    let scalar_bytes = companion_dtype
        .bit_width()
        .map_err(|error| Error::Quantization(error.to_string()))?
        / 8;
    let scale_bytes = if matches!(quantization, WeightQuantization::MxFp4) {
        groups
    } else {
        groups
            .checked_mul(scalar_bytes)
            .ok_or_else(|| Error::Quantization("entry scale row size overflowed".into()))?
    };
    let bias_bytes = if quantization.has_biases() {
        groups
            .checked_mul(scalar_bytes)
            .ok_or_else(|| Error::Quantization("entry bias row size overflowed".into()))?
    } else {
        0
    };
    let row_bytes = packed
        .checked_add(scale_bytes)
        .and_then(|bytes| bytes.checked_add(bias_bytes))
        .ok_or_else(|| Error::Quantization("packed entry row total overflowed".into()))?;
    rows.checked_mul(row_bytes)
        .ok_or_else(|| Error::Quantization("packed entry projection size overflowed".into()))
}

fn packed_binding(
    local_name: &str,
    checkpoint_key: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<WeightBinding, Error> {
    let metadata = store.source_metadata(checkpoint_key)?;
    Ok(WeightBinding::new(
        local_name,
        checkpoint_key,
        TensorSelection::Full,
        metadata.encoded_byte_len,
    )?)
}
