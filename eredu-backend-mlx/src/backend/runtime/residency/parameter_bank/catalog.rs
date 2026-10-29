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

/// Validates and lowers exact neutral bank tasks without collapsing mechanisms.
pub fn entries_from_selected_members(
    members: &[eredu_runtime::AddressableBankMember],
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<AddressableBankCatalog, Error> {
    let plans =
        eredu_runtime::plan_addressable_bank_bindings(members, store, |_task, recipe, source| {
            crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, source)
        })
        .map_err(|error| Error::Other(Box::new(error)))?;
    Ok(eredu_runtime::AddressableBankCatalog::from_binding_plans(
        plans,
    )?)
}
/// Result of replacing dense entry bindings with a disk-backed packed overlay.
pub(crate) struct QuantizedParameterBankCatalog {
    /// Store supplying synthetic packed bindings and delegating all other keys.
    pub(crate) store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    /// Entry units rebuilt against the packed store.
    pub(crate) entries: Vec<AddressableBankEntry>,
    /// Deterministic bounded-materialisation telemetry.
    pub(crate) report: WeightMaterializationReport,
}

#[cfg(test)]
pub(crate) fn quantize_entry_catalog(
    source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<AddressableBankEntry>,
    quantization: WeightQuantization,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedParameterBankCatalog, Error> {
    let selected = entries
        .iter()
        .flat_map(|entry| {
            entry.unit().bindings().iter().filter_map(move |binding| {
                binding
                    .quantization_companions()
                    .map(|_| (entry.key(), binding.name().to_owned()))
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

pub(super) fn quantize_selected_entry_catalog(
    mut source: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    mut entries: Vec<AddressableBankEntry>,
    transformations: eredu_runtime::AddressableBankTransformPlan,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedParameterBankCatalog, Error> {
    let mut report = WeightMaterializationReport::default();
    for group in transformations.into_groups() {
        let (transform, selected) = group.into_parts();
        let transformed = quantize_selected_entry_catalog_once(
            source,
            entries,
            transform.quantization(),
            transform.companion_dtype(),
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
    entries: Vec<AddressableBankEntry>,
    quantization: WeightQuantization,
    companion_dtype: &eredu_checkpoint::recipe::RecipeDtype,
    selected: &std::collections::BTreeSet<(ParameterBankKey, String)>,
    max_working_set_bytes: u64,
    source_stream: &Stream,
) -> Result<QuantizedParameterBankCatalog, Error> {
    let (plan, bindings) = eredu_runtime::prepare_addressable_bank_quantization(
        source.as_ref(),
        entries,
        eredu_runtime::AddressableBindingTransform::new(quantization, companion_dtype.clone()),
        selected,
        max_working_set_bytes,
    )
    .map_err(|error| Error::Other(Box::new(error)))?;
    let (store, report) = prepare_quantized_source(source, plan, source_stream)?;
    let rebuilt = bindings
        .bind(store.as_ref())
        .map_err(|error| Error::Other(Box::new(error)))?;
    Ok(QuantizedParameterBankCatalog {
        store,
        entries: rebuilt,
        report,
    })
}
