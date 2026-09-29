//! Architecture-authored n-gram sources and exact integer companions.
pub(super) mod gguf;
/// Header-only published table admission and retained-source binding.
pub use gguf::GgufTableSourcePlan;
/// Exact published GGUF text tensor recipes and restricted source owners.
pub mod gguf_text;
/// Layer-sized and independently addressable expert source recipes.
pub mod recipes;
mod safetensors_table;
pub mod schema;
pub(in crate::qwen4_exp) mod table;
use super::{
    config::{Config, NGramSourceLayout},
    ngram::{NGramError, NGramHashSpec},
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    rows::{PreparedRowSource, RowReadError},
    store::{
        CheckpointSource, EncodedTensorLease, PreparedCheckpointSource, PreparedTensorSource,
        ReadPolicy, RestrictedCheckpointSource, SharedCheckpointSource, StoreError,
        TensorReadRequest, TensorSelection,
    },
    StoredDtype,
};
use eredu_nn::{ParameterId, TensorElementType};
use eredu_runtime::{PreparedRowScale, RowEncoding, RowLookupSpec};
/// Metadata-only table admission with deferred bounded literal acquisition.
pub use safetensors_table::SafetensorsTableSourcePlan;
use std::{collections::BTreeMap, sync::Arc};

/// Fully described logical row source; preparation never reads table payloads.
#[derive(Clone)]
pub struct PreparedNGramTable {
    /// One compact recipe over the physical table or its declared shards.
    pub rows: Arc<PreparedRowSource>,
    /// Exact hash policy from bounded I64 reads or integer container metadata.
    pub hash: NGramHashSpec,
    /// Retained exact integer sources or container metadata after literal decoding.
    pub controls: NGramControls,
    /// Exact row decoder specification for provider construction.
    pub lookup: RowLookupSpec,
    /// Shared scalar scale for E4M3 rows; absent for dense and GGUF rows.
    pub scale: Option<PreparedRowScale>,
}

/// Original hash-control representation, retained without floating conversion.
#[derive(Clone)]
pub enum NGramControls {
    /// Restricted exact I64 tensor sources from the SafeTensors artifact.
    Safetensors(SharedCheckpointSource),
    /// Exact metadata from the same admitted GGUF checkpoint as the table.
    Gguf {
        /// Only the validated architecture/table-control keys, independent of vocabulary size.
        metadata: BTreeMap<String, eredu_gguf::MetadataValue>,
        /// Physical table provenance retained alongside its container metadata.
        table: eredu_checkpoint::store::TensorSourceProvenance,
    },
}

fn hash_spec(
    config: &Config,
    multipliers: Vec<i64>,
    moduli: Vec<i64>,
    offsets: Vec<i64>,
    total: u64,
) -> Result<NGramHashSpec, NGramArtifactError> {
    let mut used = 0u64;
    for (&modulus, &offset) in moduli.iter().zip(&offsets) {
        if modulus <= 0 || offset < 0 || offset as u64 != used {
            return Err(invalid(
                "hash offsets must partition the declared head vocabularies",
            ));
        }
        used = used
            .checked_add(modulus as u64)
            .ok_or_else(|| invalid("hash vocabulary overflow"))?;
    }
    let matches = match config.ngram.source {
        NGramSourceLayout::Safetensors {
            vocabulary_alignment: alignment,
            ..
        } => {
            alignment != 0
                && used
                    .checked_add(alignment - 1)
                    .map(|n| n / alignment * alignment)
                    == Some(total)
        }
        NGramSourceLayout::Gguf { rows } => total == rows && used <= total,
    };
    if !matches {
        return Err(invalid(
            "table rows differ from declared hash vocabulary storage",
        ));
    }
    Ok(NGramHashSpec::new(
        config.vocabulary as u64,
        *config
            .eos
            .first()
            .ok_or_else(|| invalid("missing n-gram reset token"))? as u64,
        config.ngram.order as usize,
        config.ngram.heads as usize,
        multipliers,
        moduli,
        offsets,
        total,
    )?)
}
impl std::fmt::Debug for PreparedNGramTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedNGramTable")
            .field("rows", &self.rows)
            .field("hash", &self.hash)
            .field("lookup", &self.lookup)
            .field("scale", &self.scale)
            .finish_non_exhaustive()
    }
}
/// Missing, ambiguous or malformed released table artifacts.
#[derive(Debug, thiserror::Error)]
pub enum NGramArtifactError {
    /// Missing, duplicated or malformed checkpoint structure.
    #[error("invalid qwen4_exp n-gram artifact: {0}")]
    Layout(String),
    /// Source/provenance/read admission failure.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// Bounded row recipe failure.
    #[error(transparent)]
    Rows(#[from] RowReadError),
    /// Invalid literal hash companions.
    #[error(transparent)]
    Hash(#[from] NGramError),
    /// Invalid row mechanism geometry or admitted row resource bounds.
    #[error(transparent)]
    Lookup(#[from] eredu_runtime::RowLookupError),
}
/// Shared logical row namespace for header admission and retained table binding.
pub(in crate::qwen4_exp) fn table_row_range(
    lookup: &RowLookupSpec,
    metadata: &eredu_checkpoint::recipe::RecipeMetadata,
    policy: eredu_core::residency::ResidencyPolicy,
) -> Result<eredu_core::residency::OffloadUnitRange, NGramArtifactError> {
    use eredu_core::residency::{OffloadUnitId, OffloadUnitRange};
    lookup.validate()?;
    if metadata.byte_len % lookup.rows != 0 {
        return Err(invalid("table bytes do not describe a uniform row range"));
    }
    let prefix = OffloadUnitId::new(format!(
        "parameter_bank.{}.{}.rows",
        lookup.bank, lookup.unit
    ))
    .map_err(|e| invalid(e.to_string()))?;
    OffloadUnitRange::new(
        prefix,
        0,
        lookup.rows,
        metadata.byte_len / lookup.rows,
        policy,
    )
    .map_err(|e| invalid(e.to_string()))
}
fn invalid(message: impl Into<String>) -> NGramArtifactError {
    NGramArtifactError::Layout(message.into())
}
fn retain(
    source: SharedCheckpointSource,
    keys: Vec<String>,
) -> Result<SharedCheckpointSource, NGramArtifactError> {
    let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        source,
        "qwen4_exp n-gram companions",
        keys.into_iter().collect(),
    )?);
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            Ok((
                key.clone(),
                PreparedTensorSource {
                    metadata: source.source_metadata(&key)?,
                    provenance: source.source_provenance(&key)?,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, StoreError>>()?;
    Ok(Arc::new(PreparedCheckpointSource::new(source, catalog)?))
}
fn integers(
    source: &dyn CheckpointSource,
    key: &str,
    count: usize,
) -> Result<Vec<i64>, NGramArtifactError> {
    let meta = source.source_metadata(key)?;
    if count == 0
        || count > 4096
        || meta.stored_dtype != StoredDtype::I64
        || meta.logical_shape != [count]
        || meta.physical_shape != [count]
        || meta.encoded_byte_len != count as u64 * 8
    {
        return Err(invalid(format!(
            "{key} must be a bounded exact I64 vector of length {count}"
        )));
    }
    let lease = source.acquire_lease(TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    })?;
    let bytes = lease
        .encoded_bytes()
        .ok_or_else(|| invalid("I64 companion has no exact encoded bytes"))?;
    if bytes.len() != count * 8 || !lease.bounded_read_proof().physically_bounded {
        return Err(invalid(
            "I64 companion read did not preserve its exact bound",
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight bytes")))
        .collect())
}

#[cfg(test)]
mod tests;
