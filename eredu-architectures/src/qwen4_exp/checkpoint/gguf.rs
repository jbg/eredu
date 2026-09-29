//! Table mapping from llama.cpp 2145525a4081d66ff1a87cf43ef809f95a85ac0c.
//! Decoder/vision mappings and public GGUF admission are separate from this owner.
use super::*;
use eredu_checkpoint::recipe::RecipeCatalog;
use eredu_checkpoint::schema::{GgufTensorConstraint, GgufTypeConstraint, TensorOperation};
use eredu_gguf::{Checkpoint, LogicalDtype, MetadataValue};

const TABLE: &str = "per_layer_token_embd.weight";

/// Header-only contract for the one published physical table. Binding uses an
/// existing exact source; it never opens another store or reader cache.
#[derive(Clone)]
pub struct GgufTableSourcePlan {
    checkpoint: Checkpoint,
    constraint: GgufTensorConstraint,
    controls: TableControls,
    layers: Vec<usize>,
    metadata: eredu_checkpoint::store::TensorMetadata,
    provenance: eredu_checkpoint::store::TensorSourceProvenance,
    encoding: RowEncoding,
}
impl RecipeCatalog for GgufTableSourcePlan {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        if key != TABLE {
            return Err(StoreError::UnknownTensor { key: key.into() });
        }
        let mut metadata = self.metadata.clone();
        metadata.backing_shard = None;
        Ok(metadata)
    }
}
impl GgufTableSourcePlan {
    /// Validates exact integer metadata and preserves encoded table blocks.
    pub fn prepare(checkpoint: &Checkpoint, config: &Config) -> Result<Self, NGramArtifactError> {
        let layer = *config
            .ngram
            .layers
            .first()
            .ok_or_else(|| invalid("missing injection layer"))?;
        let controls = controls(checkpoint, config, layer)?;
        let encoded = checkpoint
            .tensors()
            .filter(|t| {
                t.descriptor().name == TABLE
                    && t.descriptor()
                        .ggml_type
                        .block_and_bytes()
                        .is_ok_and(|(block, _)| block > 1)
            })
            .map(|t| t.descriptor().name.clone())
            .collect::<Vec<_>>();
        let checkpoint = checkpoint
            .clone()
            .into_tensor_representation(encoded, eredu_gguf::QuantizedTensorRepresentation::Encoded)
            .map_err(|e| invalid(e.to_string()))?;
        let (shard, tensor) = checkpoint
            .shards()
            .iter()
            .find_map(|shard| {
                shard
                    .tensors()
                    .iter()
                    .find(|tensor| tensor.descriptor().name == TABLE)
                    .map(|tensor| (shard, tensor))
            })
            .ok_or_else(|| invalid("missing GGUF per-layer token embedding table"))?;
        let descriptor = tensor.descriptor();
        let [output] = tensor.outputs() else {
            return Err(invalid(
                "encoded GGUF table must have one logical row output",
            ));
        };
        if output.name != TABLE {
            return Err(invalid("GGUF row output identity differs from the table"));
        }
        let (encoding, stored_dtype, scalar_bytes) = match output.dtype {
            LogicalDtype::F32 => (RowEncoding::Dense, StoredDtype::F32, 4),
            LogicalDtype::F16 => (RowEncoding::Dense, StoredDtype::F16, 2),
            LogicalDtype::Bf16 => (RowEncoding::Dense, StoredDtype::BF16, 2),
            LogicalDtype::U8 => (
                RowEncoding::Gguf {
                    encoding: descriptor.ggml_type,
                    endian: shard.endian(),
                },
                StoredDtype::U8,
                1,
            ),
            _ => {
                return Err(invalid(
                    "GGUF table requires a floating or encoded-byte row layout",
                ))
            }
        };
        let shape = |values: &[u64]| {
            values
                .iter()
                .map(|&n| usize::try_from(n).map_err(|_| invalid("GGUF table dimension overflow")))
                .collect::<Result<Vec<_>, _>>()
        };
        let metadata = eredu_checkpoint::store::TensorMetadata {
            name: TABLE.into(),
            logical_shape: shape(&output.shape)?,
            physical_shape: shape(&descriptor.row_major_shape())?,
            stored_dtype,
            encoded_byte_len: output
                .shape
                .iter()
                .try_fold(scalar_bytes, |n: u64, &v| n.checked_mul(v))
                .ok_or_else(|| invalid("GGUF table byte length overflow"))?,
            backing_shard: Some(shard.path().to_path_buf()),
        };
        let provenance = eredu_checkpoint::store::TensorSourceProvenance {
            catalog_key: TABLE.into(),
            physical_tensor: TABLE.into(),
            output: TABLE.into(),
            backing_shard: metadata.backing_shard.clone(),
            source_encoding: eredu_checkpoint::SourceTensorEncoding::Gguf {
                ggml_type: descriptor.ggml_type,
                endian: shard.endian(),
            },
        };
        let constraint = GgufTensorConstraint::required(
            TABLE,
            vec![
                usize::try_from(controls.rows).map_err(|_| invalid("row count overflow"))?,
                controls.width as usize,
            ],
            GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
        );
        Ok(Self {
            checkpoint,
            constraint,
            controls,
            layers: config.ngram.layers.clone(),
            metadata,
            provenance,
            encoding,
        })
    }
    /// Header view whose table output retains its exact physical encoding.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }
    /// Table constraint to combine with ordinary parameter admission.
    pub fn constraint(&self) -> &GgufTensorConstraint {
        &self.constraint
    }
    pub(in crate::qwen4_exp) fn normalized(
        &self,
        layer: usize,
    ) -> Result<super::table::TableDeclaration, NGramArtifactError> {
        if !self.layers.contains(&layer) {
            return Err(invalid(
                "GGUF table owner is not a declared injection layer",
            ));
        }
        Ok(super::table::TableDeclaration {
            layer,
            rows: self.controls.rows,
            dimensions: self.controls.width,
            encoding: self.encoding.clone(),
            recipe: DerivedWeightRecipe::source(TABLE, TensorSelection::Full),
            metadata: BTreeMap::from([(TABLE.into(), self.tensor_metadata(TABLE)?)]),
            provenance: BTreeMap::from([(TABLE.into(), self.provenance.clone())]),
            literals: super::table::HashLiterals::Admitted {
                hash: self.controls.hash.clone(),
                metadata: self.controls.metadata.clone(),
                table: self.provenance.clone(),
            },
            scale_name: None,
        })
    }
    /// Declares a logical row owner from compact admitted recipes.
    pub fn lookup_spec(
        &self,
        layer: usize,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<RowLookupSpec, NGramArtifactError> {
        self.normalized(layer)?.lookup_spec(bank, unit, output_type)
    }
    /// Header-only row mechanism contract.
    pub fn row_descriptor(
        &self,
        layer: usize,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
        limits: eredu_runtime::RowLookupLimits,
        policy: eredu_core::residency::ResidencyPolicy,
    ) -> Result<eredu_runtime::RowLookupDescriptor, NGramArtifactError> {
        self.normalized(layer)?
            .row_descriptor(bank, unit, output_type, limits, policy)
    }
    /// Binds exact integer metadata while retaining lazy compact rows.
    pub fn bind(
        &self,
        source: SharedCheckpointSource,
        layer: usize,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<PreparedNGramTable, NGramArtifactError> {
        self.normalized(layer)?
            .bind(source, bank, unit, output_type)
    }
}

/// Pure metadata validation shared by cold configuration and prepared row sources.
#[derive(Clone)]
pub(in crate::qwen4_exp) struct TableControls {
    pub metadata: BTreeMap<String, MetadataValue>,
    pub hash: NGramHashSpec,
    pub width: i32,
    pub rows: u64,
}
pub(in crate::qwen4_exp) fn controls(
    checkpoint: &Checkpoint,
    config: &Config,
    layer: usize,
) -> Result<TableControls, NGramArtifactError> {
    let metadata = checkpoint.metadata();
    if metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        != Some("qwen4exp")
    {
        return Err(invalid("expected GGUF architecture qwen4exp"));
    }
    let mut retained = BTreeMap::from([(
        "general.architecture".into(),
        metadata["general.architecture"].clone(),
    )]);
    let mut integer = |suffix: &str| -> Result<i64, NGramArtifactError> {
        let key = format!("qwen4exp.{suffix}");
        let value = metadata
            .get(&key)
            .ok_or_else(|| invalid(format!("missing {key}")))?;
        let result = value
            .as_i64()
            .ok_or_else(|| invalid(format!("{key} must be an exact integer")))?;
        retained.insert(key, value.clone());
        Ok(result)
    };
    let heads = config
        .ngram
        .order
        .checked_sub(1)
        .and_then(|n| n.checked_mul(config.ngram.heads))
        .filter(|n| *n > 0 && *n <= 4096)
        .ok_or_else(|| invalid("invalid bounded n-gram head count"))? as usize;
    if config.vocabulary <= 0
        || config.ngram.embedding_dim <= 0
        || config.ngram.embedding_dim as usize % heads != 0
        || !config.ngram.layers.contains(&layer)
    {
        return Err(invalid("invalid configured GGUF table geometry"));
    }
    let width = config.ngram.embedding_dim / heads as i32;
    for (key, expected) in [
        ("ple.ngram_size", i64::from(config.ngram.order)),
        ("ple.heads_per_ngram", i64::from(config.ngram.heads)),
        ("ple.conv_kernel", i64::from(config.ngram.kernel)),
        (
            "ple.eos_token_id",
            i64::from(
                *config
                    .eos
                    .first()
                    .ok_or_else(|| invalid("missing reset token"))?,
            ),
        ),
    ] {
        if integer(key)? != expected {
            return Err(invalid(format!(
                "GGUF {key} differs from configured n-gram semantics"
            )));
        }
    }
    // The converter emits this only when it has already seen table shards.
    // The physical descriptor below remains authoritative when it is absent.
    if metadata.contains_key("qwen4exp.embedding_length_per_layer_input")
        && integer("embedding_length_per_layer_input")? != i64::from(width)
    {
        return Err(invalid(
            "GGUF row-width metadata differs from configuration",
        ));
    }
    if metadata.contains_key("qwen4exp.ple.image_token_id") {
        let image = integer("ple.image_token_id")?;
        if image < 0
            || image >= i64::from(config.vocabulary)
            || config
                .media
                .as_ref()
                .is_some_and(|media| i64::from(media.image) != image)
        {
            return Err(invalid(
                "GGUF image token differs from configured media identity",
            ));
        }
    }
    let mut integers = |suffix: &str, count: usize| -> Result<Vec<i64>, NGramArtifactError> {
        let key = format!("qwen4exp.{suffix}");
        let value = metadata
            .get(&key)
            .ok_or_else(|| invalid(format!("missing {key}")))?;
        // Check length before allocating or copying metadata. Integer accessors
        // use checked conversion for UINT64; no hash bits pass through float.
        let array = value
            .as_array()
            .filter(|a| count > 0 && count <= 4096 && a.len() == count)
            .ok_or_else(|| invalid(format!("{key} must have exactly {count} bounded entries")))?;
        let result = array.to_i64_vec().ok_or_else(|| {
            invalid(format!(
                "{key} must contain exact signed-representable integers"
            ))
        })?;
        retained.insert(key, value.clone());
        Ok(result)
    };
    let layers = integers("ple.layers", config.ngram.layers.len())?;
    if layers
        .iter()
        .copied()
        .ne(config.ngram.layers.iter().map(|&v| v as i64))
    {
        return Err(invalid(
            "GGUF zero-based injection layers differ from configuration",
        ));
    }
    let multipliers = integers("ple.layer_multipliers", config.ngram.order as usize)?;
    let moduli = integers("ple.head_vocab_sizes", heads)?;
    let offsets = integers("ple.head_offsets", heads)?;
    let descriptor = checkpoint
        .tensors()
        .find(|t| t.descriptor().name == TABLE)
        .ok_or_else(|| invalid("missing GGUF per-layer token embedding table"))?
        .descriptor();
    let shape = descriptor.row_major_shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] != width as u64 {
        return Err(invalid(
            "GGUF table dimensions differ from configured row width",
        ));
    }
    let rows = shape[0];
    let hash = hash_spec(config, multipliers, moduli, offsets, rows)?;
    Ok(TableControls {
        metadata: retained,
        hash,
        width,
        rows,
    })
}

#[cfg(test)]
mod tests;
