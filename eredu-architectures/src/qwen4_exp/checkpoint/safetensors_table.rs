//! Header-only table contracts and deferred, bounded literal acquisition.
use super::*;
use eredu_checkpoint::validation::{CatalogTensorMetadata, SafetensorsCatalog};
use eredu_checkpoint::{recipe::RecipeCatalog, store::TensorMetadata};

/// One admitted injection table. Header admission needs no readable source;
/// binding pins physical metadata and provenance before acquiring any literals.
#[derive(Debug, Clone)]
pub struct SafetensorsTableSourcePlan {
    config: Config,
    layer: usize,
    heads: usize,
    width: i32,
    total: u64,
    pub(super) catalog: BTreeMap<String, CatalogTensorMetadata>,
    pub(super) shards: Vec<String>,
    names: [String; 3],
    pub(super) scale_name: Option<String>,
}
impl RecipeCatalog for SafetensorsTableSourcePlan {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let header = self
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        let scalar_bytes = match header.stored_dtype {
            StoredDtype::I64 => 8,
            StoredDtype::F32 => 4,
            StoredDtype::BF16 | StoredDtype::F16 => 2,
            StoredDtype::F8E4M3 => 1,
            _ => unreachable!("admitted scalar encoding"),
        };
        let encoded_byte_len = header
            .shape
            .iter()
            .try_fold(scalar_bytes, |n: u64, &v| n.checked_mul(v as u64))
            .ok_or_else(|| StoreError::Overflow {
                context: format!("n-gram header bytes for {key}"),
            })?;
        Ok(TensorMetadata {
            name: key.into(),
            logical_shape: header.shape.clone(),
            physical_shape: header.shape.clone(),
            stored_dtype: header.stored_dtype.clone(),
            encoded_byte_len,
            backing_shard: None,
        })
    }
}

impl SafetensorsTableSourcePlan {
    /// Admits aliases, shard geometry and companion encodings from headers only.
    pub fn prepare<C: SafetensorsCatalog + ?Sized>(
        source: &C,
        config: &Config,
        layer: usize,
    ) -> Result<Self, NGramArtifactError> {
        if !config.ngram.layers.contains(&layer) {
            return Err(invalid("layer has no declared n-gram injection"));
        }
        let heads = config
            .ngram
            .order
            .checked_sub(1)
            .and_then(|n| n.checked_mul(config.ngram.heads))
            .filter(|n| *n > 0 && *n <= 4096)
            .ok_or_else(|| invalid("invalid hash head geometry"))? as usize;
        let NGramSourceLayout::Safetensors {
            shards,
            vocabulary_alignment,
            ..
        } = config.ngram.source
        else {
            return Err(invalid(
                "SafeTensors table requires its original source declaration",
            ));
        };
        if shards == 0
            || vocabulary_alignment == 0
            || config.ngram.embedding_dim <= 0
            || config.ngram.embedding_dim as usize % heads != 0
            || config.eos.is_empty()
            || config.ngram.order > 4096
        {
            return Err(invalid("invalid table configuration"));
        }
        let keys = source.keys();
        let roots: Vec<_> = [
            format!("model.language_model.layers.{layer}.ple.ple_embedding"),
            format!("model.layers.{layer}.ple.ple_embedding"),
            format!("layers.{layer}.ple.ple_embedding"),
        ]
        .into_iter()
        .filter(|root| keys.iter().any(|key| key.starts_with(&format!("{root}."))))
        .collect();
        if roots.len() != 1 {
            return Err(invalid(
                "expected exactly one complete table namespace alias",
            ));
        }
        let root = &roots[0];
        let prefix = format!("{root}.ngram_embedding.shard_");
        if keys.iter().filter(|key| key.starts_with(&prefix)).count() != shards {
            return Err(invalid("table shard count differs from the declared split"));
        }
        let width = config.ngram.embedding_dim / heads as i32;
        let mut catalog = BTreeMap::new();
        let mut shard_names = Vec::with_capacity(shards);
        let mut total = 0u64;
        let mut dtype = None;
        for shard in 0..shards {
            let name = format!("{prefix}{shard}.weight");
            let meta = source.metadata(&name).map_err(invalid)?;
            if meta.shape.len() != 2
                || meta.shape[0] == 0
                || meta.shape[1] != width as usize
                || !matches!(
                    meta.stored_dtype,
                    StoredDtype::BF16 | StoredDtype::F16 | StoredDtype::F32 | StoredDtype::F8E4M3
                )
                || dtype
                    .as_ref()
                    .is_some_and(|dtype| dtype != &meta.stored_dtype)
            {
                return Err(invalid(format!("malformed table shard {name}")));
            }
            total = total
                .checked_add(meta.shape[0] as u64)
                .ok_or_else(|| invalid("table row count overflow"))?;
            dtype = Some(meta.stored_dtype.clone());
            catalog.insert(name.clone(), meta);
            shard_names.push(name);
        }
        if total > i64::MAX as u64 || total % vocabulary_alignment != 0 {
            return Err(invalid("table row count differs from vocabulary alignment"));
        }
        let names = [
            format!("{root}.layer_multipliers"),
            format!("{root}.ngram_heads_vocab_sizes"),
            format!("{root}.ngram_heads_offsets"),
        ];
        for (name, count) in names
            .iter()
            .zip([config.ngram.order as usize, heads, heads])
        {
            let meta = source.metadata(name).map_err(invalid)?;
            if meta.stored_dtype != StoredDtype::I64 || meta.shape != [count] {
                return Err(invalid(format!(
                    "{name} must be a bounded exact I64 vector of length {count}"
                )));
            }
            catalog.insert(name.clone(), meta);
        }
        let scale_name = format!("{root}.ngram_embedding.weight_scale");
        let fp8 = dtype == Some(StoredDtype::F8E4M3);
        if !fp8 && keys.contains(&scale_name) {
            return Err(invalid("dense table unexpectedly has an FP8 scale"));
        }
        let scale_name = if fp8 {
            let meta = source.metadata(&scale_name).map_err(invalid)?;
            if !matches!(meta.shape.as_slice(), [] | [1])
                || !matches!(
                    meta.stored_dtype,
                    StoredDtype::BF16 | StoredDtype::F16 | StoredDtype::F32
                )
            {
                return Err(invalid(
                    "FP8 table scale must be a single floating scalar, not an expert block scale",
                ));
            }
            catalog.insert(scale_name.clone(), meta);
            Some(scale_name)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            layer,
            heads,
            width,
            total,
            catalog,
            shards: shard_names,
            names,
            scale_name,
        })
    }
    /// Number of logical rows across the compact shard recipe.
    pub fn rows(&self) -> u64 {
        self.total
    }
    /// Width of each lookup row.
    pub fn dimensions(&self) -> i32 {
        self.width
    }
    /// Whether this table declares the independent shared E4M3 scale.
    pub fn scalar_fp8(&self) -> bool {
        self.scale_name.is_some()
    }

    pub(in crate::qwen4_exp) fn normalized(
        &self,
    ) -> Result<super::table::TableDeclaration, NGramArtifactError> {
        let scale = self
            .scale_name
            .as_ref()
            .map(|_| {
                ParameterId::new(format!(
                    "model.layers.{}.ple.ple_embedding.ngram_embedding.weight_scale",
                    self.layer
                ))
                .map_err(|e| invalid(e.to_string()))
            })
            .transpose()?;
        Ok(super::table::TableDeclaration {
            layer: self.layer,
            rows: self.total,
            dimensions: self.width,
            encoding: scale.map_or(RowEncoding::Dense, |scale| RowEncoding::ScalarE4M3 {
                scale,
            }),
            recipe: DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: self
                    .shards
                    .iter()
                    .map(|key| DerivedWeightRecipe::source(key, TensorSelection::Full))
                    .collect(),
            },
            metadata: self
                .catalog
                .keys()
                .map(|key| Ok((key.clone(), self.tensor_metadata(key)?)))
                .collect::<Result<_, StoreError>>()?,
            provenance: BTreeMap::new(),
            literals: super::table::HashLiterals::Deferred {
                config: self.config.clone(),
                names: self.names.clone(),
                heads: self.heads,
            },
            scale_name: self.scale_name.clone(),
        })
    }
    /// Declares a logical row owner from compact admitted recipes.
    pub fn lookup_spec(
        &self,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<RowLookupSpec, NGramArtifactError> {
        self.normalized()?.lookup_spec(bank, unit, output_type)
    }
    /// Header-only row mechanism contract.
    pub fn row_descriptor(
        &self,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
        limits: eredu_runtime::RowLookupLimits,
        policy: eredu_core::residency::ResidencyPolicy,
    ) -> Result<eredu_runtime::RowLookupDescriptor, NGramArtifactError> {
        self.normalized()?
            .row_descriptor(bank, unit, output_type, limits, policy)
    }
    /// Acquires bounded literals while retaining lazy compact rows.
    pub fn bind(
        &self,
        source: SharedCheckpointSource,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<PreparedNGramTable, NGramArtifactError> {
        self.normalized()?.bind(source, bank, unit, output_type)
    }
}
