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

    /// Declares the logical provider identity and row encoding without payload access.
    pub fn lookup_spec(
        &self,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<RowLookupSpec, NGramArtifactError> {
        let canonical = format!(
            "model.layers.{}.ple.ple_embedding.ngram_embedding",
            self.layer
        );
        let scale_parameter = self
            .scale_name
            .as_ref()
            .map(|_| ParameterId::new(format!("{canonical}.weight_scale")))
            .transpose()
            .map_err(|e| invalid(e.to_string()))?;
        let lookup = RowLookupSpec {
            parameter: ParameterId::new(format!("{canonical}.weight"))
                .map_err(|e| invalid(e.to_string()))?,
            bank,
            unit,
            rows: self.total,
            dimensions: self.width,
            encoding: scale_parameter
                .as_ref()
                .map_or(RowEncoding::Dense, |scale| RowEncoding::ScalarE4M3 {
                    scale: scale.clone(),
                }),
            output_type,
        };
        lookup.validate().map_err(|e| invalid(e.to_string()))?;
        Ok(lookup)
    }

    fn row_recipe(&self) -> DerivedWeightRecipe {
        DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: self
                .shards
                .iter()
                .map(|name| DerivedWeightRecipe::source(name, TensorSelection::Full))
                .collect(),
        }
    }

    /// Complete cold row mechanism contract, inferred from compact shard recipes
    /// and scalar headers. Contains no readable sources or artifact paths.
    pub fn row_descriptor(
        &self,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
        limits: eredu_runtime::RowLookupLimits,
        policy: eredu_core::residency::ResidencyPolicy,
    ) -> Result<eredu_runtime::RowLookupDescriptor, NGramArtifactError> {
        use eredu_runtime::{RowLookupDescriptor, RowScaleDescriptor, RowScaleSource};
        let lookup = self.lookup_spec(bank, unit, output_type)?;
        let metadata = self
            .row_recipe()
            .infer(self)
            .map_err(|e| invalid(e.to_string()))?;
        let range = table_row_range(&lookup, &metadata, policy)?;
        let scale = self
            .scale_name
            .as_ref()
            .map(|name| {
                let RowEncoding::ScalarE4M3 { scale } = &lookup.encoding else {
                    unreachable!("declared scalar encoding")
                };
                let metadata = self.tensor_metadata(name)?;
                let encoding = eredu_checkpoint::SourceTensorEncoding::Safetensors(
                    metadata.stored_dtype.clone(),
                );
                Ok::<_, NGramArtifactError>(RowScaleDescriptor::new(
                    scale.clone(),
                    DerivedWeightRecipe::source(name, TensorSelection::Full),
                    BTreeMap::from([(name.clone(), RowScaleSource { metadata, encoding })]),
                )?)
            })
            .transpose()?;
        Ok(RowLookupDescriptor::new(
            range, metadata, lookup, scale, limits,
        )?)
    }

    /// Pins exact physical sources, then reads only bounded integer controls and
    /// the optional scalar scale. Table bytes remain deferred to row acquisition.
    pub fn bind(
        &self,
        source: SharedCheckpointSource,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<PreparedNGramTable, NGramArtifactError> {
        let lookup = self.lookup_spec(bank, unit, output_type)?;
        let source = retain(source, self.catalog.keys().cloned().collect())?;
        // Validate every source before the first payload acquisition.
        for key in self.catalog.keys() {
            let meta = source.source_metadata(key)?;
            let admitted = self.tensor_metadata(key)?;
            if meta.logical_shape != admitted.logical_shape
                || meta.physical_shape != admitted.physical_shape
                || meta.stored_dtype != admitted.stored_dtype
                || meta.encoded_byte_len != admitted.encoded_byte_len
            {
                return Err(invalid(format!(
                    "retained SafeTensors source {key} differs from its admitted header"
                )));
            }
        }
        let rows = PreparedRowSource::new(source.clone(), self.row_recipe())?;
        let multipliers = integers(
            source.as_ref(),
            &self.names[0],
            self.config.ngram.order as usize,
        )?;
        let moduli = integers(source.as_ref(), &self.names[1], self.heads)?;
        let offsets = integers(source.as_ref(), &self.names[2], self.heads)?;
        let hash = hash_spec(&self.config, multipliers, moduli, offsets, self.total)?;
        let scale = if let Some(name) = &self.scale_name {
            let lease = source.acquire_lease(TensorReadRequest {
                key: name.clone(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })?;
            let bytes = lease
                .encoded_bytes()
                .ok_or_else(|| invalid("scale has no encoded bytes"))?;
            let value = match (&self.catalog[name].stored_dtype, bytes) {
                (StoredDtype::BF16, [a, b]) => {
                    half::bf16::from_bits(u16::from_le_bytes([*a, *b])).to_f32()
                }
                (StoredDtype::F16, [a, b]) => {
                    half::f16::from_bits(u16::from_le_bytes([*a, *b])).to_f32()
                }
                (StoredDtype::F32, [a, b, c, d]) => f32::from_le_bytes([*a, *b, *c, *d]),
                _ => {
                    return Err(invalid(
                        "scale payload length differs from its scalar dtype",
                    ))
                }
            };
            if !value.is_finite() || value <= 0. || !lease.bounded_read_proof().physically_bounded {
                return Err(invalid(
                    "FP8 table scale must be finite, positive and bounded",
                ));
            }
            Some(PreparedRowScale {
                parameter: match &lookup.encoding {
                    RowEncoding::ScalarE4M3 { scale } => scale.clone(),
                    _ => unreachable!("declared scalar scale"),
                },
                source: retain(source.clone(), vec![name.clone()])?,
                recipe: DerivedWeightRecipe::source(name, TensorSelection::Full),
            })
        } else {
            None
        };
        Ok(PreparedNGramTable {
            rows,
            hash,
            controls: NGramControls::Safetensors(retain(source, self.names.to_vec())?),
            lookup,
            scale,
        })
    }
}
