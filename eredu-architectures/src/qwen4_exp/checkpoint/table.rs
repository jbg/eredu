//! Compact row recipes and typed hash-control declarations shared by containers.
use super::*;
use eredu_checkpoint::recipe::RecipeCatalog;
use eredu_checkpoint::store::{TensorMetadata, TensorSourceProvenance};

#[derive(Clone)]
pub(in crate::qwen4_exp) enum HashLiterals {
    Deferred {
        config: Config,
        names: [String; 3],
        heads: usize,
    },
    Admitted {
        hash: NGramHashSpec,
        metadata: BTreeMap<String, eredu_gguf::MetadataValue>,
        table: TensorSourceProvenance,
    },
}

/// One immutable compact table recipe, independent of request limits and mechanisms.
#[derive(Clone)]
pub(in crate::qwen4_exp) struct TableDeclaration {
    pub layer: usize,
    pub rows: u64,
    pub dimensions: i32,
    pub encoding: RowEncoding,
    pub recipe: DerivedWeightRecipe,
    pub metadata: BTreeMap<String, TensorMetadata>,
    pub provenance: BTreeMap<String, TensorSourceProvenance>,
    pub literals: HashLiterals,
    pub scale_name: Option<String>,
}
impl RecipeCatalog for TableDeclaration {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.metadata
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}
impl TableDeclaration {
    pub fn lookup_spec(
        &self,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<RowLookupSpec, NGramArtifactError> {
        let lookup = RowLookupSpec {
            parameter: ParameterId::new(format!(
                "model.layers.{}.ple.ple_embedding.ngram_embedding.weight",
                self.layer
            ))
            .map_err(|e| invalid(e.to_string()))?,
            bank,
            unit,
            rows: self.rows,
            dimensions: self.dimensions,
            encoding: self.encoding.clone(),
            output_type,
        };
        lookup.validate().map_err(|e| invalid(e.to_string()))?;
        Ok(lookup)
    }
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
            .recipe
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
    pub fn bind(
        &self,
        source: SharedCheckpointSource,
        bank: usize,
        unit: usize,
        output_type: TensorElementType,
    ) -> Result<PreparedNGramTable, NGramArtifactError> {
        let lookup = self.lookup_spec(bank, unit, output_type)?;
        let source = retain(source, self.metadata.keys().cloned().collect())?;
        for (key, admitted) in &self.metadata {
            let meta = source.source_metadata(key)?;
            if meta.logical_shape != admitted.logical_shape
                || meta.physical_shape != admitted.physical_shape
                || meta.stored_dtype != admitted.stored_dtype
                || meta.encoded_byte_len != admitted.encoded_byte_len
            {
                return Err(invalid(format!(
                    "retained table source {key} differs from its admitted header"
                )));
            }
        }
        for (key, admitted) in &self.provenance {
            if source.source_provenance(key)? != *admitted {
                return Err(invalid(
                    "retained GGUF table source differs from its admitted header",
                ));
            }
        }
        let rows = PreparedRowSource::new(source.clone(), self.recipe.clone())?;
        let (hash, controls) = match &self.literals {
            HashLiterals::Deferred {
                config,
                names,
                heads,
            } => {
                let multipliers =
                    integers(source.as_ref(), &names[0], config.ngram.order as usize)?;
                let moduli = integers(source.as_ref(), &names[1], *heads)?;
                let offsets = integers(source.as_ref(), &names[2], *heads)?;
                (
                    hash_spec(config, multipliers, moduli, offsets, self.rows)?,
                    NGramControls::Safetensors(retain(source.clone(), names.to_vec())?),
                )
            }
            HashLiterals::Admitted {
                hash,
                metadata,
                table,
            } => (
                hash.clone(),
                NGramControls::Gguf {
                    metadata: metadata.clone(),
                    table: table.clone(),
                },
            ),
        };
        let scale = if let Some(name) = &self.scale_name {
            let lease = source.acquire_lease(TensorReadRequest {
                key: name.clone(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })?;
            let bytes = lease
                .encoded_bytes()
                .ok_or_else(|| invalid("scale has no encoded bytes"))?;
            let value = match (&self.metadata[name].stored_dtype, bytes) {
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
            controls,
            lookup,
            scale,
        })
    }
}
