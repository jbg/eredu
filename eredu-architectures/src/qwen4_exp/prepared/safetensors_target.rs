//! Complete header admission followed by exact source binding and bounded literals.
use super::*;
use crate::qwen4_exp::mtp::{PredictionLimits, PredictionSpec};
use crate::routed_text::SelectedRoutedTextRealization;
use checkpoint::SafetensorsTableSourcePlan;
use eredu_checkpoint::{
    schema::SafetensorsCheckpointPlan,
    validation::{CatalogTensorMetadata, ResolvedCheckpointPlan, SafetensorsCatalog},
};
use eredu_runtime::{AppendStreamBinding, ReplicatedTextPhysicalSource, SelectedRowLookupPlans};

/// Header-only target, prediction and vision contract. This value needs neither a
/// readable source nor backend resources. Source provenance becomes authoritative
/// at binding; the supplied source must match every admitted physical header.
#[derive(Debug, Clone)]
pub struct SafetensorsTargetPlan {
    config: Config,
    encoding: SafetensorsEncoding,
    catalog: HeaderCatalog,
    checkpoint: SafetensorsCheckpointPlan,
    resolution: ResolvedCheckpointPlan,
    tables: BTreeMap<usize, SafetensorsTableSourcePlan>,
}

#[derive(Debug, Clone)]
pub(super) struct HeaderCatalog(pub(super) BTreeMap<String, CatalogTensorMetadata>);
impl SafetensorsCatalog for HeaderCatalog {
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| format!("missing admitted header {key}"))
    }
}
impl HeaderCatalog {
    pub(super) fn tensor_bytes(metadata: &CatalogTensorMetadata) -> Option<u64> {
        use eredu_checkpoint::StoredDtype;
        let width = match metadata.stored_dtype {
            StoredDtype::I64 => 8u64,
            StoredDtype::F32 => 4,
            StoredDtype::BF16 | StoredDtype::F16 => 2,
            StoredDtype::F8E4M3 => 1,
            _ => return None,
        };
        metadata
            .shape
            .iter()
            .try_fold(width, |n, &axis| n.checked_mul(axis as u64))
    }
}
impl eredu_checkpoint::recipe::RecipeCatalog for HeaderCatalog {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        let metadata = self
            .0
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        let bytes = Self::tensor_bytes(metadata)
            .ok_or_else(|| StoreError::PreparedCatalogMismatch { key: key.into() })?;
        Ok(eredu_checkpoint::store::TensorMetadata {
            name: key.into(),
            logical_shape: metadata.shape.clone(),
            physical_shape: metadata.shape.clone(),
            stored_dtype: metadata.stored_dtype.clone(),
            encoded_byte_len: bytes,
            backing_shard: None,
        })
    }
}
impl SafetensorsTargetPlan {
    /// Projects the exact normalized input representation intent before any source access.
    /// The target and projector are selected jointly; the target-only entry rejects this intent.
    pub fn conditional_execution_plan_for_load(
        &self,
        request: &eredu_runtime::NormalizedLoadRequest,
        element: eredu_nn::TensorElementType,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
        vision: VisionPlan,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<ConditionalHeaderExecutionPlan, TargetLoadError> {
        let eredu_runtime::MediaLoadRequest::Required(media) = request.media_execution() else {
            return Err(TargetLoadError::MissingMediaPolicy);
        };
        let target_request = request
            .clone()
            .with_media_execution(eredu_runtime::MediaLoadRequest::Disabled);
        let target = self.execution_plan_for_load(&target_request, element, support, physical)?;
        Ok(target
            .with_vision(vision)?
            .with_load_processor(media.processor().clone(), media.processor_budget()))
    }

    /// Resolves the complete artifact before acquiring integer controls or scales.
    /// All later validation consumes this one immutable catalog snapshot.
    pub fn prepare<C: SafetensorsCatalog + ?Sized>(
        catalog: &C,
        config: Config,
        encoding: SafetensorsEncoding,
    ) -> Result<Self, PreparationError> {
        let mut headers = BTreeMap::new();
        for key in catalog.keys() {
            let metadata = catalog.metadata(&key).map_err(PreparationError::Contract)?;
            if headers.insert(key.clone(), metadata).is_some() {
                return Err(PreparationError::Contract(format!(
                    "duplicate header {key}"
                )));
            }
        }
        let catalog = HeaderCatalog(headers);
        let tables = config
            .ngram
            .layers
            .iter()
            .map(|&layer| {
                Ok((
                    layer,
                    SafetensorsTableSourcePlan::prepare(&catalog, &config, layer)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, PreparationError>>()?;
        let checkpoint = checkpoint::schema::safetensors_plan(&config, &encoding, &tables)
            .map_err(PreparationError::Contract)?;
        let resolution = resolve_safetensors_plan(&catalog, &checkpoint)
            .map_err(|error| PreparationError::Contract(format!("{error:?}")))?;
        Ok(Self {
            config,
            encoding,
            catalog,
            checkpoint,
            resolution,
            tables,
        })
    }

    /// Normalizes admitted layout, formats and recipes before execution policy.
    pub fn normalize(
        &self,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<TargetArtifactDeclaration, PreparationError> {
        self.declarations(physical, true)
    }
    fn declarations(
        &self,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
        exact: bool,
    ) -> Result<TargetArtifactDeclaration, PreparationError> {
        if exact
            && physical.keys().collect::<BTreeSet<_>>()
                != self.resolution.source_keys().iter().collect()
        {
            return Err(PreparationError::Contract(
                "physical declarations differ from admitted source set".into(),
            ));
        }
        for (key, entry) in &physical {
            let header = &self.catalog.0[key];
            if entry.catalog_key() != key
                || entry.source_encoding()
                    != &eredu_checkpoint::SourceTensorEncoding::Safetensors(
                        header.stored_dtype.clone(),
                    )
                || Some(entry.encoded_byte_len()) != HeaderCatalog::tensor_bytes(header)
            {
                return Err(PreparationError::Contract(format!(
                    "physical declaration differs from header {key}"
                )));
            }
        }
        let keys = self
            .resolution
            .source_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let roots = (0..self.config.layers.len())
            .map(|l| format!("model.layers.{l}.mlp.experts"))
            .chain(
                (0..self
                    .config
                    .prediction
                    .as_ref()
                    .map_or(0, |p| p.layers.len()))
                    .map(|d| format!("mtp.layers.{d}.mlp.experts")),
            );
        let (formats, recipes) =
            safetensors_format_recipes(&self.catalog, &keys, &self.config, &self.encoding, roots)?;
        let banks = recipes
            .into_iter()
            .map(|(root, recipes)| {
                Ok((
                    root,
                    Arc::new(PreparedExpertBank::new(
                        &self.catalog,
                        recipes,
                        self.config.experts.count as usize,
                    )?),
                ))
            })
            .collect::<Result<_, PreparationError>>()?;
        let mut static_recipes =
            recipes::parameter_recipes(keys.clone(), &self.config, ParameterScope::Static)
                .map_err(PreparationError::Contract)?;
        static_recipes.retain(|name, _| {
            name.starts_with("model.embed_tokens.")
                || name.starts_with("model.hyper_connection_mixer.")
                || name.starts_with("lm_head.")
        });
        let mut unit_recipes = Vec::new();
        for layer in 0..self.config.layers.len() {
            if self.tables.contains_key(&layer) {
                unit_recipes.push(
                    recipes::parameter_recipes(
                        keys.clone(),
                        &self.config,
                        ParameterScope::Lexical(layer),
                    )
                    .map_err(PreparationError::Contract)?,
                );
            }
            unit_recipes.push(
                recipes::parameter_recipes(
                    keys.clone(),
                    &self.config,
                    ParameterScope::Target(layer),
                )
                .map_err(PreparationError::Contract)?,
            );
        }
        let prediction = self
            .config
            .prediction
            .as_ref()
            .map(|_| {
                if exact {
                    SafetensorsPredictionPlan::prepare(
                        &self.catalog,
                        self.config.clone(),
                        self.encoding.clone(),
                        physical.clone(),
                    )
                } else {
                    SafetensorsPredictionPlan::prepare_metadata(
                        &self.catalog,
                        self.config.clone(),
                        self.encoding.clone(),
                    )
                }
            })
            .transpose()?;
        let metadata = self
            .catalog
            .0
            .iter()
            .map(|(key, header)| {
                (
                    key.clone(),
                    eredu_checkpoint::store::TensorMetadata {
                        name: key.clone(),
                        logical_shape: header.shape.clone(),
                        physical_shape: header.shape.clone(),
                        stored_dtype: header.stored_dtype.clone(),
                        encoded_byte_len: HeaderCatalog::tensor_bytes(header).unwrap_or_default(),
                        backing_shard: None,
                    },
                )
            })
            .collect();
        let parameters = if exact {
            super::requirements::target_parameters(
                &self.config,
                &formats,
                &self.catalog,
                &physical,
                &static_recipes,
                &unit_recipes,
                &banks,
            )?
        } else {
            Default::default()
        };
        let normalized = Arc::new(
            crate::artifact_preparation::NormalizedArtifactPreparation::from_parameters(
                metadata, physical, parameters,
            ),
        );
        Ok(TargetArtifactDeclaration {
            config: self.config.clone(),
            formats,
            normalized,
            binding_keys: self.resolution.source_keys().clone(),
            resolution: self.resolution.clone(),
            gguf_source: None,
            tables: self
                .tables
                .iter()
                .map(|(&layer, table)| Ok((layer, table.normalized()?)))
                .collect::<Result<_, checkpoint::NGramArtifactError>>()?,
            banks,
            static_recipes,
            unit_recipes,
            embedded_prediction: prediction,
            vision_reset: None,
        })
    }

    /// Exact schema, including alternate expert layouts and physical table shards.
    pub fn checkpoint(&self) -> &SafetensorsCheckpointPlan {
        &self.checkpoint
    }
    /// Admitted physical source set; binding cannot choose a different alias/layout.
    pub fn resolution(&self) -> &ResolvedCheckpointPlan {
        &self.resolution
    }
    /// Distinct family geometry retained by this admission.
    pub fn configuration(&self) -> &Config {
        &self.config
    }

    /// Declares the integrated tower from this admission's exact tensor headers.
    /// Physical identities come from the same artifact admission as target execution.
    pub fn vision_plan(
        &self,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<VisionPlan, PreparationError> {
        VisionPlan::safetensors(&self.config, &self.catalog, physical)
    }

    /// Derives executable geometry from headers without reading hash constants,
    /// scale values or weights. Literal binding remains a separate required stage.
    pub fn target_spec(&self, limits: TargetLimits) -> Result<TargetSpec, PreparationError> {
        self.target_spec_selected(limits, None)
    }
    pub(super) fn target_spec_selected(
        &self,
        limits: TargetLimits,
        selected: Option<&SelectedRoutedTextRealization>,
    ) -> Result<TargetSpec, PreparationError> {
        limits.validate_for_config(&self.config)?;
        let keys = self
            .resolution
            .source_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let roots = (0..self.config.layers.len()).map(|l| format!("model.layers.{l}.mlp.experts"));
        let (formats, _) =
            safetensors_format_recipes(&self.catalog, &keys, &self.config, &self.encoding, roots)?;
        let mut rows = BTreeMap::new();
        let mut ordinal = 0;
        for layer in 0..self.config.layers.len() {
            if let Some(table) = self.tables.get(&layer) {
                rows.insert(
                    layer,
                    table.lookup_spec(rows.len() + 1, ordinal, limits.element)?,
                );
                ordinal += 1;
            }
            ordinal += 1;
        }
        Ok(super::execution::target_spec_from_formats(
            self.config.clone(),
            &rows,
            limits,
            &formats,
            selected,
        )?)
    }

    /// Prediction geometry from admitted headers, without source acquisition.
    pub fn prediction_spec(
        &self,
        limits: PredictionLimits,
    ) -> Result<PredictionSpec, PreparationError> {
        self.prediction_parts(limits).map(|(spec, _, _)| spec)
    }

    fn prediction_parts(
        &self,
        limits: PredictionLimits,
    ) -> Result<
        (
            PredictionSpec,
            ParameterFormats,
            BTreeMap<String, Arc<PreparedExpertBank>>,
        ),
        PreparationError,
    > {
        let prediction = self
            .config
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        let keys = self
            .resolution
            .source_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let roots =
            (0..prediction.layers.len()).map(|depth| format!("mtp.layers.{depth}.mlp.experts"));
        let (formats, recipes) =
            safetensors_format_recipes(&self.catalog, &keys, &self.config, &self.encoding, roots)?;
        let banks = recipes
            .into_iter()
            .map(|(root, recipes)| {
                Ok((
                    root,
                    Arc::new(PreparedExpertBank::new(
                        &self.catalog,
                        recipes,
                        self.config.experts.count as usize,
                    )?),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, PreparationError>>()?;
        let bank = eredu_runtime::RoutedBankId::new(u32::try_from(self.tables.len() + 1).map_err(
            |_| PreparationError::Contract("prediction bank identity exceeds u32".into()),
        )?);
        let spec = PredictionSpec::from_prepared(
            &self.config,
            limits,
            bank,
            |depth| {
                expert_spec(
                    &self.config,
                    &formats,
                    &format!("mtp.layers.{depth}.mlp.experts"),
                )
            },
            |name| formats.ordinary(name),
        )?;
        Ok((spec, formats, banks))
    }

    /// Declares compact row banks without readable sources. The returned contract
    /// can be selected once and passed to the bound target's execution plan.
    pub fn row_descriptors(
        &self,
        limits: TargetLimits,
        row_limits: RowLookupLimits,
        policy: ResidencyPolicy,
    ) -> Result<eredu_runtime::RowLookupDescriptors, PreparationError> {
        let spec = self.target_spec(limits)?;
        if row_limits.requests < limits.lookup_rows {
            return Err(eredu_runtime::RowLookupError::Budget {
                resource: "target lookup requests",
                required: limits.lookup_rows as u64,
                limit: row_limits.requests as u64,
            }
            .into());
        }
        let mut entries = Vec::new();
        for (ordinal, unit) in spec.units.iter().enumerate() {
            if let UnitSpec::Lexical { layer, .. } = unit {
                entries.push(self.tables[layer].row_descriptor(
                    entries.len() + 1,
                    ordinal,
                    limits.element,
                    row_limits,
                    policy,
                )?);
            }
        }
        Ok(eredu_runtime::RowLookupDescriptors::new(
            entries,
            spec.units.len(),
        )?)
    }

    /// Projects bounded normalized load policy using headers alone, selecting row
    /// mechanisms before any source is bound. The exact request remains authoritative
    /// through subsequent ordinary mechanism selection and source binding.
    pub fn execution_plan_for_load(
        &self,
        request: &eredu_runtime::NormalizedLoadRequest,
        element: eredu_nn::TensorElementType,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<TargetPreparationPlan, TargetLoadError> {
        self.normalize(physical)?
            .execution_plan_for_load(request, element, support)
    }

    /// Retains a normalized TP/PP request through source-free mechanism selection.
    pub fn partition_execution_plan_for_load(
        &self,
        request: &eredu_runtime::NormalizedLoadRequest,
        element: eredu_nn::TensorElementType,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<TargetPartitionExecutionPlan, TargetLoadError> {
        let partition = super::partition_selection::load_partition_request(request)?;
        Ok(self
            .execution_plan_for_load(request, element, support, physical)?
            .partition(partition)?)
    }

    /// Authors complete target requirements using headers and exact physical
    /// provenance supplied by artifact admission. No readable source is accepted.
    pub fn execution_plan(
        &self,
        limits: TargetLimits,
        streams: Vec<AppendStreamBinding>,
        row_admission: SelectedRowLookupPlans,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<TargetPreparationPlan, PreparationError> {
        self.normalize(physical)?
            .execution_plan(limits, streams, row_admission)
    }

    /// Consumes header authority and pins exact metadata/provenance before reading
    /// bounded hash vectors and scalar scales. Accepts the complete admitted source
    /// or its already-resolved claimed view; it never opens an artifact.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        limits: TargetLimits,
    ) -> Result<PreparedTarget, PreparationError> {
        self.declarations(BTreeMap::new(), false)?
            .bind(source, limits)
    }
}
