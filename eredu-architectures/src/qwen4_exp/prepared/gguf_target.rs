//! Header-only GGUF target requirements and exact retained-source binding.
use super::target_preparation::validate_row_requests;
use super::*;
use crate::routed_text::SelectedRoutedTextRealization;
use checkpoint::gguf_text::GgufTextPlan;
use eredu_runtime::{AppendStreamBinding, ReplicatedTextPhysicalSource, SelectedRowLookupPlans};

/// Complete published GGUF text header contract. The pinned exporter omits MTP
/// and supplies vision separately; optional prediction uses independent SafeTensors headers.
#[derive(Clone)]
pub struct GgufTargetPlan {
    text: GgufTextPlan,
}

impl std::fmt::Debug for GgufTargetPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufTargetPlan")
            .field("configuration", self.text.config())
            .field("resolution", self.text.resolution())
            .finish_non_exhaustive()
    }
}

impl GgufTargetPlan {
    /// Projects the exact normalized input representation intent before any source access.
    /// The target and projector are selected jointly; the target-only entry rejects this intent.
    pub fn conditional_execution_plan_for_load(
        &self,
        request: &eredu_runtime::NormalizedLoadRequest,
        element: eredu_nn::TensorElementType,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
        vision: VisionPlan,
    ) -> Result<ConditionalHeaderExecutionPlan, TargetLoadError> {
        let eredu_runtime::MediaLoadRequest::Required(media) = request.media_execution() else {
            return Err(TargetLoadError::MissingMediaPolicy);
        };
        let target_request = request
            .clone()
            .with_media_execution(eredu_runtime::MediaLoadRequest::Disabled);
        let target = self.execution_plan_for_load(&target_request, element, support)?;
        Ok(target
            .with_vision(vision)?
            .with_load_processor(media.processor().clone(), media.processor_budget()))
    }

    /// Validates published metadata and derives recipes from container headers.
    pub fn prepare(checkpoint: &eredu_gguf::Checkpoint) -> Result<Self, PreparationError> {
        Ok(Self {
            text: GgufTextPlan::prepare(checkpoint)?,
        })
    }

    /// Normalizes published physical encodings and family recipes before selection.
    pub fn normalize(&self) -> Result<TargetArtifactDeclaration, PreparationError> {
        let catalog = self.text.catalog();
        let physical = catalog
            .keys()
            .into_iter()
            .map(|key| {
                let metadata = catalog.metadata(&key)?;
                let p = catalog.source_provenance(&key)?;
                let shard = p.backing_shard.ok_or_else(|| {
                    PreparationError::Contract(format!("GGUF header lacks backing shard for {key}"))
                })?;
                Ok((
                    key,
                    ReplicatedTextPhysicalSource::new(
                        p.catalog_key,
                        p.physical_tensor,
                        shard,
                        p.output,
                        p.source_encoding,
                        metadata.encoded_byte_len,
                    )
                    .map_err(|e| PreparationError::Contract(e.to_string()))?,
                ))
            })
            .collect::<Result<_, PreparationError>>()?;
        let mut static_recipes = self.text.parameter_recipes(ParameterScope::Static)?;
        if self.text.config().tied_embeddings {
            static_recipes.retain(|name, _| !name.starts_with("lm_head."));
        }
        let mut unit_recipes = Vec::new();
        for layer in 0..self.text.config().layers.len() {
            if self.text.config().ngram.layers.contains(&layer) {
                unit_recipes.push(
                    self.text
                        .parameter_recipes(ParameterScope::Lexical(layer))?,
                );
            }
            unit_recipes.push(self.text.parameter_recipes(ParameterScope::Target(layer))?);
        }
        let formats = ParameterFormats(self.text.formats.clone(), BTreeMap::new());
        let banks = self.expert_banks()?;
        let metadata = catalog
            .keys()
            .into_iter()
            .map(|key| Ok((key.clone(), catalog.metadata(&key)?)))
            .collect::<Result<_, StoreError>>()?;
        let parameters = super::requirements::target_parameters(
            self.text.config(),
            &formats,
            catalog,
            &physical,
            &static_recipes,
            &unit_recipes,
            &banks,
        )?;
        let normalized = Arc::new(
            crate::artifact_preparation::NormalizedArtifactPreparation::from_parameters(
                metadata, physical, parameters,
            ),
        );
        Ok(TargetArtifactDeclaration {
            config: self.text.config().clone(),
            formats,
            normalized,
            binding_keys: catalog.keys().into_iter().collect(),
            resolution: self.text.resolution().clone(),
            gguf_source: Some(self.text.clone()),
            tables: self
                .text
                .config()
                .ngram
                .layers
                .iter()
                .map(|&layer| Ok((layer, self.text.table.normalized(layer)?)))
                .collect::<Result<_, checkpoint::NGramArtifactError>>()?,
            banks,
            static_recipes,
            unit_recipes,
            embedded_prediction: None,
            vision_reset: self
                .text
                .checkpoint()
                .metadata()
                .get("qwen4exp.ple.image_token_id")
                .map(|v| v.as_i64().and_then(|v| u32::try_from(v).ok()).ok_or(())),
        })
    }

    /// Retained exact layouts, mappings and resolution for the ordinary source factory.
    pub fn text_plan(&self) -> &GgufTextPlan {
        &self.text
    }

    /// Derives executable geometry without opening a weight store or reading payloads.
    pub fn target_spec(&self, limits: TargetLimits) -> Result<TargetSpec, PreparationError> {
        self.target_spec_selected(limits, None)
    }
    pub(super) fn target_spec_selected(
        &self,
        limits: TargetLimits,
        selected: Option<&SelectedRoutedTextRealization>,
    ) -> Result<TargetSpec, PreparationError> {
        let config = self.text.config();
        limits.validate_for_config(config)?;
        let formats = ParameterFormats(self.text.formats.clone(), BTreeMap::new());
        let mut rows = BTreeMap::new();
        let mut ordinal = 0;
        for layer in 0..config.layers.len() {
            if config.ngram.layers.contains(&layer) {
                rows.insert(
                    layer,
                    self.text
                        .table
                        .lookup_spec(layer, rows.len() + 1, ordinal, limits.element)?,
                );
                ordinal += 1;
            }
            ordinal += 1;
        }
        Ok(super::execution::target_spec_from_formats(
            config.clone(),
            &rows,
            limits,
            &formats,
            selected,
        )?)
    }

    /// Compact row bank descriptors with exact encoding and bounded request geometry.
    pub fn row_descriptors(
        &self,
        limits: TargetLimits,
        row_limits: RowLookupLimits,
        policy: ResidencyPolicy,
    ) -> Result<eredu_runtime::RowLookupDescriptors, PreparationError> {
        let spec = self.target_spec(limits)?;
        validate_row_requests(limits, row_limits)?;
        let entries = spec
            .units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                let UnitSpec::Lexical { layer, spec } = unit else {
                    return None;
                };
                Some(self.text.table.row_descriptor(
                    *layer,
                    spec.embedding.lookup_spec().bank,
                    ordinal,
                    limits.element,
                    row_limits,
                    policy,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
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
    ) -> Result<TargetPreparationPlan, TargetLoadError> {
        self.normalize()?
            .execution_plan_for_load(request, element, support)
    }

    /// Retains a normalized TP/PP request through source-free mechanism selection.
    pub fn partition_execution_plan_for_load(
        &self,
        request: &eredu_runtime::NormalizedLoadRequest,
        element: eredu_nn::TensorElementType,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
    ) -> Result<TargetPartitionExecutionPlan, TargetLoadError> {
        let partition = super::partition_selection::load_partition_request(request)?;
        Ok(self
            .execution_plan_for_load(request, element, support)?
            .partition(partition)?)
    }

    /// Authors all target requirements against the exact source-free GGUF catalog.
    pub fn execution_plan(
        &self,
        limits: TargetLimits,
        streams: Vec<AppendStreamBinding>,
        row_admission: SelectedRowLookupPlans,
    ) -> Result<TargetPreparationPlan, PreparationError> {
        self.normalize()?
            .execution_plan(limits, streams, row_admission)
    }

    /// Pins the ordinary factory's source to every admitted physical header and
    /// provenance declaration. No source is reopened and no payload is acquired.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        limits: TargetLimits,
    ) -> Result<PreparedTarget, PreparationError> {
        self.normalize()?.bind(source, limits)
    }

    fn expert_banks(&self) -> Result<BTreeMap<String, Arc<PreparedExpertBank>>, PreparationError> {
        (0..self.text.config().layers.len())
            .map(|layer| {
                let root = format!("model.layers.{layer}.mlp.experts");
                let recipes = self
                    .text
                    .expert_bank_recipes(layer)?
                    .into_iter()
                    .map(|(name, recipe)| (name[root.len() + 1..].to_owned(), recipe))
                    .collect();
                let bank = PreparedExpertBank::new(
                    self.text.catalog(),
                    recipes,
                    self.text.config().experts.count as usize,
                )?;
                Ok((root, Arc::new(bank)))
            })
            .collect()
    }
}
