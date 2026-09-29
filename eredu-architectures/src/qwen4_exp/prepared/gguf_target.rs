//! Header-only GGUF target requirements and exact retained-source binding.
use super::*;
use crate::qwen4_exp::mtp::{PredictionLimits, PredictionSpec};
use crate::routed_text::{
    RoutedTextRequirements, RoutedTextSelectionRequest, SelectedRoutedTextRealization,
};
use checkpoint::gguf_text::GgufTextPlan;
use eredu_checkpoint::recipe::RecipeCatalog;
use eredu_runtime::{
    AppendStreamBinding, BackendMechanismCapabilities, ReplicatedTextPhysicalSource,
    SelectedRowLookupPlans,
};

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

/// Complete source-free ordinary, routed, state and row requirements.
#[derive(Clone)]
pub struct GgufTargetExecutionPlan {
    header: GgufTargetPlan,
    limits: TargetLimits,
    requirements: RoutedTextRequirements,
    capability: crate::capability::CapabilityEstimate,
    load_selection: Option<RoutedTextSelectionRequest>,
    prediction: Option<(
        SafetensorsPredictionPlan,
        PredictionSpec,
        ParameterFormats,
        eredu_runtime::StateRealizationRequirements,
    )>,
}

/// Mechanisms selected before a readable artifact source is supplied.
#[derive(Clone)]
pub struct SelectedGgufTargetExecution {
    plan: GgufTargetExecutionPlan,
    selected: SelectedRoutedTextRealization,
    prediction_state: Option<eredu_runtime::SelectedStateRealization>,
}

impl GgufTargetExecutionPlan {
    /// Exact target source declaration retained before combined role selection.
    pub fn header_plan(&self) -> &GgufTargetPlan {
        &self.header
    }
    pub(super) fn vision_reset_token_id(&self) -> Result<Option<u32>, PreparationError> {
        self.header
            .text
            .checkpoint()
            .metadata()
            .get("qwen4exp.ple.image_token_id")
            .map(|value| {
                value
                    .as_i64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or(PreparationError::VisionMismatch {
                        field: "media token IDs",
                    })
            })
            .transpose()
    }
    pub(super) fn selected_target_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<TargetSpec, PreparationError> {
        self.header
            .target_spec_selected(self.limits, Some(selected))
    }
    pub(super) fn target_spec(&self) -> Result<TargetSpec, PreparationError> {
        let mut spec = self.header.target_spec(self.limits)?;
        if let Some((companion, _, _, _)) = &self.prediction {
            spec.config.prediction = companion.configuration().prediction.clone();
        }
        Ok(spec)
    }

    pub(super) fn catalog(&self) -> &dyn eredu_checkpoint::recipe::RecipeCatalog {
        self.header.text.catalog()
    }

    /// Adds a separately admitted predictor while retaining the GGUF vocabulary.
    pub fn with_prediction(
        mut self,
        companion: SafetensorsPredictionPlan,
        limits: PredictionLimits,
        streams: Vec<AppendStreamBinding>,
    ) -> Result<Self, PreparationError> {
        if self.prediction.is_some() {
            return Err(PreparationError::PredictionAlreadyPresent);
        }
        companion.validate_target(self.header.text.config())?;
        let (spec, formats, banks) = companion.parts(limits)?;
        let keys = companion
            .resolution()
            .source_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut shared = recipes::parameter_recipes(
            keys.clone(),
            companion.configuration(),
            ParameterScope::Static,
        )
        .map_err(PreparationError::Contract)?;
        shared.retain(|name, _| name.starts_with("mtp."));
        let units = (0..spec.units.len())
            .map(|depth| {
                recipes::parameter_recipes(
                    keys.clone(),
                    companion.configuration(),
                    ParameterScope::Prediction(depth),
                )
                .map_err(PreparationError::Contract)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let catalog = PredictionCatalog {
            target: self.header.text.catalog(),
            prediction: &companion,
        };
        let (requirements, state) = super::prediction_plan::prediction_requirements(
            &spec,
            &self.target_spec()?,
            &formats,
            &catalog,
            companion.physical_sources(),
            &shared,
            &units,
            &banks,
            &self.requirements,
            streams,
        )?;
        self.capability = crate::capability::qwen4_exp_prediction(&self.target_spec()?, &spec)?;
        self.requirements = requirements;
        self.prediction = Some((companion, spec, formats, state));
        Ok(self)
    }
    /// Retained companion headers used by source preparation.
    pub fn prediction_header(&self) -> Option<&SafetensorsPredictionPlan> {
        self.prediction.as_ref().map(|(header, _, _, _)| header)
    }
    /// Independent mutable prediction state selected during cold preparation.
    pub fn prediction_state_requirements(
        &self,
    ) -> Option<&eredu_runtime::StateRealizationRequirements> {
        self.prediction.as_ref().map(|(_, _, _, state)| state)
    }
    /// Header-authored prediction construction before weight transform selection.
    pub fn prediction_spec(&self) -> Option<&PredictionSpec> {
        self.prediction.as_ref().map(|(_, spec, _, _)| spec)
    }

    pub(super) fn selected_prediction_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<PredictionSpec, PreparationError> {
        let (header, spec, formats, _) = self
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        super::prediction_plan::selected_prediction_spec(
            header.configuration(),
            spec,
            formats,
            selected,
        )
    }

    /// Adds a source-free vision role before selecting the combined graph.
    pub fn with_vision(
        self,
        vision: VisionPlan,
    ) -> Result<ConditionalHeaderExecutionPlan, PreparationError> {
        super::conditional_header::ConditionalHeaderExecutionPlan::new(
            super::conditional_header::TargetHeader::Gguf(Box::new(self)),
            vision,
        )
    }

    /// Exact normalized load policy retained through cold selection and binding.
    pub fn load_selection_request(&self) -> Option<&RoutedTextSelectionRequest> {
        self.load_selection.as_ref()
    }

    /// Header-derived context and state estimates.
    pub fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability
    }

    /// Complete geometry and exact physical declarations, without source handles.
    pub fn requirements(&self) -> &RoutedTextRequirements {
        &self.requirements
    }

    /// Selects complete target mechanisms using only retained headers.
    pub fn select(
        self,
        request: &RoutedTextSelectionRequest,
        mechanisms: &BackendMechanismCapabilities,
        prediction_mechanisms: Option<&eredu_runtime::StateMechanismCapabilities>,
    ) -> Result<SelectedGgufTargetExecution, TargetSelectionError> {
        if self
            .load_selection
            .as_ref()
            .is_some_and(|retained| retained != request)
        {
            return Err(TargetSelectionError::LoadRequestMismatch);
        }
        let prediction_state = match (self.prediction_state_requirements(), prediction_mechanisms) {
            (Some(state), Some(mechanisms)) => Some(eredu_runtime::select_state_realization(
                state,
                request.text(),
                mechanisms,
            )?),
            (None, None) => None,
            _ => return Err(TargetSelectionError::PredictionStatePresence),
        };
        let selected = crate::routed_text::select_routed_text_realization(
            &self.requirements,
            request,
            mechanisms,
        )?;
        super::execution::selected_stream_allowances(&selected, prediction_state.as_ref())?;
        Ok(SelectedGgufTargetExecution {
            plan: self,
            selected,
            prediction_state,
        })
    }

    /// Checks and pins the complete exact source set. Every payload remains lazy.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        prediction_source: Option<SharedCheckpointSource>,
    ) -> Result<TargetExecutionPlan, PreparationError> {
        let prediction = match (self.prediction, prediction_source) {
            (Some((header, spec, _, state)), Some(source)) => {
                Some((header.bind(source)?, spec, state))
            }
            (None, None) => None,
            _ => {
                return Err(PreparationError::Contract(
                    "prediction source presence differs from admitted header".into(),
                ))
            }
        };
        let target = self.header.bind(source, self.limits)?;
        let (target, prediction) = match prediction {
            Some((source, spec, state)) => {
                let target = target.with_prediction_source(source)?;
                let prediction = target.prediction(spec.limits)?;
                (target, Some((prediction, state)))
            }
            None => (target, None),
        };
        let row_sources = if let Some(rows) = self.requirements.row_lookups() {
            target.bind_rows(rows)?
        } else {
            eredu_runtime::PreparedRowLookups::new([], target.spec.units.len())?
        };
        Ok(TargetExecutionPlan {
            target,
            prediction,
            load_selection: self.load_selection,
            requirements: self.requirements,
            row_sources,
            capability: self.capability,
        })
    }
}

impl SelectedGgufTargetExecution {
    /// Exact source-free artifact contract retained by this mechanism selection.
    pub fn header_plan(&self) -> &GgufTargetPlan {
        &self.plan.header
    }

    /// Conservative selected recipe workspace from retained headers alone.
    /// Row lookup buffers are accounted for by the separate row contracts.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        let execution = crate::SelectedExecution::routed(self.selected.clone());
        if let Some(companion) = self.plan.prediction_header() {
            execution.parameter_materialization_workspace(
                &PredictionCatalog {
                    target: self.plan.header.text.catalog(),
                    prediction: companion,
                },
                None,
                mechanisms,
            )
        } else {
            execution.parameter_materialization_workspace(
                self.plan.header.text.catalog(),
                None,
                mechanisms,
            )
        }
    }

    /// Exact retained cold mechanism selection.
    pub fn selected(&self) -> &SelectedRoutedTextRealization {
        &self.selected
    }

    /// Retained companion metadata, never a reopened artifact.
    pub fn prediction_header(&self) -> Option<&SafetensorsPredictionPlan> {
        self.plan.prediction_header()
    }
    /// Exact independent state realization retained from joint cold selection.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction_state.as_ref()
    }
    /// Prediction geometry after selected parameter transformations.
    pub fn prediction_spec(&self) -> Result<PredictionSpec, PreparationError> {
        self.plan.selected_prediction_spec(&self.selected)
    }
    pub(crate) fn prediction_descriptor(
        &self,
    ) -> Result<eredu_core::ArchitectureDescriptor, PreparationError> {
        let target = self.plan.target_spec()?;
        let prediction = self.prediction_spec()?;
        super::graph::Graph::new(&target, &prediction)
            .with_prediction_observations(&target, &prediction)
    }

    /// Consumes the retained selection without requerying mechanism support.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        prediction_source: Option<SharedCheckpointSource>,
    ) -> Result<SelectedTargetExecution, PreparationError> {
        self.plan
            .bind(source, prediction_source)?
            .bind_selected(self.selected, self.prediction_state)
            .map_err(|error| PreparationError::Contract(error.to_string()))
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
    ) -> Result<GgufTargetExecutionPlan, TargetLoadError> {
        let projection = if request.has_parallel_execution() {
            super::load_policy::TargetLoadProjection::partitioned(
                request,
                self.text.config(),
                element,
            )?
        } else {
            super::load_policy::TargetLoadProjection::new(request, self.text.config(), element)?
        };
        self.execution_plan_from_projection(projection, support)
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

    fn execution_plan_from_projection(
        &self,
        projection: super::load_policy::TargetLoadProjection,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
    ) -> Result<GgufTargetExecutionPlan, TargetLoadError> {
        let spec = self.target_spec(projection.limits)?;
        let descriptors = self.row_descriptors(
            projection.limits,
            projection.row_limits(),
            ResidencyPolicy::Cacheable,
        )?;
        let rows = projection.select_rows(descriptors, support)?;
        let mut plan = self.execution_plan(projection.limits, projection.streams(&spec), rows)?;
        plan.load_selection = Some(projection.selection);
        Ok(plan)
    }

    /// Authors all target requirements against the exact source-free GGUF catalog.
    pub fn execution_plan(
        &self,
        limits: TargetLimits,
        streams: Vec<AppendStreamBinding>,
        row_admission: SelectedRowLookupPlans,
    ) -> Result<GgufTargetExecutionPlan, PreparationError> {
        let spec = self.target_spec(limits)?;
        let mut expected_rows = Vec::new();
        for (ordinal, unit) in spec.units.iter().enumerate() {
            if let UnitSpec::Lexical {
                layer,
                spec: lexical,
            } = unit
            {
                let lookup = lexical.embedding.lookup_spec();
                let descriptor = row_admission
                    .descriptors()
                    .entries()
                    .get(&lookup.parameter)
                    .ok_or_else(|| {
                        PreparationError::Contract("missing admitted lexical row bank".into())
                    })?;
                validate_row_requests(limits, descriptor.limits())?;
                expected_rows.push(self.text.table.row_descriptor(
                    *layer,
                    lookup.bank,
                    ordinal,
                    limits.element,
                    descriptor.limits(),
                    descriptor.range().policy(),
                )?);
            }
        }
        let expected_rows =
            eredu_runtime::RowLookupDescriptors::new(expected_rows, spec.units.len())?;
        if &expected_rows != row_admission.descriptors() {
            return Err(PreparationError::Contract(
                "selected row declarations differ from target headers".into(),
            ));
        }
        let physical = self
            .text
            .catalog()
            .keys()
            .into_iter()
            .map(|key| {
                let metadata = self.text.catalog().metadata(&key)?;
                let provenance = self.text.catalog().source_provenance(&key)?;
                let shard = provenance.backing_shard.ok_or_else(|| {
                    PreparationError::Contract(format!("GGUF header lacks backing shard for {key}"))
                })?;
                let physical = ReplicatedTextPhysicalSource::new(
                    provenance.catalog_key,
                    provenance.physical_tensor,
                    shard,
                    provenance.output,
                    provenance.source_encoding,
                    metadata.encoded_byte_len,
                )
                .map_err(|error| PreparationError::Contract(error.to_string()))?;
                Ok((key, physical))
            })
            .collect::<Result<BTreeMap<_, _>, PreparationError>>()?;
        let formats = ParameterFormats(self.text.formats.clone(), BTreeMap::new());
        let banks = self.expert_banks()?;
        let (static_recipes, unit_recipes) = self.parameter_recipes(&spec)?;
        let requirements = super::requirements::target_requirements(
            &spec,
            &formats,
            self.text.catalog(),
            &physical,
            &static_recipes,
            &unit_recipes,
            &banks,
            streams,
            row_admission,
        )?;
        Ok(GgufTargetExecutionPlan {
            header: self.clone(),
            limits,
            requirements,
            capability: crate::capability::qwen4_exp_target(&spec)?,
            load_selection: None,
            prediction: None,
        })
    }

    /// Pins the ordinary factory's source to every admitted physical header and
    /// provenance declaration. No source is reopened and no payload is acquired.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        limits: TargetLimits,
    ) -> Result<PreparedTarget, PreparationError> {
        let spec = self.target_spec(limits)?;
        let expert_banks = self.expert_banks()?;
        let (static_recipes, unit_recipes) = self.parameter_recipes(&spec)?;
        let weights = self.text.bind(source)?;
        let artifact = weights.source.clone();
        let formats = ParameterFormats(weights.plan.formats.clone(), BTreeMap::new());
        let tables = spec
            .units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                let UnitSpec::Lexical { layer, spec } = unit else {
                    return None;
                };
                Some(
                    weights
                        .plan
                        .table
                        .bind(
                            artifact.clone(),
                            *layer,
                            spec.embedding.lookup_spec().bank,
                            ordinal,
                            limits.element,
                        )
                        .map(|table| (*layer, table)),
                )
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        BoundTargetSpec::new(
            spec.clone(),
            tables
                .iter()
                .map(|(&layer, table)| (layer, table.hash.clone()))
                .collect(),
        )?;
        let static_parameters = PreparedParameters::new(artifact.clone(), static_recipes)?;
        let units = unit_recipes
            .into_iter()
            .map(|recipes| PreparedParameters::new(artifact.clone(), recipes))
            .collect::<Result<_, _>>()?;
        Ok(PreparedTarget {
            artifact,
            formats,
            expert_banks,
            spec,
            tables,
            static_parameters,
            units,
        })
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

    fn parameter_recipes(
        &self,
        spec: &TargetSpec,
    ) -> Result<
        (
            BTreeMap<String, DerivedWeightRecipe>,
            Vec<BTreeMap<String, DerivedWeightRecipe>>,
        ),
        PreparationError,
    > {
        let mut static_recipes = self.text.parameter_recipes(ParameterScope::Static)?;
        if self.text.config().tied_embeddings {
            static_recipes.retain(|name, _| !name.starts_with("lm_head."));
        }
        let units = spec
            .units
            .iter()
            .map(|unit| {
                self.text.parameter_recipes(match unit {
                    UnitSpec::Lexical { layer, .. } => ParameterScope::Lexical(*layer),
                    UnitSpec::Decoder { layer, .. } => ParameterScope::Target(*layer),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok((static_recipes, units))
    }
}

fn validate_row_requests(
    limits: TargetLimits,
    rows: RowLookupLimits,
) -> Result<(), PreparationError> {
    if rows.requests < limits.lookup_rows {
        return Err(eredu_runtime::RowLookupError::Budget {
            resource: "target lookup requests",
            required: limits.lookup_rows as u64,
            limit: rows.requests as u64,
        }
        .into());
    }
    Ok(())
}

/// The two source namespaces stay distinct; only metadata is composed here.
struct PredictionCatalog<'a> {
    target: &'a dyn RecipeCatalog,
    prediction: &'a SafetensorsPredictionPlan,
}
impl RecipeCatalog for PredictionCatalog<'_> {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        if self.prediction.physical_sources().contains_key(key) {
            self.prediction.tensor_metadata(key)
        } else {
            self.target.tensor_metadata(key)
        }
    }
}
