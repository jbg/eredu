//! Canonical target declarations and one source-free execution preparation path.
use super::*;
use crate::qwen4_exp::mtp::{PredictionLimits, PredictionSpec};
use crate::routed_text::{
    RoutedTextRequirements, RoutedTextSelectionRequest, SelectedRoutedTextRealization,
};
use eredu_checkpoint::recipe::RecipeCatalog;
use eredu_checkpoint::validation::ResolvedCheckpointPlan;
use eredu_runtime::{AppendStreamBinding, BackendMechanismCapabilities, SelectedRowLookupPlans};

use checkpoint::table::TableDeclaration;

/// Family declarations produced by format admission, without execution policy.
#[derive(Clone)]
pub struct TargetArtifactDeclaration {
    pub(super) config: Config,
    pub(super) formats: ParameterFormats,
    pub(super) normalized: Arc<crate::artifact_preparation::NormalizedArtifactPreparation>,
    pub(super) binding_keys: BTreeSet<String>,
    pub(super) resolution: ResolvedCheckpointPlan,
    pub(super) gguf_source: Option<checkpoint::gguf_text::GgufTextPlan>,
    pub(super) tables: BTreeMap<usize, TableDeclaration>,
    pub(super) banks: BTreeMap<String, Arc<PreparedExpertBank>>,
    pub(super) static_recipes: BTreeMap<String, DerivedWeightRecipe>,
    pub(super) unit_recipes: Vec<BTreeMap<String, DerivedWeightRecipe>>,
    pub(super) embedded_prediction: Option<SafetensorsPredictionPlan>,
    pub(super) vision_reset: Option<Result<u32, ()>>,
}
impl RecipeCatalog for TargetArtifactDeclaration {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        self.normalized.catalog().tensor_metadata(key)
    }
}
impl TargetArtifactDeclaration {
    /// Shared immutable artifact preparation contract.
    pub fn normalized(&self) -> &crate::artifact_preparation::NormalizedArtifactPreparation {
        &self.normalized
    }
    pub(crate) fn with_sources(
        mut self,
        sources: Arc<crate::artifact_preparation::ArtifactSourceDeclarations>,
    ) -> Self {
        self.normalized = Arc::new(self.normalized.as_ref().clone().with_sources(sources));
        self
    }
    /// Family configuration normalized independently of its container.
    pub fn configuration(&self) -> &Config {
        &self.config
    }
    pub(crate) fn prediction_source_keys(&self) -> Option<&BTreeSet<String>> {
        self.embedded_prediction
            .as_ref()
            .map(|h| h.resolution().source_keys())
    }
    /// Whether admission declares a prediction role in the primary artifact.
    pub fn has_embedded_prediction(&self) -> bool {
        self.embedded_prediction.is_some()
    }
    /// Prediction geometry from the retained embedded role.
    pub fn prediction_spec(
        &self,
        limits: PredictionLimits,
    ) -> Result<PredictionSpec, PreparationError> {
        self.embedded_prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?
            .prediction_spec(limits)
    }
    /// Exact source resolution retained before selection.
    pub fn resolution(&self) -> &ResolvedCheckpointPlan {
        &self.resolution
    }
    /// Physical GGUF opening declaration, when a GGUF container is admitted.
    pub fn gguf_source(&self) -> Option<&checkpoint::gguf_text::GgufTextPlan> {
        self.gguf_source.as_ref()
    }
    fn target_spec(
        &self,
        limits: TargetLimits,
        selected: Option<&SelectedRoutedTextRealization>,
    ) -> Result<TargetSpec, PreparationError> {
        limits.validate_for_config(&self.config)?;
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
            &self.formats,
            selected,
        )?)
    }
    pub(super) fn execution_plan(
        self,
        limits: TargetLimits,
        streams: Vec<AppendStreamBinding>,
        rows: SelectedRowLookupPlans,
    ) -> Result<TargetPreparationPlan, PreparationError> {
        let spec = self.target_spec(limits, None)?;
        let mut expected = Vec::new();
        for (ordinal, unit) in spec.units.iter().enumerate() {
            if let UnitSpec::Lexical { layer, spec } = unit {
                let descriptor = rows
                    .descriptors()
                    .entries()
                    .get(&spec.embedding.lookup_spec().parameter)
                    .ok_or_else(|| {
                        PreparationError::Contract("missing admitted lexical row bank".into())
                    })?;
                validate_row_requests(limits, descriptor.limits())?;
                expected.push(self.tables[layer].row_descriptor(
                    expected.len() + 1,
                    ordinal,
                    limits.element,
                    descriptor.limits(),
                    descriptor.range().policy(),
                )?);
            }
        }
        if &eredu_runtime::RowLookupDescriptors::new(expected, spec.units.len())?
            != rows.descriptors()
        {
            return Err(PreparationError::Contract(
                "selected row declarations differ from target headers".into(),
            ));
        }
        let requirements = super::requirements::target_requirements(
            &spec,
            &self.formats,
            &self,
            &self.normalized.parameters,
            &self.static_recipes,
            &self.banks,
            streams,
            rows,
        )?;
        Ok(TargetPreparationPlan {
            artifact: self,
            limits,
            requirements,
            capability: crate::capability::qwen4_exp_target(&spec)?,
            load_selection: None,
            prediction: None,
        })
    }
    pub(crate) fn execution_plan_for_load(
        self,
        request: &eredu_runtime::NormalizedLoadRequest,
        element: eredu_nn::TensorElementType,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
    ) -> Result<TargetPreparationPlan, TargetLoadError> {
        let projection = if request.has_parallel_execution() {
            super::load_policy::TargetLoadProjection::partitioned(request, &self.config, element)?
        } else {
            super::load_policy::TargetLoadProjection::new(request, &self.config, element)?
        };
        let spec = self.target_spec(projection.limits, None)?;
        let mut entries = Vec::new();
        validate_row_requests(projection.limits, projection.row_limits())?;
        for (ordinal, unit) in spec.units.iter().enumerate() {
            if let UnitSpec::Lexical { layer, .. } = unit {
                entries.push(
                    self.tables[layer]
                        .row_descriptor(
                            entries.len() + 1,
                            ordinal,
                            element,
                            projection.row_limits(),
                            ResidencyPolicy::Cacheable,
                        )
                        .map_err(PreparationError::from)?,
                );
            }
        }
        let rows = projection.select_rows(
            eredu_runtime::RowLookupDescriptors::new(entries, spec.units.len())
                .map_err(PreparationError::from)?,
            support,
        )?;
        let mut plan = self.execution_plan(projection.limits, projection.streams(&spec), rows)?;
        plan.load_selection = Some(projection.selection);
        Ok(plan)
    }
    pub(super) fn bind(
        &self,
        source: SharedCheckpointSource,
        limits: TargetLimits,
    ) -> Result<PreparedTarget, PreparationError> {
        let keys = source.source_keys().into_iter().collect::<BTreeSet<_>>();
        if keys != self.binding_keys && keys != self.normalized.source_keys().into_iter().collect()
        {
            return Err(PreparationError::Contract(
                "source set differs from admitted target headers".into(),
            ));
        }
        let source = retain(source, keys)?;
        for key in source.source_keys() {
            let expected = self.normalized.catalog().tensor_metadata(&key)?;
            let actual = source.source_metadata(&key)?;
            if actual.logical_shape != expected.logical_shape
                || actual.stored_dtype != expected.stored_dtype
                || (self.binding_keys.contains(&key)
                    && (actual.physical_shape != expected.physical_shape
                        || actual.encoded_byte_len != expected.encoded_byte_len))
            {
                return Err(PreparationError::Contract(format!(
                    "source header changed for {key}"
                )));
            }
        }
        for (key, expected) in self.normalized.physical_sources() {
            let actual = crate::replicated_text::exact_physical_source(source.as_ref(), key)
                .map_err(|e| PreparationError::Contract(e.to_string()))?;
            if &actual != expected {
                return Err(PreparationError::Contract(format!(
                    "physical source changed for {key}"
                )));
            }
        }
        let source = retain(source, self.binding_keys.clone())?;
        let spec = self.target_spec(limits, None)?;
        let mut tables = BTreeMap::new();
        for (ordinal, unit) in spec.units.iter().enumerate() {
            if let UnitSpec::Lexical { layer, spec } = unit {
                tables.insert(
                    *layer,
                    self.tables[layer].bind(
                        source.clone(),
                        spec.embedding.lookup_spec().bank,
                        ordinal,
                        limits.element,
                    )?,
                );
            }
        }
        BoundTargetSpec::new(
            spec.clone(),
            tables
                .iter()
                .map(|(&layer, t)| (layer, t.hash.clone()))
                .collect(),
        )?;
        Ok(PreparedTarget {
            prediction_recipes: self.embedded_prediction.as_ref().map(|p| p.recipes()),
            artifact: source.clone(),
            formats: self.formats.clone(),
            expert_banks: self.banks.clone(),
            spec,
            tables,
            static_parameters: PreparedParameters::new(
                source.clone(),
                self.static_recipes.clone(),
            )?,
            units: self
                .unit_recipes
                .iter()
                .map(|recipes| PreparedParameters::new(source.clone(), recipes.clone()))
                .collect::<Result<_, _>>()?,
        })
    }
}

/// One cold target preparation, independent of its admitted container.
#[derive(Clone)]
pub struct TargetPreparationPlan {
    artifact: TargetArtifactDeclaration,
    limits: TargetLimits,
    requirements: RoutedTextRequirements,
    capability: crate::capability::CapabilityEstimate,
    pub(super) load_selection: Option<RoutedTextSelectionRequest>,
    prediction: Option<(
        SafetensorsPredictionPlan,
        bool,
        PredictionSpec,
        ParameterFormats,
        eredu_runtime::StateRealizationRequirements,
    )>,
}
/// Mechanisms retained before binding target and optional prediction roles.
#[derive(Clone)]
pub struct SelectedTargetPreparation {
    plan: TargetPreparationPlan,
    selected: SelectedRoutedTextRealization,
    prediction_state: Option<eredu_runtime::SelectedStateRealization>,
}
impl TargetPreparationPlan {
    /// Immutable source authority used by physical binding.
    pub fn artifact(&self) -> &TargetArtifactDeclaration {
        &self.artifact
    }
    pub(super) fn vision_reset_token_id(&self) -> Result<Option<u32>, PreparationError> {
        self.artifact
            .vision_reset
            .transpose()
            .map_err(|_| PreparationError::VisionMismatch {
                field: "media token IDs",
            })
    }
    pub(super) fn selected_target_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<TargetSpec, PreparationError> {
        self.artifact.target_spec(self.limits, Some(selected))
    }
    pub(super) fn target_spec(&self) -> Result<TargetSpec, PreparationError> {
        let mut spec = self.artifact.target_spec(self.limits, None)?;
        if let Some((header, _, _, _, _)) = &self.prediction {
            spec.config.prediction = header.configuration().prediction.clone();
        }
        Ok(spec)
    }
    /// Adds the admitted embedded prediction role.
    pub fn with_prediction(
        self,
        limits: PredictionLimits,
        streams: Vec<AppendStreamBinding>,
    ) -> Result<Self, PreparationError> {
        if self.prediction.is_some() {
            return Err(PreparationError::Contract(
                "prediction is already prepared".into(),
            ));
        }
        let header = self
            .artifact
            .embedded_prediction
            .clone()
            .ok_or(PreparationError::MissingPrediction)?;
        self.add_prediction(header, false, limits, streams)
    }
    /// Adds a compatible separately admitted prediction role.
    pub fn with_prediction_source(
        self,
        header: SafetensorsPredictionPlan,
        limits: PredictionLimits,
        streams: Vec<AppendStreamBinding>,
    ) -> Result<Self, PreparationError> {
        if self.prediction.is_some() {
            return Err(PreparationError::PredictionAlreadyPresent);
        }
        header.validate_target(&self.artifact.config)?;
        self.add_prediction(header, true, limits, streams)
    }
    fn add_prediction(
        mut self,
        companion: SafetensorsPredictionPlan,
        separate: bool,
        limits: PredictionLimits,
        streams: Vec<AppendStreamBinding>,
    ) -> Result<Self, PreparationError> {
        let (spec, formats, banks) = companion.parts(limits)?;
        let (shared, units) = companion.recipes();
        let catalog = PredictionCatalog {
            target: &self.artifact,
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
        self.prediction = Some((companion, separate, spec, formats, state));
        Ok(self)
    }
    /// Separate prediction source authority, if required by this preparation.
    pub fn prediction_header(&self) -> Option<&SafetensorsPredictionPlan> {
        self.prediction
            .as_ref()
            .and_then(|(h, separate, _, _, _)| separate.then_some(h))
    }
    /// Independent mutable prediction state selected during cold preparation.
    pub fn prediction_state_requirements(
        &self,
    ) -> Option<&eredu_runtime::StateRealizationRequirements> {
        self.prediction.as_ref().map(|(_, _, _, _, state)| state)
    }
    /// Header-authored prediction construction before weight transform selection.
    pub fn prediction_spec(&self) -> Option<&PredictionSpec> {
        self.prediction.as_ref().map(|(_, _, spec, _, _)| spec)
    }

    pub(super) fn selected_prediction_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<PredictionSpec, PreparationError> {
        let (header, _, spec, formats, _) = self
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
        super::conditional_header::ConditionalHeaderExecutionPlan::new(self, vision)
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
    ) -> Result<SelectedTargetPreparation, TargetSelectionError> {
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
        Ok(SelectedTargetPreparation {
            plan: self,
            selected,
            prediction_state,
        })
    }

    /// Binds exact role sources before acquiring bounded family literals.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        prediction_source: Option<SharedCheckpointSource>,
    ) -> Result<TargetExecutionPlan, PreparationError> {
        let separate = self
            .prediction
            .as_ref()
            .is_some_and(|(_, separate, _, _, _)| *separate);
        if separate != prediction_source.is_some() {
            return Err(PreparationError::Contract(
                "prediction source presence differs from admitted header".into(),
            ));
        }
        // Validate the independent role before the target acquires any literals.
        let bound_prediction = match (&self.prediction, prediction_source) {
            (Some((header, _, _, _, _)), Some(source)) => Some(header.clone().bind(source)?),
            _ => None,
        };
        let mut target = self.artifact.bind(source, self.limits)?;
        if let Some(prediction) = bound_prediction {
            target = target.with_prediction_source(prediction)?;
        }
        let prediction = self
            .prediction
            .map(|(_, _, spec, _, state)| {
                Ok::<_, PreparationError>((target.prediction(spec.limits)?, state))
            })
            .transpose()?;
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
impl RecipeCatalog for TargetPreparationPlan {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        if let Some((header, _, _, _, _)) = &self.prediction {
            if header.physical_sources().contains_key(key) {
                return header.tensor_metadata(key);
            }
        }
        self.artifact.tensor_metadata(key)
    }
}
impl SelectedTargetPreparation {
    /// Exact normalized artifact retained with mechanism selection.
    pub fn artifact(&self) -> &TargetArtifactDeclaration {
        &self.plan.artifact
    }
    /// Conservative selected recipe workspace from retained metadata alone.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        crate::SelectedExecution::routed(self.selected.clone())
            .parameter_materialization_workspace(&self.plan, None, mechanisms)
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
pub(super) fn validate_row_requests(
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
