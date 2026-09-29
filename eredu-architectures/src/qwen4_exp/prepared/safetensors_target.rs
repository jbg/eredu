//! Complete header admission followed by exact source binding and bounded literals.
use super::*;
use crate::qwen4_exp::mtp::{PredictionLimits, PredictionSpec};
use crate::routed_text::{
    RoutedTextRequirements, RoutedTextSelectionRequest, SelectedRoutedTextRealization,
};
use checkpoint::SafetensorsTableSourcePlan;
use eredu_checkpoint::{
    schema::SafetensorsCheckpointPlan,
    validation::{CatalogTensorMetadata, ResolvedCheckpointPlan, SafetensorsCatalog},
};
use eredu_runtime::{
    AppendStreamBinding, BackendMechanismCapabilities, ReplicatedTextPhysicalSource,
    SelectedRowLookupPlans,
};

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

/// Complete source-free ordinary, routed and row execution requirements. Physical
/// declarations are retained exactly and checked again before any literal reads.
#[derive(Clone)]
pub struct SafetensorsTargetExecutionPlan {
    header: SafetensorsTargetPlan,
    limits: TargetLimits,
    physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    requirements: RoutedTextRequirements,
    capability: crate::capability::CapabilityEstimate,
    load_selection: Option<RoutedTextSelectionRequest>,
    prediction: Option<(
        PredictionSpec,
        ParameterFormats,
        eredu_runtime::StateRealizationRequirements,
    )>,
}

/// Exact mechanism selection retained before any readable sources are supplied.
#[derive(Clone)]
pub struct SelectedSafetensorsTargetExecution {
    plan: SafetensorsTargetExecutionPlan,
    selected: SelectedRoutedTextRealization,
    prediction_state: Option<eredu_runtime::SelectedStateRealization>,
}

impl SafetensorsTargetExecutionPlan {
    /// Exact target source declaration retained before combined role selection.
    pub fn header_plan(&self) -> &SafetensorsTargetPlan {
        &self.header
    }
    pub(super) fn selected_target_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<TargetSpec, PreparationError> {
        self.header
            .target_spec_selected(self.limits, Some(selected))
    }
    pub(super) fn target_spec(&self) -> Result<TargetSpec, PreparationError> {
        self.header.target_spec(self.limits)
    }

    pub(super) fn catalog(&self) -> &dyn eredu_checkpoint::recipe::RecipeCatalog {
        &self.header.catalog
    }

    /// Adds embedded prediction parameters and independent state using admitted headers only.
    pub fn with_prediction(
        mut self,
        limits: PredictionLimits,
        streams: Vec<AppendStreamBinding>,
    ) -> Result<Self, PreparationError> {
        if self.prediction.is_some() {
            return Err(PreparationError::Contract(
                "prediction is already prepared".into(),
            ));
        }
        let (spec, formats, banks) = self.header.prediction_parts(limits)?;
        let keys = self
            .header
            .resolution
            .source_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut shared =
            recipes::parameter_recipes(keys.clone(), &self.header.config, ParameterScope::Static)
                .map_err(PreparationError::Contract)?;
        shared.retain(|name, _| name.starts_with("mtp."));
        let units = (0..spec.units.len())
            .map(|depth| {
                recipes::parameter_recipes(
                    keys.clone(),
                    &self.header.config,
                    ParameterScope::Prediction(depth),
                )
                .map_err(PreparationError::Contract)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (requirements, state) = super::prediction_plan::prediction_requirements(
            &spec,
            &self.target_spec()?,
            &formats,
            &self.header.catalog,
            &self.physical,
            &shared,
            &units,
            &banks,
            &self.requirements,
            streams,
        )?;
        self.requirements = requirements;
        self.capability = crate::capability::qwen4_exp_prediction(&self.target_spec()?, &spec)?;
        self.prediction = Some((spec, formats, state));
        Ok(self)
    }

    /// Exact independent prediction state for cold mechanism queries.
    pub fn prediction_state_requirements(
        &self,
    ) -> Option<&eredu_runtime::StateRealizationRequirements> {
        self.prediction.as_ref().map(|(_, _, state)| state)
    }

    /// Header-authored prediction geometry before parameter transform selection.
    pub fn prediction_spec(&self) -> Option<&PredictionSpec> {
        self.prediction.as_ref().map(|(spec, _, _)| spec)
    }

    pub(super) fn selected_prediction_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<PredictionSpec, PreparationError> {
        let (spec, formats, _) = self
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        super::prediction_plan::selected_prediction_spec(
            &self.header.config,
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
            super::conditional_header::TargetHeader::Safetensors(Box::new(self)),
            vision,
        )
    }

    /// Exact normalized load policy retained through cold selection and binding.
    pub fn load_selection_request(&self) -> Option<&RoutedTextSelectionRequest> {
        self.load_selection.as_ref()
    }

    /// Header-derived context and state estimates, before source binding.
    pub fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability
    }

    /// Complete geometry and exact physical declarations; contains no source handle.
    pub fn requirements(&self) -> &RoutedTextRequirements {
        &self.requirements
    }

    /// Selects complete target mechanisms before controls or scales are acquired.
    pub fn select(
        self,
        request: &RoutedTextSelectionRequest,
        mechanisms: &BackendMechanismCapabilities,
        prediction_mechanisms: Option<&eredu_runtime::StateMechanismCapabilities>,
    ) -> Result<SelectedSafetensorsTargetExecution, TargetSelectionError> {
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
        Ok(SelectedSafetensorsTargetExecution {
            plan: self,
            selected,
            prediction_state,
        })
    }

    /// Pins and checks the exact admitted physical source set before reading
    /// bounded integer controls. Ordinary, expert and table payloads remain lazy.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
    ) -> Result<TargetExecutionPlan, PreparationError> {
        let source = retain(source.clone(), source.source_keys().into_iter().collect())?;
        for (key, expected) in &self.physical {
            let actual = crate::replicated_text::exact_physical_source(source.as_ref(), key)
                .map_err(|error| PreparationError::Contract(error.to_string()))?;
            if &actual != expected {
                return Err(PreparationError::Contract(format!(
                    "physical source changed for {key}"
                )));
            }
        }
        let target = self.header.bind(source, self.limits)?;
        let row_sources = if let Some(rows) = self.requirements.row_lookups() {
            target.bind_rows(rows)?
        } else {
            eredu_runtime::PreparedRowLookups::new([], target.spec.units.len())?
        };
        let prediction = self
            .prediction
            .map(|(spec, _, state)| {
                Ok::<_, PreparationError>((target.prediction(spec.limits)?, state))
            })
            .transpose()?;
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

impl SelectedSafetensorsTargetExecution {
    /// Exact source-free artifact contract retained by this mechanism selection.
    pub fn header_plan(&self) -> &SafetensorsTargetPlan {
        &self.plan.header
    }

    /// Conservative selected recipe workspace from retained headers alone.
    /// Row lookup buffers are accounted for by the separate row contracts.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        crate::SelectedExecution::routed(self.selected.clone()).parameter_materialization_workspace(
            &self.plan.header.catalog,
            None,
            mechanisms,
        )
    }

    /// Retained cold selection, available for inspection without payload reads.
    pub fn selected(&self) -> &SelectedRoutedTextRealization {
        &self.selected
    }

    /// Exact independent prediction state retained by cold selection.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction_state.as_ref()
    }

    /// Prediction construction after selected parameter transformations.
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

    /// Binds the retained selection without requerying or reselecting mechanisms.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
    ) -> Result<SelectedTargetExecution, PreparationError> {
        self.plan
            .bind(source)?
            .bind_selected(self.selected, self.prediction_state)
            .map_err(|error| PreparationError::Contract(error.to_string()))
    }
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
    ) -> Result<SafetensorsTargetExecutionPlan, TargetLoadError> {
        let projection = if request.has_parallel_execution() {
            super::load_policy::TargetLoadProjection::partitioned(request, &self.config, element)?
        } else {
            super::load_policy::TargetLoadProjection::new(request, &self.config, element)?
        };
        self.execution_plan_from_projection(projection, support, physical)
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

    fn execution_plan_from_projection(
        &self,
        projection: super::load_policy::TargetLoadProjection,
        support: &impl eredu_runtime::RowLookupMechanismSupport,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<SafetensorsTargetExecutionPlan, TargetLoadError> {
        let spec = self.target_spec(projection.limits)?;
        let descriptors = self.row_descriptors(
            projection.limits,
            projection.row_limits(),
            ResidencyPolicy::Cacheable,
        )?;
        let rows = projection.select_rows(descriptors, support)?;
        let mut plan =
            self.execution_plan(projection.limits, projection.streams(&spec), rows, physical)?;
        plan.load_selection = Some(projection.selection);
        Ok(plan)
    }

    /// Authors complete target requirements using headers and exact physical
    /// provenance supplied by artifact admission. No readable source is accepted.
    pub fn execution_plan(
        &self,
        limits: TargetLimits,
        streams: Vec<AppendStreamBinding>,
        row_admission: SelectedRowLookupPlans,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<SafetensorsTargetExecutionPlan, PreparationError> {
        if physical.keys().collect::<BTreeSet<_>>()
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
        let spec = self.target_spec(limits)?;
        // Check every selected compact bank against this exact header plan, not
        // just its owner geometry, before authoring ordinary/expert requirements.
        let mut expected_rows = Vec::new();
        for (ordinal, unit) in spec.units.iter().enumerate() {
            if let UnitSpec::Lexical {
                layer,
                spec: lexical,
            } = unit
            {
                let descriptor = row_admission
                    .descriptors()
                    .entries()
                    .get(&lexical.embedding.lookup_spec().parameter)
                    .ok_or_else(|| {
                        PreparationError::Contract("missing admitted lexical row bank".into())
                    })?;
                if descriptor.limits().requests < limits.lookup_rows {
                    return Err(eredu_runtime::RowLookupError::Budget {
                        resource: "target lookup requests",
                        required: limits.lookup_rows as u64,
                        limit: descriptor.limits().requests as u64,
                    }
                    .into());
                }
                expected_rows.push(self.tables[layer].row_descriptor(
                    expected_rows.len() + 1,
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
        let keys = self
            .resolution
            .source_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let roots =
            (0..self.config.layers.len()).map(|layer| format!("model.layers.{layer}.mlp.experts"));
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
        let (static_recipes, unit_recipes) = target_parameter_recipes(&keys, &self.config, &spec)?;
        let requirements = super::requirements::target_requirements(
            &spec,
            &formats,
            &self.catalog,
            &physical,
            &static_recipes,
            &unit_recipes,
            &banks,
            streams,
            row_admission,
        )?;
        Ok(SafetensorsTargetExecutionPlan {
            header: self.clone(),
            limits,
            physical,
            requirements,
            capability: crate::capability::qwen4_exp_target(&spec)?,
            load_selection: None,
            prediction: None,
        })
    }

    /// Consumes header authority and pins exact metadata/provenance before reading
    /// bounded hash vectors and scalar scales. Accepts the complete admitted source
    /// or its already-resolved claimed view; it never opens an artifact.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
        limits: TargetLimits,
    ) -> Result<PreparedTarget, PreparationError> {
        let spec = self.target_spec(limits)?;
        let keys: BTreeSet<_> = source.source_keys().into_iter().collect();
        let all: BTreeSet<_> = self.catalog.0.keys().cloned().collect();
        if keys != all && keys != *self.resolution.source_keys() {
            return Err(PreparationError::Contract(
                "source set differs from admitted SafeTensors headers".into(),
            ));
        }
        let source = retain(source, keys)?;
        for key in source.source_keys() {
            let metadata = source.source_metadata(&key)?;
            let admitted = &self.catalog.0[&key];
            if metadata.logical_shape != admitted.shape
                || metadata.stored_dtype != admitted.stored_dtype
            {
                return Err(PreparationError::Contract(format!(
                    "source header changed for {key}"
                )));
            }
            // Claimed SafeTensors tensors use the released scalar encodings.
            // Check all owners before any injection reads its literal buffers.
            if self.resolution.source_keys().contains(&key) {
                let bytes = HeaderCatalog::tensor_bytes(admitted);
                if metadata.physical_shape != admitted.shape
                    || bytes != Some(metadata.encoded_byte_len)
                {
                    return Err(PreparationError::Contract(format!(
                        "source physical header changed for {key}"
                    )));
                }
            }
        }
        let resolution = resolve_safetensors_plan(source.as_ref(), &self.checkpoint)
            .map_err(|error| PreparationError::Contract(format!("{error:?}")))?;
        if resolution.source_keys() != self.resolution.source_keys() {
            return Err(PreparationError::Contract(
                "source layout differs from admitted SafeTensors headers".into(),
            ));
        }
        let artifact: SharedCheckpointSource =
            Arc::new(ResolvedCheckpointSource::new(source, resolution));
        let config = self.config;
        let encoding = self.encoding;
        let roots = (0..config.layers.len())
            .map(|l| format!("model.layers.{l}.mlp.experts"))
            .chain(
                (0..config.prediction.as_ref().map_or(0, |p| p.layers.len()))
                    .map(|d| format!("mtp.layers.{d}.mlp.experts")),
            );
        let (formats, expert_banks) =
            prepare_safetensors_formats(&artifact, &config, &encoding, roots)?;
        let mut tables = BTreeMap::new();
        let mut ordinal = 0;
        for layer in 0..config.layers.len() {
            if let Some(table) = self.tables.get(&layer) {
                tables.insert(
                    layer,
                    table.bind(artifact.clone(), tables.len() + 1, ordinal, limits.element)?,
                );
                ordinal += 1;
            }
            ordinal += 1;
        }
        // Literal controls must satisfy the header-authored construction exactly.
        BoundTargetSpec::new(
            spec.clone(),
            tables
                .iter()
                .map(|(&layer, table)| (layer, table.hash.clone()))
                .collect(),
        )?;
        let (static_recipes, unit_recipes) =
            target_parameter_recipes(&artifact.source_keys(), &config, &spec)?;
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
}

fn target_parameter_recipes(
    keys: &[String],
    config: &Config,
    spec: &TargetSpec,
) -> Result<
    (
        BTreeMap<String, DerivedWeightRecipe>,
        Vec<BTreeMap<String, DerivedWeightRecipe>>,
    ),
    PreparationError,
> {
    let mut static_recipes =
        recipes::parameter_recipes(keys.iter().cloned(), config, ParameterScope::Static)
            .map_err(PreparationError::Contract)?;
    static_recipes.retain(|name, _| {
        name.starts_with("model.embed_tokens.")
            || name.starts_with("model.hyper_connection_mixer.")
            || name.starts_with("lm_head.")
    });
    let units = spec
        .units
        .iter()
        .map(|unit| {
            let scope = match unit {
                UnitSpec::Lexical { layer, .. } => ParameterScope::Lexical(*layer),
                UnitSpec::Decoder { layer, .. } => ParameterScope::Target(*layer),
            };
            recipes::parameter_recipes(keys.iter().cloned(), config, scope)
                .map_err(PreparationError::Contract)
        })
        .collect::<Result<_, _>>()?;
    Ok((static_recipes, units))
}
