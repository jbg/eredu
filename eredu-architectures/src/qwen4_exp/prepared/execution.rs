//! Retained target authority across cold selection and typed native binding.
use super::*;
use crate::{
    capability::CapabilityEstimate,
    qwen4_exp::target::TargetModel,
    routed_text::{
        prepare_routed_architecture_handoff, select_routed_text_realization,
        validate_selected_routed_handoff, RoutedTextArchitectureVisitor, RoutedTextDispatchError,
        RoutedTextRequirements, RoutedTextSelectionError, RoutedTextSelectionRequest,
        SelectedRoutedTextRealization,
    },
};
use eredu_nn::{AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Tensor};
use eredu_runtime::{
    BackendMechanismCapabilities, LayerRuntimeState, RuntimeAppendStreams, RuntimeStateComponents,
};

/// Payload-lazy target plan. Requirements and reports retain the exact artifact and
/// construction specification that authored them; external catalogs cannot replace it.
#[derive(Clone)]
pub struct TargetExecutionPlan {
    pub(super) target: PreparedTarget,
    pub(super) prediction: Option<(
        super::prediction::PreparedPrediction,
        eredu_runtime::StateRealizationRequirements,
    )>,
    pub(super) requirements: RoutedTextRequirements,
    pub(super) row_sources: eredu_runtime::PreparedRowLookups,
    pub(super) capability: CapabilityEstimate,
    pub(super) load_selection: Option<RoutedTextSelectionRequest>,
}
/// Cold admission failure before native allocation or checkpoint payload reads.
#[derive(Debug, thiserror::Error)]
pub enum TargetSelectionError {
    /// A retained normalized request cannot be replaced during mechanism selection.
    #[error("qwen4_exp selection differs from retained normalized load policy")]
    LoadRequestMismatch,
    /// Input readiness cannot replace the caller's retained normalized media intent.
    #[error("qwen4_exp processor selection differs from retained normalized media policy")]
    ProcessorRequestMismatch,
    /// Requested media representations or processor primitives are unavailable.
    #[error(transparent)]
    Processor(#[from] eredu_runtime::ProcessorSelectionError),
    /// Target weights, state or provider mechanisms are unavailable.
    #[error(transparent)]
    Target(#[from] RoutedTextSelectionError),
    /// Retained requirements differ from the selected cold authority.
    #[error("target selection contract: {0}")]
    Contract(String),
    /// Independent prediction state is not covered by the supplied mechanisms.
    #[error("prediction state: {0}")]
    PredictionState(#[from] eredu_runtime::ReplicatedTextSelectionError),
    /// Every prepared prediction requires its own exact state mechanism facts.
    #[error("prediction state capability presence differs from the prepared roles")]
    PredictionStatePresence,
    /// Independently selected owners cannot form a valid aggregate resource contract.
    #[error("combined state resources: {0}")]
    Resources(#[from] eredu_core::resources::ResourceDescriptionError),
}
impl TargetExecutionPlan {
    /// Exact metadata contracts supplied to generic mechanism selection.
    pub fn requirements(&self) -> &RoutedTextRequirements {
        &self.requirements
    }
    /// Exact policy retained by normalized load lowering, when selected through that path.
    pub fn load_selection_request(&self) -> Option<&RoutedTextSelectionRequest> {
        self.load_selection.as_ref()
    }
    /// Context and persistent-state accounting for this target role.
    pub fn capability_estimate(&self) -> &CapabilityEstimate {
        &self.capability
    }
    /// Independent prediction geometry for exact backend capability queries.
    pub fn prediction_state_requirements(
        &self,
    ) -> Option<&eredu_runtime::StateRealizationRequirements> {
        self.prediction.as_ref().map(|(_, state)| state)
    }
    /// Selects without a native context or any ordinary/expert/table payload read.
    pub fn select(
        self,
        request: &RoutedTextSelectionRequest,
        mechanisms: &BackendMechanismCapabilities,
        prediction_mechanisms: Option<&eredu_runtime::StateMechanismCapabilities>,
    ) -> Result<SelectedTargetExecution, TargetSelectionError> {
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
        let selected = select_routed_text_realization(&self.requirements, request, mechanisms)?;
        self.bind_selected(selected, prediction_state)
    }

    pub(super) fn bind_selected(
        self,
        selected: SelectedRoutedTextRealization,
        prediction_state: Option<eredu_runtime::SelectedStateRealization>,
    ) -> Result<SelectedTargetExecution, TargetSelectionError> {
        validate_selected_routed_handoff(&self.requirements, &selected)
            .map_err(|e| TargetSelectionError::Contract(e.to_string()))?;
        if self.prediction.is_some() != prediction_state.is_some() {
            return Err(TargetSelectionError::PredictionStatePresence);
        }
        match selected.row_lookups() {
            Some(rows) if rows.descriptors() == self.row_sources.descriptors() => {}
            None if self.row_sources.entries().is_empty() => {}
            _ => {
                return Err(TargetSelectionError::Contract(
                    "bound row sources differ from the selected descriptors".into(),
                ));
            }
        }
        let stream_allowances = selected_stream_allowances(&selected, prediction_state.as_ref())?;
        Ok(SelectedTargetExecution {
            plan: self,
            selected,
            prediction_state,
            stream_allowances,
        })
    }
}

pub(super) fn selected_stream_allowances(
    selected: &SelectedRoutedTextRealization,
    prediction_state: Option<&eredu_runtime::SelectedStateRealization>,
) -> Result<eredu_runtime::AppendStreamAllowances, TargetSelectionError> {
    let mut states = vec![eredu_runtime::execution_resources::PreparedStateResource {
        owner: "target",
        state: selected.text().state(),
    }];
    if let Some(state) = prediction_state {
        states.push(eredu_runtime::execution_resources::PreparedStateResource {
            owner: "prediction",
            state,
        });
    }
    Ok(
        eredu_runtime::execution_resources::PreparedStateResource::append_stream_allowances(
            &states,
        )?,
    )
}

/// A selected target paired with its original preparation authority. Construction
/// accepts native mechanisms, never substitute requirements, specs or source stores.
#[derive(Clone)]
pub struct SelectedTargetExecution {
    pub(super) plan: TargetExecutionPlan,
    pub(super) selected: SelectedRoutedTextRealization,
    pub(super) prediction_state: Option<eredu_runtime::SelectedStateRealization>,
    pub(super) stream_allowances: eredu_runtime::AppendStreamAllowances,
}
impl SelectedTargetExecution {
    /// Literal-aware state identity after the retained parameter transforms.
    /// Use this identity for state and speculative contracts; cold requirements
    /// describe geometry before exact integer controls are bound.
    pub fn target_state_fingerprint(&self) -> Result<String, PreparationError> {
        let spec = self.plan.target.selected_spec(&self.selected)?;
        Ok(self
            .plan
            .target
            .bind_spec(spec)?
            .state_fingerprint()
            .to_owned())
    }

    /// Exact independent prediction placement and lifecycle guarantees.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction_state.as_ref()
    }
    /// Actual retained state, weight, grouped-bank and row mechanisms.
    pub fn realization(&self) -> &SelectedRoutedTextRealization {
        &self.selected
    }
    /// Target context and exact persistent-state layout.
    pub fn capability_estimate(&self) -> &CapabilityEstimate {
        &self.plan.capability
    }
    /// Constructs with the combined K/V, fixed and append-stream profile, then
    /// delegates materialization and session assembly to the ordinary typed visitor.
    pub fn visit<B, S, V>(
        self,
        context: &<B::Tensor as Tensor>::Context,
        mut visitor: V,
    ) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState:
            AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
        V: RoutedTextArchitectureVisitor<B, S>,
    {
        let (prepared, source) = self
            .prepare_with::<B, S>(context, || visitor.construction_started())
            .map_err(|e| RoutedTextDispatchError::Architecture(e.to_string()))?;
        visitor
            .visit(prepared, source)
            .map_err(RoutedTextDispatchError::Backend)
    }

    /// Retains the concrete architecture type for target/prediction session composition.
    pub fn prepare<B, S>(
        self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            crate::routed_text::PreparedRoutedTextArchitecture<TargetModel<B>>,
            SharedCheckpointSource,
        ),
        PreparationError,
    >
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState:
            AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        self.prepare_with::<B, S>(context, || {})
    }

    fn prepare_with<B, S>(
        self,
        context: &<B::Tensor as Tensor>::Context,
        construction_started: impl FnOnce(),
    ) -> Result<
        (
            crate::routed_text::PreparedRoutedTextArchitecture<TargetModel<B>>,
            SharedCheckpointSource,
        ),
        PreparationError,
    >
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState:
            AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        let Self { plan, selected, .. } = self;
        validate_selected_routed_handoff(&plan.requirements, &selected)
            .map_err(|e| PreparationError::Contract(e.to_string()))?;
        let execution_source = crate::replicated_text::restrict_store_handoff(
            plan.requirements.text(),
            plan.target.artifact.clone(),
            crate::replicated_text::StoreHandoffScope::Primary,
        )
        .map_err(PreparationError::Contract)?;
        let spec = plan
            .target
            .selected_spec(&selected)
            .map_err(|e| PreparationError::Contract(e.to_string()))?;
        let spec = plan.target.bind_spec(spec)?;
        let identity = spec.state_fingerprint().to_owned();
        construction_started();
        let source_architecture = crate::replicated_text::selected_uses_transform(selected.text())
            .then(|| {
                plan.target
                    .bound_spec()
                    .and_then(|bound| TargetModel::<B>::new(bound, context))
            })
            .transpose()
            .map_err(|e| PreparationError::Contract(e.to_string()))?;
        let architecture = TargetModel::<B>::new(spec, context)
            .map_err(|e| PreparationError::Contract(e.to_string()))?;
        let prepared = prepare_routed_architecture_handoff::<B, S, _>(
            architecture,
            source_architecture,
            plan.requirements,
            selected,
            (!plan.row_sources.entries().is_empty()).then_some(plan.row_sources),
            plan.capability,
            "qwen4_exp_text".into(),
            identity,
            context,
        )
        .map_err(PreparationError::Contract)?;
        Ok((prepared, execution_source))
    }
}

impl PreparedTarget {
    pub(crate) fn selected_bound_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<super::BoundTargetSpec, Error> {
        self.bind_spec(self.selected_spec(selected)?)
    }

    // Retain checkpoint-native companions for unchanged projections. Only changed
    // formats adopt the shared architecture-authored output companion convention.
    pub(super) fn selected_spec(
        &self,
        selected: &SelectedRoutedTextRealization,
    ) -> Result<TargetSpec, Error> {
        target_spec_from_formats(
            self.spec.config.clone(),
            &self
                .tables
                .iter()
                .map(|(&layer, table)| (layer, table.lookup.clone()))
                .collect(),
            self.spec.limits,
            &self.formats,
            Some(selected),
        )
    }
}

/// One format projection for cold inspection and source-bound construction.
pub(super) fn target_spec_from_formats(
    config: Config,
    rows: &BTreeMap<usize, eredu_runtime::RowLookupSpec>,
    limits: TargetLimits,
    source: &ParameterFormats,
    selected: Option<&SelectedRoutedTextRealization>,
) -> Result<TargetSpec, Error> {
    use crate::routed_text::RoutedGroupedPlan;
    let formats = selected.map(|selected| {
        crate::replicated_text::selected_matrix_formats(
            selected.text().requirements(),
            selected.text(),
        )
    });
    let experts = selected
        .map(|selected| {
            let bank = selected
                .bank(eredu_runtime::RoutedBankId::new(0))
                .ok_or_else(|| Error::backend("selected target is missing its expert bank"))?;
            let RoutedGroupedPlan::Gated(experts) = bank.plan() else {
                return Err(Error::backend(
                    "selected target requires gated-product experts",
                ));
            };
            Ok(experts)
        })
        .transpose()?;
    TargetSpec::from_headers(
        config.clone(),
        rows,
        limits,
        |layer, ordinal| match experts {
            Some(experts) => experts
                .unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, ordinal)
                .cloned()
                .ok_or_else(|| Error::backend("selected target expert owner is missing")),
            None => expert_spec(
                &config,
                source,
                &format!("model.layers.{layer}.mlp.experts"),
            ),
        },
        |name| {
            let Some(formats) = &formats else {
                return source.ordinary(name);
            };
            let format = formats.get(name).copied().ok_or_else(|| {
                Error::backend(format!("target projection {name} has no selected format"))
            })?;
            if format == source.get(name)? {
                source.ordinary(name)
            } else {
                crate::linear_format::standard_linear_format(name, format)
            }
        },
    )
}
