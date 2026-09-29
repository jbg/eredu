//! Prediction requirements selected together with their retained target artifact.
use super::requirements::ParameterRequirements;
use super::*;
use crate::qwen4_exp::mtp::{PredictionLimits, PredictionSpec};
use crate::{
    routed_text::{RoutedBankRequirements, RoutedGroupedPlan, RoutedTextRequirements},
    ExpertParameterRecipe, ExpertParameterRole, ExpertRealizationPlan, ExpertResidencyCatalog,
    ExpertResidencyDistribution, ExpertResidencyUnit,
};
use eredu_runtime::{
    AppendStreamBinding, AuxiliaryModuleResidency, ExecutionGroupId, ParameterBankKey,
    ReplicatedTextParameterOwner as Owner,
};

const GROUP: &str = "prediction";
fn invalid(error: impl std::fmt::Display) -> PreparationError {
    PreparationError::Contract(error.to_string())
}
impl TargetExecutionPlan {
    /// Adds exact prediction weights to joint weight selection. Its independent
    /// state/stream declaration is admitted independently during joint selection.
    pub fn with_prediction(
        mut self,
        limits: PredictionLimits,
        streams: Vec<AppendStreamBinding>,
    ) -> Result<Self, PreparationError> {
        if self.load_selection.is_some() {
            return Err(invalid(
                "normalized target-only load requires separate joint prediction preparation",
            ));
        }
        if self.prediction.is_some() {
            return Err(invalid("prediction is already prepared"));
        }
        let prediction = self.target.prediction(limits)?;
        let physical = super::requirements::physical_sources(self.target.artifact.as_ref())?;
        let units = (0..prediction.spec.units.len())
            .map(|depth| prediction.unit(depth).map(|unit| unit.recipes().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let (requirements, state) = prediction_requirements(
            &prediction.spec,
            &self.target.spec,
            &self.target.formats,
            self.target.artifact.as_ref(),
            &physical,
            prediction.shared().recipes(),
            &units,
            &self.target.expert_banks,
            &self.requirements,
            streams,
        )?;
        self.requirements = requirements;
        self.capability =
            crate::capability::qwen4_exp_prediction(&self.target.spec, &prediction.spec)?;
        self.prediction = Some((prediction, state));
        Ok(self)
    }
}
/// Shared metadata-only authoring for bound and header-only prediction plans.
pub(super) fn prediction_requirements<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    prediction: &PredictionSpec,
    target: &TargetSpec,
    formats: &ParameterFormats,
    source: &C,
    physical: &BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource>,
    shared_recipes: &BTreeMap<String, DerivedWeightRecipe>,
    unit_recipes: &[BTreeMap<String, DerivedWeightRecipe>],
    expert_banks: &BTreeMap<String, Arc<PreparedExpertBank>>,
    target_requirements: &RoutedTextRequirements,
    streams: Vec<AppendStreamBinding>,
) -> Result<
    (
        RoutedTextRequirements,
        eredu_runtime::StateRealizationRequirements,
    ),
    PreparationError,
> {
    let state = prediction.state_layout()?;
    AppendStreamBinding::validate_layout(&state, &streams)?;
    for binding in &streams {
        if binding.lanes != prediction.limits.qsa.batch as u32
            || binding.limits.read_entries < prediction.limits.tile_blocks as usize
        {
            return Err(eredu_runtime::AppendStreamError::Geometry.into());
        }
    }
    let bank = prediction.units[0].feed_forward.feed_forward.bank;
    let mut declared = ParameterRequirements::new(source, physical, formats);
    let shared =
        AuxiliaryModuleResidency::new("qwen4_exp.prediction.shared", true).map_err(invalid)?;
    for (name, recipe) in shared_recipes {
        declared.add(
            name,
            recipe,
            Owner::StaticRole("prediction_shared".into()),
            false,
        )?;
    }
    for parameter in &mut declared.parameters {
        *parameter = parameter.clone().with_auxiliary_residency(shared.clone());
    }
    let owner = ExecutionGroupId::new(GROUP).map_err(invalid)?;
    let mut specs = BTreeMap::new();
    let mut routes = BTreeMap::new();
    let mut members = Vec::new();
    for unit in &prediction.units {
        let depth = unit.depth;
        let first = declared.parameters.len();
        let parameter_owner = Owner::ExecutionUnit {
            group: GROUP.into(),
            unit: depth,
        };
        for (name, recipe) in &unit_recipes[depth] {
            declared.add(name, recipe, parameter_owner.clone(), false)?;
        }
        let root = format!("mtp.layers.{depth}.mlp.experts");
        let complete = expert_banks
            .get(&root)
            .ok_or_else(|| invalid("missing prepared prediction bank"))?;
        for (local, recipe) in &complete.recipes {
            declared.add(
                &format!("{root}.{local}"),
                &recipe,
                parameter_owner.clone(),
                true,
            )?;
        }
        let residency =
            AuxiliaryModuleResidency::new(format!("qwen4_exp.prediction.depth.{depth}"), false)
                .map_err(invalid)?;
        for parameter in &mut declared.parameters[first..] {
            *parameter = parameter
                .clone()
                .with_auxiliary_residency(residency.clone());
        }
        specs.insert(
            (owner.clone(), depth),
            unit.feed_forward.feed_forward.experts.clone(),
        );
        routes.insert(depth, target.config.experts.selected as usize);
        for expert in 0..target.config.experts.count as usize {
            let recipes = select_expert(expert_banks, &root, expert)?;
            let parameters = recipes
                .into_iter()
                .map(|(local, recipe)| {
                    let target = format!("{root}.{local}");
                    let role = if matches!(local.as_str(), "gate_up_proj" | "down_proj")
                        && formats.get(&target)? == LinearFormat::Dense
                    {
                        ExpertParameterRole::quantizable_projection(
                            format!("{local}_scales"),
                            format!("{local}_biases"),
                        )
                    } else {
                        ExpertParameterRole::Preserved
                    };
                    ExpertParameterRecipe::new(&local, target, recipe, role).map_err(invalid)
                })
                .collect::<Result<Vec<_>, _>>()?;
            members.push(
                ExpertResidencyUnit::new(
                    ParameterBankKey::new(bank.value() as usize, depth, expert),
                    owner.clone(),
                    depth,
                    format!("mtp.layers.{depth}"),
                    ExpertResidencyDistribution::ExpertParallel,
                    parameters,
                )
                .map_err(invalid)?,
            );
        }
    }
    let catalog = ExpertResidencyCatalog::new(members)
        .map_err(invalid)?
        .with_inferred_byte_geometry(source)
        .map_err(invalid)?;
    let plan = ExpertRealizationPlan::balanced(
        target.config.experts.count as usize,
        eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(invalid)?,
            0,
        )
        .map_err(invalid)?,
        specs,
    )
    .map_err(invalid)?;
    let mut banks = target_requirements.banks().clone();
    banks.insert(
        bank,
        RoutedBankRequirements::new(owner, plan.into(), catalog, routes).map_err(invalid)?,
    );
    let identity = eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
        "qwen4_exp.target_prediction.v1",
        [
            ("target", target.geometry_fingerprint()),
            ("prediction", format!("{prediction:?}")),
        ],
    );
    let text = target_requirements
        .text()
        .clone()
        .with_auxiliary_parameters(declared.parameters, declared.derived, declared.outputs)
        .map_err(invalid)?
        .with_replicated_static_roles(["embedding", "output"])
        .map_err(invalid)?
        .with_additional_operators(prediction.required_operators())
        .with_extension_architecture_identity(identity)
        .map_err(invalid)?;
    let mut requirements = RoutedTextRequirements::new(text, banks, source).map_err(invalid)?;
    if let Some(rows) = target_requirements.row_lookups() {
        requirements = requirements
            .with_row_lookups(rows.clone())
            .map_err(invalid)?;
    }
    let access = if streams.is_empty() {
        eredu_runtime::ReplicatedTextStateAccess::Fixed
    } else {
        eredu_runtime::ReplicatedTextStateAccess::AttentionWithStreams
    };
    let state = eredu_runtime::StateRealizationRequirements::new(
        state,
        access,
        requirements.text().floating_state_source().cloned(),
        streams,
    )
    .map_err(invalid)?;
    Ok((requirements, state))
}
impl SelectedTargetExecution {
    /// Selected prediction formats and exact independent state declaration, without payload reads.
    pub fn prediction_spec(&self) -> Result<PredictionSpec, PreparationError> {
        let (prediction, _) = self
            .plan
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        selected_prediction_spec(
            &self.plan.target.spec.config,
            &prediction.spec,
            &self.plan.target.formats,
            &self.selected,
        )
    }
}

pub(super) fn selected_prediction_spec(
    config: &Config,
    prediction: &PredictionSpec,
    source_formats: &ParameterFormats,
    selected: &crate::routed_text::SelectedRoutedTextRealization,
) -> Result<PredictionSpec, PreparationError> {
    let bank = prediction.units[0].feed_forward.feed_forward.bank;
    let selected_bank = selected
        .bank(bank)
        .ok_or_else(|| invalid("selected prediction bank missing"))?;
    let RoutedGroupedPlan::Gated(experts) = selected_bank.plan() else {
        return Err(invalid("prediction requires gated experts"));
    };
    let formats: BTreeMap<_, _> = selected
        .text()
        .auxiliary_parameters()
        .iter()
        .map(|p| (p.name(), p.executable()))
        .collect();
    PredictionSpec::from_prepared(
        config,
        prediction.limits,
        bank,
        |depth| {
            experts
                .unit_spec(GROUP, depth)
                .cloned()
                .ok_or_else(|| Error::backend("selected prediction expert owner missing"))
        },
        |name| {
            let format = *formats
                .get(name)
                .ok_or_else(|| Error::backend(format!("missing prediction format {name}")))?;
            if format == source_formats.get(name)? {
                source_formats.ordinary(name)
            } else {
                crate::linear_format::standard_linear_format(name, format)
            }
        },
    )
    .map_err(PreparationError::Neural)
}

impl SelectedTargetExecution {
    /// Constructs prediction parameter owners from the retained selected tasks.
    /// Backends receive exact source/local modules without interpreting family names.
    pub fn prepare_prediction_weights<B>(
        &self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::prediction_extension::PreparedQwen4PredictionWeights<B>, PreparationError>
    where
        B: eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend
            + eredu_nn::HyperNeuralBackend,
    {
        let (prediction, _) = self
            .plan
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        crate::routed_text::validate_selected_routed_handoff(
            &self.plan.requirements,
            &self.selected,
        )
        .map_err(invalid)?;
        crate::replicated_text::validate_store_handoff(
            self.plan.requirements.text(),
            self.plan.target.artifact.as_ref(),
            crate::replicated_text::StoreHandoffScope::Complete,
        )
        .map_err(invalid)?;
        crate::prediction_extension::PreparedQwen4PredictionWeights::new(
            prediction.artifact.clone(),
            &prediction.spec,
            self.prediction_spec()?,
            None,
            self.prediction_state
                .clone()
                .ok_or_else(|| invalid("prediction state was not selected"))?,
            self.selected.text().auxiliary_materialization_tasks(),
            self.selected.text().residency(),
            context,
        )
        .map_err(invalid)
    }
}

/// Concrete target and prediction owners prepared from one retained joint selection.
/// Source authority, selected state and bank identities survive native binding together.
pub struct PreparedPredictionExecution<B>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
{
    target: crate::routed_text::PreparedRoutedTextArchitecture<
        crate::qwen4_exp::target::TargetModel<B>,
    >,
    prediction: crate::prediction_extension::PreparedQwen4PredictionWeights<B>,
    target_source: SharedCheckpointSource,
    provider_source: SharedCheckpointSource,
}
impl<B> PreparedPredictionExecution<B>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
{
    /// Bank identities moved from the shared provider collection into prediction.
    pub fn prediction_banks(&self) -> Vec<eredu_runtime::RoutedBankId> {
        self.prediction
            .spec()
            .units
            .iter()
            .map(|unit| unit.feed_forward.feed_forward.bank)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    /// Primary-only source for ordinary target materialization and strict validation.
    pub fn target_source(&self) -> &SharedCheckpointSource {
        &self.target_source
    }
    /// All declared primary and auxiliary parameters for joint provider binding.
    /// Row tables and host controls retain their independent source roles.
    pub fn provider_source(&self) -> &SharedCheckpointSource {
        &self.provider_source
    }
    /// Moves the target and prediction roles, followed by their primary target
    /// source and complete provider source, without rediscovery or reselection.
    pub fn into_parts(
        self,
    ) -> (
        crate::routed_text::PreparedRoutedTextArchitecture<
            crate::qwen4_exp::target::TargetModel<B>,
        >,
        crate::prediction_extension::PreparedQwen4PredictionWeights<B>,
        SharedCheckpointSource,
        SharedCheckpointSource,
    ) {
        (
            self.target,
            self.prediction,
            self.target_source,
            self.provider_source,
        )
    }
}
impl SelectedTargetExecution {
    /// Constructs both roles with distinct target and joint-provider source authority.
    pub fn prepare_prediction_execution<B, S>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<PreparedPredictionExecution<B>, PreparationError>
    where
        B: eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend
            + eredu_nn::HyperNeuralBackend,
        S: eredu_runtime::LayerRuntimeState<B>,
        S::LayerState: eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_runtime::RuntimeAppendStreams<B>,
    {
        let prediction = self.prepare_prediction_weights::<B>(context)?;
        // The ordinary target keeps primary-only authority. The joint provider
        // also owns prediction banks, whose physical recipes are auxiliary
        // parameters; give that provider the complete declared parameter view,
        // while retaining row tables and host controls in their separate roles.
        let provider_source = crate::replicated_text::restrict_store_handoff(
            self.plan.requirements.text(),
            self.plan.target.artifact.clone(),
            crate::replicated_text::StoreHandoffScope::Complete,
        )
        .map_err(invalid)?;
        let (target, target_source) = self.prepare::<B, S>(context)?;
        Ok(PreparedPredictionExecution {
            target,
            prediction,
            target_source,
            provider_source,
        })
    }
}

impl SelectedTargetExecution {
    /// Derives the shared speculative driver's exact pre-collapse capture contract.
    /// Bounds and target identity must match this retained joint selection.
    pub fn speculative_contract(
        &self,
        request: crate::prediction_extension::EmbeddedSpeculativeContractRequest,
    ) -> Result<crate::prediction_extension::EmbeddedSpeculativeContract, PreparationError> {
        crate::prediction_extension::qwen4_speculative_contract(
            &self.prediction_spec()?,
            &self.target_state_fingerprint()?,
            self.plan.target.spec.limits,
            request,
        )
        .map_err(invalid)
    }
}
