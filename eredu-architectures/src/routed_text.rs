//! Architecture-owned routed execution over generic grouped-bank mechanisms.

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use eredu_nn::{
    GroupSelection, GroupedGatedProductOperator, GroupedLinearOperator, GroupedNeuralBackend,
    GroupedRelu2Operator, Tensor,
};
use eredu_runtime::{
    AddressableGroupedBank, IndexedMovement, ParameterBankAcquisition, ParameterBankKey,
    RoutedBankId, RoutedExpertProvider, RoutedExpertRequest,
};

use crate::{ExpertRealizationPlan, ExpertResidencyCatalog};
mod source;
mod composite;
pub(crate) use composite::prepare_composite_handoff;
pub use source::RoutedTextConstructionSource;
mod units;
pub(crate) use units::RetainedRoutedUnits;
pub(crate) use source::{CompletedRoutedConstruction, RetainedRoutedDescription, RoutedConstructionParameters};
use crate::replicated_text::{config_source,capability_source};

/// Architecture-owned grouped equation and exact per-unit realization plan.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutedGroupedPlan {
    /// Activated selected-linear banks with owned output rows.
    Linear(ExpertRealizationPlan<eredu_nn::GroupedLinearSpec>),
    /// Gated-product grouped banks.
    Gated(ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>),
    /// ReLU-squared grouped banks.
    Relu2(ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>),
}

impl RoutedGroupedPlan {
    fn partition_routes(
        &self,
        routes: &BTreeMap<usize, usize>,
        exchange: bool,
    ) -> BTreeMap<usize, usize> {
        routes
            .iter()
            .map(|(unit, count)| {
                let replicated = match self {
                    Self::Gated(plan) => plan.unit_is_replicated(*unit),
                    Self::Relu2(plan) => plan.unit_is_replicated(*unit),
                    Self::Linear(plan) => plan.unit_is_replicated(*unit),
                };
                (*unit, if exchange && !replicated { 1 } else { *count })
            })
            .collect()
    }

    pub(crate) fn with_catalog_distribution(
        self,
        catalog: &ExpertResidencyCatalog,
    ) -> Result<Self, String> {
        Ok(match self {
            Self::Linear(plan) => Self::Linear(plan.with_catalog_distribution(catalog)?),
            Self::Gated(plan) => Self::Gated(plan.with_catalog_distribution(catalog)?),
            Self::Relu2(plan) => Self::Relu2(plan.with_catalog_distribution(catalog)?),
        })
    }

    pub(crate) fn project_local_groups(
        &self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<Vec<usize>, crate::ExpertRealizationPlanError> {
        match self {
            Self::Linear(plan) => plan.project_local_groups(topology),
            Self::Gated(plan) => plan.project_local_groups(topology),
            Self::Relu2(plan) => plan.project_local_groups(topology),
        }
    }

    /// Checkpoint-global members owned by this rank for this bank alone.
    pub fn local_global_group_indices(&self) -> &[usize] {
        match self {
            Self::Linear(plan) => plan.local_global_group_indices(),
            Self::Gated(plan) => plan.local_global_group_indices(),
            Self::Relu2(plan) => plan.local_global_group_indices(),
        }
    }

    /// Whether this bank is invoked by the specified logical unit.
    pub fn has_unit(&self, owner: &str, unit: usize) -> bool {
        match self {
            Self::Linear(plan) => plan.unit_spec(owner, unit).is_some(),
            Self::Gated(plan) => plan.unit_spec(owner, unit).is_some(),
            Self::Relu2(plan) => plan.unit_spec(owner, unit).is_some(),
        }
    }
    /// Returns the architecture-global routed member count.
    pub fn global_group_count(&self) -> usize {
        match self {
            Self::Linear(plan) => plan.global_expert_count(),
            Self::Gated(plan) => plan.global_expert_count(),
            Self::Relu2(plan) => plan.global_expert_count(),
        }
    }

    /// Returns the selected expert-axis width.
    pub fn expert_parallel_size(&self) -> usize {
        match self {
            Self::Linear(plan) => plan.expert_parallel_size(),
            Self::Gated(plan) => plan.expert_parallel_size(),
            Self::Relu2(plan) => plan.expert_parallel_size(),
        }
    }

    /// Returns this rank's coordinate in the selected expert axis.
    pub fn expert_parallel_rank(&self) -> usize {
        match self {
            Self::Linear(plan) => plan.expert_parallel_rank(),
            Self::Gated(plan) => plan.expert_parallel_rank(),
            Self::Relu2(plan) => plan.expert_parallel_rank(),
        }
    }

    /// Returns the selected-linear plan when that equation was selected.
    pub const fn linear(&self) -> Option<&ExpertRealizationPlan<eredu_nn::GroupedLinearSpec>> {
        match self {
            Self::Linear(plan) => Some(plan),
            _ => None,
        }
    }

    /// Returns the gated-product plan when that equation was selected.
    pub const fn gated(&self) -> Option<&ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>> {
        match self {
            Self::Gated(plan) => Some(plan),
            Self::Linear(_) | Self::Relu2(_) => None,
        }
    }

    /// Returns the ReLU-squared plan when that equation was selected.
    pub const fn relu2(&self) -> Option<&ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>> {
        match self {
            Self::Relu2(plan) => Some(plan),
            Self::Linear(_) | Self::Gated(_) => None,
        }
    }
}

impl From<ExpertRealizationPlan<eredu_nn::GroupedLinearSpec>> for RoutedGroupedPlan {
    fn from(plan: ExpertRealizationPlan<eredu_nn::GroupedLinearSpec>) -> Self {
        Self::Linear(plan)
    }
}

impl From<ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>> for RoutedGroupedPlan {
    fn from(plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>) -> Self {
        Self::Gated(plan)
    }
}

impl From<ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>> for RoutedGroupedPlan {
    fn from(plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>) -> Self {
        Self::Relu2(plan)
    }
}

/// Converts one architecture-owned grouped specification into the opaque
/// routed plan retained by the neutral execution driver.
pub trait RoutedGroupedSpec: Clone {
    /// Erases the concrete grouped equation after architecture preparation.
    fn into_routed_grouped_plan(plan: ExpertRealizationPlan<Self>) -> RoutedGroupedPlan;
}

impl RoutedGroupedSpec for eredu_nn::GroupedLinearSpec {
    fn into_routed_grouped_plan(plan: ExpertRealizationPlan<Self>) -> RoutedGroupedPlan {
        RoutedGroupedPlan::Linear(plan)
    }
}

impl RoutedGroupedSpec for eredu_nn::GroupedGatedProductSpec {
    fn into_routed_grouped_plan(plan: ExpertRealizationPlan<Self>) -> RoutedGroupedPlan {
        RoutedGroupedPlan::Gated(plan)
    }
}

impl RoutedGroupedSpec for eredu_nn::GroupedRelu2Spec {
    fn into_routed_grouped_plan(plan: ExpertRealizationPlan<Self>) -> RoutedGroupedPlan {
        RoutedGroupedPlan::Relu2(plan)
    }
}

mod banks;
pub use banks::{PlannedAddressableBank, PlannedResidentBank};
mod partition_units;
mod partition_source;
mod addressable_source;
pub(crate) use partition_source::RetainedPartitionResidentSource;
pub(crate) use partition_units::PartitionUnitProvider;

/// Closed alias of the actual immutable bank table produced by cold selection.
#[derive(Debug,PartialEq)]
pub(crate) struct RetainedRoutedBanks(Option<std::sync::Arc<BTreeMap<RoutedBankId,SelectedRoutedBank>>>);
impl Clone for RetainedRoutedBanks{fn clone(&self)->Self{Self(self.0.clone())}}
impl Drop for RetainedRoutedBanks{fn drop(&mut self){if let Some(owner)=self.0.take(){drop(std::sync::Arc::into_inner(owner));}}}
impl std::ops::Deref for RetainedRoutedBanks{
    type Target=BTreeMap<RoutedBankId,SelectedRoutedBank>;
    fn deref(&self)->&Self::Target{self.0.as_deref().expect("live immutable routed bank table")}
}
impl RetainedRoutedBanks{
    fn new(values:BTreeMap<RoutedBankId,SelectedRoutedBank>)->Self{Self(Some(std::sync::Arc::new(values)))}
    fn same_source(&self,other:&Self)->bool{std::sync::Arc::ptr_eq(self.0.as_ref().expect("live bank table"),other.0.as_ref().expect("live bank table"))}
    // Existing public ordinary materialization API. Checked construction uses
    // into_shared_parts and never enters this independently owned copy path.
    fn into_owned(mut self)->BTreeMap<RoutedBankId,SelectedRoutedBank>{
        let owner=self.0.take().expect("live bank table");
        match std::sync::Arc::try_unwrap(owner){Ok(values)=>values,Err(owner)=>(*owner).clone()}
    }
}

/// Checked text modules paired with every independently selected routed bank.
pub struct PreparedRoutedTextArchitecture<A> {
    text: crate::replicated_text::PreparedReplicatedTextArchitecture<A>,
    bank_residency: eredu_runtime::ParameterBankResidency,
    banks: RetainedRoutedBanks,
}

impl<A> PreparedRoutedTextArchitecture<A> {
    /// Shared text architecture and exact prepared sources.
    pub const fn text(&self) -> &crate::replicated_text::PreparedReplicatedTextArchitecture<A> {
        &self.text
    }
    /// Placement selected for independently identified banks.
    pub const fn bank_residency(&self) -> eredu_runtime::ParameterBankResidency {
        self.bank_residency
    }
    /// Complete bank facts consumed by reusable materialization mechanisms.
    pub fn banks(&self) -> &BTreeMap<RoutedBankId, SelectedRoutedBank> {
        &self.banks
    }
    /// Moves text modules and identified banks into execution.
    pub fn into_parts(
        self,
    ) -> (
        crate::replicated_text::PreparedReplicatedTextModules<A>,
        eredu_runtime::ParameterBankResidency,
        BTreeMap<RoutedBankId, SelectedRoutedBank>,
    ) {
        (self.text.into_modules(), self.bank_residency, self.banks.into_owned())
    }
    pub(crate) fn into_shared_parts(self)->(
        crate::replicated_text::PreparedReplicatedTextModules<A>,
        eredu_runtime::ParameterBankResidency,RetainedRoutedBanks,
    ) {(self.text.into_modules(),self.bank_residency,self.banks)}
}

/// Failure while pairing one selected routed realization with concrete modules.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedTextPreparationError {
    /// A participating host constructor preserves its exact failure.
    #[error(transparent)]
    Metadata(eredu_nn::Error),
    /// The admitted graph no longer matches the selected routed class.
    #[error("replicated routed text preparation is ineligible")]
    Ineligible,
    /// Requirements, artifact provenance, selected formats, or modules disagreed.
    #[error("invalid replicated routed text preparation: {0}")]
    Invalid(String),
}

fn index_materialization_tasks(
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
) -> BTreeMap<&str, &eredu_runtime::ReplicatedTextMaterializationTask> {
    let mut indexed = BTreeMap::new();
    for task in tasks {
        for companion in task.output_companions() {
            if let Some(exact) = companion.materialization_task() {
                indexed.insert(companion.name(), exact);
            }
        }
        indexed.insert(task.name(), task);
    }
    indexed
}

fn selected_member_geometry(
    unit: &crate::ExpertResidencyUnit,
    tasks: &BTreeMap<&str, &eredu_runtime::ReplicatedTextMaterializationTask>,
) -> Result<(u64, u64), String> {
    let mut source_bytes = 0u64;
    let mut selected_bytes = 0u64;
    for parameter in unit.parameters() {
        let metadata = parameter.metadata().ok_or_else(|| {
            format!(
                "addressable member {:?} omitted admitted recipe metadata",
                unit.identity()
            )
        })?;
        source_bytes = source_bytes
            .checked_add(metadata.byte_len())
            .ok_or_else(|| "addressable source byte geometry overflowed".to_owned())?;
        let task = tasks.get(parameter.logical_target()).ok_or_else(|| {
            format!(
                "addressable target {:?} has no selected parameter realization",
                parameter.logical_target()
            )
        })?;
        let parameter_bytes = if matches!(
            task.lowering(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        ) {
            if metadata.dtype() != &eredu_checkpoint::recipe::RecipeDtype::F4
                && !matches!(
                    parameter.role(),
                    crate::ExpertParameterRole::QuantizableProjection { .. }
                )
            {
                return Err(format!(
                    "addressable transform target {:?} is not a quantizable projection",
                    parameter.logical_target()
                ));
            }
            let quantization = task.executable().weight_quantization().ok_or_else(|| {
                format!(
                    "addressable transform target {:?} has no packed realization",
                    parameter.logical_target()
                )
            })?;
            if matches!(
                quantization,
                eredu_checkpoint::WeightQuantization::GgufIQuant { .. }
            ) {
                return Err(
                    "load-time addressable transformation cannot select GGUF IQuant".into(),
                );
            }
            eredu_runtime::selected_addressable_parameter_bytes(task, metadata)
                .map_err(|error| error.to_string())?
        } else {
            metadata.byte_len()
        };
        selected_bytes = selected_bytes
            .checked_add(parameter_bytes)
            .ok_or_else(|| "addressable selected byte geometry overflowed".to_owned())?;
    }
    if unit.byte_len() != Some(source_bytes) {
        return Err(format!(
            "addressable member {:?} source bytes differ from its admitted catalog",
            unit.identity()
        ));
    }
    Ok((source_bytes, selected_bytes))
}

pub(crate) fn project_addressable_members(
    catalog: &ExpertResidencyCatalog,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<Vec<eredu_runtime::AddressableBankMember>, RoutedTextPreparationError> {
    let tasks = eredu_runtime::replicated_text_materialization_tasks(selected)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    project_addressable_members_with_tasks(catalog, selected, &tasks)
}

pub(crate) fn project_addressable_members_with_tasks(
    catalog: &ExpertResidencyCatalog,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
) -> Result<Vec<eredu_runtime::AddressableBankMember>, RoutedTextPreparationError> {
    project_bank_members_with_tasks(catalog, selected, tasks, false)
}

/// Composite providers invoke both routed and separately replicated shared banks.
pub(crate) fn project_composite_bank_members_with_tasks(
    catalog: &ExpertResidencyCatalog,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
) -> Result<Vec<eredu_runtime::AddressableBankMember>, RoutedTextPreparationError> {
    project_bank_members_with_tasks(catalog, selected, tasks, true)
}

fn project_bank_members_with_tasks(
    catalog: &ExpertResidencyCatalog,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
    include_replicated: bool,
) -> Result<Vec<eredu_runtime::AddressableBankMember>, RoutedTextPreparationError> {
    let selected_tasks = index_materialization_tasks(selected.materialization_tasks());
    let mut exact_tasks = BTreeMap::new();
    for task in tasks {
        if exact_tasks.insert(task.name(), task).is_some() {
            return Err(RoutedTextPreparationError::Invalid(format!(
                "addressable materialization repeats task {:?}",
                task.name()
            )));
        }
        // Partition projection folds encoded-linear companions into their
        // primary task so the physical family stays atomic. A bank catalog,
        // however, retains one binding and recipe per stored value. Re-index
        // the exact standalone task retained by each companion under the
        // architecture topology identity; do not reconstruct its lowering.
        for companion in task.output_companions() {
            let Some(exact) = companion.materialization_task() else {
                continue;
            };
            if exact_tasks.insert(companion.name(), exact).is_some() {
                return Err(RoutedTextPreparationError::Invalid(format!(
                    "addressable materialization repeats output {:?}",
                    companion.name()
                )));
            }
        }
    }
    let mut shared_tasks = BTreeMap::new();
    let mut members = Vec::with_capacity(catalog.units().len());
    for unit in catalog.units() {
        if !include_replicated
            && unit.distribution() != crate::ExpertResidencyDistribution::ExpertParallel
        {
            continue;
        }
        let local_parameters = unit
            .parameters()
            .iter()
            .filter(|parameter| exact_tasks.contains_key(parameter.logical_target()))
            .count();
        if local_parameters == 0 {
            continue;
        }
        if local_parameters != unit.parameters().len() {
            return Err(RoutedTextPreparationError::Invalid(format!(
                "addressable member {:?} has only {local_parameters} of {} exact local tasks",
                unit.identity(),
                unit.parameters().len()
            )));
        }
        let (source_bytes, _selected_bytes) = selected_member_geometry(unit, &selected_tasks)
            .map_err(RoutedTextPreparationError::Invalid)?;
        let mut parameters = Vec::with_capacity(unit.parameters().len());
        for parameter in unit.parameters() {
            let metadata = parameter.metadata().ok_or_else(|| {
                RoutedTextPreparationError::Invalid(format!(
                    "addressable member {:?} omitted admitted recipe metadata",
                    unit.identity()
                ))
            })?;
            let task = exact_tasks
                .get(parameter.logical_target())
                .copied()
                .ok_or_else(|| {
                    RoutedTextPreparationError::Invalid(format!(
                        "addressable target {:?} has no exact materialization task",
                        parameter.logical_target()
                    ))
                })?;
            let transforms = matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            );
            let selected_bytes = if transforms {
                eredu_runtime::selected_addressable_parameter_bytes(task, metadata)
                    .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?
            } else {
                metadata.byte_len()
            };
            let companions =
                if transforms && metadata.dtype() != &eredu_checkpoint::recipe::RecipeDtype::F4 {
                    match parameter.role() {
                        crate::ExpertParameterRole::QuantizableProjection {
                            scales_binding,
                            biases_binding,
                        } => {
                            let quantization =
                                task.executable().weight_quantization().ok_or_else(|| {
                                    RoutedTextPreparationError::Invalid(format!(
                                        "addressable transform target {:?} has no packed format",
                                        task.name()
                                    ))
                                })?;
                            Some(
                                eredu_runtime::QuantizationCompanionBindings::new(
                                    scales_binding,
                                    quantization.has_biases().then(|| biases_binding.clone()),
                                )
                                .map_err(|error| {
                                    RoutedTextPreparationError::Invalid(error.to_string())
                                })?,
                            )
                        }
                        crate::ExpertParameterRole::Preserved => None,
                    }
                } else {
                    None
                };
            let shared_key = (
                parameter.logical_target(),
                metadata.shape(),
                companions.as_ref().map(|bindings| {
                    (
                        bindings.scale().to_owned(),
                        bindings.affine_bias().map(str::to_owned),
                    )
                }),
            );
            let shared_task = if let Some(shared) = shared_tasks.get(&shared_key) {
                eredu_runtime::AddressableBankTask::clone(shared)
            } else {
                let mut task = task.clone();
                if task.output_companions().is_empty() {
                    if let Some(companions) = companions.as_ref() {
                        let quantization =
                            task.executable().weight_quantization().ok_or_else(|| {
                                RoutedTextPreparationError::Invalid(format!(
                                    "addressable transform target {:?} has no packed format",
                                    task.name()
                                ))
                            })?;
                        let prefix = parameter
                            .logical_target()
                            .strip_suffix(parameter.binding_name())
                            .ok_or_else(|| {
                                RoutedTextPreparationError::Invalid(format!(
                                "addressable target {:?} does not end in its local binding {:?}",
                                parameter.logical_target(),
                                parameter.binding_name()
                            ))
                            })?;
                        let mut shape = metadata.shape().to_vec();
                        let columns = shape.last_mut().ok_or_else(|| {
                            RoutedTextPreparationError::Invalid(
                                "addressable transform companion has no group axis".into(),
                            )
                        })?;
                        *columns = columns
                            .checked_div(quantization.group_size() as usize)
                            .ok_or_else(|| {
                                RoutedTextPreparationError::Invalid(
                                    "addressable transform companion group geometry is invalid"
                                        .into(),
                                )
                            })?;
                        let owner = task.owner().parameter_group_owner().map_err(|error| {
                            RoutedTextPreparationError::Invalid(error.to_string())
                        })?;
                        let mut outputs = vec![eredu_runtime::ReplicatedTextOutputCompanion::new(
                            format!("{prefix}{}", companions.scale()),
                            eredu_nn::LinearCompanionRole::Scale,
                            shape.clone(),
                            owner.clone(),
                        )
                        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?];
                        if let Some(biases) = companions.affine_bias() {
                            outputs.push(
                                eredu_runtime::ReplicatedTextOutputCompanion::new(
                                    format!("{prefix}{biases}"),
                                    eredu_nn::LinearCompanionRole::AffineBias,
                                    shape,
                                    owner,
                                )
                                .map_err(|error| {
                                    RoutedTextPreparationError::Invalid(error.to_string())
                                })?,
                            );
                        }
                        task = task.with_output_companions(outputs).map_err(|error| {
                            RoutedTextPreparationError::Invalid(error.to_string())
                        })?;
                    }
                }
                let shared = eredu_runtime::AddressableBankTask::new(task)
                    .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
                shared_tasks.insert(shared_key, shared.clone());
                shared
            };
            parameters.push(
                eredu_runtime::AddressableBankParameter::from_shared_task(
                    parameter.binding_name(),
                    shared_task,
                    parameter.recipe().clone(),
                    metadata.clone(),
                    selected_bytes,
                    companions,
                )
                .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?,
            );
        }
        let distribution = match unit.distribution() {
            crate::ExpertResidencyDistribution::Replicated => {
                eredu_runtime::AddressableBankDistribution::Replicated
            }
            crate::ExpertResidencyDistribution::ExpertParallel => {
                eredu_runtime::AddressableBankDistribution::ExpertParallel
            }
        };
        let placement = eredu_runtime::AddressableBankMemberPlacement::new(
            unit.owner_group().clone(),
            unit.owner_unit(),
            unit.unit_path(),
            distribution,
        )
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
        let member =
            eredu_runtime::AddressableBankMember::new(unit.identity(), placement, parameters)
                .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
        if member.source_bytes() != source_bytes {
            return Err(RoutedTextPreparationError::Invalid(format!(
                "addressable member {:?} source bytes differ from its exact tasks",
                unit.identity()
            )));
        }
        members.push(member);
    }
    Ok(members)
}

pub(crate) fn validate_selected_routed_handoff(
    expected: &RoutedTextRequirements,
    selected: &SelectedRoutedTextRealization,
) -> Result<(), RoutedTextPreparationError> {
    if selected.text().requirements() != expected.text()
        || selected.banks.len() != expected.banks.len()
    {
        return Err(RoutedTextPreparationError::Invalid(
            "selected realization differs from admitted routed requirements".into(),
        ));
    }
    for (id, bank) in &expected.banks {
        let expected_plan = select_grouped_formats(bank.plan.clone(), selected.text())
            .map_err(RoutedTextPreparationError::Invalid)?;
        let Some(selected_bank) = selected.bank(*id) else {
            return Err(RoutedTextPreparationError::Invalid(format!(
                "missing routed bank {id:?}"
            )));
        };
        if selected_bank.owner_group != bank.owner_group
            || selected_bank.plan != expected_plan
            || selected_bank.catalog != bank.catalog
            || selected_bank.routes_per_token != bank.routes_per_token
            || selected_bank.routes_by_unit != bank.routes_by_unit
        {
            return Err(RoutedTextPreparationError::Invalid(format!(
                "selected bank {id:?} differs from admitted routed requirements"
            )));
        }
    }
    Ok(())
}

fn addressable_parameter_targets<O>(
    residency: eredu_runtime::ParameterBankResidency,
    _plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
) -> std::collections::BTreeSet<String>
where
    O: RoutedGroupedOperationValidation,
{
    collect_addressable_parameter_targets(residency, catalog.units().iter())
}

fn k2_horizon_args(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
) -> Option<&crate::k2_horizon::ModelArgs> {
    match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::K2Horizon(args) => Some(args),
            _ => None,
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::K2Horizon(args) => Some(args),
            _ => None,
        },
        _ => None,
    }
}

fn prepare_k2_horizon_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<crate::k2_horizon::LayeredModel<B>>, String>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| error.to_string())?;
    validate_selected_routed_handoff(&expected, &selected).map_err(|error| error.to_string())?;
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())?;
    let args = k2_horizon_args(inspection).ok_or("expected K2 Horizon configuration")?;
    let source_architecture = crate::replicated_text::selected_uses_transform(selected.text())
        .then(|| crate::k2_horizon::LayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| error.to_string())?;
    let selected_args = crate::replicated_text::selected_k2_horizon_args(args, selected.text())?;
    let identity = crate::k2_horizon::prompt_cache_architecture_fingerprint(&selected_args);
    let capability =
        crate::capability::k2_horizon(&selected_args).map_err(|error| error.to_string())?;
    let architecture = crate::k2_horizon::LayeredModel::<B>::new(selected_args, context)
        .map_err(|error| error.to_string())?;
    prepare_routed_architecture_handoff::<B, S, _>(
        architecture,
        source_architecture,
        expected,
        selected,
        capability,
        "k2_horizon".into(),
        identity,
        context,
    )
}

fn prepare_qwen_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::qwen::RoutedLayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::Qwen(args) if args.is_moe() => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::Qwen(args) if args.is_moe() => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::qwen::RoutedLayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_qwen_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::qwen::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::qwen::RoutedLayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::qwen(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

fn prepare_gpt_oss_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::gpt_oss::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::GptOss(args) => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::GptOss(args) => args,
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::gpt_oss::new_layered_model::<B>(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_gpt_oss_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::gpt_oss::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::gpt_oss::new_layered_model::<B>(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::gpt_oss(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

fn prepare_lfm2_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<crate::lfm2::LayeredModel<B>>, RoutedTextPreparationError>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::Lfm2(args)
                if args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::lfm2::LayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_lfm2_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::lfm2::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::lfm2::LayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::lfm2(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

fn prepare_kimi_linear_routed_text_architecture<B, S>(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::kimi_linear::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_runtime::RuntimeStateComponents<B> + eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(RoutedTextPreparationError::Invalid)?;
    let args = match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| crate::kimi_linear::LayeredModel::<B>::new(args.clone(), context))
        .transpose()
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let selected_args = crate::replicated_text::selected_kimi_linear_args(args, &text)
        .map_err(RoutedTextPreparationError::Invalid)?;
    let prompt_cache_architecture_identity =
        crate::kimi_linear::prompt_cache_architecture_fingerprint(&selected_args);
    let architecture = crate::kimi_linear::LayeredModel::<B>::new(selected_args, context)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let capability_estimate = crate::capability::kimi_linear(args)
        .map_err(|error| RoutedTextPreparationError::Invalid(error.to_string()))?;
    let targets =
        addressable_parameter_targets::<GatedProductOperation>(bank_residency, &plan, &catalog);
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            args.model_type.clone(),
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            context,
        )
        .map_err(RoutedTextPreparationError::Invalid)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

fn prepare_qwen_hybrid_routed_text_architecture<B, S>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::qwen::hybrid::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let inspection=source.routed_inspection();
    let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
    let retained=source::begin::<B>(&source,&selected,&store,context)?;
    if retained.is_none(){
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    }
    let args = match (
        source.architecture_plan().safetensors_architecture(),
        source.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                &args.text
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                &args.text
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let mut source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| {
            let args=config_source::source_config::<B,_>(&source,args,config_source::qwen_hybrid,context)?;
            crate::qwen::hybrid::LayeredModel::<B>::new_with_config(args,context)
        })
        .transpose()
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let selected_args=config_source::selected_config::<B,_,std::convert::Infallible>(
        &source,args,&text,config_source::qwen_hybrid,crate::replicated_text::selected_qwen_hybrid_args,|args|args.validate().map_err(|error|error.to_string()),context,
    ).map_err(source::preparation)?;
    let config_publication=config_source::publication::<B,_,std::convert::Infallible>(&source,&selected_args,&text,context)
        .map_err(source::preparation)?;
    let prompt_cache_architecture_identity =
        crate::qwen::hybrid::prompt_cache_architecture_fingerprint_with_metadata(&selected_args, metadata)
            .map_err(RoutedTextPreparationError::Metadata)?;
    let mut architecture = crate::qwen::hybrid::LayeredModel::<B>::new_with_config(selected_args, context)
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let capability_estimate=capability_source::prepare::<B,std::convert::Infallible,_>(
        &source,&text,context,||crate::capability::qwen_hybrid_text(args),
    ).map_err(source::preparation)?;
    let targets=source::targets(&retained,||addressable_parameter_targets::<GatedProductOperation>(bank_residency,plan,catalog));
    let parameters = source::parameters::<B, _>(
        &mut architecture, source_architecture.as_mut(), retained.as_ref(), &banks, context)?;
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable_metadata::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            metadata.text(&args.model_type).map_err(RoutedTextPreparationError::Metadata)?,
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            source::materialization(&retained),
            context,
        )
        .map_err(source::contract)?;
    config_publication.commit::<std::convert::Infallible>(prepared.selected()).map_err(source::preparation)?;
    capability_source::publish::<B,_,std::convert::Infallible>(&source,&prepared,context).map_err(source::preparation)?;
    source::publish::<B>(&source,prepared.selected(),&banks,&targets,&parameters,prepared.materialization_source(),context)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

fn prepare_deepseek_v3_routed_text_architecture<B, S>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<crate::deepseek::v3::Model<B>>, RoutedTextPreparationError>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::CompressedAttentionCache<B::Tensor>,
{
    let inspection=source.routed_inspection();
    let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
    let retained=source::begin::<B>(&source,&selected,&store,context)?;
    if retained.is_none(){
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    }
    let args = match (
        source.architecture_plan().safetensors_architecture(),
        source.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let mut source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| {
            let args=config_source::source_config::<B,_>(&source,args,config_source::deepseek_v3,context)?;
            crate::deepseek::v3::Model::<B>::new_with_config(args,context)
        })
        .transpose()
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let selected_args=config_source::selected_deepseek_v3::<B,std::convert::Infallible>(&source,args,&text,context)
        .map_err(source::preparation)?;
    let config_publication=config_source::publication::<B,_,std::convert::Infallible>(&source,&selected_args,&text,context)
        .map_err(source::preparation)?;
    let prompt_cache_architecture_identity =
        crate::deepseek::config::v3_architecture_fingerprint_with_metadata(&selected_args, metadata)
            .map_err(RoutedTextPreparationError::Metadata)?;
    let mut architecture = crate::deepseek::v3::Model::<B>::new_with_config(selected_args, context)
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let capability_estimate=capability_source::prepare::<B,std::convert::Infallible,_>(
        &source,&text,context,||crate::capability::deepseek_v3(args),
    ).map_err(source::preparation)?;
    let targets=source::targets(&retained,||addressable_parameter_targets::<GatedProductOperation>(bank_residency,plan,catalog));
    let parameters = source::parameters::<B, _>(
        &mut architecture, source_architecture.as_mut(), retained.as_ref(), &banks, context)?;
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable_metadata::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            metadata.text(&args.model_type).map_err(RoutedTextPreparationError::Metadata)?,
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            source::materialization(&retained),
            context,
        )
        .map_err(source::contract)?;
    config_publication.commit::<std::convert::Infallible>(prepared.selected()).map_err(source::preparation)?;
    capability_source::publish::<B,_,std::convert::Infallible>(&source,&prepared,context).map_err(source::preparation)?;
    source::publish::<B>(&source,prepared.selected(),&banks,&targets,&parameters,prepared.materialization_source(),context)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

fn prepare_deepseek_v4_routed_text_architecture<B, S>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<crate::deepseek::v4::Model<B>>, RoutedTextPreparationError>
where
    B: eredu_nn::HyperNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    let inspection=source.routed_inspection();
    let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
    let retained=source::begin::<B>(&source,&selected,&store,context)?;
    if retained.is_none(){
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    }
    let args = match (
        source.architecture_plan().safetensors_architecture(),
        source.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Gated(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let mut source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| {
            let args=config_source::source_config::<B,_>(&source,args,config_source::deepseek_v4,context)?;
            crate::deepseek::v4::Model::<B>::new_with_config(args,context)
        })
        .transpose()
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let selected_args=config_source::selected_deepseek_v4::<B,std::convert::Infallible>(&source,args,&text,context)
        .map_err(source::preparation)?;
    let config_publication=config_source::publication::<B,_,std::convert::Infallible>(&source,&selected_args,&text,context)
        .map_err(source::preparation)?;
    let prompt_cache_architecture_identity =
        crate::deepseek::config::v4_architecture_fingerprint_with_metadata(&selected_args, metadata)
            .map_err(RoutedTextPreparationError::Metadata)?;
    let mut architecture = crate::deepseek::v4::Model::<B>::new_with_config(selected_args, context)
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let capability_estimate=capability_source::prepare::<B,std::convert::Infallible,_>(
        &source,&text,context,||crate::capability::deepseek_v4(args),
    ).map_err(source::preparation)?;
    let targets=source::targets(&retained,||addressable_parameter_targets::<GatedProductOperation>(bank_residency,plan,catalog));
    let parameters = source::parameters::<B, _>(
        &mut architecture, source_architecture.as_mut(), retained.as_ref(), &banks, context)?;
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable_metadata::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            metadata.text(&args.model_type).map_err(RoutedTextPreparationError::Metadata)?,
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            source::materialization(&retained),
            context,
        )
        .map_err(source::contract)?;
    config_publication.commit::<std::convert::Infallible>(prepared.selected()).map_err(source::preparation)?;
    capability_source::publish::<B,_,std::convert::Infallible>(&source,&prepared,context).map_err(source::preparation)?;
    source::publish::<B>(&source,prepared.selected(),&banks,&targets,&parameters,prepared.materialization_source(),context)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

fn prepare_nemotron_h_routed_text_architecture<B, S>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    PreparedRoutedTextArchitecture<crate::nemotron_h::LayeredModel<B>>,
    RoutedTextPreparationError,
>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    let inspection=source.routed_inspection();
    let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
    let retained=source::begin::<B>(&source,&selected,&store,context)?;
    if retained.is_none(){
    let expected = routed_text_requirements(inspection).map_err(|error| match error {
        RoutedTextRequirementsError::Ineligible => RoutedTextPreparationError::Ineligible,
        RoutedTextRequirementsError::Invalid(detail) => RoutedTextPreparationError::Invalid(detail),
    })?;
    validate_selected_routed_handoff(&expected, &selected)?;
    }
    let args = match (
        source.architecture_plan().safetensors_architecture(),
        source.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                args
            }
            _ => return Err(RoutedTextPreparationError::Ineligible),
        },
        _ => return Err(RoutedTextPreparationError::Ineligible),
    };
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let SelectedRoutedBank { plan, catalog, .. } = &banks[&RoutedBankId::new(0)];
    let RoutedGroupedPlan::Relu2(plan) = plan else {
        return Err(RoutedTextPreparationError::Invalid(
            "selected grouped equation differs from the admitted architecture".into(),
        ));
    };
    let mut source_architecture = crate::replicated_text::selected_uses_transform(&text)
        .then(|| {
            let args=config_source::source_config::<B,_>(&source,args,config_source::nemotron_h,context)?;
            crate::nemotron_h::LayeredModel::<B>::new_with_config(args,context)
        })
        .transpose()
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let selected_args=config_source::selected_config::<B,_,std::convert::Infallible>(
        &source,args,&text,config_source::nemotron_h,crate::replicated_text::selected_nemotron_h_args,config_source::unchanged,context,
    ).map_err(source::preparation)?;
    let config_publication=config_source::publication::<B,_,std::convert::Infallible>(&source,&selected_args,&text,context)
        .map_err(source::preparation)?;
    let prompt_cache_architecture_identity =
        crate::nemotron_h::config::prompt_cache_architecture_fingerprint_with_metadata(&selected_args, metadata)
            .map_err(RoutedTextPreparationError::Metadata)?;
    let mut architecture = crate::nemotron_h::LayeredModel::<B>::new_with_config(selected_args, context)
        .map_err(|error| source::preparation(config_source::constructor_error::<B, std::convert::Infallible>(error, context)))?;
    let capability_estimate=capability_source::prepare::<B,std::convert::Infallible,_>(
        &source,&text,context,||crate::capability::nemotron_h(args),
    ).map_err(source::preparation)?;
    let targets=source::targets(&retained,||addressable_parameter_targets::<Relu2Operation>(bank_residency,plan,catalog));
    let parameters = source::parameters::<B, _>(
        &mut architecture, source_architecture.as_mut(), retained.as_ref(), &banks, context)?;
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable_metadata::<B, S, _>(
            architecture,
            source_architecture,
            text,
            capability_estimate,
            metadata.text(&args.model_type).map_err(RoutedTextPreparationError::Metadata)?,
            prompt_cache_architecture_identity,
            targets.iter().map(String::as_str),
            source::materialization(&retained),
            context,
        )
        .map_err(source::contract)?;
    config_publication.commit::<std::convert::Infallible>(prepared.selected()).map_err(source::preparation)?;
    capability_source::publish::<B,_,std::convert::Infallible>(&source,&prepared,context).map_err(source::preparation)?;
    source::publish::<B>(&source,prepared.selected(),&banks,&targets,&parameters,prepared.materialization_source(),context)?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

/// Receives one statically paired ReLU-squared routed architecture.
pub trait Relu2RoutedTextArchitectureVisitor<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Called immediately before architecture construction begins.
    fn construction_started(&mut self) {}

    /// Binds one architecture-owned prepared handoff to backend mechanisms.
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Constructs an admitted ReLU-squared routed family and invokes a generic visitor.
pub fn visit_relu2_routed_text_architecture<B, S, V>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: Relu2RoutedTextArchitectureVisitor<B, S>,
{
    visitor.construction_started();
    let prepared = prepare_nemotron_h_routed_text_architecture::<B, S>(
        &source,
        selected,
        store.clone(),
        context,
    )
    .map_err(RoutedTextDispatchError::preparation)?;
    visitor
        .visit(prepared, store)
        .map_err(RoutedTextDispatchError::Backend)
}

/// Receives one statically paired gated routed architecture.
pub trait RoutedTextArchitectureVisitor<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Called immediately before architecture construction begins.
    fn construction_started(&mut self) {}

    /// Binds one architecture-owned prepared handoff to backend mechanisms.
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Family-blind backend visitor for an exact routed prediction target.
pub trait RoutedPredictionTargetVisitor<B, S, M>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Called immediately before architecture construction begins.
    fn construction_started(&mut self) {}

    /// Receives a routed target only after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<B, S>
            + crate::prediction_extension::MaterializedPredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Supplies exact state-profile visitors while architecture dispatch selects
/// the admitted routed prediction family.
pub trait RoutedPredictionProfileDispatcher<B, M>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed construction output shared by both admitted profiles.
    type Output;
    /// Mechanism-binding failure shared by both admitted profiles.
    type Error;
    /// State selected for mixed-attention and compressed-latent execution.
    type GatedState: eredu_runtime::LayerRuntimeState<B>;
    /// State selected for pooling-attention DeepSeek-V4 execution.
    type PoolingState: eredu_runtime::LayerRuntimeState<B>;
    /// Exact visitor for mixed-attention and compressed-latent profiles.
    type GatedVisitor: RoutedPredictionTargetVisitor<
        B,
        Self::GatedState,
        M,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Exact visitor for the DeepSeek-V4 profile.
    type PoolingVisitor: RoutedPredictionTargetVisitor<
        B,
        Self::PoolingState,
        M,
        Output = Self::Output,
        Error = Self::Error,
    >;

    /// Consumes the dispatcher into its compressed-latent visitor.
    fn into_gated_visitor(self) -> Self::GatedVisitor;
    /// Consumes the dispatcher into its pooling-attention visitor.
    fn into_pooling_visitor(self) -> Self::PoolingVisitor;
}

/// Selects, constructs, and pairs an admitted routed prediction target
/// without requiring backend family inspection.
pub fn dispatch_routed_prediction_target_architecture<B, M, D>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    dispatcher: D,
) -> Result<D::Output, RoutedTextDispatchError<D::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    D: RoutedPredictionProfileDispatcher<B, M>,
    <D::GatedState as eredu_runtime::LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor>
            + eredu_runtime::RuntimeStateComponents<B>
            + eredu_nn::CompressedAttentionCache<B::Tensor>,
    <D::PoolingState as eredu_runtime::LayerRuntimeState<B>>::LayerState:
        eredu_nn::PoolingAttentionCache<B::Tensor>,
{
    let models = (
        source
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        source
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    );
    match models {
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(_))) => {
            visit_gated_routed_prediction_target_architecture::<B, D::GatedState, M, _>(
                &source,
                selected,
                extension,
                store,
                context,
                dispatcher.into_gated_visitor(),
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV4(_))) => {
            visit_pooling_routed_prediction_target_architecture::<B, D::PoolingState, M, _>(
                &source,
                selected,
                extension,
                store,
                context,
                dispatcher.into_pooling_visitor(),
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::QwenHybrid(_))) => {
            visit_qwen_hybrid_routed_prediction_target::<B,D::GatedState,M,_>(
                &source,selected,extension,store,context,dispatcher.into_gated_visitor(),
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(_))) => {
            visit_nemotron_h_routed_prediction_target::<B,D::GatedState,M,_>(
                &source,selected,extension,store,context,dispatcher.into_gated_visitor(),
            )
        }
        _ => Err(RoutedTextDispatchError::Architecture(
            "routed prediction target has no admitted family construction".into(),
        )),
    }
}

// Keep the other family's constructed model out of the selected dispatch frame.
#[inline(never)]
fn visit_qwen_hybrid_routed_prediction_target<B,S,M,V>(
    source:impl RoutedTextConstructionSource,selected:SelectedRoutedTextRealization,
    extension:crate::prediction_extension::MaterializedPredictionExtension<B,M>,
    store:eredu_checkpoint::store::RetainedCheckpointSource,
    context:&<B::Tensor as eredu_nn::Tensor>::Context,mut visitor:V,
)->Result<V::Output,RoutedTextDispatchError<V::Error>>
where B:eredu_nn::BlockwiseAttentionBackend+eredu_nn::DistributedNeuralBackend
    +eredu_nn::TensorParallelGroupedNeuralBackend+eredu_nn::HyperNeuralBackend,
    S:eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:eredu_nn::AttentionCache<B::Tensor>+eredu_runtime::RuntimeStateComponents<B>
        +eredu_nn::CompressedAttentionCache<B::Tensor>,
    M:crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V:RoutedPredictionTargetVisitor<B,S,M>,
{
    crate::decoder::ModuleMetadata::new::<B>(context).controls::<(
        SelectedRoutedTextRealization,crate::prediction_extension::MaterializedPredictionExtension<B,M>,
        eredu_checkpoint::store::RetainedCheckpointSource,
        &<B::Tensor as eredu_nn::Tensor>::Context,V,
        Result<V::Output,RoutedTextDispatchError<V::Error>>,
    )>().map_err(RoutedTextDispatchError::Metadata)?;
    visitor.construction_started();
    let prepared=prepare_qwen_hybrid_routed_text_architecture::<B,S>(
        &source,selected,store.clone(),context,
    ).map_err(RoutedTextDispatchError::preparation)?;
    visit_routed_prediction_target_architecture(prepared,extension,store,visitor)
}

// Keep the other family's constructed model out of the selected dispatch frame.
#[inline(never)]
fn visit_nemotron_h_routed_prediction_target<B,S,M,V>(
    source:impl RoutedTextConstructionSource,selected:SelectedRoutedTextRealization,
    extension:crate::prediction_extension::MaterializedPredictionExtension<B,M>,
    store:eredu_checkpoint::store::RetainedCheckpointSource,
    context:&<B::Tensor as eredu_nn::Tensor>::Context,mut visitor:V,
)->Result<V::Output,RoutedTextDispatchError<V::Error>>
where B:eredu_nn::BlockwiseAttentionBackend+eredu_nn::DistributedNeuralBackend
    +eredu_nn::TensorParallelGroupedNeuralBackend+eredu_nn::HyperNeuralBackend,
    S:eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:eredu_nn::AttentionCache<B::Tensor>+eredu_runtime::RuntimeStateComponents<B>
        +eredu_nn::CompressedAttentionCache<B::Tensor>,
    M:crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V:RoutedPredictionTargetVisitor<B,S,M>,
{
    crate::decoder::ModuleMetadata::new::<B>(context).controls::<(
        SelectedRoutedTextRealization,crate::prediction_extension::MaterializedPredictionExtension<B,M>,
        eredu_checkpoint::store::RetainedCheckpointSource,
        &<B::Tensor as eredu_nn::Tensor>::Context,V,
        Result<V::Output,RoutedTextDispatchError<V::Error>>,
    )>().map_err(RoutedTextDispatchError::Metadata)?;
    visitor.construction_started();
    let prepared=prepare_nemotron_h_routed_text_architecture::<B,S>(
        &source,selected,store.clone(),context,
    ).map_err(RoutedTextDispatchError::preparation)?;
    visit_routed_prediction_target_architecture(prepared,extension,store,visitor)
}

/// Constructs the admitted DeepSeek-V3 target, pairs its extension, and invokes generic mechanisms.
pub fn visit_gated_routed_prediction_target_architecture<B, S, M, V>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::CompressedAttentionCache<B::Tensor>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPredictionTargetVisitor<B, S, M>,
{
    visitor.construction_started();
    let prepared = prepare_deepseek_v3_routed_text_architecture::<B, S>(
        &source,
        selected,
        store.clone(),
        context,
    )
    .map_err(RoutedTextDispatchError::preparation)?;
    visit_routed_prediction_target_architecture(prepared, extension, store, visitor)
}

/// Constructs the admitted DeepSeek-V4 target, pairs its extension, and invokes generic mechanisms.
pub fn visit_pooling_routed_prediction_target_architecture<B, S, M, V>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPredictionTargetVisitor<B, S, M>,
{
    visitor.construction_started();
    let prepared = prepare_deepseek_v4_routed_text_architecture::<B, S>(
        &source,
        selected,
        store.clone(),
        context,
    )
    .map_err(RoutedTextDispatchError::preparation)?;
    visit_routed_prediction_target_architecture(prepared, extension, store, visitor)
}

/// Pairs one materialized extension with its exact routed target before backend erasure.
pub fn visit_routed_prediction_target_architecture<B, S, M, A, V>(
    prepared: PreparedRoutedTextArchitecture<A>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>
        + crate::prediction_extension::MaterializedPredictionTarget<B>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    V: RoutedPredictionTargetVisitor<B, S, M>,
{
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| RoutedTextDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension, store)
        .map_err(RoutedTextDispatchError::Backend)
}

/// Constructs an admitted pooling-attention routed family and invokes generic mechanisms.
pub fn visit_pooling_routed_text_architecture<B, S, V>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::HyperNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    V: RoutedTextArchitectureVisitor<B, S>,
{
    visitor.construction_started();
    let prepared = prepare_deepseek_v4_routed_text_architecture::<B, S>(
        &source,
        selected,
        store.clone(),
        context,
    )
    .map_err(RoutedTextDispatchError::preparation)?;
    visitor
        .visit(prepared, store)
        .map_err(RoutedTextDispatchError::Backend)
}

/// Constructs the exact admitted gated routed family and invokes a generic mechanism visitor.
pub fn visit_routed_text_architecture<B, S, V>(
    source: impl RoutedTextConstructionSource,
    selected: SelectedRoutedTextRealization,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, RoutedTextDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::CompressedAttentionCache<B::Tensor>,
    V: RoutedTextArchitectureVisitor<B, S>,
{
    let inspection = source.routed_inspection();
    visitor.construction_started();
    if k2_horizon_args(inspection).is_some() {
        let prepared = prepare_k2_horizon_routed_text_architecture::<B, S>(
            inspection,
            selected,
            store.clone(),
            context,
        )
        .map_err(RoutedTextDispatchError::Architecture)?;
        return visitor
            .visit(prepared, store)
            .map_err(RoutedTextDispatchError::Backend);
    }

    match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::Qwen(args) if args.is_moe()
            ) =>
        {
            prepare_qwen_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::Qwen(args) if args.is_moe()
            ) =>
        {
            prepare_qwen_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::GptOss(_)
            ) =>
        {
            prepare_gpt_oss_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::GptOss(_)
            ) =>
        {
            prepare_gpt_oss_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::Lfm2(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_lfm2_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::Lfm2(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_lfm2_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::KimiLinear(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_kimi_linear_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::KimiLinear(args)
                    if args.has_sparse_moe_layers()
            ) =>
        {
            prepare_kimi_linear_routed_text_architecture::<B, S>(
                inspection,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::QwenHybrid(args)
                    if args.vision.is_none()
                        && args.text.mtp_num_hidden_layers == 0
                        && args.text.is_moe()
            ) =>
        {
            prepare_qwen_hybrid_routed_text_architecture::<B, S>(
                &source,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::QwenHybrid(args)
                    if args.vision.is_none()
                        && args.text.mtp_num_hidden_layers == 0
                        && args.text.is_moe()
            ) =>
        {
            prepare_qwen_hybrid_routed_text_architecture::<B, S>(
                &source,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (Some(plan), None)
            if matches!(
                plan.model(),
                crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)
                    if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers()
            ) =>
        {
            prepare_deepseek_v3_routed_text_architecture::<B, S>(
                &source,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        (None, Some(plan))
            if matches!(
                plan.model(),
                crate::configuration::GgufModelConfig::DeepSeekV3(args)
                    if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers()
            ) =>
        {
            prepare_deepseek_v3_routed_text_architecture::<B, S>(
                &source,
                selected,
                store.clone(),
                context,
            )
            .map_err(RoutedTextDispatchError::preparation)
            .and_then(|prepared| {
                visitor
                    .visit(prepared, store)
                    .map_err(RoutedTextDispatchError::Backend)
            })
        }
        _ => Err(RoutedTextDispatchError::Architecture(
            RoutedTextPreparationError::Ineligible.to_string(),
        )),
    }
}

/// Failure while architecture-owned routed dispatch invokes a mechanism visitor.
#[derive(Debug, thiserror::Error)]
pub enum RoutedTextDispatchError<E> {
    /// A participating host constructor preserves its exact failure.
    #[error(transparent)]
    Metadata(eredu_nn::Error),
    /// Architecture admission and selected realization disagreed.
    #[error("{0}")]
    Architecture(String),
    /// Backend mechanism binding failed.
    #[error("routed text mechanism binding failed: {0}")]
    Backend(E),
}

impl<E> RoutedTextDispatchError<E>{
    fn preparation(error:RoutedTextPreparationError)->Self{
        match error{
            RoutedTextPreparationError::Metadata(cause)=>Self::Metadata(cause),
            cause=>Self::Architecture(cause.to_string()),
        }
    }
}

/// Requirements of one independently identified routed bank.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedBankRequirements {
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: RoutedGroupedPlan,
    catalog: ExpertResidencyCatalog,
    routes_per_token: usize,
    routes_by_unit: BTreeMap<usize, usize>,
}

impl RoutedBankRequirements {
    /// Execution group owning the bank's logical invocations.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }
    /// Equation, ownership, and rank-local geometry.
    pub const fn plan(&self) -> &RoutedGroupedPlan {
        &self.plan
    }
    /// Independently addressable expert recipes.
    pub const fn catalog(&self) -> &ExpertResidencyCatalog {
        &self.catalog
    }
    /// Maximum selected members in one row of this bank.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }
    /// Route cardinalities keyed by logical unit.
    pub const fn routes_by_unit(&self) -> &BTreeMap<usize, usize> {
        &self.routes_by_unit
    }
    fn routes_for_unit(&self, unit: usize) -> Option<usize> {
        self.routes_by_unit.get(&unit).copied()
    }
}

/// Exact architecture and artifact requirements for replicated routed text.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedTextRequirements {
    text: eredu_runtime::ReplicatedTextRequirements,
    banks: BTreeMap<RoutedBankId, RoutedBankRequirements>,
}

impl RoutedTextRequirements {
    pub(crate) fn with_state_layout(
        mut self,
        layout: eredu_runtime::StateLayout,
    ) -> Result<Self, eredu_runtime::ReplicatedTextContractError> {
        self.text = self.text.with_state_layout(layout)?;
        Ok(self)
    }

    /// Shared text and state requirements.
    pub const fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        &self.text
    }
    /// All independently identified bank requirements.
    pub const fn banks(&self) -> &BTreeMap<RoutedBankId, RoutedBankRequirements> {
        &self.banks
    }
    /// One bank's exact equation and source requirements.
    pub fn bank(&self, id: RoutedBankId) -> Option<&RoutedBankRequirements> {
        self.banks.get(&id)
    }
    /// Maximum route cardinality of one invocation, for reusable transport capacity.
    pub fn routes_per_token(&self) -> usize {
        self.banks
            .values()
            .map(RoutedBankRequirements::routes_per_token)
            .max()
            .unwrap_or(0)
    }
}

fn uniform_routes_by_unit<S>(
    plan: &ExpertRealizationPlan<S>,
    routes_per_token: usize,
) -> BTreeMap<usize, usize> {
    plan.unit_specs()
        .keys()
        .map(|(_, unit)| (*unit, routes_per_token))
        .collect()
}

fn validate_routes_by_unit<O>(
    plan: &ExpertRealizationPlan<O::Spec>,
    routes_by_unit: &BTreeMap<usize, usize>,
) -> Result<(), RoutedTextRequirementsError>
where
    O: RoutedGroupedOperationValidation,
{
    let planned = plan
        .unit_specs()
        .keys()
        .map(|(_, unit)| *unit)
        .collect::<std::collections::BTreeSet<_>>();
    let routed = routes_by_unit
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if planned != routed {
        return Err(RoutedTextRequirementsError::Invalid(format!(
            "route-cardinality units {routed:?} differ from grouped-bank units {planned:?}"
        )));
    }
    for ((_, unit), spec) in plan.unit_specs() {
        let routes = routes_by_unit[unit];
        let groups = usize::try_from(O::group_count(spec)).unwrap_or_default();
        if routes == 0 || routes > groups {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "grouped-bank unit {unit} selects {routes} routes from {groups} groups"
            )));
        }
    }
    Ok(())
}

pub(crate) fn gated_routed_text_requirements(
    text: eredu_runtime::ReplicatedTextRequirements,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    catalog: ExpertResidencyCatalog,
    routes_per_token: usize,
    recipe_source: &(impl eredu_checkpoint::recipe::RecipeCatalog + ?Sized),
) -> Result<RoutedTextRequirements, RoutedTextRequirementsError> {
    let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
    gated_routed_text_requirements_with_routes(
        text,
        owner_group,
        plan,
        catalog,
        routes_by_unit,
        recipe_source,
    )
}

pub(crate) fn gated_routed_text_requirements_with_routes(
    text: eredu_runtime::ReplicatedTextRequirements,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    catalog: ExpertResidencyCatalog,
    routes_by_unit: BTreeMap<usize, usize>,
    recipe_source: &(impl eredu_checkpoint::recipe::RecipeCatalog + ?Sized),
) -> Result<RoutedTextRequirements, RoutedTextRequirementsError> {
    let bank = RoutedBankRequirements::new(owner_group, plan.into(), catalog, routes_by_unit)?;
    RoutedTextRequirements::new(text, [(RoutedBankId::new(0), bank)], recipe_source)
}

fn addressable_bank_parameter_targets(
    bank_residency: &eredu_runtime::ParameterBankResidency,
    banks: &RetainedRoutedBanks,
) -> BTreeSet<String> {
    collect_addressable_parameter_targets(*bank_residency, banks.values().flat_map(|bank| bank.catalog.units()))
}

fn collect_addressable_parameter_targets<'a>(
    residency: eredu_runtime::ParameterBankResidency,
    units: impl Iterator<Item = &'a crate::ExpertResidencyUnit>,
) -> BTreeSet<String> {
    if matches!(residency, eredu_runtime::ParameterBankResidency::IndependentCache(_)) {
        units.filter(|unit| unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel)
            .flat_map(crate::ExpertResidencyUnit::parameters)
            .map(|parameter| parameter.logical_target().to_owned()).collect()
    } else { BTreeSet::new() }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_routed_architecture_handoff<B, S, A>(
    architecture: A,
    source_architecture: Option<A>,
    expected: RoutedTextRequirements,
    selected: SelectedRoutedTextRealization,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    prompt_cache_architecture_identity: String,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<A>, String>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>,
    A::StaticModules: Clone,
{
    validate_selected_routed_handoff(&expected, &selected).map_err(|error| error.to_string())?;
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let targets = addressable_bank_parameter_targets(&bank_residency, &banks);
    let prepared = crate::replicated_text::prepare_architecture_handoff_with_addressable::<B, S, _>(
        architecture,
        source_architecture,
        text,
        capability_estimate,
        effective_model_type,
        prompt_cache_architecture_identity,
        targets.iter().map(String::as_str),
        context,
    )?;
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}

/// Caller policy for one replicated routed text session.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoutedTextSelectionRequest {
    text: eredu_runtime::ReplicatedTextSelectionRequest,
    weights: eredu_runtime::WeightResidency,
}

impl RoutedTextSelectionRequest {
    /// Pairs shared text-session policy with ordinary and banked weight placement.
    pub fn new(
        text: eredu_runtime::ReplicatedTextSelectionRequest,
        weights: eredu_runtime::WeightResidency,
    ) -> Result<Self, RoutedTextSelectionError> {
        if text.residency() != weights.layers() {
            return Err(RoutedTextSelectionError {
                issues: vec![
                    "ordinary weight residency differs between text and banked policy".into(),
                ],
            });
        }
        if let Some(options) = weights.parameter_bank_cache() {
            options
                .validate()
                .map_err(|error| RoutedTextSelectionError {
                    issues: vec![error.to_string()],
                })?;
        }
        Ok(Self { text, weights })
    }

    /// Returns shared state, transform, session, topology, and completion policy.
    pub const fn text(&self) -> &eredu_runtime::ReplicatedTextSelectionRequest {
        &self.text
    }

    /// Returns ordinary and independently addressable weight placement.
    pub const fn weights(&self) -> eredu_runtime::WeightResidency {
        self.weights
    }
}

/// Selected geometry and source tasks for one routed bank.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedRoutedBank {
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: RoutedGroupedPlan,
    catalog: ExpertResidencyCatalog,
    addressable_members: Vec<eredu_runtime::AddressableBankMember>,
    routes_per_token: usize,
    routes_by_unit: BTreeMap<usize, usize>,
    partition_unit_coordinates:
        BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>,
}

impl SelectedRoutedBank {
    pub(crate) fn with_partition_geometry(
        &self,
        plan: RoutedGroupedPlan,
        catalog: ExpertResidencyCatalog,
        addressable_members: Vec<eredu_runtime::AddressableBankMember>,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Self, crate::component_partition::ComponentPartitionError> {
        let plan = plan
            .with_catalog_distribution(&self.catalog)
            .map_err(crate::component_partition::ComponentPartitionError::ParameterLayout)?;
        let partition_unit_coordinates = crate::component_partition::derive_bank_unit_coordinates(
            &self.plan,
            &plan,
            &self.catalog,
            layout,
        )?;
        Ok(Self {
            plan,
            catalog,
            addressable_members,
            partition_unit_coordinates,
            ..self.clone()
        })
    }
    /// Execution group owning this bank.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }
    /// Selected equation and ownership map.
    pub const fn plan(&self) -> &RoutedGroupedPlan {
        &self.plan
    }
    /// Atomic recipe catalog.
    pub const fn catalog(&self) -> &ExpertResidencyCatalog {
        &self.catalog
    }
    /// Exact selected materialization tasks for this bank's storage.
    pub fn addressable_members(&self) -> &[eredu_runtime::AddressableBankMember] {
        &self.addressable_members
    }
    /// Maximum selected members for one row.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }
    /// Exact route counts per logical unit.
    pub const fn routes_by_unit(&self) -> &BTreeMap<usize, usize> {
        &self.routes_by_unit
    }
    /// Scalar coordinates retained with localized grouped construction. These
    /// describe bank storage; execution-group ownership still decides invocation.
    pub fn partition_unit_coordinates(
        &self,
    ) -> &BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap> {
        &self.partition_unit_coordinates
    }
}

/// Authoritative routed realization selected before architecture construction.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedRoutedTextRealization {
    text: eredu_runtime::SelectedReplicatedTextRealization,
    bank_residency: eredu_runtime::ParameterBankResidency,
    banks: RetainedRoutedBanks,
}

impl SelectedRoutedTextRealization {
    /// Selected shared text-session realization.
    pub const fn text(&self) -> &eredu_runtime::SelectedReplicatedTextRealization {
        &self.text
    }
    /// Placement policy applied independently to each bank.
    pub const fn bank_residency(&self) -> eredu_runtime::ParameterBankResidency {
        self.bank_residency
    }
    /// All selected banks, preserving their independent identities.
    pub fn banks(&self) -> &BTreeMap<RoutedBankId, SelectedRoutedBank> {
        &self.banks
    }
    /// One selected bank's complete construction facts.
    pub fn bank(&self, id: RoutedBankId) -> Option<&SelectedRoutedBank> {
        self.banks.get(&id)
    }
    /// Maximum selected members of any one invocation.
    pub fn routes_per_token(&self) -> usize {
        self.banks
            .values()
            .map(SelectedRoutedBank::routes_per_token)
            .max()
            .unwrap_or(0)
    }
    /// Consumes the selection without losing bank identity.
    pub fn into_parts(
        self,
    ) -> (
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ParameterBankResidency,
        BTreeMap<RoutedBankId, SelectedRoutedBank>,
    ) {
        (self.text, self.bank_residency, self.banks.into_owned())
    }
    pub(crate) fn into_shared_parts(self)->(
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ParameterBankResidency,RetainedRoutedBanks,
    ){(self.text,self.bank_residency,self.banks)}
}

/// Complete fail-closed routed selection diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("replicated routed text realization is unsupported: {issues}", issues = .issues.join("; "))]
pub struct RoutedTextSelectionError {
    issues: Vec<String>,
}

impl RoutedTextSelectionError {
    /// Returns every missing mechanism in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

fn maximum_selected_compact_bytes(
    requirements: &RoutedBankRequirements,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<u64, String> {
    let tasks = index_materialization_tasks(selected.materialization_tasks());
    let mut by_unit = BTreeMap::<(String, usize), Vec<u64>>::new();
    for unit in requirements.catalog().units() {
        let (_, selected_bytes) = selected_member_geometry(unit, &tasks)?;
        by_unit
            .entry((
                unit.owner_group().as_str().to_owned(),
                unit.identity().unit(),
            ))
            .or_default()
            .push(selected_bytes);
    }
    let mut maximum = 0u64;
    for ((group, unit), mut members) in by_unit {
        let routes = requirements.routes_for_unit(unit).ok_or_else(|| {
            format!("routed unit {group:?}/{unit} has no admitted route cardinality")
        })?;
        if members.len() < routes {
            return Err(format!(
                "routed unit {group:?}/{unit} has {} bank members for {} routes per token",
                members.len(),
                routes
            ));
        }
        members.sort_unstable_by(|left, right| right.cmp(left));
        let bytes = members
            .into_iter()
            .take(routes)
            .try_fold(0u64, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| "per-row compact-bank byte geometry overflowed".to_owned())?;
        maximum = maximum.max(bytes);
    }
    if maximum == 0 {
        return Err("routed requirements contain no compact-bank geometry".into());
    }
    Ok(maximum)
}

/// Selects one routed realization without constructing modules or opening payloads.
pub fn select_routed_text_realization(
    requirements: &RoutedTextRequirements,
    request: &RoutedTextSelectionRequest,
    capabilities: &eredu_runtime::BackendMechanismCapabilities,
) -> Result<SelectedRoutedTextRealization, RoutedTextSelectionError> {
    let text = eredu_runtime::select_replicated_text_realization(
        requirements.text(),
        request.text(),
        capabilities,
    );
    let bank_residency = request.weights().parameter_banks();
    let mut issues = text
        .as_ref()
        .err()
        .map(|error| error.issues().to_vec())
        .unwrap_or_default();
    if let eredu_runtime::ParameterBankResidency::IndependentCache(options) = bank_residency {
        if !capabilities.indexed_movement() {
            issues.push("indexed selection and movement".into());
        }
        match capabilities.addressable_storage() {
            Some(storage) => {
                if !storage.bulk_access() {
                    issues.push("addressable bulk access".into());
                }
                if !storage.incremental_access() {
                    issues.push("addressable incremental access".into());
                }
                if !storage.lease_completion() {
                    issues.push("addressable lease completion".into());
                }
                if !storage.tiers().disk() {
                    issues.push("addressable disk storage".into());
                }
                if options.offload().host_budget_bytes() != Some(0) && !storage.tiers().host() {
                    issues.push("addressable host storage".into());
                }
                if !storage.tiers().device() {
                    issues.push("addressable device storage".into());
                }
                if options.compact_bank_scratch_bytes() > storage.maximum_compact_bytes() {
                    issues.push(format!(
                        "compact bank limit {} exceeds backend maximum {}",
                        options.compact_bank_scratch_bytes(),
                        storage.maximum_compact_bytes()
                    ));
                }
            }
            None => issues.push("independently addressable storage".into()),
        }
        if let Ok(selected_text) = &text {
            for (id, bank) in &requirements.banks {
                match maximum_selected_compact_bytes(bank, selected_text) {
                    Ok(bytes) if bytes > options.compact_bank_scratch_bytes() => issues.push(format!(
                        "routed bank {id:?} requires {bytes} selected compact-bank bytes for {} routes, exceeding limit {}",
                        bank.routes_per_token(), options.compact_bank_scratch_bytes()
                    )),
                    Ok(_) => {}
                    Err(issue) => issues.push(format!("routed bank {id:?}: {issue}")),
                }
            }
        }
    }
    if !issues.is_empty() {
        return Err(RoutedTextSelectionError { issues });
    }
    let text = text.expect("an empty diagnostic implies successful text selection");
    let banks =
        requirements
            .banks
            .iter()
            .map(|(id, bank)| {
                let plan = select_grouped_formats(bank.plan.clone(), &text).map_err(|issue| {
                    RoutedTextSelectionError {
                        issues: vec![format!("bank {id:?}: {issue}")],
                    }
                })?;
                let addressable_members = project_addressable_members(&bank.catalog, &text)
                    .map_err(|error| RoutedTextSelectionError {
                        issues: vec![format!("bank {id:?}: {error}")],
                    })?;
                Ok((
                    *id,
                    SelectedRoutedBank {
                        owner_group: bank.owner_group.clone(),
                        plan,
                        catalog: bank.catalog.clone(),
                        addressable_members,
                        routes_per_token: bank.routes_per_token,
                        routes_by_unit: bank.routes_by_unit.clone(),
                        partition_unit_coordinates: BTreeMap::new(),
                    },
                ))
            })
            .collect::<Result<_, RoutedTextSelectionError>>()?;
    Ok(SelectedRoutedTextRealization {
        text,
        bank_residency,
        banks:RetainedRoutedBanks::new(banks),
    })
}

fn select_grouped_formats(
    plan: RoutedGroupedPlan,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<RoutedGroupedPlan, String> {
    match plan {
        RoutedGroupedPlan::Linear(plan) => plan
            .try_map_unit_specs(|spec| {
                let projection = select_projection_format(spec.projection(), text)?;
                eredu_nn::GroupedLinearSpec::new(
                    spec.group_count(),
                    spec.input_dimensions(),
                    spec.global_output_dimensions(),
                    spec.activation(),
                    projection,
                )
                .map(|selected| selected.with_reduction(spec.reduction()))
                .and_then(|selected| selected.partition_output(spec.output_range()))
                .map_err(|error| error.to_string())
            })
            .map(RoutedGroupedPlan::Linear),
        RoutedGroupedPlan::Gated(plan) => plan
            .try_map_unit_specs(|spec| select_gated_formats(spec, text))
            .map(RoutedGroupedPlan::Gated),
        RoutedGroupedPlan::Relu2(plan) => plan
            .try_map_unit_specs(|spec| select_relu2_formats(spec, text))
            .map(RoutedGroupedPlan::Relu2),
    }
}

fn select_projection_format(
    projection: &eredu_nn::GroupedProjectionSpec,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<eredu_nn::GroupedProjectionSpec, String> {
    let weight = projection.weight().id.to_string();
    let executable = text
        .parameters()
        .iter()
        .find(|parameter| parameter.name() == weight)
        .map(eredu_runtime::SelectedParameterRealization::executable)
        .ok_or_else(|| format!("grouped projection {weight:?} has no selected realization"))?;
    if executable == projection.format().encoding() {
        return Ok(projection.clone());
    }
    eredu_nn::GroupedProjectionSpec::new(
        projection.weight().clone(),
        projection.bias().cloned(),
        if weight.ends_with(".weight") {
            crate::linear_format::standard_linear_format(&weight, executable)
        } else {
            crate::linear_format::standard_expert_format(&weight, executable)
        }
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn select_gated_formats(
    spec: eredu_nn::GroupedGatedProductSpec,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<eredu_nn::GroupedGatedProductSpec, String> {
    let layout = match spec.layout() {
        eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
            eredu_nn::GatedProductGroupLayout::Packed {
                gate_up: select_projection_format(gate_up, text)?,
                down: select_projection_format(down, text)?,
            }
        }
        eredu_nn::GatedProductGroupLayout::Independent(groups) => {
            eredu_nn::GatedProductGroupLayout::Independent(
                groups
                    .iter()
                    .map(|group| {
                        Ok(eredu_nn::GatedProductGroupParameters::new(
                            select_projection_format(group.gate(), text)?,
                            select_projection_format(group.up(), text)?,
                            select_projection_format(group.down(), text)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            )
        }
        _ => return Err("unsupported grouped gated-product layout".into()),
    };
    eredu_nn::GroupedGatedProductSpec::new(
        spec.group_count(),
        spec.input_dimensions(),
        spec.intermediate_dimensions(),
        spec.output_dimensions(),
        spec.policy(),
        layout,
    )
    .map(|selected| selected.with_reduction(spec.reduction()))
    .map_err(|error| error.to_string())
}

fn select_relu2_formats(
    spec: eredu_nn::GroupedRelu2Spec,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<eredu_nn::GroupedRelu2Spec, String> {
    eredu_nn::GroupedRelu2Spec::new(
        spec.group_count(),
        spec.hidden_dimensions(),
        spec.intermediate_dimensions(),
        select_projection_format(spec.up(), text)?,
        select_projection_format(spec.down(), text)?,
    )
    .map_err(|error| error.to_string())
}

/// Failure while deriving routed requirements from an admitted artifact.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedTextRequirementsError {
    /// The admitted graph is outside replicated routed text execution.
    #[error("architecture is not an eligible replicated routed text graph")]
    Ineligible,
    /// Artifact provenance, architecture geometry, or routed topology is invalid.
    #[error("invalid replicated routed text requirements: {0}")]
    Invalid(String),
}

/// Derives routed requirements entirely from admitted architecture facts.
///
/// Backend support and caller residency policy do not participate in this step.
pub fn routed_text_requirements(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
) -> Result<RoutedTextRequirements, RoutedTextRequirementsError> {
    inspection
        .architecture_plan()
        .validation(inspection.admission_token())
        .routed
        .get_or_init(|| derive_routed_text_requirements(inspection))
        .clone()
}

fn derive_routed_text_requirements(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
) -> Result<RoutedTextRequirements, RoutedTextRequirementsError> {
    let architecture = inspection.architecture_plan();
    if architecture.has_processor() || architecture.gguf_media_projector().is_some() {
        return Err(RoutedTextRequirementsError::Ineligible);
    }
    if let Some(args) = k2_horizon_args(inspection) {
        if !args.is_moe() {
            return Err(RoutedTextRequirementsError::Ineligible);
        }
        let invalid = |e: String| RoutedTextRequirementsError::Invalid(e);
        let text =
            crate::replicated_text::k2_horizon_replicated_text_requirements(inspection, args)
                .map_err(|e| invalid(e.to_string()))?;
        let source = crate::replicated_text::inspection_recipe_source(inspection)
            .map_err(|e| invalid(e.to_string()))?;
        let rank = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(|e| invalid(e.to_string()))?,
            0,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let plans = crate::k2_horizon::expert_realization_plans(args, rank, None)
            .map_err(|e| invalid(e.to_string()))?;
        let owner =
            eredu_runtime::ExecutionGroupId::new(crate::decoder::TEXT_DECODER_EXECUTION_GROUP)
                .map_err(|e| invalid(e.to_string()))?;
        let mut banks = Vec::new();
        for bank in [
            crate::k2_horizon::ExpertBank::FeedForward,
            crate::k2_horizon::ExpertBank::AttentionValue,
        ] {
            let Some(plan) = plans.get(&bank.id()) else {
                continue;
            };
            let routes = match bank {
                crate::k2_horizon::ExpertBank::FeedForward => args.num_experts_per_tok,
                crate::k2_horizon::ExpertBank::AttentionValue => args.mova_num_experts_per_tok,
            } as usize;
            let routes_by_unit = match plan {
                RoutedGroupedPlan::Gated(plan) => uniform_routes_by_unit(plan, routes),
                RoutedGroupedPlan::Linear(plan) => uniform_routes_by_unit(plan, routes),
                RoutedGroupedPlan::Relu2(_) => unreachable!("K2 has SwiGLU feed-forward banks"),
            };
            let catalog = crate::k2_horizon::expert_residency_catalog(source.as_ref(), args, bank)
                .map_err(invalid)?;
            banks.push((
                bank.id(),
                RoutedBankRequirements::new(owner.clone(), plan.clone(), catalog, routes_by_unit)?,
            ));
        }
        return RoutedTextRequirements::new(text, banks, source.as_ref());
    }
    enum Family<'a> {
        Qwen(&'a crate::qwen::ModelArgs),
        GptOss(&'a crate::gpt_oss::ModelArgs),
        NemotronH(&'a crate::nemotron_h::ModelArgs),
        Lfm2(&'a crate::lfm2::ModelArgs),
        KimiLinear(&'a crate::kimi_linear::ModelArgs),
        QwenHybrid(&'a crate::qwen::hybrid::HybridConfig),
        DeepSeekV3(&'a crate::deepseek::V3Args),
        DeepSeekV4(&'a crate::deepseek::V4Args),
    }
    let family = match (
        architecture.safetensors_architecture(),
        architecture.gguf_plan(),
    ) {
        (Some(plan), None) => match plan.model() {
            crate::configuration::SafetensorsModelConfig::Qwen(args) if args.is_moe() => {
                Family::Qwen(args)
            }
            crate::configuration::SafetensorsModelConfig::GptOss(args) => Family::GptOss(args),
            crate::configuration::SafetensorsModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::NemotronH(args)
            }
            crate::configuration::SafetensorsModelConfig::Lfm2(args)
                if args.has_sparse_moe_layers() =>
            {
                Family::Lfm2(args)
            }
            crate::configuration::SafetensorsModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                Family::KimiLinear(args)
            }
            crate::configuration::SafetensorsModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                Family::QwenHybrid(&args.text)
            }
            crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::DeepSeekV3(args)
            }
            crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                Family::DeepSeekV4(args)
            }
            _ => return Err(RoutedTextRequirementsError::Ineligible),
        },
        (None, Some(plan)) => match plan.model() {
            crate::configuration::GgufModelConfig::Qwen(args) if args.is_moe() => {
                Family::Qwen(args)
            }
            crate::configuration::GgufModelConfig::GptOss(args) => Family::GptOss(args),
            crate::configuration::GgufModelConfig::NemotronH(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::NemotronH(args)
            }
            crate::configuration::GgufModelConfig::Lfm2(args) if args.has_sparse_moe_layers() => {
                Family::Lfm2(args)
            }
            crate::configuration::GgufModelConfig::KimiLinear(args)
                if args.has_sparse_moe_layers() =>
            {
                Family::KimiLinear(args)
            }
            crate::configuration::GgufModelConfig::QwenHybrid(args)
                if args.vision.is_none()
                    && args.text.mtp_num_hidden_layers == 0
                    && args.text.is_moe() =>
            {
                Family::QwenHybrid(&args.text)
            }
            crate::configuration::GgufModelConfig::DeepSeekV3(args)
                if args.num_nextn_predict_layers == 0 && args.has_sparse_moe_layers() =>
            {
                Family::DeepSeekV3(args)
            }
            crate::configuration::GgufModelConfig::DeepSeekV4(args)
                if args.num_nextn_predict_layers == 0 =>
            {
                Family::DeepSeekV4(args)
            }
            _ => return Err(RoutedTextRequirementsError::Ineligible),
        },
        _ => return Err(RoutedTextRequirementsError::Ineligible),
    };
    let routes_per_token = usize::try_from(match &family {
        Family::Qwen(args) => args.num_experts_per_tok,
        Family::GptOss(args) => args.num_experts_per_tok,
        Family::NemotronH(args) => args.num_experts_per_tok,
        Family::Lfm2(args) => args.num_experts_per_tok,
        Family::KimiLinear(args) => args.num_experts_per_token,
        Family::QwenHybrid(args) => args.num_experts_per_tok,
        Family::DeepSeekV3(args) => args.num_experts_per_tok,
        Family::DeepSeekV4(args) => args.num_experts_per_tok,
    })
    .ok()
    .filter(|routes| *routes > 0)
    .ok_or_else(|| {
        RoutedTextRequirementsError::Invalid(
            "routed architecture has no positive routes-per-token cardinality".into(),
        )
    })?;
    let expected_routed_units = match &family {
        Family::Qwen(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| {
                (
                    ("text_decoder".to_owned(), unit),
                    format!("{}.layers.{unit}", args.parameter_root),
                )
            })
            .collect(),
        Family::GptOss(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| {
                (
                    ("text_decoder".to_owned(), unit),
                    format!("{}.layers.{unit}", args.parameter_root),
                )
            })
            .collect(),
        Family::NemotronH(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| **policy == crate::nemotron_h::LayerPolicy::SparseMoe)
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::Lfm2(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| policy.feed_forward == crate::lfm2::FeedForwardPolicy::SparseMoe)
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::KimiLinear(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| {
                policy.feed_forward == crate::kimi_linear::FeedForwardPolicy::SparseMoe
            })
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::QwenHybrid(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::DeepSeekV3(args) => args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| **policy == crate::deepseek::LayerPolicy::SparseMoe)
            .map(|(unit, _)| (("target".to_owned(), unit), format!("model.layers.{unit}")))
            .collect(),
        Family::DeepSeekV4(args) => (0..usize::try_from(args.num_hidden_layers).unwrap_or(0))
            .map(|unit| (("target".to_owned(), unit), format!("layers.{unit}")))
            .collect(),
    };
    let recipe_source = crate::replicated_text::inspection_recipe_source(inspection)
        .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
    if let Family::NemotronH(args) = family {
        let text =
            crate::replicated_text::nemotron_h_replicated_text_requirements(inspection, args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?
                .with_grouped_operations([eredu_runtime::GroupedOperationRequirement::Relu2]);
        let plan = crate::nemotron_h::replicated_expert_realization_plan(args)
            .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
        let catalog = crate::nemotron_h::expert_residency_catalog(recipe_source.as_ref(), args)
            .map_err(RoutedTextRequirementsError::Invalid)?;
        let owner_group = eredu_runtime::ExecutionGroupId::new("target")
            .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
        validate_expected_routed_schedule(&expected_routed_units, &plan, &catalog)?;
        validate_plan_catalog::<Relu2Operation>(&owner_group, &plan, &catalog)?;
        validate_catalog_parameter_topology::<Relu2Operation>(
            &text,
            &plan,
            &catalog,
            recipe_source.as_ref(),
        )?;
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        return RoutedTextRequirements::new(
            text,
            [(
                RoutedBankId::new(0),
                RoutedBankRequirements::new(
                    owner_group,
                    RoutedGroupedPlan::Relu2(plan),
                    catalog,
                    routes_by_unit,
                )?,
            )],
            recipe_source.as_ref(),
        );
    }
    let (text, plan, catalog, owner_group_name) = match family {
        Family::Qwen(args) => (
            crate::replicated_text::qwen_replicated_text_requirements(inspection, args),
            crate::qwen::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::qwen::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "text_decoder",
        ),
        Family::GptOss(args) => (
            crate::replicated_text::gpt_oss_replicated_text_requirements(inspection, args),
            crate::gpt_oss::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::gpt_oss::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "text_decoder",
        ),
        Family::Lfm2(args) => (
            crate::replicated_text::lfm2_replicated_text_requirements(inspection, args),
            crate::lfm2::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::lfm2::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::KimiLinear(args) => (
            crate::replicated_text::kimi_linear_replicated_text_requirements(inspection, args),
            crate::kimi_linear::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::kimi_linear::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::QwenHybrid(args) => (
            crate::replicated_text::qwen_hybrid_replicated_text_requirements(inspection, args),
            crate::qwen::hybrid::replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::qwen::hybrid::expert_residency_catalog(recipe_source.as_ref(), args)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::DeepSeekV3(args) => (
            crate::replicated_text::deepseek_v3_replicated_text_requirements(inspection, args),
            crate::deepseek::v3_replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::deepseek::v3_expert_residency_catalog(recipe_source.as_ref(), args, None)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::DeepSeekV4(args) => (
            crate::replicated_text::deepseek_v4_replicated_text_requirements(inspection, args),
            crate::deepseek::v4_replicated_expert_realization_plan(args)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string())),
            crate::deepseek::v4_expert_residency_catalog(recipe_source.as_ref(), args, None)
                .map_err(RoutedTextRequirementsError::Invalid),
            "target",
        ),
        Family::NemotronH(_) => {
            unreachable!("Nemotron-H returned through ReLU-squared requirements")
        }
    };
    let text = text
        .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?
        .with_grouped_operations([eredu_runtime::GroupedOperationRequirement::GatedProduct]);
    let plan = plan?;
    let catalog = catalog?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(owner_group_name)
        .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
    validate_expected_routed_schedule(&expected_routed_units, &plan, &catalog)?;
    validate_plan_catalog::<GatedProductOperation>(&owner_group, &plan, &catalog)?;
    validate_catalog_parameter_topology::<GatedProductOperation>(
        &text,
        &plan,
        &catalog,
        recipe_source.as_ref(),
    )?;
    let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
    RoutedTextRequirements::new(
        text,
        [(
            RoutedBankId::new(0),
            RoutedBankRequirements::new(
                owner_group,
                RoutedGroupedPlan::Gated(plan),
                catalog,
                routes_by_unit,
            )?,
        )],
        recipe_source.as_ref(),
    )
}

fn validate_expected_routed_schedule<S>(
    expected: &BTreeMap<(String, usize), String>,
    plan: &ExpertRealizationPlan<S>,
    catalog: &ExpertResidencyCatalog,
) -> Result<(), RoutedTextRequirementsError> {
    let actual = plan
        .unit_specs()
        .keys()
        .map(|(group, unit)| ((group.as_str().to_owned(), *unit), ()))
        .collect::<BTreeMap<_, _>>();
    let expected_addresses = expected
        .keys()
        .cloned()
        .map(|address| (address, ()))
        .collect::<BTreeMap<_, _>>();
    if actual != expected_addresses {
        return Err(RoutedTextRequirementsError::Invalid(format!(
            "routed plan addresses {:?} differ from architecture schedule {:?}",
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        )));
    }
    for ((group, unit), expected_path) in expected {
        let members = catalog
            .units()
            .iter()
            .filter(|member| member.owner_group().as_str() == group && member.owner_unit() == *unit)
            .collect::<Vec<_>>();
        if members.is_empty()
            || members
                .iter()
                .any(|member| member.unit_path() != expected_path)
        {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "routed catalog address {group:?}/{unit} does not use architecture path {expected_path:?}"
            )));
        }
    }
    Ok(())
}

fn validate_catalog_parameter_topology<O: RoutedGroupedOperationValidation>(
    text: &eredu_runtime::ReplicatedTextRequirements,
    plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
    recipe_source: &(impl eredu_checkpoint::recipe::RecipeCatalog + ?Sized),
) -> Result<(), RoutedTextRequirementsError> {
    let requirements = text
        .parameters()
        .iter()
        .map(|parameter| (parameter.name(), parameter))
        .collect::<BTreeMap<_, _>>();
    let mut occurrences = BTreeMap::<&str, usize>::new();
    for parameter in catalog.units().iter().flat_map(|unit| unit.parameters()) {
        *occurrences.entry(parameter.logical_target()).or_default() += 1;
    }
    let mut allowed_sources = BTreeMap::new();
    let mut member_recipes = BTreeMap::new();
    for unit in catalog.units() {
        for parameter in unit.parameters() {
            let requirement = requirements
                .get(parameter.logical_target())
                .ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} is absent from replicated parameter topology",
                        parameter.logical_target()
                    ))
                })?;
            let spec = plan
                .unit_spec(unit.owner_group().as_str(), unit.identity().unit())
                .ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} has no routed unit specification",
                        parameter.logical_target()
                    ))
                })?;
            let expected_shapes = O::member_parameter_shapes(spec, unit.identity().member())
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?;
            if let Some(expected_member_shape) = expected_shapes.get(parameter.logical_target()) {
                let occurrences = occurrences[parameter.logical_target()];
                let mut expected_logical = expected_member_shape.clone();
                if occurrences > 1 {
                    let first = expected_logical.first_mut().ok_or_else(|| {
                        RoutedTextRequirementsError::Invalid(
                            "grouped member shape has no leading expert axis".into(),
                        )
                    })?;
                    *first = occurrences;
                }
                if requirement.logical_shape() != expected_logical {
                    return Err(RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} logical shape {:?} differs from grouped shape {expected_logical:?}",
                        parameter.logical_target(),
                        requirement.logical_shape()
                    )));
                }
                let mut expected_physical = requirement
                    .physical_shape()
                    .ok_or_else(|| {
                        RoutedTextRequirementsError::Invalid(format!(
                            "bank target {:?} has no admitted physical geometry",
                            parameter.logical_target()
                        ))
                    })?
                    .to_vec();
                if occurrences > 1 {
                    let first = expected_physical.first_mut().ok_or_else(|| {
                        RoutedTextRequirementsError::Invalid(
                            "packed bank physical shape has no expert axis".into(),
                        )
                    })?;
                    if *first != occurrences {
                        return Err(RoutedTextRequirementsError::Invalid(format!(
                            "bank target {:?} physical expert axis {} differs from catalog cardinality {occurrences}",
                            parameter.logical_target(), *first
                        )));
                    }
                    *first = 1;
                }
                let metadata = parameter.metadata().ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} has no inferred recipe geometry",
                        parameter.logical_target()
                    ))
                })?;
                if metadata.shape() != expected_physical {
                    return Err(RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} recipe shape {:?} differs from admitted member shape {expected_physical:?}",
                        parameter.logical_target(), metadata.shape()
                    )));
                }
            }
            match requirement.owner() {
                eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit {
                    group,
                    unit: owner,
                } if group == unit.owner_group().as_str() && *owner == unit.owner_unit() => {}
                actual => {
                    return Err(RoutedTextRequirementsError::Invalid(format!(
                        "bank target {:?} owner {actual:?} differs from {:?}/{}",
                        parameter.logical_target(),
                        unit.owner_group().as_str(),
                        unit.owner_unit()
                    )));
                }
            }
            if !matches!(
                requirement.role(),
                eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                    | eredu_runtime::ReplicatedTextParameterRole::LinearBias
                    | eredu_runtime::ReplicatedTextParameterRole::FormatCompanion
            ) {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank target {:?} has non-linear topology role {:?}",
                    parameter.logical_target(),
                    requirement.role()
                )));
            }
            let derived = text.derived_recipes().get(parameter.logical_target());
            let allowed = allowed_sources
                .entry(parameter.logical_target())
                .or_insert_with(|| {
                    derived.map_or_else(
                        || {
                            requirement
                                .sources()
                                .iter()
                                .chain(requirement.aliases())
                                .cloned()
                                .collect::<std::collections::BTreeSet<_>>()
                        },
                        |recipe| {
                            recipe
                                .source_keys()
                                .into_iter()
                                .map(str::to_owned)
                                .collect()
                        },
                    )
                });
            let actual = parameter
                .recipe()
                .source_keys()
                .into_iter()
                .map(str::to_owned)
                .collect::<std::collections::BTreeSet<_>>();
            if actual.is_empty() || !actual.is_subset(allowed) {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank target {:?} recipe sources {actual:?} differ from admitted sources {allowed:?}",
                    parameter.logical_target()
                )));
            }
            let member = unit.identity().member();
            let member_selection = eredu_checkpoint::store::TensorSelection::Range {
                axis: 0,
                start: member,
                end: member + 1,
            };
            let exact_match = if let Some(recipe) = derived {
                let count = occurrences[parameter.logical_target()];
                if count > 1 {
                    if !member_recipes.contains_key(parameter.logical_target()) {
                        let members =
                            recipe
                                .select_bounded_members(recipe_source)
                                .map_err(|error| {
                                    RoutedTextRequirementsError::Invalid(format!(
                                        "bank target {:?} cannot select admitted members: {error}",
                                        parameter.logical_target()
                                    ))
                                })?;
                        member_recipes.insert(parameter.logical_target(), members);
                    }
                    member_recipes[parameter.logical_target()].get(member)
                        == Some(parameter.recipe())
                } else {
                    recipe
                        .select_bounded(recipe_source, member_selection.clone())
                        .map_err(|error| {
                            RoutedTextRequirementsError::Invalid(format!(
                                "bank target {:?} cannot select admitted member {member}: {error}",
                                parameter.logical_target()
                            ))
                        })?
                        == *parameter.recipe()
                }
            } else {
                requirement
                    .sources()
                    .iter()
                    .chain(requirement.aliases())
                    .filter_map(|source| {
                        let complete = eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                            source,
                            eredu_checkpoint::store::TensorSelection::Full,
                        );
                        if &complete == parameter.recipe() {
                            return Some(true);
                        }
                        complete
                            .select_bounded(recipe_source, member_selection.clone())
                            .ok()
                            .map(|selected| selected == *parameter.recipe())
                    })
                    .any(|matches| matches)
            };
            if !exact_match {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank target {:?} member {member} recipe differs from the exact admitted member recipe",
                    parameter.logical_target()
                )));
            }
        }
    }
    Ok(())
}

/// Invalid routed plan or failure from a generic execution mechanism.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedTextExecutionError {
    /// The selected constructor did not retain a resident provider source.
    #[error("selected routed constructor has no retained resident provider source")]
    ResidentSourceUnavailable,
    /// Architecture plan, catalog, unit address, or grouped geometry disagreed.
    #[error("invalid routed text execution contract: {0}")]
    Contract(String),
    /// Indexed movement, bank acquisition, grouped construction, or completion failed.
    #[error("routed text execution mechanism failed: {0}")]
    Mechanism(String),
    /// Original compute, storage, movement or completion failure.
    #[error("routed text execution mechanism failed: {0}")]
    Source(#[source] eredu_nn::Error),
}

impl From<String> for RoutedTextExecutionError {
    fn from(message: String) -> Self {
        Self::Contract(message)
    }
}
impl From<&str> for RoutedTextExecutionError {
    fn from(message: &str) -> Self {
        Self::Contract(message.into())
    }
}

impl RoutedTextExecutionError {
    /// Retains a mechanism failure without erasing its typed cause.
    pub fn from_error(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Source(eredu_nn::Error::backend_source(error))
    }
}

fn addressable_source_error<E:std::error::Error+Send+Sync+'static>(
    demands:&eredu_runtime::expert::IndexedDemandSource,cause:E)->RoutedTextExecutionError {
    match demands.funding() {
        Some(funding)=>RoutedTextExecutionError::Source(funding.metadata_source(cause)),
        None=>RoutedTextExecutionError::from_error(cause),
    }
}

fn retain_addressable_demand_failure(
    source: eredu_runtime::expert::IndexedDemandSource,
    cause: RoutedTextExecutionError,
) -> RoutedTextExecutionError {
    if source.funding().is_none() { return cause; }
    let cause = match cause {
        RoutedTextExecutionError::Source(cause) => cause,
        cause => eredu_nn::Error::backend_retained_source(cause),
    };
    RoutedTextExecutionError::Source(source.retain_error(cause))
}

fn validate_route_cardinality<T: Tensor>(
    routes: &GroupSelection<T>,
    routes_per_token: usize,
) -> Result<(), RoutedTextExecutionError> {
    let actual = routes
        .group_indices()
        .shape()
        .last()
        .copied()
        .and_then(|value| usize::try_from(value).ok());
    if actual != Some(routes_per_token) {
        return Err(RoutedTextExecutionError::Contract(format!(
            "route cardinality {actual:?} differs from selected routes per token {routes_per_token}"
        )));
    }
    Ok(())
}

/// Resident grouped execution validated against one architecture plan.
pub struct PlannedResidentGatedProduct {
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    routes_by_unit: BTreeMap<usize, usize>,
    partition_unit_coordinates:
        Option<BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>>,
}

impl PlannedResidentGatedProduct {
    pub(crate) fn with_partition_unit_coordinates(
        mut self,
        coordinates: BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>,
    ) -> Self {
        self.partition_unit_coordinates = Some(coordinates);
        self
    }

    fn unit_global_group_indices(&self, unit: usize) -> Option<&[usize]> {
        if self.plan.unit_is_replicated(unit)
            || self
                .partition_unit_coordinates
                .as_ref()
                .and_then(|coordinates| coordinates.get(&unit))
                .is_some_and(|coordinates| {
                    let experts = coordinates.experts();
                    experts.local_count() == experts.global_count()
                        && (0..experts.local_count())
                            .all(|index| experts.local_to_global(index) == Some(index))
                })
        {
            // The separately replicated shared bank uses its own global IDs.
            None
        } else {
            Some(self.plan.local_global_group_indices())
        }
    }

    /// Validates and retains one replicated routed-unit plan.
    pub fn new(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        catalog: ExpertResidencyCatalog,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        Self::new_with_routes(owner_group, plan, catalog, routes_by_unit)
    }

    fn new_with_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        catalog: ExpertResidencyCatalog,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        validate_replicated_plan(&plan)?;
        validate_plan_catalog::<GatedProductOperation>(&owner_group, &plan, &catalog)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_routes_by_unit::<GatedProductOperation>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        Ok(Self {
            owner_group,
            plan,
            routes_by_unit,
            partition_unit_coordinates: None,
        })
    }

    /// Validates and retains one exact rank-local expert-partition plan.
    pub fn new_partitioned(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        Self::new_partitioned_with_routes(owner_group, plan, routes_by_unit)
    }

    /// Validates one rank-local plan with exact per-unit route cardinality.
    pub fn new_partitioned_with_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        if plan.local_global_group_indices().is_empty() {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned routed rank owns no experts".into(),
            ));
        }
        validate_routes_by_unit::<GatedProductOperation>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        if plan
            .unit_specs()
            .keys()
            .any(|(group, _)| group != &owner_group)
        {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned routed plan names a different owner group".into(),
            ));
        }
        Ok(Self {
            owner_group,
            plan,
            routes_by_unit,
            partition_unit_coordinates: None,
        })
    }
}

impl<B> RoutedExpertProvider<B> for &PlannedResidentGatedProduct
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        partition_units::with_optional_coordinates(
            self.partition_unit_coordinates.as_ref(),
            request,
            |mut request| {
                let routes = self
                    .routes_by_unit
                    .get(&request.layer)
                    .copied()
                    .ok_or_else(|| {
                        RoutedTextExecutionError::Contract(format!(
                            "execution unit {:?}/{} has no route cardinality",
                            self.owner_group.as_str(),
                            request.layer
                        ))
                    })?;
                validate_route_cardinality(request.routes, routes)?;
                let selected = self
                    .plan
                    .unit_spec(self.owner_group.as_str(), request.layer)
                    .ok_or_else(|| {
                        RoutedTextExecutionError::Contract(format!(
                            "execution unit {:?}/{} has no grouped bank specification",
                            self.owner_group.as_str(),
                            request.layer
                        ))
                    })?;
                if selected != resident_bank.spec() {
                    return Err(RoutedTextExecutionError::Contract(format!(
                        "resident grouped bank for {:?}/{} differs from the architecture plan",
                        self.owner_group.as_str(),
                        request.layer
                    )));
                }
                eredu_runtime::with_provider_unit_observer(
                    &mut request.unit_observer,
                    request.routes.group_indices(),
                    self.unit_global_group_indices(request.layer),
                    0,
                    |observer| {
                        resident_bank.forward_grouped_with_unit_observer(
                            request.input,
                            request.routes,
                            context,
                            observer,
                        )
                    },
                )
                .map_err(RoutedTextExecutionError::from_error)
            },
        )
    }

    fn forward_compact_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        partition_units::with_optional_coordinates(
            self.partition_unit_coordinates.as_ref(),
            request,
            |mut request| {
                let selected = self
                    .plan
                    .unit_spec(self.owner_group.as_str(), request.layer)
                    .ok_or_else(|| {
                        RoutedTextExecutionError::Contract(format!(
                            "execution unit {:?}/{} has no grouped bank specification",
                            self.owner_group.as_str(),
                            request.layer
                        ))
                    })?;
                if selected != resident_bank.spec() {
                    return Err(RoutedTextExecutionError::Contract(format!(
                        "resident grouped bank for {:?}/{} differs from the architecture plan",
                        self.owner_group.as_str(),
                        request.layer
                    )));
                }
                validate_route_cardinality(request.routes, 1)?;
                eredu_runtime::with_provider_unit_observer(
                    &mut request.unit_observer,
                    request.routes.group_indices(),
                    self.unit_global_group_indices(request.layer),
                    0,
                    |observer| {
                        resident_bank.forward_grouped_with_unit_observer(
                            request.input,
                            request.routes,
                            context,
                            observer,
                        )
                    },
                )
                .map_err(RoutedTextExecutionError::from_error)
            },
        )
    }

    /// Executes an activated selected-linear bank with owned output rows.
    fn forward_linear_routed(
        &mut self,
        _: &mut B::LinearGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected provider equation is not a linear bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

impl<B> RoutedExpertProvider<B> for PlannedResidentGatedProduct
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;
    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident_bank), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_grouped(&mut &*self, resident_bank, request, context)
    }

    fn forward_compact_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident_bank), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_compact_grouped(&mut &*self, resident_bank, request, context)
    }

    fn forward_linear_routed(
        &mut self,
        operand_1: &mut B::LinearGroups,
        operand_2: RoutedExpertRequest<'_, '_, B::Tensor>,
        operand_3: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        RoutedExpertProvider::<B>::forward_linear_routed(&mut &*self, operand_1, operand_2, operand_3)
    }

    fn forward_relu2_routed(
        &mut self,
        operand_1: &mut B::Relu2Groups,
        operand_2: RoutedExpertRequest<'_, '_, B::Tensor>,
        operand_3: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        RoutedExpertProvider::<B>::forward_relu2_routed(&mut &*self, operand_1, operand_2, operand_3)
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for &PlannedResidentGatedProduct
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        partition_units::with_optional_coordinates(
            self.partition_unit_coordinates.as_ref(),
            request,
            |mut request| {
                let routes = self
                    .routes_by_unit
                    .get(&request.layer)
                    .copied()
                    .ok_or_else(|| {
                        RoutedTextExecutionError::Contract(format!(
                            "execution unit {:?}/{} has no route cardinality",
                            self.owner_group.as_str(),
                            request.layer
                        ))
                    })?;
                validate_route_cardinality(request.routes, routes)?;
                let selected = self
                    .plan
                    .unit_spec(self.owner_group.as_str(), request.layer)
                    .ok_or_else(|| {
                        RoutedTextExecutionError::Contract(format!(
                            "execution unit {:?}/{} has no grouped bank specification",
                            self.owner_group.as_str(),
                            request.layer
                        ))
                    })?;
                if selected != resident_bank.spec() {
                    return Err(RoutedTextExecutionError::Contract(format!(
                        "resident grouped bank for {:?}/{} differs from the architecture plan",
                        self.owner_group.as_str(),
                        request.layer
                    )));
                }
                eredu_runtime::with_provider_unit_observer(
                    &mut request.unit_observer,
                    request.routes.group_indices(),
                    self.unit_global_group_indices(request.layer),
                    0,
                    |observer| {
                        B::gated_product_groups_tensor_parallel_with_unit_observer(
                            resident_bank,
                            request.input,
                            request.routes,
                            partitions,
                            context,
                            observer,
                        )
                    },
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
                .map_err(RoutedTextExecutionError::from_error)
            },
        )
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        partition_units::with_optional_coordinates(
            self.partition_unit_coordinates.as_ref(),
            request,
            |mut request| {
                let selected = self
                    .plan
                    .unit_spec(self.owner_group.as_str(), request.layer)
                    .ok_or_else(|| {
                        RoutedTextExecutionError::Contract(format!(
                            "execution unit {:?}/{} has no grouped bank specification",
                            self.owner_group.as_str(),
                            request.layer
                        ))
                    })?;
                if selected != resident_bank.spec() {
                    return Err(RoutedTextExecutionError::Contract(format!(
                        "resident grouped bank for {:?}/{} differs from the architecture plan",
                        self.owner_group.as_str(),
                        request.layer
                    )));
                }
                validate_route_cardinality(request.routes, 1)?;
                eredu_runtime::with_provider_unit_observer(
                    &mut request.unit_observer,
                    request.routes.group_indices(),
                    self.unit_global_group_indices(request.layer),
                    0,
                    |observer| {
                        B::gated_product_groups_tensor_parallel_with_unit_observer(
                            resident_bank,
                            request.input,
                            request.routes,
                            partitions,
                            context,
                            observer,
                        )
                    },
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
                .map_err(RoutedTextExecutionError::from_error)
            },
        )
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for PlannedResidentGatedProduct
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident_bank), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(&mut &*self, resident_bank, request, partitions, context)
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident_bank), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(&mut &*self, resident_bank, request, partitions, context)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        operand_1: &mut B::Relu2Groups,
        operand_2: RoutedExpertRequest<'_, '_, B::Tensor>,
        operand_3: usize,
        operand_4: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(&mut &*self, operand_1, operand_2, operand_3, operand_4)
    }
}

/// Resident ReLU-squared execution validated against one architecture plan.
pub struct PlannedResidentRelu2 {
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
    routes_per_token: usize,
}

/// Provider installed on a pipeline rank whose exact local routed catalog is empty.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyPartitionRoutedExpertProvider;

impl<B> RoutedExpertProvider<B> for EmptyPartitionRoutedExpertProvider
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received gated-product work".into(),
        ))
    }

    /// Executes an activated selected-linear bank with owned output rows.
    fn forward_linear_routed(
        &mut self,
        _: &mut B::LinearGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected provider equation is not a linear bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received ReLU-squared work".into(),
        ))
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for EmptyPartitionRoutedExpertProvider
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received tensor-parallel gated-product work".into(),
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received tensor-parallel ReLU-squared work".into(),
        ))
    }
}

impl PlannedResidentRelu2 {
    /// Validates and retains one replicated routed-unit plan.
    pub fn new(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
        catalog: ExpertResidencyCatalog,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        validate_replicated_plan(&plan)?;
        validate_plan_catalog::<Relu2Operation>(&owner_group, &plan, &catalog)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        if routes_per_token == 0 {
            return Err(RoutedTextExecutionError::Contract(
                "routes per token must be positive".into(),
            ));
        }
        Ok(Self {
            owner_group,
            plan,
            routes_per_token,
        })
    }

    /// Validates and retains one exact rank-local ReLU-squared expert plan.
    pub fn new_partitioned(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        if plan.local_global_group_indices().is_empty()
            || plan
                .unit_specs()
                .keys()
                .any(|(group, _)| group != &owner_group)
            || routes_per_token == 0
        {
            return Err(RoutedTextExecutionError::Contract(
                "partitioned ReLU-squared plan has no local experts, changes owner, or has no routes"
                    .into(),
            ));
        }
        Ok(Self {
            owner_group,
            plan,
            routes_per_token,
        })
    }
}

impl<B> RoutedExpertProvider<B> for &PlannedResidentRelu2
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    /// Executes an activated selected-linear bank with owned output rows.
    fn forward_linear_routed(
        &mut self,
        _resident_bank: &mut B::LinearGroups,
        _request: RoutedExpertRequest<'_, '_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected provider equation is not a linear bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        partition_units::with_optional_coordinates(None, request, |mut request| {
        validate_route_cardinality(request.routes, self.routes_per_token)?;
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        eredu_runtime::with_provider_unit_observer(
            &mut request.unit_observer,
            request.routes.group_indices(),
            (!self.plan.unit_is_replicated(request.layer))
                .then(|| self.plan.local_global_group_indices()),
            0,
            |observer| {
                resident_bank.forward_grouped_with_unit_observer(
                    request.input,
                    request.routes,
                    context,
                    observer,
                )
            },
        )
        .map_err(RoutedTextExecutionError::from_error)
        })
    }
}

impl<B> RoutedExpertProvider<B> for PlannedResidentRelu2
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;
    fn forward_grouped(
        &mut self,
        operand_1: &mut B::GatedProductGroups,
        operand_2: RoutedExpertRequest<'_, '_, B::Tensor>,
        operand_3: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        RoutedExpertProvider::<B>::forward_grouped(&mut &*self, operand_1, operand_2, operand_3)
    }

    fn forward_linear_routed(
        &mut self,
        _resident_bank: &mut B::LinearGroups,
        _request: RoutedExpertRequest<'_, '_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        RoutedExpertProvider::<B>::forward_linear_routed(&mut &*self, _resident_bank, _request, _context)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        mut request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident_bank), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_relu2_routed(&mut &*self, resident_bank, request, context)
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for &PlannedResidentRelu2
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        partition_units::with_optional_coordinates(None, request, |mut request| {
        validate_route_cardinality(request.routes, self.routes_per_token)?;
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        if selected != resident_bank.spec() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "resident grouped bank for {:?}/{} differs from the architecture plan",
                self.owner_group.as_str(),
                request.layer
            )));
        }
        eredu_runtime::with_provider_unit_observer(
            &mut request.unit_observer,
            request.routes.group_indices(),
            (!self.plan.unit_is_replicated(request.layer))
                .then(|| self.plan.local_global_group_indices()),
            0,
            |observer| {
                B::relu2_groups_tensor_parallel_with_unit_observer(
                    resident_bank,
                    request.input,
                    request.routes,
                    partitions,
                    context,
                    observer,
                )
            },
        )
        .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
        .map_err(RoutedTextExecutionError::from_error)
        })
    }
}

impl<B> eredu_runtime::TensorParallelRoutedExpertProvider<B> for PlannedResidentRelu2
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        operand_1: &mut B::GatedProductGroups,
        operand_2: RoutedExpertRequest<'_, '_, B::Tensor>,
        operand_3: usize,
        operand_4: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(&mut &*self, operand_1, operand_2, operand_3, operand_4)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        mut request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident_bank), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        eredu_runtime::TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(&mut &*self, resident_bank, request, partitions, context)
    }
}

/// Gated-product grouped-operation projection for the neutral routed driver.
#[derive(Debug, Clone, Copy)]
pub struct GatedProductOperation;

/// ReLU-squared grouped-operation projection for the neutral routed driver.
#[derive(Debug, Clone, Copy)]
pub struct Relu2Operation;

/// Architecture-level geometry shared by resident and addressable grouped execution.
pub trait RoutedGroupedOperationValidation {
    /// Architecture-owned grouped specification.
    type Spec: Clone + PartialEq;

    /// Copies the original full specification under the outer invocation source.
    fn clone_spec_with_funding(spec:&Self::Spec,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error>;

    /// Borrows the same retained grouped equation for an addressable source.
    fn workspace_kernel(spec: &Self::Spec) -> eredu_nn::workspace::WorkspaceExpertKernel<'_>;

    /// Returns the number of groups described by a specification.
    fn group_count(spec: &Self::Spec) -> i32;

    /// Returns the output width after the selected grouped equation.
    fn output_dimensions(spec: &Self::Spec) -> i32;

    /// Returns every exact parameter target needed by one global member.
    fn member_parameter_targets(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<Vec<String>, eredu_nn::Error>;

    /// Returns exact per-member logical shapes for primary projections and biases.
    fn member_parameter_shapes(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, eredu_nn::Error>;

    /// Projects an architecture-global specification to one compact bank.
    fn compact_spec(spec: &Self::Spec, group_count: i32) -> Result<Self::Spec, eredu_nn::Error>;

    /// Uses the closed clone producer of the retained semantic specification.
    fn compact_spec_with_funding(spec:&Self::Spec,group_count:i32,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error> {
        if funding.is_some(){return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());}
        Self::compact_spec(spec,group_count)
    }
}

/// One grouped equation consumable by addressable routed composition.
pub trait RoutedGroupedOperation<B>: RoutedGroupedOperationValidation
where
    B: GroupedNeuralBackend,
{
    /// Constructs and executes the exact grouped equation over acquired storage.
    fn execute<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display;
}

/// Grouped equation that preserves tensor-parallel reduction structure.
pub trait TensorParallelRoutedGroupedOperation<B>: RoutedGroupedOperation<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    /// Constructs and executes one acquired rank-local grouped partial.
    fn execute_tensor_parallel<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        partitions: usize,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, eredu_nn::Error>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display;
}

impl RoutedGroupedOperationValidation for GatedProductOperation {
    type Spec = eredu_nn::GroupedGatedProductSpec;

    fn clone_spec_with_funding(spec:&Self::Spec,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error> {
        match funding {Some(funding)=>funding.clone_grouped_gated_product(spec),None=>Ok(spec.clone())}
    }

    fn workspace_kernel(spec: &Self::Spec) -> eredu_nn::workspace::WorkspaceExpertKernel<'_> {
        eredu_nn::workspace::WorkspaceExpertKernel::Gated(spec)
    }

    fn group_count(spec: &Self::Spec) -> i32 {
        spec.group_count()
    }

    fn output_dimensions(spec: &Self::Spec) -> i32 {
        spec.output_dimensions()
    }

    fn member_parameter_targets(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<Vec<String>, eredu_nn::Error> {
        let projections = match spec.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
                vec![gate_up, down]
            }
            eredu_nn::GatedProductGroupLayout::Independent(groups) => {
                let group = groups.get(member).ok_or_else(|| {
                    eredu_nn::Error::backend(format!(
                        "gated-product member {member} has no parameter group"
                    ))
                })?;
                vec![group.gate(), group.up(), group.down()]
            }
            _ => return Err(eredu_nn::Error::backend("unknown gated-product layout")),
        };
        Ok(projections
            .into_iter()
            .flat_map(eredu_nn::GroupedProjectionSpec::parameters)
            .map(|parameter| parameter.id.to_string())
            .collect())
    }

    fn member_parameter_shapes(
        spec: &Self::Spec,
        member: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, eredu_nn::Error> {
        let input = usize::try_from(spec.input_dimensions()).map_err(eredu_nn::Error::backend)?;
        let intermediate =
            usize::try_from(spec.intermediate_dimensions()).map_err(eredu_nn::Error::backend)?;
        let output = usize::try_from(spec.output_dimensions()).map_err(eredu_nn::Error::backend)?;
        let mut shapes = BTreeMap::new();
        let mut add = |projection: &eredu_nn::GroupedProjectionSpec, shape: Vec<usize>| {
            shapes.insert(projection.weight().id.to_string(), shape.clone());
            if let Some(bias) = projection.bias() {
                shapes.insert(bias.id.to_string(), shape[..shape.len() - 1].to_vec());
            }
            if let Some(companion_shape) = grouped_companion_shape(projection.format(), &shape) {
                if let Some(scale) = projection.format().scale() {
                    shapes.insert(scale.id.to_string(), companion_shape.clone());
                }
                if let Some(bias) = projection.format().affine_bias() {
                    shapes.insert(bias.id.to_string(), companion_shape);
                }
            }
        };
        match spec.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } => {
                add(gate_up, vec![1, intermediate * 2, input]);
                add(down, vec![1, output, intermediate]);
            }
            eredu_nn::GatedProductGroupLayout::Independent(groups) => {
                let group = groups.get(member).ok_or_else(|| {
                    eredu_nn::Error::backend(format!(
                        "gated-product member {member} has no parameter group"
                    ))
                })?;
                add(group.gate(), vec![intermediate, input]);
                add(group.up(), vec![intermediate, input]);
                add(group.down(), vec![output, intermediate]);
            }
            _ => return Err(eredu_nn::Error::backend("unknown gated-product layout")),
        }
        Ok(shapes)
    }

    fn compact_spec(spec: &Self::Spec, group_count: i32) -> Result<Self::Spec, eredu_nn::Error> {
        spec.clone()
            .with_group_geometry(group_count, spec.intermediate_dimensions())
    }
    fn compact_spec_with_funding(spec:&Self::Spec,group_count:i32,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error> {
        match funding {Some(funding)=>funding.clone_grouped_gated_product(spec)?,None=>spec.clone()}
            .with_group_geometry(group_count,spec.intermediate_dimensions())
    }
}

impl<B> RoutedGroupedOperation<B> for GatedProductOperation
where
    B: GroupedNeuralBackend,
{
    fn execute<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .gated_product_groups(acquisition, spec, context)
            .map_err(eredu_nn::Error::backend_source)?;
        groups
            .forward_grouped_with_unit_observer(input, routes, context, observer)
            .map_err(eredu_nn::Error::backend_source)
    }
}

impl<B> TensorParallelRoutedGroupedOperation<B> for GatedProductOperation
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn execute_tensor_parallel<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        partitions: usize,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, eredu_nn::Error>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .gated_product_groups(acquisition, spec, context)
            .map_err(eredu_nn::Error::backend_source)?;
        B::gated_product_groups_tensor_parallel_with_unit_observer(
            &mut groups,
            input,
            routes,
            partitions,
            context,
            observer,
        )
        .map_err(eredu_nn::Error::backend_source)
    }
}

impl RoutedGroupedOperationValidation for Relu2Operation {
    type Spec = eredu_nn::GroupedRelu2Spec;

    fn clone_spec_with_funding(spec:&Self::Spec,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error> {
        match funding {Some(funding)=>funding.clone_grouped_relu2(spec),None=>Ok(spec.clone())}
    }

    fn workspace_kernel(spec: &Self::Spec) -> eredu_nn::workspace::WorkspaceExpertKernel<'_> {
        eredu_nn::workspace::WorkspaceExpertKernel::Relu2(spec)
    }

    fn group_count(spec: &Self::Spec) -> i32 {
        spec.group_count()
    }

    fn output_dimensions(spec: &Self::Spec) -> i32 {
        spec.hidden_dimensions()
    }

    fn member_parameter_targets(
        spec: &Self::Spec,
        _: usize,
    ) -> Result<Vec<String>, eredu_nn::Error> {
        Ok([spec.up(), spec.down()]
            .into_iter()
            .flat_map(eredu_nn::GroupedProjectionSpec::parameters)
            .map(|parameter| parameter.id.to_string())
            .collect())
    }

    fn member_parameter_shapes(
        spec: &Self::Spec,
        _: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, eredu_nn::Error> {
        let hidden = usize::try_from(spec.hidden_dimensions()).map_err(eredu_nn::Error::backend)?;
        let intermediate =
            usize::try_from(spec.intermediate_dimensions()).map_err(eredu_nn::Error::backend)?;
        let mut shapes = BTreeMap::new();
        for (projection, shape) in [
            (spec.up(), vec![1, intermediate, hidden]),
            (spec.down(), vec![1, hidden, intermediate]),
        ] {
            shapes.insert(projection.weight().id.to_string(), shape.clone());
            if let Some(bias) = projection.bias() {
                shapes.insert(bias.id.to_string(), shape[..shape.len() - 1].to_vec());
            }
            if let Some(companion_shape) = grouped_companion_shape(projection.format(), &shape) {
                if let Some(scale) = projection.format().scale() {
                    shapes.insert(scale.id.to_string(), companion_shape.clone());
                }
                if let Some(bias) = projection.format().affine_bias() {
                    shapes.insert(bias.id.to_string(), companion_shape);
                }
            }
        }
        Ok(shapes)
    }

    fn compact_spec(spec: &Self::Spec, group_count: i32) -> Result<Self::Spec, eredu_nn::Error> {
        spec.clone().with_group_count(group_count)
    }
    fn compact_spec_with_funding(spec:&Self::Spec,group_count:i32,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error> {
        match funding {Some(funding)=>funding.clone_grouped_relu2(spec)?,None=>spec.clone()}
            .with_group_count(group_count)
    }
}

fn grouped_companion_shape(
    format: &eredu_nn::LinearFormatSpec,
    weight_shape: &[usize],
) -> Option<Vec<usize>> {
    let rows = *weight_shape.get(weight_shape.len().checked_sub(2)?)?;
    let columns = *weight_shape.last()?;
    let (row_groups, column_groups) = match format.encoding() {
        eredu_checkpoint::LinearFormat::Affine(config) => (
            rows,
            columns.div_ceil(usize::try_from(config.group_size).ok()?),
        ),
        eredu_checkpoint::LinearFormat::MxFp4 => (rows, columns.div_ceil(32)),
        eredu_checkpoint::LinearFormat::E4M3BlockFp8(config) => (
            format
                .row_layout()
                .scale_rows(rows, usize::try_from(config.block_rows).ok()?)
                .ok()?,
            columns.div_ceil(usize::try_from(config.block_columns).ok()?),
        ),
        eredu_checkpoint::LinearFormat::Dense
        | eredu_checkpoint::LinearFormat::GgufIQuant { .. } => return None,
    };
    let mut shape = weight_shape[..weight_shape.len() - 2].to_vec();
    shape.extend([row_groups, column_groups]);
    Some(shape)
}

impl<B> RoutedGroupedOperation<B> for Relu2Operation
where
    B: GroupedNeuralBackend,
{
    fn execute<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .relu2_groups(acquisition, spec, context)
            .map_err(eredu_nn::Error::backend_source)?;
        groups
            .forward_grouped_with_unit_observer(input, routes, context, observer)
            .map_err(eredu_nn::Error::backend_source)
    }
}

impl<B> TensorParallelRoutedGroupedOperation<B> for Relu2Operation
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn execute_tensor_parallel<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        partitions: usize,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, eredu_nn::Error>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        let mut groups = bank
            .relu2_groups(acquisition, spec, context)
            .map_err(eredu_nn::Error::backend_source)?;
        B::relu2_groups_tensor_parallel_with_unit_observer(
            &mut groups,
            input,
            routes,
            partitions,
            context,
            observer,
        )
        .map_err(eredu_nn::Error::backend_source)
    }
}

/// Addressable grouped execution driven by architecture identities.
pub struct PlannedAddressableGrouped<O, B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Movement: IndexedMovement<B>,
    O: RoutedGroupedOperation<B>,
{
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: std::sync::Arc<ExpertRealizationPlan<O::Spec>>,
    catalog: ExpertResidencyCatalog,
    bank: Bank,
    movement: Movement,
    compact_bank_scratch_bytes: u64,
    bulk_compact_bank_target_bytes: u64,
    routes_by_unit: BTreeMap<usize, usize>,
    operation: PhantomData<fn() -> (O, B)>,
}

/// Addressable gated-product execution through the shared neutral driver.
pub type PlannedAddressableGatedProduct<B, Bank, Movement> =
    PlannedAddressableGrouped<GatedProductOperation, B, Bank, Movement>;

/// Addressable ReLU-squared execution through the shared neutral driver.
pub type PlannedAddressableRelu2<B, Bank, Movement> =
    PlannedAddressableGrouped<Relu2Operation, B, Bank, Movement>;

pub(crate) fn unowned_expert_sources(
    catalog: &ExpertResidencyCatalog,
    local: &[usize],
) -> BTreeSet<String> {
    catalog
        .units()
        .iter()
        .filter(|unit| {
            unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel
                && !local.contains(&unit.identity().member())
        })
        .flat_map(|unit| unit.parameters())
        .flat_map(|parameter| parameter.recipe().source_keys())
        .map(str::to_owned)
        .collect()
}

impl<B, A, G, W> crate::partitioned_execution::PreparedRoutedPartitionedArchitecture<B, A, G, W>
where
    B: GroupedNeuralBackend,
{
    /// Constructs the complete resident collection from retained rank-local plans.
    pub fn resident_partition_providers(
        &self,
    ) -> Result<crate::prepared_execution::PartitionBankProviders<B>, RoutedTextExecutionError>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static,
    {
        self.resident_partition_source()?.ordinary::<B>()
    }

    /// Sources excluded by each bank's independent expert ownership map.
    pub fn unowned_expert_checkpoint_sources(&self) -> BTreeSet<String> {
        self.banks()
            .values()
            .flat_map(|bank| {
                unowned_expert_sources(bank.catalog(), bank.plan().local_global_group_indices())
            })
            .collect()
    }
    /// Logical targets materialized exclusively by independently addressable banks.
    pub fn addressable_logical_targets(&self) -> BTreeSet<String> {
        if !matches!(
            self.bank_residency(),
            eredu_runtime::ParameterBankResidency::IndependentCache(_)
        ) {
            return BTreeSet::new();
        }
        self.banks()
            .values()
            .flat_map(|bank| {
                bank.catalog()
                    .logical_targets()
                    .into_iter()
                    // Load-time transforms generate companions absent from the
                    // source catalog. Their retained member tasks own those
                    // destinations alongside the primary expert weights.
                    .chain(
                        bank.addressable_members()
                            .iter()
                            .flat_map(|member| member.parameters())
                            .flat_map(|parameter| parameter.task().output_companions())
                            .map(|companion| companion.name()),
                    )
                    .map(str::to_owned)
            })
            .collect()
    }
}

impl<O, B, Bank, Movement> PlannedAddressableGrouped<O, B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    O: RoutedGroupedOperation<B>,
{
    /// Validates plan/catalog coherence before any bank acquisition.
    pub fn new(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        Self::new_with_routes(
            owner_group,
            plan,
            catalog,
            selected_member_bytes,
            bank,
            movement,
            options,
            routes_by_unit,
        )
    }

    /// Validates and retains one exact rank-local addressable expert plan.
    #[allow(clippy::too_many_arguments)]
    pub fn new_partitioned(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_per_token: usize,
    ) -> Result<Self, RoutedTextExecutionError> {
        let routes_by_unit = uniform_routes_by_unit(&plan, routes_per_token);
        Self::new_partitioned_with_routes(
            owner_group,
            plan,
            catalog,
            selected_member_bytes,
            bank,
            movement,
            options,
            routes_by_unit,
        )
    }

    /// Validates local addressable members with each invocation's retained routes.
    #[allow(clippy::too_many_arguments)]
    pub fn new_partitioned_with_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        options
            .validate()
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_routes_by_unit::<O>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_partitioned_catalog_binding::<O>(
            &owner_group,
            &plan,
            &catalog,
            &selected_member_bytes,
            |key| bank.member_bytes(key),
        )?;
        Ok(Self {
            owner_group,
            plan: std::sync::Arc::new(plan),
            catalog,
            bank,
            movement,
            compact_bank_scratch_bytes: options.compact_bank_scratch_bytes(),
            bulk_compact_bank_target_bytes: options.prefill_compact_bank_target_bytes(),
            routes_by_unit,
            operation: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        validate_replicated_plan(&plan)?;
        options
            .validate()
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_routes_by_unit::<O>(&plan, &routes_by_unit)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        validate_plan_catalog::<O>(&owner_group, &plan, &catalog)
            .map_err(|error| RoutedTextExecutionError::Contract(error.to_string()))?;
        Self::from_validated_routes(
            owner_group,
            plan,
            catalog,
            selected_member_bytes,
            bank,
            movement,
            options,
            routes_by_unit,
        )
    }

    // Called only with the immutable contracts owned by a prepared architecture,
    // or after the public raw-parts constructor has established the same proofs.
    fn from_validated_routes(
        owner_group: eredu_runtime::ExecutionGroupId,
        plan: ExpertRealizationPlan<O::Spec>,
        catalog: ExpertResidencyCatalog,
        selected_member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: eredu_runtime::ParameterBankLoadOptions,
        routes_by_unit: BTreeMap<usize, usize>,
    ) -> Result<Self, RoutedTextExecutionError> {
        validate_catalog_binding::<B, Bank>(&catalog, &selected_member_bytes, &bank)?;
        Ok(Self {
            owner_group,
            plan: std::sync::Arc::new(plan),
            catalog,
            bank,
            movement,
            compact_bank_scratch_bytes: options.compact_bank_scratch_bytes(),
            bulk_compact_bank_target_bytes: options.prefill_compact_bank_target_bytes(),
            routes_by_unit,
            operation: PhantomData,
        })
    }

    /// Borrows the movement retained by the same typed provider constructor.
    pub fn indexed_movement(&self)->&Movement { &self.movement }

    /// Borrows the already selected storage owner without changing routing or placement.
    pub fn bank_storage(&self) -> &Bank {
        &self.bank
    }

    /// Returns generic storage telemetry without exposing backend types to the architecture.
    pub fn bank_report(&self) -> Result<Bank::Report, RoutedTextExecutionError> {
        self.bank
            .report()
            .map_err(RoutedTextExecutionError::from_error)
    }

    fn key_for(
        &self,
        owner_unit: usize,
        selected_identity: usize,
    ) -> Result<ParameterBankKey, RoutedTextExecutionError> {
        let global_identity = if self.plan.unit_is_replicated(owner_unit) {
            selected_identity
        } else {
            self
            .plan
            .local_global_group_indices()
            .get(selected_identity)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "selected owner-local group {selected_identity} is outside the rank-local expert plan"
                ))
            })?
        };
        self.catalog
            .units()
            .iter()
            .filter(|unit| {
                unit.owner_group() == &self.owner_group
                    && unit.identity().unit() == owner_unit
                    && unit.identity().member() == global_identity
            })
            .map(|unit| unit.identity())
            .next()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "selected group {selected_identity} has no bank key for {:?}/{owner_unit}",
                    self.owner_group.as_str()
                ))
            })
    }

    fn compact_chunk_plan(&self, spec: &O::Spec, owner_unit: usize, rows: usize,
        routes: usize, access: eredu_runtime::ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context)
        -> Result<eredu_runtime::expert::AddressableChunkPlan, RoutedTextExecutionError> {
        if let Some(metadata) = B::construction_metadata(context) {
            metadata.charge_metadata(eredu_runtime::expert::AddressableChunkPlan::control_bytes())
                .map_err(RoutedTextExecutionError::from_error)?;
        }
        let maximum = if access == eredu_runtime::ParameterBankAccess::Bulk {
            Some(self.catalog.units().iter().filter(|unit|
                unit.owner_group() == &self.owner_group && unit.identity().unit() == owner_unit)
                .filter_map(|unit| self.bank.member_bytes(unit.identity())).max()
                .ok_or_else(|| RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no selected bank byte geometry",
                    self.owner_group.as_str(), owner_unit)))?)
        } else { None };
        let members = usize::try_from(O::group_count(spec))
            .map_err(RoutedTextExecutionError::from_error)?;
        eredu_runtime::expert::AddressableChunkPlan::new(rows, routes, members, access,
            maximum, self.bulk_compact_bank_target_bytes)
            .map_err(RoutedTextExecutionError::from_error)
    }

    fn acquire_chunk(
        &mut self,
        spec: &O::Spec,
        owner_unit: usize,
        routes: &GroupSelection<B::Tensor>,
        access: eredu_runtime::ParameterBankAccess,
        census: eredu_runtime::expert::AddressableChunkCensus,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Bank::Acquisition, O::Spec, GroupSelection<B::Tensor>,
        eredu_runtime::expert::IndexedDemandSource), RoutedTextExecutionError>
    {
        let group_count = usize::try_from(O::group_count(spec)).map_err(|_| {
            RoutedTextExecutionError::Contract("grouped bank count is not representable".into())
        })?;
        if census.unit() != owner_unit || census.members() != group_count || census.access() != access {
            return Err(RoutedTextExecutionError::Contract("addressable chunk source differs from selected invocation".into()));
        }
        let demands = self
            .movement
            .index_demand_source_for_chunk(routes.group_indices(), census, context)
            .map_err(RoutedTextExecutionError::from_error)?;
        // Both derived directories and the demand-retention wrapper are paid
        // before remapping or acquisition can start. The source itself stays
        // alive through the caller's grouped completion.
        if demands.funding().is_some() {
            let count = demands.demands().len();
            let controls = [
                std::mem::size_of::<(Bank::Acquisition, O::Spec, GroupSelection<B::Tensor>,
                    eredu_runtime::expert::IndexedDemandSource)>(),
                std::mem::size_of::<Result<(Bank::Acquisition, O::Spec, GroupSelection<B::Tensor>),
                    RoutedTextExecutionError>>(),
                std::mem::size_of::<Result<B::Tensor, RoutedTextExecutionError>>(),
                std::mem::size_of::<Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>,
                    RoutedTextExecutionError>>(),
                std::mem::size_of::<(Vec<(ParameterBankKey,u64)>, Vec<(usize,usize)>)>(),
                std::mem::size_of::<(usize,u64,ParameterBankKey)>(),
                std::mem::size_of::<std::iter::Enumerate<std::slice::Iter<'_,(usize,u64)>>>() ,
            ];
            let bytes = count.checked_mul(std::mem::size_of::<(ParameterBankKey,u64)>())
                .and_then(|bytes| count.checked_mul(std::mem::size_of::<(usize,usize)>())
                    .and_then(|mapping| bytes.checked_add(mapping)))
                .and_then(|bytes| bytes.checked_add(std::mem::size_of_val(&controls)))
                .and_then(|bytes| controls.into_iter().try_fold(bytes, usize::checked_add))
                .and_then(|bytes| bytes.checked_add(
                    eredu_nn::Error::retained_source_control_bytes::<RoutedTextExecutionError>()?));
            demands.reserve_metadata(bytes)
                .map_err(RoutedTextExecutionError::Source)?;
        }
        let result = (|| {
            let values = demands.demands();
            if values.is_empty() {
                return Err(RoutedTextExecutionError::Contract(
                    "routed selection contains no bank members".into(),
                ));
            }
            if values.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
                return Err(RoutedTextExecutionError::Mechanism(
                    "indexed demand identities are not strictly ordered".into(),
                ));
            }
            let mut entries = Vec::with_capacity(values.len());
            let mut mapping = Vec::with_capacity(values.len());
            let mut scratch_bytes = 0u64;
            for (compact, &(identity, demand)) in values.iter().enumerate() {
                let key = self.key_for(owner_unit, identity)?;
                let bytes = self.bank.member_bytes(key).ok_or_else(|| {
                    RoutedTextExecutionError::Contract(format!(
                        "selected bank member {key:?} has no byte geometry"
                    ))
                })?;
                scratch_bytes = scratch_bytes.checked_add(bytes).ok_or_else(|| {
                    RoutedTextExecutionError::Contract(
                        "selected compact-bank byte geometry overflowed".into(),
                    )
                })?;
                entries.push((key, demand));
                mapping.push((identity, compact));
            }
            if scratch_bytes > self.compact_bank_scratch_bytes {
                return Err(RoutedTextExecutionError::Contract(format!(
                    "selected compact bank requires {scratch_bytes} bytes, limit is {}",
                    self.compact_bank_scratch_bytes
                )));
            }
            let compact_indices = self
                .movement
                .remap_demand_indices(routes.group_indices(), &mapping, &demands, context)
                .map_err(|cause|addressable_source_error(&demands,cause))?;
            let compact_routes = GroupSelection::new(
                compact_indices,
                self.movement.copy_route_value(routes.selected_scores(),&demands,context)
                    .map_err(|cause|addressable_source_error(&demands,cause))?,
                self.movement.copy_route_value(routes.coefficients(),&demands,context)
                    .map_err(|cause|addressable_source_error(&demands,cause))?,
            );
            let compact_count = i32::try_from(entries.len()).map_err(|_| {
                RoutedTextExecutionError::Contract("compact bank group count exceeds i32".into())
            })?;
            let compact_spec = O::compact_spec_with_funding(spec, compact_count, demands.funding())
                .map_err(RoutedTextExecutionError::Source)?;
            let bank=&mut self.bank;
            let acquisition=self.movement.with_demand_loan(&demands,|loan|
                bank.acquire_from_demand(ParameterBankAcquisition::new(&entries,access),&demands,loan,context))
                .map_err(|cause|addressable_source_error(&demands,cause))?
                .map_err(|cause|addressable_source_error(&demands,cause))?;
            Ok((acquisition, compact_spec, compact_routes))
        })();
        match result {
            Ok((acquisition, spec, routes)) => Ok((acquisition, spec, routes, demands)),
            Err(cause) => Err(retain_addressable_demand_failure(demands, cause)),
        }
    }

    fn execute_chunk(
        &mut self,
        spec: &O::Spec,
        owner_unit: usize,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        access: eredu_runtime::ParameterBankAccess,
        census: eredu_runtime::expert::AddressableChunkCensus,
        source_groups: &B::Tensor,
        token_offset: usize,
        unit_observer: &mut Option<&mut dyn eredu_runtime::RoutedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RoutedTextExecutionError> {
        let (acquisition, compact_spec, compact_routes, demands) =
            self.acquire_chunk(spec, owner_unit, routes, access, census, context)?;
        let result = (|| {
            let output = eredu_runtime::with_provider_unit_observer(
                unit_observer,
                source_groups,
                (!self.plan.unit_is_replicated(owner_unit))
                    .then(|| self.plan.local_global_group_indices()),
                token_offset,
                |observer| {
                    O::execute(
                        &mut self.bank,
                        &acquisition,
                        &compact_spec,
                        input,
                        &compact_routes,
                        observer,
                        context,
                    )
                },
            )
            .map_err(RoutedTextExecutionError::from_error)?;
            self.bank
                .complete(acquisition, &output, context)
                .map_err(RoutedTextExecutionError::from_error)?;
            Ok(output)
        })();
        match result {
            Ok(output) => { drop(demands); Ok(output) }
            Err(cause) => Err(retain_addressable_demand_failure(demands, cause)),
        }
    }

    fn with_addressable_source<'a,'observer>(
        &mut self,
        mut request: RoutedExpertRequest<'a,'observer,B::Tensor>,
        partitions: Option<usize>,
        context: &<B::Tensor as Tensor>::Context,
        run:fn(&mut Self,RoutedExpertRequest<'a,'observer,B::Tensor>,Option<usize>,
            eredu_runtime::expert::AddressableChunkPlan,Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
            &<B::Tensor as Tensor>::Context)
            ->Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>,RoutedTextExecutionError>,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>,RoutedTextExecutionError>
    {
        if let Some(metadata)=B::construction_metadata(context) {
            metadata.charge_metadata(eredu_nn::workspace::WorkspaceAddressableRegionView::control_bytes()
                .and_then(|n|n.checked_add(std::mem::size_of::<(
                    &mut Self,RoutedExpertRequest<'_, '_,B::Tensor>,Option<usize>,
                    &<B::Tensor as Tensor>::Context,
                    std::sync::Arc<ExpertRealizationPlan<O::Spec>>,
                    eredu_runtime::expert::AddressableChunkPlan,
                    Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>,RoutedTextExecutionError>,
                )>())).ok_or_else(||RoutedTextExecutionError::from_error(
                    eredu_nn::workspace::WorkspaceMetadataError::Overflow))?)
                .map_err(RoutedTextExecutionError::from_error)?;
        }
        let plan=std::sync::Arc::clone(&self.plan);
        let ((owner_group,_),spec)=plan.unit_specs().iter()
            .find(|((group,unit),_)|group.as_str()==self.owner_group.as_str() && *unit==request.layer)
            .ok_or_else(||RoutedTextExecutionError::from_error(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified))?;
        let expected_routes=self.routes_by_unit.get(&request.layer).copied()
            .ok_or_else(||RoutedTextExecutionError::from_error(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified))?;
        validate_route_cardinality(request.routes,expected_routes)?;
        let geometry=eredu_nn::workspace::ExpertRegionInputShape::inspect(
            request.input.shape(),request.routes.group_indices().shape())
            .map_err(RoutedTextExecutionError::from_error)?;
        let chunks=self.compact_chunk_plan(spec,request.layer,geometry.rows as usize,
            geometry.routes as usize,request.pass.parameter_bank_access(),context)?;
        let source=addressable_source::declaration(owner_group.as_str(),request.bank,request.layer,request.pass,
            chunks,(!plan.unit_is_replicated(request.layer)).then(||plan.local_global_group_indices()),
            O::workspace_kernel(spec),partitions,self.compact_bank_scratch_bytes,self.bulk_compact_bank_target_bytes,
            addressable_source::callback_control_bytes::<B::Tensor>()
                .ok_or_else(||RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?)
            .map_err(RoutedTextExecutionError::from_error)?;
        let input=request.input;
        let routes=request.routes;
        {
            let interested=request.unit_observer.is_some();
            let mut observe=|source:eredu_nn::workspace::WorkspaceAddressableObservationView<'_>|request.unit_observer.as_mut()
                .ok_or_else(||eredu_nn::Error::backend_source(eredu_nn::GroupedUnitError::Unavailable))?
                .observe_addressable_source(source);
            if let Some(output)=B::record_addressable_region_source(source,input,routes,context,
                interested.then_some(&mut observe as &mut dyn for<'view> FnMut(eredu_nn::workspace::WorkspaceAddressableObservationView<'view>)->Result<eredu_nn::workspace::WorkspaceAddressableObservationSource,eredu_nn::Error>))
                .map_err(RoutedTextExecutionError::from_error)? {return Ok(output);}
        }
        let mut callback=addressable_source::Callback {request:Some(request),error:None,chunks,partitions,context,run};
        // Passing the entire named frame prevents disjoint closure captures
        // from changing the native/cold descriptor's callback census.
        let mut invoke=|owner:&mut Self,funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>| {
            owner.invoke_addressable_callback(&mut callback,funding)
        };
        if let Some(metadata)=B::construction_metadata(context) {
            metadata.charge_metadata(source.callback_control_bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        let result=B::with_addressable_region(source,self,input,routes,context,|owner,loan| {
            let mut callback=eredu_runtime::expert::BorrowedIndexedInvocation::new(
                owner,|owner|&mut owner.movement,&mut invoke);
            Movement::with_invocation_source(&mut callback,
                eredu_runtime::expert::IndexedInvocationRequest{declaration:source,input,routes},loan,context)
                .map_err(RoutedTextExecutionError::from_error)?
                .map_err(|()|RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Unqualified))
        }).map_err(RoutedTextExecutionError::from_error);
        drop(invoke);
        // The actual typed failure wins after the native callback has settled or
        // retained its scope, including when that settlement itself refused.
        if let Some(cause)=callback.error { return Err(cause); }
        result?

    }

    fn invoke_addressable_callback(&mut self,callback:&mut addressable_source::Callback<'_,'_,'_,Self,B::Tensor>,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)
        ->Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>,()> {
        let request=callback.request.take().ok_or(())?;
        let result=(callback.run)(self,request,callback.partitions,callback.chunks,funding,callback.context);
        match result {Ok(output)=>Ok(output),Err(cause)=>{callback.error=Some(cause);Err(())}}
    }

    fn execute(
        &mut self,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RoutedTextExecutionError> {
        let (output,bias)=self.with_addressable_source(request,None,context,
            |owner,request,_,chunks,funding,context|owner.execute_body(request,chunks,funding,context)
                .map(|output|eredu_nn::TensorParallelGroupedOutput::new(output,None)))?.into_parts();
        if bias.is_some() {
            return Err(RoutedTextExecutionError::from_error(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified));
        }
        Ok(output)
    }

    fn execute_body(
        &mut self,
        mut request: RoutedExpertRequest<'_, '_, B::Tensor>,
        chunks: eredu_runtime::expert::AddressableChunkPlan,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RoutedTextExecutionError> {
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        let spec=O::clone_spec_with_funding(selected,funding)
            .map_err(RoutedTextExecutionError::from_error)?;
        let routes = self
            .routes_by_unit
            .get(&request.layer)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no route cardinality",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        validate_route_cardinality(request.routes, routes)?;
        let input_shape = request.input.shape();
        let route_shape = request.routes.group_indices().shape();
        let row_count = |shape: &[i32]| {
            shape[..shape.len().saturating_sub(1)]
                .iter()
                .try_fold(1usize, |total, dimension| {
                    usize::try_from(*dimension)
                        .ok()
                        .and_then(|dimension| total.checked_mul(dimension))
                })
        };
        let input_rows = (input_shape.len() >= 2)
            .then(|| row_count(input_shape))
            .flatten();
        let route_rows = (route_shape.len() >= 2)
            .then(|| row_count(route_shape))
            .flatten();
        if input_rows.is_none()
            || input_rows != route_rows
            || request.routes.selected_scores().shape() != route_shape
            || request.routes.coefficients().shape() != route_shape
        {
            return Err(RoutedTextExecutionError::Contract(format!(
                "routed input and selection shapes disagree: input={input_shape:?}, routes={route_shape:?}"
            )));
        }
        let hidden = *input_shape.last().expect("validated hidden axis");
        let selections_per_row = *route_shape.last().expect("validated selection axis");
        let rows = input_rows.expect("validated routed row geometry");
        let flat_rows = i32::try_from(rows).map_err(|_| {
            RoutedTextExecutionError::Contract("routed row count exceeds i32".into())
        })?;
        let input = request
            .input
            .reshape(&[flat_rows, hidden], context)
            .map_err(RoutedTextExecutionError::from_error)?;
        let flatten_routes = |value: &B::Tensor| {
            value
                .reshape(&[flat_rows, selections_per_row], context)
                .map_err(RoutedTextExecutionError::from_error)
        };
        let group_indices = flatten_routes(request.routes.group_indices())?;
        let selected_scores = flatten_routes(request.routes.selected_scores())?;
        let coefficients = flatten_routes(request.routes.coefficients())?;
        let access = request.pass.parameter_bank_access();
        debug_assert_eq!(chunks.workspace_source().rows,rows);
        debug_assert_eq!(chunks.workspace_source().routes,selections_per_row as usize);
        let mut outputs=match funding {
            Some(funding)=>funding.metadata_vec(chunks.len()).map_err(RoutedTextExecutionError::from_error)?,
            None=>Vec::new(),
        };
        for (chunk_index, range) in chunks.ranges().enumerate() {
            let (start, end) = (range.start, range.end);
            let select = |movement: &mut Movement, value: &B::Tensor| {
                movement
                    .select_rows(value, start, end, context)
                    .map_err(RoutedTextExecutionError::from_error)
            };
            let chunk_input = select(&mut self.movement, &input)?;
            let chunk_indices = select(&mut self.movement, &group_indices)?;
            let chunk_scores = select(&mut self.movement, &selected_scores)?;
            let chunk_coefficients = select(&mut self.movement, &coefficients)?;
            let chunk_routes = GroupSelection::new(chunk_indices, chunk_scores, chunk_coefficients);
            outputs.push(self.execute_chunk(
                &spec,
                request.layer,
                &chunk_input,
                &chunk_routes,
                access,
                eredu_runtime::expert::AddressableChunkCensus::new(
                    request.bank.value() as usize, request.layer, chunks, chunk_index, access)
                    .expect("ordinal from the selected chunk plan"),
                &group_indices,
                start,
                &mut request.unit_observer,
                context,
            )?);
        }
        let output = if outputs.len() == 1 {
            outputs.pop().expect("one routed output")
        } else {
            self.movement
                .concatenate_rows(&outputs, context)
                .map_err(RoutedTextExecutionError::from_error)?
        };
        let mut output_shape=match funding {
            Some(funding)=>{
                let mut shape=funding.metadata_vec(input_shape.len()).map_err(RoutedTextExecutionError::from_error)?;
                shape.extend_from_slice(input_shape);shape
            }
            None=>input_shape.to_vec(),
        };
        *output_shape.last_mut().expect("validated hidden axis") = O::output_dimensions(&spec);
        output
            .reshape(&output_shape, context)
            .map_err(RoutedTextExecutionError::from_error)
    }
}

impl<B, Bank, Movement> RoutedExpertProvider<B>
    for PlannedAddressableGrouped<GatedProductOperation, B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.execute(request, context)
    }

    /// Executes an activated selected-linear bank with owned output rows.
    fn forward_linear_routed(
        &mut self,
        _: &mut B::LinearGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected provider equation is not a linear bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

impl<O, B, Bank, Movement> PlannedAddressableGrouped<O, B, Bank, Movement>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    O: TensorParallelRoutedGroupedOperation<B>,
{
    fn execute_chunk_tensor_parallel(
        &mut self,
        spec: &O::Spec,
        owner_unit: usize,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        access: eredu_runtime::ParameterBankAccess,
        census: eredu_runtime::expert::AddressableChunkCensus,
        partitions: usize,
        source_groups: &B::Tensor,
        token_offset: usize,
        unit_observer: &mut Option<&mut dyn eredu_runtime::RoutedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, RoutedTextExecutionError> {
        let (acquisition, compact_spec, compact_routes, demands) =
            self.acquire_chunk(spec, owner_unit, routes, access, census, context)?;
        let result = (|| {
            let output = eredu_runtime::with_provider_unit_observer(
                unit_observer,
                source_groups,
                (!self.plan.unit_is_replicated(owner_unit))
                    .then(|| self.plan.local_global_group_indices()),
                token_offset,
                |observer| {
                    O::execute_tensor_parallel(
                        &mut self.bank,
                        &acquisition,
                        &compact_spec,
                        input,
                        &compact_routes,
                        partitions,
                        observer,
                        context,
                    )
                },
            )
            .map_err(RoutedTextExecutionError::from_error)?;
            self.bank
                .complete(acquisition, output.reducible(), context)
                .map_err(RoutedTextExecutionError::from_error)?;
            Ok(output)
        })();
        match result {
            Ok(output) => { drop(demands); Ok(output) }
            Err(cause) => Err(retain_addressable_demand_failure(demands, cause)),
        }
    }

    fn execute_tensor_parallel(
        &mut self,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, RoutedTextExecutionError> {
        self.with_addressable_source(request,Some(partitions),context,
            |owner,request,partitions,chunks,funding,context|owner.execute_tensor_parallel_body(request,
                partitions.ok_or_else(||RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Unqualified))?,
                chunks,funding,context))
    }

    fn execute_tensor_parallel_body(
        &mut self,
        mut request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        chunks: eredu_runtime::expert::AddressableChunkPlan,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::TensorParallelGroupedOutput<B::Tensor>, RoutedTextExecutionError> {
        let selected = self
            .plan
            .unit_spec(self.owner_group.as_str(), request.layer)
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no grouped bank specification",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        let spec=O::clone_spec_with_funding(selected,funding)
            .map_err(RoutedTextExecutionError::from_error)?;
        let routes = self
            .routes_by_unit
            .get(&request.layer)
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "execution unit {:?}/{} has no route cardinality",
                    self.owner_group.as_str(),
                    request.layer
                ))
            })?;
        validate_route_cardinality(request.routes, routes)?;
        let input_shape = request.input.shape();
        let route_shape = request.routes.group_indices().shape();
        let row_count = |shape: &[i32]| {
            shape[..shape.len().saturating_sub(1)]
                .iter()
                .try_fold(1usize, |total, dimension| {
                    usize::try_from(*dimension)
                        .ok()
                        .and_then(|dimension| total.checked_mul(dimension))
                })
        };
        let input_rows = (input_shape.len() >= 2)
            .then(|| row_count(input_shape))
            .flatten();
        let route_rows = (route_shape.len() >= 2)
            .then(|| row_count(route_shape))
            .flatten();
        if input_rows.is_none()
            || input_rows != route_rows
            || request.routes.selected_scores().shape() != route_shape
            || request.routes.coefficients().shape() != route_shape
        {
            return Err(RoutedTextExecutionError::Contract(format!(
                "routed input and selection shapes disagree: input={input_shape:?}, routes={route_shape:?}"
            )));
        }
        let hidden = *input_shape.last().expect("validated hidden axis");
        let selections_per_row = *route_shape.last().expect("validated selection axis");
        let rows = input_rows.expect("validated routed row geometry");
        let flat_rows = i32::try_from(rows).map_err(|_| {
            RoutedTextExecutionError::Contract("routed row count exceeds i32".into())
        })?;
        let input = request
            .input
            .reshape(&[flat_rows, hidden], context)
            .map_err(RoutedTextExecutionError::from_error)?;
        let flatten_routes = |value: &B::Tensor| {
            value
                .reshape(&[flat_rows, selections_per_row], context)
                .map_err(RoutedTextExecutionError::from_error)
        };
        let group_indices = flatten_routes(request.routes.group_indices())?;
        let selected_scores = flatten_routes(request.routes.selected_scores())?;
        let coefficients = flatten_routes(request.routes.coefficients())?;
        let access = request.pass.parameter_bank_access();
        debug_assert_eq!(chunks.workspace_source().rows,rows);
        debug_assert_eq!(chunks.workspace_source().routes,selections_per_row as usize);
        let mut reducible=match funding {
            Some(funding)=>funding.metadata_vec(chunks.len()).map_err(RoutedTextExecutionError::from_error)?,
            None=>Vec::new(),
        };
        let mut post_reduce=match funding {
            Some(funding)=>funding.metadata_vec(chunks.len()).map_err(RoutedTextExecutionError::from_error)?,
            None=>Vec::new(),
        };
        for (chunk_index, range) in chunks.ranges().enumerate() {
            let (start, end) = (range.start, range.end);
            let select = |movement: &mut Movement, value: &B::Tensor| {
                movement
                    .select_rows(value, start, end, context)
                    .map_err(RoutedTextExecutionError::from_error)
            };
            let chunk_input = select(&mut self.movement, &input)?;
            let chunk_indices = select(&mut self.movement, &group_indices)?;
            let chunk_scores = select(&mut self.movement, &selected_scores)?;
            let chunk_coefficients = select(&mut self.movement, &coefficients)?;
            let chunk_routes = GroupSelection::new(chunk_indices, chunk_scores, chunk_coefficients);
            let output = self.execute_chunk_tensor_parallel(
                &spec,
                request.layer,
                &chunk_input,
                &chunk_routes,
                access,
                eredu_runtime::expert::AddressableChunkCensus::new(
                    request.bank.value() as usize, request.layer, chunks, chunk_index, access)
                    .expect("ordinal from the selected chunk plan"),
                partitions,
                &group_indices,
                start,
                &mut request.unit_observer,
                context,
            )?;
            let (chunk_reducible, chunk_post_reduce) = output.into_parts();
            reducible.push(chunk_reducible);
            post_reduce.push(chunk_post_reduce);
        }
        let concatenate = |movement: &mut Movement, mut values: Vec<B::Tensor>| {
            if values.len() == 1 {
                Ok(values.pop().expect("one routed tensor-parallel output"))
            } else {
                movement
                    .concatenate_rows(&values, context)
                    .map_err(RoutedTextExecutionError::from_error)
            }
        };
        let reducible = concatenate(&mut self.movement, reducible)?
            .reshape(input_shape, context)
            .map_err(RoutedTextExecutionError::from_error)?;
        let post_reduce = if post_reduce.iter().all(Option::is_none) {
            None
        } else if post_reduce.iter().all(Option::is_some) {
            Some(
                concatenate(
                    &mut self.movement,
                    match funding {
                        Some(funding)=>{
                            let mut values=funding.metadata_vec(post_reduce.len()).map_err(RoutedTextExecutionError::from_error)?;
                            values.extend(post_reduce.into_iter().flatten());values
                        }
                        None=>post_reduce.into_iter().flatten().collect(),
                    },
                )?
                .reshape(input_shape, context)
                .map_err(RoutedTextExecutionError::from_error)?,
            )
        } else {
            return Err(RoutedTextExecutionError::Contract(
                "tensor-parallel compact chunks disagree on post-reduction bias".into(),
            ));
        };
        Ok(eredu_nn::TensorParallelGroupedOutput::new(
            reducible,
            post_reduce,
        ))
    }
}

impl<B, Bank, Movement> eredu_runtime::TensorParallelRoutedExpertProvider<B>
    for PlannedAddressableGrouped<GatedProductOperation, B, Bank, Movement>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.execute_tensor_parallel(request, partitions, context)
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.forward_grouped_tensor_parallel(resident_bank, request, partitions, context)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a gated-product execution plan cannot invoke a ReLU-squared bank".into(),
        ))
    }
}

impl<B, Bank, Movement> eredu_runtime::TensorParallelRoutedExpertProvider<B>
    for PlannedAddressableGrouped<Relu2Operation, B, Bank, Movement>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.execute_tensor_parallel(request, partitions, context)
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
    }
}

impl<B, Bank, Movement> RoutedExpertProvider<B>
    for PlannedAddressableGrouped<Relu2Operation, B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "a ReLU-squared execution plan cannot invoke a gated-product bank".into(),
        ))
    }

    /// Executes an activated selected-linear bank with owned output rows.
    fn forward_linear_routed(
        &mut self,
        _: &mut B::LinearGroups,
        _request: RoutedExpertRequest<'_, '_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected provider equation is not a linear bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.execute(request, context)
    }
}

fn validate_replicated_plan<S>(
    plan: &ExpertRealizationPlan<S>,
) -> Result<(), RoutedTextExecutionError> {
    if plan.expert_parallel_size() != 1
        || plan.expert_parallel_rank() != 0
        || plan.local_global_group_indices() != (0..plan.global_expert_count()).collect::<Vec<_>>()
    {
        return Err(RoutedTextExecutionError::Contract(
            "routed text execution requires a complete replicated group plan".into(),
        ));
    }
    Ok(())
}

fn validate_catalog_binding<B, Bank>(
    catalog: &ExpertResidencyCatalog,
    selected_member_bytes: &BTreeMap<ParameterBankKey, u64>,
    bank: &Bank,
) -> Result<(), RoutedTextExecutionError>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
{
    let addressable_units = catalog
        .units()
        .iter()
        .filter(|unit| unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel);
    if selected_member_bytes.len() != addressable_units.clone().count() {
        return Err(RoutedTextExecutionError::Contract(
            "selected addressable member geometry differs from the architecture catalog".into(),
        ));
    }
    for unit in addressable_units {
        let expected = selected_member_bytes
            .get(&unit.identity())
            .copied()
            .ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "bank member {:?} has no selected byte geometry",
                    unit.identity()
                ))
            })?;
        if bank.member_bytes(unit.identity()) != Some(expected) {
            return Err(RoutedTextExecutionError::Contract(format!(
                "bank member {:?} byte geometry differs from admitted {expected} bytes",
                unit.identity(),
            )));
        }
    }
    Ok(())
}

// Checked before acquiring or constructing any native bank values.
fn validate_partitioned_catalog_binding<O: RoutedGroupedOperationValidation>(
    owner_group: &eredu_runtime::ExecutionGroupId,
    plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
    selected_member_bytes: &BTreeMap<ParameterBankKey, u64>,
    member_bytes: impl Fn(ParameterBankKey) -> Option<u64>,
) -> Result<(), RoutedTextExecutionError> {
    if plan.local_global_group_indices().is_empty()
        || plan
            .unit_specs()
            .keys()
            .any(|(group, _)| group != owner_group)
    {
        return Err(RoutedTextExecutionError::Contract(
            "partitioned addressable plan has no local experts or names a different owner group"
                .into(),
        ));
    }
    let bank_id = catalog
        .units()
        .first()
        .ok_or_else(|| {
            RoutedTextExecutionError::Contract("partitioned bank catalog is empty".into())
        })?
        .identity()
        .bank();
    if catalog
        .units()
        .iter()
        .any(|unit| unit.identity().bank() != bank_id)
    {
        return Err(RoutedTextExecutionError::Contract(
            "partitioned catalog mixes bank identities".into(),
        ));
    }
    let selected_units = selected_member_bytes
        .keys()
        .map(|key| key.unit())
        .collect::<BTreeSet<_>>();
    if selected_units.is_empty() {
        return Err(RoutedTextExecutionError::Contract(
            "partitioned addressable catalog selected no local units".into(),
        ));
    }
    for unit in &selected_units {
        if plan.unit_spec(owner_group.as_str(), *unit).is_none() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "partitioned addressable selection names unplanned unit {unit}"
            )));
        }
    }
    let mut expected_keys = BTreeSet::new();
    for ((group, unit), spec) in plan
        .unit_specs()
        .iter()
        .filter(|((_, unit), _)| selected_units.contains(unit))
    {
        let local_count = usize::try_from(O::group_count(spec)).map_err(|_| {
            RoutedTextExecutionError::Contract(
                "partitioned addressable group count is not representable".into(),
            )
        })?;
        let replicated = plan.unit_is_replicated(*unit);
        if !replicated && local_count != plan.local_global_group_indices().len() {
            return Err(RoutedTextExecutionError::Contract(format!(
                "partitioned addressable unit {:?}/{unit} has {local_count} local groups, expected {}",
                group.as_str(),
                plan.local_global_group_indices().len()
            )));
        }
        for local in 0..local_count {
            let global = if replicated {
                local
            } else {
                plan.local_global_group_indices()[local]
            };
            let key = ParameterBankKey::new(bank_id, *unit, global);
            expected_keys.insert(key);
            let selected = selected_member_bytes.get(&key).copied().ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "partitioned addressable unit {:?}/{unit} is missing global expert {global}",
                    group.as_str()
                ))
            })?;
            let catalog_unit = catalog.unit(key).ok_or_else(|| {
                RoutedTextExecutionError::Contract(format!(
                    "partitioned addressable catalog is missing {key:?}"
                ))
            })?;
            let expected_distribution = if replicated {
                crate::ExpertResidencyDistribution::Replicated
            } else {
                crate::ExpertResidencyDistribution::ExpertParallel
            };
            if catalog_unit.owner_group() != group
                || catalog_unit.distribution() != expected_distribution
                || member_bytes(key) != Some(selected)
            {
                return Err(RoutedTextExecutionError::Contract(format!(
                    "partitioned addressable member {key:?} differs from selected ownership or byte geometry"
                )));
            }
        }
    }
    if selected_member_bytes
        .keys()
        .copied()
        .collect::<BTreeSet<_>>()
        != expected_keys
    {
        return Err(RoutedTextExecutionError::Contract(
            "partitioned addressable selection differs from exact local member identities".into(),
        ));
    }
    Ok(())
}

fn validate_plan_catalog<O>(
    owner_group: &eredu_runtime::ExecutionGroupId,
    plan: &ExpertRealizationPlan<O::Spec>,
    catalog: &ExpertResidencyCatalog,
) -> Result<(), RoutedTextRequirementsError>
where
    O: RoutedGroupedOperationValidation,
{
    let mut by_request = BTreeMap::<_, Vec<_>>::new();
    let mut by_owner = BTreeMap::<_, Vec<_>>::new();
    let mut by_unit = BTreeMap::<_, Vec<_>>::new();
    for unit in catalog.units() {
        by_request
            .entry((
                unit.owner_group(),
                unit.identity().unit(),
                unit.identity().member(),
            ))
            .or_default()
            .push(unit);
        by_owner
            .entry((
                unit.owner_group(),
                unit.owner_unit(),
                unit.identity().member(),
            ))
            .or_default()
            .push(unit);
        by_unit
            .entry((unit.owner_group(), unit.identity().unit()))
            .or_default()
            .push(unit);
    }
    for ((group, request_unit), spec) in plan.unit_specs() {
        if group != owner_group {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "routed unit group {:?} differs from selected group {:?}",
                group.as_str(),
                owner_group.as_str()
            )));
        }
        let member_count = usize::try_from(O::group_count(spec)).map_err(|_| {
            RoutedTextRequirementsError::Invalid(format!(
                "grouped bank for {:?}/{request_unit} has invalid group count {}",
                group.as_str(),
                O::group_count(spec)
            ))
        })?;
        for member in 0..member_count {
            let key = (group, *request_unit, member);
            let matches = by_request.get(&key).or_else(|| by_owner.get(&key));
            let matches = matches.map(Vec::as_slice).unwrap_or_default();
            let [unit] = matches else {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog must contain one member {member} for {:?}/{request_unit}",
                    group.as_str()
                )));
            };
            if unit.identity().unit() != *request_unit || unit.identity().member() != member {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog member {:?} does not use request unit {request_unit} as its key namespace",
                    unit.identity()
                )));
            }
            let expected = O::member_parameter_targets(spec, member)
                .map_err(|error| RoutedTextRequirementsError::Invalid(error.to_string()))?
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>();
            let mut actual = std::collections::BTreeSet::new();
            for parameter in unit.parameters() {
                let target = parameter.logical_target();
                let binding = parameter.binding_name();
                let prefix = target.strip_suffix(binding).ok_or_else(|| {
                    RoutedTextRequirementsError::Invalid(format!(
                        "bank binding {binding:?} does not name the suffix of logical target {target:?}"
                    ))
                })?;
                actual.insert(target.to_owned());
                if let crate::ExpertParameterRole::QuantizableProjection {
                    scales_binding,
                    biases_binding,
                } = parameter.role()
                {
                    for companion in [scales_binding, biases_binding] {
                        let target = format!("{prefix}{companion}");
                        if expected.contains(&target) {
                            actual.insert(target);
                        }
                    }
                }
            }
            if actual != expected {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog member {:?} targets {actual:?}, expected grouped targets {expected:?}",
                    unit.identity()
                )));
            }
        }
        let mut unit_facts = by_unit.get(&(group, *request_unit)).into_iter().flatten();
        if let Some(first) = unit_facts.next() {
            if unit_facts.any(|unit| {
                unit.unit_path() != first.unit_path()
                    || unit.owner_unit() != first.owner_unit()
                    || unit.distribution() != first.distribution()
            }) {
                return Err(RoutedTextRequirementsError::Invalid(format!(
                    "bank catalog members for {:?}/{request_unit} disagree on path, owner, or distribution",
                    group.as_str()
                )));
            }
        }
    }
    for unit in catalog.units() {
        let spec = plan.unit_spec(unit.owner_group().as_str(), unit.identity().unit());
        let member_count = spec
            .and_then(|spec| usize::try_from(O::group_count(spec)).ok())
            .unwrap_or_default();
        if unit.owner_group() != owner_group
            || spec.is_none()
            || unit.identity().member() >= member_count
        {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "bank catalog member {:?} is outside the replicated routed plan",
                unit.identity()
            )));
        }
        if unit.byte_len().is_none() {
            return Err(RoutedTextRequirementsError::Invalid(format!(
                "bank catalog member {:?} has no admitted byte geometry",
                unit.identity()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn partitioned_addressable_members_keep_distinct_shared_geometry() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let make_catalog = |shared_distribution, shared_owner: &eredu_runtime::ExecutionGroupId| {
            ExpertResidencyCatalog::new((0..2).flat_map(|unit| {
                let count = if unit == 0 { 2 } else { 4 };
                let owner = if unit == 0 {
                    owner.clone()
                } else {
                    shared_owner.clone()
                };
                (0..count).map(move |member| {
                    ExpertResidencyUnit::new(
                        ParameterBankKey::new(0, unit, member),
                        owner.clone(),
                        0,
                        "decoder.layers.0.mlp",
                        if unit == 0 {
                            ExpertResidencyDistribution::ExpertParallel
                        } else {
                            shared_distribution
                        },
                        [ExpertParameterRecipe::new(
                            "gate_up_proj",
                            format!("test.bank{unit}.gate_up_proj"),
                            DerivedWeightRecipe::source(
                                format!("source.{unit}.{member}"),
                                TensorSelection::Full,
                            ),
                            ExpertParameterRole::Preserved,
                        )
                        .unwrap()],
                    )
                    .unwrap()
                    .with_byte_len(100 + member as u64)
                    .unwrap()
                })
            }))
            .unwrap()
        };
        let catalog = make_catalog(ExpertResidencyDistribution::Replicated, &owner);
        let plan = ExpertRealizationPlan::balanced(
            2,
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 2, 1).unwrap(), 1).unwrap(),
            BTreeMap::from([
                (
                    (owner.clone(), 0),
                    grouped_spec().with_group_geometry(1, 8).unwrap(),
                ),
                (
                    (owner.clone(), 1),
                    grouped_spec().with_group_geometry(4, 8).unwrap(),
                ),
            ]),
        )
        .unwrap()
        .with_catalog_distribution(&catalog)
        .unwrap();
        // Rank one owns routed global ID 1 and every shared ID, including IDs
        // larger than the complete routed bank. Both invocations share owner 0.
        let bytes = BTreeMap::from([
            (ParameterBankKey::new(0, 0, 1), 101),
            (ParameterBankKey::new(0, 1, 0), 100),
            (ParameterBankKey::new(0, 1, 1), 101),
            (ParameterBankKey::new(0, 1, 2), 102),
            (ParameterBankKey::new(0, 1, 3), 103),
        ]);
        let validate = |catalog: &ExpertResidencyCatalog, selected: &BTreeMap<_, _>| {
            validate_partitioned_catalog_binding::<GatedProductOperation>(
                &owner,
                &plan,
                catalog,
                selected,
                |key| bytes.get(&key).copied(),
            )
        };
        validate(&catalog, &bytes).unwrap();
        let global_routes = BTreeMap::from([(0, 2), (1, 4)]);
        let routes = RoutedGroupedPlan::Gated(plan.clone()).partition_routes(&global_routes, true);
        assert_eq!(routes, BTreeMap::from([(0, 1), (1, 4)]));
        validate_routes_by_unit::<GatedProductOperation>(&plan, &routes).unwrap();
        for mutation in 0..5 {
            let mut invalid = bytes.clone();
            match mutation {
                0 => {
                    invalid.remove(&ParameterBankKey::new(0, 1, 3));
                }
                1 => {
                    invalid.insert(ParameterBankKey::new(0, 2, 0), 100);
                }
                2 => {
                    invalid.insert(ParameterBankKey::new(0, 0, 0), 100);
                }
                3 => {
                    invalid.insert(ParameterBankKey::new(1, 1, 0), 100);
                }
                _ => {
                    invalid.insert(ParameterBankKey::new(0, 1, 0), 999);
                }
            }
            assert!(validate(&catalog, &invalid).is_err(), "mutation {mutation}");
        }
        let wrong_distribution = make_catalog(ExpertResidencyDistribution::ExpertParallel, &owner);
        assert!(validate(&wrong_distribution, &bytes).is_err());
        let foreign = eredu_runtime::ExecutionGroupId::new("foreign").unwrap();
        let wrong_owner = make_catalog(ExpertResidencyDistribution::Replicated, &foreign);
        assert!(validate(&wrong_owner, &bytes).is_err());
    }

    #[test]
    fn independently_blocked_gate_up_companions_match_source_and_local_shapes() {
        let format = eredu_nn::LinearFormatSpec::scaled(
            eredu_checkpoint::LinearFormat::E4M3BlockFp8(
                eredu_checkpoint::BlockFp8Format::new(
                    128,
                    128,
                    eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint,
                )
                .unwrap(),
            ),
            eredu_nn::ParameterSpec::trainable("scales").unwrap(),
        )
        .unwrap()
        .with_row_layout(eredu_nn::LinearRowLayout::equal_partitions(2).unwrap())
        .unwrap();
        assert_eq!(
            super::grouped_companion_shape(&format, &[3, 518, 130]),
            Some(vec![3, 6, 2])
        );
        assert_eq!(
            super::grouped_companion_shape(&format, &[1, 6, 130]),
            Some(vec![1, 2, 2])
        );
        assert_eq!(
            super::grouped_companion_shape(&format, &[2, 512, 130]),
            Some(vec![2, 4, 2])
        );
    }

    use super::*;
    use crate::{
        ExpertParameterRecipe, ExpertParameterRole, ExpertResidencyDistribution,
        ExpertResidencyUnit,
    };
    use eredu_checkpoint::{
        recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeDtype, RecipeMetadata},
        store::{StoreError, TensorMetadata, TensorSelection},
        LinearFormat, SourceTensorEncoding, StoredDtype,
    };
    use eredu_core::{
        cache::LayerCachePolicy, AttentionPolicy, LayerSchedule, ParallelRankTopology,
        ParallelTopology,
    };
    use eredu_nn::{
        GatedProductGroupLayout, GatedProductPolicy, GroupedGatedProductSpec,
        GroupedProjectionSpec, LinearFormatSpec, ParameterSpec,
    };
    use eredu_runtime::{
        ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
        ArchitectureMergeDestination, ExecutionGraph, ExecutionUnitLayout,
        ParameterTransformConstraint, ReplicatedTextParameterOwner,
        ReplicatedTextParameterPresence, ReplicatedTextParameterRequirement,
        ReplicatedTextParameterRole, ReplicatedTextPhysicalSource, ReplicatedTextStateAccess,
        StateLayout,
    };

    fn grouped_spec() -> GroupedGatedProductSpec {
        let projection = |name| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(name).unwrap(),
                None,
                LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            )
            .unwrap()
        };
        GroupedGatedProductSpec::new(
            2,
            4,
            4,
            4,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("test.experts.gate_up_proj"),
                down: projection("test.experts.down_proj"),
            },
        )
        .unwrap()
    }

    struct TestRecipeCatalog;

    impl RecipeCatalog for TestRecipeCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            let shape = if key == "source.down" {
                vec![2, 4, 4]
            } else if key == "source.value" {
                vec![3, 6, 4]
            } else {
                vec![2, 8, 4]
            };
            Ok(TensorMetadata {
                name: key.into(),
                logical_shape: shape.clone(),
                physical_shape: shape.clone(),
                stored_dtype: StoredDtype::F16,
                encoded_byte_len: (shape.iter().product::<usize>() * 2) as u64,
                backing_shard: None,
            })
        }
    }

    struct WrongAffineCompanionCatalog;

    impl RecipeCatalog for WrongAffineCompanionCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            let shape = match key {
                "source.gate.weight" => vec![2, 8, 4],
                "source.gate.scales" => vec![2, 8, 2],
                "source.gate.biases" => vec![2, 8, 1],
                "source.down.weight" => vec![2, 4, 4],
                "source.down.scales" | "source.down.biases" => vec![2, 4, 1],
                _ => return Err(StoreError::UnknownTensor { key: key.into() }),
            };
            Ok(TensorMetadata {
                name: key.into(),
                logical_shape: shape.clone(),
                physical_shape: shape.clone(),
                stored_dtype: StoredDtype::F16,
                encoded_byte_len: (shape.iter().product::<usize>() * 2) as u64,
                backing_shard: None,
            })
        }
    }

    fn plan(
        owner: &eredu_runtime::ExecutionGroupId,
    ) -> ExpertRealizationPlan<GroupedGatedProductSpec> {
        ExpertRealizationPlan::balanced(
            2,
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
            BTreeMap::from([((owner.clone(), 0), grouped_spec())]),
        )
        .unwrap()
    }

    fn catalog(
        owner: &eredu_runtime::ExecutionGroupId,
        key_namespace: usize,
        include_down: bool,
        gate_source: &str,
    ) -> ExpertResidencyCatalog {
        ExpertResidencyCatalog::new((0..2).map(|member| {
            let selection = TensorSelection::Range {
                axis: 0,
                start: member,
                end: member + 1,
            };
            let mut parameters = vec![ExpertParameterRecipe::new(
                "gate_up_proj",
                "test.experts.gate_up_proj",
                DerivedWeightRecipe::source(gate_source, selection.clone()),
                ExpertParameterRole::Preserved,
            )
            .unwrap()];
            if include_down {
                parameters.push(
                    ExpertParameterRecipe::new(
                        "down_proj",
                        "test.experts.down_proj",
                        DerivedWeightRecipe::source("source.down", selection),
                        ExpertParameterRole::Preserved,
                    )
                    .unwrap(),
                );
            }
            ExpertResidencyUnit::new(
                ParameterBankKey::new(0, key_namespace, member),
                owner.clone(),
                0,
                "decoder.layers.0.mlp",
                ExpertResidencyDistribution::ExpertParallel,
                parameters,
            )
            .unwrap()
        }))
        .unwrap()
        .with_inferred_byte_geometry(&TestRecipeCatalog)
        .unwrap()
    }

    fn text_requirements() -> eredu_runtime::ReplicatedTextRequirements {
        text_requirements_with_linear(false)
    }

    fn text_requirements_with_linear(
        include_linear: bool,
    ) -> eredu_runtime::ReplicatedTextRequirements {
        let graph = ExecutionGraph::chain(["decoder"]).unwrap();
        let parameter = |name: &str, source: &str| {
            let shape = if name.ends_with("gate_up_proj") {
                vec![2, 8, 4]
            } else if name == "test.values.weight" {
                vec![3, 6, 4]
            } else {
                vec![2, 4, 4]
            };
            ReplicatedTextParameterRequirement::new(
                name,
                vec![source.into()],
                vec![ReplicatedTextPhysicalSource::new(
                    source,
                    source,
                    "/checkpoint/model.safetensors",
                    source,
                    SourceTensorEncoding::Safetensors(StoredDtype::F16),
                    (shape.iter().product::<usize>() * 2) as u64,
                )
                .unwrap()],
                Vec::new(),
                Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
                Some(shape.clone()),
                shape,
                LinearFormat::Dense,
                ReplicatedTextParameterRole::LinearWeight,
                ReplicatedTextParameterOwner::ExecutionUnit {
                    group: "decoder".into(),
                    unit: 0,
                },
                ReplicatedTextParameterPresence::Required,
                ParameterTransformConstraint::Linear { packed_axis: 2 },
            )
            .unwrap()
        };
        eredu_runtime::ReplicatedTextRequirements::new(
            "test.routed-catalog",
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            vec![ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            ReplicatedTextStateAccess::KeyValue,
            {
                let mut parameters = vec![
                    parameter("test.experts.gate_up_proj", "source.gate"),
                    parameter("test.experts.down_proj", "source.down"),
                ];
                if include_linear {
                    parameters.push(parameter("test.values.weight", "source.value"));
                }
                parameters
            },
        )
        .unwrap()
    }

    fn affine_plan_and_catalog(
        owner: &eredu_runtime::ExecutionGroupId,
    ) -> (
        ExpertRealizationPlan<GroupedGatedProductSpec>,
        ExpertResidencyCatalog,
    ) {
        let affine =
            LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(16, 4).unwrap());
        let projection = |weight: &str, scales: &str, biases: &str| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(weight).unwrap(),
                None,
                LinearFormatSpec::affine(
                    affine,
                    ParameterSpec::trainable(scales).unwrap(),
                    ParameterSpec::trainable(biases).unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
        };
        let spec = GroupedGatedProductSpec::new(
            2,
            4,
            4,
            4,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection(
                    "test.experts.gate_up_proj",
                    "test.experts.gate_up_proj_scales",
                    "test.experts.gate_up_proj_biases",
                ),
                down: projection(
                    "test.experts.down_proj",
                    "test.experts.down_proj_scales",
                    "test.experts.down_proj_biases",
                ),
            },
        )
        .unwrap();
        let plan = ExpertRealizationPlan::balanced(
            2,
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
            BTreeMap::from([((owner.clone(), 0), spec)]),
        )
        .unwrap();
        let bindings = [
            ("gate_up_proj", "source.gate.weight"),
            ("gate_up_proj_scales", "source.gate.scales"),
            ("gate_up_proj_biases", "source.gate.biases"),
            ("down_proj", "source.down.weight"),
            ("down_proj_scales", "source.down.scales"),
            ("down_proj_biases", "source.down.biases"),
        ];
        let catalog = ExpertResidencyCatalog::new((0..2).map(|member| {
            let selection = TensorSelection::Range {
                axis: 0,
                start: member,
                end: member + 1,
            };
            let parameters = bindings
                .iter()
                .map(|(binding, source)| {
                    ExpertParameterRecipe::new(
                        *binding,
                        format!("test.experts.{binding}"),
                        DerivedWeightRecipe::source(*source, selection.clone()),
                        ExpertParameterRole::Preserved,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            ExpertResidencyUnit::new(
                ParameterBankKey::new(0, 0, member),
                owner.clone(),
                0,
                "decoder.layers.0.mlp",
                ExpertResidencyDistribution::ExpertParallel,
                parameters,
            )
            .unwrap()
        }))
        .unwrap()
        .with_inferred_byte_geometry(&WrongAffineCompanionCatalog)
        .unwrap();
        (plan, catalog)
    }

    fn affine_text_requirements() -> eredu_runtime::ReplicatedTextRequirements {
        let graph = ExecutionGraph::chain(["decoder"]).unwrap();
        let parameter =
            |name: &str, source: &str, shape: Vec<usize>, role: ReplicatedTextParameterRole| {
                ReplicatedTextParameterRequirement::new(
                    name,
                    vec![source.into()],
                    vec![ReplicatedTextPhysicalSource::new(
                        source,
                        source,
                        "/checkpoint/model.safetensors",
                        source,
                        SourceTensorEncoding::Safetensors(StoredDtype::F16),
                        (shape.iter().product::<usize>() * 2) as u64,
                    )
                    .unwrap()],
                    Vec::new(),
                    Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
                    Some(shape.clone()),
                    shape,
                    LinearFormat::Dense,
                    role,
                    ReplicatedTextParameterOwner::ExecutionUnit {
                        group: "decoder".into(),
                        unit: 0,
                    },
                    ReplicatedTextParameterPresence::Required,
                    ParameterTransformConstraint::None,
                )
                .unwrap()
            };
        eredu_runtime::ReplicatedTextRequirements::new(
            "test.affine-routed-catalog",
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            vec![ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            ReplicatedTextStateAccess::KeyValue,
            vec![
                parameter(
                    "test.experts.gate_up_proj",
                    "source.gate.weight",
                    vec![2, 8, 4],
                    ReplicatedTextParameterRole::LinearWeight,
                ),
                parameter(
                    "test.experts.gate_up_proj_scales",
                    "source.gate.scales",
                    vec![2, 8, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
                parameter(
                    "test.experts.gate_up_proj_biases",
                    "source.gate.biases",
                    vec![2, 8, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
                parameter(
                    "test.experts.down_proj",
                    "source.down.weight",
                    vec![2, 4, 4],
                    ReplicatedTextParameterRole::LinearWeight,
                ),
                parameter(
                    "test.experts.down_proj_scales",
                    "source.down.scales",
                    vec![2, 4, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
                parameter(
                    "test.experts.down_proj_biases",
                    "source.down.biases",
                    vec![2, 4, 1],
                    ReplicatedTextParameterRole::FormatCompanion,
                ),
            ],
        )
        .unwrap()
    }

    #[test]
    fn identified_banks_keep_distinct_equations_counts_and_routes() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let gated = RoutedBankRequirements::new(
            owner.clone(),
            plan(&owner).into(),
            catalog(&owner, 0, true, "source.gate"),
            BTreeMap::from([(0, 1)]),
        )
        .unwrap();
        let linear_spec = eredu_nn::GroupedLinearSpec::new(
            3,
            4,
            6,
            eredu_nn::GroupedLinearActivation::Silu,
            GroupedProjectionSpec::new(
                ParameterSpec::trainable("test.values.weight").unwrap(),
                None,
                LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let linear_plan = ExpertRealizationPlan::balanced(
            3,
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
            BTreeMap::from([((owner.clone(), 0), linear_spec)]),
        )
        .unwrap();
        let linear_catalog = ExpertResidencyCatalog::new((0..3).map(|member| {
            ExpertResidencyUnit::new(
                ParameterBankKey::new(1, 0, member),
                owner.clone(),
                0,
                "decoder.layers.0.attention.values",
                ExpertResidencyDistribution::ExpertParallel,
                vec![ExpertParameterRecipe::new(
                    "weight",
                    "test.values.weight",
                    DerivedWeightRecipe::source(
                        "source.value",
                        TensorSelection::Range {
                            axis: 0,
                            start: member,
                            end: member + 1,
                        },
                    ),
                    ExpertParameterRole::Preserved,
                )
                .unwrap()],
            )
            .unwrap()
        }))
        .unwrap()
        .with_inferred_byte_geometry(&TestRecipeCatalog)
        .unwrap();
        let linear = RoutedBankRequirements::new(
            owner,
            linear_plan.into(),
            linear_catalog,
            BTreeMap::from([(0, 2)]),
        )
        .unwrap();
        let requirements = RoutedTextRequirements::new(
            text_requirements_with_linear(true),
            [
                (RoutedBankId::new(0), gated.clone()),
                (RoutedBankId::new(1), linear),
            ],
            &TestRecipeCatalog,
        )
        .unwrap();
        assert_eq!(requirements.banks().len(), 2);
        let value = requirements.bank(RoutedBankId::new(1)).unwrap();
        assert_eq!(value.plan().global_group_count(), 3);
        assert_eq!(value.routes_per_token(), 2);
        assert_eq!(value.catalog().units().len(), 3);
        assert_eq!(
            requirements
                .bank(RoutedBankId::new(0))
                .unwrap()
                .plan()
                .global_group_count(),
            2
        );
        assert_eq!(
            requirements.text().grouped_operations(),
            &[
                eredu_runtime::GroupedOperationRequirement::Linear,
                eredu_runtime::GroupedOperationRequirement::GatedProduct,
            ]
        );
        assert!(RoutedTextRequirements::new(
            text_requirements(),
            [
                (RoutedBankId::new(0), gated.clone()),
                (RoutedBankId::new(0), gated)
            ],
            &TestRecipeCatalog
        )
        .unwrap_err()
        .to_string()
        .contains("duplicate routed bank"));
        assert!(RoutedTextRequirements::new(text_requirements(), [], &TestRecipeCatalog).is_err());
    }

    #[test]
    fn routed_catalog_rejects_wrong_keys_missing_targets_and_unadmitted_sources() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let plan = plan(&owner);
        let wrong_key = validate_plan_catalog::<GatedProductOperation>(
            &owner,
            &plan,
            &catalog(&owner, 1, true, "source.gate"),
        )
        .unwrap_err();
        assert!(
            wrong_key.to_string().contains("key namespace"),
            "{wrong_key}"
        );

        let missing = validate_plan_catalog::<GatedProductOperation>(
            &owner,
            &plan,
            &catalog(&owner, 0, false, "source.gate"),
        )
        .unwrap_err();
        assert!(
            missing.to_string().contains("expected grouped targets"),
            "{missing}"
        );

        let wrong_source = validate_catalog_parameter_topology::<GatedProductOperation>(
            &text_requirements(),
            &plan,
            &catalog(&owner, 0, true, "source.unadmitted"),
            &TestRecipeCatalog,
        )
        .unwrap_err();
        assert!(
            wrong_source
                .to_string()
                .contains("differ from admitted sources"),
            "{wrong_source}"
        );

        let derived = DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                DerivedWeightRecipe::source("source.gate", TensorSelection::Full),
                DerivedWeightRecipe::source("source.gate.extra", TensorSelection::Full),
            ],
        };
        let derived_text = text_requirements()
            .with_derived_recipes(
                BTreeMap::from([("test.experts.gate_up_proj".into(), derived)]),
                BTreeMap::from([(
                    "test.experts.gate_up_proj".into(),
                    RecipeMetadata {
                        shape: vec![2, 16, 4],
                        dtype: RecipeDtype::F16,
                        byte_len: 256,
                    },
                )]),
            )
            .unwrap();
        let removed_source = validate_catalog_parameter_topology::<GatedProductOperation>(
            &derived_text,
            &plan,
            &catalog(&owner, 0, true, "source.gate"),
            &TestRecipeCatalog,
        )
        .expect_err("derived recipe source deletion was accepted");
        assert!(removed_source
            .to_string()
            .contains("differs from the exact admitted member recipe"));
    }

    #[test]
    fn routed_schedule_rejects_coordinated_omission_and_wrong_catalog_path() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let plan = plan(&owner);
        let catalog = catalog(&owner, 0, true, "source.gate");
        let omitted = BTreeMap::from([
            (("decoder".to_owned(), 0), "decoder.layers.0.mlp".to_owned()),
            (("decoder".to_owned(), 1), "decoder.layers.1.mlp".to_owned()),
        ]);
        let error = validate_expected_routed_schedule(&omitted, &plan, &catalog)
            .expect_err("coordinated plan/catalog omission was accepted");
        assert!(
            error.to_string().contains("architecture schedule"),
            "{error}"
        );

        let wrong_path = BTreeMap::from([(
            ("decoder".to_owned(), 0),
            "decoder.layers.0.wrong".to_owned(),
        )]);
        let error = validate_expected_routed_schedule(&wrong_path, &plan, &catalog)
            .expect_err("wrong catalog path was accepted");
        assert!(error.to_string().contains("architecture path"), "{error}");
    }

    #[test]
    fn routed_catalog_rejects_wrong_affine_companion_geometry() {
        let owner = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let (plan, catalog) = affine_plan_and_catalog(&owner);
        let error = validate_catalog_parameter_topology::<GatedProductOperation>(
            &affine_text_requirements(),
            &plan,
            &catalog,
            &WrongAffineCompanionCatalog,
        )
        .expect_err("wrong affine scale shape was accepted");
        assert!(error.to_string().contains("recipe shape"), "{error}");
    }
}

/// Selected-linear grouped equation with activation before reduction.
#[derive(Debug, Clone, Copy)]
pub struct LinearOperation;

impl RoutedGroupedOperationValidation for LinearOperation {
    type Spec = eredu_nn::GroupedLinearSpec;
    fn clone_spec_with_funding(spec:&Self::Spec,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error> {
        match funding {Some(funding)=>funding.clone_grouped_linear(spec),None=>Ok(spec.clone())}
    }

    fn workspace_kernel(spec: &Self::Spec) -> eredu_nn::workspace::WorkspaceExpertKernel<'_> {
        eredu_nn::workspace::WorkspaceExpertKernel::Linear(spec)
    }

    fn group_count(spec: &Self::Spec) -> i32 {
        spec.group_count()
    }
    fn output_dimensions(spec: &Self::Spec) -> i32 {
        spec.output_dimensions()
    }
    fn member_parameter_targets(
        spec: &Self::Spec,
        _: usize,
    ) -> Result<Vec<String>, eredu_nn::Error> {
        Ok(spec
            .projection()
            .parameters()
            .iter()
            .map(|p| p.id.to_string())
            .collect())
    }
    fn member_parameter_shapes(
        spec: &Self::Spec,
        _: usize,
    ) -> Result<BTreeMap<String, Vec<usize>>, eredu_nn::Error> {
        let projection = spec.projection();
        let shape = vec![
            1,
            spec.output_dimensions() as usize,
            spec.input_dimensions() as usize,
        ];
        let mut shapes = BTreeMap::from([(projection.weight().id.to_string(), shape.clone())]);
        if let Some(bias) = projection.bias() {
            shapes.insert(bias.id.to_string(), shape[..2].to_vec());
        }
        if let Some(companion) = grouped_companion_shape(projection.format(), &shape) {
            for parameter in [
                projection.format().scale(),
                projection.format().affine_bias(),
            ]
            .into_iter()
            .flatten()
            {
                shapes.insert(parameter.id.to_string(), companion.clone());
            }
        }
        Ok(shapes)
    }
    fn compact_spec(spec: &Self::Spec, groups: i32) -> Result<Self::Spec, eredu_nn::Error> {
        spec.clone().with_group_count(groups)
    }
    fn compact_spec_with_funding(spec:&Self::Spec,groups:i32,
        funding:Option<&eredu_nn::workspace::WorkspaceMetadataFunding>)->Result<Self::Spec,eredu_nn::Error> {
        match funding {Some(funding)=>funding.clone_grouped_linear(spec)?,None=>spec.clone()}
            .with_group_count(groups)
    }
}
impl<B: GroupedNeuralBackend> RoutedGroupedOperation<B> for LinearOperation {
    fn execute<Bank>(
        bank: &mut Bank,
        acquisition: &Bank::Acquisition,
        spec: &Self::Spec,
        input: &B::Tensor,
        routes: &GroupSelection<B::Tensor>,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
    {
        if observer.is_some() {
            return Err(eredu_nn::Error::backend_source(
                eredu_nn::GroupedUnitError::Unavailable,
            ));
        }
        let mut groups = bank
            .linear_groups(acquisition, spec, context)
            .map_err(eredu_nn::Error::backend_source)?;
        groups
            .forward_grouped(input, routes, context)
            .map_err(eredu_nn::Error::backend_source)
    }
}

/// Bounded selected-linear execution using the existing acquisition and completion lifecycle.
pub type PlannedAddressableLinear<B, Bank, Movement> =
    PlannedAddressableGrouped<LinearOperation, B, Bank, Movement>;

impl<B, Bank, Movement> RoutedExpertProvider<B> for PlannedAddressableLinear<B, Bank, Movement>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
{
    type Error = RoutedTextExecutionError;
    fn forward_linear_routed(
        &mut self,
        _: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.execute(request, context)
    }
    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected linear provider received a gated-product bank".into(),
        ))
    }
    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected linear provider received a ReLU-squared bank".into(),
        ))
    }
}
