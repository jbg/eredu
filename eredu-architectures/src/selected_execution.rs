//! Total backend-neutral selections retained between cold admission and materialization.

use std::collections::BTreeSet;

use eredu_core::{
    artifact::ArtifactAdmissionToken, CollectiveGroupId, PreparationAdmission, SessionCapabilities,
};
use eredu_runtime::{
    CommunicationManifest, ParameterBankResidency, PipelineActivationDtype,
    ReplicatedTextRequirements, SelectedReplicatedTextRealization, SelectedSpeculativeRealization,
};

use crate::{
    configuration::PredictionExtensionPlan,
    partitioned_execution::SelectedPartitionedAdmission,
    replicated_text::{
        CompositeTextRequirements, SelectedCompositeTextRealization,
        SelectedReplicatedTextExecution, SelectedReplicatedTextExecutionDispatcher,
    },
    RoutedTextRequirements, SelectedRoutedTextRealization,
};

/// Selected direct partitioned execution, including admitted communication and ownership.
pub type SelectedDensePartitionedExecution =
    SelectedPartitionedAdmission<SelectedReplicatedTextRealization, ReplicatedTextRequirements>;

/// Selected routed partitioned execution, including admitted communication and ownership.
pub type SelectedRoutedPartitionedExecution =
    SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>;

/// Selected composite partitioned execution, including admitted communication and ownership.
pub type SelectedCompositePartitionedExecution =
    SelectedPartitionedAdmission<SelectedCompositeTextRealization, CompositeTextRequirements>;

type SelectedOrdinaryExecution = SelectedReplicatedTextExecution<
    SelectedReplicatedTextRealization,
    SelectedRoutedTextRealization,
    SelectedCompositeTextRealization,
>;

#[derive(Debug, Clone)]
enum SelectedExecutionKind {
    Replicated(SelectedReplicatedTextRealization),
    Routed(SelectedRoutedTextRealization),
    Composite(SelectedCompositeTextRealization),
    PartitionedDense(SelectedDensePartitionedExecution),
    PartitionedRouted(SelectedRoutedPartitionedExecution),
    PartitionedComposite(SelectedCompositePartitionedExecution),
}

/// An owned adapter for exactly one selected execution branch.
///
/// A backend implements this trait on a concrete materializer. Dispatch is
/// monomorphized and consumes both the authoritative selection and the
/// materializer, so no branch reconstruction or dynamic dispatch is needed.
pub trait SelectedExecutionDispatcher: Sized {
    /// Completed materialization output.
    type Output;
    /// Materialization failure.
    type Error;

    /// Materializes ordinary replicated text execution.
    fn replicated(
        self,
        selected: SelectedReplicatedTextRealization,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes ordinary routed text execution.
    fn routed(self, selected: SelectedRoutedTextRealization) -> Result<Self::Output, Self::Error>;

    /// Materializes ordinary composite text execution.
    fn composite(
        self,
        selected: SelectedCompositeTextRealization,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes direct partitioned text execution.
    fn partitioned_dense(
        self,
        selected: SelectedDensePartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes routed partitioned text execution.
    fn partitioned_routed(
        self,
        selected: SelectedRoutedPartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes composite partitioned text execution.
    fn partitioned_composite(
        self,
        selected: SelectedCompositePartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;
}

/// One authoritative backend-neutral execution selection.
///
/// The semantic branch is private. Backends can only consume it through the
/// typed dispatcher, preventing a materializer from substituting or rebuilding
/// the selection after admission.
#[derive(Debug, Clone)]
pub struct SelectedExecution {
    kind: Box<SelectedExecutionKind>,
}

impl SelectedExecution {
    pub(crate) fn partition_groups(
        &self,
    ) -> Option<&[crate::partitioned_execution::PartitionedGroupRequirements]> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                Some(selected.requirements().groups())
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                Some(selected.requirements().groups())
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.requirements().groups())
            }
            _ => None,
        }
    }
    /// Exact portable rank topology of a selected partition, when present.
    pub fn parallel_topology(&self) -> Option<eredu_core::ParallelRankTopology> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                Some(selected.requirements().topology())
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                Some(selected.requirements().topology())
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.requirements().topology())
            }
            _ => None,
        }
    }

    /// Exact selected processor policy for composite execution.
    pub fn processor(&self) -> Option<&eredu_runtime::SelectedProcessorExecution> {
        match self.kind.as_ref() {
            SelectedExecutionKind::Composite(selected) => Some(selected.processor()),
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.base().processor())
            }
            _ => None,
        }
    }

    pub(crate) fn replicated(selected: SelectedReplicatedTextRealization) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::Replicated(selected)),
        }
    }

    pub(crate) fn routed(selected: SelectedRoutedTextRealization) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::Routed(selected)),
        }
    }

    pub(crate) fn composite(selected: SelectedCompositeTextRealization) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::Composite(selected)),
        }
    }

    pub(crate) fn partitioned_dense(selected: SelectedDensePartitionedExecution) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::PartitionedDense(selected)),
        }
    }

    pub(crate) fn partitioned_routed(selected: SelectedRoutedPartitionedExecution) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::PartitionedRouted(selected)),
        }
    }

    pub(crate) fn partitioned_composite(selected: SelectedCompositePartitionedExecution) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::PartitionedComposite(selected)),
        }
    }

    pub(crate) fn ordinary(selected: SelectedOrdinaryExecution) -> Self {
        struct IntoTotal;

        impl
            SelectedReplicatedTextExecutionDispatcher<
                SelectedReplicatedTextRealization,
                SelectedRoutedTextRealization,
                SelectedCompositeTextRealization,
            > for IntoTotal
        {
            type Output = SelectedExecution;
            type Error = std::convert::Infallible;

            fn replicated(
                self,
                selected: SelectedReplicatedTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(SelectedExecution::replicated(selected))
            }

            fn routed(
                self,
                selected: SelectedRoutedTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(SelectedExecution::routed(selected))
            }

            fn composite(
                self,
                selected: SelectedCompositeTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(SelectedExecution::composite(selected))
            }
        }

        selected
            .dispatch(IntoTotal)
            .expect("ordinary execution conversion is infallible")
    }

    /// Invokes exactly one backend materializer with the owned selected branch.
    pub fn dispatch<D>(self, dispatcher: D) -> Result<D::Output, D::Error>
    where
        D: SelectedExecutionDispatcher,
    {
        match *self.kind {
            SelectedExecutionKind::Replicated(selected) => dispatcher.replicated(selected),
            SelectedExecutionKind::Routed(selected) => dispatcher.routed(selected),
            SelectedExecutionKind::Composite(selected) => dispatcher.composite(selected),
            SelectedExecutionKind::PartitionedDense(selected) => {
                dispatcher.partitioned_dense(selected)
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                dispatcher.partitioned_routed(selected)
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                dispatcher.partitioned_composite(selected)
            }
        }
    }

    /// Returns the shared text-session realization selected for every execution class.
    pub fn text_realization(&self) -> &SelectedReplicatedTextRealization {
        match self.kind.as_ref() {
            SelectedExecutionKind::Replicated(selected) => selected,
            SelectedExecutionKind::Routed(selected) => selected.text(),
            SelectedExecutionKind::Composite(selected) => selected.execution(),
            SelectedExecutionKind::PartitionedDense(selected) => selected.base(),
            SelectedExecutionKind::PartitionedRouted(selected) => selected.base().text(),
            SelectedExecutionKind::PartitionedComposite(selected) => selected.base().execution(),
        }
    }

    /// Complete unsharded executable parameter accounting from retained tasks.
    /// The artifact resource profile describes the model before rank replication;
    /// native session reports separately account for the selected local placement.
    pub fn parameter_resources(
        &self,
    ) -> Result<eredu_runtime::SelectedParameterResources, eredu_runtime::BoundedResidencySizingError>
    {
        let (bank_targets, excluded) = self.bank_parameter_targets();
        let text = self.text_realization();
        let tasks = text
            .materialization_tasks()
            .iter()
            .chain(text.auxiliary_materialization_tasks())
            .cloned()
            .collect::<Vec<_>>();
        eredu_runtime::selected_parameter_resources(&tasks, &bank_targets, &excluded)
    }

    fn bank_parameter_targets(&self) -> (BTreeSet<String>, BTreeSet<String>) {
        let routed = self.routed_realization();
        let bank_targets = routed
            .into_iter()
            .flat_map(|selected| selected.banks().values())
            .flat_map(|bank| bank.addressable_members())
            .flat_map(|member| member.parameters())
            .map(|parameter| parameter.task().name().to_owned())
            .collect::<BTreeSet<_>>();
        let excluded = if routed.is_some_and(|selected| {
            matches!(
                selected.bank_residency(),
                ParameterBankResidency::IndependentCache(_)
            )
        }) {
            bank_targets.clone()
        } else {
            BTreeSet::new()
        };
        (bank_targets, excluded)
    }

    fn routed_realization(&self) -> Option<&SelectedRoutedTextRealization> {
        match self.kind.as_ref() {
            SelectedExecutionKind::Routed(selected) => Some(selected),
            SelectedExecutionKind::PartitionedRouted(selected) => Some(selected.base()),
            SelectedExecutionKind::Composite(SelectedCompositeTextRealization::Routed {
                execution,
                ..
            }) => Some(execution),
            SelectedExecutionKind::PartitionedComposite(selected) => match selected.base() {
                SelectedCompositeTextRealization::Routed { execution, .. } => Some(execution),
                SelectedCompositeTextRealization::Direct(_) => None,
            },
            _ => None,
        }
    }

    /// Source-recipe and compact-bank workspace of the retained rank selection.
    pub fn parameter_materialization_workspace(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        let (_, independent) = self.bank_parameter_targets();
        let tasks = match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => selected.materialization_tasks(),
            SelectedExecutionKind::PartitionedRouted(selected) => selected.materialization_tasks(),
            SelectedExecutionKind::PartitionedComposite(selected) => {
                selected.materialization_tasks()
            }
            _ => self.text_realization().materialization_tasks(),
        };
        let mut result = eredu_core::ParameterMaterializationWorkspace {
            ordinary_recipe_peak_bytes: 0,
            expert_member_recipe_peak_bytes: 0,
            compact_bank_bytes: 0,
            ordinary_native_peak_bytes: 0,
            expert_member_native_peak_bytes: 0,
            conversion_workspace_bytes: 0,
            ordinary_materializations: match self.text_realization().residency() {
                eredu_runtime::LayerWeightResidency::DenseDiskStream(options) => options
                    .background_queue_capacity()
                    .checked_add(1)
                    .ok_or("materialization concurrency overflow")?,
                _ => 1,
            },
        };
        let mut conversion_outputs = 0u64;
        let mut conversion_source_peak = 0u64;
        let mut ordinary_groups = Vec::<(eredu_runtime::ReplicatedTextParameterOwner, u64)>::new();
        let mut record_group = |task: &eredu_runtime::ReplicatedTextMaterializationTask,
                                bytes: u64|
         -> Result<(), String> {
            if let Some((_, total)) = ordinary_groups
                .iter_mut()
                .find(|(owner, _)| owner == task.owner())
            {
                *total = total
                    .checked_add(bytes)
                    .ok_or("native group workspace overflow")?;
            } else {
                ordinary_groups.push((task.owner().clone(), bytes));
            }
            Ok(())
        };
        let target =
            |task: &eredu_runtime::ReplicatedTextMaterializationTask| -> Result<String, String> {
                std::iter::once(task.name())
                    .chain(task.aliases().iter().map(String::as_str))
                    .find(|name| layout.is_none_or(|layout| layout.contains(name)))
                    .map(str::to_owned)
                    .ok_or_else(|| format!("missing local target {}", task.name()))
            };
        for task in tasks
            .iter()
            .filter(|task| !independent.contains(task.name()))
        {
            let recipe = eredu_runtime::placed_source_recipe(
                &task.source_recipe().map_err(|error| error.to_string())?,
                &target(task)?,
                source,
                layout,
                false,
            )?;
            let native_peak = mechanisms.recipe_materialization_workspace(&recipe, source)?;
            record_group(task, native_peak)?;
            if matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            ) {
                let output = recipe.infer(source).map_err(|error| error.to_string())?;
                conversion_outputs = conversion_outputs
                    .checked_add(
                        eredu_runtime::selected_addressable_parameter_bytes(task, &output)
                            .map_err(|error| error.to_string())?,
                    )
                    .ok_or("conversion output size overflow")?;
                conversion_source_peak = conversion_source_peak.max(native_peak);
            }
            result.ordinary_recipe_peak_bytes =
                result
                    .ordinary_recipe_peak_bytes
                    .max(eredu_runtime::placed_recipe_peak_bytes(
                        &task.source_recipe().map_err(|e| e.to_string())?,
                        &target(task)?,
                        source,
                        layout,
                        false,
                    )?);
            for companion in task.output_companions() {
                if let Some(task) = companion.materialization_task() {
                    let recipe = eredu_runtime::placed_source_recipe(
                        &task.source_recipe().map_err(|error| error.to_string())?,
                        &target(task)?,
                        source,
                        layout,
                        false,
                    )?;
                    record_group(
                        task,
                        mechanisms.recipe_materialization_workspace(&recipe, source)?,
                    )?;
                    result.ordinary_recipe_peak_bytes = result.ordinary_recipe_peak_bytes.max(
                        eredu_runtime::placed_recipe_peak_bytes(
                            &task.source_recipe().map_err(|e| e.to_string())?,
                            &target(task)?,
                            source,
                            layout,
                            false,
                        )?,
                    );
                }
            }
        }
        result.ordinary_native_peak_bytes = ordinary_groups
            .iter()
            .map(|(_, bytes)| *bytes)
            .max()
            .unwrap_or(0);
        if let Some(routed) = self.routed_realization() {
            if let ParameterBankResidency::IndependentCache(options) = routed.bank_residency() {
                for bank in routed.banks().values() {
                    let mut units = std::collections::BTreeMap::<usize, u64>::new();
                    for member in bank.addressable_members() {
                        if self.partition_groups().is_some_and(|groups| {
                            !groups.iter().any(|group| {
                                group.group() == bank.owner_group()
                                    && group.units().contains(&member.key().unit())
                            })
                        }) {
                            continue;
                        }
                        let mut retained = 0u64;
                        let mut bytes = 0u64;
                        let mut native_retained = 0u64;
                        for parameter in member.parameters() {
                            let task = parameter.task();
                            let name = target(task)?;
                            if let Some(tensor) = layout.and_then(|layout| layout.tensor(&name)) {
                                if tensor.additional_placements().iter().chain(std::iter::once(tensor.placement())).any(|p| matches!(p, eredu_runtime::TensorPlacement::Range { axis: 0, start, end } if !( *start..*end).contains(&member.key().member()))) { continue; }
                            }
                            let recipe = eredu_runtime::placed_source_recipe(
                                parameter.recipe(),
                                &name,
                                source,
                                layout,
                                true,
                            )?;
                            let peak = recipe
                                .peak_materialization_bytes(source)
                                .map_err(|e| e.to_string())?;
                            native_retained = native_retained
                                .checked_add(
                                    mechanisms.recipe_materialization_workspace(&recipe, source)?,
                                )
                                .ok_or("expert native workspace overflow")?;
                            result.expert_member_recipe_peak_bytes =
                                result.expert_member_recipe_peak_bytes.max(
                                    retained
                                        .checked_add(peak)
                                        .ok_or("expert recipe workspace overflow")?,
                                );
                            let output = recipe.infer(source).map_err(|e| e.to_string())?;
                            if matches!(
                                task.lowering(),
                                eredu_runtime::WeightLoweringKind::Transform
                                    | eredu_runtime::WeightLoweringKind::DerivedTransform
                            ) {
                                native_retained = native_retained
                                    .checked_add(
                                        eredu_runtime::selected_addressable_parameter_bytes(
                                            task, &output,
                                        )
                                        .map_err(|error| error.to_string())?,
                                    )
                                    .ok_or("expert conversion workspace overflow")?;
                            }
                            retained = retained
                                .checked_add(output.byte_len())
                                .ok_or("expert recipe output overflow")?;
                            bytes = bytes
                                .checked_add(
                                    eredu_runtime::selected_addressable_parameter_bytes(
                                        task, &output,
                                    )
                                    .map_err(|e| e.to_string())?,
                                )
                                .ok_or("expert selected bytes overflow")?;
                        }
                        result.expert_member_native_peak_bytes =
                            result.expert_member_native_peak_bytes.max(native_retained);
                        let unit = units.entry(member.key().unit()).or_default();
                        *unit = unit.checked_add(bytes).ok_or("bank unit size overflow")?;
                    }
                    let compact = units
                        .values()
                        .copied()
                        .max()
                        .unwrap_or(0)
                        .min(options.compact_bank_scratch_bytes());
                    result.compact_bank_bytes = result
                        .compact_bank_bytes
                        .checked_add(compact)
                        .ok_or("compact bank workspace overflow")?;
                }
            }
        }
        if conversion_outputs > 0 {
            result.conversion_workspace_bytes = conversion_outputs
                .checked_add(conversion_source_peak)
                .ok_or("conversion workspace overflow")?;
        }
        Ok(result)
    }

    /// Exact cold TP/PP/EP output sizes from the family's physical topology.
    pub fn partition_parameter_resources(
        &self,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
    ) -> Result<
        Option<(
            eredu_runtime::SelectedParameterResources,
            eredu_runtime::LocalModelLayout,
        )>,
        String,
    > {
        let Some(rank) = self.parallel_topology() else {
            return Ok(None);
        };
        let layout =
            crate::partitioned_execution::derive_partitioned_local_layout(parameters, rank)?;
        let (routed, independent) = self.bank_parameter_targets();
        fn size<R, Q>(
            selected: &SelectedPartitionedAdmission<R, Q>,
            parameters: &eredu_runtime::ArchitectureParameterDescription,
            layout: &eredu_runtime::LocalModelLayout,
            routed: &BTreeSet<String>,
            independent: &BTreeSet<String>,
        ) -> Result<eredu_runtime::SelectedParameterResources, String> {
            let requirements = selected.requirements();
            let owned = parameters
                .groups()
                .iter()
                .filter(|group| {
                    group
                        .owner()
                        .is_owned_by(requirements.ownership(), |group, unit| {
                            requirements.groups().iter().any(|owned| {
                                owned.group() == group && owned.units().contains(&unit)
                            })
                        })
                })
                .flat_map(|group| group.members())
                .map(|member| member.target().to_owned())
                .collect();
            eredu_runtime::selected_parameter_resources_for_layout(
                selected.materialization_tasks(),
                layout,
                &owned,
                routed,
                independent,
            )
            .map_err(|e| e.to_string())
        }
        let resources = match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                size(selected, parameters, &layout, &routed, &independent)?
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                size(selected, parameters, &layout, &routed, &independent)?
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                size(selected, parameters, &layout, &routed, &independent)?
            }
            _ => unreachable!("partition topology requires a partitioned selection"),
        };
        Ok(Some((resources, layout)))
    }

    /// Returns parameter targets moved into independently resident routed banks.
    ///
    /// Partitioned materialization owns exact rank-local tasks, so it does not
    /// use the ordinary whole-model exclusion projection.
    pub fn bounded_residency_exclusions(&self) -> BTreeSet<String> {
        let SelectedExecutionKind::Routed(selected) = self.kind.as_ref() else {
            return BTreeSet::new();
        };
        if !matches!(
            selected.bank_residency(),
            ParameterBankResidency::IndependentCache(_)
        ) {
            return BTreeSet::new();
        }
        selected
            .banks()
            .values()
            .flat_map(|bank| bank.addressable_members())
            .flat_map(|member| member.parameters())
            .map(|parameter| parameter.task().name().to_owned())
            .collect()
    }

    /// Returns the selected text realization and ordinary materialization exclusions.
    pub fn selected_bounded_residency(
        &self,
    ) -> (SelectedReplicatedTextRealization, BTreeSet<String>) {
        (
            self.text_realization().clone(),
            self.bounded_residency_exclusions(),
        )
    }

    /// Returns the admitted communication manifest for partitioned execution.
    pub fn communication_manifest(&self) -> Option<&CommunicationManifest> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                Some(selected.requirements().communication())
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                Some(selected.requirements().communication())
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.requirements().communication())
            }
            SelectedExecutionKind::Replicated(_)
            | SelectedExecutionKind::Routed(_)
            | SelectedExecutionKind::Composite(_) => None,
        }
    }

    /// Returns the admitted pipeline activation dtype for partitioned execution.
    pub fn partitioned_activation_dtype(&self) -> Option<PipelineActivationDtype> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                Some(selected.requirements().activation_dtype())
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                Some(selected.requirements().activation_dtype())
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.requirements().activation_dtype())
            }
            SelectedExecutionKind::Replicated(_)
            | SelectedExecutionKind::Routed(_)
            | SelectedExecutionKind::Composite(_) => None,
        }
    }

    /// Returns the selected session-wide publication group for partitioned execution.
    pub fn partitioned_session_group(&self) -> Option<CollectiveGroupId> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                selected.requirements().session_group()
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                selected.requirements().session_group()
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                selected.requirements().session_group()
            }
            SelectedExecutionKind::Replicated(_)
            | SelectedExecutionKind::Routed(_)
            | SelectedExecutionKind::Composite(_) => None,
        }
    }

    /// Reports whether the selected execution consumes a composite media projector.
    pub fn allows_media_projector(&self) -> bool {
        matches!(
            self.kind.as_ref(),
            SelectedExecutionKind::Composite(_) | SelectedExecutionKind::PartitionedComposite(_)
        )
    }
}

/// Complete neutral preparation selection retained until backend materialization.
///
/// This value deliberately contains no device, stream, rank binding, native
/// completion object, or backend-private token.
#[derive(Debug, Clone)]
pub struct SelectedPreparation {
    admission_token: ArtifactAdmissionToken,
    execution: SelectedExecution,
    admission: PreparationAdmission,
    prediction_extension: Option<PredictionExtensionPlan>,
    prediction_realization: Option<SelectedSpeculativeRealization>,
}

impl SelectedPreparation {
    pub(crate) const fn new(
        admission_token: ArtifactAdmissionToken,
        execution: SelectedExecution,
        admission: PreparationAdmission,
        prediction_extension: Option<PredictionExtensionPlan>,
        prediction_realization: Option<SelectedSpeculativeRealization>,
    ) -> Self {
        Self {
            admission_token,
            execution,
            admission,
            prediction_extension,
            prediction_realization,
        }
    }

    /// Opaque inspection origin retained by total cold selection.
    pub fn admission_token(&self) -> ArtifactAdmissionToken {
        self.admission_token.clone()
    }

    /// Returns the total execution selected before native work.
    pub const fn execution(&self) -> &SelectedExecution {
        &self.execution
    }

    /// Returns the retained portable admission proof.
    pub const fn admission(&self) -> PreparationAdmission {
        self.admission
    }

    /// Returns session capabilities admitted for the exact selected preparation.
    pub const fn session_capabilities(&self) -> SessionCapabilities {
        self.admission.session_capabilities()
    }

    /// Returns the selected embedded prediction-extension architecture, when present.
    pub const fn prediction_extension(&self) -> Option<&PredictionExtensionPlan> {
        self.prediction_extension.as_ref()
    }

    /// Returns the selected speculative realization for the embedded extension, when present.
    pub const fn prediction_realization(&self) -> Option<&SelectedSpeculativeRealization> {
        self.prediction_realization.as_ref()
    }

    /// Returns the shared text-session realization selected for every execution class.
    pub fn text_realization(&self) -> &SelectedReplicatedTextRealization {
        self.execution.text_realization()
    }

    /// Returns the selected text realization and ordinary materialization exclusions.
    pub fn selected_bounded_residency(
        &self,
    ) -> (SelectedReplicatedTextRealization, BTreeSet<String>) {
        self.execution.selected_bounded_residency()
    }

    /// Returns the admitted communication manifest for partitioned execution.
    pub fn communication_manifest(&self) -> Option<&CommunicationManifest> {
        self.execution.communication_manifest()
    }

    /// Returns the admitted pipeline activation dtype for partitioned execution.
    pub fn partitioned_activation_dtype(&self) -> Option<PipelineActivationDtype> {
        self.execution.partitioned_activation_dtype()
    }

    /// Returns the selected session-wide publication group for partitioned execution.
    pub fn partitioned_session_group(&self) -> Option<CollectiveGroupId> {
        self.execution.partitioned_session_group()
    }

    /// Reports whether the selected execution consumes a composite media projector.
    pub fn allows_media_projector(&self) -> bool {
        self.execution.allows_media_projector()
    }

    /// Consumes the selection into neutral materialization inputs.
    pub(crate) fn into_parts(
        self,
    ) -> (
        SelectedExecution,
        PreparationAdmission,
        Option<PredictionExtensionPlan>,
        Option<SelectedSpeculativeRealization>,
    ) {
        (
            self.execution,
            self.admission,
            self.prediction_extension,
            self.prediction_realization,
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn public_selection_types_are_owned_and_thread_safe() {
        fn assert_owned<T: Clone + Send + Sync + 'static>() {}

        assert_owned::<super::SelectedExecution>();
        assert_owned::<super::SelectedPreparation>();
    }
}
