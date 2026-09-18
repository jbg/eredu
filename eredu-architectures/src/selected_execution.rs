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

#[derive(Debug, Clone, PartialEq)]
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

/// A borrowed adapter for exactly the retained selected execution branch.
///
/// Cold inspection can read the same branch that owned construction consumes,
/// without cloning selections or reconstructing execution-class policy. The
/// dispatcher may return references with the selection's lifetime. This gives
/// descriptive access only: no materialization, completion or admission authority.
/// Arbitrary adapter implementations are not thereby allocation-free.
pub trait SelectedExecutionBorrowedDispatcher<'a>: Sized {
    /// Inspection result, which may borrow the retained selection.
    type Output;
    /// The adapter's unchanged inspection failure.
    type Error;

    /// Inspects ordinary replicated execution.
    fn replicated(
        self,
        selected: &'a SelectedReplicatedTextRealization,
    ) -> Result<Self::Output, Self::Error>;

    /// Inspects ordinary routed execution.
    fn routed(
        self,
        selected: &'a SelectedRoutedTextRealization,
    ) -> Result<Self::Output, Self::Error>;

    /// Inspects ordinary composite execution.
    fn composite(
        self,
        selected: &'a SelectedCompositeTextRealization,
    ) -> Result<Self::Output, Self::Error>;

    /// Inspects partitioned dense execution.
    fn partitioned_dense(
        self,
        selected: &'a SelectedDensePartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;

    /// Inspects partitioned routed execution.
    fn partitioned_routed(
        self,
        selected: &'a SelectedRoutedPartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;

    /// Inspects partitioned composite execution.
    fn partitioned_composite(
        self,
        selected: &'a SelectedCompositePartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;
}

/// One authoritative backend-neutral execution selection.
///
/// The semantic branch is private. Typed dispatchers lend it for descriptive
/// inspection or consume it for construction, preventing a materializer from
/// substituting or rebuilding the selection after admission.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedExecution {
    kind: Box<SelectedExecutionKind>,
}

impl SelectedExecution {
    /// Resolves effective parameter coordinates on one rank of this retained
    /// execution. This is bounded descriptive geometry, not loaded authority.
    pub fn parameter_partition_layout_for_rank(
        &self,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        parameter: &str,
        global_rank: usize,
        reservation: &mut impl eredu_core::capture::CaptureReservation,
    ) -> Result<
        Option<crate::parameter_partition::ParameterPartitionLayout>,
        eredu_core::parameters::ParameterError,
    > {
        self.parameter_partition_layout_with_index(
            parameters,
            parameter,
            global_rank,
            None,
            reservation,
        )
    }

    pub(crate) fn parameter_partition_layout_with_index(
        &self,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        parameter: &str,
        global_rank: usize,
        index: Option<&crate::parameter_partition::ParameterMemberIndex>,
        reservation: &mut impl eredu_core::capture::CaptureReservation,
    ) -> Result<
        Option<crate::parameter_partition::ParameterPartitionLayout>,
        eredu_core::parameters::ParameterError,
    > {
        use crate::partitioned_execution::selected_parameter_layout_for_rank;
        let tasks = self.text_realization().materialization_tasks();
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                selected_parameter_layout_for_rank(
                    selected.requirements(),
                    tasks,
                    parameters,
                    parameter,
                    global_rank,
                    index,
                    reservation,
                )
                .map(Some)
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                selected_parameter_layout_for_rank(
                    selected.requirements(),
                    tasks,
                    parameters,
                    parameter,
                    global_rank,
                    index,
                    reservation,
                )
                .map(Some)
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                selected_parameter_layout_for_rank(
                    selected.requirements(),
                    tasks,
                    parameters,
                    parameter,
                    global_rank,
                    index,
                    reservation,
                )
                .map(Some)
            }
            _ => Ok(None),
        }
    }

    /// Describes global scalar component coordinates for this retained partition.
    /// A nonpartitioned execution returns `None`. This does not enable capture or
    /// interventions: those require their own admitted budgets and delivery policy.
    pub fn component_partition_layout(
        &self,
        descriptor: &eredu_core::ArchitectureDescriptor,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
    ) -> Result<
        Option<crate::component_partition::ComponentPartitionLayout>,
        crate::component_partition::ComponentPartitionError,
    > {
        let Some(topology) = self.parallel_topology() else {
            return Ok(None);
        };
        self.component_partition_layout_for_rank(descriptor, parameters, topology.global_rank())
    }

    /// Describes any rank in this same retained execution selection. Ownership is
    /// compiled by the architecture's admission driver, not inferred from rank
    /// adjacency, checkpoint aliases or a backend's local native tensor shapes.
    /// The supplied descriptor/parameters must be those retained with this model.
    pub fn component_partition_layout_for_rank(&self,descriptor:&eredu_core::ArchitectureDescriptor,parameters:&eredu_runtime::ArchitectureParameterDescription,global_rank:usize)->Result<Option<crate::component_partition::ComponentPartitionLayout>,crate::component_partition::ComponentPartitionError>{self.component_partition_layout_for_rank_worker(descriptor,parameters,global_rank,crate::component_partition::construction::Destination(None))}
    /// Compiles all retained rank declarations through the shared constructor.
    pub fn component_partition_layouts(&self,descriptor:&eredu_core::ArchitectureDescriptor,parameters:&eredu_runtime::ArchitectureParameterDescription,max_ranks:usize)->Result<Option<crate::component_partition::ComponentPartitionLayouts>,crate::component_partition::ComponentPartitionError>{self.component_partition_layouts_worker(descriptor,parameters,max_ranks,crate::component_partition::construction::Destination(None))}
    pub(crate) fn component_partition_layout_for_rank_worker(
        &self,
        descriptor: &eredu_core::ArchitectureDescriptor,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        global_rank: usize,
        allocation:crate::component_partition::construction::Destination<'_>,
    ) -> Result<
        Option<crate::component_partition::ComponentPartitionLayout>,
        crate::component_partition::ComponentPartitionError,
    > {
        use crate::partitioned_execution::selected_component_layout_for_rank_worker;
        allocation.controls::<(&Self,&eredu_core::ArchitectureDescriptor,&eredu_runtime::ArchitectureParameterDescription,usize)>()?;
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                selected_component_layout_for_rank_worker(
                    selected.requirements(),
                    descriptor,
                    parameters,
                    global_rank,
                    self.routed_realization(),allocation,
                )
                .map(Some)
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                selected_component_layout_for_rank_worker(
                    selected.requirements(),
                    descriptor,
                    parameters,
                    global_rank,
                    self.routed_realization(),allocation,
                )
                .map(Some)
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                selected_component_layout_for_rank_worker(
                    selected.requirements(),
                    descriptor,
                    parameters,
                    global_rank,
                    self.routed_realization(),allocation,
                )
                .map(Some)
            }
            _ => Ok(None),
        }
    }

    /// Compiles one reusable set of component layouts for the whole retained
    /// topology. The rank bound is checked before allocating or deriving layouts.
    /// Reuse the result across selections and forwards; it contains no native state.
    pub(crate) fn component_partition_layouts_worker(
        &self,
        descriptor: &eredu_core::ArchitectureDescriptor,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        max_ranks: usize,
        allocation:crate::component_partition::construction::Destination<'_>,
    ) -> Result<
        Option<crate::component_partition::ComponentPartitionLayouts>,
        crate::component_partition::ComponentPartitionError,
    > {
        allocation.controls::<(&Self,&eredu_core::ArchitectureDescriptor,&eredu_runtime::ArchitectureParameterDescription,usize,Vec<crate::component_partition::ComponentPartitionLayout>)>()?;
        let Some(topology) = self.parallel_topology() else {
            return Ok(None);
        };
        if topology.world_size() > max_ranks {
            return Err(allocation.capture_invalid(format_args!("component rank count exceeds its bound")));
        }
        let mut layouts=allocation.vector(topology.world_size())?;
        for rank in 0..topology.world_size(){layouts.push(self.component_partition_layout_for_rank_worker(descriptor,parameters,rank,allocation)?.expect("retained partition selection"));}
        crate::component_partition::ComponentPartitionLayouts::new_worker(topology.topology(),layouts,allocation)
            .map(Some)
    }

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

    /// Invokes exactly one adapter with a borrow of the authoritative branch.
    /// The selection remains retained and unchanged for subsequent construction.
    pub fn dispatch_ref<'a, D>(&'a self, dispatcher: D) -> Result<D::Output, D::Error>
    where
        D: SelectedExecutionBorrowedDispatcher<'a>,
    {
        match self.kind.as_ref() {
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
        struct TextRealization;
        impl<'a> SelectedExecutionBorrowedDispatcher<'a> for TextRealization {
            type Output = &'a SelectedReplicatedTextRealization;
            type Error = std::convert::Infallible;
            fn replicated(
                self,
                selected: &'a SelectedReplicatedTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(selected)
            }
            fn routed(
                self,
                selected: &'a SelectedRoutedTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(selected.text())
            }
            fn composite(
                self,
                selected: &'a SelectedCompositeTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(selected.execution())
            }
            fn partitioned_dense(
                self,
                selected: &'a SelectedDensePartitionedExecution,
            ) -> Result<Self::Output, Self::Error> {
                Ok(selected.base())
            }
            fn partitioned_routed(
                self,
                selected: &'a SelectedRoutedPartitionedExecution,
            ) -> Result<Self::Output, Self::Error> {
                Ok(selected.base().text())
            }
            fn partitioned_composite(
                self,
                selected: &'a SelectedCompositePartitionedExecution,
            ) -> Result<Self::Output, Self::Error> {
                Ok(selected.base().execution())
            }
        }
        self.dispatch_ref(TextRealization)
            .unwrap_or_else(|never| match never {})
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

    /// Whether this retained selection uses independent parameter acquisition.
    /// This is descriptive load policy, not a native source or operation grant.
    pub fn has_independent_parameter_banks(&self) -> bool {
        self.routed_realization().is_some_and(|selected|
            matches!(selected.bank_residency(), ParameterBankResidency::IndependentCache(_)))
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

    /// Exact cold TP/PP/EP output sizes from the family's complete physical topology.
    /// Prediction parameters use tensor placement and remain replicated over the
    /// target's pipeline and expert axes, independently of target storage ownership.
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
                        .is_stored_by(requirements.ownership(), |group, unit| {
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
        let mut resources = match self.kind.as_ref() {
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
        let auxiliary = self.text_realization().auxiliary_materialization_tasks();
        if !auxiliary.is_empty() {
            let prediction_rank = crate::prediction_extension::tensor_rank(rank)
                .map_err(|error| error.to_string())?;
            let prediction_layout = crate::partitioned_execution::derive_partitioned_local_layout(
                parameters,
                prediction_rank,
            )?;
            let owned = parameters
                .groups()
                .iter()
                .flat_map(|group| group.members())
                .map(|member| member.target().to_owned())
                .collect();
            let auxiliary = eredu_runtime::selected_parameter_resources_for_layout(
                auxiliary,
                &prediction_layout,
                &owned,
                &routed,
                &BTreeSet::new(),
            )
            .map_err(|error| error.to_string())?;
            resources.parameter_bytes = resources
                .parameter_bytes
                .checked_add(auxiliary.parameter_bytes)
                .ok_or("rank parameter size overflow")?;
            resources.expert_bytes = resources
                .expert_bytes
                .checked_add(auxiliary.expert_bytes)
                .ok_or("rank expert size overflow")?;
            resources.pinned_bytes = resources
                .pinned_bytes
                .checked_add(auxiliary.pinned_bytes)
                .ok_or("rank pinned size overflow")?;
            resources.largest_unit_bytes = resources
                .largest_unit_bytes
                .max(auxiliary.largest_unit_bytes);
            resources.largest_adjacent_units_bytes = resources
                .largest_adjacent_units_bytes
                .max(auxiliary.largest_adjacent_units_bytes);
        }
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

    /// Exact tensor group retained by the architecture's partition selection.
    /// This is descriptive identity; native group ownership and operation
    /// qualification remain the backend consumer's responsibility.
    pub fn partitioned_tensor_group(&self) -> Option<CollectiveGroupId> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => selected.requirements().tensor_group(),
            SelectedExecutionKind::PartitionedRouted(selected) => selected.requirements().tensor_group(),
            SelectedExecutionKind::PartitionedComposite(selected) => selected.requirements().tensor_group(),
            SelectedExecutionKind::Replicated(_) | SelectedExecutionKind::Routed(_)
            | SelectedExecutionKind::Composite(_) => None,
        }
    }

    /// Exact architecture-selected output owner and publication group.
    pub fn partitioned_output_publication(&self)->Option<eredu_runtime::PartitionOutputPublication> {
        let (group,owner_rank)=match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected)=>(selected.requirements().session_group(),selected.requirements().publication_owner()),
            SelectedExecutionKind::PartitionedRouted(selected)=>(selected.requirements().session_group(),selected.requirements().publication_owner()),
            SelectedExecutionKind::PartitionedComposite(selected)=>(selected.requirements().session_group(),selected.requirements().publication_owner()),
            _=>return None,
        };
        Some(eredu_runtime::PartitionOutputPublication {group:group?,owner_rank})
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
    pub(crate) fn same_complete_selection(&self, other: &Self) -> bool {
        self.admission_token.same_admission(&other.admission_token)
            && self.execution == other.execution
            && self.admission == other.admission
            && self.prediction_realization == other.prediction_realization
            && match (&self.prediction_extension, &other.prediction_extension) {
                (Some(a), Some(b)) => a.same_admission(b),
                (None, None) => true,
                _ => false,
            }
    }

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

    /// Exact architecture-selected output owner and publication group.
    pub fn partitioned_output_publication(&self)->Option<eredu_runtime::PartitionOutputPublication> {
        self.execution.partitioned_output_publication()
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
