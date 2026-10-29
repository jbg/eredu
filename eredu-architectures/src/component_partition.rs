//! Architecture-owned observation coordinates for an exact partition selection.

use std::collections::BTreeMap;

use eredu_core::{
    capture::PartitionCaptureCombination,
    component::{ComponentCoordinateMap, ComponentGroup, ComponentWritePartition},
    ArchitectureDescriptor,
};
use eredu_runtime::{
    inspection::ObservationHookSite, ArchitectureParameterDescription, LocalModelLayout,
    LocalTensorLayout, TensorPlacement,
};

mod capture_bundle;
mod input_projection;
mod output_gate;
mod output_projection;
mod prediction;
mod residual;
mod routed;
mod streams;
mod transforms;
pub(crate) use routed::{derive_bank_unit_coordinates, derive_coordinates_for_experts};
pub use routed::{derive_routed_component_coordinates, PartitionedRoutedObservation};

/// One invocation's local scalar coordinates. Absence of coordinates means that
/// this partition does not execute the invocation, even if it stores shared weights.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionedComponentGroup {
    coordinates: Option<ComponentCoordinateMap>,
}

impl PartitionedComponentGroup {
    /// Local scalar coordinates, or no invocation on this partition.
    pub fn coordinates(&self) -> Option<&ComponentCoordinateMap> {
        self.coordinates.as_ref()
    }
}

/// One observation's semantic axis and invocation-local coordinates. A component
/// input and its scalar activations can have different axes and placements.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionedObservation {
    axis: String,
    coordinates: Option<ComponentCoordinateMap>,
    exports: bool,
    site: ObservationHookSite,
    combination: PartitionCaptureCombination,
}

impl PartitionedObservation {
    /// Declared observation axis whose coordinates this placement describes.
    pub fn axis(&self) -> &str {
        &self.axis
    }

    /// Local coordinates, or no observation invocation on this rank.
    pub fn coordinates(&self) -> Option<&ComponentCoordinateMap> {
        self.coordinates.as_ref()
    }

    /// Whether this invocation can supply an authoritative capture fragment.
    /// Replicas can still supply dependencies while publication belongs elsewhere.
    pub const fn exports(&self) -> bool {
        self.exports
    }

    /// Actual execution site whose hook is required in addition to placement.
    pub const fn site(&self) -> ObservationHookSite {
        self.site
    }

    /// How authoritative native contributions form the selected global value.
    pub const fn combination(&self) -> PartitionCaptureCombination {
        self.combination
    }
}

/// Exact component-axis placements from architecture semantics and retained rank
/// ownership. This is descriptive metadata, not distributed capture admission.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComponentPartitionLayout {
    topology: eredu_core::ParallelRankTopology,
    groups: BTreeMap<String, PartitionedComponentGroup>,
    paths: BTreeMap<String, String>,
    observations: BTreeMap<String, PartitionedObservation>,
    routed: BTreeMap<String, PartitionedRoutedObservation>,
}

impl ComponentPartitionLayout {
    pub(crate) fn register_unit_observation(
        &mut self,
        descriptor: &ArchitectureDescriptor,
        path: &str,
        local: bool,
    ) -> Result<(), ComponentPartitionError> {
        replicated_observation(
            &mut self.observations,
            descriptor,
            path,
            "hidden",
            local,
            ObservationHookSite::Unit,
        )
    }

    /// Projects an immutable global activation operation onto this invocation.
    /// Nonexporting replicas still participate; absence means this rank does not
    /// execute the invocation. This grants no distributed execution authority.
    pub fn project_intervention<'a>(
        &'a self,
        plan: &'a eredu_core::intervention::AdmittedInterventionPlan,
        operation: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        max_regions: usize,
    ) -> Result<
        Option<eredu_runtime::intervention::PartitionActivationProjection<'a>>,
        ComponentPartitionError,
    > {
        self.project_intervention_at(plan, operation, phase, prediction, None, max_regions)
    }

    /// Projects an explicitly shaped forward without changing the admitted operation.
    pub fn project_intervention_at<'a>(
        &'a self,
        plan: &'a eredu_core::intervention::AdmittedInterventionPlan,
        operation: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        invocation: Option<eredu_core::capture::CaptureInvocationShape>,
        max_regions: usize,
    ) -> Result<
        Option<eredu_runtime::intervention::PartitionActivationProjection<'a>>,
        ComponentPartitionError,
    > {
        use eredu_core::capture::CaptureError;
        let operation_plan = plan.plan().operations.get(operation).ok_or_else(|| {
            CaptureError::Invalid("unknown partition intervention operation".into())
        })?;
        if prediction >= plan.request().max_predictions
            || !operation_plan.schedule.includes(phase, prediction)
        {
            return Err(CaptureError::Invalid(
                "partition intervention is outside its admitted schedule".into(),
            )
            .into());
        }
        let point = &plan.points()[operation];
        let placement = self
            .observation(&point.path)
            .ok_or_else(|| CaptureError::MissingPath(point.path.clone()))?;
        let Some(coordinates) = placement.coordinates() else {
            return Ok(None);
        };
        let mut matching = point
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| axis.name == placement.axis());
        let Some((axis, _)) = matching.next() else {
            return Err(CaptureError::Invalid(
                "intervention has no retained observation axis".into(),
            )
            .into());
        };
        if matching.next().is_some() {
            return Err(CaptureError::Invalid(
                "intervention repeats its retained observation axis".into(),
            )
            .into());
        }
        let shape = plan
            .geometry_at(phase, prediction, invocation)?
            .resolve(&point.observation_geometry())?
            .ok_or_else(|| {
                CaptureError::Unsupported("partition intervention shape is unresolved".into())
            })?;
        let projection = eredu_runtime::intervention::PartitionActivationProjection::new_at(
            plan,
            operation,
            phase,
            prediction,
            invocation,
            &shape,
            axis,
            coordinates,
            max_regions,
        )?;
        let projection = match placement.combination() {
            PartitionCaptureCombination::Disjoint => projection,
            PartitionCaptureCombination::SumF64ToF32 => {
                projection.as_sum_term(self.topology.tensor_parallel_rank() == 0)?
            }
        };
        Ok(Some(projection))
    }
    /// Exact sparse invocation placement, including absence on this pipeline rank.
    pub fn routed_observation(&self, path: &str) -> Option<&PartitionedRoutedObservation> {
        self.routed.get(path)
    }

    /// Exact world rank whose invocation ownership and scalar storage are described.
    pub const fn topology(&self) -> eredu_core::ParallelRankTopology {
        self.topology
    }

    /// Resolves a stable component-group identity, including nonlocal invocations.
    pub fn group(&self, id: &str) -> Option<&PartitionedComponentGroup> {
        self.groups.get(id)
    }

    /// Resolves a component activation, effective activation, or actual write input.
    /// Normalized sublayer inputs are different axes and are not registered here.
    pub fn point(&self, path: &str) -> Option<&PartitionedComponentGroup> {
        self.paths.get(path).and_then(|id| self.groups.get(id))
    }

    /// Resolves retained capture ownership, including normalized inputs and
    /// readout values whose axes differ from scalar component activations.
    pub fn observation(&self, path: &str) -> Option<&PartitionedObservation> {
        self.observations.get(path)
    }

    /// Projects an admitted global component capture using this retained
    /// invocation/partition layout. `None` means the invocation is not local;
    /// a local projection may separately have no overlap with the selected slice.
    /// This grants no peer-publication or transport authority.
    pub fn project_capture(
        &self,
        plan: &eredu_core::capture::AdmittedCapturePlan,
        selection_index: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        max_fragments: usize,
    ) -> Result<Option<eredu_core::capture::CaptureSlicePartition>, ComponentPartitionError> {
        self.project_capture_at(
            plan,
            selection_index,
            phase,
            prediction,
            None,
            max_fragments,
        )
    }

    /// Projects explicit invocation axes through this retained local ownership.
    pub fn project_capture_at(
        &self,
        plan: &eredu_core::capture::AdmittedCapturePlan,
        selection_index: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        invocation: Option<eredu_core::capture::CaptureInvocationShape>,
        max_fragments: usize,
    ) -> Result<Option<eredu_core::capture::CaptureSlicePartition>, ComponentPartitionError> {
        use eredu_core::capture::{resolve_slice, CaptureError, CaptureSlicePartition};
        let selection =
            plan.plan().selections.get(selection_index).ok_or_else(|| {
                CaptureError::Invalid("unknown component capture selection".into())
            })?;
        if prediction >= plan.request().max_predictions
            || !selection.schedule.includes(phase, prediction)
        {
            return Err(CaptureError::Invalid(
                "component capture is outside its admitted schedule".into(),
            )
            .into());
        }
        let point = &plan.points()[selection_index];
        let placement = self
            .observation(&point.path)
            .ok_or_else(|| CaptureError::MissingPath(point.path.clone()))?;
        let Some(coordinates) = placement.coordinates() else {
            return Ok(None);
        };
        let axes = point.axes.as_ref().ok_or_else(|| {
            CaptureError::Unsupported("component capture axes are not declared".into())
        })?;
        let mut matching = axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| axis.name == placement.axis());
        let Some((axis, _)) = matching.next() else {
            return Err(
                CaptureError::Invalid("capture has no retained observation axis".into()).into(),
            );
        };
        if matching.next().is_some() {
            return Err(CaptureError::Invalid(
                "capture repeats its retained observation axis".into(),
            )
            .into());
        }
        let shape = plan
            .geometry_at(phase, prediction, invocation)?
            .resolve(point)?
            .ok_or_else(|| {
                CaptureError::Unsupported("component capture shape is unresolved".into())
            })?;
        let slice = resolve_slice(point, selection, &shape)?;
        CaptureSlicePartition::new(&shape, &slice, axis, coordinates, max_fragments)
            .map(Some)
            .map_err(Into::into)
    }

    fn from_components(
        descriptor: &ArchitectureDescriptor,
        components: &[ComponentGroup],
        layout: &LocalModelLayout,
        topology: eredu_core::ParallelRankTopology,
        owns_parameter_invocation: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
    ) -> Result<Self, ComponentPartitionError> {
        let mut groups = BTreeMap::new();
        let mut paths = BTreeMap::new();
        let mut observations = BTreeMap::new();
        for component in components {
            // Use the invocation's logical weight, never its shared source alias.
            // Reused checkpoint storage does not transfer execution ownership.
            let weight = component
                .write_input_projection
                .as_ref()
                .map_or(&component.write_weight, |stage| &stage.weight);
            let local = owns_parameter_invocation(weight)?;
            let coordinates = if local {
                let tensor = layout
                    .tensor(weight)
                    .ok_or_else(|| ComponentPartitionError::MissingWeight(weight.clone()))?;
                Some(derive_component_coordinates(component, tensor)?)
            } else {
                None
            };
            if groups
                .insert(
                    component.id.clone(),
                    PartitionedComponentGroup {
                        coordinates: coordinates.clone(),
                    },
                )
                .is_some()
            {
                return Err(ComponentPartitionError::DuplicateIdentity(
                    component.id.clone(),
                ));
            }
            for path in [&component.activation, &component.effective_activation]
                .into_iter()
                .chain(component.write_input.iter())
            {
                if paths.insert(path.clone(), component.id.clone()).is_some() {
                    return Err(ComponentPartitionError::DuplicateIdentity(path.clone()));
                }
                insert_observation(
                    &mut observations,
                    path,
                    PartitionedObservation {
                        axis: "component".into(),
                        coordinates: coordinates.clone(),
                        exports: local,
                        site: ObservationHookSite::Unit,
                        combination: PartitionCaptureCombination::Disjoint,
                    },
                )?;
            }
            // Shared decoder reads consume the complete normalized hidden axis
            // on each executing TP/EP rank, before scalar projection sharding.
            replicated_observation(
                &mut observations,
                descriptor,
                &component.input,
                "hidden",
                local,
                ObservationHookSite::Unit,
            )?;
            for read in component
                .reads
                .iter()
                .filter(|read| read.projection_output.is_some())
            {
                if owns_parameter_invocation(&read.weight)? != local {
                    return Err(ComponentPartitionError::InvalidPlacement(
                        read.weight.clone(),
                    ));
                }
                let tensor = if local {
                    Some(layout.tensor(&read.weight).ok_or_else(|| {
                        ComponentPartitionError::MissingWeight(read.weight.clone())
                    })?)
                } else {
                    None
                };
                input_projection::insert_read_output(&mut observations, descriptor, read, tensor)?;
            }
            for stage in component
                .reads
                .iter()
                .chain(component.output_gate.iter().map(|gate| &gate.read))
                .flat_map(|read| &read.input_projections)
            {
                input_projection::insert_observation(
                    &mut observations,
                    descriptor,
                    stage,
                    if local {
                        Some(layout.tensor(&stage.weight).ok_or_else(|| {
                            ComponentPartitionError::MissingWeight(stage.weight.clone())
                        })?)
                    } else {
                        None
                    },
                )?;
            }
            if component.output_gate.is_some() {
                output_gate::insert_observations(
                    &mut observations,
                    descriptor,
                    component,
                    layout,
                    local,
                    owns_parameter_invocation,
                )?;
            }
            if let Some(stage) = &component.write_input_projection {
                if owns_parameter_invocation(&component.write_weight)? != local {
                    return Err(ComponentPartitionError::InvalidPlacement(
                        component.write_weight.clone(),
                    ));
                }
                output_projection::insert_observations(
                    &mut observations,
                    descriptor,
                    component,
                    stage,
                    local.then_some(layout),
                )?;
            }
            if let eredu_core::component::ComponentActivation::Attention {
                query_heads,
                head_width,
                attended_head_normalization: Some(head),
                ..
            } = &component.activation_equation
            {
                let point = descriptor.observations.get(&head.input).ok_or_else(|| {
                    eredu_core::capture::CaptureError::MissingPath(head.input.clone())
                })?;
                let axes = point.axes.as_deref().ok_or_else(|| {
                    ComponentPartitionError::InvalidPlacement(component.id.clone())
                })?;
                if point.position != eredu_core::ObservationPosition::ReadOnly
                    || !matches!(axes, [batch, heads, sequence, channels]
                        if batch.dimension == eredu_core::SymbolicDimension::Batch
                            && heads.dimension == eredu_core::SymbolicDimension::Known(*query_heads)
                            && sequence.dimension == eredu_core::SymbolicDimension::Sequence
                            && channels.dimension == eredu_core::SymbolicDimension::Known(*head_width))
                {
                    return Err(ComponentPartitionError::InvalidPlacement(
                        component.id.clone(),
                    ));
                }
                let head_coordinates = coordinates
                    .as_ref()
                    .map(|coordinates| {
                        transforms::complete_groups(coordinates, *query_heads, *head_width)
                    })
                    .transpose()?;
                insert_observation(
                    &mut observations,
                    &head.input,
                    PartitionedObservation {
                        axis: axes[1].name.clone(),
                        coordinates: head_coordinates,
                        exports: local,
                        site: ObservationHookSite::Unit,
                        combination: PartitionCaptureCombination::Disjoint,
                    },
                )?;
            }
            for path in component.write_output.iter().chain(component.output.iter()) {
                replicated_observation(
                    &mut observations,
                    descriptor,
                    path,
                    "hidden",
                    local,
                    ObservationHookSite::Unit,
                )?;
                if component.write_partition == ComponentWritePartition::TensorParallelSum
                    && topology.tensor_parallel_size() > 1
                {
                    if component.output_normalization.is_some() || component.write_bias.is_some() {
                        return Err(eredu_core::capture::CaptureError::Unsupported(
                            "additive component write requires an explicit bias-free, unnormalized TP equation".into(),
                        ).into());
                    }
                    for name in [path.clone(), format!("{path}.effective")] {
                        if let Some(placement) = observations.get_mut(&name) {
                            placement.combination = PartitionCaptureCombination::SumF64ToF32;
                        }
                    }
                }
            }
        }
        Ok(Self {
            topology,
            groups,
            paths,
            observations,
            routed: BTreeMap::new(),
        })
    }

    pub(crate) fn from_ownership(
        descriptor: &ArchitectureDescriptor,
        parameters: &ArchitectureParameterDescription,
        layout: &LocalModelLayout,
        topology: eredu_core::ParallelRankTopology,
        ownership: &eredu_runtime::PartitionOwnership,
        owned_groups: &[eredu_runtime::PipelineGroupOwnership],
        publication_owner: usize,
        selected_routed: Option<&crate::SelectedRoutedTextRealization>,
        component: Option<&eredu_runtime::ComponentDefinition>,
    ) -> Result<Self, ComponentPartitionError> {
        let owns_invocation = |address: eredu_runtime::ExecutionUnitAddress| {
            let group = parameters.unit_layout().group_id(address.group());
            owned_groups.iter().any(|owned| {
                Some(owned.group()) == group && owned.units().contains(&address.index())
            })
        };
        let observation_local = |path: &str| {
            component
                .and_then(|definition| definition.observation_invocation(path))
                .map(&owns_invocation)
        };
        // This inspection-only view projects declared references onto canonical
        // placement; it allocates neither parameter storage nor execution units.
        let projected_layout;
        let layout = if let Some(definition) = component {
            if definition.parameters().invocations() != parameters.invocations() {
                return Err(ComponentPartitionError::ParameterLayout(
                    "component invocation authority differs from selected parameters".into(),
                ));
            }
            let mut projected = layout.clone();
            for (logical, usage) in definition.parameter_uses() {
                let tensor = layout
                    .tensor(&usage.target)
                    .ok_or_else(|| ComponentPartitionError::MissingWeight(usage.target.clone()))?;
                if projected
                    .tensor(logical)
                    .is_some_and(|existing| existing != tensor)
                {
                    return Err(ComponentPartitionError::InvalidPlacement(logical.clone()));
                }
                projected.insert(logical.clone(), tensor.clone());
            }
            projected_layout = projected;
            &projected_layout
        } else {
            layout
        };
        let owns_parameter_invocation = |parameter: &str| {
            if let Some(usage) =
                component.and_then(|definition| definition.parameter_uses().get(parameter))
            {
                return Ok(owns_invocation(usage.invocation));
            }
            let owned = parameters
                .groups()
                .iter()
                .find(|group| {
                    group
                        .members()
                        .iter()
                        .any(|member| member.target() == parameter)
                })
                .ok_or_else(|| ComponentPartitionError::MissingWeight(parameter.into()))?;
            Ok::<_, ComponentPartitionError>(owned.owner().is_owned_by(ownership, |group, unit| {
                owned_groups
                    .iter()
                    .any(|owned| owned.group() == group && owned.units().contains(&unit))
            }))
        };
        let Self {
            groups,
            paths,
            mut observations,
            ..
        } = Self::from_components(
            descriptor,
            &descriptor.components,
            layout,
            topology,
            &owns_parameter_invocation,
        )?;
        // Complete block boundaries and residual additions execute in their
        // containing unit after TP reduction. Their hidden values are replicated
        // on that unit's ranks, independently of contribution parameter shards.
        for node in descriptor.nodes.iter().filter(|node| {
            node.kind == eredu_core::ArchitectureNodeKind::ResidualAdd
                || (node.kind == eredu_core::ArchitectureNodeKind::DecoderBlock
                    && descriptor
                        .layer_groups
                        .iter()
                        .flat_map(|group| &group.passes)
                        .flat_map(|pass| &pass.executions)
                        .any(|execution| execution.node_id == node.id))
        }) {
            if crate::speculative_execution::speculative_capture_scope(descriptor, &node.id)?
                != eredu_core::speculative::SpeculativeCaptureScope::Target
            {
                continue;
            }
            let points = descriptor
                .observations
                .points
                .iter()
                .filter(|point| {
                    point.node_id == node.id
                        && node.observation_paths.contains(&point.path)
                        && point
                            .axes
                            .as_ref()
                            .is_some_and(|axes| axes.iter().any(|axis| axis.name == "hidden"))
                })
                .collect::<Vec<_>>();
            if points.is_empty() {
                continue;
            }
            if points
                .iter()
                .all(|point| observation_local(&point.path).is_some())
            {
                for point in points {
                    replicated_observation(
                        &mut observations,
                        descriptor,
                        &point.path,
                        "hidden",
                        observation_local(&point.path).expect("declared observation"),
                        ObservationHookSite::Unit,
                    )?;
                }
                continue;
            }
            let owner = if node.kind == eredu_core::ArchitectureNodeKind::DecoderBlock {
                capture_bundle::block_invocation_owner(descriptor, parameters, &node.id)?
            } else {
                residual_invocation_owner(descriptor, parameters, &node.id)?
            };
            let local = owner.is_owned_by(ownership, |group, unit| {
                owned_groups
                    .iter()
                    .any(|owned| owned.group() == group && owned.units().contains(&unit))
            });
            for point in points {
                replicated_observation(
                    &mut observations,
                    descriptor,
                    &point.path,
                    "hidden",
                    local,
                    ObservationHookSite::Unit,
                )?;
            }
        }
        for point in descriptor.observations.points.iter().filter(|point| {
            point.axes.as_ref().is_some_and(|axes| {
                axes.iter().any(|axis| {
                    matches!(
                        axis.dimension,
                        eredu_core::SymbolicDimension::MediaPositions
                            | eredu_core::SymbolicDimension::ProjectedMediaPositions
                    )
                })
            })
        }) {
            let owner = residual_invocation_owner(descriptor, parameters, &point.node_id)?;
            let local = owner.is_owned_by(ownership, |group, unit| {
                owned_groups
                    .iter()
                    .any(|owned| owned.group() == group && owned.units().contains(&unit))
            });
            replicated_observation(
                &mut observations,
                descriptor,
                &point.path,
                "hidden",
                local,
                ObservationHookSite::Unit,
            )?;
        }
        register_routed_boundaries(
            &mut observations,
            descriptor,
            &descriptor.routed_components,
            |node| {
                let owner = node_invocation_owner(descriptor, parameters, node)?;
                Ok(owner.is_owned_by(ownership, |group, unit| {
                    owned_groups
                        .iter()
                        .any(|owned| owned.group() == group && owned.units().contains(&unit))
                }))
            },
        )?;
        if let Some(readout) = &descriptor.component_readout {
            if let Some(streams) = &readout.stream_residual {
                streams::register(
                    &mut observations,
                    descriptor,
                    streams,
                    &owns_parameter_invocation,
                    (ownership.owns_input(), ObservationHookSite::Input),
                    (ownership.owns_output(), ObservationHookSite::Readout),
                )?;
            }
            if readout.logits != eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                return Err(eredu_core::capture::CaptureError::Invalid(
                    "component readout does not name the public final-output seam".into(),
                )
                .into());
            }
            for normalization in &readout.block_normalizations {
                let parameter = normalization
                    .normalization
                    .gain
                    .as_ref()
                    .or(normalization.normalization.bias.as_ref())
                    .ok_or_else(|| {
                        eredu_core::capture::CaptureError::Unsupported(
                            "block normalization has no parameter-owned invocation".into(),
                        )
                    })?;
                // The declared logical parameter retains the invocation owner,
                // even when its prepared source aliases a shared final gain.
                let local = owns_parameter_invocation(parameter)?;
                for path in [&normalization.input, &normalization.output] {
                    replicated_observation(
                        &mut observations,
                        descriptor,
                        path,
                        "hidden",
                        observation_local(path).unwrap_or(local),
                        ObservationHookSite::Unit,
                    )?;
                }
            }
            for residual in &readout.block_transforms {
                let invalid = || {
                    eredu_core::capture::CaptureError::Invalid(format!(
                        "block transform has no matching residual invocation: {}",
                        residual.transform_id
                    ))
                };
                let transform = descriptor
                    .component_transforms
                    .iter()
                    .find(|transform| transform.id == residual.transform_id)
                    .ok_or_else(&invalid)?;
                let point = descriptor
                    .observations
                    .get(&transform.input)
                    .ok_or_else(&invalid)?;
                let mut node = descriptor
                    .nodes
                    .iter()
                    .find(|node| node.id == point.node_id)
                    .ok_or_else(&invalid)?;
                // The readout declares an invocation, independently of parameter
                // sharing. Resolve the input's containing block and prove the
                // declared layer before registering its complete residual axes.
                for _ in 0..descriptor.nodes.len() {
                    if node.kind == eredu_core::ArchitectureNodeKind::DecoderBlock {
                        break;
                    }
                    let parent = node.parent.as_ref().ok_or_else(&invalid)?;
                    node = descriptor
                        .nodes
                        .iter()
                        .find(|node| &node.id == parent)
                        .ok_or_else(&invalid)?;
                }
                if node.kind != eredu_core::ArchitectureNodeKind::DecoderBlock
                    || node.layer_index != Some(residual.layer_index)
                {
                    return Err(invalid().into());
                }
                let owner = node_invocation_owner(descriptor, parameters, &node.id)?;
                let local = owner.is_owned_by(ownership, |group, unit| {
                    owned_groups
                        .iter()
                        .any(|owned| owned.group() == group && owned.units().contains(&unit))
                });
                let input = if point.position == eredu_core::ObservationPosition::AfterIntervention
                {
                    let input = transform
                        .input
                        .strip_suffix(".effective")
                        .ok_or_else(&invalid)?;
                    let original = descriptor.observations.get(input).ok_or_else(&invalid)?;
                    if original.position != eredu_core::ObservationPosition::BeforeIntervention
                        || original.axes != point.axes
                        || original.node_id != point.node_id
                    {
                        return Err(invalid().into());
                    }
                    input
                } else {
                    &transform.input
                };
                replicated_observation(
                    &mut observations,
                    descriptor,
                    input,
                    "hidden",
                    local,
                    ObservationHookSite::Unit,
                )?;
            }
            for write in &readout.other_writes {
                let owner = node_invocation_owner(descriptor, parameters, &write.node_id)?;
                let local = owner.is_owned_by(ownership, |group, unit| {
                    owned_groups
                        .iter()
                        .any(|owned| owned.group() == group && owned.units().contains(&unit))
                });
                for path in [&write.output, &write.effective_output]
                    .into_iter()
                    .chain(write.input.iter())
                {
                    replicated_observation(
                        &mut observations,
                        descriptor,
                        path,
                        "hidden",
                        local,
                        ObservationHookSite::Unit,
                    )?;
                }
                // A declared residual-add invocation also owns its incoming and
                // resulting hidden-state evidence. Its contribution may come
                // from an earlier modality group, but the addition executes at
                // this unit, independently of source parameter residency.
                if descriptor.nodes.iter().any(|node| {
                    node.id == write.node_id
                        && node.kind == eredu_core::ArchitectureNodeKind::ResidualAdd
                }) {
                    for point in descriptor.observations.points.iter().filter(|point| {
                        point.node_id == write.node_id
                            && point
                                .axes
                                .as_ref()
                                .is_some_and(|axes| axes.iter().any(|axis| axis.name == "hidden"))
                    }) {
                        replicated_observation(
                            &mut observations,
                            descriptor,
                            &point.path,
                            "hidden",
                            local,
                            ObservationHookSite::Unit,
                        )?;
                    }
                }
            }
            // Invocation ownership is distinct from tied parameter storage:
            // the output stage may store embeddings without executing lookup.
            replicated_observation(
                &mut observations,
                descriptor,
                &readout.embedding,
                "hidden",
                ownership.owns_input(),
                ObservationHookSite::Input,
            )?;
            for path in [&readout.residual, &readout.normalized]
                .into_iter()
                .chain(readout.projection_input.iter())
            {
                replicated_observation(
                    &mut observations,
                    descriptor,
                    path,
                    "hidden",
                    ownership.owns_output(),
                    ObservationHookSite::Readout,
                )?;
            }
            // The shared vocabulary projection completes its global gather
            // before emitting this affine score, on each output-stage rank.
            replicated_observation(
                &mut observations,
                descriptor,
                &readout.linear_scores,
                "vocabulary",
                ownership.owns_output(),
                ObservationHookSite::Readout,
            )?;
        }
        if let Some(readout) = &descriptor.component_readout {
            residual::register(
                &mut observations,
                descriptor,
                &readout.residual_cycles,
                layout,
                &owns_parameter_invocation,
            )?;
        }
        transforms::register(
            &mut observations,
            descriptor,
            layout,
            eredu_core::speculative::SpeculativeCaptureScope::Target,
            &owns_parameter_invocation,
        )?;
        // Final publication belongs to the executor independently of whether
        // this family has declared component decomposition or readout equations.
        if descriptor
            .observations
            .points
            .iter()
            .any(|point| point.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        {
            replicated_observation(
                &mut observations,
                descriptor,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                "vocabulary",
                ownership.owns_output(),
                ObservationHookSite::Publication,
            )?;
            let logits = observations
                .get_mut(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                .expect("registered logits");
            logits.exports = topology.global_rank() == publication_owner;
            if logits.exports && logits.coordinates.is_none() {
                return Err(eredu_core::capture::CaptureError::Invalid(
                    "publication owner has no final-output invocation".into(),
                )
                .into());
            }
        }
        let routed = if let Some(selected) = selected_routed {
            routed::placement::derive_observations(
                descriptor,
                parameters,
                layout,
                topology,
                owned_groups,
                selected,
            )?
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            topology,
            groups,
            paths,
            observations,
            routed,
        })
    }
}

impl eredu_runtime::intervention::PartitionActivationLayout for ComponentPartitionLayouts {
    fn routed_activation_members<'a>(
        &'a self,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
        operation: usize,
        max_members: usize,
    ) -> Result<
        Vec<eredu_runtime::intervention::PartitionRoutedActivationMember<'a>>,
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::capture::CaptureError;
        let point = plan
            .points()
            .get(operation)
            .ok_or_else(|| CaptureError::Invalid("unknown sparse operation".into()))?;
        let routed = point
            .routed_units
            .as_ref()
            .ok_or_else(|| CaptureError::Invalid("operation is not sparse".into()))?;
        let mut members = Vec::new();
        for (rank, layout) in self.layouts.iter().enumerate() {
            let retained = layout
                .routed_observation(&point.path)
                .ok_or_else(|| CaptureError::MissingPath(point.path.clone()))?;
            if retained.routing() != routed.routing
                || retained.geometry() != routed.geometry
                || retained.effective()
            {
                return Err(CaptureError::Invalid(
                    "sparse operation differs from retained invocation".into(),
                ));
            }
            if let Some(ownership) = retained.ownership() {
                if members.len() == max_members {
                    return Err(CaptureError::Invalid(
                        "sparse operation membership exceeds bound".into(),
                    ));
                }
                members.push(
                    eredu_runtime::intervention::PartitionRoutedActivationMember {
                        rank,
                        ownership,
                        input_width: retained.input_width(),
                    },
                );
            }
        }
        Ok(members)
    }
    fn activation_region_bound(
        &self,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
        operation: usize,
        _max_members: usize,
        max_regions: usize,
    ) -> Result<usize, eredu_core::capture::CaptureError> {
        use eredu_core::{capture::CaptureError, intervention::InterventionAction};
        let operation = plan
            .plan()
            .operations
            .get(operation)
            .ok_or_else(|| CaptureError::Invalid("unknown intervention operation".into()))?;
        let point = plan
            .points()
            .iter()
            .find(|point| point.path == operation.target)
            .ok_or_else(|| CaptureError::MissingPath(operation.target.clone()))?;
        let columns = matches!(
            operation.action,
            InterventionAction::MaskComponents { .. } | InterventionAction::MaskLogits { .. }
        );
        let mut regions = 0usize;
        for layout in &self.layouts {
            let placement = layout
                .observation(&point.path)
                .ok_or_else(|| CaptureError::MissingPath(point.path.clone()))?;
            let Some(coordinates) = placement.coordinates() else {
                continue;
            };
            let count = if coordinates.local_count() == 0 {
                0
            } else if coordinates.contiguous_range().is_some()
                || (columns
                    && point
                        .axes
                        .last()
                        .is_some_and(|axis| axis.name == placement.axis()))
            {
                1
            } else {
                coordinates.local_count()
            };
            regions = regions.checked_add(count).ok_or(CaptureError::Overflow)?;
        }
        // The projection enforces both bounds before retaining any additional
        // region. A wholly empty invocation still has a bounded metadata owner.
        Ok(regions.max(1).min(max_regions))
    }
    fn activation_members<'a>(
        &'a self,
        plan: &'a eredu_core::intervention::AdmittedInterventionPlan,
        operation: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        max_members: usize,
        max_regions: usize,
    ) -> Result<
        Vec<eredu_runtime::intervention::PartitionActivationMember<'a>>,
        eredu_core::capture::CaptureError,
    > {
        self.activation_members_at(
            plan,
            operation,
            phase,
            prediction,
            None,
            max_members,
            max_regions,
        )
    }

    fn activation_members_at<'a>(
        &'a self,
        plan: &'a eredu_core::intervention::AdmittedInterventionPlan,
        operation: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        invocation: Option<eredu_core::capture::CaptureInvocationShape>,
        max_members: usize,
        max_regions: usize,
    ) -> Result<
        Vec<eredu_runtime::intervention::PartitionActivationMember<'a>>,
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::capture::CaptureError;
        if max_members == 0 || max_regions == 0 {
            return Err(CaptureError::Invalid(
                "partition intervention bounds must be positive".into(),
            ));
        }
        let path = &plan
            .plan()
            .operations
            .get(operation)
            .ok_or_else(|| {
                CaptureError::Invalid("unknown partition intervention operation".into())
            })?
            .target;
        self.observation_combination(path)?;
        let mut members = Vec::new();
        let mut regions = 0usize;
        for (rank, layout) in self.layouts.iter().enumerate() {
            let projection = layout
                .project_intervention_at(
                    plan,
                    operation,
                    phase,
                    prediction,
                    invocation,
                    max_regions,
                )
                .map_err(|error| match error {
                    ComponentPartitionError::Capture(error) => error,
                    error => CaptureError::Invalid(error.to_string()),
                })?;
            if let Some(projection) = projection {
                regions = regions
                    .checked_add(projection.region_count())
                    .ok_or(CaptureError::Overflow)?;
                if members.len() >= max_members || regions > max_regions {
                    return Err(CaptureError::Invalid(
                        "partition intervention membership or region bound exceeded".into(),
                    ));
                }
                members.push(eredu_runtime::intervention::PartitionActivationMember {
                    rank,
                    projection,
                });
            }
        }
        if members.is_empty() {
            return Err(CaptureError::Invalid(
                "partition intervention has no invocation owner".into(),
            ));
        }
        Ok(members)
    }
}

/// Join a semantic operator to its logical parameter namespaces. The complete
/// residual write executes at that unit, independently of shared source storage.
fn register_routed_boundaries(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    components: &[eredu_core::component::RoutedComponentGroup],
    owns_node: impl Fn(&str) -> Result<bool, ComponentPartitionError>,
) -> Result<(), ComponentPartitionError> {
    for component in components {
        let mut boundaries = component
            .input
            .iter()
            .map(|path| (path, Some(component.input_width)))
            .chain(
                component
                    .write_output
                    .iter()
                    .map(|path| (path, Some(component.output_width))),
            )
            .chain(component.output.iter().map(|path| (path, None)))
            .collect::<Vec<_>>();
        if let Some(id) = &component.expert_output_transform {
            let transform = descriptor
                .component_transforms
                .iter()
                .find(|transform| &transform.id == id)
                .ok_or_else(|| ComponentPartitionError::InvalidPlacement(component.id.clone()))?;
            let path = transform
                .input
                .strip_suffix(".effective")
                .unwrap_or(&transform.input);
            let point = descriptor
                .observations
                .get(path)
                .ok_or_else(|| eredu_core::capture::CaptureError::MissingPath(path.into()))?;
            let routes = component
                .routes_per_token
                .ok_or_else(|| ComponentPartitionError::InvalidPlacement(component.id.clone()))?;
            if !point.axes.as_ref().is_some_and(|axes| matches!(axes.as_slice(), [token, route, channel]
                if token.dimension == eredu_core::SymbolicDimension::TokenRows
                    && route.dimension == eredu_core::SymbolicDimension::Known(routes)
                    && channel.dimension == eredu_core::SymbolicDimension::Known(component.output_width))) {
                return Err(ComponentPartitionError::InvalidPlacement(component.id.clone()));
            }
            // Each routed output is complete across tensor ranks before this
            // nonlinear transform; expert-unit shards are a different boundary.
            boundaries.push((&transform.input, Some(component.output_width)));
        }
        for (path, width) in boundaries {
            let point = descriptor
                .observations
                .get(path)
                .ok_or_else(|| eredu_core::capture::CaptureError::MissingPath(path.clone()))?;
            let axis = point.axes.as_ref().and_then(|axes| axes.last()).filter(|axis|
                matches!(axis.dimension, eredu_core::SymbolicDimension::Known(actual) if actual > 0 && width.is_none_or(|width| width == actual)))
                .ok_or_else(|| ComponentPartitionError::InvalidPlacement(component.id.clone()))?;
            let original = if point.position == eredu_core::ObservationPosition::AfterIntervention {
                path.strip_suffix(".effective").ok_or_else(|| {
                    ComponentPartitionError::InvalidPlacement(component.id.clone())
                })?
            } else {
                path.as_str()
            };
            replicated_observation(
                observations,
                descriptor,
                original,
                &axis.name,
                owns_node(&point.node_id)?,
                ObservationHookSite::Unit,
            )?;
        }
    }
    Ok(())
}

fn residual_invocation_owner<'a>(
    descriptor: &ArchitectureDescriptor,
    parameters: &'a ArchitectureParameterDescription,
    node_id: &str,
) -> Result<&'a eredu_runtime::ParameterGroupOwner, ComponentPartitionError> {
    use eredu_core::capture::CaptureError;
    let mut current = Some(node_id);
    for _ in 0..=descriptor.nodes.len() {
        let Some(id) = current else {
            break;
        };
        let node = descriptor.node(id).ok_or_else(|| {
            CaptureError::Invalid("residual ancestry contains an unknown node".into())
        })?;
        if !node.parameter_groups.is_empty() {
            return node_invocation_owner(descriptor, parameters, id);
        }
        current = node.parent.as_deref();
    }
    Err(CaptureError::Invalid("residual addition has no containing execution unit".into()).into())
}

fn node_invocation_owner<'a>(
    descriptor: &ArchitectureDescriptor,
    parameters: &'a ArchitectureParameterDescription,
    node_id: &str,
) -> Result<&'a eredu_runtime::ParameterGroupOwner, ComponentPartitionError> {
    use eredu_core::capture::CaptureError;
    let invalid = || {
        CaptureError::Invalid(format!(
            "residual write has no unique unit owner: {node_id}"
        ))
    };
    let node = descriptor
        .nodes
        .iter()
        .find(|node| node.id == node_id)
        .ok_or_else(invalid)?;
    let mut owner = None;
    for group_id in &node.parameter_groups {
        let group = descriptor
            .parameter_groups
            .iter()
            .find(|group| &group.id == group_id)
            .ok_or_else(invalid)?;
        // A declared logical group owns its invocation even when another
        // invocation (for example lexical injection) nests in its namespace.
        let named_group = parameters
            .groups()
            .iter()
            .any(|owned| owned.logical_name() == group.canonical_prefix);
        let mut matched = false;
        for owned in parameters.groups().iter().filter(|owned| {
            if named_group {
                owned.logical_name() == group.canonical_prefix
            } else {
                owned.members().iter().any(|member| {
                    member.target() == group.canonical_prefix
                        || member
                            .target()
                            .strip_prefix(&group.canonical_prefix)
                            .is_some_and(|suffix| suffix.starts_with('.'))
                })
            }
        }) {
            matched = true;
            if !matches!(
                owned.owner(),
                eredu_runtime::ParameterGroupOwner::ExecutionUnit { .. }
            ) || owner.is_some_and(|previous| previous != owned.owner())
            {
                return Err(invalid().into());
            }
            owner = Some(owned.owner());
        }
        if !matched {
            return Err(invalid().into());
        }
    }
    owner.ok_or_else(|| invalid().into())
}

fn insert_observation(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    path: &str,
    placement: PartitionedObservation,
) -> Result<(), ComponentPartitionError> {
    match observations.entry(path.into()) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(placement);
            Ok(())
        }
        std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &placement => Ok(()),
        _ => Err(ComponentPartitionError::DuplicateIdentity(path.into())),
    }
}

fn replicated_observation(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    path: &str,
    axis: &str,
    local: bool,
    site: ObservationHookSite,
) -> Result<(), ComponentPartitionError> {
    use eredu_core::{capture::CaptureError, SymbolicDimension};
    let point = descriptor
        .observations
        .points
        .iter()
        .find(|point| point.path == path)
        .ok_or_else(|| CaptureError::MissingPath(path.into()))?;
    let axes = point
        .axes
        .as_ref()
        .ok_or_else(|| CaptureError::Unsupported("replicated observation has no axes".into()))?;
    let mut matching = axes.iter().filter(|candidate| candidate.name == axis);
    let Some(eredu_core::TensorAxis {
        dimension: SymbolicDimension::Known(width),
        ..
    }) = matching.next()
    else {
        return Err(
            CaptureError::Invalid("replicated observation width is unresolved".into()).into(),
        );
    };
    if matching.next().is_some() {
        return Err(CaptureError::Invalid("replicated observation axis is repeated".into()).into());
    }
    let coordinates = local
        .then(|| ComponentCoordinateMap::range(*width, 0..*width))
        .transpose()?;
    let placement = PartitionedObservation {
        axis: axis.into(),
        coordinates,
        exports: local,
        site,
        combination: PartitionCaptureCombination::Disjoint,
    };
    insert_observation(observations, path, placement.clone())?;
    let effective = format!("{path}.effective");
    if descriptor
        .observations
        .points
        .iter()
        .any(|point| point.path == effective)
    {
        insert_observation(observations, &effective, placement)?;
    }
    Ok(())
}

/// Reusable, architecture-derived scalar layouts for one complete topology.
/// Compile once per retained execution and reuse across selections and forwards.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComponentPartitionLayouts {
    topology: eredu_core::ParallelTopology,
    layouts: Vec<ComponentPartitionLayout>,
}

impl ComponentPartitionLayouts {
    pub(crate) fn new(
        topology: eredu_core::ParallelTopology,
        layouts: Vec<ComponentPartitionLayout>,
    ) -> Result<Self, ComponentPartitionError> {
        if layouts.len() != topology.world_size()
            || layouts.iter().enumerate().any(|(rank, layout)| {
                layout.topology.global_rank() != rank || layout.topology.topology() != topology
            })
        {
            return Err(eredu_core::capture::CaptureError::Invalid(
                "component layouts do not match the complete rank topology".into(),
            )
            .into());
        }
        Ok(Self { topology, layouts })
    }

    /// Exact retained global topology.
    pub const fn topology(&self) -> eredu_core::ParallelTopology {
        self.topology
    }

    /// Component and invocation layout for a world rank.
    pub fn rank(&self, rank: usize) -> Option<&ComponentPartitionLayout> {
        self.layouts.get(rank)
    }

    /// Required execution site, agreed by every retained rank layout.
    pub fn observation_site(&self, path: &str) -> Option<ObservationHookSite> {
        let site = |layout: &ComponentPartitionLayout| {
            layout
                .observation(path)
                .map(PartitionedObservation::site)
                .or_else(|| {
                    layout
                        .routed_observation(path)
                        .map(|_| ObservationHookSite::RoutedUnits)
                })
        };
        let first = site(self.layouts.first()?)?;
        self.layouts
            .iter()
            .all(|layout| site(layout) == Some(first))
            .then_some(first)
    }

    /// Assembly equation agreed by every retained rank, including idle stages.
    pub fn observation_combination(
        &self,
        path: &str,
    ) -> Result<PartitionCaptureCombination, eredu_core::capture::CaptureError> {
        use eredu_core::capture::CaptureError;
        let mut combination = None;
        for layout in &self.layouts {
            let current = if let Some(point) = layout.observation(path) {
                point.combination()
            } else if layout.routed_observation(path).is_some() {
                PartitionCaptureCombination::Disjoint
            } else {
                return Err(CaptureError::MissingPath(path.into()));
            };
            if combination.is_some_and(|previous| previous != current) {
                return Err(CaptureError::Invalid(
                    "partition observation equations disagree".into(),
                ));
            }
            combination = Some(current);
        }
        let combination = combination.ok_or_else(|| CaptureError::MissingPath(path.into()))?;
        if combination == PartitionCaptureCombination::SumF64ToF32 {
            for layout in &self.layouts {
                let point = layout.observation(path).expect("agreed scalar equation");
                let Some(coordinates) = point.coordinates() else {
                    continue;
                };
                if coordinates.contiguous_range() != Some(0..coordinates.global_count()) {
                    return Err(CaptureError::Invalid(
                        "additive write has incomplete ordered coordinates".into(),
                    ));
                }
                for peer in layout
                    .topology
                    .tensor_parallel_peers()
                    .map_err(|error| CaptureError::Invalid(error.to_string()))?
                {
                    let other = self.layouts[peer]
                        .observation(path)
                        .expect("agreed scalar equation");
                    if other.coordinates() != Some(coordinates) || other.axis() != point.axis() {
                        return Err(CaptureError::Invalid(
                            "additive write lacks a complete tensor-parallel invocation group"
                                .into(),
                        ));
                    }
                }
            }
        }
        Ok(combination)
    }

    /// Combines semantic placement with hook coverage from the actual executor.
    /// Native collector checks remain separate and must follow this declaration.
    pub fn capture_hook_support(
        &self,
        path: &str,
        hooks: eredu_runtime::inspection::ObservationHookSupport,
    ) -> eredu_core::ObservationSupportStatus {
        let Some(site) = self.observation_site(path) else {
            return eredu_core::ObservationSupportStatus::Unverified(
                "global observation ownership is not yet implemented for this point".into(),
            );
        };
        if !hooks.supports(site) {
            return eredu_core::ObservationSupportStatus::Unverified(
                "selected architecture and executor do not declare the required internal observation hook".into(),
            );
        }
        eredu_core::ObservationSupportStatus::Supported
    }

    /// All ranks that execute a declared component observation, including
    /// replicas. Unknown observation axes have no component placement.
    pub fn capture_hook_members(&self, path: &str) -> Option<Vec<usize>> {
        self.observation_site(path)?;
        Some(
            self.layouts
                .iter()
                .enumerate()
                .filter_map(|(rank, layout)| {
                    let local = layout
                        .observation(path)
                        .is_some_and(|point| point.coordinates().is_some())
                        || layout
                            .routed_observation(path)
                            .is_some_and(|point| point.ownership().is_some());
                    local.then_some(rank)
                })
                .collect(),
        )
    }

    /// Compiles bounded expected producers for one global scalar selection.
    /// Identical disjoint coordinate maps are replicas: the lowest rank captures
    /// while every replica consumes interventions. Additive terms retain one
    /// producer per TP coordinate, eliminating only true replicas. Distinct shards
    /// with no selected overlap remain producers of explicit acknowledgments.
    /// Runtime receipt admission validates coverage and binds live work authority.
    pub fn capture_producers(
        &self,
        request: ComponentCaptureProjectionRequest<'_>,
    ) -> Result<
        Vec<eredu_runtime::capture::partition::PartitionCaptureProducer>,
        ComponentPartitionError,
    > {
        use eredu_core::capture::CaptureError;
        use eredu_runtime::capture::partition::PartitionCaptureProducer;
        if request.max_producers == 0 {
            return Err(CaptureError::Invalid("component producer bound is zero".into()).into());
        }
        let selection = request
            .plan
            .plan()
            .selections
            .get(request.selection_index)
            .ok_or_else(|| CaptureError::Invalid("unknown component capture selection".into()))?;
        let mut producers = Vec::new();
        let mut coordinates = Vec::new();
        let combination = self.observation_combination(&selection.path)?;
        let mut fragments = 0usize;
        for (rank, layout) in self.layouts.iter().enumerate() {
            let group = layout
                .observation(&selection.path)
                .ok_or_else(|| CaptureError::MissingPath(selection.path.clone()))?;
            let Some(map) = group.coordinates() else {
                continue;
            };
            if !group.exports() {
                continue;
            }
            let term = match combination {
                PartitionCaptureCombination::Disjoint => None,
                PartitionCaptureCombination::SumF64ToF32 => {
                    Some(layout.topology.tensor_parallel_rank())
                }
            };
            if coordinates.contains(&(map, term)) {
                continue;
            }
            if producers.len() == request.max_producers {
                return Err(CaptureError::Invalid(
                    "component producer count exceeds its bound".into(),
                )
                .into());
            }
            let projection = layout
                .project_capture_at(
                    request.plan,
                    request.selection_index,
                    request.phase,
                    request.prediction,
                    request.invocation,
                    request.max_fragments.saturating_sub(fragments),
                )?
                .expect("local component invocation");
            fragments = fragments
                .checked_add(projection.fragments().len())
                .ok_or(CaptureError::Overflow)?;
            if fragments > request.max_fragments {
                return Err(CaptureError::Invalid(
                    "component fragment count exceeds its bound".into(),
                )
                .into());
            }
            coordinates.push((map, term));
            producers.push(PartitionCaptureProducer { rank, projection });
        }
        if producers.is_empty() {
            return Err(CaptureError::Invalid(
                "component selection has no invocation owner".into(),
            )
            .into());
        }
        if combination == PartitionCaptureCombination::SumF64ToF32
            && producers.len() != self.topology.tensor()
        {
            return Err(CaptureError::Invalid(
                "additive capture lacks an authoritative producer for every TP term".into(),
            )
            .into());
        }
        Ok(producers)
    }
}

/// One original global selection whose producer geometry is being compiled.
/// This is cold descriptive input; it grants no native work or delivery authority.
pub struct ComponentCaptureProjectionRequest<'a> {
    /// Original admitted capture plan, shared across every rank.
    pub plan: &'a eredu_core::capture::AdmittedCapturePlan,
    /// Selection ordinal in that plan.
    pub selection_index: usize,
    /// Actual forward phase.
    pub phase: eredu_core::capture::CapturePhase,
    /// Prediction ordinal, distinct from positions selected within a tensor.
    pub prediction: u64,
    /// Independently admitted physical invocation axes; ordinary requests use None.
    pub invocation: Option<eredu_core::capture::CaptureInvocationShape>,
    /// Maximum retained distinct producer maps.
    pub max_producers: usize,
    /// Maximum native fragments summed across all chosen producers.
    pub max_fragments: usize,
}

impl eredu_runtime::capture::partition::PartitionCaptureLayout for ComponentPartitionLayouts {
    fn capture_combination(
        &self,
        path: &str,
    ) -> Result<PartitionCaptureCombination, eredu_core::capture::CaptureError> {
        self.observation_combination(path)
    }
    fn routed_capture_placement(
        &self,
        plan: &eredu_core::capture::AdmittedCapturePlan,
        index: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        limits: eredu_runtime::capture::partition::PartitionCaptureReceiptLimits,
    ) -> Result<
        eredu_runtime::capture::partition::PartitionRoutedCapturePlacement,
        eredu_core::capture::CaptureError,
    > {
        self.routed_capture_placement_at(plan, index, phase, prediction, None, limits)
    }

    fn routed_capture_placement_at(
        &self,
        plan: &eredu_core::capture::AdmittedCapturePlan,
        index: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        invocation: Option<eredu_core::capture::CaptureInvocationShape>,
        limits: eredu_runtime::capture::partition::PartitionCaptureReceiptLimits,
    ) -> Result<
        eredu_runtime::capture::partition::PartitionRoutedCapturePlacement,
        eredu_core::capture::CaptureError,
    > {
        self.routed_placement(plan, index, phase, prediction, invocation, limits)
    }

    fn capture_placement(
        &self,
        plan: &eredu_core::capture::AdmittedCapturePlan,
        index: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        limits: eredu_runtime::capture::partition::PartitionCaptureReceiptLimits,
    ) -> Result<
        eredu_runtime::capture::partition::PartitionCapturePlacement,
        eredu_core::capture::CaptureError,
    > {
        self.capture_placement_at(plan, index, phase, prediction, None, limits)
    }

    fn capture_placement_at(
        &self,
        plan: &eredu_core::capture::AdmittedCapturePlan,
        index: usize,
        phase: eredu_core::capture::CapturePhase,
        prediction: u64,
        invocation: Option<eredu_core::capture::CaptureInvocationShape>,
        limits: eredu_runtime::capture::partition::PartitionCaptureReceiptLimits,
    ) -> Result<
        eredu_runtime::capture::partition::PartitionCapturePlacement,
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::capture::CaptureError;
        let producers = self
            .capture_producers(ComponentCaptureProjectionRequest {
                invocation,
                plan,
                selection_index: index,
                phase,
                prediction,
                max_producers: limits.max_producers,
                max_fragments: limits.max_fragments,
            })
            .map_err(|error| match error {
                ComponentPartitionError::Capture(error) => error,
                error => CaptureError::Invalid(error.to_string()),
            })?;
        let path = &plan.plan().selections[index].path;
        let hook_members: Vec<_> = self
            .layouts
            .iter()
            .enumerate()
            .filter_map(|(rank, layout)| {
                layout
                    .observation(path)
                    .and_then(|group| group.coordinates())
                    .map(|_| rank)
            })
            .collect();
        let source_shapes = hook_members
            .iter()
            .map(|rank| {
                self.layouts[*rank]
                    .project_capture_at(
                        plan,
                        index,
                        phase,
                        prediction,
                        invocation,
                        limits.max_fragments,
                    )
                    .map(|projection| projection.expect("local invocation").local_shape().to_vec())
                    .map_err(|error| match error {
                        ComponentPartitionError::Capture(error) => error,
                        error => CaptureError::Invalid(error.to_string()),
                    })
            })
            .collect::<Result<_, _>>()?;
        Ok(
            eredu_runtime::capture::partition::PartitionCapturePlacement {
                producers,
                hook_members,
                source_shapes,
            },
        )
    }
}

/// Derives the actual scalar input axis of an architecture-declared write matrix.
/// Semantic units determine scalar offsets even when physical columns are packed.
/// Pipeline invocation ownership must be established separately by the caller.
pub fn derive_component_coordinates(
    component: &ComponentGroup,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    if let Some(stage) = &component.write_input_projection {
        return output_projection::component_coordinates(component.count, stage, tensor);
    }
    derive_write_coordinates(&component.write_weight, component.count, tensor)
}

fn derive_write_coordinates(
    name: &str,
    count: usize,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    derive_matrix_axis_coordinates(name, count, tensor, 1)
}

fn derive_matrix_axis_coordinates(
    name: &str,
    count: usize,
    tensor: &LocalTensorLayout,
    axis: usize,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    if tensor.global_shape().len() != 2 || axis >= 2 {
        return Err(ComponentPartitionError::InvalidPlacement(name.into()));
    }
    derive_tensor_axis_coordinates(name, count, tensor, axis)
}

fn derive_tensor_axis_coordinates(
    name: &str,
    count: usize,
    tensor: &LocalTensorLayout,
    axis: usize,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(name.into());
    let global = tensor.global_shape();
    let local = tensor.local_shape();
    if axis >= global.len()
        || global.len() != local.len()
        || global.contains(&0)
        || local.contains(&0)
        || global
            .iter()
            .zip(local)
            .enumerate()
            .any(|(index, (global, local))| index != axis && global != local)
        || !tensor.additional_placements().is_empty()
    {
        return Err(invalid());
    }
    let full = || ComponentCoordinateMap::range(count, 0..count).map_err(Into::into);
    match tensor.placement() {
        TensorPlacement::Replicated | TensorPlacement::Local if global == local => full(),
        TensorPlacement::Range {
            axis: selected_axis,
            start,
            end,
        } if *selected_axis == axis
            && start <= end
            && *end <= global[axis]
            && end - start == local[axis] =>
        {
            let units = tensor.logical_units().ok_or_else(invalid)?;
            let range = tensor.logical_range().ok_or_else(invalid)?;
            if let Some(chunk) = tensor.partition_chunk_size() {
                if chunk == 0 || global[axis].div_ceil(chunk) != units {
                    return Err(invalid());
                }
                let physical =
                    eredu_runtime::partition_chunk_range(global[axis], chunk, range.clone())
                        .map_err(|_| invalid())?;
                if physical != (*start..*end) {
                    return Err(invalid());
                }
                // FP8 retains one physical code per component. A short final
                // chunk therefore maps directly, without uniform-unit rounding.
                if global[axis] == count {
                    return ComponentCoordinateMap::range(count, physical).map_err(Into::into);
                }
                // Packed columns can use uniform semantic units only when
                // every physical chunk is complete. Partial packed encodings
                // need a separately declared scalar layout.
                if global[axis].is_multiple_of(chunk) {
                    return ComponentCoordinateMap::partition_units(count, units, range.clone())
                        .map_err(Into::into);
                }
                return Err(invalid());
            }
            // Verify the physical selection agrees with the retained semantic
            // units before expanding them to scalar component coordinates.
            if units == 0
                || !global[axis].is_multiple_of(units)
                || range.start > range.end
                || range.end > units
            {
                return Err(invalid());
            }
            let physical_per_unit = global[axis] / units;
            if *start != range.start * physical_per_unit || *end != range.end * physical_per_unit {
                return Err(invalid());
            }
            ComponentCoordinateMap::partition_units(count, units, range.clone()).map_err(Into::into)
        }
        TensorPlacement::Shard {
            axis: selected_axis,
            index,
            parts,
        } if *selected_axis == axis
            && *parts > 0
            && index < parts
            && global[axis].is_multiple_of(*parts)
            && local[axis] == global[axis] / parts =>
        {
            let units = tensor.logical_units().ok_or_else(invalid)?;
            let range = tensor.logical_range().ok_or_else(invalid)?;
            if units == 0
                || !units.is_multiple_of(*parts)
                || !global[axis].is_multiple_of(units)
                || range != &(index * (units / parts)..(index + 1) * (units / parts))
            {
                return Err(invalid());
            }
            ComponentCoordinateMap::partition_units(count, units, range.clone()).map_err(Into::into)
        }
        // An indexed physical axis is a scalar axis only when unpacked. Packed
        // layouts need an explicit semantic index map, not multiplication guesses.
        TensorPlacement::Indices {
            axis: selected_axis,
            indices,
        } if *selected_axis == axis && global[axis] == count && local[axis] == indices.len() => {
            ComponentCoordinateMap::indices(count, indices.clone()).map_err(Into::into)
        }
        _ => Err(invalid()),
    }
}

/// Component declarations and retained placement do not form an exact map.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ComponentPartitionError {
    /// The global capture admission cannot project onto this component layout.
    #[error(transparent)]
    Capture(#[from] eredu_core::capture::CaptureError),
    /// A logical invocation has no declared output projection.
    #[error(
        "component projection weight {0:?} is absent from the parameter description or layout"
    )]
    MissingWeight(String),
    /// A group or observation path is declared more than once.
    #[error("component identity {0:?} is repeated")]
    DuplicateIdentity(String),
    /// A physical selection cannot establish the requested scalar coordinates.
    #[error("projection weight {0:?} has no exact scalar component placement")]
    InvalidPlacement(String),
    /// The architecture's scalar axis or mask is malformed.
    #[error(transparent)]
    Coordinates(#[from] eredu_core::component::ComponentCoordinateError),
    /// Parallel layout could not be derived from the supplied parameter description.
    #[error("component parameter layout: {0}")]
    ParameterLayout(String),
}
