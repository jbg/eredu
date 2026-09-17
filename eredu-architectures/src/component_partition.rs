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

mod intervention_source;
pub use intervention_source::PartitionInterventionMemberLayout;
mod contiguous_capture;
pub use contiguous_capture::{ComponentPartitionCaptureSource, ComponentPartitionCaptureRank, ContiguousPartitionCaptureSource, ContiguousPartitionCaptureRank, PartitionCaptureSourceError};
mod decoder_invocations;
mod input_projection;
mod output_gate;
mod output_projection;
mod prediction;
mod routed;
mod routed_capture;
pub use routed_capture::{RoutedPartitionCaptureSource, RoutedPartitionCaptureRank, RoutedPartitionCaptureSourceError};
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

    /// The shared executor edits final publication only on its authoritative
    /// owner. Internal replicas remain independent intervention invocations.
    fn intervention_coordinates(&self) -> Option<&ComponentCoordinateMap> {
        if self.site == ObservationHookSite::Publication && !self.exports {
            None
        } else {
            self.coordinates()
        }
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
        let Some(source) = self.intervention_source(plan, operation, phase, prediction)? else {
            return Ok(None);
        };
        let point = &plan.points()[operation];
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
            source.axis(),
            source.coordinates(),
            max_regions,
        )?;
        let projection = match source.sum_offset_owner() {
            None => projection,
            Some(owner) => projection.as_sum_term(owner)?,
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

    pub(crate) fn from_admission<R>(
        descriptor: &ArchitectureDescriptor,
        parameters: &ArchitectureParameterDescription,
        layout: &LocalModelLayout,
        admission: &crate::partitioned_execution::PartitionedAdmission<R>,
        routed: Option<&crate::SelectedRoutedTextRealization>,
    ) -> Result<Self, ComponentPartitionError> {
        Self::from_ownership(
            descriptor,
            parameters,
            layout,
            admission.topology(),
            admission.ownership(),
            admission.groups(),
            admission.publication_owner(),
            routed,
        )
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
        owned_groups: &[crate::partitioned_execution::PartitionedGroupRequirements],
        publication_owner: usize,
        selected_routed: Option<&crate::SelectedRoutedTextRealization>,
    ) -> Result<Self, ComponentPartitionError> {
        let owns_parameter_invocation = |parameter: &str| {
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
        decoder_invocations::register(&mut observations, descriptor, parameters, |owner| {
            owner.is_owned_by(ownership, |group, unit| {
                owned_groups
                    .iter()
                    .any(|owned| owned.group() == group && owned.units().contains(&unit))
            })
        })?;
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
                        local,
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
            let Some(coordinates) = placement.intervention_coordinates() else {
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
        for (input, width) in component
            .input
            .iter()
            .map(|path| (path, component.input_width))
            .chain(
                component
                    .write_output
                    .iter()
                    .chain(component.output.iter())
                    .map(|path| (path, component.output_width)),
            )
        {
            let point = descriptor
                .observations
                .points
                .iter()
                .find(|point| point.path == *input)
                .ok_or_else(|| eredu_core::capture::CaptureError::MissingPath(input.clone()))?;
            if !point.axes.as_ref().is_some_and(|axes| {
                axes.iter().any(|axis| {
                    axis.name == "hidden"
                        && axis.dimension == eredu_core::SymbolicDimension::Known(width)
                })
            }) {
                return Err(ComponentPartitionError::InvalidPlacement(
                    component.id.clone(),
                ));
            }
            replicated_observation(
                observations,
                descriptor,
                input,
                "hidden",
                owns_node(&point.node_id)?,
                ObservationHookSite::Unit,
            )?;
        }
    }
    Ok(())
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
        let mut matched = false;
        for owned in parameters.groups().iter().filter(|owned| {
            owned.members().iter().any(|member| {
                member
                    .target()
                    .strip_prefix(&group.canonical_prefix)
                    .is_some_and(|suffix| suffix.starts_with('.'))
            })
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
    let placement = replicated_placement(descriptor, path, axis, local, site)?;
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

// Project exactly one declared point. Callers choose whether an explicit
// original/effective convention or exact node bindings establish other points.
fn replicated_placement(
    descriptor: &ArchitectureDescriptor,
    path: &str,
    axis: &str,
    local: bool,
    site: ObservationHookSite,
) -> Result<PartitionedObservation, ComponentPartitionError> {
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
    Ok(PartitionedObservation {
        axis: axis.into(),
        coordinates,
        exports: local,
        site,
        combination: PartitionCaptureCombination::Disjoint,
    })
}

/// A complete replicated observation, with one authoritative existing producer.
/// This describes architecture placement only; it grants no native transform,
/// transport or allocation permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletePartitionCaptureSource {
    producer: usize,
    hook_members: usize,
    world_size: usize,
    site: ObservationHookSite,
}
impl CompletePartitionCaptureSource {
    /// Lowest exporting world rank, following the ordinary replica rule.
    pub const fn producer(self) -> usize { self.producer }
    /// Whether every rank actually executes this complete observation hook.
    pub const fn has_world_hooks(self) -> bool { self.hook_members == self.world_size }
    /// Declared ordinary hook category, including explicit output publication.
    pub const fn site(self) -> ObservationHookSite { self.site }
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
        self.capture_combination_source(path).map_err(|error| error.legacy(path))
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

    /// Selects the same producer as `capture_producers` when every executing
    /// invocation has the complete ordered semantic axis. Shards, additive
    /// writes and sparse routed values require their other existing producers.
    /// This query borrows the actual retained table and allocates no metadata.
    pub fn complete_capture_source(&self, path: &str) -> Option<CompletePartitionCaptureSource> {
        let mut site = None;
        let mut producer = None;
        let mut width = None;
        let mut axis = None;
        let mut hook_members = 0usize;
        for (rank, layout) in self.layouts.iter().enumerate() {
            let point = layout.observation(path)?;
            if point.combination() != PartitionCaptureCombination::Disjoint
                || site.is_some_and(|prior| prior != point.site()) { return None; }
            site = Some(point.site());
            let Some(map) = point.coordinates() else {
                if point.exports() { return None; }
                continue;
            };
            if map.contiguous_range() != Some(0..map.global_count())
                || width.is_some_and(|prior| prior != map.global_count())
                || axis.is_some_and(|prior| prior != point.axis()) { return None; }
            width = Some(map.global_count()); axis = Some(point.axis());
            hook_members = hook_members.checked_add(1)?;
            if point.exports() && producer.is_none() { producer = Some(rank); }
        }
        Some(CompletePartitionCaptureSource { producer: producer?, hook_members,
            world_size: self.topology.world_size(), site: site? })
    }

    /// Fixed control values for one borrowed complete-source lookup. The
    /// retained layout table remains its existing owner's storage; this query
    /// creates no table, coordinate map, path string or allocation destination.
    pub fn complete_capture_source_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<(&Self, &str)>(),
            size_of::<(Option<ObservationHookSite>, Option<usize>, Option<usize>, Option<&str>, usize)>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, ComponentPartitionLayout>>>(),
            size_of::<Option<(usize, &ComponentPartitionLayout)>>(),
            size_of::<Option<&PartitionedObservation>>(),
            size_of::<Option<&ComponentCoordinateMap>>(),
            size_of::<(Option<std::ops::Range<usize>>, Option<std::ops::Range<usize>>)>(),
            size_of::<(Option<CompletePartitionCaptureSource>, CompletePartitionCaptureSource)>(),
            size_of::<(&ComponentPartitionLayout, &PartitionedObservation, &ComponentCoordinateMap, &str)>(),
            size_of::<(Option<usize>, bool, PartitionCaptureCombination, ObservationHookSite)>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
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
        let combination = self.observation_combination(&selection.path)?;
        let mut fragments = 0usize;
        for (rank, layout) in self.layouts.iter().enumerate() {
            let group = layout
                .observation(&selection.path)
                .ok_or_else(|| CaptureError::MissingPath(selection.path.clone()))?;
            if group.coordinates().is_none() {
                continue;
            }
            if !group.exports() {
                continue;
            }
            if !self.is_capture_producer(&selection.path, rank, combination) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ModelConfigurationResolver;

    #[test]
    fn routed_normalized_inputs_follow_invocation_owners_without_scalar_residuals() {
        let config = serde_json::json!({
            "model_type":"deepseek_v4", "hidden_size":8, "moe_intermediate_size":8,
            "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1,
            "head_dim":4, "qk_rope_head_dim":2, "q_lora_rank":4,
            "o_groups":2, "o_lora_rank":4, "vocab_size":16,
            "max_position_embeddings":64, "sliding_window":4, "compress_ratios":[0,4],
            "index_n_heads":2, "index_head_dim":4, "index_topk":1,
            "hc_mult":2, "hc_sinkhorn_iters":2, "n_routed_experts":2,
            "n_shared_experts":1, "num_experts_per_tok":1, "num_hash_layers":1,
            "num_nextn_predict_layers":0,
            "scoring_func":"sqrtsoftplus", "topk_method":"noaux_tc",
            "norm_topk_prob":true, "routed_scaling_factor":1.0, "swiglu_limit":4.0
        });
        let args = crate::deepseek::parse_v4_config(&config).unwrap();
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let parameters = crate::deepseek::parallel::v4_parameter_description(&args).unwrap();
        assert_eq!(descriptor.routed_components.len(), 2);
        for stage in 0..2 {
            let mut observations = BTreeMap::new();
            register_routed_boundaries(
                &mut observations,
                &descriptor,
                &descriptor.routed_components,
                |node| {
                    let owner = node_invocation_owner(&descriptor, &parameters, node)?;
                    let eredu_runtime::ParameterGroupOwner::ExecutionUnit { global_unit, .. } =
                        owner
                    else {
                        panic!("normalized routed input must have a unit owner")
                    };
                    Ok(*global_unit == stage)
                },
            )
            .unwrap();
            for component in &descriptor.routed_components {
                assert!(component.residual_scale.is_none());
                let input = component.input.as_ref().unwrap();
                let original = &observations[input];
                assert_eq!(original, &observations[&format!("{input}.effective")]);
                assert_eq!(original.site, ObservationHookSite::Unit);
                assert_eq!(original.exports, component.layer_index == stage);
                assert_eq!(
                    original
                        .coordinates
                        .as_ref()
                        .map(|map| map.contiguous_range()),
                    (component.layer_index == stage).then_some(Some(0..8)),
                );
            }
        }
        let mut invalid = descriptor.routed_components.clone();
        invalid[0].input = Some("readout.embedding".into());
        assert!(
            register_routed_boundaries(&mut BTreeMap::new(), &descriptor, &invalid, |node| {
                node_invocation_owner(&descriptor, &parameters, node).map(|_| true)
            },)
            .is_err()
        );
    }

    fn component() -> ComponentGroup {
        crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&serde_json::json!({
                "model_type":"llama", "hidden_size":8, "intermediate_size":256,
                "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2,
                "head_dim":2, "vocab_size":16, "rms_norm_eps":1e-5, "max_position_embeddings":64
            }))
            .unwrap()
            .architecture_plan()
            .architecture_descriptor()
            .components
            .into_iter()
            .find(|group| group.count == 256)
            .unwrap()
    }

    #[test]
    fn packed_write_columns_use_semantic_units_and_reject_inconsistent_placement() {
        let component = component();
        let layout = |placement, range| {
            LocalTensorLayout::new(
                "ffn",
                eredu_runtime::ParameterRole::FeedForwardIntermediate,
                vec![8, 32],
                vec![8, 8],
                placement,
                Some(4),
                Some(range),
                false,
            )
        };
        let exact = layout(
            TensorPlacement::Range {
                axis: 1,
                start: 8,
                end: 16,
            },
            1..2,
        );
        let map = derive_component_coordinates(&component, &exact).unwrap();
        assert_eq!(map.contiguous_range(), Some(64..128));
        assert_eq!(map.localize_indices(&[10, 67, 127, 150]).unwrap(), [3, 63]);
        let shard = layout(
            TensorPlacement::Shard {
                axis: 1,
                index: 1,
                parts: 4,
            },
            1..2,
        );
        assert_eq!(
            derive_component_coordinates(&component, &shard).unwrap(),
            map
        );
        let malformed_units = LocalTensorLayout::new(
            "ffn",
            eredu_runtime::ParameterRole::FeedForwardIntermediate,
            vec![8, 6],
            vec![8, 3],
            TensorPlacement::Shard {
                axis: 1,
                index: 0,
                parts: 2,
            },
            Some(4),
            Some(0..2),
            false,
        );
        assert!(derive_component_coordinates(&component, &malformed_units).is_err());
        for invalid in [
            layout(
                TensorPlacement::Range {
                    axis: 1,
                    start: 0,
                    end: 8,
                },
                1..2,
            ),
            layout(
                TensorPlacement::Range {
                    axis: 1,
                    start: 8,
                    end: 16,
                },
                1..5,
            ),
            layout(TensorPlacement::Omit, 1..2),
            exact.with_additional_placement(TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 4,
            }),
        ] {
            assert!(derive_component_coordinates(&component, &invalid).is_err());
        }
    }

    #[test]
    fn chunked_component_coordinates_retain_the_short_final_rank() {
        let layout = |start, end, range, chunk| {
            LocalTensorLayout::new(
                "ffn",
                eredu_runtime::ParameterRole::FeedForwardIntermediate,
                vec![130, 259],
                vec![130, end - start],
                TensorPlacement::Range {
                    axis: 1,
                    start,
                    end,
                },
                Some(3),
                Some(range),
                false,
            )
            .with_partition_chunk_size(Some(chunk))
        };
        let first = derive_write_coordinates("write", 259, &layout(0, 256, 0..2, 128)).unwrap();
        let last = derive_write_coordinates("write", 259, &layout(256, 259, 2..3, 128)).unwrap();
        assert_eq!(first.contiguous_range(), Some(0..256));
        assert_eq!(last.contiguous_range(), Some(256..259));
        assert_eq!(last.localize_indices(&[0, 128, 256, 258]).unwrap(), [0, 2]);
        assert!(derive_write_coordinates("write", 259, &layout(255, 259, 2..3, 128)).is_err());
        assert!(derive_write_coordinates("write", 259, &layout(256, 259, 1..2, 128)).is_err());
        assert!(derive_write_coordinates("write", 259, &layout(256, 259, 2..3, 0)).is_err());
    }

    #[test]
    fn indexed_unpacked_and_replicated_component_axes_preserve_identity() {
        let component = component();
        let layout = LocalTensorLayout::new(
            "ffn",
            eredu_runtime::ParameterRole::FeedForwardIntermediate,
            vec![8, 256],
            vec![8, 3],
            TensorPlacement::Indices {
                axis: 1,
                indices: vec![200, 2, 100],
            },
            None,
            None,
            false,
        );
        let map = derive_component_coordinates(&component, &layout).unwrap();
        assert_eq!(map.localize_indices(&[2, 100, 200]).unwrap(), [1, 2, 0]);
        let replicated = LocalTensorLayout::new(
            "ffn",
            eredu_runtime::ParameterRole::FeedForwardIntermediate,
            vec![8, 32],
            vec![8, 32],
            TensorPlacement::Replicated,
            Some(4),
            Some(1..2),
            true,
        );
        assert_eq!(
            derive_component_coordinates(&component, &replicated)
                .unwrap()
                .contiguous_range(),
            Some(0..256)
        );
    }

    #[test]
    fn replicated_producers_keep_distinct_idle_shards_and_admit_complete_receipts() {
        use eredu_core::{capture::*, *};
        use eredu_runtime::capture::partition::*;
        let topology = ParallelTopology::new(2, 1, 2, 1).unwrap();
        let layouts: Vec<_> = (0..topology.world_size())
            .map(|rank| {
                let rank = ParallelRankTopology::new(topology, rank).unwrap();
                let start = rank.tensor_parallel_rank() * 128;
                ComponentPartitionLayout {
                    topology: rank,
                    groups: BTreeMap::from([(
                        "units".into(),
                        PartitionedComponentGroup {
                            coordinates: Some(
                                ComponentCoordinateMap::range(256, start..start + 128).unwrap(),
                            ),
                        },
                    )]),
                    paths: BTreeMap::from([("units".into(), "units".into())]),
                    routed: BTreeMap::new(),
                    observations: BTreeMap::from([(
                        "units".into(),
                        PartitionedObservation {
                            axis: "component".into(),
                            coordinates: Some(
                                ComponentCoordinateMap::range(256, start..start + 128).unwrap(),
                            ),
                            exports: true,
                            site: ObservationHookSite::Unit,
                            combination: PartitionCaptureCombination::Disjoint,
                        },
                    )]),
                }
            })
            .collect();
        let mut swapped = layouts.clone();
        swapped.swap(0, 1);
        assert!(ComponentPartitionLayouts::new(topology, swapped).is_err());
        assert!(ComponentPartitionLayouts::new(topology, layouts[..3].to_vec()).is_err());
        let layouts = ComponentPartitionLayouts::new(topology, layouts).unwrap();
        assert!(layouts.rank(topology.world_size()).is_none());
        {
            use eredu_core::intervention::*;
            use eredu_runtime::intervention::PartitionActivationLayout;
            let plan = InterventionPlan {
                schema_version: 1,
                operations: vec![InterventionOperation {
                    id: "keep".into(),
                    target: "units".into(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![],
                    action: InterventionAction::MaskComponents {
                        dtype: InterventionDtype::Float32,
                        indices: vec![180, 7],
                        keep_selected: true,
                    },
                    evidence: InterventionEvidence::None,
                }],
            }
            .admit(
                &InterventionDiscovery {
                    schema_version: 1,
                    artifact_identity: "source".into(),
                    session_identity: Some("session".into()),
                    points: vec![InterventionPoint {
                        path: "units".into(),
                        node_id: "feed_forward".into(),
                        stage: InterventionStage::Activation,
                        axes: vec![
                            TensorAxis {
                                name: "sequence".into(),
                                dimension: SymbolicDimension::Sequence,
                            },
                            TensorAxis {
                                name: "component".into(),
                                dimension: SymbolicDimension::Known(256),
                            },
                        ],
                        dtypes: vec![InterventionDtype::Float32],
                        operations: vec![InterventionKind::MaskComponents],
                        score_stages: vec![],
                        prefill: ObservationSupportStatus::Supported,
                        decode: ObservationSupportStatus::Supported,
                        conditions: vec![],
                        routed_units: None,
                        routing: None,
                    }],
                },
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 2,
                    max_predictions: 3,
                },
                "run",
            )
            .unwrap();
            // Publication has one actual intervene callback despite real local
            // logits replicas on the other output-stage ranks.
            let mut publication = layouts.clone();
            for (rank, layout) in publication.layouts.iter_mut().enumerate() {
                let point = layout.observations.get_mut("units").unwrap();
                point.coordinates = Some(ComponentCoordinateMap::range(256, 0..256).unwrap());
                point.site = ObservationHookSite::Publication;
                point.exports = rank == 2;
            }
            let members = publication.activation_members(&plan, 0, CapturePhase::Prefill, 0, 4, 4).unwrap();
            assert_eq!(members.iter().map(|member| member.rank).collect::<Vec<_>>(), [2]);
            assert_eq!(members[0].projection.local_shape(), [2, 256]);
            assert_eq!(publication.activation_region_bound(&plan, 0, 4, 4).unwrap(), 1);
            assert!(publication.complete_capture_source("units").unwrap().has_world_hooks());
            for layout in &mut publication.layouts {
                layout.observations.get_mut("units").unwrap().site = ObservationHookSite::Unit;
            }
            assert_eq!(publication.activation_members(&plan, 0, CapturePhase::Prefill, 0, 4, 4).unwrap().len(), 4);
            let scoped = plan
                .plan()
                .clone()
                .admit_invocations(
                    &InterventionDiscovery {
                        schema_version: 1,
                        artifact_identity: "source".into(),
                        session_identity: Some("session".into()),
                        points: plan.points().to_vec(),
                    },
                    CaptureInvocationBounds {
                        batch: 1,
                        max_sequence: 5,
                        max_context: None,
                        max_predictions: 8,
                    },
                    "run",
                )
                .unwrap();
            for rows in [1, 3, 5] {
                let invocation = Some(CaptureInvocationShape {
                    batch: 1,
                    sequence: rows,
                    context: None,
                });
                let members = layouts
                    .activation_members_at(&scoped, 0, CapturePhase::Decode, 4, invocation, 4, 4)
                    .unwrap();
                assert_eq!(members.len(), 4);
                assert!(members
                    .iter()
                    .all(|member| member.projection.local_shape() == [rows, 128]));
            }
            assert!(layouts
                .activation_members(&scoped, 0, CapturePhase::Decode, 4, 4, 4)
                .is_err());
            let mut additive = layouts.clone();
            for layout in &mut additive.layouts {
                let point = layout.observations.get_mut("units").unwrap();
                point.coordinates = Some(ComponentCoordinateMap::range(256, 0..256).unwrap());
                point.combination = PartitionCaptureCombination::SumF64ToF32;
            }
            let mut add_plan = plan.plan().clone();
            add_plan.operations[0].action = InterventionAction::Add {
                tensor: InterventionTensor {
                    shape: vec![2, 256],
                    values: InterventionValues::Float32(vec![1.5; 512]),
                },
            };
            add_plan.operations[0].schedule.decode = false;
            let mut point = plan.points()[0].clone();
            point.operations = vec![InterventionKind::Add];
            let add_plan = add_plan
                .admit(
                    &InterventionDiscovery {
                        schema_version: 1,
                        artifact_identity: "source".into(),
                        session_identity: Some("session".into()),
                        points: vec![point],
                    },
                    plan.request(),
                    "run",
                )
                .unwrap();
            let members = additive
                .activation_members(&add_plan, 0, CapturePhase::Prefill, 0, 4, 4)
                .unwrap();
            assert_eq!(
                members.len(),
                4,
                "all TP and EP invocation members still participate"
            );
            for member in members {
                let owner = additive
                    .rank(member.rank)
                    .unwrap()
                    .topology()
                    .tensor_parallel_rank()
                    == 0;
                assert_eq!(
                    member.projection.region_count(),
                    usize::from(owner),
                    "one Add offset per true replica sum"
                );
            }
            additive.layouts[1]
                .observations
                .get_mut("units")
                .unwrap()
                .coordinates = None;
            assert!(additive
                .activation_members(&add_plan, 0, CapturePhase::Prefill, 0, 4, 4)
                .is_err());
            for (phase, prediction, rows) in
                [(CapturePhase::Prefill, 0, 2), (CapturePhase::Decode, 1, 1)]
            {
                let members = layouts
                    .activation_members(&plan, 0, phase, prediction, 4, 4)
                    .unwrap();
                assert_eq!(
                    members.iter().map(|member| member.rank).collect::<Vec<_>>(),
                    [0, 1, 2, 3]
                );
                for member in members {
                    assert_eq!(member.projection.local_shape(), [rows, 128]);
                    assert_eq!(member.projection.region_count(), 1);
                }
                assert!(layouts
                    .activation_members(&plan, 0, phase, prediction, 3, 4)
                    .is_err());
                assert!(layouts
                    .activation_members(&plan, 0, phase, prediction, 4, 3)
                    .is_err());
                let mut replicas = layouts.clone();
                for rank in &mut replicas.layouts {
                    rank.observations.get_mut("units").unwrap().exports = false;
                }
                assert_eq!(
                    replicas
                        .activation_members(&plan, 0, phase, prediction, 4, 4)
                        .unwrap()
                        .len(),
                    4,
                    "capture export selection cannot suppress interventions on replicas"
                );
                replicas.layouts[1]
                    .observations
                    .get_mut("units")
                    .unwrap()
                    .coordinates = None;
                assert_eq!(
                    replicas
                        .activation_members(&plan, 0, phase, prediction, 4, 4)
                        .unwrap()
                        .iter()
                        .map(|member| member.rank)
                        .collect::<Vec<_>>(),
                    [0, 2, 3]
                );
                assert!(replicas
                    .rank(1)
                    .unwrap()
                    .project_intervention(&plan, 0, phase, prediction, 4)
                    .unwrap()
                    .is_none());
                for rank in &mut replicas.layouts {
                    rank.observations.get_mut("units").unwrap().coordinates = None;
                }
                assert!(replicas
                    .activation_members(&plan, 0, phase, prediction, 4, 4)
                    .is_err());
            }
        }
        // Complete geometric ownership alone cannot enable a callback that the
        // selected unit executor does not invoke. Readout/publication facts must
        // not accidentally authorize internal unit capture.
        use eredu_core::ObservationSupportStatus;
        use eredu_runtime::inspection::ObservationHookSupport;
        let outer = ObservationHookSupport::internal(true, false, true).with_publication(true);
        assert!(matches!(
            layouts.capture_hook_support("units", outer),
            ObservationSupportStatus::Unverified(_)
        ));
        assert_eq!(
            layouts.capture_hook_support("units", outer.with_units(true)),
            ObservationSupportStatus::Supported
        );
        assert!(matches!(
            layouts.capture_hook_support("unknown", outer.with_units(true)),
            ObservationSupportStatus::Unverified(_)
        ));
        for empty in [false, true] {
            let point = ObservationPoint {
                path: "units".into(),
                node_id: "feed_forward".into(),
                meaning: "post-gating units".into(),
                value_type: ObservationValueType::Tensor,
                dtype: ObservationDtype::Floating,
                axes: Some(vec![
                    TensorAxis {
                        name: "sequence".into(),
                        dimension: SymbolicDimension::Sequence,
                    },
                    TensorAxis {
                        name: "component".into(),
                        dimension: SymbolicDimension::Known(256),
                    },
                ]),
                prefill: true,
                decode: true,
                requirements: vec![ObservationRequirement::ActivationHooks],
                position: ObservationPosition::BeforeIntervention,
                retained_bytes: None,
                host_bytes: None,
            };
            let capabilities = CaptureCapabilities {
                transformations: vec![CaptureTransformKind::Slice],
                max_histogram_bins: 0,
                physical_native_limit: false,
                conditions: vec![],
            };
            let support = ObservationSupportReport {
                schema_version: 1,
                capture: capabilities.clone(),
                points: vec![ObservationSupport {
                    path: "units".into(),
                    prefill: ObservationSupportStatus::Supported,
                    decode: ObservationSupportStatus::Supported,
                    floating_to_f32: true,
                }],
            };
            let usage = CaptureUsage {
                captures: 16,
                retained_bytes: 1_000_000,
                host_bytes: 1_000_000,
                encoded_bytes: 1_000_000,
            };
            let plan = CapturePlan {
                schema_version: 1,
                selections: vec![CaptureSelection {
                    id: "selected".into(),
                    path: "units".into(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![CaptureSlice {
                        axis: "component".into(),
                        start: 180,
                        end: if empty { 180 } else { 181 },
                        stride: 1,
                    }],
                    transform: CaptureTransform::Slice,
                }],
                limits: CaptureLimits {
                    per_step: usage,
                    cumulative: usage,
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            }
            .admit(
                &ObservationCatalog {
                    schema_version: 1,
                    points: vec![point],
                    completeness: DescriptionCompleteness::Complete,
                },
                &support,
                &capabilities,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 2,
                    max_predictions: 1,
                },
            )
            .unwrap();
            let scoped = plan
                .plan()
                .clone()
                .admit_invocations(
                    &ObservationCatalog {
                        schema_version: 1,
                        points: plan.points().to_vec(),
                        completeness: DescriptionCompleteness::Complete,
                    },
                    &support,
                    &capabilities,
                    CaptureInvocationBounds {
                        batch: 1,
                        max_sequence: 5,
                        max_context: Some(32),
                        max_predictions: 8,
                    },
                )
                .unwrap();
            let limits = PartitionCaptureReceiptLimits {
                max_producers: 2,
                max_fragments: usize::from(!empty),
                max_record_bytes: 8192,
            };
            for rows in [1, 3, 5] {
                let invocation = Some(CaptureInvocationShape {
                    batch: 1,
                    sequence: rows,
                    context: Some(19),
                });
                let projected = layouts
                    .capture_placement_at(&scoped, 0, CapturePhase::Decode, 4, invocation, limits)
                    .unwrap();
                assert_eq!(projected.hook_members, [0, 1, 2, 3]);
                assert!(projected
                    .source_shapes
                    .iter()
                    .all(|shape| shape == &[rows, 128]));
                assert!(projected
                    .producers
                    .iter()
                    .all(|producer| producer.projection.global_shape() == [rows, 256]));
                assert!(layouts
                    .capture_placement_at(&plan, 0, CapturePhase::Prefill, 0, invocation, limits)
                    .is_err());
            }
            assert!(layouts
                .capture_placement(&scoped, 0, CapturePhase::Decode, 4, limits)
                .is_err());
            let request = |max_producers, max_fragments| ComponentCaptureProjectionRequest {
                invocation: None,
                plan: &plan,
                selection_index: 0,
                phase: CapturePhase::Prefill,
                prediction: 0,
                max_producers,
                max_fragments,
            };
            let producers = layouts
                .capture_producers(request(2, usize::from(!empty)))
                .unwrap();
            let contiguous = layouts.contiguous_capture_source("units").unwrap();
            assert_eq!(contiguous.axis(), "component");
            assert_eq!(contiguous.width(), 256);
            assert_eq!(contiguous.producer_count(), producers.len());
            assert_eq!((0..contiguous.world_size()).filter_map(|rank| contiguous.rank(rank))
                .filter(|row| row.produces).map(|row| row.rank).collect::<Vec<_>>(),
                producers.iter().map(|row| row.rank).collect::<Vec<_>>());
            for rank in 0..contiguous.world_size() {
                assert_eq!(contiguous.rank(rank).unwrap().coordinates,
                    layouts.layouts[rank].observation("units").unwrap().coordinates().unwrap().contiguous_range().unwrap());
            }
            // Selection-empty sources remain explicit producers. Their actual
            // tensor source is not confused with an absent local invocation.
            assert_eq!(contiguous.producer_count(), 2);
            assert!(contiguous.rank(contiguous.world_size()).is_none());
            assert!(ContiguousPartitionCaptureSource::control_bytes().unwrap() > 0);
            let mut additive = layouts.clone();
            for layout in &mut additive.layouts {
                let point = layout.observations.get_mut("units").unwrap();
                point.coordinates = Some(ComponentCoordinateMap::range(256, 0..256).unwrap());
                point.combination = PartitionCaptureCombination::SumF64ToF32;
            }
            assert_eq!(
                additive.capture_combination("units").unwrap(),
                PartitionCaptureCombination::SumF64ToF32
            );
            let terms = additive
                .capture_producers(request(2, 2 * usize::from(!empty)))
                .unwrap();
            assert_eq!(
                terms.iter().map(|term| term.rank).collect::<Vec<_>>(),
                [0, 2],
                "retain every TP term but eliminate EP replicas"
            );
            assert!(terms
                .iter()
                .all(|term| term.projection.local_shape() == [2, 256]));
            let source = additive.contiguous_capture_source("units").unwrap();
            assert_eq!(source.combination(), PartitionCaptureCombination::SumF64ToF32);
            assert_eq!((0..source.world_size()).filter_map(|rank| source.rank(rank))
                .filter(|row| row.produces).map(|row| row.rank).collect::<Vec<_>>(), [0, 2]);
            assert!(additive.capture_producers(request(1, 2)).is_err());
            additive.layouts[1]
                .observations
                .get_mut("units")
                .unwrap()
                .coordinates = None;
            assert!(additive.capture_producers(request(2, 2)).is_err());
            assert_eq!(additive.contiguous_capture_source("units").unwrap_err(),
                PartitionCaptureSourceError::AdditiveGroup);
            additive.layouts[1]
                .observations
                .get_mut("units")
                .unwrap()
                .combination = PartitionCaptureCombination::Disjoint;
            assert!(additive.capture_combination("units").is_err());
            // An authoritative seam can belong to a later rank, independently
            // of which identical source map would win ordinary replica dedup.
            let mut authoritative = layouts.clone();
            for (rank, layout) in authoritative.layouts.iter_mut().enumerate() {
                let source = layout.observations.get_mut("units").unwrap();
                source.coordinates = Some(ComponentCoordinateMap::range(256, 0..256).unwrap());
                source.exports = rank == 3;
            }
            assert_eq!(
                authoritative.capture_hook_members("units").unwrap(),
                [0, 1, 2, 3]
            );
            let exported = authoritative
                .capture_producers(request(1, usize::from(!empty)))
                .unwrap();
            assert_eq!(exported.len(), 1);
            assert_eq!(exported[0].rank, 3);
            let source = authoritative.contiguous_capture_source("units").unwrap();
            assert_eq!(source.producer_count(), 1);
            assert!(source.rank(3).unwrap().produces);
            assert!(!source.rank(0).unwrap().produces);
            assert_eq!(source.rank(0).unwrap().coordinates, 0..256);
            let placement = layouts
                .capture_placement(
                    &plan,
                    0,
                    CapturePhase::Prefill,
                    0,
                    PartitionCaptureReceiptLimits {
                        max_producers: 2,
                        max_fragments: usize::from(!empty),
                        max_record_bytes: 8192,
                    },
                )
                .unwrap();
            assert_eq!(
                placement.hook_members,
                [0, 1, 2, 3],
                "replicas must still agree source failures"
            );
            assert_eq!(
                placement
                    .producers
                    .iter()
                    .map(|p| p.rank)
                    .collect::<Vec<_>>(),
                producers.iter().map(|p| p.rank).collect::<Vec<_>>()
            );
            assert_eq!(
                producers.len(),
                2,
                "EP replicas must not duplicate receipts"
            );
            for producer in &producers {
                let map = layouts
                    .rank(producer.rank)
                    .unwrap()
                    .point("units")
                    .unwrap()
                    .coordinates()
                    .unwrap();
                assert!(
                    (0..producer.rank).all(|rank| layouts
                        .rank(rank)
                        .unwrap()
                        .point("units")
                        .unwrap()
                        .coordinates()
                        .unwrap()
                        != map),
                    "choose the lowest rank for each exact coordinate map"
                );
            }
            assert_eq!(
                producers
                    .iter()
                    .filter(|p| p.projection.fragments().is_empty())
                    .count(),
                if empty { 2 } else { 1 }
            );
            assert!(layouts.capture_producers(request(1, 16)).is_err());
            if !empty {
                assert!(layouts.capture_producers(request(2, 0)).is_err());
            }
            let mut ledger = CaptureLedger::new(&plan);
            let context = PartitionCaptureContext {
                invocation: None,
                artifact_identity: "source".into(),
                execution_identity: "selected-world".into(),
                run_identity: "run".into(),
                overlay_identity: None,
                capture_plan_identity: plan.identity().into(),
                selection_index: 0,
                phase: CapturePhase::Prefill,
                prediction: 0,
                forward_epoch: 1,
            };
            PartitionCaptureReceiptPlan::new(
                plan,
                context,
                producers,
                topology.world_size(),
                PartitionCaptureReceiptLimits {
                    max_producers: 2,
                    max_fragments: 1,
                    max_record_bytes: 4096,
                },
                &mut ledger,
            )
            .unwrap();
        }
    }
}
