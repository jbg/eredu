//! Sparse invocation placement from retained bank policy and logical parameters.
use super::*;
use eredu_core::capture::{
    AdmittedCapturePlan, CaptureError, CapturePhase, CaptureSlicePartition,
    RoutedUnitCaptureOwnership, RoutedUnitGeometry,
};
use eredu_runtime::capture::partition::{
    PartitionCaptureReceiptLimits, PartitionRoutedCapturePlacement, PartitionRoutedCaptureProducer,
    PartitionRoutedCaptureSource,
};

/// A declared routed observation on one retained rank. Ownership is absent on
/// ranks that do not execute this invocation; stored shared weights grant none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedRoutedObservation {
    routing: String,
    effective: bool,
    geometry: RoutedUnitGeometry,
    input_width: u64,
    ownership: Option<RoutedUnitCaptureOwnership>,
}
impl PartitionedRoutedObservation {
    /// Canonical provider invocation, distinct from the selected observation path.
    pub fn routing(&self) -> &str {
        &self.routing
    }
    /// Whether this point describes effective values before down projection.
    pub const fn effective(&self) -> bool {
        self.effective
    }
    /// Exact global expert/route/scalar geometry.
    pub const fn geometry(&self) -> RoutedUnitGeometry {
        self.geometry
    }
    /// Provider read width, before internal native chunking or route expansion.
    pub const fn input_width(&self) -> u64 {
        self.input_width
    }
    /// Retained local coordinates and publication source, or no local invocation.
    /// These declarations grant no capture, native work or completion authority.
    pub fn ownership(&self) -> Option<&RoutedUnitCaptureOwnership> {
        self.ownership.as_ref()
    }
}

pub(in crate::component_partition) fn derive_observations(
    descriptor: &ArchitectureDescriptor,
    parameters: &ArchitectureParameterDescription,
    layout: &LocalModelLayout,
    topology: eredu_core::ParallelRankTopology,
    owned_groups: &[crate::partitioned_execution::PartitionedGroupRequirements],
    selected: &crate::SelectedRoutedTextRealization,
) -> Result<BTreeMap<String, PartitionedRoutedObservation>, ComponentPartitionError> {
    use crate::routed_text::RoutedGroupedPlan;
    let mut observations = BTreeMap::new();
    for component in &descriptor.routed_components {
        let invalid = || ComponentPartitionError::InvalidPlacement(component.id.clone());
        let node = descriptor.node(&component.node_id).ok_or_else(invalid)?;
        // RoutedExperts is a child operation whose parameter ownership is
        // explicitly inherited from its enclosing MixtureOfExperts invocation.
        let invocation = if node.parameter_groups.is_empty()
            && node.kind == eredu_core::ArchitectureNodeKind::RoutedExperts
        {
            let parent = node
                .parent
                .as_deref()
                .and_then(|id| descriptor.node(id))
                .ok_or_else(invalid)?;
            if parent.kind != eredu_core::ArchitectureNodeKind::MixtureOfExperts {
                return Err(invalid());
            }
            parent
        } else {
            node
        };
        let eredu_runtime::ParameterGroupOwner::ExecutionUnit {
            group, global_unit, ..
        } = super::super::node_invocation_owner(descriptor, parameters, &invocation.id)?
        else {
            return Err(invalid());
        };
        let Some(bank) = selected.bank(eredu_runtime::RoutedBankId::new(component.bank)) else {
            // Discovery may also describe an extension absent from this selection.
            // Its missing placement cannot authorize capture through this executor.
            continue;
        };
        if bank.owner_group() != group {
            continue;
        }
        let (cache_unit, distribution) = catalog_invocation(component, bank, group, *global_unit)?;
        let dimensions = match bank.plan() {
            RoutedGroupedPlan::Gated(plan) => {
                let spec = plan
                    .unit_spec(group.as_str(), cache_unit)
                    .ok_or_else(invalid)?;
                (
                    spec.group_count(),
                    spec.intermediate_dimensions(),
                    spec.input_dimensions(),
                    spec.output_dimensions(),
                )
            }
            RoutedGroupedPlan::Relu2(plan) => {
                let spec = plan
                    .unit_spec(group.as_str(), cache_unit)
                    .ok_or_else(invalid)?;
                (
                    spec.group_count(),
                    spec.intermediate_dimensions(),
                    spec.hidden_dimensions(),
                    spec.hidden_dimensions(),
                )
            }
            RoutedGroupedPlan::Linear(_) => return Err(invalid()),
        };
        if [dimensions.0, dimensions.1, dimensions.2, dimensions.3]
            .into_iter()
            .zip([
                component.expert_count,
                component.units_per_expert,
                component.input_width,
                component.output_width,
            ])
            .any(|(actual, expected)| usize::try_from(actual).ok() != Some(expected))
        {
            return Err(invalid());
        }
        let routes = bank
            .routes_by_unit()
            .get(&cache_unit)
            .copied()
            .ok_or_else(invalid)?;
        if component
            .routes_per_token
            .is_some_and(|declared| declared != routes)
        {
            return Err(invalid());
        }
        let geometry = RoutedUnitGeometry {
            experts: component.expert_count as u64,
            units_per_expert: component.units_per_expert as u64,
            routes_per_token: routes as u64,
        };
        geometry.components()?;
        let local = owned_groups
            .iter()
            .any(|owned| owned.group() == group && owned.units().contains(global_unit));
        let ownership = if local {
            let exchanged = distribution == crate::ExpertResidencyDistribution::ExpertParallel;
            let groups = if exchanged {
                if bank.plan().global_group_count() != component.expert_count {
                    return Err(invalid());
                }
                bank.plan()
                    .project_local_groups(topology)
                    .map_err(|error| ComponentPartitionError::ParameterLayout(error.to_string()))?
            } else {
                (0..component.expert_count).collect()
            };
            Some(RoutedUnitCaptureOwnership {
                coordinates: derive_coordinates_for_experts(component, layout, &groups)?,
                // The selected partition driver replicates each logical input on
                // the EP axis. Publish one source peer's routes, while budgeting
                // and validating all peers that contribute received native rows.
                source_peer: (exchanged && topology.expert_parallel_size() > 1).then_some(0),
                source_peers: if exchanged {
                    topology.expert_parallel_size() as u64
                } else {
                    1
                },
            })
        } else {
            None
        };
        for (path, effective) in [
            (&component.activation, false),
            (&component.effective_activation, true),
        ] {
            let point = descriptor
                .observations
                .points
                .iter()
                .find(|point| &point.path == path)
                .ok_or_else(invalid)?;
            if !matches!(&point.value_type, eredu_core::ObservationValueType::RoutedUnits { routing, geometry: declared } if routing == &component.routing && declared == &geometry)
                || point.position
                    != if effective {
                        eredu_core::ObservationPosition::AfterIntervention
                    } else {
                        eredu_core::ObservationPosition::BeforeIntervention
                    }
            {
                return Err(invalid());
            }
            if observations
                .insert(
                    path.clone(),
                    PartitionedRoutedObservation {
                        routing: component.routing.clone(),
                        effective,
                        geometry,
                        input_width: component.input_width as u64,
                        ownership: ownership.clone(),
                    },
                )
                .is_some()
            {
                return Err(ComponentPartitionError::DuplicateIdentity(path.clone()));
            }
        }
    }
    Ok(observations)
}

/// Join a logical component to its exact retained provider address. Multiple
/// invocations may belong to the same decoder unit and bank, with independent
/// cache keys and expert-axis distribution (for example an always-on branch).
fn catalog_invocation(
    component: &RoutedComponentGroup,
    bank: &crate::SelectedRoutedBank,
    owner: &eredu_runtime::ExecutionGroupId,
    owner_unit: usize,
) -> Result<(usize, crate::ExpertResidencyDistribution), ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(component.id.clone());
    let mut invocation = None;
    let mut members = std::collections::BTreeSet::new();
    for unit in bank
        .catalog()
        .units()
        .iter()
        .filter(|unit| unit.owner_group() == owner && unit.owner_unit() == owner_unit)
    {
        let key = unit.identity();
        let write = match &component.write_weight {
            RoutedComponentParameter::Packed { name } => Some(name),
            RoutedComponentParameter::Independent { names } => {
                names.get(key.member()).and_then(Option::as_ref)
            }
        };
        let Some(write) = write else { continue };
        if !unit
            .parameters()
            .iter()
            .any(|parameter| parameter.logical_target() == write.parameter)
        {
            continue;
        }
        let address = (key.unit(), unit.distribution());
        if u32::try_from(key.bank()).ok() != Some(component.bank)
            || key.member() >= component.expert_count
            || invocation.is_some_and(|previous| previous != address)
            || !members.insert(key.member())
        {
            return Err(invalid());
        }
        invocation = Some(address);
    }
    if members.len() != component.expert_count {
        return Err(invalid());
    }
    invocation.ok_or_else(invalid)
}

/// Prediction units carry resident banks on every execution replica. Their local
/// write layout still shards scalar columns on the tensor axis; no EP exchange
/// or target-bank policy participates in this invocation.
pub(in crate::component_partition) fn prediction_observations(
    descriptor: &ArchitectureDescriptor,
    components: &[eredu_core::component::RoutedComponentGroup],
    layout: &LocalModelLayout,
) -> Result<BTreeMap<String, PartitionedRoutedObservation>, ComponentPartitionError> {
    let mut observations = BTreeMap::new();
    for component in components {
        let invalid = || ComponentPartitionError::InvalidPlacement(component.id.clone());
        let ownership = RoutedUnitCaptureOwnership {
            coordinates: derive_coordinates_for_experts(
                component,
                layout,
                &(0..component.expert_count).collect::<Vec<_>>(),
            )?,
            source_peer: None,
            source_peers: 1,
        };
        for (path, effective) in [
            (&component.activation, false),
            (&component.effective_activation, true),
        ] {
            let point = descriptor
                .observations
                .points
                .iter()
                .find(|point| &point.path == path)
                .ok_or_else(invalid)?;
            let eredu_core::ObservationValueType::RoutedUnits { routing, geometry } =
                &point.value_type
            else {
                return Err(invalid());
            };
            if routing != &component.routing
                || geometry.experts != component.expert_count as u64
                || geometry.units_per_expert != component.units_per_expert as u64
                || point.position
                    != if effective {
                        eredu_core::ObservationPosition::AfterIntervention
                    } else {
                        eredu_core::ObservationPosition::BeforeIntervention
                    }
            {
                return Err(invalid());
            }
            geometry.components()?;
            if observations
                .insert(
                    path.clone(),
                    PartitionedRoutedObservation {
                        routing: routing.clone(),
                        geometry: *geometry,
                        effective,
                        input_width: component.input_width as u64,
                        ownership: Some(ownership.clone()),
                    },
                )
                .is_some()
            {
                return Err(ComponentPartitionError::DuplicateIdentity(path.clone()));
            }
        }
    }
    Ok(observations)
}

impl ComponentPartitionLayouts {
    pub(in crate::component_partition) fn routed_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<eredu_core::capture::CaptureInvocationShape>,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionRoutedCapturePlacement, CaptureError> {
        let invalid = |message: &str| CaptureError::Invalid(message.into());
        let selection = plan
            .plan()
            .selections
            .get(index)
            .ok_or_else(|| invalid("unknown routed capture selection"))?;
        if prediction >= plan.request().max_predictions
            || !selection.schedule.includes(phase, prediction)
        {
            return Err(invalid("routed capture is outside its admitted schedule"));
        }
        if limits.max_producers == 0 {
            return Err(invalid("routed producer bound is zero"));
        }
        let point = &plan.points()[index];
        let source = self.routed_capture_source(&selection.path)
            .map_err(|cause| cause.legacy(&selection.path))?;
        if !matches!(
            selection.transform,
            eredu_core::capture::CaptureTransform::RoutedUnits
        ) || point.position
            != if source.effective() {
                eredu_core::ObservationPosition::AfterIntervention
            } else {
                eredu_core::ObservationPosition::BeforeIntervention
            }
            || !matches!(&point.value_type, eredu_core::ObservationValueType::RoutedUnits { routing, geometry } if routing == source.routing() && geometry == &source.geometry())
        {
            return Err(invalid("routed capture differs from the retained bank"));
        }
        let shape = plan
            .geometry_at(phase, prediction, invocation)?
            .resolve(point)?
            .ok_or_else(|| {
                CaptureError::Unsupported("routed capture shape is unresolved".into())
            })?;
        if shape.len() != 3
            || shape[1..]
                != [
                    source.geometry().routes_per_token,
                    source.geometry().units_per_expert,
                ]
        {
            return Err(invalid("routed capture axes differ from the selected bank"));
        }
        let slice = eredu_core::capture::resolve_slice(point, selection, &shape)?;
        let mut sources = Vec::new();
        let mut producers = Vec::new();
        let mut fragments = 0usize;
        for rank in 0..source.world_size() {
            let Some(local) = source.rank(rank) else { continue; };
            let ownership = local.ownership;
            sources.push(PartitionRoutedCaptureSource {
                rank,
                ownership: ownership.clone(),
                input_width: source.input_width(),
            });
            if !local.produces { continue; }
            let map = &ownership.coordinates;
            if producers.len() == limits.max_producers {
                return Err(invalid("routed producer count exceeds its bound"));
            }
            let projection = CaptureSlicePartition::new(
                &shape,
                &slice,
                2,
                map.units(),
                limits.max_fragments.saturating_sub(fragments),
            )?;
            fragments = fragments
                .checked_add(projection.fragments().len())
                .ok_or(CaptureError::Overflow)?;
            if fragments > limits.max_fragments {
                return Err(invalid("routed fragment count exceeds its bound"));
            }
            producers.push(PartitionRoutedCaptureProducer {
                rank,
                projection,
                ownership: ownership.clone(),
            });
        }
        if producers.is_empty() {
            return Err(invalid("routed capture has no selected invocation owner"));
        }
        Ok(PartitionRoutedCapturePlacement {
            routing: source.routing().into(),
            effective: source.effective(),
            producers,
            sources,
        })
    }
}
