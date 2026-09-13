//! Coordinates retained by the selected provider, before native construction.
use super::*;
use crate::routed_text::RoutedGroupedPlan;

enum Writes<'a> {
    Packed(&'a str),
    Independent(Vec<&'a str>),
}
struct Geometry<'a> {
    experts: usize,
    units: usize,
    output: usize,
    writes: Writes<'a>,
}
impl<'a> Geometry<'a> {
    fn gated(spec: &'a eredu_nn::GroupedGatedProductSpec) -> Result<Self, ComponentPartitionError> {
        Ok(Self {
            experts: spec.group_count() as usize,
            units: spec.intermediate_dimensions() as usize,
            output: spec.output_dimensions() as usize,
            writes: match spec.layout() {
                eredu_nn::GatedProductGroupLayout::Packed { down, .. } => {
                    Writes::Packed(down.weight().id.as_str())
                }
                eredu_nn::GatedProductGroupLayout::Independent(groups) => Writes::Independent(
                    groups
                        .iter()
                        .map(|group| group.down().weight().id.as_str())
                        .collect(),
                ),
                _ => {
                    return Err(ComponentPartitionError::InvalidPlacement(
                        "unknown grouped write layout".into(),
                    ))
                }
            },
        })
    }
    fn relu2(spec: &'a eredu_nn::GroupedRelu2Spec) -> Self {
        Self {
            experts: spec.group_count() as usize,
            units: spec.intermediate_dimensions() as usize,
            output: spec.hidden_dimensions() as usize,
            writes: Writes::Packed(spec.down().weight().id.as_str()),
        }
    }
}

/// Joins the original selected equation with its localized construction and
/// exact physical layouts. No backend tensor or discovery-name parsing is used.
pub(crate) fn derive_bank_unit_coordinates(
    global: &RoutedGroupedPlan,
    local: &RoutedGroupedPlan,
    catalog: &crate::ExpertResidencyCatalog,
    layout: &LocalModelLayout,
) -> Result<BTreeMap<usize, RoutedComponentCoordinateMap>, ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement("selected routed bank".into());
    if global.global_group_count() != local.global_group_count() {
        return Err(invalid());
    }
    let routed_experts = ComponentCoordinateMap::indices(
        local.global_group_count(),
        local.local_global_group_indices().to_vec(),
    )?;
    let mut result = BTreeMap::new();
    let mut insert = |owner: &eredu_runtime::ExecutionGroupId,
                      unit,
                      global: Geometry<'_>,
                      local: Geometry<'_>| {
        // Equal-sized shared and routed banks still have different execution
        // ownership. The retained catalog, not a dimension comparison, decides.
        let mut distribution = None;
        let mut members = std::collections::BTreeSet::new();
        for entry in catalog
            .units()
            .iter()
            .filter(|entry| entry.owner_group() == owner && entry.identity().unit() == unit)
        {
            if distribution.is_some_and(|previous| previous != entry.distribution())
                || entry.identity().member() >= global.experts
                || !members.insert(entry.identity().member())
            {
                return Err(invalid());
            }
            distribution = Some(entry.distribution());
        }
        if members.len() != global.experts {
            return Err(invalid());
        }
        let experts = match distribution.ok_or_else(invalid)? {
            crate::ExpertResidencyDistribution::ExpertParallel
                if global.experts == routed_experts.global_count() =>
            {
                routed_experts.clone()
            }
            crate::ExpertResidencyDistribution::Replicated => {
                ComponentCoordinateMap::range(global.experts, 0..global.experts)?
            }
            _ => return Err(invalid()),
        };
        if global.units == 0
            || global.output != local.output
            || (experts.local_count() > 0 && local.experts != experts.local_count())
        {
            return Err(invalid());
        }
        let units = if experts.local_count() == 0 {
            ComponentCoordinateMap::range(global.units, 0..0)?
        } else {
            let units = match global.writes {
                Writes::Packed(name) => {
                    let tensor = layout
                        .tensor(name)
                        .ok_or_else(|| ComponentPartitionError::MissingWeight(name.into()))?;
                    let (stored, units) = packed_coordinates(
                        name,
                        global.experts,
                        global.units,
                        global.output,
                        tensor,
                    )?;
                    if local_experts_missing(&experts, &stored) {
                        return Err(invalid());
                    }
                    units
                }
                Writes::Independent(names) => {
                    if names.len() != global.experts {
                        return Err(invalid());
                    }
                    let mut units = None;
                    for expert in local_expert_indices(&experts) {
                        let name = names[expert];
                        let tensor = layout
                            .tensor(name)
                            .ok_or_else(|| ComponentPartitionError::MissingWeight(name.into()))?;
                        if tensor.global_shape().first() != Some(&global.output) {
                            return Err(invalid());
                        }
                        let next = derive_write_coordinates(name, global.units, tensor)?;
                        if units.as_ref().is_some_and(|units| *units != next) {
                            return Err(invalid());
                        }
                        units = Some(next);
                    }
                    units.ok_or_else(invalid)?
                }
            };
            if units.local_count() != local.units {
                return Err(invalid());
            }
            units
        };
        if result
            .insert(
                unit,
                RoutedComponentCoordinateMap::new(experts.clone(), units),
            )
            .is_some()
        {
            return Err(invalid());
        }
        Ok(())
    };
    match (global, local) {
        (RoutedGroupedPlan::Gated(global), RoutedGroupedPlan::Gated(local)) => {
            for ((owner, unit), spec) in local.unit_specs() {
                let original = global
                    .unit_spec(owner.as_str(), *unit)
                    .ok_or_else(invalid)?;
                insert(
                    owner,
                    *unit,
                    Geometry::gated(original)?,
                    Geometry::gated(spec)?,
                )?;
            }
        }
        (RoutedGroupedPlan::Relu2(global), RoutedGroupedPlan::Relu2(local)) => {
            for ((owner, unit), spec) in local.unit_specs() {
                let original = global
                    .unit_spec(owner.as_str(), *unit)
                    .ok_or_else(invalid)?;
                insert(
                    owner,
                    *unit,
                    Geometry::relu2(original),
                    Geometry::relu2(spec),
                )?;
            }
        }
        // Selected-linear outputs are not hidden FFN units.
        (RoutedGroupedPlan::Linear(_), RoutedGroupedPlan::Linear(_)) => {}
        _ => return Err(invalid()),
    }
    Ok(result)
}

fn local_expert_indices(map: &ComponentCoordinateMap) -> impl Iterator<Item = usize> + '_ {
    (0..map.local_count()).map(|index| map.local_to_global(index).expect("checked coordinate map"))
}
fn local_experts_missing(
    selected: &ComponentCoordinateMap,
    stored: &ComponentCoordinateMap,
) -> bool {
    local_expert_indices(selected).any(|expert| stored.global_to_local(expert).is_none())
}
