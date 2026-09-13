//! Sparse expert-unit identities and effective parameter coordinates.
use super::{
    ComponentActivation, ComponentNormalization, ComponentReadRole, ComponentRowMapping,
    ComponentScalar,
};
use crate::parameters::{LoadedParameter, ParameterDiscovery, ParameterRegion};
use serde::{Deserialize, Serialize};

/// Independent global expert and within-expert unit coordinates for a local
/// bank. This describes a Cartesian placement, never participation or admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedComponentCoordinateMap {
    experts: super::ComponentCoordinateMap,
    units: super::ComponentCoordinateMap,
}
impl RoutedComponentCoordinateMap {
    /// Combines already checked expert and scalar-unit axes. Both maps retain
    /// local order, including explicit permutations and empty local selections.
    pub fn new(
        experts: super::ComponentCoordinateMap,
        units: super::ComponentCoordinateMap,
    ) -> Self {
        Self { experts, units }
    }
    /// Local expert order in the checkpoint-global expert namespace.
    pub fn experts(&self) -> &super::ComponentCoordinateMap {
        &self.experts
    }
    /// Local scalar order within each checkpoint-global expert.
    pub fn units(&self) -> &super::ComponentCoordinateMap {
        &self.units
    }
    /// Resolves a local expert/unit pair without expanding the bank product.
    pub fn local_to_global(&self, expert: usize, unit: usize) -> Option<(usize, usize)> {
        Some((
            self.experts.local_to_global(expert)?,
            self.units.local_to_global(unit)?,
        ))
    }
    /// Resolves a global expert/unit pair, when both are present on this rank.
    pub fn global_to_local(&self, expert: usize, unit: usize) -> Option<(usize, usize)> {
        Some((
            self.experts.global_to_local(expert)?,
            self.units.global_to_local(unit)?,
        ))
    }
}

/// One unit in one checkpoint-global expert at one logical invocation.
/// Route slots and tokens describe participation, and are not part of this identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RoutedComponentId {
    /// Stable architecture-declared routed component group.
    pub group: String,
    /// Checkpoint-global expert ordinal, never a compact provider index.
    pub expert: usize,
    /// Zero-based unit within that expert.
    pub index: usize,
}

/// Architecture parameter identity, distinct from loaded operation support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedComponentParameterName {
    /// Canonical effective parameter slot.
    pub parameter: String,
    /// Architecture group owning this slot.
    pub parameter_group: String,
    /// Declared shared source. Loaded discovery supplies actual alias authority.
    pub shared_parameter: String,
}

/// Exact effective expert-axis layout, independent of the checkpoint encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoutedComponentParameter {
    /// Leading expert axis, followed by matrix or bias-vector dimensions.
    Packed {
        /// One effective bank parameter.
        name: RoutedComponentParameterName,
    },
    /// Separately named parameters in global expert order. A missing entry is
    /// permitted for an absent per-expert bias, never for a required matrix.
    Independent {
        /// Exact expert-indexed slots, without inferred naming rules.
        names: Vec<Option<RoutedComponentParameterName>>,
    },
}

/// A read branch within each expert; mappings address per-expert projection rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedComponentRead {
    /// Input, activated gate, or multiplicative value branch.
    pub role: ComponentReadRole,
    /// Exact matrix slot(s).
    pub weight: RoutedComponentParameter,
    /// Optional affine bias slot(s); quantization companions are not affine biases.
    pub bias: Option<RoutedComponentParameter>,
    /// Number of logical output rows, including other segments of a fused matrix.
    pub projection_rows: usize,
    /// Unit-to-row map, including fused gate/up offsets.
    pub rows: ComponentRowMapping,
}

/// Compact declaration of units in a routed bank. Only selected routes produce
/// activations; inactive experts have no measured row. This is topology, not a
/// claim of loaded capture, intervention, parameter, or distributed support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedComponentGroup {
    /// Stable group identity; invocations sharing weights remain distinct.
    pub id: String,
    /// Owning logical routed-expert node.
    pub node_id: String,
    /// Logical layer/invocation ordinal.
    pub layer_index: usize,
    /// Architecture-declared bank ordinal within the invocation.
    pub bank: u32,
    /// Number of global experts.
    pub expert_count: usize,
    /// Exact route count per token for this invocation, when explicitly declared.
    /// Always-on grouped branches can differ from their parent's routed top-k.
    /// Older declarations inherit the enclosing routing node's cardinality.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routes_per_token: Option<usize>,
    /// Scalar units within each expert.
    pub units_per_expert: usize,
    /// Width consumed by expert read matrices.
    pub input_width: usize,
    /// Width produced by expert down projections.
    pub output_width: usize,
    /// Canonical routed invocation path used by the unit observer.
    pub routing: String,
    /// Original selected-unit boundary, before down projection and route weighting.
    pub activation: String,
    /// Effective selected units before optional projection input quantization.
    pub effective_activation: String,
    /// Actual transformed down-projection input, when separately declared.
    pub write_input: Option<String>,
    /// Complete route-weighted bank write after expert and tensor reductions,
    /// before a separately declared output transform. This excludes other banks
    /// or shared branches. Its `.effective` companion is the consumed write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_output: Option<String>,
    /// Bank output after declared transforms, before combining with other writes.
    /// Join this boundary to `ComponentTensorTransform` for the exact equation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Normalized input boundary, when declared by the architecture.
    pub input: Option<String>,
    /// Read branches, including both gate and value for a gated bank.
    pub reads: Vec<RoutedComponentRead>,
    /// Effective down projection(s), logically `[expert, output, unit]` when packed.
    pub write_weight: RoutedComponentParameter,
    /// Separate per-expert additive output bias, weighted by each route coefficient.
    pub write_bias: Option<RoutedComponentParameter>,
    /// Exact activation, clamps and offsets applied before down projection.
    pub activation_equation: ComponentActivation,
    /// Exact normalization preceding the expert reads, if described.
    pub input_normalization: Option<ComponentNormalization>,
    /// Scalar applied after route-weighted expert reduction at residual addition.
    /// Absent when that residual transformation has not been described as scalar.
    pub residual_scale: Option<ComponentScalar>,
}

/// A checked region joined to the actual loaded slot. It grants no execution
/// authority: submit it through the ordinary admitted parameter query/edit APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedComponentSelection<'a> {
    /// Actual loaded shape, dtype, alias identity and operation support.
    pub parameter: &'a LoadedParameter,
    /// Exact unsqueezed region in that slot, suitable for queries or overlays.
    pub region: ParameterRegion,
}

impl RoutedComponentGroup {
    /// Checked global component index used by compact sparse activation masks.
    /// This identity is independent of token participation and native route order.
    pub fn component_index(&self, id: &RoutedComponentId) -> Result<u32, RoutedComponentError> {
        self.validate_id(id)?;
        id.expert
            .checked_mul(self.units_per_expert)
            .and_then(|n| n.checked_add(id.index))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(RoutedComponentError::Geometry)
    }
    /// Validates invocation, expert and unit identity independently of participation.
    pub fn contains(&self, id: &RoutedComponentId) -> bool {
        id.group == self.id && id.expert < self.expert_count && id.index < self.units_per_expert
    }

    fn validate_id(&self, id: &RoutedComponentId) -> Result<(), RoutedComponentError> {
        if !self.contains(id) || self.input_width == 0 || self.output_width == 0 {
            return Err(RoutedComponentError::Identity);
        }
        Ok(())
    }

    /// Joins one read row (or declared row range) to loaded effective geometry.
    pub fn read_weight<'a>(
        &self,
        id: &RoutedComponentId,
        role: ComponentReadRole,
        loaded: &'a ParameterDiscovery,
    ) -> Result<RoutedComponentSelection<'a>, RoutedComponentError> {
        self.validate_id(id)?;
        let read = self.read(role)?;
        let rows = read
            .rows
            .row_range(id.index)
            .ok_or(RoutedComponentError::Geometry)?;
        self.select(
            &read.weight,
            id.expert,
            &[read.projection_rows, self.input_width],
            &[rows.start, 0],
            &[rows.len(), self.input_width],
            loaded,
        )?
        .ok_or(RoutedComponentError::Geometry)
    }

    /// Resolves an optional read bias; absence is distinct from missing loaded data.
    pub fn read_bias<'a>(
        &self,
        id: &RoutedComponentId,
        role: ComponentReadRole,
        loaded: &'a ParameterDiscovery,
    ) -> Result<Option<RoutedComponentSelection<'a>>, RoutedComponentError> {
        self.validate_id(id)?;
        let read = self.read(role)?;
        let Some(bias) = &read.bias else {
            return Ok(None);
        };
        let rows = read
            .rows
            .row_range(id.index)
            .ok_or(RoutedComponentError::Geometry)?;
        self.select(
            bias,
            id.expert,
            &[read.projection_rows],
            &[rows.start],
            &[rows.len()],
            loaded,
        )
    }

    /// Resolves the selected expert's down-projection column.
    pub fn write_column<'a>(
        &self,
        id: &RoutedComponentId,
        loaded: &'a ParameterDiscovery,
    ) -> Result<RoutedComponentSelection<'a>, RoutedComponentError> {
        self.validate_id(id)?;
        self.select(
            &self.write_weight,
            id.expert,
            &[self.output_width, self.units_per_expert],
            &[0, id.index],
            &[self.output_width, 1],
            loaded,
        )?
        .ok_or(RoutedComponentError::Geometry)
    }

    /// Resolves a whole expert output bias. This is one term per selected route,
    /// not one term per unit, and must be multiplied by that route's coefficient.
    pub fn write_bias<'a>(
        &self,
        id: &RoutedComponentId,
        loaded: &'a ParameterDiscovery,
    ) -> Result<Option<RoutedComponentSelection<'a>>, RoutedComponentError> {
        self.validate_id(id)?;
        match &self.write_bias {
            Some(bias) => self.select(
                bias,
                id.expert,
                &[self.output_width],
                &[0],
                &[self.output_width],
                loaded,
            ),
            None => Ok(None),
        }
    }

    fn read(&self, role: ComponentReadRole) -> Result<&RoutedComponentRead, RoutedComponentError> {
        let mut matches = self.reads.iter().filter(|read| read.role == role);
        let read = matches.next().ok_or(RoutedComponentError::ReadRole)?;
        if matches.next().is_some() {
            return Err(RoutedComponentError::Geometry);
        }
        Ok(read)
    }

    fn select<'a>(
        &self,
        mapping: &RoutedComponentParameter,
        expert: usize,
        shape: &[usize],
        starts: &[usize],
        extents: &[usize],
        loaded: &'a ParameterDiscovery,
    ) -> Result<Option<RoutedComponentSelection<'a>>, RoutedComponentError> {
        select_parameter(
            self.expert_count,
            mapping,
            expert,
            shape,
            starts,
            extents,
            loaded,
        )
    }
}

pub(super) fn select_parameter<'a>(
    expert_count: usize,
    mapping: &RoutedComponentParameter,
    expert: usize,
    shape: &[usize],
    starts: &[usize],
    extents: &[usize],
    loaded: &'a ParameterDiscovery,
) -> Result<Option<RoutedComponentSelection<'a>>, RoutedComponentError> {
    if expert >= expert_count {
        return Err(RoutedComponentError::Geometry);
    }
    let (name, packed) = match mapping {
        RoutedComponentParameter::Packed { name } => (name, true),
        RoutedComponentParameter::Independent { names } => {
            if names.len() != expert_count {
                return Err(RoutedComponentError::Geometry);
            }
            let Some(name) = names.get(expert).and_then(Option::as_ref) else {
                return Ok(None);
            };
            (name, false)
        }
    };
    let convert = |values: &[usize]| {
        values
            .iter()
            .map(|&v| u64::try_from(v).map_err(|_| RoutedComponentError::Geometry))
            .collect::<Result<Vec<_>, _>>()
    };
    let mut expected = convert(shape)?;
    let mut region = ParameterRegion {
        starts: convert(starts)?,
        shape: convert(extents)?,
    };
    if packed {
        expected.insert(
            0,
            u64::try_from(expert_count).map_err(|_| RoutedComponentError::Geometry)?,
        );
        region.starts.insert(
            0,
            u64::try_from(expert).map_err(|_| RoutedComponentError::Geometry)?,
        );
        region.shape.insert(0, 1);
    }
    let mut parameters = loaded.parameters.iter().filter(|p| p.id == name.parameter);
    let parameter = parameters
        .next()
        .ok_or_else(|| RoutedComponentError::MissingParameter(name.parameter.clone()))?;
    if parameters.next().is_some()
        || parameter.shape != expected
        || region.validate(&expected).is_err()
    {
        return Err(RoutedComponentError::Geometry);
    }
    Ok(Some(RoutedComponentSelection { parameter, region }))
}

/// Invalid topology join; never represents a measured zero or admitted operation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RoutedComponentError {
    /// Invocation, expert or unit is outside the declared group.
    #[error("invalid routed component identity")]
    Identity,
    /// A requested read branch is not part of this component equation.
    #[error("routed component has no requested read role")]
    ReadRole,
    /// Malformed/overflowed mapping or incompatible actual loaded geometry.
    #[error("routed component parameter geometry differs from the declaration")]
    Geometry,
    /// The declared effective slot is absent from actual loaded discovery.
    #[error("routed component parameter {0:?} is absent from loaded discovery")]
    MissingParameter(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        capture::CaptureUsage, component::ComponentNonlinearity, intervention::InterventionDtype,
        parameters::ProjectionInputTransform,
    };

    fn mapping(parameter: &str) -> RoutedComponentParameter {
        RoutedComponentParameter::Packed {
            name: RoutedComponentParameterName {
                parameter: parameter.into(),
                shared_parameter: parameter.into(),
                parameter_group: "bank".into(),
            },
        }
    }
    fn fixture() -> (RoutedComponentGroup, ParameterDiscovery, RoutedComponentId) {
        let group = RoutedComponentGroup {
            routes_per_token: None,
            id: "layer.bank.units".into(),
            node_id: "layer.bank".into(),
            layer_index: 4,
            bank: 2,
            expert_count: 3,
            units_per_expert: 5,
            input_width: 4,
            output_width: 6,
            routing: "layer.routing".into(),
            activation: "layer.routing.units".into(),
            effective_activation: "layer.routing.units.effective".into(),
            write_input: None,
            write_output: None,
            output: None,
            input: None,
            reads: vec![
                RoutedComponentRead {
                    role: ComponentReadRole::Gate,
                    weight: mapping("gate_up"),
                    bias: Some(mapping("read_bias")),
                    projection_rows: 10,
                    rows: ComponentRowMapping::Direct { offset: 0 },
                },
                RoutedComponentRead {
                    role: ComponentReadRole::Value,
                    weight: mapping("gate_up"),
                    bias: Some(mapping("read_bias")),
                    projection_rows: 10,
                    rows: ComponentRowMapping::Direct { offset: 5 },
                },
            ],
            write_weight: mapping("down"),
            write_bias: Some(mapping("down_bias")),
            activation_equation: ComponentActivation::Gated {
                activation: ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                },
                gate_upper_bound: None,
                value_absolute_bound: None,
                value_offset: ComponentScalar::new(0.0),
            },
            input_normalization: None,
            residual_scale: Some(ComponentScalar::new(1.0)),
        };
        let loaded = ParameterDiscovery {
            identity: "loaded-edit-2".into(),
            artifact_identity: "artifact".into(),
            overlay_identity: Some("edit-2".into()),
            usage: CaptureUsage::default(),
            coordination_usage: Default::default(),
            parameters: [
                ("gate_up", vec![3, 10, 4]),
                ("read_bias", vec![3, 10]),
                ("down", vec![3, 6, 5]),
                ("down_bias", vec![3, 6]),
            ]
            .into_iter()
            .map(|(id, shape)| LoadedParameter {
                id: id.into(),
                shared_id: format!("actual-shared:{id}"),
                shape,
                dtype: Some(InterventionDtype::Float32),
                supported: true,
                access: None,
                condition: "edited effective parameter".into(),
                input_transform: ProjectionInputTransform::Identity,
            })
            .collect(),
        };
        let id = RoutedComponentId {
            group: group.id.clone(),
            expert: 2,
            index: 3,
        };
        (group, loaded, id)
    }

    #[test]
    fn packed_expert_regions_preserve_all_axes_fused_reads_and_actual_aliases() {
        let (group, loaded, id) = fixture();
        for (role, row) in [(ComponentReadRole::Gate, 3), (ComponentReadRole::Value, 8)] {
            let selected = group.read_weight(&id, role, &loaded).unwrap();
            assert_eq!(
                selected.region,
                ParameterRegion {
                    starts: vec![2, row, 0],
                    shape: vec![1, 1, 4]
                }
            );
            assert_eq!(selected.parameter.shared_id, "actual-shared:gate_up");
            let bias = group.read_bias(&id, role, &loaded).unwrap().unwrap();
            assert_eq!(
                bias.region,
                ParameterRegion {
                    starts: vec![2, row],
                    shape: vec![1, 1]
                }
            );
        }
        assert_eq!(
            group.write_column(&id, &loaded).unwrap().region,
            ParameterRegion {
                starts: vec![2, 0, 3],
                shape: vec![1, 6, 1]
            }
        );
        assert_eq!(
            group.write_bias(&id, &loaded).unwrap().unwrap().region,
            ParameterRegion {
                starts: vec![2, 0],
                shape: vec![1, 6]
            }
        );
        let encoded = serde_json::to_string(&group).unwrap();
        assert_eq!(
            serde_json::from_str::<RoutedComponentGroup>(&encoded).unwrap(),
            group
        );
    }

    #[test]
    fn independent_expert_bias_absence_is_distinct_from_missing_loaded_values() {
        let (mut group, mut loaded, mut id) = fixture();
        let RoutedComponentParameter::Packed { name } = mapping("individual_bias") else {
            unreachable!()
        };
        group.write_bias = Some(RoutedComponentParameter::Independent {
            names: vec![None, Some(name), None],
        });
        id.expert = 0;
        assert_eq!(group.write_bias(&id, &loaded).unwrap(), None);
        id.expert = 1;
        assert!(matches!(
            group.write_bias(&id, &loaded),
            Err(RoutedComponentError::MissingParameter(_))
        ));
        let mut parameter = loaded.parameters[3].clone();
        parameter.id = "individual_bias".into();
        parameter.shape = vec![6];
        parameter.supported = false;
        loaded.parameters.push(parameter);
        let selected = group.write_bias(&id, &loaded).unwrap().unwrap();
        assert_eq!(
            selected.region,
            ParameterRegion {
                starts: vec![0],
                shape: vec![6]
            }
        );
        assert!(
            !selected.parameter.access().query,
            "a topology join cannot upgrade loaded capability"
        );
    }

    #[test]
    fn malformed_component_or_loaded_geometry_fails_before_producing_regions() {
        let (group, loaded, id) = fixture();
        for invalid in [
            RoutedComponentId {
                expert: 3,
                ..id.clone()
            },
            RoutedComponentId {
                index: 5,
                ..id.clone()
            },
            RoutedComponentId {
                group: "other invocation".into(),
                ..id.clone()
            },
        ] {
            assert_eq!(
                group.write_column(&invalid, &loaded).unwrap_err(),
                RoutedComponentError::Identity
            );
        }
        assert_eq!(
            group
                .read_weight(&id, ComponentReadRole::Key, &loaded)
                .unwrap_err(),
            RoutedComponentError::ReadRole
        );
        let mut invalid = loaded.clone();
        invalid.parameters[2].shape = vec![6, 15];
        assert_eq!(
            group.write_column(&id, &invalid).unwrap_err(),
            RoutedComponentError::Geometry
        );
        let mut invalid_group = group.clone();
        invalid_group.reads[0].rows = ComponentRowMapping::Direct { offset: usize::MAX };
        assert_eq!(
            invalid_group
                .read_weight(&id, ComponentReadRole::Gate, &loaded)
                .unwrap_err(),
            RoutedComponentError::Geometry
        );
        invalid_group = group.clone();
        invalid_group.reads.push(group.reads[0].clone());
        assert_eq!(
            invalid_group
                .read_weight(&id, ComponentReadRole::Gate, &loaded)
                .unwrap_err(),
            RoutedComponentError::Geometry
        );
        let mut missing = loaded;
        missing.parameters.retain(|p| p.id != "down");
        assert!(matches!(
            group.write_column(&id, &missing),
            Err(RoutedComponentError::MissingParameter(_))
        ));
    }
}
