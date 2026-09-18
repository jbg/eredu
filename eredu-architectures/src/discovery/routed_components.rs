//! Expert-unit topology projected from canonical bank construction specifications.
use super::*;
use eredu_core::component::*;
use eredu_nn::{
    GatedProductGroupLayout, GroupedGatedProductSpec, GroupedProjectionSpec, GroupedRelu2Spec,
    ParameterSpec,
};

pub(super) fn declare_captures(g: &mut Builder) {
    let groups: Vec<_> = g
        .descriptor
        .routed_components
        .iter()
        .chain(
            g.descriptor
                .component_scopes
                .iter()
                .flat_map(|scope| &scope.routed_components),
        )
        .cloned()
        .collect();
    for group in groups {
        let mut node = g.descriptor.node(&group.node_id);
        let routes = group.routes_per_token.unwrap_or_else(|| loop {
            match node {
                Some(n) if n.moe.is_some() => break n.moe.as_ref().unwrap().selected_experts,
                Some(n) => node = n.parent.as_ref().and_then(|p| g.descriptor.node(p)),
                None => break 0,
            }
        });
        if routes == 0 {
            g.partial("routed-unit capture has no declared route cardinality");
            continue;
        }
        let geometry = eredu_core::capture::RoutedUnitGeometry {
            experts: group.expert_count as u64,
            units_per_expert: group.units_per_expert as u64,
            routes_per_token: routes as u64,
        };
        if let Ok(components) = geometry.components().and_then(|n| {
            usize::try_from(n).map_err(|_| {
                eredu_core::capture::CaptureError::Invalid(
                    "sparse component extent overflow".into(),
                )
            })
        }) {
            use eredu_core::intervention::*;
            g.interventions.push(InterventionPoint {
                path: group.activation.clone(), node_id: group.node_id.clone(),
                stage: InterventionStage::Activation,
                axes: vec![
                    TensorAxis { name: "token".into(), dimension: SymbolicDimension::TokenRows },
                    TensorAxis { name: "component".into(), dimension: SymbolicDimension::Known(components) },
                ],
                dtypes: vec![InterventionDtype::Float32, InterventionDtype::Float16, InterventionDtype::Bfloat16],
                operations: vec![InterventionKind::Zero, InterventionKind::Scale, InterventionKind::Mask, InterventionKind::MaskComponents, InterventionKind::Replace, InterventionKind::Add],
                score_stages: vec![],
                prefill: ObservationSupportStatus::Unverified("requires loaded sparse-unit mechanisms".into()),
                decode: ObservationSupportStatus::Unverified("requires loaded sparse-unit mechanisms".into()),
                conditions: vec!["Global component = expert * units_per_expert + unit; edits follow current routing, before down projection and route weighting".into(), "Use original/effective RoutedUnits captures for attributed evidence".into()],
                routing: None,
                routed_units: Some(RoutedUnitInterventionPoint { routing: group.routing.clone(), geometry }),
            });
        }
        for (path, position) in [
            (group.activation, ObservationPosition::BeforeIntervention),
            (
                group.effective_activation,
                ObservationPosition::AfterIntervention,
            ),
        ] {
            g.get_mut(&group.node_id)
                .observation_paths
                .push(path.clone());
            g.descriptor.observations.points.push(ObservationPoint {
                path, node_id: group.node_id.clone(),
                meaning: "Actual selected expert units before down projection and route weighting; original route identity is retained".into(),
                value_type: ObservationValueType::RoutedUnits { routing: group.routing.clone(), geometry },
                dtype: ObservationDtype::Floating,
                axes: Some(vec![
                    TensorAxis { name: "token".into(), dimension: SymbolicDimension::TokenRows },
                    TensorAxis { name: "route".into(), dimension: SymbolicDimension::Known(routes) },
                    TensorAxis { name: "component".into(), dimension: SymbolicDimension::Known(group.units_per_expert) },
                ]),
                prefill: true, decode: true,
                requirements: vec![ObservationRequirement::ActivationHooks],
                position, retained_bytes: None, host_bytes: None,
            });
        }
    }
}

pub(super) fn rms(gain: String, epsilon: f32, offset: f32) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(epsilon),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(offset),
        bias: None,
        groups: 1,
    }
}

pub(super) struct Site<'a> {
    pub node: &'a str,
    pub layer: usize,
    pub routing: &'a str,
    pub input: Option<String>,
    pub normalization: Option<ComponentNormalization>,
    pub residual_scale: Option<ComponentScalar>,
}

pub(super) fn decoder<C: crate::decoder::Config>(
    g: &mut Builder,
    config: &C,
    bank: impl Fn(usize) -> Result<GroupedGatedProductSpec, eredu_nn::Error>,
) {
    for layer in 0..config.num_hidden_layers() as usize {
        let path = format!("{}.layers.{layer}", config.parameter_root());
        let Some(points) = config.routed_observation_points(&path, layer, None).expect("ordinary routed point construction") else {
            continue;
        };
        let Some(point) = points.bank(eredu_runtime::RoutedBankId::new(0)) else {
            continue;
        };
        let mut normalization = rms(
            format!(
                "{path}.{}.weight",
                config.block_parameter_fields().post_attention_norm
            ),
            config.rms_norm_epsilon(),
            config.normalization_offset(),
        );
        normalization.groups = config.normalization_groups().unwrap_or(1) as usize;
        gated(
            g,
            Site {
                node: &format!("decoder.layers.{layer}.feed_forward"),
                layer,
                routing: point.path(),
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(normalization),
                residual_scale: Some(ComponentScalar::new(1.0)),
            },
            bank(layer),
        );
    }
}

fn name(spec: &ParameterSpec, group: &str) -> RoutedComponentParameterName {
    RoutedComponentParameterName {
        parameter: spec.id.to_string(),
        shared_parameter: spec.alias_of.as_ref().unwrap_or(&spec.id).to_string(),
        parameter_group: group.into(),
    }
}
fn packed(
    spec: &GroupedProjectionSpec,
    group: &str,
) -> (RoutedComponentParameter, Option<RoutedComponentParameter>) {
    (
        RoutedComponentParameter::Packed {
            name: name(spec.weight(), group),
        },
        spec.bias().map(|bias| RoutedComponentParameter::Packed {
            name: name(bias, group),
        }),
    )
}
fn independent<'a>(
    specs: impl Iterator<Item = &'a GroupedProjectionSpec>,
    group: &str,
) -> (RoutedComponentParameter, Option<RoutedComponentParameter>) {
    let (weights, biases): (Vec<_>, Vec<_>) = specs
        .map(|spec| {
            (
                Some(name(spec.weight(), group)),
                spec.bias().map(|bias| name(bias, group)),
            )
        })
        .unzip();
    let bias = biases
        .iter()
        .any(Option::is_some)
        .then_some(RoutedComponentParameter::Independent { names: biases });
    (
        RoutedComponentParameter::Independent { names: weights },
        bias,
    )
}

fn group_name(g: &Builder, node: &str) -> Option<String> {
    g.descriptor.node(node)?.parameter_groups.first().cloned()
}
fn finish(
    g: &mut Builder,
    site: Site<'_>,
    experts: usize,
    units: usize,
    input_width: usize,
    output_width: usize,
    reads: Vec<RoutedComponentRead>,
    write: (RoutedComponentParameter, Option<RoutedComponentParameter>),
    equation: ComponentActivation,
) {
    let routed = format!("{}.routed", site.node);
    let node_id = if g.descriptor.node(&routed).is_some() {
        routed
    } else {
        site.node.into()
    };
    g.descriptor.routed_components.push(RoutedComponentGroup {
        routes_per_token: None,
        id: format!("{}.expert_units", site.node),
        node_id,
        layer_index: site.layer,
        bank: 0,
        expert_count: experts,
        units_per_expert: units,
        input_width,
        output_width,
        routing: site.routing.into(),
        activation: format!("{}.units", site.routing),
        effective_activation: format!("{}.units.effective", site.routing),
        write_input: None,
        write_output: None,
        output: None,
        input: site.input,
        reads,
        write_weight: write.0,
        write_bias: write.1,
        activation_equation: equation,
        input_normalization: site.normalization,
        residual_scale: site.residual_scale,
    });
}

pub(super) fn gated(
    g: &mut Builder,
    site: Site<'_>,
    spec: Result<GroupedGatedProductSpec, eredu_nn::Error>,
) {
    let spec = match spec {
        Ok(spec) => spec,
        Err(error) => {
            g.partial(format!("expert-unit declaration unavailable: {error}"));
            return;
        }
    };
    let Some(group) = group_name(g, site.node) else {
        g.partial("expert-unit declaration has no owning parameter group");
        return;
    };
    let units = spec.intermediate_dimensions() as usize;
    let policy = spec.policy();
    let activation = match policy.activation() {
        eredu_nn::GatedProductActivation::Silu => ComponentNonlinearity::Silu {
            multiplier: ComponentScalar::new(policy.sigmoid_multiplier()),
        },
        eredu_nn::GatedProductActivation::GeluApproximate => ComponentNonlinearity::GeluApproximate,
        _ => {
            g.partial("expert-unit activation equation is not described");
            return;
        }
    };
    let read = |role,
                pair: (RoutedComponentParameter, Option<RoutedComponentParameter>),
                projection_rows,
                offset| RoutedComponentRead {
        role,
        weight: pair.0,
        bias: pair.1,
        projection_rows,
        rows: ComponentRowMapping::Direct { offset },
    };
    let (reads, write) = match spec.layout() {
        GatedProductGroupLayout::Packed { gate_up, down } => {
            let Some(rows) = units.checked_mul(2) else {
                g.partial("expert-unit fused row geometry overflow");
                return;
            };
            (
                vec![
                    read(ComponentReadRole::Gate, packed(gate_up, &group), rows, 0),
                    read(
                        ComponentReadRole::Value,
                        packed(gate_up, &group),
                        rows,
                        units,
                    ),
                ],
                packed(down, &group),
            )
        }
        GatedProductGroupLayout::Independent(experts) => (
            vec![
                read(
                    ComponentReadRole::Gate,
                    independent(experts.iter().map(|e| e.gate()), &group),
                    units,
                    0,
                ),
                read(
                    ComponentReadRole::Value,
                    independent(experts.iter().map(|e| e.up()), &group),
                    units,
                    0,
                ),
            ],
            independent(experts.iter().map(|e| e.down()), &group),
        ),
        _ => {
            g.partial("expert-unit parameter layout is not described");
            return;
        }
    };
    finish(
        g,
        site,
        spec.group_count() as usize,
        units,
        spec.input_dimensions() as usize,
        spec.output_dimensions() as usize,
        reads,
        write,
        ComponentActivation::Gated {
            activation,
            gate_upper_bound: policy.gate_upper_bound().map(ComponentScalar::new),
            value_absolute_bound: policy.up_absolute_bound().map(ComponentScalar::new),
            value_offset: ComponentScalar::new(policy.up_offset()),
        },
    );
}

pub(super) fn relu2(
    g: &mut Builder,
    site: Site<'_>,
    spec: Result<GroupedRelu2Spec, eredu_nn::Error>,
) {
    let spec = match spec {
        Ok(spec) => spec,
        Err(error) => {
            g.partial(format!("expert-unit declaration unavailable: {error}"));
            return;
        }
    };
    let Some(group) = group_name(g, site.node) else {
        g.partial("expert-unit declaration has no owning parameter group");
        return;
    };
    let units = spec.intermediate_dimensions() as usize;
    let (weight, bias) = packed(spec.up(), &group);
    finish(
        g,
        site,
        spec.group_count() as usize,
        units,
        spec.hidden_dimensions() as usize,
        spec.hidden_dimensions() as usize,
        vec![RoutedComponentRead {
            role: ComponentReadRole::Input,
            weight,
            bias,
            projection_rows: units,
            rows: ComponentRowMapping::Direct { offset: 0 },
        }],
        packed(spec.down(), &group),
        ComponentActivation::Unary {
            activation: ComponentNonlinearity::ReluSquared,
        },
    );
}
