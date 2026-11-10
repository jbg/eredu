//! Ordinary decoder attention ownership, head geometry and physical projection slots.

use super::*;
use eredu_nn::AttentionHeadGeometry;

/// Component-major Q/K/V or independent projections with a separately supplied value.
#[derive(Debug, Clone)]
pub(crate) enum AttentionInputParameters {
    Split {
        query: LinearSpec,
        key: LinearSpec,
        value: Option<LinearSpec>,
    },
    Fused {
        projection: LinearSpec,
        layout: FusedProjectionLayout,
    },
}

/// Parameter ownership is independent of invocation policy and mutable attention state.
#[derive(Debug, Clone)]
pub(crate) struct AttentionParameters {
    pub geometry: AttentionHeadGeometry,
    pub input: AttentionInputParameters,
    pub gate: Option<(LinearSpec, OutputGateActivation)>,
    pub output: LinearSpec,
}
impl AttentionParameters {
    pub fn new(config: &impl Config, layer: usize) -> Result<Self, Error> {
        let fields = config.block_parameter_fields().validate()?;
        let prefix = format!(
            "{}.layers.{layer}.{}",
            config.parameter_root(),
            fields.attention
        );
        let geometry = AttentionHeadGeometry::new(
            config.num_attention_heads(),
            config.num_key_value_heads(),
            config.head_dim(),
            config.head_dim(),
        )?;
        let hidden = config.hidden_size();
        let linear = |field, input, output, bias| {
            topology::linear(config, &prefix, field, input, output, bias)
        };
        let input = match config.attention_projection_layout() {
            AttentionProjectionLayout::Split => AttentionInputParameters::Split {
                query: linear(
                    fields.attention_query,
                    hidden,
                    geometry.query_width(),
                    config.attention_bias(AttentionProjection::Query),
                )?,
                key: linear(
                    fields.attention_key,
                    hidden,
                    geometry.key_width(),
                    config.attention_bias(AttentionProjection::Key),
                )?,
                value: (!config.external_attention_value(layer))
                    .then(|| {
                        linear(
                            fields.attention_value,
                            hidden,
                            geometry.value_width(),
                            config.attention_bias(AttentionProjection::Value),
                        )
                    })
                    .transpose()?,
            },
            AttentionProjectionLayout::Fused { field } => {
                if field.trim().is_empty() || config.external_attention_value(layer) {
                    return Err(Error::backend(
                        "fused QKV requires a named projection and ordinary values",
                    ));
                }
                let biases = [
                    AttentionProjection::Query,
                    AttentionProjection::Key,
                    AttentionProjection::Value,
                ]
                .map(|p| config.attention_bias(p));
                if biases.iter().any(|bias| *bias != biases[0]) {
                    return Err(Error::backend(
                        "fused QKV projection requires identical query/key/value bias policy",
                    ));
                }
                let layout = FusedProjectionLayout::new([
                    FusedProjectionSegment::new("query", geometry.query_width())?,
                    FusedProjectionSegment::new("key", geometry.key_width())?,
                    FusedProjectionSegment::new("value", geometry.value_width())?,
                ])?;
                AttentionInputParameters::Fused {
                    projection: linear(field, hidden, layout.output_width(), biases[0])?,
                    layout,
                }
            }
        };
        let spec = Self {
            geometry,
            input,
            gate: config
                .attention_output_gate()
                .map(|(field, activation)| -> Result<_, Error> {
                    Ok((
                        linear(field, hidden, geometry.attended_width(), false)?,
                        activation,
                    ))
                })
                .transpose()?,
            output: linear(
                fields.attention_output,
                geometry.attended_width(),
                hidden,
                config.attention_bias(AttentionProjection::Output),
            )?,
        };
        eredu_nn::validate_logical_parameter_scope(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    /// Affines in checkpoint/construction order, with absent provider slots omitted.
    pub fn projections(&self) -> Vec<&LinearSpec> {
        let mut p = match &self.input {
            AttentionInputParameters::Split { query, key, value } => {
                let mut p = vec![query, key];
                p.extend(value.iter());
                p
            }
            AttentionInputParameters::Fused { projection, .. } => vec![projection],
        };
        p.extend(self.gate.iter().map(|(spec, _)| spec));
        p.push(&self.output);
        p
    }
    /// The physical fused row segments consumed by construction and placement.
    pub fn fused_segments(&self) -> Option<Vec<std::ops::Range<usize>>> {
        let AttentionInputParameters::Fused { layout, .. } = &self.input else {
            return None;
        };
        Some(
            layout
                .ranges()
                .map(|range| range.start as usize..range.end as usize)
                .collect(),
        )
    }

    pub fn partitioned_projections(&self) -> Vec<(&LinearSpec, MemberSharding)> {
        let mut p = match &self.input {
            AttentionInputParameters::Split { query, key, value } => {
                let rows = MemberSharding::Partitioned { axis: 0 };
                let mut p = vec![(query, rows.clone()), (key, rows.clone())];
                p.extend(value.iter().map(|spec| (spec, rows.clone())));
                p
            }
            AttentionInputParameters::Fused { projection, .. } => vec![(
                projection,
                MemberSharding::PartitionedSegments {
                    axis: 0,
                    segments: self.fused_segments().expect("fused layout"),
                },
            )],
        };
        p.extend(
            self.gate
                .iter()
                .map(|(spec, _)| (spec, MemberSharding::Partitioned { axis: 0 })),
        );
        p.push((&self.output, MemberSharding::Partitioned { axis: 1 }));
        p
    }
    fn validate_geometry(&self) -> Result<(), Error> {
        let g = self.geometry;
        let hidden = self.output.output;
        let affine = |spec: &LinearSpec, rows| spec.input == hidden && spec.output == rows;
        let valid = match &self.input {
            AttentionInputParameters::Split { query, key, value } => {
                affine(query, g.query_width())
                    && affine(key, g.key_width())
                    && value
                        .as_ref()
                        .is_none_or(|spec| affine(spec, g.value_width()))
            }
            AttentionInputParameters::Fused { projection, layout } => {
                let expected = [
                    ("query", g.query_width()),
                    ("key", g.key_width()),
                    ("value", g.value_width()),
                ];
                layout.segments().len() == expected.len()
                    && layout
                        .segments()
                        .iter()
                        .zip(expected)
                        .all(|(s, (name, width))| s.name() == name && s.width() == width)
                    && affine(projection, layout.output_width())
            }
        };
        if g.key_dimensions() != g.value_dimensions()
            || !valid
            || self.output.input != g.attended_width()
            || self
                .gate
                .as_ref()
                .is_some_and(|(spec, _)| !affine(spec, g.attended_width()))
        {
            return Err(Error::backend(
                "incompatible ordinary attention projection geometry",
            ));
        }
        Ok(())
    }
}
impl DeclareLogicalParameters for AttentionParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = Vec::new();
        for projection in self.projections() {
            parameters.extend(projection.logical_parameters()?);
        }
        Ok(parameters)
    }
}
