//! Decoder/latent projections and normalization before expert accumulation.
use super::*;

pub(super) struct CompressedSite {
    pub input: String,
    pub normalization: ComponentNormalization,
    pub expert_transform: String,
}

pub(super) fn declare(
    g: &mut Builder,
    c: &DecoderConfig,
    layer: usize,
    node: &str,
    path: &str,
    parameter: &str,
) -> Option<CompressedSite> {
    let declaration = c.routed(layer).expect("validated Muse routed declaration");
    let compression = declaration.compression()?;
    let scalar = c.scalar(layer).expect("validated Muse scalar declaration");
    let norms = &compression.norms;
    let latent = compression.input.output as usize;
    let width = compression.input.input as usize;
    let boundary = |field| format!("{path}.feed_forward.{field}");
    let matrix = |projection: &eredu_nn::LinearSpec| {
        declared_transform_parameter(&projection.weight, format!("parameters:{parameter}"))
    };
    let rows = |channels| {
        vec![
            TensorAxis {
                name: "token".into(),
                dimension: SymbolicDimension::TokenRows,
            },
            TensorAxis {
                name: "channel".into(),
                dimension: SymbolicDimension::Known(channels),
            },
        ]
    };
    for field in ["compressed_input", "expert_input", "expert_mixture"] {
        g.component_observation(
            node,
            boundary(field),
            "Muse compressed expert token rows",
            rows(latent),
        );
    }
    for field in ["expert_outputs", "normalized_experts"] {
        g.component_observation(
            node,
            boundary(field),
            "Complete selected expert outputs before coefficient weighting",
            vec![
                TensorAxis {
                    name: "token".into(),
                    dimension: SymbolicDimension::TokenRows,
                },
                TensorAxis {
                    name: "route".into(),
                    dimension: SymbolicDimension::Known(
                        declaration.selector().selection().top_k() as usize
                    ),
                },
                TensorAxis {
                    name: "channel".into(),
                    dimension: SymbolicDimension::Known(latent),
                },
            ],
        );
    }
    let mut transform = |id: String, input: String, output: String, equation| {
        g.descriptor
            .component_transforms
            .push(ComponentTensorTransform {
                id,
                node_id: node.into(),
                input,
                effective_output: Some(format!("{output}.effective")),
                output,
                equation,
            });
    };
    transform(
        format!("{path}.pre_feedforward_layernorm"),
        format!("{path}.attention.residual.effective"),
        boundary("input"),
        ComponentTensorTransformEquation::Normalization {
            normalization: declared_target_norm(&scalar.pre_feed_forward),
        },
    );
    transform(
        format!("{parameter}.pre_expert_proj"),
        format!("{}.effective", boundary("input")),
        boundary("compressed_input"),
        ComponentTensorTransformEquation::AffineProjection {
            weight: matrix(&compression.input),
            bias: None,
            input_width: width,
            output_width: latent,
        },
    );
    let normalization = declared_target_norm(&norms[0]);
    transform(
        format!("{parameter}.pre_expert_norm"),
        format!("{}.effective", boundary("compressed_input")),
        boundary("expert_input"),
        ComponentTensorTransformEquation::Normalization {
            normalization: normalization.clone(),
        },
    );
    let expert_transform = format!("{parameter}.experts.post_expert_norm");
    transform(
        expert_transform.clone(),
        format!("{}.effective", boundary("expert_outputs")),
        boundary("normalized_experts"),
        ComponentTensorTransformEquation::Normalization {
            normalization: declared_target_norm(&norms[1]),
        },
    );
    transform(
        format!("{parameter}.post_expert_proj"),
        format!("{}.effective", boundary("expert_mixture")),
        boundary("write"),
        ComponentTensorTransformEquation::AffineProjection {
            weight: matrix(&compression.output),
            bias: None,
            input_width: latent,
            output_width: width,
        },
    );
    Some(CompressedSite {
        input: boundary("expert_input"),
        normalization,
        expert_transform,
    })
}
