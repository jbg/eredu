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
    node: &str,
    path: &str,
    parameter: &str,
) -> Option<CompressedSite> {
    let crate::muse::RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. } =
        c.routed_feed_forward
    else {
        return None;
    };
    let latent = expert_width as usize;
    let width = c.hidden_size as usize;
    let boundary = |field| format!("{path}.feed_forward.{field}");
    let matrix = |field: &str| ComponentTransformParameter {
        parameter: format!("{parameter}.{field}.weight"),
        shared_parameter: format!("{parameter}.{field}.weight"),
        parameter_group: format!("parameters:{parameter}"),
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
                    dimension: SymbolicDimension::Known(c.num_experts_per_tok as usize),
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
            normalization: norm(
                c,
                Some(format!("{path}.pre_feedforward_layernorm.weight")),
                NormStage::Input,
            ),
        },
    );
    transform(
        format!("{parameter}.pre_expert_proj"),
        format!("{}.effective", boundary("input")),
        boundary("compressed_input"),
        ComponentTensorTransformEquation::AffineProjection {
            weight: matrix("pre_expert_proj"),
            bias: None,
            input_width: width,
            output_width: latent,
        },
    );
    let normalization = norm(
        c,
        Some(format!("{parameter}.pre_expert_norm.weight")),
        NormStage::Input,
    );
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
            normalization: norm(
                c,
                Some(format!("{expert_transform}.weight")),
                NormStage::FeedForwardOutput,
            ),
        },
    );
    transform(
        format!("{parameter}.post_expert_proj"),
        format!("{}.effective", boundary("expert_mixture")),
        boundary("write"),
        ComponentTensorTransformEquation::AffineProjection {
            weight: matrix("post_expert_proj"),
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
