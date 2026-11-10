//! Gated-delta reads, fused causal convolution and consumed output channels.
use super::*;

pub(super) fn declare(
    g: &mut Builder,
    declaration: &crate::gated_delta::GatedDeltaMixerSpec,
    normalization: &eredu_nn::NormalizationConstructionSpec,
    layer: usize,
    path: &str,
    node: &str,
) {
    let key_heads = declaration.geometry.key_heads() as usize;
    let value_heads = declaration.geometry.value_heads() as usize;
    let key_dim = declaration.geometry.key_dimensions() as usize;
    let value_dim = declaration.geometry.value_dimensions() as usize;
    let count = declaration.input_gate.output as usize;
    let qkv_width = declaration.input_qkv.output as usize;
    let key_width = declaration.geometry.key_width() as usize;
    let prefix = format!("{path}.linear_attn");
    let group = format!("parameters:{prefix}");
    let head_rows = |offset, read_head_width, repeats| ComponentRowMapping::HeadRows {
        offset,
        component_head_width: value_dim,
        read_head_width,
        component_heads_per_read_head: repeats,
        read_head_stride: None,
    };
    let qk_norm = || {
        Some(ComponentHeadNormalization {
            output_scale: ComponentScalar::new(1.0),
            heads: key_heads,
            head_width: key_dim,
            independent_gains: false,
            normalization: ComponentNormalization {
                kind: ComponentNormalizationKind::L2,
                epsilon: ComponentScalar::new(declaration.l2_epsilon),
                gain: None,
                gain_offset: ComponentScalar::new(0.0),
                bias: None,
                groups: 1,
            },
        })
    };
    let read = |projection: &eredu_nn::LinearSpec, role, rows, head_normalization, suffix: &str| {
        ComponentRead {
            projection_output: Some(format!("{path}.mixer.{suffix}.projected")),
            head_normalization,
            ..declared_component_read(projection, group.clone(), role, rows)
        }
    };
    let reads = vec![
        read(
            &declaration.input_qkv,
            ComponentReadRole::Query,
            head_rows(0, key_dim, value_heads / key_heads),
            qk_norm(),
            "qkv",
        ),
        read(
            &declaration.input_qkv,
            ComponentReadRole::Key,
            head_rows(key_width, key_dim, value_heads / key_heads),
            qk_norm(),
            "qkv",
        ),
        read(
            &declaration.input_qkv,
            ComponentReadRole::Value,
            ComponentRowMapping::Direct {
                offset: 2 * key_width,
            },
            None,
            "qkv",
        ),
        read(
            &declaration.input_gate,
            ComponentReadRole::OutputGate,
            ComponentRowMapping::Direct { offset: 0 },
            None,
            "gate",
        ),
        read(
            &declaration.input_beta,
            ComponentReadRole::Update,
            head_rows(0, 1, 1),
            None,
            "update",
        ),
        read(
            &declaration.input_decay,
            ComponentReadRole::Decay,
            head_rows(0, 1, 1),
            None,
            "decay",
        ),
    ];
    for (suffix, width) in [
        ("qkv", qkv_width),
        ("gate", count),
        ("update", value_heads),
        ("decay", value_heads),
    ] {
        g.read_only_observation(
            node,
            format!("{path}.mixer.{suffix}.projected"),
            "Actual affine recurrent read before convolution, normalization or nonlinearity",
            axes(width),
        );
    }
    let convolved = format!("{path}.mixer.qkv.convolved");
    g.read_only_observation(
        node,
        convolved.clone(),
        "Causal SiLU-convolved Q/K/V before head normalization",
        axes(qkv_width),
    );
    g.descriptor
        .component_transforms
        .push(ComponentTensorTransform {
            id: format!("{node}.qkv.convolution"),
            node_id: node.into(),
            input: format!("{path}.mixer.qkv.projected"),
            output: convolved,
            effective_output: None,
            equation: ComponentTensorTransformEquation::CausalDepthwiseConvolution {
                kernel: declared_transform_parameter(
                    &declaration.convolution.weight,
                    group.clone(),
                ),
                channels: qkv_width,
                taps: declaration.convolution.kernel_size as usize,
                residual: false,
                activation: Some(ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                }),
            },
        });
    let write = declaration.output.weight.id.as_str().to_owned();
    g.descriptor.components.push(ComponentGroup {
        id: format!("{node}.channels"),
        node_id: node.into(),
        layer_index: layer,
        count,
        activation: format!("{path}.mixer.channels"),
        effective_activation: format!("{path}.mixer.channels.effective"),
        write_input: Some(format!("{path}.mixer.write_input")),
        write_output: Some(format!("{path}.mixer.write")),
        output: Some(format!("{path}.mixer.output")),
        input: format!("{path}.mixer.input"),
        write_partition: ComponentWritePartition::Complete,
        reads,
        routed_reads: vec![],
        shared_write_weight: declaration
            .output
            .weight
            .alias_of
            .as_ref()
            .unwrap_or(&declaration.output.weight.id)
            .as_str()
            .to_owned(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: group.clone(),
        write_bias: declaration
            .output
            .bias
            .as_ref()
            .map(|bias| bias.id.as_str().to_owned()),
        activation_equation: ComponentActivation::GatedDeltaAttention {
            key_heads,
            value_heads,
            key_head_width: key_dim,
            value_head_width: value_dim,
            decay_layout: ComponentDeltaDecay::Head,
            query_scale: ComponentScalar::new((key_dim as f32).sqrt().recip()),
            key_scale: ComponentScalar::new(1.0),
            decay_rate: declared_transform_parameter(&declaration.transition_log, group.clone()),
            decay_bias: declared_transform_parameter(&declaration.decay_bias, group.clone()),
            channel_normalization: ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads: value_heads,
                head_width: value_dim,
                independent_gains: false,
                normalization: ComponentNormalization {
                    kind: ComponentNormalizationKind::Rms,
                    epsilon: ComponentScalar::new(declaration.output_epsilon),
                    gain: Some(declaration.normalization_weight.id.as_str().to_owned()),
                    gain_offset: ComponentScalar::new(0.0),
                    bias: None,
                    groups: 1,
                },
            },
            output_gate: match declaration.output_gate {
                eredu_nn::OutputGateActivation::Silu => ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                },
                eredu_nn::OutputGateActivation::Sigmoid => ComponentNonlinearity::Sigmoid,
            },
        },
        input_normalization: declared_normalization(normalization),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut shape = axes(count);
    shape[2].name = "component".into();
    g.component_observation(node, format!("{path}.mixer.channels"),
        "Recurrent value channels after head RMS normalization and SiLU gating, before output projection", shape);
}
