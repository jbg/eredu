//! Gated-delta reads, fused causal convolution and consumed output channels.
use super::*;

pub(super) fn declare(g: &mut Builder, c: &HybridConfig, layer: usize, path: &str, node: &str) {
    let key_heads = c.linear_num_key_heads as usize;
    let value_heads = c.linear_num_value_heads as usize;
    let key_dim = c.linear_key_head_dim as usize;
    let value_dim = c.linear_value_head_dim as usize;
    let key_width = key_heads * key_dim;
    let count = value_heads * value_dim;
    let qkv_width = 2 * key_width + count;
    let prefix = format!("{path}.linear_attn");
    let group = format!("parameters:{prefix}");
    let parameter = |suffix: &str| ComponentTransformParameter {
        parameter: format!("{prefix}.{suffix}"),
        shared_parameter: format!("{prefix}.{suffix}"),
        parameter_group: group.clone(),
    };
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
                epsilon: ComponentScalar::new(1e-6),
                gain: None,
                gain_offset: ComponentScalar::new(0.0),
                bias: None,
                groups: 1,
            },
        })
    };
    let read = |field: &str, role, rows, head_normalization, suffix: &str| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            shared_weight: weight.clone(),
            weight,
            parameter_group: group.clone(),
            bias: None,
            role,
            rows,
            head_normalization,
            input_projections: vec![],
            projection_output: Some(format!("{path}.mixer.{suffix}.projected")),
        }
    };
    let reads = vec![
        read(
            "in_proj_qkv",
            ComponentReadRole::Query,
            head_rows(0, key_dim, value_heads / key_heads),
            qk_norm(),
            "qkv",
        ),
        read(
            "in_proj_qkv",
            ComponentReadRole::Key,
            head_rows(key_width, key_dim, value_heads / key_heads),
            qk_norm(),
            "qkv",
        ),
        read(
            "in_proj_qkv",
            ComponentReadRole::Value,
            ComponentRowMapping::Direct {
                offset: 2 * key_width,
            },
            None,
            "qkv",
        ),
        read(
            "in_proj_z",
            ComponentReadRole::OutputGate,
            ComponentRowMapping::Direct { offset: 0 },
            None,
            "gate",
        ),
        read(
            "in_proj_b",
            ComponentReadRole::Update,
            head_rows(0, 1, 1),
            None,
            "update",
        ),
        read(
            "in_proj_a",
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
                kernel: parameter("conv1d.weight"),
                channels: qkv_width,
                taps: c.linear_conv_kernel_dim as usize,
                residual: false,
                activation: Some(ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                }),
            },
        });
    let write = format!("{prefix}.out_proj.weight");
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
        shared_write_weight: write.clone(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: group.clone(),
        write_bias: None,
        activation_equation: ComponentActivation::GatedDeltaAttention {
            key_heads,
            value_heads,
            key_head_width: key_dim,
            value_head_width: value_dim,
            decay_layout: ComponentDeltaDecay::Head,
            query_scale: ComponentScalar::new((key_dim as f32).sqrt().recip()),
            key_scale: ComponentScalar::new(1.0),
            decay_rate: parameter("A_log"),
            decay_bias: parameter("dt_bias"),
            channel_normalization: ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads: value_heads,
                head_width: value_dim,
                independent_gains: false,
                normalization: ComponentNormalization {
                    kind: ComponentNormalizationKind::Rms,
                    epsilon: ComponentScalar::new(1e-6),
                    gain: Some(format!("{prefix}.norm.weight")),
                    gain_offset: ComponentScalar::new(0.0),
                    bias: None,
                    groups: 1,
                },
            },
            output_gate: ComponentNonlinearity::Silu {
                multiplier: ComponentScalar::new(1.0),
            },
        },
        input_normalization: normalization(c, format!("{path}.input_layernorm.weight")),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut shape = axes(count);
    shape[2].name = "component".into();
    g.component_observation(node, format!("{path}.mixer.channels"),
        "Recurrent value channels after head RMS normalization and SiLU gating, before output projection", shape);
}
