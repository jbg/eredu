//! Source-free Gemma audio parameter geometry.

/// Logical audio geometry shared by checkpoint admission and resource ownership.
pub(crate) struct AudioParameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub format: Option<eredu_checkpoint::LinearFormat>,
    pub optional: bool,
}
impl AudioParameter {
    fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
}
fn safe(name: impl Into<String>, shape: Vec<usize>) -> AudioParameter {
    AudioParameter {
        name: name.into(),
        shape,
        format: None,
        optional: false,
    }
}
fn add_matrix(
    output: &mut Vec<AudioParameter>,
    name: &str,
    shape: Vec<usize>,
    format: eredu_checkpoint::LinearFormat,
) -> Result<(), String> {
    output.push(AudioParameter {
        name: name.into(),
        shape,
        format: Some(format),
        optional: false,
    });
    Ok(())
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Gemma 4 {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Gemma 4 {name} geometry overflows"))
}

pub(crate) fn declaration(
    config: &super::AudioConfig,
    text_hidden: usize,
    layers: std::ops::Range<usize>,
) -> Result<Vec<AudioParameter>, String> {
    let mut parameters = Vec::new();
    let output = &mut parameters;
    let hidden = dimension(config.hidden_size, "audio hidden size")?;
    let [first, second] = config.subsampling_conv_channels.as_slice() else {
        return Err("Gemma 4 audio requires two subsampling channels".into());
    };
    let first = dimension(*first, "first audio channel count")?;
    let second = dimension(*second, "second audio channel count")?;
    let projection = dimension(config.output_proj_dims, "audio output width")?;
    for (name, shape) in [
        (
            "model.audio_tower.subsample_conv_projection.layer0.conv.weight",
            vec![first, 3, 3, 1],
        ),
        (
            "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
            vec![first],
        ),
        (
            "model.audio_tower.subsample_conv_projection.layer1.conv.weight",
            vec![second, 3, 3, first],
        ),
        (
            "model.audio_tower.subsample_conv_projection.layer1.norm.weight",
            vec![second],
        ),
    ] {
        output.push(safe(name, shape));
    }
    let input_projection = "model.audio_tower.subsample_conv_projection.input_proj_linear.weight";
    let input_width = checked_mul(32, second, "audio subsampling width")?;
    add_matrix(
        output,
        input_projection,
        vec![hidden, input_width],
        config.linear_format_for(input_projection, input_width as i32),
    )?;
    let output_projection = "model.audio_tower.output_proj.weight";
    add_matrix(
        output,
        output_projection,
        vec![projection, hidden],
        config.linear_format_for(output_projection, hidden as i32),
    )?;
    output.push(safe("model.audio_tower.output_proj.bias", vec![projection]).optional());
    let count = dimension(config.num_hidden_layers, "audio layer count")?;
    if layers.start > layers.end || layers.end > count {
        return Err("Gemma 4 audio parameter layers exceed the declared stack".into());
    }
    for layer in layers {
        let root = format!("model.audio_tower.layers.{layer}");
        for local in [
            "feed_forward1.pre_layer_norm.weight",
            "feed_forward1.post_layer_norm.weight",
            "norm_pre_attn.weight",
            "norm_post_attn.weight",
            "lconv1d.pre_layer_norm.weight",
            "lconv1d.conv_norm.weight",
            "feed_forward2.pre_layer_norm.weight",
            "feed_forward2.post_layer_norm.weight",
            "norm_out.weight",
        ] {
            output.push(safe(format!("{root}.{local}"), vec![hidden]));
        }
        let four_hidden = checked_mul(4, hidden, "audio feed-forward width")?;
        let two_hidden = checked_mul(2, hidden, "audio convolution gate width")?;
        for (local, shape, input) in [
            (
                "feed_forward1.ffw_layer_1",
                vec![four_hidden, hidden],
                hidden,
            ),
            (
                "feed_forward1.ffw_layer_2",
                vec![hidden, four_hidden],
                four_hidden,
            ),
            ("self_attn.q_proj", vec![hidden, hidden], hidden),
            ("self_attn.k_proj", vec![hidden, hidden], hidden),
            ("self_attn.v_proj", vec![hidden, hidden], hidden),
            ("self_attn.post", vec![hidden, hidden], hidden),
            ("lconv1d.linear_start", vec![two_hidden, hidden], hidden),
            ("lconv1d.linear_end", vec![hidden, hidden], hidden),
            (
                "feed_forward2.ffw_layer_1",
                vec![four_hidden, hidden],
                hidden,
            ),
            (
                "feed_forward2.ffw_layer_2",
                vec![hidden, four_hidden],
                four_hidden,
            ),
        ] {
            add_audio_clipped_matrix(output, config, &format!("{root}.{local}"), shape, input)?;
        }
        output.extend([
            safe(
                format!("{root}.self_attn.relative_k_proj.weight"),
                vec![hidden, hidden],
            ),
            safe(
                format!("{root}.self_attn.per_dim_scale"),
                vec![config.head_dim() as usize],
            ),
            safe(
                format!("{root}.lconv1d.depthwise_conv1d.weight"),
                vec![
                    hidden,
                    dimension(config.conv_kernel_size, "audio convolution kernel")?,
                    1,
                ],
            ),
        ]);
    }
    let media_projection = "model.embed_audio.embedding_projection.weight";
    add_matrix(
        output,
        media_projection,
        vec![text_hidden, projection],
        config.linear_format_for(media_projection, projection as i32),
    )?;
    Ok(parameters)
}
fn add_audio_clipped_matrix(
    output: &mut Vec<AudioParameter>,
    config: &super::AudioConfig,
    prefix: &str,
    shape: Vec<usize>,
    input: usize,
) -> Result<(), String> {
    let weight = format!("{prefix}.linear.weight");
    add_matrix(
        output,
        &weight,
        shape,
        config.linear_format_for(&weight, input as i32),
    )?;
    for local in ["input_min", "input_max", "output_min", "output_max"] {
        output.push(safe(format!("{prefix}.{local}"), Vec::new()));
    }
    Ok(())
}
