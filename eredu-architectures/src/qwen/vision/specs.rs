//! Parameter identities and geometry shared by vision declarations and native construction.

use super::VisionConfig;
use eredu_nn::{multimodal::FlattenedPatchSpec, EmbeddingSpec, Error, LinearSpec, ParameterSpec};

pub(super) fn rooted(root: &str, relative: &str) -> String {
    if root.is_empty() {
        relative.to_owned()
    } else {
        format!("{root}.{relative}")
    }
}

pub(super) struct LayerNormSpec {
    pub dimensions: i32,
    pub weight: ParameterSpec,
    pub bias: ParameterSpec,
}

pub(super) fn norm_spec(prefix: &str, dimensions: i32) -> Result<LayerNormSpec, Error> {
    Ok(LayerNormSpec {
        dimensions,
        weight: ParameterSpec::trainable(format!("{prefix}.weight")).map_err(Error::backend)?,
        bias: ParameterSpec::trainable(format!("{prefix}.bias")).map_err(Error::backend)?,
    })
}

pub(super) struct PatchSpec {
    pub geometry: FlattenedPatchSpec,
    pub weight: ParameterSpec,
    pub bias: ParameterSpec,
}
impl PatchSpec {
    pub fn weight_shape(&self) -> [i32; 5] {
        [
            self.geometry.output,
            self.geometry.channels,
            self.geometry.temporal,
            self.geometry.height,
            self.geometry.width,
        ]
    }
}
pub(super) fn patch_spec(config: &VisionConfig, root: &str) -> Result<PatchSpec, Error> {
    Ok(PatchSpec {
        geometry: FlattenedPatchSpec {
            channels: config.in_channels,
            temporal: config.temporal_patch_size,
            height: config.patch_size,
            width: config.patch_size,
            output: config.hidden_size,
        },
        weight: ParameterSpec::trainable(rooted(root, "patch_embed.proj.weight"))
            .map_err(Error::backend)?,
        bias: ParameterSpec::trainable(rooted(root, "patch_embed.proj.bias"))
            .map_err(Error::backend)?,
    })
}

pub(super) fn position_spec(config: &VisionConfig, root: &str) -> Result<EmbeddingSpec, Error> {
    let name = rooted(root, "pos_embed.weight");
    Ok(EmbeddingSpec {
        vocabulary: config.num_position_embeddings,
        dimensions: config.hidden_size,
        weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(
            &name,
            eredu_checkpoint::LinearFormat::Dense,
        )?,
    })
}

fn linear_spec(
    config: &VisionConfig,
    prefix: &str,
    input: i32,
    output: i32,
) -> Result<LinearSpec, Error> {
    let weight = format!("{prefix}.weight");
    let relative = ["blocks.", "merger.", "deepstack_merger_list."]
        .into_iter()
        .find_map(|marker| weight.find(marker).map(|index| &weight[index..]))
        .unwrap_or(&weight);
    Ok(LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
        bias: Some(ParameterSpec::trainable(format!("{prefix}.bias")).map_err(Error::backend)?),
        format: crate::linear_format::standard_linear_format(
            &weight,
            config.linear_format(relative),
        )?,
    })
}

pub(super) fn block_projection_specs(
    config: &VisionConfig,
    root: &str,
    layer: usize,
    heads: i32,
    intermediate: i32,
) -> Result<[LinearSpec; 4], Error> {
    let prefix = rooted(root, &format!("blocks.{layer}"));
    let width = heads
        .checked_mul(
            config
                .hidden_size
                .checked_div(config.num_heads)
                .ok_or_else(|| Error::backend("vision head geometry is invalid"))?,
        )
        .ok_or_else(|| Error::backend("vision attention width overflowed"))?;
    Ok([
        linear_spec(
            config,
            &format!("{prefix}.attn.qkv"),
            config.hidden_size,
            width
                .checked_mul(3)
                .ok_or_else(|| Error::backend("vision QKV width overflowed"))?,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.attn.proj"),
            width,
            config.hidden_size,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.mlp.linear_fc1"),
            config.hidden_size,
            intermediate,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.mlp.linear_fc2"),
            intermediate,
            config.hidden_size,
        )?,
    ])
}

pub(super) fn merger_geometry(
    config: &VisionConfig,
    postshuffle: bool,
) -> Result<(i32, i32, i32), Error> {
    let unit = config
        .spatial_merge_size
        .checked_mul(config.spatial_merge_size)
        .ok_or_else(|| Error::backend("vision merge area overflowed"))?;
    let width = config
        .hidden_size
        .checked_mul(unit)
        .ok_or_else(|| Error::backend("vision merger width overflowed"))?;
    Ok((
        unit,
        width,
        if postshuffle {
            width
        } else {
            config.hidden_size
        },
    ))
}

pub(super) fn merger_projection_specs(
    config: &VisionConfig,
    prefix: &str,
    intermediate: i32,
) -> Result<[LinearSpec; 2], Error> {
    let (_, width, _) = merger_geometry(config, false)?;
    Ok([
        linear_spec(config, &format!("{prefix}.linear_fc1"), width, intermediate)?,
        linear_spec(
            config,
            &format!("{prefix}.linear_fc2"),
            intermediate,
            config.out_hidden_size,
        )?,
    ])
}
