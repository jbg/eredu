//! Muse-Glimmer vision tower, layerwise state, and language projection.

use std::collections::HashMap;

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        concatenate_axis,
        indexing::{NewAxis, TryIndexOp},
        mean_axis, rsqrt, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::linear::unloaded_maybe_quantized_linear,
    backend::mlx::runtime::{
        cache::ConcatKeyValueCache, checkpoint::quantization::WeightQuantization,
    },
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum VisionAttentionPolicy {
    Window,
    Full,
}

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub(crate) hidden_size: i32,
    pub(crate) intermediate_size: i32,
    pub(crate) num_heads: i32,
    pub(crate) patch_size: i32,
    pub(crate) temporal_patch_size: i32,
    pub(crate) merge_size: i32,
    pub(crate) pos_height: i32,
    pub(crate) pos_width: i32,
    pub(crate) layer_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) schedule: Vec<VisionAttentionPolicy>,
    pub(crate) quantized_weight_configs: HashMap<String, WeightQuantization>,
    pub(crate) language_hidden_size: i32,
    pub(crate) projector_hidden_size: i32,
}

#[derive(Deserialize)]
struct VisionConfigSource {
    model_type: String,
    hidden_size: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    num_hidden_layers: i32,
    patch_size: i32,
    patch_temporal: i32,
    merge_size: i32,
    pos_emb_height: i32,
    pos_emb_width: i32,
    max_position_embeddings: i32,
    layer_norm_eps: f32,
    hidden_act: String,
    layer_types: Vec<String>,
    rope_parameters: HashMap<String, Value>,
}

impl VisionConfig {
    pub(crate) fn from_hf_value(value: &Value, language_hidden_size: i32) -> Result<Self, Error> {
        let source: VisionConfigSource =
            serde_json::from_value(value.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!(
                    "invalid Muse-Glimmer vision_config: {error}"
                ))
            })?;
        if source.model_type != "muse_glimmer_vision" {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer vision model_type must be muse_glimmer_vision, got {:?}",
                source.model_type
            )));
        }
        if source.hidden_act != "gelu"
            || source.hidden_size <= 0
            || source.intermediate_size <= 0
            || source.num_attention_heads <= 0
            || source.hidden_size % source.num_attention_heads != 0
            || source.patch_size <= 0
            || source.patch_temporal <= 0
            || source.merge_size <= 0
            || source.pos_emb_height <= 0
            || source.pos_emb_width <= 0
            || source.pos_emb_height * source.pos_emb_width != source.max_position_embeddings
            || source.layer_norm_eps <= 0.0
        {
            return Err(Error::UnsupportedArchitecture(
                "Muse-Glimmer vision geometry or activation is invalid".into(),
            ));
        }
        let layer_count = usize::try_from(source.num_hidden_layers)
            .ok()
            .filter(|count| *count > 0)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Muse-Glimmer vision num_hidden_layers must be positive".into(),
                )
            })?;
        if source.layer_types.len() != layer_count {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer vision layer_types has {} entries, expected {layer_count}",
                source.layer_types.len()
            )));
        }
        let schedule = source
            .layer_types
            .iter()
            .enumerate()
            .map(|(index, kind)| match kind.as_str() {
                "window_attention" => Ok(VisionAttentionPolicy::Window),
                "full_attention" => Ok(VisionAttentionPolicy::Full),
                _ => Err(Error::UnsupportedArchitecture(format!(
                    "Muse-Glimmer vision layer {index} has unsupported type {kind:?}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rope_theta = source
            .rope_parameters
            .get("rope_theta")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Muse-Glimmer vision rope_parameters.rope_theta must be positive".into(),
                )
            })?;
        if source
            .rope_parameters
            .get("rope_type")
            .and_then(Value::as_str)
            != Some("default")
        {
            return Err(Error::UnsupportedArchitecture(
                "Muse-Glimmer vision supports only default 2-D RoPE".into(),
            ));
        }
        Ok(Self {
            hidden_size: source.hidden_size,
            intermediate_size: source.intermediate_size,
            num_heads: source.num_attention_heads,
            patch_size: source.patch_size,
            temporal_patch_size: source.patch_temporal,
            merge_size: source.merge_size,
            pos_height: source.pos_emb_height,
            pos_width: source.pos_emb_width,
            layer_norm_eps: source.layer_norm_eps,
            rope_theta,
            schedule,
            quantized_weight_configs: HashMap::new(),
            language_hidden_size,
            projector_hidden_size: 4096,
        })
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.schedule.len()
    }

    /// Reconstructs the image-only tower described by the official Muse-Glimmer
    /// projector GGUF. The converter folds the two temporal patch slices into
    /// one image patch matrix, so this source deliberately uses a temporal
    /// extent of one and is never admitted for video.
    pub(crate) fn from_gguf_metadata(
        metadata: &HashMap<String, GgufMetadataValue>,
        language_hidden_size: i32,
    ) -> Result<Self, Error> {
        let string = |key: &str| {
            metadata
                .get(key)
                .and_then(GgufMetadataValue::as_str)
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Muse-Glimmer projector GGUF is missing string metadata {key:?}"
                    ))
                })
        };
        let integer = |key: &str| {
            metadata
                .get(key)
                .and_then(GgufMetadataValue::as_i64)
                .and_then(|value| i32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Muse-Glimmer projector GGUF requires positive integer metadata {key:?}"
                    ))
                })
        };
        let float = |key: &str| {
            metadata
                .get(key)
                .and_then(GgufMetadataValue::as_f32)
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Muse-Glimmer projector GGUF requires positive float metadata {key:?}"
                    ))
                })
        };
        if string("general.architecture")? != "clip"
            || string("general.type")? != "mmproj"
            || string("clip.projector_type")? != "muse-glimmer"
            || metadata
                .get("clip.has_vision_encoder")
                .and_then(|value| match value {
                    GgufMetadataValue::Bool(value) => Some(*value),
                    _ => None,
                })
                != Some(true)
        {
            return Err(Error::UnsupportedArchitecture(
                "GGUF sidecar is not a Muse-Glimmer vision projector".into(),
            ));
        }
        let hidden_size = integer("clip.vision.embedding_length")?;
        let intermediate_size = integer("clip.vision.feed_forward_length")?;
        let layer_count = integer("clip.vision.block_count")?;
        let num_heads = integer("clip.vision.attention.head_count")?;
        let patch_size = integer("clip.vision.patch_size")?;
        let merge_size = integer("clip.vision.spatial_merge_size")?;
        let image_size = integer("clip.vision.image_size")?;
        let projection_dim = integer("clip.vision.projection_dim")?;
        if hidden_size != 1536
            || intermediate_size != 8960
            || layer_count != 50
            || num_heads != 16
            || patch_size != 14
            || merge_size != 2
            || image_size != 896
            || projection_dim != language_hidden_size
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer projector GGUF geometry is incompatible: hidden={hidden_size}, intermediate={intermediate_size}, layers={layer_count}, heads={num_heads}, patch={patch_size}, merge={merge_size}, image={image_size}, projection={projection_dim}, language={language_hidden_size}"
            )));
        }
        let schedule = (0..layer_count)
            .map(|index| {
                if index == layer_count - 1 || index % 4 == 3 {
                    VisionAttentionPolicy::Full
                } else {
                    VisionAttentionPolicy::Window
                }
            })
            .collect();
        Ok(Self {
            hidden_size,
            intermediate_size,
            num_heads,
            patch_size,
            // The official GGUF conversion sums the two temporal slices.
            temporal_patch_size: 1,
            merge_size,
            pos_height: 32,
            pos_width: 32,
            layer_norm_eps: float("clip.vision.attention.layer_norm_epsilon")?,
            rope_theta: 10_000.0,
            schedule,
            quantized_weight_configs: HashMap::new(),
            language_hidden_size,
            projector_hidden_size: 4096,
        })
    }

    pub(crate) fn apply_load_time_quantization(&mut self, quantization: WeightQuantization) {
        let aligned = |input: i32| input > 0 && input % quantization.group_size() == 0;
        self.quantized_weight_configs.clear();
        for index in 0..self.layer_count() {
            for (name, input) in [
                (
                    format!("vision_tower.layers.{index}.attn.q_proj.weight"),
                    self.hidden_size,
                ),
                (
                    format!("vision_tower.layers.{index}.attn.k_proj.weight"),
                    self.hidden_size,
                ),
                (
                    format!("vision_tower.layers.{index}.attn.v_proj.weight"),
                    self.hidden_size,
                ),
                (
                    format!("vision_tower.layers.{index}.attn.proj.weight"),
                    self.hidden_size,
                ),
                (
                    format!("vision_tower.layers.{index}.mlp.fc1.weight"),
                    self.hidden_size,
                ),
                (
                    format!("vision_tower.layers.{index}.mlp.fc2.weight"),
                    self.intermediate_size,
                ),
            ] {
                if aligned(input) {
                    self.quantized_weight_configs.insert(name, quantization);
                }
            }
        }
        for (name, input) in [
            (
                "vision_tower.patch_embedder.patch_embedding.weight",
                self.temporal_patch_size * 3 * self.patch_size * self.patch_size,
            ),
            (
                "vision_adapter.fc1.weight",
                self.hidden_size * self.merge_size * self.merge_size,
            ),
            ("vision_adapter.fc2.weight", self.projector_hidden_size),
            ("vision_projection.weight", self.projector_hidden_size),
        ] {
            if aligned(input) {
                self.quantized_weight_configs
                    .insert(name.into(), quantization);
            }
        }
    }

    fn quantization(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs.get(name).copied()
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct CenteredLayerNorm {
    #[param]
    pub(crate) weight: Param<Array>,
    #[param]
    pub(crate) bias: Param<Array>,
    eps: f32,
}

impl CenteredLayerNorm {
    fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::unloaded(&[dim], Dtype::Float32, stream)?,
            bias: Param::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    fn forward(&self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mean = mean_axis(input, -1, true, stream)?;
        let centered = input.subtract(mean, stream)?;
        let variance = mean_axis(&centered.square(stream)?, -1, true, stream)?;
        centered
            .multiply(
                rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
                stream,
            )?
            .multiply(&*self.weight, stream)?
            .add(&*self.bias, stream)?
            .as_dtype(input.dtype(), stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct PatchEmbedder {
    #[param]
    pub(crate) patch_embedding: MaybeQuantized<nn::Linear>,
    #[param]
    pub(crate) position_embedding_table: nn::Embedding,
    input_dim: i32,
}

impl PatchEmbedder {
    fn new(config: &VisionConfig, stream: &Stream) -> Result<Self, Exception> {
        let input_dim = config.temporal_patch_size * 3 * config.patch_size * config.patch_size;
        Ok(Self {
            patch_embedding: unloaded_maybe_quantized_linear(
                input_dim,
                config.hidden_size,
                false,
                config.quantization("vision_tower.patch_embedder.patch_embedding.weight"),
                stream,
            )?,
            position_embedding_table: nn::Embedding::new(
                config.pos_height * config.pos_width,
                config.hidden_size,
            )?,
            input_dim,
        })
    }

    fn forward(
        &mut self,
        pixels: &Array,
        grid: &[(i32, i32, i32)],
        config: &VisionConfig,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if pixels.shape().len() != 2 || pixels.dim(1) != self.input_dim {
            return Err(Exception::custom(format!(
                "Muse-Glimmer vision pixels must be shaped [patches, {}], got {:?}",
                self.input_dim,
                pixels.shape()
            )));
        }
        let hidden = self.patch_embedding.forward(pixels, stream)?;
        let positions = interpolated_positions(
            &mut self.position_embedding_table,
            grid,
            config.pos_height,
            config.pos_width,
            stream,
        )?;
        hidden.add(positions.as_dtype(hidden.dtype(), stream)?, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionTowerStatic {
    #[param]
    pub(crate) patch_embedder: PatchEmbedder,
    #[param]
    pub(crate) ln_pre: CenteredLayerNorm,
    #[param]
    pub(crate) ln_post: CenteredLayerNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionAdapter {
    #[param]
    pub(crate) fc1: MaybeQuantized<nn::Linear>,
    #[param]
    pub(crate) fc2: MaybeQuantized<nn::Linear>,
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionStatic {
    #[param]
    pub(crate) vision_tower: VisionTowerStatic,
    #[param]
    pub(crate) vision_adapter: VisionAdapter,
    #[param]
    pub(crate) vision_projection: MaybeQuantized<nn::Linear>,
    pub(crate) config: VisionConfig,
}

impl VisionStatic {
    pub(crate) fn new(
        mut config: VisionConfig,
        projector_hidden: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        config.projector_hidden_size = projector_hidden;
        let shuffled = config.hidden_size * config.merge_size * config.merge_size;
        Ok(Self {
            vision_tower: VisionTowerStatic {
                patch_embedder: PatchEmbedder::new(&config, stream)?,
                ln_pre: CenteredLayerNorm::new(config.hidden_size, config.layer_norm_eps, stream)?,
                ln_post: CenteredLayerNorm::new(config.hidden_size, config.layer_norm_eps, stream)?,
            },
            vision_adapter: VisionAdapter {
                fc1: unloaded_maybe_quantized_linear(
                    shuffled,
                    projector_hidden,
                    false,
                    config.quantization("vision_adapter.fc1.weight"),
                    stream,
                )?,
                fc2: unloaded_maybe_quantized_linear(
                    projector_hidden,
                    projector_hidden,
                    false,
                    config.quantization("vision_adapter.fc2.weight"),
                    stream,
                )?,
            },
            vision_projection: unloaded_maybe_quantized_linear(
                projector_hidden,
                config.language_hidden_size,
                false,
                config.quantization("vision_projection.weight"),
                stream,
            )?,
            config,
        })
    }

    pub(crate) fn begin(
        &mut self,
        pixels: &Array,
        grid_array: &Array,
        stream: &Stream,
    ) -> Result<(Array, VisionState), Exception> {
        let grid = grid_from_array(grid_array, stream)?;
        validate_grid(&grid, pixels, self.config.merge_size)?;
        let hidden =
            self.vision_tower
                .patch_embedder
                .forward(pixels, &grid, &self.config, stream)?;
        let hidden = self.vision_tower.ln_pre.forward(&hidden, stream)?;
        let full_chunks = full_chunk_lengths(&grid);
        let (window_index, window_chunks) =
            crate::backend::mlx::architectures::qwen::vl::vision::vision_window_index(
                &grid,
                1,
                self.config.pos_height * self.config.patch_size,
                self.config.patch_size,
            )?;
        let index = Array::from_slice(&window_index, &[window_index.len() as i32]);
        let hidden = hidden.try_index_device((&index, ..), stream)?;
        let (cos, sin) = rotary_embeddings(&grid, &self.config, stream)?;
        Ok((
            hidden,
            VisionState {
                grid,
                full_chunks,
                window_chunks,
                window_index,
                cos: cos.try_index_device((&index, ..), stream)?,
                sin: sin.try_index_device((&index, ..), stream)?,
            },
        ))
    }

    /// Reconstructs parameter-free placement state on a downstream pipeline
    /// owner. The encoded activation itself is supplied by pipeline transport,
    /// so patch embedding is deliberately not repeated here.
    pub(crate) fn continuation_state(
        &self,
        grid_array: &Array,
        stream: &Stream,
    ) -> Result<VisionState, Exception> {
        let grid = grid_from_array(grid_array, stream)?;
        if grid.is_empty()
            || grid.iter().any(|&(t, h, w)| {
                t <= 0
                    || h <= 0
                    || w <= 0
                    || h % self.config.merge_size != 0
                    || w % self.config.merge_size != 0
            })
        {
            return Err(Exception::custom(
                "Muse-Glimmer continuation grid must be positive and spatially divisible by merge_size",
            ));
        }
        let full_chunks = full_chunk_lengths(&grid);
        let (window_index, window_chunks) =
            crate::backend::mlx::architectures::qwen::vl::vision::vision_window_index(
                &grid,
                1,
                self.config.pos_height * self.config.patch_size,
                self.config.patch_size,
            )?;
        let index = Array::from_slice(&window_index, &[window_index.len() as i32]);
        let (cos, sin) = rotary_embeddings(&grid, &self.config, stream)?;
        Ok(VisionState {
            grid,
            full_chunks,
            window_chunks,
            window_index,
            cos: cos.try_index_device((&index, ..), stream)?,
            sin: sin.try_index_device((&index, ..), stream)?,
        })
    }

    pub(crate) fn forward_block(
        &self,
        block: &mut VisionBlock,
        index: usize,
        hidden: &Array,
        state: &VisionState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let chunks = match self.config.schedule.get(index) {
            Some(VisionAttentionPolicy::Full) => &state.full_chunks,
            Some(VisionAttentionPolicy::Window) => &state.window_chunks,
            None => {
                return Err(Exception::custom(format!(
                    "Muse-Glimmer vision layer {index} is outside the schedule"
                )))
            }
        };
        block.forward(hidden, chunks, &state.cos, &state.sin, stream)
    }

    pub(crate) fn forward_block_tensor_parallel(
        &self,
        block: &mut VisionBlock,
        index: usize,
        hidden: &Array,
        state: &VisionState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let chunks = match self.config.schedule.get(index) {
            Some(VisionAttentionPolicy::Full) => &state.full_chunks,
            Some(VisionAttentionPolicy::Window) => &state.window_chunks,
            None => {
                return Err(Exception::custom(format!(
                    "Muse-Glimmer vision layer {index} is outside the schedule"
                )))
            }
        };
        block.forward_tensor_parallel(hidden, chunks, &state.cos, &state.sin, group, stream)
    }

    pub(crate) fn finish(
        &mut self,
        hidden: &Array,
        state: &VisionState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let reverse = crate::backend::mlx::architectures::qwen::vl::vision::reverse_permutation(
            &state.window_index,
        );
        let reverse = Array::from_slice(&reverse, &[reverse.len() as i32]);
        let hidden = hidden.try_index_device((&reverse, ..), stream)?;
        let hidden = self.vision_tower.ln_post.forward(&hidden, stream)?;
        let hidden = pixel_shuffle(&hidden, &state.grid, self.config.merge_size, stream)?;
        let hidden = nn::gelu(self.vision_adapter.fc1.forward(&hidden, stream)?, stream)?;
        let hidden = nn::gelu(self.vision_adapter.fc2.forward(&hidden, stream)?, stream)?;
        let hidden = self.vision_projection.forward(&hidden, stream)?;
        weightless_rms_norm(&hidden, 1e-5, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionAttention {
    #[param]
    pub(crate) q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub(crate) k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub(crate) v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub(crate) proj: MaybeQuantized<nn::Linear>,
    heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn new(config: &VisionConfig, index: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_heads(config, index, config.num_heads, stream)
    }

    fn new_with_heads(
        config: &VisionConfig,
        index: usize,
        heads: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("vision_tower.layers.{index}.attn");
        let head_dim = config.hidden_size / config.num_heads;
        let local_width = heads * head_dim;
        let column = |name: &str| {
            unloaded_maybe_quantized_linear(
                config.hidden_size,
                local_width,
                true,
                config.quantization(&format!("{prefix}.{name}.weight")),
                stream,
            )
        };
        Ok(Self {
            q_proj: column("q_proj")?,
            k_proj: column("k_proj")?,
            v_proj: column("v_proj")?,
            proj: unloaded_maybe_quantized_linear(
                local_width,
                config.hidden_size,
                true,
                config.quantization(&format!("{prefix}.proj.weight")),
                stream,
            )?,
            heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        chunks: &[i32],
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let seq = hidden.dim(0);
        let mut q = self
            .q_proj
            .forward(hidden, stream)?
            .reshape(&[seq, self.heads, self.head_dim], stream)?;
        let mut k = self
            .k_proj
            .forward(hidden, stream)?
            .reshape(&[seq, self.heads, self.head_dim], stream)?;
        let v = self
            .v_proj
            .forward(hidden, stream)?
            .reshape(&[seq, self.heads, self.head_dim], stream)?;
        (q, k) = apply_rotary(q, k, cos, sin, stream)?;
        let mut outputs = Vec::with_capacity(chunks.len());
        let mut start = 0;
        for &length in chunks {
            let end = start + length;
            let prepare = |value: &Array| -> Result<Array, Exception> {
                value
                    .try_index_device((start..end, .., ..), stream)?
                    .transpose_axes(&[1, 0, 2], stream)?
                    .try_index_device((NewAxis, .., .., ..), stream)
            };
            let output = crate::backend::mlx::nn::tensor::scaled_dot_product_attention(
                prepare(&q)?,
                prepare(&k)?,
                prepare(&v)?,
                Option::<ConcatKeyValueCache>::None,
                self.scale,
                None,
                stream,
            )?
            .try_index_device((0, .., .., ..), stream)?
            .transpose_axes(&[1, 0, 2], stream)?
            .reshape(&[length, self.heads * self.head_dim], stream)?;
            outputs.push(output);
            start = end;
        }
        if start != seq {
            return Err(Exception::custom(format!(
                "Muse-Glimmer vision chunks cover {start} tokens, expected {seq}"
            )));
        }
        self.proj
            .forward(&concatenate_axis(&outputs, 0, stream)?, stream)
    }

    fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        chunks: &[i32],
        cos: &Array,
        sin: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut partial = self.forward(hidden, chunks, cos, sin, stream)?;
        if let Some(bias) = linear_bias(&self.proj) {
            partial = partial.subtract(bias, stream)?;
        }
        let mut output = safemlx::distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = linear_bias(&self.proj) {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionMlp {
    #[param]
    pub(crate) fc1: MaybeQuantized<nn::Linear>,
    #[param]
    pub(crate) fc2: MaybeQuantized<nn::Linear>,
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionBlock {
    #[param]
    pub(crate) norm1: CenteredLayerNorm,
    #[param]
    pub(crate) attn: VisionAttention,
    #[param]
    pub(crate) norm2: CenteredLayerNorm,
    #[param]
    pub(crate) mlp: VisionMlp,
}

impl VisionBlock {
    pub(crate) fn new(
        config: &VisionConfig,
        index: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("vision_tower.layers.{index}.mlp");
        Ok(Self {
            norm1: CenteredLayerNorm::new(config.hidden_size, config.layer_norm_eps, stream)?,
            attn: VisionAttention::new(config, index, stream)?,
            norm2: CenteredLayerNorm::new(config.hidden_size, config.layer_norm_eps, stream)?,
            mlp: VisionMlp {
                fc1: unloaded_maybe_quantized_linear(
                    config.hidden_size,
                    config.intermediate_size,
                    true,
                    config.quantization(&format!("{prefix}.fc1.weight")),
                    stream,
                )?,
                fc2: unloaded_maybe_quantized_linear(
                    config.intermediate_size,
                    config.hidden_size,
                    true,
                    config.quantization(&format!("{prefix}.fc2.weight")),
                    stream,
                )?,
            },
        })
    }

    pub(crate) fn new_tensor_parallel(
        config: &VisionConfig,
        index: usize,
        local_heads: i32,
        local_intermediate: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("vision_tower.layers.{index}.mlp");
        Ok(Self {
            norm1: CenteredLayerNorm::new(config.hidden_size, config.layer_norm_eps, stream)?,
            attn: VisionAttention::new_with_heads(config, index, local_heads, stream)?,
            norm2: CenteredLayerNorm::new(config.hidden_size, config.layer_norm_eps, stream)?,
            mlp: VisionMlp {
                fc1: unloaded_maybe_quantized_linear(
                    config.hidden_size,
                    local_intermediate,
                    true,
                    config.quantization(&format!("{prefix}.fc1.weight")),
                    stream,
                )?,
                fc2: unloaded_maybe_quantized_linear(
                    local_intermediate,
                    config.hidden_size,
                    true,
                    config.quantization(&format!("{prefix}.fc2.weight")),
                    stream,
                )?,
            },
        })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        chunks: &[i32],
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let norm = self.norm1.forward(hidden, stream)?;
        let hidden = hidden.add(self.attn.forward(&norm, chunks, cos, sin, stream)?, stream)?;
        let norm = self.norm2.forward(&hidden, stream)?;
        let mlp = self.mlp.fc1.forward(&norm, stream)?;
        let mlp = nn::gelu(mlp, stream)?;
        hidden.add(self.mlp.fc2.forward(&mlp, stream)?, stream)
    }

    fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        chunks: &[i32],
        cos: &Array,
        sin: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let norm = self.norm1.forward(hidden, stream)?;
        let attention = self
            .attn
            .forward_tensor_parallel(&norm, chunks, cos, sin, group, stream)?;
        let hidden = hidden.add(attention, stream)?;
        let norm = self.norm2.forward(&hidden, stream)?;
        let mlp = nn::gelu(self.mlp.fc1.forward(&norm, stream)?, stream)?;
        let mut partial = self.mlp.fc2.forward(&mlp, stream)?;
        if let Some(bias) = linear_bias(&self.mlp.fc2) {
            partial = partial.subtract(bias, stream)?;
        }
        let mut output = safemlx::distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = linear_bias(&self.mlp.fc2) {
            output = output.add(bias, stream)?;
        }
        hidden.add(output, stream)
    }
}

fn linear_bias(linear: &MaybeQuantized<nn::Linear>) -> Option<&Array> {
    match linear {
        MaybeQuantized::Original(linear) => linear.bias.as_ref().as_ref(),
        MaybeQuantized::Quantized(linear) => linear.inner.bias.as_ref().as_ref(),
    }
}

pub(crate) struct VisionState {
    grid: Vec<(i32, i32, i32)>,
    full_chunks: Vec<i32>,
    window_chunks: Vec<i32>,
    window_index: Vec<i32>,
    cos: Array,
    sin: Array,
}

fn grid_from_array(array: &Array, stream: &Stream) -> Result<Vec<(i32, i32, i32)>, Exception> {
    crate::backend::mlx::architectures::qwen::vl::vision::grid_thw_from_array(array, stream)
}

fn validate_grid(grid: &[(i32, i32, i32)], pixels: &Array, merge: i32) -> Result<(), Exception> {
    let patches: i32 = grid.iter().map(|(t, h, w)| t * h * w).sum();
    if patches != pixels.dim(0) {
        return Err(Exception::custom(format!(
            "Muse-Glimmer vision grid describes {patches} patches, tensor has {}",
            pixels.dim(0)
        )));
    }
    if grid.is_empty()
        || grid
            .iter()
            .any(|&(t, h, w)| t <= 0 || h <= 0 || w <= 0 || h % merge != 0 || w % merge != 0)
    {
        return Err(Exception::custom(
            "Muse-Glimmer vision grid must be positive and spatially divisible by merge_size",
        ));
    }
    Ok(())
}

fn full_chunk_lengths(grid: &[(i32, i32, i32)]) -> Vec<i32> {
    grid.iter()
        .flat_map(|&(t, h, w)| std::iter::repeat_n(h * w, t as usize))
        .collect()
}

fn interpolated_positions(
    embedding: &mut nn::Embedding,
    grid: &[(i32, i32, i32)],
    side_h: i32,
    side_w: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let mut indices = [Vec::<u32>::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut weights = [Vec::<f32>::new(), Vec::new(), Vec::new(), Vec::new()];
    for &(t, h, w) in grid {
        for _ in 0..t {
            for y in 0..h {
                let gy = (y as f32 + 0.5) * side_h as f32 / h as f32 - 0.5;
                let y0_raw = gy.floor() as i32;
                let y1_raw = y0_raw + 1;
                let yf = gy - y0_raw as f32;
                for x in 0..w {
                    let gx = (x as f32 + 0.5) * side_w as f32 / w as f32 - 0.5;
                    let x0_raw = gx.floor() as i32;
                    let x1_raw = x0_raw + 1;
                    let xf = gx - x0_raw as f32;
                    for (corner, yy, xx, weight) in [
                        (0, y0_raw, x0_raw, (1.0 - yf) * (1.0 - xf)),
                        (1, y0_raw, x1_raw, (1.0 - yf) * xf),
                        (2, y1_raw, x0_raw, yf * (1.0 - xf)),
                        (3, y1_raw, x1_raw, yf * xf),
                    ] {
                        let valid = yy >= 0 && yy < side_h && xx >= 0 && xx < side_w;
                        let yy = yy.clamp(0, side_h - 1);
                        let xx = xx.clamp(0, side_w - 1);
                        indices[corner].push((yy * side_w + xx) as u32);
                        weights[corner].push(if valid { weight } else { 0.0 });
                    }
                }
            }
        }
    }
    let len = indices[0].len() as i32;
    let mut output: Option<Array> = None;
    for corner in 0..4 {
        let index = Array::from_slice(&indices[corner], &[len]);
        let weight = Array::from_slice(&weights[corner], &[len, 1]);
        let value = embedding
            .forward(&index, stream)?
            .multiply(weight, stream)?;
        output = Some(match output {
            Some(current) => current.add(value, stream)?,
            None => value,
        });
    }
    output.ok_or_else(|| Exception::custom("Muse-Glimmer vision grid is empty"))
}

fn rotary_embeddings(
    grid: &[(i32, i32, i32)],
    config: &VisionConfig,
    _stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let head_dim = config.hidden_size / config.num_heads;
    let spatial_dim = head_dim / 2;
    if spatial_dim % 2 != 0 {
        return Err(Exception::custom(
            "Muse-Glimmer vision RoPE spatial dimension must be even",
        ));
    }
    let inv = (0..spatial_dim)
        .step_by(2)
        .map(|index| 1.0 / config.rope_theta.powf(index as f32 / spatial_dim as f32))
        .collect::<Vec<_>>();
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for &(t, h, w) in grid {
        for _ in 0..t {
            for y in 0..h {
                for x in 0..w {
                    let fw = inv
                        .iter()
                        .map(|value| (x + 1) as f32 * value)
                        .collect::<Vec<_>>();
                    let fh = inv
                        .iter()
                        .map(|value| (y + 1) as f32 * value)
                        .collect::<Vec<_>>();
                    for angle in fw.iter().chain(&fh).chain(&fw).chain(&fh) {
                        cos.push(angle.cos());
                        sin.push(angle.sin());
                    }
                }
            }
        }
    }
    let seq = cos.len() as i32 / head_dim;
    Ok((
        Array::from_slice(&cos, &[seq, head_dim]),
        Array::from_slice(&sin, &[seq, head_dim]),
    ))
}

fn rotate_half(value: &Array, stream: &Stream) -> Result<Array, Exception> {
    let half = value.dim(-1) / 2;
    let first = value.try_index_device((.., .., ..half), stream)?;
    let second = value.try_index_device((.., .., half..), stream)?;
    concatenate_axis(
        &[second.multiply(Array::from_f32(-1.0), stream)?, first],
        -1,
        stream,
    )
}

fn apply_rotary(
    q: Array,
    k: Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let cos = cos
        .as_dtype(Dtype::Float32, stream)?
        .try_index_device((.., NewAxis, ..), stream)?;
    let sin = sin
        .as_dtype(Dtype::Float32, stream)?
        .try_index_device((.., NewAxis, ..), stream)?;
    let rotate = |value: Array, dtype: Dtype| -> Result<Array, Exception> {
        let value = value.as_dtype(Dtype::Float32, stream)?;
        value
            .multiply(&cos, stream)?
            .add(rotate_half(&value, stream)?.multiply(&sin, stream)?, stream)?
            .as_dtype(dtype, stream)
    };
    Ok((rotate(q, q_dtype)?, rotate(k, k_dtype)?))
}

fn pixel_shuffle(
    hidden: &Array,
    grid: &[(i32, i32, i32)],
    factor: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dim = hidden.dim(-1);
    let mut offset = 0;
    let mut outputs = Vec::new();
    for &(t, h, w) in grid {
        let count = t * h * w;
        let chunk = hidden.try_index_device((offset..offset + count, ..), stream)?;
        let mut permutation = Vec::with_capacity(count as usize);
        for frame in 0..t {
            for outer_y in 0..h / factor {
                for outer_x in 0..w / factor {
                    for inner_y in 0..factor {
                        for inner_x in 0..factor {
                            permutation.push(
                                frame * h * w
                                    + (outer_y * factor + inner_y) * w
                                    + outer_x * factor
                                    + inner_x,
                            );
                        }
                    }
                }
            }
        }
        let index = Array::from_slice(&permutation, &[count]);
        let output_count = t * (h / factor) * (w / factor);
        outputs.push(
            chunk
                .try_index_device((&index, ..), stream)?
                .reshape(&[output_count, factor * factor, dim], stream)?
                .transpose_axes(&[0, 2, 1], stream)?
                .reshape(&[output_count, dim * factor * factor], stream)?,
        );
        offset += count;
    }
    concatenate_axis(&outputs, 0, stream)
}

fn weightless_rms_norm(hidden: &Array, eps: f32, stream: &Stream) -> Result<Array, Exception> {
    super::rms_norm_without_scale(hidden, eps, stream)
}
