//! Backend-neutral Muse-Glimmer vision tower and language adapter.

use eredu_nn::{
    multimodal::{
        multi_axis_rotary_embeddings, MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    },
    sequence_layout::{
        attention_chunk_lengths, bilinear_interpolation_samples, inverse_permutation,
        patch_positions, validate_patch_grid, window_partition, InterpolationMode, PatchTraversal,
    },
    EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator, LinearSpec, NeuralBackend,
    Parameter, ParameterSpec, Parameterized, Tensor,
};

use super::{VisionAttentionPolicy, VisionConfig};

/// Prepared flattened RGB patches and their image/video grid metadata.
pub struct VisionInput<'a, T> {
    /// Flattened patches shaped `[patches, temporal * 3 * patch_size^2]`.
    pub pixels: &'a T,
    /// Packed media grids in `(time, height, width)` order.
    pub grid: &'a [(i32, i32, i32)],
}

/// Parameter-free placement values retained between vision blocks.
#[derive(Debug, Clone)]
pub struct VisionState<T> {
    grid: Vec<(i32, i32, i32)>,
    full_chunks: Vec<i32>,
    window_chunks: Vec<i32>,
    window_permutation: Vec<i32>,
    cosine: T,
    sine: T,
}

impl<T> VisionState<T> {
    /// Packed full-attention chunk lengths, one per temporal item.
    pub fn full_chunks(&self) -> &[i32] {
        &self.full_chunks
    }

    /// Window-attention chunk lengths after the stable window permutation.
    pub fn window_chunks(&self) -> &[i32] {
        &self.window_chunks
    }

    /// Stable permutation from raster order to window order.
    pub fn window_permutation(&self) -> &[i32] {
        &self.window_permutation
    }

    /// Returns backend tensors that must remain live while blocks are submitted.
    pub fn retained_values(&self) -> [&T; 2] {
        [&self.cosine, &self.sine]
    }

    fn chunks(&self, policy: VisionAttentionPolicy) -> &[i32] {
        match policy {
            VisionAttentionPolicy::Full => &self.full_chunks,
            VisionAttentionPolicy::Window => &self.window_chunks,
        }
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct CenteredLayerNorm<B: NeuralBackend> {
    weight: Parameter<B::Tensor>,
    bias: Parameter<B::Tensor>,
    #[parameter(skip)]
    epsilon: f32,
}

impl<B: NeuralBackend> CenteredLayerNorm<B> {
    fn new(
        prefix: &str,
        dimensions: i32,
        epsilon: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let parameter = |suffix: &str| {
            Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.{suffix}")).map_err(Error::backend)?,
                &[dimensions],
                context,
            )
        };
        Ok(Self {
            weight: parameter("weight")?,
            bias: parameter("bias")?,
            epsilon,
        })
    }

    fn forward(
        &self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::Tensor::layer_norm(
            input,
            Some(self.weight.as_ref()),
            Some(self.bias.as_ref()),
            self.epsilon,
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct PatchEmbedder<B: NeuralBackend> {
    projection: B::Linear,
    position_table: B::Embedding,
    #[parameter(skip)]
    input_width: i32,
}

impl<B: NeuralBackend> PatchEmbedder<B> {
    fn new(config: &VisionConfig, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let input_width = config.temporal_patch_size * 3 * config.patch_size * config.patch_size;
        let weight = "model.vision_tower.patch_embedder.patch_embedding.weight";
        let position = "model.vision_tower.patch_embedder.position_embedding_table.weight";
        Ok(Self {
            projection: B::linear(
                LinearSpec {
                    input: input_width,
                    output: config.hidden_size,
                    weight: ParameterSpec::trainable(weight).map_err(Error::backend)?,
                    bias: None,
                    format: config.linear_format_for(weight),
                },
                context,
            )?,
            position_table: B::embedding(
                EmbeddingSpec {
                    vocabulary: config.position_height * config.position_width,
                    dimensions: config.hidden_size,
                    weight: ParameterSpec::trainable(position).map_err(Error::backend)?,
                    quantization: config.weight_quantization_for(position),
                },
                context,
            )?,
            input_width,
        })
    }

    fn forward(
        &mut self,
        pixels: &B::Tensor,
        grid: &[(i32, i32, i32)],
        config: &VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if pixels.shape().len() != 2 || pixels.dim(1) != self.input_width {
            return Err(Error::backend(format!(
                "Muse-Glimmer vision pixels must be shaped [patches, {}], got {:?}",
                self.input_width,
                pixels.shape()
            )));
        }
        let hidden = self.projection.forward(pixels, context)?;
        hidden.add(
            &interpolated_positions::<B>(
                &mut self.position_table,
                grid,
                config.position_height,
                config.position_width,
                context,
            )?,
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct VisionAttention<B: NeuralBackend> {
    query: B::Linear,
    key: B::Linear,
    value: B::Linear,
    output: B::Linear,
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    scale: f32,
}

impl<B: NeuralBackend> VisionAttention<B> {
    fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.vision_tower.layers.{layer}.attn");
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: Some(
                        ParameterSpec::trainable(format!("{prefix}.{field}.bias"))
                            .map_err(Error::backend)?,
                    ),
                    format: config.linear_format_for(&weight),
                },
                context,
            )
        };
        let head_dimensions = config.hidden_size / config.num_heads;
        Ok(Self {
            query: linear("q_proj", config.hidden_size, config.hidden_size)?,
            key: linear("k_proj", config.hidden_size, config.hidden_size)?,
            value: linear("v_proj", config.hidden_size, config.hidden_size)?,
            output: linear("proj", config.hidden_size, config.hidden_size)?,
            heads: config.num_heads,
            head_dimensions,
            scale: (head_dimensions as f32).sqrt().recip(),
        })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cosine: &B::Tensor,
        sine: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let sequence = hidden.dim(0);
        let query = self
            .query
            .forward(hidden, context)?
            .reshape(&[sequence, self.heads, self.head_dimensions], context)?;
        let key = self
            .key
            .forward(hidden, context)?
            .reshape(&[sequence, self.heads, self.head_dimensions], context)?;
        let value = self
            .value
            .forward(hidden, context)?
            .reshape(&[sequence, self.heads, self.head_dimensions], context)?;
        let query = apply_rotary(&query, cosine, sine, context)?;
        let key = apply_rotary(&key, cosine, sine, context)?;
        let mut outputs = Vec::with_capacity(chunks.len());
        let mut start = 0;
        for &length in chunks {
            if length <= 0 || start + length > sequence {
                return Err(Error::backend(
                    "invalid Muse-Glimmer vision attention chunk",
                ));
            }
            let end = start + length;
            let prepare = |value: &B::Tensor| {
                value
                    .index(
                        &[Index::Range(start, end), Index::Full, Index::Full],
                        context,
                    )?
                    .transpose_axes(&[1, 0, 2], context)?
                    .expand_dims(0, context)
            };
            let output = B::attention(
                prepare(&query)?,
                prepare(&key)?,
                prepare(&value)?,
                self.scale,
                None,
                context,
            )?
            .squeeze_axes(&[0], context)?
            .transpose_axes(&[1, 0, 2], context)?
            .reshape(&[length, self.heads * self.head_dimensions], context)?;
            outputs.push(output);
            start = end;
        }
        if start != sequence {
            return Err(Error::backend(format!(
                "Muse-Glimmer vision chunks cover {start} tokens, expected {sequence}"
            )));
        }
        self.output
            .forward(&B::Tensor::concatenate(&outputs, 0, context)?, context)
    }
}

/// One exact Muse-Glimmer vision transformer block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionBlock<B: NeuralBackend> {
    norm1: CenteredLayerNorm<B>,
    attention: VisionAttention<B>,
    norm2: CenteredLayerNorm<B>,
    fc1: B::Linear,
    fc2: B::Linear,
}

impl<B: NeuralBackend> VisionBlock<B> {
    /// Builds one unloaded vision block.
    pub fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.vision_tower.layers.{layer}");
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.mlp.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: Some(
                        ParameterSpec::trainable(format!("{prefix}.mlp.{field}.bias"))
                            .map_err(Error::backend)?,
                    ),
                    format: config.linear_format_for(&weight),
                },
                context,
            )
        };
        Ok(Self {
            norm1: CenteredLayerNorm::new(
                &format!("{prefix}.norm1"),
                config.hidden_size,
                config.layer_norm_eps,
                context,
            )?,
            attention: VisionAttention::new(config, layer, context)?,
            norm2: CenteredLayerNorm::new(
                &format!("{prefix}.norm2"),
                config.hidden_size,
                config.layer_norm_eps,
                context,
            )?,
            fc1: linear("fc1", config.hidden_size, config.intermediate_size)?,
            fc2: linear("fc2", config.intermediate_size, config.hidden_size)?,
        })
    }

    /// Applies this block using architecture-selected attention chunks.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cosine: &B::Tensor,
        sine: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normalized = self.norm1.forward(hidden, context)?;
        let hidden = hidden.add(
            &self
                .attention
                .forward(&normalized, chunks, cosine, sine, context)?,
            context,
        )?;
        let normalized = self.norm2.forward(&hidden, context)?;
        let feed_forward = B::Tensor::gelu(&self.fc1.forward(&normalized, context)?, context)?;
        hidden.add(&self.fc2.forward(&feed_forward, context)?, context)
    }

    /// Applies this block through its normalized full/window schedule entry.
    pub fn forward_scheduled(
        &mut self,
        hidden: &B::Tensor,
        policy: VisionAttentionPolicy,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward(
            hidden,
            state.chunks(policy),
            &state.cosine,
            &state.sine,
            context,
        )
    }
}

/// Pinned patch, position, normalization, merge-adapter, and projection modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionStatic<B: NeuralBackend> {
    patch_embedder: PatchEmbedder<B>,
    pre_norm: CenteredLayerNorm<B>,
    post_norm: CenteredLayerNorm<B>,
    adapter_fc1: B::Linear,
    adapter_fc2: B::Linear,
    projection: B::Linear,
    #[parameter(skip)]
    config: VisionConfig,
}

impl<B: NeuralBackend> VisionStatic<B> {
    /// Builds unloaded pinned vision modules.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let linear = |name: &str, input, output| {
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                    bias: None,
                    format: config.linear_format_for(name),
                },
                context,
            )
        };
        let shuffled = config.hidden_size * config.merge_size * config.merge_size;
        Ok(Self {
            patch_embedder: PatchEmbedder::new(&config, context)?,
            pre_norm: CenteredLayerNorm::new(
                "model.vision_tower.ln_pre",
                config.hidden_size,
                config.layer_norm_eps,
                context,
            )?,
            post_norm: CenteredLayerNorm::new(
                "model.vision_tower.ln_post",
                config.hidden_size,
                config.layer_norm_eps,
                context,
            )?,
            adapter_fc1: linear(
                "model.vision_adapter.fc1.weight",
                shuffled,
                config.projector_hidden_size,
            )?,
            adapter_fc2: linear(
                "model.vision_adapter.fc2.weight",
                config.projector_hidden_size,
                config.projector_hidden_size,
            )?,
            projection: linear(
                "model.vision_projection.weight",
                config.projector_hidden_size,
                config.language_hidden_size,
            )?,
            config,
        })
    }

    /// Embeds and window-permutes patches and constructs reusable placement state.
    pub fn begin(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        validate_patch_grid(
            input.grid,
            self.config.merge_size,
            Some(input.pixels.dim(0)),
        )
        .map_err(Error::backend)?;
        let hidden =
            self.patch_embedder
                .forward(input.pixels, input.grid, &self.config, context)?;
        let hidden = self.pre_norm.forward(&hidden, context)?;
        let full_chunks = attention_chunk_lengths(input.grid).map_err(Error::backend)?;
        let window = window_partition(
            input.grid,
            1,
            self.config.position_height * self.config.patch_size,
            self.config.patch_size,
        )
        .map_err(Error::backend)?;
        let permutation = B::Tensor::from_i32_slice(
            &window.permutation,
            &[window.permutation.len() as i32],
            context,
        )?;
        let hidden = hidden.take_axis(&permutation, 0, context)?;
        let (cosine, sine) = rotary_embeddings::<B::Tensor>(input.grid, &self.config, context)?;
        Ok((
            hidden,
            VisionState {
                grid: input.grid.to_vec(),
                full_chunks,
                window_chunks: window.chunk_lengths,
                window_permutation: window.permutation,
                cosine: cosine.take_axis(&permutation, 0, context)?,
                sine: sine.take_axis(&permutation, 0, context)?,
            },
        ))
    }

    /// Restores raster order, pixel-shuffles, adapts, projects, and normalizes.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let reverse = inverse_permutation(&state.window_permutation).map_err(Error::backend)?;
        let reverse = B::Tensor::from_i32_slice(&reverse, &[reverse.len() as i32], context)?;
        let hidden = hidden.take_axis(&reverse, 0, context)?;
        let hidden = self.post_norm.forward(&hidden, context)?;
        let hidden = pixel_shuffle(&hidden, &state.grid, self.config.merge_size, context)?;
        let hidden = B::Tensor::gelu(&self.adapter_fc1.forward(&hidden, context)?, context)?;
        let hidden = B::Tensor::gelu(&self.adapter_fc2.forward(&hidden, context)?, context)?;
        let hidden = self.projection.forward(&hidden, context)?;
        B::rms_norm_without_weight(&hidden, 1e-5, context)
    }
}

/// Resident neutral vision tower; streamed runtimes reuse its static phases and block units.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionTower<B: NeuralBackend> {
    /// Pinned media modules.
    pub static_modules: VisionStatic<B>,
    /// Independently streamable encoder blocks.
    pub blocks: Vec<VisionBlock<B>>,
    #[parameter(skip)]
    schedule: Vec<VisionAttentionPolicy>,
}

impl<B: NeuralBackend> VisionTower<B> {
    /// Builds the complete unloaded native vision tower and language adapter.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let blocks = (0..config.layer_count())
            .map(|layer| VisionBlock::new(&config, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
        let schedule = config.schedule.clone();
        Ok(Self {
            static_modules: VisionStatic::new(config, context)?,
            blocks,
            schedule,
        })
    }

    /// Runs the complete resident tower through the same reusable phases.
    pub fn forward(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (mut hidden, state) = self.static_modules.begin(input, context)?;
        for layer in 0..self.blocks.len() {
            hidden = self.blocks[layer].forward_scheduled(
                &hidden,
                self.schedule[layer],
                &state,
                context,
            )?;
        }
        self.static_modules.finish(&hidden, &state, context)
    }
}

fn interpolated_positions<B: NeuralBackend>(
    embedding: &mut B::Embedding,
    grid: &[(i32, i32, i32)],
    height: i32,
    width: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let samples = bilinear_interpolation_samples(
        grid,
        height,
        width,
        InterpolationMode::HalfPixel,
        PatchTraversal::Raster,
    )
    .map_err(Error::backend)?;
    let length = samples.len() as i32;
    let mut output: Option<B::Tensor> = None;
    for corner in 0..4 {
        let indices = samples
            .iter()
            .map(|sample| sample.indices[corner] as i32)
            .collect::<Vec<_>>();
        let weights = samples
            .iter()
            .map(|sample| sample.weights[corner])
            .collect::<Vec<_>>();
        let indices = B::Tensor::from_i32_slice(&indices, &[length], context)?;
        let weights = B::Tensor::from_f32_slice(&weights, &[length, 1], context)?;
        let value = embedding
            .forward(&indices, context)?
            .multiply(&weights, context)?;
        output = Some(match output {
            Some(current) => current.add(&value, context)?,
            None => value,
        });
    }
    output.ok_or_else(|| Error::backend("Muse-Glimmer vision grid is empty"))
}

fn rotary_embeddings<T: Tensor>(
    grid: &[(i32, i32, i32)],
    config: &VisionConfig,
    context: &T::Context,
) -> Result<(T, T), Error> {
    let positions = patch_positions(grid, PatchTraversal::Raster).map_err(Error::backend)?;
    let positions = positions
        .into_iter()
        .flat_map(|[_, y, x]| [x, y])
        .collect::<Vec<_>>();
    let position_ids = T::from_i32_slice(&positions, &[positions.len() as i32 / 2, 2], context)?;
    let axis_dimensions = (config.hidden_size / config.num_heads) / 2;
    multi_axis_rotary_embeddings(
        &position_ids,
        &MultiAxisRotarySpec {
            axes: vec![
                RotaryAxisSpec {
                    dimensions: axis_dimensions,
                    position_offset: 1,
                },
                RotaryAxisSpec {
                    dimensions: axis_dimensions,
                    position_offset: 1,
                },
            ],
            base: config.rope_theta,
            minimum_position: 0,
            layout: MultiAxisRotaryLayout::SplitHalves,
        },
        context,
    )
}

fn apply_rotary<T: Tensor>(
    value: &T,
    cosine: &T,
    sine: &T,
    context: &T::Context,
) -> Result<T, Error> {
    let half = value.dim(2) / 2;
    let first = value.index(&[Index::Full, Index::Full, Index::Range(0, half)], context)?;
    let second = value.index(
        &[Index::Full, Index::Full, Index::Range(half, value.dim(2))],
        context,
    )?;
    let rotated = T::concatenate(&[second.multiply_scalar(-1.0, context)?, first], 2, context)?;
    let cosine = cosine.expand_dims(1, context)?;
    let sine = sine.expand_dims(1, context)?;
    value
        .multiply(&cosine, context)?
        .add(&rotated.multiply(&sine, context)?, context)
}

fn pixel_shuffle<T: Tensor>(
    hidden: &T,
    grid: &[(i32, i32, i32)],
    factor: i32,
    context: &T::Context,
) -> Result<T, Error> {
    let dimensions = hidden.dim(1);
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(grid.len());
    for &(time, height, width) in grid {
        let count = time * height * width;
        let chunk = hidden.index(
            &[Index::Range(offset, offset + count), Index::Full],
            context,
        )?;
        let mut permutation = Vec::with_capacity(count as usize);
        for frame in 0..time {
            for outer_y in 0..height / factor {
                for outer_x in 0..width / factor {
                    for inner_y in 0..factor {
                        for inner_x in 0..factor {
                            permutation.push(
                                frame * height * width
                                    + (outer_y * factor + inner_y) * width
                                    + outer_x * factor
                                    + inner_x,
                            );
                        }
                    }
                }
            }
        }
        let permutation = T::from_i32_slice(&permutation, &[count], context)?;
        let output_count = time * (height / factor) * (width / factor);
        outputs.push(
            chunk
                .take_axis(&permutation, 0, context)?
                .reshape(&[output_count, factor * factor, dimensions], context)?
                .transpose_axes(&[0, 2, 1], context)?
                .reshape(&[output_count, dimensions * factor * factor], context)?,
        );
        offset += count;
    }
    if offset != hidden.dim(0) {
        return Err(Error::backend(
            "Muse-Glimmer pixel shuffle grid does not cover hidden states",
        ));
    }
    T::concatenate(&outputs, 0, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_preserve_released_window_and_shuffle_order() {
        let grid = [(1, 4, 4), (2, 2, 4)];
        assert_eq!(attention_chunk_lengths(&grid).unwrap(), [16, 8, 8]);
        let layout = window_partition(&grid, 1, 8, 2).unwrap();
        assert_eq!(layout.permutation.len(), 32);
        assert_eq!(inverse_permutation(&layout.permutation).unwrap().len(), 32);

        let factor = 2;
        let mut permutation = Vec::new();
        for outer_y in 0..2 {
            for outer_x in 0..2 {
                for inner_y in 0..factor {
                    for inner_x in 0..factor {
                        permutation
                            .push((outer_y * factor + inner_y) * 4 + outer_x * factor + inner_x);
                    }
                }
            }
        }
        assert_eq!(
            permutation,
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
    }
}
