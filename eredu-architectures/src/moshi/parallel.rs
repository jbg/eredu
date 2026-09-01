//! Semantic tensor-parallel parameter plans and rank-local Moshi geometry.

use std::{collections::BTreeMap, ops::Range};

use crate::decoder;
use eredu_nn::NeuralBackend;
use eredu_runtime::{
    module_parameter_group, LayerRuntimeState, MemberSharding, ParallelPlanError,
    ParameterGroupSpec, ParameterRole, StateLayout, StateSegmentLifetime, StateSegmentSpec,
    TensorPlacement,
};

use super::{
    LayeredModel, MoshiConfig, MoshiTransformerConfig, StaticModules, Unit, DEPTH_STATE_SEGMENT,
    TEMPORAL_STATE_SEGMENT,
};

/// Exact collective cardinality for one tensor-parallel realtime traversal.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MoshiCollectiveCount {
    /// Row-parallel reductions in canonical execution order.
    pub all_sum: usize,
    /// Uneven vocabulary gathers in canonical text-then-depth order.
    pub all_gather: usize,
}

/// Derives the backend-independent collective oracle for one traversal.
///
/// `executed_depth_slices` permits fully forced tails to state their shorter
/// traversal explicitly while rejecting impossible cardinalities.
pub fn collective_count(
    config: &MoshiConfig,
    executed_depth_slices: usize,
) -> Result<MoshiCollectiveCount, ParallelPlanError> {
    let depth_slices = config.frame_schedule().depth_audio_codebooks();
    if executed_depth_slices > depth_slices {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "executed Moshi depth slices {executed_depth_slices} exceed {depth_slices}"
        )));
    }
    let temporal_layers = usize::try_from(config.temporal().num_hidden_layers()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi temporal layer count exceeds usize".into())
    })?;
    let depth_layers =
        usize::try_from(config.depth_template().num_hidden_layers()).map_err(|_| {
            ParallelPlanError::InvalidGroup("Moshi depth layer count exceeds usize".into())
        })?;
    collective_count_components(
        config.frame_schedule().total_audio_codebooks(),
        temporal_layers,
        executed_depth_slices,
        depth_layers,
    )
}

fn collective_count_components(
    audio_embedding_tables: usize,
    temporal_layers: usize,
    executed_depth_slices: usize,
    depth_layers: usize,
) -> Result<MoshiCollectiveCount, ParallelPlanError> {
    let overflow = |operation: &str| {
        ParallelPlanError::InvalidGroup(format!(
            "Moshi collective count overflowed while {operation}"
        ))
    };
    let all_gather = executed_depth_slices
        .checked_add(1)
        .ok_or_else(|| overflow("including the text vocabulary gather"))?;
    let embedding_sums = audio_embedding_tables
        .checked_add(1)
        .ok_or_else(|| overflow("including the text embedding"))?;
    let temporal_sums = temporal_layers
        .checked_mul(2)
        .ok_or_else(|| overflow("counting temporal row projections"))?;
    let depth_block_sums = depth_layers
        .checked_mul(2)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| overflow("counting one depth slice"))?;
    let depth_sums = executed_depth_slices
        .checked_mul(depth_block_sums)
        .ok_or_else(|| overflow("counting executed depth slices"))?;
    let all_sum = embedding_sums
        .checked_add(temporal_sums)
        .and_then(|count| count.checked_add(depth_sums))
        .ok_or_else(|| overflow("combining all-sum phases"))?;
    Ok(MoshiCollectiveCount {
        all_sum,
        all_gather,
    })
}

/// Rank-local widths used to construct one shared decoder block.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalTransformerGeometry {
    attention_heads: i32,
    gated_hidden_size: i32,
}

impl LocalTransformerGeometry {
    /// Rank-local complete attention heads.
    pub const fn attention_heads(&self) -> i32 {
        self.attention_heads
    }

    /// Rank-local width of each fused SwiGLU component.
    pub const fn gated_hidden_size(&self) -> i32 {
        self.gated_hidden_size
    }
}

/// Complete rank-local construction and state geometry for one Moshi rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    temporal: Vec<LocalTransformerGeometry>,
    depth: Vec<Vec<LocalTransformerGeometry>>,
    vocabulary_ranges: BTreeMap<String, Range<usize>>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

impl LocalGeometry {
    /// Local temporal block geometry in execution order.
    pub fn temporal(&self) -> &[LocalTransformerGeometry] {
        &self.temporal
    }

    /// Local depth transformer geometry by slice then layer.
    pub fn depth(&self) -> &[Vec<LocalTransformerGeometry>] {
        &self.depth
    }

    /// Exact vocabulary range owned by one embedding or output parameter.
    pub fn vocabulary_range(&self, target: &str) -> Option<&Range<usize>> {
        self.vocabulary_ranges.get(target)
    }

    /// Persistent-temporal plus reusable frame-local depth cache geometry.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Validates that every local width, vocabulary range, and state slot was
    /// derived from this exact normalized Moshi configuration.
    pub fn validate_for(&self, config: &MoshiConfig) -> Result<(), ParallelPlanError> {
        let temporal_layers =
            usize::try_from(config.temporal().num_hidden_layers()).map_err(|_| {
                ParallelPlanError::InvalidGroup("Moshi temporal layer count exceeds usize".into())
            })?;
        let depth_slices = config.frame_schedule().depth_audio_codebooks();
        let depth_layers =
            usize::try_from(config.depth_template().num_hidden_layers()).map_err(|_| {
                ParallelPlanError::InvalidGroup("Moshi depth layer count exceeds usize".into())
            })?;
        if self.architecture_fingerprint != config.architecture_fingerprint()
            || self.temporal.len() != temporal_layers
            || self.depth.len() != depth_slices
            || self.depth.iter().any(|slice| slice.len() != depth_layers)
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Moshi geometry belongs to a different normalized configuration".into(),
            ));
        }

        let expected_vocabularies = vocabulary_global_rows(config)?;
        if self.vocabulary_ranges.len() != expected_vocabularies.len() {
            return Err(ParallelPlanError::InvalidTensor(
                "rank-local Moshi vocabulary ownership is incomplete".into(),
            ));
        }
        for (target, global_rows) in expected_vocabularies {
            let range = self.vocabulary_ranges.get(&target).ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "rank-local Moshi geometry is missing vocabulary range for {target}"
                ))
            })?;
            if range.start >= range.end || range.end > global_rows {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "rank-local Moshi vocabulary range {range:?} for {target} exceeds {global_rows} rows"
                )));
            }
        }

        let temporal_heads = self
            .temporal
            .iter()
            .map(LocalTransformerGeometry::attention_heads)
            .collect::<Vec<_>>();
        let depth_heads = self.depth.first().ok_or_else(|| {
            ParallelPlanError::InvalidGroup("rank-local Moshi geometry has no depth slices".into())
        })?;
        if self.depth.iter().any(|slice| slice != depth_heads) {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Moshi depth slices have inconsistent geometry".into(),
            ));
        }
        let expected_state = local_state_layout(
            config,
            &temporal_heads,
            &depth_heads
                .iter()
                .map(LocalTransformerGeometry::attention_heads)
                .collect::<Vec<_>>(),
        )?;
        if expected_state != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Moshi state layout drifted from transformer geometry".into(),
            ));
        }
        Ok(())
    }

    /// Builds one rank-local temporal decoder configuration.
    pub fn temporal_config(
        &self,
        global: &MoshiTransformerConfig,
        layer: usize,
    ) -> Result<MoshiTransformerConfig, ParallelPlanError> {
        local_config(global, self.temporal.get(layer), "temporal", layer)
    }

    /// Builds one rank-local depth decoder configuration.
    pub fn depth_config(
        &self,
        global: &MoshiTransformerConfig,
        slice: usize,
        layer: usize,
    ) -> Result<MoshiTransformerConfig, ParallelPlanError> {
        local_config(
            global,
            self.depth.get(slice).and_then(|layers| layers.get(layer)),
            "depth",
            layer,
        )
    }

    /// Builds one rank-local execution unit using the canonical Moshi unit
    /// types and shared decoder block implementation.
    pub fn build_unit<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        config: &MoshiConfig,
        group: usize,
        index: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Unit<B>, eredu_nn::Error> {
        match group {
            0 => {
                let local = self
                    .temporal_config(config.temporal(), index)
                    .map_err(eredu_nn::Error::backend)?;
                super::block::build(&local, index, context).map(Unit::Temporal)
            }
            1 => super::DepthSlice::new_parallel(config, index, self, context).map(Unit::Depth),
            _ => Err(eredu_nn::Error::backend(format!(
                "Moshi execution group {group} is outside 0..2"
            ))),
        }
    }
}

fn local_config(
    global: &MoshiTransformerConfig,
    geometry: Option<&LocalTransformerGeometry>,
    stack: &str,
    layer: usize,
) -> Result<MoshiTransformerConfig, ParallelPlanError> {
    let geometry = geometry.ok_or_else(|| {
        ParallelPlanError::InvalidGroup(format!(
            "missing rank-local Moshi {stack} geometry for layer {layer}"
        ))
    })?;
    global
        .with_parallel_geometry(geometry.attention_heads, geometry.gated_hidden_size)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

/// Executes one rank-local temporal block while preserving the canonical
/// Moshi mask and state-slot semantics.
#[allow(clippy::too_many_arguments)]
pub fn forward_temporal_block_parallel<B, S>(
    block: &mut crate::decoder::TransformerBlock<B>,
    index: usize,
    hidden: &B::Tensor,
    state: &mut S,
    forward: &mut super::ForwardContext<B::Tensor>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    super::block::forward_parallel(
        block,
        index,
        hidden,
        forward.temporal_mask(),
        forward.allow_sliding_prefill(),
        state,
        parallel,
        context,
    )
}

/// Declares pinned embedding, normalization, and text-output parameter groups.
pub fn static_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = Vec::with_capacity(modules.embeddings.tables.len() + 2);
    for (index, table) in modules.embeddings.tables.iter().enumerate() {
        let name = if index == 0 {
            "text_emb".to_string()
        } else {
            format!("audio_embs.{}", index - 1)
        };
        groups.push(module_parameter_group::<B::Tensor, _>(
            name,
            ParameterRole::Vocabulary,
            &table.embedding,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "Moshi embedding parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    groups.push(module_parameter_group::<B::Tensor, _>(
        "out_norm",
        ParameterRole::Replicated,
        &modules.output_norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?);
    groups.push(module_parameter_group::<B::Tensor, _>(
        "text_linear",
        ParameterRole::Vocabulary,
        &modules.text_output,
        |_, shape| {
            if shape.is_empty() {
                Err(ParallelPlanError::InvalidTensor(
                    "Moshi text output parameter is scalar".into(),
                ))
            } else {
                Ok(MemberSharding::Balanced { axis: 0 })
            }
        },
    )?);
    Ok(groups)
}

/// Declares all semantic parameter groups for one temporal block or depth slice.
pub fn unit_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    unit: &Unit<B>,
    config: &MoshiConfig,
    group: usize,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    match (group, unit) {
        (0, Unit::Temporal(block)) => {
            decoder::layer_parallel_parameter_groups(block, config.temporal(), index)
        }
        (1, Unit::Depth(slice)) => {
            let prefix = format!("depformer.slices.{index}");
            let mut groups = vec![
                module_parameter_group::<B::Tensor, _>(
                    format!("{prefix}.emb"),
                    ParameterRole::Vocabulary,
                    &slice.embedding,
                    |_, shape| {
                        (!shape.is_empty())
                            .then_some(MemberSharding::Balanced { axis: 0 })
                            .ok_or_else(|| {
                                ParallelPlanError::InvalidTensor(format!(
                                    "Moshi depth embedding {prefix}.emb is scalar"
                                ))
                            })
                    },
                )?,
                module_parameter_group::<B::Tensor, _>(
                    format!("{prefix}.linear_in"),
                    ParameterRole::Replicated,
                    &slice.input,
                    |_, _| Ok(MemberSharding::Replicated),
                )?,
                module_parameter_group::<B::Tensor, _>(
                    format!("{prefix}.linear_out"),
                    ParameterRole::Vocabulary,
                    &slice.output,
                    |_, shape| {
                        (!shape.is_empty())
                            .then_some(MemberSharding::Balanced { axis: 0 })
                            .ok_or_else(|| {
                                ParallelPlanError::InvalidTensor(format!(
                                    "Moshi depth output {prefix}.linear_out is scalar"
                                ))
                            })
                    },
                )?,
            ];
            let transformer = config
                .depth_transformer(index)
                .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
            for (layer, block) in slice.blocks.iter().enumerate() {
                groups.extend(decoder::layer_parallel_parameter_groups(
                    block,
                    &transformer,
                    layer,
                )?);
            }
            Ok(groups)
        }
        _ => Err(ParallelPlanError::InvalidGroup(format!(
            "Moshi unit does not match group {group} index {index}"
        ))),
    }
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    /// Declares pinned parameter groups for semantic tensor-parallel planning.
    pub fn static_parameter_groups(&self) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        static_parameter_groups(self.static_modules())
    }
}

/// Derives and validates all rank-local Moshi widths from a resolved placement.
///
/// `aliases` maps logical aliases to canonical owners. Every alias present in
/// the model layout must have the same local shape and placement as its owner.
pub fn local_geometry<'a>(
    config: &MoshiConfig,
    layout: &eredu_runtime::LocalModelLayout,
    aliases: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<LocalGeometry, ParallelPlanError> {
    validate_alias_layouts(layout, aliases)?;

    let temporal_layers = usize::try_from(config.temporal().num_hidden_layers()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi temporal layer count exceeds usize".into())
    })?;
    let temporal = (0..temporal_layers)
        .map(|layer| local_block_geometry(config.temporal(), layer, layout))
        .collect::<Result<Vec<_>, _>>()?;

    let depth = (0..config.frame_schedule().depth_audio_codebooks())
        .map(|slice| {
            let transformer = config
                .depth_transformer(slice)
                .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
            (0..transformer.num_hidden_layers() as usize)
                .map(|layer| local_block_geometry(&transformer, layer, layout))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut vocabulary_ranges = BTreeMap::new();
    for target in vocabulary_targets(config) {
        let tensor = required_tensor(layout, &target)?;
        let range = vocabulary_range(tensor, &target)?;
        vocabulary_ranges.insert(target, range);
    }

    let temporal_heads = temporal
        .iter()
        .map(LocalTransformerGeometry::attention_heads)
        .collect::<Vec<_>>();
    let depth_heads = depth
        .first()
        .ok_or_else(|| ParallelPlanError::InvalidGroup("Moshi has no depth slices".into()))?;
    if depth.iter().any(|slice| {
        slice
            .iter()
            .map(LocalTransformerGeometry::attention_heads)
            .ne(depth_heads
                .iter()
                .map(LocalTransformerGeometry::attention_heads))
    }) {
        return Err(ParallelPlanError::InvalidGroup(
            "Moshi depth slices resolved different rank-local cache geometry".into(),
        ));
    }
    let state_layout = local_state_layout(
        config,
        &temporal_heads,
        &depth_heads
            .iter()
            .map(LocalTransformerGeometry::attention_heads)
            .collect::<Vec<_>>(),
    )?;
    let geometry = LocalGeometry {
        temporal,
        depth,
        vocabulary_ranges,
        state_layout,
        architecture_fingerprint: config.architecture_fingerprint().to_owned(),
    };
    geometry.validate_for(config)?;
    Ok(geometry)
}

fn vocabulary_global_rows(
    config: &MoshiConfig,
) -> Result<BTreeMap<String, usize>, ParallelPlanError> {
    let text = usize::try_from(config.text_vocabulary_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi text vocabulary exceeds usize".into())
    })?;
    let audio = usize::try_from(config.audio_vocabulary_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi audio vocabulary exceeds usize".into())
    })?;
    let text_embedding = text.checked_add(1).ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Moshi text embedding vocabulary overflowed".into())
    })?;
    let audio_embedding = audio.checked_add(1).ok_or_else(|| {
        ParallelPlanError::InvalidGroup("Moshi audio embedding vocabulary overflowed".into())
    })?;
    let mut rows = BTreeMap::from([
        ("text_emb.weight".into(), text_embedding),
        ("text_linear.weight".into(), text),
    ]);
    rows.extend(
        (0..config.frame_schedule().total_audio_codebooks())
            .map(|codebook| (format!("audio_embs.{codebook}.weight"), audio_embedding)),
    );
    for slice in 0..config.frame_schedule().depth_audio_codebooks() {
        rows.insert(
            format!("depformer.slices.{slice}.emb.weight"),
            if slice == 0 {
                text_embedding
            } else {
                audio_embedding
            },
        );
        rows.insert(format!("depformer.slices.{slice}.linear_out.weight"), audio);
    }
    Ok(rows)
}

fn vocabulary_targets(config: &MoshiConfig) -> Vec<String> {
    let mut targets = vec!["text_emb.weight".into(), "text_linear.weight".into()];
    targets.extend(
        (0..config.frame_schedule().total_audio_codebooks())
            .map(|codebook| format!("audio_embs.{codebook}.weight")),
    );
    for slice in 0..config.frame_schedule().depth_audio_codebooks() {
        targets.push(format!("depformer.slices.{slice}.emb.weight"));
        targets.push(format!("depformer.slices.{slice}.linear_out.weight"));
    }
    targets
}

fn vocabulary_range(
    tensor: &eredu_runtime::LocalTensorLayout,
    target: &str,
) -> Result<Range<usize>, ParallelPlanError> {
    match tensor.placement() {
        TensorPlacement::Range {
            axis: 0,
            start,
            end,
        } if start < end => Ok(*start..*end),
        TensorPlacement::Replicated => {
            let vocabulary = tensor.global_shape().first().copied().ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "Moshi vocabulary tensor {target} is scalar"
                ))
            })?;
            Ok(0..vocabulary)
        }
        placement => Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi vocabulary tensor {target} has invalid placement {placement:?}"
        ))),
    }
}

fn local_block_geometry(
    config: &MoshiTransformerConfig,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<LocalTransformerGeometry, ParallelPlanError> {
    if layer >= config.num_hidden_layers() as usize {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Moshi layer {layer} is outside {}",
            config.num_hidden_layers()
        )));
    }
    let root = format!("{}.layers.{layer}", config.parameter_root());
    let attention = required_parameter(layout, &format!("{root}.self_attn.in_proj"))?;
    let attention_width = local_axis(attention, 0, "fused QKV width")?;
    if !attention_width.is_multiple_of(3) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} local fused QKV width {attention_width} is not divisible by three"
        )));
    }
    let query_width = attention_width / 3;
    let head = usize::try_from(config.head_dim()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Moshi head dimension exceeds usize".into())
    })?;
    if query_width == 0 || !query_width.is_multiple_of(head) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} local query width {query_width} splits head dimension {head}"
        )));
    }
    let output = required_parameter(layout, &format!("{root}.self_attn.out_proj"))?;
    if local_axis(output, 1, "attention output input width")? != query_width {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} attention row projection does not consume its local head partition"
        )));
    }

    let gating = required_parameter(layout, &format!("{root}.gating.linear_in"))?;
    let fused_gating = local_axis(gating, 0, "fused gating width")?;
    if !fused_gating.is_multiple_of(2) {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} local fused gating width {fused_gating} is not even"
        )));
    }
    let gated_hidden_size = fused_gating / 2;
    let down = required_parameter(layout, &format!("{root}.gating.linear_out"))?;
    if local_axis(down, 1, "gating output input width")? != gated_hidden_size {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Moshi {root} gating row projection does not consume its local fused partition"
        )));
    }

    validate_companion_group(layout, attention)?;
    validate_companion_group(layout, gating)?;
    Ok(LocalTransformerGeometry {
        attention_heads: i32::try_from(query_width / head).map_err(|_| {
            ParallelPlanError::InvalidTensor("Moshi local head count exceeds i32".into())
        })?,
        gated_hidden_size: i32::try_from(gated_hidden_size).map_err(|_| {
            ParallelPlanError::InvalidTensor("Moshi local gated width exceeds i32".into())
        })?,
    })
}

fn required_parameter<'a>(
    layout: &'a eredu_runtime::LocalModelLayout,
    parameter: &str,
) -> Result<&'a eredu_runtime::LocalTensorLayout, ParallelPlanError> {
    layout
        .tensor(&format!("{parameter}.weight"))
        .or_else(|| layout.tensor(parameter))
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "missing rank-local Moshi layout for {parameter}"
            ))
        })
}

fn required_tensor<'a>(
    layout: &'a eredu_runtime::LocalModelLayout,
    target: &str,
) -> Result<&'a eredu_runtime::LocalTensorLayout, ParallelPlanError> {
    layout.tensor(target).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing rank-local Moshi layout for {target}"))
    })
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<usize, ParallelPlanError> {
    tensor.local_shape().get(axis).copied().ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "Moshi {label} tensor has no axis {axis}: {:?}",
            tensor.local_shape()
        ))
    })
}

fn validate_companion_group(
    layout: &eredu_runtime::LocalModelLayout,
    representative: &eredu_runtime::LocalTensorLayout,
) -> Result<(), ParallelPlanError> {
    let expected = representative.logical_range();
    for (target, member) in layout
        .tensors()
        .filter(|(_, member)| member.logical_name() == representative.logical_name())
    {
        if member.logical_range() != expected {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Moshi quantization companion {target} does not share logical range {expected:?}"
            )));
        }
    }
    Ok(())
}

fn validate_alias_layouts<'a>(
    layout: &eredu_runtime::LocalModelLayout,
    aliases: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<(), ParallelPlanError> {
    for (alias, owner) in aliases {
        let alias_layout = required_tensor(layout, alias)?;
        let owner_layout = required_tensor(layout, owner)?;
        if alias_layout.global_shape() != owner_layout.global_shape()
            || alias_layout.local_shape() != owner_layout.local_shape()
            || alias_layout.placement() != owner_layout.placement()
        {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Moshi alias {alias} does not share owner {owner} placement"
            )));
        }
    }
    Ok(())
}

fn local_state_layout(
    config: &MoshiConfig,
    temporal_heads: &[i32],
    depth_heads: &[i32],
) -> Result<StateLayout, ParallelPlanError> {
    let temporal = decoder::cache_layout_with_key_value_heads(
        config.temporal(),
        temporal_heads.iter().copied(),
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let depth = decoder::cache_layout_with_key_value_heads(
        config.depth_template(),
        depth_heads.iter().copied(),
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let temporal_count = temporal.len();
    let depth_count = depth.len();
    let policies = temporal
        .iter()
        .cloned()
        .chain(depth.iter().cloned())
        .collect::<Vec<_>>();
    let schedule = eredu_core::LayerSchedule::new(policies.len(), policies)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    StateLayout::segmented(
        schedule,
        [
            StateSegmentSpec::new(
                TEMPORAL_STATE_SEGMENT,
                0..temporal_count,
                StateSegmentLifetime::Persistent,
                0,
            )
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
            StateSegmentSpec::new(
                DEPTH_STATE_SEGMENT,
                temporal_count..temporal_count + depth_count,
                StateSegmentLifetime::FrameLocal,
                0,
            )
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        ],
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout};

    fn tiny_config() -> MoshiConfig {
        MoshiConfig::from_json(
            r#"{
                "model_type":"moshi", "dim":32, "text_card":101,
                "n_q":4, "dep_q":3, "generated_audio_codebooks":2, "card":64,
                "num_heads":4, "num_layers":2, "dim_feedforward":48,
                "causal":true, "context":7, "max_period":10000.0,
                "positional_embedding":"rope", "depformer_dim":24,
                "depformer_dim_feedforward":36, "depformer_num_heads":4,
                "depformer_num_layers":2, "depformer_context":3,
                "depformer_max_period":10000.0, "depformer_pos_emb":"none",
                "delays":[0,0,1,2,1]
            }"#,
        )
        .unwrap()
    }

    fn insert(
        layout: &mut LocalModelLayout,
        target: String,
        logical: String,
        role: ParameterRole,
        global: Vec<usize>,
        local: Vec<usize>,
        placement: TensorPlacement,
        units: Option<usize>,
        range: Option<Range<usize>>,
    ) {
        layout.insert(
            target,
            LocalTensorLayout::new(logical, role, global, local, placement, units, range, false),
        );
    }

    fn insert_vocab(layout: &mut LocalModelLayout, target: String, rows: usize, width: usize) {
        let end = rows.div_ceil(2);
        insert(
            layout,
            target.clone(),
            target,
            ParameterRole::Vocabulary,
            vec![rows, width],
            vec![end, width],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end,
            },
            None,
            None,
        );
    }

    fn insert_block(layout: &mut LocalModelLayout, config: &MoshiTransformerConfig, layer: usize) {
        let root = format!("{}.layers.{layer}", config.parameter_root());
        let hidden = config.hidden_size() as usize;
        let query = config.num_attention_heads() as usize * config.head_dim() as usize / 2;
        let heads = config.num_attention_heads() as usize;
        let attention_logical = format!("{root}.self_attn.projections");
        insert(
            layout,
            format!("{root}.self_attn.in_proj.weight"),
            attention_logical.clone(),
            ParameterRole::AttentionHeads,
            vec![3 * hidden, hidden],
            vec![3 * query, hidden],
            TensorPlacement::Indices {
                axis: 0,
                indices: (0..3 * query).collect(),
            },
            Some(heads),
            Some(0..heads / 2),
        );
        insert(
            layout,
            format!("{root}.self_attn.out_proj.weight"),
            attention_logical,
            ParameterRole::AttentionHeads,
            vec![hidden, hidden],
            vec![hidden, query],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: query,
            },
            Some(heads),
            Some(0..heads / 2),
        );
        let gated = config.gated_hidden_size() as usize;
        let local_gated = gated / 2;
        let gating_logical = format!("{root}.gating.projections");
        insert(
            layout,
            format!("{root}.gating.linear_in.weight"),
            gating_logical.clone(),
            ParameterRole::FeedForwardIntermediate,
            vec![2 * gated, hidden],
            vec![2 * local_gated, hidden],
            TensorPlacement::Indices {
                axis: 0,
                indices: (0..2 * local_gated).collect(),
            },
            Some(gated),
            Some(0..local_gated),
        );
        insert(
            layout,
            format!("{root}.gating.linear_out.weight"),
            gating_logical,
            ParameterRole::FeedForwardIntermediate,
            vec![hidden, gated],
            vec![hidden, local_gated],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: local_gated,
            },
            Some(gated),
            Some(0..local_gated),
        );
    }

    fn local_layout(config: &MoshiConfig) -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        let temporal_width = config.temporal().hidden_size() as usize;
        insert_vocab(
            &mut layout,
            "text_emb.weight".into(),
            config.text_vocabulary_size() as usize + 1,
            temporal_width,
        );
        insert_vocab(
            &mut layout,
            "text_linear.weight".into(),
            config.text_vocabulary_size() as usize,
            temporal_width,
        );
        for codebook in 0..config.frame_schedule().total_audio_codebooks() {
            insert_vocab(
                &mut layout,
                format!("audio_embs.{codebook}.weight"),
                config.audio_vocabulary_size() as usize + 1,
                temporal_width,
            );
        }
        for layer in 0..config.temporal().num_hidden_layers() as usize {
            insert_block(&mut layout, config.temporal(), layer);
        }
        for slice in 0..config.frame_schedule().depth_audio_codebooks() {
            let transformer = config.depth_transformer(slice).unwrap();
            let input_rows = if slice == 0 {
                config.text_vocabulary_size()
            } else {
                config.audio_vocabulary_size()
            } as usize
                + 1;
            insert_vocab(
                &mut layout,
                format!("depformer.slices.{slice}.emb.weight"),
                input_rows,
                transformer.hidden_size() as usize,
            );
            insert_vocab(
                &mut layout,
                format!("depformer.slices.{slice}.linear_out.weight"),
                config.audio_vocabulary_size() as usize,
                transformer.hidden_size() as usize,
            );
            for layer in 0..transformer.num_hidden_layers() as usize {
                insert_block(&mut layout, &transformer, layer);
            }
        }
        layout
    }

    #[test]
    fn local_geometry_derives_heads_fused_widths_vocab_ranges_and_state() {
        let config = tiny_config();
        let layout = local_layout(&config);
        let local = local_geometry(&config, &layout, std::iter::empty()).unwrap();
        assert_eq!(local.temporal().len(), 2);
        assert_eq!(local.temporal()[0].attention_heads(), 2);
        assert_eq!(
            local.temporal()[0].gated_hidden_size(),
            config.temporal().gated_hidden_size() / 2
        );
        assert_eq!(local.depth().len(), 3);
        assert_eq!(local.depth()[0][0].attention_heads(), 2);
        assert_eq!(
            local
                .temporal_config(config.temporal(), 0)
                .unwrap()
                .num_attention_heads(),
            2
        );
        assert_eq!(local.state_layout().segments()[0].id().as_str(), "temporal");
        assert_eq!(local.state_layout().segments()[1].id().as_str(), "depth");
        assert_eq!(
            local.vocabulary_range("text_linear.weight").unwrap().end,
            (config.text_vocabulary_size() as usize).div_ceil(2)
        );
    }

    #[test]
    fn collective_oracle_tracks_forced_depth_tail_cardinality() {
        let config = tiny_config();
        let full_depth = config.frame_schedule().depth_audio_codebooks();
        let skipped = collective_count(&config, 0).unwrap();
        let full = collective_count(&config, full_depth).unwrap();
        let depth_layers = usize::try_from(config.depth_template().num_hidden_layers()).unwrap();
        assert_eq!(skipped.all_gather, 1);
        assert_eq!(full.all_gather, 1 + full_depth);
        assert_eq!(
            full.all_sum - skipped.all_sum,
            full_depth * (1 + 2 * depth_layers)
        );
        assert!(collective_count(&config, full_depth + 1).is_err());
    }

    #[test]
    fn collective_oracle_rejects_every_arithmetic_overflow() {
        for (audio, temporal, slices, depth, operation) in [
            (usize::MAX, 0, 0, 0, "text embedding"),
            (0, usize::MAX, 0, 0, "temporal row projections"),
            (0, 0, 1, usize::MAX, "one depth slice"),
            (0, 0, usize::MAX / 3 + 1, 1, "executed depth slices"),
            (usize::MAX - 1, 1, 0, 0, "all-sum phases"),
            (0, 0, usize::MAX, 0, "text vocabulary gather"),
        ] {
            let error = collective_count_components(audio, temporal, slices, depth).unwrap_err();
            assert!(
                error.to_string().contains(operation),
                "unexpected overflow diagnostic: {error}"
            );
        }
    }

    #[test]
    fn local_geometry_rejects_alias_placement_drift() {
        let config = tiny_config();
        let mut layout = local_layout(&config);
        insert(
            &mut layout,
            "owner.weight".into(),
            "owner".into(),
            ParameterRole::Replicated,
            vec![32],
            vec![32],
            TensorPlacement::Replicated,
            None,
            None,
        );
        insert(
            &mut layout,
            "alias.weight".into(),
            "alias".into(),
            ParameterRole::Replicated,
            vec![32],
            vec![16],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 16,
            },
            None,
            None,
        );
        let error =
            local_geometry(&config, &layout, [("alias.weight", "owner.weight")]).unwrap_err();
        assert!(error.to_string().contains("does not share owner"));
    }

    #[test]
    fn local_geometry_rejects_quantization_companion_partition_drift() {
        let config = tiny_config();
        let mut layout = local_layout(&config);
        let root = "transformer.layers.0.self_attn";
        let hidden = config.temporal().hidden_size() as usize;
        insert(
            &mut layout,
            format!("{root}.in_proj.scales"),
            format!("{root}.projections"),
            ParameterRole::AttentionHeads,
            vec![3 * hidden, hidden / 4],
            vec![3 * hidden / 2, hidden / 4],
            TensorPlacement::Indices {
                axis: 0,
                indices: (0..3 * hidden / 2).collect(),
            },
            Some(config.temporal().num_attention_heads() as usize),
            Some(1..3),
        );
        let error = local_geometry(&config, &layout, std::iter::empty()).unwrap_err();
        assert!(error.to_string().contains("quantization companion"));
    }

    #[test]
    fn local_geometry_rejects_incomplete_fused_head_segments() {
        let config = tiny_config();
        let mut layout = local_layout(&config);
        let root = "transformer.layers.0.self_attn";
        let hidden = config.temporal().hidden_size() as usize;
        let local_fused = 3 * hidden / 2 - 1;
        insert(
            &mut layout,
            format!("{root}.in_proj.weight"),
            format!("{root}.projections"),
            ParameterRole::AttentionHeads,
            vec![3 * hidden, hidden],
            vec![local_fused, hidden],
            TensorPlacement::Indices {
                axis: 0,
                indices: (0..local_fused).collect(),
            },
            Some(config.temporal().num_attention_heads() as usize),
            Some(0..2),
        );
        let error = local_geometry(&config, &layout, std::iter::empty()).unwrap_err();
        assert!(error.to_string().contains("fused QKV width"));
    }
}
