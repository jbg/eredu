//! Joint projection alignment while retaining complete grouped-query heads.

use super::parameter_metadata::{DeclarationDestination, ParameterGroupError};
use super::{AttentionProjectionLayout, Config};
use eredu_checkpoint::LinearFormat;
use eredu_runtime::{MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionPartition {
    heads: usize,
    heads_per_chunk: usize,
}

impl AttentionPartition {
    pub(crate) fn new(config: &impl Config, layer: usize) -> Result<Self, ParallelPlanError> {
        Self::new_with(config, layer, DeclarationDestination(None))
            .map_err(ParameterGroupError::ordinary)
    }

    pub(crate) fn new_with(
        config: &impl Config,
        layer: usize,
        destination: DeclarationDestination<'_>,
    ) -> Result<Self, ParameterGroupError> {
        destination.controls::<Self>()?;
        let invalid = |message: &str| destination.group_error(format_args!("{message}"));
        let heads = usize::try_from(config.num_key_value_heads())
            .map_err(|_| invalid("attention key/value head count is negative"))?;
        let queries = usize::try_from(config.num_attention_heads())
            .map_err(|_| invalid("attention query head count is negative"))?;
        let width = usize::try_from(config.head_dim())
            .map_err(|_| invalid("attention head width is negative"))?;
        if heads == 0 || queries == 0 || width == 0 || !queries.is_multiple_of(heads) {
            return Err(invalid(
                "attention partition requires complete positive GQA groups",
            ));
        }
        let query_width = (queries / heads)
            .checked_mul(width)
            .ok_or_else(|| invalid("attention group width overflows"))?;
        let fields = config
            .block_parameter_fields()
            .validate_with(|args| destination.group_error(args))?;
        let prefix = destination.text(format_args!(
            "{}.layers.{layer}.{}",
            config.parameter_root(),
            fields.attention
        ))?;
        let format = |field: &str| -> Result<LinearFormat, ParameterGroupError> {
            let name = destination.text(format_args!("{prefix}.{field}.weight"))?;
            destination.linear_format(config, &name)
        };
        let (query, key, value) = match config.attention_projection_layout() {
            AttentionProjectionLayout::Split => (
                format(fields.attention_query)?,
                format(fields.attention_key)?,
                match destination.0 {
                    Some(context) => config
                        .attention_value_format_with_metadata(layer, context)
                        .map_err(ParameterGroupError::Metadata)?,
                    None => config.attention_value_format(layer),
                },
            ),
            AttentionProjectionLayout::Fused { field } => {
                let fused = format(field)?;
                (fused, fused, fused)
            }
        };
        let mut heads_per_chunk = 1usize;
        let mut constrain =
            |elements: usize, alignment: usize| -> Result<(), ParameterGroupError> {
                if alignment == 0 {
                    return Err(invalid("attention encoding alignment is zero"));
                }
                let required = alignment / gcd(elements, alignment);
                heads_per_chunk = (heads_per_chunk / gcd(heads_per_chunk, required))
                    .checked_mul(required)
                    .ok_or_else(|| invalid("joint attention alignment overflows"))?;
                Ok(())
            };
        for (format, elements) in [(query, query_width), (key, width), (value, width)] {
            constrain(elements, row_alignment_with(format, destination)?)?;
        }
        if let Some((field, _)) = config.attention_output_gate() {
            constrain(
                query_width,
                row_alignment_with(format(field)?, destination)?,
            )?;
        }
        constrain(
            query_width,
            usize::try_from(crate::linear_format::input_partition_alignment(format(
                fields.attention_output,
            )?))
            .map_err(|_| invalid("attention output alignment is negative"))?,
        )?;
        Ok(Self {
            heads,
            heads_per_chunk,
        })
    }

    /// Uniform declarations remain valid before partial chunks are attached.
    pub(crate) fn preferred_units(self) -> usize {
        if self.heads.is_multiple_of(self.heads_per_chunk) {
            self.heads / self.heads_per_chunk
        } else {
            1
        }
    }

    pub(crate) fn head_range(
        self,
        parts: usize,
        rank: usize,
    ) -> Result<std::ops::Range<usize>, ParallelPlanError> {
        let logical = eredu_core::balanced_contiguous_range(
            self.heads.div_ceil(self.heads_per_chunk),
            parts,
            rank,
            false,
        )
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        eredu_runtime::partition_chunk_range(self.heads, self.heads_per_chunk, logical).map_err(Into::into)
    }

    /// Applies the same head ownership to projections, per-head parameters and
    /// separately supplied value banks. Companion widths use primary geometry.
    pub(crate) fn apply(
        self,
        group: ParameterGroupSpec,
        format_of: impl Fn(&str) -> LinearFormat,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        self.apply_with(
            group,
            |name| Ok(format_of(name)),
            DeclarationDestination(None),
        )
        .map_err(ParameterGroupError::ordinary)
    }

    pub(crate) fn apply_with(
        self,
        group: ParameterGroupSpec,
        format_of: impl Fn(&str) -> Result<LinearFormat, ParameterGroupError>,
        destination: DeclarationDestination<'_>,
    ) -> Result<ParameterGroupSpec, ParameterGroupError> {
        destination.controls::<Self>()?;
        if !matches!(
            group.role(),
            ParameterRole::AttentionHeads | ParameterRole::ExpertOutput
        ) {
            return Ok(group);
        }
        if self.heads.is_multiple_of(self.heads_per_chunk) {
            return destination.repartition(group, self.preferred_units());
        }
        destination.chunks(
            group,
            self.heads.div_ceil(self.heads_per_chunk),
            |member, members| {
                let (axis, segments) = match member.sharding() {
                    MemberSharding::Partitioned { axis } => (*axis, None),
                    MemberSharding::PartitionedSegments { axis, segments } => {
                        (*axis, Some(segments))
                    }
                    _ => {
                        return Err(destination.tensor_error(format_args!(
                            "attention member has no shared head partition"
                        )));
                    }
                };
                let fp8_owner = match member.linear_companion_of() {
                    Some(owner) => match format_of(owner)? {
                        LinearFormat::E4M3BlockFp8(format) => Some((owner, format)),
                        _ => None,
                    },
                    None => None,
                };
                let (primary, divisor) = if let Some((owner, format)) = fp8_owner {
                    format
                        .validate()
                        .map_err(|error| destination.tensor_error(format_args!("{error}")))?;
                    let primary = members
                        .iter()
                        .find(|candidate| candidate.target() == owner)
                        .ok_or_else(|| {
                            destination.tensor_error(format_args!(
                                "attention scale has no primary in its group"
                            ))
                        })?;
                    let divisor = match primary.global_shape().len().checked_sub(axis) {
                        Some(2) => format.block_rows as usize,
                        Some(1) => format.block_columns as usize,
                        _ => {
                            return Err(destination.tensor_error(format_args!(
                                "attention scale does not partition a matrix axis"
                            )));
                        }
                    };
                    (primary, divisor)
                } else {
                    (member, 1)
                };
                let extent = match (primary.sharding(), segments) {
                    (MemberSharding::PartitionedSegments { segments, .. }, _) => {
                        let first = segments
                            .first()
                            .ok_or_else(|| {
                                destination
                                    .tensor_error(format_args!("attention segments are empty"))
                            })?
                            .len();
                        if segments.iter().any(|segment| segment.len() != first) {
                            return Err(destination.tensor_error(format_args!(
                                "partial attention segments require distinct chunk widths"
                            )));
                        }
                        first
                    }
                    _ => *primary.global_shape().get(axis).ok_or_else(|| {
                        destination.tensor_error(format_args!("attention partition axis is absent"))
                    })?,
                };
                let numerator = extent.checked_mul(self.heads_per_chunk).ok_or_else(|| {
                    destination.tensor_error(format_args!("attention physical chunk overflows"))
                })?;
                let denominator = self.heads.checked_mul(divisor).ok_or_else(|| {
                    destination.tensor_error(format_args!("attention scale divisor overflows"))
                })?;
                if !numerator.is_multiple_of(denominator) {
                    return Err(destination.tensor_error(format_args!(
                        "attention chunk splits a physical encoding block"
                    )));
                }
                Ok(numerator / denominator)
            },
        )
    }
}

fn row_alignment_with(
    format: LinearFormat,
    destination: DeclarationDestination<'_>,
) -> Result<usize, ParameterGroupError> {
    match format {
        LinearFormat::E4M3BlockFp8(format) => {
            format
                .validate()
                .map_err(|cause| destination.group_error(format_args!("{cause}")))?;
            Ok(format.block_rows as usize)
        }
        _ => Ok(1),
    }
}

fn gcd(mut first: usize, mut second: usize) -> usize {
    while second != 0 {
        (first, second) = (second, first % second);
    }
    first
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding};
    use eredu_core::{ParallelRankTopology, ParallelTopology};

    #[test]
    fn attention_head_tails_keep_joint_projection_scale_and_state_ownership() {
        let source: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/k2_horizon/reference.json"
        )))
        .unwrap();
        for encoding in [
            BlockFp8ScaleEncoding::FloatingPoint,
            BlockFp8ScaleEncoding::Ue8m0,
        ] {
            let mut config = source["mova"]["config"].clone();
            config["hidden_size"] = 130.into();
            config["num_attention_heads"] = 66.into();
            config["num_key_value_heads"] = 33.into();
            config["query_key_norm"] = true.into();
            let mut args = crate::k2_horizon::model_args_from_config_value(&config).unwrap();
            let fp8 = LinearFormat::E4M3BlockFp8(BlockFp8Format::new(128, 128, encoding).unwrap());
            // The middle layer keeps dense projections, so its local head count
            // differs from the adjacent FP8 layers. PP must preserve both states.
            for layer in [0, 2] {
                for field in ["q_proj", "k_proj", "v_proj", "o_proj", "v_experts"] {
                    args.formats.insert(
                        format!("model.layers.{layer}.self_attn.{field}.weight"),
                        fp8,
                    );
                }
            }
            let description = crate::k2_horizon::parameter_description(&args).unwrap();
            let topology = ParallelTopology::new(2, 2, 2, 1).unwrap();
            for rank in 0..topology.world_size() {
                let rank = ParallelRankTopology::new(topology, rank).unwrap();
                let tail = rank.tensor_parallel_rank() == 1;
                let expected_heads = if tail {
                    vec![1, 16, 1]
                } else {
                    vec![32, 17, 32]
                };
                let layout = crate::partitioned_execution::derive_partitioned_local_layout(
                    &description,
                    rank,
                )
                .unwrap();
                for layer in 0..3 {
                    let policy = AttentionPartition::new(&args, layer).unwrap();
                    let head_range = policy.head_range(2, rank.tensor_parallel_rank()).unwrap();
                    assert_eq!(head_range.len(), expected_heads[layer] as usize);
                    let local = crate::k2_horizon::local_block_args(&args, layer, &layout).unwrap();
                    assert_eq!(local.num_key_value_heads(), expected_heads[layer]);
                    assert_eq!(local.num_attention_heads(), 2 * expected_heads[layer]);
                    for (field, heads) in [
                        ("q_norm", 2 * expected_heads[layer]),
                        ("k_norm", expected_heads[layer]),
                    ] {
                        let name = format!("model.layers.{layer}.self_attn.{field}.weight");
                        assert_eq!(
                            layout.tensor(&name).unwrap().local_shape(),
                            [heads as usize * 4]
                        );
                    }
                    if layer != 1 {
                        for (field, axis, width, scale_width) in [
                            (
                                "q_proj",
                                0,
                                if tail { 8 } else { 256 },
                                if tail { 1 } else { 2 },
                            ),
                            ("k_proj", 0, if tail { 4 } else { 128 }, 1),
                            (
                                "o_proj",
                                1,
                                if tail { 8 } else { 256 },
                                if tail { 1 } else { 2 },
                            ),
                        ] {
                            let name = format!("model.layers.{layer}.self_attn.{field}");
                            assert_eq!(
                                layout
                                    .tensor(&format!("{name}.weight"))
                                    .unwrap()
                                    .local_shape()[axis],
                                width
                            );
                            assert_eq!(
                                layout
                                    .tensor(&format!("{name}.weight_scale_inv"))
                                    .unwrap()
                                    .local_shape()[axis],
                                scale_width
                            );
                        }
                    }
                }
                for owned in [0..2, 2..3] {
                    let geometry =
                        super::super::partition_local_geometry(&args, &layout, owned).unwrap();
                    let expected = eredu_runtime::StateLayout::new(
                        super::super::cache_layout_with_key_value_heads(
                            &args,
                            expected_heads.clone(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    assert_eq!(geometry.complete_state_layout(), &expected);
                }
            }
        }
    }
}
