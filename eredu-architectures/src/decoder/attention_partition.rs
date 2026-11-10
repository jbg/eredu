//! Joint projection alignment while retaining complete grouped-query heads.

use super::topology::{AttentionInputParameters, AttentionParameters};
use eredu_checkpoint::LinearFormat;
use eredu_runtime::{MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionPartition {
    heads: usize,
    heads_per_chunk: usize,
}

impl AttentionPartition {
    pub(crate) fn from_parameters(
        parameters: &AttentionParameters,
        implicit_value_format: LinearFormat,
    ) -> Result<Self, ParallelPlanError> {
        let invalid = |message: &str| ParallelPlanError::InvalidGroup(message.into());
        let geometry = parameters.geometry;
        let heads = geometry.key_value_heads() as usize;
        let width = geometry.key_dimensions() as usize;
        let query_width = geometry.query_width() as usize / heads;
        let (query, key, value) = match &parameters.input {
            AttentionInputParameters::Split { query, key, value } => (
                query.format.encoding(),
                key.format.encoding(),
                value
                    .as_ref()
                    .map_or_else(|| implicit_value_format, |spec| spec.format.encoding()),
            ),
            AttentionInputParameters::Fused { projection, .. } => {
                let format = projection.format.encoding();
                (format, format, format)
            }
        };
        let mut heads_per_chunk = 1usize;
        let mut constrain = |elements: usize, alignment: usize| -> Result<(), ParallelPlanError> {
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
            constrain(elements, row_alignment(format)?)?;
        }
        if let Some((gate, _)) = &parameters.gate {
            constrain(query_width, row_alignment(gate.format.encoding())?)?;
        }
        constrain(
            query_width,
            usize::try_from(crate::linear_format::input_partition_alignment(
                parameters.output.format.encoding(),
            ))
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
        eredu_runtime::partition_chunk_range(self.heads, self.heads_per_chunk, logical)
    }

    /// Applies the same head ownership to projections, per-head parameters and
    /// separately supplied value banks. Companion widths use primary geometry.
    pub(crate) fn apply(
        self,
        group: ParameterGroupSpec,
        format_of: impl Fn(&str) -> LinearFormat,
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        if !matches!(
            group.role(),
            ParameterRole::AttentionHeads | ParameterRole::ExpertOutput
        ) {
            return Ok(group);
        }
        if self.heads.is_multiple_of(self.heads_per_chunk) {
            return ParameterGroupSpec::partitioned(
                group.logical_name(),
                group.role(),
                self.preferred_units(),
                group.members().iter().cloned(),
            );
        }
        let members = group.members().to_vec();
        eredu_runtime::partition_parameter_group_chunks(
            group,
            self.heads.div_ceil(self.heads_per_chunk),
            |member| {
                let (axis, segments) = match member.sharding() {
                    MemberSharding::Partitioned { axis } => (*axis, None),
                    MemberSharding::PartitionedSegments { axis, segments } => {
                        (*axis, Some(segments))
                    }
                    _ => {
                        return Err(ParallelPlanError::InvalidTensor(
                            "attention member has no shared head partition".into(),
                        ));
                    }
                };
                let fp8_owner = member.linear_companion_of().and_then(|owner| {
                    match format_of(owner.as_str()) {
                        LinearFormat::E4M3BlockFp8(format) => Some((owner, format)),
                        _ => None,
                    }
                });
                let (primary, divisor) = if let Some((owner, format)) = fp8_owner {
                    format
                        .validate()
                        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
                    let primary = members
                        .iter()
                        .find(|candidate| candidate.target() == owner.as_str())
                        .ok_or_else(|| {
                            ParallelPlanError::InvalidTensor(
                                "attention scale has no primary in its group".into(),
                            )
                        })?;
                    let divisor = match primary.global_shape().len().checked_sub(axis) {
                        Some(2) => format.block_rows as usize,
                        Some(1) => format.block_columns as usize,
                        _ => {
                            return Err(ParallelPlanError::InvalidTensor(
                                "attention scale does not partition a matrix axis".into(),
                            ));
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
                                ParallelPlanError::InvalidTensor(
                                    "attention segments are empty".into(),
                                )
                            })?
                            .len();
                        if segments.iter().any(|segment| segment.len() != first) {
                            return Err(ParallelPlanError::InvalidTensor(
                                "partial attention segments require distinct chunk widths".into(),
                            ));
                        }
                        first
                    }
                    _ => *primary.global_shape().get(axis).ok_or_else(|| {
                        ParallelPlanError::InvalidTensor(
                            "attention partition axis is absent".into(),
                        )
                    })?,
                };
                let numerator = extent.checked_mul(self.heads_per_chunk).ok_or_else(|| {
                    ParallelPlanError::InvalidTensor("attention physical chunk overflows".into())
                })?;
                let denominator = self.heads.checked_mul(divisor).ok_or_else(|| {
                    ParallelPlanError::InvalidTensor("attention scale divisor overflows".into())
                })?;
                if !numerator.is_multiple_of(denominator) {
                    return Err(ParallelPlanError::InvalidTensor(
                        "attention chunk splits a physical encoding block".into(),
                    ));
                }
                Ok(numerator / denominator)
            },
        )
    }
}

fn row_alignment(format: LinearFormat) -> Result<usize, ParallelPlanError> {
    match format {
        LinearFormat::E4M3BlockFp8(format) => {
            format
                .validate()
                .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
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
