//! Pure parallel topology coordinates and membership validation.

use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Logical parallel axis.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelAxis {
    /// Tensor parallelism.
    Tensor,
    /// Pipeline parallelism.
    Pipeline,
    /// Expert parallelism.
    Expert,
    /// Data parallelism.
    Data,
}

/// Coordinate of one rank in a four-dimensional topology.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ParallelCoordinates {
    /// Tensor coordinate.
    pub tensor: usize,
    /// Pipeline coordinate.
    pub pipeline: usize,
    /// Expert coordinate.
    pub expert: usize,
    /// Data coordinate.
    pub data: usize,
}

/// Validated sizes for every parallel axis.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct ParallelTopology {
    /// Tensor-parallel size.
    pub tensor: usize,
    /// Pipeline-parallel size.
    pub pipeline: usize,
    /// Expert-parallel size.
    pub expert: usize,
    /// Data-parallel size.
    pub data: usize,
}

impl<'de> Deserialize<'de> for ParallelTopology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawTopology {
            tensor: usize,
            pipeline: usize,
            expert: usize,
            data: usize,
        }

        let raw = RawTopology::deserialize(deserializer)?;
        Self::new(raw.tensor, raw.pipeline, raw.expert, raw.data).map_err(serde::de::Error::custom)
    }
}

impl ParallelTopology {
    /// Validates positive sizes and a representable world size.
    pub fn new(
        tensor: usize,
        pipeline: usize,
        expert: usize,
        data: usize,
    ) -> Result<Self, TopologyError> {
        let sizes = [tensor, pipeline, expert, data];
        if sizes.contains(&0) {
            return Err(TopologyError::ZeroAxis);
        }
        sizes
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .ok_or(TopologyError::WorldSizeOverflow)?;
        Ok(Self {
            tensor,
            pipeline,
            expert,
            data,
        })
    }
    /// Total rank count.
    pub fn world_size(self) -> usize {
        self.tensor * self.pipeline * self.expert * self.data
    }
    /// Converts a pipeline-major, tensor, expert-minor rank to coordinates.
    ///
    /// Data parallelism is the outermost dimension. Within one data replica,
    /// `rank = ((pipeline * tensor_size) + tensor) * expert_size + expert`.
    pub fn coordinates(self, rank: usize) -> Result<ParallelCoordinates, TopologyError> {
        if rank >= self.world_size() {
            return Err(TopologyError::RankOutOfRange {
                rank,
                world_size: self.world_size(),
            });
        }
        let expert = rank % self.expert;
        let rank = rank / self.expert;
        let tensor = rank % self.tensor;
        let rank = rank / self.tensor;
        let pipeline = rank % self.pipeline;
        Ok(ParallelCoordinates {
            tensor,
            pipeline,
            expert,
            data: rank / self.pipeline,
        })
    }
    /// Returns all ranks matching the supplied coordinate on non-selected axes.
    pub fn axis_members(
        self,
        rank: usize,
        axis: ParallelAxis,
    ) -> Result<Vec<usize>, TopologyError> {
        let mut coordinates = self.coordinates(rank)?;
        let size = match axis {
            ParallelAxis::Tensor => self.tensor,
            ParallelAxis::Pipeline => self.pipeline,
            ParallelAxis::Expert => self.expert,
            ParallelAxis::Data => self.data,
        };
        (0..size)
            .map(|coordinate| {
                match axis {
                    ParallelAxis::Tensor => coordinates.tensor = coordinate,
                    ParallelAxis::Pipeline => coordinates.pipeline = coordinate,
                    ParallelAxis::Expert => coordinates.expert = coordinate,
                    ParallelAxis::Data => coordinates.data = coordinate,
                }
                self.rank_for(coordinates)
            })
            .collect()
    }

    /// Resolves Cartesian coordinates to the unique global rank.
    pub fn rank_for(self, coordinates: ParallelCoordinates) -> Result<usize, TopologyError> {
        for (coordinate, size, axis) in [
            (coordinates.tensor, self.tensor, ParallelAxis::Tensor),
            (coordinates.pipeline, self.pipeline, ParallelAxis::Pipeline),
            (coordinates.expert, self.expert, ParallelAxis::Expert),
            (coordinates.data, self.data, ParallelAxis::Data),
        ] {
            if coordinate >= size {
                return Err(TopologyError::CoordinateOutOfRange {
                    axis,
                    coordinate,
                    size,
                });
            }
        }
        coordinates
            .data
            .checked_mul(self.pipeline)
            .and_then(|rank| rank.checked_add(coordinates.pipeline))
            .and_then(|rank| rank.checked_mul(self.tensor))
            .and_then(|rank| rank.checked_add(coordinates.tensor))
            .and_then(|rank| rank.checked_mul(self.expert))
            .and_then(|rank| rank.checked_add(coordinates.expert))
            .ok_or(TopologyError::WorldSizeOverflow)
    }
}

/// One validated rank in a backend-neutral Cartesian topology.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct ParallelRankTopology {
    /// Number of ranks in the complete topology.
    pub world_size: usize,
    /// Global rank represented by this value.
    pub global_rank: usize,
    /// Tensor-parallel rank count.
    pub tensor_parallel_size: usize,
    /// Tensor-parallel coordinate.
    pub tensor_parallel_rank: usize,
    /// Pipeline-parallel rank count.
    pub pipeline_parallel_size: usize,
    /// Pipeline-parallel coordinate.
    pub pipeline_parallel_rank: usize,
    /// Expert-parallel rank count.
    pub expert_parallel_size: usize,
    /// Expert-parallel coordinate.
    pub expert_parallel_rank: usize,
    /// Data-parallel rank count.
    pub data_parallel_size: usize,
    /// Data-parallel coordinate.
    pub data_parallel_rank: usize,
}

impl ParallelRankTopology {
    /// Validates and resolves one rank in `topology`.
    pub fn new(topology: ParallelTopology, global_rank: usize) -> Result<Self, TopologyError> {
        let topology = ParallelTopology::new(
            topology.tensor,
            topology.pipeline,
            topology.expert,
            topology.data,
        )?;
        let coordinates = topology.coordinates(global_rank)?;
        Ok(Self {
            world_size: topology.world_size(),
            global_rank,
            tensor_parallel_size: topology.tensor,
            tensor_parallel_rank: coordinates.tensor,
            pipeline_parallel_size: topology.pipeline,
            pipeline_parallel_rank: coordinates.pipeline,
            expert_parallel_size: topology.expert,
            expert_parallel_rank: coordinates.expert,
            data_parallel_size: topology.data,
            data_parallel_rank: coordinates.data,
        })
    }

    /// Returns the complete topology shape.
    pub fn topology(self) -> ParallelTopology {
        ParallelTopology {
            tensor: self.tensor_parallel_size,
            pipeline: self.pipeline_parallel_size,
            expert: self.expert_parallel_size,
            data: self.data_parallel_size,
        }
    }

    /// Returns whether every parallel dimension is a singleton.
    pub const fn is_replicated(self) -> bool {
        self.world_size == 1
    }

    /// Returns whether an axis contains more than one rank.
    pub const fn is_axis_active(self, axis: ParallelAxis) -> bool {
        match axis {
            ParallelAxis::Tensor => self.tensor_parallel_size > 1,
            ParallelAxis::Pipeline => self.pipeline_parallel_size > 1,
            ParallelAxis::Expert => self.expert_parallel_size > 1,
            ParallelAxis::Data => self.data_parallel_size > 1,
        }
    }

    /// Returns this rank's Cartesian coordinates.
    pub const fn coordinates(self) -> ParallelCoordinates {
        ParallelCoordinates {
            tensor: self.tensor_parallel_rank,
            pipeline: self.pipeline_parallel_rank,
            expert: self.expert_parallel_rank,
            data: self.data_parallel_rank,
        }
    }

    /// Resolves Cartesian coordinates to a global rank.
    pub fn global_rank_for(self, coordinates: ParallelCoordinates) -> Result<usize, TopologyError> {
        self.topology().rank_for(coordinates)
    }

    /// Returns topology-derived membership in one communication axis.
    pub fn subgroup(self, axis: ParallelAxis) -> Result<SubgroupMembership, TopologyError> {
        let coordinates = self.coordinates();
        let global_ranks = self.topology().axis_members(self.global_rank, axis)?;
        let rank = match axis {
            ParallelAxis::Tensor => coordinates.tensor,
            ParallelAxis::Pipeline => coordinates.pipeline,
            ParallelAxis::Expert => coordinates.expert,
            ParallelAxis::Data => coordinates.data,
        };
        let size = global_ranks.len();
        let color = subgroup_color(self.topology(), coordinates, axis)?;
        if global_ranks.get(rank).copied() != Some(self.global_rank) {
            return Err(TopologyError::SubgroupIdentity { axis });
        }
        Ok(SubgroupMembership {
            axis,
            color,
            rank,
            size,
            global_ranks,
        })
    }

    /// Ordered global ranks participating in tensor collectives with this rank.
    pub fn tensor_parallel_peers(self) -> Result<Vec<usize>, TopologyError> {
        Ok(self.subgroup(ParallelAxis::Tensor)?.global_ranks)
    }

    /// Ordered global ranks participating in expert exchange with this rank.
    pub fn expert_parallel_peers(self) -> Result<Vec<usize>, TopologyError> {
        Ok(self.subgroup(ParallelAxis::Expert)?.global_ranks)
    }

    /// Preceding pipeline rank with matching coordinates on other axes.
    pub fn pipeline_predecessor(self) -> Result<Option<usize>, TopologyError> {
        if self.pipeline_parallel_rank == 0 {
            return Ok(None);
        }
        self.global_rank_for(ParallelCoordinates {
            pipeline: self.pipeline_parallel_rank - 1,
            ..self.coordinates()
        })
        .map(Some)
    }

    /// Succeeding pipeline rank with matching coordinates on other axes.
    pub fn pipeline_successor(self) -> Result<Option<usize>, TopologyError> {
        if self.pipeline_parallel_rank + 1 == self.pipeline_parallel_size {
            return Ok(None);
        }
        self.global_rank_for(ParallelCoordinates {
            pipeline: self.pipeline_parallel_rank + 1,
            ..self.coordinates()
        })
        .map(Some)
    }

    /// Whether this rank owns the stage-local embedding.
    pub const fn owns_embedding(self) -> bool {
        self.pipeline_parallel_rank == 0
    }

    /// Whether this rank owns the stage-local output head.
    pub const fn owns_output_head(self) -> bool {
        self.pipeline_parallel_rank + 1 == self.pipeline_parallel_size
    }

    /// This pipeline stage's balanced decoder-layer range.
    pub fn layer_range(self, layers: usize) -> Result<Range<usize>, TopologyError> {
        balanced_contiguous_range(
            layers,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
            false,
        )
    }

    /// This expert rank's balanced routed-expert range.
    pub fn expert_range(self, experts: usize) -> Result<Range<usize>, TopologyError> {
        balanced_contiguous_range(
            experts,
            self.expert_parallel_size,
            self.expert_parallel_rank,
            false,
        )
    }

    /// Whether this rank owns `layer` under balanced pipeline placement.
    pub fn owns_layer(self, layers: usize, layer: usize) -> Result<bool, TopologyError> {
        Ok(self.layer_range(layers)?.contains(&layer))
    }

    /// Whether this rank owns `expert` under balanced expert placement.
    pub fn owns_expert(self, experts: usize, expert: usize) -> Result<bool, TopologyError> {
        Ok(self.expert_range(experts)?.contains(&expert))
    }

    /// Validates optional layer/expert geometry before payload access.
    pub fn preflight(
        self,
        decoder_layers: Option<usize>,
        routed_experts: Option<usize>,
    ) -> Result<TopologyPreflightReport, TopologyError> {
        let local_layer_range = match decoder_layers {
            Some(layers) => Some(self.layer_range(layers)?),
            None if self.pipeline_parallel_size > 1 => {
                return Err(TopologyError::MissingLayerCount)
            }
            None => None,
        };
        let local_expert_range = match routed_experts {
            Some(experts) => Some(self.expert_range(experts)?),
            None if self.expert_parallel_size > 1 => return Err(TopologyError::MissingExpertCount),
            None => None,
        };
        Ok(TopologyPreflightReport {
            topology: self,
            tensor_subgroup: self.subgroup(ParallelAxis::Tensor)?,
            pipeline_subgroup: self.subgroup(ParallelAxis::Pipeline)?,
            expert_subgroup: self.subgroup(ParallelAxis::Expert)?,
            data_subgroup: self.subgroup(ParallelAxis::Data)?,
            local_layer_range,
            local_expert_range,
            owns_embedding: self.owns_embedding(),
            owns_output_head: self.owns_output_head(),
        })
    }
}

impl<'de> Deserialize<'de> for ParallelRankTopology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawRankTopology {
            world_size: usize,
            global_rank: usize,
            tensor_parallel_size: usize,
            tensor_parallel_rank: usize,
            pipeline_parallel_size: usize,
            pipeline_parallel_rank: usize,
            expert_parallel_size: usize,
            expert_parallel_rank: usize,
            data_parallel_size: usize,
            data_parallel_rank: usize,
        }
        let raw = RawRankTopology::deserialize(deserializer)?;
        let topology = ParallelTopology::new(
            raw.tensor_parallel_size,
            raw.pipeline_parallel_size,
            raw.expert_parallel_size,
            raw.data_parallel_size,
        )
        .map_err(serde::de::Error::custom)?;
        let value = Self::new(topology, raw.global_rank).map_err(serde::de::Error::custom)?;
        if value.world_size != raw.world_size
            || value.tensor_parallel_rank != raw.tensor_parallel_rank
            || value.pipeline_parallel_rank != raw.pipeline_parallel_rank
            || value.expert_parallel_rank != raw.expert_parallel_rank
            || value.data_parallel_rank != raw.data_parallel_rank
        {
            return Err(serde::de::Error::custom(
                "parallel rank topology contains inconsistent derived fields",
            ));
        }
        Ok(value)
    }
}

/// Topology-derived membership of one rank in an axis subgroup.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubgroupMembership {
    /// Axis represented by this subgroup.
    pub axis: ParallelAxis,
    /// Deterministic subgroup color.
    pub color: usize,
    /// Rank within the subgroup.
    pub rank: usize,
    /// Number of subgroup ranks.
    pub size: usize,
    /// Ordered global ranks.
    pub global_ranks: Vec<usize>,
}

/// Weight-independent ownership report for one parallel rank.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct TopologyPreflightReport {
    /// Complete rank topology.
    pub topology: ParallelRankTopology,
    /// Tensor subgroup.
    pub tensor_subgroup: SubgroupMembership,
    /// Pipeline subgroup.
    pub pipeline_subgroup: SubgroupMembership,
    /// Expert subgroup.
    pub expert_subgroup: SubgroupMembership,
    /// Data subgroup.
    pub data_subgroup: SubgroupMembership,
    /// Locally owned layer range.
    pub local_layer_range: Option<Range<usize>>,
    /// Locally owned expert range.
    pub local_expert_range: Option<Range<usize>>,
    /// Whether the embedding is local.
    pub owns_embedding: bool,
    /// Whether the output head is local.
    pub owns_output_head: bool,
}

fn subgroup_color(
    topology: ParallelTopology,
    coordinates: ParallelCoordinates,
    axis: ParallelAxis,
) -> Result<usize, TopologyError> {
    let dimensions = [
        (ParallelAxis::Data, coordinates.data, topology.data),
        (
            ParallelAxis::Pipeline,
            coordinates.pipeline,
            topology.pipeline,
        ),
        (ParallelAxis::Tensor, coordinates.tensor, topology.tensor),
        (ParallelAxis::Expert, coordinates.expert, topology.expert),
    ];
    dimensions
        .into_iter()
        .filter(|(candidate, _, _)| *candidate != axis)
        .try_fold(0usize, |color, (_, coordinate, size)| {
            color
                .checked_mul(size)
                .and_then(|value| value.checked_add(coordinate))
                .ok_or(TopologyError::WorldSizeOverflow)
        })
}

/// Computes a deterministic balanced contiguous range.
pub fn balanced_contiguous_range(
    total: usize,
    parts: usize,
    index: usize,
    allow_empty: bool,
) -> Result<Range<usize>, TopologyError> {
    if parts == 0 {
        return Err(TopologyError::ZeroPartitions);
    }
    if index >= parts {
        return Err(TopologyError::PartitionOutOfRange { index, parts });
    }
    if !allow_empty && total < parts {
        return Err(TopologyError::EmptyPartition { total, parts });
    }
    let base = total / parts;
    let extra = total % parts;
    let start = index
        .checked_mul(base)
        .and_then(|value| value.checked_add(index.min(extra)))
        .ok_or(TopologyError::PartitionOverflow)?;
    let end = start
        .checked_add(base + usize::from(index < extra))
        .ok_or(TopologyError::PartitionOverflow)?;
    Ok(start..end)
}

/// Topology validation error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum TopologyError {
    /// An axis size was zero.
    #[error("parallel topology axis sizes must be positive")]
    ZeroAxis,
    /// Product of axis sizes overflowed.
    #[error("parallel topology world size overflows usize")]
    WorldSizeOverflow,
    /// Rank is outside the world.
    #[error("rank {rank} is outside world size {world_size}")]
    RankOutOfRange {
        /// Invalid rank.
        rank: usize,
        /// World size.
        world_size: usize,
    },
    /// A coordinate is outside its axis.
    #[error("{axis:?} coordinate {coordinate} is outside axis size {size}")]
    CoordinateOutOfRange {
        /// Invalid axis.
        axis: ParallelAxis,
        /// Invalid coordinate.
        coordinate: usize,
        /// Axis size.
        size: usize,
    },
    /// Subgroup membership did not map back to the represented rank.
    #[error("{axis:?} subgroup geometry does not map back to the represented rank")]
    SubgroupIdentity {
        /// Invalid subgroup axis.
        axis: ParallelAxis,
    },
    /// Pipeline preflight omitted layer geometry.
    #[error("pipeline topology preflight requires the decoder-layer count")]
    MissingLayerCount,
    /// Expert preflight omitted expert geometry.
    #[error("expert topology preflight requires the routed-expert count")]
    MissingExpertCount,
    /// Partition count is zero.
    #[error("partition count must be nonzero")]
    ZeroPartitions,
    /// Partition index is outside the partition count.
    #[error("partition index {index} is outside {parts} parts")]
    PartitionOutOfRange {
        /// Invalid index.
        index: usize,
        /// Partition count.
        parts: usize,
    },
    /// Non-empty partitions were requested with too few items.
    #[error("cannot divide {total} items among {parts} non-empty partitions")]
    EmptyPartition {
        /// Item count.
        total: usize,
        /// Partition count.
        parts: usize,
    },
    /// Partition offset calculation overflowed.
    #[error("balanced range calculation overflowed usize")]
    PartitionOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_topology_is_the_authoritative_coordinate_and_membership_plan() {
        let topology = ParallelTopology::new(2, 3, 2, 2).unwrap();
        let rank = ParallelRankTopology::new(topology, 22).unwrap();
        assert_eq!(
            rank.coordinates(),
            ParallelCoordinates {
                tensor: 1,
                pipeline: 2,
                expert: 0,
                data: 1,
            }
        );
        assert_eq!(rank.global_rank_for(rank.coordinates()).unwrap(), 22);
        assert_eq!(
            rank.subgroup(ParallelAxis::Tensor).unwrap().global_ranks,
            [20, 22]
        );
        assert_eq!(
            rank.subgroup(ParallelAxis::Pipeline).unwrap().global_ranks,
            [14, 18, 22]
        );
        assert_eq!(
            rank.subgroup(ParallelAxis::Expert).unwrap().global_ranks,
            [22, 23]
        );
        assert_eq!(
            rank.subgroup(ParallelAxis::Data).unwrap().global_ranks,
            [10, 22]
        );
        assert_eq!(rank.pipeline_predecessor().unwrap(), Some(18));
        assert_eq!(rank.pipeline_successor().unwrap(), None);
        assert!(!rank.owns_embedding());
        assert!(rank.owns_output_head());

        let report = rank.preflight(Some(7), Some(5)).unwrap();
        assert_eq!(report.local_layer_range, Some(5..7));
        assert_eq!(report.local_expert_range, Some(0..3));
        assert_eq!(report.data_subgroup.global_ranks, [10, 22]);
    }

    #[test]
    fn rank_mapping_is_exhaustive_for_all_axes() {
        let topology = ParallelTopology::new(3, 2, 2, 2).unwrap();
        for global_rank in 0..topology.world_size() {
            let rank = ParallelRankTopology::new(topology, global_rank).unwrap();
            assert_eq!(
                rank.global_rank_for(rank.coordinates()).unwrap(),
                global_rank
            );
            for axis in [
                ParallelAxis::Tensor,
                ParallelAxis::Pipeline,
                ParallelAxis::Expert,
                ParallelAxis::Data,
            ] {
                let subgroup = rank.subgroup(axis).unwrap();
                assert_eq!(subgroup.global_ranks[subgroup.rank], global_rank);
                assert_eq!(
                    subgroup.global_ranks,
                    topology.axis_members(global_rank, axis).unwrap()
                );
            }
        }
    }

    #[test]
    fn balanced_ranges_and_preflight_fail_closed() {
        let ranges = (0..3)
            .map(|index| balanced_contiguous_range(8, 3, index, false).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ranges, [0..3, 3..6, 6..8]);
        assert!(balanced_contiguous_range(2, 3, 0, false).is_err());
        assert_eq!(balanced_contiguous_range(2, 3, 2, true).unwrap(), 2..2);

        let pipeline =
            ParallelRankTopology::new(ParallelTopology::new(1, 4, 1, 1).unwrap(), 0).unwrap();
        assert_eq!(
            pipeline.preflight(None, None),
            Err(TopologyError::MissingLayerCount)
        );
        let expert =
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 4, 1).unwrap(), 0).unwrap();
        assert_eq!(
            expert.preflight(None, None),
            Err(TopologyError::MissingExpertCount)
        );
    }

    #[test]
    fn deserialization_cannot_bypass_shape_or_derived_rank_validation() {
        assert!(serde_json::from_str::<ParallelTopology>(
            r#"{"tensor":0,"pipeline":1,"expert":1,"data":1}"#
        )
        .is_err());

        let rank =
            ParallelRankTopology::new(ParallelTopology::new(2, 2, 2, 2).unwrap(), 9).unwrap();
        let encoded = serde_json::to_string(&rank).unwrap();
        assert_eq!(
            serde_json::from_str::<ParallelRankTopology>(&encoded).unwrap(),
            rank
        );

        let mut inconsistent = serde_json::to_value(rank).unwrap();
        inconsistent["tensor_parallel_rank"] = serde_json::json!(1);
        assert!(serde_json::from_value::<ParallelRankTopology>(inconsistent).is_err());
    }
}
