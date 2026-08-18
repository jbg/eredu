//! Pure parallel topology coordinates, membership, and placement validation.

use serde::{Deserialize, Serialize};

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
        let origin = self.coordinates(rank)?;
        Ok((0..self.world_size())
            .filter(|candidate| {
                let c = self.coordinates(*candidate).expect("bounded");
                match axis {
                    ParallelAxis::Tensor => {
                        (c.pipeline, c.expert, c.data)
                            == (origin.pipeline, origin.expert, origin.data)
                    }
                    ParallelAxis::Pipeline => {
                        (c.tensor, c.expert, c.data) == (origin.tensor, origin.expert, origin.data)
                    }
                    ParallelAxis::Expert => {
                        (c.tensor, c.pipeline, c.data)
                            == (origin.tensor, origin.pipeline, origin.data)
                    }
                    ParallelAxis::Data => {
                        (c.tensor, c.pipeline, c.expert)
                            == (origin.tensor, origin.pipeline, origin.expert)
                    }
                }
            })
            .collect())
    }
}

/// Portable tensor placement description.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorPlacement {
    /// Identical copy on every member.
    Replicated,
    /// One dimension partitioned over an axis.
    Sharded {
        /// Tensor dimension.
        dimension: usize,
        /// Parallel axis.
        axis: ParallelAxis,
    },
    /// Tensor is owned by exactly one rank.
    RankLocal {
        /// Owning world rank.
        rank: usize,
    },
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_coordinates_and_membership() {
        let topology = ParallelTopology::new(2, 2, 2, 1).unwrap();
        assert_eq!(
            topology.coordinates(5).unwrap(),
            ParallelCoordinates {
                tensor: 0,
                pipeline: 1,
                expert: 1,
                data: 0
            }
        );
        assert_eq!(
            topology.axis_members(5, ParallelAxis::Tensor).unwrap(),
            vec![5, 7]
        );
        assert!(topology.coordinates(8).is_err());
    }

    #[test]
    fn deserialization_cannot_bypass_topology_validation() {
        assert!(serde_json::from_str::<ParallelTopology>(
            r#"{"tensor":0,"pipeline":1,"expert":1,"data":1}"#
        )
        .is_err());
    }
}
