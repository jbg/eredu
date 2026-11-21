//! MLX allocator and row movement on the native rank queue.

use super::*;
use crate::backend::runtime::distributed::boundary::MlxPartitionTensorAllocator;
use crate::composition::mlx::distributed::expert::MlxExpertRouteTensorMovement;

impl eredu_core::checkpoint_mechanisms::TextRankResources<MlxNeuralBackend>
    for MlxLocalTextBinding<'_>
{
    type Allocator = MlxPartitionTensorAllocator;
    type ExpertMovement = MlxExpertRouteTensorMovement;

    fn rank_allocator(&self) -> Self::Allocator {
        MlxPartitionTensorAllocator
    }

    fn rank_expert_movement(&self) -> Self::ExpertMovement {
        MlxExpertRouteTensorMovement::new(self.stream)
    }
}
