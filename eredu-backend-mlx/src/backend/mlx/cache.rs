//! MLX runtime topology conversion for the neutral prompt-cache contract.

use crate::backend::mlx::MlxParallelContext;
use eredu_core::cache::PromptCacheTopology;

pub fn prompt_cache_topology(topology: MlxParallelContext) -> PromptCacheTopology {
    PromptCacheTopology {
        pipeline: (topology.pipeline_parallel_size > 1).then_some((
            topology.pipeline_parallel_size,
            topology.pipeline_parallel_rank,
        )),
        tensor_parallel: (topology.tensor_parallel_size > 1)
            .then_some((topology.tensor_parallel_size, topology.tensor_parallel_rank)),
        expert_parallel: (topology.expert_parallel_size > 1)
            .then_some((topology.expert_parallel_size, topology.expert_parallel_rank)),
        expert_parallel_cache_replicated: true,
    }
}
