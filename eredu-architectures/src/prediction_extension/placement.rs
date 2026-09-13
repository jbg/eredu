//! Placement retained from actual neutral prediction-module preparation.
use super::*;
use std::sync::{Arc, OnceLock};

pub(crate) type PredictionPlacementSlot = Arc<OnceLock<Arc<PreparedPredictionPlacement>>>;

/// Exact placement used by the prepared prediction materializer. This is semantic
/// metadata, not capture authority or a claim that a particular hook is implemented.
#[derive(Debug, Clone)]
pub struct PreparedPredictionPlacement {
    topology: ParallelRankTopology,
    parameters: Option<Arc<eredu_runtime::ArchitectureParameterDescription>>,
    layout: Option<Arc<LocalModelLayout>>,
}
impl PreparedPredictionPlacement {
    pub(crate) fn from_prepared(
        topology: ParallelRankTopology,
        parameters: Option<Arc<eredu_runtime::ArchitectureParameterDescription>>,
        layout: Option<Arc<LocalModelLayout>>,
    ) -> Self {
        Self {
            topology,
            parameters,
            layout,
        }
    }

    /// Global execution topology, including replicas outside the tensor axis.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.topology
    }
    /// The global parameter declaration used to compile sharded prediction units.
    /// An unsharded complete module has no partition parameter compiler.
    pub fn parameters(&self) -> Option<&eredu_runtime::ArchitectureParameterDescription> {
        self.parameters.as_deref()
    }
    /// The very same local placement lent to native module materialization.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.layout.as_deref()
    }
    /// Projects a peer through the same tensor-only placement compiler. Prediction
    /// units are replicated over the other axes by their prepared construction.
    pub fn layout_for_rank(&self, rank: usize) -> Result<Option<Arc<LocalModelLayout>>, String> {
        let topology = ParallelRankTopology::new(self.topology.topology(), rank)
            .map_err(|error| error.to_string())?;
        if topology.tensor_parallel_rank() == self.topology.tensor_parallel_rank() {
            return Ok(self.layout.clone());
        }
        let Some(parameters) = &self.parameters else {
            return Ok(None);
        };
        let tensor = tensor_rank(topology).map_err(|error| error.to_string())?;
        crate::partitioned_execution::derive_partitioned_local_layout(parameters, tensor)
            .map(|layout| Some(Arc::new(layout)))
    }
}
impl<B> PreparedPredictionExtension<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    pub(crate) fn retained_placement(
        &self,
        topology: ParallelRankTopology,
    ) -> Arc<PreparedPredictionPlacement> {
        let (parameters, layout) = match self {
            Self::DeepSeekV3 {
                parameters, layout, ..
            }
            | Self::DeepSeekV4 {
                parameters, layout, ..
            }
            | Self::DeepSeekV4Dspark {
                parameters, layout, ..
            }
            | Self::Inkling {
                parameters, layout, ..
            }
            | Self::QwenHybrid {
                parameters, layout, ..
            }
            | Self::NemotronH {
                parameters, layout, ..
            } => (Some(Arc::clone(parameters)), Some(Arc::clone(layout))),
        };
        Arc::new(PreparedPredictionPlacement::from_prepared(
            topology, parameters, layout,
        ))
    }
}
