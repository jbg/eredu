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
    construction: Option<Arc<construction::PreparedPredictionConstruction>>,
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
            construction: None,
        }
    }

    pub(super) fn construction(
        &self,
    ) -> Option<&Arc<construction::PreparedPredictionConstruction>> {
        self.construction.as_ref()
    }
    pub(super) fn retained_layout(&self) -> Option<Arc<LocalModelLayout>> {
        self.layout.clone()
    }
    pub(super) fn retained_parameters(
        &self,
    ) -> Option<Arc<eredu_runtime::ArchitectureParameterDescription>> {
        self.parameters.clone()
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
        self.layout_projection(
            rank,
            crate::partitioned_execution::source_allocation::Allocation(None),
        )
        .map(|value| {
            value.map(|value| match value {
                PredictionLayout::Retained(layout) => Arc::clone(layout),
                PredictionLayout::Derived(layout) => Arc::new(layout),
            })
        })
        .map_err(crate::partitioned_execution::source_allocation::Cause::ordinary)
    }
    pub(crate) fn layout_projection(
        &self,
        rank: usize,
        allocation: crate::partitioned_execution::source_allocation::Allocation<'_>,
    ) -> Result<Option<PredictionLayout<'_>>, crate::partitioned_execution::source_allocation::Cause>
    {
        allocation.controls::<(
            &Self,
            usize,
            ParallelRankTopology,
            Option<PredictionLayout<'_>>,
        )>()?;
        let topology = ParallelRankTopology::new(self.topology.topology(), rank)
            .map_err(|error| allocation.error(format_args!("{error}")))?;
        if topology.tensor_parallel_rank() == self.topology.tensor_parallel_rank() {
            return Ok(self.layout.as_ref().map(PredictionLayout::Retained));
        }
        let Some(parameters) = &self.parameters else {
            return Ok(None);
        };
        let tensor =
            tensor_rank(topology).map_err(|error| allocation.error(format_args!("{error}")))?;
        crate::partitioned_execution::local_layout::local_layout_worker(
            parameters,
            tensor.tensor_parallel_rank(),
            tensor.tensor_parallel_size(),
            tensor.expert_parallel_rank(),
            tensor.expert_parallel_size(),
            allocation,
        )
        .map(|layout| Some(PredictionLayout::Derived(layout)))
    }
}
/// Temporary source projection: exact retained layout or newly produced peer.
/// The enclosing constructor owns the account until this scratch retires.
pub(crate) enum PredictionLayout<'a> {
    Retained(&'a Arc<LocalModelLayout>),
    Derived(LocalModelLayout),
}
impl PredictionLayout<'_> {
    pub(crate) fn layout(&self) -> &LocalModelLayout {
        match self {
            Self::Retained(value) => value,
            Self::Derived(value) => value,
        }
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
        let mut placement =
            PreparedPredictionPlacement::from_prepared(topology, parameters, layout);
        if let Self::DeepSeekV3 { construction, .. }
        | Self::DeepSeekV4 { construction, .. }
        | Self::DeepSeekV4Dspark { construction, .. }
        | Self::QwenHybrid { construction, .. }
        | Self::Inkling { construction, .. }
        | Self::NemotronH { construction, .. } = self
        {
            placement.construction = Some(construction.clone());
        }
        Arc::new(placement)
    }
}
