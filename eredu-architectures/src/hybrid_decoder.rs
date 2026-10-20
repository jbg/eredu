//! Shared lifecycle for heterogeneous stateful text decoders.

use eredu_nn::{EmbeddingOperator, Error, LinearOperator, NeuralBackend, Tensor};

use crate::decoder::{
    DecoderBoundary, NormalizedBoundary, SequentialGroup, StaticModules, TARGET_EXECUTION_GROUP,
};

/// Pinned decoder modules with architecture-owned shared execution parameters.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct HybridStaticModules<B: NeuralBackend, E, D: DecoderBoundary<B> = NormalizedBoundary<B>> {
    /// Shared token embedding, normalization, and vocabulary projection.
    pub base: StaticModules<B, D>,
    /// Architecture-owned pinned modules shared by execution units.
    pub extension: E,
}

impl<B: NeuralBackend, E: Clone, D: DecoderBoundary<B> + Clone> Clone
    for HybridStaticModules<B, E, D>
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            extension: self.extension.clone(),
        }
    }
}

/// Common hybrid execution shell with an optional architecture-owned extension.
/// Family modules retain their operator policies and block equations; this shell
/// owns embedding/finalization and one stable target group.
pub struct HybridDecoder<B: NeuralBackend, E = (), D: DecoderBoundary<B> = NormalizedBoundary<B>> {
    static_modules: HybridStaticModules<B, E, D>,
    group: SequentialGroup,
}

impl<B: NeuralBackend, E: Clone, D: DecoderBoundary<B> + Clone> Clone for HybridDecoder<B, E, D> {
    fn clone(&self) -> Self {
        Self {
            static_modules: self.static_modules.clone(),
            group: self.group.clone(),
        }
    }
}

impl<B: NeuralBackend, D: DecoderBoundary<B>> HybridDecoder<B, (), D> {
    /// Owns prepared pinned modules and one validated heterogeneous target group.
    pub fn new(
        static_modules: StaticModules<B, D>,
        parameter_root: &'static str,
        units: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            static_modules: HybridStaticModules {
                base: static_modules,
                extension: (),
            },
            group: SequentialGroup::new(TARGET_EXECUTION_GROUP, parameter_root, units)?,
        })
    }

    /// Attaches one pinned extension without reconstructing the decoder modules.
    pub fn with_static_extension<E>(self, extension: E) -> HybridDecoder<B, E, D> {
        HybridDecoder {
            static_modules: HybridStaticModules {
                base: self.static_modules.base,
                extension,
            },
            group: self.group,
        }
    }
}

impl<B: NeuralBackend, E, D: DecoderBoundary<B>> HybridDecoder<B, E, D> {
    /// Borrows all pinned modules, including the architecture extension.
    pub const fn extended_static_modules(&self) -> &HybridStaticModules<B, E, D> {
        &self.static_modules
    }

    /// Mutably borrows all pinned modules.
    pub fn extended_static_modules_mut(&mut self) -> &mut HybridStaticModules<B, E, D> {
        &mut self.static_modules
    }

    /// Consumes the execution shell and returns all pinned modules.
    pub fn into_extended_static_modules(self) -> HybridStaticModules<B, E, D> {
        self.static_modules
    }

    /// Borrows the shared embedding, final normalization, and output head.
    pub const fn static_modules(&self) -> &StaticModules<B, D> {
        &self.static_modules.base
    }

    /// Mutably borrows the shared embedding, final normalization, and output head.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B, D> {
        &mut self.static_modules.base
    }

    /// Consumes the graph shell and returns its pinned modules.
    pub fn into_static_modules(self) -> StaticModules<B, D> {
        self.static_modules.base
    }

    /// Builds the target execution graph.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        self.group.execution_graph()
    }

    /// Returns the number of units in the target group.
    pub fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.group.unit_count(group)
    }

    /// Returns one stable family-owned parameter path after validating its address.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        self.group.unit_path(group, index)
    }

    /// Selects the initial activation for the target group.
    pub fn begin_group<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.group.begin(group, initial, dependencies)
    }

    /// Applies the architecture's residual readout and tied or separate vocabulary projection.
    pub fn finish_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_logits_instrumented(
            hidden,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Projects only the final position of an unobserved causal-text pass.
    pub fn finish_text_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_logits(
            &crate::decoder::final_hidden_position(hidden, context)?,
            context,
        )
    }

    /// Applies the selected readout while preserving semantic observation boundaries.
    pub fn finish_logits_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .base
            .finish_instrumented(hidden, context, instrumentation)
    }

    /// Applies the same readout with the selected vocabulary collective.
    pub fn finish_logits_parallel_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        self.static_modules.base.finish_parallel_instrumented(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }

    /// Projects an already normalized hidden state through the shared vocabulary head.
    pub fn project_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &mut self.static_modules.base.lm_head {
            Some(head) => head.forward(hidden, context),
            None => self
                .static_modules
                .base
                .embeddings
                .as_linear(hidden, context),
        }
    }
}
