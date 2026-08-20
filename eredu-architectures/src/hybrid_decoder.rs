//! Shared lifecycle for heterogeneous stateful text decoders.

use eredu_nn::{
    EmbeddingOperator, Error, LinearOperator, NeuralBackend, NormalizationOperator, Tensor,
};

use crate::decoder::{
    SequentialGroup, SequentialPredictionGroups, StaticModuleSpec, StaticModules,
};

enum HybridExecutionGroups {
    Target(SequentialGroup),
    TargetAndPrediction(SequentialPredictionGroups),
}

/// Static modules and one ordinary target execution group shared by hybrid decoders.
///
/// Family modules retain their closed operator policies and block equations. This
/// assembly owns only the common embedding/finalization and stable layered-group
/// lifecycle used by those blocks.
pub struct HybridDecoder<B: NeuralBackend> {
    static_modules: StaticModules<B>,
    groups: HybridExecutionGroups,
}

impl<B: NeuralBackend> HybridDecoder<B> {
    /// Builds pinned modules and one validated heterogeneous target group.
    pub fn new(
        static_spec: StaticModuleSpec,
        parameter_root: &'static str,
        units: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            static_modules: StaticModules::from_spec(static_spec, context)?,
            groups: HybridExecutionGroups::Target(SequentialGroup::new(
                "target",
                parameter_root,
                units,
            )?),
        })
    }

    /// Builds pinned modules plus target and equally sized appended prediction groups.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prediction_groups(
        static_spec: StaticModuleSpec,
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_parameter_root: &'static str,
        prediction_groups: usize,
        prediction_units: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            static_modules: StaticModules::from_spec(static_spec, context)?,
            groups: HybridExecutionGroups::TargetAndPrediction(
                SequentialPredictionGroups::new_pattern(
                    target_parameter_root,
                    target_units,
                    prediction_parameter_root,
                    prediction_groups,
                    prediction_units,
                )?,
            ),
        })
    }

    /// Borrows the shared embedding, final normalization, and output head.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows the shared embedding, final normalization, and output head.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Builds the target execution graph.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        match &self.groups {
            HybridExecutionGroups::Target(group) => group.execution_graph(),
            HybridExecutionGroups::TargetAndPrediction(groups) => groups.execution_graph(),
        }
    }

    /// Returns the number of units in the target group.
    pub fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        match &self.groups {
            HybridExecutionGroups::Target(target) => target.unit_count(group),
            HybridExecutionGroups::TargetAndPrediction(groups) => groups.unit_count(group),
        }
    }

    /// Returns one stable family-owned parameter path after validating its address.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        match &self.groups {
            HybridExecutionGroups::Target(target) => target.unit_path(group, index),
            HybridExecutionGroups::TargetAndPrediction(groups) => groups.unit_path(group, index),
        }
    }

    /// Selects the initial activation for the target group.
    pub fn begin_group<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        match &self.groups {
            HybridExecutionGroups::Target(target) => target.begin(group, initial, dependencies),
            HybridExecutionGroups::TargetAndPrediction(groups) => {
                groups.begin(group, initial, dependencies)
            }
        }
    }

    /// Applies final normalization and the tied or separate vocabulary projection.
    pub fn finish_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.norm.forward(hidden, context)?;
        match &mut self.static_modules.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self.static_modules.embeddings.as_linear(&hidden, context),
        }
    }

    /// Projects an already normalized hidden state through the shared vocabulary head.
    pub fn project_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &mut self.static_modules.lm_head {
            Some(head) => head.forward(hidden, context),
            None => self.static_modules.embeddings.as_linear(hidden, context),
        }
    }
}
