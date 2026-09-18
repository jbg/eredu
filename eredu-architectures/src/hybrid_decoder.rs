//! Shared lifecycle for heterogeneous stateful text decoders.

use eredu_nn::{EmbeddingOperator, Error, LinearOperator, NeuralBackend, Tensor};

use crate::decoder::{
    SequentialGroup, SequentialPredictionGroups, StaticModuleSpec, StaticModules,
    TARGET_EXECUTION_GROUP,
};

enum HybridExecutionGroups {
    Target(SequentialGroup),
    TargetAndPrediction(SequentialPredictionGroups),
}

/// Pinned decoder modules with architecture-owned shared execution parameters.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct HybridStaticModules<B: NeuralBackend, E> {
    /// Shared token embedding, normalization, and vocabulary projection.
    pub base: StaticModules<B>,
    /// Architecture-owned pinned modules shared by execution units.
    pub extension: E,
}

impl<B: NeuralBackend, E: Clone> Clone for HybridStaticModules<B, E> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            extension: self.extension.clone(),
        }
    }
}

/// Common hybrid execution shell with an optional architecture-owned extension.
/// Family modules retain their operator policies and block equations; this shell
/// owns embedding/finalization and the stable target/prediction group lifecycle.
pub struct HybridDecoder<B: NeuralBackend, E = ()> {
    static_modules: HybridStaticModules<B, E>,
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
        let metadata =
            B::construction_metadata(context).filter(|context| context.uses_checked_metadata());
        if let Some(metadata) = metadata {
            metadata.charge_metadata(
                size_of::<Self>()
                    + size_of::<Result<Self, Error>>()
                    + size_of::<HybridStaticModules<B, ()>>()
                    + size_of::<HybridExecutionGroups>(),
            )?;
        }
        Ok(Self {
            static_modules: HybridStaticModules {
                base: StaticModules::from_spec(static_spec, context)?,
                extension: (),
            },
            groups: HybridExecutionGroups::Target(match metadata {
                Some(metadata) => SequentialGroup::new_with_metadata(
                    TARGET_EXECUTION_GROUP,
                    parameter_root,
                    units,
                    metadata,
                )?,
                None => SequentialGroup::new(TARGET_EXECUTION_GROUP, parameter_root, units)?,
            }),
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
        let metadata =
            B::construction_metadata(context).filter(|context| context.uses_checked_metadata());
        if let Some(metadata) = metadata {
            metadata.charge_metadata(
                size_of::<Self>()
                    + size_of::<Result<Self, Error>>()
                    + size_of::<HybridStaticModules<B, ()>>()
                    + size_of::<HybridExecutionGroups>(),
            )?;
        }
        Ok(Self {
            static_modules: HybridStaticModules {
                base: StaticModules::from_spec(static_spec, context)?,
                extension: (),
            },
            groups: HybridExecutionGroups::TargetAndPrediction(
                SequentialPredictionGroups::new_pattern_with_metadata(
                    target_parameter_root,
                    target_units,
                    prediction_parameter_root,
                    prediction_groups,
                    prediction_units,
                    metadata,
                )?,
            ),
        })
    }

    /// Attaches one pinned extension without reconstructing the decoder modules.
    pub fn with_static_extension<E>(self, extension: E) -> HybridDecoder<B, E> {
        HybridDecoder {
            static_modules: HybridStaticModules {
                base: self.static_modules.base,
                extension,
            },
            groups: self.groups,
        }
    }
}

impl<B: NeuralBackend, E> HybridDecoder<B, E> {
    /// Borrows all pinned modules, including the architecture extension.
    pub const fn extended_static_modules(&self) -> &HybridStaticModules<B, E> {
        &self.static_modules
    }

    /// Mutably borrows all pinned modules.
    pub fn extended_static_modules_mut(&mut self) -> &mut HybridStaticModules<B, E> {
        &mut self.static_modules
    }

    /// Consumes the execution shell and returns all pinned modules.
    pub fn into_extended_static_modules(self) -> HybridStaticModules<B, E> {
        self.static_modules
    }

    /// Borrows the shared embedding, final normalization, and output head.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules.base
    }

    /// Mutably borrows the shared embedding, final normalization, and output head.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules.base
    }

    /// Consumes the graph shell and returns its pinned modules.
    pub fn into_static_modules(self) -> StaticModules<B> {
        self.static_modules.base
    }

    /// Builds the target execution graph.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        match &self.groups {
            HybridExecutionGroups::Target(group) => group.execution_graph(),
            HybridExecutionGroups::TargetAndPrediction(groups) => groups.execution_graph(),
        }
    }



    /// Returns the number of prediction depths declared by the constructed graph.
    pub fn prediction_count(&self) -> usize {
        match &self.groups {
            HybridExecutionGroups::Target(_) => 0,
            HybridExecutionGroups::TargetAndPrediction(groups) => groups.prediction_count(),
        }
    }

    /// Returns stable prediction-group identities in prediction-depth order.
    pub fn prediction_execution_groups(&self) -> Vec<String> {
        match &self.groups {
            HybridExecutionGroups::Target(_) => Vec::new(),
            HybridExecutionGroups::TargetAndPrediction(groups) => {
                groups.prediction_execution_groups()
            }
        }
    }

    /// Returns the number of units in the target group.
    pub fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        match &self.groups {
            HybridExecutionGroups::Target(target) => target.unit_count(group, metadata_context),
            HybridExecutionGroups::TargetAndPrediction(groups) => groups.unit_count(group, metadata_context),
        }
    }

    /// Returns one stable family-owned parameter path after validating its address.
    pub fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        match &self.groups {
            HybridExecutionGroups::Target(target) => target.unit_path(group, index, metadata_context),
            HybridExecutionGroups::TargetAndPrediction(groups) => groups.unit_path(group, index, metadata_context),
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
        self.finish_logits_instrumented(
            hidden,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn finish_logits_instrumented(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .base
            .finish_instrumented(hidden, context, instrumentation)
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
