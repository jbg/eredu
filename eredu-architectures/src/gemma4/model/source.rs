//! Exact initial Gemma declarations retained by the completed model source.
use super::*;
use crate::{decoder::identity::Metadata, replicated_text::SharedCompositeConfig};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};

#[derive(Clone, Debug)]
pub(crate) struct RetainedModelSource {
    pub(super) args: SharedCompositeConfig<FamilyConfig>,
    graph: SharedCompositeConfig<Graph>,
}
#[derive(Debug)]
pub(super) struct Graph {
    pub(super) state: StateLayout,
    parallel: Option<SharedCompositeConfig<LocalGeometry>>,
    partition_state: Option<SharedCompositeConfig<StateLayout>>,
    partition_offset: usize,
    partition_media_inputs: [bool; 2],
    expert: Option<SharedCompositeConfig<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>>,
    pub(super) description: ArchitectureParameterDescription,
    pub(super) transports: [eredu_runtime::ArchitectureGroupTransport; 3],
    pub(super) fingerprint: String,
}
fn same_owner<T>(left: Option<&SharedCompositeConfig<T>>, right: Option<&SharedCompositeConfig<T>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => std::ptr::eq(&**left, &**right),
        (None, None) => true,
        _ => false,
    }
}
impl RetainedModelSource {
    pub(crate) fn matches_admission(&self, admission: &FamilyConfig) -> bool {
        std::ptr::eq(&*self.args, admission)
    }
    pub(crate) fn effective_model_type(&self) -> &str { self.args.effective_model_type() }
    pub(crate) fn fingerprint(&self) -> &str { &self.graph.fingerprint }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    /// Only the actual initial constructor creates declarations. Cold quotation
    /// consumes this same immutable owner after the successful contract handoff.
    pub(crate) fn prepare_source(
        &mut self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RetainedModelSource, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        self.args.validate().map_err(Error::backend)?;
        let graph = Graph {
            state: self.state_layout_impl()?,
            parallel: self.parallel_geometry.clone(),
            partition_state: self.partition_state.clone(),
            partition_offset: self.partition_state_offset,
            partition_media_inputs: self.partition_media_inputs,
            expert: self.expert_realization.clone(),
            description: self.parameter_description_impl(context)?,
            transports: std::array::from_fn(|group| self.canonical_group_transport(group)),
            fingerprint: self.args.architecture_fingerprint(),
        };
        let source = RetainedModelSource {
            args: self.args.clone(),
            graph: SharedCompositeConfig::new(graph, B::construction_metadata(context))?,
        };
        self.source = Some(source.clone());
        Ok(source)
    }

    pub(super) fn checked_graph(&self, metadata: Metadata<'_>) -> Result<&Graph, Error> {
        metadata.controls::<(&Self, Option<&RetainedModelSource>, &Graph, bool)>()?;
        let source = self.source.as_ref().ok_or(WorkspaceMetadataError::Unqualified)?;
        if !std::ptr::eq(&*self.args, &*source.args)
            || !same_owner(self.parallel_geometry.as_ref(), source.graph.parallel.as_ref())
            || !same_owner(self.partition_state.as_ref(), source.graph.partition_state.as_ref())
            || !same_owner(self.expert_realization.as_ref(), source.graph.expert.as_ref())
            || self.partition_state_offset != source.graph.partition_offset
            || self.partition_media_inputs != source.graph.partition_media_inputs
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Ok(&source.graph)
    }

    pub(crate) fn new_with_source(
        source: RetainedModelSource,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, RetainedModelSource, SharedCompositeConfig<FamilyConfig>,
            model_text::StaticTextModules<B>, Result<Self, Error>)>()?;
        metadata.require::<B>("Gemma 4", crate::operator_requirements::GEMMA4)?;
        let text = match source.graph.parallel.as_ref() {
            Some(geometry) => model_text::StaticTextModules::new_parallel(&source.args.text, geometry, context)?,
            None => model_text::StaticTextModules::new(&source.args.text, context)?,
        };
        let mut model = Self::with_shared_text(source.args.clone(), text, source.graph.parallel.clone(), context)?;
        model.partition_state = source.graph.partition_state.clone();
        model.partition_state_offset = source.graph.partition_offset;
        model.partition_media_inputs = source.graph.partition_media_inputs;
        model.expert_realization = source.graph.expert.clone();
        model.source = Some(source);
        Ok(model)
    }

    pub(super) fn source_state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &WorkspaceContext,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        let metadata=Metadata::new(Some(context));
        metadata.controls::<(&Self, &eredu_runtime::PartitionState,
            eredu_core::cache::PromptCacheTopology, eredu_runtime::ModelStateIdentity,
            usize, usize, usize)>()?;
        let graph=self.checked_graph(metadata)?;
        let count=self.args.text.num_hidden_layers();
        let start=state.global_layer_offset();
        let end=start.checked_add(state.layout().len())
            .ok_or_else(||metadata.error(format_args!("Gemma 4 owned state range overflowed")))?;
        if end>count {
            return Err(metadata.error(format_args!("Gemma 4 owns state layers {start}..{end}, outside {count} layers")));
        }
        eredu_runtime::ModelStateIdentity::new_with_diagnostic(
            metadata.text("gemma4")?,metadata.text(self.args.effective_model_type())?,
            metadata.text(&graph.fingerprint)?,count,start,0,topology,
            |message|metadata.prompt_error(message),
        )
    }
}

#[cfg(test)]
mod tests;
