//! Exact completed Qwen-VL constructor and local declarations for cold execution.
use super::*;
use crate::{decoder::identity::Metadata, replicated_text::SharedCompositeConfig};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct RetainedModelSource {
    args: SharedCompositeConfig<ModelArgs>,
    graph: SharedCompositeConfig<Graph>,
}
#[derive(Debug)]
pub(super) struct Graph {
    pub(super) state: StateLayout,
    parallel: Option<Arc<LocalGeometry>>,
    partition: Option<Arc<super::super::PartitionLocalGeometry>>,
    pub(super) description: ArchitectureParameterDescription,
    fingerprint: String,
}
fn same_owner<T>(left: Option<&Arc<T>>, right: Option<&Arc<T>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}
impl RetainedModelSource {
    pub(super) fn execution_graph(
        &self,
        context: Option<&WorkspaceContext>,
    ) -> Result<ExecutionGraph, Error> {
        match context {
            Some(context) => self.graph.description.graph().clone_with_metadata(context),
            None => Ok(self.graph.description.graph().clone()),
        }
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    /// Only the completed initial constructor publishes these declarations.
    /// Later quotation retains the exact owners, never equal replacement config.
    pub(crate) fn prepare_source(
        &mut self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RetainedModelSource, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        let graph = Graph {
            state: self.state_layout_impl()?,
            parallel: self.parallel_geometry.clone(),
            partition: self.partition_geometry.clone(),
            description: self.parameter_description_impl(context)?,
            fingerprint: super::super::prompt_cache_architecture_fingerprint(&self.args),
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
        let source = self
            .source
            .as_ref()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if !std::ptr::eq(&*self.args, &*source.args)
            || !same_owner(
                self.parallel_geometry.as_ref(),
                source.graph.parallel.as_ref(),
            )
            || !same_owner(
                self.partition_geometry.as_ref(),
                source.graph.partition.as_ref(),
            )
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Ok(&source.graph)
    }

    pub(crate) fn new_with_source(
        source: RetainedModelSource,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = Metadata::new(B::construction_metadata(context));
        metadata.controls::<(
            Self,
            RetainedModelSource,
            SharedCompositeConfig<ModelArgs>,
            Option<Arc<LocalGeometry>>,
            Option<Arc<super::super::PartitionLocalGeometry>>,
            Result<Self, Error>,
        )>()?;
        Self::from_shared_config(
            source.args.clone(),
            source.graph.parallel.clone(),
            source.graph.partition.clone(),
            Some(source),
            context,
        )
    }

    pub(super) fn source_state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &WorkspaceContext,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        let metadata = Metadata::new(Some(context));
        metadata.controls::<(
            &Self,
            &eredu_runtime::PartitionState,
            eredu_core::cache::PromptCacheTopology,
            eredu_runtime::ModelStateIdentity,
            usize,
            usize,
            usize,
        )>()?;
        let graph = self.checked_graph(metadata)?;
        let count = usize::try_from(self.args.text.num_hidden_layers)
            .map_err(|_| metadata.error(format_args!("invalid text layer count")))?;
        let start = state.global_layer_offset();
        let end = start
            .checked_add(state.layout().len())
            .ok_or_else(|| metadata.error(format_args!("Qwen3-VL owned layer range overflowed")))?;
        if end > count {
            return Err(metadata.error(format_args!(
                "Qwen3-VL owns layers {start}..{end}, outside {count} layers"
            )));
        }
        eredu_runtime::ModelStateIdentity::new_with_diagnostic(
            metadata.text(self.args.model_kind().canonical_name())?,
            metadata.text(self.args.effective_model_type())?,
            metadata.text(&graph.fingerprint)?,
            count,
            start,
            0,
            topology,
            |message| metadata.prompt_error(message),
        )
    }
}

#[cfg(test)]
mod tests;
