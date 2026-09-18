//! Immutable V3 partition construction shared with its later cold quotation.
use super::*;
use crate::replicated_text::{config_source::ConfigOwner, SharedCompositeConfig};
use crate::routed_text::{RetainedRoutedDescription, RetainedRoutedUnits, RoutedConstructionParameters};

#[derive(Clone)]
pub(crate) struct RetainedParallelModelSource(SharedCompositeConfig<Graph>);
struct Graph {
    args: SharedCompositeConfig<V3Args>,
    static_spec: StaticModuleSpec,
    geometry: Arc<super::super::parallel::V3LocalGeometry>,
    expert: Option<Arc<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>>,
    start: usize,
    parameters: RetainedRoutedDescription,
    units: Option<RetainedRoutedUnits>,
}
impl<B> Model<B>
where B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
{
    // Called only by the initial ordinary partition constructor. The model
    // receives the exact same immutable descriptions subsequently lent to quotes.
    pub(crate) fn prepare_parallel_source(
        &mut self, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RetainedParallelModelSource, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        let geometry = self.parallel_geometry.clone()
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
        let args = SharedCompositeConfig::new((*self.args).clone(), B::construction_metadata(context))?;
        let parameters = match &self.construction_parameters {
            Some(source) => source.clone(),
            None => RetainedRoutedDescription::from_completed(
                eredu_runtime::ArchitectureParameters::parameter_description(self, context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description, B::construction_metadata(context)))?),
        };
        let units = match &self.construction_units {
            Some(source) => Some(source.clone()),
            None => self.prepare_construction_units(None, context)?,
        };
        let source = RetainedParallelModelSource(SharedCompositeConfig::new(Graph {
            static_spec: static_spec_view(&args).to_owned(), args: args.clone(), geometry,
            expert: self.expert_realization.clone(), start: self.partition_target_start,
            parameters: parameters.clone(), units: units.clone(),
        }, B::construction_metadata(context))?);
        self.args = ConfigOwner::Shared(args);
        self.construction_parameters = Some(parameters);
        self.construction_units = units;
        Ok(source)
    }

    pub(crate) fn from_retained_parallel_source(
        source: RetainedParallelModelSource, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, RetainedParallelModelSource, StaticModuleSpec,
            SharedCompositeConfig<V3Args>, Result<Self, Error>)>()?;
        metadata.require::<B>("DeepSeek-V3 tensor parallelism", crate::operator_requirements::DEEPSEEK_V3
            .union(eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL))?;
        let graph = &source.0;
        let spec = match B::construction_metadata(context) {
            Some(metadata) => graph.static_spec.borrowed().to_owned_with_metadata(metadata)?,
            None => graph.static_spec.borrowed().to_owned(),
        };
        let static_modules = StaticModules::from_parallel_spec(spec,
            graph.geometry.embedding_range().clone(), Some(graph.geometry.output_range().clone()), context)?;
        Ok(Self {
            args: ConfigOwner::Shared(graph.args.clone()),
            groups: prediction_groups(&graph.args, B::construction_metadata(context))?,
            static_modules, construction_parameters: Some(graph.parameters.clone()),
            construction_units: graph.units.clone(), parallel_geometry: Some(graph.geometry.clone()),
            expert_realization: graph.expert.clone(), partition_target_start: graph.start,
        })
    }
}
