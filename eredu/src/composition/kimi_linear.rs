//! Neutral Kimi Linear/Kimi Linear-MoE composition over MLX execution policies.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
    sync::Arc,
};

use eredu_architectures::kimi_linear::{Block, LayeredModel, ModelArgs};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource, WeightQuantization};
use eredu_nn::{ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized};
use eredu_runtime::{
    ActivationObserver, CacheResidencyPolicy, CausalModel, DenseDiskStreamReport, ExecutionGraph,
    ExecutionUnitLayout, ExpertIdentity, LayerWeightResidency, LayeredArchitecture,
    LayeredForwardState, LayerwiseModelMetadata, LayerwiseRuntime, OffloadUnit, PagedCacheOptions,
    ParallelLayeredArchitecture, ParallelModelInfo, ResidencyReport, StaticUnitBindings,
    WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

use crate::{
    backend::mlx::{
        error::Error,
        nn::{
            parallel::{VocabParallelEmbedding, VocabParallelLmHead},
            shared::{MlxBackend, MlxModule, MlxNamedModule},
        },
        runtime::{
            cache::{
                residency::{
                    load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
                },
                state::MlxHybridState,
            },
            checkpoint::{
                binding::{
                    binding_bytes, build_module_bindings,
                    build_module_bindings_with_recipes_excluding, canonical_checkpoint_name,
                    populate_module_from_lease_excluding,
                },
                binding_plan::{BindingPlan, PlannedBinding},
                load::{gguf_quantization_configs, GgufTensorNames},
                quantization::should_quantize_on_load,
                store::open_gguf_checkpoint_source,
            },
            execution::{
                generic::{
                    prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                    MlxUnitFactory,
                },
                layerwise::{
                    open_safetensors_weight_store, quantize_parameterized_store,
                    shard_layer_bindings,
                },
            },
            media::input,
            residency::expert_cache::ExpertCatalogEntry,
            residency::expert_cache::{ExpertCache, ExpertCacheReport},
            residency::expert_provider::{
                CachedGatedProductBankSpec, CachedGatedProductExpertProvider,
                ExpertExecutorProvider,
            },
        },
    },
    core::cache::{
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
        PromptCacheTopology,
    },
};

type NeutralBlock = Block<MlxBackend>;
type NeutralArchitecture = LayeredModel<MlxBackend>;
type ResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralBlock>,
>;
type BoundedRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralBlock, KimiLinearUnitFactory>,
>;
type ParallelResidentRuntime = LayerwiseRuntime<
    KimiLinearParallelComposition,
    MlxBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralBlock>,
>;
type ParallelBoundedRuntime = LayerwiseRuntime<
    KimiLinearParallelComposition,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralBlock, KimiLinearParallelUnitFactory>,
>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "Array")]
#[doc(hidden)]
#[cfg(test)]
pub struct KimiLinearCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxBackend>,
    pub layers: Vec<NeutralBlock>,
}

#[cfg(test)]
impl KimiLinearCheckpointTemplate {
    /// Builds one neutral full-parameter template for checkpoint tooling.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let layers = (0..args.num_hidden_layers as usize)
            .map(|index| {
                Block::new(&args, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: architecture.static_modules().clone(),
            layers,
        })
    }
}

struct NeutralKimiLinearObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
}

impl eredu_runtime::ActivationObserver<Array, eredu_nn::Error> for NeutralKimiLinearObserver<'_> {
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe(path, value)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, eredu_nn::Error> {
        self.inner
            .intervene(path, value)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe_routing(routing)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

#[derive(Clone)]
struct KimiLinearUnitFactory {
    args: ModelArgs,
    external_experts: bool,
}

/// Pipeline/loading adapter over the same neutral Kimi Linear blocks used by resident
/// and bounded execution.
pub(crate) struct KimiLinearPipelineAdapter {
    args: ModelArgs,
    static_modules: eredu_architectures::decoder::StaticModules<MlxBackend>,
    external_experts: bool,
}

impl KimiLinearPipelineAdapter {
    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(Self {
            args,
            static_modules: architecture.static_modules().clone(),
            external_experts: false,
        })
    }

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.external_experts = true;
        Ok(adapter)
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub(crate) fn static_units(
        &self,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = Vec::new();
        if select("kimi_linear.static.embedding") {
            units.push(StaticUnitBindings::new(
                "kimi_linear.static.embedding",
                build_module_bindings(
                    &MlxModule::new(self.static_modules.embeddings.clone()),
                    "",
                    store,
                )?,
            )?);
        }
        if select("kimi_linear.static.norm") {
            units.push(StaticUnitBindings::new(
                "kimi_linear.static.norm",
                build_module_bindings(
                    &MlxModule::new(self.static_modules.norm.clone()),
                    "",
                    store,
                )?,
            )?);
        }
        if select("kimi_linear.static.output") {
            if let Some(head) = &self.static_modules.lm_head {
                units.push(StaticUnitBindings::new(
                    "kimi_linear.static.output",
                    build_module_bindings(&MlxModule::new(head.clone()), "", store)?,
                )?);
            }
        }
        Ok(units)
    }

    pub(crate) fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::Parallel(format!(
                "Kimi Linear has no execution group {group}"
            )))
        }
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<NeutralBlock>, Error> {
        self.layer_count(group)?;
        Ok(MlxModule::new(
            Block::new(&self.args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        ))
    }

    pub(crate) fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &MlxModule<NeutralBlock>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let recipes = unit_recipes(store, &self.args, index, !self.external_experts)?;
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && name.contains(".mlp.experts.")
        })
        .map_err(Into::into)
    }

    pub(crate) fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub(crate) fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        if !self.args.has_sparse_moe_layers() {
            return Err(Error::Parallel(
                "Kimi Linear PP+EP requires a sparse-MoE checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        _build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        for group in eredu_architectures::kimi_linear::static_parallel_parameter_groups::<MlxBackend>(
            &self.static_modules,
        )? {
            planner.register(group)?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let block = Block::<MlxBackend>::new(&self.args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for group in eredu_architectures::kimi_linear::layer_parallel_parameter_groups(
                &block, &self.args, index,
            )? {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<MlxModule<NeutralBlock>, Error> {
        self.layer_count(group)?;
        let block = match layout {
            Some(layout) => Block::new_with_geometry(
                &self.args,
                index,
                eredu_architectures::kimi_linear::local_block_geometry(&self.args, index, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                stream,
            ),
            None => Block::new(&self.args, index, stream),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(MlxModule::new(block))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &MlxModule<NeutralBlock>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        let bindings = self.layer_bindings(group, index, &global, store)?;
        match layout {
            Some(layout) => {
                shard_layer_bindings(bindings, &format!("model.layers.{index}"), store, layout)
            }
            None => Ok(bindings),
        }
    }
}

impl MlxUnitFactory<NeutralBlock> for KimiLinearUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<NeutralBlock, Error> {
        Block::new(&self.args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".mlp.experts.")
        })?;
        Ok(())
    }
}

#[derive(Clone)]
struct KimiLinearParallelUnitFactory {
    args: ModelArgs,
    geometries: Arc<Vec<eredu_architectures::kimi_linear::BlockGeometry>>,
    external_experts: bool,
}

impl MlxUnitFactory<NeutralBlock> for KimiLinearParallelUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<NeutralBlock, Error> {
        Block::new_with_geometry(
            &self.args,
            index,
            *self.geometries.get(index).ok_or_else(|| {
                Error::Parallel(format!(
                    "parallel Kimi Linear unit {index} is not configured"
                ))
            })?,
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".mlp.experts.")
        })?;
        Ok(())
    }
}

struct KimiLinearParallelComposition {
    architecture: NeutralArchitecture,
    parallel_embedding: Option<MlxNamedModule<VocabParallelEmbedding>>,
    parallel_lm_head: Option<MlxNamedModule<VocabParallelLmHead>>,
    geometries: Option<Arc<Vec<eredu_architectures::kimi_linear::BlockGeometry>>>,
    state_layout: Option<eredu_runtime::StateLayout>,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl KimiLinearParallelComposition {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            architecture: NeutralArchitecture::new(args, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            parallel_embedding: None,
            parallel_lm_head: None,
            geometries: None,
            state_layout: None,
            topology: None,
        })
    }

    const fn args(&self) -> &ModelArgs {
        self.architecture.args()
    }

    fn configure(
        &mut self,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let geometries = (0..self.args().num_hidden_layers as usize)
            .map(|layer| {
                eredu_architectures::kimi_linear::local_block_geometry(self.args(), layer, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let state_geometry =
            eredu_architectures::kimi_linear::local_state_geometry(self.args(), layout)
                .map_err(|error| Error::Parallel(error.to_string()))?;
        self.state_layout = Some(
            eredu_architectures::kimi_linear::state_layout_with_geometry(
                self.args(),
                &state_geometry,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
        );
        self.parallel_embedding = Some(MlxNamedModule::new(
            VocabParallelEmbedding::unloaded(
                self.args().vocab_size as usize,
                self.args().hidden_size,
                self.args()
                    .weight_quantization_for("model.embed_tokens.weight"),
                build,
                stream,
            )?,
            ParameterSpec::trainable("model.embed_tokens.weight")
                .map_err(|error| Error::Parallel(error.to_string()))?,
            None,
        )?);
        if !self.args().tie_word_embeddings {
            self.parallel_lm_head = Some(MlxNamedModule::new(
                VocabParallelLmHead::unloaded(
                    self.args().hidden_size,
                    self.args().vocab_size as usize,
                    self.args().weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )?,
                ParameterSpec::trainable("lm_head.weight")
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )?);
        }
        self.geometries = Some(Arc::new(geometries));
        self.topology = Some(build.topology());
        Ok(())
    }

    fn unit_factory(&self, external_experts: bool) -> Result<KimiLinearParallelUnitFactory, Error> {
        Ok(KimiLinearParallelUnitFactory {
            args: self.args().clone(),
            geometries: Arc::clone(self.geometries.as_ref().ok_or_else(|| {
                Error::Parallel("parallel Kimi Linear geometry is not configured".into())
            })?),
            external_experts,
        })
    }

    fn local_state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        self.state_layout
            .clone()
            .ok_or_else(|| Error::Parallel("parallel Kimi Linear state is not configured".into()))
    }

    fn execution_context<'a>(
        &self,
        group: &'a safemlx::distributed::Group,
        stream: &'a Stream,
    ) -> Result<
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'a>,
        Error,
    > {
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
            self.topology.ok_or_else(|| {
                Error::Parallel("parallel Kimi Linear topology is not configured".into())
            })?,
            group,
            stream,
        )
    }
}

impl Parameterized<Array> for KimiLinearParallelComposition {
    fn visit_parameters<'a, V: ParameterVisitor<'a, Array>>(&'a self, visitor: &mut V) {
        if let Some(embedding) = &self.parallel_embedding {
            embedding.visit_parameters(visitor);
        }
        self.architecture
            .static_modules()
            .norm
            .visit_parameters(visitor);
        if let Some(head) = &self.parallel_lm_head {
            head.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, Array>>(&'a mut self, visitor: &mut V) {
        if let Some(embedding) = &mut self.parallel_embedding {
            embedding.visit_parameters_mut(visitor);
        }
        self.architecture
            .static_modules_mut()
            .norm
            .visit_parameters_mut(visitor);
        if let Some(head) = &mut self.parallel_lm_head {
            head.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        if let Some(embedding) = &mut self.parallel_embedding {
            embedding.set_trainable(trainable);
        }
        self.architecture
            .static_modules_mut()
            .norm
            .set_trainable(trainable);
        if let Some(head) = &mut self.parallel_lm_head {
            head.set_trainable(trainable);
        }
    }
}

impl LayeredArchitecture<MlxBackend, MlxHybridState> for KimiLinearParallelComposition {
    type Input<'a> = eredu_architectures::decoder::LayeredInput<'a, Array>;
    type StaticModules = Self;
    type Unit = NeutralBlock;
    type ForwardContext = eredu_architectures::kimi_linear::ForwardContext<Array>;
    type RetainedContextValues<'a> = std::option::Iter<'a, Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.args().model_type
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::chain(["target"]).map_err(Into::into)
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            usize::try_from(self.args().num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid Kimi Linear layer count".into()))
        } else {
            Err(Error::Parallel(format!(
                "Kimi Linear has no execution group {group}"
            )))
        }
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::Parallel(format!("Kimi Linear has no unit {index}")));
        }
        Ok(format!("model.layers.{index}"))
    }
    fn static_modules(&self) -> &Self {
        self
    }
    fn static_modules_mut(&mut self) -> &mut Self {
        self
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<NeutralBlock, Error> {
        self.unit_path(group, index)?;
        Block::new_with_geometry(
            self.args(),
            index,
            *self
                .geometries
                .as_ref()
                .and_then(|values| values.get(index))
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "parallel Kimi Linear unit {index} is not configured"
                    ))
                })?,
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
    fn begin_forward<'a>(
        &mut self,
        _: Self::Input<'a>,
        _: &mut MlxHybridState,
        _: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        Err(Error::Parallel(
            "parallel Kimi Linear requires collective execution".into(),
        ))
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        _: &mut MlxHybridState,
        _: &mut Self::ForwardContext,
        _: &Stream,
    ) -> Result<Array, Error> {
        if group == 0 && dependencies.is_empty() {
            Ok(initial.clone())
        } else {
            Err(Error::Parallel(format!(
                "Kimi Linear group {group} received {} dependencies",
                dependencies.len()
            )))
        }
    }
    fn forward_unit(
        &mut self,
        _: usize,
        _: usize,
        _: &mut NeutralBlock,
        _: &Array,
        _: &mut MlxHybridState,
        _: &mut Self::ForwardContext,
        _: &Stream,
    ) -> Result<Array, Error> {
        Err(Error::Parallel(
            "parallel Kimi Linear requires collective execution".into(),
        ))
    }
    fn finish_forward(
        &mut self,
        _: &Array,
        _: &mut MlxHybridState,
        _: &Self::ForwardContext,
        _: &Stream,
    ) -> Result<Array, Error> {
        Err(Error::Parallel(
            "parallel Kimi Linear requires collective execution".into(),
        ))
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::retained_context_values(&self.architecture, forward, group, index)
    }
}

impl ParallelLayeredArchitecture<MlxBackend, MlxHybridState> for KimiLinearParallelComposition {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .parallel_embedding
            .as_mut()
            .ok_or_else(|| {
                Error::Parallel("parallel Kimi Linear embedding is not configured".into())
            })?
            .forward(input.tokens, &execution)?;
        let layout = self.local_state_layout()?;
        self.architecture
            .begin_embedded_with_layout(hidden, input.mask, state, &layout, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
    fn forward_unit_parallel(
        &mut self,
        _: usize,
        index: usize,
        unit: &mut NeutralBlock,
        hidden: &Array,
        state: &mut MlxHybridState,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.architecture
            .forward_block_parallel(index, unit, hidden, state, forward, group, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _: &mut MlxHybridState,
        _: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .architecture
            .static_modules_mut()
            .norm
            .forward(hidden, stream)?;
        let logits = match &mut self.parallel_lm_head {
            Some(head) => head.forward(&hidden, &execution)?,
            None => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| {
                    Error::Parallel("parallel Kimi Linear embedding is not configured".into())
                })?
                .project_logits(&hidden, &execution)?,
        };
        logits.all_gather(&execution)
    }
}

enum KimiLinearExecution {
    Resident(Box<ResidentRuntime>),
    Layerwise(Box<BoundedRuntime>),
    TensorParallelResident(Box<ParallelResidentRuntime>),
    TensorParallelLayerwise(Box<ParallelBoundedRuntime>),
}

fn execution_layout(layer_count: usize) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["target"])
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    ExecutionUnitLayout::new(&graph, [layer_count])
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn load_model_args(model_dir: &Path) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    eredu_architectures::kimi_linear::model_args_from_config_reader(file)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn resolve_store(
    store: Arc<dyn CheckpointSource>,
    args: &ModelArgs,
) -> Result<Arc<dyn CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend
            != eredu_checkpoint::store::WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = eredu_architectures::kimi_linear::safetensors_plan(args)
        .map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "Kimi Linear checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn expert_source_from_normalized(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: i32,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, Error> {
    let runtime = projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|candidate| normalized.contains_key(candidate))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Kimi Linear checkpoint is missing expert {expert} projection under {prefix}"
            ))
        })?;
    let key = normalized
        .get(&runtime)
        .expect("normalized Kimi expert key exists");
    Ok(DerivedWeightRecipe::source(
        key,
        eredu_checkpoint::store::TensorSelection::Full,
    ))
}

fn expert_source(
    store: &dyn CheckpointSource,
    prefix: &str,
    expert: i32,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, Error> {
    expert_source_from_normalized(
        &normalized_checkpoint_keys(store),
        prefix,
        expert,
        projections,
    )
}

fn normalized_checkpoint_keys(store: &dyn CheckpointSource) -> BTreeMap<String, String> {
    store
        .source_keys()
        .into_iter()
        .map(|raw| {
            let runtime = canonical_checkpoint_name(&raw).replace(".block_sparse_moe.", ".mlp.");
            (runtime, raw)
        })
        .collect()
}

fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
    include_experts: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    use eredu_checkpoint::store::TensorSelection;

    let mut recipes = BTreeMap::new();
    let root = format!("model.layers.{layer}");
    let normalized = normalized_checkpoint_keys(store);
    for (runtime, raw) in &normalized {
        if runtime.starts_with(&format!("{root}.mlp."))
            && !runtime.contains(".mlp.experts.")
            && runtime != raw
        {
            recipes.insert(
                runtime.clone(),
                DerivedWeightRecipe::source(raw, TensorSelection::Full),
            );
        }
    }
    let attention = format!("{root}.self_attn");
    let projection = (args.kda_config.num_heads * args.kda_config.head_dim) as usize;
    for local in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
        let name = format!("{attention}.{local}");
        if store.source_metadata(&name).is_ok() {
            recipes.insert(
                name.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(&name, TensorSelection::Full)),
                    shape: vec![
                        projection,
                        1,
                        args.kda_config.short_conv_kernel_size as usize,
                    ],
                },
            );
        }
    }
    let a_log = format!("{attention}.A_log");
    if store.source_metadata(&a_log).is_ok() {
        let mut recipe = DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source(&a_log, TensorSelection::Full)),
            shape: vec![1, 1, args.kda_config.num_heads as usize, 1],
        };
        if store.source_diagnostics()?.backend == eredu_checkpoint::store::WeightStoreBackend::Gguf
        {
            recipe = DerivedWeightRecipe::NegLog {
                input: Box::new(recipe),
            };
        }
        recipes.insert(a_log, recipe);
    }
    let policy = args.layer_policy(layer).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("Kimi Linear has no layer policy {layer}"))
    })?;
    if !include_experts
        || policy.feed_forward != eredu_architectures::kimi_linear::FeedForwardPolicy::SparseMoe
    {
        return Ok(recipes);
    }
    let prefix = format!("{root}.mlp.experts");
    let gate_up = format!("{prefix}.gate_up_proj");
    let down = format!("{prefix}.down_proj");
    if let (Some(gate_up_source), Some(down_source)) =
        (normalized.get(&gate_up), normalized.get(&down))
    {
        if gate_up_source != &gate_up {
            recipes.insert(
                gate_up.clone(),
                DerivedWeightRecipe::source(gate_up_source, TensorSelection::Full),
            );
        }
        if down_source != &down {
            recipes.insert(
                down.clone(),
                DerivedWeightRecipe::source(down_source, TensorSelection::Full),
            );
        }
        return Ok(recipes);
    }
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    if normalized.contains_key(&gate) && normalized.contains_key(&up) {
        recipes.insert(
            gate_up,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: [gate, up]
                    .into_iter()
                    .map(|runtime| {
                        DerivedWeightRecipe::source(
                            normalized.get(&runtime).expect("normalized projection key"),
                            TensorSelection::Full,
                        )
                    })
                    .collect(),
            },
        );
        return Ok(recipes);
    }
    let mut gate_up_inputs = Vec::new();
    let mut down_inputs = Vec::new();
    for expert in 0..args.num_experts {
        gate_up_inputs.push(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                expert_source_from_normalized(&normalized, &prefix, expert, &["w1", "gate_proj"])?,
                expert_source_from_normalized(&normalized, &prefix, expert, &["w3", "up_proj"])?,
            ],
        });
        down_inputs.push(expert_source_from_normalized(
            &normalized,
            &prefix,
            expert,
            &["w2", "down_proj"],
        )?);
    }
    recipes.insert(
        gate_up,
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: gate_up_inputs,
        },
    );
    recipes.insert(
        down,
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: down_inputs,
        },
    );
    Ok(recipes)
}

fn planned_expert_binding(
    name: impl Into<String>,
    recipe: DerivedWeightRecipe,
    store: &dyn CheckpointSource,
) -> Result<PlannedBinding, Error> {
    let metadata = recipe.infer(store)?;
    Ok(PlannedBinding {
        target_name: name.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    })
}

pub(crate) fn expert_catalog(
    args: &ModelArgs,
    store: &dyn CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    use eredu_checkpoint::store::TensorSelection;

    if !args.has_sparse_moe_layers() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching requires Kimi Linear-MoE".into(),
        ));
    }
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut entries = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != eredu_architectures::kimi_linear::FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let packed_gate_up = format!("{prefix}.gate_up_proj");
        let packed_down = format!("{prefix}.down_proj");
        for expert in 0..args.num_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let selection = TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            };
            let mut planned = Vec::new();
            if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
                for (name, key) in [
                    ("gate_up_proj", packed_gate_up.clone()),
                    ("down_proj", packed_down.clone()),
                ] {
                    planned.push(planned_expert_binding(
                        name,
                        DerivedWeightRecipe::source(key, selection.clone()),
                        store,
                    )?);
                }
                for (name, key) in [
                    ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
                    ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
                    ("down_proj_scales", format!("{packed_down}_scales")),
                    ("down_proj_biases", format!("{packed_down}_biases")),
                ] {
                    if keys.contains(&key) {
                        planned.push(planned_expert_binding(
                            name,
                            DerivedWeightRecipe::source(key, selection.clone()),
                            store,
                        )?);
                    }
                }
            } else if keys.contains(&format!("{prefix}.gate_proj"))
                && keys.contains(&format!("{prefix}.up_proj"))
                && keys.contains(&packed_down)
            {
                planned.push(planned_expert_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                format!("{prefix}.gate_proj"),
                                selection.clone(),
                            ),
                            DerivedWeightRecipe::source(
                                format!("{prefix}.up_proj"),
                                selection.clone(),
                            ),
                        ],
                    },
                    store,
                )?);
                planned.push(planned_expert_binding(
                    "down_proj",
                    DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
                    store,
                )?);
                for suffix in ["_scales", "_biases"] {
                    let gate = format!("{prefix}.gate_proj{suffix}");
                    let up = format!("{prefix}.up_proj{suffix}");
                    if keys.contains(&gate) && keys.contains(&up) {
                        planned.push(planned_expert_binding(
                            &format!("gate_up_proj{suffix}"),
                            DerivedWeightRecipe::Concatenate {
                                axis: 1,
                                inputs: vec![
                                    DerivedWeightRecipe::source(gate, selection.clone()),
                                    DerivedWeightRecipe::source(up, selection.clone()),
                                ],
                            },
                            store,
                        )?);
                    }
                    let down = format!("{packed_down}{suffix}");
                    if keys.contains(&down) {
                        planned.push(planned_expert_binding(
                            &format!("down_proj{suffix}"),
                            DerivedWeightRecipe::source(down, selection.clone()),
                            store,
                        )?);
                    }
                }
            } else {
                if args.weight_quantization_for(&packed_gate_up).is_some()
                    || args.weight_quantization_for(&packed_down).is_some()
                {
                    return Err(Error::Quantization(
                        "split Kimi Linear experts cannot be lazily load-time quantized".into(),
                    ));
                }
                let gate = expert_source(store, &prefix, expert as i32, &["w1", "gate_proj"])?;
                let up = expert_source(store, &prefix, expert as i32, &["w3", "up_proj"])?;
                let down = expert_source(store, &prefix, expert as i32, &["w2", "down_proj"])?;
                planned.extend([
                    planned_expert_binding(
                        "gate_up_proj",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![gate, up],
                            }],
                        },
                        store,
                    )?,
                    planned_expert_binding(
                        "down_proj",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![down],
                        },
                        store,
                    )?,
                ]);
            }
            let bindings = BindingPlan::new(planned)
                .and_then(|plan| plan.build_bindings(store))
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total
                    .checked_add(binding.expected_bytes())
                    .ok_or_else(|| Error::UnsupportedArchitecture("expert bytes overflowed".into()))
            })?;
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

fn cached_provider<'a>(
    cache: &'a ExpertCache,
    args: &'a ModelArgs,
) -> CachedGatedProductExpertProvider<'a, impl FnMut(usize) -> CachedGatedProductBankSpec + 'a> {
    CachedGatedProductExpertProvider::new(cache, move |layer| {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        CachedGatedProductBankSpec {
            hidden_dimensions: args.hidden_size,
            intermediate_dimensions: args.moe_intermediate_size,
            gate_up_quantization: args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
            down_quantization: args.weight_quantization_for(&format!("{prefix}.down_proj")),
            gate_up_bias: false,
            down_bias: false,
            policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        }
    })
}

fn load_neutral(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<KimiLinearModel, Error> {
    let count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Kimi Linear layer count".into()))?;
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        architecture.static_modules_mut(),
        KimiLinearUnitFactory {
            args: args.clone(),
            external_experts,
        },
        execution_layout(count)?,
        options,
        stream,
        weights_stream,
        move |key| {
            key.ends_with(".rotary_emb.inv_freq")
                || (external_experts && key.contains(".mlp.experts."))
        },
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |index, unit, store, _| {
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                unit_recipes(store, &binding_args, index, !external_experts)?,
                |name| external_experts && name.contains(".mlp.experts."),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization);
    metadata.set_materialization(materialization);
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let execution = if options.is_fully_resident() {
        KimiLinearExecution::Resident(Box::new(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        )))
    } else {
        KimiLinearExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(KimiLinearModel {
        args,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: None,
        parallel_rank: None,
    })
}

fn load_neutral_parallel(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<KimiLinearModel, Error> {
    let count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::Parallel("invalid Kimi Linear layer count".into()))?;
    let mut composition = KimiLinearParallelComposition::new(args.clone(), stream)?;
    let mut planner = build.planner();
    for group in eredu_architectures::kimi_linear::static_parallel_parameter_groups::<MlxBackend>(
        composition.architecture.static_modules(),
    )? {
        planner.register(group)?;
    }
    for layer in 0..count {
        let block = Block::<MlxBackend>::new(&args, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        for group in
            eredu_architectures::kimi_linear::layer_parallel_parameter_groups(&block, &args, layer)?
        {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Kimi Linear declared no tensor-parallel parameters".into(),
        ));
    }
    composition.configure(build, &layout, stream)?;
    let state_layout = composition.local_state_layout()?;
    let factory = composition.unit_factory(external_experts)?;

    let global_static = MlxModule::new(composition.architecture.static_modules().clone());
    let global_static_bindings = build_module_bindings(&global_static, "", store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for layer in 0..count {
        let block = Block::<MlxBackend>::new(&args, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let bindings = build_module_bindings_with_recipes_excluding(
            &MlxModule::new(block),
            "",
            store.as_ref(),
            unit_recipes(store.as_ref(), &args, layer, !external_experts)?,
            |name| external_experts && name.contains(".mlp.experts."),
        )?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&bindings)?)
            .ok_or_else(|| {
                Error::Parallel("global Kimi Linear parameter bytes overflowed".into())
            })?;
    }

    let shared_layout = Arc::new(layout);
    let static_layout = Arc::clone(&shared_layout);
    let unit_layout = Arc::clone(&shared_layout);
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut composition,
        factory,
        execution_layout(count)?,
        options,
        stream,
        weights_stream,
        move |key| {
            key.ends_with(".rotary_emb.inv_freq")
                || (external_experts && key.contains(".mlp.experts."))
        },
        move |modules, store| {
            let global = MlxModule::new(modules.architecture.static_modules().clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, "", store, &static_layout)
        },
        move |layer, _local, store, stream| {
            let global = Block::<MlxBackend>::new(&binding_args, layer, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bindings = build_module_bindings_with_recipes_excluding(
                &MlxModule::new(global),
                "",
                store,
                unit_recipes(store, &binding_args, layer, !external_experts)?,
                |name| external_experts && name.contains(".mlp.experts."),
            )?;
            shard_layer_bindings(
                bindings,
                &format!("model.layers.{layer}"),
                store,
                &unit_layout,
            )
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("local Kimi Linear parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("device Kimi Linear parameter bytes overflowed".into()))?;
    let info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        shared_layout
            .tensors()
            .map(|(target, _)| target.to_owned())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if options.is_fully_resident() {
            local_parameter_bytes
        } else {
            metadata.static_device_bytes()
        },
        maximum_device_parameter_bytes,
    );
    let rank =
        crate::backend::mlx::cache::prompt_cache_topology(build.topology()).cache_rank_identity();
    let execution = if options.is_fully_resident() {
        KimiLinearExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new(
            composition,
            policy.into_resident(stream)?,
        )))
    } else {
        KimiLinearExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            composition,
            policy,
        )))
    };
    Ok(KimiLinearModel {
        args,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: Some(info),
        parallel_rank: rank,
    })
}

fn quantize_store(
    store: Arc<dyn CheckpointSource>,
    args: &ModelArgs,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target = args.clone();
    target.weight_quantization = Some(quantization);
    target.quantized_weight_configs = None;
    let source = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let destination = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let source_args = args.clone();
    let target_args = target.clone();
    let (store, report) = quantize_parameterized_store(
        store,
        source.static_modules(),
        destination.static_modules(),
        move |index, stream| {
            Block::<MlxBackend>::new(&source_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        move |index, stream| {
            Block::<MlxBackend>::new(&target_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        usize::try_from(args.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture("invalid Kimi Linear layer count".into())
        })?,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

/// Kimi Linear causal model whose equations are owned by `eredu-architectures`.
pub struct KimiLinearModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    execution: KimiLinearExecution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    parallel_rank: Option<crate::CacheRankIdentity>,
}

impl KimiLinearModel {
    /// Returns validated family policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns canonical residency metadata.
    pub const fn residency_metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Returns parallel metadata when a distributed binder supplied it.
    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Creates device-resident heterogeneous state.
    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone())
            .expect("validated Kimi Linear state must be realizable by MLX")
    }

    /// Creates state with the requested attention residency.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxHybridState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                MlxHybridState::paged(self.state_layout.clone(), manager, self.parallel_rank)
                    .map_err(Into::into)
            }
        }
    }

    /// Returns cache residency telemetry.
    pub fn cache_residency_report(
        &self,
        cache: &MlxHybridState,
    ) -> Result<Option<eredu_runtime::CacheResidencyReport>, Error> {
        cache.residency_report().map_err(Into::into)
    }

    /// Returns weight residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().residency_report(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().residency_report(),
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.policy().residency_report()
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().residency_report()
            }
        }
    }

    /// Returns disk streaming telemetry when enabled.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            KimiLinearExecution::Resident(_) => Ok(None),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().dense_stream_report(),
            KimiLinearExecution::TensorParallelResident(_) => Ok(None),
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().dense_stream_report()
            }
        }
    }

    /// Returns independent expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the persistent checkpoint source.
    pub fn checkpoint_store(&self) -> &dyn CheckpointSource {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().checkpoint_store(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().checkpoint_store(),
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.policy().checkpoint_store()
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().checkpoint_store()
            }
        }
    }

    pub(crate) fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.policy().checkpoint_store_arc()
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().checkpoint_store_arc()
            }
        }
    }

    /// Returns the canonical prompt-cache fingerprint.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(eredu_architectures::kimi_linear::prompt_cache_architecture_fingerprint(&self.args))
    }

    /// Returns the exact rank-local state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.state_layout.layers().clone())
    }

    /// Returns the canonical prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        eredu_architectures::kimi_linear::state_identity(
            &self.args,
            &self.state_layout,
            0,
            self.parallel_info
                .as_ref()
                .map_or_else(PromptCacheTopology::default, |info| {
                    crate::backend::mlx::cache::prompt_cache_topology(info.topology())
                }),
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
        .prompt_cache_identity(&self.state_layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Persists paged attention and all fixed components atomically.
    pub fn save_prompt_cache(
        &self,
        cache: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.save_prompt_cache_with_identity(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            &self.prompt_cache_model_identity()?,
        )
    }

    pub(crate) fn save_prompt_cache_with_identity(
        &self,
        cache: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        identity: &PromptCacheModelIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(&descriptor, identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    /// Opens paged attention and restores all fixed components.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MlxHybridState, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        self.load_prompt_cache_with_identity(
            directory,
            expected,
            prefix_token_ids,
            options,
            &identity,
            stream,
        )
    }

    pub(crate) fn load_prompt_cache_with_identity(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        identity: &PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(MlxHybridState, PromptCacheManifest), Error> {
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut cache =
            MlxHybridState::paged(self.state_layout.clone(), manager, self.parallel_rank)?;
        cache.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Exception::custom("prompt-cache prefix exceeds i32"))?,
            &identity.layer_prefix_offsets,
        )?;
        Ok((cache, manifest))
    }

    /// Executes embedding, the scheduled physical blocks, normalization, and logits.
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let result = {
                let mut provider = cached_provider(&expert_cache, &args);
                self.forward_with_provider(tokens, None, cache, &mut provider, stream)
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        let input = eredu_architectures::decoder::LayeredInput { tokens, mask: None };
        match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.forward(input, cache, stream),
            KimiLinearExecution::Layerwise(runtime) => runtime.forward(input, cache, stream),
            KimiLinearExecution::TensorParallelResident(_)
            | KimiLinearExecution::TensorParallelLayerwise(_) => {
                return Err(Error::Parallel(
                    "tensor-parallel Kimi Linear requires collective execution".into(),
                ))
            }
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn forward_with_provider<P>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let input = eredu_architectures::decoder::LayeredInput { tokens, mask };
        let hook = |architecture: &mut NeutralArchitecture,
                    group: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &Array,
                    state: &mut MlxHybridState,
                    forward: &mut eredu_architectures::kimi_linear::ForwardContext<Array>,
                    context: &Stream| {
            if group != 0 {
                return Err(eredu_nn::Error::backend(format!(
                    "Kimi Linear received execution group {group}"
                )));
            }
            architecture.forward_block_with_feed_forward(
                index,
                block,
                hidden,
                state,
                forward,
                context,
                |policy, normalized, context| {
                    policy.forward_with_provider(normalized, pass, context, provider)
                },
            )
        };
        match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            KimiLinearExecution::Layerwise(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            _ => {
                return Err(Error::Parallel(
                    "tensor-parallel Kimi Linear expert cache requires collective execution".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Executes a replicated neutral pass with activation intervention and
    /// normalized routing observations.
    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let expert_cache = self.expert_cache.take();
        let result = {
            let mut observer = NeutralKimiLinearObserver { inner: observer };
            match expert_cache.as_ref() {
                Some(expert_cache) => {
                    let args = self.args.clone();
                    let mut provider = cached_provider(expert_cache, &args);
                    self.forward_observed_with_provider(
                        tokens,
                        mask,
                        cache,
                        &mut provider,
                        stream,
                        &mut observer,
                    )
                }
                None => {
                    let mut provider = eredu_runtime::ResidentExpertProvider;
                    self.forward_observed_with_provider(
                        tokens,
                        mask,
                        cache,
                        &mut provider,
                        stream,
                        &mut observer,
                    )
                }
            }
        };
        self.expert_cache = expert_cache;
        result
    }

    fn forward_observed_with_provider<P>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
        observer: &mut NeutralKimiLinearObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let expert_count = self.args.num_experts;
        let input = eredu_architectures::decoder::LayeredInput { tokens, mask };
        let hook = |architecture: &mut NeutralArchitecture,
                    group: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &Array,
                    state: &mut MlxHybridState,
                    forward: &mut eredu_architectures::kimi_linear::ForwardContext<Array>,
                    context: &Stream| {
            if group != 0 {
                return Err(eredu_nn::Error::backend(format!(
                    "Kimi Linear received execution group {group}"
                )));
            }
            let path = format!("model.layers.{index}");
            observer.observe(&format!("{path}.input"), hidden)?;
            let output = architecture.forward_block_with_feed_forward(
                index,
                block,
                hidden,
                state,
                forward,
                context,
                |policy, normalized, context| {
                    policy.forward_observed_with_provider(
                        &format!("{path}.mlp"),
                        expert_count,
                        normalized,
                        pass,
                        context,
                        observer,
                        provider,
                    )
                },
            )?;
            eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &output)
        };
        match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            KimiLinearExecution::Layerwise(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            _ => {
                return Err(Error::Parallel(
                    "tensor-parallel Kimi Linear observation requires distributed observation"
                        .into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Runs the neutral decoder while delegating routed experts to an
    /// external placement policy such as expert parallelism.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        self.forward_with_provider(tokens, mask, cache, &mut provider, stream)
    }

    /// Runs TP-sharded neutral blocks and dense operators while delegating
    /// routed experts to the matching expert-parallel subgroup.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let input = eredu_architectures::decoder::LayeredInput { tokens, mask };
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        let hook = |composition: &mut KimiLinearParallelComposition,
                    group_index: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &Array,
                    state: &mut MlxHybridState,
                    forward: &mut eredu_architectures::kimi_linear::ForwardContext<Array>,
                    parallel: &safemlx::distributed::Group,
                    context: &Stream| {
            if group_index != 0 {
                return Err(Error::Parallel(format!(
                    "Kimi Linear received execution group {group_index}"
                )));
            }
            composition
                .architecture
                .forward_block_parallel_with_feed_forward(
                    index,
                    block,
                    hidden,
                    state,
                    forward,
                    parallel,
                    context,
                    |policy, normalized, parallel, context| {
                        policy
                            .forward_parallel_with_provider(
                                normalized,
                                pass,
                                parallel,
                                context,
                                &mut provider,
                            )
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                    },
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        };
        match &mut self.execution {
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.forward_parallel_with_unit_executor(input, cache, group, stream, hook)
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.forward_parallel_with_unit_executor(input, cache, group, stream, hook)
            }
            _ => {
                return Err(Error::Parallel(
                    "Kimi Linear was not loaded for tensor plus expert parallelism".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Clears transient decoder weights for bounded execution.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        match &self.execution {
            KimiLinearExecution::Resident(_) => Ok(()),
            KimiLinearExecution::Layerwise(runtime) => {
                runtime.policy().clear_device_group("target")
            }
            KimiLinearExecution::TensorParallelResident(_) => Ok(()),
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().clear_device_group("target")
            }
        }
    }

    /// Executes a rank-local tensor-parallel forward pass.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let input = eredu_architectures::decoder::LayeredInput { tokens, mask: None };
        match &mut self.execution {
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.forward_parallel(input, cache, group, stream)
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.forward_parallel(input, cache, group, stream)
            }
            _ => {
                return Err(Error::Parallel(
                    "Kimi Linear was not loaded for tensor parallelism".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl CausalModel<MlxHybridState> for KimiLinearModel {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads SafeTensors Kimi Linear through one neutral model object.
pub fn load_kimi_linear_model(
    model_dir: impl AsRef<Path>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let model_dir = model_dir.as_ref();
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    let args = load_model_args(model_dir)?;
    let quantize = quantization
        .map(|requested| {
            should_quantize_on_load("Kimi Linear", args.weight_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    let store = resolve_store(store, &args)?;
    if let Some(quantization) = quantize {
        let (store, target, report) = quantize_store(store, &args, quantization, stream)?;
        let mut model = load_neutral(
            store,
            target,
            options,
            stream,
            weights_stream,
            Some(report),
            expert_options.is_some(),
        )?;
        if let Some(expert_options) = expert_options {
            attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let mut model = load_neutral(
        store,
        args,
        options,
        stream,
        weights_stream,
        None,
        expert_options.is_some(),
    )?;
    if let Some(expert_options) = expert_options {
        attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_expert_cache(
    model: &mut KimiLinearModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = expert_catalog(&model.args, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors Kimi Linear through generalized tensor-parallel placement.
pub(crate) fn load_kimi_linear_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = load_model_args(model_dir)?;
    let options = options.into();
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    let store = resolve_store(store, &args)?;
    load_neutral_parallel(store, args, options, build, stream, weights_stream, false)
}

/// Loads only nonexpert Kimi Linear weights for an external expert-placement policy.
pub(crate) fn load_external_expert_base_with_store(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let store = resolve_store(store, &args)?;
    load_neutral(store, args, options, stream, weights_stream, None, true)
}

/// Loads tensor-partitioned nonexpert Kimi Linear weights for TP+EP execution.
pub(crate) fn load_external_expert_parallel_base_with_store(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let store = resolve_store(store, &args)?;
    load_neutral_parallel(store, args, options, build, stream, weights_stream, true)
}

struct GgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::kimi_linear::GgufTensorCatalog for GgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        self.0.any_gguf_tensor(predicate)
    }
}

pub(crate) struct PreparedGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<PreparedGguf, Error> {
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) => value.as_str(),
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF general.architecture must be a string".into(),
            ))
        }
    };
    if architecture != "kimi-linear" {
        return Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear GGUF loader received architecture {architecture:?}"
        )));
    }
    let gguf_architecture = crate::core::GgufArchitecture::resolve(architecture)?;
    crate::composition::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let mut args = eredu_architectures::kimi_linear::model_args_from_gguf_catalog(
        &GgufCatalog(checkpoint),
        metadata,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let translate = eredu_architectures::kimi_linear::translate_gguf_weight_name;
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate)?;
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != eredu_architectures::kimi_linear::FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        if let Some(config) = configs.remove(&format!("{prefix}.gate_proj")) {
            configs.remove(&format!("{prefix}.up_proj"));
            configs.insert(format!("{prefix}.gate_up_proj"), config);
        }
    }
    args.quantized_weight_configs = Some(configs);
    args.weight_quantization = None;
    args.validate()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(PreparedGguf {
        args,
        eos_token_ids: crate::backend::mlx::gguf_eos_token_ids(metadata)?,
    })
}

/// Loads a GGUF checkpoint through the same neutral Kimi Linear model object.
pub(crate) fn load_kimi_linear_gguf_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(KimiLinearModel, Vec<u32>), Error> {
    let prepared = prepare_gguf(checkpoint, metadata)?;
    let expert_options = residency.expert_cache();
    let plan = eredu_architectures::kimi_linear::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        &plan,
        eredu_architectures::kimi_linear::translate_gguf_weight_name,
        residency.max_mapped_shards(),
    )?);
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_store(store, &prepared.args, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, prepared.args, None),
    };
    let mut model = load_neutral(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        materialization,
        expert_options.is_some(),
    )?;
    if let Some(expert_options) = expert_options {
        attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok((model, prepared.eos_token_ids))
}

/// Loads GGUF Kimi Linear with tensor-parallel placement.
pub(crate) fn load_kimi_linear_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(KimiLinearModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared = prepare_gguf(checkpoint, metadata)?;
    let plan = eredu_architectures::kimi_linear::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        &plan,
        eredu_architectures::kimi_linear::translate_gguf_weight_name,
        options.max_mapped_shards(),
    )?);
    let model = load_neutral_parallel(
        store,
        prepared.args,
        options,
        build,
        stream,
        weights_stream,
        false,
    )?;
    Ok((model, prepared.eos_token_ids))
}
