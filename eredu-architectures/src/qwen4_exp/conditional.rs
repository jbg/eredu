//! Shared graph traversal of retained vision blocks and the ordinary Flash-Next target.
use super::{
    media::{MediaIngress, PreparedMediaInput},
    target::{BoundTargetSpec, TargetForward, TargetInput, TargetModel, Unit as TargetUnit},
};
use crate::qwen::vision::{VisionBlock, VisionConfig, VisionState, VisionStatic};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, EmbeddingOperator, Error, GroupedNeuralBackend,
    Parameterized, Tensor,
};
use eredu_runtime::*;

/// Stable optional encoder group, preceding the ordinary target group.
pub const VISION_GROUP: &str = "vision";
mod composite;
mod partitioned;
pub mod prefill;
const TARGET_INDEX: usize = 0;
const VISION_INDEX: usize = 1;

pub(crate) fn graph() -> Result<ExecutionGraph, Error> {
    // Flat parameter-bank coordinates stay target-first. Dependency order is
    // independent of declaration order, so the encoder still executes first.
    ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::with_dependencies(
                crate::decoder::TARGET_EXECUTION_GROUP,
                [VISION_GROUP],
            ),
            ExecutionGroupSpec::root(VISION_GROUP),
        ],
        crate::decoder::TARGET_EXECUTION_GROUP,
    )
    .map_err(Error::backend)
}
pub(crate) fn vision_transport() -> ArchitectureGroupTransport {
    ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::VisionEncoder,
        first_owner_static_roles: vec!["vision".into()],
        last_owner_static_roles: vec!["vision".into()],
        merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: Some(ArchitectureParallelSubgroup::TensorSharded),
        request_optional: true,
    }
}

/// Pinned modules. The target owns the single vocabulary and residual readout.
pub struct ConditionalStatic<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    target: TargetModel<B>,
    vision: VisionStatic<B>,
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend + Clone> Clone for ConditionalStatic<B> {
    fn clone(&self) -> Self {
        Self {
            target: self.target.clone(),
            vision: self.vision.clone(),
        }
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> Parameterized<B::Tensor>
    for ConditionalStatic<B>
{
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, B::Tensor>>(
        &'a self,
        visitor: &mut V,
    ) {
        self.target
            .decoder
            .static_modules()
            .visit_parameters(visitor);
        self.vision.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.target
            .decoder
            .static_modules_mut()
            .visit_parameters_mut(visitor);
        self.vision.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.target
            .decoder
            .static_modules_mut()
            .set_trainable(trainable);
        self.vision.set_trainable(trainable);
    }
}
/// One independently materialized encoder block or target execution unit.
#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum ConditionalUnit<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    /// Shared Qwen vision equation.
    Vision(VisionBlock<B>),
    /// Existing recurrent/indexed/lexical target equation.
    Target(TargetUnit<B>),
}
/// One ordinary target pass, or an initial admitted media prompt.
pub enum ConditionalInput<'a, T> {
    /// Text decode or already assembled bounded target input.
    Target(TargetInput<'a, T>),
    /// Prepared media context; the shared graph runs only required encoder blocks.
    Media(&'a PreparedMediaInput<T>),
    /// One absolute target span, optionally reusing a completed encoder result.
    MediaChunk {
        /// Exact request prepared once before advancement.
        prepared: &'a PreparedMediaInput<T>,
        /// Absolute decoder span within the request.
        range: std::ops::Range<i32>,
        /// Completed encoder output from the same request.
        projected: Option<&'a T>,
    },
}
/// Request-owned encoder context and the target's exact original-ID provenance.
pub struct ConditionalForward<T> {
    media: Option<PreparedMediaInput<T>>,
    range: Option<std::ops::Range<i32>>,
    vision: Option<VisionState<T>>,
    initial: Option<T>,
    projected: Option<T>,
    target: Option<TargetForward<T>>,
    incoming_target: Option<super::input::RequestBoundary<T>>,
}

/// Request/encoder tensor roles in an active conditional forward context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderContextTensorRole {
    /// Prepared whole-request media products.
    Prepared(super::media::PreparedMediaTensorRole),
    /// Rotary products and intermediate features retained between encoder units.
    Vision(crate::qwen::vision::VisionStateTensorRole),
    /// Initial encoder activation retained by the forward context.
    Initial,
    /// Final encoder projection retained for target execution.
    Projected,
}
impl<T: Tensor> ConditionalForward<T> {
    /// Named request/encoder roots owned by this forward context. The unit's
    /// current activation and the original admission-proof input have different
    /// owners; callers can join those borrowed roots before making one survey.
    /// Target state, parameters and graph-internal scratch are excluded.
    pub fn encoder_storage_values(&self) -> impl Iterator<Item = (EncoderContextTensorRole, &T)> {
        self.media
            .iter()
            .flat_map(|media| media.storage_values())
            .map(|(role, value)| (EncoderContextTensorRole::Prepared(role), value))
            .chain(
                self.vision
                    .iter()
                    .flat_map(|vision| vision.storage_values())
                    .map(|(role, value)| (EncoderContextTensorRole::Vision(role), value)),
            )
            .chain(
                self.initial
                    .iter()
                    .map(|value| (EncoderContextTensorRole::Initial, value)),
            )
            .chain(
                self.projected
                    .iter()
                    .map(|value| (EncoderContextTensorRole::Projected, value)),
            )
    }
}
/// Portable composition; parameter residency and submission remain shared mechanisms.
pub struct ConditionalModel<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    modules: ConditionalStatic<B>,
    ingress: MediaIngress,
    vision: VisionConfig,
    vision_local: Option<Vec<(i32, i32)>>,
    vision_input_owner: bool,
    partition_state: Option<PartitionState>,
    global_parameters: Option<ArchitectureParameterDescription>,
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ConditionalModel<B> {
    /// Constructs pinned modules only; encoder and target units are built on demand.
    pub fn new(
        spec: BoundTargetSpec,
        ingress: MediaIngress,
        vision: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        // Formats may change at selection; semantic geometry must match the retained role.
        let mut expected = ingress.vision().config().clone();
        let mut actual = vision.clone();
        expected.linear_formats.clear();
        actual.linear_formats.clear();
        if expected != actual || vision.out_hidden_size != spec.geometry().config.hidden_size {
            return Err(Error::backend(
                "conditional vision geometry differs from retained ingress",
            ));
        }
        super::media::MediaIngress::new(spec.geometry(), ingress.vision().clone())
            .map_err(Error::backend_source)?;
        Ok(Self {
            modules: ConditionalStatic {
                target: TargetModel::new(spec, context)?,
                vision: VisionStatic::new_with_root(vision.clone(), "model.visual", context)?,
            },
            ingress,
            vision,
            vision_local: None,
            vision_input_owner: true,
            partition_state: None,
            global_parameters: None,
        })
    }
    fn begin_media<S>(
        &mut self,
        input: &PreparedMediaInput<B::Tensor>,
        range: std::ops::Range<i32>,
        projected: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForward<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState:
            AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        let invalid =
            |reason| Error::backend_source(super::media::MediaInputError::Geometry(reason));
        input
            .validate_policy(&self.ingress)
            .map_err(Error::backend_source)?;
        let expected = self
            .partition_state
            .as_ref()
            .map(|state| state.layout().clone())
            .map(Ok)
            .unwrap_or_else(|| self.modules.target.spec.state_layout())?;
        if state.layout() != &expected {
            return Err(invalid("media state layout differs from target"));
        }
        if range.start < 0
            || range.start >= range.end
            || range.end > input.token_ids().dim(1)
            || range.end - range.start > self.modules.target.spec.limits.qsa.tokens
        {
            return Err(invalid("media span exceeds the admitted target invocation"));
        }
        self.modules
            .target
            .spec
            .limits
            .validate_history(range.start, range.end - range.start)
            .map_err(Error::backend_source)?;
        for index in 0..expected.len() {
            if state.layer(index).map_err(Error::backend)?.position() != range.start {
                return Err(invalid("media span differs from committed target prefix"));
            }
        }
        let vision = input.with_vision_input(|vision| {
            if range.start > 0 && vision.is_some() && projected.is_none() {
                if !self.vision_input_owner {
                    // Every local target frontier already equals range.start.
                    // This proves the initial media wave committed. Receiving
                    // target owners resume assembled residuals and need no
                    // encoder payload of their own for later prompt chunks.
                    return Ok(None);
                }
                return Err(invalid(
                    "media continuation is missing its completed encoder output",
                ));
            }
            if projected.is_some() {
                return Ok(None);
            }
            vision
                .map(|input| {
                    if self.vision_input_owner {
                        self.modules.vision.begin(input, context)
                    } else {
                        let state = self.modules.vision.prepare_state(input.grid, context)?;
                        let sequence = state
                            .full_chunks()
                            .iter()
                            .try_fold(0_i32, |total, length| total.checked_add(*length))
                            .ok_or_else(|| {
                                Error::backend("vision continuation sequence exceeds i32")
                            })?;
                        let hidden = B::Tensor::full_f32(
                            0.0,
                            &[sequence, self.vision.hidden_size],
                            context,
                        )?;
                        Ok((hidden, state))
                    }
                })
                .transpose()
        })?;
        if let Some((hidden, vision)) = vision {
            return Ok(LayeredForwardState {
                hidden: hidden.clone(),
                context: ConditionalForward {
                    media: Some(input.clone()),
                    range: Some(range),
                    vision: Some(vision),
                    initial: Some(hidden),
                    projected: None,
                    target: None,
                    incoming_target: None,
                },
            });
        }
        if self
            .partition_state
            .as_ref()
            .is_some_and(|state| state.global_layer_offset() != 0)
        {
            let geometry = self.modules.target.spec.boundary.geometry;
            let hidden = B::Tensor::full_f32(
                0.0,
                &[
                    input.token_ids().dim(0),
                    range.end - range.start,
                    geometry.streams(),
                    geometry.hidden_size(),
                ],
                context,
            )?;
            return Ok(LayeredForwardState {
                hidden,
                context: ConditionalForward {
                    media: Some(input.clone()),
                    range: Some(range),
                    vision: None,
                    initial: None,
                    projected: projected.cloned(),
                    target: None,
                    incoming_target: None,
                },
            });
        }
        let media = input
            .assemble(
                range,
                projected,
                |ids| {
                    self.modules
                        .target
                        .decoder
                        .static_modules_mut()
                        .embeddings
                        .forward(ids, context)
                },
                context,
            )
            .map_err(Error::backend_source)?;
        let next = media
            .with_target_input(|input| self.modules.target.begin_forward(input, state, context))?;
        Ok(LayeredForwardState {
            hidden: next.hidden,
            context: ConditionalForward {
                media: None,
                range: None,
                vision: None,
                initial: None,
                projected: projected.cloned(),
                target: Some(next.context),
                incoming_target: None,
            },
        })
    }
    /// Exact selected target/vision parameter geometry and media semantics.
    pub fn state_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "qwen4_exp.conditional.v1",
            [
                ("target", self.modules.target.state_fingerprint().to_owned()),
                ("media", self.ingress.identity().to_owned()),
                (
                    "vision",
                    crate::qwen::vision::prompt_cache_architecture_fingerprint(&self.vision),
                ),
            ],
        )
    }
    /// Retained request policy; projected requests bypass the vision group.
    pub fn ingress(&self) -> &MediaIngress {
        &self.ingress
    }
    fn unit_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.modules.target.spec.units.len()),
            1 => Ok(self.vision.layer_count()),
            _ => Err(Error::backend("conditional execution group out of bounds")),
        }
    }
    fn check_unit(&self, group: usize, index: usize) -> Result<(), Error> {
        if index >= self.unit_count(group)? {
            return Err(Error::backend("conditional unit out of bounds"));
        }
        Ok(())
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ArchitectureParameters<B>
    for ConditionalModel<B>
{
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        self.modules.target.state_layout()
    }
    fn state_identity(
        &self,
        state: &PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error> {
        self.modules
            .target
            .state_identity(state, topology.clone())?;
        ModelStateIdentity::new(
            "qwen4_exp",
            "qwen4_exp",
            self.state_fingerprint(),
            self.modules.target.spec.units.len(),
            state.global_layer_offset(),
            0,
            topology,
        )
        .map_err(Error::backend)
    }
    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        if let Some(parameters) = &self.global_parameters {
            return Ok(parameters.clone());
        }
        let graph = graph()?;
        let layout = ExecutionUnitLayout::new(&graph, [self.unit_count(0)?, self.unit_count(1)?])
            .map_err(Error::backend)?;
        let mut groups = self
            .modules
            .target
            .parameter_description(context)?
            .groups()
            .to_vec();
        for group in crate::qwen::vision::static_parallel_parameter_groups(
            &self.modules.vision,
            &self.vision,
            "model.visual",
        )
        .map_err(Error::backend)?
        {
            groups.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("vision"),
                group,
            ));
        }
        for index in 0..self.vision.layer_count() {
            let module =
                VisionBlock::<B>::new_with_root(&self.vision, "model.visual", index, context)?;
            for group in crate::qwen::vision::block_parallel_parameter_groups(
                &module,
                &self.vision,
                "model.visual",
                index,
            )
            .map_err(Error::backend)?
            {
                groups.push(OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(layout.group_id(1).unwrap().clone(), index),
                    group,
                ));
            }
        }
        ArchitectureParameterDescription::new(
            &graph,
            &layout,
            groups.iter().map(|g| g.group().clone()),
            groups.clone(),
        )
        .map_err(Error::backend)
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        let mut recipes = self.modules.target.static_parameter_recipes(source)?;
        recipes.extend(self.ingress.vision().static_parameters().recipes().clone());
        Ok(recipes)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.modules.target.visit_static_parameters(visitor)?;
        visitor.visit("vision", &self.modules.vision)
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.modules.target.visit_static_parameters_mut(visitor)?;
        visitor.visit_mut("vision", &mut self.modules.vision)
    }
}
impl<B, S> LayeredArchitecture<B, S> for ConditionalModel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    type StaticModules = ConditionalStatic<B>;
    type Unit = ConditionalUnit<B>;
    type Input<'a>
        = ConditionalInput<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type ForwardContext = ConditionalForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Error> {
        self.unit_count(group)?;
        Ok((group == TARGET_INDEX)
            .then(|| eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.into()))
    }
    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_unit_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
            observer,
        )
    }
    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.modules.target.finish_forward_observed(
            hidden,
            state,
            forward
                .target
                .as_ref()
                .ok_or_else(|| Error::backend("missing target context"))?,
            context,
            observer,
        )
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.unit_count(group)
    }
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        if group == VISION_INDEX {
            vision_transport()
        } else {
            crate::transport::decoder()
        }
    }
    fn primary_execution_group(&self) -> &str {
        crate::decoder::TARGET_EXECUTION_GROUP
    }
    fn state_partition_plan(&self, layout: &StateLayout) -> ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(TARGET_INDEX, layout)
    }
    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        if group == TARGET_INDEX {
            index
        } else {
            0
        }
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        self.check_unit(group, index)?;
        Ok(if group == VISION_INDEX {
            format!("model.visual.blocks.{index}")
        } else {
            self.modules.target.spec.units[index].path()
        })
    }
    fn static_modules(&self) -> &Self::StaticModules {
        &self.modules
    }
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.modules
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.check_unit(group, index)?;
        if group == VISION_INDEX {
            let (heads, intermediate) = self
                .vision_local
                .as_ref()
                .map(|local| local[index])
                .unwrap_or((self.vision.num_heads, self.vision.intermediate_size));
            Ok(ConditionalUnit::Vision(
                VisionBlock::new_parallel_with_root(
                    &self.vision,
                    "model.visual",
                    index,
                    heads,
                    intermediate,
                    context,
                )?,
            ))
        } else {
            Ok(ConditionalUnit::Target(
                self.modules.target.construct_unit(index, context)?,
            ))
        }
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let empty = |target| ConditionalForward {
            media: None,
            range: None,
            vision: None,
            initial: None,
            projected: None,
            target: Some(target),
            incoming_target: None,
        };
        match input {
            ConditionalInput::Target(input) => {
                if let Some(partition) = self
                    .partition_state
                    .as_ref()
                    .filter(|partition| partition.global_layer_offset() != 0)
                {
                    if state.layout() != partition.layout()
                        || input.batch <= 0
                        || input.batch > self.modules.target.spec.limits.qsa.batch
                        || input.tokens <= 0
                        || input.tokens > self.modules.target.spec.limits.qsa.tokens
                    {
                        return Err(Error::backend(
                            "conditional continuation differs from selected state or invocation",
                        ));
                    }
                    let offset = state.layer(0).map_err(Error::backend)?.position();
                    for index in 1..partition.layout().len() {
                        if state.layer(index).map_err(Error::backend)?.position() != offset {
                            return Err(Error::backend(
                                "conditional continuation has inconsistent local prefixes",
                            ));
                        }
                    }
                    self.modules
                        .target
                        .spec
                        .limits
                        .validate_history(offset, input.tokens)
                        .map_err(Error::backend_source)?;
                    let geometry = self.modules.target.spec.boundary.geometry;
                    return Ok(LayeredForwardState {
                        hidden: B::Tensor::full_f32(
                            0.0,
                            &[
                                input.batch,
                                input.tokens,
                                geometry.streams(),
                                geometry.hidden_size(),
                            ],
                            context,
                        )?,
                        context: ConditionalForward {
                            media: None,
                            range: None,
                            vision: None,
                            initial: None,
                            projected: None,
                            target: None,
                            incoming_target: None,
                        },
                    });
                }
                let next = self.modules.target.begin_forward(input, state, context)?;
                Ok(LayeredForwardState {
                    hidden: next.hidden,
                    context: empty(next.context),
                })
            }
            ConditionalInput::Media(input) => {
                self.begin_media(input, 0..input.token_ids().dim(1), None, state, context)
            }
            ConditionalInput::MediaChunk {
                prepared,
                range,
                projected,
            } => self.begin_media(prepared, range, projected, state, context),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == TARGET_INDEX || (group == VISION_INDEX && forward.vision.is_some())
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == VISION_INDEX {
            return Ok(forward.initial.as_ref().unwrap_or(initial).clone());
        }
        if let Some(media) = &forward.media {
            let projected = dependencies.first().copied().or(forward.projected.as_ref());
            let input = media
                .assemble(
                    forward
                        .range
                        .clone()
                        .ok_or_else(|| Error::backend("missing media span"))?,
                    projected,
                    |ids| {
                        self.modules
                            .target
                            .decoder
                            .static_modules_mut()
                            .embeddings
                            .forward(ids, context)
                    },
                    context,
                )
                .map_err(Error::backend_source)?;
            let next = input.with_target_input(|input| {
                self.modules.target.begin_forward(input, state, context)
            })?;
            forward.target = Some(next.context);
            return Ok(next.hidden);
        }
        Ok(initial.clone())
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
        )
    }
    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == VISION_INDEX {
            let Some(vision) = forward.vision.as_mut() else {
                return Ok(hidden.clone());
            };
            let output = self.modules.vision.finish(hidden, vision, context)?;
            forward.projected = Some(output.embeddings.clone());
            Ok(output.embeddings)
        } else {
            self.modules.target.complete_execution_group(
                0,
                hidden,
                state,
                forward
                    .target
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing target context"))?,
                context,
            )
        }
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.modules.target.finish_forward(
            hidden,
            state,
            forward
                .target
                .as_ref()
                .ok_or_else(|| Error::backend("missing target context"))?,
            context,
        )
    }
    fn projects_final_text_position() -> bool {
        true
    }
    fn finish_text_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.modules.target.finish_text_forward(
            hidden,
            state,
            forward
                .target
                .as_ref()
                .ok_or_else(|| Error::backend("missing target context"))?,
            context,
        )
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward
            .target
            .as_ref()
            .and_then(<TargetModel<B> as LayeredArchitecture<B, S>>::prediction_target_capture)
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = vec![];
        values.extend(forward.encoder_storage_values().map(|(_, value)| value));
        if let Some(target) = &forward.target {
            values.extend(
                <TargetModel<B> as LayeredArchitecture<B, S>>::retained_context_values(
                    &self.modules.target,
                    target,
                    0,
                    0,
                ),
            );
        }
        values.into_iter()
    }
}
impl<B, S> RoutedLayeredArchitecture<B, S> for ConditionalModel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
    }
    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.check_unit(group, index)?;
        match (group, unit) {
            (VISION_INDEX, ConditionalUnit::Vision(block)) => self.modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing vision context"))?,
                context,
            ),
            (TARGET_INDEX, ConditionalUnit::Target(unit)) => {
                self.modules.target.forward_unit_observed_with_provider(
                    0,
                    index,
                    unit,
                    hidden,
                    state,
                    forward
                        .target
                        .as_mut()
                        .ok_or_else(|| Error::backend("missing target context"))?,
                    pass,
                    provider,
                    context,
                    observer,
                )
            }
            _ => Err(Error::backend("conditional unit/group mismatch")),
        }
    }
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.check_unit(group, index)?;
        match (group, unit) {
            (VISION_INDEX, ConditionalUnit::Vision(block)) => {
                let vision = forward
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing vision context"))?;
                self.modules
                    .vision
                    .forward_block(block, index, hidden, vision, context)
            }
            (TARGET_INDEX, ConditionalUnit::Target(unit)) => {
                self.modules.target.forward_unit_with_provider(
                    0,
                    index,
                    unit,
                    hidden,
                    state,
                    forward
                        .target
                        .as_mut()
                        .ok_or_else(|| Error::backend("missing target context"))?,
                    pass,
                    provider,
                    context,
                )
            }
            _ => Err(Error::backend("conditional unit/group mismatch")),
        }
    }
}
impl<B, S> ReplicatedTextArchitecture<B, S> for ConditionalModel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        ConditionalInput::Target(
            <TargetModel<B> as ReplicatedTextArchitecture<B, S>>::text_input(tokens, mask),
        )
    }
    fn supports_chunked_prefill() -> bool {
        true
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend>
    crate::prediction_extension::Qwen4PredictionTarget<B> for ConditionalModel<B>
{
    fn prediction_embedding(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.modules.target.prediction_embedding(tokens, context)
    }
    fn prediction_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.modules
            .target
            .prediction_logits(hidden, context, instrumentation)
    }
}
