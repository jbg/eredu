//! Shared graph traversal of retained vision blocks and the ordinary Flash-Next target.
use super::{
    media::{MediaIngress, PreparedMediaInput},
    target::{BoundTargetSpec, TargetForward, TargetInput, TargetKernel, Unit as TargetUnit},
};
use crate::qwen::vision::{VisionBlock, VisionConfig, VisionState, VisionStatic};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, EmbeddingOperator, Error, GroupedNeuralBackend,
    Tensor,
};
use eredu_runtime::*;

/// Stable optional encoder group, preceding the ordinary target group.
pub const VISION_GROUP: &str = "vision";
mod composite;
mod partitioned;
pub mod prefill;
const TARGET_INDEX: usize = 0;
const VISION_INDEX: usize = 1;

fn graph() -> Result<ExecutionGraph, Error> {
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
/// Composes retained target storage and vision geometry without native resources.
pub fn parameter_description(
    target: &ComponentDefinition,
    vision: &VisionConfig,
) -> Result<ArchitectureParameterDescription, Error> {
    Ok(definition(target, vision)?.parameters().clone())
}

/// Validates target and optional vision invocations before native construction.
pub fn definition(
    target: &ComponentDefinition,
    vision: &VisionConfig,
) -> Result<ComponentDefinition, Error> {
    definition_with_parameters(target, vision, declare_parameters(target, vision)?)
}

fn definition_with_parameters(
    target: &ComponentDefinition,
    vision: &VisionConfig,
    parameters: ArchitectureParameterDescription,
) -> Result<ComponentDefinition, Error> {
    let mut units = (0..target.unit_count(TARGET_INDEX)?)
        .map(|index| {
            target
                .unit(TARGET_INDEX, index)
                .cloned()
                .map_err(Error::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    units.extend((0..vision.layer_count()).map(|index| ComponentUnit {
        path: format!("model.visual.blocks.{index}"),
        state: 0..0,
    }));
    ComponentDefinition::new(
        "qwen4_exp_conditional",
        parameters,
        units,
        vec![
            target.group_transport(TARGET_INDEX)?.clone(),
            vision_transport(),
        ],
        crate::decoder::TARGET_EXECUTION_GROUP,
        vec![],
        target.state_layout().clone(),
        target.state_partition_plan().clone(),
    )
    .map_err(Error::from)
}

fn declare_parameters(
    target: &ComponentDefinition,
    vision: &VisionConfig,
) -> Result<ArchitectureParameterDescription, Error> {
    let graph = graph()?;
    let layout = ExecutionUnitLayout::new(
        &graph,
        [target.unit_count(TARGET_INDEX)?, vision.layer_count()],
    )
    .map_err(Error::backend)?;
    let mut groups = target.parameters().groups().to_vec();
    for group in crate::qwen::vision::static_parallel_parameter_groups(vision, "model.visual")
        .map_err(Error::backend)?
    {
        groups.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("vision"),
            group,
        ));
    }
    for index in 0..vision.layer_count() {
        for group in
            crate::qwen::vision::block_parallel_parameter_groups(vision, "model.visual", index)
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

fn vision_transport() -> ArchitectureGroupTransport {
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

pub(crate) fn validate_ingress(
    spec: &super::target::TargetSpec,
    ingress: &MediaIngress,
    vision: &VisionConfig,
) -> Result<(), Error> {
    // Formats may change at selection; semantic geometry must match the retained role.
    let mut expected = ingress.vision().config().clone();
    let mut actual = vision.clone();
    expected.linear_formats.clear();
    actual.linear_formats.clear();
    if expected != actual || vision.out_hidden_size != spec.configuration().hidden_size {
        return Err(Error::backend(
            "conditional vision geometry differs from retained ingress",
        ));
    }
    super::media::MediaIngress::new(spec, ingress.vision().clone())
        .map_err(Error::backend_source)?;
    Ok(())
}

/// Pinned modules. The target owns the single vocabulary and residual readout.
#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ConditionalStatic<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    target: TargetKernel<B>,
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
    /// Admitted prepared input using the same target/media equations.
    Prepared(
        crate::composite_execution::PreparedCompositeInput<
            'a,
            T,
            super::media::MediaInputPartPlan<T>,
        >,
    ),
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
/// Conditional execution through the shared validated component adapter.
pub type ConditionalModel<B> = ComponentAdapter<
    std::sync::Arc<ComponentDefinition>,
    ConditionalKernel<B>,
    KernelParameterModules,
>;

/// Family media assembly and target/vision equations under the shared lifecycle.
pub struct ConditionalKernel<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    modules: ConditionalStatic<B>,
    ingress: MediaIngress,
    vision: VisionConfig,
    vision_local: Option<Vec<(i32, i32)>>,
    vision_input_owner: bool,
    partition_state: Option<PartitionState>,
    definition: std::sync::Arc<ComponentDefinition>,
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ConditionalKernel<B> {
    /// Assembles the complete shared runtime adapter from finalized geometry.
    pub fn into_model(self) -> Result<ConditionalModel<B>, Error> {
        ComponentAdapter::with_pinned_resources::<B>(self.definition.clone(), self)
            .map_err(Error::from)
    }
    /// Retains selected ownership and exact native pinned resources on that same adapter.
    pub fn into_partition_model<G, A>(
        self,
        partition: &ArchitecturePartition<G, A>,
    ) -> Result<ConditionalModel<B>, Error> {
        let kernel = self.select_partition_resources(partition)?;
        ComponentAdapter::with_partition_resources::<B, G, A>(
            kernel.definition.clone(),
            kernel,
            partition,
        )
        .map_err(Error::from)
    }
    /// Constructs pinned modules only; encoder and target units are built on demand.
    pub fn new(
        spec: BoundTargetSpec,
        ingress: MediaIngress,
        vision: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        validate_ingress(spec.geometry(), &ingress, &vision)?;
        let definition = std::sync::Arc::new(definition(&spec.geometry().definition()?, &vision)?);
        Ok(Self {
            modules: ConditionalStatic {
                target: TargetKernel::new(spec, context)?,
                vision: VisionStatic::new_with_root(vision.clone(), "model.visual", context)?,
            },
            ingress,
            vision,
            vision_local: None,
            vision_input_owner: true,
            partition_state: None,
            definition,
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
                        .statics
                        .embedding_mut()?
                        .forward(ids, context)
                },
                context,
            )
            .map_err(Error::backend_source)?;
        let next =
            media.with_target_input(|input| self.modules.target.begin(input, state, context))?;
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
    pub(crate) fn target_boundary_schema(
        &self,
    ) -> Result<super::input::RequestBoundarySchema, Error> {
        self.modules.target.spec.boundary_schema()
    }
    /// Retained target/vision invocation, parameter, state and transport authority.
    pub fn definition(&self) -> &ComponentDefinition {
        &self.definition
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ComponentPinnedResources<B>
    for ConditionalKernel<B>
{
    type Modules = ConditionalStatic<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.modules
    }
    fn visit_pinned<V: StaticParameterVisitor<B>>(&self, visitor: &mut V) -> Result<(), V::Error> {
        self.modules.target.visit_pinned(visitor)?;
        if self.modules.vision.has_parameters() {
            visitor.visit("vision", &self.modules.vision)?;
        }
        Ok(())
    }
    fn visit_pinned_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.modules.target.visit_pinned_mut(visitor)?;
        if self.modules.vision.has_parameters() {
            visitor.visit_mut("vision", &mut self.modules.vision)?;
        }
        Ok(())
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend>
    crate::prediction_extension::Qwen4PredictionTarget<B> for ConditionalKernel<B>
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

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ComponentKernel<B>
    for ConditionalKernel<B>
{
    type Input<'a>
        = ConditionalInput<'a, B::Tensor>
    where
        Self: 'a;
    type Unit = ConditionalUnit<B>;
    type ForwardContext = ConditionalForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
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
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        let group = address.group();
        let index = address.index();
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
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == TARGET_INDEX || (group == VISION_INDEX && forward.vision.is_some())
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward
            .target
            .as_ref()
            .and_then(<TargetKernel<B> as ComponentKernel<B>>::prediction_target_capture)
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = vec![];
        values.extend(forward.encoder_storage_values().map(|(_, value)| value));
        if let Some(target) = &forward.target {
            values.extend(<TargetKernel<B> as ComponentKernel<B>>::retained_values(
                &self.modules.target,
                target,
                self.modules
                    .target
                    .definition()
                    .unit_address(0, 0)
                    .expect("target invocation"),
            ));
        }
        values.into_iter()
    }
}
impl<B, S> ComponentEquations<B, S> for ConditionalKernel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Error> {
        Ok((group == TARGET_INDEX)
            .then(|| eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.into()))
    }
    fn execute_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
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
        self.execute_routed_observed(
            address,
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
    fn finish_observed<O>(
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
        self.modules.target.finish_observed(
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
    fn begin<'a>(
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
            ConditionalInput::Prepared(input) => self.begin_prepared(input, state, context),
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
                let next = self.modules.target.begin(input, state, context)?;
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
    fn begin_group(
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
                            .statics
                            .embedding_mut()?
                            .forward(ids, context)
                    },
                    context,
                )
                .map_err(Error::backend_source)?;
            let next = input
                .with_target_input(|input| self.modules.target.begin(input, state, context))?;
            forward.target = Some(next.context);
            return Ok(next.hidden);
        }
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.execute_routed(
            address,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
        )
    }
    fn complete_group(
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
            self.modules.target.complete_group(
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
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.modules.target.finish(
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
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.modules.target.finish_text(
            hidden,
            state,
            forward
                .target
                .as_ref()
                .ok_or_else(|| Error::backend("missing target context"))?,
            context,
        )
    }
}
impl<B, S> RoutedComponentKernel<B, S> for ConditionalKernel<B>
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
    fn execute_routed_observed<P, O>(
        &mut self,
        address: ExecutionUnitAddress,
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
        let group = address.group();
        let index = address.index();
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
                self.modules.target.execute_routed_observed(
                    self.modules.target.definition().unit_address(0, index)?,
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
    fn execute_routed<P>(
        &mut self,
        address: ExecutionUnitAddress,
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
        let group = address.group();
        let index = address.index();
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
            (TARGET_INDEX, ConditionalUnit::Target(unit)) => self.modules.target.execute_routed(
                self.modules.target.definition().unit_address(0, index)?,
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
            ),
            _ => Err(Error::backend("conditional unit/group mismatch")),
        }
    }
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TextComponentKernel<B>
    for ConditionalKernel<B>
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        ConditionalInput::Target(<TargetKernel<B> as TextComponentKernel<B>>::text_input(
            tokens, mask,
        ))
    }
    fn supports_chunked_prefill() -> bool {
        true
    }
}
