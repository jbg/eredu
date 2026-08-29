//! Thin DeepSeek-V4 architecture policy.

use std::{num::NonZeroU32, sync::Arc};

use eredu_core::{
    cache::{
        LayerCachePolicy, MutableStateResidency, PoolingStateComponent, StateResidencyClass,
        StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
    },
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::{
    EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, GatedProductExpertBankOperator,
    HyperHead, HyperHeadSpec, HyperNeuralBackend, Index, LinearOperator, LinearSpec,
    NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Parameterized,
    PoolingAttentionCache, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, ModelStateIdentity, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, PartitionedLayeredArchitecture, RoutedExpertProvider,
    RoutedLayeredArchitecture, StateLayout, StateSegmentLifetime, StateSegmentSpec,
};

use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};

use crate::decoder::{
    SequentialPredictionGroups, StaticModuleSpec, StaticModules as TextStaticModules,
};

use super::{
    block::V4Block,
    moe::MoePolicy,
    mtp::{EmbeddedInput, ForwardMode, RetainedValues, V4PredictionLayer},
    DsparkConfig, ExpertFormat, V4Args, V4AttentionPolicy,
};

/// Declares V4 cache identity independently of concrete state storage.
pub fn state_identity(
    args: &V4Args,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    args.validate().map_err(Error::backend)?;
    topology.validate().map_err(Error::backend)?;
    let layer_count = usize::try_from(
        args.num_hidden_layers
            .checked_add(args.num_nextn_predict_layers)
            .ok_or_else(|| Error::backend("V4 state layer count overflowed"))?,
    )
    .map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("V4 owned state range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "V4 owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(ModelStateIdentity {
        model_family: "deepseek_v4".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: super::v4_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        topology,
    })
}

/// Pinned DSpark projections and heads shared by its ordinary draft blocks.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DsparkStatic<B: HyperNeuralBackend> {
    main_projection: B::Linear,
    main_norm: B::Normalization,
    output_norm: B::Normalization,
    hyper_head: HyperHead<B>,
    markov_embedding: B::Embedding,
    markov_output: B::Linear,
    confidence_head: B::Linear,
}

impl<B, S> RoutedLayeredArchitecture<B, S> for Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: PoolingAttentionCache<B::Tensor>,
{
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        Model::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, pass, provider, context,
        )
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: PoolingAttentionCache<B::Tensor>,
{
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        Model::forward_unit_parallel_with_provider(
            self, group, index, unit, hidden, state, forward, pass, parallel, provider, context,
        )
    }
}

/// One target, sequential MTP, or fused DSpark execution unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    /// Ordinary target decoder block.
    Target(V4Block<B>),
    /// One sequential embedded prediction layer.
    Prediction(V4PredictionLayer<B>),
    /// One ordinary local-attention block in the fused DSpark chain.
    Dspark(V4Block<B>),
}

impl<B> Unit<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    /// Returns the exact routed bank specification retained by this realized unit.
    pub fn expert_bank_spec(&self) -> &eredu_nn::GatedProductExpertBankSpec {
        match self {
            Self::Target(block) | Self::Dspark(block) => block.feed_forward.experts.spec(),
            Self::Prediction(prediction) => prediction.decoder.feed_forward.experts.spec(),
        }
    }
}

/// V4 pinned modules shared by resident and bounded layer execution.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: HyperNeuralBackend> {
    /// Shared embedding, final normalization, and vocabulary head lifecycle.
    pub text: TextStaticModules<B>,
    /// Learned collapse from hyper-connection streams to final hidden state.
    pub hyper_head: HyperHead<B>,
    /// Optional fused-drafter projections and heads.
    pub dspark: Option<DsparkStatic<B>>,
}

/// V4 values retained for one target-model forward.
pub struct ForwardContext<T> {
    input_ids: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    target_capture: Option<T>,
    draft_logits: Option<T>,
    draft_hidden: Option<T>,
    captures: Vec<Option<T>>,
}

/// Family-owned schema for immutable V4 target context crossing pipeline ranks.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetBoundarySchema {
    hidden_size: i32,
    activation_hidden_size: i32,
    capture_count: usize,
}

impl TargetBoundarySchema {
    /// Derives the schema for the configured DSpark target captures.
    pub fn from_args(args: &V4Args) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let activation_hidden_size = args
            .hidden_size
            .checked_mul(args.hc_mult)
            .ok_or_else(|| Error::backend("V4 transport activation width overflowed"))?;
        Ok(Self {
            hidden_size: args.hidden_size,
            activation_hidden_size,
            capture_count: args
                .dspark
                .as_ref()
                .map_or(0, |config| config.target_layer_ids.len()),
        })
    }

    /// Returns the flattened hyper-stream width transported between partitions.
    pub const fn activation_hidden_size(self) -> i32 {
        self.activation_hidden_size
    }

    /// Returns the number of configured capture tensors.
    pub const fn capture_count(self) -> usize {
        self.capture_count
    }
}

impl eredu_runtime::ArchitectureBoundary for TargetBoundarySchema {
    type Boundary<T> = TargetBoundary<T>;

    const IDENTITY: &'static str = "deepseek_v4.target";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.activation_hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        let mut specs = vec![eredu_runtime::BoundaryTensorSpec::new(
            "tokens",
            [Dim::Batch, Dim::Sequence],
            Dtype::Uint32,
        )];
        specs.extend((0..self.capture_count).map(|index| {
            eredu_runtime::BoundaryTensorSpec::new(
                format!("capture.{index}"),
                [Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                Dtype::Activation,
            )
        }));
        specs
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        Ok(TargetBoundary {
            input_ids: tensors.next().expect("validated target token ids"),
            captures: tensors.collect(),
        })
    }

    /// Encodes tokens followed by configured captures after validation.
    fn encode<T>(
        &self,
        boundary: TargetBoundary<T>,
    ) -> Result<Vec<T>, eredu_runtime::ArchitectureBoundaryError> {
        if boundary.captures.len() != self.capture_count {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "deepseek_v4.target.captures",
                expected: self.capture_count,
                actual: boundary.captures.len(),
            });
        }
        Ok(std::iter::once(boundary.input_ids)
            .chain(boundary.captures)
            .collect())
    }
}

/// Typed V4 target context transported alongside the evolving activation.
pub struct TargetBoundary<T> {
    /// Original target token identities.
    pub input_ids: T,
    /// Captures selected for DSpark target conditioning.
    pub captures: Vec<T>,
}

impl<T> TargetBoundary<T> {
    /// Creates a target boundary from tokens and configured captures.
    pub const fn new(input_ids: T, captures: Vec<T>) -> Self {
        Self {
            input_ids,
            captures,
        }
    }
}

/// Target input owned by either the first or a downstream V4 partition.
pub enum TargetPartitionInput<'a, T> {
    /// Token identities embedded by the input-owning partition.
    Tokens(&'a T),
    /// Evolving activation plus immutable target context from an upstream rank.
    Hidden {
        /// Flattened transported hyper-stream activation.
        hidden: T,
        /// Original token identities and configured captures.
        boundary: TargetBoundary<T>,
    },
}

/// Architecture-prepared target activation and cross-partition context.
pub struct TargetPartitionForward<T> {
    /// Hyper-stream activation consumed by target blocks.
    pub hidden: T,
    /// Typed immutable/mutable target boundary values.
    pub boundary: TargetBoundary<T>,
}

/// Architecture-owned completion of one routed V4 target partition.
pub enum TargetPartitionOutput<T> {
    /// Final-owner logits and the hidden/capture value consumed by drafting.
    Final {
        /// Complete target logits.
        logits: T,
        /// Target hidden state or concatenated DSpark captures.
        draft_hidden: T,
    },
    /// Transport-visible activation and typed context for the next owner.
    Boundary {
        /// Flattened hyper-stream activation.
        hidden: T,
        /// Original token ids and configured target captures.
        boundary: TargetBoundary<T>,
    },
}

impl<T> ForwardContext<T> {
    /// Borrows the final target hidden state or configured DSpark captures.
    pub const fn target_capture(&self) -> Option<&T> {
        self.target_capture.as_ref()
    }

    /// Borrows logits emitted by a sequential or fused draft pass.
    pub const fn draft_logits(&self) -> Option<&T> {
        self.draft_logits.as_ref()
    }

    /// Borrows the hidden state emitted by sequential prediction execution.
    pub const fn draft_hidden(&self) -> Option<&T> {
        self.draft_hidden.as_ref()
    }
}

/// Thin V4 target-model architecture over the shared layered runtime.
pub struct Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    args: V4Args,
    static_modules: StaticModules<B>,
    groups: SequentialPredictionGroups,
    parallel_geometry: Option<Arc<super::parallel::V4LocalGeometry>>,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>>,
}

impl<B> eredu_runtime::ArchitectureParameters<B> for Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl()
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Self::DefinitionError> {
        state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }

    fn parameter_description(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Self::DefinitionError> {
        super::parallel::v4_parameter_description(&self.args).map_err(Error::backend)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        visitor.visit("embedding", &self.static_modules.text.embeddings)?;
        visitor.visit("norm", &self.static_modules.text.norm)?;
        if let Some(head) = &self.static_modules.text.lm_head {
            visitor.visit("output", head)?;
        }
        visitor.visit("hyper_head", &self.static_modules.hyper_head)?;
        if let Some(dspark) = &self.static_modules.dspark {
            visitor.visit("mtp", dspark)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules.text.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.text.norm)?;
        if let Some(head) = &mut self.static_modules.text.lm_head {
            visitor.visit_mut("output", head)?;
        }
        visitor.visit_mut("hyper_head", &mut self.static_modules.hyper_head)?;
        if let Some(dspark) = &mut self.static_modules.dspark {
            visitor.visit_mut("mtp", dspark)?;
        }
        Ok(())
    }
}

impl<B> Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    /// Enters a target partition using the canonical layered execution context.
    pub fn begin_routed_target_partition(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let prepared = self.begin_partition_target_inner(input, parallel, context)?;
        let TargetBoundary {
            input_ids,
            captures,
        } = prepared.boundary;
        Ok(LayeredForwardState {
            hidden: prepared.hidden,
            context: ForwardContext {
                embedded: input_ids.clone(),
                input_ids,
                mask: mask.cloned(),
                mode: ForwardMode::Target,
                target_capture: None,
                draft_logits: None,
                draft_hidden: None,
                captures: captures.into_iter().map(Some).collect(),
            },
        })
    }

    /// Reconstitutes the typed cross-partition target boundary after layered execution.
    pub fn routed_target_boundary(
        &self,
        forward: &ForwardContext<B::Tensor>,
    ) -> Result<TargetBoundary<B::Tensor>, Error> {
        let captures = forward
            .captures
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, capture)| {
                capture.ok_or_else(|| {
                    Error::backend(format!("missing V4 target capture slot {index}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TargetBoundary::new(forward.input_ids.clone(), captures))
    }

    /// Completes a routed target partition without exposing hyper-stream
    /// flattening or DSpark capture selection to distributed composition.
    pub fn finish_routed_target_partition(
        &mut self,
        hidden: &B::Tensor,
        forward: &ForwardContext<B::Tensor>,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TargetPartitionOutput<B::Tensor>, Error> {
        let boundary = self.routed_target_boundary(forward)?;
        if owns_output {
            let draft_hidden = if self.args.dspark.is_some() {
                B::Tensor::concatenate(&boundary.captures, -1, context)?
            } else {
                hidden.clone()
            };
            let logits = match parallel {
                Some(parallel) => {
                    self.finish_partition_target_parallel(hidden, parallel, context)?
                }
                None => self.finish_partition_target(hidden, context)?,
            };
            Ok(TargetPartitionOutput::Final {
                logits,
                draft_hidden,
            })
        } else {
            let hidden = hidden.reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    self.args.hc_mult * self.args.hidden_size,
                ],
                context,
            )?;
            Ok(TargetPartitionOutput::Boundary { hidden, boundary })
        }
    }

    fn begin_partition_target_inner(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TargetPartitionForward<B::Tensor>, Error> {
        match input {
            TargetPartitionInput::Tokens(tokens) => {
                let hidden = match parallel {
                    Some(parallel) => self.pipeline_embed_parallel(tokens, parallel, context)?,
                    None => self.pipeline_embed(tokens, context)?,
                };
                let captures = (0..self
                    .args
                    .dspark
                    .as_ref()
                    .map_or(0, |config| config.target_layer_ids.len()))
                    .map(|_| {
                        B::Tensor::full_f32(
                            0.0,
                            &[tokens.dim(0), tokens.dim(1), self.args.hidden_size],
                            context,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TargetPartitionForward {
                    hidden,
                    boundary: TargetBoundary::new(tokens.clone(), captures),
                })
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let hidden = hidden.reshape(
                    &[
                        hidden.dim(0),
                        hidden.dim(1),
                        self.args.hc_mult,
                        self.args.hidden_size,
                    ],
                    context,
                )?;
                Ok(TargetPartitionForward { hidden, boundary })
            }
        }
    }

    /// Finishes the serial target partition through the architecture output boundary.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish(hidden, context)
    }

    /// Finishes the tensor-parallel target partition through the architecture output boundary.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
    }

    /// Builds unloaded pinned V4 modules.
    pub fn new(args: V4Args, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "DeepSeek-V4",
            crate::operator_requirements::DEEPSEEK_V4,
        )?;
        args.validate().map_err(Error::backend)?;
        let text = TextStaticModules::from_spec(text_static_spec(&args), context)?;
        Ok(Self {
            groups: prediction_groups(&args)?,
            static_modules: static_modules(&args, text, context)?,
            args,
            parallel_geometry: None,
            expert_realization: None,
        })
    }

    /// Builds unloaded V4 modules using one authoritative rank-local plan.
    pub fn new_parallel(
        args: V4Args,
        geometry: super::parallel::V4LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "DeepSeek-V4",
            crate::operator_requirements::DEEPSEEK_V4
                .union(eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL),
        )?;
        args.validate().map_err(Error::backend)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let text = TextStaticModules::from_parallel_spec(
            text_static_spec(&args),
            geometry.embedding_range().clone(),
            Some(geometry.output_range().clone()),
            context,
        )?;
        Ok(Self {
            groups: prediction_groups(&args)?,
            static_modules: static_modules(&args, text, context)?,
            args,
            parallel_geometry: Some(Arc::new(geometry)),
            expert_realization: None,
        })
    }

    /// Returns the state layout matching the modules this instance builds.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        match &self.parallel_geometry {
            Some(geometry) => Ok(geometry.state_layout().clone()),
            None => state_layout(&self.args),
        }
    }

    /// Borrows rank-local geometry for resident and streamed unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<Arc<super::parallel::V4LocalGeometry>> {
        self.parallel_geometry.clone()
    }

    /// Installs the architecture-derived expert realization used by every unit factory.
    pub fn install_expert_realization(
        &mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>,
    ) {
        self.expert_realization = Some(realization);
    }

    /// Returns the normalized V4 arguments.
    pub const fn args(&self) -> &V4Args {
        &self.args
    }

    /// Returns the prediction depth count declared by the execution graph.
    pub fn mtp_len(&self) -> usize {
        self.groups.prediction_count()
    }

    /// Borrows pinned modules for checkpoint binding.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned modules for checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Constructs one target, sequential MTP, or DSpark unit from this model's
    /// authoritative global or rank-local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.groups.unit_count(group)?;
        let args = self
            .parallel_geometry
            .as_ref()
            .map_or(&self.args, |geometry| geometry.args());
        let mut unit = if group == 0 {
            Unit::Target(V4Block::new(args, index, context)?)
        } else if self.args.dspark.is_some() {
            let global =
                usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)? + group - 1;
            Unit::Dspark(V4Block::new_at(
                args,
                global,
                &format!("mtp.{}", group - 1),
                context,
            )?)
        } else {
            Unit::Prediction(V4PredictionLayer::new(args, group - 1, context)?)
        };
        if let Some(realization) = &self.expert_realization {
            let owner_group = if group == 0 {
                "target".to_owned()
            } else {
                format!("mtp.{}", group - 1)
            };
            let spec = realization.unit_spec(&owner_group, index).ok_or_else(|| {
                Error::backend(format!(
                    "V4 expert realization has no bank for {owner_group}.{index}"
                ))
            })?;
            let feed_forward = match &mut unit {
                Unit::Target(block) | Unit::Dspark(block) => &mut block.feed_forward,
                Unit::Prediction(prediction) => &mut prediction.decoder.feed_forward,
            };
            feed_forward.experts = B::gated_product_expert_bank(spec.clone(), context)?;
        }
        Ok(unit)
    }

    /// Embeds tokens and broadcasts them across hyper-connection streams for
    /// a pipeline-partitioned target pass.
    pub fn pipeline_embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(tokens, context)?;
        broadcast_streams::<B>(&embedded, &self.args, context)
    }

    /// Applies vocabulary-parallel lookup and architecture-owned stream broadcast.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.static_modules.text.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        broadcast_streams::<B>(&embedded, &self.args, context)
    }

    /// Collapses hyper streams, applies the final norm, and projects logits on
    /// the final pipeline stage.
    pub fn pipeline_finish(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.hyper_head.forward(hidden, context)?;
        let hidden = self.static_modules.text.norm.forward(&hidden, context)?;
        self.static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head")
            .forward(&hidden, context)
    }

    /// Collapses hyper streams and applies the final target normalization
    /// without projecting vocabulary logits.
    pub fn pipeline_finish_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.hyper_head.forward(hidden, context)?;
        self.static_modules.text.norm.forward(&hidden, context)
    }

    /// Applies the complete tensor-parallel target output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.pipeline_finish_hidden(hidden, context)?;
        B::vocabulary_parallel_project(
            self.static_modules
                .text
                .lm_head
                .as_mut()
                .expect("validated V4 models have an untied output head"),
            &hidden,
            parallel,
            context,
        )
    }

    /// Executes one sequential embedded-prediction unit owned by the final
    /// pipeline stage.
    pub fn pipeline_forward_prediction<C>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
    {
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(tokens, context)?;
        let output_head = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head");
        match unit {
            Unit::Prediction(unit) => {
                unit.forward(hidden, &embedded, tokens, cache, output_head, context)
            }
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes one sequential embedded-prediction unit with runtime-supplied
    /// routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_with_provider<C, P>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(tokens, context)?;
        let output_head = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head");
        match unit {
            Unit::Prediction(unit) => unit.forward_with_provider(
                hidden,
                &embedded,
                tokens,
                cache,
                output_head,
                pass,
                provider,
                context,
            ),
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes a rank-local sequential predictor using the architecture-owned
    /// vocabulary shards and the backend-neutral collective hooks.
    pub fn pipeline_forward_prediction_neutral_parallel<C>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
    {
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.static_modules.text.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let head = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head");
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel(
                hidden,
                &embedded,
                tokens,
                cache,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
                |value, context| B::vocabulary_parallel_project(head, value, parallel, context),
            ),
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes a rank-local sequential predictor with supplied experts and
    /// architecture-owned vocabulary shards.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_neutral_parallel_with_provider<C, P>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.static_modules.text.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let head = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head");
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel_with_provider(
                hidden,
                &embedded,
                tokens,
                cache,
                pass,
                provider,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
                |value, context| B::vocabulary_parallel_project(head, value, parallel, context),
            ),
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Converts the transport-visible target capture into the internal
    /// head-expanded activation consumed by one V4 predictor.
    pub fn begin_partition_prediction_hidden(
        &self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if hidden.shape().len() == 3 {
            hidden.reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    self.args.hc_mult,
                    self.args.hidden_size,
                ],
                context,
            )
        } else {
            Ok(hidden.clone())
        }
    }

    /// Converts one completed V4 predictor result back to the canonical
    /// transport-visible hidden width.
    pub fn finish_partition_prediction_output(
        &self,
        output: super::mtp::PredictionOutput<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error> {
        let hidden = output.hidden.reshape(
            &[
                output.hidden.dim(0),
                output.hidden.dim(1),
                self.args.hc_mult * self.args.hidden_size,
            ],
            context,
        )?;
        Ok(super::mtp::PredictionOutput {
            logits: output.logits,
            hidden,
            tokens: output.tokens,
        })
    }

    /// Selects the shifted prefix used to warm sequential prediction caches.
    pub fn prepare_partition_prediction_replay(
        &self,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<(B::Tensor, B::Tensor)>, Error> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(None);
        }
        let hidden = self.begin_partition_prediction_hidden(hidden, context)?;
        let hidden = hidden.index(
            &[
                Index::Full,
                Index::Range(0, sequence - 1),
                Index::Full,
                Index::Full,
            ],
            context,
        )?;
        let next = tokens.index(&[Index::Full, Index::Range(1, sequence)], context)?;
        Ok(Some((hidden, next)))
    }

    /// Rebuilds the fused DSpark context caches from concatenated target-layer
    /// captures. The caller controls transactionality by choosing the cache
    /// slice supplied here.
    pub fn pipeline_prefill_dspark_context<C, M>(
        &mut self,
        units: &mut [M],
        captures: &B::Tensor,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
    {
        if units.len() != caches.len() {
            return Err(Error::backend("DSpark unit/cache count mismatch"));
        }
        let dspark = self
            .static_modules
            .dspark
            .as_mut()
            .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
        let main = dspark
            .main_norm
            .forward(&dspark.main_projection.forward(captures, context)?, context)?;
        let hidden = broadcast_streams::<B>(&main, &self.args, context)?;
        for (unit, cache) in units.iter_mut().zip(caches) {
            let Unit::Dspark(unit) = unit.as_mut() else {
                return Err(Error::backend("DSpark context received a non-DSpark unit"));
            };
            unit.prefill_attention_cache(&hidden, cache, context)?;
        }
        Ok(())
    }

    /// Executes one transactional fused DSpark proposal block.
    pub fn pipeline_dspark_proposal<C, M>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
    {
        self.pipeline_dspark_proposal_inner(
            units,
            anchor,
            capacity,
            caches,
            context,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward(hidden, tokens, Some(mask), Some(cache), context)
            },
        )
    }

    /// Executes one transactional fused DSpark proposal block with
    /// runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_dspark_proposal_with_provider<C, M, P>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.pipeline_dspark_proposal_inner(
            units,
            anchor,
            capacity,
            caches,
            context,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward_with_provider(
                    hidden,
                    tokens,
                    Some(mask),
                    Some(cache),
                    pass,
                    provider,
                    context,
                )
            },
        )
    }

    /// Executes a tensor-partitioned DSpark proposal using the
    /// architecture-owned vocabulary shards.
    pub fn pipeline_dspark_proposal_neutral_parallel<C, M>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
    {
        self.pipeline_dspark_proposal_neutral_parallel_inner(
            units,
            anchor,
            capacity,
            caches,
            parallel,
            context,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward_parallel(
                    hidden,
                    tokens,
                    Some(mask),
                    Some(cache),
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )
            },
        )
    }

    /// Executes a tensor-partitioned DSpark proposal with supplied experts and
    /// architecture-owned vocabulary shards.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_dspark_proposal_neutral_parallel_with_provider<C, M, P>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.pipeline_dspark_proposal_neutral_parallel_inner(
            units,
            anchor,
            capacity,
            caches,
            parallel,
            context,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward_parallel_with_provider(
                    hidden,
                    tokens,
                    Some(mask),
                    Some(cache),
                    pass,
                    provider,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline_dspark_proposal_neutral_parallel_inner<C, M, F>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        mut forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        F: FnMut(
            &mut V4Block<B>,
            &B::Tensor,
            &B::Tensor,
            &B::Tensor,
            &mut C,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let config = self
            .args
            .dspark
            .as_ref()
            .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
        if capacity == 0 || anchor.shape().len() != 2 || anchor.dim(1) != 1 {
            return Err(Error::backend(
                "DSpark proposal requires a positive capacity and [batch, 1] anchor",
            ));
        }
        if units.len() != caches.len() {
            return Err(Error::backend("DSpark unit/cache count mismatch"));
        }
        let input_ids = if capacity == 1 {
            anchor.clone()
        } else {
            let noise = B::Tensor::full_i32(
                config.noise_token_id,
                &[
                    anchor.dim(0),
                    i32::try_from(capacity - 1).map_err(Error::backend)?,
                ],
                context,
            )?;
            B::Tensor::concatenate(&[anchor.clone(), noise], 1, context)?
        };
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.static_modules.text.embeddings,
            &input_ids,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let mut hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
        for (unit, cache) in units.iter_mut().zip(caches) {
            let Unit::Dspark(unit) = unit.as_mut() else {
                return Err(Error::backend("DSpark proposal received a non-DSpark unit"));
            };
            let keys = (cache.offset() + i32::try_from(capacity).map_err(Error::backend)?)
                .min(self.args.sliding_window);
            let mask = B::Tensor::full_f32(
                0.0,
                &[i32::try_from(capacity).map_err(Error::backend)?, keys],
                context,
            )?;
            hidden = forward(unit, &hidden, &input_ids, &mask, cache, context)?;
        }
        let dspark = self
            .static_modules
            .dspark
            .as_mut()
            .expect("validated DSpark static modules");
        let collapsed = dspark.hyper_head.forward(&hidden, context)?;
        let normalized = dspark.output_norm.forward(&collapsed, context)?;
        let mut logits = B::vocabulary_parallel_project(
            self.static_modules
                .text
                .lm_head
                .as_mut()
                .expect("validated V4 models have an untied output head"),
            &normalized,
            parallel,
            context,
        )?;
        let markov = dspark.markov_embedding.forward(anchor, context)?;
        let adjustment = dspark.markov_output.forward(&markov, context)?;
        logits = logits.add(&adjustment.broadcast_to(logits.shape(), context)?, context)?;
        Ok(logits)
    }

    fn pipeline_dspark_proposal_inner<C, M, F>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
        mut forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        F: FnMut(
            &mut V4Block<B>,
            &B::Tensor,
            &B::Tensor,
            &B::Tensor,
            &mut C,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let config = self
            .args
            .dspark
            .as_ref()
            .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
        if capacity == 0 || anchor.shape().len() != 2 || anchor.dim(1) != 1 {
            return Err(Error::backend(
                "DSpark proposal requires a positive capacity and [batch, 1] anchor",
            ));
        }
        if units.len() != caches.len() {
            return Err(Error::backend("DSpark unit/cache count mismatch"));
        }
        let input_ids = if capacity == 1 {
            anchor.clone()
        } else {
            let noise = B::Tensor::full_i32(
                config.noise_token_id,
                &[
                    anchor.dim(0),
                    i32::try_from(capacity - 1).map_err(Error::backend)?,
                ],
                context,
            )?;
            B::Tensor::concatenate(&[anchor.clone(), noise], 1, context)?
        };
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(&input_ids, context)?;
        let mut hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
        for (unit, cache) in units.iter_mut().zip(caches) {
            let Unit::Dspark(unit) = unit.as_mut() else {
                return Err(Error::backend("DSpark proposal received a non-DSpark unit"));
            };
            let keys = (cache.offset() + i32::try_from(capacity).map_err(Error::backend)?)
                .min(self.args.sliding_window);
            let mask = B::Tensor::full_f32(
                0.0,
                &[i32::try_from(capacity).map_err(Error::backend)?, keys],
                context,
            )?;
            hidden = forward(unit, &hidden, &input_ids, &mask, cache, context)?;
        }
        let dspark = self
            .static_modules
            .dspark
            .as_mut()
            .expect("validated DSpark static modules");
        let collapsed = dspark.hyper_head.forward(&hidden, context)?;
        let normalized = dspark.output_norm.forward(&collapsed, context)?;
        let mut logits = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head")
            .forward(&normalized, context)?;
        let markov = dspark.markov_embedding.forward(anchor, context)?;
        let adjustment = dspark.markov_output.forward(&markov, context)?;
        logits = logits.add(&adjustment.broadcast_to(logits.shape(), context)?, context)?;
        Ok(logits)
    }

    /// Executes one target or prediction unit with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => {
                let hidden = unit.forward_with_provider(
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    pass,
                    provider,
                    context,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        forward.captures[position] =
                            Some(B::Tensor::mean_axis(&hidden, 2, false, context)?);
                    }
                }
                Ok(hidden)
            }
            Unit::Prediction(unit) if group > 0 => {
                let output_head = self
                    .static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head");
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    output_head,
                    pass,
                    provider,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            Unit::Dspark(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        Ok(hidden.clone())
                    }
                    ForwardMode::DsparkProposal => unit.forward_with_provider(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        pass,
                        provider,
                        context,
                    ),
                    _ => Err(Error::backend("DSpark unit selected outside DSpark mode")),
                }
            }
            _ => Err(Error::backend(format!(
                "V4 execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one tensor-partitioned target, MTP, or DSpark unit while an
    /// external provider owns routed experts. Tensor reductions and
    /// vocabulary collectives remain scoped to `parallel`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_parallel_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => {
                let hidden = unit.forward_parallel_with_provider(
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    pass,
                    provider,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        forward.captures[position] =
                            Some(B::Tensor::mean_axis(&hidden, 2, false, context)?);
                    }
                }
                Ok(hidden)
            }
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let head = self
                    .static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head");
                let output = unit.forward_parallel_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    pass,
                    provider,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                    |value, context| B::vocabulary_parallel_project(head, value, parallel, context),
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            Unit::Dspark(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        Ok(hidden.clone())
                    }
                    ForwardMode::DsparkProposal => unit.forward_parallel_with_provider(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        pass,
                        provider,
                        context,
                        |value, context| B::sum_parallel(value, parallel, context),
                    ),
                    _ => Err(Error::backend("DSpark unit selected outside DSpark mode")),
                }
            }
            _ => Err(Error::backend(format!(
                "V4 parallel execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one graph unit with stable target/MTP/DSpark observation and
    /// intervention points.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_observed<S, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match unit {
            Unit::Target(unit) if group == 0 => {
                let output = unit.forward_observed(
                    &format!("layers.{index}"),
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    context,
                    observer,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        let capture = B::Tensor::mean_axis(&output, 2, false, context)?;
                        observer
                            .observe(&format!("dspark.target_captures.{position}"), &capture)?;
                        forward.captures[position] = Some(capture);
                    }
                }
                Ok(output)
            }
            Unit::Prediction(_) | Unit::Dspark(_) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let output = <Self as LayeredArchitecture<B, S>>::forward_unit(
                    self, group, index, unit, hidden, state, forward, context,
                )?;
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output,
                )
            }
            _ => Err(Error::backend(format!(
                "V4 observed execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one observed graph unit with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_observed_with_provider<S, O, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match unit {
            Unit::Target(unit) if group == 0 => {
                let output = unit.forward_observed_with_provider(
                    &format!("layers.{index}"),
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    pass,
                    provider,
                    context,
                    observer,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        let capture = B::Tensor::mean_axis(&output, 2, false, context)?;
                        observer
                            .observe(&format!("dspark.target_captures.{position}"), &capture)?;
                        forward.captures[position] = Some(capture);
                    }
                }
                Ok(output)
            }
            Unit::Prediction(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output_head = self
                    .static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head");
                let output = unit.forward_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    output_head,
                    pass,
                    provider,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output.hidden,
                )
            }
            Unit::Dspark(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                let output = match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        hidden.clone()
                    }
                    ForwardMode::DsparkProposal => unit.forward_with_provider(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        pass,
                        provider,
                        context,
                    )?,
                    _ => return Err(Error::backend("DSpark unit selected outside DSpark mode")),
                };
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output,
                )
            }
            _ => Err(Error::backend(format!(
                "V4 observed execution unit does not match group {group}"
            ))),
        }
    }
}

fn text_static_spec(args: &V4Args) -> StaticModuleSpec {
    StaticModuleSpec {
        embedding_weight: "embed.weight".into(),
        normalization_weight: "norm.weight".into(),
        head_weight: "head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.rms_norm_eps,
        normalization_offset: 0.0,
        embedding_quantization: None,
        head_format: args.linear_format_for("head.weight"),
        tied_head: false,
    }
}

fn prediction_groups(args: &V4Args) -> Result<SequentialPredictionGroups, Error> {
    SequentialPredictionGroups::new(
        "layers",
        usize::try_from(args.num_hidden_layers).map_err(Error::backend)?,
        (0..usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?)
            .map(|depth| format!("mtp.{depth}")),
    )
}

fn static_modules<B>(
    args: &V4Args,
    text: TextStaticModules<B>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<StaticModules<B>, Error>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    let hyper_head = HyperHead::new(
        HyperHeadSpec {
            streams: args.hc_mult,
            hidden_size: args.hidden_size,
            norm_epsilon: args.rms_norm_eps,
            epsilon: args.hc_eps,
            function: parameter("hc_head_fn")?,
            base: parameter("hc_head_base")?,
            scale: parameter("hc_head_scale")?,
        },
        context,
    )?;
    let dspark = args
        .dspark
        .as_ref()
        .map(|config| DsparkStatic::new(args, config, context))
        .transpose()?;
    Ok(StaticModules {
        text,
        hyper_head,
        dspark,
    })
}

impl<B: HyperNeuralBackend> DsparkStatic<B> {
    fn new(
        args: &V4Args,
        config: &DsparkConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let last = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)? - 1;
        let norm = |name: String| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    parameter(name)?,
                ),
                context,
            )
        };
        Ok(Self {
            main_projection: projection::<B>(
                "mtp.0.main_proj.weight",
                args.hidden_size
                    * i32::try_from(config.target_layer_ids.len()).map_err(Error::backend)?,
                args.hidden_size,
                args.linear_format_for("mtp.0.main_proj.weight"),
                context,
            )?,
            main_norm: norm("mtp.0.main_norm.weight".into())?,
            output_norm: norm(format!("mtp.{last}.norm.weight"))?,
            hyper_head: HyperHead::new(
                HyperHeadSpec {
                    streams: args.hc_mult,
                    hidden_size: args.hidden_size,
                    norm_epsilon: args.rms_norm_eps,
                    epsilon: args.hc_eps,
                    function: parameter(format!("mtp.{last}.hc_head_fn"))?,
                    base: parameter(format!("mtp.{last}.hc_head_base"))?,
                    scale: parameter(format!("mtp.{last}.hc_head_scale"))?,
                },
                context,
            )?,
            markov_embedding: B::embedding(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: config.markov_rank,
                    weight: parameter(format!("mtp.{last}.markov_head.markov_w1.weight"))?,
                    format: crate::linear_format::standard_linear_format(
                        &format!("mtp.{last}.markov_head.markov_w1.weight"),
                        LinearFormat::Dense,
                    )?,
                },
                context,
            )?,
            markov_output: projection::<B>(
                format!("mtp.{last}.markov_head.markov_w2.weight"),
                config.markov_rank,
                args.vocab_size,
                args.linear_format_for(&format!("mtp.{last}.markov_head.markov_w2.weight")),
                context,
            )?,
            confidence_head: projection::<B>(
                format!("mtp.{last}.confidence_head.proj.weight"),
                args.hidden_size + config.markov_rank,
                1,
                args.linear_format_for(&format!("mtp.{last}.confidence_head.proj.weight")),
                context,
            )?,
        })
    }
}

impl<B, S> LayeredArchitecture<B, S> for Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: PoolingAttentionCache<B::Tensor>,
{
    type Input<'a> = EmbeddedInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = RetainedValues<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        if group == 0 {
            target_group_transport()
        } else {
            let mut transport = crate::transport::prediction();
            if group == 1 {
                transport.first_owner_static_roles.push("mtp".into());
            }
            transport
        }
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_with_output_state(
            0,
            self.args.num_hidden_layers as usize,
            layout,
        )
    }

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        self.groups.execution_graph()
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        self.groups.unit_count(group)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        self.groups.unit_path(group, index)
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        self.construct_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self.state_layout_impl()?;
        if state.layout() != &expected {
            return Err(Error::backend(format!(
                "V4 runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let (input_ids, embedded, hidden, mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => {
                let embedded = self
                    .static_modules
                    .text
                    .embeddings
                    .forward(tokens, context)?;
                let hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
                (
                    tokens.clone(),
                    embedded,
                    hidden,
                    mask.cloned(),
                    ForwardMode::Target,
                )
            }
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if self.args.dspark.is_some() || depth >= self.groups.prediction_count() {
                    return Err(Error::backend(format!(
                        "V4 sequential prediction depth {depth} is unavailable"
                    )));
                }
                (
                    tokens.clone(),
                    self.static_modules
                        .text
                        .embeddings
                        .forward(tokens, context)?,
                    hidden.clone(),
                    None,
                    ForwardMode::Draft(depth),
                )
            }
            EmbeddedInput::DsparkContext { captures } => {
                let dspark = self
                    .static_modules
                    .dspark
                    .as_mut()
                    .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
                let main = dspark
                    .main_norm
                    .forward(&dspark.main_projection.forward(captures, context)?, context)?;
                let hidden = broadcast_streams::<B>(&main, &self.args, context)?;
                (
                    captures.clone(),
                    main,
                    hidden,
                    None,
                    ForwardMode::DsparkContext,
                )
            }
            EmbeddedInput::DsparkProposal { anchor, capacity } => {
                let config = self
                    .args
                    .dspark
                    .as_ref()
                    .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
                if capacity == 0 || anchor.shape().len() != 2 || anchor.dim(1) != 1 {
                    return Err(Error::backend(
                        "DSpark proposal requires a positive capacity and [batch, 1] anchor",
                    ));
                }
                let input_ids = if capacity == 1 {
                    anchor.clone()
                } else {
                    let noise = B::Tensor::full_i32(
                        config.noise_token_id,
                        &[
                            anchor.dim(0),
                            i32::try_from(capacity - 1).map_err(Error::backend)?,
                        ],
                        context,
                    )?;
                    B::Tensor::concatenate(&[anchor.clone(), noise], 1, context)?
                };
                let embedded = self
                    .static_modules
                    .text
                    .embeddings
                    .forward(&input_ids, context)?;
                let hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
                let draft_start =
                    usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?;
                let offset = state.layer(draft_start).map_err(Error::backend)?.offset();
                let keys = (offset + i32::try_from(capacity).map_err(Error::backend)?)
                    .min(self.args.sliding_window);
                let mask = Some(B::Tensor::full_f32(
                    0.0,
                    &[i32::try_from(capacity).map_err(Error::backend)?, keys],
                    context,
                )?);
                (
                    input_ids,
                    embedded,
                    hidden,
                    mask,
                    ForwardMode::DsparkProposal,
                )
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                input_ids,
                embedded,
                mask,
                mode,
                target_capture: None,
                draft_logits: None,
                draft_hidden: None,
                captures: self
                    .args
                    .dspark
                    .as_ref()
                    .map_or_else(Vec::new, |config| vec![None; config.target_layer_ids.len()]),
            },
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.groups.begin(group, initial, dependencies)
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 0,
            ForwardMode::Draft(depth) => group == depth + 1,
            ForwardMode::DsparkContext | ForwardMode::DsparkProposal => group > 0,
        }
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
    ) -> Result<B::Tensor, Self::Error> {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => {
                let hidden = unit.forward(
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    context,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        forward.captures[position] =
                            Some(B::Tensor::mean_axis(&hidden, 2, false, context)?);
                    }
                }
                Ok(hidden)
            }
            Unit::Prediction(unit) if group > 0 => {
                let output_head = self
                    .static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head");
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    output_head,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            Unit::Dspark(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        Ok(hidden.clone())
                    }
                    ForwardMode::DsparkProposal => unit.forward(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        context,
                    ),
                    _ => Err(Error::backend("DSpark unit selected outside DSpark mode")),
                }
            }
            _ => Err(Error::backend(format!(
                "V4 execution unit does not match group {group}"
            ))),
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 && matches!(forward.mode, ForwardMode::Target) {
            forward.target_capture = if self.args.dspark.is_some() {
                Some(B::Tensor::concatenate(
                    &forward
                        .captures
                        .iter()
                        .map(|capture| {
                            capture.clone().ok_or_else(|| {
                                Error::backend("configured DSpark target capture was not produced")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    -1,
                    context,
                )?)
            } else {
                Some(hidden.clone())
            };
        }
        if group == self.groups.prediction_count()
            && matches!(forward.mode, ForwardMode::DsparkProposal)
        {
            let dspark = self
                .static_modules
                .dspark
                .as_mut()
                .expect("DSpark proposal mode has pinned DSpark modules");
            let collapsed = dspark.hyper_head.forward(hidden, context)?;
            let normalized = dspark.output_norm.forward(&collapsed, context)?;
            let mut logits = self
                .static_modules
                .text
                .lm_head
                .as_mut()
                .expect("validated V4 models have an untied output head")
                .forward(&normalized, context)?;
            let anchor = forward.input_ids.index(
                &[eredu_nn::Index::Full, eredu_nn::Index::Range(0, 1)],
                context,
            )?;
            let markov = dspark.markov_embedding.forward(&anchor, context)?;
            let adjustment = dspark.markov_output.forward(&markov, context)?;
            let adjustment = adjustment.broadcast_to(logits.shape(), context)?;
            logits = logits.add(&adjustment, context)?;
            forward.draft_logits = Some(logits);
        }
        if group > 0 && matches!(forward.mode, ForwardMode::Draft(_)) {
            forward.draft_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => {
                let hidden = self.static_modules.hyper_head.forward(hidden, context)?;
                let hidden = self.static_modules.text.norm.forward(&hidden, context)?;
                self.static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head")
                    .forward(&hidden, context)
            }
            ForwardMode::Draft(_) | ForwardMode::DsparkProposal => forward
                .draft_logits
                .clone()
                .ok_or_else(|| Error::backend("V4 draft group produced no logits")),
            ForwardMode::DsparkContext => Ok(hidden.clone()),
        }
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.input_ids),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_capture.as_ref(),
            forward.draft_logits.as_ref(),
            forward.draft_hidden.as_ref(),
        ])
        .with_extras(&forward.captures)
    }
}

fn target_group_transport() -> eredu_runtime::ArchitectureGroupTransport {
    let mut transport = crate::transport::decoder();
    transport.last_owner_static_roles.push("hyper_head".into());
    transport
}

impl<B, S> ParallelLayeredArchitecture<B, S> for Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: PoolingAttentionCache<B::Tensor>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| Error::backend("V4 model was not built with local geometry"))?
            .state_layout()
            .clone();
        if state.layout() != &expected {
            return Err(Error::backend(format!(
                "V4 runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let (input_ids, embedded, hidden, mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => {
                let embedded = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.text.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                let hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
                (
                    tokens.clone(),
                    embedded,
                    hidden,
                    mask.cloned(),
                    ForwardMode::Target,
                )
            }
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if self.args.dspark.is_some() || depth >= self.groups.prediction_count() {
                    return Err(Error::backend(format!(
                        "V4 sequential prediction depth {depth} is unavailable"
                    )));
                }
                (
                    tokens.clone(),
                    B::vocabulary_parallel_lookup(
                        &mut self.static_modules.text.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?,
                    hidden.clone(),
                    None,
                    ForwardMode::Draft(depth),
                )
            }
            EmbeddedInput::DsparkContext { captures } => {
                let dspark = self
                    .static_modules
                    .dspark
                    .as_mut()
                    .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
                let main = dspark
                    .main_norm
                    .forward(&dspark.main_projection.forward(captures, context)?, context)?;
                let hidden = broadcast_streams::<B>(&main, &self.args, context)?;
                (
                    captures.clone(),
                    main,
                    hidden,
                    None,
                    ForwardMode::DsparkContext,
                )
            }
            EmbeddedInput::DsparkProposal { anchor, capacity } => {
                let config = self
                    .args
                    .dspark
                    .as_ref()
                    .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
                if capacity == 0 || anchor.shape().len() != 2 || anchor.dim(1) != 1 {
                    return Err(Error::backend(
                        "DSpark proposal requires a positive capacity and [batch, 1] anchor",
                    ));
                }
                let input_ids = if capacity == 1 {
                    anchor.clone()
                } else {
                    let noise = B::Tensor::full_i32(
                        config.noise_token_id,
                        &[
                            anchor.dim(0),
                            i32::try_from(capacity - 1).map_err(Error::backend)?,
                        ],
                        context,
                    )?;
                    B::Tensor::concatenate(&[anchor.clone(), noise], 1, context)?
                };
                let embedded = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.text.embeddings,
                    &input_ids,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                let hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
                let draft_start =
                    usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?;
                let offset = state.layer(draft_start).map_err(Error::backend)?.offset();
                let keys = (offset + i32::try_from(capacity).map_err(Error::backend)?)
                    .min(self.args.sliding_window);
                let mask = Some(B::Tensor::full_f32(
                    0.0,
                    &[i32::try_from(capacity).map_err(Error::backend)?, keys],
                    context,
                )?);
                (
                    input_ids,
                    embedded,
                    hidden,
                    mask,
                    ForwardMode::DsparkProposal,
                )
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                input_ids,
                embedded,
                mask,
                mode,
                target_capture: None,
                draft_logits: None,
                draft_hidden: None,
                captures: self
                    .args
                    .dspark
                    .as_ref()
                    .map_or_else(Vec::new, |config| vec![None; config.target_layer_ids.len()]),
            },
        })
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => {
                let hidden = unit.forward_parallel(
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        forward.captures[position] =
                            Some(B::Tensor::mean_axis(&hidden, 2, false, context)?);
                    }
                }
                Ok(hidden)
            }
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_parallel(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                    |value, context| {
                        B::vocabulary_parallel_project(
                            self.static_modules
                                .text
                                .lm_head
                                .as_mut()
                                .expect("validated V4 models have an untied output head"),
                            value,
                            parallel,
                            context,
                        )
                    },
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            Unit::Dspark(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        Ok(hidden.clone())
                    }
                    ForwardMode::DsparkProposal => unit.forward_parallel(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        context,
                        |value, context| B::sum_parallel(value, parallel, context),
                    ),
                    _ => Err(Error::backend("DSpark unit selected outside DSpark mode")),
                }
            }
            _ => Err(Error::backend(format!(
                "V4 parallel execution unit does not match group {group}"
            ))),
        }
    }

    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 && matches!(forward.mode, ForwardMode::Target) {
            forward.target_capture = if self.args.dspark.is_some() {
                Some(B::Tensor::concatenate(
                    &forward
                        .captures
                        .iter()
                        .map(|capture| {
                            capture.clone().ok_or_else(|| {
                                Error::backend("configured DSpark target capture was not produced")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    -1,
                    context,
                )?)
            } else {
                Some(hidden.clone())
            };
        }
        if group == self.groups.prediction_count()
            && matches!(forward.mode, ForwardMode::DsparkProposal)
        {
            let dspark = self
                .static_modules
                .dspark
                .as_mut()
                .expect("DSpark proposal mode has pinned DSpark modules");
            let collapsed = dspark.hyper_head.forward(hidden, context)?;
            let normalized = dspark.output_norm.forward(&collapsed, context)?;
            let mut logits = B::vocabulary_parallel_project(
                self.static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head"),
                &normalized,
                parallel,
                context,
            )?;
            let anchor = forward.input_ids.index(
                &[eredu_nn::Index::Full, eredu_nn::Index::Range(0, 1)],
                context,
            )?;
            let markov = dspark.markov_embedding.forward(&anchor, context)?;
            let adjustment = dspark.markov_output.forward(&markov, context)?;
            logits = logits.add(&adjustment.broadcast_to(logits.shape(), context)?, context)?;
            forward.draft_logits = Some(logits);
        }
        if group > 0 && matches!(forward.mode, ForwardMode::Draft(_)) {
            forward.draft_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => {
                let hidden = self.static_modules.hyper_head.forward(hidden, context)?;
                let hidden = self.static_modules.text.norm.forward(&hidden, context)?;
                B::vocabulary_parallel_project(
                    self.static_modules
                        .text
                        .lm_head
                        .as_mut()
                        .expect("validated V4 models have an untied output head"),
                    &hidden,
                    parallel,
                    context,
                )
            }
            ForwardMode::Draft(_) | ForwardMode::DsparkProposal => forward
                .draft_logits
                .clone()
                .ok_or_else(|| Error::backend("V4 draft group produced no logits")),
            ForwardMode::DsparkContext => Ok(hidden.clone()),
        }
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: PoolingAttentionCache<B::Tensor>,
{
    type Boundary = TargetBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        TargetBoundarySchema::from_args(self.args())
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        _state: &mut S,
        _expected: &StateLayout,
        _first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let input = match input {
            LayeredPartitionInput::Tokens(tokens) => TargetPartitionInput::Tokens(tokens),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => TargetPartitionInput::Hidden {
                hidden,
                boundary: auxiliary,
            },
        };
        self.begin_routed_target_partition(input, mask, None, context)
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        _state: &mut S,
        _expected: &StateLayout,
        _first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let input = match input {
            LayeredPartitionInput::Tokens(tokens) => TargetPartitionInput::Tokens(tokens),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => TargetPartitionInput::Hidden {
                hidden,
                boundary: auxiliary,
            },
        };
        self.begin_routed_target_partition(input, mask, Some(parallel), context)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TargetBoundary<B::Tensor>>, Self::Error> {
        match self.finish_routed_target_partition(
            hidden,
            forward,
            owns_output,
            parallel,
            context,
        )? {
            TargetPartitionOutput::Final {
                logits,
                draft_hidden,
            } => Ok(LayeredPartitionOutput::Final {
                output: logits,
                retained: Some(draft_hidden),
            }),
            TargetPartitionOutput::Boundary { hidden, boundary } => {
                Ok(LayeredPartitionOutput::Boundary {
                    hidden,
                    auxiliary: boundary,
                })
            }
        }
    }
}

/// Builds the shared learned or token-selected MoE policy for one target
/// layer. The caller supplies token-table IDs to `RouteSource::Selected` for
/// hash layers.
pub fn moe_policy(args: &V4Args, layer: usize) -> Result<MoePolicy, Error> {
    moe_policy_at(args, layer, &format!("layers.{layer}.ffn"))
}

/// Returns the architecture-owned routed expert specification for a target or MTP layer.
pub fn expert_bank_spec(
    args: &V4Args,
    layer: usize,
) -> Result<eredu_nn::GatedProductExpertBankSpec, Error> {
    let root = if layer < args.num_hidden_layers as usize {
        format!("layers.{layer}.ffn")
    } else {
        format!("mtp.{}.ffn", layer - args.num_hidden_layers as usize)
    };
    crate::deepseek::moe::expert_bank_spec(&moe_policy_at(args, layer, &root)?)
}

pub(crate) fn localized_expert_bank_spec(
    args: &V4Args,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<eredu_nn::GatedProductExpertBankSpec, Error> {
    let mut spec = expert_bank_spec(args, layer)?;
    spec.expert_count = expert_count;
    spec.intermediate_dimensions = intermediate_dimensions;
    spec.validate()?;
    Ok(spec)
}

pub(crate) fn moe_policy_at(args: &V4Args, layer: usize, root: &str) -> Result<MoePolicy, Error> {
    if layer
        >= usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
            .map_err(Error::backend)?
    {
        return Err(Error::backend(format!(
            "V4 target or prediction layer {layer} is out of range"
        )));
    }
    let expert_format = match args.expert_format {
        ExpertFormat::Dense => LinearFormat::Dense,
        ExpertFormat::MxFp4 => LinearFormat::MxFp4,
        ExpertFormat::BlockFp8 => match args.linear_format {
            format @ LinearFormat::E4M3BlockFp8(_) => format,
            _ => LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0)
                    .map_err(Error::backend)?,
            ),
        },
    };
    Ok(MoePolicy {
        layer,
        hidden: args.hidden_size,
        expert_count: args.n_routed_experts,
        routes_per_token: args.num_experts_per_tok,
        expert_width: args.moe_intermediate_size,
        shared_width: args
            .moe_intermediate_size
            .checked_mul(args.n_shared_experts)
            .ok_or_else(|| Error::backend("V4 shared expert width overflowed"))?,
        scoring: eredu_nn::RoutingScoring::SqrtSoftplus,
        normalize_routes: args.norm_topk_prob,
        normalization_epsilon: 1e-20,
        routed_scaling: args.routed_scaling_factor,
        expert_groups: 1,
        selected_groups: 1,
        router_weight: format!("{root}.gate.weight"),
        correction_bias: (layer >= args.num_hash_layers as usize)
            .then(|| format!("{root}.gate.bias")),
        expert_gate_up: format!("{root}.switch_mlp.gate_up_proj"),
        expert_down: format!("{root}.switch_mlp.down_proj"),
        shared_gate: format!("{root}.shared_experts.w1.weight"),
        shared_up: format!("{root}.shared_experts.w3.weight"),
        shared_down: format!("{root}.shared_experts.w2.weight"),
        shared_gate_format: args.linear_format_for(&format!("{root}.shared_experts.w1.weight")),
        shared_up_format: args.linear_format_for(&format!("{root}.shared_experts.w3.weight")),
        shared_down_format: args.linear_format_for(&format!("{root}.shared_experts.w2.weight")),
        expert_gate_up_format: args
            .linear_formats
            .get(&format!("{root}.switch_mlp.gate_up_proj"))
            .copied()
            .unwrap_or(expert_format),
        expert_down_format: args
            .linear_formats
            .get(&format!("{root}.switch_mlp.down_proj"))
            .copied()
            .unwrap_or(expert_format),
        shared_limit: None,
        limit: args.swiglu_limit,
    })
}

/// Declares bounded local keys and every append-only pooling component for V4
/// target and prediction layers.
pub fn state_layout(args: &V4Args) -> Result<StateLayout, Error> {
    args.validate().map_err(Error::backend)?;
    let layers = usize::try_from(
        args.num_hidden_layers
            .checked_add(args.num_nextn_predict_layers)
            .ok_or_else(|| Error::backend("V4 total state layer count overflowed"))?,
    )
    .map_err(Error::backend)?;
    let attention =
        AttentionPolicy::sliding(u32::try_from(args.sliding_window).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let policies = (0..layers)
        .map(|layer| {
            let fixed = match args.attention_policy(layer) {
                Some(V4AttentionPolicy::Local) => Vec::new(),
                Some(V4AttentionPolicy::Compressed { ratio }) => {
                    let mut tensors = pooling_stream(0, ratio, args.head_dim, ratio == 4)?;
                    if ratio == 4 {
                        tensors.extend(pooling_stream(1, ratio, args.index_head_dim, true)?);
                    }
                    tensors
                }
                None => return Err(Error::backend(format!("missing V4 layer policy {layer}"))),
            };
            if fixed.is_empty() {
                LayerCachePolicy::key_only(attention, 1, args.head_dim).map_err(Error::backend)
            } else {
                LayerCachePolicy::key_only_with_fixed_state(attention, 1, args.head_dim, fixed)
                    .map_err(Error::backend)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let target_layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let mut segments = vec![StateSegmentSpec::new(
        super::TARGET_STATE_SEGMENT,
        0..target_layers,
        StateSegmentLifetime::Persistent,
        0,
    )
    .map_err(Error::backend)?];
    if layers > target_layers {
        segments.push(
            StateSegmentSpec::new(
                super::PREDICTION_STATE_SEGMENT,
                target_layers..layers,
                StateSegmentLifetime::Persistent,
                if args.dspark.is_some() { 0 } else { -1 },
            )
            .map_err(Error::backend)?,
        );
    }
    StateLayout::segmented(
        LayerSchedule::new(layers, policies).map_err(Error::backend)?,
        segments,
    )
    .map_err(Error::backend)
}

fn pooling_stream(
    stream: u32,
    ratio: i32,
    pooled_width: i32,
    overlapping: bool,
) -> Result<Vec<StateTensorPolicy>, Error> {
    let ratio = NonZeroU32::new(u32::try_from(ratio).map_err(Error::backend)?)
        .ok_or_else(|| Error::backend("V4 pooling ratio must be positive"))?;
    let source_width = if overlapping {
        pooled_width
            .checked_mul(2)
            .ok_or_else(|| Error::backend("V4 pooling source width overflowed"))?
    } else {
        pooled_width
    };
    let role = |component| StateTensorRole::Pooling { stream, component };
    let pending = |component| {
        StateTensorPolicy::new(
            role(component),
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensRem(ratio),
                StateTensorDimension::fixed(source_width).map_err(Error::backend)?,
            ],
            StateTensorDtype::Floating,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .map(|policy| policy.when_prefix_remainder_nonzero(ratio))
        .map_err(Error::backend)
    };
    let mut tensors = vec![
        pending(PoolingStateComponent::PendingValues)?,
        pending(PoolingStateComponent::PendingGates)?,
        StateTensorPolicy::new_with_residency(
            role(PoolingStateComponent::Pooled),
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(ratio),
                StateTensorDimension::fixed(pooled_width).map_err(Error::backend)?,
            ],
            StateTensorDtype::Floating,
            StateResidencyClass::SealablePaged,
        )
        .map(|policy| policy.when_prefix_at_least(ratio))
        .map_err(Error::backend)?,
    ];
    if overlapping {
        for component in [
            PoolingStateComponent::OverlapValues,
            PoolingStateComponent::OverlapGates,
        ] {
            tensors.push(
                StateTensorPolicy::new(
                    role(component),
                    vec![
                        StateTensorDimension::Batch,
                        StateTensorDimension::Fixed(ratio),
                        StateTensorDimension::fixed(pooled_width).map_err(Error::backend)?,
                    ],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map(|policy| policy.when_prefix_at_least(ratio))
                .map_err(Error::backend)?,
            );
        }
    }
    Ok(tensors)
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}

fn projection<B: eredu_nn::NeuralBackend>(
    name: impl Into<String>,
    input: i32,
    output: i32,
    format: LinearFormat,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, Error> {
    let name = name.into();
    B::linear(
        LinearSpec {
            input,
            output,
            weight: parameter(&name)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(&name, format)?,
        },
        context,
    )
}

fn broadcast_streams<B: HyperNeuralBackend>(
    hidden: &B::Tensor,
    args: &V4Args,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    hidden.expand_dims(2, context)?.broadcast_to(
        &[hidden.dim(0), hidden.dim(1), args.hc_mult, args.hidden_size],
        context,
    )
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDtype};

    #[test]
    fn configured_capture_count_owns_wire_cardinality() {
        let schema = TargetBoundarySchema {
            hidden_size: 24,
            activation_hidden_size: 72,
            capture_count: 2,
        };
        assert_eq!(schema.activation_hidden_size(), 72);
        let tensors = schema.wire_schema().unwrap().resolve(1, 5).unwrap();
        assert_eq!(tensors.primary().shape(), [1, 5, 72]);
        assert_eq!(tensors.auxiliary().len(), 3);
        assert_eq!(tensors.auxiliary()[0].dtype(), BoundaryTensorDtype::Uint32);
        assert_eq!(tensors.auxiliary()[1].role(), "capture.0");
        assert_eq!(tensors.auxiliary()[1].shape(), [1, 5, 24]);
        assert_eq!(tensors.auxiliary()[2].role(), "capture.1");
    }

    #[test]
    fn target_transport_owns_the_distinct_hyper_head_role() {
        let transport = target_group_transport();
        assert_eq!(
            transport.last_owner_static_roles,
            ["norm", "output", "hyper_head"]
        );
    }
}
