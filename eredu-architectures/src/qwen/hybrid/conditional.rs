//! Conditional-generation graph over the shared Qwen vision tower and hybrid decoder.

mod observation;

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend, Index,
    Parameterized, Tensor,
};
use eredu_runtime::{
    ExecutionUnitLayout, LayerRuntimeState, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, ParameterProvider, ResidentExpertProvider, RuntimeStateComponents,
    StateLayout,
};

use eredu_runtime::{ComponentEquations as _, ParallelComponentKernel as _};

use crate::decoder::{DecoderStaticResources, PartitionStaticModules};
use crate::qwen::ingress::{prepare_input, prepared_semantic_tokens, ProjectedTokenPolicy};
use crate::qwen::vision::{VisionBlock, VisionInput, VisionMode, VisionState, VisionStatic};
use crate::qwen::vl::InputPart;
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    media_plan::QwenInputPartPlan,
};

use super::{Block, ConditionalLocalGeometry, ForwardMode, ParsedHybridConfig, PredictionUnit};

/// Stable execution-group identity for conditional Qwen vision ingress.
pub const VISION_EXECUTION_GROUP: &str = "vision";

enum PreparedPart<T> {
    Text {
        tokens: T,
        embeddings: T,
        ready: bool,
    },
    Media {
        tokens: T,
    },
}

/// Pinned hybrid text and shared-vision modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ConditionalStaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Hybrid token embedding, final normalization, and vocabulary head.
    pub text: PartitionStaticModules<B>,
    /// Shared Qwen patch, position, and merger modules.
    pub vision: VisionStatic<B>,
    /// Canonical fusion and normalization shared by all prediction depths.
    pub prediction: Option<super::PredictionShared<B>>,
}

/// One conditional vision, target-text, or MTP unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum ConditionalUnit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Shared vision block.
    Vision(VisionBlock<B>),
    /// Hybrid recurrent/full-attention target block.
    Target(Block<B>),
    /// Configured hybrid prediction depth.
    Prediction(PredictionUnit<B>),
}

impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for ConditionalKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn routed_unit_observations(&self) -> bool {
        self.prediction_steps == 0
    }
    fn routed_sparse_observations(&self) -> bool {
        self.prediction_steps == 0
    }
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_target_observed(
            address, unit, hidden, state, forward, provider, context, observer,
        )
    }

    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        match group {
            0 => <Self as eredu_runtime::ComponentEquations<B, S>>::execute(
                self, address, unit, hidden, state, forward, context,
            ),
            _ => ConditionalKernel::forward_unit_with_provider(
                self, group, index, unit, hidden, state, forward, provider, context,
            ),
        }
    }
}

impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for ConditionalKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        self.prediction_steps == 0
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        self.prediction_steps == 0
    }
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_target_parallel_observed(
            address, unit, hidden, state, forward, provider, parallel, context, observer,
        )
    }

    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        match group {
            0 => <Self as eredu_runtime::ParallelComponentKernel<B, S>>::execute_parallel(
                self, address, unit, hidden, state, forward, parallel, context,
            ),
            _ => ConditionalKernel::forward_unit_with_provider_parallel(
                self, group, index, unit, hidden, state, forward, provider, parallel, context,
            ),
        }
    }
}

/// Target media input or one embedded prediction request.
pub enum ConditionalInput<'a, T> {
    /// Admitted prepared input using the same target/media equations.
    Prepared(crate::composite_execution::PreparedCompositeInput<'a, T, QwenInputPartPlan>),
    /// Ordered text/media target input.
    Target {
        /// Ordered semantic segments.
        parts: &'a [InputPart<'a, T>],
        /// Concatenated flattened patches for media segments.
        pixels: Option<&'a T>,
        /// Optional explicit decoder attention mask.
        mask: Option<&'a T>,
    },
    /// One configured MTP depth.
    Draft {
        /// Next-token identities.
        tokens: &'a T,
        /// Target or prior-depth hidden state.
        hidden: &'a T,
        /// Zero-based prediction depth.
        depth: usize,
    },
}

impl<B, S> crate::composite_execution::ParallelCompositeArchitecture<B, S>
    for ConditionalLayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
}

impl<B, S> CompositeArchitecture<B, S> for ConditionalLayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn supports_chunked_text_prefill() -> bool {
        super::CHUNKED_TEXT_PREFILL
    }

    fn prepared_input<'a>(
        input: crate::composite_execution::PreparedCompositeInput<
            'a,
            B::Tensor,
            Self::InputPartPlan,
        >,
    ) -> Self::Input<'a> {
        ConditionalInput::Prepared(input)
    }
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<
        crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    > {
        match input {
            ConditionalInput::Prepared(input) => Some(*input),
            _ => None,
        }
    }

    type InputPartPlan = QwenInputPartPlan;
    type PrefillRequest =
        crate::composite_execution::SingleCompositePrefillRequest<B::Tensor, Self::InputPartPlan>;

    fn prepare_prefill_request(
        prepared: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Self::Error> {
        crate::composite_execution::SingleCompositePrefillRequest::new(prepared, admitted)
    }
    type AdmissionConfig = ParsedHybridConfig;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.kernel().parsed.clone()
    }

    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<
        crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        eredu_core::CapabilityError,
    > {
        crate::media_plan::admit_qwen_hybrid_input(config, input, inspector)
    }

    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::Tensor::concatenate(
            &prepared_semantic_tokens(input, ProjectedTokenPolicy::Placeholder(0), context)?,
            1,
            context,
        )
    }

    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        group == 1
            || (group == 0
                && input
                    .admitted()
                    .parts()
                    .iter()
                    .any(|part| matches!(part, QwenInputPartPlan::Media { .. })))
    }

    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group == 0 {
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    QwenInputPartPlan::Media { shape, .. } => Some(shape.decoder_positions),
                    _ => None,
                })
                .try_fold(0_u64, |total, positions| total.checked_add(positions))
                .ok_or_else(|| "conditional Qwen projected media positions overflowed".to_owned())?
        } else {
            input.admitted().decoder_positions()
        };
        i32::try_from(positions)
            .map_err(|_| "conditional Qwen prepared group boundary sequence exceeds i32".to_owned())
    }

    fn prepared_group_continuation_geometry(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<crate::composite_execution::CompositeContinuationGeometry>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = input
            .admitted()
            .parts()
            .iter()
            .filter_map(|part| match part {
                QwenInputPartPlan::Media { ingress, .. } => Some(&ingress.patch_grid),
                _ => None,
            })
            .flatten()
            .try_fold(0_i64, |total, &(time, height, width)| {
                i64::from(time)
                    .checked_mul(i64::from(height))
                    .and_then(|area| area.checked_mul(i64::from(width)))
                    .and_then(|patches| total.checked_add(patches))
                    .ok_or_else(|| {
                        "conditional Qwen continuation patch geometry overflowed".to_owned()
                    })
            })?;
        let patches = i32::try_from(patches)
            .map_err(|_| "conditional Qwen continuation patch count exceeds i32".to_owned())?;
        let width = self
            .kernel()
            .parsed
            .vision
            .as_ref()
            .ok_or_else(|| "conditional Qwen continuation has no vision config".to_owned())?
            .hidden_size;
        Ok(Some((patches, width).into()))
    }

    fn prepared_group_continuation_batched(&self, group: usize) -> bool {
        group != 0
    }

    fn prepared_group_collective_waves(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        let pipeline_stages = stage_units.len();
        if group != 0 || tensor_partitions <= 1 || pipeline_stages <= 1 {
            return Ok(None);
        }
        let vision = self.kernel().parsed.vision.as_ref().ok_or_else(|| {
            "conditional Qwen collective schedule has no vision config".to_owned()
        })?;
        let mut patch_positions = 0_u64;
        let mut projected_positions = 0_u64;
        for part in input.admitted().parts() {
            let QwenInputPartPlan::Media { ingress, shape } = part else {
                continue;
            };
            projected_positions = projected_positions
                .checked_add(shape.decoder_positions)
                .ok_or_else(|| "conditional Qwen projected positions overflowed".to_owned())?;
            for &(time, height, width) in &ingress.patch_grid {
                let patches = u64::try_from(time)
                    .ok()
                    .and_then(|time| {
                        u64::try_from(height)
                            .ok()
                            .and_then(|height| time.checked_mul(height))
                    })
                    .and_then(|area| {
                        u64::try_from(width)
                            .ok()
                            .and_then(|width| area.checked_mul(width))
                    })
                    .ok_or_else(|| "conditional Qwen patch-grid geometry overflowed".to_owned())?;
                patch_positions = patch_positions
                    .checked_add(patches)
                    .ok_or_else(|| "conditional Qwen patch positions overflowed".to_owned())?;
            }
        }
        let patch_positions = i32::try_from(patch_positions)
            .map_err(|_| "conditional Qwen patch positions exceed i32".to_owned())?;
        let projected_positions = i32::try_from(projected_positions)
            .map_err(|_| "conditional Qwen projected positions exceed i32".to_owned())?;
        if patch_positions == 0 || projected_positions == 0 {
            return Ok(Some(vec![Vec::new(); pipeline_stages]));
        }
        let layers = vision.layer_count();
        if layers < pipeline_stages {
            return Err(
                "conditional Qwen vision collective schedule has fewer units than PP stages".into(),
            );
        }
        let patch_shape = vec![patch_positions, vision.hidden_size];
        let projected_shape = vec![projected_positions, vision.out_hidden_size];
        let mut ingress_operations = Vec::new();
        for (part, plan) in input
            .prepared()
            .parts()
            .iter()
            .zip(input.admitted().parts())
        {
            if let QwenInputPartPlan::TextTokens { positions } = plan {
                let batch = part.payload().value().dim(0);
                let positions = i32::try_from(*positions)
                    .map_err(|_| "conditional Qwen text positions exceed i32".to_owned())?;
                ingress_operations.push(
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: vec![batch, positions, self.kernel().parsed.text.hidden_size],
                    },
                );
            }
        }
        let mut stages = Vec::with_capacity(pipeline_stages);
        for range in &stage_units[..pipeline_stages] {
            let mut operations = ingress_operations.clone();
            for layer in range.clone() {
                operations.extend([
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: patch_shape.clone(),
                    },
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: patch_shape.clone(),
                    },
                ]);
                if vision
                    .layer_policy(layer)
                    .is_some_and(|policy| policy.deepstack_merger.is_some())
                {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: projected_shape.clone(),
                    });
                }
            }
            if range.end == layers {
                operations.push(crate::composite_execution::CompositeTensorCollective::Sum {
                    shape: projected_shape.clone(),
                });
            }
            stages.push(operations);
        }
        Ok(Some(stages))
    }

    fn partition_boundary_schema(
        &self,
        source_group: usize,
        destination_group: usize,
        selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        batch: i32,
        source_sequence: i32,
        group_sequences: &[i32],
        continuation: Option<(i32, i32)>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Self::Error> {
        let vision_edge = source_group == 0 && matches!(destination_group, 0 | 1);
        let decoder_continuation = source_group == 1 && destination_group == 1;
        if !vision_edge && !decoder_continuation {
            return Ok(None);
        }
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        let vision_continuation = source_group == 0 && destination_group == 0;
        let vision = self
            .kernel()
            .parsed
            .vision
            .as_ref()
            .ok_or_else(|| Error::backend("conditional Qwen vision boundary has no config"))?;
        let primary = if vision_continuation {
            eredu_runtime::BoundaryTensorSpec::new(
                "hidden",
                [Dim::Sequence, Dim::Fixed(vision.hidden_size)],
                Dtype::Activation,
            )
        } else {
            eredu_runtime::BoundaryTensorSpec::new(
                "hidden",
                [
                    Dim::Batch,
                    Dim::Sequence,
                    Dim::Fixed(self.kernel().parsed.text.hidden_size),
                ],
                Dtype::Activation,
            )
        };
        let schema = eredu_runtime::BoundaryWireSchema::new(
            if vision_continuation {
                "qwen_conditional.vision_continuation"
            } else if decoder_continuation {
                "qwen_conditional.decoder"
            } else {
                "qwen_conditional.vision_to_decoder"
            },
            primary,
            (0..selected.auxiliary().len()).map(|index| {
                eredu_runtime::BoundaryTensorSpec::new(
                    format!("deepstack.{index}"),
                    [
                        Dim::Batch,
                        Dim::Sequence,
                        Dim::Fixed(self.kernel().parsed.text.hidden_size),
                    ],
                    Dtype::Activation,
                )
            }),
        )
        .map_err(|error| Error::backend(error.to_string()))?;
        let primary_sequence = if vision_continuation {
            continuation
                .ok_or_else(|| {
                    Error::backend("conditional Qwen continuation has no patch geometry")
                })?
                .0
        } else {
            source_sequence
        };
        let vision_sequence = *group_sequences.first().ok_or_else(|| {
            Error::backend("conditional Qwen boundary has no vision sequence geometry")
        })?;
        // Vision edges carry compact media features. Decoder continuations
        // carry contributions expanded to the declared decoder extent; receiving
        // owners do not need the original visual mask or input embeddings.
        let deepstack_sequence = if decoder_continuation {
            source_sequence
        } else if vision_sequence > 0 {
            vision_sequence
        } else {
            source_sequence
        };
        schema
            .resolve_each(
                batch,
                std::iter::once(primary_sequence).chain(std::iter::repeat_n(
                    deepstack_sequence,
                    selected.auxiliary().len(),
                )),
            )
            .map(Some)
            .map_err(|error| Error::backend(error.to_string()))
    }

    fn partition_boundary_values(
        &self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>, Self::Error> {
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let deepstack = if source_group == destination_group {
            forward
                .vision_state
                .as_ref()
                .ok_or_else(|| Error::backend("conditional Qwen continuation has no vision state"))?
                .deepstack_features()
        } else {
            &forward.deepstack
        };
        if deepstack.len() != schema.auxiliary().len() {
            return Err(Error::backend(
                "conditional Qwen vision boundary has incomplete DeepStack context",
            ));
        }
        let mut values = Vec::with_capacity(1 + deepstack.len());
        values.push(
            eredu_runtime::ArchitectureBoundaryValue::new(schema.primary().role(), hidden.clone())
                .map_err(|error| Error::backend(error.to_string()))?,
        );
        for (spec, value) in schema.auxiliary().iter().zip(deepstack) {
            values.push(
                eredu_runtime::ArchitectureBoundaryValue::new(spec.role(), value.clone())
                    .map_err(|error| Error::backend(error.to_string()))?,
            );
        }
        Ok(Some(values))
    }

    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        if source_group == 1 && destination_group == 1 {
            if values.len() != 1 + schema.auxiliary().len() {
                return Err(Error::backend(format!(
                    "conditional Qwen decoder boundary has {} values, expected {}",
                    values.len(),
                    1 + schema.auxiliary().len()
                )));
            }
            let mut values = values.into_iter();
            let hidden = values.next().expect("validated decoder boundary primary");
            let boundary = eredu_runtime::ArchitectureBoundary::decode(
                &self.kernel().pipeline_boundary_schema(),
                values.collect(),
            )
            .map_err(|error| Error::backend(error.to_string()))?;
            forward.deepstack = boundary.deepstack;
            return Ok(Some(hidden));
        }
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let expected = 1 + schema.auxiliary().len();
        if values.len() != expected {
            return Err(Error::backend(format!(
                "conditional Qwen vision boundary has {} values, expected {expected}",
                values.len()
            )));
        }
        let mut values = values.into_iter();
        let hidden = values.next().expect("validated boundary primary");
        if source_group == destination_group {
            let vision = forward.vision_state.as_mut().ok_or_else(|| {
                Error::backend("conditional Qwen continuation has no vision state")
            })?;
            let mut retained = vision
                .retained_values()
                .take(2)
                .cloned()
                .collect::<Vec<_>>();
            retained.extend(values);
            vision.replace_retained_values(retained)?;
        } else {
            forward.deepstack = values.collect();
            forward.vision_output = Some(hidden.clone());
        }
        Ok(Some(hidden))
    }
}

/// Request-local conditional execution values.
pub struct ConditionalForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    parts: Vec<PreparedPart<T>>,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
    visual_mask: Option<T>,
    target_hidden: Option<T>,
}

/// Request state transported while placed owners execute conditional vision.
pub struct ConditionalPipelineVisionState<T> {
    /// Current shared-vision activation, or a decoder-width placeholder for text-only input.
    pub hidden: T,
    parts: Vec<PreparedPart<T>>,
    mask: Option<T>,
    vision: Option<VisionState<T>>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
}

/// Decoder-facing conditional values after the placed vision group completes.
pub struct ConditionalPipelinePrepared<T> {
    /// Assembled decoder-width activation.
    pub hidden: T,
    /// Optional explicit or causal decoder mask.
    pub mask: Option<T>,
    /// Fixed-shape DeepStack additions indexed by decoder layer.
    pub deepstack: Vec<T>,
}

impl<T: Clone> ConditionalPipelinePrepared<T> {
    /// Converts target preparation into the canonical layered forward state.
    pub fn into_layered_forward(self) -> LayeredForwardState<T, ConditionalForwardContext<T>> {
        LayeredForwardState {
            hidden: self.hidden.clone(),
            context: ConditionalForwardContext {
                tokens: self.hidden.clone(),
                embedded: self.hidden,
                mask: self.mask,
                mode: ForwardMode::Target,
                parts: Vec::new(),
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                deepstack: self.deepstack,
                visual_mask: None,
                target_hidden: None,
            },
        }
    }

    /// Recovers the decoder boundary from a completed layered forward.
    pub fn from_layered_forward(
        forward: LayeredForwardState<T, ConditionalForwardContext<T>>,
    ) -> (T, ConditionalPipelineBoundary<T>) {
        (
            forward.hidden,
            ConditionalPipelineBoundary {
                deepstack: forward.context.deepstack,
            },
        )
    }
}

/// Family-owned schema for conditional decoder values crossing pipeline ranks.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ConditionalPipelineBoundarySchema {
    hidden_size: i32,
    deepstack_count: usize,
}

impl ConditionalPipelineBoundarySchema {
    /// Declares the target/DeepStack boundary without constructing native resources.
    pub fn for_config(parsed: &ParsedHybridConfig) -> Result<Self, Error> {
        let vision = parsed
            .vision
            .as_ref()
            .ok_or_else(|| Error::backend("conditional Qwen requires vision config"))?;
        vision
            .validate_for(VisionMode::WindowScheduled)
            .map_err(Error::backend)?;
        let schema = Self {
            hidden_size: parsed.text.hidden_size,
            deepstack_count: vision.deepstack_layer_count(),
        };
        eredu_runtime::ArchitectureBoundary::wire_schema(&schema).map_err(Error::backend_source)?;
        Ok(schema)
    }

    /// Returns the expected number of transported DeepStack tensors.
    pub const fn deepstack_count(self) -> usize {
        self.deepstack_count
    }
}

impl eredu_runtime::ArchitectureBoundary for ConditionalPipelineBoundarySchema {
    type Boundary<T> = ConditionalPipelineBoundary<T>;

    const IDENTITY: &'static str = "qwen_conditional.decoder";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        (0..self.deepstack_count)
            .map(|index| {
                eredu_runtime::BoundaryTensorSpec::new(
                    format!("deepstack.{index}"),
                    [Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                    Dtype::Activation,
                )
            })
            .collect()
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(ConditionalPipelineBoundary { deepstack: tensors })
    }

    /// Encodes typed conditional decoder context after validating cardinality.
    fn encode<T>(
        &self,
        boundary: ConditionalPipelineBoundary<T>,
    ) -> Result<
        Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,
        eredu_runtime::ArchitectureBoundaryError,
    > {
        if boundary.deepstack.len() != self.deepstack_count {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "qwen_conditional.decoder",
                expected: self.deepstack_count,
                actual: boundary.deepstack.len(),
            });
        }
        boundary
            .deepstack
            .into_iter()
            .enumerate()
            .map(|(index, tensor)| {
                eredu_runtime::ArchitectureBoundaryValue::new(format!("deepstack.{index}"), tensor)
            })
            .collect()
    }
}

/// Typed immutable decoder context transported between conditional partitions.
pub struct ConditionalPipelineBoundary<T> {
    /// Per-layer DeepStack additions.
    pub deepstack: Vec<T>,
}

impl<T> ConditionalPipelineBoundary<T> {
    /// Splits prepared state into its evolving activation and immutable boundary.
    pub fn from_prepared(prepared: ConditionalPipelinePrepared<T>) -> (T, Self) {
        (
            prepared.hidden,
            Self {
                deepstack: prepared.deepstack,
            },
        )
    }

    /// Reconstructs decoder state on a downstream partition.
    pub fn into_prepared(self, hidden: T) -> ConditionalPipelinePrepared<T> {
        ConditionalPipelinePrepared {
            hidden,
            mask: None,
            deepstack: self.deepstack,
        }
    }
}

/// Conditional target input from the input owner or an upstream pipeline rank.
pub enum ConditionalPartitionInput<'a, T> {
    /// Text token identities entering the target embedding boundary.
    Tokens {
        /// Token identities.
        tokens: &'a T,
        /// Existing target cache offset.
        offset: i32,
    },
    /// Evolving activation plus typed immutable target context.
    Hidden {
        /// Upstream target activation.
        hidden: T,
        /// Family-owned DeepStack boundary.
        boundary: ConditionalPipelineBoundary<T>,
    },
}

impl<T> ConditionalForwardContext<T> {
    /// Hidden state captured after the selected text execution group.
    pub const fn target_hidden(&self) -> Option<&T> {
        self.target_hidden.as_ref()
    }
}

/// Shared execution adapter over conditional Qwen equations and selected pinned resources.
pub type ConditionalLayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    ConditionalKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Conditional Qwen semantic equations, native resource factories and media geometry.
pub struct ConditionalKernel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    parsed: ParsedHybridConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    static_modules: ConditionalStaticModules<B>,
    prediction_steps: usize,
    owns_input: bool,
    parallel_geometry: Option<std::sync::Arc<ConditionalLocalGeometry>>,
    partition_state_layout: Option<StateLayout>,
    partition_state_offset: usize,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> eredu_runtime::ComponentKernel<B>
    for ConditionalKernel<B>
{
    type Input<'a> = ConditionalInput<'a, B::Tensor>;
    type Unit = ConditionalUnit<B>;
    type ForwardContext = ConditionalForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::Error> {
        let target_layers =
            usize::try_from(self.parsed.text.num_hidden_layers).map_err(Error::backend)?;
        let prediction_layers =
            usize::try_from(self.parsed.text.mtp_num_hidden_layers).map_err(Error::backend)?;
        let layer_count = target_layers
            .checked_add(prediction_layers)
            .ok_or_else(|| Error::backend("conditional Qwen state layer count overflowed"))?;
        let global_layer_start = state.global_layer_offset();
        let global_layer_end = global_layer_start
            .checked_add(state.layout().len())
            .ok_or_else(|| Error::backend("conditional Qwen owned layer range overflowed"))?;
        if global_layer_end > layer_count {
            return Err(Error::backend(format!(
                "conditional Qwen owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
            )));
        }
        eredu_runtime::ModelStateIdentity::new(
            self.parsed.text.variant.model_kind().canonical_name(),
            self.parsed.text.model_type.clone(),
            super::conditional_prompt_cache_architecture_fingerprint(&self.parsed),
            layer_count,
            global_layer_start,
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
        let recipes = super::static_recipes(source)?;
        crate::static_parameters::module_recipes(&self.static_modules, recipes)
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.construct_unit(address, context)
    }
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 1 || (group == 0 && forward.vision_state.is_some()),
            ForwardMode::Draft(depth) => group == depth + 2,
        }
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = vec![&forward.tokens, &forward.embedded];
        for part in &forward.parts {
            match part {
                PreparedPart::Text {
                    tokens, embeddings, ..
                } => values.extend([tokens, embeddings]),
                PreparedPart::Media { tokens } => values.push(tokens),
            }
        }
        values.extend(forward.mask.iter());
        values.extend(forward.vision_initial.iter());
        values.extend(forward.vision_output.iter());
        values.extend(forward.deepstack.iter());
        values.extend(forward.visual_mask.iter());
        values.extend(forward.target_hidden.iter());
        if let Some(state) = &forward.vision_state {
            values.extend(state.retained_values());
        }
        values.into_iter()
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.target_hidden()
    }
    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(Some(vec![
            forward.tokens.dim(0),
            forward.tokens.dim(1),
            self.parsed.text.hidden_size,
        ]))
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ComponentPinnedResources<B> for ConditionalKernel<B>
{
    type Modules = ConditionalStaticModules<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.static_modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.static_modules
    }
    fn visit_pinned<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        if let Some(shared) = &self.static_modules.prediction {
            visitor.visit("mtp", shared)?;
        }
        if self.static_modules.vision.has_parameters() {
            visitor.visit("vision", &self.static_modules.vision)?;
        }
        self.static_modules.text.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(shared) = &mut self.static_modules.prediction {
            visitor.visit_mut("mtp", shared)?;
        }
        if self.static_modules.vision.has_parameters() {
            visitor.visit_mut("vision", &mut self.static_modules.vision)?;
        }
        self.static_modules
            .text
            .decoder_resources_mut()
            .visit(visitor)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> ConditionalKernel<B> {
    /// Prepares one routed target decoder partition with architecture-owned
    /// shape validation and causal-mask construction.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_routed_target_partition(
        &mut self,
        input: ConditionalPartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        batch: i32,
        sequence: i32,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>, Error> {
        let mut prepared = self.begin_partition_target_inner(input, parallel, context)?;
        let expected = [batch, sequence, self.parsed.text.hidden_size];
        if prepared.hidden.shape() != expected {
            return Err(Error::backend(format!(
                "conditional Qwen3.5 decoder input is shaped {:?}, expected {expected:?}",
                prepared.hidden.shape()
            )));
        }
        prepared.mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => None,
        };
        Ok(prepared.into_layered_forward())
    }

    fn begin_partition_target_inner(
        &mut self,
        input: ConditionalPartitionInput<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelinePrepared<B::Tensor>, Error> {
        match input {
            ConditionalPartitionInput::Tokens { tokens, offset } => {
                let parts = [InputPart::Text(tokens)];
                let state = match parallel {
                    Some(parallel) => self.begin_pipeline_target_parallel(
                        &parts, None, None, offset, parallel, context,
                    ),
                    None => self.begin_pipeline_target(&parts, None, None, offset, context),
                }?;
                self.finish_pipeline_target(state, parallel, context)
            }
            ConditionalPartitionInput::Hidden { hidden, boundary } => {
                Ok(boundary.into_prepared(hidden))
            }
        }
    }

    /// Finishes the serial target output boundary.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_pipeline_logits(hidden, context)
    }

    /// Finishes the tensor-parallel target output boundary.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, true, parallel, context)
    }

    /// Embeds tokens for an adapter-owned prediction unit without entering the target graph.
    pub fn begin_partition_prediction_embedding(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.lookup_tokens(tokens, None, context)
    }

    /// Embeds tokens for an adapter-owned prediction unit with the selected TP layout.
    pub fn begin_partition_prediction_embedding_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_embed_parallel(tokens, parallel, context)
    }

    /// Projects an already-normalized prediction hidden value through the shared vocabulary head.
    pub fn finish_partition_prediction(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(
            hidden,
            false,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn project_prediction_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(hidden, false, parallel, context, instrumentation)
    }

    /// Projects an already-normalized prediction hidden value through the selected TP head.
    pub fn finish_partition_prediction_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, false, parallel, context)
    }

    /// Returns the normalized family configuration owned by this architecture.
    pub const fn args(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    /// Returns the architecture-owned vision, target, and prediction traversal.
    pub fn unit_layout(&self) -> Result<ExecutionUnitLayout, Error> {
        Ok(self.definition.parameters().unit_layout().clone())
    }

    /// Borrows canonical invocation, parameter, transport and state ownership.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    /// Returns the family-owned activation schema transported between stages.
    pub fn pipeline_boundary_schema(&self) -> ConditionalPipelineBoundarySchema {
        ConditionalPipelineBoundarySchema::for_config(&self.parsed)
            .expect("validated conditional definition")
    }

    /// Builds the exact configured vision/text/MTP graph.
    pub fn from_definition(
        parsed: ParsedHybridConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalLayeredModel<B>, Error> {
        Self::materialize(parsed, definition, None, None, context)
    }

    /// Applies the architecture-owned tensor-parallel token embedding boundary.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.lookup_tokens(tokens, Some(parallel), context)
    }

    /// Applies the architecture-owned tensor-parallel output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        normalize: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(
            hidden,
            normalize,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Builds the conditional graph with planner-derived local modules.
    pub fn from_parallel_definition(
        parsed: ParsedHybridConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: ConditionalLocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalLayeredModel<B>, Error> {
        Self::materialize(parsed, definition, Some(geometry), None, context)
    }

    /// Materializes exactly the pinned roots and unit factories owned by an authoritative partition.
    pub fn from_partition(
        parsed: ParsedHybridConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: ConditionalLocalGeometry,
        partition: &eredu_runtime::ArchitecturePartition<
            super::ConditionalPartitionLocalGeometry,
            ConditionalPipelineBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalLayeredModel<B>, Error> {
        Self::materialize(parsed, definition, Some(geometry), Some(partition), context)
    }

    fn materialize(
        parsed: ParsedHybridConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: Option<ConditionalLocalGeometry>,
        partition: Option<
            &eredu_runtime::ArchitecturePartition<
                super::ConditionalPartitionLocalGeometry,
                ConditionalPipelineBoundarySchema,
            >,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalLayeredModel<B>, Error> {
        let definition =
            super::select_conditional_definition(&parsed, definition, geometry.as_ref())?;
        let ownership = if let Some(partition) = partition {
            super::ConditionalPartitionLocalFoundation::from_partition(&parsed, partition)
                .map_err(Error::backend_source)?;
            partition
                .local_geometry()
                .validate_complete_geometry(
                    &parsed,
                    geometry.as_ref().ok_or_else(|| {
                        Error::backend("selected conditional Qwen has no complete geometry")
                    })?,
                )
                .map_err(Error::backend_source)?;
            Some(definition.select_partition(partition)?)
        } else {
            None
        };
        crate::operator_requirements::require::<B>(
            "Qwen hybrid conditional",
            crate::operator_requirements::QWEN_HYBRID.union(crate::operator_requirements::QWEN_VL),
        )?;
        if geometry.is_some() {
            crate::operator_requirements::require::<B>(
                "Qwen hybrid tensor parallelism",
                eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL,
            )?;
        }
        let prediction_steps = definition.prediction_groups().len();
        let text = PartitionStaticModules::<B>::from_selected_spec(
            super::static_module_spec(&parsed.text),
            geometry.as_ref().map(|geometry| {
                (
                    geometry.text().embedding_range().clone(),
                    geometry.text().output_range().cloned(),
                )
            }),
            |role| {
                ownership.as_ref().is_none_or(|selection| {
                    selection
                        .owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(role.into()))
                })
            },
            context,
        )?;
        let prediction = (prediction_steps > 0)
            .then(|| super::PredictionShared::new(&parsed.text, context))
            .transpose()?;
        let vision = parsed
            .vision
            .as_ref()
            .expect("validated conditional vision")
            .clone();
        let selection = match partition {
            Some(partition) => crate::qwen::vision::VisionStaticSelection::from_parameter_bindings(
                vision,
                "model.visual",
                eredu_runtime::ExecutionGroupId::new(VISION_EXECUTION_GROUP)
                    .map_err(Error::backend)?,
                "vision",
                partition.parameter_bindings(),
            ),
            None => crate::qwen::vision::VisionStaticSelection::all(vision, "model.visual"),
        }
        .map_err(Error::backend_source)?;
        let complete_widths;
        let widths = match &geometry {
            Some(geometry) => geometry.merger_widths(),
            None => {
                complete_widths = selection
                    .unsharded_merger_intermediates()
                    .map_err(Error::backend_source)?;
                &complete_widths
            }
        };
        let vision = VisionStatic::from_selection(selection, widths, context)?;
        let selected_state = partition.and_then(|partition| partition.state());
        let kernel = Self {
            parsed,
            definition: std::sync::Arc::clone(&definition),
            static_modules: ConditionalStaticModules {
                text,
                vision,
                prediction,
            },
            prediction_steps,
            owns_input: partition.is_none_or(|partition| partition.ownership().owns_input()),
            parallel_geometry: geometry.map(std::sync::Arc::new),
            partition_state_layout: selected_state.map(|state| state.layout().clone()),
            partition_state_offset: selected_state.map_or(0, |state| state.global_layer_offset()),
            expert_realization: None,
        };
        match partition {
            Some(partition) => {
                eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
                    definition, kernel, partition,
                )
            }
            None => eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel),
        }
        .map_err(Error::from)
    }

    /// Restricts a low-level state view independently of parameter ownership.
    /// Retained parameter ownership is independent of this state view.
    pub fn with_partition_state_layout(
        mut self,
        global_offset: usize,
        layout: StateLayout,
    ) -> Result<Self, Error> {
        if self.parallel_geometry.is_none() || layout.is_empty() {
            return Err(Error::backend(
                "conditional Qwen partition state requires parallel geometry and owned units",
            ));
        }
        let end = global_offset
            .checked_add(layout.len())
            .ok_or_else(|| Error::backend("conditional Qwen partition state range overflowed"))?;
        let expected = self
            .definition
            .state_layout()
            .slice(global_offset..end)
            .map_err(Error::backend)?;
        if layout != expected {
            return Err(Error::backend(
                "conditional Qwen partition state geometry mismatch",
            ));
        }
        if self
            .partition_state_layout
            .as_ref()
            .is_some_and(|retained| {
                retained != &layout || self.partition_state_offset != global_offset
            })
        {
            return Err(Error::backend(
                "conditional Qwen partition state selection is incompatible",
            ));
        }
        self.partition_state_layout = Some(layout);
        self.partition_state_offset = global_offset;
        Ok(self)
    }

    /// Normalized composite policy.
    pub const fn parsed(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    /// Returns the prediction depth count declared by the constructed execution graph.
    pub const fn mtp_len(&self) -> usize {
        self.prediction_steps
    }

    /// Shares planner-owned local geometry with placed unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<ConditionalLocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Retains exact local bank specifications without narrowing global routing.
    pub(crate) fn install_expert_realization(
        &mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) {
        self.expert_realization = Some(realization);
    }

    fn construct_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalUnit<B>, Error> {
        let (group, index) = (address.group(), address.index());
        match group {
            0 => {
                let vision = self.parsed.vision.as_ref().expect("validated vision");
                match self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.vision_block(index))
                {
                    Some((heads, intermediate)) => VisionBlock::new_parallel_with_root(
                        vision,
                        "model.visual",
                        index,
                        heads,
                        intermediate,
                        context,
                    ),
                    None => VisionBlock::new_with_root(vision, "model.visual", index, context),
                }
                .map(ConditionalUnit::Vision)
            }
            1 => {
                let config = self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.text().target(index))
                    .unwrap_or(&self.parsed.text);
                let spec = self
                    .expert_realization
                    .as_ref()
                    .map(|plan| {
                        plan.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, index)
                            .cloned()
                            .ok_or_else(|| {
                                Error::backend(
                                    "conditional Qwen partition has no selected expert unit",
                                )
                            })
                    })
                    .transpose()?;
                Block::new_with_routed_spec(config, index, spec, context)
                    .map(ConditionalUnit::Target)
            }
            _ => {
                let config = self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.text().prediction(group - 2))
                    .unwrap_or(&self.parsed.text);
                PredictionUnit::new(config, group - 2, context).map(ConditionalUnit::Prediction)
            }
        }
    }

    fn begin_request<S>(
        &mut self,
        input: ConditionalInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if parallel.is_some() && self.parallel_geometry.is_none() {
            return Err(Error::backend("conditional Qwen3.5 has no local geometry"));
        }
        let expected = self
            .partition_state_layout
            .as_ref()
            .unwrap_or_else(|| self.definition.state_layout());
        if state.layout() != expected {
            return Err(Error::backend("conditional Qwen3.5 state layout mismatch"));
        }
        match input {
            ConditionalInput::Prepared(input) => {
                self.begin_prepared(input, state, parallel, context)
            }
            ConditionalInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.prediction_steps {
                    return Err(Error::backend("conditional Qwen3.5 MTP depth is invalid"));
                }
                let embedded = self.lookup_tokens(tokens, parallel, context)?;
                let state_index = self.state_index(depth + 2, 0)?;
                let offset = state.layer(state_index).map_err(Error::backend)?.position();
                let mask = if tokens.dim(1) > 1 {
                    Some(B::causal_mask(tokens.dim(1), offset, None, context)?)
                } else {
                    None
                };
                Ok(LayeredForwardState {
                    hidden: hidden.clone(),
                    context: ConditionalForwardContext {
                        tokens: tokens.clone(),
                        embedded,
                        mask,
                        mode: ForwardMode::Draft(depth),
                        parts: Vec::new(),
                        vision_state: None,
                        vision_initial: None,
                        vision_output: None,
                        deepstack: Vec::new(),
                        visual_mask: None,
                        target_hidden: None,
                    },
                })
            }
            ConditionalInput::Target {
                parts,
                pixels,
                mask,
            } => {
                let (prepared, grids) = self.prepare_parts(parts, parallel, context)?;
                let has_media = !grids.is_empty();
                if has_media != pixels.is_some() {
                    return Err(Error::backend(
                        "conditional Qwen3.5 pixels and media metadata must appear together",
                    ));
                }
                let offset = state.layer(0).map_err(Error::backend)?.position();
                if has_media && offset != 0 {
                    return Err(Error::backend(
                        "conditional Qwen3.5 media cannot append to populated state",
                    ));
                }
                let (vision_initial, vision_state) = match pixels {
                    Some(pixels) => {
                        let (hidden, state) =
                            self.prepare_vision_request(pixels, &grids, context)?;
                        (Some(hidden), Some(state))
                    }
                    None => (None, None),
                };
                let assembled = self.assemble(&prepared, None, context);
                let sequence = prepared
                    .iter()
                    .map(|part| match part {
                        PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => {
                            tokens.dim(1)
                        }
                    })
                    .sum::<i32>();
                let decoder_mask = match mask {
                    Some(mask) => Some(mask.clone()),
                    None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
                    None => None,
                };
                let (tokens, embedded, error) = match assembled {
                    Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
                    Err(error) => (None, None, Some(error)),
                };
                let hidden = match vision_initial
                    .as_ref()
                    .cloned()
                    .or_else(|| embedded.clone())
                {
                    Some(hidden) => hidden,
                    None if !self.owns_input => B::Tensor::full_f32(
                        0.0,
                        &[
                            hidden_token_placeholder(&prepared).dim(0),
                            sequence,
                            self.parsed.text.hidden_size,
                        ],
                        context,
                    )?,
                    None => {
                        return Err(
                            error.unwrap_or_else(|| Error::backend("empty conditional input"))
                        )
                    }
                };
                let embedded = embedded.unwrap_or_else(|| hidden_embedding_placeholder(&prepared));
                let deepstack = (0..self
                    .parsed
                    .vision
                    .as_ref()
                    .expect("validated conditional vision")
                    .deepstack_layers()
                    .len())
                    .map(|_| embedded.zeros_like(context))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(LayeredForwardState {
                    hidden,
                    context: ConditionalForwardContext {
                        tokens: tokens.unwrap_or_else(|| hidden_token_placeholder(&prepared)),
                        embedded,
                        mask: decoder_mask,
                        mode: ForwardMode::Target,
                        parts: prepared,
                        vision_state,
                        vision_initial,
                        vision_output: None,
                        deepstack,
                        visual_mask: None,
                        target_hidden: None,
                    },
                })
            }
        }
    }

    fn lookup_tokens(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embedding = self
            .static_modules
            .text
            .embeddings
            .as_mut()
            .ok_or_else(|| {
                Error::backend("conditional Qwen does not own token embedding parameters")
            })?;
        match parallel {
            Some(parallel) => B::vocabulary_parallel_lookup(
                embedding,
                tokens,
                EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            ),
            None => embedding.forward(tokens, context),
        }
    }

    fn prepare_vision_request(
        &mut self,
        pixels: &B::Tensor,
        grids: &[(i32, i32, i32)],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        if self.static_modules.vision.owns_ingress() {
            return self.static_modules.vision.begin(
                VisionInput {
                    pixels,
                    grid: grids,
                },
                context,
            );
        }
        let config = self.parsed.vision.as_ref().expect("validated vision");
        eredu_nn::sequence_layout::validate_patch_grid(
            grids,
            config.spatial_merge_size,
            Some(pixels.dim(0)),
        )
        .map_err(Error::backend)?;
        let state = self.static_modules.vision.prepare_state(grids, context)?;
        let hidden = B::Tensor::full_f32(0.0, &[pixels.dim(0), config.hidden_size], context)?;
        Ok((hidden, state))
    }

    fn prepare_parts(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        if parts.is_empty() {
            return Err(Error::backend("conditional Qwen3.5 input has no parts"));
        }
        let mut grids = Vec::new();
        let prepared = parts
            .iter()
            .map(|part| match part {
                InputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: if self.owns_input {
                        self.lookup_tokens(tokens, parallel, context)?
                    } else {
                        B::Tensor::full_f32(
                            0.0,
                            &[tokens.dim(0), tokens.dim(1), self.parsed.text.hidden_size],
                            context,
                        )?
                    },
                    ready: self.owns_input,
                }),
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.parsed.text.hidden_size]
                    {
                        return Err(Error::backend(
                            "conditional Qwen3.5 projected input geometry mismatch",
                        ));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                        ready: true,
                    })
                }
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok((prepared, grids))
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        let media_tokens = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Media { tokens } => tokens.dim(1),
                PreparedPart::Text { .. } => 0,
            })
            .sum::<i32>();
        let valid_vision_shape = match vision {
            Some(value) => value.shape() == [1, media_tokens, self.parsed.text.hidden_size],
            None => media_tokens == 0,
        };
        if !valid_vision_shape {
            return Err(Error::backend(format!(
                "conditional Qwen3.5 media placeholders require [1, {media_tokens}, {}], got {:?}",
                self.parsed.text.hidden_size,
                vision.map(|value| value.shape()),
            )));
        }
        let mut offset = 0;
        let mut embeddings = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value,
                    ready,
                    ..
                } => {
                    if !ready {
                        return Err(Error::backend(
                            "conditional Qwen text embeddings await their input owner",
                        ));
                    }
                    embeddings.push(value.clone());
                }
                PreparedPart::Media { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(vision.expect("validated vision output").index(
                        &[
                            Index::Full,
                            Index::Range(offset, offset + length),
                            Index::Full,
                        ],
                        context,
                    )?);
                    offset += length;
                }
            }
        }
        let ordered = parts
            .iter()
            .zip(&embeddings)
            .map(|(part, embeddings)| OrderedInputPart {
                token_ids: match part {
                    PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens,
                },
                embeddings,
            })
            .collect::<Vec<_>>();
        assemble_ordered_inputs(&ordered, self.parsed.text.hidden_size, context)
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        let state = &self.definition.unit(group, index)?.state;
        if state.len() != 1 {
            return Err(Error::backend(
                "conditional Qwen unit has no mutable text state",
            ));
        }
        let local = state
            .start
            .checked_sub(self.partition_state_offset)
            .ok_or_else(|| {
                Error::backend("conditional Qwen state precedes its selected PP offset")
            })?;
        if self
            .partition_state_layout
            .as_ref()
            .is_some_and(|layout| local >= layout.len())
        {
            return Err(Error::backend(
                "conditional Qwen state follows its selected PP range",
            ));
        }
        Ok(local)
    }

    /// Starts one pipeline target request before any placed vision block runs.
    pub fn begin_pipeline_target(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        pixels: Option<&B::Tensor>,
        mask: Option<&B::Tensor>,
        offset: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_target_inner(parts, pixels, mask, offset, None, context)
    }

    /// Starts the same target request with rank-local vocabulary lookup.
    pub fn begin_pipeline_target_parallel(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        pixels: Option<&B::Tensor>,
        mask: Option<&B::Tensor>,
        offset: i32,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_target_inner(parts, pixels, mask, offset, Some(parallel), context)
    }

    fn begin_pipeline_target_inner(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        pixels: Option<&B::Tensor>,
        mask: Option<&B::Tensor>,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelineVisionState<B::Tensor>, Error> {
        let (parts, grids) = self.prepare_parts(parts, parallel, context)?;
        let has_media = !grids.is_empty();
        if has_media != pixels.is_some() {
            return Err(Error::backend(
                "conditional Qwen3.5 pixels and media metadata must appear together",
            ));
        }
        if has_media && offset != 0 {
            return Err(Error::backend(
                "conditional Qwen3.5 media cannot append to populated state",
            ));
        }
        let (vision_initial, vision) = match pixels {
            Some(pixels) => {
                let (hidden, state) = self.prepare_vision_request(pixels, &grids, context)?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => None,
        };
        let assembled = self.assemble(&parts, None, context);
        let hidden = match vision_initial {
            Some(hidden) => hidden,
            None => assembled?.embeddings,
        };
        Ok(ConditionalPipelineVisionState {
            hidden,
            parts,
            mask,
            vision,
            vision_output: None,
            deepstack: Vec::new(),
        })
    }

    /// Whether the request has shared-vision work.
    pub fn pipeline_vision_active(state: &ConditionalPipelineVisionState<B::Tensor>) -> bool {
        state.vision.is_some()
    }

    /// Exports the tensors required by the next placed vision owner.
    pub fn pipeline_retained_values(
        state: &ConditionalPipelineVisionState<B::Tensor>,
    ) -> Vec<B::Tensor> {
        let mut values = vec![state.hidden.clone()];
        values.extend(state.mask.iter().cloned());
        for part in &state.parts {
            match part {
                PreparedPart::Text {
                    tokens, embeddings, ..
                } => {
                    values.extend([tokens.clone(), embeddings.clone()]);
                }
                PreparedPart::Media { tokens } => values.push(tokens.clone()),
            }
        }
        if let Some(vision) = &state.vision {
            values.extend(vision.retained_values().cloned());
        }
        values
    }

    /// Installs values transported from the previous placed vision owner.
    pub fn replace_pipeline_retained_values(
        state: &mut ConditionalPipelineVisionState<B::Tensor>,
        values: Vec<B::Tensor>,
    ) -> Result<(), Error> {
        let expected = Self::pipeline_retained_values(state).len();
        if values.len() != expected {
            return Err(Error::backend(format!(
                "conditional Qwen3.5 pipeline continuation received {} tensors, expected {expected}",
                values.len()
            )));
        }
        let mut values = values.into_iter();
        state.hidden = values.next().expect("validated hidden");
        if state.mask.is_some() {
            state.mask = Some(values.next().expect("validated mask"));
        }
        for part in &mut state.parts {
            match part {
                PreparedPart::Text {
                    tokens,
                    embeddings,
                    ready,
                } => {
                    *ready = true;
                    *tokens = values.next().expect("validated tokens");
                    *embeddings = values.next().expect("validated embeddings");
                }
                PreparedPart::Media { tokens } => {
                    *tokens = values.next().expect("validated media tokens");
                }
            }
        }
        if let Some(vision) = &mut state.vision {
            vision.replace_retained_values(values.collect())?;
        }
        Ok(())
    }

    /// Executes one placed shared-vision block.
    pub fn forward_pipeline_vision(
        &mut self,
        index: usize,
        block: &mut VisionBlock<B>,
        state: &mut ConditionalPipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        let vision = state
            .vision
            .as_mut()
            .ok_or_else(|| Error::backend("missing conditional pipeline vision state"))?;
        state.hidden = match parallel {
            Some(parallel) => self.static_modules.vision.forward_block_parallel(
                block,
                index,
                &state.hidden,
                vision,
                parallel,
                context,
            )?,
            None => self.static_modules.vision.forward_block(
                block,
                index,
                &state.hidden,
                vision,
                context,
            )?,
        };
        Ok(())
    }

    /// Finishes the placed vision projector without assembling decoder ingress.
    pub fn complete_pipeline_vision(
        &mut self,
        state: &mut ConditionalPipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        if state.vision_output.is_some() || state.vision.is_none() {
            return Ok(());
        }
        let output = match &mut state.vision {
            Some(vision) => Some(match parallel {
                Some(parallel) => self.static_modules.vision.finish_parallel(
                    &state.hidden,
                    vision,
                    parallel,
                    context,
                )?,
                None => self
                    .static_modules
                    .vision
                    .finish(&state.hidden, vision, context)?,
            }),
            None => None,
        };
        if let Some(output) = output {
            state.vision_output = Some(output.embeddings);
            state.deepstack = output.deepstack_features;
        }
        Ok(())
    }

    /// Returns the completed placed projector output.
    pub fn pipeline_vision_output(
        state: &ConditionalPipelineVisionState<B::Tensor>,
    ) -> Option<&B::Tensor> {
        state.vision_output.as_ref()
    }

    /// Replaces the completed placed projector output before decoder assembly.
    pub fn replace_pipeline_vision_output(
        state: &mut ConditionalPipelineVisionState<B::Tensor>,
        output: B::Tensor,
    ) {
        state.vision_output = Some(output);
    }

    /// Finishes the shared merger and emits a fixed-shape decoder payload.
    pub fn finish_pipeline_target(
        &mut self,
        mut state: ConditionalPipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelinePrepared<B::Tensor>, Error> {
        self.complete_pipeline_vision(&mut state, parallel, context)?;
        let assembled = self.assemble(&state.parts, state.vision_output.as_ref(), context)?;
        let deepstack = if state.vision_output.is_some() {
            let image = self.parsed.image_token_id.expect("validated image token");
            let video = self.parsed.video_token_id.expect("validated video token");
            let visual = assembled
                .token_ids
                .equal_i32(image, context)?
                .logical_or(&assembled.token_ids.equal_i32(video, context)?, context)?;
            state
                .deepstack
                .into_iter()
                .map(|features| {
                    assembled.embeddings.zeros_like(context)?.masked_scatter(
                        &visual,
                        &features.index(&[Index::At(0), Index::Full, Index::Full], context)?,
                        context,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?
        } else {
            (0..self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .deepstack_layer_count())
                .map(|_| assembled.embeddings.zeros_like(context))
                .collect::<Result<Vec<_>, Error>>()?
        };
        Ok(ConditionalPipelinePrepared {
            hidden: assembled.embeddings,
            mask: state.mask,
            deepstack,
        })
    }

    fn finish_readout(
        &mut self,
        hidden: &B::Tensor,
        normalize: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let resources = self.static_modules.text.decoder_resources_mut();
        if normalize {
            resources.finish_parallel(hidden, parallel, context, instrumentation)
        } else {
            resources.project(hidden, parallel, context, instrumentation)
        }
    }

    /// Applies the conditional target normalization and vocabulary projection.
    pub fn finish_pipeline_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(
            hidden,
            true,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Executes one text unit through a runtime-owned routed-expert provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut ConditionalUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ConditionalForwardContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let state_index = self.state_index(group, index)?;
        let output = match unit {
            ConditionalUnit::Target(block) if group == 1 => block.forward_with_provider(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            )?,
            ConditionalUnit::Prediction(unit) if group >= 2 => unit.forward_with_provider(
                self.static_modules
                    .prediction
                    .as_mut()
                    .ok_or_else(|| Error::backend("Qwen prediction has no shared modules"))?,
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            )?,
            _ => return Err(Error::backend("conditional Qwen3.5 unit/group mismatch")),
        };
        self.add_deepstack(
            group,
            index,
            output,
            forward,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    fn deepstack_contribution(
        hidden: &B::Tensor,
        features: &B::Tensor,
        visual_mask: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if features.shape() == hidden.shape() {
            return Ok(features.clone());
        }
        let mask = visual_mask.ok_or_else(|| Error::backend("missing conditional visual mask"))?;
        let source = features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
        hidden
            .zeros_like(context)?
            .masked_scatter(mask, &source, context)
    }

    /// Adds the exact prepared vision contribution at its target-layer boundary.
    /// The disabled path uses the same borrowed feature value as ordinary execution.
    fn add_deepstack(
        &self,
        group: usize,
        index: usize,
        output: B::Tensor,
        forward: &ConditionalForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let Some(features) = (group == 1).then(|| forward.deepstack.get(index)).flatten() else {
            return Ok(output);
        };
        instrumentation.observe("deepstack.input", &output)?;
        let contribution =
            Self::deepstack_contribution(&output, features, forward.visual_mask.as_ref(), context)?;
        let contribution = instrumentation.apply("deepstack.output", contribution)?;
        let output = output.add(&contribution, context)?;
        instrumentation.apply("deepstack.residual", output)
    }

    /// Executes one text unit with local projections and required collectives.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider_parallel<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut ConditionalUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ConditionalForwardContext<B::Tensor>,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let state_index = self.state_index(group, index)?;
        let output = match unit {
            ConditionalUnit::Target(block) if group == 1 => block.forward_parallel(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            )?,
            ConditionalUnit::Prediction(unit) if group >= 2 => unit.forward_parallel(
                self.static_modules
                    .prediction
                    .as_mut()
                    .ok_or_else(|| Error::backend("Qwen prediction has no shared modules"))?,
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            )?,
            _ => {
                return Err(Error::backend(
                    "conditional Qwen3.5 parallel unit/group mismatch",
                ))
            }
        };
        self.add_deepstack(
            group,
            index,
            output,
            forward,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

impl<B, S> eredu_runtime::ComponentEquations<B, S> for ConditionalKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(target, target, target)
    }
    fn execute_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_target_observed(
            address,
            unit,
            hidden,
            state,
            forward,
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
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if matches!(forward.mode, ForwardMode::Target) {
            self.finish_target_observed(hidden, None, context, observer)
        } else {
            self.finish(hidden, state, forward, context)
        }
    }

    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Error> {
        Ok((group == 1).then(|| eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.to_owned()))
    }

    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Error> {
        Ok((group == 0).then(|| eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH.to_owned()))
    }

    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_request(input, state, None, context)
    }

    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match group {
            0 => Ok(forward.vision_initial.as_ref().unwrap_or(initial).clone()),
            1 if matches!(forward.mode, ForwardMode::Target) => {
                let vision = forward
                    .vision_output
                    .as_ref()
                    .and_then(|_| dependencies.first().copied())
                    .or(forward.vision_output.as_ref());
                let assembled = self.assemble(&forward.parts, vision, context)?;
                let image = self.parsed.image_token_id.expect("validated image token");
                let video = self.parsed.video_token_id.expect("validated video token");
                forward.visual_mask = if forward.deepstack.is_empty() {
                    None
                } else {
                    Some(
                        assembled
                            .token_ids
                            .equal_i32(image, context)?
                            .logical_or(&assembled.token_ids.equal_i32(video, context)?, context)?,
                    )
                };
                forward.tokens = assembled.token_ids;
                forward.embedded = assembled.embeddings.clone();
                Ok(assembled.embeddings)
            }
            _ => Ok(dependencies.first().copied().unwrap_or(initial).clone()),
        }
    }

    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (group, index) = (address.group(), address.index());
        if group == 0 {
            let ConditionalUnit::Vision(block) = unit else {
                return Err(Error::backend("conditional vision unit/group mismatch"));
            };
            self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing conditional vision state"))?,
                context,
            )
        } else {
            self.forward_unit_with_provider(
                group,
                index,
                unit,
                hidden,
                state,
                forward,
                &mut ResidentExpertProvider,
                context,
            )
        }
    }

    fn complete_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 0 {
            if let Some(vision_state) = forward.vision_state.as_mut() {
                let output = self
                    .static_modules
                    .vision
                    .finish(hidden, vision_state, context)?;
                forward.deepstack = output.deepstack_features;
                forward.vision_output = Some(output.embeddings);
                return Ok(forward.vision_output.as_ref().unwrap().clone());
            }
        }
        if group == 1 || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 2) {
            forward.target_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
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
        self.finish(
            &crate::decoder::final_hidden_position(hidden, context)?,
            state,
            forward,
            context,
        )
    }

    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(
            hidden,
            matches!(forward.mode, ForwardMode::Target),
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for ConditionalKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(target, target, target)
    }
    fn execute_parallel_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_target_parallel_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            &mut ResidentExpertProvider,
            parallel,
            context,
            observer,
        )
    }
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if matches!(forward.mode, ForwardMode::Target) {
            self.finish_target_observed(hidden, Some(parallel), context, observer)
        } else {
            self.finish_parallel(hidden, state, forward, parallel, context)
        }
    }

    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_request(input, state, Some(parallel), context)
    }

    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (group, index) = (address.group(), address.index());
        if group == 0 {
            let ConditionalUnit::Vision(block) = unit else {
                return Err(Error::backend("conditional vision unit/group mismatch"));
            };
            self.static_modules.vision.forward_block_parallel(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing conditional vision state"))?,
                parallel,
                context,
            )
        } else {
            self.forward_unit_with_provider_parallel(
                group,
                index,
                unit,
                hidden,
                state,
                forward,
                &mut ResidentExpertProvider,
                parallel,
                context,
            )
        }
    }

    fn complete_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 0 {
            if let Some(vision_state) = forward.vision_state.as_mut() {
                let output = self.static_modules.vision.finish_parallel(
                    hidden,
                    vision_state,
                    parallel,
                    context,
                )?;
                forward.deepstack = output.deepstack_features;
                forward.vision_output = Some(output.embeddings);
                return Ok(forward.vision_output.as_ref().unwrap().clone());
            }
        }
        if group == 1 || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 2) {
            forward.target_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend("conditional Qwen3.5 has no local geometry"));
        }
        self.finish_readout(
            hidden,
            matches!(forward.mode, ForwardMode::Target),
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for ConditionalKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        let target = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(target, target, target)
    }
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, ConditionalPipelineBoundary<B::Tensor>>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let output = match parallel {
            Some(parallel) => {
                self.finish_parallel_observed(hidden, state, forward, parallel, context, observer)?
            }
            None => self.finish_observed(hidden, state, forward, context, observer)?,
        };
        Ok(LayeredPartitionOutput::Final {
            output,
            retained: Some(hidden.clone()),
        })
    }

    type Boundary = ConditionalPipelineBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(self.pipeline_boundary_schema())
    }

    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let (input, batch, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => (
                ConditionalPartitionInput::Tokens {
                    tokens,
                    offset: state.layer(0).map_err(Error::backend)?.position(),
                },
                tokens.dim(0),
                tokens.dim(1),
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                let batch = hidden.dim(0);
                let sequence = hidden.dim(1);
                (
                    ConditionalPartitionInput::Hidden {
                        hidden,
                        boundary: auxiliary,
                    },
                    batch,
                    sequence,
                )
            }
        };
        let offset = state.layer(0).map_err(Error::backend)?.position();
        self.begin_routed_target_partition(
            input,
            mask,
            batch,
            sequence,
            offset,
            selected.parallel,
            selected.native,
        )
    }

    fn enter_partition_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(initial.clone())
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        LayeredPartitionOutput<B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
        Self::Error,
    > {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: Some(hidden.clone()),
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: ConditionalPipelineBoundary {
                    deepstack: forward
                        .deepstack
                        .iter()
                        .map(|features| {
                            Self::deepstack_contribution(
                                hidden,
                                features,
                                forward.visual_mask.as_ref(),
                                context,
                            )
                        })
                        .collect::<Result<_, _>>()?,
                },
            })
        }
    }
}

fn hidden_token_placeholder<T: Clone>(parts: &[PreparedPart<T>]) -> T {
    match &parts[0] {
        PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.clone(),
    }
}

fn hidden_embedding_placeholder<T: Clone>(parts: &[PreparedPart<T>]) -> T {
    parts
        .iter()
        .find_map(|part| match part {
            PreparedPart::Text { embeddings, .. } => Some(embeddings.clone()),
            PreparedPart::Media { .. } => None,
        })
        .unwrap_or_else(|| hidden_token_placeholder(parts))
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> ConditionalKernel<B> {
    fn begin_prepared<'a, S>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, QwenInputPartPlan>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        prepare_input(input, ProjectedTokenPolicy::Placeholder(0), context)?.with_parts(
            |parts, pixels| {
                let input = ConditionalInput::Target {
                    parts,
                    pixels,
                    mask: None,
                };
                self.begin_request(input, state, parallel, context)
            },
        )
    }
}
