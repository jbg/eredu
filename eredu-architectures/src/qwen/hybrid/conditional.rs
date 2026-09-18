//! Conditional-generation graph over the shared Qwen vision tower and hybrid decoder.

use super::HybridConfig;

mod construction;
pub(crate) use construction::RetainedConditionalUnits;

mod media_prefill;
mod observation;

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend, Index,
    LinearOperator, NormalizationOperator, Parameterized, Tensor,
    multimodal::{OrderedInputPart, assemble_ordered_inputs},
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, LayerRuntimeState,
    LayeredArchitecture, LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput,
    OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture,
    PartitionedLayeredArchitecture, ResidentExpertProvider, RoutedExpertProvider,
    RoutedLayeredArchitecture, RuntimeStateComponents, StateLayout,
};

use crate::qwen::vision::{VisionBlock, VisionInput, VisionMode, VisionState, VisionStatic};
use crate::qwen::vl::InputPart;
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    media_plan::QwenHybridInputPartPlan,
};

use super::{
    Block, ConditionalLocalGeometry, ForwardMode, ParsedHybridConfig, PredictionUnit, Unit,
};

/// Stable execution-group identity for conditional Qwen vision ingress.
pub const VISION_EXECUTION_GROUP: &str = "vision";

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Media { tokens: T },
}

/// Pinned hybrid text and shared-vision modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ConditionalStaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Hybrid token embedding, final normalization, and vocabulary head.
    pub text: crate::decoder::StaticModules<B>,
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

impl<B, S> RoutedLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
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
    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_target_observed(
            group, index, unit, hidden, state, forward, provider, context, observer,
        )
    }

    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match group {
            0 => <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            ),
            _ => ConditionalLayeredModel::forward_unit_with_provider(
                self, group, index, unit, hidden, state, forward, provider, context,
            ),
        }
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
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
    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_target_parallel_observed(
            group, index, unit, hidden, state, forward, provider, parallel, context, observer,
        )
    }

    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match group {
            0 => <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            ),
            _ => ConditionalLayeredModel::forward_unit_with_provider_parallel(
                self, group, index, unit, hidden, state, forward, provider, parallel, context,
            ),
        }
    }
}

/// Target media input or one embedded prediction request.
pub enum ConditionalInput<'a, T> {
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

enum PreparedInputKind {
    Text(usize),
    Projected(usize, usize),
    Image(usize, usize),
    Video(usize, usize),
}

/// Architecture-owned tensor assembly for one admitted conditional Qwen request.
pub struct PreparedInput<T> {
    tokens: Vec<T>,
    grids: Vec<Vec<(i32, i32, i32)>>,
    pixels: Option<T>,
    kinds: Vec<PreparedInputKind>,
    projected: Vec<Option<T>>,
}

// The shared worker has no public forwarding capability. Only its complete
// grid-retaining result constructs PreparedInput; retained spans use the smaller
// pending type and borrow semantic grids from their admitted source.
struct PreparedInputAssembly<T> {
    tokens: Vec<T>,
    grids: Vec<Vec<(i32, i32, i32)>>,
    pixels: Option<T>,
    kinds: Vec<PreparedInputKind>,
    projected: Vec<Option<T>>,
}
struct PreparedPendingInput<T> {
    tokens: Vec<T>,
    pixels: Option<T>,
    kinds: Vec<PreparedInputKind>,
    projected: Vec<Option<T>>,
}

impl<T: Tensor> PreparedInput<T> {
    /// Borrows the assembled request through the canonical target vocabulary.
    pub fn with_target_input<R>(&self, apply: impl FnOnce(ConditionalInput<'_, T>) -> R) -> R {
        self.with_target_input_destination(crate::decoder::identity::Metadata::new(None), apply)
            .expect("ordinary borrowed input construction is infallible")
    }

    fn with_target_input_destination<R>(
        &self,
        metadata: crate::decoder::identity::Metadata<'_>,
        apply: impl FnOnce(ConditionalInput<'_, T>) -> R,
    ) -> Result<R, Error> {
        metadata.controls::<(Vec<InputPart<'_, T>>, R)>()?;
        if let Some(context) = metadata.context() {
            context.charge_metadata(std::mem::size_of_val(&apply))?;
        }
        let mut parts = metadata.vector(self.kinds.len())?;
        parts.extend(self.kinds.iter().map(|kind| {
            match *kind {
                PreparedInputKind::Text(token) => InputPart::Text(&self.tokens[token]),
                PreparedInputKind::Projected(token, original) => InputPart::Projected {
                    tokens: &self.tokens[token],
                    embeddings: self.projected[original]
                        .as_ref()
                        .expect("projected input retains its embeddings"),
                },
                PreparedInputKind::Image(token, grid) => InputPart::Image {
                    tokens: &self.tokens[token],
                    grid: &self.grids[grid],
                },
                PreparedInputKind::Video(token, grid) => InputPart::Video {
                    tokens: &self.tokens[token],
                    grid: &self.grids[grid],
                },
            }
        }));
        Ok(apply(ConditionalInput::Target {
            parts: &parts,
            pixels: self.pixels.as_ref(),
            mask: None,
        }))
    }

    /// Concatenates semantic token identity in decoder order.
    pub fn token_ids(&self, context: &T::Context) -> Result<T, Error> {
        T::concatenate(&self.tokens, 1, context)
    }
}

fn visit_semantic_tokens<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>,
    visitor: &mut dyn FnMut(crate::composite_execution::PredictionTokenPart<'_, T>) -> Result<(), Error>,
) -> Result<(), Error> {
    use crate::composite_execution::PredictionTokenPart;
    let metadata = crate::decoder::identity::Metadata::new(input.metadata());
    for (part, plan) in input.prepared().parts().iter().zip(input.qwen_parts()) {
        let token = if plan.role == crate::media_plan::qwen::QwenPartRole::Tokens {
            let eredu_runtime::PreparedInputPayload::TokenIds(value) = part.payload() else {
                return Err(metadata.error(format_args!("conditional Qwen text admission lost token payload")));
            };
            PredictionTokenPart::Tokens(value)
        } else {
            PredictionTokenPart::Repeated {
                token: if plan.role == crate::media_plan::qwen::QwenPartRole::Projected { 0 } else { plan.placeholder },
                positions: plan.positions,
            }
        };
        visitor(token)?;
    }
    Ok(())
}
fn prepared_semantic_tokens<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>,
    context: &T::Context,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<Vec<T>, Error> {
    metadata.controls::<(Vec<T>, T, [i32; 2])>()?;
    let mut tokens = metadata.vector(input.prepared().len())?;
    visit_semantic_tokens(input, &mut |part| {
        tokens.push(crate::composite_execution::prediction_tokens::materialize(part, context, metadata)?);
        Ok(())
    })?;
    Ok(tokens)
}

/// Materializes Qwen placeholder IDs, patch grids, and ordered target segments
/// from an architecture admission.
pub fn prepare_input<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>,
    context: &T::Context,
) -> Result<PreparedInput<T>, Error> {
    prepare_input_with_metadata(input, context, None)
}

fn prepare_input_with_metadata<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>,
    context: &T::Context,
    metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<PreparedInput<T>, Error> {
    let metadata = crate::decoder::identity::Metadata::new(metadata);
    metadata.controls::<(PreparedInput<T>, PreparedInputAssembly<T>)>()?;
    let assembled = assemble_input(input, true, context, metadata)?;
    Ok(PreparedInput {
        tokens: assembled.tokens,
        grids: assembled.grids,
        pixels: assembled.pixels,
        kinds: assembled.kinds,
        projected: assembled.projected,
    })
}

fn prepare_pending_input<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>,
    context: &T::Context,
    metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<PreparedPendingInput<T>, Error> {
    let metadata = crate::decoder::identity::Metadata::new(metadata);
    metadata.controls::<(PreparedPendingInput<T>, PreparedInputAssembly<T>)>()?;
    let assembled = assemble_input(input, false, context, metadata)?;
    debug_assert!(assembled.grids.is_empty());
    Ok(PreparedPendingInput {
        tokens: assembled.tokens,
        pixels: assembled.pixels,
        kinds: assembled.kinds,
        projected: assembled.projected,
    })
}

fn assemble_input<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenHybridInputPartPlan>,
    retain_grids: bool,
    context: &T::Context,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<PreparedInputAssembly<T>, Error> {
    metadata.controls::<(PreparedInputAssembly<T>, PreparedInputKind, [i32; 2])>()?;
    let prepared = input.prepared();
    if let Some(admitted) = input.admitted().ordinary() {
        if prepared.identity() != admitted.identity() || prepared.len() != admitted.parts().len() {
            return Err(metadata.error(format_args!(
                "conditional Qwen prepared input no longer matches its admission"
            )));
        }
    }
    let tokens = prepared_semantic_tokens(input, context, metadata)?;
    let media_count = input
        .qwen_parts()
        .filter(|part| part.role == crate::media_plan::qwen::QwenPartRole::Encoded)
        .count();
    let mut grids = metadata.vector(if retain_grids { media_count } else { 0 })?;
    let mut grid_count = 0usize;
    let mut pixels = metadata.vector(media_count)?;
    let mut kinds = metadata.vector(prepared.len())?;
    let mut projected = metadata.vector(prepared.len())?;
    for (original, (part, plan)) in prepared.parts().iter().zip(input.qwen_parts()).enumerate() {
        match plan.role {
            crate::media_plan::qwen::QwenPartRole::Tokens => {
                kinds.push(PreparedInputKind::Text(original));
                projected.push(None);
            }
            crate::media_plan::qwen::QwenPartRole::Projected => {
                let eredu_runtime::PreparedInputPayload::Embeddings(value) = part.payload() else {
                    return Err(metadata.error(format_args!(
                        "conditional Qwen projected part lost its embeddings"
                    )));
                };

                kinds.push(PreparedInputKind::Projected(original, original));
                projected.push(Some(value.clone()));
            }
            crate::media_plan::qwen::QwenPartRole::Encoded => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(metadata.error(format_args!(
                        "conditional Qwen admitted media lost its tensor payload"
                    )));
                };

                if retain_grids {
                    let mut grid = metadata.vector(plan.grid.iter().count())?;
                    grid.extend(plan.grid.iter());
                    grids.push(grid);
                }
                pixels.push(value.clone());
                let token_index = original;
                let grid_index = grid_count;
                grid_count += 1;
                kinds.push(match part.modality() {
                    eredu_core::InputModality::Image => {
                        PreparedInputKind::Image(token_index, grid_index)
                    }
                    eredu_core::InputModality::Video => {
                        PreparedInputKind::Video(token_index, grid_index)
                    }
                    _ => {
                        return Err(metadata.error(format_args!(
                            "conditional Qwen admission contains an unsupported modality"
                        )));
                    }
                });
                projected.push(None);
            }
        }
    }
    let pixels = match pixels.len() {
        0 => None,
        1 => pixels.pop(),
        _ => Some(T::concatenate(&pixels, 0, context)?),
    };
    Ok(PreparedInputAssembly {
        tokens,
        grids,
        pixels,
        kinds,
        projected,
    })
}

impl<B, S> CompositeArchitecture<B, S> for ConditionalLayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type InputPartPlan = QwenHybridInputPartPlan;
    type AdmissionConfig = crate::replicated_text::SharedCompositeConfig<ParsedHybridConfig>;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.parsed.clone()
    }

    fn retain_admission_config_with_metadata(
        config: &Self::AdmissionConfig,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::AdmissionConfig, Error> {
        context.charge_metadata(std::mem::size_of::<(
            Self::AdmissionConfig,
            Result<Self::AdmissionConfig, Error>,
        )>())?;
        // This aliases the selected immutable owner, including its custody.
        Ok(config.clone())
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

    fn admit_prepared_input_with_metadata(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>, Error> {
        crate::media_plan::admission::qwen_hybrid(config, input, inspector, context)
    }

    fn visit_prepared_prediction_tokens(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        visitor: &mut dyn FnMut(crate::composite_execution::PredictionTokenPart<'_, B::Tensor>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        visit_semantic_tokens(input, visitor)
    }

    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        group == 1
            || (group == 0
                && input
                    .qwen_parts()
                    .any(|part| part.role == crate::media_plan::qwen::QwenPartRole::Encoded))
    }

    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group == 0 {
            input
                .qwen_parts()
                .filter_map(|part| {
                    (part.role == crate::media_plan::qwen::QwenPartRole::Encoded)
                        .then_some(part.positions)
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
    ) -> Result<Option<(i32, i32)>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = input
            .qwen_parts()
            .filter(|part| part.role == crate::media_plan::qwen::QwenPartRole::Encoded)
            .flat_map(|part| part.grid.iter())
            .try_fold(0_i64, |total, (time, height, width)| {
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
            .parsed
            .vision
            .as_ref()
            .ok_or_else(|| "conditional Qwen continuation has no vision config".to_owned())?
            .hidden_size;
        Ok(Some((patches, width)))
    }

    fn prepared_group_continuation_batched(&self, group: usize) -> bool {
        group != 0
    }

    fn prepared_group_collective_waves(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        pipeline_stages: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, Error>
    {
        let destination=crate::composite_execution::graph::Destination(context);
        destination.controls::<(&Self,usize,PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,usize,usize)>()?;
        self.ingress_group_collective_waves(group, input, tensor_partitions, pipeline_stages, true, destination)
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
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Self::Error> {
        let destination = crate::composite_execution::graph::Destination(metadata);
        destination.controls::<(&Self, usize, usize, &eredu_runtime::ResolvedBoundaryWireSchema,
            i32, i32, &[i32], Option<(i32, i32)>, eredu_runtime::BoundaryTensorSpec,
            eredu_runtime::BoundaryWireSchema, Vec<eredu_runtime::BoundaryTensorSpec>, Vec<i32>,
            Option<eredu_runtime::ResolvedBoundaryWireSchema>)>()?;
        let vision_edge = source_group == 0 && matches!(destination_group, 0 | 1);
        let decoder_continuation = source_group == 1 && destination_group == 1;
        if !vision_edge && !decoder_continuation {
            return Ok(None);
        }
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        let vision_continuation = source_group == 0 && destination_group == 0;
        let vision = self
            .parsed
            .vision
            .as_ref()
            .ok_or_else(|| destination.error(format_args!("conditional Qwen vision boundary has no config")))?;
        let primary = if vision_continuation {
            destination.boundary_spec("hidden", &[Dim::Sequence, Dim::Fixed(vision.hidden_size)], Dtype::Activation)?
        } else {
            destination.boundary_spec("hidden", &[Dim::Batch, Dim::Sequence,
                Dim::Fixed(self.parsed.text.hidden_size)], Dtype::Activation)?
        };
        let mut auxiliary = destination.vector(selected.auxiliary().len())?;
        for index in 0..selected.auxiliary().len() {
            let role = destination.text(format_args!("deepstack.{index}"))?;
            auxiliary.push(destination.boundary_spec(&role, &[Dim::Batch, Dim::Sequence,
                Dim::Fixed(self.parsed.text.hidden_size)], Dtype::Activation)?);
        }
        let schema = destination.boundary_schema(
            if vision_continuation { "qwen_conditional.vision_continuation" }
            else if decoder_continuation { "qwen_conditional.decoder" }
            else { "qwen_conditional.vision_to_decoder" },
            primary, auxiliary,
        )?;
        let primary_sequence = if vision_continuation {
            continuation
                .ok_or_else(|| {
                    destination.error(format_args!("conditional Qwen continuation has no patch geometry"))
                })?
                .0
        } else {
            source_sequence
        };
        let vision_sequence = *group_sequences.first().ok_or_else(|| {
            destination.error(format_args!("conditional Qwen boundary has no vision sequence geometry"))
        })?;
        // Vision boundaries retain compact projected-media features. A decoder
        // producer expands additions to its actual rows before transport, so a
        // downstream decoder needs no source token mask or whole-media extent.
        // This also covers text-only/empty-media spans of a retained request.
        let deepstack_sequence = if decoder_continuation {
            source_sequence
        } else if vision_sequence > 0 {
            vision_sequence
        } else {
            source_sequence
        };
        let sequences = destination.collect(std::iter::once(primary_sequence).chain(
            std::iter::repeat_n(deepstack_sequence, selected.auxiliary().len())))?;
        destination.resolve_boundary(&schema, batch, &sequences).map(Some)
    }

    fn partition_boundary_values(
        &self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>, Self::Error> {
        let destination = crate::composite_execution::graph::Destination(metadata);
        destination.controls::<(&Self, usize, usize, &eredu_runtime::ResolvedBoundaryWireSchema,
            &B::Tensor, &Self::ForwardContext, Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>,
            Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>)>()?;
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let deepstack = if source_group == destination_group {
            forward
                .vision_state
                .as_ref()
                .ok_or_else(|| destination.error(format_args!("conditional Qwen continuation has no vision state")))?
                .deepstack_features()
        } else {
            &forward.deepstack
        };
        if deepstack.len() != schema.auxiliary().len() {
            return Err(destination.error(format_args!(
                "conditional Qwen vision boundary has incomplete DeepStack context")));
        }
        let count = deepstack.len().checked_add(1)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let mut values = destination.vector(count)?;
        values.push(destination.boundary_value(schema.primary().role(), hidden.clone())?);
        for (spec, value) in schema.auxiliary().iter().zip(deepstack) {
            values.push(destination.boundary_value(spec.role(), value.clone())?);
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
                &self.pipeline_boundary_schema(),
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

    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.target_hidden()
    }

    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(Some(vec![
            forward.tokens()?.dim(0),
            forward.tokens()?.dim(1),
            self.parsed.text.hidden_size,
        ]))
    }

    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let metadata = crate::decoder::identity::Metadata::new(
            input
                .metadata()
                .or_else(|| B::construction_metadata(context)),
        );
        prepare_input_with_metadata(input, context, metadata.context())?
            .with_target_input_destination(metadata, |input| {
                self.begin_forward_with_metadata(input, state, None, context, metadata)
            })?
    }

    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let metadata = crate::decoder::identity::Metadata::new(
            input.metadata().or_else(|| B::construction_metadata(context)),
        );
        prepare_input_with_metadata(input, context, metadata.context())?
            .with_target_input_destination(metadata, |input| {
                self.begin_forward_with_metadata(input, state, Some(parallel), context, metadata)
            })?
    }
}

/// Request-local conditional execution values.
pub struct ConditionalForwardContext<T> {
    tokens: Option<T>,
    embedded: Option<T>,
    pending_media: Option<media_prefill::PendingMedia<T>>,
    media_span: bool,
    span_assembled: bool,
    mask: Option<T>,
    mode: ForwardMode,
    parts: Vec<PreparedPart<T>>,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
    visual_mask: Option<T>,
    target_hidden: Option<T>,
    // Enclosing source account survives the paid forward containers.
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
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
                metadata: None,
                pending_media: None,
                media_span: false,
                span_assembled: false,
                tokens: Some(self.hidden.clone()),
                embedded: Some(self.hidden),
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

    /// Recovers the decoder boundary from an already prepared decoder forward.
    /// This is the inverse of `into_layered_forward`; a raw composite vision
    /// context must first use the architecture's partition finish operation to
    /// expand compact features with its actual visual mask.
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

    fn wire_schema_with_metadata(&self,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<eredu_runtime::BoundaryWireSchema,Error>{
        use eredu_runtime::{BoundaryTensorDimension as Dim,BoundaryTensorDtype as Dtype};
        crate::boundary_metadata::schema(context,Self::IDENTITY,self.hidden_size,&[],
            Some(("deepstack",self.deepstack_count,&[Dim::Batch,Dim::Sequence,Dim::Fixed(self.hidden_size)],Dtype::Activation)))
    }
    fn encode_with_metadata<T>(&self,boundary:Self::Boundary<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,Error>{
        crate::boundary_metadata::encode(context,Self::IDENTITY,[],boundary.deepstack,self.deepstack_count,"deepstack")
    }
    fn decode_with_metadata<T>(&self,tensors:Vec<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Self::Boundary<T>,Error>{
        let ([],deepstack)=crate::boundary_metadata::decode(context,Self::IDENTITY,tensors,self.deepstack_count)?;
        Ok(ConditionalPipelineBoundary{deepstack})
    }

}

/// Typed immutable decoder context transported between conditional partitions.
pub struct ConditionalPipelineBoundary<T> {
    /// Per-layer additions expanded to the transported decoder's actual rows.
    /// Vision-boundary compact features are normalized before constructing this
    /// decoder boundary; no source visual mask is required after receipt.
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
    fn visit_values<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        for value in self.tokens.iter() {
            visitor(value);
        }
        for value in self.embedded.iter() {
            visitor(value);
        }
        if let Some(pending) = &self.pending_media {
            for value in pending.values() {
                visitor(value);
            }
        }
        for value in self.mask.iter() {
            visitor(value);
        }
        for value in self.vision_initial.iter() {
            visitor(value);
        }
        for value in self.vision_output.iter() {
            visitor(value);
        }
        for value in self.deepstack.iter() {
            visitor(value);
        }
        for value in self.visual_mask.iter() {
            visitor(value);
        }
        for value in self.target_hidden.iter() {
            visitor(value);
        }
        if let Some(state) = &self.vision_state {
            for value in state.retained_values() {
                visitor(value);
            }
        }
    }

    fn tokens(&self) -> Result<&T, Error> {
        self.tokens
            .as_ref()
            .ok_or_else(|| Error::backend("conditional decoder tokens are pending ingress"))
    }
    fn embedded(&self) -> Result<&T, Error> {
        self.embedded
            .as_ref()
            .ok_or_else(|| Error::backend("conditional decoder embedding is pending ingress"))
    }
    /// Hidden state captured after the selected text execution group.
    pub const fn target_hidden(&self) -> Option<&T> {
        self.target_hidden.as_ref()
    }
}

/// One neutral conditional Qwen3.5 graph.
pub struct ConditionalLayeredModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    parsed: crate::replicated_text::SharedCompositeConfig<ParsedHybridConfig>,
    construction_units: Option<RetainedConditionalUnits>,
    static_modules: ConditionalStaticModules<B>,
    target_layers: usize,
    prediction_steps: usize,
    execution_graph: ExecutionGraph,
    parallel_geometry: Option<std::sync::Arc<ConditionalLocalGeometry>>,
    partition_state_layout: Option<StateLayout>,
    partition_state_offset: usize,
    expert_realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ArchitectureParameters<B> for ConditionalLayeredModel<B>
{
    type DefinitionError = Error;


    fn state_layout(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Self::DefinitionError> {
match context { Some(context) => {
        match &self.parallel_geometry {
            Some(geometry) => geometry.state_layout().clone_workspace(context),
            None => super::state_layout_with_metadata(&self.parsed.text, context),
        }
    }, None => {
        self.state_layout_impl()
    } }
}


    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
match context { Some(context) => {
        self.state_identity_destination(
            state,
            topology,
            crate::decoder::identity::Metadata::new(Some(context)),
        )
    }, None => {
        self.state_identity_destination(
            state,
            topology,
            crate::decoder::identity::Metadata::new(None),
        )
    } }
}

    fn parameter_description(&self, context: &<B::Tensor as Tensor>::Context)
        -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Self::DefinitionError> {
        crate::decoder::ModuleMetadata::new::<B>(context).controls::<(
            &Self, &<B::Tensor as Tensor>::Context, std::borrow::Cow<'_, ArchitectureParameterDescription>,
        )>()?;
        self.parameter_description_impl(context).map(std::borrow::Cow::Owned)
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

    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        eredu_nn::Parameterized::retained_value_slot_bound(&self.static_modules)
    }

    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        eredu_nn::Parameterized::visit_retained_values(&self.static_modules, visitor)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        if let Some(shared) = &self.static_modules.prediction {
            visitor.visit("mtp", shared)?;
        }
        visitor.visit("vision", &self.static_modules.vision)?;
        visitor.visit("embedding", &self.static_modules.text.embeddings)?;
        visitor.visit("norm", &self.static_modules.text.norm)?;
        if let Some(head) = &self.static_modules.text.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(shared) = &mut self.static_modules.prediction {
            visitor.visit_mut("mtp", shared)?;
        }
        visitor.visit_mut("vision", &mut self.static_modules.vision)?;
        visitor.visit_mut("embedding", &mut self.static_modules.text.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.text.norm)?;
        if let Some(head) = &mut self.static_modules.text.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> ConditionalLayeredModel<B> {
    fn begin_forward_with_metadata<S>(
        &mut self,
        input: ConditionalInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        metadata.controls::<(
            Option<&B::ParallelContext>,
            LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>,
            Vec<B::Tensor>,
            [Index; 3],
        )>()?;
        let complete;
        let expected = match (
            self.partition_state_layout.as_ref(),
            self.parallel_geometry.as_ref(),
        ) {
            (Some(layout), _) => layout,
            (None, Some(geometry)) => geometry.state_layout(),
            (None, None) if parallel.is_some() => {
                return Err(metadata.error(format_args!("conditional Qwen3.5 model has no local geometry")));
            }
            (None, None) => {
                complete = match metadata.context() {
                    Some(metadata) => {
                        super::state_layout_with_metadata(&self.parsed.text, metadata)?
                    }
                    None => super::state_layout(&self.parsed.text).map_err(Error::backend)?,
                };
                &complete
            }
        };
        if state.layout() != expected {
            return Err(
                crate::composite_execution::graph::Destination(metadata.context())
                    .error(format_args!("conditional Qwen3.5 state layout mismatch")),
            );
        }
        match input {
            ConditionalInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.prediction_steps {
                    return Err(
                        metadata.error(format_args!("conditional Qwen3.5 MTP depth is invalid"))
                    );
                }
                let embedded = match parallel {
                    Some(parallel) => B::vocabulary_parallel_lookup(
                        &mut self.static_modules.text.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?,
                    None => self.static_modules.text.embeddings.forward(tokens, context)?,
                };
                let state_index = self.target_layers + depth;
                let offset = state
                    .layer(state_index)
                    .map_err(|cause| metadata.source(cause))?
                    .position();
                let mask = if tokens.dim(1) > 1 {
                    Some(B::causal_mask(tokens.dim(1), offset, None, context)?)
                } else {
                    None
                };
                Ok(LayeredForwardState {
                    hidden: hidden.clone(),
                    context: ConditionalForwardContext {
                        metadata: metadata.context().cloned(),
                        pending_media: None,
                        media_span: false,
                        span_assembled: false,
                        tokens: Some(tokens.clone()),
                        embedded: Some(embedded),
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
                let (prepared, grids) =
                    self.prepare_parts_with_metadata(parts, parallel, context, metadata)?;
                let has_media = !grids.is_empty();
                if has_media != pixels.is_some() {
                    return Err(metadata.error(format_args!(
                        "conditional Qwen3.5 pixels and media metadata must appear together"
                    )));
                }
                let offset = state
                    .layer(0)
                    .map_err(|cause| metadata.source(cause))?
                    .position();
                if has_media && offset != 0 {
                    return Err(metadata.error(format_args!(
                        "conditional Qwen3.5 media cannot append to populated state"
                    )));
                }
                let (vision_initial, vision_state) = match pixels {
                    Some(pixels) => {
                        let (hidden, state) = self.static_modules.vision.begin(
                            VisionInput {
                                pixels,
                                grid: &grids,
                            },
                            context,
                        )?;
                        (Some(hidden), Some(state))
                    }
                    None => (None, None),
                };
                let assembled = self.assemble_with_metadata(&prepared, None, context, metadata);
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
                let hidden = vision_initial
                    .as_ref()
                    .cloned()
                    .or_else(|| embedded.clone())
                    .ok_or_else(|| {
                        error.unwrap_or_else(|| {
                            metadata.error(format_args!("empty conditional input"))
                        })
                    })?;
                let embedded = embedded.unwrap_or_else(|| hidden_embedding_placeholder(&prepared));
                let deepstack_count = self
                    .parsed
                    .vision
                    .as_ref()
                    .expect("validated conditional vision")
                    .deepstack_layers()
                    .len();
                let mut deepstack = metadata.vector(deepstack_count)?;
                for _ in 0..deepstack_count {
                    deepstack.push(embedded.zeros_like(context)?);
                }
                Ok(LayeredForwardState {
                    hidden,
                    context: ConditionalForwardContext {
                        metadata: metadata.context().cloned(),
                        pending_media: None,
                        media_span: false,
                        span_assembled: false,
                        tokens: Some(tokens.unwrap_or_else(|| hidden_token_placeholder(&prepared))),
                        embedded: Some(embedded),
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

    #[allow(clippy::too_many_arguments)]
    fn begin_distributed_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        _first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if state.layout() != expected {
            return Err(Error::backend(
                "conditional Qwen partition state layout mismatch",
            ));
        }
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
        self.begin_routed_target_partition(input, mask, batch, sequence, offset, parallel, context)
    }

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
        self.static_modules.text.embeddings.forward(tokens, context)
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
        match &mut self.static_modules.text.lm_head {
            Some(head) => head.forward(hidden, context),
            None => self
                .static_modules
                .text
                .embeddings
                .as_linear(hidden, context),
        }
    }

    pub(crate) fn project_prediction_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .text
            .project_instrumented(hidden, parallel, context, instrumentation)
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

    fn state_identity_destination(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        metadata.controls::<(
            eredu_runtime::ModelStateIdentity,
            &Self,
            &eredu_runtime::PartitionState,
            eredu_core::cache::PromptCacheTopology,
        )>()?;
        let target_layers = usize::try_from(self.parsed.text.num_hidden_layers)
            .map_err(|cause| metadata.source(cause))?;
        let prediction_layers = usize::try_from(self.parsed.text.mtp_num_hidden_layers)
            .map_err(|cause| metadata.source(cause))?;
        let layer_count = target_layers
            .checked_add(prediction_layers)
            .ok_or_else(|| {
                metadata.error(format_args!(
                    "conditional Qwen state layer count overflowed"
                ))
            })?;
        let global_layer_start = state.global_layer_offset();
        let global_layer_end = global_layer_start
            .checked_add(state.layout().len())
            .ok_or_else(|| {
                metadata.error(format_args!(
                    "conditional Qwen owned layer range overflowed"
                ))
            })?;
        if global_layer_end > layer_count {
            return Err(metadata.error(format_args!(
                "conditional Qwen owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
            )));
        }
        eredu_runtime::ModelStateIdentity::new_with_diagnostic(
            metadata.text(self.parsed.text.variant.model_kind().canonical_name())?,
            metadata.text(&self.parsed.text.model_type)?,
            super::conditional_prompt_cache_architecture_fingerprint_with_metadata(
                &self.parsed,
                metadata,
            )?,
            layer_count,
            global_layer_start,
            0,
            topology,
            |message| metadata.prompt_error(message),
        )
    }

    /// Returns the normalized family configuration owned by this architecture.
    pub fn args(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    fn canonical_execution_graph(&self) -> Result<ExecutionGraph, Error> {
        self.canonical_execution_graph_destination(None)
    }

    fn canonical_execution_graph_destination(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<ExecutionGraph, Error> {
        match context {
            Some(context) => self.execution_graph.clone_with_metadata(context),
            None => Ok(self.execution_graph.clone()),
        }
    }

    fn build_execution_graph(
        prediction_steps: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<ExecutionGraph, Error> {
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<(ExecutionGraph, String)>()?;
        let count = prediction_steps
            .checked_add(2)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let mut groups = destination.vector(count)?;
        groups.push(destination.group(VISION_EXECUTION_GROUP, &[])?);
        groups.push(destination.group(
            crate::decoder::TARGET_EXECUTION_GROUP,
            &[VISION_EXECUTION_GROUP],
        )?);
        let mut output = destination.text(format_args!("target"))?;
        for depth in 0..prediction_steps {
            let id = destination.text(format_args!("mtp.{depth}"))?;
            groups.push(destination.group(&id, &[&output])?);
            output = id;
        }
        destination.finish(groups, &output)
    }

    fn canonical_group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.canonical_group_unit_count_destination(group, None)
    }

    fn canonical_group_unit_count_destination(
        &self,
        group: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Error> {
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<(usize, &Self)>()?;
        match group {
            0 => Ok(self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .layer_count()),
            1 => Ok(self.target_layers),
            group if group < self.prediction_steps + 2 => Ok(1),
            _ => Err(destination.error(format_args!(
                "conditional Qwen3.5 execution group is invalid"
            ))),
        }
    }

    /// Returns the architecture-owned vision, target, and prediction traversal.
    pub fn unit_layout(&self) -> Result<ExecutionUnitLayout, Error> {
        let graph = self.canonical_execution_graph()?;
        let counts = (0..graph.groups().len())
            .map(|group| self.canonical_group_unit_count(group))
            .collect::<Result<Vec<_>, _>>()?;
        ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)
    }

    /// Returns the family-owned activation schema transported between stages.
    pub fn pipeline_boundary_schema(&self) -> ConditionalPipelineBoundarySchema {
        ConditionalPipelineBoundarySchema {
            hidden_size: self.parsed.text.hidden_size,
            deepstack_count: self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .deepstack_layer_count(),
        }
    }

    /// Describes vision, target, and prediction parameters with explicit
    /// canonical graph ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let metadata = B::construction_metadata(context);
        let graph = self.canonical_execution_graph_destination(metadata)?;
        let vision = self.parsed.vision.as_ref().expect("validated vision");
        let destination = crate::composite_execution::graph::Destination(metadata);
        destination.controls::<(
            ArchitectureParameterDescription,
            Vec<OwnedParameterGroupSpec>,
            Vec<eredu_runtime::ParameterGroupSpec>,
        )>()?;
        let mut counts = destination.vector(graph.groups().len())?;
        for group in 0..graph.groups().len() {
            counts.push(self.canonical_group_unit_count_destination(group, metadata)?);
        }
        let layout = destination.layout(&graph, &counts)?;
        // A partition-local model contains already-local tensor shapes. Keep
        // parameter authority checkpoint-global so selected TP segments are
        // validated once instead of being applied again to local modules.
        destination.controls::<(
            Option<Result<(crate::decoder::StaticModules<B>, VisionStatic<B>), Error>>,
            Option<(crate::decoder::StaticModules<B>, VisionStatic<B>)>,
        )>()?;
        let global_static = self.parallel_geometry.is_some().then(|| {
            Ok::<_, Error>((
                super::LayeredModel::<B>::prepare_static_modules(&self.parsed.text, context)?.base,
                VisionStatic::new_with_config(
                    crate::replicated_text::CompositeModelConfig::Retained(self.parsed.project(
                        |parsed| {
                            parsed
                                .vision
                                .as_ref()
                                .expect("validated immutable vision config")
                        },
                        metadata,
                    )?),
                    "model.visual",
                    context,
                )?,
            ))
        });
        let global_static = global_static.transpose()?;
        let (text_modules, vision_modules) = global_static.as_ref().map_or(
            (&self.static_modules.text, &self.static_modules.vision),
            |(text, vision)| (text, vision),
        );
        let text_static = crate::decoder::static_parallel_parameter_groups_with_metadata::<B>(
            &text_modules.embeddings,
            &text_modules.norm,
            text_modules.lm_head.as_ref(),
            "model",
            metadata,
        )
        .map_err(|cause| cause.into_neural())?;
        let vision_static =
            crate::qwen::vision::owned_static_parallel_parameter_groups_with_metadata::<B>(
                vision_modules,
                vision,
                "model.visual",
                destination.group_id(VISION_EXECUTION_GROUP)?,
                "vision",
                metadata,
            )?;
        let mut owned = destination.vector(text_static.len())?;
        for (index, group) in text_static.into_iter().enumerate() {
            let roles: &[&str] = if index == 0 {
                if self.parsed.text.tie_word_embeddings {
                    &["embedding", "mtp", "output"]
                } else {
                    &["embedding", "mtp"]
                }
            } else {
                match index {
                    1 => &["norm"],
                    _ => &["output"],
                }
            };
            owned.push(OwnedParameterGroupSpec::new(
                destination.static_owner(roles)?,
                group,
            ));
        }
        destination.append_owned(&mut owned, vision_static)?;
        if let Some(shared) = &self.static_modules.prediction {
            let groups = super::parallel::prediction_shared_parameter_groups_with_metadata(
                shared,
                self.prediction_steps,
                metadata,
            )?;
            destination.append_owned(&mut owned, groups)?;
        }
        for group_index in 0..layout.group_count() {
            let group_id = layout
                .group_id(group_index)
                .expect("conditional layout group");
            let count = layout
                .group_range(group_index)
                .expect("conditional layout range")
                .len();
            for index in 0..count {
                let unit = if self.parallel_geometry.is_some() {
                    match group_index {
                        0 => ConditionalUnit::Vision(VisionBlock::new_with_root(
                            vision,
                            "model.visual",
                            index,
                            context,
                        )?),
                        1 => {
                            ConditionalUnit::Target(Block::new(&self.parsed.text, index, context)?)
                        }
                        _ => ConditionalUnit::Prediction(PredictionUnit::new(
                            &self.parsed.text,
                            group_index - 2,
                            context,
                        )?),
                    }
                } else {
                    self.construct_unit(group_index, index, context)?
                };
                let groups = match unit {
                    ConditionalUnit::Vision(block) => {
                        crate::qwen::vision::block_parallel_parameter_groups_with_metadata(
                            &block,
                            vision,
                            "model.visual",
                            index,
                            metadata,
                        )
                    }
                    ConditionalUnit::Target(block) => {
                        super::parallel::unit_parallel_parameter_groups_with_metadata(
                            &Unit::Target(block),
                            &self.parsed.text,
                            0,
                            index,
                            metadata,
                        )
                    }
                    ConditionalUnit::Prediction(prediction) => {
                        super::parallel::unit_parallel_parameter_groups_with_metadata(
                            &Unit::Prediction(prediction),
                            &self.parsed.text,
                            group_index - 1,
                            index,
                            metadata,
                        )
                    }
                }?;
                destination.append_unit(&mut owned, groups, group_id, index)?;
            }
        }
        destination.description(graph, layout, owned)
    }

    /// Builds the exact configured vision/text/MTP graph.
    pub fn new(
        parsed: ParsedHybridConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_config(
            crate::replicated_text::CompositeModelConfig::Owned(parsed),
            context,
        )
    }

    pub(crate) fn config_owner(
        &self,
    ) -> &crate::replicated_text::SharedCompositeConfig<ParsedHybridConfig> {
        &self.parsed
    }

    pub(crate) fn new_with_config(
        parsed: crate::replicated_text::CompositeModelConfig<ParsedHybridConfig>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if let Some(metadata) = B::construction_metadata(context) {
            metadata.charge_metadata(
                std::mem::size_of_val(&parsed) + std::mem::size_of::<Result<Self, Error>>(),
            )?;
        }
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        crate::operator_requirements::require_with_metadata::<B>(
            "Qwen hybrid conditional",
            crate::operator_requirements::QWEN_HYBRID.union(crate::operator_requirements::QWEN_VL),
            metadata,
        )?;
        let vision = parsed.vision.as_ref().ok_or_else(|| {
            metadata.error(format_args!("conditional Qwen3.5 requires vision config"))
        })?;
        match B::construction_metadata(context).filter(|context| context.uses_checked_metadata()) {
            Some(context) => {
                vision.validate_for_with_metadata(VisionMode::WindowScheduled, context)?
            }
            None => vision
                .validate_for(VisionMode::WindowScheduled)
                .map_err(Error::backend)?,
        }
        let text = super::LayeredModel::<B>::prepare_static_modules(&parsed.text, context)?;
        let target_layers =
            usize::try_from(parsed.text.num_hidden_layers).map_err(Error::backend)?;
        let prediction_steps =
            usize::try_from(parsed.text.mtp_num_hidden_layers).map_err(Error::backend)?;
        let parsed = parsed.into_shared(B::construction_metadata(context))?;
        let vision = parsed.project(
            |parsed| {
                parsed
                    .vision
                    .as_ref()
                    .expect("validated immutable vision config")
            },
            B::construction_metadata(context),
        )?;
        Ok(Self {
            parsed,
            construction_units: None,
            static_modules: ConditionalStaticModules {
                text: text.base,
                prediction: text.extension,
                vision: VisionStatic::new_with_config(
                    crate::replicated_text::CompositeModelConfig::Retained(vision),
                    "model.visual",
                    context,
                )?,
            },
            target_layers,
            prediction_steps,
            execution_graph: Self::build_execution_graph(prediction_steps, B::construction_metadata(context))?,
            parallel_geometry: None,
            partition_state_layout: None,
            partition_state_offset: 0,
            expert_realization: None,
        })
    }

    /// Applies the architecture-owned tensor-parallel token embedding boundary.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            &mut self.static_modules.text.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )
    }

    /// Applies the architecture-owned tensor-parallel output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        normalize: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = if normalize {
            self.static_modules.text.norm.forward(hidden, context)?
        } else {
            hidden.clone()
        };
        match &mut self.static_modules.text.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }
    }

    /// Builds the conditional graph with planner-derived local modules.
    pub fn new_parallel(
        parsed: ParsedHybridConfig,
        geometry: ConditionalLocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen hybrid conditional",
            crate::operator_requirements::QWEN_HYBRID.union(crate::operator_requirements::QWEN_VL),
        )?;
        geometry.validate_for(&parsed).map_err(Error::backend)?;
        let vision_config = parsed
            .vision
            .clone()
            .ok_or_else(|| Error::backend("conditional Qwen3.5 requires vision config"))?;
        vision_config
            .validate_for(VisionMode::WindowScheduled)
            .map_err(Error::backend)?;
        let text_model = super::LayeredModel::<B>::new_parallel(
            parsed.text.clone(),
            geometry.text().clone(),
            context,
        )?;
        let target_layers =
            usize::try_from(parsed.text.num_hidden_layers).map_err(Error::backend)?;
        let prediction_steps =
            usize::try_from(parsed.text.mtp_num_hidden_layers).map_err(Error::backend)?;
        Ok(Self {
            parsed: crate::replicated_text::SharedCompositeConfig::new(
                parsed,
                B::construction_metadata(context),
            )?,
            construction_units: None,
            static_modules: {
                let text = text_model.into_extended_static_modules();
                ConditionalStaticModules {
                    text: text.base,
                    prediction: text.extension,
                    vision: VisionStatic::new_parallel_with_root(
                        vision_config,
                        "model.visual",
                        geometry.merger_widths(),
                        context,
                    )?,
                }
            },
            target_layers,
            prediction_steps,
            execution_graph: Self::build_execution_graph(prediction_steps, B::construction_metadata(context))?,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
            partition_state_layout: None,
            partition_state_offset: 0,
            expert_realization: None,
        })
    }

    /// Restricts execution state to the exact PP-owned target slice while
    /// retaining the complete TP-local unit construction geometry.
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
        self.partition_state_layout = Some(layout);
        self.partition_state_offset = global_offset;
        Ok(self)
    }

    /// Normalized composite policy.
    pub fn parsed(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    /// Returns the prediction depth count declared by the constructed execution graph.
    pub const fn mtp_len(&self) -> usize {
        self.prediction_steps
    }

    /// Returns replicated or planner-derived target/MTP state geometry.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(
                || super::state_layout(&self.parsed.text).map_err(Error::backend),
                Ok,
            )
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
        if !self.construction_units.as_ref().is_some_and(|source| source.matches_realization(&realization)) {
            self.construction_units = None;
        }
        self.expert_realization = Some(realization);
    }

    /// Constructs one canonical vision, target, or prediction unit using this
    /// model's replicated or planner-derived local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalUnit<B>, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(&Self, usize, usize, ConditionalUnit<B>, Option<&ConditionalLocalGeometry>, Option<&HybridConfig>, &HybridConfig)>()?;
        if let Some(source) = &self.construction_units {
            source.validate(self, context)?;
            if group == 1 {
                let spec = source.targets().get(index).ok_or_else(|| metadata.error(format_args!("conditional target unit is outside retained source")))?;
                return spec.instantiate::<B>(self.target_construction_config(index), context).map(ConditionalUnit::Target);
            }
            if group >= 2 {
                if index != 0 { return Err(metadata.error(format_args!("conditional prediction unit is outside retained source"))); }
                let spec = source.predictions().get(group - 2).ok_or_else(|| metadata.error(format_args!("conditional prediction group is outside retained source")))?;
                return spec.instantiate::<B>(context).map(ConditionalUnit::Prediction);
            }
        }
        let count = match group {
            0 => self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .layer_count(),
            1 => self.target_layers,
            group if group < self.prediction_steps + 2 => 1,
            _ => return Err(metadata.error(format_args!("conditional Qwen3.5 group is invalid"))),
        };
        if index >= count {
            return Err(metadata.error(format_args!("conditional Qwen3.5 unit is outside its group")));
        }
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

    fn prepare_parts(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        self.prepare_parts_with_metadata(
            parts,
            None,
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }

    fn prepare_parts_with_metadata(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        metadata.controls::<(
            Option<&B::ParallelContext>,
            Vec<PreparedPart<B::Tensor>>,
            Vec<(i32, i32, i32)>,
            PreparedPart<B::Tensor>,
        )>()?;
        if parts.is_empty() {
            return Err(metadata.error(format_args!("conditional Qwen3.5 input has no parts")));
        }
        let grid_count = parts.iter().try_fold(0usize, |count, part| {
            let rows = match part {
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => grid.len(),
                _ => 0,
            };
            count
                .checked_add(rows)
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)
        })?;
        let mut grids = metadata.vector(grid_count)?;
        let mut prepared = metadata.vector(parts.len())?;
        for part in parts {
            let value: Result<PreparedPart<B::Tensor>, Error> = match part {
                InputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: match parallel {
                        Some(parallel) => B::vocabulary_parallel_lookup(
                            &mut self.static_modules.text.embeddings,
                            tokens,
                            EmbeddingLookupPolicy::Strict,
                            parallel,
                            context,
                        )?,
                        None => self.static_modules.text.embeddings.forward(tokens, context)?,
                    },
                }),
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.parsed.text.hidden_size]
                    {
                        return Err(metadata.error(format_args!(
                            "conditional Qwen3.5 projected input geometry mismatch"
                        )));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            };
            prepared.push(value?);
        }
        Ok((prepared, grids))
    }

    fn prepare_parts_parallel(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        self.prepare_parts_with_metadata(
            parts,
            Some(parallel),
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        self.assemble_with_metadata(
            parts,
            vision,
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }
    fn assemble_with_metadata(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        metadata.controls::<(
            eredu_nn::multimodal::OrderedModelInput<B::Tensor>,
            Vec<B::Tensor>,
            Vec<OrderedInputPart<'_, B::Tensor>>,
            [Index; 3],
        )>()?;
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
            return Err(metadata.error(format_args!(
                "conditional Qwen3.5 media placeholders require [1, {media_tokens}, {}], got {:?}",
                self.parsed.text.hidden_size,
                vision.map(|value| value.shape()),
            )));
        }
        let mut offset = 0;
        let mut embeddings = metadata.vector(parts.len())?;
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
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
        let mut ordered = metadata.vector(parts.len())?;
        ordered.extend(
            parts
                .iter()
                .zip(&embeddings)
                .map(|(part, embeddings)| OrderedInputPart {
                    token_ids: match part {
                        PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => {
                            tokens
                        }
                    },
                    embeddings,
                }),
        );
        match metadata.context() {
            Some(metadata) => eredu_nn::multimodal::assemble_ordered_inputs_with_metadata(
                &ordered,
                self.parsed.text.hidden_size,
                context,
                metadata,
            ),
            None => assemble_ordered_inputs(&ordered, self.parsed.text.hidden_size, context),
        }
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        let global = if group == 1 {
            index
        } else if group >= 2 && group < self.prediction_steps + 2 && index == 0 {
            self.target_layers + group - 2
        } else {
            return Err(Error::backend(
                "conditional Qwen3.5 state address is invalid",
            ));
        };
        global
            .checked_sub(self.partition_state_offset)
            .ok_or_else(|| Error::backend("conditional Qwen state precedes its selected PP offset"))
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
        let (parts, grids) = match parallel {
            Some(parallel) => self.prepare_parts_parallel(parts, parallel, context)?,
            None => self.prepare_parts(parts, context)?,
        };
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
                let (hidden, state) = self.static_modules.vision.begin(
                    VisionInput {
                        pixels,
                        grid: &grids,
                    },
                    context,
                )?;
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
                PreparedPart::Text { tokens, embeddings } => {
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
                PreparedPart::Text { tokens, embeddings } => {
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

    /// Applies the conditional target normalization and vocabulary projection.
    pub fn finish_pipeline_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
        match &mut self.static_modules.text.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self
                .static_modules
                .text
                .embeddings
                .as_linear(&hidden, context),
        }
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
        P: eredu_runtime::RoutedExpertProvider<B>,
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
                forward.embedded()?,
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

    // One equation for a compact feature's actual decoder-row contribution.
    // Decoder boundaries carry this representation; vision boundaries stay compact.
    fn decoder_deepstack_contribution(
        features: &B::Tensor,
        hidden: &B::Tensor,
        visual_mask: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if features.shape() == hidden.shape() {
            return Ok(features.clone());
        }
        let source = features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
        hidden.zeros_like(context)?.masked_scatter(
            visual_mask.ok_or_else(|| Error::backend("missing conditional visual mask"))?,
            &source,
            context,
        )
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
        let output = if features.shape() == output.shape() {
            if instrumentation.enabled() {
                let contribution = instrumentation.apply("deepstack.output", features.clone())?;
                output.add(&contribution, context)?
            } else {
                output.add(features, context)?
            }
        } else {
            let contribution = Self::decoder_deepstack_contribution(
                features,
                &output,
                forward.visual_mask.as_ref(),
                context,
            )?;
            let contribution = instrumentation.apply("deepstack.output", contribution)?;
            output.add(&contribution, context)?
        };
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
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
                forward.embedded()?,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            )?,
            _ => {
                return Err(Error::backend(
                    "conditional Qwen3.5 parallel unit/group mismatch",
                ));
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

impl<B, S> LayeredArchitecture<B, S> for ConditionalLayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn media_prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata=crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(&Self,Option<&eredu_nn::workspace::WorkspaceContext>,usize,usize,String,Vec<eredu_runtime::layered::PrefillObservationDeclaration>,std::ops::Range<usize>,Option<eredu_runtime::RoutedObservationPoints>,Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>,Error>)>()?;

        // The validated fresh-cache source preserves projected/placeholder rows and causal recurrent/KV state.
        let units = <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 1, metadata_context)?;
        let mut declarations = crate::decoder::media_prefill_observation_declarations(
            (0..units).map(|index| <Self as LayeredArchitecture<B, S>>::unit_path(self, 1, index, metadata_context)), metadata_context)?;
        // The retained-media ingress changes decoder inputs and positions, but
        // executes the same row-local routed banks as the ordinary target.
        // Reuse those exact architecture declarations; encoder hooks remain
        // outside this decoder contract.
        let ordinary=<Self as LayeredArchitecture<B,S>>::prefill_observation_declarations(self,metadata_context)?;
        metadata.controls::<(Vec<eredu_runtime::layered::PrefillObservationDeclaration>,usize)>()?;
        let selected = ordinary.iter().filter(|declaration| declaration.flattens_batch_tokens());
        metadata.borrowed_controls(&selected)?;
        let count = selected.count();
        if let Some(context) = metadata.context() { context.reserve_metadata_vec(&mut declarations, count)?; }
        let selected = ordinary.into_iter().filter(|declaration| declaration.flattens_batch_tokens());
        metadata.borrowed_controls(&selected)?;
        declarations.extend(selected);
        Ok(declarations)
    }

    fn prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata=crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(&Self,Option<&eredu_nn::workspace::WorkspaceContext>,usize,usize,String,Vec<eredu_runtime::layered::PrefillObservationDeclaration>,std::ops::Range<usize>,Option<eredu_runtime::RoutedObservationPoints>,Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>,Error>)>()?;

        // The actual group1 target retains its KV or convolution/recurrent
        // state across causal text chunks. Vision and later MTP groups are not
        // declared; existing prediction hook availability stays unchanged.
        let units = <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 1, metadata_context)?;
        let mut declarations = crate::decoder::ordinary_prefill_observation_declarations(
            (0..units).map(|index| <Self as LayeredArchitecture<B, S>>::unit_path(self, 1, index, metadata_context)),
            true, metadata_context)?;
        // Same target bank invocation as observed execution; its expert equations are row-local.
        for index in 0..units {
            let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, 1, index, metadata_context)?;
            if self.parsed.text.num_experts > 0 {
                crate::decoder::append_routed_prefill_path(&mut declarations, &metadata.format(format_args!("{path}.mlp"))?, metadata_context)?;
            }
        }
        Ok(declarations)
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(target, target, target)
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
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_target_observed(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
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
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if matches!(forward.mode, ForwardMode::Target) {
            self.finish_target_observed(hidden, None, context, observer)
        } else {
            self.finish_forward(hidden, state, forward, context)
        }
    }

    type Input<'a> = ConditionalInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        match input {
            ConditionalInput::Target { parts, .. } => {
                crate::prefill::segmented_token_shape(parts.iter().map(|part| match part {
                    InputPart::Text(tokens)
                    | InputPart::Image { tokens, .. }
                    | InputPart::Video { tokens, .. }
                    | InputPart::Projected { tokens, .. } => *tokens,
                }))
                .map(Some)
            }
            ConditionalInput::Draft { .. } => Ok(None),
        }
    }

    type StaticModules = ConditionalStaticModules<B>;
    type Unit = ConditionalUnit<B>;
    type ForwardContext = ConditionalForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        match group {
            0 => crate::transport::vision_declaration().into_owned(),
            1 => crate::transport::decoder(),
            _ => conditional_prediction_group_transport(group),
        }
    }

    fn group_transport_matches(
        &self,
        group: usize,
        expected: &eredu_runtime::ArchitectureGroupTransport,
    ) -> bool {
        match group {
            0 => crate::transport::vision_declaration().matches(expected),
            1 => crate::transport::decoder_declaration().matches(expected),
            _ => conditional_prediction_group_declaration(group).matches(expected),
        }
    }

    fn primary_execution_group(&self) -> &str {
        crate::decoder::TARGET_EXECUTION_GROUP
    }

    fn prediction_execution_groups(&self) -> Vec<String> {
        (0..self.prediction_steps)
            .map(|depth| format!("mtp.{depth}"))
            .collect()
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_with_output_state(1, self.target_layers, layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            &self.execution_graph,
        ))
    }

    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        self.canonical_group_unit_count_destination(group, metadata_context)
    }



    fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        if index >= <Self as LayeredArchitecture<B, S>>::group_unit_count(self, group, metadata_context)? {
            return Err(metadata.error(format_args!("{}", "conditional Qwen3.5 unit is outside its group")));
        }
        match group {
            0 => metadata.text(format_args!("model.visual.blocks.{index}")),
            1 => metadata.text(format_args!("model.layers.{index}")),
            _ => metadata.text(format_args!("mtp.layers.{}", group - 2)),
        }
    }

    fn group_input_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path((group == 1).then_some(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH))
    }

    fn group_output_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path((group == 0).then_some(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH))
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
    ) -> Result<Self::Unit, Error> {
        self.construct_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_forward_with_metadata(
            input,
            state,
            None,
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }

    fn begin_execution_group(
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
                if forward.span_assembled {
                    return Ok(initial.clone());
                }
                let vision = if forward.media_span {
                    forward.vision_output.as_ref()
                } else {
                    forward
                        .vision_output
                        .as_ref()
                        .and_then(|_| dependencies.first().copied())
                        .or(forward.vision_output.as_ref())
                };
                let assembled = self.assemble_with_metadata(
                    &forward.parts,
                    vision,
                    context,
                    crate::decoder::identity::Metadata::new(
                        forward
                            .metadata
                            .as_ref()
                            .or_else(|| B::construction_metadata(context)),
                    ),
                )?;
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
                forward.tokens = Some(assembled.token_ids);
                forward.embedded = Some(assembled.embeddings.clone());
                Ok(assembled.embeddings)
            }
            _ => Ok(dependencies.first().copied().unwrap_or(initial).clone()),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 1 || (group == 0 && forward.vision_state.is_some()),
            ForwardMode::Draft(depth) => group == depth + 2,
        }
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            1 => index,
            _ => self.target_layers + group - 2,
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
    ) -> Result<B::Tensor, Error> {
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

    fn complete_execution_group(
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

    fn select_readout_positions(
        &self,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        crate::readout::select_readout_positions(hidden, demand, 1, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = match forward.mode {
            ForwardMode::Target => self.static_modules.text.norm.forward(hidden, context)?,
            ForwardMode::Draft(_) => hidden.clone(),
        };
        match &mut self.static_modules.text.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self
                .static_modules
                .text
                .embeddings
                .as_linear(&hidden, context),
        }
    }

    fn forward_metadata(
        &self,
        forward: &Self::ForwardContext,
    ) -> Option<eredu_runtime::layered::LayeredMetadata<Self::Error>> {
        forward.metadata.as_ref().map(|context| {
            eredu_runtime::layered::LayeredMetadata::new(context, |error| error)
        })
    }

    fn visit_retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
        visitor: &mut dyn FnMut(&'a B::Tensor),
    ) {
        forward.visit_values(visitor);
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        forward.visit_values(&mut |value| values.push(value));
        values.into_iter()
    }
}

fn conditional_prediction_group_transport(
    group: usize,
) -> eredu_runtime::ArchitectureGroupTransport {
    conditional_prediction_group_declaration(group).into_owned()
}

fn conditional_prediction_group_declaration(
    group: usize,
) -> eredu_runtime::ArchitectureGroupTransportDeclaration<'static> {
    let mut transport = crate::transport::prediction_declaration();
    if group == 2 {
        transport.first_owner_static_roles = &["mtp"];
    }
    transport
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "the transport contract tests stay adjacent to the transport declaration"
)]
mod transport_tests {
    use super::{ConditionalPipelineBoundarySchema, conditional_prediction_group_transport};
    use eredu_runtime::ArchitectureBoundary;

    #[test]
    fn conditional_deepstack_count_owns_wire_cardinality() {
        let schema = ConditionalPipelineBoundarySchema {
            hidden_size: 48,
            deepstack_count: 3,
        };
        let tensors = schema.wire_schema().unwrap().resolve(2, 4).unwrap();
        assert_eq!(tensors.primary().shape(), [2, 4, 48]);
        assert_eq!(tensors.auxiliary().len(), 3);
        assert_eq!(tensors.auxiliary()[0].role(), "deepstack.0");
        assert_eq!(tensors.auxiliary()[2].shape(), [2, 4, 48]);
    }

    #[test]
    fn first_conditional_prediction_group_owns_shared_mtp_embedding_role_once() {
        assert_eq!(
            conditional_prediction_group_transport(2).first_owner_static_roles,
            ["mtp"]
        );
        assert!(
            conditional_prediction_group_transport(3)
                .first_owner_static_roles
                .is_empty()
        );
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        let target = self.prediction_steps == 0;
        eredu_runtime::inspection::ObservationHookSupport::internal(target, target, target)
    }
    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
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
            group,
            index,
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
    fn finish_forward_parallel_observed<O>(
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
            self.finish_forward_parallel(hidden, state, forward, parallel, context)
        }
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_forward_with_metadata(
            input,
            state,
            Some(parallel),
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
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
    ) -> Result<B::Tensor, Error> {
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

    fn complete_execution_group_parallel(
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

    fn finish_forward_parallel(
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
        let hidden = match forward.mode {
            ForwardMode::Target => self.static_modules.text.norm.forward(hidden, context)?,
            ForwardMode::Draft(_) => hidden.clone(),
        };
        match &mut self.static_modules.text.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
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
            Some(parallel) => self.finish_forward_parallel_observed(
                hidden, state, forward, parallel, context, observer,
            )?,
            None => self.finish_forward_observed(hidden, state, forward, context, observer)?,
        };
        Ok(LayeredPartitionOutput::Final {
            output,
            retained: Some(hidden.clone()),
        })
    }

    type Boundary = ConditionalPipelineBoundarySchema;

    fn boundary_schema(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Self::Boundary, Self::Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &Self, Option<&eredu_nn::workspace::WorkspaceContext>,
                Self::Boundary, Result<Self::Boundary, Self::Error>,
            )>())?;
        }

        Ok(self.pipeline_boundary_schema())
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_distributed_partition(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            None,
            context,
        )
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_distributed_partition(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            Some(parallel),
            context,
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
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: Some(hidden.clone()),
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: ConditionalPipelineBoundary {
                    // These values are additions, not already-applied hidden
                    // state. Keep their global layer indices; the downstream
                    // partition applies only its own remaining layer entries.
                    deepstack: forward
                        .deepstack
                        .iter()
                        .map(|features| {
                            Self::decoder_deepstack_contribution(
                                features,
                                hidden,
                                forward.visual_mask.as_ref(),
                                context,
                            )
                        })
                        .collect::<Result<Vec<_>, Error>>()?,
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

#[cfg(test)]
mod paid_boundary_metadata_tests {
    use super::*;
    #[test]
    fn conditional_boundary_preserves_empty_and_multiple_deepstack_values(){
        for count in [0,3]{
            crate::boundary_metadata::tests::check(ConditionalPipelineBoundarySchema{
                hidden_size:8,deepstack_count:count},(0..count).map(|n|31+n as i32*7).collect());
        }
    }
}
