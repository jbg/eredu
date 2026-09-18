//! Composite Qwen3-VL lifecycle over the shared vision tower and ordinary Qwen decoder.

mod observation;
mod source;
mod partition_metadata;
pub(crate) use source::RetainedModelSource;

mod media_prefill;
pub use media_prefill::MediaPrefillPlan;

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend, Index,
    LinearOperator, NormalizationOperator, Parameterized, RotaryPosition, Tensor,
    multimodal::{OrderedInputPart, assemble_ordered_inputs},
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExpertPass, LayerRuntimeState,
    LayeredArchitecture, LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput,
    OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture,
    PartitionedLayeredArchitecture, RoutedExpertProvider, RoutedLayeredArchitecture,
    RuntimeStateComponents, StateLayout,
};

use crate::qwen::vision::{VisionBlock, VisionInput, VisionMode, VisionState, VisionStatic};
use crate::qwen::{self, AttentionInput};
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    media_plan::QwenVlInputPartPlan,
};

use super::{
    LocalGeometry, ModelArgs, PositionPart, mrope_embeddings, multimodal_position_ids,
    position_ids_tensor,
};

/// Stable execution-group identity for Qwen3-VL vision ingress.
pub const VISION_EXECUTION_GROUP: &str = "vision";
/// Stable execution-group identity for Qwen3-VL text decoding.
pub const TEXT_EXECUTION_GROUP: &str = "text_decoder";

/// One semantic segment in decoder order.
pub enum InputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image placeholders and their unmerged patch grids.
    Image {
        /// Placeholder IDs shaped `[1, merged_patches]`.
        tokens: &'a T,
        /// One or more `(time, height, width)` patch grids.
        grid: &'a [(i32, i32, i32)],
    },
    /// Video placeholders and their unmerged patch grids.
    Video {
        /// Placeholder IDs shaped `[1, merged_patches]`.
        tokens: &'a T,
        /// One or more `(time, height, width)` patch grids.
        grid: &'a [(i32, i32, i32)],
    },
    /// Already projected decoder-width embeddings.
    Projected {
        /// Semantic token identities.
        tokens: &'a T,
        /// Embeddings shaped `[1, sequence, hidden]`.
        embeddings: &'a T,
    },
}

/// Prepared text and optional model-native visual input.
pub struct ModelInput<'a, T> {
    /// Ordered text and media segments.
    pub parts: &'a [InputPart<'a, T>],
    /// Flattened patches for every image/video part in order.
    pub pixels: Option<&'a T>,
    /// Optional explicit text attention mask.
    pub mask: Option<&'a T>,
}

enum PreparedInputKind {
    Text(usize),
    Projected(usize, usize),
    Image(usize, usize),
    Video(usize, usize),
}

/// Architecture-owned tensor assembly for one admitted Qwen3-VL request.
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

impl<T> PreparedInput<T> {
    /// Borrows the assembled request through the canonical model input vocabulary.
    pub fn with_model_input<R>(&self, apply: impl FnOnce(ModelInput<'_, T>) -> R) -> R {
        self.with_model_input_destination(crate::decoder::identity::Metadata::new(None), apply)
            .expect("ordinary borrowed input construction is infallible")
    }

    fn with_model_input_destination<R>(
        &self,
        metadata: crate::decoder::identity::Metadata<'_>,
        apply: impl FnOnce(ModelInput<'_, T>) -> R,
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
        Ok(apply(ModelInput {
            parts: &parts,
            pixels: self.pixels.as_ref(),
            mask: None,
        }))
    }
}

/// Materializes Qwen3-VL placeholder IDs, patch grids, and ordered segments
/// from an architecture admission.
pub fn prepare_input<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenVlInputPartPlan>,
    context: &T::Context,
) -> Result<PreparedInput<T>, Error> {
    prepare_input_with_metadata(input, context, None)
}

fn prepare_input_with_metadata<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenVlInputPartPlan>,
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
    input: PreparedCompositeInput<'_, T, QwenVlInputPartPlan>,
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
    input: PreparedCompositeInput<'_, T, QwenVlInputPartPlan>,
    retain_grids: bool,
    context: &T::Context,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<PreparedInputAssembly<T>, Error> {
    metadata.controls::<(PreparedInputAssembly<T>, PreparedInputKind, [i32; 2])>()?;
    let prepared = input.prepared();
    if let Some(admitted) = input.admitted().legacy() {
        if prepared.identity() != admitted.identity() || prepared.len() != admitted.parts().len() {
            return Err(metadata.error(format_args!(
                "Qwen3-VL prepared input no longer matches its admission"
            )));
        }
    }

    let mut tokens = metadata.vector(prepared.len())?;
    let media_count = input
        .qwen_parts()
        .filter(|part| part.role == crate::media_plan::qwen::QwenPartRole::Encoded)
        .count();
    let mut grids = metadata.vector(if retain_grids { media_count } else { 0 })?;
    let mut grid_count = 0usize;
    let mut pixels = metadata.vector(media_count)?;
    let mut kinds = metadata.vector(prepared.len())?;
    let mut projected = metadata.vector(prepared.len())?;
    for (part_index, (part, plan)) in prepared.parts().iter().zip(input.qwen_parts()).enumerate() {
        match plan.role {
            crate::media_plan::qwen::QwenPartRole::Tokens => {
                let eredu_runtime::PreparedInputPayload::TokenIds(value) = part.payload() else {
                    return Err(metadata.error(format_args!(
                        "Qwen3-VL admitted text part lost its token payload"
                    )));
                };
                tokens.push(value.clone());
                kinds.push(PreparedInputKind::Text(tokens.len() - 1));
                projected.push(None);
            }
            crate::media_plan::qwen::QwenPartRole::Projected => {
                let eredu_runtime::PreparedInputPayload::Embeddings(value) = part.payload() else {
                    return Err(metadata.error(format_args!(
                        "Qwen3-VL admitted projected part lost its embedding payload"
                    )));
                };
                let positions = i32::try_from(plan.positions).map_err(|_| {
                    metadata.error(format_args!("Qwen3-VL projected span exceeds I32"))
                })?;
                tokens.push(T::full_u32(0, &[1, positions], context)?);
                kinds.push(PreparedInputKind::Projected(tokens.len() - 1, part_index));
                projected.push(Some(value.clone()));
            }
            crate::media_plan::qwen::QwenPartRole::Encoded => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(metadata.error(format_args!(
                        "Qwen3-VL admitted media part lost its tensor payload"
                    )));
                };
                let count = i32::try_from(plan.positions)
                    .map_err(|_| metadata.error(format_args!("Qwen3-VL media span exceeds I32")))?;
                let token = plan.placeholder;
                tokens.push(T::full_u32(token, &[1, count], context)?);
                if retain_grids {
                    let mut grid = metadata.vector(plan.grid.iter().count())?;
                    grid.extend(plan.grid.iter());
                    grids.push(grid);
                }
                pixels.push(value.clone());
                let token_index = tokens.len() - 1;
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
                            "Qwen3-VL media admission contains an unsupported modality"
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

impl<B, S> CompositeArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type InputPartPlan = QwenVlInputPartPlan;
    type AdmissionConfig = crate::replicated_text::SharedCompositeConfig<ModelArgs>;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.args.clone()
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
        crate::media_plan::admit_qwen_vl_input(config, input, inspector)
    }

    fn admit_prepared_input_with_metadata(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>, Error> {
        crate::media_plan::admission::qwen_vl(config, input, inspector, context)
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
                .ok_or_else(|| "Qwen3-VL projected media positions overflowed".to_owned())?
        } else {
            input.admitted().decoder_positions()
        };
        i32::try_from(positions)
            .map_err(|_| "Qwen3-VL prepared group boundary sequence exceeds i32".to_owned())
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
            .try_fold(0_u64, |total, (time, height, width)| {
                u64::try_from(time)
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
                    .and_then(|patches| total.checked_add(patches))
                    .ok_or_else(|| "Qwen3-VL continuation patch geometry overflowed".to_owned())
            })?;
        let patches = i32::try_from(patches)
            .map_err(|_| "Qwen3-VL continuation patch count exceeds i32".to_owned())?;
        Ok((patches > 0).then_some((patches, self.args.vision.hidden_size)))
    }

    fn prepared_group_collective_waves(
        &self, group:usize, input:PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,
        tensor_partitions:usize,pipeline_stages:usize,
        context:Option<&eredu_nn::workspace::WorkspaceContext>,
    )->Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>,Error> {
        let destination=crate::composite_execution::graph::Destination(context);
        destination.controls::<(&Self,usize,PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,
            usize,usize)>()?;
        self.ingress_group_collective_waves(group,input,tensor_partitions,pipeline_stages,true,destination)
    }

    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<(usize, usize), Self::Error> {
        // The tensor-parallel attention contribution precedes expert routing;
        // the routed activation contribution remains partial until its exact
        // post-exchange row reduction.
        Ok(qwen_vl_routed_tensor_reductions())
    }

    fn partition_boundary_schema(
        &self,source_group:usize,destination_group:usize,selected:&eredu_runtime::ResolvedBoundaryWireSchema,
        batch:i32,source_sequence:i32,group_sequences:&[i32],continuation:Option<(i32,i32)>,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    )->Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>,Error> {
        self.partition_boundary_schema_in(source_group,destination_group,selected,batch,source_sequence,
            group_sequences,continuation,crate::composite_execution::graph::Destination(metadata))
    }
    fn partition_boundary_values(
        &self,source_group:usize,destination_group:usize,schema:&eredu_runtime::ResolvedBoundaryWireSchema,
        hidden:&B::Tensor,forward:&Self::ForwardContext,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    )->Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>,Error> {
        self.partition_boundary_values_in(source_group,destination_group,schema,hidden,forward,
            crate::composite_execution::graph::Destination(metadata))
    }

    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        let metadata=forward.metadata.clone();
        let destination=crate::composite_execution::graph::Destination(metadata.as_ref());
        destination.controls::<(Vec<B::Tensor>,std::vec::IntoIter<B::Tensor>,PipelineBoundary<B::Tensor>,
            PipelineBoundarySchema,Option<B::Tensor>)>()?;
        if source_group == 1 && destination_group == 1 {
            if values.len() != 1 + schema.auxiliary().len() {
                return Err(destination.error(format_args!(
                    "Qwen3-VL decoder boundary has {} values, expected {}",
                    values.len(),
                    1 + schema.auxiliary().len()
                )));
            }
            let mut values = values.into_iter();
            let hidden = values.next().expect("validated decoder boundary primary");
            let schema=PipelineBoundarySchema::from_args(&self.args);
            let values=destination.collect(values)?;
            let boundary=match metadata.as_ref() {
                Some(context)=>eredu_runtime::ArchitectureBoundary::decode_with_metadata(&schema,values,context)?,
                None=>eredu_runtime::ArchitectureBoundary::decode(&schema,values).map_err(Error::backend_retained_source)?,
            };
            forward.rotary = Some((boundary.cosine, boundary.sine));
            forward.position_delta = Some(boundary.position_delta);
            forward.deepstack = boundary.deepstack;
            return Ok(Some(hidden));
        }
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let expected = 1 + schema.auxiliary().len();
        if values.len() != expected {
            return Err(destination.error(format_args!(
                "Qwen3-VL vision boundary has {} values, expected {expected}",
                values.len()
            )));
        }
        let mut values = values.into_iter();
        let hidden = values.next().expect("validated boundary primary");
        if source_group == destination_group {
            let vision = forward
                .vision_state
                .as_mut()
                .ok_or_else(|| destination.error(format_args!("Qwen3-VL continuation has no vision state")))?;
            let retained=destination.collect(vision.retained_values().take(2).cloned().chain(values))?;
            vision.replace_retained_values(retained)?;
        } else {
            forward.deepstack = destination.collect(values)?;
            forward.vision_output = Some(hidden.clone());
        }
        Ok(Some(hidden))
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
            .with_model_input_destination(metadata, |input| {
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
            .with_model_input_destination(metadata, |input| {
                self.begin_forward_with_metadata(input, state, Some(parallel), context, metadata)
            })?
    }
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Media { tokens: T },
}

/// Pinned text and shared-vision modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Ordinary Qwen embeddings, final norm, and vocabulary head.
    pub text: qwen::StaticModules<B>,
    /// Qwen shared vision patch, position, and merger modules.
    pub vision: VisionStatic<B>,
}

/// One streamable vision or ordinary Qwen text unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Shared vision transformer block.
    Vision(VisionBlock<B>),
    /// Existing neutral ordinary Qwen dense-or-MoE block.
    Text(qwen::RoutedTransformerBlock<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_components(
            group, index, unit, hidden, state, forward, pass, provider, context, observer,
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
        _pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        LayeredModel::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, provider, context,
        )
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }
    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
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
        self.forward_components_parallel(
            group, index, unit, hidden, state, forward, pass, provider, parallel, context, observer,
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
        _pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        LayeredModel::forward_unit_with_provider_parallel(
            self, group, index, unit, hidden, state, forward, provider, parallel, context,
        )
    }
}

/// Architecture-owned values retained for one complete pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    tokens: Option<T>,
    parts: Vec<PreparedPart<T>>,
    rotary: Option<(T, T)>,
    position_delta: Option<T>,
    pending_media: Option<media_prefill::PendingMedia<T>>,
    media_span: bool,
    span_assembled: bool,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
    visual_mask: Option<T>,
    // Enclosing source account survives the paid forward containers.
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}

impl<T> ForwardContext<T> {
    fn visit_values<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        for value in self.mask.iter() {
            visitor(value);
        }
        for value in self.tokens.iter() {
            visitor(value);
        }
        if let Some((cosine, sine)) = &self.rotary {
            for value in [cosine, sine] {
                visitor(value);
            }
        }
        for value in self.position_delta.iter() {
            visitor(value);
        }
        if let Some(pending) = &self.pending_media {
            for value in pending.values() {
                visitor(value);
            }
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
        if let Some(state) = &self.vision_state {
            for value in state.retained_values() {
                visitor(value);
            }
        }
        for part in &self.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    visitor(tokens);
                    visitor(embeddings);
                }
                PreparedPart::Media { tokens } => visitor(tokens),
            }
        }
    }

    fn rotary(&self) -> Result<&(T, T), Error> {
        self.rotary
            .as_ref()
            .ok_or_else(|| Error::backend("decoder rotary state is not prepared"))
    }
}

/// Request-scoped state transported while pipeline owners execute the shared
/// vision tower.
pub struct PipelineVisionState<T> {
    /// Current vision activation.
    pub hidden: T,
    parts: Vec<PreparedPart<T>>,
    rotary: (T, T),
    delta: T,
    mask: Option<T>,
    vision: Option<VisionState<T>>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
}

/// Decoder-facing values produced after the placed vision group completes.
pub struct PipelinePrepared<T> {
    /// Assembled text-width activation.
    pub hidden: T,
    /// Text mRoPE cosine.
    pub cosine: T,
    /// Text mRoPE sine.
    pub sine: T,
    /// Persisted decode position delta.
    pub position_delta: T,
    /// Optional explicit or causal mask.
    pub mask: Option<T>,
    /// Selected raw DeepStack features.
    pub deepstack: Vec<T>,
    /// Image-or-video placeholder mask used for DeepStack scatter.
    pub visual_mask: Option<T>,
}

impl<T: Clone> PipelinePrepared<T> {
    /// Converts decoder preparation into the canonical layered forward state.
    pub fn into_layered_forward(self) -> (LayeredForwardState<T, ForwardContext<T>>, T) {
        let position_delta = self.position_delta;
        (
            LayeredForwardState {
                hidden: self.hidden,
                context: ForwardContext {
                    metadata: None,
                    mask: self.mask,
                    tokens: None,
                    parts: Vec::new(),
                    rotary: Some((self.cosine, self.sine)),
                    position_delta: Some(position_delta.clone()),
                    pending_media: None,
                    media_span: false,
                    span_assembled: false,
                    vision_state: None,
                    vision_initial: None,
                    vision_output: None,
                    deepstack: self.deepstack,
                    visual_mask: self.visual_mask,
                },
            },
            position_delta,
        )
    }

    /// Recovers the transport boundary from a completed layered forward.
    pub fn from_layered_forward(
        forward: LayeredForwardState<T, ForwardContext<T>>,
        position_delta: T,
    ) -> (T, PipelineBoundary<T>) {
        let (cosine, sine) = forward
            .context
            .rotary
            .expect("completed decoder has rotary state");
        (
            forward.hidden,
            PipelineBoundary {
                cosine,
                sine,
                position_delta,
                deepstack: forward.context.deepstack,
            },
        )
    }
}

/// Family-owned schema for decoder values transported between pipeline ranks.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PipelineBoundarySchema {
    head_dim: i32,
    hidden_size: i32,
    deepstack_count: usize,
}

impl PipelineBoundarySchema {
    /// Derives the boundary schema from the normalized multimodal model.
    pub fn from_args(args: &ModelArgs) -> Self {
        Self {
            head_dim: args.text.head_dim,
            hidden_size: args.text.hidden_size,
            deepstack_count: args.vision.deepstack_layer_count(),
        }
    }

    /// Returns the configured number of DeepStack feature tensors.
    pub const fn deepstack_count(self) -> usize {
        self.deepstack_count
    }
}

/// Exact learned vision context transported between partition owners.
pub fn vision_partition_boundary_schema(
    args: &ModelArgs,
    continuation: bool,
    deepstack_count: usize,
) -> Result<eredu_runtime::BoundaryWireSchema, eredu_runtime::ArchitectureBoundaryError> {
    partition_metadata::ordinary_vision_schema(args, continuation, deepstack_count)
}

/// Exact projected-vision context consumed by the first decoder owner.
pub fn vision_dependency_boundary_schema(
    args: &ModelArgs,
) -> Result<eredu_runtime::BoundaryWireSchema, eredu_runtime::ArchitectureBoundaryError> {
    vision_partition_boundary_schema(args, false, args.vision.deepstack_layer_count())
}

impl eredu_runtime::ArchitectureBoundary for PipelineBoundarySchema {
    type Boundary<T> = PipelineBoundary<T>;

    const IDENTITY: &'static str = "qwen_vl.decoder";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        self.primary_declaration()
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        self.auxiliary_declarations()
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        Ok(PipelineBoundary {
            cosine: tensors.next().expect("validated mRoPE cosine"),
            sine: tensors.next().expect("validated mRoPE sine"),
            position_delta: tensors.next().expect("validated position delta"),
            deepstack: tensors.collect(),
        })
    }

    /// Encodes a typed boundary after validating its configured cardinality.
    fn encode<T>(
        &self,
        boundary: PipelineBoundary<T>,
    ) -> Result<
        Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,
        eredu_runtime::ArchitectureBoundaryError,
    > {
        if boundary.deepstack.len() != self.deepstack_count {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "qwen_vl.decoder.deepstack",
                expected: self.deepstack_count,
                actual: boundary.deepstack.len(),
            });
        }
        let mut values = Vec::with_capacity(3 + boundary.deepstack.len());
        values.push(eredu_runtime::ArchitectureBoundaryValue::new(
            "cosine",
            boundary.cosine,
        )?);
        values.push(eredu_runtime::ArchitectureBoundaryValue::new(
            "sine",
            boundary.sine,
        )?);
        values.push(eredu_runtime::ArchitectureBoundaryValue::new(
            "position_delta",
            boundary.position_delta,
        )?);
        for (index, tensor) in boundary.deepstack.into_iter().enumerate() {
            values.push(eredu_runtime::ArchitectureBoundaryValue::new(
                format!("deepstack.{index}"),
                tensor,
            )?);
        }
        Ok(values)
    }

    fn wire_schema_with_metadata(&self,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<eredu_runtime::BoundaryWireSchema,Error>{
        self.wire_declaration_with_metadata(context)
    }
    fn encode_with_metadata<T>(&self,boundary:Self::Boundary<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,Error>{
        crate::boundary_metadata::encode(context,Self::IDENTITY,
            [("cosine",boundary.cosine),("sine",boundary.sine),("position_delta",boundary.position_delta)],
            boundary.deepstack,self.deepstack_count,"deepstack")
    }
    fn decode_with_metadata<T>(&self,tensors:Vec<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Self::Boundary<T>,Error>{
        let ([cosine,sine,position_delta],deepstack)=crate::boundary_metadata::decode(context,Self::IDENTITY,tensors,self.deepstack_count)?;
        Ok(PipelineBoundary{cosine,sine,position_delta,deepstack})
    }

}

/// Typed immutable decoder context transported between Qwen-VL partitions.
pub struct PipelineBoundary<T> {
    /// Text mRoPE cosine.
    pub cosine: T,
    /// Text mRoPE sine.
    pub sine: T,
    /// Persisted decode position delta.
    pub position_delta: T,
    /// Per-layer DeepStack features.
    pub deepstack: Vec<T>,
}

impl<T> PipelineBoundary<T> {
    /// Splits prepared decoder state into its evolving activation and boundary.
    pub fn from_prepared(prepared: PipelinePrepared<T>) -> (T, Self) {
        (
            prepared.hidden,
            Self {
                cosine: prepared.cosine,
                sine: prepared.sine,
                position_delta: prepared.position_delta,
                deepstack: prepared.deepstack,
            },
        )
    }

    /// Reconstructs decoder state on a downstream partition.
    pub fn into_prepared(self, hidden: T) -> PipelinePrepared<T> {
        PipelinePrepared {
            hidden,
            cosine: self.cosine,
            sine: self.sine,
            position_delta: self.position_delta,
            mask: None,
            deepstack: self.deepstack,
            visual_mask: None,
        }
    }
}

/// Decoder partition input from the input owner or an upstream pipeline rank.
pub enum PipelinePartitionInput<'a, T> {
    /// Text token identities entering the architecture-owned embedding boundary.
    Tokens {
        /// Token identities.
        tokens: &'a T,
        /// Existing decoder cache offset.
        offset: i32,
        /// Persisted multimodal position delta after a media prefill.
        position_delta: Option<&'a T>,
    },
    /// Evolving activation plus typed immutable decoder context.
    Hidden {
        /// Upstream decoder activation.
        hidden: T,
        /// Family-owned boundary context.
        boundary: PipelineBoundary<T>,
    },
}

const fn qwen_vl_routed_tensor_reductions() -> (usize, usize) {
    (1, 1)
}

/// One neutral composite model for dense and MoE Qwen3-VL.
pub struct LayeredModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: crate::replicated_text::SharedCompositeConfig<ModelArgs>,
    source: Option<RetainedModelSource>,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
    partition_geometry: Option<std::sync::Arc<super::PartitionLocalGeometry>>,
    execution_graph: ExecutionGraph,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ArchitectureParameters<B> for LayeredModel<B>
{
    type DefinitionError = Error;


    fn state_layout(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Self::DefinitionError> {
match context { Some(context) => {
        if self.source.is_some() {
            return self.checked_graph(crate::decoder::identity::Metadata::new(Some(context)))?
                .state.clone_workspace(context);
        }
        match &self.parallel_geometry {
            Some(geometry) => geometry.state_layout().clone_workspace(context),
            None => super::state_layout_with_metadata(&self.args, context),
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
        if self.source.is_some() {
            return self.source_state_identity(state, topology, context);
        }
        super::state_identity_with_metadata(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
            context,
        )
    }, None => {
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
        .map_err(|error| Error::backend(error.to_string()))
    } }
}


    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Self::DefinitionError> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(&Self, std::borrow::Cow<'_, ArchitectureParameterDescription>)>()?;
        if self.source.is_some() {
            return Ok(std::borrow::Cow::Borrowed(&self.checked_graph(metadata)?.description));
        }
        self.parameter_description_impl(context).map(std::borrow::Cow::Owned)
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        Ok(super::static_recipes(source))
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
        visitor.visit_mut("vision", &mut self.static_modules.vision)?;
        visitor.visit_mut("embedding", &mut self.static_modules.text.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.text.norm)?;
        if let Some(head) = &mut self.static_modules.text.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    fn build_execution_graph(
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<ExecutionGraph, Error> {
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<ExecutionGraph>()?;
        let mut groups = destination.vector(2)?;
        groups.push(destination.group(VISION_EXECUTION_GROUP, &[])?);
        groups.push(destination.group(TEXT_EXECUTION_GROUP, &[VISION_EXECUTION_GROUP])?);
        destination.finish(groups, TEXT_EXECUTION_GROUP)
    }

    fn begin_forward_with_metadata<S>(
        &mut self,
        input: ModelInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        metadata.controls::<(
            Option<&B::ParallelContext>,
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            Vec<B::Tensor>,
            [Index; 3],
        )>()?;
        let expected = match self.partition_geometry.as_deref() {
            Some(geometry) => match metadata.context() {
                Some(context) => geometry.local_state_layout_with_metadata(context)?,
                None => geometry
                    .local_state_layout()
                    .map_err(|error| Error::backend(error.to_string()))?,
            },
            None => match parallel {
                Some(_) => {
                    let layout = self.parallel_geometry.as_ref()
                        .ok_or_else(|| metadata.error(format_args!("Qwen3-VL model has no local geometry")))?
                        .state_layout();
                    match metadata.context() {
                        Some(context) => layout.clone_workspace(context)?,
                        None => layout.clone(),
                    }
                }
                None => match metadata.context() {
                    Some(metadata) => {
                        <Self as eredu_runtime::ArchitectureParameters<B>>::state_layout(
                            self, Some(metadata),
                        )?
                    }
                    None => self.state_layout_impl()?,
                },
            },
        };
        if state.layout() != &expected {
            return Err(
                crate::composite_execution::graph::Destination(metadata.context())
                    .error(format_args!("Qwen3-VL runtime state layout mismatch")),
            );
        }
        let owns_position_delta = expected.layer(0).is_some_and(|policy| {
            policy
                .fixed_state()
                .iter()
                .any(|tensor| tensor.role == StateTensorRole::PositionDelta)
        });
        let (parts, grids) = self.prepare_parts_with_metadata(input.parts, parallel, context, metadata)?;
        let visual_mask = self.prepared_visual_mask_with_metadata(&parts, context, metadata)?;
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(metadata.error(format_args!(
                "Qwen3-VL pixels and media metadata must appear together"
            )));
        }
        let (vision_initial, vision_state) = match input.pixels {
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
        let assembled = self.assemble_with_metadata(&parts, None, context, metadata);
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let state_layer = state.layer(0).map_err(|cause| metadata.source(cause))?;
        let offset = state_layer.position();
        let mut position_parts = metadata.vector(input.parts.len())?;
        for part in input.parts {
            match part {
                InputPart::Text(tokens) | InputPart::Projected { tokens, .. } => {
                    position_parts.push(PositionPart::Text(tokens.dim(1)))
                }
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => {
                    position_parts.push(PositionPart::Media(grid))
                }
            }
        }
        let (mut positions, computed_delta) =
            super::positions::multimodal_position_ids_with_metadata(
                &position_parts,
                self.args.vision.spatial_merge_size,
                sequence,
                metadata.context(),
            )?;
        if media && offset != 0 {
            return Err(metadata.error(format_args!(
                "Qwen3-VL media input cannot append to a populated cache"
            )));
        }
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = super::positions::position_ids_tensor_with_metadata::<B::Tensor>(
            std::array::from_fn(|axis| positions[axis].as_slice()),
            context,
            metadata.context(),
        )?;
        let position_delta = if owns_position_delta {
            let delta = state_layer
                .fixed_component(StateTensorRole::PositionDelta)
                .map_err(|cause| metadata.source(cause))?;
            if media || delta.is_none() {
                *delta = Some(B::Tensor::full_i32(computed_delta, &[1], context)?);
            } else if let Some(delta) = delta.as_ref() {
                positions = positions.add(delta, context)?;
            }
            delta
                .as_ref()
                .expect("Qwen3-VL installs position delta")
                .clone()
        } else {
            // PositionDelta is architecture-global state owned by text layer zero.
            // A later PP stage still prepares its rank-local media group from the
            // immutable request, but receives the authoritative persisted value
            // with the decoder boundary before it executes any text unit.
            B::Tensor::full_i32(computed_delta, &[1], context)?
        };
        let rotary = super::positions::mrope_embeddings_with_metadata(
            &positions,
            self.args.text.head_dim,
            self.args.text.rope_theta,
            &self.args.mrope_section,
            context,
            metadata.context(),
        )?;
        let mask = if let Some(mask) = input.mask {
            Some(mask.clone())
        } else if sequence > 1 {
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let (assembled_tokens, assembled_hidden, assembled_error) = match assembled {
            Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
            Err(error) => (None, None, Some(error)),
        };
        let hidden = vision_initial
            .as_ref()
            .cloned()
            .or(assembled_hidden)
            .ok_or_else(|| {
                assembled_error
                    .unwrap_or_else(|| metadata.error(format_args!("empty Qwen3-VL input")))
            })?;
        let deepstack_count = if media {
            0
        } else {
            self.args.vision.deepstack_layer_count()
        };
        let mut deepstack = metadata.vector(deepstack_count)?;
        for _ in 0..deepstack_count {
            deepstack.push(hidden.zeros_like(context)?);
        }
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                metadata: metadata.context().cloned(),
                mask,
                tokens: assembled_tokens,
                parts,
                rotary: Some(rotary),
                position_delta: Some(position_delta),
                pending_media: None,
                media_span: false,
                span_assembled: false,
                vision_state,
                vision_initial,
                vision_output: None,
                deepstack,
                visual_mask,
            },
        })
    }

    fn add_deepstack(
        &self,
        index: usize,
        output: B::Tensor,
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let Some(features) = forward.deepstack.get(index) else {
            return Ok(output);
        };
        instrumentation.observe("deepstack.input", &output)?;
        let output = if features.shape() == output.shape() {
            if instrumentation.enabled() {
                output.add(
                    &instrumentation.apply("deepstack.output", features.clone())?,
                    context,
                )?
            } else {
                output.add(features, context)?
            }
        } else {
            let source = features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
            let contribution = output.zeros_like(context)?.masked_scatter(
                forward
                    .visual_mask
                    .as_ref()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL visual mask"))?,
                &source,
                context,
            )?;
            output.add(
                &instrumentation.apply("deepstack.output", contribution)?,
                context,
            )?
        };
        instrumentation.apply("deepstack.residual", output)
    }

    fn text_state_ordinal(&self, global_unit: usize) -> Result<usize, Error> {
        let Some(geometry) = self.partition_geometry.as_deref() else {
            return Ok(global_unit);
        };
        let owned = geometry.text_units();
        if !owned.contains(&global_unit) {
            return Err(Error::backend(format!(
                "Qwen3-VL text unit {global_unit} is outside local state ownership {owned:?}"
            )));
        }
        Ok(global_unit - owned.start)
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_distributed_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, PipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if state.layout() != expected {
            return Err(Error::backend("Qwen3-VL partition state layout mismatch"));
        }
        let local_state_ordinal = 0;
        let offset = state
            .layer(local_state_ordinal)
            .map_err(Error::backend)?
            .position();
        let persisted_delta = if matches!(&input, LayeredPartitionInput::Tokens(_)) {
            state
                .layer(local_state_ordinal)
                .map_err(Error::backend)?
                .fixed_component(StateTensorRole::PositionDelta)
                .map_err(Error::backend)?
                .clone()
        } else {
            None
        };
        let (input, batch, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => (
                PipelinePartitionInput::Tokens {
                    tokens,
                    offset,
                    position_delta: persisted_delta.as_ref(),
                },
                tokens.dim(0),
                tokens.dim(1),
            ),
            LayeredPartitionInput::Hidden {
                hidden,
                auxiliary: boundary,
            } => {
                let batch = hidden.dim(0);
                let sequence = hidden.dim(1);
                (
                    PipelinePartitionInput::Hidden { hidden, boundary },
                    batch,
                    sequence,
                )
            }
        };
        let (forward, position_delta) = self
            .begin_routed_text_partition(input, mask, batch, sequence, offset, parallel, context)?;
        if first_state_ordinal == 0 {
            *state
                .layer(local_state_ordinal)
                .map_err(Error::backend)?
                .fixed_component(StateTensorRole::PositionDelta)
                .map_err(Error::backend)? = Some(position_delta);
        }
        Ok(forward)
    }

    /// Prepares one routed decoder partition, including DeepStack defaults,
    /// shape validation, and architecture-owned causal masking.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_routed_text_partition(
        &mut self,
        input: PipelinePartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        batch: i32,
        sequence: i32,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            B::Tensor,
        ),
        Error,
    > {
        let mut prepared = self.begin_partition_text_inner(input, parallel, context)?;
        let deepstack_count = self.args.vision.deepstack_layer_count();
        if prepared.deepstack.is_empty() {
            let zero = prepared.hidden.zeros_like(context)?;
            prepared.deepstack = vec![zero; deepstack_count];
        } else if prepared.deepstack.len() != deepstack_count {
            return Err(Error::backend(format!(
                "Qwen3-VL prepared {} DeepStack tensors, expected {deepstack_count}",
                prepared.deepstack.len()
            )));
        }
        let expected = [batch, sequence, self.args.text.hidden_size];
        if prepared.hidden.shape() != expected {
            return Err(Error::backend(format!(
                "Qwen3-VL decoder input is shaped {:?}, expected {expected:?}",
                prepared.hidden.shape()
            )));
        }
        prepared.mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => prepared.mask,
        };
        Ok(prepared.into_layered_forward())
    }

    fn begin_partition_text_inner(
        &mut self,
        input: PipelinePartitionInput<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelinePrepared<B::Tensor>, Error> {
        match input {
            PipelinePartitionInput::Tokens {
                tokens,
                offset,
                position_delta,
            } => {
                let parts = [InputPart::Text(tokens)];
                let input = ModelInput {
                    parts: &parts,
                    pixels: None,
                    mask: None,
                };
                let state = match parallel {
                    Some(parallel) => self.begin_pipeline_parallel(
                        input,
                        offset,
                        position_delta,
                        parallel,
                        context,
                    ),
                    None => self.begin_pipeline(input, offset, position_delta, context),
                }?;
                self.finish_pipeline(state, parallel, context)
            }
            PipelinePartitionInput::Hidden { hidden, boundary } => {
                Ok(boundary.into_prepared(hidden))
            }
        }
    }

    /// Finishes the serial text output boundary.
    pub fn finish_partition_text(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_pipeline_logits(hidden, context)
    }

    /// Finishes the tensor-parallel text output boundary.
    pub fn finish_partition_text_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
    }

    /// Builds unloaded modules with their canonical checkpoint identities.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        Self::new_with_config(
            crate::replicated_text::CompositeModelConfig::Owned(args),
            context,
        )
    }

    pub(crate) fn config_owner(&self) -> &crate::replicated_text::SharedCompositeConfig<ModelArgs> {
        &self.args
    }

    pub(crate) fn new_with_config(
        args: crate::replicated_text::CompositeModelConfig<ModelArgs>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(crate::replicated_text::CompositeModelConfig<ModelArgs>, Result<Self, Error>)>()?;
        let args = args.into_shared(B::construction_metadata(context))?;
        Self::from_shared_config(args, None, None, None, context)
    }

    /// Builds the composite graph with planner-derived text and vision modules.
    pub fn new_parallel(
        args: ModelArgs,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let args = crate::replicated_text::SharedCompositeConfig::new(args, B::construction_metadata(context))?;
        Self::from_shared_config(args, Some(std::sync::Arc::new(geometry)), None, None, context)
    }

    // Ordinary and retained-source constructors share the actual module workers.
    // Cold construction aliases validated geometry and projects its config owner;
    // it never clones the nested configuration or reconstructs local geometry.
    fn from_shared_config(
        args: crate::replicated_text::SharedCompositeConfig<ModelArgs>,
        parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
        partition_geometry: Option<std::sync::Arc<super::PartitionLocalGeometry>>,
        source: Option<RetainedModelSource>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(Self, crate::replicated_text::SharedCompositeConfig<ModelArgs>,
            Option<std::sync::Arc<LocalGeometry>>, Option<std::sync::Arc<super::PartitionLocalGeometry>>,
            Option<RetainedModelSource>, qwen::StaticModules<B>, VisionStatic<B>,
            crate::replicated_text::SharedCompositeConfig<crate::qwen::vision::VisionConfig>,
            Result<Self, Error>)>()?;
        crate::operator_requirements::require_with_metadata::<B>(
            "Qwen3-VL", crate::operator_requirements::QWEN_VL, metadata,
        )?;
        match metadata.context().filter(|context| context.uses_checked_metadata()) {
            Some(context) => args.vision.validate_for_with_metadata(VisionMode::DeepStack, context)?,
            None => args.vision.validate_for(VisionMode::DeepStack).map_err(Error::backend)?,
        }
        let text = match &parallel_geometry {
            Some(geometry) => qwen::StaticModules::new_parallel(&args.text, geometry.text(), context)?,
            None => qwen::StaticModules::new(&args.text, context)?,
        };
        let vision_config = crate::replicated_text::CompositeModelConfig::Retained(
            args.project(|args| &args.vision, metadata.context())?,
        );
        let vision = match &parallel_geometry {
            Some(geometry) => VisionStatic::new_parallel_with_config(
                vision_config, "model.visual", geometry.merger_widths(), context,
            )?,
            None => VisionStatic::new_with_config(vision_config, "model.visual", context)?,
        };
        let execution_graph = match &source {
            Some(source) => source.execution_graph(metadata.context())?,
            None => Self::build_execution_graph(metadata.context())?,
        };
        Ok(Self { args, source, static_modules: StaticModules { text, vision },
            parallel_geometry, partition_geometry, execution_graph })
    }

    /// Retains the already validated PP/TP/EP-local geometry used by placed
    /// unit factories while leaving static modules under their original
    /// checkpoint-global parameter authority.
    pub fn with_partition_geometry(mut self, geometry: super::PartitionLocalGeometry) -> Self {
        self.partition_geometry = Some(std::sync::Arc::new(geometry));
        self
    }

    /// Returns normalized nested text and vision policy.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Describes shared-vision and text-decoder parameters with canonical
    /// graph-unit ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let metadata = B::construction_metadata(context);
        let graph = self.canonical_execution_graph(metadata)?;
        let destination = crate::composite_execution::graph::Destination(metadata);
        destination.controls::<(
            ArchitectureParameterDescription,
            Vec<OwnedParameterGroupSpec>,
            Vec<eredu_runtime::ParameterGroupSpec>,
        )>()?;
        destination.controls::<[usize; 2]>()?;
        let counts = [
            self.canonical_group_unit_count(0, metadata)?,
            self.canonical_group_unit_count(1, metadata)?,
        ];
        let layout = destination.layout(&graph, &counts)?;
        // A partition-local model contains already-local tensor shapes. Its
        // parameter authority must nevertheless continue to describe the
        // immutable checkpoint-global groups selected at admission; applying
        // TP placement to the local shapes would shard them a second time.
        destination.controls::<(
            Option<Result<(crate::decoder::StaticModules<B>, VisionStatic<B>), Error>>,
            Option<(crate::decoder::StaticModules<B>, VisionStatic<B>)>,
        )>()?;
        let global_static = self.parallel_geometry.is_some().then(|| {
            Ok::<_, Error>((
                qwen::StaticModules::new(&self.args.text, context)?,
                VisionStatic::new_with_config(
                    crate::replicated_text::CompositeModelConfig::Retained(
                        self.args.project(|args| &args.vision, metadata)?,
                    ),
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
            &self.args.text.parameter_root,
            metadata,
        )
        .map_err(|cause| cause.into_neural())?;
        let vision_static =
            crate::qwen::vision::owned_static_parallel_parameter_groups_with_metadata::<B>(
                vision_modules,
                &self.args.vision,
                "model.visual",
                destination.group_id(VISION_EXECUTION_GROUP)?,
                "vision",
                metadata,
            )?;
        let mut owned = destination.vector(text_static.len())?;
        for (index, group) in text_static.into_iter().enumerate() {
            let roles: &[&str] = if index == 0 && self.args.text.tie_word_embeddings {
                &["embedding", "output"]
            } else {
                match index {
                    0 => &["embedding"],
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
        for (group_index, &count) in counts.iter().enumerate() {
            let group_id = layout.group_id(group_index).expect("Qwen3-VL layout group");
            for index in 0..count {
                let unit = if self.parallel_geometry.is_some() {
                    match group_index {
                        0 => Unit::Vision(VisionBlock::new_with_root(
                            &self.args.vision,
                            "model.visual",
                            index,
                            context,
                        )?),
                        1 => Unit::Text(qwen::new_routed_block(&self.args.text, index, context)?),
                        _ => unreachable!("Qwen3-VL has two execution groups"),
                    }
                } else {
                    self.construct_unit(group_index, index, context)?
                };
                let groups = match unit {
                    Unit::Vision(block) => {
                        crate::qwen::vision::block_parallel_parameter_groups_with_metadata(
                            &block,
                            &self.args.vision,
                            "model.visual",
                            index,
                            metadata,
                        )
                    }
                    Unit::Text(block) => {
                        qwen::routed_layer_parallel_parameter_groups_with_metadata(
                            &block,
                            &self.args.text,
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

    fn canonical_group_unit_count(
        &self,
        group: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Error> {
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<(usize, &Self)>()?;
        match group {
            0 => Ok(self.args.vision.layer_count()),
            1 => usize::try_from(self.args.text.num_hidden_layers)
                .map_err(|cause| destination.error(format_args!("{cause}"))),
            _ => Err(destination.error(format_args!("Qwen3-VL has two execution groups"))),
        }
    }

    fn canonical_execution_graph(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<ExecutionGraph, Error> {
        match context {
            Some(context) => self.execution_graph.clone_with_metadata(context),
            None => Ok(self.execution_graph.clone()),
        }
    }

    /// Returns replicated or planner-derived decoder state geometry.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(
                || super::state_layout(&self.args).map_err(Error::backend),
                Ok,
            )
    }

    /// Applies the architecture-owned tensor-parallel target output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
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

    /// Shares planner-owned geometry with placed unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical vision or text unit using this model's
    /// replicated or planner-derived local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        let count = match group {
            0 => self.args.vision.layer_count(),
            1 => usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend)?,
            _ => return Err(Error::backend("Qwen3-VL has two execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Qwen3-VL unit is outside its group"));
        }
        match group {
            0 => {
                let geometry = self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.vision_block(index));
                Ok(Unit::Vision(match geometry {
                    Some((heads, intermediate)) => VisionBlock::new_parallel_with_root(
                        &self.args.vision,
                        "model.visual",
                        index,
                        heads,
                        intermediate,
                        context,
                    )?,
                    None => VisionBlock::new_with_root(
                        &self.args.vision,
                        "model.visual",
                        index,
                        context,
                    )?,
                }))
            }
            1 => {
                let local = self
                    .partition_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.text_block(index))
                    .or_else(|| {
                        self.parallel_geometry
                            .as_ref()
                            .and_then(|geometry| geometry.text().block(index))
                    });
                Ok(Unit::Text(match local {
                    Some(local) if self.args.text.is_moe() => {
                        qwen::new_partitioned_routed_block(&self.args.text, local, index, context)?
                    }
                    Some(local) => qwen::new_routed_block(local, index, context)?,
                    None => qwen::new_routed_block(&self.args.text, index, context)?,
                }))
            }
            _ => unreachable!(),
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
            return Err(metadata.error(format_args!("Qwen3-VL input has no ordered parts")));
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
                            &mut self.static_modules.text.embeddings, tokens,
                            EmbeddingLookupPolicy::Strict, parallel, context,
                        )?,
                        None => self.static_modules.text.embeddings.forward(tokens, context)?,
                    },
                }),
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(metadata
                            .error(format_args!("Qwen3-VL projected input geometry mismatch")));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
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
        self.prepare_parts_with_metadata(parts, Some(parallel), context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)))
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
                _ => 0,
            })
            .sum::<i32>();
        match vision {
            Some(value) if value.shape() == [1, media_tokens, self.args.text.hidden_size] => {}
            None if media_tokens == 0 => {}
            Some(value) => {
                return Err(metadata.error(format_args!(
                    "Qwen3-VL vision output {:?} does not match {media_tokens} placeholders",
                    value.shape()
                )));
            }
            None => {
                return Err(metadata.error(format_args!(
                    "Qwen3-VL media placeholders require vision output"
                )));
            }
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
                self.args.text.hidden_size,
                context,
                metadata,
            ),
            None => assemble_ordered_inputs(&ordered, self.args.text.hidden_size, context),
        }
    }

    fn prepared_visual_mask(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        self.prepared_visual_mask_with_metadata(
            parts,
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }
    fn prepared_visual_mask_with_metadata(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<Option<B::Tensor>, Error> {
        metadata.controls::<(Vec<B::Tensor>, Option<B::Tensor>)>()?;
        if self.args.vision.deepstack_layer_count() == 0 {
            return Ok(None);
        }
        let mut tokens = metadata.vector(parts.len())?;
        tokens.extend(parts.iter().map(|part| match part {
            PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.clone(),
        }));
        let tokens = B::Tensor::concatenate(&tokens, 1, context)?;
        Ok(Some(
            tokens
                .equal_i32(self.args.image_token_id, context)?
                .logical_or(
                    &tokens.equal_i32(self.args.video_token_id, context)?,
                    context,
                )?,
        ))
    }

    fn ensure_visual_mask(
        &self,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        if !forward.deepstack.is_empty() && forward.visual_mask.is_none() {
            forward.visual_mask = match forward.tokens.as_ref() {
                Some(tokens) => Some(
                    tokens
                        .equal_i32(self.args.image_token_id, context)?
                        .logical_or(
                            &tokens.equal_i32(self.args.video_token_id, context)?,
                            context,
                        )?,
                ),
                None if !forward.parts.is_empty() => self.prepared_visual_mask_with_metadata(
                    &forward.parts,
                    context,
                    crate::decoder::identity::Metadata::new(
                        forward
                            .metadata
                            .as_ref()
                            .or_else(|| B::construction_metadata(context)),
                    ),
                )?,
                None => {
                    return Err(
                        crate::decoder::identity::Metadata::new(B::construction_metadata(context))
                            .error(format_args!(
                                "Qwen3-VL compact DeepStack state has no visual-mask authority"
                            )),
                    );
                }
            };
        }
        Ok(())
    }

    fn finish_logits(
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

    /// Prepares a pipeline request without binding it to a concrete cache
    /// container. The caller supplies the current token offset and persisted
    /// multimodal delta from the generic state slots.
    pub fn begin_pipeline<'a>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        offset: i32,
        persisted_delta: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_inner(input, offset, persisted_delta, None, context)
    }

    /// Prepares the same placed request with rank-local vocabulary lookup.
    pub fn begin_pipeline_parallel<'a>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        offset: i32,
        persisted_delta: Option<&B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_inner(input, offset, persisted_delta, Some(parallel), context)
    }

    fn begin_pipeline_inner<'a>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        offset: i32,
        persisted_delta: Option<&B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelineVisionState<B::Tensor>, Error> {
        let (parts, grids) = match parallel {
            Some(parallel) => self.prepare_parts_parallel(input.parts, parallel, context)?,
            None => self.prepare_parts(input.parts, context)?,
        };
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(Error::backend(
                "Qwen3-VL pixels and media metadata must appear together",
            ));
        }
        if media && offset != 0 {
            return Err(Error::backend(
                "Qwen3-VL media input cannot append to a populated cache",
            ));
        }
        let (vision_initial, vision) = match input.pixels {
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
        let position_parts = input
            .parts
            .iter()
            .map(|part| match part {
                InputPart::Text(tokens) | InputPart::Projected { tokens, .. } => {
                    PositionPart::Text(tokens.dim(1))
                }
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => {
                    PositionPart::Media(grid)
                }
            })
            .collect::<Vec<_>>();
        let (mut positions, computed_delta) = multimodal_position_ids(
            &position_parts,
            self.args.vision.spatial_merge_size,
            sequence,
        )
        .map_err(Error::backend)?;
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = position_ids_tensor::<B::Tensor>(&positions, context)?;
        let delta = match (media, persisted_delta) {
            (false, Some(delta)) => delta.clone(),
            _ => B::Tensor::full_i32(computed_delta, &[1], context)?,
        };
        if !media {
            positions = positions.add(&delta, context)?;
        }
        let rotary = mrope_embeddings(
            &positions,
            self.args.text.head_dim,
            self.args.text.rope_theta,
            &self.args.mrope_section,
            context,
        )?;
        let mask = if let Some(mask) = input.mask {
            Some(mask.clone())
        } else if sequence > 1 {
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let assembled = self.assemble(&parts, None, context);
        let hidden = match vision_initial {
            Some(hidden) => hidden,
            None => assembled?.embeddings,
        };
        Ok(PipelineVisionState {
            hidden,
            parts,
            rotary,
            delta,
            mask,
            vision,
            vision_output: None,
            deepstack: Vec::new(),
        })
    }

    /// Whether this request owns visual component work.
    pub fn pipeline_vision_active(state: &PipelineVisionState<B::Tensor>) -> bool {
        state.vision.is_some()
    }

    /// Exports all request tensors needed by a downstream vision owner.
    pub fn pipeline_retained_values(state: &PipelineVisionState<B::Tensor>) -> Vec<B::Tensor> {
        let mut values = vec![
            state.hidden.clone(),
            state.rotary.0.clone(),
            state.rotary.1.clone(),
            state.delta.clone(),
        ];
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

    /// Replaces request tensors with values transported from the previous
    /// component owner. Structure is validated against the locally rebuilt
    /// parameter-free request state.
    pub fn replace_pipeline_retained_values(
        state: &mut PipelineVisionState<B::Tensor>,
        values: Vec<B::Tensor>,
    ) -> Result<(), Error> {
        let fixed = 4
            + usize::from(state.mask.is_some())
            + state
                .parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Text { .. } => 2,
                    PreparedPart::Media { .. } => 1,
                })
                .sum::<usize>();
        let minimum = fixed + usize::from(state.vision.is_some()) * 2;
        if values.len() < minimum || (state.vision.is_none() && values.len() != fixed) {
            return Err(Error::backend(format!(
                "Qwen3-VL pipeline continuation received {} tensors, expected at least {minimum}",
                values.len(),
            )));
        }
        let mut values = values.into_iter();
        state.hidden = values.next().expect("validated hidden");
        state.rotary.0 = values.next().expect("validated cosine");
        state.rotary.1 = values.next().expect("validated sine");
        state.delta = values.next().expect("validated delta");
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
        state: &mut PipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        let vision = state
            .vision
            .as_mut()
            .ok_or_else(|| Error::backend("missing Qwen3-VL pipeline vision state"))?;
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
        state: &mut PipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        if state.vision_output.is_some() || state.vision.is_none() {
            return Ok(());
        }
        if let Some(vision) = &mut state.vision {
            let output = match parallel {
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
            };
            state.vision_output = Some(output.embeddings);
            state.deepstack = output.deepstack_features;
        }
        Ok(())
    }

    /// Returns the completed placed projector output.
    pub fn pipeline_vision_output(state: &PipelineVisionState<B::Tensor>) -> Option<&B::Tensor> {
        state.vision_output.as_ref()
    }

    /// Replaces the completed placed projector output before decoder assembly.
    pub fn replace_pipeline_vision_output(
        state: &mut PipelineVisionState<B::Tensor>,
        output: B::Tensor,
    ) {
        state.vision_output = Some(output);
    }

    /// Finishes the vision projector and assembles decoder-width input.
    pub fn finish_pipeline(
        &mut self,
        mut state: PipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelinePrepared<B::Tensor>, Error> {
        self.complete_pipeline_vision(&mut state, parallel, context)?;
        let assembled = self.assemble(&state.parts, state.vision_output.as_ref(), context)?;
        let visual_mask = if state.deepstack.is_empty() {
            None
        } else {
            Some(
                assembled
                    .token_ids
                    .equal_i32(self.args.image_token_id, context)?
                    .logical_or(
                        &assembled
                            .token_ids
                            .equal_i32(self.args.video_token_id, context)?,
                        context,
                    )?,
            )
        };
        let deepstack = match visual_mask.as_ref() {
            Some(mask) => state
                .deepstack
                .into_iter()
                .map(|features| {
                    assembled.embeddings.zeros_like(context)?.masked_scatter(
                        mask,
                        &features.index(&[Index::At(0), Index::Full, Index::Full], context)?,
                        context,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?,
            None => state.deepstack,
        };
        Ok(PipelinePrepared {
            hidden: assembled.embeddings,
            cosine: state.rotary.0,
            sine: state.rotary.1,
            position_delta: state.delta,
            mask: state.mask,
            deepstack,
            visual_mask: None,
        })
    }

    /// Applies the shared final norm and vocabulary projection.
    pub fn finish_pipeline_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_logits(hidden, context)
    }

    /// Executes one unit while routing ordinary-Qwen MoE banks through a
    /// runtime-owned provider. Vision units retain the shared vision path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                context,
            ),
            (1, Unit::Text(block)) => {
                if forward
                    .deepstack
                    .get(index)
                    .is_some_and(|features| !features.shape().iter().eq(hidden.shape()))
                {
                    self.ensure_visual_mask(forward, context)?;
                }
                let state_ordinal = self.text_state_ordinal(index)?;
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let mask = forward.mask.as_ref();
                let (cosine, sine) = forward.rotary()?;
                let output = block.forward_routed_observed(
                    index,
                    AttentionInput {
                        hidden,
                        mask,
                        cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings { cosine, sine }),
                    },
                    pass,
                    provider,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::disabled(),
                    None,
                )?;
                let output = self.add_deepstack(
                    index,
                    output,
                    forward,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::disabled(),
                )?;
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL unit/group mismatch")),
        }
    }

    /// Executes one local unit through the runtime-owned routed expert provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider_parallel<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block_parallel(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                parallel,
                context,
            ),
            (1, Unit::Text(block)) => {
                if forward
                    .deepstack
                    .get(index)
                    .is_some_and(|features| !features.shape().iter().eq(hidden.shape()))
                {
                    self.ensure_visual_mask(forward, context)?;
                }
                let state_ordinal = self.text_state_ordinal(index)?;
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let mask = forward.mask.as_ref();
                let (cosine, sine) = forward.rotary()?;
                let output = block.forward_routed_parallel_observed(
                    index,
                    AttentionInput {
                        hidden,
                        mask,
                        cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings { cosine, sine }),
                    },
                    pass,
                    provider,
                    parallel,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::disabled(),
                    None,
                )?;
                let output = self.add_deepstack(
                    index,
                    output,
                    forward,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::disabled(),
                )?;
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL parallel unit/group mismatch")),
        }
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn media_prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata=crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(&Self,Option<&eredu_nn::workspace::WorkspaceContext>,usize,usize,String,Vec<eredu_runtime::layered::PrefillObservationDeclaration>,std::ops::Range<usize>,Option<eredu_runtime::RoutedObservationPoints>,Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>,Error>)>()?;

        // The validated source retains its complete MRoPE/DeepStack placement; only decoder rows are declared.
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

        // Only the actual group1 ordinary text target is declared. Its KV
        // prefix and persisted position delta retain causal rotary offsets;
        // media/deepstack assembly and vision hooks are outside this proof.
        let units = <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 1, metadata_context)?;
        let mut declarations = crate::decoder::ordinary_prefill_observation_declarations(
            (0..units).map(|index| <Self as LayeredArchitecture<B, S>>::unit_path(self, 1, index, metadata_context)),
            true, metadata_context)?;
        // Same target bank invocation as observed execution; its expert equations are row-local.
        for index in 0..units {
            let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, 1, index, metadata_context)?;
            if let Some(points) = self.args.text.routed_observation_points(&path, index, metadata_context)? {
                crate::decoder::append_routed_prefill_observations(&mut declarations, &points, metadata_context)?;
            }
        }
        Ok(declarations)
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
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
        let pass = <Self as RoutedLayeredArchitecture<B, S>>::expert_pass_for_unit(
            self, group, index, hidden, forward,
        );
        self.forward_components(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            observer,
        )
    }
    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.finish_components(hidden, None, context, observer)
    }

    type Input<'a> = ModelInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::segmented_token_shape(input.parts.iter().map(|part| match part {
            InputPart::Text(tokens)
            | InputPart::Image { tokens, .. }
            | InputPart::Video { tokens, .. }
            | InputPart::Projected { tokens, .. } => *tokens,
        }))
        .map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        if group == 0 {
            crate::transport::vision_declaration().into_owned()
        } else {
            crate::transport::decoder()
        }
    }

    fn group_transport_matches(
        &self,
        group: usize,
        expected: &eredu_runtime::ArchitectureGroupTransport,
    ) -> bool {
        if group == 0 {
            crate::transport::vision_declaration().matches(expected)
        } else {
            crate::transport::decoder_declaration().matches(expected)
        }
    }

    fn primary_execution_group(&self) -> &str {
        TEXT_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(1, layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            &self.execution_graph,
        ))
    }

    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        self.canonical_group_unit_count(group, metadata_context)
    }



    fn unit_path(&self,group:usize,index:usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>)->Result<String,Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        self.unit_path_in(group,index,crate::composite_execution::graph::Destination(metadata_context))
    }

    fn group_input_observation_path(&self,group:usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>)->Result<Option<String>,Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        Self::group_path_in(group==1,eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH,
            crate::composite_execution::graph::Destination(metadata_context))
    }

    fn group_output_observation_path(&self,group:usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>)->Result<Option<String>,Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        Self::group_path_in(group==0,eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH,
            crate::composite_execution::graph::Destination(metadata_context))
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
    ) -> Result<B::Tensor, Self::Error> {
        match group {
            0 => Ok(forward.vision_initial.as_ref().unwrap_or(initial).clone()),
            1 => {
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
                forward.visual_mask = if forward.deepstack.is_empty() {
                    None
                } else {
                    Some(
                        assembled
                            .token_ids
                            .equal_i32(self.args.image_token_id, context)?
                            .logical_or(
                                &assembled
                                    .token_ids
                                    .equal_i32(self.args.video_token_id, context)?,
                                context,
                            )?,
                    )
                };
                if let Some(visual_mask) = forward.visual_mask.as_ref() {
                    for features in &mut forward.deepstack {
                        if !features.shape().iter().eq(assembled.embeddings.shape()) {
                            *features = assembled.embeddings.zeros_like(context)?.masked_scatter(
                                visual_mask,
                                &features
                                    .index(&[Index::At(0), Index::Full, Index::Full], context)?,
                                context,
                            )?;
                        }
                    }
                }
                forward.tokens = Some(assembled.token_ids);
                Ok(assembled.embeddings)
            }
            _ => Err(Error::backend("invalid Qwen3-VL execution group")),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 1 || (group == 0 && forward.vision_state.is_some())
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            1 => index,
            _ => index,
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
        match (group, unit) {
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                context,
            ),
            (1, Unit::Text(block)) => {
                if forward
                    .deepstack
                    .get(index)
                    .is_some_and(|features| !features.shape().iter().eq(hidden.shape()))
                {
                    self.ensure_visual_mask(forward, context)?;
                }
                let state_ordinal = self.text_state_ordinal(index)?;
                let output = block.forward(
                    AttentionInput {
                        hidden,
                        mask: forward.mask.as_ref(),
                        cache: Some(state.layer(state_ordinal).map_err(|error| {
                            Error::backend(format!(
                                "Qwen3-VL text group {group} unit {index} cache: {error}"
                            ))
                        })?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings {
                            cosine: &forward.rotary()?.0,
                            sine: &forward.rotary()?.1,
                        }),
                    },
                    context,
                )?;
                let output = self.add_deepstack(
                    index,
                    output,
                    forward,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::disabled(),
                )?;
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL unit/group mismatch")),
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
        if group == 0 {
            if let Some(vision_state) = forward.vision_state.as_mut() {
                let output = self
                    .static_modules
                    .vision
                    .finish(hidden, vision_state, context)?;
                forward.deepstack = output.deepstack_features;
                forward.vision_output = Some(output.embeddings);
                return Ok(forward
                    .vision_output
                    .as_ref()
                    .expect("installed vision output")
                    .clone());
            }
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
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_logits(hidden, context)
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

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
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
        let pass = <Self as RoutedLayeredArchitecture<B, S>>::expert_pass_for_unit(
            self, group, index, hidden, forward,
        );
        self.forward_components_parallel(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
            observer,
        )
    }
    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.finish_components(hidden, Some(parallel), context, observer)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_forward_with_metadata(input, state, Some(parallel), context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)))
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
        self.forward_unit_with_provider_parallel(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
        )
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
        Ok(hidden.clone())
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend("Qwen3-VL model has no local geometry"));
        }
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
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

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, PipelineBoundary<B::Tensor>>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        Ok(LayeredPartitionOutput::Final {
            output: self.finish_components(hidden, parallel, context, observer)?,
            retained: None,
        })
    }

    type Boundary = PipelineBoundarySchema;

    fn boundary_schema(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Self::Boundary, Self::Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &Self, Option<&eredu_nn::workspace::WorkspaceContext>,
                Self::Boundary, Result<Self::Boundary, Self::Error>,
            )>())?;
        }
        Ok(PipelineBoundarySchema::from_args(self.args()))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, PipelineBoundary<B::Tensor>>,
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
        input: LayeredPartitionInput<'a, B::Tensor, PipelineBoundary<B::Tensor>>,
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, PipelineBoundary<B::Tensor>>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: PipelineBoundary {
                    cosine: forward.rotary()?.0.clone(),
                    sine: forward.rotary()?.1.clone(),
                    position_delta: forward
                        .position_delta
                        .as_ref()
                        .ok_or_else(|| Error::backend("decoder has no position delta"))?
                        .clone(),
                    deepstack: forward.deepstack.clone(),
                },
            })
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDtype};

    #[test]
    fn mrope_and_deepstack_wire_geometry_is_family_owned() {
        let schema = PipelineBoundarySchema {
            head_dim: 8,
            hidden_size: 32,
            deepstack_count: 2,
        };
        let tensors = schema.wire_schema().unwrap().resolve(2, 5).unwrap();
        assert_eq!(tensors.primary().shape(), [2, 5, 32]);
        assert_eq!(tensors.auxiliary().len(), 5);
        assert_eq!(tensors.auxiliary()[0].shape(), [5, 8]);
        assert_eq!(tensors.auxiliary()[2].dtype(), BoundaryTensorDtype::Int32);
        assert_eq!(tensors.auxiliary()[3].role(), "deepstack.0");
        assert_eq!(tensors.auxiliary()[3].shape(), [2, 5, 32]);
    }

    #[test]
    fn routed_partition_declares_attention_and_expert_output_sums() {
        assert_eq!(qwen_vl_routed_tensor_reductions(), (1, 1));
    }
}

#[cfg(test)]
mod paid_boundary_metadata_tests {
    use super::*;
    #[test]
    fn vision_boundary_keeps_rotary_delta_and_deepstack_roles(){
        for count in [0,3]{
            crate::boundary_metadata::tests::check(PipelineBoundarySchema{
                head_dim:4,hidden_size:8,deepstack_count:count},(0..3+count).map(|n|43+n as i32*13).collect());
        }
    }
}
