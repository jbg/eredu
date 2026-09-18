//! One neutral Muse-Glimmer multimodal model for resident and bounded runtimes.

mod media_prefill;
mod observation;
pub use media_prefill::MediaPrefillPlan;

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, Error, GroupedNeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, ExpertPass,
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, OwnedParameterGroupSpec, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, ParameterGroupOwner, PartitionedLayeredArchitecture,
    RoutedExpertProvider, RoutedLayeredArchitecture, StateLayout,
};

use crate::{
    composite_execution::{
        CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation, PreparedCompositeInput,
    },
    media_plan::MuseGlimmerInputPartPlan,
};

use super::{
    layer_parameter_groups, state_layout, static_parameter_groups, vision_layer_parameter_groups,
    vision_static_parameter_groups, DecoderConfig, LocalGeometry,
    StaticModules as TextStaticModules, TransformerBlock, VisionBlock, VisionInput, VisionState,
    VisionStatic,
};

/// Stable execution-group identity for Muse-Glimmer vision ingress.
pub const VISION_EXECUTION_GROUP: &str = "vision_encoder";
/// Stable execution-group identity for Muse-Glimmer text decoding.
pub const TEXT_EXECUTION_GROUP: &str = "text_decoder";

/// Shared cold and loaded transport ownership for an optional Muse vision root.
pub(crate) fn vision_group_transport(
    args: &DecoderConfig,
) -> eredu_runtime::ArchitectureGroupTransport {
    eredu_runtime::ArchitectureGroupTransport {
        placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
        kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        first_owner_static_roles: args
            .vision_config
            .as_ref()
            .map_or_else(Vec::new, |_| vec!["vision".into()]),
        last_owner_static_roles: Vec::new(),
        merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
        request_optional: true,
    }
}

/// Exact flattened patch or projected-media wire geometry for a vision edge.
pub(crate) fn vision_partition_boundary_schema(
    args: &DecoderConfig, continuation: bool,
    metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
    use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
    let destination = crate::composite_execution::graph::Destination(metadata);
    destination.controls::<(&DecoderConfig, bool, [Dim; 2], eredu_runtime::BoundaryTensorSpec,
        eredu_runtime::BoundaryWireSchema, Vec<eredu_runtime::BoundaryTensorSpec>)>()?;
    let vision = args.vision_config.as_ref().ok_or_else(|| destination.error(format_args!(
        "Muse vision boundary has no vision configuration")))?;
    let primary = destination.boundary_spec("hidden", &[
        Dim::Sequence, Dim::Fixed(if continuation { vision.hidden_size } else { args.hidden_size }),
    ], Dtype::Activation)?;
    destination.boundary_schema(
        if continuation { "muse.vision_continuation" } else { "muse.vision_to_decoder" },
        primary, destination.vector(0)?,
    )
}

/// Proves one DFlash assistant against a target and returns exact ordered capture paths.
pub fn external_assistant_capture_request(
    target: &DecoderConfig,
    assistant: &super::assistant::DFlashConfig,
) -> Result<ExternalPredictionCaptureRequest, String> {
    let proof = assistant
        .prove_compatibility(target)
        .map_err(|error| error.to_string())?;
    let target_layers = proof.target_layer_ids().to_vec().into_boxed_slice();
    let target_paths = target_layers
        .iter()
        .map(|index| format!("model.layers.{index}.output"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(ExternalPredictionCaptureRequest::MuseGlimmerDFlash {
        target_layers,
        target_paths,
    })
}

/// One ordered decoder-ingress segment.
pub enum DecoderInputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image/video placeholder token IDs matching the next projected media span.
    Media(&'a T),
}

/// Prepared ordered text/media request.
pub struct ModelInput<'a, T> {
    /// Ordered token segments at decoder ingress.
    pub parts: &'a [DecoderInputPart<'a, T>],
    /// Optional packed raw image/video patches and host grid metadata.
    pub vision: Option<VisionInput<'a, T>>,
    /// Optional explicit decoder attention mask.
    pub mask: Option<&'a T>,
}

/// Architecture-prepared Muse-Glimmer decoder and vision ingress.
pub struct PreparedCompositeIngress<T> {
    tokens: Vec<T>,
    media: Vec<bool>,
    pixels: Option<T>,
    grid: Vec<(i32, i32, i32)>,
}

impl<T> PreparedCompositeIngress<T> {
    /// Borrows ordered decoder segments with architecture-created placeholders.
    pub fn decoder_parts(&self) -> Vec<DecoderInputPart<'_, T>> {
        self.tokens
            .iter()
            .zip(&self.media)
            .map(|(tokens, media)| {
                if *media {
                    DecoderInputPart::Media(tokens)
                } else {
                    DecoderInputPart::Text(tokens)
                }
            })
            .collect()
    }

    /// Borrows the packed vision input when media is present.
    pub fn vision_input(&self) -> Option<VisionInput<'_, T>> {
        self.pixels.as_ref().map(|pixels| VisionInput {
            pixels,
            grid: &self.grid,
        })
    }
}

/// Interprets one admitted Muse-Glimmer input using neutral tensor operations.
pub fn prepare_composite_ingress<B>(
    input: PreparedCompositeInput<'_, B::Tensor, MuseGlimmerInputPartPlan>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedCompositeIngress<B::Tensor>, Error>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    let prepared = input.prepared();
    let admitted = input
        .admitted()
        .ordinary()
        .expect("this family retains ordinary admission");
    if prepared.identity() != admitted.identity() || prepared.len() != admitted.parts().len() {
        return Err(Error::backend(
            "Muse-Glimmer prepared input no longer matches its admission",
        ));
    }

    let mut tokens = Vec::with_capacity(prepared.len());
    let mut media = Vec::with_capacity(prepared.len());
    let mut pixels = Vec::new();
    let mut grid = Vec::new();
    for (part, plan) in prepared.parts().iter().zip(admitted.parts()) {
        match plan {
            MuseGlimmerInputPartPlan::TextTokens { .. } => {
                let eredu_runtime::PreparedInputPayload::TokenIds(value) = part.payload() else {
                    return Err(Error::backend(
                        "Muse-Glimmer admitted text part lost its token payload",
                    ));
                };
                tokens.push(value.clone());
                media.push(false);
            }
            MuseGlimmerInputPartPlan::Vision { ingress, .. } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Muse-Glimmer admitted media part lost its tensor payload",
                    ));
                };
                let count = i32::try_from(ingress.placeholder_count)
                    .map_err(|_| Error::backend("Muse-Glimmer media span exceeds I32"))?;
                let token = ingress.placeholder_token_id;
                tokens.push(B::Tensor::full_u32(token, &[1, count], context)?);
                media.push(true);
                pixels.push(value.clone());
                grid.extend_from_slice(&ingress.patch_grid);
            }
        }
    }
    let pixels = match pixels.len() {
        0 => None,
        1 => pixels.pop(),
        _ => Some(B::Tensor::concatenate(&pixels, 0, context)?),
    };
    Ok(PreparedCompositeIngress {
        tokens,
        media,
        pixels,
        grid,
    })
}

impl<B, S> CompositeArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type InputPartPlan = MuseGlimmerInputPartPlan;
    type AdmissionConfig = DecoderConfig;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.args.clone()
    }

    fn external_assistant_target_profile_ref(config:&Self::AdmissionConfig)
        ->Option<crate::external_assistant::ExternalAssistantTargetProfileRef<'_>> {
        Some(crate::external_assistant::ExternalAssistantTargetProfileRef::MuseGlimmer(config))
    }

    fn external_assistant_target_profile(
        config: &Self::AdmissionConfig,
    ) -> Option<crate::external_assistant::ExternalAssistantTargetProfile> {
        Some(crate::external_assistant::ExternalAssistantTargetProfileRef::MuseGlimmer(config).to_owned())
    }

    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<
        crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        eredu_core::CapabilityError,
    > {
        crate::media_plan::admit_muse_glimmer_input(config, input, inspector)
    }

    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        group != 0
            || input
                .admitted()
                .ordinary()
                .expect("this family retains ordinary admission")
                .parts()
                .iter()
                .any(|part| matches!(part, MuseGlimmerInputPartPlan::Vision { .. }))
    }

    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group == 0 {
            input
                .admitted()
                .ordinary()
                .expect("this family retains ordinary admission")
                .parts()
                .iter()
                .filter_map(|part| match part {
                    MuseGlimmerInputPartPlan::Vision { ingress, .. } => {
                        Some(ingress.placeholder_count)
                    }
                    _ => None,
                })
                .try_fold(0_u64, |total, positions| total.checked_add(positions))
                .ok_or_else(|| "Muse projected media positions overflowed".to_owned())?
        } else {
            input
                .admitted()
                .ordinary()
                .expect("this family retains ordinary admission")
                .decoder_positions()
        };
        i32::try_from(positions)
            .map_err(|_| "Muse prepared boundary sequence exceeds i32".to_owned())
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
            .admitted()
            .ordinary()
            .expect("this family retains ordinary admission")
            .parts()
            .iter()
            .filter_map(|part| match part {
                MuseGlimmerInputPartPlan::Vision { ingress, .. } => Some(&ingress.patch_grid),
                _ => None,
            })
            .flatten()
            .try_fold(0_u64, |total, &(time, height, width)| {
                [time, height, width]
                    .into_iter()
                    .try_fold(1_u64, |product, value| {
                        u64::try_from(value)
                            .ok()
                            .and_then(|value| product.checked_mul(value))
                    })
                    .and_then(|patches| total.checked_add(patches))
                    .ok_or_else(|| "Muse continuation patch geometry overflowed".to_owned())
            })?;
        if patches == 0 {
            return Ok(None);
        }
        let patches = i32::try_from(patches)
            .map_err(|_| "Muse continuation patch count exceeds i32".to_owned())?;
        let width = self
            .args
            .vision_config
            .as_ref()
            .ok_or_else(|| "Muse continuation has no vision configuration".to_owned())?
            .hidden_size;
        Ok(Some((patches, width)))
    }

    fn prepared_group_continuation_batched(&self, group: usize) -> bool {
        group != 0
    }

    fn partition_boundary_schema(
        &self, source_group: usize, destination_group: usize,
        _selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        batch: i32, source_sequence: i32, _group_sequences: &[i32],
        continuation: Option<(i32, i32)>,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Error> {
        let destination = crate::composite_execution::graph::Destination(metadata);
        destination.controls::<(&Self, usize, usize, i32, i32, Option<(i32, i32)>,
            eredu_runtime::BoundaryWireSchema, Option<eredu_runtime::ResolvedBoundaryWireSchema>)>()?;
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let sequence = continuation.map_or(source_sequence, |(sequence, _)| sequence);
        let schema = vision_partition_boundary_schema(&self.args, source_group == destination_group, metadata)?;
        destination.resolve_boundary(&schema, batch, &[sequence]).map(Some)
    }

    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
    ) -> Result<Option<B::Tensor>, Error> {
        if !matches!((source_group, destination_group), (0, 0) | (0, 1) | (1, 1)) {
            return Ok(None);
        }
        if values.len() != 1 || !schema.auxiliary().is_empty() {
            return Err(Error::backend(
                "Muse boundary must contain exactly its primary activation",
            ));
        }
        // A retained-media invocation owns its actual projected cut before
        // decoder span construction. Ordinary boundary ownership is unchanged.
        let value = values.into_iter().next();
        if source_group == 0 && destination_group == 1 && forward.pending_media.is_some() {
            forward.media_output = value.clone();
        }
        Ok(value)
    }

    fn prepared_group_collective_waves(
        &self,
        group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        pipeline_stages: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, Error>
    {
        // Muse-Glimmer's vision blocks and projector are replicated equations.
        // Under TP+PP they still execute on every tensor rank, but emit no tensor
        // collective; the explicit empty waves keep that fact architecture-owned.
        let destination=crate::composite_execution::graph::Destination(context);
        destination.controls::<(&Self,usize,PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,
            usize,usize,Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>)>()?;
        if group != 0 || tensor_partitions <= 1 || pipeline_stages <= 1 { return Ok(None); }
        destination.collect((0..pipeline_stages).map(|_| Vec::new())).map(Some)
    }

    fn prepared_primary_ingress_collectives(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, Error> {
        let destination=crate::composite_execution::graph::Destination(context);
        destination.controls::<(&Self,PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,usize)>()?;
        crate::composite_execution::segmented_token_ingress_collectives_in(
            input
                .admitted()
                .ordinary()
                .expect("this family retains ordinary admission")
                .parts()
                .iter()
                .filter_map(|part| match part {
                    MuseGlimmerInputPartPlan::TextTokens { positions } => Some(*positions),
                    MuseGlimmerInputPartPlan::Vision { .. } => None,
                }),
            self.args.hidden_size,
            tensor_partitions,
            destination,
        )
    }

    fn primary_ingress_collectives_pending(&self, forward: &Self::ForwardContext) -> bool {
        forward
            .parts
            .iter()
            .any(|part| matches!(part, PreparedPart::PendingText { .. }))
    }

    fn external_prediction_capture_paths(
        request: &ExternalPredictionCaptureRequest,
    ) -> Result<Option<Vec<String>>, Self::Error> {
        external_capture_paths(request, crate::decoder::ModuleMetadata::ordinary())
    }

    fn external_prediction_capture_paths_with_metadata(
        request: &ExternalPredictionCaptureRequest,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<Vec<String>>, Error> {
        external_capture_paths(request, crate::decoder::ModuleMetadata::funded(context))
    }

    fn external_prediction_capture(
        request: &ExternalPredictionCaptureRequest,
        forward: &Self::ForwardContext,
        observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        external_capture(request, forward, observed, crate::decoder::ModuleMetadata::ordinary())
    }

    fn external_prediction_capture_with_metadata(
        request: &ExternalPredictionCaptureRequest,
        forward: &Self::ForwardContext,
        observed: Vec<B::Tensor>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Error> {
        external_capture(request, forward, observed, crate::decoder::ModuleMetadata::funded(context))
    }

    fn external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        match operation {
            ExternalPredictionTargetOperation::TokenEmbeddings(tokens) => {
                self.token_embeddings(tokens, context).map(Some)
            }
            ExternalPredictionTargetOperation::ProjectLogits(hidden) => {
                self.project_logits(hidden, context).map(Some)
            }
        }
    }

    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts();
        <Self as LayeredArchitecture<B, S>>::begin_forward(
            self,
            ModelInput {
                parts: &decoder_parts,
                vision: prepared.vision_input(),
                mask: None,
            },
            state,
            context,
        )
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
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts();
        <Self as ParallelLayeredArchitecture<B, S>>::begin_forward_parallel(
            self,
            ModelInput {
                parts: &decoder_parts,
                vision: prepared.vision_input(),
                mask: None,
            },
            state,
            parallel,
            context,
        )
    }
}

/// Typed decoder input for one pipeline partition.
pub enum TextPartitionInput<'a, T> {
    /// Token identities owned by the first decoder partition.
    Tokens(&'a T),
    /// Embedded activation received from an upstream decoder partition.
    Hidden(T),
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    PendingText { tokens: T },
    Media { tokens: T },
}

/// Forward-pass values retained across streamed unit submissions.
pub struct ForwardContext<T> {
    mask: Option<T>,
    parts: Vec<PreparedPart<T>>,
    vision: Option<VisionState<T>>,
    pending_media: Option<PreparedCompositeIngress<T>>,
    media_output: Option<T>,
    media_span: bool,
}

/// Pinned text and media modules shared by every storage policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Text embedding, final norm, and output head.
    pub text: TextStaticModules<B>,
    /// Optional patch/position modules, merge adapter, and language projection.
    pub vision: Option<VisionStatic<B>>,
}

/// A streamable native-vision block or decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Vision encoder block.
    Vision(VisionBlock<B>),
    /// Text decoder block.
    Text(TransformerBlock<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
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
        self.forward_text_observed(
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
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (1, Unit::Text(unit)) => self.forward_text_unit_with_provider(
                index, unit, hidden, state, forward, pass, provider, context,
            ),
            (_, unit) => <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            ),
        }
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
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
        self.forward_text_parallel_observed(
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
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (1, Unit::Text(unit)) => self.forward_text_unit_parallel_with_provider(
                index, unit, hidden, state, forward, pass, provider, parallel, context,
            ),
            (_, unit) => <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            ),
        }
    }
}

/// The same architecture object used by resident and bounded runtimes.
pub struct LayeredModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: DecoderConfig,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
    partition_state_offset: usize,
    expert_realization:
        Option<std::sync::Arc<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>>,
    execution_graph: ExecutionGraph,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ArchitectureParameters<B> for LayeredModel<B>
{
    type DefinitionError = Error;

    fn state_layout(&self,metadata:Option<&eredu_nn::workspace::WorkspaceContext>)->Result<StateLayout,Self::DefinitionError>{
        match (metadata,&self.parallel_geometry){
            (Some(context),Some(geometry))=>geometry.state_layout().clone_workspace(context),
            (Some(context),None)=>super::graph::state_layout_with_metadata(&self.args,context),
            (None,_)=>self.state_layout_impl(),
        }
    }
    fn state_identity(&self,state:&eredu_runtime::PartitionState,topology:eredu_core::cache::PromptCacheTopology,metadata:Option<&eredu_nn::workspace::WorkspaceContext>)->Result<eredu_runtime::ModelStateIdentity,Self::DefinitionError>{
        super::state_identity_in(&self.args,state.layout(),state.global_layer_offset(),topology,crate::decoder::identity::Metadata::new(metadata))
    }
    fn parameter_description(&self,context:&<B::Tensor as Tensor>::Context)->Result<std::borrow::Cow<'_,ArchitectureParameterDescription>,Self::DefinitionError>{
        self.parameter_description_impl(B::construction_metadata(context)).map(std::borrow::Cow::Owned)
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        super::static_safetensors_recipes(&self.args, source)
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
        if let Some(vision) = &self.static_modules.vision {
            visitor.visit("vision", vision)?;
        }
        visitor.visit("embedding", &self.static_modules.text.embeddings)?;
        visitor.visit("norm", &self.static_modules.text.final_norm)?;
        if let Some(head) = &self.static_modules.text.head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(vision) = &mut self.static_modules.vision {
            visitor.visit_mut("vision", vision)?;
        }
        visitor.visit_mut("embedding", &mut self.static_modules.text.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.text.final_norm)?;
        if let Some(head) = &mut self.static_modules.text.head {
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

    #[allow(clippy::too_many_arguments)]
    fn begin_text_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout() != expected {
            return Err(Error::backend(
                "Muse-Glimmer partition state layout mismatch",
            ));
        }
        let (input, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                (TextPartitionInput::Tokens(tokens), tokens.dim(1))
            }
            LayeredPartitionInput::Hidden { hidden, .. } => {
                let sequence = hidden.dim(1);
                (TextPartitionInput::Hidden(hidden), sequence)
            }
        };
        let offset = state
            .layer(self.local_state_ordinal(first_state_ordinal)?)
            .map_err(Error::backend)?
            .offset();
        self.begin_routed_text_partition(input, mask, sequence, offset, parallel, context)
    }

    /// Builds unloaded pinned modules.
    pub fn new(
        args: DecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Muse-Glimmer",
            crate::operator_requirements::MUSE_GLIMMER,
        )?;
        let static_modules = StaticModules {
            text: TextStaticModules::new(&args, context)?,
            vision: args
                .vision_config
                .clone()
                .map(|vision| VisionStatic::new(vision, context))
                .transpose()?,
        };
        Ok(Self {
            args,
            static_modules,
            execution_graph: Self::build_execution_graph(B::construction_metadata(context))?,
            parallel_geometry: None,
            partition_state_offset: 0,
            expert_realization: None,
        })
    }

    /// Builds the canonical multimodal graph with planner-derived text geometry.
    pub fn new_parallel(
        args: DecoderConfig,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Muse-Glimmer",
            crate::operator_requirements::MUSE_GLIMMER,
        )?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let static_modules = StaticModules {
            text: TextStaticModules::new_parallel(&args, &geometry, context)?,
            vision: args
                .vision_config
                .clone()
                .map(|vision| VisionStatic::new(vision, context))
                .transpose()?,
        };
        Ok(Self {
            args,
            static_modules,
            execution_graph: Self::build_execution_graph(B::construction_metadata(context))?,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
            partition_state_offset: 0,
            expert_realization: None,
        })
    }

    /// Retains the architecture-global ordinal of this pipeline partition's first text state.
    pub(crate) fn with_partition_state_offset(mut self, offset: usize) -> Result<Self, Error> {
        if offset >= self.args.num_hidden_layers as usize {
            return Err(Error::backend(
                "Muse-Glimmer partition state offset is outside the decoder",
            ));
        }
        self.partition_state_offset = offset;
        Ok(self)
    }

    fn local_state_ordinal(&self, global: usize) -> Result<usize, Error> {
        global
            .checked_sub(self.partition_state_offset)
            .ok_or_else(|| Error::backend("Muse-Glimmer unit precedes the partition state offset"))
    }

    fn validate_partition_state<S: LayerRuntimeState<B>>(&self, state: &S) -> Result<(), Error> {
        let complete = self.state_layout_impl()?;
        let end = self
            .partition_state_offset
            .checked_add(state.layout().len())
            .ok_or_else(|| Error::backend("Muse-Glimmer partition state interval overflow"))?;
        let expected = complete
            .slice(self.partition_state_offset..end)
            .map_err(Error::backend)?;
        if state.layout() != &expected {
            return Err(Error::backend(
                "Muse-Glimmer rank-local state layout mismatch",
            ));
        }
        Ok(())
    }

    /// Binds the exact selected compact expert banks used by partition-local units.
    pub fn with_expert_realization(
        mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        if !self.args.is_moe() {
            return Err(Error::backend(
                "dense Muse-Glimmer cannot bind an expert realization",
            ));
        }
        self.expert_realization = Some(std::sync::Arc::new(realization));
        Ok(self)
    }

    /// Returns normalized configuration.
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
    }

    /// Describes pinned multimodal modules and each vision/text graph unit with
    /// explicit neutral ownership.
    fn parameter_description_impl(&self,context:Option<&eredu_nn::workspace::WorkspaceContext>)->Result<ArchitectureParameterDescription,Error>{
        use crate::decoder::parameter_metadata::{DeclarationDestination,ParameterGroupError};
        let context=context.filter(|context|context.uses_checked_metadata());
        let destination=DeclarationDestination(context);
        let metadata=crate::decoder::identity::Metadata::new(context);
        metadata.controls::<(&Self,[usize;2],ExecutionGraph,ExecutionUnitLayout,Vec<OwnedParameterGroupSpec>,Vec<eredu_runtime::ParameterGroupSpec>,ParameterGroupOwner,ArchitectureParameterDescription,eredu_runtime::ExecutionGroupId,std::ops::Range<usize>,Vec<(eredu_runtime::ExecutionGroupId,usize)>,Vec<String>)>()?;
        let graph=match context{Some(context)=>self.execution_graph.clone_with_metadata(context)?,None=>self.execution_graph.clone()};
        let counts=[self.args.vision_config.as_ref().map_or(0,|vision|vision.layer_count()),self.args.num_hidden_layers as usize];
        let layout=match context{Some(context)=>ExecutionUnitLayout::new_with_metadata(&graph,&counts,context)?,None=>ExecutionUnitLayout::new(&graph,counts).map_err(Error::backend)?};
        let text_static=super::parallel::static_parameter_groups_in(&self.args,destination).map_err(ParameterGroupError::into_neural)?;
        let vision_static=super::parallel::vision_static_parameter_groups_in(&self.args,destination).map_err(ParameterGroupError::into_neural)?;
        let mut owned=metadata.vector(text_static.len()+vision_static.len())?;
        for (index,group) in text_static.into_iter().enumerate(){
            let owner=if index==0 && self.args.tie_word_embeddings {
                let mut roles=metadata.vector(2)?;roles.push(metadata.text("embedding")?);roles.push(metadata.text("output")?);ParameterGroupOwner::StaticAnyOf(roles)
            }else{ParameterGroupOwner::static_role(metadata.text(match index{0=>"embedding",1=>"norm",_=>"output"})?)};
            owned.push(OwnedParameterGroupSpec::new(owner,group));
        }
        let copy_id=|id:&eredu_runtime::ExecutionGroupId|eredu_runtime::ExecutionGroupId::new(metadata.text(id.as_str())?).map_err(|cause|metadata.source(cause));
        metadata.borrowed_controls(&copy_id)?;
        for group in vision_static {
            let indices=super::parallel::vision_static_consumer_units(&self.args,group.members()[0].target()).expect("declared Muse vision static group has consumers");
            let mut consumers=metadata.vector(indices.len())?;
            for index in indices {consumers.push((copy_id(layout.group_id(0).expect("Muse vision group"))?,index));}
            let owner=ParameterGroupOwner::StaticUnitConsumers{role:metadata.text("vision")?,consumers};
            owned.push(OwnedParameterGroupSpec::new(owner,group));
        }
        for (group_index,&count) in counts.iter().enumerate(){
            let owner=layout.group_id(group_index).expect("Muse layout group");
            for index in 0..count {
                let groups=if group_index==0 {super::parallel::vision_layer_parameter_groups_in(&self.args,index,destination)}else{super::parallel::layer_parameter_groups_in(&self.args,index,destination)}.map_err(ParameterGroupError::into_neural)?;
                destination.reserve(&mut owned,groups.len()).map_err(ParameterGroupError::into_neural)?;
                for group in groups {owned.push(OwnedParameterGroupSpec::new(ParameterGroupOwner::execution_unit(copy_id(owner)?,index),group));}
            }
        }
        match context {Some(context)=>ArchitectureParameterDescription::from_owned_with_metadata(graph,layout,owned,context),None=>ArchitectureParameterDescription::from_owned(graph,layout,owned).map_err(Error::backend)}
    }

    /// Returns the replicated or planner-derived mutable-state layout.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| state_layout(&self.args).map_err(Error::backend), Ok)
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

    /// Applies final normalization and architecture-owned tensor-parallel
    /// vocabulary projection, including released logit transforms.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.final_hidden(hidden, context)?;
        let logits = match &mut self.static_modules.text.head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context)?,
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            )?,
        };
        self.static_modules.text.finish_logits(logits, context)
    }

    /// Enters the serial decoder partition through the family embedding boundary.
    pub fn begin_partition_text(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.token_embeddings(tokens, context)
    }

    /// Enters the tensor-parallel decoder partition through lookup and input normalization.
    pub fn begin_partition_text_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.pipeline_embed_parallel(tokens, parallel, context)?;
        self.static_modules
            .text
            .normalize_embeddings(&hidden, context)
    }

    /// Resumes an already embedded decoder partition in the canonical layered context.
    pub fn resume_partition_text(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
    ) -> LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>> {
        LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                parts: Vec::new(),
                vision: None,
                pending_media: None,
                media_output: None,
                media_span: false,
            },
        }
    }

    /// Enters or resumes a routed text partition with architecture-owned
    /// layer-local causal-mask construction, including retained sliding history.
    pub fn begin_routed_text_partition(
        &mut self,
        input: TextPartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        _sequence: i32,
        _offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let hidden = match input {
            TextPartitionInput::Tokens(tokens) => match parallel {
                Some(parallel) => self.begin_partition_text_parallel(tokens, parallel, context)?,
                None => self.begin_partition_text(tokens, context)?,
            },
            TextPartitionInput::Hidden(hidden) => hidden,
        };
        Ok(self.resume_partition_text(hidden, explicit_mask.cloned()))
    }

    /// Finishes the serial decoder partition through the family output boundary.
    pub fn finish_partition_text(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_logits(hidden, context)
    }

    /// Finishes the tensor-parallel decoder partition through the family output boundary.
    pub fn finish_partition_text_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
    }

    /// Returns planner-derived geometry for a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_deref()
    }

    /// Shares authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Starts a text-only pass from a rank-local vocabulary embedding shard.
    pub fn begin_parallel_text<S: LayerRuntimeState<B>>(
        &mut self,
        tokens: &B::Tensor,
        embeddings: B::Tensor,
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout().len() != self.args.num_hidden_layers as usize {
            return Err(Error::backend(
                "Muse-Glimmer rank-local state layout mismatch",
            ));
        }
        let hidden = self
            .static_modules
            .text
            .normalize_embeddings(&embeddings, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: None,
                parts: vec![PreparedPart::Text {
                    tokens: tokens.clone(),
                    embeddings,
                }],
                vision: None,
                pending_media: None,
                media_output: None,
                media_span: false,
            },
        })
    }

    /// Runs replicated media ingress and assembles rank-local text embeddings
    /// before the tensor-parallel decoder traversal.
    pub fn begin_parallel_input<S: LayerRuntimeState<B>>(
        &mut self,
        input: ModelInput<'_, B::Tensor>,
        text_embeddings: Vec<B::Tensor>,
        vision_blocks: &mut [VisionBlock<B>],
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout().len() != self.args.num_hidden_layers as usize {
            return Err(Error::backend(
                "Muse-Glimmer rank-local state layout mismatch",
            ));
        }
        let mut embeddings = text_embeddings.into_iter();
        let parts = input
            .parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self.static_modules.text.normalize_embeddings(
                        &embeddings.next().ok_or_else(|| {
                            Error::backend("Muse-Glimmer parallel text embedding is missing")
                        })?,
                        context,
                    )?,
                }),
                DecoderInputPart::Media(tokens) => Ok(PreparedPart::Media {
                    tokens: (*tokens).clone(),
                }),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if embeddings.next().is_some() {
            return Err(Error::backend(
                "Muse-Glimmer parallel input has excess text embeddings",
            ));
        }
        let hidden =
            match input.vision {
                Some(vision) => {
                    let config = self.args.vision_config.as_ref().ok_or_else(|| {
                        Error::backend("Muse-Glimmer model has no vision projector")
                    })?;
                    if vision_blocks.len() != config.layer_count() {
                        return Err(Error::backend(
                            "Muse-Glimmer parallel vision block count mismatch",
                        ));
                    }
                    let vision_static = self.static_modules.vision.as_mut().ok_or_else(|| {
                        Error::backend("Muse-Glimmer model has no vision modules")
                    })?;
                    let (mut hidden, vision_state) = vision_static.begin(vision, context)?;
                    for (index, block) in vision_blocks.iter_mut().enumerate() {
                        hidden = block.forward_scheduled(
                            &hidden,
                            config.schedule[index],
                            &vision_state,
                            context,
                        )?;
                    }
                    let media = vision_static.finish(&hidden, &vision_state, context)?;
                    self.assemble(&parts, Some(&media), context)?
                }
                None => self.assemble(&parts, None, context)?,
            };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                parts,
                vision: None,
                pending_media: None,
                media_output: None,
                media_span: false,
            },
        })
    }

    /// Executes one decoder block with rank-local projections and collectives.
    pub fn forward_text_unit_parallel<S: LayerRuntimeState<B>>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        unit.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            parallel,
            context,
        )
    }

    /// Executes one rank-local decoder block with a runtime-owned routed bank.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_unit_parallel_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_parallel_with_provider(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            pass,
            provider,
            parallel,
            context,
        )
    }

    /// Applies the replicated final normalization before a sharded output head.
    pub fn final_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.final_hidden(hidden, context)
    }

    /// Applies released output scaling and softcapping after vocab gather.
    pub fn finish_parallel_logits(
        &self,
        logits: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.finish_logits(logits, context)
    }

    /// Applies the target's ordinary token embedding and input normalization.
    pub fn token_embeddings(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.embed(tokens, context)
    }

    /// Applies the target-owned final normalization and vocabulary head.
    pub fn project_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.logits(hidden, context)
    }

    /// Executes one text unit while delegating its routed bank to runtime residency.
    pub fn forward_text_unit_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_with_provider(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            pass,
            provider,
            context,
        )
    }

    fn prepare_parts(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        if parts.is_empty() {
            return Err(Error::backend("Muse-Glimmer input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self.static_modules.text.embed(tokens, context)?,
                }),
                DecoderInputPart::Media(tokens) => {
                    if tokens.shape().len() != 2 {
                        return Err(Error::backend(
                            "Muse-Glimmer media token IDs must have rank two",
                        ));
                    }
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            })
            .collect()
    }

    fn prepare_parts_parallel(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        defer_embeddings: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        if parts.is_empty() {
            return Err(Error::backend("Muse-Glimmer input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) if defer_embeddings => {
                    Ok(PreparedPart::PendingText {
                        tokens: (*tokens).clone(),
                    })
                }
                DecoderInputPart::Text(tokens) => {
                    let embeddings = B::vocabulary_parallel_lookup(
                        &mut self.static_modules.text.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?;
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: self
                            .static_modules
                            .text
                            .normalize_embeddings(&embeddings, context)?,
                    })
                }
                DecoderInputPart::Media(tokens) => {
                    if tokens.shape().len() != 2 {
                        return Err(Error::backend(
                            "Muse-Glimmer media token IDs must have rank two",
                        ));
                    }
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            })
            .collect()
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let media_tokens = parts
            .iter()
            .filter_map(|part| match part {
                PreparedPart::Media { tokens } => Some(tokens.dim(1)),
                PreparedPart::Text { .. } | PreparedPart::PendingText { .. } => None,
            })
            .sum::<i32>();
        match vision {
            Some(vision) if vision.shape() != [media_tokens, self.args.hidden_size] => {
                return Err(Error::backend(format!(
                    "Muse-Glimmer projected media has shape {:?}, expected [{media_tokens}, {}]",
                    vision.shape(),
                    self.args.hidden_size
                )));
            }
            None if media_tokens != 0 => {
                return Err(Error::backend(
                    "Muse-Glimmer media placeholders require projected media",
                ));
            }
            _ => {}
        }
        let mut owned_embeddings = Vec::with_capacity(parts.len());
        let mut offset = 0;
        for part in parts {
            match part {
                PreparedPart::PendingText { .. } => {
                    return Err(Error::backend(
                        "Muse text embeddings must be completed in the decoder ingress wave",
                    ));
                }
                PreparedPart::Text { embeddings, .. } => {
                    owned_embeddings.push(embeddings.clone());
                }
                PreparedPart::Media { tokens } => {
                    let length = tokens.dim(1);
                    let media = vision
                        .expect("validated media exists")
                        .index(
                            &[
                                eredu_nn::Index::Range(offset, offset + length),
                                eredu_nn::Index::Full,
                            ],
                            context,
                        )?
                        .expand_dims(0, context)?;
                    owned_embeddings.push(media);
                    offset += length;
                }
            }
        }
        let batch = parts
            .first()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. }
                | PreparedPart::PendingText { tokens }
                | PreparedPart::Media { tokens } => tokens.dim(0),
            })
            .ok_or_else(|| Error::backend("Muse-Glimmer input has no ordered parts"))?;
        for (index, (part, embeddings)) in parts.iter().zip(&owned_embeddings).enumerate() {
            let tokens = match part {
                PreparedPart::Text { tokens, .. }
                | PreparedPart::PendingText { tokens }
                | PreparedPart::Media { tokens } => tokens,
            };
            if tokens.shape().len() != 2
                || embeddings.shape().len() != 3
                || tokens.dim(0) != batch
                || embeddings.dim(0) != batch
                || tokens.dim(1) != embeddings.dim(1)
                || embeddings.dim(2) != self.args.hidden_size
            {
                return Err(Error::backend(format!(
                    "Muse-Glimmer ordered input part {index} has incompatible token/embedding shapes {:?} and {:?}",
                    tokens.shape(),
                    embeddings.shape()
                )));
            }
        }
        B::Tensor::concatenate(&owned_embeddings, 1, context)
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn media_prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata=crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(&Self,Option<&eredu_nn::workspace::WorkspaceContext>,usize,usize,String,Vec<eredu_runtime::layered::PrefillObservationDeclaration>,std::ops::Range<usize>,Option<eredu_runtime::RoutedObservationPoints>,Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>,Error>)>()?;

        // The validated ingress preserves media-prefix placement and the same causal/sliding decoder cache offsets.
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

        // Ordinary text uses causal per-layer KV offsets/window masks and fixed
        // RoPE/NoPE, followed by row-local gates, norms and dense/routed experts.
        // Group 1 owns these real hooks; vision and external assistant equations
        // acquire no row declaration from this ordinary target companion.
        let units = <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 1, metadata_context)?;
        let mut declarations = crate::decoder::ordinary_prefill_observation_declarations(
            (0..units).map(|index| <Self as LayeredArchitecture<B, S>>::unit_path(self, 1, index, metadata_context)),
            true, metadata_context)?;
        // Same target bank invocation as observed execution; its expert equations are row-local.
        for index in 0..units {
            let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, 1, index, metadata_context)?;
            if self.args.num_experts > 0 {
                crate::decoder::append_routed_prefill_path(&mut declarations, &metadata.format(format_args!("{path}.routing"))?, metadata_context)?;
            }
        }
        Ok(declarations)
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn observes_unit_boundaries(&self, group: usize, _index: usize) -> bool {
        group == 1
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
        self.forward_text_observed(
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
        self.finish_text_observed(hidden, None, context, observer)
    }

    type Input<'a> = ModelInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::segmented_token_shape(input.parts.iter().map(|part| match part {
            DecoderInputPart::Text(tokens) | DecoderInputPart::Media(tokens) => *tokens,
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
            vision_group_transport(&self.args)
        } else {
            crate::transport::decoder()
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

        match group {
            0 => Ok(self.parallel_geometry.as_ref().map_or_else(
                || {
                    self.args
                        .vision_config
                        .as_ref()
                        .map_or(0, |vision| vision.layer_count())
                },
                |geometry| geometry.vision_layers(),
            )),
            1 => Ok(self.args.num_hidden_layers as usize),
            _ => Err(metadata.error(format_args!("{}", "Muse-Glimmer has two execution groups"))),
        }
    }

    fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        let count = match group {
            0 => self
                .args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.layer_count()),
            1 => self.args.num_hidden_layers as usize,
            _ => return Err(metadata.error(format_args!("{}", "Muse-Glimmer has two execution groups"))),
        };
        if index >= count {
            return Err(metadata.error(format_args!("{}", "Muse-Glimmer unit is outside its group")));
        }
        match group {
            0 => metadata.text(format_args!("model.vision_tower.layers.{index}")),
            1 => metadata.text(format_args!("model.layers.{index}")),
            _ => unreachable!(),
        }
    }

    fn group_output_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path((group == 0).then_some(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH))
    }

    fn group_input_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path((group == 1).then_some(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH))
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
        match group {
            0 => Ok(Unit::Vision(VisionBlock::new(
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?,
                index,
                context,
            )?)),
            1 => {
                let args = self
                    .parallel_geometry
                    .as_ref()
                    .map(|geometry| {
                        geometry.text_block(index).ok_or_else(|| {
                            Error::backend(format!(
                                "Muse-Glimmer text unit {index} has no rank-local geometry"
                            ))
                        })
                    })
                    .transpose()?
                    .unwrap_or(&self.args);
                let routed_spec = self
                    .expert_realization
                    .as_ref()
                    .and_then(|plan| plan.unit_spec(TEXT_EXECUTION_GROUP, index))
                    .cloned();
                if self.expert_realization.is_some() && routed_spec.is_none() {
                    return Err(Error::backend(format!(
                        "Muse-Glimmer expert realization omits text unit {index}"
                    )));
                }
                Ok(Unit::Text(TransformerBlock::new_with_routed_spec(
                    args,
                    index,
                    routed_spec,
                    context,
                )?))
            }
            _ => Err(Error::backend("Muse-Glimmer has two execution groups")),
        }
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            1 => index,
            _ => index,
        }
    }

    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _ordinal: usize,
    ) -> std::ops::Range<usize> {
        match group {
            0 => 0..0,
            1 => index..index + 1,
            _ => 0..0,
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.validate_partition_state(state)?;
        let parts = self.prepare_parts(input.parts, context)?;
        let (hidden, vision) = match input.vision {
            Some(vision) => {
                let (hidden, state) = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .begin(vision, context)?;
                (hidden, Some(state))
            }
            None => (self.assemble(&parts, None, context)?, None),
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                parts,
                vision,
                pending_media: None,
                media_output: None,
                media_span: false,
            },
        })
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
        if group == 1 && forward.media_span {
            return Ok(initial.clone());
        }
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1, [vision_or_assembled]) if forward.vision.is_some() => {
                self.assemble(&forward.parts, Some(vision_or_assembled), context)
            }
            (1, []) if forward.vision.is_none() => Ok(initial.clone()),
            (1, [assembled]) => Ok((*assembled).clone()),
            _ => Err(Error::backend(
                "invalid Muse-Glimmer execution dependencies",
            )),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group != 0 || forward.vision.is_some()
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
            (0, Unit::Vision(unit)) => unit.forward_scheduled(
                hidden,
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .schedule[index],
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer vision state is missing"))?,
                context,
            ),
            (1, Unit::Text(unit)) => unit.forward(
                hidden,
                forward.mask.as_ref(),
                Some(
                    state
                        .layer(self.local_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                context,
            ),
            _ => Err(Error::backend("Muse-Glimmer unit/group mismatch")),
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
        match (group, forward.vision.as_ref()) {
            (0, Some(vision)) => {
                let media = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .finish(hidden, vision, context)?;
                if forward.pending_media.is_some() {
                    forward.media_output = Some(media.clone());
                }
                Ok(media)
            }
            (0, None) | (1, _) => Ok(hidden.clone()),
            _ => Err(Error::backend("invalid Muse-Glimmer execution group")),
        }
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
        self.static_modules.text.logits(hidden, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(forward.mask.iter());
        values.extend(forward.media_output.iter());
        if let Some(pending) = &forward.pending_media {
            values.extend(pending.tokens.iter());
            values.extend(pending.pixels.iter());
        }
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    values.extend([tokens, embeddings]);
                }
                PreparedPart::Media { tokens } | PreparedPart::PendingText { tokens } => {
                    values.push(tokens)
                }
            }
        }
        if let Some(vision) = &forward.vision {
            values.extend(vision.retained_values());
        }
        values.into_iter()
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
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
        self.forward_text_parallel_observed(
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
        self.finish_text_observed(hidden, Some(parallel), context, observer)
    }

    fn begin_execution_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 1 {
            for part in &mut forward.parts {
                if let PreparedPart::PendingText { tokens } = part {
                    let embeddings = B::vocabulary_parallel_lookup(
                        &mut self.static_modules.text.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?;
                    *part = PreparedPart::Text {
                        tokens: tokens.clone(),
                        embeddings: self
                            .static_modules
                            .text
                            .normalize_embeddings(&embeddings, context)?,
                    };
                }
            }
        }
        self.begin_execution_group(group, initial, dependencies, state, forward, context)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Muse-Glimmer model was not built with rank-local geometry",
            ));
        }
        self.validate_partition_state(state)?;
        // Text lookup reductions belong to the primary decoder wave, after the
        // replicated vision equations. Other pipeline ranks can then enter the
        // same exact segmented sums before the decoder's routed collectives.
        let parts =
            self.prepare_parts_parallel(input.parts, input.vision.is_some(), parallel, context)?;
        let (hidden, vision) = match input.vision {
            Some(vision) => {
                let (hidden, state) = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .begin(vision, context)?;
                (hidden, Some(state))
            }
            None => (self.assemble(&parts, None, context)?, None),
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                parts,
                vision,
                pending_media: None,
                media_output: None,
                media_span: false,
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
        match (group, unit) {
            (0, Unit::Vision(unit)) => unit.forward_scheduled(
                hidden,
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .schedule[index],
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer vision state is missing"))?,
                context,
            ),
            (1, Unit::Text(unit)) => self
                .forward_text_unit_parallel(index, unit, hidden, state, forward, parallel, context),
            _ => Err(Error::backend("Muse-Glimmer parallel unit/group mismatch")),
        }
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
            return Err(Error::backend(
                "Muse-Glimmer model was not built with rank-local geometry",
            ));
        }
        self.static_modules.text.logits_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
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
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        Ok(LayeredPartitionOutput::Final {
            output: self.finish_text_observed(hidden, parallel, context, observer)?,
            retained: None,
        })
    }

    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Self::Boundary, Self::Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &Self, Option<&eredu_nn::workspace::WorkspaceContext>,
                Self::Boundary, Result<Self::Boundary, Self::Error>,
            )>())?;
        }

        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args().hidden_size,
        ))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_text_partition(
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
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_text_partition(
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
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Self::Error> {
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
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
}


fn external_capture_paths(
    request: &ExternalPredictionCaptureRequest,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<Option<Vec<String>>, Error> {
    if !matches!(request, ExternalPredictionCaptureRequest::MuseGlimmerDFlash { .. }) {
        return Ok(None);
    }
    request.collect_paths(metadata).map(Some)
}

fn external_capture<T: Clone>(
    request: &ExternalPredictionCaptureRequest,
    forward: &ForwardContext<T>,
    observed: Vec<T>,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<Option<ExternalPredictionTargetCapture<T>>, Error> {
    metadata.controls::<(Vec<T>, ExternalPredictionTargetCapture<T>, &ForwardContext<T>)>()?;
    let _ = forward;
    let ExternalPredictionCaptureRequest::MuseGlimmerDFlash { target_layers, .. } = request else {
        return Ok(None);
    };
    if observed.len() != target_layers.len() {
        return Err(metadata.error(format_args!(
            "Muse-Glimmer DFlash capture expected {} target states, received {}",
            target_layers.len(), observed.len()
        )));
    }
    Ok(Some(ExternalPredictionTargetCapture::MuseGlimmerDFlash { target_states: observed }))
}
