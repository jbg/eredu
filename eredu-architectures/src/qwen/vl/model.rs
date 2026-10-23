//! Composite Qwen3-VL lifecycle over the shared vision tower and ordinary Qwen decoder.

mod observation;

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend, Index,
    Parameterized, RotaryPosition, Tensor,
};
use eredu_runtime::{
    ExpertPass, LayerRuntimeState, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, ParameterProvider, RuntimeStateComponents, StateLayout,
};

use crate::decoder::DecoderStaticResources;
use crate::qwen::vision::{VisionBlock, VisionInput, VisionState, VisionStatic};
use crate::qwen::{self, AttentionInput};
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    media_plan::QwenVlInputPartPlan,
};
use eredu_runtime::{ComponentEquations as _, ParallelComponentKernel as _};

use super::{
    mrope_embeddings, multimodal_position_ids, position_ids_tensor, LocalGeometry, ModelArgs,
    PositionPart,
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

impl<T> PreparedInput<T> {
    /// Borrows the assembled request through the canonical model input vocabulary.
    pub fn with_model_input<R>(&self, apply: impl FnOnce(ModelInput<'_, T>) -> R) -> R {
        let parts = self
            .kinds
            .iter()
            .map(|kind| match *kind {
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
            })
            .collect::<Vec<_>>();
        apply(ModelInput {
            parts: &parts,
            pixels: self.pixels.as_ref(),
            mask: None,
        })
    }
}

/// Materializes Qwen3-VL placeholder IDs, patch grids, and ordered segments
/// from an architecture admission.
pub fn prepare_input<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenVlInputPartPlan>,
    context: &T::Context,
) -> Result<PreparedInput<T>, Error> {
    let prepared = input.prepared();
    let admitted = input.admitted();
    if prepared.identity() != admitted.identity() || prepared.len() != admitted.parts().len() {
        return Err(Error::backend(
            "Qwen3-VL prepared input no longer matches its admission",
        ));
    }

    let mut tokens = Vec::with_capacity(prepared.len());
    let mut grids = Vec::new();
    let mut pixels = Vec::new();
    let mut kinds = Vec::with_capacity(prepared.len());
    let mut projected = Vec::with_capacity(prepared.len());
    for (part_index, (part, plan)) in prepared.parts().iter().zip(admitted.parts()).enumerate() {
        match plan {
            QwenVlInputPartPlan::TextTokens { .. } => {
                let eredu_runtime::PreparedInputPayload::TokenIds(value) = part.payload() else {
                    return Err(Error::backend(
                        "Qwen3-VL admitted text part lost its token payload",
                    ));
                };
                tokens.push(value.clone());
                kinds.push(PreparedInputKind::Text(tokens.len() - 1));
                projected.push(None);
            }
            QwenVlInputPartPlan::ProjectedText { positions } => {
                let eredu_runtime::PreparedInputPayload::Embeddings(value) = part.payload() else {
                    return Err(Error::backend(
                        "Qwen3-VL admitted projected part lost its embedding payload",
                    ));
                };
                let positions = i32::try_from(*positions)
                    .map_err(|_| Error::backend("Qwen3-VL projected span exceeds I32"))?;
                tokens.push(T::full_i32(0, &[1, positions], context)?);
                kinds.push(PreparedInputKind::Projected(tokens.len() - 1, part_index));
                projected.push(Some(value.clone()));
            }
            QwenVlInputPartPlan::Media { ingress, .. } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Qwen3-VL admitted media part lost its tensor payload",
                    ));
                };
                let count = i32::try_from(ingress.placeholder_count)
                    .map_err(|_| Error::backend("Qwen3-VL media span exceeds I32"))?;
                let token = i32::try_from(ingress.placeholder_token_id)
                    .map_err(|_| Error::backend("Qwen3-VL placeholder ID exceeds I32"))?;
                tokens.push(T::full_i32(token, &[1, count], context)?);
                grids.push(ingress.patch_grid.clone());
                pixels.push(value.clone());
                let token_index = tokens.len() - 1;
                let grid_index = grids.len() - 1;
                kinds.push(match part.modality() {
                    eredu_core::InputModality::Image => {
                        PreparedInputKind::Image(token_index, grid_index)
                    }
                    eredu_core::InputModality::Video => {
                        PreparedInputKind::Video(token_index, grid_index)
                    }
                    _ => {
                        return Err(Error::backend(
                            "Qwen3-VL media admission contains an unsupported modality",
                        ));
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
    Ok(PreparedInput {
        tokens,
        grids,
        pixels,
        kinds,
        projected,
    })
}

impl<B, S> crate::composite_execution::ParallelCompositeArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
}

impl<B, S> CompositeArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a> {
        crate::composite_execution::CompositeModelInput::Prepared(input)
    }
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<
        crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    > {
        match input {
            crate::composite_execution::CompositeModelInput::Prepared(input) => Some(*input),
            _ => None,
        }
    }

    type InputPartPlan = QwenVlInputPartPlan;
    type PrefillRequest =
        crate::composite_execution::SingleCompositePrefillRequest<B::Tensor, Self::InputPartPlan>;

    fn prepare_prefill_request(
        prepared: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Self::Error> {
        crate::composite_execution::SingleCompositePrefillRequest::new(prepared, admitted)
    }
    type AdmissionConfig = ModelArgs;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.kernel().args.clone()
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
                    .any(|part| matches!(part, QwenVlInputPartPlan::Media { .. })))
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
                    QwenVlInputPartPlan::Media { shape, .. } => Some(shape.decoder_positions),
                    _ => None,
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
    ) -> Result<Option<crate::composite_execution::CompositeContinuationGeometry>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = input
            .admitted()
            .parts()
            .iter()
            .filter_map(|part| match part {
                QwenVlInputPartPlan::Media { ingress, .. } => Some(&ingress.patch_grid),
                _ => None,
            })
            .flatten()
            .try_fold(0_u64, |total, &(time, height, width)| {
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
        Ok((patches > 0).then_some((patches, self.kernel().args.vision.hidden_size).into()))
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
        let mut patch_positions = 0_u64;
        let mut projected_positions = 0_u64;
        for part in input.admitted().parts() {
            let QwenVlInputPartPlan::Media { ingress, shape } = part else {
                continue;
            };
            projected_positions = projected_positions
                .checked_add(shape.decoder_positions)
                .ok_or_else(|| "Qwen3-VL projected media positions overflowed".to_owned())?;
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
                    .ok_or_else(|| "Qwen3-VL patch-grid geometry overflowed".to_owned())?;
                patch_positions = patch_positions
                    .checked_add(patches)
                    .ok_or_else(|| "Qwen3-VL patch positions overflowed".to_owned())?;
            }
        }
        let patch_positions = i32::try_from(patch_positions)
            .map_err(|_| "Qwen3-VL patch positions exceed i32".to_owned())?;
        let projected_positions = i32::try_from(projected_positions)
            .map_err(|_| "Qwen3-VL projected positions exceed i32".to_owned())?;
        if patch_positions == 0 || projected_positions == 0 {
            return Ok(Some(vec![Vec::new(); pipeline_stages]));
        }
        let layers = self.kernel().args.vision.layer_count();
        if layers < pipeline_stages {
            return Err(
                "Qwen3-VL vision collective schedule has fewer units than PP stages".into(),
            );
        }
        let patch_shape = vec![patch_positions, self.kernel().args.vision.hidden_size];
        let projected_shape = vec![
            projected_positions,
            self.kernel().args.vision.out_hidden_size,
        ];
        let mut ingress_operations = Vec::new();
        for (part, plan) in input
            .prepared()
            .parts()
            .iter()
            .zip(input.admitted().parts())
        {
            if let QwenVlInputPartPlan::TextTokens { positions } = plan {
                let batch = part.payload().value().dim(0);
                let positions = i32::try_from(*positions)
                    .map_err(|_| "Qwen3-VL text positions exceed i32".to_owned())?;
                ingress_operations.push(
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: vec![batch, positions, self.kernel().args.text.hidden_size],
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
                if self
                    .kernel()
                    .args
                    .vision
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

    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Self::Error> {
        // The tensor-parallel attention contribution precedes expert routing;
        // the routed activation contribution remains partial until its exact
        // post-exchange row reduction.
        Ok(qwen_vl_routed_tensor_reductions())
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
        if source_group == 1 && destination_group == 1 {
            let _media_sequence = *group_sequences.first().ok_or_else(|| {
                Error::backend("Qwen3-VL decoder continuation has no vision sequence authority")
            })?;
            let schema = eredu_runtime::ArchitectureBoundary::wire_schema(
                &PipelineBoundarySchema::from_args(&self.kernel().args),
            )
            .map_err(|error| Error::backend(error.to_string()))?;
            let auxiliary = selected.auxiliary().len();
            if auxiliary < 3 {
                return Err(Error::backend(
                    "Qwen3-VL decoder continuation omitted mRoPE boundary roles",
                ));
            }
            return schema
                .resolve_each(
                    batch,
                    std::iter::once(source_sequence)
                        .chain(std::iter::repeat_n(source_sequence, 3))
                        .chain(std::iter::repeat_n(source_sequence, auxiliary - 3)),
                )
                .map(Some)
                .map_err(|error| Error::backend(error.to_string()));
        }
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let same_group = source_group == destination_group;
        let schema = vision_partition_boundary_schema(
            &self.kernel().args,
            same_group,
            selected.auxiliary().len(),
        )
        .map_err(|error| Error::backend(error.to_string()))?;
        let primary_sequence = if same_group {
            continuation
                .ok_or_else(|| Error::backend("Qwen3-VL continuation has no patch geometry"))?
                .0
        } else {
            source_sequence
        };
        schema
            .resolve_each(
                batch,
                std::iter::once(primary_sequence).chain(std::iter::repeat_n(
                    source_sequence,
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
                .ok_or_else(|| Error::backend("Qwen3-VL continuation has no vision state"))?
                .deepstack_features()
        } else {
            &forward.deepstack
        };
        if deepstack.len() != schema.auxiliary().len() {
            return Err(Error::backend(
                "Qwen3-VL vision boundary has incomplete DeepStack context",
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
                    "Qwen3-VL decoder boundary has {} values, expected {}",
                    values.len(),
                    1 + schema.auxiliary().len()
                )));
            }
            let mut values = values.into_iter();
            let hidden = values.next().expect("validated decoder boundary primary");
            let boundary = eredu_runtime::ArchitectureBoundary::decode(
                &PipelineBoundarySchema::from_args(&self.kernel().args),
                values.collect(),
            )
            .map_err(|error| Error::backend(error.to_string()))?;
            forward.rotary = (boundary.cosine, boundary.sine);
            forward.position_delta = boundary.position_delta;
            forward.deepstack = boundary.deepstack;
            return Ok(Some(hidden));
        }
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let expected = 1 + schema.auxiliary().len();
        if values.len() != expected {
            return Err(Error::backend(format!(
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
                .ok_or_else(|| Error::backend("Qwen3-VL continuation has no vision state"))?;
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

/// Pinned text and shared-vision modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Ordinary Qwen embeddings, final norm, and vocabulary head.
    pub text: crate::decoder::PartitionStaticModules<B>,
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

impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for VlKernel<B>
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
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let (group, index) = (address.group(), address.index());
        self.forward_components(
            group, index, unit, hidden, state, forward, pass, provider, context, observer,
        )
    }

    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        VlKernel::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, provider, context,
        )
    }
}

impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for VlKernel<B>
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
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let (group, index) = (address.group(), address.index());
        self.forward_components_parallel(
            group, index, unit, hidden, state, forward, pass, provider, parallel, context, observer,
        )
    }

    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        VlKernel::forward_unit_with_provider_parallel(
            self, group, index, unit, hidden, state, forward, provider, parallel, context,
        )
    }
}

/// Architecture-owned values retained for one complete pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    tokens: Option<T>,
    parts: Vec<PreparedPart<T>>,
    rotary: (T, T),
    position_delta: T,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
    visual_mask: Option<T>,
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
                    mask: self.mask,
                    tokens: None,
                    parts: Vec::new(),
                    rotary: (self.cosine, self.sine),
                    position_delta: position_delta.clone(),
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
        (
            forward.hidden,
            PipelineBoundary {
                cosine: forward.context.rotary.0,
                sine: forward.context.rotary.1,
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
    use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
    eredu_runtime::BoundaryWireSchema::new(
        if continuation {
            "qwen_vl.vision_continuation"
        } else {
            "qwen_vl.vision_to_decoder"
        },
        eredu_runtime::BoundaryTensorSpec::new(
            "hidden",
            [
                Dim::Batch,
                Dim::Sequence,
                Dim::Fixed(if continuation {
                    args.vision.hidden_size
                } else {
                    args.text.hidden_size
                }),
            ],
            Dtype::Activation,
        ),
        (0..deepstack_count).map(|index| {
            eredu_runtime::BoundaryTensorSpec::new(
                format!("deepstack.{index}"),
                [Dim::Batch, Dim::Sequence, Dim::Fixed(args.text.hidden_size)],
                Dtype::Activation,
            )
        }),
    )
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
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        let mut specs = vec![
            eredu_runtime::BoundaryTensorSpec::new(
                "cosine",
                [Dim::Sequence, Dim::Fixed(self.head_dim)],
                Dtype::Activation,
            ),
            eredu_runtime::BoundaryTensorSpec::new(
                "sine",
                [Dim::Sequence, Dim::Fixed(self.head_dim)],
                Dtype::Activation,
            ),
            eredu_runtime::BoundaryTensorSpec::new("position_delta", [Dim::Fixed(1)], Dtype::Int32),
        ];
        specs.extend((0..self.deepstack_count).map(|index| {
            eredu_runtime::BoundaryTensorSpec::new(
                format!("deepstack.{index}"),
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

fn qwen_vl_routed_tensor_reductions() -> eredu_runtime::RoutedTensorReductions {
    eredu_runtime::RoutedTensorReductions::hidden(1, 1)
}

/// Shared execution adapter for full and selected Qwen3-VL components.
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    VlKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Qwen3-VL equations, media geometry and typed resource factories.
pub struct VlKernel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: ModelArgs,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    owns_input: bool,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
    partition_geometry: Option<std::sync::Arc<super::PartitionLocalGeometry>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> eredu_runtime::ComponentKernel<B>
    for VlKernel<B>
{
    type Input<'a> = crate::composite_execution::CompositeModelInput<
        'a,
        B::Tensor,
        QwenVlInputPartPlan,
        ModelInput<'a, B::Tensor>,
    >;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
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
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
        .map_err(Error::backend_source)
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        crate::static_parameters::module_recipes(
            &self.static_modules,
            super::static_recipes(source),
        )
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        self.construct_unit(address, context)
    }
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 1 || (group == 0 && forward.vision_state.is_some())
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(forward.mask.iter());
        values.extend(forward.tokens.iter());
        values.extend([&forward.rotary.0, &forward.rotary.1]);
        values.push(&forward.position_delta);
        values.extend(forward.vision_initial.iter());
        values.extend(forward.vision_output.iter());
        values.extend(forward.deepstack.iter());
        values.extend(forward.visual_mask.iter());
        if let Some(state) = &forward.vision_state {
            values.extend(state.retained_values());
        }
        for part in &forward.parts {
            match part {
                PreparedPart::Text {
                    tokens, embeddings, ..
                } => values.extend([tokens, embeddings]),
                PreparedPart::Media { tokens } => values.push(tokens),
            }
        }
        values.into_iter()
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ComponentPinnedResources<B> for VlKernel<B>
{
    type Modules = StaticModules<B>;
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
        if self.static_modules.vision.has_parameters() {
            visitor.visit("vision", &self.static_modules.vision)?;
        }
        self.static_modules.text.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if self.static_modules.vision.has_parameters() {
            visitor.visit_mut("vision", &mut self.static_modules.vision)?;
        }
        self.static_modules
            .text
            .decoder_resources_mut()
            .visit(visitor)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> VlKernel<B> {
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

    /// Builds all canonical pinned resources from a validated declaration.
    pub fn new(
        args: ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, None, None, context)
    }

    /// Builds complete planner-derived text and vision modules.
    pub fn new_parallel(
        args: ModelArgs,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, Some(geometry), None, context)
    }

    /// Materializes exactly the pinned roots and factories owned by a cold partition.
    pub fn from_partition(
        args: ModelArgs,
        geometry: LocalGeometry,
        partition: &eredu_runtime::ArchitecturePartition<
            super::PartitionLocalGeometry,
            PipelineBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, Some(geometry), Some(partition), context)
    }

    fn materialize(
        args: ModelArgs,
        geometry: Option<LocalGeometry>,
        partition: Option<
            &eredu_runtime::ArchitecturePartition<
                super::PartitionLocalGeometry,
                PipelineBoundarySchema,
            >,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        let definition =
            std::sync::Arc::new(super::component_definition(&args, geometry.as_ref())?);
        if let Some(partition) = partition {
            super::PartitionLocalFoundation::from_partition(&args, partition)
                .map_err(Error::backend_source)?;
            let complete = geometry.as_ref().ok_or_else(|| {
                Error::backend("selected Qwen3-VL has no complete local geometry")
            })?;
            partition
                .local_geometry()
                .validate_complete_geometry(&args, complete)
                .map_err(Error::backend_source)?;
            if partition.graph() != definition.parameters().graph()
                || partition.unit_layout() != definition.parameters().unit_layout()
                || partition.parameter_bindings() != definition.parameters().select_owned(partition)
            {
                return Err(Error::backend(
                    "Qwen3-VL partition ownership differs from its definition",
                ));
            }
        }
        crate::operator_requirements::require::<B>(
            "Qwen3-VL",
            crate::operator_requirements::QWEN_VL,
        )?;
        let stores = |role: &str| {
            partition.is_none_or(|partition| {
                partition
                    .parameter_bindings()
                    .iter()
                    .any(|group| group.owner().accepts_static_role(role))
            })
        };
        let (embedding, head, normalization) =
            crate::decoder::StaticModuleSpec::from_config(&args.text).construction_specs()?;
        let embeddings = stores("embedding")
            .then(|| match &geometry {
                Some(geometry) => B::vocabulary_parallel_embedding(
                    embedding,
                    geometry.text().embedding_range().clone(),
                    context,
                ),
                None => B::embedding(embedding, context),
            })
            .transpose()?;
        let norm = stores("norm")
            .then(|| {
                crate::decoder::NormalizedBoundary::<B>::new(normalization, context)
                    .map(|boundary| boundary.norm)
            })
            .transpose()?;
        let lm_head = if stores("output") {
            head.map(|spec| match &geometry {
                Some(geometry) => B::vocabulary_parallel_linear(
                    spec,
                    geometry.text().output_range().cloned().ok_or_else(|| {
                        Error::backend("Qwen3-VL output owner has no vocabulary range")
                    })?,
                    context,
                ),
                None => B::linear(spec, context),
            })
            .transpose()?
        } else {
            None
        };
        let selection = match partition {
            Some(partition) => crate::qwen::vision::VisionStaticSelection::from_parameter_bindings(
                args.vision.clone(),
                "model.visual",
                eredu_runtime::ExecutionGroupId::new(VISION_EXECUTION_GROUP)
                    .map_err(Error::backend_source)?,
                "vision",
                partition.parameter_bindings(),
            ),
            None => {
                crate::qwen::vision::VisionStaticSelection::all(args.vision.clone(), "model.visual")
            }
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
        let kernel = Self {
            args,
            definition: std::sync::Arc::clone(&definition),
            owns_input: partition.is_none_or(|partition| partition.ownership().owns_input()),
            static_modules: StaticModules {
                text: crate::decoder::PartitionStaticModules {
                    embeddings,
                    norm,
                    lm_head,
                },
                vision,
            },
            parallel_geometry: geometry.map(std::sync::Arc::new),
            partition_geometry: partition
                .map(|partition| std::sync::Arc::new(partition.local_geometry().clone())),
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

    /// Retains a low-level unit/state view independently of pinned resource ownership.
    pub fn with_partition_geometry(
        mut self,
        geometry: super::PartitionLocalGeometry,
    ) -> Result<Self, Error> {
        geometry
            .validate_for(&self.args)
            .map_err(Error::backend_source)?;
        if let Some(complete) = self.parallel_geometry.as_deref() {
            geometry
                .validate_complete_geometry(&self.args, complete)
                .map_err(Error::backend_source)?;
        }
        if self
            .partition_geometry
            .as_deref()
            .is_some_and(|retained| retained != &geometry)
        {
            return Err(Error::backend(
                "Qwen3-VL retained partition geometry is incompatible",
            ));
        }
        self.partition_geometry = Some(std::sync::Arc::new(geometry));
        Ok(self)
    }

    /// Returns normalized nested text and vision policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Borrows canonical invocation, parameter, transport and state ownership.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        Ok(self.definition.state_layout().clone())
    }

    /// Applies the architecture-owned tensor-parallel target output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_readout(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Shares planner-owned geometry with placed unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    fn construct_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        let (group, index) = (address.group(), address.index());
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

    fn begin_request<S>(
        &mut self,
        input: ModelInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if parallel.is_some()
            && self.parallel_geometry.is_none()
            && self.partition_geometry.is_none()
        {
            return Err(Error::backend("Qwen3-VL model has no local geometry"));
        }

        let expected = match self.partition_geometry.as_deref() {
            Some(geometry) => geometry
                .local_state_layout()
                .map_err(|error| Error::backend(error.to_string()))?,
            None => self.state_layout_impl()?,
        };
        if state.layout() != &expected {
            return Err(Error::backend("Qwen3-VL runtime state layout mismatch"));
        }
        let owns_position_delta = expected.layer(0).is_some_and(|policy| {
            policy
                .fixed_state()
                .iter()
                .any(|tensor| tensor.role == StateTensorRole::PositionDelta)
        });
        let (parts, grids) = self.prepare_parts(input.parts, parallel, context)?;
        let visual_mask = self.prepared_visual_mask(&parts, context)?;
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(Error::backend(
                "Qwen3-VL pixels and media metadata must appear together",
            ));
        }
        let (vision_initial, vision_state) = match input.pixels {
            Some(pixels) => {
                let (hidden, state) = self.prepare_vision_request(pixels, &grids, context)?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let assembled = self.assemble(&parts, None, context);
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let state_layer = state.layer(0).map_err(Error::backend)?;
        let offset = state_layer.position();
        let mut position_parts = Vec::with_capacity(input.parts.len());
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
        let (mut positions, computed_delta) = multimodal_position_ids(
            &position_parts,
            self.args.vision.spatial_merge_size,
            sequence,
        )
        .map_err(Error::backend)?;
        if media && offset != 0 {
            return Err(Error::backend(
                "Qwen3-VL media input cannot append to a populated cache",
            ));
        }
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = position_ids_tensor::<B::Tensor>(&positions, context)?;
        let position_delta = if owns_position_delta {
            let delta = state_layer
                .fixed_component(StateTensorRole::PositionDelta)
                .map_err(Error::backend)?;
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
        let (assembled_tokens, assembled_hidden, assembled_error) = match assembled {
            Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
            Err(error) => (None, None, Some(error)),
        };
        let hidden = match vision_initial.as_ref().cloned().or(assembled_hidden) {
            Some(hidden) => hidden,
            None if !self.owns_input() => {
                let tokens = match &parts[0] {
                    PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens,
                };
                B::Tensor::full_f32(
                    0.0,
                    &[tokens.dim(0), sequence, self.args.text.hidden_size],
                    context,
                )?
            }
            None => {
                return Err(
                    assembled_error.unwrap_or_else(|| Error::backend("empty Qwen3-VL input"))
                )
            }
        };
        let deepstack = if media {
            Vec::new()
        } else {
            (0..self.args.vision.deepstack_layer_count())
                .map(|_| hidden.zeros_like(context))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                tokens: assembled_tokens,
                parts,
                rotary,
                position_delta,
                vision_state,
                vision_initial,
                vision_output: None,
                deepstack,
                visual_mask,
            },
        })
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
            .ok_or_else(|| Error::backend("Qwen3-VL does not own token embeddings"))?;
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

    fn owns_input(&self) -> bool {
        self.owns_input
    }

    fn prepare_parts(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        if parts.is_empty() {
            return Err(Error::backend("Qwen3-VL input has no ordered parts"));
        }
        let mut grids = Vec::new();
        let mut prepared = Vec::with_capacity(parts.len());
        for part in parts {
            prepared.push(match part {
                InputPart::Text(tokens) => {
                    let ready = self.owns_input();
                    let embeddings = if ready {
                        self.lookup_tokens(tokens, parallel, context)?
                    } else {
                        B::Tensor::full_f32(
                            0.0,
                            &[tokens.dim(0), tokens.dim(1), self.args.text.hidden_size],
                            context,
                        )?
                    };
                    PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings,
                        ready,
                    }
                }
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    }
                }
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(Error::backend("Qwen3-VL projected input geometry mismatch"));
                    }
                    PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                        ready: true,
                    }
                }
            });
        }
        Ok((prepared, grids))
    }

    fn prepare_vision_request(
        &mut self,
        pixels: &B::Tensor,
        grids: &[(i32, i32, i32)],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        let input = VisionInput {
            pixels,
            grid: grids,
        };
        if self.static_modules.vision.owns_ingress() {
            self.static_modules.vision.begin(input, context)
        } else {
            eredu_nn::sequence_layout::validate_patch_grid(
                grids,
                self.args.vision.spatial_merge_size,
                Some(pixels.dim(0)),
            )
            .map_err(Error::backend)?;
            let state = self.static_modules.vision.prepare_state(grids, context)?;
            let hidden =
                B::Tensor::full_f32(0.0, &[pixels.dim(0), self.args.vision.hidden_size], context)?;
            Ok((hidden, state))
        }
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        if parts
            .iter()
            .any(|part| matches!(part, PreparedPart::Text { ready: false, .. }))
        {
            return Err(Error::backend(
                "Qwen3-VL text products are not available on this owner",
            ));
        }
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
                return Err(Error::backend(format!(
                    "Qwen3-VL vision output {:?} does not match {media_tokens} placeholders",
                    value.shape()
                )))
            }
            None => {
                return Err(Error::backend(
                    "Qwen3-VL media placeholders require vision output",
                ))
            }
        }
        let mut offset = 0;
        let mut embeddings = Vec::with_capacity(parts.len());
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
        assemble_ordered_inputs(&ordered, self.args.text.hidden_size, context)
    }

    fn prepared_visual_mask(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        if self.args.vision.deepstack_layer_count() == 0 {
            return Ok(None);
        }
        let tokens = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => {
                    tokens.clone()
                }
            })
            .collect::<Vec<_>>();
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
                None if !forward.parts.is_empty() => {
                    self.prepared_visual_mask(&forward.parts, context)?
                }
                None => {
                    return Err(Error::backend(
                        "Qwen3-VL compact DeepStack state has no visual-mask authority",
                    ))
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
        self.finish_readout(
            hidden,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    fn finish_readout(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .text
            .decoder_resources_mut()
            .finish_parallel(hidden, parallel, context, instrumentation)
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
            Some(parallel) => self.prepare_parts(input.parts, Some(parallel), context)?,
            None => self.prepare_parts(input.parts, None, context)?,
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
            None if !self.owns_input() => {
                B::Tensor::full_f32(0.0, &[1, sequence, self.args.text.hidden_size], context)?
            }
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
        P: ParameterProvider<B>,
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
                let cosine = &forward.rotary.0;
                let sine = &forward.rotary.1;
                let output = block.body.forward_with_feed_forward(
                    block.output_norm.as_mut(),
                    AttentionInput {
                        selected_positions: None,
                        hidden,
                        mask,
                        cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings { cosine, sine }),
                    },
                    context,
                    |policy, normalized, context| {
                        let shape = normalized.shape().to_vec();
                        let flat = normalized.reshape(
                            &[-1, normalized.dim(normalized.shape().len() - 1)],
                            context,
                        )?;
                        policy
                            .forward_with_provider(index, pass, &flat, context, provider)?
                            .reshape(&shape, context)
                    },
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
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
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
                let cosine = &forward.rotary.0;
                let sine = &forward.rotary.1;
                let output = block.body.forward_tensor_parallel_with_feed_forward(
                    block.output_norm.as_mut(),
                    AttentionInput {
                        selected_positions: None,
                        hidden,
                        mask,
                        cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings { cosine, sine }),
                    },
                    parallel,
                    context,
                    |policy, normalized, context| {
                        let shape = normalized.shape().to_vec();
                        let flat = normalized.reshape(
                            &[-1, normalized.dim(normalized.shape().len() - 1)],
                            context,
                        )?;
                        policy
                            .forward_with_provider_parallel(
                                index, pass, &flat, parallel, context, provider,
                            )?
                            .reshape(&shape, context)
                    },
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

impl<B, S> eredu_runtime::ComponentEquations<B, S> for VlKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
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
        let (group, index) = (address.group(), address.index());
        let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
            self, address, hidden, forward,
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
    fn finish_observed<O>(
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

    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok((group == 1).then(|| eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.to_owned()))
    }

    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok((group == 0).then(|| eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH.to_owned()))
    }

    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        match input {
            crate::composite_execution::CompositeModelInput::Native(input) => {
                self.begin_request(input, state, None, context)
            }
            crate::composite_execution::CompositeModelInput::Prepared(input) => {
                self.begin_prepared(input, state, context)
            }
        }
    }

    fn begin_group(
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
                let vision = forward
                    .vision_output
                    .as_ref()
                    .and_then(|_| dependencies.first().copied())
                    .or(forward.vision_output.as_ref());
                let assembled = self.assemble(&forward.parts, vision, context)?;
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

    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_unit_with_provider(
            address.group(),
            address.index(),
            unit,
            hidden,
            state,
            forward,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
        )
    }

    fn complete_group(
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

    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_logits(hidden, context)
    }
}

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for VlKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
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
        let (group, index) = (address.group(), address.index());
        let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
            self, address, hidden, forward,
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
    fn finish_parallel_observed<O>(
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

    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        match input {
            crate::composite_execution::CompositeModelInput::Native(input) => {
                self.begin_request(input, state, Some(parallel), context)
            }
            crate::composite_execution::CompositeModelInput::Prepared(input) => {
                self.begin_prepared_parallel(input, state, parallel, context)
            }
        }
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
    ) -> Result<B::Tensor, Self::Error> {
        let (group, index) = (address.group(), address.index());
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

    fn complete_group_parallel(
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

    fn finish_parallel(
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
        self.finish_readout(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for VlKernel<B>
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

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(PipelineBoundarySchema::from_args(self.args()))
    }

    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, PipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
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
        let (forward, position_delta) = self.begin_routed_text_partition(
            input,
            mask,
            batch,
            sequence,
            offset,
            selected.parallel,
            selected.native,
        )?;
        if selected.first_state_ordinal == 0 {
            *state
                .layer(local_state_ordinal)
                .map_err(Error::backend)?
                .fixed_component(StateTensorRole::PositionDelta)
                .map_err(Error::backend)? = Some(position_delta);
        }
        Ok(forward)
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
                    self.finish_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: PipelineBoundary {
                    cosine: forward.rotary.0.clone(),
                    sine: forward.rotary.1.clone(),
                    position_delta: forward.position_delta.clone(),
                    deepstack: forward.deepstack.clone(),
                },
            })
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> VlKernel<B> {
    fn begin_prepared<'a, S>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, QwenVlInputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        prepare_input(input, context)?
            .with_model_input(|input| self.begin_request(input, state, None, context))
    }
    fn begin_prepared_parallel<'a, S>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, QwenVlInputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        prepare_input(input, context)?
            .with_model_input(|input| self.begin_request(input, state, Some(parallel), context))
    }
}
