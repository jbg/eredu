//! Exact local perception state and post-subsampling pipeline boundaries.
use super::*;
use eredu_nn::TensorElementType;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    ArchitecturePartition, BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype,
    BoundaryTensorSpec, LayeredPartitionInput, LayeredPartitionOutput,
};
use std::ops::Range;

/// Frame identity retained with every post-subsampling activation.
#[derive(Clone, Debug)]
pub struct PerceptionBoundary<T> {
    /// I32 `[5]`: start low/high, end low/high (u64 bit limbs), encoded count.
    pub frame_frontier: T,
}

/// Exact perception wire contract; all boundaries contain encoder-width features.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerceptionBoundarySchema {
    width: i32,
}
impl PerceptionBoundarySchema {
    /// Declares transport without native allocation.
    pub fn new(dimensions: PerceptionDimensions) -> Self {
        Self {
            width: dimensions.block.width,
        }
    }
}
impl ArchitectureBoundary for PerceptionBoundarySchema {
    type Boundary<T> = PerceptionBoundary<T>;
    const IDENTITY: &'static str = "nemotron_voicechat.perception.v1";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::primary_activation(self.width)
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        vec![BoundaryTensorSpec::new(
            "frame_frontier",
            [Dim::Fixed(5)],
            Dtype::Int32,
        )]
    }
    fn encode<T>(
        &self,
        boundary: PerceptionBoundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(vec![ArchitectureBoundaryValue::new(
            "frame_frontier",
            boundary.frame_frontier,
        )?])
    }
    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<PerceptionBoundary<T>, ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(PerceptionBoundary {
            frame_frontier: tensors.into_iter().next().expect("validated count"),
        })
    }
}

pub(super) struct PerceptionOwnership {
    pub(super) blocks: Range<usize>,
    subsampling: bool,
    projection: bool,
    local_state: Option<StateLayout>,
}

fn ownership<G>(
    definition: &ComponentDefinition,
    dimensions: PerceptionDimensions,
    partition: &ArchitecturePartition<G, PerceptionBoundarySchema>,
) -> Result<PerceptionOwnership, PerceptionError> {
    dimensions
        .block
        .width
        .checked_add(dimensions.output_width)
        .ok_or(PerceptionError::Configuration(
            "packed perception width exceeds I32",
        ))?;
    let description = definition.parameters();
    if partition.graph() != description.graph()
        || partition.unit_layout() != description.unit_layout()
        || partition.boundary_schema() != &PerceptionBoundarySchema::new(dimensions)
        || !description.matches_partition_bindings(partition)
    {
        return Err(PerceptionError::Configuration(
            "perception partition graph, parameters or boundary differs",
        ));
    }
    let mut subsampling = false;
    let mut projection = false;
    let mut blocks = None;
    for group in partition.groups() {
        match group.group_index() {
            0 if group.global_units() == (0..1) => subsampling = true,
            1 => blocks = Some(group.global_units()),
            2 if group.global_units() == (0..1) => projection = true,
            _ => {
                return Err(PerceptionError::Configuration(
                    "perception partition selects unknown units",
                ))
            }
        }
    }
    let blocks = blocks.unwrap_or(dimensions.layers..dimensions.layers);
    if (blocks.is_empty() && !subsampling && !projection)
        || (subsampling && !blocks.is_empty() && blocks.start != 0)
        || (projection && !blocks.is_empty() && blocks.end != dimensions.layers)
        || (subsampling && projection && blocks != (0..dimensions.layers))
        || partition.ownership().owns_input() != subsampling
        || partition.ownership().owns_output() != projection
    {
        return Err(PerceptionError::Configuration(
            "perception stage is not a contiguous input/output ownership chain",
        ));
    }
    let local_state = if blocks.is_empty() {
        None
    } else {
        Some(definition.state_layout().slice(blocks.clone())?)
    };
    match (partition.state(), &local_state) {
        (Some(actual), Some(expected))
            if actual.global_layers() == blocks && actual.layout() == expected => {}
        (None, None) => {}
        _ => {
            return Err(PerceptionError::State(
                "perception partition cache geometry differs",
            ))
        }
    }
    Ok(PerceptionOwnership {
        blocks,
        subsampling,
        projection,
        local_state,
    })
}

/// Validates selected units, state, physical placement and transport without native resources.
pub fn validate_perception_partition<G>(
    definition: &ComponentDefinition,
    dimensions: PerceptionDimensions,
    layout: Option<&LocalModelLayout>,
    partition: &ArchitecturePartition<G, PerceptionBoundarySchema>,
) -> Result<(), PerceptionError> {
    if let Some(layout) = layout {
        perception_local_dimensions(dimensions, layout)?;
    }
    ownership(definition, dimensions, partition).map(|_| ())
}

impl<B: NeuralBackend> PerceptionKernel<B> {
    /// Constructs only the parameters and cache owners selected for one stage.
    pub fn new_partition<G>(
        definition: ComponentDefinition,
        dimensions: PerceptionDimensions,
        layout: Option<&LocalModelLayout>,
        partition: &ArchitecturePartition<G, PerceptionBoundarySchema>,
    ) -> Result<
        eredu_runtime::ComponentAdapter<eredu_runtime::ComponentDefinition, Self>,
        PerceptionError,
    > {
        let mut architecture = match layout {
            Some(layout) => Self::new_parallel(dimensions, layout)?,
            None => Self::new(dimensions)?,
        };
        architecture.partition = Some(ownership(&definition, dimensions, partition)?);
        Ok(ComponentAdapter::new(definition, architecture)?.select_partition(partition)?)
    }
    /// Validates a restored stage cache without materializing learned parameters.
    /// Every cache belongs to this rank's exact interval and uses released F32 geometry.
    pub fn validate_parent(
        &self,
        state: &PerceptionRuntimeState<B::Tensor>,
    ) -> Result<(), PerceptionError> {
        let selected = self
            .partition
            .as_ref()
            .ok_or(PerceptionError::Configuration(
                "perception parent validation requires selected ownership",
            ))?;
        let layout = selected
            .local_state
            .clone()
            .unwrap_or(perception_state_layout(self.dimensions)?);
        if state.partition_blocks.as_ref() != Some(&selected.blocks)
            || state.partition_roles != Some((selected.subsampling, selected.projection))
            || state.layout != layout
            || state.value.dimensions != Some(self.dimensions)
            || state.value.blocks.len() != selected.blocks.len()
        {
            return Err(PerceptionError::State(
                "perception branch belongs to different stage ownership",
            ));
        }
        let batch = state.value.blocks.iter().find_map(|block| {
            block
                .channel()
                .and_then(|value| value.shape().first().copied())
        });
        for block in &state.value.blocks {
            block.validate_retained(self.dimensions.block, batch, state.value.frames)?;
        }
        Ok(())
    }
    /// Validates ownership and every retained cache against the caller's batch
    /// before any stage enters parameter acquisition.
    pub fn validate_parent_for_batch(
        &self,
        state: &PerceptionRuntimeState<B::Tensor>,
        batch: i32,
    ) -> Result<(), PerceptionError> {
        if batch <= 0 {
            return Err(PerceptionError::State("perception batch must be positive"));
        }
        self.validate_parent(state)?;
        for block in &state.value.blocks {
            block.validate_retained(self.dimensions.block, Some(batch), state.value.frames)?;
        }
        Ok(())
    }
    fn encoded_dimensions(
        &self,
        input: &PerceptionInput<'_, B::Tensor>,
    ) -> Result<(i32, i32), PerceptionError> {
        let shape = input.mel.shape();
        if shape.len() != 3
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[2] != self.dimensions.subsampling.features
            || input.mel.element_type() != Some(TensorElementType::F32)
        {
            return Err(PerceptionError::State(
                "perception pipeline requires F32 frame-major mel ingress",
            ));
        }
        let drop = i32::try_from(input.drop_preencoded)
            .map_err(|_| PerceptionError::State("subsampling overlap exceeds I32"))?;
        let encoded = self
            .dimensions
            .subsampling
            .output_frames(shape[1])?
            .checked_sub(drop)
            .filter(|count| *count > 0)
            .ok_or(PerceptionError::State(
                "subsampling overlap consumes every output frame",
            ))?;
        Ok((shape[0], encoded))
    }
    /// Prepares first-owner mel or an exact upstream encoder activation before cache mutation.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_prepared_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, PerceptionBoundary<B::Tensor>>,
        drop_preencoded: usize,
        encoded_count: Option<i32>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        expected: Option<&StateLayout>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, PerceptionForward<B::Tensor>>, PerceptionError> {
        self.validate_parent(state)?;
        let selected = self
            .partition
            .as_ref()
            .ok_or(PerceptionError::Configuration(
                "perception entry requires selected ownership",
            ))?;
        let expected_entry = if selected.subsampling {
            None
        } else {
            selected.local_state.as_ref()
        };
        if self.is_parallel() != parallel.is_some()
            || expected != expected_entry
            || state.partition_blocks.as_ref() != Some(&selected.blocks)
            || state.partition_roles != Some((selected.subsampling, selected.projection))
            || state.value.dimensions != Some(self.dimensions)
            || state.value.blocks.len() != selected.blocks.len()
            || selected
                .local_state
                .as_ref()
                .is_some_and(|layout| &state.layout != layout)
        {
            return Err(PerceptionError::State(
                "perception stage state or collective selection differs",
            ));
        }
        if let (Some(local), Some(parallel)) = (&self.local_blocks, parallel) {
            if B::parallel_size(parallel) != (self.dimensions.block.heads / local[0].heads) as usize
            {
                return Err(PerceptionError::Configuration(
                    "perception tensor degree differs from communicator",
                ));
            }
        }
        let initial_frames = state.value.frames;
        let (hidden, next_frames) = match input {
            LayeredPartitionInput::Tokens(mel) => {
                if !selected.subsampling {
                    return Err(PerceptionError::State(
                        "mel ingress requires the subsampling owner",
                    ));
                }
                let (_, count) = self.encoded_dimensions(&PerceptionInput {
                    mel,
                    drop_preencoded,
                })?;
                if encoded_count.is_some_and(|expected| expected != count) {
                    return Err(PerceptionError::State("encoded request count differs"));
                }
                let next = initial_frames
                    .checked_add(count as u64)
                    .ok_or(PerceptionError::State("perception frame clock overflow"))?;
                (mel.clone(), next)
            }
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                if selected.subsampling
                    || hidden.shape().len() != 3
                    || hidden.dim(0) <= 0
                    || hidden.dim(1) <= 0
                    || hidden.dim(2) != self.dimensions.block.width
                    || hidden.element_type() != Some(TensorElementType::F32)
                    || auxiliary.frame_frontier.shape() != [5]
                    || auxiliary.frame_frontier.element_type() != Some(TensorElementType::I32)
                {
                    return Err(PerceptionError::State(
                        "invalid perception boundary geometry",
                    ));
                }
                let words = auxiliary.frame_frontier.to_i32_vec(context)?;
                if words.len() != 5 {
                    return Err(PerceptionError::State(
                        "invalid perception frontier payload length",
                    ));
                }
                let start = words[0] as u32 as u64 | ((words[1] as u32 as u64) << 32);
                let end = words[2] as u32 as u64 | ((words[3] as u32 as u64) << 32);
                if start != initial_frames
                    || words[4] != hidden.dim(1)
                    || encoded_count.is_some_and(|count| count != words[4])
                    || start.checked_add(words[4] as u64) != Some(end)
                {
                    return Err(PerceptionError::State(
                        "perception boundary frame frontier differs",
                    ));
                }
                (hidden, end)
            }
        };
        for block in &state.value.blocks {
            block.validate_retained(self.dimensions.block, Some(hidden.dim(0)), initial_frames)?;
        }
        Ok(LayeredForwardState {
            hidden,
            context: PerceptionForward {
                encoder: None,
                drop_preencoded,
                next_frames: Some(next_frames),
                initial_frames,
                projected: false,
            },
        })
    }
}

impl<T: Tensor> PerceptionRuntimeState<T> {
    /// Creates initial local caches or a checked independent candidate from an existing stage.
    pub fn for_partition<B: NeuralBackend<Tensor = T>>(
        architecture: &PerceptionKernel<B>,
        previous: Option<&Self>,
    ) -> Result<Self, PerceptionError> {
        let selected = architecture
            .partition
            .as_ref()
            .ok_or(PerceptionError::Configuration(
                "perception state requires a selected partition",
            ))?;
        let layout = selected
            .local_state
            .clone()
            .unwrap_or(perception_state_layout(architecture.dimensions)?);
        if let Some(previous) = previous {
            architecture.validate_parent(previous)?;
            return Ok(previous.clone());
        }
        Ok(Self {
            layout,
            partition_blocks: Some(selected.blocks.clone()),
            partition_roles: Some((selected.subsampling, selected.projection)),
            value: PerceptionState {
                dimensions: Some(architecture.dimensions),
                blocks: vec![ConformerState::default(); selected.blocks.len()],
                frames: 0,
            },
        })
    }
    /// Exact globally indexed Conformer interval, absent for an ordinary full-stack state.
    pub fn global_blocks(&self) -> Option<Range<usize>> {
        self.partition_blocks.clone()
    }
    /// Exact local Conformer cache replicas owned by this candidate.
    pub fn blocks(&self) -> &[ConformerState<T>] {
        self.value.blocks()
    }
    /// Candidate encoded-frame frontier, independent of mel and live audio clocks.
    pub fn frames(&self) -> u64 {
        self.value.frames()
    }
}

impl<B: NeuralBackend>
    eredu_runtime::PartitionComponentKernel<
        B,
        crate::voicechat::perception_layered::PerceptionRuntimeState<B::Tensor>,
    > for PerceptionKernel<B>
{
    type Boundary = PerceptionBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, PerceptionError> {
        Ok(PerceptionBoundarySchema::new(self.dimensions))
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, PerceptionBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, PerceptionError> {
        if selected.first_state_ordinal != 0 || mask.is_some() {
            return Err(PerceptionError::State(
                "perception partition has no external mask or nonlocal cache origin",
            ));
        }
        self.begin_prepared_partition(
            input,
            0,
            None,
            state,
            selected.expected,
            selected.parallel,
            selected.native,
        )
    }

    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        _: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &mut Self::ForwardContext,
        _: Option<&B::ParallelContext>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, PerceptionError> {
        if group == 2 {
            forward.encoder = Some(initial.clone());
        }
        Ok(initial.clone())
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        forward: &Self::ForwardContext,
        owns_output: bool,
        _: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, PerceptionBoundary<B::Tensor>>, PerceptionError>
    {
        let selected = self
            .partition
            .as_ref()
            .ok_or(PerceptionError::Configuration(
                "perception stage ownership is absent",
            ))?;
        if owns_output != selected.projection || forward.projected != owns_output {
            return Err(PerceptionError::State(
                "perception output requires its completed projection",
            ));
        }
        let end = forward.next_frames.ok_or(PerceptionError::State(
            "perception stage has no encoded frame frontier",
        ))?;
        let start = forward.initial_frames;
        let count = i32::try_from(
            end.checked_sub(start)
                .ok_or(PerceptionError::State("perception frontier regressed"))?,
        )
        .map_err(|_| PerceptionError::State("perception invocation frame count exceeds I32"))?;
        if count <= 0 || hidden.dim(1) != count {
            return Err(PerceptionError::State(
                "perception final sequence differs from encoded count",
            ));
        }
        let result = if owns_output {
            let encoder = forward.encoder.as_ref().ok_or(PerceptionError::State(
                "projection omitted raw encoder capture",
            ))?;
            let packed = B::Tensor::concatenate(&[encoder.clone(), hidden.clone()], 2, context)?;
            LayeredPartitionOutput::Final {
                output: packed,
                retained: None,
            }
        } else {
            let words = [
                start as u32 as i32,
                (start >> 32) as u32 as i32,
                end as u32 as i32,
                (end >> 32) as u32 as i32,
                count,
            ];
            LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: PerceptionBoundary {
                    frame_frontier: B::Tensor::from_i32_slice(&words, &[5], context)?,
                },
            }
        };
        state.value.frames = end;
        Ok(result)
    }
}

impl<B: NeuralBackend>
    eredu_runtime::PipelineComponentKernel<
        B,
        crate::voicechat::perception_layered::PerceptionRuntimeState<B::Tensor>,
    > for PerceptionKernel<B>
{
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), PerceptionError> {
        self.encoded_dimensions(input)
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, PerceptionBoundary<B::Tensor>)>,
        state: &mut PerceptionRuntimeState<B::Tensor>,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        _: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, PerceptionError>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, PerceptionError> + ?Sized,
    {
        if selected.first_state_ordinal != 0 {
            return Err(PerceptionError::State(
                "perception cache ordinals must be local",
            ));
        }
        let (batch, count) = self.encoded_dimensions(&input)?;
        let ingress = match incoming {
            Some((hidden, auxiliary)) => {
                if hidden.shape().get(..2) != Some(&[batch, count]) {
                    return Err(PerceptionError::State(
                        "perception boundary differs from request cadence",
                    ));
                }
                LayeredPartitionInput::Hidden { hidden, auxiliary }
            }
            None => LayeredPartitionInput::Tokens(input.mel),
        };
        self.begin_prepared_partition(
            ingress,
            input.drop_preencoded,
            Some(count),
            state,
            selected.expected,
            selected.parallel,
            selected.native,
        )
    }
    fn output_width(&self) -> i32 {
        self.dimensions.block.width + self.dimensions.output_width
    }

    fn tensor_vocabulary_sharded(&self) -> bool {
        false
    }
}

/// Separates the shared publication tensor into RNNT and language-model features.
pub fn unpack_perception_pipeline_output<T: Tensor>(
    dimensions: PerceptionDimensions,
    packed: &T,
    context: &T::Context,
) -> Result<(T, T), PerceptionError> {
    let width = dimensions
        .block
        .width
        .checked_add(dimensions.output_width)
        .ok_or(PerceptionError::Configuration(
            "packed perception width exceeds I32",
        ))?;
    if packed.shape().len() != 3
        || packed.dim(0) <= 0
        || packed.dim(1) <= 0
        || packed.dim(2) != width
    {
        return Err(PerceptionError::State(
            "published perception feature geometry differs",
        ));
    }
    Ok((
        packed
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(0, dimensions.block.width),
                ],
                context,
            )?
            .compact(context)?,
        packed
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(dimensions.block.width, width),
                ],
                context,
            )?
            .compact(context)?,
    ))
}
