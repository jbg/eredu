//! Typed supplied-embedding transport for causal and character transformer stages.
use super::*;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    ArchitecturePartition, BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype,
    BoundaryTensorSpec, LayeredPartitionInput, LayeredPartitionOutput,
};
use std::ops::Range;

/// Exact auxiliary tensors carried with a transformer activation.
#[derive(Debug, Clone)]
pub struct TransformerBoundary<T> {
    /// I32 `[2]`: initial causal offset and stack kind (causal=1, character=0).
    pub position: T,
    /// I32 `[batch, sequence]` zero/one visibility, present only for characters.
    pub visibility: Option<T>,
}

/// Architecture-owned wire contract, independent of native transport precision.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TransformerBoundarySchema {
    hidden: i32,
    causal: bool,
}
impl TransformerBoundarySchema {
    /// Declares transport from the retained normalized transformer policy.
    pub fn new(config: &TransformerConfig) -> Self {
        Self {
            hidden: config.hidden_size(),
            causal: config.geometry().causal,
        }
    }
}
impl ArchitectureBoundary for TransformerBoundarySchema {
    type Boundary<T> = TransformerBoundary<T>;
    const IDENTITY: &'static str = "nemotron_voicechat.tts_transformer.v1";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::primary_activation(self.hidden)
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        let mut result = vec![BoundaryTensorSpec::new(
            "position",
            [Dim::Fixed(2)],
            Dtype::Int32,
        )];
        if !self.causal {
            result.push(BoundaryTensorSpec::new(
                "visibility",
                [Dim::Batch, Dim::Sequence],
                Dtype::Int32,
            ));
        }
        result
    }
    fn encode<T>(
        &self,
        boundary: TransformerBoundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        if boundary.visibility.is_some() == self.causal {
            return Err(ArchitectureBoundaryError::Invalid {
                boundary: Self::IDENTITY,
                detail: "transformer stack and visibility carrier disagree".into(),
            });
        }
        let mut values = vec![ArchitectureBoundaryValue::new(
            "position",
            boundary.position,
        )?];
        if let Some(keys) = boundary.visibility {
            values.push(ArchitectureBoundaryValue::new("visibility", keys)?);
        }
        Ok(values)
    }
    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<TransformerBoundary<T>, ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        Ok(TransformerBoundary {
            position: tensors.next().expect("validated count"),
            visibility: tensors.next(),
        })
    }
}

pub(super) struct TransformerOwnership {
    pub(super) blocks: Range<usize>,
    input: bool,
    output: bool,
    pub(super) local_state: Option<StateLayout>,
}

pub(super) fn validate_partition<G>(
    definition: &ComponentDefinition,
    config: &TransformerConfig,
    partition: &ArchitecturePartition<G, TransformerBoundarySchema>,
) -> Result<TransformerOwnership, TransformerError> {
    let description = definition.parameters();
    if partition.graph() != description.graph()
        || partition.unit_layout() != description.unit_layout()
        || partition.boundary_schema() != &TransformerBoundarySchema::new(config)
    {
        return Err(TransformerError::Configuration(
            "TTS partition belongs to a different graph or boundary",
        ));
    }
    let mut blocks = None;
    let mut norm = false;
    for group in partition.groups() {
        match group.group_index() {
            0 => blocks = Some(group.global_units()),
            1 if group.global_units() == (0..1) => norm = true,
            _ => {
                return Err(TransformerError::Configuration(
                    "TTS partition selects an unknown unit",
                ))
            }
        }
    }
    let blocks =
        blocks.unwrap_or(config.geometry().num_hidden_layers..config.geometry().num_hidden_layers);
    let input = partition.ownership().owns_input();
    let output = partition.ownership().owns_output();
    if input != (!blocks.is_empty() && blocks.start == 0) || output != norm {
        return Err(TransformerError::Configuration(
            "TTS partition input/output ownership differs from selected units",
        ));
    }
    if norm && !blocks.is_empty() && blocks.end != config.geometry().num_hidden_layers {
        return Err(TransformerError::Configuration(
            "TTS normalization cannot precede remaining blocks",
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
            return Err(TransformerError::Configuration(
                "TTS partition state differs from its owned block range",
            ))
        }
    }
    if !description.matches_partition_bindings(partition) {
        return Err(TransformerError::Configuration(
            "TTS partition parameter authority differs from its units",
        ));
    }
    Ok(TransformerOwnership {
        blocks,
        input,
        output,
        local_state,
    })
}

impl<B: NeuralBackend> TransformerKernel<B> {
    /// Exact attention-cache allocation for this selected component. Character
    /// encoding and normalization-only stages allocate no attention-cache slots.
    pub fn attention_cache_layout(&self) -> Option<&StateLayout> {
        if !self.config.geometry().causal {
            return None;
        }
        self.partition
            .as_ref()
            .map_or(Some(&self.state_layout), |selected| {
                selected.local_state.as_ref()
            })
    }

    /// Constructs only units authorized by an exact canonical architecture partition.
    /// `layout` is present when these units also have tensor-parallel projections.
    pub fn from_partition<G>(
        definition: ComponentDefinition,
        config: TransformerConfig,
        layout: Option<&LocalModelLayout>,
        partition: &ArchitecturePartition<G, TransformerBoundarySchema>,
    ) -> Result<ComponentAdapter<ComponentDefinition, Self>, TransformerError> {
        let mut model = match layout {
            Some(layout) => Self::new_parallel(config, layout)?,
            None => Self::new(config)?,
        };
        if definition.state_layout() != &model.state_layout {
            return Err(TransformerError::Configuration(
                "TTS definition cache geometry differs from selected execution",
            ));
        }
        model.partition = Some(validate_partition(&definition, &model.config, partition)?);
        Ok(ComponentAdapter::new(definition, model)?.select_partition(partition)?)
    }

    /// Prepares one shared-driver stage, including explicitly stateless norm stages.
    /// First-owner `Tokens` contains supplied embeddings, rather than token IDs.
    /// Character scaling is applied only to that ingress form. Upstream hidden
    /// states retain exact key visibility and causal position through the boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_prepared_partition<
        C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
        D: DerefMut<Target = [C]>,
    >(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TransformerBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        expected: Option<&StateLayout>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, TransformerForward<B::Tensor>>, TransformerError>
    {
        let selected = self
            .partition
            .as_ref()
            .ok_or(TransformerError::Configuration(
                "TTS partition invocation requires selected ownership",
            ))?;
        if self.is_parallel() != parallel.is_some()
            || selected.local_state.as_ref() != expected
            || selected
                .local_state
                .as_ref()
                .is_some_and(|layout| layout != &state.layout)
            || state.cache_base != selected.blocks.start
            || state.caches.len()
                != if self.config.geometry().causal {
                    selected.blocks.len()
                } else {
                    0
                }
        {
            return Err(TransformerError::Configuration(
                "TTS partition state or collective selection differs",
            ));
        }
        let causal = self.config.geometry().causal;
        let retained_position = if causal {
            self.validate_causal_state(state)?
        } else {
            0
        };
        let (hidden, keys, offset, ingress) = match input {
            LayeredPartitionInput::Tokens(embedded) => {
                if !selected.input {
                    return Err(TransformerError::Configuration(
                        "TTS embeddings require the input owner",
                    ));
                }
                let offset = retained_position;
                (embedded.clone(), mask.cloned(), offset, true)
            }
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                if selected.input
                    || mask.is_some()
                    || auxiliary.position.shape() != [2]
                    || auxiliary.position.element_type() != Some(TensorElementType::I32)
                    || auxiliary.visibility.is_some() == causal
                {
                    return Err(TransformerError::Configuration(
                        "invalid TTS pipeline ingress or boundary geometry",
                    ));
                }
                let position = auxiliary.position.to_i32_vec(context)?;
                if position.len() != 2 || position[1] != i32::from(causal) {
                    return Err(TransformerError::Configuration(
                        "TTS pipeline boundary names a different stack",
                    ));
                }
                let keys = match auxiliary.visibility {
                    Some(visibility) => {
                        if visibility.element_type() != Some(TensorElementType::I32)
                            || visibility.shape() != &hidden.shape()[..hidden.shape().len().min(2)]
                            || visibility
                                .to_i32_vec(context)?
                                .iter()
                                .any(|&n| n != 0 && n != 1)
                        {
                            return Err(TransformerError::Configuration(
                                "invalid TTS character visibility boundary",
                            ));
                        }
                        Some(visibility.equal_i32(1, context)?)
                    }
                    None => None,
                };
                (hidden, keys, position[0], false)
            }
        };
        let sequence = validate_input(&self.config, &hidden)?;
        if offset != retained_position {
            return Err(TransformerError::Configuration(
                "TTS boundary position differs from the retained stage frontier",
            ));
        }
        if offset < 0
            || (!causal && offset != 0)
            || offset
                .checked_add(sequence)
                .is_none_or(|end| end as usize > self.config.geometry().max_position_embeddings)
            || (causal && keys.is_some())
        {
            return Err(TransformerError::Configuration(
                "TTS partition position exceeds its retained capacity",
            ));
        }
        if causal {
            for (index, cache) in state.caches.iter().enumerate() {
                let policy = self
                    .config
                    .attention_schedule()
                    .get(selected.blocks.start + index)
                    .expect("validated owned range");
                if cache.offset() != offset
                    || cache.max_size() != policy.window().map(|n| n.get() as i32)
                {
                    return Err(TransformerError::Configuration(
                        "TTS partition cache frontier or retention differs from boundary",
                    ));
                }
            }
        } else if keys.as_ref().is_none_or(|keys| {
            keys.shape() != &hidden.shape()[..2]
                || keys.element_type() != Some(TensorElementType::Bool)
        }) {
            return Err(TransformerError::Configuration(
                "TTS character partition requires matching boolean visibility",
            ));
        }
        let dtype = hidden
            .element_type()
            .expect("validated activation precision");
        let hidden = if ingress && !causal {
            let scale =
                B::Tensor::full_f32((self.config.hidden_size() as f32).sqrt(), &[], context)?
                    .cast_float(dtype, context)?;
            hidden.multiply(&scale, context)?
        } else {
            hidden
        };
        Ok(LayeredForwardState {
            hidden,
            context: TransformerForward {
                keys,
                mask: None,
                sequence,
                offset,
                dtype,
                normalized: false,
            },
        })
    }
}

impl<'a, B: NeuralBackend, C: RuntimeLayerState<B>, D: DerefMut<Target = [C]>>
    TransformerRuntimeState<'a, B, C, D>
{
    /// Borrows only caches owned by the selected stage; a norm stage owns none.
    pub fn for_partition(
        architecture: &TransformerKernel<B>,
        caches: D,
    ) -> Result<Self, TransformerError>
    where
        C: AttentionCache<B::Tensor>,
    {
        let selected = architecture
            .partition
            .as_ref()
            .ok_or(TransformerError::Configuration(
                "TTS local state requires selected partition ownership",
            ))?;
        let causal = architecture.config.geometry().causal && !selected.blocks.is_empty();
        if caches.len() != if causal { selected.blocks.len() } else { 0 } {
            return Err(TransformerError::Configuration(
                "TTS stage cache count differs from selected blocks",
            ));
        }
        Ok(Self {
            layout: selected
                .local_state
                .clone()
                .unwrap_or_else(|| architecture.state_layout.clone()),
            position: caches.first().map_or(0, |cache| cache.offset()),
            identity: architecture.config.architecture_fingerprint(),
            caches,
            causal,
            cache_base: selected.blocks.start,
            backend: PhantomData,
        })
    }
}

impl<
        B: NeuralBackend,
        C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
        D: DerefMut<Target = [C]>,
    > eredu_runtime::PartitionComponentKernel<B, TransformerRuntimeState<'_, B, C, D>>
    for TransformerKernel<B>
{
    type Boundary = TransformerBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, TransformerError> {
        Ok(TransformerBoundarySchema::new(&self.config))
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TransformerBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, TransformerError> {
        if selected.first_state_ordinal != 0 {
            return Err(TransformerError::Configuration(
                "TTS partition cache ordinals must begin at zero",
            ));
        }
        self.begin_prepared_partition(
            input,
            mask,
            state,
            selected.expected,
            selected.parallel,
            selected.native,
        )
    }

    fn enter_partition_group(
        &mut self,
        _: usize,
        initial: &B::Tensor,
        _: &mut TransformerRuntimeState<'_, B, C, D>,
        _: &mut Self::ForwardContext,
        _: Option<&B::ParallelContext>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TransformerError> {
        Ok(initial.clone())
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        forward: &Self::ForwardContext,
        owns_output: bool,
        _: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TransformerBoundary<B::Tensor>>, TransformerError>
    {
        let selected = self
            .partition
            .as_ref()
            .ok_or(TransformerError::Configuration("missing TTS partition"))?;
        if owns_output != selected.output || owns_output != forward.normalized {
            return Err(TransformerError::Configuration(
                "TTS output requires exactly one completed final normalization",
            ));
        }
        if owns_output {
            if self.config.geometry().causal {
                state.position = forward.offset + forward.sequence;
            }
            return Ok(LayeredPartitionOutput::Final {
                output: hidden.clone(),
                retained: None,
            });
        }
        let visibility = match &forward.keys {
            Some(keys) => Some(B::Tensor::where_condition(
                keys,
                &B::Tensor::full_i32(1, keys.shape(), context)?,
                &B::Tensor::full_i32(0, keys.shape(), context)?,
                context,
            )?),
            None => None,
        };
        let output = LayeredPartitionOutput::Boundary {
            hidden: hidden.clone(),
            auxiliary: TransformerBoundary {
                position: B::Tensor::from_i32_slice(
                    &[forward.offset, i32::from(self.config.geometry().causal)],
                    &[2],
                    context,
                )?,
                visibility,
            },
        };
        if self.config.geometry().causal {
            state.position = forward.offset + forward.sequence;
        }
        Ok(output)
    }
}
