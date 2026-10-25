//! Family invocation transport over shared selected text execution.
use super::{
    model::{ForwardContext, TextInput, TextKernel},
    FamilyConfig,
};
use crate::decoder::{self, ComponentInstrumentation, Config, PartitionStaticModules};
use eredu_core::decoding::AttentionInvocation;
use eredu_nn::{AttentionCache, DistributedNeuralBackend, Error, Tensor};
use eredu_runtime::{
    ComponentAdapter, ComponentPartitionContext, KernelParameterModules, LayerRuntimeState,
    LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput, NoAuxiliaryBoundary,
    PartitionComponentKernel, PipelineComponentKernel, StateLayout,
};
/// Immutable mode and original prefix position carried with the complete canvas.
#[derive(Debug, Clone)]
pub struct TextBoundary<T> {
    /// Three I32 values: mode code, prefix position and activation precision.
    pub invocation: T,
}
/// Checked mode/frontier wire geometry over full residual activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextBoundarySchema {
    hidden: i32,
}
impl TextBoundarySchema {
    /// Declares the global residual width.
    pub fn new(hidden: i32) -> Self {
        Self { hidden }
    }
}
impl eredu_runtime::ArchitectureBoundary for TextBoundarySchema {
    type Boundary<T> = TextBoundary<T>;
    const IDENTITY: &'static str = "nemotron_labs_diffusion.shared_text";
    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden)
    }
    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        vec![eredu_runtime::BoundaryTensorSpec::new(
            "invocation",
            [eredu_runtime::BoundaryTensorDimension::Fixed(3)],
            eredu_runtime::BoundaryTensorDtype::Int32,
        )]
    }
    fn encode<T>(
        &self,
        boundary: TextBoundary<T>,
    ) -> Result<
        Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,
        eredu_runtime::ArchitectureBoundaryError,
    > {
        Ok(vec![eredu_runtime::ArchitectureBoundaryValue::new(
            "invocation",
            boundary.invocation,
        )?])
    }
    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<TextBoundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(TextBoundary {
            invocation: tensors
                .into_iter()
                .next()
                .expect("validated boundary cardinality"),
        })
    }
}
fn mode_code(invocation: AttentionInvocation) -> Result<i32, Error> {
    match invocation {
        AttentionInvocation::CAUSAL => Ok(0),
        AttentionInvocation::DIFFUSION => Ok(1),
        AttentionInvocation::DRAFT => Ok(2),
        _ => Err(Error::backend("invalid shared text invocation")),
    }
}
fn decode<T: Tensor>(
    boundary: TextBoundary<T>,
    context: &T::Context,
) -> Result<(AttentionInvocation, i32, eredu_nn::TensorElementType), Error> {
    if boundary.invocation.shape() != [3]
        || boundary.invocation.element_type() != Some(eredu_nn::TensorElementType::I32)
    {
        return Err(Error::backend("invalid shared text invocation wire"));
    }
    let values = boundary.invocation.to_i32_vec(context)?;
    if values.len() != 3 || values[1] < 0 {
        return Err(Error::backend("invalid shared text prefix frontier"));
    }
    let invocation = match values[0] {
        0 => AttentionInvocation::CAUSAL,
        1 => AttentionInvocation::DIFFUSION,
        2 => AttentionInvocation::DRAFT,
        _ => return Err(Error::backend("invalid shared text mode code")),
    };
    Ok((invocation, values[1], precision_from_code(values[2])?))
}

pub(crate) fn precision_code<T: Tensor>(tensor: &T) -> Result<i32, Error> {
    match tensor.element_type() {
        Some(eredu_nn::TensorElementType::F32) => Ok(0),
        Some(eredu_nn::TensorElementType::F16) => Ok(1),
        Some(eredu_nn::TensorElementType::Bf16) => Ok(2),
        _ => Err(Error::backend(
            "unsupported shared-backbone activation precision",
        )),
    }
}
pub(crate) fn precision_from_code(code: i32) -> Result<eredu_nn::TensorElementType, Error> {
    match code {
        0 => Ok(eredu_nn::TensorElementType::F32),
        1 => Ok(eredu_nn::TensorElementType::F16),
        2 => Ok(eredu_nn::TensorElementType::Bf16),
        _ => Err(Error::backend(
            "invalid shared-backbone activation precision",
        )),
    }
}
/// Prepared residual ingress and request-local visibility.
pub struct PreparedPartitionInvocation<T: 'static> {
    /// Owned residual ingress, ready for the ordinary partition driver.
    pub input: LayeredPartitionInput<'static, T, TextBoundary<T>>,
    /// Request visibility used unchanged by every owner.
    pub mask: Option<T>,
}
/// Selected physical resources and global-to-local geometry for the shared text kernel.
pub type PartitionedTextKernel<B> =
    TextKernel<B, PartitionStaticModules<B>, decoder::PartitionLocalGeometry<FamilyConfig>>;
/// Text partition using the same validated adapter and equations as complete execution.
pub type PartitionedTextModel<B> = ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    PartitionedTextKernel<B>,
    KernelParameterModules,
>;

impl<B: DistributedNeuralBackend>
    TextKernel<B, PartitionStaticModules<B>, decoder::PartitionLocalGeometry<FamilyConfig>>
{
    /// Materializes only resources authorized by the selected declaration.
    pub fn from_partition(
        config: FamilyConfig,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        partition: &eredu_runtime::ArchitecturePartition<
            decoder::PartitionLocalGeometry<FamilyConfig>,
            TextBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionedTextModel<B>, Error> {
        let definition = std::sync::Arc::new(super::definition::text_partition(
            &config,
            parameters,
            partition.local_geometry(),
        )?);
        definition.select_partition(partition)?;
        if partition.boundary_schema() != &TextBoundarySchema::new(config.hidden_size()) {
            return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                component: definition.id().to_owned(),
                resource: "partition boundary geometry".into(),
            }
            .into());
        }
        let base_definition = std::sync::Arc::new(decoder::component_definition(
            &config,
            parameters.clone(),
            partition.local_geometry().complete_state_layout().clone(),
        )?);
        let base =
            decoder::DecoderKernel::from_partition(config, base_definition, partition, context)?
                .into_parts()
                .1;
        let kernel = Self {
            definition: definition.clone(),
            base,
        };
        ComponentAdapter::with_partition_resources::<B, _, _>(definition, kernel, partition)
            .map_err(Error::from)
    }
    fn resume_partition<S, O>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let frontier = state.layer(0).map_err(Error::backend)?.offset();
        let (input, invocation, position, length) = match input {
            LayeredPartitionInput::Tokens(tokens) => (
                LayeredPartitionInput::Tokens(tokens),
                AttentionInvocation::CAUSAL,
                frontier,
                tokens.dim(1),
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                let (invocation, position, element) = decode(auxiliary, context)?;
                let hidden = hidden.cast_float(element, context)?;
                if position != frontier {
                    return Err(Error::backend(
                        "pipeline owners disagree on canonical prefix frontier",
                    ));
                }
                let length = hidden.dim(1);
                (
                    LayeredPartitionInput::Hidden {
                        hidden,
                        auxiliary: NoAuxiliaryBoundary,
                    },
                    invocation,
                    position,
                    length,
                )
            }
        };
        let mask = self.prepare_mask(mask, length, invocation, state, context)?;
        let forward = PartitionComponentKernel::begin_partition_observed(
            &mut self.base,
            input,
            mask.as_ref(),
            state,
            ComponentPartitionContext {
                expected: Some(expected),
                first_state_ordinal: first,
                parallel,
                native: context,
            },
            observer,
        )?;
        Ok(Self::invocation_forward(forward, invocation, position))
    }
    /// Embeds first-owner tokens and retains immutable mode, frontier and precision.
    pub fn prepare_partition_input<S>(
        &mut self,
        input: TextInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedPartitionInvocation<B::Tensor>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
    {
        let expected = state.layout().clone();
        let forward = self.begin_input(
            input,
            None,
            state,
            &expected,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )?;
        let boundary = TextBoundary {
            invocation: B::Tensor::from_i32_slice(
                &[
                    mode_code(forward.context.invocation)?,
                    forward.context.position,
                    precision_code(&forward.hidden)?,
                ],
                &[3],
                context,
            )?,
        };
        Ok(PreparedPartitionInvocation {
            input: LayeredPartitionInput::Hidden {
                hidden: forward.hidden,
                auxiliary: boundary,
            },
            mask: forward.context.mask,
        })
    }
    fn begin_input<S, O>(
        &mut self,
        input: TextInput<'_, B::Tensor>,
        incoming: Option<(B::Tensor, TextBoundary<B::Tensor>)>,
        state: &mut S,
        expected: &StateLayout,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        mode_code(input.invocation)?;
        if input.feedback.is_some()
            || input.tokens.shape().len() != 2
            || input.tokens.dim(0) <= 0
            || input.tokens.dim(1) <= 0
        {
            return Err(Error::backend("invalid shared text partition input"));
        }
        if let Some((hidden, boundary)) = incoming {
            if hidden.shape()
                != [
                    input.tokens.dim(0),
                    input.tokens.dim(1),
                    self.config().hidden_size(),
                ]
            {
                return Err(Error::backend(
                    "pipeline activation does not carry the complete requested token span",
                ));
            }
            let (actual, position, _) = decode(boundary.clone(), context)?;
            if actual != input.invocation
                || position != state.layer(0).map_err(Error::backend)?.offset()
            {
                return Err(Error::backend(
                    "pipeline owners disagree on invocation mode or causal frontier",
                ));
            }
            let mut forward = self.resume_partition(
                LayeredPartitionInput::Hidden {
                    hidden,
                    auxiliary: boundary,
                },
                input.mask,
                state,
                expected,
                0,
                parallel,
                context,
                observer,
            )?;
            if forward.context.invocation != input.invocation {
                return Err(Error::backend(
                    "pipeline owners disagree on requested attention mode",
                ));
            }
            forward.context.invocation = input.invocation;
            return Ok(forward);
        }
        let position = state.layer(0).map_err(Error::backend)?.offset();
        let mask = self.prepare_mask(
            input.mask,
            input.tokens.dim(1),
            input.invocation,
            state,
            context,
        )?;
        let forward = PartitionComponentKernel::begin_partition_observed(
            &mut self.base,
            LayeredPartitionInput::Tokens(input.tokens),
            mask.as_ref(),
            state,
            ComponentPartitionContext {
                expected: Some(expected),
                first_state_ordinal: 0,
                parallel,
                native: context,
            },
            observer,
        )?;
        Ok(Self::invocation_forward(
            forward,
            input.invocation,
            position,
        ))
    }
}
impl<B, S> PartitionComponentKernel<B, S> for PartitionedTextKernel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
{
    type Boundary = TextBoundarySchema;
    fn partition_observation_hooks(
        &self,
        _: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as eredu_runtime::ComponentEquations<B, S>>::observation_hooks(self)
    }
    fn boundary_schema(&self) -> Result<Self::Boundary, Error> {
        Ok(TextBoundarySchema::new(self.config().hidden_size()))
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_partition_observed(
            input,
            mask,
            state,
            selected,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_partition_observed<O>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.resume_partition(
            input,
            mask,
            state,
            selected.expected.ok_or_else(|| {
                Error::backend("shared text partition requires selected state geometry")
            })?,
            selected.first_state_ordinal,
            selected.parallel,
            selected.native,
            observer,
        )
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error> {
        self.finish_partition_observed(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: TextBoundary {
                    invocation: B::Tensor::from_i32_slice(
                        &[
                            mode_code(forward.invocation)?,
                            forward.position,
                            precision_code(hidden)?,
                        ],
                        &[3],
                        context,
                    )?,
                },
            });
        }
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        Ok(LayeredPartitionOutput::Final {
            output: self.base.finish_local_instrumented(
                hidden,
                parallel,
                context,
                &mut ComponentInstrumentation::new("readout", &mut observer),
            )?,
            retained: None,
        })
    }
}
impl<B, S> PipelineComponentKernel<B, S> for PartitionedTextKernel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
{
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }
    fn output_width(&self) -> i32 {
        self.config().vocabulary_size()
    }
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Error> {
        if input.tokens.shape().len() != 2 || input.tokens.dim(0) <= 0 || input.tokens.dim(1) <= 0 {
            return Err(Error::backend("invalid shared text pipeline token shape"));
        }
        Ok((input.tokens.dim(0), input.tokens.dim(1)))
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, TextBoundary<B::Tensor>)>,
        state: &mut S,
        selected: ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.begin_input(
            input,
            incoming,
            state,
            selected.expected.ok_or_else(|| {
                Error::backend("shared text pipeline requires selected state geometry")
            })?,
            selected.parallel,
            selected.native,
            observer,
        )
    }
}
