//! Shared-stack invocation and vision-span transport across composite cuts.
use super::*;
use crate::composite_execution::ParallelCompositeArchitecture;
use eredu_nn::TensorParallelGroupedNeuralBackend;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype, BoundaryTensorSpec,
    LayeredPartitionInput, LayeredPartitionOutput, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, PartitionedLayeredArchitecture,
};

use eredu_runtime::{
    ParallelComponentKernel as _, ParallelRoutedComponentKernel as _, PartitionComponentKernel as _,
};

/// Shared text activation plus invocation and semantic image spans.
pub struct Boundary<T> {
    /// Architecture invocation phase and authoritative context frontier.
    pub invocation: T,
    /// Image interval for each query row, or `[-1,-1]` for ordinary text.
    pub vision_spans: T,
}
/// Exact composite decoder transport, independent of native tensors.
#[derive(Clone, Copy)]
pub struct BoundarySchema {
    hidden: i32,
}
impl BoundarySchema {
    /// Declares shared text activation and invocation/span metadata at a pipeline cut.
    pub const fn new(hidden: i32) -> Self {
        Self { hidden }
    }
}
impl ArchitectureBoundary for BoundarySchema {
    type Boundary<T> = Boundary<T>;
    const IDENTITY: &'static str = "diffusion_gemma.composite_text";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::primary_activation(self.hidden)
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        vec![
            BoundaryTensorSpec::new("invocation", [Dim::Fixed(2)], Dtype::Int32),
            BoundaryTensorSpec::new("vision_spans", [Dim::Sequence, Dim::Fixed(2)], Dtype::Int32),
        ]
    }
    fn encode<T>(
        &self,
        value: Boundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(vec![
            ArchitectureBoundaryValue::new("invocation", value.invocation)?,
            ArchitectureBoundaryValue::new("vision_spans", value.vision_spans)?,
        ])
    }
    fn decode<T>(&self, values: Vec<T>) -> Result<Boundary<T>, ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &values)?;
        let mut values = values.into_iter();
        Ok(Boundary {
            invocation: values.next().unwrap(),
            vision_spans: values.next().unwrap(),
        })
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> Model<B> {
    /// Declares only locally owned shared text parameters and replicated image equations.
    pub fn new_parallel(
        config: FamilyConfig,
        geometry: super::super::parallel::CompositeGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = std::sync::Arc::new(component_definition(&config, Some(&geometry))?);
        let text = TextKernel::materialize(config.clone(), Some(geometry.text.clone()), context)?;
        let image = ImageStatic::new(&config, context)?;
        Ok(Self {
            config,
            definition,
            statics: StaticModules { text, image },
            geometry: Some(geometry),
        })
    }

    /// Retains the admitted compact expert realization without changing invocation geometry.
    pub fn with_expert_realization(
        mut self,
        plan: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        self.statics.text.set_expert_realization(plan)?;
        Ok(self)
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for Model<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <TextKernel<B> as eredu_runtime::ParallelComponentKernel<B, S>>::parallel_observation_hooks(
            &self.statics.text,
        )
    }
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_input(input, state, true, context)
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
    ) -> Result<B::Tensor, Error> {
        if group != 1 {
            return self.begin_execution_group(
                group,
                initial,
                dependencies,
                state,
                forward,
                context,
            );
        }
        if let Some(output) = dependencies.first() {
            forward.vision_output = Some((*output).clone());
        }
        if !forward.deferred {
            return Ok(initial.clone());
        }
        forward.deferred = false;
        if let Some(feedback) = &forward.feedback {
            let prepared = self.statics.text.begin_parallel(
                TextInput::Refinement {
                    tokens: &forward.tokens,
                    feedback: feedback.as_ref(),
                    mask: None,
                },
                state,
                parallel,
                context,
            )?;
            forward.text = prepared.context;
            return Ok(prepared.hidden);
        }
        let embeddings =
            self.statics
                .text
                .encode_context_tokens_parallel(&forward.tokens, parallel, context)?;
        self.assemble_embeddings(forward, &embeddings, context)
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
        self.forward_unit_parallel_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
        )
    }
    fn forward_unit_parallel_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
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
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_parallel_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
            observer,
        )
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics
            .text
            .finish_parallel(hidden, state, &forward.text, parallel, context)
    }
    fn finish_forward_parallel_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.statics.text.finish_parallel_observed(
            hidden,
            state,
            &forward.text,
            parallel,
            context,
            observer,
        )
    }
}
impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for Model<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }
    fn forward_unit_parallel_with_provider<P: eredu_runtime::TensorParallelParameterProvider<B>>(
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
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::fmt::Display,
    {
        self.forward_unit_parallel_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn forward_unit_parallel_observed_with_provider<P, O>(
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
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match (group, unit) {
            (0, Unit::Vision(unit)) => self.statics.image.tower.forward_layer(
                unit,
                hidden,
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("missing image invocation"))?,
                context,
            ),
            (1, Unit::Text(unit)) => self.statics.text.execute_parallel_routed_observed(
                self.statics.text.definition().unit_address(0, index)?,
                unit,
                hidden,
                state,
                &mut forward.text,
                pass,
                provider,
                parallel,
                context,
                observer,
            ),
            _ => Err(Error::backend("unit/group mismatch")),
        }
    }
}
impl<B, S> ParallelCompositeArchitecture<B, S> for Model<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Gemma4InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_forward_parallel(input, state, parallel, context)
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for Model<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Boundary = BoundarySchema;
    fn boundary_schema(&self) -> Result<BoundarySchema, Error> {
        Ok(BoundarySchema {
            hidden: self.config.text.hidden_size,
        })
    }
    fn partition_observation_hooks(
        &self,
        parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        <TextKernel<B> as eredu_runtime::PartitionComponentKernel<B, S>>::partition_observation_hooks(
            &self.statics.text,
            parallel,
        )
    }
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, Boundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.resume(input, mask, state, expected, first, None, context)
    }
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, Boundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.resume(input, mask, state, expected, first, Some(parallel), context)
    }
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if let Some(auxiliary) = forward.pending_text_boundary.take() {
            if group != 1 {
                return Err(Error::backend(
                    "text continuation entered another execution group",
                ));
            }
            let geometry = self
                .geometry
                .as_ref()
                .ok_or_else(|| Error::backend("text continuation has no retained ownership"))?;
            let expected = geometry.text.state_layout().clone();
            let first = geometry.text.units().start;
            let resumed = self.resume(
                LayeredPartitionInput::Hidden {
                    hidden: initial.clone(),
                    auxiliary,
                },
                None,
                state,
                &expected,
                first,
                parallel,
                context,
            )?;
            *forward = resumed.context;
            return Ok(resumed.hidden);
        }
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, Boundary<B::Tensor>>, Error> {
        self.finish_cut(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn finish_partition_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, Boundary<B::Tensor>>, Error> {
        self.finish_cut(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            observer,
        )
    }
}
impl<B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend> Model<B> {
    fn resume<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, Boundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        let origin = self
            .geometry
            .as_ref()
            .ok_or_else(|| Error::backend("composite cut has no retained geometry"))?
            .text
            .units()
            .start;
        if first != origin {
            return Err(Error::backend(
                "composite state origin differs from retained ownership",
            ));
        }
        let (input, spans, visibility) = match input {
            LayeredPartitionInput::Tokens(tokens) => (
                LayeredPartitionInput::Tokens(tokens),
                vec![(-1, -1); tokens.dim(1) as usize],
                mask.cloned(),
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                let words = auxiliary.invocation.to_i32_vec(context)?;
                if words.len() != 2
                    || words[1] < 0
                    || auxiliary.vision_spans.shape() != [hidden.dim(1), 2]
                {
                    return Err(Error::backend(
                        "composite invocation metadata geometry is invalid",
                    ));
                }
                let spans = auxiliary
                    .vision_spans
                    .to_i32_vec(context)?
                    .chunks_exact(2)
                    .map(|row| (row[0], row[1]))
                    .collect::<Vec<_>>();
                let visibility = if words[0] == 0 {
                    let sequence = hidden.dim(1);
                    let frontier = words[1];
                    let mut values = Vec::new();
                    for (query, (start, end)) in spans.iter().enumerate() {
                        if (*start, *end) != (-1, -1)
                            && (*start < 0
                                || *end <= *start
                                || *end > sequence
                                || !(*start..*end).contains(&(query as i32)))
                        {
                            return Err(Error::backend("transported image span is invalid"));
                        }
                        for key in 0..frontier + sequence {
                            values.push(i32::from(
                                self.config.encoder_attention == EncoderAttention::Bidirectional
                                    || key <= frontier + query as i32
                                    || (*start >= 0
                                        && key >= frontier + *start
                                        && key < frontier + *end),
                            ));
                        }
                    }
                    Some(
                        B::Tensor::from_i32_slice(
                            &values,
                            &[1, 1, sequence, frontier + sequence],
                            context,
                        )?
                        .equal_i32(1, context)?,
                    )
                } else {
                    None
                };
                (
                    LayeredPartitionInput::Hidden {
                        hidden,
                        auxiliary: super::super::boundary::TextBoundary {
                            invocation: auxiliary.invocation,
                        },
                    },
                    spans,
                    visibility,
                )
            }
        };
        let prepared = self.statics.text.resume_partition(
            input,
            visibility.as_ref(),
            state,
            expected,
            0,
            parallel,
            context,
        )?;
        let tokens = B::Tensor::from_i32_slice(
            &vec![0; prepared.hidden.dim(1) as usize],
            &[1, prepared.hidden.dim(1)],
            context,
        )?;
        Ok(LayeredForwardState {
            hidden: prepared.hidden,
            context: ForwardContext {
                text: prepared.context,
                segments: Vec::new(),
                vision: None,
                vision_initial: None,
                vision_output: None,
                tokens,
                feedback: None,
                deferred: false,
                vision_spans: spans,
                pending_text_boundary: None,
            },
        })
    }
    fn finish_cut<S, O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, Boundary<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match self.statics.text.finish_partition_observed(
            hidden,
            state,
            &forward.text,
            owns_output,
            parallel,
            context,
            observer,
        )? {
            LayeredPartitionOutput::Final { output, retained } => {
                Ok(LayeredPartitionOutput::Final { output, retained })
            }
            LayeredPartitionOutput::Boundary { hidden, auxiliary } => {
                let values = forward
                    .vision_spans
                    .iter()
                    .flat_map(|(a, b)| [*a, *b])
                    .collect::<Vec<_>>();
                Ok(LayeredPartitionOutput::Boundary {
                    hidden,
                    auxiliary: Boundary {
                        invocation: auxiliary.invocation,
                        vision_spans: B::Tensor::from_i32_slice(
                            &values,
                            &[forward.vision_spans.len() as i32, 2],
                            context,
                        )?,
                    },
                })
            }
        }
    }
}
