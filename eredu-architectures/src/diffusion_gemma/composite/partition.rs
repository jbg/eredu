//! Shared-stack invocation and vision-span transport across composite cuts.
use super::*;
use crate::composite_execution::ParallelCompositeArchitecture;
use eredu_nn::TensorParallelGroupedNeuralBackend;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype, BoundaryTensorSpec,
    LayeredPartitionInput, LayeredPartitionOutput,
};

use eredu_runtime::PartitionComponentKernel as _;

/// Shared text activation plus invocation and semantic image spans.
pub struct Boundary<T> {
    /// Architecture invocation phase and authoritative context frontier.
    pub invocation: T,
    /// Image interval for each query row, or `[-1,-1]` for ordinary text.
    pub vision_spans: T,
}
/// Exact composite decoder transport, independent of native tensors.
#[derive(Clone, Copy, PartialEq, Eq)]
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

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> CompositeKernel<B> {
    /// Materializes the exact resources selected by an authoritative composite partition.
    pub fn from_partition(
        config: FamilyConfig,
        partition: &eredu_runtime::ArchitecturePartition<
            super::super::parallel::CompositeGeometry,
            BoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Model<B>, Error> {
        let geometry = partition.local_geometry();
        let definition = std::sync::Arc::new(component_definition(&config, Some(geometry))?);
        definition.select_partition(partition)?;
        if partition.boundary_schema() != &BoundarySchema::new(config.text.hidden_size) {
            return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                component: definition.id().to_owned(),
                resource: "partition boundary geometry".into(),
            }
            .into());
        }
        for (group, expected) in [
            (0, geometry.vision_units.clone()),
            (1, geometry.text.units()),
        ] {
            let selected = partition
                .groups()
                .iter()
                .find(|g| g.group_index() == group)
                .map(|g| g.global_units())
                .unwrap_or(0..0);
            if selected != expected {
                return Err(Error::backend(
                    "composite partition differs from retained local units",
                ));
            }
        }
        let roles = definition.select_complete_pinned_roles(
            partition.parameter_bindings(),
            &["vision", "vision_projection"],
        )?;
        let mut geometry = geometry.clone();
        geometry.text = geometry.text.select_pinned_resources(
            &config,
            &definition,
            partition.parameter_bindings(),
        )?;
        let text = TextKernel::materialize(config.clone(), Some(geometry.text.clone()), context)?;
        let image = ImageStatic::materialize(
            &config,
            roles.contains("vision"),
            roles.contains("vision_projection"),
            context,
        )?;
        let kernel = Self {
            config,
            definition: std::sync::Arc::clone(&definition),
            statics: StaticModules { text, image },
            geometry: Some(geometry),
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }

    /// Retains the admitted compact expert realization without changing invocation geometry.
    pub fn set_expert_realization(
        &mut self,
        plan: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<(), Error> {
        self.statics.text.set_expert_realization(plan)
    }
}

impl<B, S> ParallelCompositeArchitecture<B, S> for Model<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
}
impl<B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend> CompositeKernel<B> {
    pub(super) fn resume<S>(
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
    pub(super) fn finish_cut<S, O>(
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
