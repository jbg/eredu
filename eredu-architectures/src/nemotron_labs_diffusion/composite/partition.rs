//! Retained composite ownership; each stage constructs only selected parameters.
use super::*;
use crate::nemotron_labs_diffusion::partitioned::{
    PartitionedTextKernel, TextBoundary, TextBoundarySchema,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ArchitecturePartition, LayeredPartitionInput,
    LayeredPartitionOutput, LocalModelLayout, PartitionOwnership, StateLayout,
};
use std::ops::Range;

/// Backend-free local geometry and independently selected image/text ranges.
#[derive(Debug, Clone)]
pub struct Geometry {
    pub(super) layout: LocalModelLayout,
    pub(super) vision: Range<usize>,
    pub(super) projection: bool,
    pub(super) text: Option<crate::decoder::PartitionLocalGeometry<FamilyConfig>>,
    pub(super) complete_state: StateLayout,
}
impl Geometry {
    /// Exact source-free rank-local parameter placement retained by this geometry.
    pub fn local_layout(&self) -> &LocalModelLayout {
        &self.layout
    }

    /// Complete tensor-local decoder state before pipeline slicing.
    pub fn complete_state(&self) -> &StateLayout {
        &self.complete_state
    }
}
/// Validates physical tensor shards and cache geometry without native allocations.
pub fn geometry(
    config: &FamilyConfig,
    layout: LocalModelLayout,
    vision: Range<usize>,
    projection: bool,
    text: Option<Range<usize>>,
) -> Result<Geometry, Error> {
    config.validate_config()?;
    let image = config
        .vision
        .as_ref()
        .ok_or_else(|| Error::backend("composite partition requires Pixtral"))?;
    if vision.start > vision.end || vision.end > image.num_hidden_layers as usize {
        return Err(Error::backend(
            "vision ownership exceeds the admitted tower",
        ));
    }
    for index in vision.clone() {
        crate::nemotron_labs_diffusion::parallel::vision_local_widths(image, index, &layout)
            .map_err(Error::backend)?;
    }
    let parameters =
        crate::nemotron_labs_diffusion::parallel::composite_parameter_description(config)
            .map_err(Error::backend)?;
    let complete = crate::decoder::partition_local_geometry(
        config,
        parameters.invocations(),
        &layout,
        0..config.num_hidden_layers() as usize,
    )
    .map_err(Error::backend)?;
    let complete_state = complete.complete_state_layout().clone();
    let text = text
        .map(|range| {
            crate::decoder::partition_local_geometry(
                config,
                parameters.invocations(),
                &layout,
                range,
            )
            .map_err(Error::backend)
        })
        .transpose()?;
    Ok(Geometry {
        layout,
        vision,
        projection,
        text,
        complete_state,
    })
}
/// Pinned modules reflect exact first/last owners, including stateless vision stages.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: DistributedNeuralBackend> {
    /// Shared text embedding/readout roles, only where text units are selected.
    pub text: Option<PartitionedTextKernel<B>>,
    /// Present only on the first image owner.
    pub image_embedding: Option<ImageEmbedding<B>>,
}
/// Selected composite specialization of the ordinary shared family kernel.
pub type SelectedCompositeKernel<B> = super::CompositeKernel<B, StaticModules<B>>;
/// Shared runtime adapter retaining exact selected ownership.
pub type Model<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    SelectedCompositeKernel<B>,
    eredu_runtime::KernelParameterModules,
>;
impl<B: DistributedNeuralBackend> super::CompositeResources<B> for StaticModules<B> {
    const OBSERVED_PARALLEL_READOUT: bool = false;
    type TextModules = crate::decoder::PartitionStaticModules<B>;
    type TextGeometry = crate::decoder::PartitionLocalGeometry<FamilyConfig>;
    fn text(&self) -> Option<&super::CompositeTextKernel<B, Self>> {
        self.text.as_ref()
    }
    fn text_mut(&mut self) -> Option<&mut super::CompositeTextKernel<B, Self>> {
        self.text.as_mut()
    }
    fn image(&self) -> Option<&ImageEmbedding<B>> {
        self.image_embedding.as_ref()
    }
    fn image_mut(&mut self) -> Option<&mut ImageEmbedding<B>> {
        self.image_embedding.as_mut()
    }
}
impl<B: DistributedNeuralBackend> super::CompositeKernel<B, StaticModules<B>> {
    /// Validates retained ownership before materializing only its physical resources.
    pub fn from_partition(
        config: FamilyConfig,
        parameters: &ArchitectureParameterDescription,
        partition: &ArchitecturePartition<Geometry, TextBoundarySchema>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Model<B>, Error> {
        let definition = std::sync::Arc::new(
            crate::nemotron_labs_diffusion::definition::composite_partition(
                &config,
                parameters,
                partition.local_geometry(),
            )?,
        );
        definition.select_partition(partition)?;
        if partition.boundary_schema() != &TextBoundarySchema::new(config.hidden_size()) {
            return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                component: definition.id().to_owned(),
                resource: "partition boundary geometry".into(),
            }
            .into());
        }
        let geometry = partition.local_geometry().clone();
        let vision = partition
            .groups()
            .iter()
            .find(|group| group.group().as_str() == "vision")
            .map(|group| group.global_units().clone())
            .unwrap_or(0..0);
        let text_range = partition
            .groups()
            .iter()
            .find(|group| group.group().as_str() == crate::decoder::TEXT_DECODER_EXECUTION_GROUP)
            .map(|group| group.global_units().clone());
        let projection = partition
            .groups()
            .iter()
            .find(|group| group.group().as_str() == "vision_projection")
            .map(|group| group.global_units().clone());
        if projection.as_ref().is_some_and(|range| *range != (0..1))
            || projection.is_some() != geometry.projection
        {
            return Err(Error::backend(
                "projector unit differs from selected ownership",
            ));
        }
        if vision != geometry.vision
            || text_range.as_ref() != geometry.text.as_ref().map(|g| g.owned_units()).as_ref()
        {
            return Err(Error::backend(
                "composite parameter and execution ownership disagree",
            ));
        }
        let ownership = partition.ownership();
        if ownership.static_roles().iter().any(|role| {
            !["embedding", "norm", "output", "vision_embedding"].contains(&role.as_str())
        }) || !ownership.replicated_static_roles().is_empty()
        {
            return Err(Error::backend(
                "composite partition has undeclared static replicas",
            ));
        }
        if ownership.owns_static_role("norm") != ownership.owns_static_role("output")
            || ownership.owns_output() != ownership.owns_static_role("output")
        {
            return Err(Error::backend(
                "text output ownership requires its normalization and untied head",
            ));
        }
        let text = if let Some(local) = &geometry.text {
            let description =
                crate::nemotron_labs_diffusion::parallel::parameter_description(&config)
                    .map_err(Error::backend)?;
            let text_ownership = PartitionOwnership::new(
                ownership.owns_static_role("embedding"),
                ownership.owns_static_role("output"),
                ownership
                    .static_roles()
                    .iter()
                    .filter(|role| ["embedding", "norm", "output"].contains(&role.as_str()))
                    .cloned(),
            )
            .map_err(Error::backend)?;
            let text_definition = crate::nemotron_labs_diffusion::definition::text_partition(
                &config,
                &description,
                local,
            )?;
            let text_partition = ArchitecturePartition::from_description(
                &description,
                [(
                    crate::decoder::TEXT_DECODER_EXECUTION_GROUP,
                    local.owned_units(),
                )],
                text_ownership,
                text_definition.state_layout(),
                text_definition.state_partition_plan(),
                local.clone(),
                TextBoundarySchema::new(config.hidden_size()),
            )
            .map_err(Error::backend)?;
            if partition.state() != text_partition.state() {
                return Err(Error::backend(
                    "composite text state differs from retained local ownership",
                ));
            }
            Some(
                PartitionedTextKernel::from_partition(
                    config.clone(),
                    &description,
                    &text_partition,
                    context,
                )?
                .into_parts()
                .1,
            )
        } else {
            if partition.state().is_some()
                || ownership.owns_static_role("embedding")
                || ownership.owns_static_role("norm")
                || ownership.owns_static_role("output")
            {
                return Err(Error::backend(
                    "stateless image owner retains text parameters or state",
                ));
            }
            None
        };
        let image_embedding = ownership
            .owns_static_role("vision_embedding")
            .then(|| ImageEmbedding::new(&config, context))
            .transpose()?;
        let kernel = Self {
            config,
            geometry: Some(geometry),
            local_layout: None,
            backend: std::marker::PhantomData,
            definition: definition.clone(),
            statics: StaticModules {
                text,
                image_embedding,
            },
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }
}
impl<B, S> CompositeArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a> {
        input
    }
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<
        crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    > {
        Some(*input)
    }

    type InputPartPlan = InputPartPlan;
    type PrefillRequest = SingleCompositePrefillRequest<B::Tensor, InputPartPlan>;
    type AdmissionConfig = FamilyConfig;
    fn supports_chunked_text_prefill() -> bool {
        true
    }
    fn supports_attention_invocation(_: &FamilyConfig, invocation: AttentionInvocation) -> bool {
        [
            AttentionInvocation::CAUSAL,
            AttentionInvocation::DIFFUSION,
            AttentionInvocation::DRAFT,
        ]
        .contains(&invocation)
    }
    fn block_refinement_geometry(
        config: &FamilyConfig,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        <super::Model<B> as CompositeArchitecture<B, S>>::block_refinement_geometry(config)
    }
    fn prepare_prefill_request(
        prepared: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<InputPartPlan>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Error> {
        SingleCompositePrefillRequest::new(prepared, admitted)
    }
    fn admission_config(&self) -> FamilyConfig {
        self.kernel().config.clone()
    }
    fn prepared_prediction_token_ids(
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        <super::Model<B> as CompositeArchitecture<B, S>>::prepared_prediction_token_ids(
            input, context,
        )
    }
    fn admit_prepared_input(
        config: &FamilyConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<InputPartPlan>, eredu_core::CapabilityError> {
        admit_input(config, input, inspector)
    }
    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
    ) -> bool {
        group == 2
            || ((group == 0 || group == 1)
                && input
                    .admitted()
                    .parts()
                    .iter()
                    .any(|part| matches!(part, InputPartPlan::Image { native: true, .. })))
    }

    fn primary_ingress_collectives_pending(&self, forward: &Self::ForwardContext) -> bool {
        self.kernel().statics.text.is_some() && forward.text.is_none()
    }
    fn prepared_primary_ingress_collectives(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        Ok(Some(if tensor_partitions > 1 {
            vec![crate::composite_execution::CompositeTensorCollective::Sum {
                shape: vec![
                    1,
                    i32::try_from(input.admitted().decoder_positions())
                        .map_err(|_| "decoder extent exceeds i32")?,
                    self.kernel().config.hidden_size(),
                ],
            }]
        } else {
            Vec::new()
        }))
    }
    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
    ) -> Result<i32, String> {
        if group == 0 {
            return native_patch_positions(input);
        }
        let positions = if group == 1 {
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    InputPartPlan::Image {
                        grid, native: true, ..
                    } => Some(
                        (grid.0 as u64) * (grid.1 as u64)
                            / u64::from(self.kernel().config.spatial_merge_size).pow(2),
                    ),
                    _ => None,
                })
                .try_fold(0u64, |n, count| n.checked_add(count))
                .ok_or("projector extent overflow")?
        } else {
            input.admitted().decoder_positions()
        };
        i32::try_from(positions).map_err(|_| "group extent exceeds i32".into())
    }
    fn prepared_group_continuation_batched(&self, _: usize) -> bool {
        false
    }
    fn partition_boundary_values(
        &self,
        source: usize,
        _: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        hidden: &B::Tensor,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>>, Error> {
        encode_image_boundary(source, schema, hidden, context)
    }
    fn accept_partition_boundary(
        &mut self,
        source: usize,
        _: usize,
        _: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        _: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        accept_image_boundary(source, values, context)
    }
    fn partition_boundary_schema(
        &self,
        source: usize,
        destination: usize,
        selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        batch: i32,
        sequence: i32,
        _: &[i32],
        continuation: Option<(i32, i32)>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Error> {
        resolve_group_boundary(
            &self.kernel().config,
            source,
            destination,
            selected,
            batch,
            sequence,
            continuation,
        )
    }
    fn prepared_group_continuation_geometry(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
    ) -> Result<Option<crate::composite_execution::CompositeContinuationGeometry>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = native_patch_positions(input)?;
        Ok((patches > 0).then_some(
            (
                patches,
                self.kernel()
                    .config
                    .vision
                    .as_ref()
                    .expect("validated tower")
                    .hidden_size,
            )
                .into(),
        ))
    }
    fn prepared_group_collective_waves(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, InputPartPlan>,
        tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        let pipeline_stages = stage_units.len();
        if tensor_partitions <= 1 {
            return Ok(None);
        }
        if group == 2 {
            let sequence = i32::try_from(input.admitted().decoder_positions())
                .map_err(|_| "decoder sequence exceeds i32")?;
            let width = self.kernel().config.hidden_size();
            let vocabulary = self.kernel().config.vocabulary_size() as usize;
            let counts = (0..tensor_partitions)
                .map(|rank| {
                    eredu_core::balanced_contiguous_range(
                        vocabulary,
                        tensor_partitions,
                        rank,
                        false,
                    )
                    .map(|range| range.len())
                    .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let last = self.kernel().config.num_hidden_layers() as usize;
            return Ok(Some(stage_units.iter().map(|range| {
                let mut operations = Vec::new();
                if !range.is_empty() && range.start == 0 {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                }
                for _ in range.clone() {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                    if input.attention_invocation().is_some_and(|invocation| invocation.adapters == eredu_core::decoding::AdapterScope::Draft) {
                        if let Some(adapter) = self.kernel().config.draft_adapter() {
                            operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, adapter.rank()] });
                        }
                    }
                    operations.push(crate::composite_execution::CompositeTensorCollective::Sum { shape: vec![1, sequence, width] });
                }
                if !range.is_empty() && range.end == last {
                    operations.push(crate::composite_execution::CompositeTensorCollective::Gather { shape: vec![1, sequence, vocabulary as i32], counts: counts.clone(), axis: 2 });
                }
                operations
            }).collect()));
        }
        if group != 0 {
            return Ok(None);
        }
        if pipeline_stages == 0 {
            return Err("Pixtral collective schedule requires a positive stage count".into());
        }
        let patches = native_patch_positions(input)?;
        let vision = self
            .kernel()
            .config
            .vision
            .as_ref()
            .expect("validated tower");
        let mut stages = Vec::with_capacity(pipeline_stages);
        for range in &stage_units[..pipeline_stages] {
            stages.push(if patches == 0 {
                Vec::new()
            } else {
                vec![
                    crate::composite_execution::CompositeTensorCollective::Sum {
                        shape: vec![1, patches, vision.hidden_size]
                    };
                    2 * range.len()
                ]
            });
        }
        Ok(Some(stages))
    }
}
impl<B, S> ParallelCompositeArchitecture<B, S> for Model<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
}
impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for SelectedCompositeKernel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Boundary = TextBoundarySchema;
    fn boundary_schema(&self) -> Result<TextBoundarySchema, Error> {
        Ok(TextBoundarySchema::new(self.config.hidden_size()))
    }
    fn partition_observation_hooks(
        &self,
        _: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as eredu_runtime::ComponentEquations<B, S>>::observation_hooks(self)
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_partition_observed(
            input,
            mask,
            state,
            selected,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_partition_observed<
        'a,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let context = selected.native;
        let expected = selected
            .expected
            .ok_or_else(|| Error::backend("text boundary requires selected state geometry"))?;
        let first = selected.first_state_ordinal;
        let parallel = selected.parallel;
        let origin = self
            .geometry
            .as_ref()
            .expect("retained selected geometry")
            .text
            .as_ref()
            .ok_or_else(|| {
                Error::backend("stateless image owner cannot resume a decoder boundary")
            })?
            .owned_units()
            .start;
        if first != origin {
            return Err(Error::backend(
                "decoder boundary origin differs from retained local ownership",
            ));
        }
        let result = eredu_runtime::PartitionComponentKernel::begin_partition_observed(
            self.text()?,
            input,
            mask,
            state,
            eredu_runtime::ComponentPartitionContext {
                expected: Some(expected),
                first_state_ordinal: 0,
                parallel,
                native: context,
            },
            observer,
        )?;
        let tokens = B::Tensor::from_i32_slice(
            &vec![0; result.hidden.dim(0) as usize * result.hidden.dim(1) as usize],
            &[result.hidden.dim(0), result.hidden.dim(1)],
            context,
        )?;
        let invocation = result.context.invocation;
        Ok(LayeredForwardState {
            hidden: result.hidden,
            context: ForwardContext {
                tokens,
                segments: Vec::new(),
                image_initial: None,
                image_state: None,
                image_output: None,
                text: Some(result.context),
                invocation,
            },
        })
    }
    fn enter_partition_group(
        &mut self,
        _: usize,
        initial: &B::Tensor,
        _: &mut S,
        _: &mut Self::ForwardContext,
        _: Option<&B::ParallelContext>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error> {
        self.text()?.finish_partition_observed(
            hidden,
            state,
            forward
                .text
                .as_ref()
                .ok_or_else(|| Error::backend("text boundary has no settled invocation"))?,
            owns_output,
            parallel,
            context,
            observer,
        )
    }
}
impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for SelectedCompositeKernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _: eredu_runtime::ExpertPass,
        _: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        eredu_runtime::ParallelComponentKernel::execute_parallel(
            self, address, unit, hidden, state, forward, parallel, context,
        )
    }
}
