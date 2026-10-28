//! Finite encoder pipeline ownership and exact cut-dependent channel/cadence transport.
use crate::{
    ear_encoder::{invalid, EncoderConfig, EncoderUnitSpec},
    ear_encoder_layered::{encoder_local_dimensions, EncoderForward, EncoderKernel},
    Error,
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    ArchitecturePartition, BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype,
    BoundaryTensorSpec, CommunicationRouteId, ComponentAdapter, ComponentDefinition,
    LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput, LocalModelLayout,
    PartitionBoundaryRoute, PartitionComponentKernel, PipelineComponentKernel,
    ResolvedBoundaryWireSchema, StatelessComponentState,
};
use std::{collections::BTreeMap, ops::Range};

fn boundary_error(error: ArchitectureBoundaryError) -> Error {
    eredu_nn::Error::backend_source(error).into()
}

/// Request identity and exact completed global cut accompanying an activation.
#[derive(Clone, Debug)]
pub struct EncoderBoundary<T> {
    /// I32 `[3]`: next global unit, original batch, original spectral frame count.
    pub request: T,
}

/// Typed channel-first transport geometry after a specific number of completed units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderBoundarySchema {
    config: EncoderConfig,
    cut: usize,
    channels: i32,
    reduction: i32,
}
impl EncoderBoundarySchema {
    /// Canonical request declaration retained in the architecture partition.
    pub fn new(config: &EncoderConfig) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            config: config.clone(),
            cut: 0,
            channels: config.fft_size + 2,
            reduction: 1,
        })
    }
    /// Declares one intermediate cut; the final frame-major result is published separately.
    pub fn at_cut(&self, cut: usize) -> Result<Self, Error> {
        let specs = self.config.unit_specs()?;
        if cut >= specs.len() {
            return Err(invalid("encoder boundary cut is not intermediate"));
        }
        let mut value = Self::new(&self.config)?;
        for spec in &specs[..cut] {
            match *spec {
                EncoderUnitSpec::Input { output, .. } => value.channels = output,
                EncoderUnitSpec::Block { .. } => {}
                EncoderUnitSpec::Downsample { output, rate, .. } => {
                    value.channels = output;
                    value.reduction = value
                        .reduction
                        .checked_mul(rate)
                        .ok_or_else(|| invalid("encoder boundary cadence overflows"))?;
                }
            }
        }
        value.cut = cut;
        Ok(value)
    }
    /// Number of globally completed units at this boundary.
    pub fn cut(&self) -> usize {
        self.cut
    }
    /// Channel width retained by the complete activation.
    pub fn channels(&self) -> i32 {
        self.channels
    }
    /// Number of original spectral frames per frame on this route.
    pub fn frame_reduction(&self) -> i32 {
        self.reduction
    }
    /// Resolves the exact cadence from the original request geometry.
    pub fn resolve(
        &self,
        batch: i32,
        spectral_frames: i32,
    ) -> Result<ResolvedBoundaryWireSchema, Error> {
        if batch <= 0 || spectral_frames < self.config.frame_reduction()? {
            return Err(invalid(
                "encoder boundary request contains no complete latent frame",
            ));
        }
        self.wire_schema()
            .and_then(|schema| schema.resolve(batch, spectral_frames / self.reduction))
            .map_err(boundary_error)
    }
}
impl ArchitectureBoundary for EncoderBoundarySchema {
    type Boundary<T> = EncoderBoundary<T>;
    const IDENTITY: &'static str = "ear_codec.encoder.v1";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::new(
            "hidden",
            [Dim::Batch, Dim::Fixed(self.channels), Dim::Sequence],
            Dtype::Activation,
        )
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        vec![BoundaryTensorSpec::new(
            "request",
            [Dim::Fixed(3)],
            Dtype::Int32,
        )]
    }
    fn encode<T>(
        &self,
        boundary: EncoderBoundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(vec![ArchitectureBoundaryValue::new(
            "request",
            boundary.request,
        )?])
    }
    fn decode<T>(&self, tensors: Vec<T>) -> Result<EncoderBoundary<T>, ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(EncoderBoundary {
            request: tensors.into_iter().next().expect("validated cardinality"),
        })
    }
}

/// Exact architecture-owned route identifiers and global unit cuts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderPipelineCuts {
    schema: EncoderBoundarySchema,
    routes: BTreeMap<CommunicationRouteId, usize>,
}
impl EncoderPipelineCuts {
    /// Validates all route identities and cuts without allocating native resources.
    pub fn new(
        config: &EncoderConfig,
        routes: impl IntoIterator<Item = (CommunicationRouteId, usize)>,
    ) -> Result<Self, Error> {
        let schema = EncoderBoundarySchema::new(config)?;
        let mut selected = BTreeMap::new();
        for (route, cut) in routes {
            if cut == 0 {
                return Err(invalid(
                    "encoder pipeline route cannot precede input projection",
                ));
            }
            schema.at_cut(cut)?;
            if selected.insert(route, cut).is_some() {
                return Err(invalid("duplicate encoder pipeline route identity"));
            }
        }
        Ok(Self {
            schema,
            routes: selected,
        })
    }
    /// Exact finite encoder geometry to which every retained route belongs.
    pub fn config(&self) -> &EncoderConfig {
        &self.schema.config
    }
    /// Exact retained route-to-cut mapping for cold manifest validation.
    pub fn routes(&self) -> impl Iterator<Item = (CommunicationRouteId, usize)> + '_ {
        self.routes.iter().map(|(&route, &cut)| (route, cut))
    }
    /// Number of exact retained route declarations.
    pub fn len(&self) -> usize {
        self.routes.len()
    }
    /// Whether the retained plan has no transported cuts.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
    /// Exact cut associated with one opaque route.
    pub fn cut(&self, route: CommunicationRouteId) -> Result<usize, Error> {
        self.routes
            .get(&route)
            .copied()
            .ok_or_else(|| invalid("encoder pipeline route has no retained cut"))
    }
    /// Typed channel geometry for one retained route.
    pub fn schema(&self, route: CommunicationRouteId) -> Result<EncoderBoundarySchema, Error> {
        self.schema.at_cut(self.cut(route)?)
    }
    /// Exact request-relative frame geometry for one retained route.
    pub fn resolved_schema(
        &self,
        route: CommunicationRouteId,
        batch: i32,
        spectral_frames: i32,
    ) -> Result<ResolvedBoundaryWireSchema, Error> {
        self.schema(route)?.resolve(batch, spectral_frames)
    }
}

pub(crate) struct EncoderPartition {
    range: Range<usize>,
    cuts: EncoderPipelineCuts,
}
fn ownership<G>(
    definition: &ComponentDefinition,
    config: &EncoderConfig,
    partition: &ArchitecturePartition<G, EncoderBoundarySchema>,
) -> Result<Range<usize>, Error> {
    let description = definition.parameters();
    let groups = partition.groups();
    if partition.graph() != description.graph()
        || partition.unit_layout() != description.unit_layout()
        || partition.boundary_schema() != &EncoderBoundarySchema::new(config)?
        || !description.matches_partition_bindings(partition)
        || partition.state().is_some()
        || groups.len() != 1
        || groups[0].group_index() != 0
        || !partition.ownership().static_roles().is_empty()
        || !partition.ownership().replicated_static_roles().is_empty()
    {
        return Err(invalid(
            "encoder partition graph, parameters, boundary or stateless ownership differs",
        ));
    }
    let range = groups[0].global_units();
    let count = description.unit_layout().len();
    if range.is_empty()
        || range.end > count
        || partition.ownership().owns_input() != (range.start == 0)
        || partition.ownership().owns_output() != (range.end == count)
    {
        return Err(invalid(
            "encoder stage is not an exact contiguous input/output ownership range",
        ));
    }
    Ok(range)
}
/// Checks stage ownership and complete tensor placement before materialization.
pub fn validate_encoder_partition<G>(
    definition: &ComponentDefinition,
    config: &EncoderConfig,
    prefix: &str,
    layout: Option<&LocalModelLayout>,
    partition: &ArchitecturePartition<G, EncoderBoundarySchema>,
) -> Result<(), Error> {
    if let Some(layout) = layout {
        encoder_local_dimensions(config, prefix, layout)?;
    }
    ownership(definition, config, partition).map(|_| ())
}
impl<B: NeuralBackend> EncoderKernel<B> {
    /// Consumes the retained definition and selects exact stage units and route geometry.
    pub fn new_partition<G>(
        definition: ComponentDefinition,
        config: EncoderConfig,
        prefix: &str,
        layout: Option<&LocalModelLayout>,
        partition: &ArchitecturePartition<G, EncoderBoundarySchema>,
        cuts: &EncoderPipelineCuts,
    ) -> Result<ComponentAdapter<ComponentDefinition, Self>, Error> {
        let range = ownership(&definition, &config, partition)?;
        if cuts.schema != EncoderBoundarySchema::new(&config)? {
            return Err(invalid("encoder pipeline cuts belong to another geometry"));
        }
        let mut value = match layout {
            Some(layout) => Self::new_parallel(config, prefix, layout)?,
            None => Self::new(config, prefix)?,
        };
        value.partition = Some(EncoderPartition {
            range,
            cuts: cuts.clone(),
        });
        Ok(ComponentAdapter::new(definition, value)?.select_partition(partition)?)
    }

    fn selected(&self) -> Result<&EncoderPartition, Error> {
        self.partition
            .as_ref()
            .ok_or_else(|| invalid("encoder pipeline requires selected stage ownership"))
    }
    fn request_dimensions(&self, spectrum: &B::Tensor) -> Result<(i32, i32), Error> {
        if !matches!(spectrum.shape(), [batch,channels,frames] if *batch>0 && *channels==self.config.fft_size+2 && *frames>=self.config.frame_reduction()?)
            || spectrum.element_type() != Some(TensorElementType::F32)
        {
            return Err(invalid(
                "encoder pipeline requires a complete F32 channel-first spectral request",
            ));
        }
        Ok((spectrum.dim(0), spectrum.dim(2)))
    }
}
impl<B: DistributedNeuralBackend> EncoderKernel<B> {
    /// Validates request identity and transported cut before any learned unit is acquired.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_prepared_partition(
        &self,
        input: LayeredPartitionInput<'_, B::Tensor, EncoderBoundary<B::Tensor>>,
        request: Option<(i32, i32)>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, EncoderForward>, Error> {
        let selected = self.selected()?;
        if self.is_parallel() != parallel.is_some() {
            return Err(invalid(
                "encoder partition state or collective selection differs",
            ));
        }
        if let Some(parallel) = parallel {
            self.validate_parallel(parallel)?;
        }
        let (hidden, batch, spectral_frames) = match input {
            LayeredPartitionInput::Tokens(spectrum) => {
                if selected.range.start != 0 {
                    return Err(invalid("encoder spectrum requires the input owner"));
                }
                let (batch, frames) = self.request_dimensions(spectrum)?;
                if request.is_some_and(|expected| expected != (batch, frames)) {
                    return Err(invalid("encoder original request dimensions differ"));
                }
                (spectrum.clone(), batch, frames)
            }
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                if selected.range.start == 0
                    || auxiliary.request.shape() != [3]
                    || auxiliary.request.element_type() != Some(TensorElementType::I32)
                {
                    return Err(invalid(
                        "encoder boundary metadata or ingress ownership differs",
                    ));
                }
                let words = auxiliary.request.to_i32_vec(context)?;
                if words.len() != 3
                    || words[0] != selected.range.start as i32
                    || request.is_some_and(|expected| expected != (words[1], words[2]))
                {
                    return Err(invalid("encoder boundary cut or original request differs"));
                }
                let schema =
                    EncoderBoundarySchema::new(&self.config)?.at_cut(selected.range.start)?;
                let expected = schema.resolve(words[1], words[2])?;
                if hidden.shape() != expected.primary().shape()
                    || hidden.element_type() != Some(TensorElementType::F32)
                {
                    return Err(invalid(
                        "encoder boundary channel geometry, cadence or precision differs",
                    ));
                }
                (hidden, words[1], words[2])
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: EncoderForward {
                batch,
                spectral_frames,
                next_unit: selected.range.start,
            },
        })
    }
}
impl<B: DistributedNeuralBackend>
    PartitionComponentKernel<B, eredu_runtime::StatelessComponentState> for EncoderKernel<B>
{
    type Boundary = EncoderBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Error> {
        EncoderBoundarySchema::new(&self.config)
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, EncoderBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        _: &mut StatelessComponentState,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, EncoderForward>, Error> {
        if mask.is_some() || selected.first_state_ordinal != 0 || selected.expected.is_some() {
            return Err(invalid(
                "finite encoder partition has no mask or mutable state origin",
            ));
        }
        self.begin_prepared_partition(input, None, selected.parallel, selected.native)
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &EncoderForward,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, EncoderBoundary<B::Tensor>>, Error> {
        let selected = self.selected()?;
        if forward.next_unit != selected.range.end
            || owns_output != (selected.range.end == self.specs.len())
            || self.is_parallel() != parallel.is_some()
        {
            return Err(invalid(
                "encoder stage did not complete its selected units or output ownership",
            ));
        }
        if owns_output {
            if hidden.shape()
                != [
                    forward.batch,
                    forward.spectral_frames / self.config.frame_reduction()?,
                    self.config.latent_size,
                ]
                || hidden.element_type() != Some(TensorElementType::F32)
            {
                return Err(invalid(
                    "encoder published latent geometry or precision differs",
                ));
            }
            Ok(LayeredPartitionOutput::Final {
                output: hidden.clone(),
                retained: None,
            })
        } else {
            let schema = EncoderBoundarySchema::new(&self.config)?.at_cut(selected.range.end)?;
            if hidden.shape()
                != schema
                    .resolve(forward.batch, forward.spectral_frames)?
                    .primary()
                    .shape()
                || hidden.element_type() != Some(TensorElementType::F32)
            {
                return Err(invalid("encoder egress shape differs from completed cut"));
            }
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: EncoderBoundary {
                    request: B::Tensor::from_i32_slice(
                        &[
                            selected.range.end as i32,
                            forward.batch,
                            forward.spectral_frames,
                        ],
                        &[3],
                        context,
                    )?,
                },
            })
        }
    }
}
impl<B: DistributedNeuralBackend> PipelineComponentKernel<B, eredu_runtime::StatelessComponentState>
    for EncoderKernel<B>
{
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Error> {
        self.request_dimensions(input)
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, EncoderBoundary<B::Tensor>)>,
        _: &mut StatelessComponentState,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        _: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, EncoderForward>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if selected.first_state_ordinal != 0 {
            return Err(invalid(
                "finite encoder pipeline has no nonlocal mutable state origin",
            ));
        }
        let request = self.request_dimensions(input)?;
        let ingress = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input),
        };
        self.begin_prepared_partition(ingress, Some(request), selected.parallel, selected.native)
    }
    fn output_width(&self) -> i32 {
        self.config.latent_size
    }
    fn output_shape(
        &self,
        _: &Self::Input<'_>,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, Error> {
        if batch <= 0 || sequence < self.config.frame_reduction()? {
            return Err(invalid(
                "encoder publication requires complete latent frames",
            ));
        }
        Ok(vec![
            batch,
            sequence / self.config.frame_reduction()?,
            self.config.latent_size,
        ])
    }
    fn route_schema(&self, route: &PartitionBoundaryRoute) -> Result<Self::Boundary, Error> {
        if route.source_group != 0 || route.destination_group != 0 {
            return Err(invalid(
                "encoder route refers to an unknown execution group",
            ));
        }
        self.selected()?.cuts.schema(route.route)
    }
    fn resolved_route_schema(
        &self,
        route: &PartitionBoundaryRoute,
        batch: i32,
        sequence: i32,
    ) -> Result<ResolvedBoundaryWireSchema, Error> {
        self.route_schema(route)?.resolve(batch, sequence)
    }

    fn tensor_vocabulary_sharded(&self) -> bool {
        false
    }
}
