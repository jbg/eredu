//! Streaming decoder pipeline ownership and exact cut-dependent channel/cadence transport.
use crate::{
    ear_decoder::{invalid, DecoderConfig, DecoderState, DecoderUnitSpec, DecoderUnitState},
    ear_decoder_layered::{
        decoder_local_dimensions, decoder_parameter_description, decoder_state_layout,
        DecoderForward, DecoderKernel, DecoderRuntimeState,
    },
    streaming::StreamPhase,
    Error,
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    ArchitecturePartition, BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype,
    BoundaryTensorSpec, CommunicationRouteId, ComponentAdapter, ComponentDefinition,
    LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput, LocalModelLayout,
    PartitionBoundaryRoute, PartitionComponentKernel, PipelineComponentKernel,
    ResolvedBoundaryWireSchema, StateLayout,
};
use std::{collections::BTreeMap, ops::Range};

fn boundary_error(error: ArchitectureBoundaryError) -> Error {
    eredu_nn::Error::backend_source(error).into()
}

/// Request identity and exact completed global cut accompanying an activation.
#[derive(Clone, Debug)]
pub struct DecoderBoundary<T> {
    /// I32 `[3]`: next global unit, original batch, original latent frame count.
    pub request: T,
}

/// Typed channel-first transport geometry after a specific number of completed units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderBoundarySchema {
    config: DecoderConfig,
    cut: usize,
    channels: i32,
    expansion: i32,
}
impl DecoderBoundarySchema {
    /// Canonical request declaration retained in the architecture partition.
    pub fn new(config: &DecoderConfig) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            config: config.clone(),
            cut: 0,
            channels: config.latent_size,
            expansion: 1,
        })
    }
    /// Declares one intermediate cut; the final spectral result is published separately.
    pub fn at_cut(&self, cut: usize) -> Result<Self, Error> {
        let specs = self.config.unit_specs()?;
        if cut >= specs.len() {
            return Err(invalid("decoder boundary cut is not intermediate"));
        }
        let mut value = Self::new(&self.config)?;
        for spec in &specs[..cut] {
            match *spec {
                DecoderUnitSpec::Output { output, .. } => value.channels = output,
                DecoderUnitSpec::Block { .. } => {}
                DecoderUnitSpec::Upsample { output, rate, .. } => {
                    value.channels = output;
                    value.expansion = value
                        .expansion
                        .checked_mul(rate)
                        .ok_or_else(|| invalid("decoder boundary cadence overflows"))?;
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
    /// Number of route frames produced per original latent frame.
    pub fn frame_expansion(&self) -> i32 {
        self.expansion
    }
    /// Resolves the exact cadence from the original request geometry.
    pub fn resolve(
        &self,
        batch: i32,
        latent_frames: i32,
    ) -> Result<ResolvedBoundaryWireSchema, Error> {
        if batch <= 0 || latent_frames <= 0 {
            return Err(invalid(
                "decoder boundary request contains no complete latent frame",
            ));
        }
        let frames = latent_frames
            .checked_mul(self.expansion)
            .ok_or_else(|| invalid("decoder boundary frame count overflows"))?;
        self.wire_schema()
            .and_then(|schema| schema.resolve(batch, frames))
            .map_err(boundary_error)
    }
}
impl ArchitectureBoundary for DecoderBoundarySchema {
    type Boundary<T> = DecoderBoundary<T>;
    const IDENTITY: &'static str = "ear_codec.decoder.v1";
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
        boundary: DecoderBoundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(vec![ArchitectureBoundaryValue::new(
            "request",
            boundary.request,
        )?])
    }
    fn decode<T>(&self, tensors: Vec<T>) -> Result<DecoderBoundary<T>, ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(DecoderBoundary {
            request: tensors.into_iter().next().expect("validated cardinality"),
        })
    }
}

/// Exact architecture-owned route identifiers and global unit cuts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderPipelineCuts {
    schema: DecoderBoundarySchema,
    routes: BTreeMap<CommunicationRouteId, usize>,
}
impl DecoderPipelineCuts {
    /// Validates all route identities and cuts without allocating native resources.
    pub fn new(
        config: &DecoderConfig,
        routes: impl IntoIterator<Item = (CommunicationRouteId, usize)>,
    ) -> Result<Self, Error> {
        let schema = DecoderBoundarySchema::new(config)?;
        let mut selected = BTreeMap::new();
        for (route, cut) in routes {
            if cut == 0 {
                return Err(invalid(
                    "decoder pipeline route cannot precede input upsampling",
                ));
            }
            schema.at_cut(cut)?;
            if selected.insert(route, cut).is_some() {
                return Err(invalid("duplicate decoder pipeline route identity"));
            }
        }
        Ok(Self {
            schema,
            routes: selected,
        })
    }
    /// Exact streaming decoder geometry to which every retained route belongs.
    pub fn config(&self) -> &DecoderConfig {
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
            .ok_or_else(|| invalid("decoder pipeline route has no retained cut"))
    }
    /// Typed channel geometry for one retained route.
    pub fn schema(&self, route: CommunicationRouteId) -> Result<DecoderBoundarySchema, Error> {
        self.schema.at_cut(self.cut(route)?)
    }
    /// Exact request-relative frame geometry for one retained route.
    pub fn resolved_schema(
        &self,
        route: CommunicationRouteId,
        batch: i32,
        latent_frames: i32,
    ) -> Result<ResolvedBoundaryWireSchema, Error> {
        self.schema(route)?.resolve(batch, latent_frames)
    }
}

pub(crate) struct DecoderPartition {
    pub(crate) range: Range<usize>,
    cuts: DecoderPipelineCuts,
    layout: StateLayout,
}
fn ownership<G>(
    config: &DecoderConfig,
    prefix: &str,
    partition: &ArchitecturePartition<G, DecoderBoundarySchema>,
) -> Result<Range<usize>, Error> {
    let description = decoder_parameter_description(config, prefix)?;
    let groups = partition.groups();
    if partition.graph() != description.graph()
        || partition.unit_layout() != description.unit_layout()
        || partition.boundary_schema() != &DecoderBoundarySchema::new(config)?
        || partition.parameter_bindings() != description.select_owned(partition)
        || groups.len() != 1
        || groups[0].group_index() != 0
        || !partition.ownership().static_roles().is_empty()
        || !partition.ownership().replicated_static_roles().is_empty()
    {
        return Err(invalid(
            "decoder partition graph, parameters or boundary ownership differs",
        ));
    }
    let range = groups[0].global_units();
    let count = config.unit_specs()?.len();
    if range.is_empty()
        || range.end > count
        || partition.ownership().owns_input() != (range.start == 0)
        || partition.ownership().owns_output() != (range.end == count)
    {
        return Err(invalid(
            "decoder stage is not an exact contiguous input/output ownership range",
        ));
    }
    let state = partition
        .state()
        .ok_or_else(|| invalid("decoder partition lacks selected cache geometry"))?;
    if state.global_layers() != range
        || state.layout() != &decoder_state_layout(config)?.slice(range.clone())?
    {
        return Err(invalid(
            "decoder partition cache geometry differs from its selected units",
        ));
    }
    Ok(range)
}
/// Checks stage ownership and complete tensor placement before materialization.
pub fn validate_decoder_partition<G>(
    config: &DecoderConfig,
    prefix: &str,
    layout: Option<&LocalModelLayout>,
    partition: &ArchitecturePartition<G, DecoderBoundarySchema>,
) -> Result<(), Error> {
    if let Some(layout) = layout {
        decoder_local_dimensions(config, prefix, layout)?;
    }
    ownership(config, prefix, partition).map(|_| ())
}
impl<B: NeuralBackend> DecoderKernel<B> {
    /// Selects exact unit ownership and retained route geometry for one pipeline stage.
    pub fn new_partition<G>(
        config: DecoderConfig,
        prefix: &str,
        layout: Option<&LocalModelLayout>,
        partition: &ArchitecturePartition<G, DecoderBoundarySchema>,
        cuts: &DecoderPipelineCuts,
    ) -> Result<ComponentAdapter<ComponentDefinition, Self>, Error> {
        let range = ownership(&config, prefix, partition)?;
        if cuts.schema != DecoderBoundarySchema::new(&config)? {
            return Err(invalid("decoder pipeline cuts belong to another geometry"));
        }
        let mut value = match layout {
            Some(layout) => Self::new_parallel(config, prefix, layout)?,
            None => Self::new(config, prefix)?,
        };
        let layout = decoder_state_layout(&value.config)?.slice(range.clone())?;
        value.partition = Some(DecoderPartition {
            range,
            cuts: cuts.clone(),
            layout,
        });
        Ok(value.into_component()?.select_partition(partition)?)
    }

    fn selected(&self) -> Result<&DecoderPartition, Error> {
        self.partition
            .as_ref()
            .ok_or_else(|| invalid("decoder pipeline requires selected stage ownership"))
    }
    fn request_dimensions(&self, latent: &B::Tensor) -> Result<(i32, i32), Error> {
        if !matches!(latent.shape(), [batch,frames,channels] if *batch>0 && *frames>0 && *channels==self.config.latent_size)
            || latent.element_type() != Some(TensorElementType::F32)
        {
            return Err(invalid(
                "decoder pipeline requires nonempty F32 frame-major latent input",
            ));
        }
        expanded_frames(&self.config, latent.dim(1))?;
        Ok((latent.dim(0), latent.dim(1)))
    }
    pub(crate) fn state_index(
        &self,
        state: &DecoderRuntimeState<B::Tensor>,
        index: usize,
    ) -> Result<usize, Error> {
        let start = match (&self.partition, &state.partition) {
            (Some(selected), Some(identity)) if identity.matches(self, selected) => {
                selected.range.start
            }
            (None, None) => 0,
            _ => {
                return Err(invalid(
                    "decoder state belongs to different stage ownership",
                ))
            }
        };
        index
            .checked_sub(start)
            .filter(|local| *local < state.value.units.len())
            .ok_or_else(|| invalid("decoder state omits selected unit"))
    }
}
fn expanded_frames(config: &DecoderConfig, frames: i32) -> Result<i32, Error> {
    if frames <= 0 {
        return Err(invalid("decoder requires nonempty latent frames"));
    }
    config
        .rates
        .iter()
        .try_fold(frames, |frames, rate| frames.checked_mul(*rate))
        .ok_or_else(|| invalid("decoder spectral frame count overflows"))
}

/// Immutable identity of the state owned by one stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecoderPartitionStateIdentity {
    config: DecoderConfig,
    prefix: String,
    range: Range<usize>,
}
impl DecoderPartitionStateIdentity {
    fn new<B: NeuralBackend>(architecture: &DecoderKernel<B>, selected: &DecoderPartition) -> Self {
        Self {
            config: architecture.config.clone(),
            prefix: architecture.prefix.clone(),
            range: selected.range.clone(),
        }
    }
    fn matches<B: NeuralBackend>(
        &self,
        architecture: &DecoderKernel<B>,
        selected: &DecoderPartition,
    ) -> bool {
        self.config == architecture.config
            && self.prefix == architecture.prefix
            && self.range == selected.range
    }
}
impl<T: Tensor> DecoderRuntimeState<T> {
    /// Initializes only this stage's state or validates and clones its immutable parent carry.
    pub fn for_partition<B: NeuralBackend<Tensor = T>>(
        architecture: &DecoderKernel<B>,
        previous: Option<&Self>,
    ) -> Result<Self, Error> {
        let selected = architecture.selected()?;
        if let Some(previous) = previous {
            previous.validate_partition(architecture)?;
            return Ok(previous.clone());
        }
        let units = architecture.specs[selected.range.clone()]
            .iter()
            .copied()
            .map(DecoderUnitSpec::initial_state)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layout: selected.layout.clone(),
            value: DecoderState {
                units,
                batch: None,
                phase: StreamPhase::AcceptingInput,
            },
            partition: Some(DecoderPartitionStateIdentity::new(architecture, selected)),
        })
    }
    /// Checks exact stage ownership, unit geometry and lifecycle without parameter access.
    pub fn validate_partition<B: NeuralBackend<Tensor = T>>(
        &self,
        architecture: &DecoderKernel<B>,
    ) -> Result<(), Error> {
        let selected = architecture.selected()?;
        if !self
            .partition
            .as_ref()
            .is_some_and(|identity| identity.matches(architecture, selected))
            || self.layout != selected.layout
            || self.value.units.len() != selected.range.len()
            || self
                .value
                .units
                .iter()
                .zip(&architecture.specs[selected.range.clone()])
                .any(|(state, spec)| {
                    state.spec != *spec
                        || state.phase != self.value.phase
                        || state.batch != self.value.batch
                })
        {
            return Err(invalid(
                "decoder mutable state belongs to another stage, geometry or lifecycle",
            ));
        }
        Ok(())
    }
    /// Validates stage identity and its batch/lifecycle against enclosing waveform carry.
    /// Empty accepting states have no retained batch until their first learned step.
    pub fn validate_stream_partition<B: NeuralBackend<Tensor = T>>(
        &self,
        architecture: &DecoderKernel<B>,
        batch: i32,
        phase: StreamPhase,
    ) -> Result<(), Error> {
        self.validate_partition(architecture)?;
        if batch <= 0
            || self.value.batch.is_some_and(|retained| retained != batch)
            || self.value.phase != phase
        {
            return Err(invalid(
                "decoder stage batch or lifecycle differs from waveform carry",
            ));
        }
        Ok(())
    }
    /// Exact globally indexed unit interval represented by this state owner.
    pub fn global_units(&self) -> Range<usize> {
        self.partition
            .as_ref()
            .map(|identity| identity.range.clone())
            .unwrap_or(0..self.value.units.len())
    }
    /// Individually retained state entries in local execution order.
    pub fn units(&self) -> &[DecoderUnitState<T>] {
        &self.value.units
    }
    /// Finite-stream lifecycle of the retained state owner.
    pub fn phase(&self) -> StreamPhase {
        self.value.phase
    }
    /// Releases only this stage's causal carry without acquiring parameters or producing a tail.
    pub fn finish_partition<B: NeuralBackend<Tensor = T>>(
        &self,
        architecture: &DecoderKernel<B>,
    ) -> Result<Self, Error> {
        self.validate_partition(architecture)?;
        let mut next = Self::for_partition(architecture, None)?;
        next.value.phase = StreamPhase::Finished;
        next.value.batch = self.value.batch;
        for (state, previous) in next.value.units.iter_mut().zip(&self.value.units) {
            state.phase = StreamPhase::Finished;
            state.batch = previous.batch;
        }
        Ok(next)
    }
}
impl<T: Tensor> eredu_nn::RetainedTensorRoots<T> for DecoderRuntimeState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        eredu_nn::RetainedTensorRoots::visit_retained_tensors(&self.value, visitor);
    }
}
use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate},
    BackendFailure,
};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};
impl SnapshotValue for DecoderPartitionStateIdentity {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.config.estimate_snapshot_value(),
            Ok(SnapshotEstimate {
                retained_bytes: self.prefix.capacity() as u64,
                copy_bytes: self.prefix.len() as u64,
            }),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut prefix = String::new();
        prefix
            .try_reserve_exact(self.prefix.len())
            .map_err(BackendFailure::from_error)?;
        prefix.push_str(&self.prefix);
        Ok(Self {
            config: self.config.copy_snapshot_value(retention)?,
            prefix,
            range: self.range.clone(),
        })
    }
}
impl<T: Tensor + SnapshotValue> SnapshotValue for DecoderRuntimeState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = self
            .layout
            .logical_metadata_bytes()
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        snapshot_value_estimate::<Self>([
            self.value.estimate_snapshot_value(),
            self.partition.estimate_snapshot_value(),
            Ok(SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes: bytes,
            }),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            layout: self.layout.clone(),
            value: self.value.copy_snapshot_value(retention)?,
            partition: self.partition.copy_snapshot_value(retention)?,
        })
    }
}

impl<B: DistributedNeuralBackend> DecoderKernel<B> {
    /// Validates the exact stage cache, request and upstream cut before parameter acquisition.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_prepared_partition(
        &self,
        input: LayeredPartitionInput<'_, B::Tensor, DecoderBoundary<B::Tensor>>,
        request: Option<(i32, i32)>,
        state: &mut DecoderRuntimeState<B::Tensor>,
        expected: Option<&StateLayout>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, DecoderForward>, Error> {
        let selected = self.selected()?;
        state.validate_partition(self)?;
        if expected != Some(&selected.layout) || self.is_parallel() != parallel.is_some() {
            return Err(invalid(
                "decoder selected cache layout or collective placement differs",
            ));
        }
        if state.value.phase == StreamPhase::Finished {
            return Err(Error::StreamFinished);
        }
        if let Some(parallel) = parallel {
            self.validate_parallel(parallel)?;
        }
        let (hidden, batch, latent_frames) = match input {
            LayeredPartitionInput::Tokens(latent) => {
                if selected.range.start != 0 {
                    return Err(invalid("latent ingress requires the decoder input owner"));
                }
                let (batch, frames) = self.request_dimensions(latent)?;
                if request.is_some_and(|expected| expected != (batch, frames)) {
                    return Err(invalid("decoder original request dimensions differ"));
                }
                (latent.swap_axes(1, 2, context)?, batch, frames)
            }
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                if selected.range.start == 0
                    || auxiliary.request.shape() != [3]
                    || auxiliary.request.element_type() != Some(TensorElementType::I32)
                {
                    return Err(invalid(
                        "decoder boundary metadata or ingress ownership differs",
                    ));
                }
                let words = auxiliary.request.to_i32_vec(context)?;
                if words.len() != 3
                    || words[0] != selected.range.start as i32
                    || request.is_some_and(|expected| expected != (words[1], words[2]))
                {
                    return Err(invalid("decoder boundary cut or original request differs"));
                }
                expanded_frames(&self.config, words[2])?;
                let schema =
                    DecoderBoundarySchema::new(&self.config)?.at_cut(selected.range.start)?;
                if hidden.shape() != schema.resolve(words[1], words[2])?.primary().shape()
                    || hidden.element_type() != Some(TensorElementType::F32)
                {
                    return Err(invalid(
                        "decoder boundary channel geometry, cadence or precision differs",
                    ));
                }
                (hidden, words[1], words[2])
            }
        };
        if state.value.batch.is_some_and(|previous| previous != batch) {
            return Err(invalid(
                "decoder pipeline batch changed with retained carry",
            ));
        }
        state.value.batch = Some(batch);
        Ok(LayeredForwardState {
            hidden,
            context: DecoderForward {
                batch,
                latent_frames,
                next_unit: selected.range.start,
            },
        })
    }
}
impl<B: DistributedNeuralBackend> PartitionComponentKernel<B> for DecoderKernel<B> {
    type Boundary = DecoderBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Error> {
        DecoderBoundarySchema::new(&self.config)
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, DecoderBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut DecoderRuntimeState<B::Tensor>,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, DecoderForward>, Error> {
        if mask.is_some() || selected.first_state_ordinal != 0 {
            return Err(invalid(
                "decoder partition requires its local cache origin and no external mask",
            ));
        }
        self.begin_prepared_partition(
            input,
            None,
            state,
            selected.expected,
            selected.parallel,
            selected.native,
        )
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut DecoderRuntimeState<B::Tensor>,
        forward: &DecoderForward,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, DecoderBoundary<B::Tensor>>, Error> {
        let selected = self.selected()?;
        if forward.next_unit != selected.range.end
            || owns_output != (selected.range.end == self.specs.len())
            || self.is_parallel() != parallel.is_some()
        {
            return Err(invalid(
                "decoder stage did not complete its selected units or output ownership",
            ));
        }
        state.validate_partition(self)?;
        if owns_output {
            if hidden.shape()
                != [
                    forward.batch,
                    self.config.fft_size + 2,
                    expanded_frames(&self.config, forward.latent_frames)?,
                ]
                || hidden.element_type() != Some(TensorElementType::F32)
            {
                return Err(invalid(
                    "decoder published spectral geometry or precision differs",
                ));
            }
            Ok(LayeredPartitionOutput::Final {
                output: hidden.clone(),
                retained: None,
            })
        } else {
            let schema = DecoderBoundarySchema::new(&self.config)?.at_cut(selected.range.end)?;
            if hidden.shape()
                != schema
                    .resolve(forward.batch, forward.latent_frames)?
                    .primary()
                    .shape()
                || hidden.element_type() != Some(TensorElementType::F32)
            {
                return Err(invalid(
                    "decoder egress shape differs from its completed cut",
                ));
            }
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: DecoderBoundary {
                    request: B::Tensor::from_i32_slice(
                        &[
                            selected.range.end as i32,
                            forward.batch,
                            forward.latent_frames,
                        ],
                        &[3],
                        context,
                    )?,
                },
            })
        }
    }
}
impl<B: DistributedNeuralBackend> PipelineComponentKernel<B> for DecoderKernel<B> {
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Error> {
        self.request_dimensions(input)
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, DecoderBoundary<B::Tensor>)>,
        state: &mut DecoderRuntimeState<B::Tensor>,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        _: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, DecoderForward>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if selected.first_state_ordinal != 0 {
            return Err(invalid(
                "decoder pipeline requires the selected local cache origin",
            ));
        }
        let request = self.request_dimensions(input)?;
        let ingress = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input),
        };
        self.begin_prepared_partition(
            ingress,
            Some(request),
            state,
            selected.expected,
            selected.parallel,
            selected.native,
        )
    }
    fn output_width(&self) -> i32 {
        self.config.fft_size + 2
    }
    fn output_shape(
        &self,
        _: &Self::Input<'_>,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, Error> {
        if batch <= 0 {
            return Err(invalid("decoder publication requires nonempty batch"));
        }
        Ok(vec![
            batch,
            self.config.fft_size + 2,
            expanded_frames(&self.config, sequence)?,
        ])
    }
    fn route_schema(&self, route: &PartitionBoundaryRoute) -> Result<Self::Boundary, Error> {
        if route.source_group != 0 || route.destination_group != 0 {
            return Err(invalid(
                "decoder route refers to an unknown execution group",
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
