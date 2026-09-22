//! Native local child ceilings of an architecture-owned dynamic region.
use super::*;
use eredu_nn::workspace::{WorkspaceExpertRegion, WorkspaceExpertRegionView};
use eredu_nn::{GroupSelection, Tensor};
use std::mem::size_of;
#[path = "expert_region/movement.rs"]
mod movement;
pub(crate) use movement::{ExpertMovementKind, ExpertMovementQuote};
#[path = "expert_region/transport.rs"]
mod transport;
use transport::ExpertTransferGeometry;
pub(crate) use transport::{ExpertTransferProfile, ExpertTransportQuote};
#[path = "expert_region/aggregate.rs"]
mod aggregate;
pub(crate) use aggregate::ExpertRegionAggregate;
#[path = "expert_region/counts.rs"]
mod counts;
pub(crate) use counts::ExpertCountQuote;
#[path = "expert_region/provider.rs"]
mod provider;
pub(crate) use provider::ExpertProviderQuote;
#[path = "expert_region/provider_wave.rs"]
mod provider_wave;
pub(crate) use provider_wave::ExpertProviderWaveQuote;
#[path = "expert_region/inactive_wave.rs"]
mod inactive_wave;
pub(crate) use inactive_wave::ExpertInactiveWaveQuote;
#[path = "expert_region/reorder.rs"]
mod reorder;
pub(crate) use reorder::ExpertReorderEnvelope;
#[path = "expert_region/local_transport.rs"]
mod local_transport;
pub(crate) use local_transport::{ExpertLocalStage, ExpertLocalStageBound, ExpertLocalTransport};

#[path = "expert_region/observation.rs"]
mod observation;
pub(crate) use observation::{
    ExpertLocalObservationSource, SourceObserver as GroupedSourceObserver,
};

/// The complete region owns this numerical child plus movement and transport
/// sources. This partial source alone is never returned as a complete op fact.
pub(crate) struct ExpertLocalQuote {
    pub(crate) declaration: WorkspaceExpertRegion,
    pub(crate) inputs: Vec<WorkspaceLayout>,
    pub(crate) outputs: Vec<WorkspaceLayout>,
    pub(crate) movement: Option<ExpertMovementQuote>,
    pub(crate) transport: Option<ExpertTransportQuote>,
    pub(crate) aggregate: Option<ExpertRegionAggregate>,
    pub(crate) counts: Option<ExpertCountQuote>,
    pub(crate) provider: Option<ExpertProviderQuote>,
    pub(crate) maximum_rows: usize,
    pub(crate) capacity: BoundaryStageCapacity,
    pub(crate) maximum_births: usize,
    pub(crate) controls: u64,
    pub(crate) kernels: usize,
    pub(crate) empty: SpeculativeNumericalRecipe,
    pub(crate) selected: Option<SpeculativeNumericalRecipe>,
    pub(crate) addressable: Option<AddressableQuoteRef>,
    /// Actual ordinary bank equation; never interpreted as an Original role.
    pub(crate) ordinary_addressable: Option<std::rc::Rc<OrdinaryAddressableProgram>>,
    pub(crate) ordinary_validations: (usize, resident_recipe::CertifiedSpanStorage),
    /// Component maxima across genuine non-indexed local row alternatives.
    /// With an indexed program this contains only the actual empty branch.
    pub(crate) ordinary_local: Option<super::ordinary::OrdinaryParallelControls>,
    /// Per-domain peak of complete actual local alternatives. Indexed native
    /// discovery and source-copy populations must be present before qualification.
    pub(crate) local_scratch: Option<eredu_nn::workspace::WorkspaceAllocationPopulation>,
    pub(crate) mechanism: ResidentExecutionMechanisms,
    observation: Option<ExpertLocalObservationSource>,
    pub(crate) capture_publications: usize,
    observation_descriptor: Option<eredu_nn::workspace::WorkspaceExpertObservationSource>,
    source: OriginalParallelSource,
}
impl ExpertLocalQuote {
    pub(crate) fn prepare(
        source: &OriginalParallelSource,
        operation: WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
        addressable: Option<&AddressableSources>,
    ) -> Result<Self, Error> {
        let observation = match operation.kind {
            WorkspaceOperationKindView::ExpertRegion(region) => region
                .observation()
                .map(|value| {
                    ExpertLocalObservationSource::from_descriptor(value)
                        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)
                })
                .transpose()?,
            _ => None,
        };
        Self::prepare_with_observation(source, operation, mechanism, observation, addressable)
    }
    pub(crate) fn prepare_with_observation(
        source: &OriginalParallelSource,
        operation: WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
        observation: Option<ExpertLocalObservationSource>,
        addressable: Option<&AddressableSources>,
    ) -> Result<Self, Error> {
        Self::prepare_sources(source, operation, mechanism, observation, addressable, None)
    }
    pub(crate) fn prepare_ordinary(
        source: &OriginalParallelSource,
        operation: WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
        addressable: Option<&OrdinaryAddressableSources>,
    ) -> Result<Self, Error> {
        let observation = match operation.kind {
            WorkspaceOperationKindView::ExpertRegion(region) => region
                .observation()
                .map(|value| {
                    ExpertLocalObservationSource::from_descriptor(value)
                        .ok_or(WorkspaceMetadataError::Unqualified)
                })
                .transpose()?,
            _ => None,
        };
        Self::prepare_sources(source, operation, mechanism, observation, None, addressable)
    }
    fn prepare_sources(
        source: &OriginalParallelSource,
        operation: WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
        observation: Option<ExpertLocalObservationSource>,
        addressable: Option<&AddressableSources>,
        ordinary: Option<&OrdinaryAddressableSources>,
    ) -> Result<Self, Error> {
        let context =
            WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())?;
        context.charge_metadata(size_of::<(
            Self,
            WorkspaceContext,
            Result<Self, Error>,
            WorkspaceExpertRegionView<'_>,
            BoundaryStageCapacity,
            [Option<SpeculativeNumericalRecipe>; 2],
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "expert child differs from its retained region source"
            ))
        };
        let WorkspaceOperationKindView::ExpertRegion(declaration) = operation.kind else {
            return Err(invalid());
        };
        let observation_descriptor = declaration.observation();
        let view = declaration.as_view();
        view.validate()?;
        if operation.inputs.len() < 4
            || operation.outputs.len()
                != 1 + usize::from(view.kernel.separate_bias(view.tensor_partitions))
        {
            return Err(invalid());
        }
        let actual = source
            .communication_source()
            .map_err(|cause| source.neural_error(cause))?;
        let selected = actual
            .source()
            .manifest()
            .select_group_operation(
                view.group,
                eredu_runtime::CommunicationOperation::VariableAllToAll,
            )
            .map_err(|cause| context.metadata_source(cause))?;
        let descriptor = selected.descriptor();
        if descriptor.members().len() != view.peers || descriptor.local_index() != Some(view.rank) {
            return Err(invalid());
        }
        let first = operation.inputs.get(0).ok_or_else(invalid)?;
        if first.shape().last().copied() != Some(view.kernel.dimensions().0)
            || first.representation().is_none()
        {
            return Err(invalid());
        }
        let mut inputs = context.metadata_vec(operation.inputs.len())?;
        for value in operation.inputs.iter() {
            inputs.push(
                context
                    .layout(value.shape(), value.dtype())?
                    .with_representation(value.representation()),
            );
        }
        let declaration = view.retain(&context)?;
        let maximum_rows = view.maximum_received_rows().ok_or_else(invalid)?;
        let mut outputs = context.metadata_vec(operation.outputs.len())?;
        let (empty, mut local_scratch) = local_trace_with_population(
            &declaration,
            &inputs,
            0,
            mechanism,
            source.funding(),
            observation,
            true,
            |values| {
                if maximum_rows == 0 {
                    for value in values {
                        outputs.push(value.layout().clone());
                    }
                }
                Ok(())
            },
        )?;
        if addressable.is_some() && ordinary.is_some()
            || ordinary.is_some() && mechanism.allocation().original_storage
            || addressable.is_some() && !mechanism.allocation().original_storage
        {
            return Err(invalid());
        }
        let ordinary_addressable = match (view.addressable, ordinary) {
            (Some(_), Some(source)) => Some(source.local_program(operation)?),
            _ => None,
        };
        let addressable = match view.addressable {
            Some(_) if ordinary_addressable.is_none() => {
                Some(addressable.ok_or_else(invalid)?.local_quote(operation)?)
            }
            _ => None,
        };
        let mut selected = None;
        let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
        let mut capacity = boundary::capacity(empty, runtime, &context)?;
        let mut maximum_births = empty.storage.maximum_births();
        let mut controls = empty.controls;
        let mut kernels = empty.kernels;
        let candidates = super::super::grouped::expert_row_candidates(view.kernel, maximum_rows);
        context.charge_metadata(
            std::mem::size_of_val(&candidates)
                .checked_add(size_of::<Option<usize>>())
                .ok_or_else(invalid)?,
        )?;
        let mut capture_publications = 0;
        let validation_population = |recipe: SpeculativeNumericalRecipe| {
            let roots = recipe.completion.validation_roots;
            (
                roots,
                if roots == 0 {
                    resident_recipe::CertifiedSpanStorage::default()
                } else {
                    recipe.storage
                },
            )
        };
        let mut ordinary_validations = validation_population(empty);
        let mut ordinary_local = super::ordinary::OrdinaryParallelControls::numerical(empty);
        if let Some(local) = &ordinary_addressable {
            let storage = local.storage().ok_or_else(|| {
                context.metadata_error(format_args!(
                    "{}",
                    local
                        .missing_source()
                        .unwrap_or("ordinary local indexed payload is incomplete")
                ))
            })?;
            // The qualified ordinary indexed producer is CPU-only. Its
            // numerical/discovery storage shares the actual CPU allocator;
            // residency transfers retain separate domain requirements.
            if !matches!(mechanism, ResidentExecutionMechanisms::Cpu { .. }) {
                return Err(invalid());
            }
            let indexed = super::scratch::native_cpu(
                &context,
                mechanism,
                usize::try_from(storage.mutable_bytes()).map_err(|_| invalid())?,
                storage.maximum_births(),
            )?;
            local_scratch = local_scratch
                .as_ref()
                .map(|empty| context.peak_scratch_populations(&[empty, &indexed]))
                .transpose()?;
            let (roots, producers) = local.validation_population().ok_or_else(invalid)?;
            ordinary_validations.0 = ordinary_validations.0.max(roots);
            ordinary_validations.1 = ordinary_validations.1.union(producers);
            capacity.backing = capacity
                .backing
                .max(usize::try_from(storage.mutable_bytes()).map_err(|_| invalid())?);
            maximum_births = maximum_births.max(storage.maximum_births());
            capture_publications =
                capture_publications.max(local.capture_publications().ok_or_else(invalid)?);
            outputs.extend(local.local_outputs().ok_or_else(invalid)?.iter().cloned());
        } else if let Some(local) = &addressable {
            let recipe = local.local_numerical().ok_or_else(invalid)?;
            let native = local.native_capacity();
            // Every Original child producer uses the retained prepared
            // allocator, including its separately scoped source copies.
            let indexed = super::scratch::native_cpu(
                &context,
                mechanism,
                native.backing,
                recipe.storage.maximum_births(),
            )?;
            local_scratch = local_scratch
                .as_ref()
                .map(|empty| context.peak_scratch_populations(&[empty, &indexed]))
                .transpose()?;

            capacity.graph = capacity.graph.max(native.graph);
            capacity.records = capacity.records.max(native.records);
            capacity.backing = capacity.backing.max(native.backing);
            maximum_births = maximum_births.max(recipe.storage.maximum_births());
            controls = controls.max(recipe.controls);
            kernels = kernels.max(recipe.kernels);
            capture_publications = capture_publications.max(local.publications());
            outputs.extend(local.outputs.iter().cloned());
            selected = Some(recipe);
        } else {
            for rows in candidates {
                if let Some(observation) = observation {
                    capture_publications = capture_publications.max(observation.publications(
                        declaration.kernel(),
                        rows,
                        mechanism,
                    )?);
                }
                let (recipe, scratch) = local_trace_with_population(
                    &declaration,
                    &inputs,
                    rows,
                    mechanism,
                    source.funding(),
                    observation,
                    true,
                    |values| {
                        if rows == maximum_rows {
                            outputs.clear();
                            for value in values {
                                outputs.push(value.layout().clone());
                            }
                        }
                        Ok(())
                    },
                )?;
                local_scratch = match (local_scratch.as_ref(), scratch.as_ref()) {
                    (Some(prior), Some(actual)) => {
                        Some(context.peak_scratch_populations(&[prior, actual])?)
                    }
                    _ => None,
                };
                if rows == maximum_rows {
                    selected = Some(recipe);
                }
                ordinary_local = ordinary_local
                    .zip(super::ordinary::OrdinaryParallelControls::numerical(recipe))
                    .map(|(prior, actual)| prior.union(actual));
                let (roots, producers) = validation_population(recipe);
                ordinary_validations.0 = ordinary_validations.0.max(roots);
                ordinary_validations.1 = ordinary_validations.1.union(producers);
                let candidate = boundary::capacity(recipe, runtime, &context)?;
                capacity.graph = capacity.graph.max(candidate.graph);
                capacity.records = capacity.records.max(candidate.records);
                capacity.backing = capacity.backing.max(candidate.backing);
                maximum_births = maximum_births.max(recipe.storage.maximum_births());
                controls = controls.max(recipe.controls);
                kernels = kernels.max(recipe.kernels);
            }
        }
        let mut value = Self {
            declaration,
            inputs,
            outputs,
            movement: None,
            transport: None,
            aggregate: None,
            counts: None,
            provider: None,
            maximum_rows,
            capacity,
            maximum_births,
            controls,
            kernels,
            empty,
            selected,
            addressable,
            ordinary_addressable,
            ordinary_validations,
            ordinary_local,
            local_scratch,
            mechanism,
            observation,
            capture_publications,
            observation_descriptor,
            source: source.clone(),
        };
        value.movement = Some(ExpertMovementQuote::prepare(&value)?);
        value.transport = Some(ExpertTransportQuote::prepare(&value)?);
        value.counts = Some(ExpertCountQuote::prepare(&value)?);
        value.provider = Some(ExpertProviderQuote::prepare(&value)?);
        value.aggregate = Some(ExpertRegionAggregate::prepare(&value)?);
        Ok(value)
    }
    pub(crate) fn parent_capture_population(
        &self,
    ) -> crate::backend::array_copy::CaptureNativePopulation {
        self.observation
            .map(|source| source.parent_population(self.capture_publications))
            .unwrap_or_default()
    }
    pub(crate) fn matches_operation(&self, region: &WorkspaceExpertRegion) -> bool {
        self.matches(region.as_view()) && self.observation_descriptor == region.observation()
    }
    pub(crate) fn matches_source(&self, source: &OriginalParallelSource) -> bool {
        self.source.same_source(source)
    }
    pub(crate) fn source(&self) -> &OriginalParallelSource {
        &self.source
    }
    pub(crate) fn matches(&self, view: WorkspaceExpertRegionView<'_>) -> bool {
        self.declaration.as_view() == view
    }
    /// Actual completed rows use the same ordinary recorder and must fit every
    /// retained source ceiling before any native child constructor is entered.
    pub(crate) fn actual(
        &self,
        rows: usize,
        native: &[safemlx::Array],
    ) -> Result<SpeculativeNumericalRecipe, Error> {
        let context = WorkspaceContext::new_with_metadata_funding(
            self.mechanism,
            self.source.funding().clone(),
        )?;
        context.charge_metadata(size_of::<(
            usize,
            SpeculativeNumericalRecipe,
            BoundaryStageCapacity,
            Result<SpeculativeNumericalRecipe, Error>,
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "completed expert rows exceed their retained child source"
            ))
        };
        if ((self.addressable.is_some() || self.ordinary_addressable.is_some()) && rows != 0)
            || rows > self.maximum_rows
            || native.len().checked_add(1) != Some(self.inputs.len())
        {
            return Err(invalid());
        }
        let mut projection = ExistingArrayProjection::with_source_count(&context, native.len())
            .map_err(|cause| context.metadata_source(cause))?;
        let mut actual = context.metadata_vec(self.inputs.len())?;
        for (index, array) in native.iter().enumerate() {
            if index == 1 {
                actual.push(context.layout(
                    &[i32::try_from(rows).map_err(|_| invalid())?, 1],
                    WorkspaceDtype::Int32,
                )?);
            }
            let selected = &self.inputs[if index == 0 { 0 } else { index + 1 }];
            if (index >= 3 && array.shape() != selected.shape())
                || crate::backend::nn::workspace::byte_view::Dtype::from_layout(selected.as_view())
                    .is_none_or(|dtype| dtype.native() != array.dtype())
            {
                return Err(invalid());
            }
            let value = projection.project(array)?;
            actual.push(value.layout().clone());
        }
        if !projection.is_complete() {
            return Err(invalid());
        }
        let (recipe, _) = local_trace_with_population(
            &self.declaration,
            &actual,
            rows,
            self.mechanism,
            self.source.funding(),
            self.observation,
            false,
            |_| Ok(()),
        )?;
        let capacity = boundary::capacity(
            recipe,
            self.source
                .agreement_inputs()
                .ok_or_else(invalid)?
                .runtime(),
            &context,
        )?;
        if capacity.graph > self.capacity.graph
            || capacity.records > self.capacity.records
            || capacity.backing > self.capacity.backing
            || recipe.storage.maximum_births() > self.maximum_births
            || recipe.controls > self.controls
            || recipe.kernels > self.kernels
        {
            return Err(invalid());
        }
        Ok(recipe)
    }
}
fn local_trace(
    declaration: &WorkspaceExpertRegion,
    source: &[WorkspaceLayout],
    rows: usize,
    mechanism: ResidentExecutionMechanisms,
    funding: &HostMetadataFunding,
) -> Result<SpeculativeNumericalRecipe, Error> {
    local_trace_with_population(
        declaration,
        source,
        rows,
        mechanism,
        funding,
        None,
        false,
        |_| Ok(()),
    )
    .map(|(recipe, _)| recipe)
}
#[cfg(test)]
fn local_trace_with_outputs<F>(
    declaration: &WorkspaceExpertRegion,
    source: &[WorkspaceLayout],
    rows: usize,
    mechanism: ResidentExecutionMechanisms,
    funding: &HostMetadataFunding,
    observation: Option<ExpertLocalObservationSource>,
    outputs: F,
) -> Result<SpeculativeNumericalRecipe, Error>
where
    F: FnOnce(&[WorkspaceTensor]) -> Result<(), Error>,
{
    local_trace_with_population(
        declaration,
        source,
        rows,
        mechanism,
        funding,
        observation,
        false,
        outputs,
    )
    .map(|(recipe, _)| recipe)
}
fn local_trace_with_population<F>(
    declaration: &WorkspaceExpertRegion,
    source: &[WorkspaceLayout],
    rows: usize,
    mechanism: ResidentExecutionMechanisms,
    funding: &HostMetadataFunding,
    observation: Option<ExpertLocalObservationSource>,
    retain_scratch: bool,
    outputs: F,
) -> Result<
    (
        SpeculativeNumericalRecipe,
        Option<eredu_nn::workspace::WorkspaceAllocationPopulation>,
    ),
    Error,
>
where
    F: FnOnce(&[WorkspaceTensor]) -> Result<(), Error>,
{
    let observe_outputs = outputs;
    let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())?;
    context.charge_metadata(size_of::<(
        WorkspaceContext,
        WorkspaceTraceReport,
        Vec<WorkspaceTensor>,
        Vec<&WorkspaceTensor>,
        GroupSelection<WorkspaceTensor>,
        Result<
            (
                SpeculativeNumericalRecipe,
                Option<eredu_nn::workspace::WorkspaceAllocationPopulation>,
            ),
            Error,
        >,
        Option<eredu_nn::workspace::WorkspaceAllocationPopulation>,
        bool,
    )>())?;
    let invalid = || {
        context.metadata_error(format_args!(
            "expert local equation differs from its retained constructor"
        ))
    };
    let rows = i32::try_from(rows).map_err(|_| invalid())?;
    let first = source.first().ok_or_else(invalid)?;
    let input = WorkspaceTensor::existing(
        context
            .layout(
                &[rows, declaration.as_view().kernel.dimensions().0],
                first.dtype(),
            )?
            .with_representation(first.representation()),
        &context,
    )?;
    let mut observed = observation
        .map(|source| observation::SourceObserver::new(source, &input, &context))
        .transpose()?;
    if rows == 0 {
        context.begin_span();
        let mut output = input.zeros_like(&context)?;
        if matches!(
            declaration.as_view().kernel,
            eredu_nn::workspace::WorkspaceExpertKernel::Linear(_)
        ) {
            output = output.reshape(&[0, declaration.as_view().kernel.dimensions().1], &context)?;
        }
        let bias = declaration
            .as_view()
            .kernel
            .separate_bias(declaration.as_view().tensor_partitions);
        let mut outputs = context.metadata_vec(1 + usize::from(bias))?;
        if bias {
            outputs.push(output.clone());
        }
        outputs.push(output);
        observe_outputs(&outputs)?;
        let scratch = if retain_scratch {
            context.new_allocation_scratch()?
        } else {
            None
        };
        let recipe = observation::finish(&context, &outputs, observed, mechanism)?;
        return Ok((recipe, scratch));
    }
    let mut parameters = context.metadata_vec(source.len().checked_sub(4).ok_or_else(invalid)?)?;
    for value in &source[4..] {
        parameters.push(WorkspaceTensor::existing(
            context
                .layout(value.shape(), value.dtype())?
                .with_representation(value.representation()),
            &context,
        )?);
    }
    let mut borrowed = context.metadata_vec(parameters.len())?;
    borrowed.extend(parameters.iter());
    let ids =
        WorkspaceTensor::existing(context.layout(&[rows, 1], WorkspaceDtype::Int32)?, &context)?;
    let scalar = |index: usize| -> Result<WorkspaceTensor, Error> {
        let layout = source.get(index).ok_or_else(invalid)?;
        WorkspaceTensor::existing(
            context
                .layout(&[rows, 1], layout.dtype())?
                .with_representation(layout.representation()),
            &context,
        )
    };
    let routes = GroupSelection::new(ids, scalar(2)?, scalar(3)?);
    context.begin_span();
    if let Some(observer) = &mut observed {
        observer.bind_groups(&routes)?;
    }
    let output = declaration.kernel().trace_with_parameters(
        &borrowed,
        &input,
        &routes,
        declaration.as_view().tensor_partitions,
        &context,
        observed
            .as_mut()
            .map(|observer| observer as &mut dyn eredu_nn::GroupedUnitObserver<WorkspaceTensor>),
    )?;
    let (partial, bias) = output.into_parts();
    let mut outputs = context.metadata_vec(1 + usize::from(bias.is_some()))?;
    outputs.push(partial);
    outputs.extend(bias);
    observe_outputs(&outputs)?;
    let scratch = if retain_scratch {
        context.new_allocation_scratch()?
    } else {
        None
    };
    let recipe = observation::finish(&context, &outputs, observed, mechanism)?;
    Ok((recipe, scratch))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[path = "expert_region/census_tests.rs"]
mod census_tests;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
