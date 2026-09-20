//! One retained numerical/constructor/read source for an actual bank region.
use super::parameters::ParameterRows;
use super::*;
use crate::backend::runtime::residency::parameter_bank::{
    IndexedBankSource, IndexedChunkLayout, IndexedResidencyPlan,
};
use eredu_runtime::expert::AddressableChunkCensus;
use eredu_runtime::working_memory::{HostSourceConstructionFacts, WorkingMemoryPool};

#[derive(Debug)]
pub(crate) struct RetainedAddressableDeclaration {
    value: std::rc::Rc<WorkspaceAddressableRegion>,
    rows: usize,
}
impl RetainedAddressableDeclaration {
    pub(crate) fn as_view(&self) -> WorkspaceAddressableRegionView<'_> {
        let mut view = self.value.as_view();
        view.chunks.rows = self.rows;
        view
    }
    fn observation(&self) -> Option<WorkspaceAddressableObservationSource> { self.value.observation() }
}
pub(crate) struct AddressableQuote {
    pub(crate) declaration: RetainedAddressableDeclaration,
    pub(crate) inputs: Vec<WorkspaceLayout>,
    pub(crate) outputs: Vec<WorkspaceLayout>,
    /// Completed equation before separately scoped source-copy producers.
    /// Local row alternatives compose this population, then join the actual
    /// admitted maximum residency source once.
    pub(super) equation: SpeculativeNumericalRecipe,
    pub(crate) numerical: SpeculativeNumericalRecipe,
    pub(crate) capacity: BoundaryStageCapacity,
    pub(crate) host_bytes: u64,
    pub(crate) capture_publications:usize,
    pub(crate) constructor_facts: HostSourceConstructionFacts,
    pub(crate) read_facts: Option<HostSourceConstructionFacts>,
    pub(crate) residency: IndexedResidencyPlan,
    parameters: std::rc::Rc<ParameterRows>,
    mechanism: ResidentExecutionMechanisms,
}
impl std::fmt::Debug for AddressableQuote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddressableQuote")
            .field("declaration", &self.declaration)
            .field("numerical", &self.numerical)
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}
impl AddressableQuote {
    pub(crate) fn prepare(
        bank: &IndexedBankSource,
        operation: WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
        runtime: &safemlx::PreparedInputRuntime,
        pool: Option<&WorkingMemoryPool>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "addressable source, native inputs or completed outputs differ"
            ))
        };
        context.charge_metadata(size_of::<(
            Self,
            WorkspaceContext,
            Result<Self, Error>,
            AddressableNumericalPopulation,
            SpeculativeNumericalRecipe,
            AddressableParentSource,
            AddressableChildSource,
            IndexedChunkLayout,
            HostSourceConstructionFacts,
            [usize; 10],
        )>())?;
        let WorkspaceOperationKindView::AddressableRegion(declaration) = operation.kind else {
            return Err(invalid());
        };
        let source = declaration.as_view();
        source.validate()?;
        let roles = 1 + usize::from(source.kernel.separate_bias(source.tensor_partitions));
        if operation.inputs.len() != 4 || operation.outputs.len() != roles {
            return Err(invalid());
        }
        let mut inputs = context.metadata_vec(4)?;
        for input in operation.inputs.iter() {
            inputs.push(
                context
                    .layout(input.shape(), input.dtype())?
                    .with_representation(input.representation()),
            );
        }
        let parameters = ParameterRows::prepare(bank, source, mechanism, &context)?;
        let first = parameters.identity.first().ok_or_else(invalid)?;
        // This is an actual manager/read constructor proof, not a tensor shape
        // substitute. The accepted role must still bind the corresponding banks.
        let residency = bank
            .inspect_original_residency(first, pool, funding)
            .map_err(|cause| context.metadata_source(cause))?;
        let constructor_facts = residency.constructor_facts().ok_or_else(invalid)?;
        let read_facts = residency.read_facts();
        if residency.requires_reads() && read_facts.is_none() {
            return Err(invalid());
        }
        let observation=declaration.observation();
        if observation.is_some_and(|v|v.unit_dtype.is_none()) {return Err(invalid());}
        let source_groups=observation.map(|_|super::observation::source_groups(source,&inputs,mechanism,funding,&context)).transpose()?;
        let mut capture_publications=0usize;
        let fields = parameters.fields;
        // Ordinary compact concatenate is same-type. Verify each actual
        // source row before using a cardinality ceiling; source identities and
        // replacement Slice recipes remain retained independently below.
        for member in 1..source.chunks.members {
            for field in 0..fields {
                if parameters.rows[member * fields + field].layout != parameters.rows[field].layout
                    || parameters.rows[member * fields + field].layout.representation() != parameters.rows[field].layout.representation()
                {
                    return Err(context.metadata_error(format_args!(
                        "addressable compact rows require their exact common physical layout"
                    )));
                }
            }
        }
        let chunks = first.plan().len();
        let mut aggregate: Option<AddressableNumericalPopulation> = None;
        let mut parent_outputs = context.metadata_vec(roles)?;
        let mut host_bytes = 0u64;
        // Only two row classes exist: the exact full chunk and final tail.
        // Each class enumerates its possible compact cardinalities, never
        // selected-ID subsets or model-family variants.
        let mut classes: Vec<(usize, usize)> = context.metadata_vec(2)?;
        for ordinal in 0..chunks {
            let rows = first.plan().range(ordinal).ok_or_else(invalid)?.len();
            if let Some((_, count)) = classes.iter_mut().find(|(n, _)| *n == rows) {
                *count += 1;
            } else {
                classes.push((rows, 1usize));
            }
        }
        for (rows, occurrences) in classes {
            let child_inputs =
                normalized_inputs(source, &inputs, rows, mechanism, funding, &context)?;
            let maximum = rows
                .checked_mul(source.chunks.routes)
                .ok_or_else(invalid)?
                .min(source.chunks.members);
            let mut branch: Option<AddressableNumericalPopulation> = None;
            let mut branch_host = 0u64;
            let mut branch_publications=0usize;
            for members in 1..=maximum {
                let mut selected =
                    context.metadata_vec(members.checked_mul(fields).ok_or_else(invalid)?)?;
                for row in &parameters.rows[..members * fields] {
                    selected.push(
                        context
                            .layout(row.layout.shape(), row.layout.dtype())?
                            .with_representation(row.layout.representation()),
                    );
                }
                let child = AddressableChildSource::prepare_with_observation(
                    source,
                    &child_inputs,
                    members,
                    &selected,
                    mechanism,
                    funding,observation,source_groups.as_ref(),
                )?;
                branch_publications=branch_publications.max(child.capture.publications);
                let recipe = SpeculativeNumericalRecipe::inspect_owned_child_with_capture(
                    &child.report,
                    roles,
                    mechanism,
                    &context,child.capture,
                )?;
                let mut population = AddressableNumericalPopulation::from_recipe(recipe, true)
                    .ok_or_else(invalid)?;
                // Replacements are sliced by their exact original row before
                // concatenate. Each field contributes at most its actual
                // replacement population and at most the selected cardinality.
                for field in 0..fields {
                    let mut slice: Option<AddressableNumericalPopulation> = None;
                    let mut count = 0usize;
                    for member in 0..source.chunks.members {
                        if let Some(recipe) = parameters.rows[member * fields + field].slice {
                            count = count.checked_add(1).ok_or_else(invalid)?;
                            let candidate =
                                AddressableNumericalPopulation::from_recipe(recipe, false)
                                    .ok_or_else(invalid)?;
                            slice = Some(match slice {
                                None => candidate,
                                Some(prior) => prior.union(candidate).ok_or_else(invalid)?,
                            });
                        }
                    }
                    if let Some(slice) = slice {
                        for _ in 0..members.min(count) {
                            population = population.append(slice).ok_or_else(invalid)?;
                        }
                    }
                }
                branch = Some(match branch {
                    None => population,
                    Some(prior) => prior.union(population).ok_or_else(invalid)?,
                });
                branch_host =
                    branch_host.max(child.report.host_workspace_bytes.ok_or_else(invalid)?);
                if rows == source.chunks.rows.min(source.chunks.chunk_rows) {
                    if parent_outputs.is_empty() {
                        for layout in &child.output_layouts {
                            parent_outputs.push(
                                context
                                    .layout(layout.shape(), layout.dtype())?
                                    .with_representation(layout.representation()),
                            );
                        }
                    } else if parent_outputs.len()!=child.output_layouts.len() || parent_outputs.iter().zip(&child.output_layouts).any(|(a,b)|a!=b||a.representation()!=b.representation()) {
                        return Err(invalid());
                    }
                }
            }
            capture_publications=capture_publications.checked_add(branch_publications.checked_mul(occurrences).ok_or_else(invalid)?)
                .ok_or_else(invalid)?;
            let branch = branch.ok_or_else(invalid)?;
            for _ in 0..occurrences {
                aggregate = Some(match aggregate {
                    None => branch,
                    Some(prior) => prior.append(branch).ok_or_else(invalid)?,
                });
            }
            host_bytes = host_bytes
                .checked_add(
                    branch_host
                        .checked_mul(occurrences as u64)
                        .ok_or_else(invalid)?,
                )
                .ok_or_else(invalid)?;
        }
        let parent =
            AddressableParentSource::prepare(source, &inputs, &parent_outputs, mechanism, funding)?;
        if parent.child_invocations != chunks
            || parent.retained_child_outputs != chunks.checked_mul(roles).ok_or_else(invalid)?
        {
            return Err(invalid());
        }
        let parent_recipe = SpeculativeNumericalRecipe::inspect_owned_child(
            &parent.report,
            roles,
            mechanism,
            &context,
        )?;
        let parent_population = AddressableNumericalPopulation::from_recipe(parent_recipe, false)
            .ok_or_else(invalid)?;
        let population = parent_population
            .append(aggregate.ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        let mut id_completions = 0usize;
        for ordinal in 0..chunks {
            let census = AddressableChunkCensus::new(
                first.bank(),
                first.unit(),
                first.plan(),
                ordinal,
                first.access(),
            )
            .ok_or_else(invalid)?;
            let indexed = IndexedChunkLayout::inspect(runtime, census).ok_or_else(invalid)?;
            host_bytes = host_bytes
                .checked_add(indexed.host_bytes() as u64)
                .ok_or_else(invalid)?;
            id_completions = id_completions
                .checked_add(indexed.parent_completions())
                .ok_or_else(invalid)?;
        }
        host_bytes = host_bytes
            .checked_add(parent.report.host_workspace_bytes.ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        let equation = population.finish(roles, id_completions, &context)?;
        let numerical = equation.with_indexed_source(&residency, &context)?;
        let backing = safemlx::OriginalBufferBudget::metal_population_layout(
            runtime,
            usize::try_from(numerical.storage.mutable_bytes()).map_err(|_| invalid())?,
            numerical.storage.maximum_births(),
        )
        .map_err(|cause| context.metadata_source(cause))?;
        let capacity = BoundaryStageCapacity {
            graph: numerical.graph_capacity,
            records: numerical.record_capacity,
            backing: backing.capacity(),
        };
        for (actual, expected) in parent.output_layouts.iter().zip(operation.outputs.iter()) {
            if actual.shape() != expected.shape() || actual.dtype() != expected.dtype() {
                return Err(invalid());
            }
        }
        let mut value=Self {
            declaration: {
                context.charge_metadata(super::sources::shared_bytes::<WorkspaceAddressableRegion>().ok_or_else(invalid)?)?;
                RetainedAddressableDeclaration { value: std::rc::Rc::new(declaration.retain(&context)?), rows: source.chunks.rows }
            },
            inputs,
            outputs: parent.output_layouts,
            equation,
            numerical,
            capacity,
            host_bytes,capture_publications,
            constructor_facts,
            read_facts,
            residency,
            parameters: { context.charge_metadata(super::sources::shared_bytes::<ParameterRows>().ok_or_else(invalid)?)?; std::rc::Rc::new(parameters) },
            mechanism,
        };
        let native=crate::backend::submission_recovery::addressable::control_bytes(&value)
            .map_err(|cause|context.metadata_source(cause))?;
        let callback=crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement::invocation_control_bytes()
            .ok_or_else(invalid)?.checked_add(source.callback_control_bytes).ok_or_else(invalid)?;
        value.host_bytes=value.host_bytes.checked_add(u64::try_from(native).map_err(|_|invalid())?)
            .and_then(|n|n.checked_add(u64::try_from(callback).ok()?)).ok_or_else(invalid)?;
        Ok(value)
    }
    fn specialization_frames() -> Option<usize> {
        let frames = [size_of::<(Self, Result<Self, Error>, &Self, usize, &super::sources::LocalEnvelope)>(),
            size_of::<WorkspaceAddressableRegionView<'_>>(), size_of::<RetainedAddressableDeclaration>(),
            size_of::<[i32; 2]>(), size_of::<Vec<WorkspaceLayout>>() * 2,
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, WorkspaceLayout>>>(),
            size_of::<std::slice::Iter<'_, WorkspaceLayout>>(), size_of::<HostMetadataFunding>(),
            size_of::<HostSourceConstructionFacts>(), size_of::<Option<HostSourceConstructionFacts>>(),
            size_of::<(&AddressableQuoteRef, usize, &HostMetadataFunding, AddressableQuoteRef, Result<AddressableQuoteRef, Error>)>()];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn specialization_control_bytes(&self) -> Option<usize> {
        let roles = self.outputs.len();
        let frames = [Self::specialization_frames()?,
            WorkspaceContext::construction_bytes::<ResidentExecutionMechanisms>()?,
            WorkspaceLayout::construction_bytes(2)?.checked_mul(4usize.checked_add(roles)?)?,
            super::sources::shared_bytes::<Self>()?,
            self.residency.for_rows_control_bytes()?,
            WorkspaceContext::metadata_vec_bytes::<WorkspaceLayout>(4)?,
            WorkspaceContext::metadata_vec_bytes::<WorkspaceLayout>(roles)?];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Selects only row geometry from a previously qualified local envelope.
    /// Physical parameter rows remain shared; no numerical constructor or
    /// admission policy is replayed at execution time.
    pub(super) fn for_rows(&self, rows: usize, envelope: &super::sources::LocalEnvelope, funding: &HostMetadataFunding) -> Result<Self, Error> {
        let context = WorkspaceContext::new_with_metadata_funding(self.mechanism, funding.clone())?;
        let invalid = || context.metadata_error(format_args!("local indexed rows exceed their retained source"));
        context.charge_metadata(Self::specialization_frames().ok_or_else(invalid)?)?;
        let mut source = self.declaration.as_view();
        if rows == 0 || rows > source.chunks.rows { return Err(invalid()); }
        source.chunks.rows = rows;
        let declaration = RetainedAddressableDeclaration { value: self.declaration.value.clone(), rows };
        let mut inputs = context.metadata_vec(self.inputs.len())?;
        let mut outputs = context.metadata_vec(self.outputs.len())?;
        for (index, layout) in self.inputs.iter().enumerate() {
            inputs.push(context.layout(&[i32::try_from(rows).map_err(|_| invalid())?,
                if index == 0 { source.kernel.dimensions().0 } else { 1 }], layout.dtype())?
                .with_representation(layout.representation()));
        }
        for layout in &self.outputs {
            outputs.push(context.layout(&[i32::try_from(rows).map_err(|_| invalid())?, source.kernel.dimensions().1], layout.dtype())?
                .with_representation(layout.representation()));
        }
        let residency = self.residency.for_rows(rows, funding).map_err(|cause| context.metadata_source(cause))?;
        let constructor_facts = residency.constructor_facts().ok_or_else(invalid)?;
        let read_facts = residency.read_facts();
        Ok(Self { declaration, inputs, outputs, equation: envelope.equation,
            numerical: envelope.numerical, capacity: envelope.capacity,
            host_bytes: envelope.host_bytes, capture_publications: envelope.capture_publications,
            constructor_facts, read_facts, residency, parameters: self.parameters.clone(), mechanism: self.mechanism })
    }
    pub(crate) fn matches(&self, operation: WorkspaceOperationView<'_>) -> bool {
        matches!(operation.kind,WorkspaceOperationKindView::AddressableRegion(value) if value.as_view()==self.declaration.as_view() && value.observation()==self.declaration.observation())
            && operation.inputs.len() == self.inputs.len()
            && operation.outputs.len() == self.outputs.len()
            && operation.inputs.iter().zip(&self.inputs).all(|(a, b)| {
                a.shape() == b.shape()
                    && a.dtype() == b.dtype()
                    && a.representation() == b.representation()
            })
            && operation
                .outputs
                .iter()
                .zip(&self.outputs)
                .all(|(a, b)| a.shape() == b.shape() && a.dtype() == b.dtype())
    }
    pub(crate) fn allocation(&self) -> NativeAllocationFacts {
        self.mechanism.allocation()
    }
    pub(crate) fn identity(
        &self,
    ) -> &crate::backend::runtime::residency::parameter_bank::IndexedBindingIdentity {
        &self.parameters.identity
    }
}
pub(super) fn normalized_inputs(
    source: WorkspaceAddressableRegionView<'_>,
    inputs: &[WorkspaceLayout],
    rows: usize,
    mechanism: ResidentExecutionMechanisms,
    funding: &HostMetadataFunding,
    owner: &WorkspaceContext,
) -> Result<Vec<WorkspaceLayout>, Error> {
    let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())?;
    let invalid = || owner.metadata_error(format_args!("addressable normalization source differs"));
    let width = source.kernel.dimensions().0;
    let mut output = owner.metadata_vec(4)?;
    for (index, layout) in inputs.iter().enumerate() {
        if index == 1 {
            // Existing remap worker copies its exact private I32 destination.
            output.push(owner.layout(
                &[
                    i32::try_from(rows).map_err(|_| invalid())?,
                    i32::try_from(source.chunks.routes).map_err(|_| invalid())?,
                ],
                WorkspaceDtype::Int32,
            )?);
            continue;
        }
        let value = WorkspaceTensor::existing(
            context
                .layout(layout.shape(), layout.dtype())?
                .with_representation(layout.representation()),
            &context,
        )?;
        let value = value
            .reshape(
                &[
                    i32::try_from(source.chunks.rows).map_err(|_| invalid())?,
                    if index == 0 {
                        width
                    } else {
                        i32::try_from(source.chunks.routes).map_err(|_| invalid())?
                    },
                ],
                &context,
            )?
            .narrow_axis(0, 0, i32::try_from(rows).map_err(|_| invalid())?, &context)?;
        output.push(
            owner
                .layout(value.shape(), value.layout().dtype())?
                .with_representation(value.layout().representation()),
        );
    }
    Ok(output)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
