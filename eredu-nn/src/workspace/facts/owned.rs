//! Finite owned emission from the same selected, borrowed mechanism facts.
//!
//! The context supplies its already admitted metadata remainder. This module
//! prices only its actual buffers and controls, never tensor backing or an
//! execution grant. Unknown facts stay unknown, with no ordinary fallback.

use super::*;
use crate::workspace::{
    Error, HostMetadataFunding, HostMetadataFundingError, WorkspaceEffectError, WorkspaceHostBound,
    WorkspaceMechanisms, metadata_funding, validate_workspace_host_assumptions,
    validate_workspace_output_storage, validate_workspace_tensor_declaration,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    marker::PhantomData,
    mem::{size_of, size_of_val},
    string::FromUtf8Error,
};

#[derive(Debug, thiserror::Error)]
pub(in crate::workspace) enum FactPreparationError<E> {
    #[error("selected finite workspace facts failed")]
    Mechanism(#[source] E),
    #[error("finite workspace fact layout overflow")]
    Overflow,
    #[error("finite workspace facts changed between count and emission")]
    Changed,
    #[error(transparent)]
    Reserve(#[from] TryReserveError),
    #[error(transparent)]
    Effect(#[from] WorkspaceEffectError),
    #[error(transparent)]
    Utf8(#[from] FromUtf8Error),
}

/// Capacity/overflow can return before even the retained-error constructor is
/// admitted. The enclosing context must use its own prepared rejection channel
/// for those fixed cases. Retained causes already consumed their exact control
/// amount and need no formatting, cloning or second error allocation.
#[derive(Debug)]
pub(in crate::workspace) enum FactEmissionFailure {
    Capacity { required: usize, available: usize },
    Overflow,
    Retained(Error),
    Funding(HostMetadataFundingError),
}

#[derive(Debug)]
pub(in crate::workspace) struct OwnedOperationFacts {
    pub(in crate::workspace) outputs: Vec<WorkspaceOutputEffect>,
    pub(in crate::workspace) aliases: Vec<usize>,
    pub(in crate::workspace) scratch_bytes: u64,
    pub(in crate::workspace) output_sources:
        Vec<Option<super::super::WorkspaceAllocationPopulation>>,
    pub(in crate::workspace) scratch_source: Option<super::super::WorkspaceAllocationPopulation>,
    pub(in crate::workspace) scratch_allocations:
        Option<Vec<super::super::WorkspaceScratchAllocation>>,
    pub(in crate::workspace) assumptions: String,
}

impl OwnedOperationFacts {
    pub(in crate::workspace) fn output(
        &self,
        index: usize,
    ) -> Option<WorkspaceOutputStorageView<'_>> {
        self.outputs.get(index)?.as_view(&self.aliases)
    }
}

#[derive(Debug)]
pub(in crate::workspace) struct EmittedWorkspaceFacts {
    pub(in crate::workspace) tensor: Option<OwnedOperationFacts>,
    pub(in crate::workspace) host: Option<WorkspaceHostBound>,
    /// Exact cumulative debit, including buffers which retire before return.
    pub(in crate::workspace) charged_bytes: usize,
}

/// A pure per-invocation layout. The facts remain bound to the actual provider
/// and operation at emission by its repeated complete validation and exact
/// equality with both counted fact headers. No source witness is manufactured.
#[derive(Debug)]
pub(in crate::workspace) struct WorkspaceFactPreparation<E> {
    tensor: Option<WorkspaceOperationFacts>,
    host: Option<WorkspaceHostFacts>,
    buffer_bytes: usize,
    scratch_layout: Option<(usize, usize)>,
    output_source_count: usize,
    charged_bytes: usize,
    error: PhantomData<fn() -> E>,
}

impl<E: std::error::Error + Send + Sync + 'static> WorkspaceFactPreparation<E> {
    /// Does not allocate, format a cause, copy a shape or emit any destination.
    pub(in crate::workspace) fn inspect<M: WorkspaceFactMechanisms<Error = E> + ?Sized>(
        operation: WorkspaceOperationView<'_>,
        mechanism: &M,
    ) -> Result<Self, FactPreparationError<E>> {
        let tensor = mechanism
            .operation_facts(operation)
            .map_err(FactPreparationError::Mechanism)?;
        if let Some(tensor) = tensor {
            if tensor.layout.outputs != operation.outputs.len()
                || tensor.layout.assumption_bytes == 0
            {
                return Err(WorkspaceEffectError::Incomplete.into());
            }
        }
        let host = mechanism
            .host_facts(operation)
            .map_err(FactPreparationError::Mechanism)?;
        let mut buffer_bytes = 0usize;
        if let Some(tensor) = tensor {
            for bytes in [
                array_bytes::<WorkspaceOutputEffect, E>(tensor.layout.outputs)?,
                array_bytes::<usize, E>(tensor.layout.aliases)?,
                array_bytes::<u8, E>(tensor.layout.assumption_bytes)?,
            ] {
                buffer_bytes = add(buffer_bytes, bytes)?;
            }
        }
        if let Some(host) = host {
            buffer_bytes = add(buffer_bytes, array_bytes::<u8, E>(host.assumption_bytes)?)?;
        }
        let mut output_source_count = 0usize;
        for index in 0..operation.outputs.len() {
            if let Some(source) = mechanism
                .output_allocations(operation, index)
                .map_err(FactPreparationError::Mechanism)?
            {
                if tensor.is_none() || source.domain_population().is_some() {
                    return Err(FactPreparationError::Changed);
                }
                output_source_count = add(output_source_count, 1)?;
            }
        }
        if output_source_count != 0 {
            buffer_bytes = add(
                buffer_bytes,
                array_bytes::<Option<super::super::WorkspaceAllocationPopulation>, E>(
                    operation.outputs.len(),
                )?,
            )?;
        }
        let scratch_layout = mechanism
            .scratch_allocations(operation)
            .map_err(FactPreparationError::Mechanism)?
            .map(|rows| scratch_layout::<E>(rows, tensor.map(|value| value.scratch_bytes)))
            .transpose()?;
        if let Some((_, bytes)) = scratch_layout {
            buffer_bytes = add(buffer_bytes, bytes)?;
        }
        let charged_bytes = add(
            control_bytes::<E>().ok_or(FactPreparationError::Overflow)?,
            buffer_bytes,
        )?;
        Ok(Self {
            tensor,
            host,
            buffer_bytes,
            scratch_layout,
            output_source_count,
            charged_bytes,
            error: PhantomData,
        })
    }

    pub(in crate::workspace) fn headers(
        &self,
    ) -> (Option<WorkspaceOperationFacts>, Option<WorkspaceHostFacts>) {
        (self.tensor, self.host)
    }

    pub(in crate::workspace) fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }

    // Private: the erased worker debits both controls and all buffers before
    // entry. Prefix values remain locals and retire on every failed emission.
    fn construct<M: WorkspaceFactMechanisms<Error = E> + ?Sized>(
        self,
        operation: WorkspaceOperationView<'_>,
        mechanism: &M,
    ) -> Result<EmittedWorkspaceFacts, FactPreparationError<E>> {
        let tensor = match self.tensor {
            Some(facts) => {
                let mut outputs =
                    initialized::<_, E>(facts.layout.outputs, WorkspaceOutputEffect::Allocate(0))?;
                let mut aliases = initialized::<_, E>(facts.layout.aliases, 0usize)?;
                let mut assumptions = initialized::<_, E>(facts.layout.assumption_bytes, 0u8)?;
                let emitted = mechanism
                    .write_operation_facts(
                        operation,
                        WorkspaceEffectDestination {
                            outputs: &mut outputs,
                            aliases: &mut aliases,
                            assumptions: &mut assumptions,
                        },
                    )
                    .map_err(FactPreparationError::Mechanism)?;
                if emitted != self.tensor {
                    return Err(FactPreparationError::Changed);
                }
                let assumptions = String::from_utf8(assumptions)?;
                validate_workspace_tensor_declaration(
                    outputs.len(),
                    operation.outputs.len(),
                    &assumptions,
                )?;
                for (index, output) in outputs.iter().copied().enumerate() {
                    let output = output
                        .as_view(&aliases)
                        .ok_or(FactPreparationError::Changed)?;
                    let layout = operation
                        .outputs
                        .get(index)
                        .ok_or(FactPreparationError::Changed)?;
                    validate_workspace_output_storage(
                        output,
                        layout,
                        index,
                        operation.inputs.len(),
                    )?;
                }
                let population = mechanism
                    .scratch_allocations(operation)
                    .map_err(FactPreparationError::Mechanism)?;
                let layout = population
                    .map(|rows| scratch_layout::<E>(rows, Some(facts.scratch_bytes)))
                    .transpose()?;
                if layout != self.scratch_layout {
                    return Err(FactPreparationError::Changed);
                }
                let scratch_allocations = population
                    .map(|rows| {
                        let rows = rows.allocations();
                        let mut owned = Vec::new();
                        owned.try_reserve_exact(rows.len())?;
                        owned.extend(rows.iter().cloned());
                        Ok::<_, FactPreparationError<E>>(owned)
                    })
                    .transpose()?;
                let mut output_sources = Vec::new();
                if self.output_source_count != 0 {
                    output_sources.try_reserve_exact(operation.outputs.len())?;
                }
                let mut count = 0usize;
                for index in 0..operation.outputs.len() {
                    let source = mechanism
                        .output_allocations(operation, index)
                        .map_err(FactPreparationError::Mechanism)?;
                    if let Some(source) = source {
                        let bytes = match outputs[index].as_view(&aliases) {
                            Some(WorkspaceOutputStorageView::Allocate(bytes))
                            | Some(WorkspaceOutputStorageView::AllocateOrAliasInputs {
                                bytes,
                                ..
                            }) => bytes,
                            _ => return Err(FactPreparationError::Changed),
                        };
                        if source.domain_population().is_some()
                            || source.backing_bytes() != Some(bytes)
                        {
                            return Err(FactPreparationError::Changed);
                        }
                        count = add(count, 1)?;
                    }
                    if self.output_source_count != 0 {
                        output_sources.push(source.cloned());
                    }
                }
                if count != self.output_source_count {
                    return Err(FactPreparationError::Changed);
                }
                Some(OwnedOperationFacts {
                    outputs,
                    aliases,
                    scratch_bytes: facts.scratch_bytes,
                    scratch_allocations,
                    scratch_source: population.cloned(),
                    output_sources,
                    assumptions,
                })
            }
            None => None,
        };
        let host = match self.host {
            Some(facts) => {
                let mut assumptions = initialized::<_, E>(facts.assumption_bytes, 0u8)?;
                let emitted = mechanism
                    .write_host_facts(
                        operation,
                        WorkspaceHostDestination {
                            assumptions: &mut assumptions,
                        },
                    )
                    .map_err(FactPreparationError::Mechanism)?;
                if emitted != self.host {
                    return Err(FactPreparationError::Changed);
                }
                let assumptions = String::from_utf8(assumptions)?;
                validate_workspace_host_assumptions(&assumptions)?;
                Some(WorkspaceHostBound {
                    bytes: facts.bytes,
                    assumptions,
                })
            }
            None => None,
        };
        Ok(EmittedWorkspaceFacts {
            tensor,
            host,
            charged_bytes: self.charged_bytes,
        })
    }
}

/// Implemented on M itself, so the context coerces two loans of the same Rc<M>
/// to ordinary and finite interfaces without constructing a second provider.
pub(in crate::workspace) trait FiniteWorkspaceFacts: std::fmt::Debug {
    fn emission_control_bytes(&self) -> Option<usize>;
    fn emit(
        &self,
        operation: WorkspaceOperationView<'_>,
        remaining_bytes: &mut usize,
        funding: Option<&HostMetadataFunding>,
        context: &super::super::WorkspaceContext,
    ) -> Result<EmittedWorkspaceFacts, FactEmissionFailure>;
}

impl<M> FiniteWorkspaceFacts for M
where
    M: WorkspaceMechanisms + WorkspaceFactMechanisms,
    M::Error: std::error::Error + Send + Sync + 'static,
{
    fn emission_control_bytes(&self) -> Option<usize> {
        control_bytes::<M::Error>()
    }

    fn emit(
        &self,
        operation: WorkspaceOperationView<'_>,
        remaining_bytes: &mut usize,
        funding: Option<&HostMetadataFunding>,
        context: &super::super::WorkspaceContext,
    ) -> Result<EmittedWorkspaceFacts, FactEmissionFailure> {
        let controls = control_bytes::<M::Error>().ok_or(FactEmissionFailure::Overflow)?;
        debit(remaining_bytes, controls, funding)?;
        // The exact canonical source owner is paid before mechanism dispatch;
        // error erasure does not format or copy its diagnostic.
        self.with_prepared_facts_context(operation, context, |mechanism| {
            let prepared = WorkspaceFactPreparation::inspect(operation, mechanism)
                .map_err(retain::<M::Error>)?;
            debit(remaining_bytes, prepared.buffer_bytes, funding)?;
            prepared
                .construct(operation, mechanism)
                .map_err(retain::<M::Error>)
        })
        .map_err(|cause| retain::<M::Error>(FactPreparationError::Mechanism(cause)))?
    }
}

fn retain<E: std::error::Error + Send + Sync + 'static>(
    cause: FactPreparationError<E>,
) -> FactEmissionFailure {
    FactEmissionFailure::Retained(Error::backend_retained_source(cause))
}

fn debit(
    remaining: &mut usize,
    bytes: usize,
    funding: Option<&HostMetadataFunding>,
) -> Result<(), FactEmissionFailure> {
    let next = remaining
        .checked_sub(bytes)
        .ok_or(FactEmissionFailure::Capacity {
            required: bytes,
            available: *remaining,
        })?;
    if let Some(funding) = funding {
        funding
            .reserve_metadata(bytes)
            .map_err(FactEmissionFailure::Funding)?;
    }
    *remaining = next;
    Ok(())
}

fn add<E>(left: usize, right: usize) -> Result<usize, FactPreparationError<E>> {
    left.checked_add(right)
        .ok_or(FactPreparationError::Overflow)
}

fn array_bytes<T, E>(count: usize) -> Result<usize, FactPreparationError<E>> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| FactPreparationError::Overflow)
}

fn initialized<T: Clone, E>(count: usize, value: T) -> Result<Vec<T>, FactPreparationError<E>> {
    array_bytes::<T, E>(count)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    values.resize(count, value);
    Ok(values)
}

fn control_bytes<E: std::error::Error + Send + Sync + 'static>() -> Option<usize> {
    let parts = [
        size_of::<WorkspaceFactPreparation<E>>(),
        size_of::<Result<WorkspaceFactPreparation<E>, FactPreparationError<E>>>(),
        size_of::<Option<WorkspaceOperationFacts>>(),
        size_of::<Option<WorkspaceHostFacts>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<&dyn WorkspaceFactMechanisms<Error = E>>(),
        size_of::<(
            &mut usize,
            Option<&HostMetadataFunding>,
            &super::super::WorkspaceContext,
        )>(),
        size_of::<Result<Result<EmittedWorkspaceFacts, FactEmissionFailure>, E>>(),
        size_of::<WorkspaceEffectDestination<'_>>(),
        size_of::<WorkspaceHostDestination<'_>>(),
        size_of::<Vec<WorkspaceOutputEffect>>(),
        size_of::<Vec<usize>>(),
        size_of::<Vec<u8>>() * 2,
        size_of::<String>() * 2,
        size_of::<Option<Vec<super::super::WorkspaceScratchAllocation>>>(),
        size_of::<super::super::WorkspaceScratchAllocation>() * 2,
        size_of::<Option<&[super::super::WorkspaceScratchAllocation]>>(),
        size_of::<OwnedOperationFacts>(),
        size_of::<Option<OwnedOperationFacts>>(),
        size_of::<WorkspaceHostBound>(),
        size_of::<Option<WorkspaceHostBound>>(),
        size_of::<EmittedWorkspaceFacts>(),
        size_of::<Result<EmittedWorkspaceFacts, FactPreparationError<E>>>(),
        size_of::<Result<EmittedWorkspaceFacts, FactEmissionFailure>>(),
        size_of::<FactPreparationError<E>>(),
        size_of::<FactEmissionFailure>(),
        metadata_funding::reservation_control_bytes(),
        size_of::<Result<(), WorkspaceEffectError>>(),
        size_of::<Result<String, FromUtf8Error>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<Vec<WorkspaceOutputEffect>, FactPreparationError<E>>>(),
        size_of::<Result<Vec<usize>, FactPreparationError<E>>>(),
        size_of::<Result<Vec<u8>, FactPreparationError<E>>>(),
        size_of::<Layout>(),
        size_of::<usize>() * 4,
        Error::retained_source_construction_bytes::<FactPreparationError<E>>()?,
        size_of::<Error>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

fn scratch_layout<E>(
    population: &super::super::WorkspaceAllocationPopulation,
    expected: Option<u64>,
) -> Result<(usize, usize), FactPreparationError<E>> {
    let rows = population.allocations();
    let mut bytes = array_bytes::<super::super::WorkspaceScratchAllocation, E>(rows.len())?;
    for row in rows {
        if let Some(placement) = &row.placement {
            bytes = add(
                bytes,
                usize::try_from(
                    placement
                        .backing_bytes()
                        .map_err(|_| FactPreparationError::Changed)?,
                )
                .map_err(|_| FactPreparationError::Overflow)?,
            )?;
        }
    }
    if expected != population.backing_bytes() {
        return Err(FactPreparationError::Changed);
    }
    Ok((rows.len(), bytes))
}
