//! Borrowed projection requirements for one actual saved-components source.
use crate::backend::error::Error;
use crate::backend::nn::workspace::OwnedArrayProjection;
use crate::backend::nn::workspace::{
    MlxMetalWorkspaceMechanisms, ResidentExecutionMechanisms, ProjectedNativeStorage, ProjectionSourceError,
    ProjectionSourceLayout,
};
use crate::backend::runtime::cache::state::PreparedResidentDecoderCopy;
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::WorkspaceContext;
use eredu_nn::workspace::WorkspaceTensor;
use safemlx::{Array, ImmutableHostTransferBuffer};
use std::sync::Arc;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum SnapshotProjectionCause {
    #[error(transparent)]
    Source(#[from] ProjectionSourceError),
    #[error(transparent)]
    Paged(#[from] crate::backend::runtime::cache::residency::CacheSourceError),
    #[error(transparent)]
    PagedWork(#[from] crate::backend::runtime::cache::residency::CacheSourceFailure),
    #[error("snapshot projection storage layout overflows")]
    Overflow,
    #[error("snapshot projection source geometry changed")]
    Changed,
    #[error("snapshot projection source geometry changed: {0}")]
    ChangedAt(&'static str),
    #[error("snapshot projection vector reservation failed")]
    Reserve(#[from] std::collections::TryReserveError),
}

/// Physical source kind is preserved through cold snapshot projection. This
/// borrowed descriptor supplies no numerical execution or materialization grant.
#[derive(Clone, Copy)]
pub(crate) enum SnapshotOperand<'a> {
    Array(&'a Array),
    Host(&'a Arc<ImmutableHostTransferBuffer>),
}

/// Sources are visited under their real lexical ownership. References cannot
/// escape a manager loan. This supplies no numerical or completed-source grant.
pub(crate) trait SnapshotArraySources {
    fn visit_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause>;
    fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        self.visit_arrays(&mut |array| visitor(SnapshotOperand::Array(array)))
    }
    fn source_controls(&self) -> Result<usize, SnapshotProjectionCause> {
        Ok(0)
    }
    fn copy_source_count(&self) -> usize {
        0
    }
    fn pin_sources(
        &self,
        _projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<(), SnapshotProjectionCause> {
        Ok(())
    }
}
impl SnapshotArraySources for PreparedResidentDecoderCopy<'_> {
    fn visit_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        let mut failure = None;
        self.visit_operands(&mut |array| {
            if failure.is_none() {
                failure = visitor(array).err();
            }
        })?;
        failure.map_or(Ok(()), Err)
    }
    fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        match self.paged_source() {
            Some(source) => source.visit_operands(visitor),
            None => self.visit_arrays(&mut |array| visitor(SnapshotOperand::Array(array))),
        }
    }
    fn source_controls(&self) -> Result<usize, SnapshotProjectionCause> {
        self.paged_source()
            .map_or(Ok(0), SnapshotArraySources::source_controls)
    }
    fn copy_source_count(&self) -> usize {
        self.paged_source()
            .map_or(0, SnapshotArraySources::copy_source_count)
    }
    fn pin_sources(
        &self,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<(), SnapshotProjectionCause> {
        self.paged_source()
            .map_or(Ok(()), |source| source.pin_sources(projection))
    }
}

/// Exact borrowed source selection. Numerical buffers, source inventory pins,
/// registered binding and the later private copy graph are separate components.
pub(crate) struct SnapshotProjectionPlan<'a> {
    decoder: &'a dyn SnapshotArraySources,
    key: Option<&'a Array>,
    pending: Option<&'a Array>,
    geometry: Geometry,
    bytes: usize,
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct Geometry {
    operands: usize,
    host_operands: usize,
    shape_dimensions: usize,
    source_bytes: usize,
    source_controls: usize,
    source_pins: usize,
}
pub(crate) struct ProjectedSnapshotInputs {
    pub(crate) context: WorkspaceContext,
    pub(crate) inputs: Vec<WorkspaceTensor>,
    pub(crate) native: ProjectedNativeStorage,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ProjectionFailure {
    #[source]
    cause: SnapshotProjectionCause,
    // Core deallocates its typed source Box before this last host token.
    _host: HostPreparationAuthority,
}

impl<'a> SnapshotProjectionPlan<'a> {
    /// No context, shape vector, native clone or formatted error is created.
    /// The decoder plan already borrows the settled, selected representation;
    /// the key and pending token are the same optional actual source operands.
    pub(crate) fn inspect(
        decoder: &'a dyn SnapshotArraySources,
        key: Option<&'a Array>,
        pending: Option<&'a Array>,
    ) -> Result<Self, SnapshotProjectionCause> {
        let geometry = measure(decoder, key, pending)?;
        let inventory = OwnedArrayProjection::construction_bytes(geometry.operands)
            .ok_or(SnapshotProjectionCause::Overflow)?;
        let fixed = [
            WorkspaceContext::construction_bytes::<ResidentExecutionMechanisms>()
                .ok_or(SnapshotProjectionCause::Overflow)?,
            inventory,
            geometry.source_controls,
            if geometry.source_pins == 0 {
                0
            } else {
                OwnedArrayProjection::copy_source_control_bytes(geometry.source_pins)
                    .ok_or(SnapshotProjectionCause::Overflow)?
            },
            geometry.source_bytes,
            Layout::array::<WorkspaceTensor>(geometry.operands)
                .map_err(|_| SnapshotProjectionCause::Overflow)?
                .size(),
            size_of::<Self>(),
            size_of::<Geometry>(),
            size_of::<ProjectedSnapshotInputs>(),
            size_of::<Result<Self, SnapshotProjectionCause>>(),
            size_of::<Result<ProjectedSnapshotInputs, SnapshotProjectionCause>>(),
            size_of::<Result<ProjectedSnapshotInputs, Error>>(),
            size_of::<Option<SnapshotProjectionCause>>(),
            size_of::<SnapshotProjectionCause>(),
            size_of::<ProjectionFailure>(),
            size_of::<Vec<WorkspaceTensor>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<(&dyn SnapshotArraySources, Option<&Array>, Option<&Array>)>(),
            size_of::<&mut dyn FnMut(&Array)>(),
            size_of::<SnapshotOperand<'_>>(),
            size_of::<bool>(),
            size_of::<&mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>>(),
            size_of::<std::iter::Chain<std::option::IntoIter<&Array>, std::option::IntoIter<&Array>>>(
            ),
            BackendFailure::source_retention_peak_bytes::<ProjectionFailure>()
                .ok_or(SnapshotProjectionCause::Overflow)?,
        ];
        let bytes = fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)
            .ok_or(SnapshotProjectionCause::Overflow)?;
        Ok(Self {
            decoder,
            key,
            pending,
            geometry,
            bytes,
        })
    }
    /// Exact source-count/rank-derived contribution, not whole snapshot fit.
    pub(crate) fn requested_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn operand_count(&self) -> usize {
        self.geometry.operands
    }
    pub(crate) fn array_operand_count(&self) -> usize {
        self.geometry.operands - self.geometry.host_operands
    }
    /// Reuses actual borrowed operands for adjoining cold component queries.
    pub(crate) fn visit_sources(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), SnapshotProjectionCause> {
        visit(self.decoder, self.key, self.pending, &mut |array| {
            visitor(array);
            Ok(())
        })
    }
    /// Typed physical operands for adjoining transfer-aware copy producers.
    /// Legacy array-only callers retain their explicit Host refusal.
    pub(crate) fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        visit_operands(self.decoder, self.key, self.pending, visitor)
    }
    /// Consumes the same source selection under its admitted host lifetime.
    /// Revalidation precedes metadata allocation. A later failure retires its
    /// prefix before returning an error that retains this same host custody.
    pub(crate) fn construct(
        self,
        mechanisms: MlxMetalWorkspaceMechanisms,
        host: &HostPreparationAuthority,
    ) -> Result<ProjectedSnapshotInputs, Error> {
        self.construct_selected(ResidentExecutionMechanisms::Metal(mechanisms), host, false)
    }
    /// Resumed managers retain exact immutable Host roots through the complete
    /// source owner; only actual Array operands enter the isolated-copy proof.
    pub(crate) fn construct_resume_arrays(
        self,
        mechanisms: ResidentExecutionMechanisms,
        host: &HostPreparationAuthority,
    ) -> Result<ProjectedSnapshotInputs, Error> {
        self.construct_selected(mechanisms, host, true)
    }
    fn construct_selected(
        self,
        mechanisms: ResidentExecutionMechanisms,
        host: &HostPreparationAuthority,
        arrays_only: bool,
    ) -> Result<ProjectedSnapshotInputs, Error> {
        let result = (|| -> Result<ProjectedSnapshotInputs, SnapshotProjectionCause> {
            if measure(self.decoder, self.key, self.pending)? != self.geometry {
                return Err(SnapshotProjectionCause::ChangedAt("projection operand/control census"));
            }
            let context = WorkspaceContext::new(mechanisms);
            // Installed before any manager callback: a failed alias or pin
            // prefix retires only after every lexical source loan has returned.
            let mut destination = None;
            let mut inputs = Vec::new();
            inputs.try_reserve_exact(self.geometry.operands)?;
            {
                let mut projection = OwnedArrayProjection::prepare_with_host(
                    &mut destination,
                    &context,
                    self.geometry.operands,
                    host,
                )?;
                if self.geometry.source_pins != 0 {
                    projection.prepare_copy_sources(self.geometry.source_pins)?;
                    self.decoder.pin_sources(&mut projection)?;
                }
                self.visit_operands(&mut |source| {
                    if arrays_only && matches!(source, SnapshotOperand::Host(_)) {
                        return Ok(());
                    }
                    let input = match source {
                        SnapshotOperand::Array(array) => projection.project_prepared(array)?,
                        SnapshotOperand::Host(host) => projection.project_host::<4>(host)?,
                    };
                    inputs.push(input);
                    Ok(())
                })?;
            }
            let native = destination.expect("complete owned snapshot projection");
            Ok(ProjectedSnapshotInputs {
                context,
                inputs,
                native,
            })
        })();
        result.map_err(|cause| retain_failure(cause, host))
    }
}
fn visit(
    decoder: &dyn SnapshotArraySources,
    key: Option<&Array>,
    pending: Option<&Array>,
    visitor: &mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>,
) -> Result<(), SnapshotProjectionCause> {
    decoder.visit_arrays(visitor)?;
    for array in key.into_iter().chain(pending) {
        visitor(array)?;
    }
    Ok(())
}
fn visit_operands(
    decoder: &dyn SnapshotArraySources,
    key: Option<&Array>,
    pending: Option<&Array>,
    visitor: &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
) -> Result<(), SnapshotProjectionCause> {
    decoder.visit_operands(visitor)?;
    for array in key.into_iter().chain(pending) {
        visitor(SnapshotOperand::Array(array))?;
    }
    Ok(())
}
fn measure(
    decoder: &dyn SnapshotArraySources,
    key: Option<&Array>,
    pending: Option<&Array>,
) -> Result<Geometry, SnapshotProjectionCause> {
    let mut geometry = Geometry {
        operands: 0,
        host_operands: 0,
        shape_dimensions: 0,
        source_bytes: 0,
        source_controls: decoder.source_controls()?,
        source_pins: decoder.copy_source_count(),
    };
    visit_operands(decoder, key, pending, &mut |operand| {
        let (rank, bytes) = match operand {
            SnapshotOperand::Array(array) => (
                ProjectionSourceLayout::inspect(array)?.rank(),
                OwnedArrayProjection::source_bytes(array)?,
            ),
            SnapshotOperand::Host(host) => {
                let descriptor = host
                    .try_fixed_descriptor::<4>()
                    .map_err(ProjectionSourceError::from)?;
                geometry.host_operands = geometry
                    .host_operands
                    .checked_add(1)
                    .ok_or(SnapshotProjectionCause::Overflow)?;
                (
                    descriptor.shape().len(),
                    OwnedArrayProjection::host_source_bytes::<4>(host)?,
                )
            }
        };
        geometry.operands = geometry
            .operands
            .checked_add(1)
            .ok_or(SnapshotProjectionCause::Overflow)?;
        geometry.shape_dimensions = geometry
            .shape_dimensions
            .checked_add(rank)
            .ok_or(SnapshotProjectionCause::Overflow)?;
        geometry.source_bytes = geometry
            .source_bytes
            .checked_add(bytes)
            .ok_or(SnapshotProjectionCause::Overflow)?;
        Ok(())
    })?;
    Ok(geometry)
}

pub(crate) fn retain_failure(
    cause: SnapshotProjectionCause,
    host: &HostPreparationAuthority,
) -> Error {
    Error::StorageSource(BackendFailure::from_error(ProjectionFailure {
        cause,
        _host: host.clone(),
    }))
}

impl SnapshotProjectionCause {
    pub(crate) fn into_error(self) -> Error {
        Error::StorageSource(BackendFailure::from_error(self))
    }
}
