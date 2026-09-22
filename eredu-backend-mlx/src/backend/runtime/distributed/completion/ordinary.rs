//! The ordinary communication worker under its actual admitted Work owner.
use super::*;
use crate::backend::{
    nn::{
        shared::{MlxNeuralBackend, OrdinaryExecutionOwner},
        workspace::OrdinaryCallControls,
    },
    submission_recovery::PreparedRecovery,
};
use eredu_core::HostPreparationAuthority;
use safemlx::error::Exception;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error("ordinary communication output iterator has no exact retained population")]
    Population,
    #[error("ordinary communication host destination allocation failed: {0}")]
    Capacity(#[source] std::collections::TryReserveError),
    #[error("ordinary communication recovery preparation failed: {0}")]
    Recovery(#[source] safemlx::SubmissionScopeOwnerCause),
    #[error("ordinary communication housekeeping preparation failed: {0}")]
    Housekeeping(#[source] safemlx::HousekeepingRegistrationCause),
    #[error("ordinary communication native operation failed: {0}")]
    Native(#[source] Exception),
    #[error("ordinary communication completion failed; native resources remain retained")]
    Completion,
    #[error(
        "bounded communication deadline exceeds the host monotonic clock range; live work was quarantined safely"
    )]
    Deadline,
    #[error("MLX communication has no native cancellation; timed-out work was quarantined safely")]
    Cancellation,
    #[error("ordinary communication completed readout differs from its source")]
    Readout,
    #[error("ordinary communication completed scalar read failed: {0}")]
    Readback(#[source] safemlx::error::CompletedReadbackError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    _host: HostPreparationAuthority,
}
pub(super) fn failure(cause: Cause, host: &HostPreparationAuthority) -> Exception {
    Exception::from_retained_source(Failure {
        cause,
        _host: host.clone(),
    })
}
enum CompletedSourceFailure {
    Native(Exception),
    Readback(safemlx::error::CompletedReadbackError),
}
impl CompletedSourceFailure {
    fn retained(self, host: &HostPreparationAuthority) -> Exception {
        failure(
            match self {
                Self::Native(cause) => Cause::Native(cause),
                Self::Readback(cause) => Cause::Readback(cause),
            },
            host,
        )
    }
    fn ordinary(self) -> Exception {
        match self {
            Self::Native(cause) => cause,
            Self::Readback(cause) => Exception::from_source(cause),
        }
    }
}
/// A signalled event can remain attached after its completion owner is waited.
/// Observe this array's actual readiness before the nonpolling descriptor query;
/// neither path evaluates an unscheduled source or waits for pending work.
fn completed_source(value: &Array) -> Result<safemlx::EvaluatedArray<'_>, CompletedSourceFailure> {
    match value
        .try_observe_availability()
        .map_err(CompletedSourceFailure::Native)?
    {
        Some(true) => {}
        Some(false) => {
            return Err(CompletedSourceFailure::Readback(
                safemlx::error::CompletedReadbackError::Attribution,
            ));
        }
        None => {
            return Err(CompletedSourceFailure::Readback(
                safemlx::error::CompletedReadbackError::Busy,
            ));
        }
    }
    value
        .try_completed()
        .map_err(CompletedSourceFailure::Readback)
}
fn completed_source_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<&Array>(),
        size_of::<CompletedSourceFailure>(),
        size_of::<Result<safemlx::EvaluatedArray<'_>, CompletedSourceFailure>>(),
        size_of::<(CompletedSourceFailure, &HostPreparationAuthority)>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn completed_array<'a>(
    value: &'a Array,
    host: &HostPreparationAuthority,
) -> Result<safemlx::EvaluatedArray<'a>, Exception> {
    completed_source(value).map_err(|cause| cause.retained(host))
}
pub(super) fn completed_scalar<T: safemlx::ArrayElement + Copy + Default>(
    value: &Array,
    host: &HostPreparationAuthority,
) -> Result<T, Exception> {
    let mut output = [T::default(); 1];
    completed_array(value, host)?
        .try_copy_into(&mut output)
        .map_err(|cause| failure(Cause::Readback(cause), host))?;
    Ok(output[0])
}
pub(crate) fn ordinary_completed_i32_scalar(value: &Array) -> Result<i32, Exception> {
    if let Some(owner) = crate::backend::nn::shared::current_ordinary_execution_owner()? {
        return completed_scalar(value, owner.host());
    }
    let mut output = [0i32; 1];
    completed_source(value)
        .map_err(CompletedSourceFailure::ordinary)?
        .try_copy_into(&mut output)
        .map_err(Exception::from_source)?;
    Ok(output[0])
}
pub(crate) fn ordinary_completed_i32_scalar_control_bytes() -> Option<usize> {
    scalar_read_controls::<i32>()
}
fn scalar_read_controls<T: safemlx::ArrayElement + Copy>() -> Option<usize> {
    let parts = [
        Array::ordinary_availability_control_bytes()?,
        Array::completed_borrow_control_bytes()?,
        safemlx::EvaluatedArray::completed_readback_control_bytes::<T>()?,
        size_of::<(&Array, &HostPreparationAuthority)>(),
        size_of::<[T; 1]>(),
        size_of::<Result<T, Exception>>(),
        completed_source_control_bytes()?,
        size_of::<Option<OrdinaryExecutionOwner>>(),
        size_of::<Failure>(),
        Exception::retained_source_control_bytes::<Failure>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(super) fn prepare<'a>(
    owner: OrdinaryExecutionOwner,
    outputs: impl Iterator<Item = &'a Array>,
    arrays: Vec<Array>,
    counts: Vec<Vec<usize>>,
    groups: Vec<Group>,
    routes: Vec<CommunicationRouteRealization>,
    streams: Vec<Stream>,
) -> Result<
    (
        Vec<Array>,
        Recovery<NativeOwner>,
        destinations::Destination<MlxCommunicationCompletion>,
    ),
    Exception,
> {
    let host = owner.host().clone();
    let (count, upper) = outputs.size_hint();
    if upper != Some(count) {
        return Err(failure(Cause::Population, &host));
    }
    let mut retained = Vec::new();
    retained
        .try_reserve_exact(count)
        .map_err(|cause| failure(Cause::Capacity(cause), &host))?;
    for output in outputs {
        if retained.len() == count {
            return Err(failure(Cause::Population, &host));
        }
        retained.push(
            output
                .try_clone_handle()
                .map_err(|cause| failure(Cause::Native(cause), &host))?,
        );
    }
    if retained.len() != count {
        return Err(failure(Cause::Population, &host));
    }
    let registry = destinations::Destination::ordinary(host.clone());
    let orphan = destinations::Destination::ordinary(host.clone());
    let housekeeping = safemlx::PreparedThreadRuntimeHousekeeping::new(
        communication::reap_communication_orphans,
        host.clone(),
    )
    .try_register()
    .map_err(|failed| {
        let (cause, prepared) = failed.into_parts();
        let error = failure(Cause::Housekeeping(cause), &host);
        drop(prepared);
        error
    })?;
    let mut resources = NativeResources::from_owned(
        arrays,
        counts,
        groups,
        routes,
        streams,
        None,
        Some(registry),
    );
    *resources.housekeeping.borrow_mut() = Some(housekeeping);
    resources.ordinary = Some(owner.clone());
    let prepared = PreparedRecovery::new(resources, host.clone()).map_err(|failed| {
        let error = failure(Cause::Recovery(failed.cause), &host);
        drop((failed.retention, failed.custody));
        error
    })?;
    let mut recovery = prepared.try_begin().map_err(|failed| {
        // Native rejected before beginning this preparation. Preserve the
        // refused preparation until the paid, source-retaining error exists.
        let error = failure(Cause::Recovery(failed.cause), &host);
        drop(failed.pending);
        error
    })?;
    recovery
        .configure_scope(|scope| scope.bind_physical_observer(owner.observer()))
        .map_err(|cause| failure(Cause::Native(cause), &host))?;
    Ok((retained, recovery, orphan))
}

fn shared<T>() -> Option<usize> {
    Some(
        Layout::new::<[usize; 2]>()
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}
fn vector<T>(count: usize) -> Option<usize> {
    Layout::array::<T>(count)
        .ok()?
        .size()
        .checked_add(size_of::<Vec<T>>())
}
impl MlxCommunicationCompletion {
    /// Exact ordinary header row and reusable readback destinations. Borrowed
    /// native header aliases and expected-byte clones are counted by the outer
    /// boundary producer. Polling only overwrites these fixed scratch buffers.
    pub(crate) fn ordinary_boundary_headers_control_bytes(
        headers: usize,
        bytes: usize,
    ) -> Option<usize> {
        let parts = [
            vector::<communication::BoundaryHeaderResolution>(headers)?,
            bytes,
            size_of::<BoundaryHeaders>(),
            size_of::<Option<HostPreparationAuthority>>(),
            size_of::<std::vec::IntoIter<(Array, Vec<u8>)>>(),
            size_of::<std::cell::RefMut<'static, Vec<u8>>>(),
            size_of::<std::cell::BorrowMutError>(),
            completed_source_control_bytes()?.checked_mul(headers)?,
            Array::ordinary_availability_control_bytes()?.checked_mul(headers)?,
            Array::completed_borrow_control_bytes()?.checked_mul(headers)?,
            safemlx::EvaluatedArray::completed_readback_control_bytes::<u8>()?
                .checked_mul(headers)?,
            size_of::<Failure>(),
            Exception::retained_source_control_bytes::<Failure>()?,
            size_of::<(&Array, &[u8], &HostPreparationAuthority)>(),
            size_of::<(usize, Option<usize>)>(),
            size_of::<Result<(), Exception>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// One actual retained Bool result cell and either the completed I32
    /// agreement reader or F32 flag reader, including an escaped refusal.
    pub(crate) fn ordinary_scalar_result_control_bytes() -> Option<usize> {
        let parts = [
            shared::<Cell<Option<bool>>>()?,
            size_of::<BoolResult>(),
            size_of::<communication::MlxFailureAgreement>(),
            size_of::<communication::MlxCommunicationFlag>(),
            size_of::<(Array, i32, bool)>(),
            scalar_read_controls::<i32>()?.max(scalar_read_controls::<f32>()?),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// The ordinary submit worker's exact retained containers, Array/Stream
    /// clones, recovery, registry, quarantine and housekeeping destinations.
    /// Counts describe supplied Vec capacities. Nested Group/route clone data,
    /// host-result attachments and native Eval are separate producer sources.
    pub(crate) fn ordinary_submit_call_controls(
        outputs: usize,
        layout: prepared::CompletionResourceLayout<'_>,
    ) -> Option<OrdinaryCallControls> {
        let parts = [
            shared::<NativeResources>()?,
            shared::<Event>()?,
            vector::<Array>(outputs)?,
            vector::<Array>(layout.arrays)?,
            vector::<Vec<usize>>(layout.counts.len())?,
            vector::<Group>(layout.groups)?,
            vector::<CommunicationRouteRealization>(layout.routes)?,
            vector::<Stream>(layout.streams)?,
            Array::ordinary_clone_control_bytes()?
                .checked_mul(outputs.checked_add(layout.arrays)?)?,
            Stream::ordinary_clone_control_bytes()?.checked_mul(layout.streams)?,
            usize::try_from(
                PreparedRecovery::<NativeOwner, HostPreparationAuthority>::control_bytes()?,
            )
            .ok()?,
            safemlx::PreparedThreadRuntimeHousekeeping::<HostPreparationAuthority>::control_bytes(
            )?,
            destinations::Destination::<std::rc::Weak<NativeResources>>::control_bytes()?,
            destinations::Destination::<Self>::control_bytes()?,
            Exception::retained_source_control_bytes::<Failure>()?,
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<NativeResources>(),
            size_of::<NativeOwner>(),
            size_of::<OrdinaryExecutionOwner>(),
            size_of::<Option<OrdinaryExecutionOwner>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<(usize, Option<usize>)>(),
            size_of::<std::slice::Iter<'static, Array>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<
                Result<
                    (
                        Vec<Array>,
                        Recovery<NativeOwner>,
                        destinations::Destination<Self>,
                    ),
                    Exception,
                >,
            >(),
            size_of::<(
                Vec<Array>,
                Vec<Vec<usize>>,
                Vec<Group>,
                Vec<CommunicationRouteRealization>,
                Vec<Stream>,
            )>(),
            original::completion_controls()?,
            size_of::<destinations::Destinations<std::rc::Weak<NativeResources>>>(),
            size_of::<communication::CommunicationOrphanQuarantine>(),
        ];
        let mut bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?;
        for &count in layout.counts {
            bytes = bytes.checked_add(vector::<usize>(count)?)?;
        }
        let mut controls = MlxNeuralBackend::ordinary_completion_call_controls(outputs)?;
        controls.metadata_bytes = controls
            .metadata_bytes
            .checked_add(u64::try_from(bytes).ok()?)?;
        Some(controls)
    }

    /// Actual common collective/packed-world retention: one output, input and
    /// output aliases, one selected Group and one Stream. The same Group clone
    /// source includes its real logical routing and manifest destinations.
    pub(crate) fn ordinary_collective_completion_call_controls(
        group: &Group,
        counts: &[usize],
    ) -> Option<OrdinaryCallControls> {
        let mut controls = Self::ordinary_submit_call_controls(
            1,
            prepared::CompletionResourceLayout {
                arrays: 2,
                counts,
                groups: 1,
                routes: 0,
                streams: 1,
            },
        )?;
        controls.metadata_bytes = controls
            .metadata_bytes
            .checked_add(u64::try_from(group.retention_copy_bytes()?).ok()?)?;
        Some(controls)
    }
}
