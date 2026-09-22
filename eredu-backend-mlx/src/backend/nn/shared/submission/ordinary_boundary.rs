//! Caller populations of the ordinary role-exact send/receive worker.
use super::*;
use crate::backend::error::Error;
use crate::backend::nn::workspace::OrdinaryCallControls;
use crate::backend::runtime::distributed::{
    completion::prepared::CompletionResourceLayout, topology::CommunicationRouteEndpoint,
};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

fn vector<T>(count: usize) -> Option<usize> {
    Layout::array::<T>(count)
        .ok()?
        .size()
        .checked_add(size_of::<Vec<T>>())
}
#[derive(Debug, thiserror::Error)]
#[error("ordinary boundary frame population differs from its retained route")]
struct PopulationFailure {
    _host: Option<eredu_core::HostPreparationAuthority>,
}
pub(super) fn validate_count(
    route: &CommunicationRouteRealization,
    frames: usize,
) -> Result<usize, safemlx::error::Exception> {
    let maximum = route
        .descriptor()
        .requirement()
        .limits()
        .map(|limits| limits.max_tensors());
    if frames != 0 && maximum.is_some_and(|maximum| frames <= maximum) {
        if let Some(retained) = frames.checked_mul(4) {
            return Ok(retained);
        }
    }
    let host = current_ordinary_execution_owner()?.map(|owner| owner.host().clone());
    Err(safemlx::error::Exception::from_retained_source(
        PopulationFailure { _host: host },
    ))
}
impl MlxNeuralBackend {
    /// Same worker as `send_receive`: `header_bytes` is the sum of exact
    /// borrowed header lengths across these frames. Native header seeds and encoding/transport/decoding graphs
    /// are separate populations from these caller and completed-reader controls.
    /// Summing one-frame quotations is a conservative envelope for a bundle;
    /// fixed group/route/completion owners then leave unused allowance.
    pub(crate) fn ordinary_boundary_call_controls(
        route: &CommunicationRouteRealization,
        frames: usize,
        header_bytes: usize,
    ) -> Option<OrdinaryCallControls> {
        if frames == 0 {
            return None;
        }
        let group = route.group()?;
        let maximum = route.descriptor().requirement().limits()?.max_tensors();
        if frames > maximum {
            return None;
        }
        let receiving = match route.endpoint()? {
            CommunicationRouteEndpoint::Source => false,
            CommunicationRouteEndpoint::Destination => true,
        };
        // One bundled C vector grows geometrically. Per-frame quotations use
        // the authenticated route cap for this source, never sum smaller root
        // vectors as though they were the actual bundled caller.
        let root_envelope = maximum.checked_mul(if receiving { 3 } else { 2 })?;
        let retained = frames.checked_mul(4)?;
        let mut calls = MlxCommunicationCompletion::ordinary_submit_call_controls(
            root_envelope,
            CompletionResourceLayout {
                arrays: retained,
                counts: &[],
                groups: 1,
                routes: 1,
                streams: 1,
            },
        )?;
        let parts = [
            // Logical validation aliases and exact input/frame destinations.
            vector::<MlxTensor>(frames)?,
            vector::<Array>(frames)?,
            vector::<Vec<u8>>(frames)?,
            // Submitted endpoints, returned values, and the pre-submission
            // borrowed-root vector coexist with the retained array destination.
            vector::<Array>(frames)?,
            vector::<Array>(frames)?,
            vector::<&Array>(retained)?,
            vector::<MlxTensor>(frames)?,
            crate::backend::nn::boundary_frame::control_bytes::<
                crate::backend::nn::boundary_frame::Native<'_>,
            >()?
            .checked_mul(frames)?,
            group.retention_copy_bytes()?,
            route.retention_copy_bytes()?,
            // The role-exact header bytes move into this worker; its final
            // completion source retains the prepaid custody for that lifetime.
            header_bytes,
            size_of::<Vec<RoleExactBoundaryValue<MlxTensor>>>(),
            size_of::<(&CommunicationRouteRealization, &Group, &Stream)>(),
            size_of::<RoleExactBoundaryValue<MlxTensor>>(),
            size_of::<(Vec<u8>, MlxTensor)>(),
            size_of::<(Array, Array)>(),
            size_of::<(usize, i32, bool)>(),
            size_of::<Result<Array, safemlx::error::Exception>>(),
            size_of::<Submission<Vec<MlxTensor>, MlxCommunicationCompletion>>(),
            size_of::<Result<Submission<Vec<MlxTensor>, MlxNeuralCommunicationCompletion>, Error>>(
            ),
            size_of::<std::slice::Iter<'static, RoleExactBoundaryValue<MlxTensor>>>(),
            size_of::<std::slice::Iter<'static, Array>>(),
            size_of::<std::vec::IntoIter<Array>>(),
            size_of::<std::vec::IntoIter<Vec<u8>>>(),
            size_of::<PopulationFailure>(),
            safemlx::error::Exception::retained_source_control_bytes::<PopulationFailure>()?,
        ];
        let mut bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?;
        if receiving {
            bytes = bytes
                .checked_add(
                    crate::backend::nn::boundary_frame::control_bytes::<
                        crate::backend::nn::boundary_frame::Native<'_>,
                    >()?
                    .checked_add(size_of::<(Array, Array)>())?
                    .checked_mul(frames)?,
                )?
                .checked_add(vector::<Array>(frames)?)?
                .checked_add(vector::<(Array, Vec<u8>)>(frames)?)?
                .checked_add(header_bytes)?
                .checked_add(
                    MlxCommunicationCompletion::ordinary_boundary_headers_control_bytes(
                        frames,
                        header_bytes,
                    )?,
                )?;
        }
        calls.metadata_bytes = calls
            .metadata_bytes
            .checked_add(u64::try_from(bytes).ok()?)?;
        Some(calls)
    }
}
