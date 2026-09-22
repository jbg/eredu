//! Prospective producers for the one controlled record family.
use super::*;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::atomic::AtomicUsize,
};

/// Fixed original record-construction failure. No diagnostic allocation is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RecordConstructionCause {
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error("control record host allocation failed")]
    HostAllocation,
    #[error("control record attribution is invalid")]
    Attribution,
}

/// A refused record producer retains its actual account through error retirement.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct RecordConstructionError {
    #[source]
    cause: RecordConstructionCause,
    _funding: HostMetadataFunding,
}
impl RecordConstructionError {
    /// The original fixed cause, borrowed without detaching its account.
    pub fn cause(&self) -> RecordConstructionCause {
        self.cause
    }
    /// The exact original funding refusal, when that producer was refused.
    pub fn funding_failure(&self) -> Option<HostMetadataFundingError> {
        match self.cause {
            RecordConstructionCause::Funding(error) => Some(error),
            _ => None,
        }
    }
    pub(crate) fn retain(cause: RecordConstructionCause, funding: &HostMetadataFunding) -> Self {
        Self {
            cause,
            _funding: funding.clone(),
        }
    }
}

pub(in crate::api::control) type Result<T> = std::result::Result<T, RecordConstructionCause>;
fn overflow() -> RecordConstructionCause {
    HostMetadataFundingError::Overflow.into()
}
fn total(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(size_of_val(parts), |sum, &part| sum.checked_add(part))
        .ok_or_else(overflow)
}
pub(in crate::api::control) fn controls(
    funding: &HostMetadataFunding,
    parts: &[usize],
) -> Result<()> {
    let shared = [
        total(parts)?,
        HostMetadataFunding::reservation_control_bytes(),
        size_of::<RecordConstructionError>(),
        size_of::<RecordConstructionCause>(),
        size_of::<HostMetadataFunding>(),
        size_of::<(&HostMetadataFunding, &[usize])>(),
        size_of::<(usize, Layout)>(),
        size_of::<Option<usize>>(),
        size_of::<Result<()>>(),
        size_of::<std::collections::TryReserveError>(),
        size_of::<String>(),
        size_of::<Result<String>>(),
        size_of::<(&str, &mut String)>(),
        size_of::<Option<String>>(),
    ];
    funding.reserve_metadata(total(&shared)?)?;
    Ok(())
}
pub(in crate::api::control) fn vector<T>(
    count: usize,
    funding: &HostMetadataFunding,
) -> Result<Vec<T>> {
    let bytes = Layout::array::<T>(count).map_err(|_| overflow())?.size();
    let parts = [
        bytes,
        size_of::<Vec<T>>(),
        size_of::<Result<Vec<T>>>(),
        size_of::<(usize, &HostMetadataFunding)>(),
        size_of::<std::collections::TryReserveError>(),
    ];
    funding.reserve_metadata(total(&parts)?)?;
    let mut destination = Vec::new();
    destination
        .try_reserve_exact(count)
        .map_err(|_| RecordConstructionCause::HostAllocation)?;
    Ok(destination)
}
pub(in crate::api::control) fn string_capacity(
    length: usize,
    funding: &HostMetadataFunding,
) -> Result<String> {
    funding.reserve_metadata(length)?;
    let mut destination = String::new();
    destination
        .try_reserve_exact(length)
        .map_err(|_| RecordConstructionCause::HostAllocation)?;
    Ok(destination)
}
pub(in crate::api::control) fn string(
    source: &str,
    funding: &HostMetadataFunding,
) -> Result<String> {
    let mut destination = string_capacity(source.len(), funding)?;
    destination.push_str(source);
    Ok(destination)
}
pub(in crate::api::control) fn optional(
    source: &Option<String>,
    funding: &HostMetadataFunding,
) -> Result<Option<String>> {
    source
        .as_deref()
        .map(|source| string(source, funding))
        .transpose()
}
pub(in crate::api::control) fn instrumentation(
    capture: &Option<String>,
    intervention: &Option<String>,
    funding: &HostMetadataFunding,
) -> Result<PreparedInstrumentationRecord> {
    Ok(match (capture, intervention) {
        (Some(capture), Some(intervention)) => PreparedInstrumentationRecord::Intervened {
            capture_plan_id: string(capture, funding)?,
            intervention_plan_id: string(intervention, funding)?,
        },
        (Some(capture), None) => PreparedInstrumentationRecord::Captured {
            plan_id: string(capture, funding)?,
        },
        (None, None) => PreparedInstrumentationRecord::Unobserved,
        _ => return Err(RecordConstructionCause::Attribution),
    })
}
pub(in crate::api::control) fn shell<T>(funding: &HostMetadataFunding) -> Result<()> {
    let bytes = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map_err(|_| overflow())?
        .0
        .pad_to_align()
        .size();
    funding.reserve_metadata(bytes)?;
    Ok(())
}

/// The raw DTO stays within the enclosing funded record producer. Its input
/// remains borrowed; copying never adopts caller buffers into record custody.
pub(in crate::api::control) fn intervention_request(
    source: &eredu_core::intervention::InterventionPlan,
    funding: &HostMetadataFunding,
) -> Result<eredu_core::intervention::InterventionPlan> {
    use eredu_core::intervention::PreparedInterventionRequestCopy;
    let failure = |error| match error {
        CapturePlanCopyError::Overflow => {
            RecordConstructionCause::Funding(HostMetadataFundingError::Overflow)
        }
        CapturePlanCopyError::Allocation(_) | CapturePlanCopyError::Capacity => {
            RecordConstructionCause::HostAllocation
        }
        CapturePlanCopyError::Identity => RecordConstructionCause::Attribution,
    };
    controls(
        funding,
        &[
            size_of::<PreparedInterventionRequestCopy<'_>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<eredu_core::intervention::InterventionPlan>>(),
            PreparedInterventionRequestCopy::inspection_control_bytes().ok_or_else(overflow)?,
        ],
    )?;
    let prepared = PreparedInterventionRequestCopy::inspect(source).map_err(failure)?;
    funding.reserve_metadata(prepared.required_bytes())?;
    funding.reserve_metadata(
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().ok_or_else(overflow)?,
    )?;
    let host = HostPreparationAuthority::retain(funding.clone());
    prepared.copy(&host).map_err(failure)
}
