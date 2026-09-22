//! Prospective storage for the shared declaration worker.
use super::*;
use crate::{HostMetadataFunding, HostMetadataFundingError};
use std::{alloc::Layout, fmt, mem::size_of};

/// Fixed storage refusal while producing a capture declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CaptureAdmissionStorageError {
    /// The actual retained metadata account refused the next producer.
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    /// The host allocator refused the concrete destination.
    #[error("capture declaration host allocation failed")]
    Allocation,
    /// The selected host allocator supplied an unexpected capacity.
    #[error("capture declaration destination capacity differs from its plan")]
    Capacity,
}

#[derive(Clone, Copy)]
pub(crate) struct Allocation<'a>(pub(crate) Option<&'a HostMetadataFunding>);
impl Allocation<'_> {
    pub(crate) fn reserve(self, bytes: usize) -> Result<(), CaptureError> {
        if let Some(funding) = self.0 {
            funding
                .reserve_metadata(bytes)
                .map_err(CaptureAdmissionStorageError::from)?;
        }
        Ok(())
    }
    pub(crate) fn vector<T>(self, count: usize) -> Result<Vec<T>, CaptureError> {
        let bytes = Layout::array::<T>(count)
            .map_err(|_| CaptureError::Overflow)?
            .size()
            .checked_add(size_of::<Vec<T>>() * 2)
            .and_then(|n| n.checked_add(size_of::<Result<Vec<T>, CaptureError>>()))
            .and_then(|n| n.checked_add(size_of::<std::collections::TryReserveError>()))
            .ok_or(CaptureError::Overflow)?;
        self.reserve(bytes)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| CaptureAdmissionStorageError::Allocation)?;
        if size_of::<T>() != 0 && result.capacity() != count {
            return Err(CaptureAdmissionStorageError::Capacity.into());
        }
        Ok(result)
    }
    pub(crate) fn text(self, value: &str) -> Result<String, CaptureError> {
        let mut bytes = self.vector(value.len())?;
        bytes.extend_from_slice(value.as_bytes());
        Ok(String::from_utf8(bytes).expect("copied UTF-8"))
    }
    pub(crate) fn format(self, value: fmt::Arguments<'_>) -> Result<String, CaptureError> {
        use fmt::Write;
        struct Counter(usize);
        impl fmt::Write for Counter {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                self.0 = self.0.checked_add(value.len()).ok_or(fmt::Error)?;
                Ok(())
            }
        }
        self.reserve(size_of::<Counter>() + size_of::<fmt::Arguments<'_>>() * 2)?;
        let mut counter = Counter(0);
        counter
            .write_fmt(value)
            .map_err(|_| CaptureError::Overflow)?;
        let mut result = String::from_utf8(self.vector(counter.0)?).expect("empty UTF-8");
        result
            .write_fmt(value)
            .expect("closed capture diagnostic formatting");
        debug_assert_eq!(result.len(), counter.0);
        Ok(result)
    }
    pub(crate) fn plan(self, value: &CapturePlan) -> Result<CapturePlan, CaptureError> {
        use super::super::plan_copy::Worker;
        let controls = size_of::<Worker>() * 2 + size_of::<CapturePlan>() * 2;
        let mut count = Worker::with_controls(false, controls);
        drop(count.raw_plan(value).map_err(copy_error)?);
        self.reserve(count.bytes())?;
        Worker::with_controls(true, controls)
            .raw_plan(value)
            .map_err(copy_error)
    }
    pub(crate) fn point(self, value: &ObservationPoint) -> Result<ObservationPoint, CaptureError> {
        use super::super::plan_copy::Worker;
        let controls = size_of::<Worker>() * 2 + size_of::<ObservationPoint>() * 2;
        let mut count = Worker::with_controls(false, controls);
        drop(count.point(value).map_err(copy_error)?);
        self.reserve(count.bytes())?;
        Worker::with_controls(true, controls)
            .point(value)
            .map_err(copy_error)
    }
}
fn copy_error(error: CapturePlanCopyError) -> CaptureError {
    match error {
        CapturePlanCopyError::Overflow => CaptureError::Overflow,
        CapturePlanCopyError::Allocation(_) => CaptureAdmissionStorageError::Allocation.into(),
        CapturePlanCopyError::Capacity => CaptureAdmissionStorageError::Capacity.into(),
        CapturePlanCopyError::Identity => {
            unreachable!("point copy has no admission or identity worker")
        }
    }
}
