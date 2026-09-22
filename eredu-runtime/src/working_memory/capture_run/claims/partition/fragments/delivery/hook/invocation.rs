//! One ordinary projected native callback under its original fragment owner.
use super::observe::{LocalCause, PartitionLocalCaptureFailure};
use super::*;
use crate::capture::partition::{
    PartitionCaptureProgramError, PartitionInvocationCaptureGeometry,
    PartitionInvocationCaptureKind,
};
use crate::capture::{FundedCaptureError, ScheduledCaptureBackend};

impl PartitionLocalCaptureHook {
    pub(crate) fn observe_invocation_program<T, E: std::error::Error + Send + Sync + 'static>(
        self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        value: &T,
    ) -> Result<Self, PartitionCaptureProgramError> {
        self.observe_invocation(backend, value).map_err(|failure| {
            let source = failure._hook.source().clone();
            let metadata = failure._hook.funding().clone();
            PartitionCaptureProgramError::local(failure, source, metadata)
        })
    }
    /// Spend the actual local source allowance before the existing typed worker
    /// receives a Host claim. The result remains in its original fragment slot;
    /// transport and the final scheduled record still belong to the caller.
    pub(crate) fn observe_invocation<T, E: std::error::Error + Send + Sync + 'static>(
        mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        value: &T,
    ) -> Result<Self, PartitionLocalCaptureFailure<E>> {
        let result = (|| {
            self.metadata
                .reserve_metadata(control_bytes::<T, E>().ok_or(LocalCause::Identity)?)?;
            let receipt = &self.receipt;
            let bank = &mut self.bank;
            let rank = bank.allowance.local_rank();
            let source = PartitionInvocationCaptureGeometry::from_receipt(receipt, rank, 0)?;
            let (dtype, usage) = match source.kind() {
                PartitionInvocationCaptureKind::Tensor(geometry) => (
                    backend
                        .validate_source(value, geometry)
                        .map_err(FundedCaptureError::Backend)?,
                    backend.estimate(value, geometry)?,
                ),
                PartitionInvocationCaptureKind::Summary(geometry) => (
                    backend.validate_summary_source(value, geometry)?,
                    backend.estimate_summary(geometry)?,
                ),
                PartitionInvocationCaptureKind::Histogram(geometry) => (
                    backend.validate_histogram_source(value, geometry)?,
                    backend.estimate_histogram(geometry)?,
                ),
            };
            let destination = bank.take_local(
                receipt,
                0,
                &dtype,
                PartitionCaptureNativeEstimate {
                    capture: usage,
                    generated_creation_bytes: 0,
                },
            )?;
            let (claim, mut loan) = destination.into_parts();
            loan.quota_mut().reserve_quota(usage)?;
            let completed = match claim {
                PartitionFragmentDestination::Routed(_) => return Err(LocalCause::Identity),
                PartitionFragmentDestination::Tensor(claim) => PartitionFragmentValue::Tensor(
                    backend
                        .transform(value, claim)
                        .map_err(FundedCaptureError::Backend)?,
                ),
                PartitionFragmentDestination::Summary(claim) => {
                    PartitionFragmentValue::Summary(backend.transform_summary(value, claim)?)
                }
                PartitionFragmentDestination::Histogram(claim) => {
                    PartitionFragmentValue::Histogram(backend.transform_histogram(value, claim)?)
                }
            };
            bank.record(receipt, rank, 0, &dtype, usage, completed)?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(self),
            Err(cause) => Err(PartitionLocalCaptureFailure { cause, _hook: self }),
        }
    }
}

fn control_bytes<T, E: std::error::Error + Send + Sync + 'static>() -> Option<usize> {
    let parts = [
        PartitionInvocationCaptureGeometry::control_bytes()?,
        crate::capture::partition::PartitionCaptureLocalHook::control_bytes()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<PartitionLocalCaptureFailure<E>>(
        )?,
        size_of::<PartitionLocalCaptureHook>() * 2,
        size_of::<PartitionLocalCaptureFailure<E>>(),
        size_of::<LocalCause<E>>(),
        size_of::<Result<PartitionLocalCaptureHook, PartitionLocalCaptureFailure<E>>>(),
        size_of::<Result<(), LocalCause<E>>>(),
        size_of::<(
            PartitionLocalCaptureHook,
            &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
            &T,
        )>(),
        size_of::<(
            &mut PartitionLocalCaptureHook,
            &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
            &T,
        )>(),
        size_of::<crate::capture::partition::PartitionCaptureLocalHook>() * 2,
        size_of::<PartitionCaptureProgramError>(),
        size_of::<
            Result<
                crate::capture::partition::PartitionCaptureLocalHook,
                PartitionCaptureProgramError,
            >,
        >(),
        size_of::<NativePartitionFragmentDestination<'_, '_>>(),
        size_of::<PartitionFragmentDestination<'_, '_>>(),
        size_of::<
            Result<NativePartitionFragmentDestination<'_, '_>, PartitionFragmentDestinationError>,
        >(),
        size_of::<(
            PartitionFragmentDestination<'_, '_>,
            PreparedPartitionFragmentLoan,
        )>(),
        size_of::<PreparedPartitionFragmentLoan>(),
        size_of::<PartitionCaptureNativeEstimate>(),
        size_of::<TensorDtype>(),
        size_of::<CaptureUsage>(),
        size_of::<(TensorDtype, CaptureUsage)>(),
        size_of::<Result<TensorDtype, E>>(),
        size_of::<Result<TensorDtype, FundedCaptureError<E>>>(),
        size_of::<Result<CaptureUsage, CaptureError>>(),
        size_of::<PartitionFragmentValue>(),
        size_of::<CaptureTensorClaim<'_, '_>>(),
        size_of::<ClaimedCaptureTensor>(),
        size_of::<Result<ClaimedCaptureTensor, E>>(),
        size_of::<CaptureSummaryClaim<'_, '_>>(),
        size_of::<ClaimedCaptureSummary>(),
        size_of::<Result<ClaimedCaptureSummary, FundedCaptureError<E>>>(),
        size_of::<CaptureHistogramClaim<'_, '_>>(),
        size_of::<ClaimedCaptureHistogram>(),
        size_of::<Result<ClaimedCaptureHistogram, FundedCaptureError<E>>>(),
        size_of::<SharedCapturePlan>(),
        size_of::<HostMetadataFunding>(),
        size_of::<FundedCaptureError<E>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
