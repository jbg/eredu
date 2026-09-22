use super::*;

mod autoregressive_transaction;
mod cache_binding;
pub(super) mod binding;
mod composite;
mod mechanisms;
pub(in crate::composition::mlx) mod parameter_reset;
pub(super) mod prepared_parameters;
mod replicated;

pub(super) use composite::*;
pub(super) use mechanisms::*;
pub(super) use replicated::*;

/// Retains both failures without treating an earlier state-preserving rejection
/// as authority to retry after a failed ownership recovery.
#[derive(Debug, thiserror::Error)]
#[error("prediction operation failed: {operation}; state recovery failed: {recovery}")]
struct PredictionStateRecoveryFailure {
    operation: Error,
    #[source]
    recovery: Error,
}

fn finish_prediction_state_operation<T>(
    operation: Result<T, Error>,
    recovery: Result<(), Error>,
) -> Result<T, Error> {
    finish_prediction_state_operation_with_metadata(operation, recovery, None)
}

fn finish_prediction_state_operation_with_metadata<T>(
    operation: Result<T, Error>,
    recovery: Result<(), Error>,
    funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
) -> Result<T, Error> {
    match (operation, recovery) {
        (Err(operation), Err(recovery)) => {
            let failure = PredictionStateRecoveryFailure {
                operation,
                recovery,
            };
            Err(match funding {
                Some(funding) => Error::Neural(funding.metadata_source(failure)),
                None => Error::Other(Box::new(failure)),
            })
        }
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Ok(output), Ok(())) => Ok(output),
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;

    #[test]
    fn prediction_recovery_failure_retains_both_causes_and_revokes_retry_evidence() {
        let operation = || Error::before_model_mutation(std::io::Error::from_raw_os_error(22));
        let recovery = || Error::Exception(Exception::custom("injected state recovery failure"));
        assert_eq!(finish_prediction_state_operation(Ok(7), Ok(())).unwrap(), 7);
        assert!(
            finish_prediction_state_operation::<()>(Err(operation()), Ok(()))
                .unwrap_err()
                .model_state_preserved()
        );
        assert!(
            !finish_prediction_state_operation(Ok(7), Err(recovery()))
                .unwrap_err()
                .model_state_preserved()
        );
        let failure =
            finish_prediction_state_operation::<()>(Err(operation()), Err(recovery())).unwrap_err();
        assert!(!failure.model_state_preserved());
        let Error::Other(inner) = &failure else {
            panic!("combined typed error")
        };
        let combined = inner
            .downcast_ref::<PredictionStateRecoveryFailure>()
            .unwrap();
        assert!(combined.operation.model_state_preserved());
        let boundary = eredu_core::BackendFailure::from_error(failure);
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&boundary);
        let mut found = false;
        while let Some(cause) = source {
            found |= cause
                .downcast_ref::<Exception>()
                .is_some_and(|native| native.what() == "injected state recovery failure");
            source = cause.source();
        }
        assert!(found, "recovery failure must survive the portable boundary");
    }
}

/// Shared allocation-free policy facts for the selected control mechanism.
fn native_control_policy_support(
    partitioned: bool,
    distributed_agreement: bool,
    prediction_present: bool,
) -> eredu_core::execution_control::ControlSupport<&'static str> {
    use eredu_core::execution_control::ControlSupport;
    match native_control_policy_rejection(partitioned, distributed_agreement, prediction_present) {
        Some(reason) => ControlSupport::Unsupported { reason },
        None => ControlSupport::Supported,
    }
}

// One predicate supplies both admission and independently rendered diagnostics.
fn native_control_policy_rejection(
    partitioned: bool,
    distributed_agreement: bool,
    prediction_present: bool,
) -> Option<&'static str> {
    if partitioned && !distributed_agreement {
        return Some("partitioned state copies require bounded all-rank preparation agreement");
    }
    if prediction_present {
        return Some("native text snapshots do not support selected speculative execution");
    }
    None
}

fn require_native_control_policy(
    partitioned: bool,
    distributed_agreement: bool,
    prediction_present: bool,
) -> Result<(), Error> {
    match native_control_policy_support(partitioned, distributed_agreement, prediction_present) {
        eredu_core::execution_control::ControlSupport::Supported => Ok(()),
        eredu_core::execution_control::ControlSupport::Unsupported { reason } => {
            Err(Error::ArchitectureModel(reason.into()))
        }
    }
}

pub(crate) use mechanisms::{
    NativeOpeningRows, NativeOpeningRowsOwner, NativeOpeningRowsPlan, RetiredOpeningRow,
    SealedOpeningRows, ordinary_model_completion_call_controls,
};

use eredu_nn::workspace::WorkspaceMetadataAllocation;
