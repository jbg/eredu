//! Exact typed control-slot construction after fresh host admission.

mod exchange;
use super::{MlxNativeTextState, NativeMemoryRetention};
use crate::backend::error::Error;
use crate::composition::mlx::replicated_text::PreparedDenseControlBindingError;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::replicated_session::{PreparedControlBindingError, ReplicatedTextControlState};
pub(crate) use exchange::{OriginalControlExchangePlan, PreparedControlExchange};

/// Fixed local causes. The sealed native resume entry retains host custody on
/// the resulting Error, including conversion/validation failures after inputs
/// have been consumed. No formatted diagnostic or nested source box is needed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PreparedControlSlotError {
    #[error("prepared control-slot construction is unavailable")]
    Unknown,
    #[error("native control state type differs")]
    Type,
    #[error(transparent)]
    Source(exchange::ExchangeSourceCause),
    #[error(transparent)]
    Exchange(
        #[from]
        eredu_runtime::replicated_session::PreparedControlExchangeError<eredu_nn::Error, Error>,
    ),
    #[error("{0}")]
    Policy(&'static str),
    #[error(transparent)]
    State(#[from] PreparedDenseControlBindingError),
    #[error(transparent)]
    Binding(#[from] PreparedControlBindingError),
}

pub(crate) fn error(cause: impl Into<PreparedControlSlotError>) -> Error {
    Error::Other(Box::new(cause.into()))
}

impl MlxNativeTextState {
    /// One actual typed Box allocation, its stack value and return transports,
    /// plus the fixed local failure source. Source table/identity payloads,
    /// pending input, native Q/B and the outer retained error are priced by
    /// their existing owners. This query neither copies nor adopts that storage.
    pub(crate) fn prepared_control_bytes<S: 'static>() -> Option<usize> {
        let parts = [
            std::mem::size_of::<ReplicatedTextControlState<S>>(), // Box allocation
            std::mem::size_of::<ReplicatedTextControlState<S>>(), // constructed value
            std::mem::size_of::<Result<ReplicatedTextControlState<S>, PreparedControlBindingError>>(
            ),
            std::mem::size_of::<Result<S, PreparedDenseControlBindingError>>(),
            std::mem::size_of::<Box<ReplicatedTextControlState<S>>>(),
            std::mem::size_of::<Box<dyn std::any::Any>>(),
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<Result<(), PreparedControlBindingError>>(),
            std::mem::size_of::<
                Result<
                    (),
                    eredu_runtime::replicated_session::PreparedControlExchangeError<
                        eredu_nn::Error,
                        Error,
                    >,
                >,
            >(),
            std::mem::size_of::<(
                &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
                &HostPreparationAuthority,
            )>(),
            std::mem::size_of::<PreparedControlSlotError>(), // one Error::Other source
            std::mem::size_of::<Box<PreparedControlSlotError>>(),
            std::mem::size_of::<Error>(),
            std::mem::size_of::<HostPreparationAuthority>(),
            std::mem::size_of::<NativeMemoryRetention>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// The Box's concrete type stays unchanged for checked downcast/exchange.
    /// H is a constructor lifetime, never a numerical or publication grant.
    pub(crate) fn from_prepared<S: 'static>(
        state: ReplicatedTextControlState<S>,
        host: &HostPreparationAuthority,
    ) -> Self {
        Self {
            state: Box::new(state),
            displaced_placement: None,
            memory_retention: NativeMemoryRetention::default(),
            host_preparation: Some(host.clone()),
        }
    }
}
