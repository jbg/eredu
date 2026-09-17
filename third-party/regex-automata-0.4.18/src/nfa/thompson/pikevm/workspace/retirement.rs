//! Consuming retirement of a failed workspace while its source is still borrowed.
use super::{capacities, Buffer, PrepareError, Requirements};
use alloc::collections::TryReserveError;
use core::fmt;

/// Owned diagnostics and the actual reserve cause after all partial buffers retire.
/// No source borrow or reusable storage escapes this value. Retiring buffers does
/// not change any enclosing account; callers must keep its allowance alive until
/// this method completes and retain their original error charge as required.
#[derive(Debug)]
pub struct RetiredPrepareError {
    requirements: Requirements,
    capacities: [usize; 7],
    buffer: Buffer,
    cause: TryReserveError,
}
impl PrepareError<'_> {
    /// Destroy the actual partial cache before returning its owned cause.
    pub fn retire(self) -> RetiredPrepareError {
        let Self {
            source: _,
            requirements,
            buffer,
            cause,
            partial,
        } = self;
        let capacities = capacities(&partial);
        drop(partial);
        RetiredPrepareError {
            requirements,
            capacities,
            buffer,
            cause,
        }
    }
}
impl RetiredPrepareError {
    /// Original requested layouts, including preparation and retirement controls.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Capacities which have already retired, in Buffer declaration order.
    pub fn retired_capacities(&self) -> [usize; 7] {
        self.capacities
    }
    /// Actual failed destination.
    pub fn buffer(&self) -> Buffer {
        self.buffer
    }
    /// The unchanged error returned by its real reserve attempt.
    pub fn cause(&self) -> &TryReserveError {
        &self.cause
    }
}
impl fmt::Display for RetiredPrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
#[cfg(feature = "std")]
impl std::error::Error for RetiredPrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[cfg(all(test, not(feature = "internal-instrument-pikevm")))]
mod tests {
    use super::*;
    use crate::nfa::thompson::pikevm::PikeVM;
    use alloc::string::ToString;
    #[test]
    fn each_actual_partial_frontier_retires_before_owned_error_escapes_source()
    {
        for buffer in Buffer::ALL {
            let (error, before, diagnostic) = {
                let source = PikeVM::new(r"(?:αβ|ab)+").unwrap();
                let mut plan = source.workspace_plan().unwrap();
                plan.fail = Some(buffer);
                let error = plan.prepare().unwrap_err();
                let before = error.capacities();
                let diagnostic = error.cause().to_string();
                (error.retire(), before, diagnostic)
            };
            assert_eq!(error.retired_capacities(), before);
            assert_eq!(error.buffer(), buffer);
            assert_eq!(error.cause().to_string(), diagnostic);
            assert!(error.requirements().capacity(buffer) > 0);
        }
    }
}
