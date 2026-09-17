//! Owned terminal diagnostics after synchronous workspace retirement.
use super::{pike, Buffer, PrepareCause, PrepareError, Requirements};
use alloc::collections::TryReserveError;
use core::fmt;

#[derive(Debug)]
enum Cause {
    Reserve {
        buffer: Buffer,
        cause: TryReserveError,
    },
    Delegate {
        ordinal: usize,
        cause: pike::RetiredPrepareError,
    },
    Profile(pike::PlanError),
}
/// Actual owned failure after all outer, completed and partial delegate storage
/// has retired. This retains no source borrow and offers no retry or cache.
/// The caller's original account must surround retirement; no refund occurs here.
#[derive(Debug)]
pub struct RetiredPrepareError {
    requirements: Requirements,
    capacities: [usize; 5],
    completed_delegates: usize,
    cause: Cause,
}
impl PrepareError<'_> {
    /// Consume the partial owners under the source borrow, then return the cause.
    pub fn retire(self) -> RetiredPrepareError {
        let Self {
            source: _,
            requirements,
            partial,
            cause,
        } = self;
        let capacities = partial.capacities();
        let completed_delegates = partial.delegates.len();
        let cause = match cause {
            PrepareCause::Reserve { buffer, cause } => Cause::Reserve { buffer, cause },
            PrepareCause::Delegate { ordinal, cause } => Cause::Delegate {
                ordinal,
                cause: cause.retire(),
            },
            PrepareCause::Profile(cause) => Cause::Profile(cause),
        };
        drop(partial);
        RetiredPrepareError {
            requirements,
            capacities,
            completed_delegates,
            cause,
        }
    }
}
impl RetiredPrepareError {
    /// Original requested layouts; retirement does not alter this diagnostic.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Actual outer capacities which have already retired, in Buffer order.
    pub fn retired_capacities(&self) -> [usize; 5] {
        self.capacities
    }
    /// Number of completed delegate owners retired alongside the failure.
    pub fn retired_delegates(&self) -> usize {
        self.completed_delegates
    }
    /// Failed outer reserve target, when applicable.
    pub fn buffer(&self) -> Option<Buffer> {
        match self.cause {
            Cause::Reserve { buffer, .. } => Some(buffer),
            _ => None,
        }
    }
    /// Actual nested reserve failure and its delegate ordinal, after retirement.
    pub fn delegate_error(&self) -> Option<(usize, &pike::RetiredPrepareError)> {
        match &self.cause {
            Cause::Delegate { ordinal, cause } => Some((*ordinal, cause)),
            _ => None,
        }
    }
    /// The real reserve cause, with no reconstructed or boxed diagnostic.
    pub fn reserve_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Reserve { cause, .. } => Some(cause),
            Cause::Delegate { cause, .. } => Some(cause.cause()),
            Cause::Profile(_) => None,
        }
    }
}
impl fmt::Display for RetiredPrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Reserve { cause, .. } => cause.fmt(f),
            Cause::Delegate { cause, .. } => cause.fmt(f),
            Cause::Profile(cause) => cause.fmt(f),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for RetiredPrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Some(cause) = self.reserve_error() {
            return Some(cause);
        }
        match &self.cause {
            Cause::Profile(cause) => Some(cause),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Source;
    use alloc::string::ToString;
    #[test]
    fn outer_and_nested_real_errors_outlive_source_after_consuming_retirement() {
        let nested = [
            pike::Buffer::Epsilon,
            pike::Buffer::CurrentDense,
            pike::Buffer::CurrentSparse,
            pike::Buffer::NextDense,
            pike::Buffer::NextSparse,
            pike::Buffer::CurrentSlots,
            pike::Buffer::NextSlots,
        ];
        for outer in Buffer::ALL {
            let (error, capacities, diagnostic) = {
                let source = Source::new(r"(a+)(?![bc])|\s+").unwrap();
                let mut plan = source.plan().unwrap();
                plan.fail = Some(outer);
                let error = plan.prepare().unwrap_err();
                let capacities = error.capacities();
                let diagnostic = error.reserve_error().unwrap().to_string();
                (error.retire(), capacities, diagnostic)
            };
            assert_eq!(error.buffer(), Some(outer));
            assert_eq!(error.retired_capacities(), capacities);
            assert_eq!(error.retired_delegates(), 0);
            assert_eq!(error.reserve_error().unwrap().to_string(), diagnostic);
        }
        for target in nested {
            let (error, outer, prefix, diagnostic) = {
                let source = Source::new(r"(a+)(?![bc])|\s+").unwrap();
                let mut plan = source.plan().unwrap();
                plan.delegate_fail = Some((1, target));
                let error = plan.prepare().unwrap_err();
                assert_eq!(error.completed_delegates(), 1);
                let outer = error.capacities();
                let prefix = error.delegate_error().unwrap().1.capacities();
                let diagnostic = error.reserve_error().unwrap().to_string();
                (error.retire(), outer, prefix, diagnostic)
            };
            assert_eq!(error.retired_capacities(), outer);
            assert_eq!(error.retired_delegates(), 1);
            let (ordinal, nested) = error.delegate_error().unwrap();
            assert_eq!(ordinal, 1);
            assert_eq!(nested.buffer(), target);
            assert_eq!(nested.retired_capacities(), prefix);
            assert_eq!(error.reserve_error().unwrap().to_string(), diagnostic);
            assert!(core::ptr::eq(
                error.reserve_error().unwrap(),
                nested.cause()
            ));
        }
    }
}
