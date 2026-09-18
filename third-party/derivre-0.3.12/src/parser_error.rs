//! Closed construction errors. Reporting a refused producer never allocates.
use crate::{ParserAllocationFailure, ParserAllocationFunding, ParserStorageError};
use std::{error::Error, fmt};

pub type ParserResult<T, E = ParserError> = Result<T, E>;

pub struct ParserError {
    kind: Kind,
}
enum Kind {
    Funding(ParserAllocationFailure),
    Storage(ParserStorageError),
    Expression(crate::raw::PreparedExprError),
    Message {
        text: String,
        funding: ParserAllocationFunding,
    },
    Cause {
        cause: Box<dyn Error + Send + Sync>,
        funding: ParserAllocationFunding,
    },
    Context {
        text: String,
        source: Box<ParserError>,
        funding: ParserAllocationFunding,
    },
}
impl ParserError {
    pub fn message(args: fmt::Arguments<'_>, funding: &ParserAllocationFunding) -> Self {
        match funding.try_format(args) {
            Ok(text) => Self {
                kind: Kind::Message {
                    text,
                    funding: funding.clone(),
                },
            },
            Err(error) => error.into(),
        }
    }
    pub fn cause<E: Error + Send + Sync + 'static>(
        cause: E,
        funding: &ParserAllocationFunding,
    ) -> Self {
        // A dependency may use a fixed marker. Restore the first original
        // refusal before attempting any diagnostic storage.
        if let Some(error) = funding.failure() {
            return error.into();
        }
        match funding.try_box(cause) {
            Ok(cause) => Self {
                kind: Kind::Cause {
                    cause,
                    funding: funding.clone(),
                },
            },
            Err(error) => error.into(),
        }
    }
    pub fn context(self, context: impl fmt::Display, funding: &ParserAllocationFunding) -> Self {
        if self.is_storage_failure() {
            return self;
        }
        let text = match funding.try_format(format_args!("{context}")) {
            Ok(text) => text,
            Err(error) => return error.into(),
        };
        self.retain_context(text, funding)
    }
    /// Formats a source annotation without erasing the original typed cause.
    /// Refused producers never enter either formatting pass.
    pub fn annotate(
        self,
        prefix: impl fmt::Display,
        suffix: impl fmt::Display,
        funding: &ParserAllocationFunding,
    ) -> Self {
        if self.is_storage_failure() {
            return self;
        }
        let text = match funding.try_format(format_args!("{prefix}{self}{suffix}")) {
            Ok(text) => text,
            Err(error) => return error.into(),
        };
        self.retain_context(text, funding)
    }
    fn retain_context(self, text: String, funding: &ParserAllocationFunding) -> Self {
        match funding.try_box(self) {
            Ok(source) => Self {
                kind: Kind::Context {
                    text,
                    source,
                    funding: funding.clone(),
                },
            },
            Err(error) => error.into(),
        }
    }
    pub fn chain(&self) -> Chain<'_> {
        Chain(Some(self))
    }
    pub fn downcast_ref<E: Error + 'static>(&self) -> Option<&E> {
        self.chain().find_map(|cause| cause.downcast_ref::<E>())
    }
    pub fn is_storage_failure(&self) -> bool {
        crate::is_parser_storage_failure(self)
    }
}
impl fmt::Display for ParserError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            Kind::Funding(error) => error.fmt(output),
            Kind::Storage(error) => error.fmt(output),
            Kind::Expression(error) => error.fmt(output),
            Kind::Message { text, .. } | Kind::Context { text, .. } => output.write_str(text),
            Kind::Cause { cause, .. } => cause.fmt(output),
        }
    }
}
impl fmt::Debug for ParserError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, output)
    }
}
impl Error for ParserError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            Kind::Funding(error) => Some(error),
            Kind::Storage(error) => Some(error),
            Kind::Expression(error) => Some(error),
            Kind::Cause { cause, .. } => Some(cause.as_ref()),
            Kind::Context { source, .. } => Some(source.as_ref()),
            Kind::Message { .. } => None,
        }
    }
}
impl From<ParserAllocationFailure> for ParserError {
    fn from(error: ParserAllocationFailure) -> Self {
        Self {
            kind: Kind::Funding(error),
        }
    }
}
impl From<ParserStorageError> for ParserError {
    fn from(error: ParserStorageError) -> Self {
        Self {
            kind: Kind::Storage(error),
        }
    }
}
impl From<crate::raw::PreparedExprError> for ParserError {
    fn from(error: crate::raw::PreparedExprError) -> Self {
        Self {
            kind: Kind::Expression(error),
        }
    }
}
pub struct Chain<'a>(Option<&'a (dyn Error + 'static)>);
impl<'a> Iterator for Chain<'a> {
    type Item = &'a (dyn Error + 'static);
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.0.take()?;
        self.0 = current.source();
        Some(current)
    }
}

#[macro_export]
macro_rules! parser_bail {
    ($funding:expr, $($args:tt)*) => { return Err($crate::parser_error!($funding, $($args)*)) };
}
#[macro_export]
macro_rules! parser_ensure {
    ($funding:expr, $condition:expr, $($args:tt)*) => { if !$condition { $crate::parser_bail!($funding, $($args)*); } };
}

impl From<crate::raw::ExprEncodingError> for ParserError {
    fn from(error: crate::raw::ExprEncodingError) -> Self {
        crate::raw::PreparedExprError::Encoding(error).into()
    }
}
impl From<crate::raw::HashConsCapacityError> for ParserError {
    fn from(error: crate::raw::HashConsCapacityError) -> Self {
        crate::raw::ExprEncodingError::Storage(error).into()
    }
}

#[macro_export]
macro_rules! parser_error {
    ($funding:expr, $format:literal $(, $args:expr)* $(,)?) => { $crate::ParserError::message(format_args!($format $(, $args)*), $funding) };
    ($funding:expr, $message:expr $(,)?) => { $crate::ParserError::message(format_args!("{}", $message), $funding) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    #[derive(Debug)]
    struct Refused;
    impl fmt::Display for Refused {
        fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
            output.write_str("refused")
        }
    }
    impl Error for Refused {}

    #[derive(Debug)]
    struct Source {
        payer_retired: Arc<AtomicBool>,
        source_retired: Arc<AtomicBool>,
    }
    impl fmt::Display for Source {
        fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
            output.write_str("syntax")
        }
    }
    impl Error for Source {}
    impl Drop for Source {
        fn drop(&mut self) {
            assert!(!self.payer_retired.load(Ordering::SeqCst));
            self.source_retired.store(true, Ordering::SeqCst);
        }
    }
    struct Payer(Arc<AtomicBool>);
    impl Drop for Payer {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn annotated_diagnostic_retains_typed_source_until_before_payer_retirement() {
        let payer_retired = Arc::new(AtomicBool::new(false));
        let source_retired = Arc::new(AtomicBool::new(false));
        let payer = Payer(payer_retired.clone());
        let funding = ParserAllocationFunding::prepare(move |_| {
            let _ = &payer;
            Ok::<_, Refused>(())
        })
        .unwrap();
        let error = ParserError::cause(
            Source {
                payer_retired: payer_retired.clone(),
                source_retired: source_retired.clone(),
            },
            &funding,
        )
        .annotate("at 2(3): ", "\nsource line", &funding);
        assert_eq!(error.to_string(), "at 2(3): syntax\nsource line");
        assert!(error.downcast_ref::<Source>().is_some());
        drop(funding);
        assert!(!payer_retired.load(Ordering::SeqCst));
        assert!(!source_retired.load(Ordering::SeqCst));
        drop(error);
        assert!(source_retired.load(Ordering::SeqCst));
        assert!(payer_retired.load(Ordering::SeqCst));
    }

    #[test]
    fn refused_diagnostic_growth_keeps_original_cause_and_never_formats_again() {
        struct MustNotFormat;
        impl fmt::Display for MustNotFormat {
            fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
                panic!("refused diagnostic formatted")
            }
        }
        let refuse = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let funding = ParserAllocationFunding::prepare({
            let refuse = refuse.clone();
            let calls = calls.clone();
            move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                if refuse.load(Ordering::SeqCst) {
                    Err(Refused)
                } else {
                    Ok(())
                }
            }
        })
        .unwrap();
        refuse.store(true, Ordering::SeqCst);
        let error = ParserError::message(format_args!("{}", MustNotFormat), &funding);
        let before = calls.load(Ordering::SeqCst);
        let error = error
            .annotate(MustNotFormat, MustNotFormat, &funding)
            .context(MustNotFormat, &funding);
        assert_eq!(calls.load(Ordering::SeqCst), before);
        drop(funding);
        assert!(error.downcast_ref::<Refused>().is_some());
        assert_eq!(error.to_string(), "refused");
    }
}
