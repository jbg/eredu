use alloc::string::String;

use regex_automata::meta;

/// An error that occurred during parsing or compiling a regular expression.
#[non_exhaustive]
#[derive(Clone, PartialEq)]
pub enum Error {
    /// A syntax error.
    Syntax(String),
    /// The compiled program exceeded the set size
    /// limit. The argument is the size limit imposed by
    /// [`RegexBuilder::size_limit`](crate::RegexBuilder::size_limit). Even
    /// when not configured explicitly, it defaults to a reasonable limit.
    ///
    /// If you're getting this error, it occurred because your regex has been
    /// compiled to an intermediate state that is too big. It is important to
    /// note that exceeding this limit does _not_ mean the regex is too big to
    /// _work_, but rather, the regex is big enough that it may wind up being
    /// surprisingly slow when used in a search. In other words, this error is
    /// meant to be a practical heuristic for avoiding a performance footgun,
    /// and especially so for the case where the regex pattern is coming from
    /// an untrusted source.
    ///
    /// There are generally two ways to move forward if you hit this error.
    /// The first is to find some way to use a smaller regex. The second is to
    /// increase the size limit via `RegexBuilder::size_limit`. However, if
    /// your regex pattern is not from a trusted source, then neither of these
    /// approaches may be appropriate. Instead, you'll have to determine just
    /// how big of a regex you want to allow.
    CompiledTooBig(usize),
    /// Original source storage admission or host allocation failure.
    Allocation(crate::allocation::AllocationError),
}

impl Error {
    pub(crate) fn from_meta_build_error_with_allocations(
        err: meta::BuildError,
        policy: &dyn crate::allocation::Allocation,
    ) -> Error {
        if let Some(error) = err.allocation_error() {
            return Error::Allocation(error);
        }
        let allocator = crate::allocation::Allocator::new(policy);
        if let Some(size_limit) = err.size_limit() {
            Error::CompiledTooBig(size_limit)
        } else if let Some(ref err) = err.syntax_error() {
            match allocator.format(format_args!("{err}")) {
                Ok(text) => Error::Syntax(text),
                Err(error) => Error::Allocation(error),
            }
        } else {
            // This is a little suspect. Technically there are more ways for
            // a meta regex to fail to build other than "exceeded size limit"
            // and "syntax error." For example, if there are too many states
            // or even too many patterns. But in practice this is probably
            // good enough. The worst thing that happens is that Error::Syntax
            // represents an error that isn't technically a syntax error, but
            // the actual message will still be shown. So... it's not too bad.
            //
            // We really should have made the Error type in the regex crate
            // completely opaque. Rookie mistake.
            match allocator.format(format_args!("{err}")) {
                Ok(text) => Error::Syntax(text),
                Err(error) => Error::Allocation(error),
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            _ => None,
        }
    }
    // TODO: Remove this method entirely on the next breaking semver release.
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match *self {
            Error::Syntax(ref err) => err,
            Error::CompiledTooBig(_) => "compiled program too big",
            Error::Allocation(_) => "regex source allocation refused",
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Error::Syntax(ref err) => err.fmt(f),
            Error::Allocation(ref err) => err.fmt(f),
            Error::CompiledTooBig(limit) => {
                write!(f, "Compiled regex exceeds size limit of {limit} bytes.",)
            }
        }
    }
}

// We implement our own Debug implementation so that we show nicer syntax
// errors when people use `Regex::new(...).unwrap()`. It's a little weird,
// but the `Syntax` variant is already storing a `String` anyway, so we might
// as well format it nicely.
impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Error::Syntax(ref err) => {
                let hr = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
                writeln!(f, "Syntax(")?;
                writeln!(f, "{hr}")?;
                writeln!(f, "{err}")?;
                writeln!(f, "{hr}")?;
                write!(f, ")")?;
                Ok(())
            }
            Error::Allocation(ref error) => f.debug_tuple("Allocation").field(error).finish(),
            Error::CompiledTooBig(limit) => f.debug_tuple("CompiledTooBig").field(&limit).finish(),
        }
    }
}

impl From<crate::allocation::AllocationError> for Error {
    fn from(error: crate::allocation::AllocationError) -> Self {
        Self::Allocation(error)
    }
}
