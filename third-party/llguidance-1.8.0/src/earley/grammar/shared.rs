//! Immutable compiled grammar ownership shared by every parser entry point.
use super::CGrammar;
use derivre::{ParserAllocationFailure, ParserStorageError};
use std::{
    alloc::Layout,
    error::Error,
    fmt,
    mem::{size_of, size_of_val},
    ops::Deref,
    sync::{Arc, atomic::AtomicUsize},
};

/// The compiled graph is immutable. Cloning this owner never copies its graph.
/// No raw Arc, weak owner or mutable compiled grammar escapes this boundary.
#[derive(Clone)]
pub struct SharedGrammar(Option<Arc<CGrammar>>);

#[derive(Debug)]
enum Cause {
    Funding(ParserAllocationFailure),
    Storage(ParserStorageError),
}

/// Failed publication retains the original compiled graph and its payer.
pub struct SharedGrammarFailure {
    cause: Cause,
    _grammar: CGrammar,
}
impl fmt::Debug for SharedGrammarFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedGrammarFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl fmt::Display for SharedGrammarFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Funding(cause) => fmt::Display::fmt(cause, f),
            Cause::Storage(cause) => fmt::Display::fmt(cause, f),
        }
    }
}
impl Error for SharedGrammarFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(match &self.cause {
            Cause::Funding(cause) => cause,
            Cause::Storage(cause) => cause,
        })
    }
}
impl SharedGrammar {
    /// Exact shared allocation, separate from the graph's original allocations.
    pub fn shell_bytes() -> Option<usize> {
        Some(
            Layout::new::<[AtomicUsize; 2]>()
                .extend(Layout::new::<CGrammar>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
        )
    }
    /// Fund the shared owner through the graph's actual compiler account before
    /// publication. Explicit unenforced compilation uses this same worker.
    pub fn new(grammar: CGrammar) -> Result<Self, SharedGrammarFailure> {
        let result = (|| {
            let parts = [
                Self::shell_bytes().ok_or_else(|| {
                    Cause::Storage(grammar.compilation_funding.storage_overflow())
                })?,
                size_of::<Self>(),
                size_of::<CGrammar>(),
                size_of::<Option<CGrammar>>(),
                size_of::<Arc<CGrammar>>(),
                size_of::<Option<Arc<CGrammar>>>(),
                size_of::<SharedGrammarFailure>(),
                size_of::<Cause>(),
                size_of::<Result<Self, SharedGrammarFailure>>(),
                size_of::<Result<(), Cause>>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| Cause::Storage(grammar.compilation_funding.storage_overflow()))?;
            grammar
                .compilation_funding
                .reserve(bytes)
                .map_err(Cause::Funding)
        })();
        match result {
            Ok(()) => Ok(Self(Some(Arc::new(grammar)))),
            Err(cause) => Err(SharedGrammarFailure { cause, _grammar: grammar }),
        }
    }
    /// Exact immutable allocation identity, independent of graph equality.
    pub fn same_grammar(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live grammar"),
            other.0.as_ref().expect("live grammar"),
        )
    }
    #[cfg(test)]
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(self.0.as_ref().expect("live grammar"))
    }
}
impl Deref for SharedGrammar {
    type Target = CGrammar;
    fn deref(&self) -> &CGrammar {
        self.0.as_deref().expect("live grammar")
    }
}
impl fmt::Debug for SharedGrammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedGrammar")
            .field("symbols", &self.symbols.len())
            .finish_non_exhaustive()
    }
}
impl Drop for SharedGrammar {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    #[derive(Debug)]
    struct Refused;
    impl fmt::Display for Refused {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("original compiler account refused")
        }
    }
    impl Error for Refused {}
    struct Retired(Arc<AtomicBool>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    fn compile() -> (
        CGrammar,
        derivre::ParserAllocationFunding,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
    ) {
        let refused = Arc::new(AtomicBool::new(false));
        let retired = Arc::new(AtomicBool::new(false));
        let funding = derivre::ParserAllocationFunding::prepare({
            let refused = refused.clone();
            let guard = Retired(retired.clone());
            move |_| {
                let _ = &guard;
                if refused.load(Ordering::SeqCst) {
                    Err(Refused)
                } else {
                    Ok(())
                }
            }
        })
        .unwrap();
        let limits = crate::api::ParserLimits::default();
        let mut builder =
            crate::grammar_builder::GrammarBuilder::new(None, limits.clone(), funding.clone())
                .unwrap();
        builder
            .add_grammar(
                crate::api::LLGuidanceOptions::default(),
                derivre::RegexAst::NoMatch,
            )
            .unwrap();
        let word = builder.string("value:23\n").unwrap();
        builder.set_start_node(word).unwrap();
        let grammar = builder
            .grammar
            .compile(builder.regex.spec, &limits, funding.clone())
            .unwrap();
        (grammar, funding, refused, retired)
    }
    #[test]
    fn shared_graph_aliases_retain_the_original_compiler_account() {
        let (grammar, funding, _, retired) = compile();
        let shared = SharedGrammar::new(grammar).unwrap();
        let alias = shared.clone();
        assert!(shared.same_grammar(&alias));
        assert!(std::ptr::eq(&*shared, &*alias));
        assert!(shared.retained_capacity_bytes(&derivre::ParserAllocationFunding::unenforced()).unwrap() > 0);
        drop((shared, funding));
        assert!(!retired.load(Ordering::SeqCst));
        let last = alias.clone();
        std::thread::scope(|threads| {
            threads.spawn(move || drop(alias));
            threads.spawn(move || drop(last));
        });
        assert!(retired.load(Ordering::SeqCst));
    }
    #[test]
    fn publication_refusal_keeps_the_compiled_graph_and_actual_error() {
        let (grammar, funding, refused, retired) = compile();
        refused.store(true, Ordering::SeqCst);
        let failure = SharedGrammar::new(grammar).unwrap_err();
        let mut cause: &(dyn Error + 'static) = &failure;
        while !cause.is::<Refused>() {
            cause = cause.source().expect("original typed refusal");
        }
        assert!(failure._grammar.retained_capacity_bytes(&derivre::ParserAllocationFunding::unenforced()).unwrap() > 0);
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }
}
