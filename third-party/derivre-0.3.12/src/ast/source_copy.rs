//! Source-qualified expression copy; subsequent derivative growth is separate.
use super::{
    storage::{ExpressionStorage, PreparedExprSet},
    ExprRef, ExprSet, UnicodeCache,
};
use crate::{
    hashcons::{HashConsCopyFailure, HashConsCopyPlan, HashConsPreparedSourcePlan},
    pp::{PrettyPrinter, PrinterCopyFailure, PrinterCopyPlan},
    RandomState,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    hash::BuildHasher,
    mem::{size_of, size_of_val},
};

/// Checked actual source population, including complete map allocations and keys.
#[derive(Clone, Copy, Debug)]
pub struct ExprSetCopyRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl ExprSetCopyRequirements {
    /// Actual independently copied buffers; no allowance for future growth.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed planning, copying, hashing and failure representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local constructor requirement.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Loan of the actual expression source and its existing child constructors.
pub struct ExprSetCopyPlan<'a> {
    source: &'a ExprSet,
    exprs: HashConsCopyPlan<'a>,
    printer: PrinterCopyPlan<'a>,
    requirements: ExprSetCopyRequirements,
    key_words: usize,
}
impl fmt::Debug for ExprSetCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExprSetCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
struct Partial {
    exprs: Option<ExpressionStorage>,
    weights: Vec<(u32, u32)>,
    printer: Option<PrettyPrinter>,
    unicode: UnicodeCache,
    pending_key: Vec<(char, char)>,
}
#[derive(Debug)]
enum Cause {
    Source,
    Overflow,
    Capacity,
    Vector(TryReserveError),
    Table(hashbrown::TryReserveError),
    HashCons(HashConsCopyFailure),
    Printer(PrinterCopyFailure),
}
/// Owns all completed and failed child allocations until actual retirement.
pub struct ExprSetCopyFailure {
    cause: Cause,
    partial: Option<Partial>,
}
impl fmt::Debug for ExprSetCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExprSetCopyFailure")
            .field("cause", &self.cause)
            .field("retains_prefix", &self.partial.is_some())
            .finish()
    }
}
impl fmt::Display for ExprSetCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source => f.write_str("expression source violates its arena provenance"),
            Cause::Overflow => f.write_str("expression copy geometry overflow"),
            Cause::Capacity => f.write_str("expression copy capacity differs from source"),
            Cause::Vector(e) => fmt::Display::fmt(e, f),
            Cause::Table(e) => fmt::Display::fmt(e, f),
            Cause::HashCons(e) => fmt::Display::fmt(e, f),
            Cause::Printer(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for ExprSetCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Vector(e) => Some(e),
            Cause::Table(e) => Some(e),
            Cause::HashCons(e) => Some(e),
            Cause::Printer(e) => Some(e),
            _ => None,
        }
    }
}
fn bytes<T>(len: usize) -> Result<usize, Cause> {
    Layout::array::<T>(len)
        .map(|l| l.size())
        .map_err(|_| Cause::Overflow)
}
impl ExprSet {
    /// Actual retained source buffers. This is inspection, not a construction
    /// bound or permission to allocate a copy.
    pub fn retained_capacity_bytes(&self) -> Option<usize> {
        let mut bytes = self.exprs.retained_capacity_bytes().ok()?
            .checked_add(self.pp.retained_capacity_bytes())?
            .checked_add(self.unicode_cache.allocation_size())?
            .checked_add(self.expr_weight.capacity().checked_mul(size_of::<(u32, u32)>())?)?;
        for key in self.unicode_cache.keys() {
            bytes = bytes.checked_add(key.capacity().checked_mul(size_of::<(char, char)>())?)?;
        }
        Some(bytes)
    }

    /// Complete existing fixed source-copy and prepared-table inspection frames,
    /// without walking the source or allocating a destination.
    pub fn prepared_source_inspection_control_bytes() -> Option<usize> {
        ExprSetPreparedSourcePlan::control_bytes()
    }
    /// Prepares a copy of actual IDs, weights, Unicode memo keys, printer mapping
    /// and scalar configuration. Does not compile or derive new expressions.
    pub fn source_copy_plan(&self) -> Result<ExprSetCopyPlan<'_>, ExprSetCopyFailure> {
        ExprSetCopyPlan::prepare(self).map_err(|cause| ExprSetCopyFailure {
            cause,
            partial: None,
        })
    }

    /// Quotes a complete independent source with the same metadata and a finite
    /// hash-cons destination from the source's real table/backing layout.
    /// This does not quote future simplifier scratch, weights, or parser work.
    pub fn prepared_source_plan(
        &self,
    ) -> Result<ExprSetPreparedSourcePlan<'_>, ExprSetCopyFailure> {
        ExprSetPreparedSourcePlan::prepare(self).map_err(|cause| ExprSetCopyFailure {
            cause,
            partial: None,
        })
    }
}

/// Complete prepared expression-source constructor requirement.
#[derive(Clone, Copy, Debug)]
pub struct ExprSetPreparedSourceRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl ExprSetPreparedSourceRequirements {
    /// Actual independently constructed source, finite table and encoding buffers.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed source/copy/completed-owner and typed failure representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local constructor requirement, without future execution work.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Loan of one expression source and the same source's finite table destination.
pub struct ExprSetPreparedSourcePlan<'a> {
    copy: ExprSetCopyPlan<'a>,
    table: HashConsPreparedSourcePlan<'a>,
    requirements: ExprSetPreparedSourceRequirements,
}
impl fmt::Debug for ExprSetPreparedSourcePlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExprSetPreparedSourcePlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
impl<'a> ExprSetPreparedSourcePlan<'a> {
    fn wrapper_controls() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<ExprSetPreparedSourceRequirements>(),
            size_of::<PreparedExprSet>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, ExprSetCopyFailure>>(),
            size_of::<Result<PreparedExprSet, ExprSetCopyFailure>>(),
            size_of::<HashConsPreparedSourcePlan<'a>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    fn control_bytes() -> Option<usize> {
        Self::wrapper_controls()?
            .checked_add(
                ExprSetCopyPlan::control_bytes()?.checked_sub(HashConsCopyPlan::control_bytes()?)?,
            )
            .and_then(|n| n.checked_add(HashConsPreparedSourcePlan::inspection_control_bytes()?))
    }
    fn prepare(source: &'a ExprSet) -> Result<Self, Cause> {
        if source.alphabet_size == 0
            || source.alphabet_size > 256
            || source.alphabet_words != source.alphabet_size.div_ceil(32)
            || source.len() <= ExprRef::NON_EMPTY_BYTE_STRING.as_usize()
            || !matches!(source.get(ExprRef::EMPTY_STRING), super::Expr::EmptyString)
            || !matches!(source.get(ExprRef::NO_MATCH), super::Expr::NoMatch)
            || !matches!(source.get(ExprRef::ANY_BYTE), super::Expr::ByteSet(bytes) if bytes.len() == source.alphabet_words)
            || !matches!(
                source.get(ExprRef::ANY_BYTE_STRING),
                super::Expr::Repeat(_, ExprRef::ANY_BYTE, 0, u32::MAX)
            )
            || !matches!(
                source.get(ExprRef::NON_EMPTY_BYTE_STRING),
                super::Expr::Repeat(_, ExprRef::ANY_BYTE, 1, u32::MAX)
            )
        {
            return Err(Cause::Source);
        }
        for i in 1..source.len() {
            let id = ExprRef::new(u32::try_from(i).map_err(|_| Cause::Overflow)?);
            if source
                .get_args(id)
                .iter()
                .any(|child| !child.is_valid() || child.as_usize() >= i)
            {
                return Err(Cause::Source);
            }
        }
        let copy = ExprSetCopyPlan::prepare(source)?;
        let table = source
            .exprs
            .prepared_source_plan()
            .map_err(Cause::HashCons)?;
        let original = copy.exprs.requirements();
        let prepared = table.requirements();
        let buffers = copy
            .requirements
            .buffers
            .checked_sub(original.buffer_bytes())
            .and_then(|n| n.checked_add(prepared.buffer_bytes()))
            .ok_or(Cause::Overflow)?;
        let controls = Self::wrapper_controls()
            .and_then(|n| {
                n.checked_add(
                    copy.requirements
                        .controls
                        .checked_sub(original.control_bytes())?,
                )
            })
            .and_then(|n| n.checked_add(prepared.control_bytes()))
            .ok_or(Cause::Overflow)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        Ok(Self {
            copy,
            table,
            requirements: ExprSetPreparedSourceRequirements {
                buffers,
                controls,
                total,
            },
        })
    }
    /// Requirements for this exact source and its complete independent owner.
    pub fn requirements(&self) -> ExprSetPreparedSourceRequirements {
        self.requirements
    }
    /// Constructs the table once and consumes the same ordinary metadata-copy
    /// worker. A later failure retains the completed finite table in its prefix.
    pub fn compile(self) -> Result<PreparedExprSet, ExprSetCopyFailure> {
        let table = self.table;
        let inner = self
            .copy
            .compile_with_storage(|_| table.compile().map(ExpressionStorage))?;
        Ok(PreparedExprSet { inner })
    }
}
impl<'a> ExprSetCopyPlan<'a> {
    fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<ExprSetCopyRequirements>(),
            size_of::<ExprSetCopyFailure>(),
            size_of::<Partial>(),
            size_of::<Cause>(),
            size_of::<ExprSet>(),
            size_of::<ExpressionStorage>(),
            size_of::<Result<ExpressionStorage, HashConsCopyFailure>>(),
            size_of::<Option<Partial>>(),
            size_of::<UnicodeCache>(),
            size_of::<Vec<(char, char)>>(),
            size_of::<Vec<(u32, u32)>>(),
            size_of::<RandomState>(),
            size_of::<<RandomState as BuildHasher>::Hasher>(),
            size_of::<hashbrown::hash_map::Iter<'_, Vec<(char, char)>, ExprRef>>(),
            size_of::<hashbrown::hash_map::Keys<'_, Vec<(char, char)>, ExprRef>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, ExprSetCopyFailure>>(),
            size_of::<Result<ExprSet, ExprSetCopyFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<(), hashbrown::TryReserveError>>(),
            size_of::<Option<ExprRef>>(),
            size_of::<(&ExprSet, &mut Partial)>(),
            size_of::<Result<usize, Cause>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| n.checked_add(HashConsCopyPlan::control_bytes()?))
            .and_then(|n| n.checked_add(PrinterCopyPlan::control_bytes()?))
    }
    fn prepare(source: &'a ExprSet) -> Result<Self, Cause> {
        let exprs = source.exprs.source_copy_plan().map_err(Cause::HashCons)?;
        let printer = source.pp.source_copy_plan().ok_or(Cause::Overflow)?;
        let mut buffers = exprs
            .requirements()
            .buffer_bytes()
            .checked_add(printer.buffer_bytes())
            .and_then(|n| n.checked_add(source.unicode_cache.allocation_size()))
            .ok_or(Cause::Overflow)?;
        buffers = buffers
            .checked_add(bytes::<(u32, u32)>(source.expr_weight.len())?)
            .ok_or(Cause::Overflow)?;
        let mut key_words = 0usize;
        for key in source.unicode_cache.keys() {
            key_words = key_words.checked_add(key.len()).ok_or(Cause::Overflow)?;
            buffers = buffers
                .checked_add(bytes::<(char, char)>(key.len())?)
                .ok_or(Cause::Overflow)?;
        }
        let controls = Self::control_bytes().ok_or(Cause::Overflow)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        Ok(Self {
            source,
            exprs,
            printer,
            key_words,
            requirements: ExprSetCopyRequirements {
                buffers,
                controls,
                total,
            },
        })
    }
    /// Exact local source-copy population.
    pub fn requirements(&self) -> ExprSetCopyRequirements {
        self.requirements
    }
    /// Runs the same fallible constructor used by ordinary Clone.
    pub fn compile(self) -> Result<ExprSet, ExprSetCopyFailure> {
        self.compile_with_funding(crate::ParserAllocationFunding::unenforced())
    }
    pub(crate) fn compile_with_funding(self, funding: crate::ParserAllocationFunding) -> Result<ExprSet, ExprSetCopyFailure> {
        let words = self.source.exprs.0.max_words();
        let encoding = self.source.exprs.0.max_encoded_words();
        self.compile_with_storage(|plan| plan.compile().map(|source| ExpressionStorage::copied(source, words, encoding, funding)))
    }
    fn compile_with_storage(
        self,
        construct: impl FnOnce(HashConsCopyPlan<'a>) -> Result<ExpressionStorage, HashConsCopyFailure>,
    ) -> Result<ExprSet, ExprSetCopyFailure> {
        let source = self.source;
        let mut partial = Partial {
            unicode: UnicodeCache::with_hasher(source.unicode_cache.hasher().clone()),
            exprs: None,
            weights: Vec::new(),
            printer: None,
            pending_key: Vec::new(),
        };
        let mut key_words = self.key_words;
        let result = (|| -> Result<(), Cause> {
            partial.exprs = Some(construct(self.exprs).map_err(Cause::HashCons)?);
            partial
                .weights
                .try_reserve_exact(source.expr_weight.len())
                .map_err(Cause::Vector)?;
            if partial.weights.capacity() != source.expr_weight.len() {
                return Err(Cause::Capacity);
            }
            partial.weights.extend_from_slice(&source.expr_weight);
            partial.printer = Some(self.printer.compile().map_err(Cause::Printer)?);
            partial
                .unicode
                .try_reserve(source.unicode_cache.capacity())
                .map_err(Cause::Table)?;
            if partial.unicode.capacity() != source.unicode_cache.capacity()
                || partial.unicode.allocation_size() != source.unicode_cache.allocation_size()
            {
                return Err(Cause::Capacity);
            }
            for (key, value) in &source.unicode_cache {
                key_words = key_words.checked_sub(key.len()).ok_or(Cause::Capacity)?;
                partial
                    .pending_key
                    .try_reserve_exact(key.len())
                    .map_err(Cause::Vector)?;
                if partial.pending_key.capacity() != key.len() {
                    return Err(Cause::Capacity);
                }
                partial.pending_key.extend_from_slice(key);
                if partial.unicode.len() == partial.unicode.capacity() {
                    return Err(Cause::Capacity);
                }
                let key = std::mem::take(&mut partial.pending_key);
                if partial.unicode.insert(key, *value).is_some() {
                    return Err(Cause::Capacity);
                }
            }
            if key_words != 0 {
                return Err(Cause::Capacity);
            }
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(ExprSetCopyFailure {
                cause,
                partial: Some(partial),
            });
        }
        Ok(ExprSet {
            exprs: partial.exprs.take().expect("completed expression source"),
            expr_weight: partial.weights,
            alphabet_size: source.alphabet_size,
            alphabet_words: source.alphabet_words,
            digits: source.digits,
            digit_dot: source.digit_dot,
            cost: source.cost,
            pp: partial.printer.take().expect("completed printer source"),
            optimize: source.optimize,
            unicode_cache: partial.unicode,
            any_unicode_star: source.any_unicode_star,
            any_unicode: source.any_unicode,
            any_unicode_non_nl: source.any_unicode_non_nl,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RegexBuilder;
    #[test]
    fn prepared_expression_copy_keeps_finite_table_on_late_metadata_refusal() {
        let mut builder = RegexBuilder::new(crate::ParserAllocationFunding::unenforced()).unwrap();
        builder.mk_regex("[α-ω]+").unwrap();
        builder.mk_regex("[Ж-Я]+").unwrap();
        let source = builder.into_exprset();
        assert!(source.unicode_cache.len() >= 2);
        let plan = source.prepared_source_plan().unwrap();
        let requirements = plan.requirements();
        let prepared = plan.compile().unwrap();
        let copy = prepared.source();
        let actual = copy.exprs.retained_capacity_bytes().unwrap()
            + copy.expr_weight.capacity() * size_of::<(u32, u32)>()
            + copy.pp.source_copy_plan().unwrap().buffer_bytes()
            + copy.unicode_cache.allocation_size()
            + copy
                .unicode_cache
                .keys()
                .map(|key| key.capacity() * size_of::<(char, char)>())
                .sum::<usize>();
        assert_eq!(actual, requirements.buffer_bytes());
        assert_eq!(copy.digits, source.digits);
        assert_eq!(copy.expr_weight, source.expr_weight);
        assert_eq!(copy.unicode_cache, source.unicode_cache);
        let mut refused = source.prepared_source_plan().unwrap();
        refused.copy.key_words = source.unicode_cache.keys().next().unwrap().len();
        let failure = match refused.compile() {
            Err(error) => error,
            Ok(_) => panic!("incomplete source key population was accepted"),
        };
        assert!(matches!(failure.cause, Cause::Capacity));
        let entries = source.len();
        drop(source);
        drop(prepared);
        let partial = failure.partial.as_ref().unwrap();
        assert!(matches!(
            partial.exprs,
            Some(ExpressionStorage(_))
        ));
        assert_eq!(partial.exprs.as_ref().unwrap().len(), entries);
        assert_eq!(partial.unicode.len(), 1);
        assert!(partial.printer.is_some());
    }
    #[test]
    fn expression_source_copy_retains_ids_mapping_and_partial_unicode_key_population() {
        let mut builder = RegexBuilder::new(crate::ParserAllocationFunding::unenforced()).unwrap();
        builder.mk_regex("[α-ω]+").unwrap();
        builder.mk_regex("[Ж-Я]+").unwrap();
        let source = builder.into_exprset();
        assert!(source.unicode_cache.len() >= 2);
        let plan = source.source_copy_plan().unwrap();
        let quote = plan.requirements();
        let copy = plan.compile().unwrap();
        let mut actual = copy.exprs.retained_capacity_bytes().unwrap()
            + copy.expr_weight.capacity() * size_of::<(u32, u32)>()
            + copy.pp.source_copy_plan().unwrap().buffer_bytes()
            + copy.unicode_cache.allocation_size();
        for key in copy.unicode_cache.keys() {
            actual += key.capacity() * size_of::<(char, char)>();
        }
        assert_eq!(actual, quote.buffer_bytes());
        assert_eq!(copy.expr_weight, source.expr_weight);
        assert_eq!(copy.unicode_cache, source.unicode_cache);
        assert_eq!(copy.digits, source.digits);
        assert_eq!(copy.cost, source.cost);
        assert_eq!(copy.optimize, source.optimize);
        for id in 0..source.exprs.len() {
            assert_eq!(copy.exprs.get(id as u32), source.exprs.get(id as u32));
        }
        let mut failing = source.source_copy_plan().unwrap();
        failing.key_words = source.unicode_cache.keys().next().unwrap().len();
        let failure = match failing.compile() {
            Err(e) => e,
            Ok(_) => panic!("incomplete key population accepted"),
        };
        assert!(matches!(failure.cause, Cause::Capacity));
        assert_eq!(failure.partial.as_ref().unwrap().unicode.len(), 1);
        let expected_entries = source.exprs.len();
        drop(copy);
        drop(source);
        let partial = failure.partial.as_ref().unwrap();
        assert_eq!(partial.exprs.as_ref().unwrap().len(), expected_entries);
        assert!(partial.printer.is_some());
        assert!(partial.unicode.keys().next().unwrap().capacity() > 0);
        assert!(partial.pending_key.is_empty());
    }
}
