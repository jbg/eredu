//! Paid actual Earley scratch and initial prediction destinations.
mod advance;
mod agenda;
mod bias;
mod capture;
mod copy;
mod force;
mod row;
mod scan;
mod session;
pub use session::{PreparedTokenParser, PreparedTokenParserError};
mod speculation;
mod token;
mod validate;
use super::{seed_predictions, CGrammar, SharedGrammar, GrammarStackNode, Item, LexemeSet, ParamValue, Scratch};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};
use toktrie::{
    SimpleVob, TokenMaskConstructionFailure, TokenMaskConstructionPlan, TokenMaskSourceError,
};
#[derive(Debug)]
enum Cause<E> {
    Overflow,
    Capacity,
    Source,
    Token(token::TokenApplicationFailure),
    Session(session::Failure),
    Step { limit: usize, actual: usize },
    DecodeSource(toktrie::RawTokenDecodeError),
    Decode(toktrie::RawTokenDecodeFailure),
    Lexer(crate::earley::PreparedLexerOperationError<E>),
    Funding(E),
    Allocation(TryReserveError),
    MaskSource(TokenMaskSourceError),
    Mask(TokenMaskConstructionFailure),
}
/// A failed seed retains masks, scratch, prediction buffers and the immutable
/// declaration. Its enclosing original owner retains the funding authority.
pub struct PreparedEarleySeedError<E> {
    cause: Cause<E>,
    pub(super) prefix: PreparedEarleySeed,
}
impl<E: fmt::Debug> fmt::Debug for PreparedEarleySeedError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedEarleySeedError")
            .field("cause", &self.cause)
            .field("prefix", &self.prefix)
            .finish()
    }
}
impl<E: fmt::Display> fmt::Display for PreparedEarleySeedError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Token(error) => fmt::Display::fmt(error, f),
            Cause::Session(error) => fmt::Display::fmt(error, f),
            Cause::Step { limit, actual } => {
                write!(f, "Earley step created {actual} items; limit is {limit}")
            }
            Cause::Overflow => f.write_str("Earley seed destination geometry overflow"),
            Cause::Source => f.write_str("Earley initial agenda source is unavailable"),
            Cause::Capacity => f.write_str("Earley seed destination exceeded exact capacity"),
            Cause::Funding(e) => fmt::Display::fmt(e, f),
            Cause::DecodeSource(e) => fmt::Display::fmt(e, f),
            Cause::Decode(e) => fmt::Display::fmt(e, f),
            Cause::Lexer(e) => fmt::Display::fmt(e, f),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
            Cause::MaskSource(e) => fmt::Display::fmt(e, f),
            Cause::Mask(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedEarleySeedError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Token(error) => Some(error),
            Cause::Session(error) => Some(error),
            Cause::Funding(e) => Some(e),
            Cause::DecodeSource(e) => Some(e),
            Cause::Decode(e) => Some(e),
            Cause::Lexer(e) => Some(e),
            Cause::Allocation(e) => Some(e),
            Cause::MaskSource(e) => Some(e),
            Cause::Mask(e) => Some(e),
            _ => None,
        }
    }
}
/// The real Scratch representation after root-stack and start-rule predictions.
/// It can close the initial agenda and lexical row, but is not a runnable Parser.
pub struct PreparedEarleySeed {
    pub(super) scratch: Option<Scratch>,
    lexemes: Option<LexemeSet>,
    grammars: Option<SimpleVob>,
    pub(super) rows: Vec<super::Row>,
    pub(super) row_infos: Vec<super::RowInfo>,
    pub(super) lexer_stack: Vec<super::LexerState>,
    pub(super) initial_selection: Option<LexemeSet>,
    row_complete: bool,
    rows_valid_end: usize,
    token_mask: Option<SimpleVob>,
    mask_key: Option<super::bias::Key>,
    bytes: Vec<u8>,
    byte_to_token_idx: Vec<u32>,
    lexer_stack_flush_position: usize,
    lexer_stack_top_eos: bool,
    max_items_in_row: usize,
    lexeme_bytes: Vec<u8>,
    backtrack_bytes: usize,
    token_idx: usize,
    max_all_items: usize,
    last_force_bytes_len: usize,
    stats: super::ParserStats,
    capture_raw: Vec<u8>,
    capture_pending: Option<Vec<u8>>,
    captures: Vec<agenda::InitialCapture>,
    agenda_closed: bool,
    grammar: SharedGrammar,
}
impl fmt::Debug for PreparedEarleySeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedEarleySeed")
            .field("initial_items", &self.initial_item_count())
            .field("has_scratch", &self.scratch.is_some())
            .finish()
    }
}
fn allocation_bytes<T>(total: usize) -> Option<usize> {
    Some(Layout::array::<T>(total).ok()?.size())
}
fn reserve<T, F: crate::earley::PreparedFunding<Error = E>, E>(
    values: &mut Vec<T>,
    total: usize,
    funding: &F,
) -> Result<(), Cause<E>> {
    if total <= values.capacity() {
        return Ok(());
    }
    funding.reserve(
        allocation_bytes::<T>(total).ok_or(Cause::Overflow)?,
    )
    .map_err(Cause::Funding)?;
    values
        .try_reserve_exact(total - values.len())
        .map_err(Cause::Allocation)?;
    if values.capacity() != total {
        return Err(Cause::Capacity);
    }
    Ok(())
}
fn mask<F: crate::earley::PreparedFunding<Error = E>, E>(bits: usize, funding: &F) -> Result<SimpleVob, Cause<E>> {
    let plan = TokenMaskConstructionPlan::zeroed(bits).map_err(Cause::MaskSource)?;
    funding.reserve(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
    plan.compile().map_err(Cause::Mask)
}
impl PreparedEarleySeed {
    fn vacant(grammar: SharedGrammar) -> Self {
        Self {
            scratch: None,
            lexemes: None,
            grammars: None,
            rows: Vec::new(),
            row_infos: Vec::new(),
            lexer_stack: Vec::new(),
            initial_selection: None,
            row_complete: false,
            rows_valid_end: 0,
            token_mask: None,
            mask_key: None,
            bytes: Vec::new(),
            byte_to_token_idx: Vec::new(),
            lexer_stack_flush_position: 0,
            lexer_stack_top_eos: false,
            max_items_in_row: usize::MAX,
            lexeme_bytes: Vec::new(),
            backtrack_bytes: 0,
            token_idx: 0,
            max_all_items: usize::MAX,
            last_force_bytes_len: usize::MAX,
            stats: super::ParserStats::default(),
            capture_raw: Vec::new(),
            capture_pending: None,
            captures: Vec::new(),
            agenda_closed: false,
            grammar,
        }
    }
    fn controls<F, E>() -> Option<usize> {
        let parts = [
            TokenMaskConstructionPlan::inspection_control_bytes()?,
            size_of::<Self>(),
            size_of::<Scratch>(),
            size_of::<Cause<E>>(),
            size_of::<PreparedEarleySeedError<E>>(),
            size_of::<SharedGrammar>(),
            size_of::<F>(),
            size_of::<TokenMaskConstructionPlan<'_>>(),
            size_of::<Result<Self, PreparedEarleySeedError<E>>>(),
            size_of::<Result<(), Cause<E>>>(),
            size_of::<Result<(), E>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<SimpleVob, Cause<E>>>(),
            size_of::<Result<SimpleVob, TokenMaskConstructionFailure>>(),
            size_of::<Result<TokenMaskConstructionPlan<'_>, TokenMaskSourceError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<(&mut Scratch, &F)>(),
            size_of::<(&CGrammar, &mut Scratch)>(),
            size_of::<(Item, ParamValue, usize, usize)>(),
            size_of::<GrammarStackNode>(),
            size_of::<std::slice::Iter<'_, super::RhsPtr>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Option<usize>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Consumes the caller's actual immutable declaration alias, reserves each
    /// destination, then invokes the same Scratch and initial prediction workers.
    pub fn prepare<F: crate::earley::PreparedFunding<Error = E>, E>(
        grammar: SharedGrammar,
        funding: &F,
    ) -> Result<Self, PreparedEarleySeedError<E>> {
        let mut owner = Self::vacant(grammar);
        let result = (|| -> Result<(), Cause<E>> {
            funding.reserve(Self::controls::<F, E>().ok_or(Cause::Overflow)?).map_err(Cause::Funding)?;
            owner.lexemes = Some(LexemeSet::from_owned_vob(mask(
                owner.grammar.lexer_spec().lexemes.len(),
                funding,
            )?));
            owner.grammars = Some(mask(
                owner.grammar.lexer_spec().skip_by_class.len(),
                funding,
            )?);
            owner.scratch = Some(Scratch::from_masks(
                owner.grammar.clone(),
                owner.lexemes.take().expect("lexemes"),
                owner.grammars.take().expect("grammars"),
            ));
            let scratch = owner.scratch.as_mut().expect("scratch");
            reserve(&mut scratch.grammar_stack, 1, funding)?;
            scratch.grammar_stack.push(GrammarStackNode::root());
            seed_predictions(&owner.grammar, |item, param| {
                if !scratch.contains_item_arg(item, param) {
                    let count = scratch.row_end.checked_add(1).ok_or(Cause::Overflow)?;
                    reserve(&mut scratch.items, count, funding)?;
                    if scratch.parametric {
                        reserve(&mut scratch.item_args, count, funding)?;
                    }
                    scratch.put_item(item, param);
                    scratch.row_end = count;
                }
                Ok(())
            })
        })();
        match result {
            Ok(()) => Ok(owner),
            Err(cause) => Err(PreparedEarleySeedError {
                cause,
                prefix: owner,
            }),
        }
    }
    /// Number of actual chart items, including predictions added by agenda closure.
    pub fn initial_item_count(&self) -> usize {
        self.scratch.as_ref().map_or(0, |s| s.row_end)
    }
    /// Borrowed declaration used by those actual prediction destinations.
    pub fn grammar(&self) -> &CGrammar {
        &self.grammar
    }
}

impl<E> Cause<E> {
 fn frame(error: crate::earley::FrameError<E>) -> Self { match error { crate::earley::FrameError::Overflow => Self::Overflow, crate::earley::FrameError::Funding(error) => Self::Funding(error) } }
}
