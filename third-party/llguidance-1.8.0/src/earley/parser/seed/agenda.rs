//! Initial chart closure using the ordinary agenda and paid reached destinations.
use super::super::{
    agenda, CGrammar, SharedGrammar, CSymIdx, GrammarStackNode, Item, Lexeme, ParamValue, Scratch,
};
use super::{reserve, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
};

// Capture names remain in the independently paid immutable declaration. Initial
// occurrences have empty bytes; later definitive captures own decoded bytes.
#[derive(Debug)]
pub(super) struct InitialCapture {
    pub(super) symbol: CSymIdx,
    pub(super) stop: bool,
    pub(super) bytes: Vec<u8>,
}
impl InitialCapture {
    pub(super) fn name<'a>(&self, grammar: &'a CGrammar) -> &'a str {
        let props = &grammar.sym_data(self.symbol).props;
        if self.stop {
            props.stop_capture_name.as_deref()
        } else {
            props.capture_name.as_deref()
        }
        .expect("retained initial capture name")
    }
}
struct Context<'a, F> {
    trie: Option<&'a toktrie::TokTrie>,
    token: usize,
    owner: &'a mut PreparedEarleySeed,
    funding: &'a F,
}
impl<F: crate::earley::PreparedFunding<Error = E>, E> Context<'_, F> {
    fn capture(&mut self, symbol: CSymIdx, stop: bool) -> Result<(), Cause<E>> {
        let entry = InitialCapture {
            symbol,
            stop,
            bytes: Vec::new(),
        };
        let name = entry.name(&self.owner.grammar);
        let previous = self
            .owner
            .captures
            .iter()
            .rev()
            .find(|v| v.name(&self.owner.grammar) == name)
            .map(|previous| previous.bytes.as_slice());
        if !agenda::capture_changed(name, &[], previous) {
            return Ok(());
        }
        let total = self
            .owner
            .captures
            .len()
            .checked_add(1)
            .ok_or(Cause::Overflow)?;
        reserve(&mut self.owner.captures, total, self.funding)?;
        self.owner.captures.push(entry);
        Ok(())
    }
}
impl<F: crate::earley::PreparedFunding<Error = E>, E> agenda::Context for Context<'_, F> {
    type Error = Cause<E>;
    fn scratch(&self) -> &Scratch {
        self.owner.scratch.as_ref().expect("seed scratch")
    }
    fn scratch_mut(&mut self) -> &mut Scratch {
        self.owner.scratch.as_mut().expect("seed scratch")
    }
    fn token_idx(&self) -> usize {
        self.token
    }
    fn row_items(&self, row: usize) -> Result<Range<usize>, Self::Error> {
        self.owner
            .rows
            .get(row)
            .map(|row| row.item_indices())
            .ok_or(Cause::Source)
    }
    fn add(&mut self, item: Item, arg: ParamValue, _info: &'static str) -> Result<(), Self::Error> {
        let scratch = self.owner.scratch.as_mut().expect("seed scratch");
        if scratch.contains_item_arg(item, arg) {
            return Ok(());
        }
        let count = scratch.row_end.checked_add(1).ok_or(Cause::Overflow)?;
        u32::try_from(count).map_err(|_| Cause::Overflow)?;
        reserve(&mut scratch.items, count, self.funding)?;
        if scratch.parametric {
            reserve(&mut scratch.item_args, count, self.funding)?;
        }
        scratch.put_item(item, arg);
        scratch.row_end = count;
        Ok(())
    }
    fn push_grammar(&mut self, node: GrammarStackNode) -> Result<(), Self::Error> {
        let scratch = self.owner.scratch.as_mut().expect("seed scratch");
        let count = scratch
            .grammar_stack
            .len()
            .checked_add(1)
            .ok_or(Cause::Overflow)?;
        u32::try_from(count).map_err(|_| Cause::Overflow)?;
        reserve(&mut scratch.grammar_stack, count, self.funding)?;
        // Same publication as ordinary Scratch, without its formatting/logging.
        scratch.put_grammar_stack(node);
        Ok(())
    }
    fn captures(
        &mut self,
        item: Item,
        row: usize,
        lexeme: &Lexeme,
        scanned: bool,
    ) -> Result<(), Self::Error> {
        let initial = row == 0 && !scanned && lexeme.is_bogus();
        if !initial && self.trie.is_none() {
            return Err(Cause::Source);
        }
        let grammar = self.owner.grammar.clone();
        agenda::capture_targets(&grammar, item, row, scanned, |symbol, is_lexeme, start| {
            let props = &grammar.sym_data(symbol).props;
            for stop in [true, false] {
                if if stop {
                    props.stop_capture_name.is_some()
                } else {
                    props.capture_name.is_some()
                } {
                    if initial {
                        if is_lexeme || start != 0 {
                            return Err(Cause::Source);
                        }
                        self.capture(symbol, stop)?;
                    } else {
                        self.owner.capture_source(
                            self.trie.expect("capture trie"),
                            symbol,
                            stop,
                            start,
                            row,
                            lexeme,
                            is_lexeme,
                            self.funding,
                        )?;
                    }
                }
            }
            Ok(())
        })
    }
    fn nullable_capture(&mut self, symbol: CSymIdx) -> Result<(), Self::Error> {
        self.capture(symbol, false)
    }
    fn trace_item(&self, _index: usize) {}
}
impl PreparedEarleySeed {
    /// Closes the actual initial prediction agenda with the ordinary completion,
    /// nested-grammar, parametric and nullable rules. A failed destination stays
    /// in the returned owner; this does not yet publish a lexer-backed row.
    pub fn close_initial_agenda<F: crate::earley::PreparedFunding<Error = E>, E>(
        mut self,
        funding: &F,
    ) -> Result<Self, PreparedEarleySeedError<E>> {
        let result = (|| -> Result<(), Cause<E>> {
            let parts = [
                size_of::<Self>(),
                size_of::<Context<'_, F>>(),
                size_of::<F>(),
                size_of::<PreparedEarleySeedError<E>>(),
                size_of::<Cause<E>>(),
                size_of::<Result<Self, PreparedEarleySeedError<E>>>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<Result<(), E>>(),
                size_of::<Result<Range<usize>, Cause<E>>>(),
                size_of::<InitialCapture>(),
                size_of::<SharedGrammar>(),
                size_of::<GrammarStackNode>(),
                size_of::<Lexeme>(),
                size_of::<(usize, usize, usize, usize, Item, ParamValue, CSymIdx, bool)>(),
                size_of::<(&CGrammar, &mut Context<'_, F>, Item, usize, bool)>(),
                size_of::<std::slice::Iter<'_, super::super::RhsPtr>>(),
                size_of::<std::slice::Iter<'_, crate::earley::grammar::ParamCond>>(),
                size_of::<std::iter::Rev<std::slice::Iter<'_, InitialCapture>>>(),
                size_of::<Result<u32, std::num::TryFromIntError>>(),
                size_of::<Result<(), std::collections::TryReserveError>>(),
                size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>(),
                size_of::<Option<&[u8]>>(),
                size_of::<Range<usize>>(),
            ];
            funding.reserve(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            let scratch = self.scratch.as_ref().ok_or(Cause::Source)?;
            if self.agenda_closed || scratch.row_start != 0 || !scratch.definitive {
                return Err(Cause::Source);
            }
            agenda::run(
                &mut Context {
                    owner: &mut self,
                    funding,
                    trie: None,
                    token: 0,
                },
                0,
                &Lexeme::bogus(),
            )?;
            self.agenda_closed = true;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(self),
            Err(cause) => Err(PreparedEarleySeedError {
                cause,
                prefix: self,
            }),
        }
    }
    /// Whether the initial agenda completed before any lexical row publication.
    pub fn initial_agenda_closed(&self) -> bool {
        self.agenda_closed
    }
    /// Actual lexical selection produced by the shared initial agenda.
    pub fn initial_lexemes(&self) -> Option<&super::super::LexemeSet> {
        self.agenda_closed.then(|| {
            &self
                .scratch
                .as_ref()
                .expect("closed scratch")
                .push_allowed_lexemes
        })
    }
    /// Captures emitted by initial completion/nullable rules, in ordinary order.
    /// Names borrow the paid declaration; subsequent definitive captures retain
    /// their actual decoded bytes in the same ordered record population.
    pub fn initial_captures(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.captures
            .iter()
            .map(|entry| (entry.name(&self.grammar), entry.bytes.as_slice()))
    }
}

pub(super) fn run<F: crate::earley::PreparedFunding<Error = E>, E>(
    owner: &mut PreparedEarleySeed,
    row: usize,
    lexeme: &Lexeme,
    trie: &toktrie::TokTrie,
    token: usize,
    funding: &F,
) -> Result<(), Cause<E>> {
    let parts = [
        PreparedEarleySeed::controls::<F, E>().ok_or(Cause::Overflow)?,
        size_of::<Context<'_, F>>(),
        size_of::<SharedGrammar>(),
        size_of::<GrammarStackNode>(),
        size_of::<(usize, usize, usize, Item, ParamValue, CSymIdx, bool)>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<Result<(), Cause<E>>>(),
    ];
    funding.reserve(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Cause::Overflow)?,
    )
    .map_err(Cause::Funding)?;
    agenda::run(
        &mut Context {
            owner,
            funding,
            trie: Some(trie),
            token,
        },
        row,
        lexeme,
    )
}
