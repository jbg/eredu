//! Paid initial Row/RowInfo/LexerState publication through the retained lexer.
use super::super::{row, LexemeClass, LexemeSet, LexerState, Row, RowInfo};
use super::{reserve, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use crate::earley::{PreparedLexer, PreparedLexerOperationError};
use derivre::StateID;
use std::mem::{size_of, size_of_val};
use toktrie::{TokenMaskConstructionFailure, TokenMaskConstructionPlan, TokenMaskSourceError};

impl PreparedEarleySeed {
    /// Publishes the initial chart through the supplied original lexical owner.
    /// Composition retains that lexer beside this seed and supplies the same
    /// declaration; this mechanism does not grant runnable parser authority.
    /// On error the seed owns its partial rows and the caller keeps the lexer.
    pub fn publish_initial_row<F: Fn(usize) -> Result<(), E>, E>(
        mut self,
        lexer: &mut PreparedLexer,
        funding: &F,
    ) -> Result<Self, PreparedEarleySeedError<E>> {
        let result = (|| -> Result<(), Cause<E>> {
            let parts = [
                size_of::<Self>(),
                size_of::<Row>(),
                size_of::<RowInfo>(),
                size_of::<LexerState>(),
                size_of::<PreparedEarleySeedError<E>>(),
                size_of::<PreparedLexerOperationError<E>>(),
                size_of::<Cause<E>>(),
                size_of::<F>(),
                size_of::<(&mut Self, &mut PreparedLexer, &F)>(),
                size_of::<Result<Self, PreparedEarleySeedError<E>>>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<Result<(), E>>(),
                size_of::<Result<StateID, PreparedLexerOperationError<E>>>(),
                size_of::<(StateID, usize, LexemeClass)>(),
                TokenMaskConstructionPlan::inspection_control_bytes().ok_or(Cause::Overflow)?,
                size_of::<TokenMaskConstructionPlan<'_>>(),
                size_of::<Result<TokenMaskConstructionPlan<'_>, TokenMaskSourceError>>(),
                size_of::<Result<toktrie::SimpleVob, TokenMaskConstructionFailure>>(),
                size_of::<Result<(), std::collections::TryReserveError>>(),
                size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>(),
                size_of::<Option<&crate::earley::regexvec::StateDesc>>(),
            ];
            funding(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            let scratch = self.scratch.as_ref().ok_or(Cause::Source)?;
            if !self.agenda_closed
                || self.row_complete
                || !self.rows.is_empty()
                || !self.lexer_stack.is_empty()
                || scratch.row_start != 0
                || scratch.row_len() == 0
            {
                return Err(Cause::Source);
            }
            reserve(&mut self.lexer_stack, 1, funding)?;
            self.lexer_stack.push(LexerState {
                row_idx: 0,
                lexer_state: StateID::DEAD,
                byte: None,
            });
            let scratch = self.scratch.as_mut().expect("initial scratch");
            row::add_skip(scratch, LexemeClass::ROOT, true);
            let state = lexer
                .start_state(&scratch.push_allowed_lexemes, funding)
                .map_err(Cause::Lexer)?;
            reserve(&mut self.rows, 1, funding)?;
            row::store(&mut self.rows, 0, scratch.work_row(state));
            reserve(&mut self.row_infos, 1, funding)?;
            row::store_info(&mut self.row_infos, 0, 0, 0);
            if !self.grammar.lexer_spec().allow_initial_skip {
                let possible = &lexer
                    .vector()
                    .state_desc(state)
                    .ok_or(Cause::Source)?
                    .possible;
                let plan = possible.copy_plan().map_err(Cause::MaskSource)?;
                funding(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
                self.initial_selection = Some(LexemeSet::from_owned_vob(
                    plan.compile().map_err(Cause::Mask)?,
                ));
                let selected = self
                    .initial_selection
                    .as_mut()
                    .expect("initial skip selection");
                selected.remove(self.grammar.lexer_spec().skip_id(LexemeClass::ROOT));
                let next = lexer.start_state(selected, funding).map_err(Cause::Lexer)?;
                self.rows[0].lexer_start_state = next;
                self.initial_selection = None;
            }
            self.lexer_stack[0].lexer_state = self.rows[0].lexer_start_state;
            self.stats.rows = 1;
            self.stats.all_items = self.scratch.as_ref().expect("initial chart").row_len();
            self.row_complete = true;
            self.rows_valid_end = self.rows.len();
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
    /// Initial state after actual row publication and optional skip removal.
    pub fn initial_lexer_state(&self) -> Option<StateID> {
        self.row_complete.then(|| self.rows[0].lexer_start_state)
    }
}
