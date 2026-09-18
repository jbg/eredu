//! One completion/prediction/nullable agenda for ordinary and paid chart owners.
use super::{
    CGrammar, CSymIdx, CSymbol, GrammarStackNode, Item, Lexeme, ParamValue, Scratch, DEBUG,
};
use std::{ops::Range, sync::Arc};

pub(super) trait Context {
    type Error;
    fn scratch(&self) -> &Scratch;
    fn scratch_mut(&mut self) -> &mut Scratch;
    fn token_idx(&self) -> usize;
    fn row_items(&self, row: usize) -> Result<Range<usize>, Self::Error>;
    fn add(&mut self, item: Item, arg: ParamValue, info: &'static str) -> Result<(), Self::Error>;
    fn push_grammar(&mut self, node: GrammarStackNode) -> Result<(), Self::Error>;
    fn captures(
        &mut self,
        item: Item,
        row: usize,
        lexeme: &Lexeme,
        scanned: bool,
    ) -> Result<(), Self::Error>;
    fn nullable_capture(&mut self, symbol: CSymIdx) -> Result<(), Self::Error>;
    fn trace_item(&self, index: usize);
    fn add_origin(
        &mut self,
        item: Item,
        origin: usize,
        info: &'static str,
    ) -> Result<(), Self::Error> {
        let scratch = self.scratch();
        let arg = if scratch.parametric {
            scratch.item_args[origin]
        } else {
            ParamValue::default()
        };
        self.add(item, arg, info)
    }
}

pub(super) fn grammar_node(
    grammar: &CGrammar,
    scratch: &Scratch,
    symbol: &CSymbol,
    row: usize,
    token: usize,
) -> GrammarStackNode {
    assert!(symbol.rules.len() == 1);
    let start_item = Item::new(symbol.rules[0], row);
    assert!(grammar.sym_idx_dot(start_item.advance_dot().rhs_ptr()) == CSymIdx::NULL);
    let nested = grammar.sym_data_dot(start_item.rhs_ptr());
    GrammarStackNode {
        back_ptr: scratch.push_grm_top,
        token_horizon: symbol.props.max_tokens.saturating_add(token) as u32,
        grammar_id: nested.props.grammar_id,
        start_item,
        start_item_idx: usize::MAX,
    }
}

// Targets and source order are shared; the context owns actual capture storage.
pub(super) fn capture_targets<E>(
    grammar: &CGrammar,
    item: Item,
    row: usize,
    scanned: bool,
    mut emit: impl FnMut(CSymIdx, bool, usize) -> Result<(), E>,
) -> Result<(), E> {
    let rule = item.rhs_ptr();
    if grammar.sym_idx_dot(rule) == CSymIdx::NULL {
        emit(grammar.sym_idx_lhs(rule), false, item.start_pos())?;
    }
    if scanned {
        let previous = item.rewind_dot();
        let symbol = grammar.sym_idx_dot(previous.rhs_ptr());
        assert!(symbol != CSymIdx::NULL);
        emit(symbol, true, row)?;
    }
    Ok(())
}

pub(super) fn capture_changed(name: &str, bytes: &[u8], previous: Option<&[u8]>) -> bool {
    name.starts_with("__LIST_APPEND:") || previous != Some(bytes)
}

pub(super) fn run<C: Context>(
    context: &mut C,
    row: usize,
    lexeme: &Lexeme,
) -> Result<(), C::Error> {
    // This is only an immutable alias of the actual chart's retained declaration.
    // Keeping it separate permits destination mutation without cloning symbols.
    let grammar = context.scratch().grammar.clone();
    let mut pointer = context.scratch().row_start;
    let scanned_end = if pointer == 0 {
        0
    } else {
        context.scratch().row_end
    };
    context.scratch_mut().push_allowed_lexemes.clear();
    context
        .scratch_mut()
        .push_allowed_grammar_ids
        .set_all(false);
    while pointer < context.scratch().row_end {
        let index = pointer;
        let item = context.scratch().items[index];
        pointer += 1;
        context.trace_item(index);
        let dot = item.rhs_ptr();
        let next = grammar.sym_idx_dot(dot);
        if context.scratch().definitive {
            let scanned = pointer <= scanned_end;
            if scanned || next == CSymIdx::NULL {
                context.captures(item, row, lexeme, scanned)?;
            }
        }
        if next == CSymIdx::NULL {
            let lhs = grammar.sym_idx_lhs(dot);
            if item.start_pos() < row {
                for origin in context.row_items(item.start_pos())? {
                    let item = context.scratch().items[origin];
                    if grammar.sym_idx_dot(item.rhs_ptr()) == lhs {
                        context.add_origin(item.advance_dot(), origin, "complete")?;
                    }
                }
            }
        } else {
            let symbol = grammar.sym_data(next);
            if let Some(lexeme) = symbol.lexeme {
                context
                    .scratch_mut()
                    .push_allowed_grammar_ids
                    .set(symbol.props.grammar_id.as_usize(), true);
                context.scratch_mut().push_allowed_lexemes.add(lexeme);
            }
            let mut conditional_nullable = false;
            if symbol.gen_grammar.is_some() {
                let mut node = grammar_node(
                    &grammar,
                    context.scratch(),
                    symbol,
                    row,
                    context.token_idx(),
                );
                context.add_origin(node.start_item, index, "gen_grammar")?;
                node.start_item_idx = context
                    .scratch()
                    .find_item(node.start_item)
                    .expect("predicted nested grammar item");
                context.push_grammar(node)?;
            } else if context.scratch().parametric {
                let param = grammar
                    .param_value_dot(dot)
                    .eval(context.scratch().item_args[index]);
                conditional_nullable = !symbol.cond_nullable.is_empty()
                    && symbol.cond_nullable.iter().any(|c| c.eval(param));
                for rule in 0..symbol.rules.len() {
                    if symbol.rules_cond[rule].eval(param) {
                        context.add(
                            Item::new(symbol.rules[rule], row),
                            param,
                            "predict_parametric",
                        )?;
                    }
                }
            } else {
                for rule in &symbol.rules {
                    context.add_origin(Item::new(*rule, row), index, "predict")?;
                }
            }
            if symbol.is_nullable || conditional_nullable {
                context.add_origin(item.advance_dot(), index, "null")?;
                if context.scratch().definitive && symbol.props.capture_name.is_some() {
                    context.nullable_capture(next)?;
                }
            }
        }
    }
    Ok(())
}

impl Context for super::ParserState {
    type Error = std::convert::Infallible;
    fn scratch(&self) -> &Scratch {
        &self.scratch
    }
    fn scratch_mut(&mut self) -> &mut Scratch {
        &mut self.scratch
    }
    fn token_idx(&self) -> usize {
        self.token_idx
    }
    fn row_items(&self, row: usize) -> Result<Range<usize>, Self::Error> {
        Ok(self.rows[row].item_indices())
    }
    fn add(&mut self, item: Item, arg: ParamValue, info: &'static str) -> Result<(), Self::Error> {
        self.scratch.add_unique_arg(item, info, arg);
        Ok(())
    }
    fn push_grammar(&mut self, node: GrammarStackNode) -> Result<(), Self::Error> {
        self.scratch.push_grammar_stack(node);
        Ok(())
    }
    fn captures(
        &mut self,
        item: Item,
        row: usize,
        lexeme: &Lexeme,
        scanned: bool,
    ) -> Result<(), Self::Error> {
        self.process_captures(item, row, lexeme, scanned);
        Ok(())
    }
    fn nullable_capture(&mut self, symbol: CSymIdx) -> Result<(), Self::Error> {
        let name = self
            .grammar
            .sym_data(symbol)
            .props
            .capture_name
            .as_ref()
            .expect("nullable capture");
        debug!("      capture: {} NULL", name);
        self.captures.push((name.clone(), vec![]));
        Ok(())
    }
    fn trace_item(&self, index: usize) {
        debug_def!(self, "    agenda: {}", self.item_to_string(index));
    }
}
