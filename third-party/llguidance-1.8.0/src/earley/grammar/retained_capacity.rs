//! Actual completed compiler capacity, never a construction or copy allowance.
use super::*;
use super::source_copy::Cause;
use std::mem::size_of;

impl CGrammar {
    /// Inspects actual retained source allocations under the caller's payer.
    /// This does not grant destination-copy or execution authority.
    pub fn retained_capacity_bytes(&self, funding: &derivre::ParserAllocationFunding)
        -> Result<usize, CompiledGrammarCopyFailure> {
        let inspect = || -> Result<usize, Cause> {
        use derivre::prepared_funding::{PreparedFunding, Scope};
        let scope = Scope::new(funding).map_err(|error| Cause::Condition(ConditionSourceError::frame(error)))?;
        let _frame = scope.frame(size_of::<(
            &CGrammar, &derivre::ParserAllocationFunding, usize, usize,
            std::slice::Iter<'_, CSymbol>,
            std::iter::Chain<std::slice::Iter<'_, ParamCond>, std::slice::Iter<'_, ParamCond>>,
            [&Option<String>; 2], Option<usize>, Result<usize, CompiledGrammarCopyFailure>, Cause,
        )>()).map_err(|error| Cause::Condition(ConditionSourceError::frame(error)))?;
        let lexer = self.lexer_spec.retained_capacity_bytes(&scope).map_err(Cause::Lexer)?;
        let mut bytes = (|| {
            lexer
                .checked_add(self.symbols.capacity().checked_mul(size_of::<CSymbol>())?)?
                .checked_add(self.rhs_elements.capacity().checked_mul(size_of::<CSymIdx>())?)?
                .checked_add(self.rhs_params.capacity().checked_mul(size_of::<ParamExpr>())?)?
                .checked_add(self.rhs_ptr_to_sym_idx.capacity().checked_mul(size_of::<CSymIdx>())?)?
                .checked_add(self.rhs_ptr_to_sym_flags.capacity().checked_mul(size_of::<SymFlags>())?)
        })().ok_or(Cause::Overflow)?;
        for symbol in &self.symbols {
            bytes = (|| bytes.checked_add(symbol.name.capacity())?
                .checked_add(symbol.rules.capacity().checked_mul(size_of::<RhsPtr>())?)?
                .checked_add(symbol.rules_cond.capacity().checked_mul(size_of::<ParamCond>())?)?
                .checked_add(symbol.cond_nullable.capacity().checked_mul(size_of::<ParamCond>())?))()
                .ok_or(Cause::Overflow)?;
            for value in [&symbol.props.capture_name, &symbol.props.stop_capture_name]
                .into_iter().flatten() {
                bytes = bytes.checked_add(value.capacity()).ok_or(Cause::Overflow)?;
            }
            if let Some(grammar) = &symbol.gen_grammar {
                let GrammarId::Name(name) = &grammar.grammar;
                bytes = bytes.checked_add(name.capacity()).ok_or(Cause::Overflow)?;
            }
            for condition in symbol.cond_nullable.iter().chain(&symbol.rules_cond) {
                bytes = bytes.checked_add(condition.condition_copy_plan(&scope).map_err(Cause::Condition)?.requirements().buffer_bytes())
                    .ok_or(Cause::Overflow)?;
            }
        }
        Ok(bytes)
        };
        inspect().map_err(CompiledGrammarCopyFailure::inspection)
    }
}

#[cfg(all(test, feature = "lark"))]
mod tests {
    use super::*;
    use crate::{
        api::{GrammarInit, TopLevelGrammar},
        Logger,
    };

    #[test]
    fn source_capacity_includes_spare_buffers_and_compact_copy_has_its_own_receipt() {
        let mut source = GrammarInit::Serialized(TopLevelGrammar::from_lark(
            "start: word word\nword: /[a-z]+/".into(),
        ))
        .to_cgrammar(
            None,
            &mut Logger::new(0, 0),
            ParserLimits::default(),
            &[],
            derivre::ParserAllocationFunding::unenforced(),
        )
        .unwrap();
        let before = source.retained_capacity_bytes(&derivre::ParserAllocationFunding::unenforced()).unwrap();
        let old_symbols = source.symbols.capacity();
        source.symbols.reserve_exact(old_symbols + 19);
        let added_symbols = source.symbols.capacity() - old_symbols;
        let old_name = source.symbols[0].name.capacity();
        source.symbols[0].name.reserve_exact(old_name + 31);
        let added_name = source.symbols[0].name.capacity() - old_name;
        assert_eq!(
            source.retained_capacity_bytes(&derivre::ParserAllocationFunding::unenforced()).unwrap(),
            before + added_symbols * size_of::<CSymbol>() + added_name
        );
        let plan = source.source_copy_plan(&derivre::ParserAllocationFunding::unenforced()).unwrap();
        let compact_bytes = plan.requirements().retained_bytes();
        let copy = plan.compile().unwrap();
        assert_eq!(copy.retained_capacity_bytes(&derivre::ParserAllocationFunding::unenforced()).unwrap(), compact_bytes);
        assert!(source.retained_capacity_bytes(&derivre::ParserAllocationFunding::unenforced()).unwrap() > compact_bytes);
        assert_eq!(copy.rhs_elements, source.rhs_elements);
        assert_eq!(copy.rhs_params, source.rhs_params);
        assert_eq!(copy.start(), source.start());
    }
}
