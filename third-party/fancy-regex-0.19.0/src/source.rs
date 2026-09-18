//! Shared parse, optimization and analysis policy for regex source construction.
use crate::{
    allocation::{Allocation, Context, Unenforced},
    analyze::{analyze_with_allocations, AnalyzeContext, Info},
    compile::CompileOptions,
    optimize::optimize_with_allocations,
    parse::Parser,
    BytesMode, CompileError, ExprTree, RegexOptions, Result,
};

pub(crate) struct ParsedSource {
    pub(crate) tree: ExprTree,
    pub(crate) context: AnalyzeContext,
}
impl ParsedSource {
    pub(crate) fn new(pattern: &str, options: &RegexOptions) -> Result<Self> {
        Self::new_with_allocations(pattern, options, &Unenforced)
    }
    pub(crate) fn new_with_allocations(
        pattern: &str,
        options: &RegexOptions,
        allocation: &dyn Allocation,
    ) -> Result<Self> {
        let mut tree =
            Parser::parse_with_allocations(pattern, options.compute_flags(), allocation)?;
        let runtime = options.hard_regex_runtime_options;
        let explicit_capture_group_0 = if runtime.find_not_empty {
            false
        } else {
            optimize_with_allocations(&mut tree, allocation)?
        };
        Ok(Self {
            tree,
            context: AnalyzeContext {
                explicit_capture_group_0,
                find_not_empty: runtime.find_not_empty,
                disallow_empty_match_at_eof_after_newline: runtime
                    .disallow_empty_match_at_eof_after_newline,
                allow_input_assertion_overrides: runtime.allow_input_assertion_overrides,
            },
        })
    }
    pub(crate) fn analyze(&self) -> Result<Info<'_>> {
        self.analyze_with_allocations(&Unenforced)
    }
    pub(crate) fn analyze_with_allocations(&self, allocation: &dyn Allocation) -> Result<Info<'_>> {
        let info = analyze_with_allocations(&self.tree, self.context.clone(), allocation)?;
        if self.context.find_not_empty && info.const_size && info.min_size == 0 {
            return Err(crate::Error::CompileError(
                Context::new(allocation)
                    .storage
                    .boxed(CompileError::PatternCanNeverMatch)?,
            ));
        }
        Ok(info)
    }
    pub(crate) fn compile_options(&self, options: &RegexOptions) -> CompileOptions {
        CompileOptions {
            anchored: crate::analyze::can_compile_as_anchored(&self.tree.expr),
            contains_subroutines: self.tree.contains_subroutines,
            seek_filter: options.seek_filter,
            disallow_empty_match_at_eof_after_newline: self
                .context
                .disallow_empty_match_at_eof_after_newline,
            bytes_mode: options.bytes_mode,
            unicode: options.syntaxc.get_unicode()
                && !matches!(options.bytes_mode, BytesMode::Ascii),
            delegate_size_limit: options.delegate_size_limit,
            delegate_dfa_size_limit: options.delegate_dfa_size_limit,
        }
    }
}
