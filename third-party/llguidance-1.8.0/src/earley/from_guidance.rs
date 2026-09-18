use std::fmt::Write;
use std::vec;

use super::grammar::SymIdx;
use super::lexerspec::LexerSpec;
use super::{CGrammar, Grammar};
use crate::api::{GrammarId, GrammarInit, GrammarWithLexer, ParserLimits, TopLevelGrammar};
use crate::earley::lexerspec::LexemeClass;
use crate::GrammarBuilder;
use crate::Instant;
use crate::{loginfo, JsonCompileOptions, Logger};
use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};
use derivre::SourceHashMap as HashMap;
use toktrie::TokTrie;

struct CompileCtx<'t> {
    builder: Option<GrammarBuilder<'t>>,
    grammar_by_idx: HashMap<GrammarId, usize>,
    grammar_roots: Vec<(SymIdx, LexemeClass)>,
    funding: derivre::ParserAllocationFunding,
}

impl CompileCtx<'_> {
    fn run_one(&mut self, input: GrammarWithLexer) -> Result<(SymIdx, LexemeClass)> {
        let builder = std::mem::take(&mut self.builder).unwrap();

        let res = if let Some(lark) = input.lark_grammar {
            #[cfg(feature = "lark")]
            {
                use crate::lark::lark_to_llguidance;
                ensure!(&self.funding,
                    input.json_schema.is_none(),
                    "cannot have both lark_grammar and json_schema"
                );
                lark_to_llguidance(builder, &lark)?
            }
            #[cfg(not(feature = "lark"))]
            {
                let _ = lark;
                bail!(&self.funding, "lark_grammar is not supported in this build")
            }
        } else if let Some(json_schema) = input.json_schema {
            JsonCompileOptions::new(&builder.funding)?.json_to_llg_with_overrides(builder, json_schema)?
        } else {
            bail!(&self.funding, "grammar must have either lark_grammar or json_schema");
        };

        res.builder.check_limits()?;

        let grammar_id = res.builder.grammar.sym_props(res.start_node).grammar_id;

        // restore builder
        self.builder = Some(res.builder);

        Ok((res.start_node, grammar_id))
    }

    fn run(mut self, mut input: TopLevelGrammar) -> Result<(Grammar, LexerSpec)> {
        for (idx, grm) in input.grammars.iter_mut().enumerate() {
            if grm.lark_grammar.is_none() && grm.json_schema.is_none() {
                bail!(&self.funding, "grammar must have either lark_grammar or json_schema");
            }
            if let Some(n) = grm.name.take() {
                let n = GrammarId::Name(n);
                if self.grammar_by_idx.contains_key(&n) {
                    bail!(&self.funding, "duplicate grammar name: {}", n);
                }
                self.funding.try_insert(&mut self.grammar_by_idx, n, idx)?;
            }
        }

        for (idx, grm) in input.grammars.into_iter().enumerate() {
            let v = self.run_one(grm)?;
            self.grammar_roots[idx] = v;
        }

        let builder = self.builder.unwrap();
        let warnings = builder.get_warnings()?;
        let mut grammar = builder.grammar;
        let mut lexer_spec = builder.regex.spec;

        grammar.resolve_grammar_refs(&mut lexer_spec, |id| {
            self.grammar_by_idx
                .get(id)
                .map(|index| self.grammar_roots[*index])
        })?;

        assert!(lexer_spec.grammar_warnings.is_empty());
        lexer_spec.grammar_warnings = warnings;

        Ok((grammar, lexer_spec))
    }
}

#[derive(Debug, Clone)]
pub enum ValidationResult {
    Valid,
    Warnings(Vec<String>),
    Error(String),
}

impl ValidationResult {
    pub fn from_warning(w: Vec<String>) -> Self {
        if w.is_empty() {
            ValidationResult::Valid
        } else {
            ValidationResult::Warnings(w)
        }
    }

    pub fn into_tuple(self) -> (bool, Vec<String>) {
        match self {
            ValidationResult::Valid => (false, vec![]),
            ValidationResult::Warnings(w) => (false, w),
            ValidationResult::Error(e) => (true, vec![e]),
        }
    }

    pub fn into_error(self) -> Option<String> {
        match self {
            ValidationResult::Valid => None,
            ValidationResult::Warnings(_) => None,
            ValidationResult::Error(e) => Some(e),
        }
    }

    pub fn render(&self, with_warnings: bool) -> String {
        match self {
            ValidationResult::Valid => String::new(),
            ValidationResult::Warnings(w) => {
                if with_warnings {
                    w.iter()
                        .map(|w| format!("WARNING: {w}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    String::new()
                }
            }
            ValidationResult::Error(e) => format!("ERROR: {e}"),
        }
    }
}

/// An immutable-grammar construction failure and its original allocation
/// account. Partial builders retire before this owner can release their charge.
#[derive(Debug)]
pub struct GrammarCompilationError {
    cause: derivre::ParserError,
    funding: derivre::ParserAllocationFunding,
}
impl std::fmt::Display for GrammarCompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for GrammarCompilationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
pub(crate) fn is_grammar_storage_failure(error: &derivre::ParserError) -> bool {
    derivre::is_parser_storage_failure(error)
        || error.chain().any(|cause| cause.is::<serde_json::allocation::AllocationError>()
            || cause.is::<serde_json::bounded_events::PlanError>()
            || cause.is::<referencing::allocation::AllocationError>()
            || matches!(cause.downcast_ref::<referencing::Error>(),
                Some(referencing::Error::Unqualified(_) | referencing::Error::MetaSchema(_))))
}

impl GrammarCompilationError {
    /// Storage failure is distinct from unsupported grammar syntax and must
    /// not trigger a less constrained schema compilation fallback.
    pub fn is_storage_failure(&self) -> bool {
        self.funding.failure().is_some()
            || is_grammar_storage_failure(&self.cause)
    }
    /// The concrete first funding refusal remains in its originally paid slot.
    pub fn funding_failure(&self) -> Option<derivre::ParserAllocationFailure> {
        self.funding.failure()
    }
}

impl GrammarInit {
    pub fn to_internal(
        self,
        tok_trie: Option<&TokTrie>,
        limits: ParserLimits,
        funding: derivre::ParserAllocationFunding,
    ) -> Result<(Grammar, LexerSpec)> {
        match self {
            GrammarInit::Internal(g, l) => Ok((g, l)),

            GrammarInit::Serialized(input) => {
                ensure!(&funding, !input.grammars.is_empty(), "empty grammars array");

                let builder = GrammarBuilder::new(tok_trie, limits.clone(), funding.clone())?;
                let mut grammar_roots = Vec::new();
                funding.try_grow_vec(&mut grammar_roots, input.grammars.len())?;
                grammar_roots.resize(input.grammars.len(), (SymIdx::BOGUS, LexemeClass::ROOT));

                let ctx = CompileCtx {
                    builder: Some(builder),
                    grammar_by_idx: HashMap::default(),
                    grammar_roots,
                    funding,
                };

                ctx.run(input)
            }
        }
    }

    pub fn validate(
        self,
        tok_trie: Option<&TokTrie>,
        limits: ParserLimits,
        funding: derivre::ParserAllocationFunding,
    ) -> ValidationResult {
        match self.to_internal(tok_trie, limits, funding) {
            Ok((_, lex_spec)) => ValidationResult::from_warning(lex_spec.render_warnings()),
            Err(e) => ValidationResult::Error(e.to_string()),
        }
    }

    pub fn to_cgrammar(
        self,
        tok_trie: Option<&TokTrie>,
        logger: &mut Logger,
        limits: ParserLimits,
        extra_lexemes: &[String],
        funding: derivre::ParserAllocationFunding,
    ) -> Result<CGrammar, GrammarCompilationError> {
        let result = (|| -> Result<CGrammar> {
            let t0 = Instant::now();
            let (grammar, mut lexer_spec) =
                self.to_internal(tok_trie, limits.clone(), funding.clone())?;
            lexer_spec.add_extra_lexemes(extra_lexemes)?;
            compile_grammar(t0, grammar, lexer_spec, logger, &limits, funding.clone())
        })();
        match result {
            Ok(mut grammar) => {
                grammar.compilation_funding = funding;
                Ok(grammar)
            }
            Err(cause) => Err(GrammarCompilationError { cause, funding }),
        }
    }
}

fn compile_grammar(
    t0: Instant,
    mut grammar: Grammar,
    lexer_spec: LexerSpec,
    logger: &mut Logger,
    limits: &ParserLimits,
    funding: derivre::ParserAllocationFunding,
) -> Result<CGrammar> {
    let log_grammar = logger.level_enabled(3) || (logger.level_enabled(2) && grammar.is_small());
    if log_grammar {
        writeln!(
            logger.info_logger(),
            "{:?}\n{}\n",
            lexer_spec,
            grammar.to_string(Some(&lexer_spec))
        )
        .unwrap();
    } else if logger.level_enabled(2) {
        writeln!(
            logger.info_logger(),
            "Grammar: (skipping body; log_level=3 will print it); {}",
            grammar.stats()
        )
        .unwrap();
    }

    let t1 = Instant::now();
    grammar = grammar.optimize()?;

    if log_grammar {
        write!(
            logger.info_logger(),
            "  == Optimize ==>\n{}",
            grammar.to_string(Some(&lexer_spec))
        )
        .unwrap();
    } else if logger.level_enabled(2) {
        writeln!(logger.info_logger(), "  ==> {}", grammar.stats()).unwrap();
    }

    let grammars = grammar.compile(lexer_spec, limits, funding)?;

    loginfo!(
        logger,
        "build grammar: {:?}; optimize: {:?}",
        t1 - t0,
        t1.elapsed()
    );

    Ok(grammars)
}
