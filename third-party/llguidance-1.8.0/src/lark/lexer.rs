use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

use crate::api::{RegexExt, SkipSpec};
use crate::earley::{
    PreparedLexer,
    regexvec::{LexemeSet, prepared::PreparedRegexVector},
};
use derivre::{ParserAllocationFunding, RegexAst, SourceHashMap as HashMap};
use derivre::{
    ParserError, ParserResult as Result, parser_bail as bail, parser_ensure as ensure,
    parser_error as anyhow,
};
use serde_json::Value;
use toktrie::TokenMaskConstructionPlan;

use crate::{
    api::ParserLimits,
    earley::{lexer::LexerResult, lexerspec::LexerSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
#[allow(clippy::upper_case_acronyms)]
pub enum Token {
    KwIgnore,
    KwImport,
    KwOverride,
    KwDeclare,
    KwJson,
    KwRegex,
    KwLLGuidance,
    KwIf,
    KwLark,
    Colon,
    DoubleColon, // ::
    Equals,
    Comma,
    Dot,
    DotDot,
    Arrow,
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    LBracket, // [
    RBracket, // ]
    Tilde,
    Hash,
    Underscore, // _ (is not a rule)
    // regexps
    Op, // + * ?
    String,
    Regexp,
    Rule,
    Token,
    Number,
    HexNumber,
    Newline,
    VBar,
    And,          // &
    SpecialToken, // <something>
    GrammarRef,   // @grammar_id or @7
    // special
    SKIP,
    EOF,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lst = Self::LITERAL_TOKENS;
        if let Some((_, literal)) = lst.iter().find(|(t, _)| t == self) {
            write!(f, "'{literal}'")
        } else {
            match self {
                Token::Op => write!(f, "'+' or '*' or '?'"),
                Token::Rule => write!(f, "'rule_name'"),
                Token::Token => write!(f, "'TOKEN_NAME'"),
                Token::String => write!(f, "\"string...\""),
                Token::Regexp => write!(f, "/regexp.../"),
                Token::Number => write!(f, "a number"),
                Token::HexNumber => write!(f, "a 0x-hex-number"),
                Token::Newline => write!(f, "'\\n'"),
                Token::SpecialToken => write!(f, "<special_token>"),
                Token::GrammarRef => write!(f, "@grammar_name"),
                _ => write!(f, "{self:?}"),
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub enum LexemeValue {
    #[default]
    None,
    String(String),
    Json(Value),
    Regex(RegexExt),
}

impl Display for LexemeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexemeValue::String(s) => {
                use std::fmt::Write;
                f.write_char('"')?;
                for chunk in s.as_bytes()[..s.len().min(100)].utf8_chunks() {
                    for ch in chunk
                        .valid()
                        .chars()
                        .chain((!chunk.invalid().is_empty()).then_some('\u{fffd}'))
                    {
                        if ch == '\'' {
                            f.write_char(ch)?;
                        } else {
                            for ch in ch.escape_debug() {
                                f.write_char(ch)?;
                            }
                        }
                    }
                }
                if s.len() > 100 {
                    f.write_str("...")?;
                }
                f.write_char('"')
            }
            _ => write!(f, "{{ ...json... }}"),
        }
    }
}

/// Represents a lexeme with its token type, value, and position.
#[derive(Debug)]
pub struct Lexeme {
    pub token: Token,
    pub value: LexemeValue,
    pub line: usize,
    pub column: usize,
}

impl Lexeme {
    pub fn take(&mut self) -> Self {
        Lexeme {
            token: self.token,
            value: std::mem::take(&mut self.value),
            line: self.line,
            column: self.column,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Location {
    pub line: usize,
    pub column: usize,
    pub src: Rc<String>,
    pub funding: ParserAllocationFunding,
}

pub(crate) struct HighlightLocation<'a> {
    source: &'a str,
    line: usize,
    column: usize,
}
pub(crate) fn highlight_location(
    source: &str,
    line: usize,
    column: usize,
) -> HighlightLocation<'_> {
    HighlightLocation {
        source,
        line,
        column,
    }
}
impl Display for HighlightLocation<'_> {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let start = self.line.saturating_sub(3);
        let count = self.line.saturating_add(2).saturating_sub(start);
        for (index, line) in self.source.lines().enumerate().skip(start).take(count) {
            let actual = index + 1;
            writeln!(output, "{actual:>4} | {line}")?;
            if actual == self.line {
                let digits = actual.checked_ilog10().unwrap_or(0) as usize + 1;
                let width = digits
                    .max(4)
                    .saturating_add(3)
                    .saturating_add(self.column.saturating_sub(1));
                writeln!(output, "{:width$}^", "")?;
            }
        }
        Ok(())
    }
}

impl Location {
    pub fn augment(&self, error: ParserError) -> ParserError {
        if crate::earley::is_grammar_storage_failure(&error) {
            return error;
        }
        error.annotate(
            format_args!("at {}({}): ", self.line, self.column),
            format_args!(
                "\n{}",
                highlight_location(&self.src, self.line, self.column)
            ),
            &self.funding,
        )
    }
}

impl Token {
    const LITERAL_TOKENS: &'static [(Token, &'static str)] = &[
        (Token::Arrow, "->"),
        (Token::Colon, ":"),
        (Token::Comma, ","),
        (Token::Dot, "."),
        (Token::DotDot, ".."),
        (Token::KwDeclare, "%declare"),
        (Token::KwLLGuidance, "%llguidance"),
        (Token::KwIgnore, "%ignore"),
        (Token::KwImport, "%import"),
        (Token::KwOverride, "%override"),
        (Token::KwJson, "%json"),
        (Token::KwRegex, "%regex"),
        (Token::KwLark, "%lark"),
        (Token::KwIf, "%if"),
        (Token::LParen, "("),
        (Token::RParen, ")"),
        (Token::LBrace, "{"),
        (Token::RBrace, "}"),
        (Token::LBracket, "["),
        (Token::RBracket, "]"),
        (Token::Tilde, "~"),
        (Token::VBar, "|"),
        (Token::And, "&"),
        (Token::Equals, "="),
        (Token::Hash, "#"),
        (Token::Underscore, "_"),
        (Token::DoubleColon, "::"),
    ];

    const REGEX_TOKENS: &'static [(Token, &'static str)] = &[
        (Token::Op, r"[+*?]"),
        (Token::Rule, r"!?[_?]?[a-z][_a-z0-9\-]*"),
        (Token::Token, r"_?[A-Z][_A-Z0-9\-]*"),
        // use JSON string syntax
        (
            Token::String,
            r#""(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4}|x[a-fA-F0-9]{2})|[^\"\\\x00-\x1F\x7F])*"(i|)"#,
        ),
        (Token::Regexp, r#"/(\\.|[^/\\])+/[imslux]*"#),
        (Token::Number, r#"[+-]?[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?"#),
        (Token::HexNumber, r#"0[xX][0-9a-fA-F]+"#),
        (Token::Newline, r"(\r?\n)+[ \t]*"),
        (Token::SpecialToken, r"<[^<>\s]+>"),
        (Token::GrammarRef, r"@[a-zA-Z0-9_\-]+"),
    ];
}

pub fn lex_lark(input: &str, funding: &ParserAllocationFunding) -> Result<Vec<Lexeme>> {
    let reserve = |bytes| funding.reserve(bytes);
    let comment_or_ws = funding.try_copy_str(r"((#|//)[^\n]*)|[ \t]+")?;
    let mut spec = LexerSpec::new(funding.clone())?;
    let cls =
        spec.setup_lexeme_class_with_skip(SkipSpec::unbounded(RegexAst::Regex(comment_or_ws)))?;
    let mut lexeme_idx_to_token = HashMap::default();
    funding.try_insert(&mut lexeme_idx_to_token, spec.skip_id(cls), Token::SKIP)?;
    for (token, literal) in Token::LITERAL_TOKENS {
        let l = spec.add_simple_literal(
            funding.try_format(format_args!("{token:?}"))?,
            literal,
            false,
        )?;
        funding.try_insert(&mut lexeme_idx_to_token, l, *token)?;
    }
    for (token, regexp) in Token::REGEX_TOKENS {
        let l = spec.add_greedy_lexeme(
            funding.try_format(format_args!("{token:?}"))?,
            RegexAst::Regex(funding.try_copy_str(regexp)?),
            false,
            None,
            usize::MAX,
        )?;
        funding.try_insert(&mut lexeme_idx_to_token, l, *token)?;
    }
    let mut limits = ParserLimits::default();
    funding.reserve(
        LexerSpec::root_source_inspection_control_bytes()
            .ok_or_else(|| funding.storage_overflow())?,
    )?;
    let root_plan = spec
        .root_source_plan()
        .map_err(|error| derivre::ParserError::cause(error, funding))?;
    funding.reserve(root_plan.requirements().required_bytes())?;
    let roots = root_plan
        .compile()
        .map_err(|error| derivre::ParserError::cause(error, funding))?;
    let mask_plan = TokenMaskConstructionPlan::zeroed(spec.lexemes.len())
        .map_err(|error| derivre::ParserError::cause(error, funding))?;
    funding.reserve(mask_plan.requirements().required_bytes())?;
    let mut mask = mask_plan
        .compile()
        .map_err(|error| derivre::ParserError::cause(error, funding))?;
    mask.set_all(true);
    let all_lexemes = LexemeSet::from_owned_vob(mask);
    let input_source = roots
        .ordinary(spec.regex_builder.into_exprset())
        .map_err(|error| derivre::ParserError::cause(error, funding))?;
    let vector = PreparedRegexVector::prepare_with_backing(
        input_source,
        &mut limits,
        Some(funding.clone()),
        &reserve,
    )
    .map_err(|error| derivre::ParserError::cause(error, funding))?;
    let mut lexer = PreparedLexer::prepare(vector, &reserve)
        .map_err(|error| derivre::ParserError::cause(error, funding))?;
    let state0 = lexer
        .start_state(&all_lexemes, &reserve)
        .map_err(|error| derivre::ParserError::cause(error, funding))?;
    let mut line_no = 1;
    let mut column_no = 1;
    let mut curr_lexeme = Lexeme {
        token: Token::EOF,
        value: LexemeValue::default(),
        line: 1,
        column: 1,
    };
    let mut state = state0;
    let mut lexemes = Vec::new();
    let mut start_idx = 0;

    let input = funding.try_format(format_args!("{input}\n"))?;
    let input_bytes = input.as_bytes();

    let mut idx = 0;

    while idx <= input_bytes.len() {
        let mut b = b'\n';
        let res = if idx == input_bytes.len() {
            lexer
                .try_lexeme_end(state, &reserve)
                .map_err(|error| derivre::ParserError::cause(error, funding))?
        } else {
            b = input_bytes[idx];
            lexer
                .advance(state, b, &reserve)
                .map_err(|error| derivre::ParserError::cause(error, funding))?
        };

        match res {
            LexerResult::Error => {
                bail!(
                    funding,
                    "{}({}): lexer error\n{}",
                    line_no,
                    column_no,
                    highlight_location(&input, line_no, column_no)
                );
            }
            LexerResult::SpecialToken(_) => {
                bail!(
                    funding,
                    "{}({}): lexer special token\n{}",
                    line_no,
                    column_no,
                    highlight_location(&input, line_no, column_no)
                );
            }
            LexerResult::State(s, _) => {
                state = s;
            }
            LexerResult::Lexeme(p) => {
                let transition_byte = if p.byte_next_row { p.byte } else { None };
                let lx_idx = crate::earley::lexer::matching(&spec.lexemes, p.idx, |state| {
                    lexer.vector().state_desc(state)
                })
                .and_then(|matches| matches.first())
                .ok_or_else(|| {
                    anyhow!(funding, "fixed Lark lexical source has no matching token")
                })?;

                let token = lexeme_idx_to_token[&lx_idx];
                curr_lexeme.token = token;
                let mut end_idx = if p.byte_next_row || p.byte.is_none() {
                    idx
                } else {
                    idx + 1
                };

                let raw_value = &input[start_idx..end_idx];

                curr_lexeme.value = if token == Token::KwJson
                    || token == Token::KwLLGuidance
                    || token == Token::KwRegex
                {
                    let inp_slice = &input_bytes[end_idx..];
                    let (lexeme_value, n_bytes) = if token == Token::KwRegex {
                        let (v, n) = super::json_value::parse_regex(inp_slice, funding)?;
                        (LexemeValue::Regex(v), n)
                    } else {
                        let (v, n) = super::json_value::parse(inp_slice, funding)?;
                        (LexemeValue::Json(v), n)
                    };

                    start_idx = end_idx;
                    end_idx += n_bytes;
                    for &b in &input_bytes[start_idx..end_idx - 1] {
                        if b == b'\n' {
                            line_no += 1;
                            column_no = 1;
                        } else {
                            column_no += 1;
                        }
                    }
                    // make sure we account the line ending properly at the end of the loop
                    idx = end_idx - 1;
                    b = input_bytes[idx];
                    lexeme_value
                } else {
                    LexemeValue::String(funding.try_copy_str(raw_value)?)
                };

                start_idx = end_idx;

                // println!("lex: {:?}", curr_lexeme);

                if curr_lexeme.token != Token::SKIP {
                    funding.try_push(&mut lexemes, curr_lexeme.take())?;
                }

                state = lexer
                    .start_state(&all_lexemes, &reserve)
                    .map_err(|error| derivre::ParserError::cause(error, funding))?;
                state = lexer
                    .transition_start_state(state, transition_byte, &reserve)
                    .map_err(|error| derivre::ParserError::cause(error, funding))?;

                curr_lexeme.line = line_no;
                curr_lexeme.column = column_no;
            }
        }

        if b == b'\n' {
            line_no += 1;
            column_no = 1;
        } else {
            column_no += 1;
        }
        idx += 1;
    }

    Ok(lexemes)
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn highlights_keep_two_context_lines_and_unicode_source_text() {
        let source = "one\ntwø\nthree\nfour\nfive\nsix";
        assert_eq!(format!("{}", highlight_location(source, 1, 2)),
            "   1 | one\n        ^\n   2 | twø\n   3 | three\n");
        assert_eq!(format!("{}", highlight_location(source, 6, 1)),
            "   4 | four\n   5 | five\n   6 | six\n       ^\n");
    }
}
