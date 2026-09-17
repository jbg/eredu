//! Source-bound token and literal construction through the ordinary lexer.
//!
//! This is the default-syntax lexical frontend, not a complete template compiler.
//! A plan derives all capacities from its borrowed source and owns no allocation.
//! Construction performs each destination reserve once, retaining all destinations
//! and source borrows on every recoverable error. It does not grant admission.
//!
//! ```compile_fail
//! use minijinja::bounded::frontend::{Plan, WhitespaceConfig};
//! let tokens = {
//!     let source = String::from("{{ 'hello' }}");
//!     Plan::inspect(&source, "name", false, WhitespaceConfig::default())
//!         .unwrap().construct().unwrap()
//! };
//! println!("{}", tokens.len());
//! ```
#![forbid(unsafe_code)]

use std::alloc::Layout;
use std::collections::TryReserveError;
use std::fmt;
use std::mem::size_of;

use crate::compiler::lexer::literal::{parse_number, LexError, Lexeme, Number};
use crate::compiler::lexer::scanner::{Scanner, Syntax};
pub use crate::compiler::lexer::WhitespaceConfig;
pub use crate::compiler::tokens::Span;
use crate::error::ErrorKind;
use crate::utils::literal::{decode, Failure as EscapeFailure, Output};

/// A checked source geometry is not a claim that its literals are valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// Source offsets, error columns or a destination Layout cannot be represented.
    Geometry,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("lexical source geometry cannot be represented")
    }
}
impl std::error::Error for PlanError {}

/// One actual, independently reserved destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buffer {
    /// Token records with inline wide integers and source borrows.
    Tokens,
    /// UTF-8 decoded escaped literals, including any failing literal prefix.
    Literals,
    /// Reused underscore-stripped number bytes.
    NumericScratch,
}

/// Exact requested heap layouts and concrete inline representations.
#[derive(Clone, Copy, Debug)]
pub struct Requirements {
    tokens: usize,
    literal_bytes: usize,
    numeric_bytes: usize,
    heap_bytes: usize,
}
impl Requirements {
    /// Maximum token records encountered before a definite scanner/escape error.
    pub fn tokens(self) -> usize {
        self.tokens
    }
    /// Decoded UTF-8 bytes through that same prefix.
    pub fn literal_bytes(self) -> usize {
        self.literal_bytes
    }
    /// Maximum stripped number length needing scratch in that prefix.
    pub fn numeric_bytes(self) -> usize {
        self.numeric_bytes
    }
    /// Sum of the three checked requested array layouts; no allocator rounding claim.
    pub fn heap_bytes(self) -> usize {
        self.heap_bytes
    }
    /// Closed source-plan representation, separate from retained construction results.
    pub fn plan_control_bytes(self) -> usize {
        size_of::<Plan<'static>>()
    }
    /// Larger of the concrete retained success/error representations, including Vec shells.
    pub fn retained_control_bytes(self) -> usize {
        size_of::<Tokenized<'static>>().max(size_of::<ConstructError<'static>>())
    }
    /// Concrete scanner state; literal/standard-number conversion uses fixed stack storage.
    pub fn scanner_control_bytes(self) -> usize {
        size_of::<Scanner<'static>>()
    }
}

fn requirements(
    tokens: usize,
    literal_bytes: usize,
    numeric_bytes: usize,
) -> Result<Requirements, PlanError> {
    let t = Layout::array::<Record<'static>>(tokens)
        .map_err(|_| PlanError::Geometry)?
        .size();
    let b = Layout::array::<u8>(literal_bytes)
        .map_err(|_| PlanError::Geometry)?
        .size();
    let n = Layout::array::<u8>(numeric_bytes)
        .map_err(|_| PlanError::Geometry)?
        .size();
    let heap_bytes = t
        .checked_add(b)
        .and_then(|v| v.checked_add(n))
        .ok_or(PlanError::Geometry)?;
    Ok(Requirements {
        tokens,
        literal_bytes,
        numeric_bytes,
        heap_bytes,
    })
}

pub(crate) fn check_source_geometry(source: &str) -> Result<(), PlanError> {
    // Ordinary spans have u32 offsets and saturating u16 columns. Reserve
    // one representable column/byte for their syntax-error extension.
    if source.len() >= u32::MAX as usize || source.len() > isize::MAX as usize {
        return Err(PlanError::Geometry);
    }
    let mut column = 0usize;
    for c in source.chars() {
        if c == '\n' {
            column = 0;
        } else {
            column += 1;
            if column >= u16::MAX as usize {
                return Err(PlanError::Geometry);
            }
        }
    }
    Ok(())
}

/// A source-derived plan. It cannot be built from caller capacities or rebound.
/// Default delimiters are static: no SyntaxConfig global is initialized.
#[derive(Debug)]
pub struct Plan<'s> {
    source: &'s str,
    filename: &'s str,
    in_expr: bool,
    whitespace: WhitespaceConfig,
    requirements: Requirements,
}
impl<'s> Plan<'s> {
    /// Scan the exact source without allocating. Literal validity is reported by
    /// construction in ordinary source order, after reserves. Custom syntax is
    /// not an input of this default-syntax API and remains an unfinished profile.
    pub fn inspect(
        source: &'s str,
        filename: &'s str,
        in_expr: bool,
        whitespace: WhitespaceConfig,
    ) -> Result<Self, PlanError> {
        check_source_geometry(source)?;
        let mut scanner = Scanner::new(source, filename, in_expr, Syntax::Default, whitespace);
        let (mut tokens, mut literal_bytes, mut numeric_bytes) = (0usize, 0usize, 0usize);
        while let Ok(Some((token, _))) = scanner.next_token() {
            tokens = tokens.checked_add(1).ok_or(PlanError::Geometry)?;
            match token {
                Lexeme::Escaped(raw) => match decode(raw, Output::Count(&mut literal_bytes)) {
                    Ok(()) => (),
                    Err(EscapeFailure::BadEscape) => break,
                    Err(EscapeFailure::Capacity) => return Err(PlanError::Geometry),
                },
                Lexeme::Number(n) if n.has_underscore => {
                    numeric_bytes = numeric_bytes.max(n.raw.bytes().filter(|&b| b != b'_').count());
                }
                _ => (),
            }
        }
        Ok(Self {
            source,
            filename,
            in_expr,
            whitespace,
            requirements: requirements(tokens, literal_bytes, numeric_bytes)?,
        })
    }
    /// Exact requested destinations and actual control layouts.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Construct once, retaining all already reserved destinations on errors.
    pub fn construct(self) -> Result<Tokenized<'s>, ConstructError<'s>> {
        self.construct_inner(None)
    }

    fn construct_inner(self, fail: Option<Buffer>) -> Result<Tokenized<'s>, ConstructError<'s>> {
        let mut owner = Tokenized {
            source: self.source,
            filename: self.filename,
            requirements: self.requirements,
            records: Vec::new(),
            literals: Vec::new(),
            numeric: Vec::new(),
        };
        macro_rules! reserve {
            ($field:ident, $count:expr, $buffer:expr) => {{
                // Only private unit tests select failure; the request reaches this
                // actual destination's one try_reserve_exact, never another Vec.
                let count = if fail == Some($buffer) {
                    usize::MAX
                } else {
                    $count
                };
                if let Err(error) = owner.$field.try_reserve_exact(count) {
                    return Err(ConstructError {
                        owner,
                        cause: Cause::Reserve {
                            buffer: $buffer,
                            error,
                        },
                    });
                }
            }};
        }
        reserve!(records, self.requirements.tokens, Buffer::Tokens);
        reserve!(literals, self.requirements.literal_bytes, Buffer::Literals);
        reserve!(
            numeric,
            self.requirements.numeric_bytes,
            Buffer::NumericScratch
        );
        let mut scanner = Scanner::new(
            self.source,
            self.filename,
            self.in_expr,
            Syntax::Default,
            self.whitespace,
        );
        loop {
            let next = match scanner.next_token() {
                Ok(next) => next,
                Err(error) => {
                    return Err(ConstructError {
                        owner,
                        cause: if error.empty_stack {
                            Cause::ExpressionEnd
                        } else {
                            Cause::Lexical(LexicalError(error))
                        },
                    })
                }
            };
            let Some((token, span)) = next else {
                break;
            };
            let value = match token {
                Lexeme::Escaped(raw) => {
                    let start = owner.literals.len();
                    if let Err(error) = decode(
                        raw,
                        Output::Packed {
                            bytes: &mut owner.literals,
                            limit: self.requirements.literal_bytes,
                        },
                    ) {
                        let cause = match error {
                            EscapeFailure::BadEscape => Cause::Lexical(LexicalError(LexError {
                                empty_stack: false,
                                kind: ErrorKind::BadEscape,
                                detail: None,
                                span: None,
                            })),
                            EscapeFailure::Capacity => Cause::Capacity(Buffer::Literals),
                        };
                        return Err(ConstructError { owner, cause });
                    }
                    Value::String {
                        start,
                        end: owner.literals.len(),
                    }
                }
                Lexeme::Number(number) => {
                    owner.numeric.clear();
                    let raw = if number.has_underscore {
                        for byte in number.raw.bytes().filter(|&b| b != b'_') {
                            if owner.numeric.len() >= self.requirements.numeric_bytes
                                || owner.numeric.len() == owner.numeric.capacity()
                            {
                                return Err(ConstructError {
                                    owner,
                                    cause: Cause::Capacity(Buffer::NumericScratch),
                                });
                            }
                            owner.numeric.push(byte);
                        }
                        std::str::from_utf8(&owner.numeric).expect("ASCII numeric scanner")
                    } else {
                        number.raw
                    };
                    let value = match parse_number(raw, number) {
                        Ok(value) => value,
                        Err(detail) => {
                            return Err(ConstructError {
                                owner,
                                cause: Cause::Lexical(LexicalError(scanner.syntax_error(detail))),
                            })
                        }
                    };
                    Value::Number(value)
                }
                token => Value::Raw(token),
            };
            if owner.records.len() >= self.requirements.tokens
                || owner.records.len() == owner.records.capacity()
            {
                return Err(ConstructError {
                    owner,
                    cause: Cause::Capacity(Buffer::Tokens),
                });
            }
            owner.records.push(Record { value, span });
        }
        debug_assert_eq!(owner.records.len(), self.requirements.tokens);
        debug_assert_eq!(owner.literals.len(), self.requirements.literal_bytes);
        Ok(owner)
    }
}

#[derive(Debug)]
struct Record<'s> {
    value: Value<'s>,
    span: Span,
}
#[derive(Debug)]
enum Value<'s> {
    Raw(Lexeme<'s>),
    String { start: usize, end: usize },
    Number(Number),
}

/// Retained source-bound tokens. No Clone, raw Vec, ordinary Token or mutable escape.
#[derive(Debug)]
pub struct Tokenized<'s> {
    source: &'s str,
    filename: &'s str,
    requirements: Requirements,
    records: Vec<Record<'s>>,
    literals: Vec<u8>,
    numeric: Vec<u8>,
}
impl<'s> Tokenized<'s> {
    /// Original complete source, retaining exact identity through its borrow.
    pub fn source(&self) -> &'s str {
        self.source
    }
    /// Original filename borrow.
    pub fn filename(&self) -> &'s str {
        self.filename
    }
    /// Exact requested layouts.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Number of fully materialized records (also meaningful on a failure prefix).
    pub fn len(&self) -> usize {
        self.records.len()
    }
    /// Whether no record completed.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    /// Read-only token projections, including source spans and decoded text.
    pub fn tokens(&self) -> impl ExactSizeIterator<Item = TokenView<'_, 's>> {
        self.records.iter().map(move |record| TokenView {
            record,
            literals: &self.literals,
        })
    }
    /// Observed capacities for diagnostics; these do not authorize construction.
    pub fn capacities(&self) -> [usize; 3] {
        [
            self.records.capacity(),
            self.literals.capacity(),
            self.numeric.capacity(),
        ]
    }
}

/// An allocation-free ordinary lexical error, with the same kind/detail/span.
#[derive(Debug)]
pub struct LexicalError(LexError);
impl LexicalError {
    /// Ordinary error kind.
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }
    /// Ordinary static error detail (none for bad escapes).
    pub fn detail(&self) -> Option<&'static str> {
        self.0.detail
    }
    /// Ordinary extended syntax span; bad escapes have no source annotation.
    pub fn span(&self) -> Option<Span> {
        self.0.span.map(|mut span| {
            if span.start_col == span.end_col {
                span.end_col += 1;
                span.end_offset += 1;
            }
            span
        })
    }
}
impl fmt::Display for LexicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.detail().unwrap_or("bad escape"))
    }
}
impl std::error::Error for LexicalError {}

/// Actual reserve errors and ordered lexical failures retain their full owner.
#[derive(Debug)]
pub enum Cause {
    /// The named actual destination failed its sole reservation.
    Reserve {
        /// Destination attempted after preceding reservations succeeded.
        buffer: Buffer,
        /// Original TryReserveError, without allocating error erasure.
        error: TryReserveError,
    },
    /// The shared lexical/literal worker's actual failure.
    Lexical(LexicalError),
    /// A preallocated destination would be exceeded; no growth occurs.
    Capacity(Buffer),
    /// Content follows an explicit expression terminator. Ordinary standalone
    /// tokenization panics with an empty stack here; closed construction rejects.
    ExpressionEnd,
}
/// By-value failure holding every actual partial/completed allocation and source borrow.
#[derive(Debug)]
pub struct ConstructError<'s> {
    owner: Tokenized<'s>,
    cause: Cause,
}
impl<'s> ConstructError<'s> {
    /// Closed actual cause.
    pub fn cause(&self) -> &Cause {
        &self.cause
    }
    /// Read-only retained prefix and capacities.
    pub fn partial(&self) -> &Tokenized<'s> {
        &self.owner
    }
}
impl fmt::Display for ConstructError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Reserve { buffer, error } => {
                write!(f, "lexical {buffer:?} reserve failed: {error}")
            }
            Cause::Lexical(error) => error.fmt(f),
            Cause::ExpressionEnd => f.write_str("content follows the expression terminator"),
            Cause::Capacity(buffer) => write!(f, "lexical {buffer:?} capacity exhausted"),
        }
    }
}
impl std::error::Error for ConstructError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Reserve { error, .. } => Some(error),
            Cause::Lexical(error) => Some(error),
            Cause::Capacity(_) | Cause::ExpressionEnd => None,
        }
    }
}

/// A borrowed projection that cannot outlive either token storage or its source.
#[derive(Clone, Copy)]
pub struct TokenView<'a, 's> {
    record: &'a Record<'s>,
    literals: &'a [u8],
}
impl<'a, 's> TokenView<'a, 's> {
    /// Exact scanner span.
    pub fn span(self) -> Span {
        self.record.span
    }
    /// Borrowed template/identifier/string text, including decoded owned strings.
    pub fn text(self) -> Option<&'a str> {
        match &self.record.value {
            Value::Raw(Lexeme::TemplateData(s) | Lexeme::Ident(s) | Lexeme::Str(s)) => Some(s),
            Value::String { start, end } => {
                Some(std::str::from_utf8(&self.literals[*start..*end]).expect("decoded UTF-8"))
            }
            _ => None,
        }
    }
    /// Integer value with the ordinary distinction available from kind().
    pub fn integer(self) -> Option<u128> {
        match self.record.value {
            Value::Number(Number::Int(n)) => Some(n as u128),
            Value::Number(Number::Int128(n)) => Some(n),
            _ => None,
        }
    }
    /// Actual standard-library float value (including ordinary overflow to infinity).
    pub fn float(self) -> Option<f64> {
        match self.record.value {
            Value::Number(Number::Float(n)) => Some(n),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;

/// Ordinary token discriminants, independent of storage ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    /// Ordinary TemplateData token.
    TemplateData,
    /// Ordinary VariableStart token.
    VariableStart,
    /// Ordinary VariableEnd token.
    VariableEnd,
    /// Ordinary BlockStart token.
    BlockStart,
    /// Ordinary BlockEnd token.
    BlockEnd,
    /// Ordinary Ident token.
    Ident,
    /// Ordinary Str token.
    Str,
    /// Ordinary String token.
    String,
    /// Ordinary Int token.
    Int,
    /// Ordinary Int128 token.
    Int128,
    /// Ordinary Float token.
    Float,
    /// Ordinary Plus token.
    Plus,
    /// Ordinary Minus token.
    Minus,
    /// Ordinary Mul token.
    Mul,
    /// Ordinary Div token.
    Div,
    /// Ordinary FloorDiv token.
    FloorDiv,
    /// Ordinary Pow token.
    Pow,
    /// Ordinary Mod token.
    Mod,
    /// Ordinary Dot token.
    Dot,
    /// Ordinary Comma token.
    Comma,
    /// Ordinary Colon token.
    Colon,
    /// Ordinary Tilde token.
    Tilde,
    /// Ordinary Assign token.
    Assign,
    /// Ordinary Pipe token.
    Pipe,
    /// Ordinary Eq token.
    Eq,
    /// Ordinary Ne token.
    Ne,
    /// Ordinary Gt token.
    Gt,
    /// Ordinary Gte token.
    Gte,
    /// Ordinary Lt token.
    Lt,
    /// Ordinary Lte token.
    Lte,
    /// Ordinary BracketOpen token.
    BracketOpen,
    /// Ordinary BracketClose token.
    BracketClose,
    /// Ordinary ParenOpen token.
    ParenOpen,
    /// Ordinary ParenClose token.
    ParenClose,
    /// Ordinary BraceOpen token.
    BraceOpen,
    /// Ordinary BraceClose token.
    BraceClose,
}

impl TokenView<'_, '_> {
    /// Ordinary token kind, preserving Str/String and Int/Int128 distinctions.
    pub fn kind(self) -> TokenKind {
        match &self.record.value {
            Value::Raw(Lexeme::TemplateData(_)) => TokenKind::TemplateData,
            Value::Raw(Lexeme::VariableStart) => TokenKind::VariableStart,
            Value::Raw(Lexeme::VariableEnd) => TokenKind::VariableEnd,
            Value::Raw(Lexeme::BlockStart) => TokenKind::BlockStart,
            Value::Raw(Lexeme::BlockEnd) => TokenKind::BlockEnd,
            Value::Raw(Lexeme::Ident(_)) => TokenKind::Ident,
            Value::Raw(Lexeme::Str(_)) => TokenKind::Str,
            Value::String { .. } => TokenKind::String,
            Value::Number(Number::Int(_)) => TokenKind::Int,
            Value::Number(Number::Int128(_)) => TokenKind::Int128,
            Value::Number(Number::Float(_)) => TokenKind::Float,
            Value::Raw(Lexeme::Plus) => TokenKind::Plus,
            Value::Raw(Lexeme::Minus) => TokenKind::Minus,
            Value::Raw(Lexeme::Mul) => TokenKind::Mul,
            Value::Raw(Lexeme::Div) => TokenKind::Div,
            Value::Raw(Lexeme::FloorDiv) => TokenKind::FloorDiv,
            Value::Raw(Lexeme::Pow) => TokenKind::Pow,
            Value::Raw(Lexeme::Mod) => TokenKind::Mod,
            Value::Raw(Lexeme::Dot) => TokenKind::Dot,
            Value::Raw(Lexeme::Comma) => TokenKind::Comma,
            Value::Raw(Lexeme::Colon) => TokenKind::Colon,
            Value::Raw(Lexeme::Tilde) => TokenKind::Tilde,
            Value::Raw(Lexeme::Assign) => TokenKind::Assign,
            Value::Raw(Lexeme::Pipe) => TokenKind::Pipe,
            Value::Raw(Lexeme::Eq) => TokenKind::Eq,
            Value::Raw(Lexeme::Ne) => TokenKind::Ne,
            Value::Raw(Lexeme::Gt) => TokenKind::Gt,
            Value::Raw(Lexeme::Gte) => TokenKind::Gte,
            Value::Raw(Lexeme::Lt) => TokenKind::Lt,
            Value::Raw(Lexeme::Lte) => TokenKind::Lte,
            Value::Raw(Lexeme::BracketOpen) => TokenKind::BracketOpen,
            Value::Raw(Lexeme::BracketClose) => TokenKind::BracketClose,
            Value::Raw(Lexeme::ParenOpen) => TokenKind::ParenOpen,
            Value::Raw(Lexeme::ParenClose) => TokenKind::ParenClose,
            Value::Raw(Lexeme::BraceOpen) => TokenKind::BraceOpen,
            Value::Raw(Lexeme::BraceClose) => TokenKind::BraceClose,
            Value::Raw(Lexeme::Escaped(_) | Lexeme::Number(_)) => {
                unreachable!("materialized literals")
            }
        }
    }
}
