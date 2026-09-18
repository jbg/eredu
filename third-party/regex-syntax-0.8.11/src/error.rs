use crate::{ast, hir};

/// This error type encompasses any error that can be returned by this crate.
///
/// This error type is marked as `non_exhaustive`. This means that adding a
/// new variant is not considered a breaking change.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    /// An error that occurred while translating concrete syntax into abstract
    /// syntax (AST).
    Parse(ast::Error),
    /// An error that occurred while translating abstract syntax into a high
    /// level intermediate representation (HIR).
    Translate(hir::Error),
}

impl From<ast::Error> for Error {
    fn from(err: ast::Error) -> Error {
        Error::Parse(err)
    }
}

impl From<hir::Error> for Error {
    fn from(err: hir::Error) -> Error {
        Error::Translate(err)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Error::Parse(ref x) => x.fmt(f),
            Error::Translate(ref x) => x.fmt(f),
        }
    }
}

/// A helper type for formatting nice error messages.
///
/// This type is responsible for reporting regex parse errors in a nice human
/// readable format. Most of its complexity is from interspersing notational
/// markers pointing out the position where an error occurred.
#[derive(Debug)]
pub struct Formatter<'e, E> {
    /// The original regex pattern in which the error occurred.
    pattern: &'e str,
    /// The error kind. It must impl fmt::Display.
    err: &'e E,
    /// The primary span of the error.
    span: &'e ast::Span,
    /// An auxiliary and optional span, in case the error needs to point to
    /// two locations (e.g., when reporting a duplicate capture group name).
    aux_span: Option<&'e ast::Span>,
}

impl<'e> From<&'e ast::Error> for Formatter<'e, ast::ErrorKind> {
    fn from(err: &'e ast::Error) -> Self {
        Formatter {
            pattern: err.pattern(),
            err: err.kind(),
            span: err.span(),
            aux_span: err.auxiliary_span(),
        }
    }
}

impl<'e> From<&'e hir::Error> for Formatter<'e, hir::ErrorKind> {
    fn from(err: &'e hir::Error) -> Self {
        Formatter {
            pattern: err.pattern(),
            err: err.kind(),
            span: err.span(),
            aux_span: None,
        }
    }
}

impl<'e, E: core::fmt::Display> core::fmt::Display for Formatter<'e, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Errors have at most two spans. Keep them on the stack and stream
        // notation directly into the caller's writer. Formatting a funded
        // refusal must not allocate an unfunded second diagnostic tree.
        let mut spans = [Some(self.span), self.aux_span];
        if let (Some(a), Some(b)) = (spans[0], spans[1]) {
            if a > b {
                spans.swap(0, 1);
            }
        }
        let multi_line = self.pattern.contains('\n');
        let mut line_count = self.pattern.lines().count();
        if self.pattern.ends_with('\n') {
            line_count += 1;
        }
        let width = if line_count <= 1 {
            0
        } else {
            let mut digits = 1;
            while line_count >= 10 {
                digits += 1;
                line_count /= 10;
            }
            digits
        };
        writeln!(f, "regex parse error:")?;
        if multi_line {
            repeat(f, "~", 79)?;
            writeln!(f)?;
        }
        for (i, line) in self.pattern.lines().enumerate() {
            if width > 0 {
                write!(f, "{:>width$}: ", i + 1)?;
            } else {
                f.write_str("    ")?;
            }
            writeln!(f, "{line}")?;
            let mut annotated = false;
            let mut position = 0;
            for span in spans
                .into_iter()
                .flatten()
                .filter(|span| span.is_one_line() && span.start.line == i + 1)
            {
                if !annotated {
                    repeat(f, " ", if width == 0 { 4 } else { width + 2 })?;
                    annotated = true;
                }
                let start = span.start.column - 1;
                if position < start {
                    repeat(f, " ", start - position)?;
                    position = start;
                }
                let length =
                    span.end.column.saturating_sub(span.start.column).max(1);
                repeat(f, "^", length)?;
                position += length;
            }
            if annotated {
                writeln!(f)?;
            }
        }
        if multi_line {
            repeat(f, "~", 79)?;
            writeln!(f)?;
            for span in spans
                .into_iter()
                .flatten()
                .filter(|span| !span.is_one_line())
            {
                writeln!(
                    f,
                    "on line {} (column {}) through line {} (column {})",
                    span.start.line,
                    span.start.column,
                    span.end.line,
                    span.end.column - 1
                )?;
            }
        }
        write!(f, "error: {}", self.err)
    }
}

fn repeat(
    f: &mut core::fmt::Formatter<'_>,
    text: &str,
    count: usize,
) -> core::fmt::Result {
    for _ in 0..count {
        f.write_str(text)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use crate::ast::parse::Parser;

    fn assert_panic_message(pattern: &str, expected_msg: &str) {
        let result = Parser::new().parse(pattern);
        match result {
            Ok(_) => {
                panic!("regex should not have parsed");
            }
            Err(err) => {
                assert_eq!(err.to_string(), expected_msg.trim());
            }
        }
    }

    // See: https://github.com/rust-lang/regex/issues/464
    #[test]
    fn regression_464() {
        let err = Parser::new().parse("a{\n").unwrap_err();
        // This test checks that the error formatter doesn't panic.
        assert!(!err.to_string().is_empty());
    }

    // See: https://github.com/rust-lang/regex/issues/545
    #[test]
    fn repetition_quantifier_expects_a_valid_decimal() {
        assert_panic_message(
            r"\\u{[^}]*}",
            r#"
regex parse error:
    \\u{[^}]*}
        ^
error: repetition quantifier expects a valid decimal
"#,
        );
    }
}
