fn class_for(c: char) -> Option<&'static str> {
    match c {
        'd' => Some("0-9"),
        'w' => Some("0-9a-zA-Z_"),
        's' => Some(" \\t\\n\\r\\f\\v"),
        _ => None,
    }
}

/// Make sure given regex can be used inside /.../ in Lark syntax.
/// Also if `use_ascii.contains('d')` replace `\d` with `[0-9]` and `\D` with `[^0-9]`.
/// Similarly for `\w`/`\W` (`[0-9a-zA-Z_]`) and `\s`/`\S` (`[ \t\n\r\f\v]`).
/// For standard Unicode Python3 or Rust regex crate semantics `use_ascii = ""`
/// For JavaScript or JSON Schema semantics `use_ascii = "dw"`
/// For Python2 or byte patters in Python3 semantics `use_ascii = "dws"`
/// More flags may be added in future.
pub fn regex_to_lark<'a>(rx: &'a str, use_ascii: &'a str) -> RegexToLark<'a> {
    RegexToLark { rx, use_ascii }
}

/// Borrowed rendering of the existing escape/ASCII-class policy. Destination
/// ownership and its allocation policy belong to the caller.
pub struct RegexToLark<'a> { rx: &'a str, use_ascii: &'a str }
impl std::fmt::Display for RegexToLark<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut is_q = false;
    for c in self.rx.chars() {
        let prev_q = is_q;
        is_q = false;
        match c {
            // make sure we don't terminate on /
            '/' => std::fmt::Write::write_str(f, "\\/")?,

            // these are optional, but nice
            '\n' => std::fmt::Write::write_str(f, "\\n")?,
            '\r' => std::fmt::Write::write_str(f, "\\r")?,
            '\t' => std::fmt::Write::write_str(f, "\\t")?,

            '\\' if !prev_q => {
                is_q = true;
            }

            'd' | 'w' | 's' | 'D' | 'W' | 'S' if prev_q => {
                let c2 = c.to_ascii_lowercase();
                if self.use_ascii.contains(c2) {
                    let class = class_for(c2).unwrap();
                    std::fmt::Write::write_char(f, '[')?;
                    if c != c2 {
                        std::fmt::Write::write_char(f, '^')?;
                    }
                    std::fmt::Write::write_str(f, class)?;
                    std::fmt::Write::write_char(f, ']')?;
                } else {
                    std::fmt::Write::write_char(f, '\\')?;
                    std::fmt::Write::write_char(f, c)?;
                }
            }

            _ => {
                if prev_q {
                    std::fmt::Write::write_char(f, '\\')?;
                }
                std::fmt::Write::write_char(f, c)?;
            }
        }
    }
    Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digit_conversion_with_ascii() {
        // \d => [0-9], \D => [^0-9]
        assert_eq!(regex_to_lark(r"\d", "d").to_string(), "[0-9]");
        assert_eq!(regex_to_lark(r"\D", "d").to_string(), "[^0-9]");
    }

    #[test]
    fn test_word_conversion_with_ascii() {
        // Only convert if use_ascii contains corresponding letter.
        assert_eq!(regex_to_lark(r"\w", "w").to_string(), "[0-9a-zA-Z_]");
        assert_eq!(regex_to_lark(r"\W", "w").to_string(), "[^0-9a-zA-Z_]");
    }

    #[test]
    fn test_space_conversion_with_ascii() {
        // \s and \S should convert accordingly.
        assert_eq!(regex_to_lark(r"\s", "s").to_string(), "[ \\t\\n\\r\\f\\v]");
        assert_eq!(regex_to_lark(r"\S", "s").to_string(), "[^ \\t\\n\\r\\f\\v]");
    }

    #[test]
    fn test_no_conversion_when_missing_in_use_ascii() {
        // If the ascii flag doesn't contain the letter, leave escape as-is.
        assert_eq!(regex_to_lark(r"\d", "").to_string(), r"\d");
        assert_eq!(regex_to_lark(r"\w", "d").to_string(), r"\w");
    }

    #[test]
    fn test_escaped_slashes_and_whitespace() {
        // '/' should be escaped; newline, tab, carriage return are escaped.
        let input = "/a\nb\rc\td";
        let expected = r"\/a\nb\rc\td";
        assert_eq!(regex_to_lark(input, "dws").to_string(), expected);
    }

    #[test]
    fn test_combined_conversions() {
        // Combined sequence with all conversions.
        let input = r"\d\w\s\D\W\S";
        let expected = "[0-9][0-9a-zA-Z_][ \\t\\n\\r\\f\\v][^0-9][^0-9a-zA-Z_][^ \\t\\n\\r\\f\\v]";
        assert_eq!(regex_to_lark(input, "dws").to_string(), expected);
    }

    #[test]
    fn test_miscellaneous_escapes() {
        // \X and \@ are not recognized as special, so they should pass through.
        assert_eq!(regex_to_lark(r"\X", "").to_string(), r"\X");
        assert_eq!(regex_to_lark(r"\@", "").to_string(), r"\@");

        // Forward slash is escaped.
        assert_eq!(regex_to_lark(r"/", "").to_string(), r"\/");
        assert_eq!(regex_to_lark(r"\/", "").to_string(), r"\/");
        assert_eq!(regex_to_lark(r"\//", "").to_string(), r"\/\/");
        assert_eq!(regex_to_lark(r"/\//", "").to_string(), r"\/\/\/");

        // Double backslash should be preserved.
        assert_eq!(regex_to_lark(r"\\", "").to_string(), r"\\");

        // Quotes should pass through unchanged.
        assert_eq!(regex_to_lark("\"", "").to_string(), "\"");
        assert_eq!(regex_to_lark(r#"a"b"#, "").to_string(), r#"a"b"#);
    }
}
