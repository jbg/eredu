//! One non-overlapping UTF-8 replacement traversal for both filter destinations.
#![forbid(unsafe_code)]
use std::fmt::{self, Write};
pub(crate) fn plain<W: Write>(output: &mut W, value: &str, from: &str, to: &str) -> fmt::Result {
    let mut end = 0;
    // The empty pattern visits UTF-8 boundaries exactly as str::replace does.
    for (start, matched) in value.match_indices(from) {
        output.write_str(&value[end..start])?;
        output.write_str(to)?;
        end = start + matched.len();
    }
    output.write_str(&value[end..])
}
pub(crate) fn control_bytes<W: Write>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<(&mut W, &str, &str, &str)>(),
        size_of::<std::str::MatchIndices<'_, &str>>(),
        size_of::<Option<(usize, &str)>>(),
        size_of::<(usize, usize, &str)>(),
        size_of::<fmt::Result>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
#[cfg(test)]
mod tests {
    #[test]
    fn replacement_segments_match_standard_nonoverlapping_unicode_replacement() {
        for value in ["", "aaa", "é\u{301}界🦀\0é", "  \u{2003} x  "] {
            for from in ["", "aa", "é", "\0", "🦀", "absent"] {
                for to in ["", "·", "界\0", "aaa", "\u{2003}"] {
                    let mut actual = String::new();
                    super::plain(&mut actual, value, from, to).unwrap();
                    assert_eq!(actual, value.replace(from, to));
                }
            }
        }
    }
}
