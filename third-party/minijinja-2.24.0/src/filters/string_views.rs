//! Borrowed string workers shared with ordinary filters. The caller chooses
//! whether each resulting slice is materialized or retained as an input view.
use std::ops::Range;

pub(crate) enum SplitParts<'a> {
    Whitespace(std::str::SplitWhitespace<'a>),
    Separator(std::str::Split<'a, &'a str>),
    LimitedWhitespace(crate::utils::SplitWhitespaceN<'a>),
    LimitedSeparator(std::str::SplitN<'a,&'a str>),
}
impl<'a> Iterator for SplitParts<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Whitespace(parts) => parts.next(),
            Self::Separator(parts) => parts.next(),
            Self::LimitedWhitespace(parts)=>parts.next(),
            Self::LimitedSeparator(parts)=>parts.next(),
        }
    }
}
pub(crate) fn split_parts<'a>(value: &'a str, separator: Option<&'a str>) -> SplitParts<'a> {
    split_parts_limit(value,separator,None)
}
pub(crate) fn split_parts_limit<'a>(value:&'a str,separator:Option<&'a str>,limit:Option<usize>)->SplitParts<'a>{
    match (separator,limit){
        (Some(separator),None)=>SplitParts::Separator(value.split(separator)),
        (None,None)=>SplitParts::Whitespace(value.split_whitespace()),
        (Some(separator),Some(limit))=>SplitParts::LimitedSeparator(value.splitn(limit,separator)),
        (None,Some(limit))=>SplitParts::LimitedWhitespace(crate::utils::SplitWhitespaceN::new(value,limit)),
    }
}

fn strip_character(characters: Option<&str>, character: char) -> bool {
    match characters {
        Some(characters) => characters.chars().any(|candidate| candidate == character),
        None => character.is_whitespace(),
    }
}
/// Character sets have Unicode scalar semantics, including an empty set. This
/// borrows the same predicate used by the standard directional trim workers.
pub(crate) fn strip_range(
    value: &str,
    characters: Option<&str>,
    left: bool,
    right: bool,
) -> Range<usize> {
    let start = if left {
        value.len()
            - value
                .trim_start_matches(|ch| strip_character(characters, ch))
                .len()
    } else {
        0
    };
    let end = if right {
        start
            + value[start..]
                .trim_end_matches(|ch| strip_character(characters, ch))
                .len()
    } else {
        value.len()
    };
    start..end
}
pub(crate) fn string_view_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<SplitParts<'_>>(),
        size_of::<crate::utils::SplitWhitespaceN<'_>>(),
        size_of::<(Option<usize>,usize)>(),
        size_of::<Option<&str>>(),
        size_of::<(&str, Option<&str>, bool, bool)>(),
        size_of::<(&str, usize, usize, Range<usize>)>(),
        // The standard trim pattern is a scalar predicate; its closure captures
        // the optional borrowed set, whose nested Unicode iterator is included.
        size_of::<(Option<&str>, std::str::CharIndices<'_>, char, bool)>(),
        size_of::<(Option<&str>, std::str::Chars<'_>, Option<char>, char, bool)>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
