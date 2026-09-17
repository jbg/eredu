use crate::utils::SysRegex;
use crate::{Offsets, Result};
use regex::Regex;

// One shared byte-span coverage worker for ordinary patterns and original E.
// A match iterator is consumed once; a failure is terminal. Empty input yields
// the existing empty non-match without advancing the engine.
pub(crate) struct Coverage<I> {
    matches: I,
    previous: usize,
    len: usize,
    pending: Option<Offsets>,
    finished: bool,
}
pub(crate) fn coverage<I>(len: usize, matches: I) -> Coverage<I> {
    Coverage {
        matches,
        previous: 0,
        len,
        pending: None,
        finished: false,
    }
}
impl<I, E> Iterator for Coverage<I>
where
    I: Iterator<Item = std::result::Result<Offsets, E>>,
{
    type Item = std::result::Result<(Offsets, bool), E>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.len == 0 {
            self.finished = true;
            return Some(Ok(((0, 0), false)));
        }
        if let Some(offsets) = self.pending.take() {
            self.previous = offsets.1;
            return Some(Ok((offsets, true)));
        }
        match self.matches.next() {
            Some(Err(error)) => {
                self.finished = true;
                Some(Err(error))
            }
            Some(Ok((start, end))) => {
                if self.previous != start {
                    self.pending = Some((start, end));
                    Some(Ok(((self.previous, start), false)))
                } else {
                    self.previous = end;
                    Some(Ok(((start, end), true)))
                }
            }
            None => {
                self.finished = true;
                if self.previous != self.len {
                    Some(Ok(((self.previous, self.len), false)))
                } else {
                    None
                }
            }
        }
    }
}

/// Pattern used to split a NormalizedString
pub trait Pattern {
    /// Slice the given string in a list of pattern match positions, with
    /// a boolean indicating whether this is a match or not.
    ///
    /// This method *must* cover the whole string in its outputs, with
    /// contiguous ordered slices.
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>>;
}

impl Pattern for char {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let is_char = |c: char| -> bool { c == *self };
        is_char.find_matches(inside)
    }
}

impl Pattern for &str {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if self.is_empty() {
            // If we try to find the matches with an empty string, just don't match anything
            return Ok(vec![((0, inside.chars().count()), false)]);
        }

        let re = Regex::new(&regex::escape(self))?;
        (&re).find_matches(inside)
    }
}

impl Pattern for &String {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let s: &str = self;
        s.find_matches(inside)
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }
        let matches = self
            .find_iter(inside)
            .map(|m| Ok::<_, std::convert::Infallible>((m.start(), m.end())));
        let mut splits = Vec::with_capacity(inside.len());
        for item in coverage(inside.len(), matches) {
            match item {
                Ok(item) => splits.push(item),
                Err(never) => match never {},
            }
        }
        Ok(splits)
    }
}

impl Pattern for &SysRegex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }
        let matches = self
            .find_iter(inside)
            .map(Ok::<_, std::convert::Infallible>);
        let mut splits = Vec::with_capacity(inside.len());
        for item in coverage(inside.len(), matches) {
            match item {
                Ok(item) => splits.push(item),
                Err(never) => match never {},
            }
        }
        Ok(splits)
    }
}

// Shared scalar-predicate enumeration; offsets always refer to original UTF-8 bytes.
// Ordinary Pattern collects the same coverage which original E visits directly.
pub(crate) struct CharMatches<'a, 'p, F> {
    chars: std::str::CharIndices<'a>,
    predicate: &'p F,
}
impl<F: Fn(char) -> bool> Iterator for CharMatches<'_, '_, F> {
    type Item = std::result::Result<Offsets, std::convert::Infallible>;
    fn next(&mut self) -> Option<Self::Item> {
        for (start, c) in self.chars.by_ref() {
            if (self.predicate)(c) {
                return Some(Ok((start, start + c.len_utf8())));
            }
        }
        None
    }
}
pub(crate) fn char_coverage<'a, 'p, F: Fn(char) -> bool>(
    inside: &'a str,
    predicate: &'p F,
) -> Coverage<CharMatches<'a, 'p, F>> {
    coverage(
        inside.len(),
        CharMatches {
            chars: inside.char_indices(),
            predicate,
        },
    )
}
impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        Ok(char_coverage(inside, self)
            .map(|item| match item {
                Ok(item) => item,
                Err(never) => match never {},
            })
            .collect())
    }
}

/// Invert the `is_match` flags for the wrapped Pattern. This is useful
/// for example when we use a regex that matches words instead of a delimiter,
/// and we want to match the delimiter.
pub struct Invert<P: Pattern>(pub P);
impl<P: Pattern> Pattern for Invert<P> {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        Ok(self
            .0
            .find_matches(inside)?
            .into_iter()
            .map(|(offsets, flag)| (offsets, !flag))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    macro_rules! do_test {
        ($inside: expr, $pattern: expr => @ERROR) => {
            assert!($pattern.find_matches($inside).is_err());
        };
        ($inside: expr, $pattern: expr => $result: expr) => {
            assert_eq!($pattern.find_matches($inside).unwrap(), $result);
            assert_eq!(
                Invert($pattern).find_matches($inside).unwrap(),
                $result
                    .into_iter()
                    .map(|v: (Offsets, bool)| (v.0, !v.1))
                    .collect::<Vec<_>>()
            );
        };
    }

    #[test]
    fn shared_coverage_preserves_empty_and_terminal_error_without_advancing_again() {
        let empty = coverage(
            0,
            std::iter::from_fn(|| -> Option<std::result::Result<Offsets, ()>> {
                panic!("empty input must not advance the matcher")
            }),
        )
        .collect::<Vec<_>>();
        assert_eq!(empty, vec![Ok(((0, 0), false))]);
        let mut input = vec![Ok((2, 4)), Err(7u8), Ok((5, 6))].into_iter();
        let mut covered = coverage(8, &mut input);
        assert_eq!(covered.next(), Some(Ok(((0, 2), false))));
        assert_eq!(covered.next(), Some(Ok(((2, 4), true))));
        assert_eq!(covered.next(), Some(Err(7)));
        assert_eq!(covered.next(), None);
        drop(covered);
        assert_eq!(input.next(), Some(Ok((5, 6))));
    }

    #[test]
    fn char() {
        do_test!("aba", 'a' => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", 'a' => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("aabbb", 'a' => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("", 'a' => vec![((0, 0), false)]);
        do_test!("aaa", 'b' => vec![((0, 3), false)]);
    }

    #[test]
    fn str() {
        do_test!("aba", "a" => vec![((0, 1), true), ((1, 2), false), ((2, 3), true)]);
        do_test!("bbbba", "a" => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("aabbb", "a" => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("aabbb", "ab" => vec![((0, 1), false), ((1, 3), true), ((3, 5), false)]);
        do_test!("aabbab", "ab" =>
            vec![((0, 1), false), ((1, 3), true), ((3, 4), false), ((4, 6), true)]
        );
        do_test!("", "" => vec![((0, 0), false)]);
        do_test!("aaa", "" => vec![((0, 3), false)]);
        do_test!("aaa", "b" => vec![((0, 3), false)]);
    }

    #[test]
    fn functions() {
        let is_b = |c| c == 'b';
        do_test!("aba", is_b => vec![((0, 1), false), ((1, 2), true), ((2, 3), false)]);
        do_test!("aaaab", is_b => vec![((0, 4), false), ((4, 5), true)]);
        do_test!("bbaaa", is_b => vec![((0, 1), true), ((1, 2), true), ((2, 5), false)]);
        do_test!("", is_b => vec![((0, 0), false)]);
        do_test!("aaa", is_b => vec![((0, 3), false)]);
    }

    #[test]
    fn regex() {
        let is_whitespace = Regex::new(r"\s+").unwrap();
        do_test!("a   b", &is_whitespace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);
        do_test!("   a   b   ", &is_whitespace =>
            vec![((0, 3), true), ((3, 4), false), ((4, 7), true), ((7, 8), false), ((8, 11), true)]
        );
        do_test!("", &is_whitespace => vec![((0, 0), false)]);
        do_test!("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace =>
            vec![((0, 16), false), ((16, 17), true), ((17, 45), false)]
        );
        do_test!("aaa", &is_whitespace => vec![((0, 3), false)]);
    }

    #[test]
    fn sys_regex() {
        let is_whitespace = SysRegex::new(r"\s+").unwrap();
        do_test!("a   b", &is_whitespace => vec![((0, 1), false), ((1, 4), true), ((4, 5), false)]);
        do_test!("   a   b   ", &is_whitespace =>
            vec![((0, 3), true), ((3, 4), false), ((4, 7), true), ((7, 8), false), ((8, 11), true)]
        );
        do_test!("", &is_whitespace => vec![((0, 0), false)]);
        do_test!("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace =>
            vec![((0, 16), false), ((16, 17), true), ((17, 45), false)]
        );
        do_test!("aaa", &is_whitespace => vec![((0, 3), false)]);
    }
}

#[cfg(test)]
mod numeric_tests {
    use super::*;
    #[test]
    fn shared_numeric_coverage_matches_scalar_boundaries_for_all_unicode_in_batches() {
        for start in (0..=0x10ffffu32).step_by(4096) {
            let text: String = (start..=(start + 4095).min(0x10ffff))
                .filter_map(char::from_u32)
                .collect();
            let mut expected = Vec::new();
            let mut end = 0;
            for (at, c) in text.char_indices() {
                if c.is_numeric() {
                    if end < at {
                        expected.push(((end, at), false));
                    }
                    end = at + c.len_utf8();
                    expected.push(((at, end), true));
                }
            }
            if end < text.len() {
                expected.push(((end, text.len()), false));
            }
            if text.is_empty() {
                expected.push(((0, 0), false));
            }
            assert_eq!(char::is_numeric.find_matches(&text).unwrap(), expected);
        }
    }
    #[test]
    fn predicate_invocations_keep_empty_and_scalar_order_without_extra_probes() {
        let seen = std::cell::RefCell::new(Vec::new());
        let predicate = |c| {
            seen.borrow_mut().push(c);
            char::is_numeric(c)
        };
        assert_eq!(predicate.find_matches("").unwrap(), vec![((0, 0), false)]);
        assert!(seen.borrow().is_empty());
        let text = "a²12¼Ⅷ٣\u{301}z";
        let _ = predicate.find_matches(text).unwrap();
        assert_eq!(*seen.borrow(), text.chars().collect::<Vec<_>>());
    }
}
