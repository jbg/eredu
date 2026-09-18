use crate::tokenizer::pattern::Pattern;
use crate::{Offsets, Result};
use onig::Regex;
use std::error::Error;

#[derive(Debug)]
pub(super) struct GeneralRegex {
    regex: Regex,
    pattern: String,
}

impl GeneralRegex {
    pub(super) fn pattern(&self) -> &str { &self.pattern }
    pub fn find_iter<'a>(&'a self, inside: &'a str) -> impl Iterator<Item = Result<Offsets>> + 'a {
        self.regex.find_iter(inside).map(Ok)
    }

    pub fn new(
        regex_str: &str,
    ) -> std::result::Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Ok(Self {
            regex: Regex::new(regex_str)?,
            pattern: regex_str.to_owned(),
        })
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for (start, end) in self.find_iter(inside) {
            if prev != start {
                splits.push(((prev, start), false));
            }
            splits.push(((start, end), true));
            prev = end;
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false))
        }
        Ok(splits)
    }
}
