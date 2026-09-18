use crate::tokenizer::pattern::Pattern;
use crate::Offsets;
use fancy_regex::Regex;
use std::error::Error;

#[derive(Debug)]
pub(super) struct GeneralRegex {
    regex: Regex,
}

impl GeneralRegex {
    pub(super) fn pattern(&self) -> &str { self.regex.as_str() }

    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches(self.regex.find_iter(inside))
    }

    pub fn new(regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Ok(Self {
            regex: Regex::new(regex_str)?,
        })
    }
}

pub struct Matches<'r, 't>(fancy_regex::Matches<'r, 't, str>);

impl Iterator for Matches<'_, '_> {
    type Item = crate::Result<(usize, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|m| m.map(|m| (m.start(), m.end())).map_err(|e| Box::new(e) as crate::Error))
    }
}

impl Pattern for &Regex {
    fn find_matches(
        &self,
        inside: &str,
    ) -> Result<Vec<(Offsets, bool)>, Box<dyn Error + Send + Sync + 'static>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for match_ in self.find_iter(inside) {
            let match_ = match_?;
            let start = match_.start();
            let end = match_.end();
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
