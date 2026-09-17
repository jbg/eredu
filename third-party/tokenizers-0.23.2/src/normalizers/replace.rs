use crate::tokenizer::Decoder;
use crate::tokenizer::pattern::{Pattern, coverage};
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::SysRegex;
use serde::{Deserialize, Serialize};

/// Represents the different patterns that `Replace` can use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub enum ReplacePattern {
    String(String),
    Regex(String),
}

impl From<String> for ReplacePattern {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for ReplacePattern {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

/// We use this custom deserializer to provide the value for `regex` for `Replace`
#[doc(hidden)]
#[derive(Deserialize)]
#[serde(tag = "type")]
struct ReplaceDeserializer {
    pattern: ReplacePattern,
    content: String,
}

impl std::convert::TryFrom<ReplaceDeserializer> for Replace {
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn try_from(v: ReplaceDeserializer) -> Result<Self> {
        Self::new(v.pattern, v.content)
    }
}

/// This normalizer will take a `pattern` (for now only a String)
/// and replace every occurrence with `content`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", try_from = "ReplaceDeserializer")]
pub struct Replace {
    pattern: ReplacePattern,
    pub content: String,
    #[serde(skip)]
    regex: Option<SysRegex>,
}

impl Clone for Replace {
    fn clone(&self) -> Self {
        Self::new(self.pattern.clone(), &self.content).unwrap()
    }
}

impl PartialEq for Replace {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.content == other.content
    }
}

impl Replace {
    /// Borrows the exact literal/regex pattern without compiling another matcher.
    pub fn pattern(&self) -> &ReplacePattern {
        &self.pattern
    }

    pub fn new<I: Into<ReplacePattern>, C: Into<String>>(pattern: I, content: C) -> Result<Self> {
        let pattern: ReplacePattern = pattern.into();
        match pattern {
            ReplacePattern::String(pattern) => {
                Ok(Self::from_literal_parts(pattern, content.into()))
            }
            ReplacePattern::Regex(pattern) => {
                let regex = SysRegex::new(&pattern)?;
                Ok(Self {
                    pattern: ReplacePattern::Regex(pattern),
                    content: content.into(),
                    regex: Some(regex),
                })
            }
        }
    }

    // The original compiler moves its two paid String destinations through the
    // same constructor as ordinary literal replacement. No matcher is allocated.
    pub(crate) fn from_literal_parts(pattern: String, content: String) -> Self {
        Self {
            pattern: ReplacePattern::String(pattern),
            content,
            regex: None,
        }
    }
}

// Literal and regex forms share Replace's existing span consumer. Literal
// match_indices has the same nonoverlapping UTF-8 boundaries as an escaped
// regex, including empty-pattern boundaries; coverage preserves empty-input
// behavior and fills every unmatched span.
struct Matches<'a>(&'a Replace);
impl Pattern for Matches<'_> {
    fn find_matches(&self, inside: &str) -> Result<Vec<(crate::Offsets, bool)>> {
        if let Some(regex) = &self.0.regex {
            return regex.find_matches(inside);
        }
        let ReplacePattern::String(pattern) = &self.0.pattern else {
            unreachable!("regex replacement retains its matcher")
        };
        let matches = inside
            .match_indices(pattern.as_str())
            .map(|(start, matched)| {
                Ok::<_, std::convert::Infallible>((start, start + matched.len()))
            });
        let mut spans = Vec::with_capacity(inside.len());
        for item in coverage(inside.len(), matches) {
            match item {
                Ok(item) => spans.push(item),
                Err(never) => match never {},
            }
        }
        Ok(spans)
    }
}

impl Normalizer for Replace {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.replace(Matches(self), &self.content)
    }
}

impl Decoder for Replace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        tokens
            .into_iter()
            .map(|token| -> Result<String> {
                let mut new_token = "".to_string();

                for ((start, stop), is_match) in Matches(self).find_matches(&token)? {
                    if is_match {
                        new_token.push_str(&self.content);
                    } else {
                        new_token.push_str(&token[start..stop]);
                    }
                }
                Ok(new_token)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace() {
        let original = "This is a ''test''";
        let normalized = "This is a \"test\"";

        let mut n = NormalizedString::from(original);
        Replace::new("''", "\"").unwrap().normalize(&mut n).unwrap();

        assert_eq!(&n.get(), &normalized);
    }

    #[test]
    fn test_replace_regex() {
        let original = "This     is   a         test";
        let normalized = "This is a test";

        let mut n = NormalizedString::from(original);
        Replace::new(ReplacePattern::Regex(r"\s+".into()), ' ')
            .unwrap()
            .normalize(&mut n)
            .unwrap();

        assert_eq!(&n.get(), &normalized);
    }

    #[test]
    fn serialization() {
        let replace = Replace::new("Hello", "Hey").unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"}"#;
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);

        let replace = Replace::new(ReplacePattern::Regex(r"\s+".into()), ' ').unwrap();
        let replace_s = r#"{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "}"#;
        assert_eq!(serde_json::to_string(&replace).unwrap(), replace_s);
        assert_eq!(serde_json::from_str::<Replace>(replace_s).unwrap(), replace);
    }

    #[test]
    fn test_replace_decode() {
        let original = vec!["hello".to_string(), "_hello".to_string()];
        let replace = Replace::new("_", " ").unwrap();
        assert_eq!(
            replace.decode_chain(original).unwrap(),
            vec!["hello", " hello"]
        );
    }
    #[test]
    fn literal_replacement_matches_escaped_regex_for_unicode_empty_and_overlapping_patterns() {
        for pattern in ["", "a", "aa", "▁", "é", ".*[]", "\n", "🙂"] {
            for input in ["", "aaaa", "é▁a🙂▁é", "x.*[]y\n", "a\0é"] {
                for content in ["", " ", "🙂x"] {
                    let literal = Replace::new(pattern, content).unwrap();
                    let regex =
                        Replace::new(ReplacePattern::Regex(regex::escape(pattern)), content)
                            .unwrap();
                    assert!(literal.regex.is_none());
                    assert_eq!(
                        literal.decode_chain(vec![input.to_owned()]).unwrap(),
                        regex.decode_chain(vec![input.to_owned()]).unwrap(),
                        "{pattern:?} {input:?}"
                    );
                    let mut a = NormalizedString::from(input);
                    let mut b = a.clone();
                    literal.normalize(&mut a).unwrap();
                    regex.normalize(&mut b).unwrap();
                    assert_eq!(a, b, "{pattern:?} {input:?}");
                    assert_eq!(literal.clone(), literal);
                    let encoded = serde_json::to_string(&literal).unwrap();
                    assert_eq!(serde_json::from_str::<Replace>(&encoded).unwrap(), literal);
                }
            }
        }
    }
}
