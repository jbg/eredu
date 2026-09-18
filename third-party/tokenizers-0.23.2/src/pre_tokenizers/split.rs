use crate::utils::SysRegex;
use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{
    pattern::Invert, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior,
};

/// Represents the different patterns that `Split` can use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub enum SplitPattern {
    String(String),
    Regex(String),
}

impl From<String> for SplitPattern {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for SplitPattern {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

/// Borrowed source spelling and its literal/regex interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SplitPatternRef<'a> { String(&'a str), Regex(&'a str) }

#[derive(Debug, Clone)]
pub struct Split {
    literal: Option<String>,
    regex: SysRegex,
    pub behavior: SplitDelimiterBehavior,
    pub invert: bool,
}
impl Serialize for Split {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct View<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            pattern: SplitPatternRef<'a>,
            behavior: SplitDelimiterBehavior,
            invert: bool,
        }
        View { kind: "Split", pattern: self.pattern(), behavior: self.behavior, invert: self.invert }
            .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for Split {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        enum Type { Split }
        #[derive(Deserialize)]
        struct Input {
            #[serde(rename = "type")]
            _type: Type,
            pattern: SplitPattern,
            behavior: SplitDelimiterBehavior,
            invert: bool,
        }
        let input = Input::deserialize(deserializer)?;
        Self::new(input.pattern, input.behavior, input.invert).map_err(serde::de::Error::custom)
    }
}
impl PartialEq for Split {
    fn eq(&self, other: &Self) -> bool {
        self.pattern() == other.pattern() && self.behavior == other.behavior && self.invert == other.invert
    }
}
impl Split {
    pub fn new<I: Into<SplitPattern>>(pattern: I, behavior: SplitDelimiterBehavior, invert: bool) -> Result<Self> {
        let (literal, regex) = match pattern.into() {
            SplitPattern::String(s) => { let regex = SysRegex::new(&regex::escape(&s))?; (Some(s), regex) }
            SplitPattern::Regex(s) => (None, SysRegex::new(&s)?),
        };
        Ok(Self { literal, regex, behavior, invert })
    }
    /// Borrows the immutable spelling actually used by this regex program.
    pub fn pattern(&self) -> SplitPatternRef<'_> {
        match &self.literal { Some(value) => SplitPatternRef::String(value), None => SplitPatternRef::Regex(self.regex.pattern()) }
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn from_regex(regex: SysRegex) -> Self {
        Self { literal: None, regex, behavior: SplitDelimiterBehavior::Isolated, invert: false }
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn workspace_plan(&self) -> Option<std::result::Result<fancy_regex::workspace::Plan<'_>, fancy_regex::workspace::PlanError>> {
        if self.literal.is_some() || self.invert || self.behavior != SplitDelimiterBehavior::Isolated { return None; }
        self.regex.workspace_plan()
    }
    #[cfg(feature = "fancy-regex")]
    pub(crate) fn compiled_control_bytes() -> Option<usize> {
        SysRegex::control_bytes()?.checked_add(std::mem::size_of::<Self>())
    }
}

impl PreTokenizer for Split {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let matcher = self.regex.matcher()?;
        if self.invert {
            pretokenized.split(|_, normalized| normalized.split(Invert(&matcher), self.behavior))
        } else {
            pretokenized.split(|_, normalized| normalized.split(&matcher, self.behavior))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};
    use SplitDelimiterBehavior::*;

    #[test]
    fn basic() {
        let tests = vec![
            (
                Removed,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Isolated,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithPrevious,
                "How are you doing?",
                vec![
                    ("How ", (0, 4)),
                    ("are ", (4, 8)),
                    ("you ", (8, 12)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                MergedWithNext,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" are", (3, 7)),
                    (" you", (7, 11)),
                    (" doing", (11, 17)),
                    ("?", (17, 18)),
                ],
            ),
            (
                Contiguous,
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    (" ", (3, 4)),
                    ("are", (4, 7)),
                    (" ", (7, 8)),
                    ("you", (8, 11)),
                    (" ", (11, 12)),
                    ("doing?", (12, 18)),
                ],
            ),
        ];

        // use whitespace regex
        let regex = SplitPattern::Regex(r"\w+|[^\w\s]+".into());

        for (behavior, s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            let pretok = Split::new(regex.clone(), behavior, true).unwrap();
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    #[test]
    fn regex_string() {
        let mut pretok_str_for_regex = PreTokenizedString::from("Hey, man!");
        let mut pretok_str_for_string = pretok_str_for_regex.clone();

        // pre-tokenizer splits on " " - one from Regex, one from string
        let pretokenizer_regex = Split::new(
            SplitPattern::Regex(r"\s+".into()),
            SplitDelimiterBehavior::Removed,
            false,
        )
        .unwrap();
        let pretokenizer_string = Split::new(" ", SplitDelimiterBehavior::Removed, false).unwrap();

        pretokenizer_regex
            .pre_tokenize(&mut pretok_str_for_regex)
            .unwrap();
        pretokenizer_string
            .pre_tokenize(&mut pretok_str_for_string)
            .unwrap();

        assert_eq!(pretok_str_for_regex, pretok_str_for_string);
    }

    #[test]
    fn invert() {
        let mut pretok_str = PreTokenizedString::from("Hello Hello Hello");
        let mut pretok_str_for_invert = pretok_str.clone();

        // one pre-tokenizer splits on " " - one splits inverted on "Hello"
        let pretokenizer = Split::new(" ", SplitDelimiterBehavior::Removed, false).unwrap();
        let pretokenizer_invert =
            Split::new("Hello", SplitDelimiterBehavior::Removed, true).unwrap();

        pretokenizer.pre_tokenize(&mut pretok_str).unwrap();
        pretokenizer_invert
            .pre_tokenize(&mut pretok_str_for_invert)
            .unwrap();

        assert_eq!(pretok_str, pretok_str_for_invert);
    }

    #[test]
    fn serialization() {
        use SplitDelimiterBehavior::*;

        let split = Split::new("Hello", Removed, true).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}"#;
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);

        let split = Split::new(SplitPattern::Regex(r"\s+".into()), Isolated, false).unwrap();
        let split_s =
            r#"{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false}"#;
        assert_eq!(serde_json::to_string(&split).unwrap(), split_s);
        assert_eq!(serde_json::from_str::<Split>(split_s).unwrap(), split);
    }
}
