use crate::tokenizer::{pattern::Invert, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use crate::utils::{macro_rules_attribute, SysRegex};

pub(crate) const PATTERN: &str = r"\w+|[^\w\s]+";

/// Unicode words and punctuation, using one retained regex source.
#[derive(Clone, Debug)]
pub struct Whitespace { regex: SysRegex }
impl Default for Whitespace {
    fn default() -> Self { Self { regex: SysRegex::new(PATTERN).expect("valid Whitespace pattern") } }
}
impl PartialEq for Whitespace { fn eq(&self, _: &Self) -> bool { true } }
impl Eq for Whitespace {}
impl Whitespace {
    #[cfg(feature="fancy-regex")]
    pub(crate) fn from_regex(regex: SysRegex) -> Self { Self { regex } }
    #[cfg(feature="fancy-regex")]
    pub(crate) fn workspace_plan(&self) -> Option<std::result::Result<fancy_regex::workspace::Plan<'_>, fancy_regex::workspace::PlanError>> { self.regex.workspace_plan() }
}
impl serde::Serialize for Whitespace {
    fn serialize<S: serde::Serializer>(&self, serializer:S)->std::result::Result<S::Ok,S::Error> {
        use serde::ser::SerializeStruct;
        let mut out=serializer.serialize_struct("Whitespace",1)?;
        out.serialize_field("type","Whitespace")?; out.end()
    }
}
impl<'de> serde::Deserialize<'de> for Whitespace {
    fn deserialize<D:serde::Deserializer<'de>>(de:D)->std::result::Result<Self,D::Error> {
        #[derive(serde::Deserialize)] enum Kind { Whitespace }
        #[derive(serde::Deserialize)] struct Input { #[serde(rename="type")] _kind:Kind }
        let _=Input::deserialize(de)?;
        Ok(Self::default())
    }
}
impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let matcher=self.regex.matcher()?;
        pretokenized.split(|_, normalized| normalized.split(Invert(&matcher), SplitDelimiterBehavior::Removed))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct WhitespaceSplit;

impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(char::is_whitespace, SplitDelimiterBehavior::Removed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};

    #[test]
    fn basic() {
        let tests = vec![
            (
                "Hey man!",
                vec![("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))],
            ),
            (
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            ("\n", vec![]),
        ];
        let pretok = Whitespace::default();
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
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
    fn whitespace_split() {
        let tests = vec![
            ("Hey man!", vec![("Hey", (0, 3)), ("man!", (4, 8))]),
            (
                "Hey, man, Good?",
                vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
            ),
        ];
        let pretok = WhitespaceSplit;
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
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
}
