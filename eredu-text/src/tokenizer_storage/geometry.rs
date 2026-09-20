//! Streaming inspection of the public JSON vocabulary formats.
use serde::{
    Deserialize, Deserializer,
    de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use std::fmt;

#[derive(Deserialize)]
struct Input {
    model: Model,
    #[serde(default)]
    added_tokens: Count,
}
#[derive(Deserialize)]
struct Model {
    vocab: Vocabulary,
}
#[derive(Default)]
struct Count(usize);
impl<'de> Deserialize<'de> for Count {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Counter;
        impl<'de> Visitor<'de> for Counter {
            type Value = Count;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an array")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Count, A::Error> {
                let mut count = 0usize;
                while seq.next_element::<IgnoredAny>()?.is_some() {
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| de::Error::custom("vocabulary count overflow"))?;
                }
                Ok(Count(count))
            }
        }
        d.deserialize_seq(Counter)
    }
}
struct Vocabulary {
    count: usize,
    extent: u64,
}
impl<'de> Deserialize<'de> for Vocabulary {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Vocab;
        impl<'de> Visitor<'de> for Vocab {
            type Value = Vocabulary;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a token-ID map or scored vocabulary array")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Vocabulary, A::Error> {
                let mut result = Vocabulary {
                    count: 0,
                    extent: 0,
                };
                while map.next_key::<IgnoredAny>()?.is_some() {
                    result.extent = result.extent.max(u64::from(map.next_value::<u32>()?) + 1);
                    result.count = result
                        .count
                        .checked_add(1)
                        .ok_or_else(|| de::Error::custom("vocabulary count overflow"))?;
                }
                Ok(result)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vocabulary, A::Error> {
                let mut count = 0usize;
                while seq.next_element::<IgnoredAny>()?.is_some() {
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| de::Error::custom("vocabulary count overflow"))?;
                }
                Ok(Vocabulary {
                    count,
                    extent: count as u64,
                })
            }
        }
        d.deserialize_any(Vocab)
    }
}
pub(super) fn domain_extent(bytes: &[u8]) -> Result<usize, serde_json::Error> {
    let input: Input = serde_json::from_slice(bytes)?;
    let extent = input
        .model
        .vocab
        .count
        .checked_add(input.added_tokens.0)
        .and_then(|added_extent| {
            usize::try_from(input.model.vocab.extent)
                .ok()
                .map(|model_extent| model_extent.max(added_extent))
        })
        .ok_or_else(|| <serde_json::Error as de::Error>::custom("vocabulary extent overflow"))?;
    Ok(extent)
}
