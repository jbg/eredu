use super::{ReverseVocabulary, WordLevel, WordLevelBuilder};
use ahash::AHashSet;
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

pub(crate) struct Serialization<'a> {
    source: &'a WordLevel,
    entries: Entries<'a>,
}
impl WordLevel {
    pub(crate) fn serialization_with_allocations<'a>(
        &'a self,
        policy: &dyn serde_json::allocation::Allocation,
    ) -> std::result::Result<Serialization<'a>, serde_json::allocation::AllocationError> {
        let allocator = serde_json::allocation::Allocator::new(policy);
        allocator.reserve(
            std::mem::size_of::<Serialization<'_>>()
                + std::mem::size_of::<std::slice::Iter<'_, (&u32, &String)>>(),
        )?;
        let entries = prepare_entries(&self.vocab_r, policy)?;
        Ok(Serialization {
            source: self,
            entries,
        })
    }
}
impl Serialize for WordLevel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialization_with_allocations(&serde_json::allocation::Unenforced)
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}
impl Serialize for Serialization<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut model = serializer.serialize_struct("WordLevel", 3)?;
        model.serialize_field("type", "WordLevel")?;
        model.serialize_field("vocab", &self.entries)?;
        model.serialize_field("unk_token", &self.source.unk_token)?;
        model.end()
    }
}

impl<'de> Deserialize<'de> for WordLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "WordLevel",
            &["type", "vocab", "unk_token"],
            WordLevelVisitor,
        )
    }
}

struct WordLevelVisitor;
impl<'de> Visitor<'de> for WordLevelVisitor {
    type Value = WordLevel;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct WordLevel")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = WordLevelBuilder::new();
        let mut missing_fields = vec![
            // for retrocompatibility the "type" field is not mandatory
            "unk_token",
            "vocab",
        ]
        .into_iter()
        .collect::<AHashSet<_>>();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "vocab" => builder = builder.vocab(map.next_value()?),
                "unk_token" => builder = builder.unk_token(map.next_value()?),
                "type" => match map.next_value()? {
                    "WordLevel" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"WordLevel",
                        ))
                    }
                },
                _ => {}
            }
            missing_fields.remove::<str>(&key);
        }

        if !missing_fields.is_empty() {
            Err(serde::de::Error::missing_field(
                missing_fields.iter().next().unwrap(),
            ))
        } else {
            Ok(builder.build().map_err(serde::de::Error::custom)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::models::wordlevel::{Vocab, WordLevel, WordLevelBuilder};

    #[test]
    fn serde() {
        let wl = WordLevel::default();
        let wl_s = r#"{"type":"WordLevel","vocab":{},"unk_token":"<unk>"}"#;

        assert_eq!(serde_json::to_string(&wl).unwrap(), wl_s);
        assert_eq!(serde_json::from_str::<WordLevel>(wl_s).unwrap(), wl);
    }

    #[test]
    fn incomplete_vocab() {
        let vocab: Vocab = [("<unk>".into(), 0), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let wordlevel = WordLevelBuilder::default()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let wl_s = r#"{"type":"WordLevel","vocab":{"<unk>":0,"b":2},"unk_token":"<unk>"}"#;
        assert_eq!(serde_json::to_string(&wordlevel).unwrap(), wl_s);
        assert_eq!(serde_json::from_str::<WordLevel>(wl_s).unwrap(), wordlevel);
    }

    #[test]
    fn deserialization_should_fail() {
        let missing_unk = r#"{"type":"WordLevel","vocab":{}}"#;
        assert!(serde_json::from_str::<WordLevel>(missing_unk)
            .unwrap_err()
            .to_string()
            .starts_with("missing field `unk_token`"));

        let wrong_type = r#"{"type":"WordPiece","vocab":{}}"#;
        assert!(serde_json::from_str::<WordLevel>(wrong_type)
            .unwrap_err()
            .to_string()
            .starts_with("invalid value: string \"WordPiece\", expected WordLevel"));
    }
}

enum Entries<'a> {
    Dense(&'a ReverseVocabulary),
    Sparse(Vec<(&'a u32, &'a String)>),
}
impl Serialize for Entries<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Dense(vocabulary) => serializer.collect_map((0..vocabulary.len()).map(|id| {
                let id = id as u32;
                (vocabulary.get(&id).expect("checked dense vocabulary"), id)
            })),
            Self::Sparse(entries) => {
                serializer.collect_map(entries.iter().map(|(id, token)| (token, id)))
            }
        }
    }
}
fn prepare_entries<'a>(
    vocabulary: &'a ReverseVocabulary,
    policy: &dyn serde_json::allocation::Allocation,
) -> std::result::Result<Entries<'a>, serde_json::allocation::AllocationError> {
    // Distinct keys whose count is n and every value is below n are exactly
    // 0..n. Both policies use the same source-derived storage choice.
    if vocabulary
        .keys()
        .all(|&id| (id as usize) < vocabulary.len())
    {
        return Ok(Entries::Dense(vocabulary));
    }
    let mut entries = Vec::new();
    serde_json::allocation::Allocator::new(policy).grow(&mut entries, vocabulary.len())?;
    entries.extend(vocabulary.iter());
    entries.sort_unstable_by_key(|(id, _)| **id);
    Ok(Entries::Sparse(entries))
}
/// Serialization visits the actual sparse IDs rather than traversing holes.
pub(super) struct OrderedVocabulary<'a>(pub &'a ReverseVocabulary);
impl Serialize for OrderedVocabulary<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        prepare_entries(self.0, &serde_json::allocation::Unenforced)
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}
