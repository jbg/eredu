//! Canonical immutable template storage used by builders, serde and source compilation.
pub(crate) mod compiler;
use super::{visit, PostProcessor, Result};
use crate::Encoding;
pub use compiler::Failure as TemplateCompileFailure;
use serde::{
    ser::{SerializeMap, SerializeSeq, SerializeStruct},
    Serialize, Serializer,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextRange {
    pub(crate) start: usize,
    pub(crate) len: usize,
}
impl TextRange {
    pub(crate) fn text<'a>(&self, bytes: &'a [u8]) -> &'a str {
        std::str::from_utf8(&bytes[self.start..self.start + self.len])
            .expect("validated template UTF-8")
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Sequence(usize),
    Special(usize),
    MissingSpecial(TextRange),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    pub(crate) kind: Kind,
    pub(crate) type_id: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Special {
    pub(crate) key: TextRange,
    pub(crate) id: TextRange,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

/// Immutable packed template shared by every construction and encoding path.
/// Cloning and serialization are explicit ordinary output operations.
///
/// ```compile_fail
/// use tokenizers::processors::template::compiled::TemplateProcessing;
/// fn mutate(value: &mut TemplateProcessing) { value.set_single("$A"); }
/// ```
#[derive(Clone, Debug)]
pub struct TemplateProcessing {
    pub(crate) pieces: Vec<Piece>,
    pub(crate) specials: Vec<Special>,
    pub(crate) ids: Vec<u32>,
    pub(crate) tokens: Vec<TextRange>,
    pub(crate) bytes: Vec<u8>,
    pub(crate) single_len: usize,
    pub(crate) added: [usize; 2],
}
impl TemplateProcessing {
    pub(crate) fn configuration_comparison_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        size_of::<[&Self; 2]>()
            .checked_add(size_of::<[&Special; 2]>())?
            .checked_add(size_of::<[&Piece; 2]>())?
            .checked_add(size_of::<[usize; 4]>())?
            .checked_add(size_of::<[TextRange; 2]>())
    }
    pub(crate) fn empty() -> Self {
        Self {
            pieces: Vec::new(),
            specials: Vec::new(),
            ids: Vec::new(),
            tokens: Vec::new(),
            bytes: Vec::new(),
            single_len: 0,
            added: [0; 2],
        }
    }
    fn pieces(&self, pair: bool) -> &[Piece] {
        if pair {
            &self.pieces[self.single_len..]
        } else {
            &self.pieces[..self.single_len]
        }
    }
    pub(crate) fn views(&self, pair: bool) -> impl Iterator<Item = visit::PieceView<'_>> {
        self.pieces(pair).iter().filter_map(move |piece| {
            Some(match piece.kind {
                Kind::Sequence(index) => visit::PieceView::Sequence {
                    index,
                    type_id: piece.type_id,
                },
                Kind::MissingSpecial(_) => return None,
                Kind::Special(index) => {
                    let special = &self.specials[index];
                    let range = special.start..special.start + special.len;
                    visit::PieceView::Special {
                        ids: &self.ids[range.clone()],
                        tokens: visit::Texts(&self.tokens[range], &self.bytes),
                        type_id: piece.type_id,
                    }
                }
            })
        })
    }
    pub(crate) fn single_ids(&self, add: bool) -> Option<usize> {
        let mut count = 0usize;
        for piece in self.pieces(false) {
            if add && matches!(piece.kind, Kind::MissingSpecial(_)) {
                return None;
            }
            if let Kind::Sequence(index) = piece.kind {
                if index != 0 {
                    return None;
                }
                count = count.checked_add(1)?;
            }
        }
        (count == 1).then_some(if add { self.added[0] } else { 0 })
    }
    pub(crate) fn visit_single<'a, E>(
        &'a self,
        add: bool,
        sink: impl FnMut(visit::PieceView<'a>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        visit::visit(self.views(false), add, sink)
    }
    pub(crate) fn visit_control_bytes() -> Option<usize> {
        [
            std::mem::size_of_val(&Self::empty().views(false)),
            std::mem::size_of::<visit::PieceView<'_>>(),
            std::mem::size_of::<Option<visit::PieceView<'_>>>(),
            std::mem::size_of::<visit::Texts<'_>>(),
        ]
        .iter()
        .try_fold(0usize, |n, v| n.checked_add(*v))
    }
    /// Actual retained destination capacities, in compiler reserve order.
    pub fn capacities(&self) -> [usize; 5] {
        [
            self.pieces.capacity(),
            self.specials.capacity(),
            self.ids.capacity(),
            self.tokens.capacity(),
            self.bytes.capacity(),
        ]
    }
}
impl PartialEq for TemplateProcessing {
    fn eq(&self, other: &Self) -> bool {
        if self.added != other.added
            || self.single_len != other.single_len
            || self.pieces.len() != other.pieces.len()
            || self.specials.len() != other.specials.len()
        {
            return false;
        }
        let same_special = |a: &Special, b: &Special| {
            a.key.text(&self.bytes) == b.key.text(&other.bytes)
                && a.id.text(&self.bytes) == b.id.text(&other.bytes)
                && self.ids[a.start..a.start + a.len] == other.ids[b.start..b.start + b.len]
                && self.tokens[a.start..a.start + a.len]
                    .iter()
                    .map(|t| t.text(&self.bytes))
                    .eq(other.tokens[b.start..b.start + b.len]
                        .iter()
                        .map(|t| t.text(&other.bytes)))
        };
        self.specials
            .iter()
            .all(|a| other.specials.iter().any(|b| same_special(a, b)))
            && self.pieces.iter().zip(&other.pieces).all(|(a, b)| {
                a.type_id == b.type_id
                    && match (a.kind, b.kind) {
                        (Kind::Sequence(a), Kind::Sequence(b)) => a == b,
                        (Kind::Special(a), Kind::Special(b)) => {
                            same_special(&self.specials[a], &other.specials[b])
                        }
                        (Kind::MissingSpecial(a), Kind::MissingSpecial(b)) => {
                            a.text(&self.bytes) == b.text(&other.bytes)
                        }
                        _ => false,
                    }
            })
    }
}
impl Eq for TemplateProcessing {}
impl PostProcessor for TemplateProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.added[usize::from(is_pair)]
    }
    fn process_encodings(&self, encodings: Vec<Encoding>, add: bool) -> Result<Vec<Encoding>> {
        let pair = match encodings.len() {
            1 => false,
            2 => true,
            _ => todo!(),
        };
        if add {
            for piece in self.pieces(pair) {
                if let Kind::MissingSpecial(key) = piece.kind {
                    return Err(format!(
                        "missing template special token {}",
                        key.text(&self.bytes)
                    )
                    .into());
                }
            }
        }
        visit::encodings(self.views(pair), encodings, add)
    }
}
struct Pieces<'a>(&'a TemplateProcessing, bool);
impl Serialize for Pieces<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        enum Piece<'a> {
            Sequence { id: &'static str, type_id: u32 },
            SpecialToken { id: &'a str, type_id: u32 },
        }
        let mut seq = serializer.serialize_seq(Some(self.0.pieces(self.1).len()))?;
        for p in self.0.pieces(self.1) {
            let v = match p.kind {
                Kind::Sequence(i) => Piece::Sequence {
                    id: if i == 0 { "A" } else { "B" },
                    type_id: p.type_id,
                },
                Kind::MissingSpecial(key) => Piece::SpecialToken {
                    id: key.text(&self.0.bytes),
                    type_id: p.type_id,
                },
                Kind::Special(i) => Piece::SpecialToken {
                    id: self.0.specials[i].key.text(&self.0.bytes),
                    type_id: p.type_id,
                },
            };
            seq.serialize_element(&v)?;
        }
        seq.end()
    }
}
struct Tokens<'a>(&'a TemplateProcessing, &'a Special);
impl Serialize for Tokens<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.1.len))?;
        for r in &self.0.tokens[self.1.start..self.1.start + self.1.len] {
            seq.serialize_element(r.text(&self.0.bytes))?;
        }
        seq.end()
    }
}
struct Specials<'a>(&'a TemplateProcessing);
impl Serialize for Specials<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Special<'a> {
            id: &'a str,
            ids: &'a [u32],
            tokens: Tokens<'a>,
        }
        let mut map = serializer.serialize_map(Some(self.0.specials.len()))?;
        for s in &self.0.specials {
            map.serialize_entry(
                s.key.text(&self.0.bytes),
                &Special {
                    id: s.id.text(&self.0.bytes),
                    ids: &self.0.ids[s.start..s.start + s.len],
                    tokens: Tokens(self.0, s),
                },
            )?;
        }
        map.end()
    }
}
impl Serialize for TemplateProcessing {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("TemplateProcessing", 4)?;
        s.serialize_field("type", "TemplateProcessing")?;
        s.serialize_field("single", &Pieces(self, false))?;
        s.serialize_field("pair", &Pieces(self, true))?;
        s.serialize_field("special_tokens", &Specials(self))?;
        s.end()
    }
}
