//! Packed source storage. Only the aggregate's borrowed compiler constructs it.
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

/// Immutable packed template. Original construction is crate-private; ordinary
/// Clone/serialization remain unmanaged compatibility operations.
///
/// ```compile_fail
/// use tokenizers::processors::template::compiled::CompiledTemplate;
/// fn mutate(value: &mut CompiledTemplate) { value.set_single("$A"); }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledTemplate {
    pub(crate) pieces: Vec<Piece>,
    pub(crate) specials: Vec<Special>,
    pub(crate) ids: Vec<u32>,
    pub(crate) tokens: Vec<TextRange>,
    pub(crate) bytes: Vec<u8>,
    pub(crate) single_len: usize,
    pub(crate) added: [usize; 2],
}
impl CompiledTemplate {
    pub(crate) fn configuration_comparison_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<(&Self, &super::TemplateProcessing)>(),
            size_of::<std::array::IntoIter<(bool, &Vec<super::Piece>), 2>>(),
            size_of::<(bool, &[super::Piece])>(),
            size_of::<&[Piece]>(),
            size_of::<
                std::iter::Zip<std::slice::Iter<'_, Piece>, std::slice::Iter<'_, super::Piece>>,
            >(),
            size_of::<(&Piece, &super::Piece)>(),
            size_of::<Option<&super::SpecialToken>>(),
            size_of::<&Special>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::slice::Iter<'_, TextRange>>(),
            size_of::<&[u8]>(),
            size_of::<std::slice::Iter<'_, String>>(),
            size_of::<bool>(),
        ]
        .iter().copied()
        .try_fold(0usize, usize::checked_add)
    }
    pub(crate) fn matches_template(&self, other: &super::TemplateProcessing) -> bool {
        if self.added != [other.added_single, other.added_pair] {
            return false;
        }
        for (pair, template) in [(false, &other.single.0), (true, &other.pair.0)] {
            let own = self.pieces(pair);
            if own.len() != template.len() {
                return false;
            }
            for (a, b) in own.iter().zip(template) {
                match (a.kind, b) {
                    (Kind::Sequence(index), super::Piece::Sequence { id, type_id })
                        if index == usize::from(*id != super::Sequence::A)
                            && a.type_id == *type_id => {}
                    (Kind::Special(index), super::Piece::SpecialToken { id, type_id }) => {
                        let Some(token) = other.special_tokens.0.get(id) else {
                            return false;
                        };
                        let own = &self.specials[index];
                        let range = own.start..own.start + own.len;
                        if a.type_id != *type_id
                            || self.ids[range.clone()] != token.ids
                            || !self.tokens[range]
                                .iter()
                                .map(|t| t.text(&self.bytes))
                                .eq(token.tokens.iter().map(String::as_str))
                        {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
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
    pub(crate) fn views(
        &self,
        pair: bool,
    ) -> impl Iterator<Item = visit::PieceView<'_>> {
        self.pieces(pair).iter().map(move |piece| match piece.kind {
            Kind::Sequence(index) => visit::PieceView::Sequence {
                index,
                type_id: piece.type_id,
            },
            Kind::Special(index) => {
                let special = &self.specials[index];
                let range = special.start..special.start + special.len;
                visit::PieceView::Special {
                    ids: &self.ids[range.clone()],
                    tokens: visit::Texts::Packed(
                        &self.tokens[range],
                        &self.bytes,
                    ),
                    type_id: piece.type_id,
                }
            }
        })
    }
    pub(crate) fn single_ids(&self, add: bool) -> Option<usize> {
        let mut count = 0usize;
        for piece in self.pieces(false) {
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
impl PostProcessor for CompiledTemplate {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.added[usize::from(is_pair)]
    }
    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add: bool,
    ) -> Result<Vec<Encoding>> {
        let pair = match encodings.len() {
            1 => false,
            2 => true,
            _ => todo!(),
        };
        visit::encodings(self.views(pair), encodings, add)
    }
}
struct Pieces<'a>(&'a CompiledTemplate, bool);
impl Serialize for Pieces<'_> {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        enum Piece<'a> {
            Sequence { id: &'static str, type_id: u32 },
            SpecialToken { id: &'a str, type_id: u32 },
        }
        let mut seq =
            serializer.serialize_seq(Some(self.0.pieces(self.1).len()))?;
        for p in self.0.pieces(self.1) {
            let v = match p.kind {
                Kind::Sequence(i) => Piece::Sequence {
                    id: if i == 0 { "A" } else { "B" },
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
struct Tokens<'a>(&'a CompiledTemplate, &'a Special);
impl Serialize for Tokens<'_> {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.1.len))?;
        for r in &self.0.tokens[self.1.start..self.1.start + self.1.len] {
            seq.serialize_element(r.text(&self.0.bytes))?;
        }
        seq.end()
    }
}
struct Specials<'a>(&'a CompiledTemplate);
impl Serialize for Specials<'_> {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
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
impl Serialize for CompiledTemplate {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("TemplateProcessing", 4)?;
        s.serialize_field("type", "TemplateProcessing")?;
        s.serialize_field("single", &Pieces(self, false))?;
        s.serialize_field("pair", &Pieces(self, true))?;
        s.serialize_field("special_tokens", &Specials(self))?;
        s.end()
    }
}
