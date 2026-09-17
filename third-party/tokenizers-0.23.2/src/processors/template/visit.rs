//! One ordered projection used by ordinary Encoding and original ID destinations.
use super::*;

#[derive(Clone, Copy)]
pub(crate) enum Texts<'a> {
    Legacy(&'a [String]),
    Packed(&'a [compiled::TextRange], &'a [u8]),
}
impl Texts<'_> {
    pub(crate) fn owned(&self) -> Vec<String> {
        match self {
            Self::Legacy(tokens) => tokens.to_vec(),
            Self::Packed(ranges, bytes) => {
                ranges.iter().map(|r| r.text(bytes).to_owned()).collect()
            }
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) enum PieceView<'a> {
    Sequence {
        index: usize,
        type_id: u32,
    },
    Special {
        ids: &'a [u32],
        tokens: Texts<'a>,
        type_id: u32,
    },
}

pub(crate) fn visit<'a, E>(
    pieces: impl Iterator<Item = PieceView<'a>>,
    add_special_tokens: bool,
    mut sink: impl FnMut(PieceView<'a>) -> StdResult<(), E>,
) -> StdResult<(), E> {
    for piece in pieces {
        if !add_special_tokens && matches!(piece, PieceView::Special { .. }) {
            continue;
        }
        sink(piece)?;
    }
    Ok(())
}

pub(super) fn encodings<'a>(
    pieces: impl Iterator<Item = PieceView<'a>>,
    mut encodings: Vec<Encoding>,
    add_special_tokens: bool,
) -> Result<Vec<Encoding>> {
    let mut output = Vec::new();
    visit(pieces, add_special_tokens, |piece| {
        let encoding = match piece {
            PieceView::Sequence { index, type_id } => {
                let encoding = &mut encodings[index];
                encoding.set_type_ids(vec![type_id; encoding.len()]);
                encoding.set_sequence_id(index);
                encoding.clone()
            }
            PieceView::Special {
                ids,
                tokens,
                type_id,
            } => {
                let len = ids.len();
                Encoding::new(
                    ids.to_vec(),
                    std::iter::repeat_n(type_id, len).collect(),
                    tokens.owned(),
                    std::iter::repeat_n(None, len).collect(),
                    std::iter::repeat_n((0, 0), len).collect(),
                    std::iter::repeat_n(1, len).collect(),
                    std::iter::repeat_n(1, len).collect(),
                    vec![],
                    AHashMap::new(),
                )
            }
        };
        output.push(encoding);
        Ok::<(), crate::Error>(())
    })?;
    Ok(output)
}
