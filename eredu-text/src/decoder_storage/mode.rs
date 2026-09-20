//! Selects Eredu's fixed-buffer kernels from public tokenizer configuration.

use tokenizers::{decoders::DecoderWrapper as D, pre_tokenizers::metaspace::PrependScheme};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FallbackOrder {
    ReplaceFirst,
    ReplaceLast,
    ByteLevelLast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Join,
    ByteLevel,
    Metaspace {
        replacement: char,
        remove_first: bool,
    },
    Fallback {
        order: FallbackOrder,
        strip: bool,
    },
}

impl Mode {
    pub(crate) fn inspect(decoder: Option<&D>) -> Option<Self> {
        let Some(decoder) = decoder else {
            return Some(Self::Join);
        };
        let decoders = match decoder {
            D::Sequence(sequence) => sequence.get_decoders(),
            single => std::slice::from_ref(single),
        };
        match decoders {
            [D::ByteLevel(_)] => return Some(Self::ByteLevel),
            [D::Metaspace(meta)] => {
                return Some(Self::Metaspace {
                    replacement: meta.get_replacement(),
                    remove_first: meta.get_prepend_scheme() != PrependScheme::Never,
                });
            }
            _ => {}
        }
        let (body, strip) = match decoders.split_last() {
            Some((D::Strip(value), rest))
                if value.content == ' ' && value.start == 1 && value.stop == 0 =>
            {
                (rest, true)
            }
            _ => (decoders, false),
        };
        let (replacement, order) = match body {
            [D::Replace(replace), D::ByteFallback(_), D::Fuse(_)] => {
                (replace, FallbackOrder::ReplaceFirst)
            }
            [D::ByteFallback(_), D::Fuse(_), D::Replace(replace)] => {
                (replace, FallbackOrder::ReplaceLast)
            }
            [
                D::ByteFallback(_),
                D::Fuse(_),
                D::Replace(replace),
                D::ByteLevel(_),
            ] => (replace, FallbackOrder::ByteLevelLast),
            _ => return None,
        };
        // Replace's pattern has no public getter. Its published serde format
        // distinguishes literal replacement from regular-expression semantics.
        let config = serde_json::to_value(replacement).ok()?;
        (replacement.content == " " && config["pattern"]["String"] == "▁")
            .then_some(Self::Fallback { order, strip })
    }
}
