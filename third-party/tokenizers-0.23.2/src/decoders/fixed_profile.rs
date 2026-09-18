//! Shared finite decoder forms for borrowed admission and fixed-destination decoding.
use super::DecoderWrapper;
use crate::normalizers::replace::ReplacePattern;

/// Exact component facts used by the fixed-destination decoder classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    /// Existing byte-to-Unicode inverse decoder.
    ByteLevel,
    /// Hexadecimal fallback token decoder.
    ByteFallback,
    /// Concatenates decoded token pieces.
    Fuse,
    /// Literal U+2581 replacement with one ASCII space.
    MarkerReplace,
    /// Removes one leading ASCII space and no trailing characters.
    InitialSpaceStrip,
    /// Ordinary Metaspace replacement and first-token removal policy.
    Metaspace { replacement: char, remove_first: bool },
    /// Another component or parameter configuration.
    Other,
}
impl Component {
    /// Borrows actual component fields without serialization or matcher construction.
    pub fn from_decoder(decoder: &DecoderWrapper) -> Self {
        match decoder {
            DecoderWrapper::ByteLevel(_) => Self::ByteLevel,
            DecoderWrapper::ByteFallback(_) => Self::ByteFallback,
            DecoderWrapper::Fuse(_) => Self::Fuse,
            DecoderWrapper::Metaspace(value) => Self::Metaspace {
                replacement: value.get_replacement(),
                remove_first: value.get_prepend_scheme() != crate::pre_tokenizers::metaspace::PrependScheme::Never,
            },
            DecoderWrapper::Replace(value)
                if matches!(value.pattern(), ReplacePattern::String(pattern) if pattern == "▁")
                    && value.content == " " =>
            {
                Self::MarkerReplace
            }
            DecoderWrapper::Strip(value)
                if value.content == ' ' && value.start == 1 && value.stop == 0 =>
            {
                Self::InitialSpaceStrip
            }
            _ => Self::Other,
        }
    }
}
/// The selected literal replacement order relative to fallback byte decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackOrder {
    /// Replace marker scalars in token pieces before fallback decoding.
    ReplaceFirst,
    /// Replace marker scalars after fallback decoding and concatenation.
    ReplaceLast,
    /// Apply ByteLevel after replacement of the concatenated fallback output.
    ByteLevelLast,
}
/// Exact existing fixed-destination decoder forms; no allocation authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// No explicit decoder: join token spellings with spaces.
    Join,
    /// A single ByteLevel decoder.
    ByteLevel,
    /// Replace markers in each retained token, removing them from the first
    /// token when the original prepend policy selects that behavior.
    Metaspace { replacement: char, remove_first: bool },
    /// One supported fallback ordering, optionally stripping its first space.
    Fallback {
        /// Position of literal replacement and optional ByteLevel conversion.
        order: FallbackOrder,
        /// Remove one leading space after the complete sequence.
        strip: bool,
    },
}
impl Profile {
    /// Classifies exact ordered component facts, without constructing components.
    pub fn sequence(components: &[Component]) -> Option<Self> {
        use Component::*;
        if components == [ByteLevel] {
            return Some(Self::ByteLevel);
        }
        if let [Metaspace { replacement, remove_first }] = components {
            return Some(Self::Metaspace { replacement: *replacement, remove_first: *remove_first });
        }
        let (components, strip) = match components.split_last() {
            Some((InitialSpaceStrip, rest)) => (rest, true),
            _ => (components, false),
        };
        let order = match components {
            [MarkerReplace, ByteFallback, Fuse] => FallbackOrder::ReplaceFirst,
            [ByteFallback, Fuse, MarkerReplace] => FallbackOrder::ReplaceLast,
            [ByteFallback, Fuse, MarkerReplace, ByteLevel] => FallbackOrder::ByteLevelLast,
            _ => return None,
        };
        Some(Self::Fallback { order, strip })
    }
    /// Applies the same classifier to the actual retained decoder objects.
    pub fn from_decoder(decoder: Option<&DecoderWrapper>) -> Option<Self> {
        let Some(decoder) = decoder else {
            return Some(Self::Join);
        };
        match decoder {
            DecoderWrapper::ByteLevel(_) => Some(Self::ByteLevel),
            DecoderWrapper::Metaspace(_) => Self::sequence(&[Component::from_decoder(decoder)]),
            DecoderWrapper::Sequence(sequence) => {
                let values = sequence.get_decoders();
                let mut components = [Component::Other; 5];
                let selected = components.get_mut(..values.len())?;
                for (destination, value) in selected.iter_mut().zip(values) {
                    *destination = Component::from_decoder(value);
                }
                Self::sequence(selected)
            }
            _ => None,
        }
    }
    /// Named fixed classifier storage, facts and result transport.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            std::mem::size_of::<[Component; 5]>(),
            std::mem::size_of::<Component>(),
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Option<Self>>(),
            std::mem::size_of::<(&[Component], bool)>(),
        ];
        parts
            .iter()
            .try_fold(std::mem::size_of_val(&parts), |sum, &part| {
                sum.checked_add(part)
            })
    }
}
