//! Complete retained configuration serialized by the ordinary source workers.
//!
//! The caller retains the borrowed allocation account until the returned bytes
//! or failed prefix retire. This does not adopt the already-existing tokenizer.
use super::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer,
};
use crate::models::{bpe::BpeSerialization, wordlevel::WordLevelSerialization};
use serde::{Serialize, Serializer};
use serde_json::allocation::Allocator;
pub use serde_json::allocation::{Allocation, AllocationError, Unenforced};
use std::{fmt, io, mem::size_of};

/// Fixed source-preparation failure or actual JSON writer diagnostic.
#[derive(Debug)]
pub enum Error {
    /// The source account or host allocator refused a reached destination.
    Allocation(AllocationError),
    /// A component's serializer does not yet expose its original producers.
    Unqualified(&'static str),
    /// The original JSON serializer's diagnostic.
    Json(serde_json::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation(e) => e.fmt(f),
            Self::Unqualified(component) => {
                write!(f, "unqualified tokenizer source serializer: {component}")
            }
            Self::Json(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}
impl From<AllocationError> for Error {
    fn from(value: AllocationError) -> Self {
        Self::Allocation(value)
    }
}
/// Failure retains the actual serialized prefix; its payer remains caller-owned.
#[derive(Debug)]
pub struct Failure {
    cause: Error,
    bytes: Vec<u8>,
}
impl Failure {
    /// Borrow the original typed failure.
    pub fn cause(&self) -> &Error {
        &self.cause
    }
    /// Actual initialized bytes before the failed write.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Actual retained output backing, including spare capacity.
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

struct Output<'a> {
    bytes: Vec<u8>,
    allocation: &'a dyn Allocation,
    failure: Option<AllocationError>,
}
impl io::Write for Output<'_> {
    fn write(&mut self, source: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::ErrorKind::OutOfMemory.into());
        }
        let result = self
            .bytes
            .len()
            .checked_add(source.len())
            .ok_or(AllocationError::SizeOverflow)
            .and_then(|required| Allocator::new(self.allocation).grow(&mut self.bytes, required));
        if let Err(error) = result {
            self.failure = Some(error);
            return Err(io::ErrorKind::OutOfMemory.into());
        }
        self.bytes.extend_from_slice(source);
        Ok(source.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

enum Model<'a> {
    Bpe(BpeSerialization<'a>),
    WordLevel(WordLevelSerialization<'a>),
    Borrowed(&'a ModelWrapper),
}
impl Serialize for Model<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Bpe(value) => value.serialize(serializer),
            Self::WordLevel(value) => value.serialize(serializer),
            Self::Borrowed(value) => value.serialize(serializer),
        }
    }
}
struct Root<'a> {
    source: &'a Tokenizer,
    model: Model<'a>,
}
impl Serialize for Root<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.source.serialize_with_model(serializer, &self.model)
    }
}

// These are concrete serializer/input/result representations. The recursive
// component walk pays each active container before descending and allocates no
// census or cloned component graph. Source-owned strings/vectors remain borrows.
fn frame<T>(allocation: &dyn Allocation) -> Result<(), Error> {
    let parts = [
        size_of::<&T>(),
        size_of::<std::slice::Iter<'_, T>>(),
        size_of::<Result<(), Error>>(),
        size_of::<serde_json::Serializer<&mut Output<'_>>>(),
        size_of::<serde_json::ser::Compound<'_, &mut Output<'_>, serde_json::ser::CompactFormatter>>(
        ),
    ];
    let bytes = parts
        .iter()
        .copied()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
        .ok_or(AllocationError::SizeOverflow)?;
    allocation.reserve(bytes)?;
    Ok(())
}
// Every nested component creates at least one JSON container. Beyond the
// original compiler's fixed JSON depth no complete source can be accepted;
// refuse before recursive serialization could exceed that same profile.
fn component_depth(depth: usize) -> Result<(), Error> {
    if depth >= crate::utils::borrowed_json::DEPTH {
        return Err(Error::Unqualified(
            "configuration nesting exceeds the original JSON compiler",
        ));
    }
    Ok(())
}
fn normalizer(
    value: &NormalizerWrapper,
    depth: usize,
    allocation: &dyn Allocation,
) -> Result<(), Error> {
    component_depth(depth)?;
    frame::<NormalizerWrapper>(allocation)?;
    match value {
        NormalizerWrapper::Sequence(sequence) => {
            for value in sequence.as_ref() {
                normalizer(value, depth + 1, allocation)?;
            }
        }
        NormalizerWrapper::Precompiled(_) => {
            return Err(Error::Unqualified("precompiled normalizer serialization"))
        }
        _ => {}
    }
    Ok(())
}
fn pre_tokenizer(
    value: &PreTokenizerWrapper,
    depth: usize,
    allocation: &dyn Allocation,
) -> Result<(), Error> {
    component_depth(depth)?;
    frame::<PreTokenizerWrapper>(allocation)?;
    if let PreTokenizerWrapper::Sequence(sequence) = value {
        for value in sequence.as_ref() {
            pre_tokenizer(value, depth + 1, allocation)?;
        }
    }
    Ok(())
}
fn post_processor(
    value: &PostProcessorWrapper,
    depth: usize,
    allocation: &dyn Allocation,
) -> Result<(), Error> {
    component_depth(depth)?;
    frame::<PostProcessorWrapper>(allocation)?;
    if let PostProcessorWrapper::Sequence(sequence) = value {
        for value in sequence.as_ref() {
            post_processor(value, depth + 1, allocation)?;
        }
    }
    Ok(())
}
fn decoder(value: &DecoderWrapper, depth: usize, allocation: &dyn Allocation) -> Result<(), Error> {
    component_depth(depth)?;
    frame::<DecoderWrapper>(allocation)?;
    if let DecoderWrapper::Sequence(sequence) = value {
        for value in sequence.get_decoders() {
            decoder(value, depth + 1, allocation)?;
        }
    }
    Ok(())
}

/// Serialize the complete immutable tokenizer configuration through its same
/// ordinary field/model workers. Every new scratch/output destination is paid
/// prospectively; no vocabulary-only reconstruction or source mutation occurs.
pub fn serialize(source: &Tokenizer, allocation: &dyn Allocation) -> Result<Vec<u8>, Failure> {
    let mut output = Output {
        bytes: Vec::new(),
        allocation,
        failure: None,
    };
    let result = (|| {
        let controls = [
            size_of::<Output<'_>>(),
            size_of::<Root<'_>>(),
            size_of::<Model<'_>>(),
            size_of::<Failure>(),
            size_of::<Result<Vec<u8>, Failure>>(),
            size_of::<serde_json::Error>(),
            serde_json::Error::io_storage_bytes(),
        ];
        allocation.reserve(
            controls
                .iter()
                .copied()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(AllocationError::SizeOverflow)?,
        )?;
        frame::<Tokenizer>(allocation)?;
        if let Some(value) = source.get_normalizer() {
            normalizer(value, 0, allocation)?;
        }
        if let Some(value) = source.get_pre_tokenizer() {
            pre_tokenizer(value, 0, allocation)?;
        }
        if let Some(value) = source.get_post_processor() {
            post_processor(value, 0, allocation)?;
        }
        if let Some(value) = source.get_decoder() {
            decoder(value, 0, allocation)?;
        }
        let model = match source.get_model() {
            ModelWrapper::BPE(value) => {
                Model::Bpe(value.serialization_with_allocations(allocation)?)
            }
            ModelWrapper::WordLevel(value) => {
                Model::WordLevel(value.serialization_with_allocations(allocation)?)
            }
            value @ ModelWrapper::Unigram(_) => Model::Borrowed(value),
            ModelWrapper::WordPiece(_) => {
                return Err(Error::Unqualified(
                    "WordPiece ordered vocabulary serialization",
                ))
            }
        };
        let root = Root { source, model };
        let result = root.serialize(&mut serde_json::Serializer::new(&mut output));
        if let Some(error) = output.failure {
            return Err(Error::Allocation(error));
        }
        result.map_err(Error::Json)
    })();
    match result {
        Ok(()) => Ok(output.bytes),
        Err(cause) => Err(Failure {
            cause,
            bytes: output.bytes,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    struct Meter {
        calls: Cell<usize>,
        refuse: usize,
    }
    impl Allocation for Meter {
        fn reserve(&self, _: usize) -> Result<(), AllocationError> {
            let call = self.calls.get();
            assert!(call <= self.refuse, "serialization continued after refusal");
            self.calls.set(call + 1);
            if call == self.refuse {
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    fn source(model: serde_json::Value) -> Tokenizer {
        let value = serde_json::json!({"version":"1.0","model":model,"added_tokens":[
            {"id":4,"content":"<tool>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
            "normalizer":{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Prepend","prepend":" "}]},
            "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"}]},
            "decoder":{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},
            "post_processor":{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}
        });
        Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap()
    }
    #[test]
    fn sparse_wordlevel_serialization_uses_actual_ids_without_traversing_holes() {
        let source=Tokenizer::from_bytes(br#"{"model":{"type":"WordLevel","vocab":{"[UNK]":0,"last":4000000000,"middle":50},"unk_token":"[UNK]"}}"#).unwrap();
        let meter = Meter {
            calls: Cell::new(0),
            refuse: usize::MAX,
        };
        let actual = serialize(&source, &meter).unwrap();
        let text = std::str::from_utf8(&actual).unwrap();
        assert!(text.contains(r#""vocab":{"[UNK]":0,"middle":50,"last":4000000000}"#));
        for refuse in 0..meter.calls.get() {
            let payer = Meter {
                calls: Cell::new(0),
                refuse,
            };
            let error = serialize(&source, &payer).unwrap_err();
            assert!(matches!(
                error.cause(),
                Error::Allocation(AllocationError::Refused)
            ));
            assert_eq!(payer.calls.get(), refuse + 1);
            assert!(actual.starts_with(error.bytes()));
        }
    }
    #[test]
    fn nested_component_source_refuses_at_the_original_compiler_depth() {
        let mut source = Tokenizer::new(crate::models::bpe::BPE::default());
        let mut normalizer = NormalizerWrapper::NFC(crate::normalizers::NFC);
        for _ in 0..crate::utils::borrowed_json::DEPTH {
            normalizer =
                NormalizerWrapper::Sequence(crate::normalizers::Sequence::new(vec![normalizer]));
        }
        source.with_normalizer(Some(normalizer)).unwrap();
        let error = serialize(&source, &Unenforced).unwrap_err();
        assert!(matches!(error.cause(), Error::Unqualified(_)));
        assert_eq!(error.capacity(), 0);
    }
    #[test]
    fn complete_configuration_uses_original_serializer_and_retains_refused_prefix() {
        for model in [
            serde_json::json!({"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"[UNK]":3},"merges":[["a","b"]],"unk_token":"[UNK]","fuse_unk":true,"byte_fallback":false,"ignore_merges":false}),
            serde_json::json!({"type":"WordLevel","vocab":{"a":0,"b":1,"ab":2,"[UNK]":3},"unk_token":"[UNK]"}),
            serde_json::json!({"type":"Unigram","vocab":[["[UNK]",0.0],["a",-0.5],["b",-0.7],["ab",-0.4]],"unk_id":0,"byte_fallback":true}),
        ] {
            let source = source(model);
            let expected = serde_json::to_vec(&source).unwrap();
            let meter = Meter {
                calls: Cell::new(0),
                refuse: usize::MAX,
            };
            let actual = serialize(&source, &meter).unwrap();
            assert_eq!(actual, expected);
            let restored = Tokenizer::from_bytes(&actual).unwrap();
            for input in ["ab a", "<tool>ab", "é a"] {
                assert_eq!(
                    source.encode(input, true).unwrap().get_ids(),
                    restored.encode(input, true).unwrap().get_ids()
                );
            }
            let mut retained_prefix = false;
            for refuse in 0..meter.calls.get() {
                let payer = Meter {
                    calls: Cell::new(0),
                    refuse,
                };
                let error = serialize(&source, &payer).unwrap_err();
                assert!(matches!(
                    error.cause(),
                    Error::Allocation(AllocationError::Refused)
                ));
                assert_eq!(payer.calls.get(), refuse + 1);
                assert!(expected.starts_with(error.bytes()));
                assert!(error.capacity() >= error.bytes().len());
                retained_prefix |= !error.bytes().is_empty();
            }
            assert!(retained_prefix);
            assert_eq!(serde_json::to_vec(&source).unwrap(), expected);
        }
    }
}
