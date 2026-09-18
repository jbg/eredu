//! Ordered composition of the existing primitive normalization workers.
use super::{Leaves, Shape};
use crate::normalizers::NormalizerWrapper as N;
use std::{collections::TryReserveError, fmt, mem::size_of, sync::OnceLock};
use unicode_normalization_alignments::workspace as nfc;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Step {
    Nfc,
    Lowercase,
    Prepend(usize),
    Replace { pattern: usize, content: usize },
}
/// Cold checked geometry only. Actual NFC allocations require the genuine
/// intermediate source Plan, checked against these reserved destination facts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub(crate) output: usize,
    pub(crate) text: usize,
    nfc: Option<nfc::Requirements>,
}
impl Bounds {
    pub(crate) fn new(bytes: usize) -> Self {
        Self {
            output: bytes,
            text: bytes,
            nfc: None,
        }
    }
    pub(crate) fn step(&mut self, step: Step) -> Option<()> {
        if self.output == 0 {
            return Some(());
        }
        self.output = match step {
            Step::Nfc => {
                let r = nfc::utf8_requirements(self.output).ok()?;
                if self
                    .nfc
                    .is_none_or(|old| old.buffer_bytes() < r.buffer_bytes())
                {
                    self.nfc = Some(r);
                }
                r.text_capacity()
            }
            Step::Lowercase => self.output.checked_mul(lowercase_ratio())?,
            Step::Prepend(bytes) => self.output.checked_add(bytes)?,
            Step::Replace { pattern, content } => {
                crate::normalizers::Replace::literal_bound(self.output, pattern, content)?
            }
        };
        self.text = self.text.max(self.output);
        Some(())
    }
    pub(crate) fn buffers(self) -> Option<usize> {
        self.text
            .checked_mul(2)?
            .checked_add(self.nfc.map_or(0, |r| r.buffer_bytes()))
    }
    pub(crate) fn capacities(self) -> [usize; 3] {
        let scalars = self.nfc.map_or(0, |r| r.scalar_capacity());
        [scalars, scalars, self.text * 2]
    }
    pub(crate) fn controls(self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Storage>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Result<Storage, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Leaves<'_>>(),
            size_of::<Step>(),
            size_of::<Shape<'_>>(),
            size_of::<std::str::CharIndices<'_>>(),
            size_of::<std::char::ToLowercase>(),
            size_of::<(char, usize, usize, usize, usize, bool)>(),
            size_of::<(&str, &mut String)>(),
            size_of::<[usize; 3]>(),
            self.nfc.map_or(0, |r| r.control_bytes()),
            crate::normalizers::Replace::literal_control_bytes(),
        ]
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add)
    }
}
/// The maximum is computed from the same selected Rust Unicode mapping, once,
/// using fixed controls only. No guessed UTF-8 multiplier or allocated table.
fn lowercase_ratio() -> usize {
    static RATIO: OnceLock<usize> = OnceLock::new();
    *RATIO.get_or_init(|| {
        (0..=0x10ffff)
            .filter_map(char::from_u32)
            .fold(1, |ratio, c| {
                let bytes: usize = crate::tokenizer::normalizer::lowercase_changes(c)
                    .map(|(c, _)| c.len_utf8())
                    .sum();
                ratio.max(bytes.div_ceil(c.len_utf8()))
            })
    })
}
impl Shape<'_> {
    pub(crate) fn pipeline_bounds(self, bytes: usize) -> Option<Bounds> {
        let mut bounds = Bounds::new(bytes);
        for leaf in Leaves::new(self.pipeline, false) {
            let step = match leaf.ok()? {
                N::NFC(_) => Step::Nfc,
                N::Lowercase(_) => Step::Lowercase,
                N::Prepend(value) if !self.remove_prefixes => Step::Prepend(value.prepend.len()),
                N::Prepend(_) => continue,
                N::Replace(value) => {
                    let (pattern, content) = value.literal_parts()?;
                    Step::Replace {
                        pattern: pattern.len(),
                        content: content.len(),
                    }
                }
                _ => return None,
            };
            bounds.step(step)?;
        }
        bounds.buffers()?;
        Some(bounds)
    }
}
#[derive(Debug)]
pub enum Cause {
    Reserve(TryReserveError),
    Nfc(nfc::PrepareFailure),
    Capacity,
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => e.fmt(f),
            Self::Nfc(e) => e.fmt(f),
            Self::Capacity => f.write_str("ordered normalization destination bound exceeded"),
        }
    }
}
impl std::error::Error for Cause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(e) => Some(e),
            Self::Nfc(e) => Some(e),
            Self::Capacity => None,
        }
    }
}
#[derive(Debug)]
pub struct Failure {
    pub(crate) cause: Cause,
    storage: Storage,
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
impl Failure {
    pub(crate) fn new(cause: Cause, storage: Storage) -> Self {
        Self { cause, storage }
    }
    pub fn capacities(&self) -> [usize; 3] {
        self.storage.capacities()
    }
    pub fn reserve_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Reserve(e) => Some(e),
            Cause::Nfc(e) => Some(e.reserve_error()),
            Cause::Capacity => None,
        }
    }
}
#[derive(Debug)]
pub(crate) struct Storage {
    buffers: [String; 2],
    bounds: Bounds,
    failure: Option<usize>,
}
impl Storage {
    pub(crate) fn prepare(bounds: Bounds, failure: Option<usize>) -> Result<Self, Failure> {
        let mut storage = Self {
            buffers: [String::new(), String::new()],
            bounds,
            failure,
        };
        for at in 0..2 {
            let requested = if failure == Some(at) {
                usize::MAX
            } else {
                bounds.text
            };
            if let Err(cause) = storage.buffers[at].try_reserve_exact(requested) {
                return Err(Failure {
                    cause: Cause::Reserve(cause),
                    storage,
                });
            }
        }
        Ok(storage)
    }
    pub(crate) fn capacities(&self) -> [usize; 3] {
        [
            0,
            0,
            self.buffers[0].capacity() + self.buffers[1].capacity(),
        ]
    }
    pub(crate) fn apply(&mut self, shape: Shape<'_>, input: &str) -> Result<(&str, usize), Cause> {
        if input.len() > self.bounds.text {
            return Err(Cause::Capacity);
        }
        self.buffers[0].clear();
        self.buffers[0].push_str(input);
        if input.is_empty() {
            return Ok((&self.buffers[0], 0));
        }
        let mut initial = input.chars().next().map_or(0, char::len_utf8);
        let mut current = 0;
        for leaf in Leaves::new(shape.pipeline, false) {
            let leaf = leaf.map_err(|_| Cause::Capacity)?;
            if shape.remove_prefixes && matches!(leaf, N::Prepend(_)) {
                continue;
            }
            let [a, b] = &mut self.buffers;
            let (source, dest) = if current == 0 { (&*a, b) } else { (&*b, a) };
            dest.clear();
            match leaf {
                N::NFC(_) => {
                    let plan = nfc::Plan::new(source, "").map_err(|_| Cause::Capacity)?;
                    #[cfg(feature = "tokenizer-compiler-test-support")]
                    let plan = match self.failure {
                        Some(2) => plan.fail_reservation(nfc::Buffer::Decomposition),
                        Some(3) => plan.fail_reservation(nfc::Buffer::Recomposition),
                        Some(4) => plan.fail_reservation(nfc::Buffer::Text),
                        _ => plan,
                    };
                    let r = plan.requirements();
                    if self.bounds.nfc.is_none_or(|limit| {
                        r.buffer_bytes() > limit.buffer_bytes()
                            || r.control_bytes() > limit.control_bytes()
                    }) {
                        return Err(Cause::Capacity);
                    }
                    let mut workspace = plan.prepare().map_err(Cause::Nfc)?;
                    workspace
                        .normalize_with_initial_origin(0..source.len(), initial)
                        .map_err(|_| Cause::Capacity)?;
                    let (text, frontier) = workspace.normalized();
                    if text.len() > self.bounds.text {
                        return Err(Cause::Capacity);
                    }
                    dest.push_str(text);
                    initial = frontier;
                }
                N::Lowercase(_) => {
                    let mut frontier = 0;
                    for (at, c) in source.char_indices() {
                        for (c, _) in crate::tokenizer::normalizer::lowercase_changes(c) {
                            dest.push(c);
                        }
                        if at < initial {
                            frontier = dest.len();
                        }
                    }
                    initial = frontier;
                }
                N::Prepend(value) => {
                    if !source.is_empty() {
                        dest.push_str(&value.prepend);
                    }
                    dest.push_str(source);
                    if initial != 0 {
                        initial += value.prepend.len();
                    }
                }
                N::Replace(value) => {
                    initial = value.write_literal_with_initial_origin(source, initial, dest);
                }
                _ => return Err(Cause::Capacity),
            }
            assert!(
                dest.len() <= self.bounds.text && dest.capacity() <= self.bounds.text,
                "checked ordered normalization capacity"
            );
            current = 1 - current;
        }
        Ok((&self.buffers[current], initial))
    }
}
