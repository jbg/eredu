//! One selected-token scoring equation for ordinary execution and paid tracing.
//!
//! IDs select results, never the normalization domain. Every native scalar read
//! is explicit; the same driver exposes the exact successful-path frontiers.
use super::capture_tensor::{CaptureCompletion, CaptureTensorNativeError as Error};
use eredu_core::capture::{CaptureCandidate, CaptureTokenScore};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTensor};
use safemlx::{Array, Stream};
mod native;
mod trace;
mod population;
pub(crate) use population::TokenScorePopulation;

pub(crate) const TOKEN_SCORE_CHUNK: i32 = 1024;

/// Actual source vocabulary and borrowed immutable requested IDs. This is a
/// numerical program, not admission, source provenance, or a host destination.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TokenScoreProgram<'a> {
    vocabulary: i32,
    ids: &'a [u32],
    source_shape: Option<[usize; 3]>,
}
#[derive(Clone, Copy)]
enum Unary {
    CastF32,
    Finite,
    MaskU32,
    Sum,
    Maximum,
    Exp,
    Argmax,
}
#[derive(Clone, Copy)]
enum Binary {
    Subtract,
    Greater,
}
#[derive(Clone, Copy)]
enum Read {
    Finite(u32),
    Rank,
    Winner { excluded: u32, vocabulary: u32 },
}
impl Read {
    // Only the metadata/counting adapters consume this representative scalar.
    // Native reads always use the completed value. This chooses no allocation
    // branch: all candidate IDs have identical scalar-view geometry.
    fn representative(self) -> u32 {
        match self {
            Self::Finite(n) => n,
            Self::Rank => 0,
            Self::Winner {
                excluded,
                vocabulary,
            } => (excluded + 1) % vocabulary,
        }
    }
}
trait Kernel {
    type Value;
    fn flatten(&mut self, source: &Self::Value, vocabulary: i32) -> Result<Self::Value, Error>;
    fn terminal(&mut self, source: &Self::Value) -> Result<Self::Value, Error>;
    fn unary(&mut self, kind: Unary, source: &Self::Value) -> Result<Self::Value, Error>;
    fn scalar(&mut self, value: f32) -> Result<Self::Value, Error>;
    fn interval(&mut self, row: &Self::Value, start: i32, end: i32) -> Result<Self::Value, Error>;
    fn at(&mut self, row: &Self::Value, id: u32) -> Result<Self::Value, Error>;
    fn binary(
        &mut self,
        kind: Binary,
        left: &Self::Value,
        right: &Self::Value,
    ) -> Result<Self::Value, Error>;
    fn exclude(
        &mut self,
        row: &Self::Value,
        id: u32,
        value: &Self::Value,
    ) -> Result<Self::Value, Error>;
    fn read_f32(&mut self, value: &Self::Value) -> Result<f64, Error>;
    fn read_u32(&mut self, value: &Self::Value, kind: Read) -> Result<u32, Error>;
}
impl<'a> TokenScoreProgram<'a> {
    pub(crate) fn new(vocabulary: i32, ids: &'a [u32]) -> Result<Self, Error> {
        if vocabulary <= 0 || ids.iter().any(|&id| id >= vocabulary as u32) {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            vocabulary,
            ids,
            source_shape: None,
        })
    }
    pub(crate) fn from_geometry(
        geometry: &eredu_core::capture::CaptureTokenScoreGeometry<'a>,
    ) -> Result<Self, Error> {
        let mut program = Self::new(
            i32::try_from(geometry.vocabulary()).map_err(|_| Error::GeometryOverflow)?,
            geometry.token_ids(),
        )?;
        program.source_shape = Some(*geometry.source_shape());
        Ok(program)
    }
    pub(crate) fn validate_source(self, source: &Array) -> Result<(), Error> {
        if source.shape().last() != Some(&self.vocabulary)
            || source.size() == 0
            || !matches!(
                source.dtype(),
                safemlx::Dtype::Float32 | safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16
            )
            || self.source_shape.is_some_and(|shape| {
                source.ndim() != shape.len()
                    || source
                        .shape()
                        .iter()
                        .zip(shape)
                        .any(|(&actual, expected)| usize::try_from(actual).ok() != Some(expected))
            })
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
    pub(crate) fn validate_workspace(
        self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        use eredu_nn::Tensor;
        context.validate_values([source])?;
        if source.shape().last() != Some(&self.vocabulary)
            || source.layout().dtype() != eredu_nn::workspace::WorkspaceDtype::Float32
            || self.source_shape.is_some_and(|shape| {
                source.shape().len() != shape.len()
                    || source
                        .shape()
                        .iter()
                        .zip(shape)
                        .any(|(&actual, expected)| usize::try_from(actual).ok() != Some(expected))
            })
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(())
    }
    pub(crate) fn vocabulary(self) -> i32 {
        self.vocabulary
    }
    pub(crate) fn ids(self) -> &'a [u32] {
        self.ids
    }

    fn run<K: Kernel>(
        self,
        source: &K::Value,
        kernel: &mut K,
        mut allowed: impl FnMut(u32) -> bool,
        mut emit: impl FnMut(CaptureTokenScore) -> Result<(), Error>,
    ) -> Result<f64, Error> {
        let flat = kernel.flatten(source, self.vocabulary)?;
        let selected = kernel.terminal(&flat)?;
        let row = kernel.unary(Unary::CastF32, &selected)?;
        let finite = kernel.unary(Unary::Finite, &row)?;
        let mask = kernel.unary(Unary::MaskU32, &finite)?;
        let count = kernel.unary(Unary::Sum, &mask)?;
        if kernel.read_u32(&count, Read::Finite(self.vocabulary as u32))? != self.vocabulary as u32
        {
            return Err(Error::NonfiniteScores);
        }
        let maximum = kernel.unary(Unary::Maximum, &row)?;
        let maximum = kernel.read_f32(&maximum)?;
        let mut total = 0.0f64;
        let mut correction = 0.0f64;
        for start in (0..self.vocabulary).step_by(TOKEN_SCORE_CHUNK as usize) {
            let end = start.saturating_add(TOKEN_SCORE_CHUNK).min(self.vocabulary);
            let chunk = kernel.interval(&row, start, end)?;
            let peak = kernel.scalar(maximum as f32)?;
            let centered = kernel.binary(Binary::Subtract, &chunk, &peak)?;
            let exponentials = kernel.unary(Unary::Exp, &centered)?;
            let sum = kernel.unary(Unary::Sum, &exponentials)?;
            let sum = kernel.read_f32(&sum)?;
            let next = total + sum;
            correction += if total.abs() >= sum.abs() {
                (total - next) + sum
            } else {
                (sum - next) + total
            };
            total = next;
        }
        let log_mass = (total + correction).ln();
        for &id in self.ids {
            let selected = kernel.at(&row, id)?;
            let score = kernel.read_f32(&selected)? as f32;
            let score_scalar = kernel.scalar(score)?;
            let larger = kernel.binary(Binary::Greater, &row, &score_scalar)?;
            let mask = kernel.unary(Unary::MaskU32, &larger)?;
            let count = kernel.unary(Unary::Sum, &mask)?;
            let rank = 1 + u64::from(kernel.read_u32(&count, Read::Rank)?);
            let strongest_alternative = if self.vocabulary == 1 {
                None
            } else {
                let excluded = kernel.scalar(f32::NEG_INFINITY)?;
                let alternatives = kernel.exclude(&row, id, &excluded)?;
                let winner = kernel.unary(Unary::Argmax, &alternatives)?;
                let winner = kernel.read_u32(
                    &winner,
                    Read::Winner {
                        excluded: id,
                        vocabulary: self.vocabulary as u32,
                    },
                )?;
                let selected = kernel.at(&row, winner)?;
                let value = kernel.read_f32(&selected)? as f32;
                Some(CaptureCandidate {
                    token_id: winner,
                    score: value,
                    allowed: allowed(winner),
                })
            };
            emit(CaptureTokenScore {
                target: CaptureCandidate {
                    token_id: id,
                    score,
                    allowed: allowed(id),
                },
                log_probability: (f64::from(score) - maximum) - log_mass,
                rank,
                strongest_alternative,
            })?;
        }
        Ok(maximum + log_mass)
    }

    /// The caller authenticates/pins its source and fixed host claim first.
    /// Every intermediate is retained before any subsequent fallible operation;
    /// each scalar uses the explicitly supplied ordinary/original completion.
    pub(crate) fn execute(
        self,
        source: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        retain: &mut dyn FnMut(&Array) -> Result<(), Error>,
        host_read: &mut dyn FnMut(usize),
        allowed: impl FnMut(u32) -> bool,
        emit: impl FnMut(CaptureTokenScore) -> Result<(), Error>,
    ) -> Result<f64, Error> {
        self.validate_source(source)?;
        completion.validate()?;
        self.run(
            source,
            &mut native::Native {
                stream,
                completion,
                retain,
                host_read,
            },
            allowed,
            emit,
        )
    }
    pub(crate) fn trace(
        self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<(), Error> {
        use eredu_nn::Tensor;
        self.validate_workspace(source, context)?;
        self.run(
            source,
            &mut trace::Trace { context, retained },
            |_| true,
            |_| Ok(()),
        )?;
        Ok(())
    }
}
