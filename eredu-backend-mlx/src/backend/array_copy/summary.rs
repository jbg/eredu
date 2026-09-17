//! Existing finite-only summary equation, shared by native work and cold tracing.
use super::capture_tensor::{CaptureCompletion, CaptureTensorNativeError as Error};
use eredu_core::capture::CaptureSummary;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTensor};
use safemlx::{Array, Stream};
mod native;
mod population;
mod trace;
pub(crate) use population::SummaryPopulation;

pub(crate) const SUMMARY_CHUNK: i32 = 1024;
#[derive(Clone, Copy, Debug)]
pub(crate) struct SummaryProgram {
    elements: i32,
}
#[derive(Clone, Copy)]
enum Unary {
    CastF32,
    Finite,
    Nan,
    PositiveInfinity,
    NegativeInfinity,
    MaskU32,
    Sum,
    Minimum,
    Maximum,
}
#[derive(Clone, Copy)]
enum Binary {
    Divide,
    Multiply,
}
#[derive(Clone, Copy)]
enum ScalarRead {
    Minimum,
    Maximum,
    Moment,
}
impl ScalarRead {
    fn representative(self) -> f64 {
        match self {
            Self::Minimum => -1.0,
            Self::Maximum => 1.0,
            Self::Moment => 0.0,
        }
    }
}
trait Kernel {
    type Value;
    fn interval(
        &mut self,
        source: &Self::Value,
        start: i32,
        end: i32,
    ) -> Result<Self::Value, Error>;
    fn unary(&mut self, kind: Unary, source: &Self::Value) -> Result<Self::Value, Error>;
    fn scalar(&mut self, value: f32) -> Result<Self::Value, Error>;
    fn select(
        &mut self,
        mask: &Self::Value,
        source: &Self::Value,
        other: &Self::Value,
    ) -> Result<Self::Value, Error>;
    fn binary(
        &mut self,
        kind: Binary,
        left: &Self::Value,
        right: &Self::Value,
    ) -> Result<Self::Value, Error>;
    fn read_u32(&mut self, value: &Self::Value, representative: u32) -> Result<u32, Error>;
    fn read_f32(&mut self, value: &Self::Value, kind: ScalarRead) -> Result<f64, Error>;
}
impl SummaryProgram {
    pub(crate) fn new(elements: i32) -> Result<Self, Error> {
        if elements < 0 {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { elements })
    }
    fn count<K: Kernel>(
        kernel: &mut K,
        mask: &K::Value,
        representative: u32,
    ) -> Result<u64, Error> {
        let mask = kernel.unary(Unary::MaskU32, mask)?;
        let sum = kernel.unary(Unary::Sum, &mask)?;
        Ok(u64::from(kernel.read_u32(&sum, representative)?))
    }
    /// Metadata/count adapters choose finite>0, min=-1/max=1 so both actual
    /// conditional suffixes are included. Native reads choose the real branch.
    /// The alternate paths only omit these enumerated operations; shapes and
    /// primitive storage classes of the common prefix are unchanged.
    fn run<K: Kernel>(self, flat: &K::Value, kernel: &mut K) -> Result<CaptureSummary, Error> {
        let mut out = CaptureSummary {
            elements: self.elements as u64,
            finite: 0,
            non_finite: 0,
            nan: 0,
            positive_infinity: 0,
            negative_infinity: 0,
            min: None,
            max: None,
            mean: None,
            rms: None,
        };
        let mut sum = 0.0f64;
        let mut squares = 0.0f64;
        for start in (0..self.elements).step_by(SUMMARY_CHUNK as usize) {
            let end = start.saturating_add(SUMMARY_CHUNK).min(self.elements);
            let selected = kernel.interval(flat, start, end)?;
            let chunk = kernel.unary(Unary::CastF32, &selected)?;
            let finite = kernel.unary(Unary::Finite, &chunk)?;
            let count = Self::count(kernel, &finite, (end - start) as u32)?;
            out.finite += count;
            let mask = kernel.unary(Unary::Nan, &chunk)?;
            out.nan += Self::count(kernel, &mask, 0)?;
            let mask = kernel.unary(Unary::PositiveInfinity, &chunk)?;
            out.positive_infinity += Self::count(kernel, &mask, 0)?;
            let mask = kernel.unary(Unary::NegativeInfinity, &chunk)?;
            out.negative_infinity += Self::count(kernel, &mask, 0)?;
            if count == 0 {
                continue;
            }
            let infinity = kernel.scalar(f32::INFINITY)?;
            let selected = kernel.select(&finite, &chunk, &infinity)?;
            let minimum = kernel.unary(Unary::Minimum, &selected)?;
            let min = kernel.read_f32(&minimum, ScalarRead::Minimum)?;
            let infinity = kernel.scalar(f32::NEG_INFINITY)?;
            let selected = kernel.select(&finite, &chunk, &infinity)?;
            let maximum = kernel.unary(Unary::Maximum, &selected)?;
            let max = kernel.read_f32(&maximum, ScalarRead::Maximum)?;
            out.min = Some(out.min.map_or(min, |value| value.min(min)));
            out.max = Some(out.max.map_or(max, |value| value.max(max)));
            let scale = min.abs().max(max.abs());
            if scale != 0.0 {
                // Construct only when consumed. Previously this Where was born
                // before min/max and left unevaluated for an all-zero chunk.
                let zero = kernel.scalar(0.0)?;
                let clean = kernel.select(&finite, &chunk, &zero)?;
                let divisor = kernel.scalar(scale as f32)?;
                let scaled = kernel.binary(Binary::Divide, &clean, &divisor)?;
                let total = kernel.unary(Unary::Sum, &scaled)?;
                sum += kernel.read_f32(&total, ScalarRead::Moment)? * scale;
                let squared = kernel.binary(Binary::Multiply, &scaled, &scaled)?;
                let total = kernel.unary(Unary::Sum, &squared)?;
                squares += kernel.read_f32(&total, ScalarRead::Moment)? * scale * scale;
            }
        }
        if out.finite != 0 {
            out.mean = Some(sum / out.finite as f64);
            out.rms = Some((squares / out.finite as f64).sqrt());
        }
        out.non_finite = out.elements - out.finite;
        Ok(out)
    }
    pub(crate) fn execute(
        self,
        flat: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        retain: &mut dyn FnMut(&Array) -> Result<(), Error>,
        host_read: &mut dyn FnMut(usize),
    ) -> Result<CaptureSummary, Error> {
        if flat.shape() != [self.elements] {
            return Err(Error::ShapeMismatch);
        }
        if matches!(completion, CaptureCompletion::Original(_))
            && !matches!(
                flat.dtype(),
                safemlx::Dtype::Float32 | safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16
            )
        {
            return Err(Error::UnsupportedDtype(flat.dtype()));
        }
        // Empty summaries make no native submission; identity remains mandatory.
        completion.validate_identity()?;
        if self.elements != 0 {
            completion.validate()?;
        }
        self.run(
            flat,
            &mut native::Native {
                stream,
                completion,
                retain,
                host_read,
            },
        )
    }
    pub(crate) fn trace(
        self,
        flat: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<(), Error> {
        use eredu_nn::Tensor;
        context.validate_values([flat])?;
        if flat.shape() != [self.elements]
            || flat.layout().dtype() != eredu_nn::workspace::WorkspaceDtype::Float32
        {
            return Err(Error::ShapeMismatch);
        }
        self.run(flat, &mut trace::Trace { context, retained })?;
        Ok(())
    }
}
