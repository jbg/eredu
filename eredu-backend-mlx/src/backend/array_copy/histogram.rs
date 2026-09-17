//! Fixed-edge histogram equation shared by actual native work and cold tracing.
use super::capture_tensor::{CaptureCompletion, CaptureTensorNativeError as Error};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTensor};
use safemlx::{Array, Stream};
mod native;
mod population;
mod trace;
pub(crate) use population::HistogramPopulation;
const CHUNK: i32 = 1024;
#[derive(Clone, Copy, Debug)]
pub(crate) struct HistogramProgram<'a> {
    elements: i32,
    edges: &'a [f32],
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HistogramTotals {
    pub(crate) below: u64,
    pub(crate) above: u64,
    pub(crate) non_finite: u64,
}
#[derive(Clone, Copy)]
enum Unary {
    CastF32,
    Finite,
    Not,
    MaskU32,
    Sum,
}
#[derive(Clone, Copy)]
enum Binary {
    Less,
    Greater,
    GreaterEqual,
    LessEqual,
    And,
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
    fn binary(
        &mut self,
        kind: Binary,
        left: &Self::Value,
        right: &Self::Value,
    ) -> Result<Self::Value, Error>;
    fn read(&mut self, value: &Self::Value) -> Result<u32, Error>;
}
impl<'a> HistogramProgram<'a> {
    pub(crate) fn new(elements: i32, edges: &'a [f32]) -> Result<Self, Error> {
        if elements < 0
            || edges.len() < 2
            || edges.iter().any(|e| !e.is_finite())
            || edges.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { elements, edges })
    }
    pub(crate) fn bins(self) -> usize {
        self.edges.len() - 1
    }
    fn count<K: Kernel>(kernel: &mut K, mask: &K::Value) -> Result<u64, Error> {
        let mask = kernel.unary(Unary::MaskU32, mask)?;
        let sum = kernel.unary(Unary::Sum, &mask)?;
        Ok(u64::from(kernel.read(&sum)?))
    }
    // No data-dependent branches: bin geometry alone chooses the inclusive last
    // upper edge. NaN/Inf are counted separately and cannot satisfy finite edges.
    fn run<K: Kernel>(
        self,
        flat: &K::Value,
        kernel: &mut K,
        record: &mut dyn FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<HistogramTotals, Error> {
        let mut totals = HistogramTotals::default();
        for start in (0..self.elements).step_by(CHUNK as usize) {
            let end = start.saturating_add(CHUNK).min(self.elements);
            let selected = kernel.interval(flat, start, end)?;
            let chunk = kernel.unary(Unary::CastF32, &selected)?;
            let finite = kernel.unary(Unary::Finite, &chunk)?;
            let nonfinite = kernel.unary(Unary::Not, &finite)?;
            totals.non_finite += Self::count(kernel, &nonfinite)?;
            let edge = kernel.scalar(self.edges[0])?;
            let below = kernel.binary(Binary::Less, &chunk, &edge)?;
            let below = kernel.binary(Binary::And, &below, &finite)?;
            totals.below += Self::count(kernel, &below)?;
            let edge = kernel.scalar(self.edges[self.bins()])?;
            let above = kernel.binary(Binary::Greater, &chunk, &edge)?;
            let above = kernel.binary(Binary::And, &above, &finite)?;
            totals.above += Self::count(kernel, &above)?;
            for (index, bounds) in self.edges.windows(2).enumerate() {
                let edge = kernel.scalar(bounds[0])?;
                let lower = kernel.binary(Binary::GreaterEqual, &chunk, &edge)?;
                let edge = kernel.scalar(bounds[1])?;
                let upper = kernel.binary(
                    if index + 1 == self.bins() {
                        Binary::LessEqual
                    } else {
                        Binary::Less
                    },
                    &chunk,
                    &edge,
                )?;
                let inside = kernel.binary(Binary::And, &lower, &upper)?;
                record(index, Self::count(kernel, &inside)?)?;
            }
        }
        Ok(totals)
    }
    pub(crate) fn execute(
        self,
        flat: &Array,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        retain: &mut dyn FnMut(&Array) -> Result<(), Error>,
        host_read: &mut dyn FnMut(usize),
        record: &mut dyn FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<HistogramTotals, Error> {
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
            record,
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
        self.run(
            flat,
            &mut trace::Trace { context, retained },
            &mut |_, _| Ok(()),
        )?;
        Ok(())
    }
}
