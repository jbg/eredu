use super::*;
use safemlx::{EvaluatedArray, OperationEvent, PreparedArrayClone};
use std::mem::{size_of, size_of_val};
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SummaryPopulation {
    pub(crate) retained_outputs: usize,
    pub(crate) scalar_completions: usize,
    pub(crate) static_indices: usize,
}
impl SummaryPopulation {
    fn output(&mut self) -> Result<(), Error> {
        self.retained_outputs = self
            .retained_outputs
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        Ok(())
    }
    fn read(&mut self) -> Result<(), Error> {
        self.scalar_completions = self
            .scalar_completions
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        Ok(())
    }
}
impl Kernel for SummaryPopulation {
    type Value = ();
    fn interval(&mut self, _: &(), _: i32, _: i32) -> Result<(), Error> {
        self.static_indices = self
            .static_indices
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        self.output()
    }
    fn unary(&mut self, _: Unary, _: &()) -> Result<(), Error> {
        self.output()
    }
    fn scalar(&mut self, _: f32) -> Result<(), Error> {
        self.output()
    }
    fn select(&mut self, _: &(), _: &(), _: &()) -> Result<(), Error> {
        self.output()
    }
    fn binary(&mut self, _: Binary, _: &(), _: &()) -> Result<(), Error> {
        self.output()
    }
    fn read_u32(&mut self, _: &(), representative: u32) -> Result<u32, Error> {
        self.read()?;
        Ok(representative)
    }
    fn read_f32(&mut self, _: &(), kind: ScalarRead) -> Result<f64, Error> {
        self.read()?;
        Ok(kind.representative())
    }
}
impl SummaryProgram {
    pub(crate) fn population(self) -> Result<SummaryPopulation, Error> {
        let mut population = SummaryPopulation::default();
        self.run(&(), &mut population)?;
        Ok(population)
    }
    /// Numerical leaf only; source selection, publication/pin, fixed host claim
    /// and actual Scope/Graph/Record are separately required by its consumer.
    pub(crate) fn control_bytes(self) -> Option<usize> {
        let population = self.population().ok()?;
        let frames = [
            size_of::<Self>(),
            size_of::<SummaryPopulation>(),
            size_of::<native::Native<'static, 'static>>(),
            size_of::<CaptureSummary>(),
            size_of::<Result<CaptureSummary, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<f64, Error>>(),
            size_of::<Result<u32, Error>>(),
            size_of::<(f64, f64, f64, f64, f64, u64, i32, i32)>(),
            size_of::<Unary>(),
            size_of::<Binary>(),
            size_of::<ScalarRead>(),
            size_of::<std::iter::StepBy<std::ops::Range<i32>>>(),
            // Outer chunk locals: interval, cast, finite, three classification
            // masks, two infinity scalars, two masked extrema, min and max.
            size_of::<(
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
            )>(),
            // Nonzero-scale suffix: zero/clean/divisor/scaled/sum/square/sum.
            size_of::<(Array, Array, Array, Array, Array, Array, Array)>(),
            size_of::<(Array, Array)>(), // count helper's mask cast and sum
            size_of::<Result<Array, Error>>(), // native adapter -> retained -> caller
            size_of::<Result<Array, Error>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?
            .checked_add(
                population.retained_outputs.checked_mul(
                    PreparedArrayClone::control_bytes()?
                        .checked_add(Array::inspection_clone_handle_bytes())?,
                )?,
            )?
            .checked_add(
                population
                    .static_indices
                    .checked_mul(safemlx::ops::indexing::inline_basic_index_control_bytes()?)?,
            )?
            .checked_add(
                population.scalar_completions.checked_mul(
                    OperationEvent::nested_completion_control_bytes::<1>()?
                        .checked_add(EvaluatedArray::iteration_control_bytes::<f32>()?)?
                        .checked_add(EvaluatedArray::iteration_control_bytes::<u32>()?)?,
                )?,
            )
    }
}
