use super::*;
use safemlx::{EvaluatedArray, OperationEvent, PreparedArrayClone};
use std::mem::{size_of, size_of_val};
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HistogramPopulation {
    pub(crate) retained_outputs: usize,
    pub(crate) scalar_completions: usize,
    pub(crate) static_indices: usize,
}
impl HistogramPopulation {
    fn output(&mut self) -> Result<(), Error> {
        self.retained_outputs = self
            .retained_outputs
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        Ok(())
    }
}
impl Kernel for HistogramPopulation {
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
    fn binary(&mut self, _: Binary, _: &(), _: &()) -> Result<(), Error> {
        self.output()
    }
    fn read(&mut self, _: &()) -> Result<u32, Error> {
        self.scalar_completions = self
            .scalar_completions
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        Ok(0)
    }
}
impl HistogramProgram<'_> {
    pub(crate) fn population(self) -> Result<HistogramPopulation, Error> {
        let mut population = HistogramPopulation::default();
        self.run(&(), &mut population, &mut |_, _| Ok(()))?;
        Ok(population)
    }
    pub(crate) fn control_bytes(self) -> Option<usize> {
        let population = self.population().ok()?;
        let frames = [
            size_of::<Self>(),
            size_of::<HistogramPopulation>(),
            size_of::<native::Native<'static, 'static>>(),
            size_of::<HistogramTotals>(),
            size_of::<Result<HistogramTotals, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<u32, Error>>(),
            size_of::<(i32, i32, usize, &[f32])>(),
            size_of::<Unary>(),
            size_of::<Binary>(),
            size_of::<std::iter::StepBy<std::ops::Range<i32>>>(),
            size_of::<std::iter::Enumerate<std::slice::Windows<'static, f32>>>(),
            // Interval, cast, finite, nonfinite, two edge/compare/mask triples stay live
            // in the outer chunk. The inner bin has five values; count holds two more.
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
            )>(),
            size_of::<(Array, Array, Array, Array, Array)>(),
            size_of::<(Array, Array)>(),
            size_of::<Result<Array, Error>>(),
            size_of::<Result<Array, Error>>(),
            size_of::<&mut dyn FnMut(usize, u64) -> Result<(), Error>>(),
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
                        .checked_add(EvaluatedArray::iteration_control_bytes::<u32>()?)?,
                )?,
            )
    }
}
