//! Fixed control/root population executes the same scalar loop without tensors.
use super::*;
use safemlx::{EvaluatedArray, OperationEvent, PreparedArrayClone};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TokenScorePopulation {
    pub(crate) retained_outputs: usize,
    pub(crate) scalar_completions: usize,
    pub(crate) exclusion_aliases: usize,
    pub(crate) static_indices: usize,
}
impl TokenScorePopulation {
    fn output(&mut self) -> Result<(), Error> {
        self.retained_outputs = self
            .retained_outputs
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        Ok(())
    }
    fn index(&mut self) -> Result<(), Error> {
        self.static_indices = self
            .static_indices
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        self.output()
    }
    fn read(&mut self) -> Result<(), Error> {
        self.scalar_completions = self
            .scalar_completions
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        Ok(())
    }
}
impl Kernel for TokenScorePopulation {
    type Value = ();
    fn flatten(&mut self, _: &(), _: i32) -> Result<(), Error> {
        self.output()
    }
    fn terminal(&mut self, _: &()) -> Result<(), Error> {
        self.index()
    }
    fn unary(&mut self, _: Unary, _: &()) -> Result<(), Error> {
        self.output()
    }
    fn scalar(&mut self, _: f32) -> Result<(), Error> {
        self.output()
    }
    fn interval(&mut self, _: &(), _: i32, _: i32) -> Result<(), Error> {
        self.index()
    }
    fn at(&mut self, _: &(), _: u32) -> Result<(), Error> {
        self.index()
    }
    fn binary(&mut self, _: Binary, _: &(), _: &()) -> Result<(), Error> {
        self.output()
    }
    fn exclude(&mut self, _: &(), _: u32, _: &()) -> Result<(), Error> {
        self.exclusion_aliases = self
            .exclusion_aliases
            .checked_add(1)
            .ok_or(Error::GeometryOverflow)?;
        // The scalar-[1] reshape is exposed by the trace and enclosed by the
        // actual indexed-update worker; retaining it is conservative natively.
        self.output()?;
        self.output()
    }
    fn read_f32(&mut self, _: &()) -> Result<f64, Error> {
        self.read()?;
        Ok(0.0)
    }
    fn read_u32(&mut self, _: &(), kind: Read) -> Result<u32, Error> {
        self.read()?;
        Ok(kind.representative())
    }
}
impl TokenScoreProgram<'_> {
    pub(crate) fn population(self) -> Result<TokenScorePopulation, Error> {
        let mut result = TokenScorePopulation::default();
        self.run(&(), &mut result, |_| true, |_| Ok(()))?;
        Ok(result)
    }
    /// Only the closed numerical worker's extra fixed controls. Source
    /// publication, claim/destination and enclosing role/Graph/Record banks are
    /// independently mandatory. Querying this grants none of those resources.
    pub(crate) fn control_bytes(self) -> Option<usize> {
        let population = self.population().ok()?;
        let clones = population
            .retained_outputs
            .checked_add(population.exclusion_aliases)?;
        let clone = PreparedArrayClone::control_bytes()?
            .checked_add(Array::inspection_clone_handle_bytes())?;
        let frames = [
            size_of::<Self>(),
            size_of::<TokenScorePopulation>(),
            size_of::<native::Native<'static, 'static>>(),
            size_of::<CaptureTokenScore>(),
            size_of::<Option<CaptureCandidate>>(),
            size_of::<Result<f64, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<(f64, f64, f64, f64, f64, f64)>(),
            size_of::<(i32, i32, u32, u64)>(),
            size_of::<Unary>(),
            size_of::<Binary>(),
            size_of::<Read>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?
            .checked_add(clones.checked_mul(clone)?)?
            .checked_add(
                population
                    .static_indices
                    .checked_mul(safemlx::ops::indexing::inline_basic_index_control_bytes()?)?,
            )?
            .checked_add(population.exclusion_aliases.checked_mul(
                safemlx::ops::indexing::inline_scalar_index_update_control_bytes()?,
            )?)?
            .checked_add(
                population.scalar_completions.checked_mul(
                    OperationEvent::nested_completion_control_bytes::<1>()?
                        .checked_add(EvaluatedArray::iteration_control_bytes::<f32>()?)?
                        .checked_add(EvaluatedArray::iteration_control_bytes::<u32>()?)?,
                )?,
            )
    }
}
