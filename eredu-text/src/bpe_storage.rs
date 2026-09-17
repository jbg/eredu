//! Packed model components; no tokenizer construction or accounting authority.
use std::{collections::TryReserveError, fmt, mem::size_of};
use tokenizers::models::bpe::{
    BpeCompileError, BpeCompileFailure, BpeCompilePlan, BpeCompileRequirements, BPE,
};

/// Fixed source/profile error classification from the concrete packed compiler.
pub use tokenizers::models::bpe::BpeCompileErrorKind as BpeModelSourceErrorKind;

/// Fixed source error; no owned input, partial buffer or retry capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BpeModelSourceError(BpeCompileError);
impl BpeModelSourceError {
    /// Exact parser/profile/semantic classification.
    pub fn kind(&self) -> BpeModelSourceErrorKind {
        self.0.kind
    }
    /// Byte offset in the original borrowed model object.
    pub fn offset(&self) -> usize {
        self.0.offset
    }
    fn overflow() -> Self {
        Self(BpeCompileError {
            kind: BpeModelSourceErrorKind::Overflow,
            offset: 0,
        })
    }
}
impl fmt::Display for BpeModelSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for BpeModelSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Source-derived payload and named adapter controls; this grants no budget.
#[derive(Debug, Clone, Copy)]
pub struct BpeModelRequirements {
    compiler: BpeCompileRequirements,
    controls: usize,
    total: usize,
}
impl BpeModelRequirements {
    /// Four Vec and up to three String payload capacities, including discarded slots.
    pub fn buffer_bytes(&self) -> usize {
        self.compiler.buffer_bytes()
    }
    /// Concrete HF and text adapter representations and return overlaps.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked payload plus all named construction controls.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}

/// Borrows exact immutable model JSON and permits one packed construction attempt.
///
/// ```compile_fail
/// # use eredu_text::bpe_storage::BpeModelPlan;
/// fn twice(plan:BpeModelPlan<'_>) { let _ = plan.compile(); let _ = plan.compile(); }
/// ```
#[derive(Debug)]
pub struct BpeModelPlan<'a> {
    compiler: BpeCompilePlan<'a>,
    requirements: BpeModelRequirements,
}
impl<'a> BpeModelPlan<'a> {
    /// Checks a BPE model object without constructing a tokenizer or allocating.
    pub fn prepare_model_json(input: &'a [u8]) -> Result<Self, BpeModelSourceError> {
        let compiler = BpeCompilePlan::prepare_model_json(input).map_err(BpeModelSourceError)?;
        let facts = compiler.requirements();
        let adapters = [
            size_of::<Self>(),
            size_of::<Result<Self, BpeModelSourceError>>(),
            size_of::<BpeModelRequirements>(),
            size_of::<PreparedBpeModel>(),
            size_of::<BpeModelCompileFailure>(),
            size_of::<BpeModelSourceError>(),
            size_of::<Result<PreparedBpeModel, BpeModelCompileFailure>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(BpeModelSourceError::overflow)?;
        let controls = facts
            .control_bytes()
            .checked_add(adapters)
            .ok_or_else(BpeModelSourceError::overflow)?;
        let total = facts
            .required_bytes()
            .checked_add(adapters)
            .ok_or_else(BpeModelSourceError::overflow)?;
        Ok(Self {
            compiler,
            requirements: BpeModelRequirements {
                compiler: facts,
                controls,
                total,
            },
        })
    }
    /// Exact immutable requirements, without taking the model input or a grant.
    pub fn requirements(&self) -> BpeModelRequirements {
        self.requirements
    }
    /// Moves the existing compiler's same model/partial failure into concrete wrappers.
    pub fn compile(self) -> Result<PreparedBpeModel, BpeModelCompileFailure> {
        match self.compiler.compile() {
            Ok(model) => Ok(PreparedBpeModel(model)),
            Err(error) => Err(BpeModelCompileFailure(error)),
        }
    }
    /// Development-only real capacity overflow before a named compiler reserve.
    #[cfg(feature = "packed-bpe-test-support")]
    #[doc(hidden)]
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        self.compiler = self.compiler.fail_reservation(stage);
        self
    }
}

/// A concrete packed model with scalar readonly access and no raw HF escape.
///
/// The component itself grants no accounting or tokenization permission.
///
/// ```compile_fail
/// # use eredu_text::bpe_storage::PreparedBpeModel;
/// fn copy(model:PreparedBpeModel) { let _ = model.clone(); }
/// ```
/// ```compile_fail
/// # use eredu_text::bpe_storage::PreparedBpeModel;
/// fn mutate(mut model:PreparedBpeModel) { model.dropout=Some(0.5); }
/// ```
pub struct PreparedBpeModel(BPE);
impl PreparedBpeModel {
    /// Canonical forward vocabulary size.
    pub fn token_count(&self) -> usize {
        self.0.vocabulary().len()
    }
    /// Looks up a spelling without copying or mutating the model.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.0.vocabulary().token_id(token)
    }
    /// Borrows the canonical spelling without allocating a String.
    pub fn spelling(&self, id: u32) -> Option<&str> {
        self.0.vocabulary().spelling(id)
    }
    /// Borrows forward IDs directly; no dense sparse-ID extent traversal.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.vocabulary().ids()
    }
}
impl fmt::Debug for PreparedBpeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedBpeModel")
            .field("token_count", &self.token_count())
            .finish()
    }
}

/// Actual one-attempt failure retaining all partial vectors and strings.
/// No Clone, retry, raw partial accessor or consuming extraction is provided.
#[derive(Debug)]
pub struct BpeModelCompileFailure(BpeCompileFailure);
impl BpeModelCompileFailure {
    /// Copies only a fixed source cause, retaining every partial buffer here.
    pub fn source_error(&self) -> Option<BpeModelSourceError> {
        self.0.source_error().copied().map(BpeModelSourceError)
    }
    /// Borrows the real allocator/capacity-overflow failure.
    pub fn allocation_error(&self) -> Option<&TryReserveError> {
        self.0.allocation_error()
    }
    /// Actual four vector and three optional string capacities.
    pub fn buffer_capacities(&self) -> [usize; 7] {
        self.0.buffer_capacities()
    }
    /// Actual retained payload bytes, excluding allocator metadata.
    pub fn allocated_bytes(&self) -> Option<usize> {
        self.0.allocated_bytes()
    }
}
impl fmt::Display for BpeModelCompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for BpeModelCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[cfg(test)]
mod tests;
