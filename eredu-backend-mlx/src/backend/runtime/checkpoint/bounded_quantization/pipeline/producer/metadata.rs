//! Admitted finite recipe inference and native shape storage for one tile.
use eredu_checkpoint::{
    recipe::{
        DerivedWeightRecipe, RecipeInferenceError, RecipeInferenceInput, RecipeInferencePlan,
        RecipeMetadata,
    },
    store::CheckpointSource,
};
use eredu_runtime::working_memory::{
    InitializedSharedNative, MemoryLedger, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
};

/// The source and recipe remain borrowed until their actual finite inference.
/// Encoded-read range construction, native work and runtime birth are separate.
pub(super) struct MetadataPlan<'a>(RecipeInferencePlan<'a, dyn CheckpointSource + 'a>);
impl<'a> MetadataPlan<'a> {
    pub(super) fn new(
        recipe: &'a DerivedWeightRecipe,
        source: &'a dyn CheckpointSource,
    ) -> Result<Self, WorkingMemoryError> {
        RecipeInferencePlan::new(RecipeInferenceInput::Derived(recipe), source)
            .map(Self)
            .ok_or(WorkingMemoryError::UnknownBound)
    }
    pub(super) fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        MemoryLedger::shared_native_initialization_required_bytes(self)
    }
    pub(super) fn prepare(
        self,
        pool: &MemoryLedger,
    ) -> Result<InitializedSharedNative<Metadata>, SharedNativeInitializationError<Self>> {
        pool.initialize_shared_native(self)
    }
}

#[derive(Debug)]
pub(super) struct Metadata {
    inferred: RecipeMetadata,
    shape: Vec<i32>,
    _custody: SharedNativeInitializationCustody,
}
impl Metadata {
    pub(super) fn inferred(&self) -> &RecipeMetadata {
        &self.inferred
    }
    pub(super) fn shape(&self) -> &[i32] {
        &self.shape
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error("tile recipe inference: {0}")]
    Inference(#[from] RecipeInferenceError),
    #[error("tile native shape reserve failed")]
    Reserve(#[source] TryReserveError),
    #[error("tile dimension {dimension} at axis {axis} is outside native i32 geometry")]
    Dimension { axis: usize, dimension: usize },
}

#[derive(Debug, thiserror::Error)]
#[error("tile metadata: {cause}")]
pub(super) struct ConstructionError {
    #[source]
    cause: Cause,
    _metadata: Option<RecipeMetadata>,
    _shape: Vec<i32>,
    _custody: SharedNativeInitializationCustody,
}
impl SharedNativeInitializer for MetadataPlan<'_> {
    type Output = Metadata;
    type Error = ConstructionError;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let layout = self.0.layout();
        let controls = [
            layout.required_bytes(),
            safemlx::EvaluatedArray::completed_readback_control_bytes::<safemlx::complex64>()
                .ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<Result<(), safemlx::error::NativeBytesCopyError>>(),
            Layout::array::<i32>(layout.shape_capacity())
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            size_of::<Metadata>(),
            size_of::<ConstructionError>(),
            size_of::<Cause>(),
            size_of::<Option<RecipeMetadata>>(),
            size_of::<Vec<i32>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, usize>>>(),
            size_of::<Result<i32, std::num::TryFromIntError>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Metadata, ConstructionError> {
        let capacity = self.0.layout().shape_capacity();
        let mut metadata = None;
        let mut shape = Vec::new();
        let result = (|| {
            metadata = Some(self.0.infer()?);
            shape.try_reserve_exact(capacity).map_err(Cause::Reserve)?;
            for (axis, &dimension) in metadata
                .as_ref()
                .expect("inferred metadata")
                .shape()
                .iter()
                .enumerate()
            {
                shape.push(
                    i32::try_from(dimension).map_err(|_| Cause::Dimension { axis, dimension })?,
                );
            }
            Ok::<_, Cause>(())
        })();
        match result {
            Ok(()) => Ok(Metadata {
                inferred: metadata.expect("inferred metadata"),
                shape,
                _custody: custody,
            }),
            Err(cause) => Err(ConstructionError {
                cause,
                _metadata: metadata,
                _shape: shape,
                _custody: custody,
            }),
        }
    }
}

#[cfg(test)]
mod tests;
