//! Bounded checkpoint recipe execution over native operations.

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeError, RecipeMetadata},
    store::{
        CheckpointLease, CheckpointSource, ReadPolicy, StoreError, TensorReadRequest,
        TensorSelection,
    },
};

/// Checks each node against the catalog and the backend's representation limits.
/// The memoizer may reuse results for a fixed validator and immutable catalog.
pub fn preflight<C: RecipeCatalog + ?Sized, E>(
    recipe: &DerivedWeightRecipe,
    catalog: &C,
    validate: &impl Fn(&DerivedWeightRecipe, &RecipeMetadata) -> Result<(), E>,
    metadata_error: &impl Fn(RecipeError) -> E,
    memoize: &impl Fn(&DerivedWeightRecipe, &dyn Fn() -> Result<(), E>) -> Result<(), E>,
) -> Result<(), E> {
    memoize(recipe, &|| {
        let metadata = recipe.infer(catalog).map_err(metadata_error)?;
        validate(recipe, &metadata)?;
        for operand in recipe.operands() {
            preflight(operand, catalog, validate, metadata_error, memoize)?;
        }
        Ok(())
    })
}

/// Native operations on an ordinary recipe's already-admitted data.
pub trait RecipeExecution {
    /// Native tensor produced by a source or operation.
    type Value;
    /// Native guards that hold sources until dependent work completes. Dropping
    /// a guard on failure must be safe even when native work is unsettled.
    type Retained;
    /// Failure with its original checkpoint store error available as a source.
    type Error: std::error::Error + 'static;

    /// Imports exactly the acquired source selection and retains any pending work.
    fn import(
        &self,
        lease: CheckpointLease,
        retained: &mut Vec<Self::Retained>,
    ) -> Result<Self::Value, Self::Error>;

    /// Applies one non-source node to its realized operands in recipe order.
    fn apply(
        &self,
        operation: &DerivedWeightRecipe,
        operands: Vec<Self::Value>,
        retained: &mut Vec<Self::Retained>,
    ) -> Result<Self::Value, Self::Error>;

    /// Translates a bounded-read failure without discarding the store error.
    fn store_error(&self, error: StoreError) -> Self::Error;

    /// Whether this failure permits settling earlier join operands and retrying.
    /// Only source capacity exhaustion in an owned import mode is retryable.
    fn can_retry_join_operand(&self, _error: &Self::Error) -> bool {
        false
    }

    /// Settles this value and releases its guards while keeping the value usable.
    fn detach_join_operand(
        &self,
        _value: &Self::Value,
        _retained: &mut Vec<Self::Retained>,
    ) -> Result<(), Self::Error> {
        unreachable!("recipe executor has not opted in to detaching join operands")
    }
}

/// Walks an ordinary recipe in operand order under bounded source leases.
/// The caller retains the returned work through native completion. Partial
/// work uses the backend guards for cleanup on error.
pub fn execute<N: RecipeExecution>(
    recipe: &DerivedWeightRecipe,
    source: &dyn CheckpointSource,
    native: &N,
    retained: &mut Vec<N::Retained>,
) -> Result<N::Value, N::Error> {
    if let DerivedWeightRecipe::Source { key, selection } = recipe {
        let lease = source
            .acquire_lease(TensorReadRequest {
                key: key.clone(),
                selection: selection.clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .map_err(|error| native.store_error(error))?;
        return native.import(lease, retained);
    }
    let operands = match recipe {
        DerivedWeightRecipe::Concatenate { inputs, .. }
        | DerivedWeightRecipe::Stack { inputs, .. } => {
            execute_join_operands(inputs, source, native, retained)?
        }
        _ => recipe
            .operands()
            .map(|operand| execute(operand, source, native, retained))
            .collect::<Result<_, _>>()?,
    };
    native.apply(recipe, operands, retained)
}

fn execute_join_operands<N: RecipeExecution>(
    inputs: &[DerivedWeightRecipe],
    source: &dyn CheckpointSource,
    native: &N,
    retained: &mut Vec<N::Retained>,
) -> Result<Vec<N::Value>, N::Error> {
    let mut pending = Vec::<(N::Value, Vec<N::Retained>)>::with_capacity(inputs.len());
    let mut detach_remaining = false;
    for input in inputs {
        loop {
            let mut guards = Vec::new();
            match execute(input, source, native, &mut guards) {
                Ok(value) => {
                    if detach_remaining && !guards.is_empty() {
                        native.detach_join_operand(&value, &mut guards)?;
                    }
                    pending.push((value, guards));
                    break;
                }
                Err(error)
                    if !detach_remaining
                        && !pending.is_empty()
                        && native.can_retry_join_operand(&error) =>
                {
                    // The earlier operands can be evaluated independently.
                    // Release their source leases before retrying this operand.
                    drop(guards);
                    for (value, guards) in &mut pending {
                        if !guards.is_empty() {
                            native.detach_join_operand(value, guards)?;
                        }
                    }
                    detach_remaining = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(pending
        .into_iter()
        .map(|(value, guards)| {
            retained.extend(guards);
            value
        })
        .collect())
}

/// Conservative native tensor and host-index buffers of the recipe materializer.
/// A metadata-only bound covers the ordinary path even when binding later selects
/// direct encoded initialization. The upper bound can
/// retain every source until completion; count all intermediates, source copies,
/// and the final contiguous output even when the allocator can recycle them.
/// The bound covers tensor and index payloads.
pub fn conservative_workspace_bytes<C: RecipeCatalog + ?Sized>(
    recipe: &DerivedWeightRecipe,
    source: &C,
) -> Result<u64, String> {
    fn add(left: u64, right: u64) -> Result<u64, String> {
        left.checked_add(right)
            .ok_or_else(|| "native recipe workspace overflow".into())
    }
    fn buffers<C: RecipeCatalog + ?Sized>(
        recipe: &DerivedWeightRecipe,
        source: &C,
    ) -> Result<u64, String> {
        let output = recipe
            .infer(source)
            .map_err(|error| error.to_string())?
            .byte_len();
        match recipe {
            DerivedWeightRecipe::DecodeTensorFp8 { weight, scale }
            | DerivedWeightRecipe::ScaleF32 { weight, scale } => add(
                add(buffers(weight, source)?, buffers(scale, source)?)?,
                output
                    .checked_mul(2)
                    .ok_or("scaled tensor workspace overflow")?,
            ),
            DerivedWeightRecipe::Source { .. } => add(output, output),
            DerivedWeightRecipe::Stack { inputs, .. }
            | DerivedWeightRecipe::Concatenate { inputs, .. } => inputs
                .iter()
                .try_fold(output, |bytes, input| add(bytes, buffers(input, source)?)),
            DerivedWeightRecipe::Select { input, selection } => {
                let indices = match selection {
                    TensorSelection::Full => 0,
                    TensorSelection::Range { start, end, .. } => end - start,
                    TensorSelection::Indices { indices, .. } => indices.len(),
                    TensorSelection::Contiguous { shape, .. } => shape
                        .iter()
                        .try_fold(1usize, |n, d| n.checked_mul(*d))
                        .ok_or("native index count overflow")?,
                };
                let index_bytes = (indices as u64)
                    .checked_mul(8)
                    .ok_or("native index buffers overflow")?;
                add(add(buffers(input, source)?, output)?, index_bytes)
            }
            DerivedWeightRecipe::NegLog { input } => {
                // Negation output plus the sign-validation boolean tensor.
                let metadata = input.infer(source).map_err(|error| error.to_string())?;
                let elements = metadata
                    .shape()
                    .iter()
                    .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
                    .ok_or("native validation size overflow")?;
                add(
                    add(
                        buffers(input, source)?,
                        output
                            .checked_mul(2)
                            .ok_or("native neg-log size overflow")?,
                    )?,
                    elements,
                )
            }
            DerivedWeightRecipe::Reshape { input, .. }
            | DerivedWeightRecipe::Transpose { input, .. }
            | DerivedWeightRecipe::Cast { input, .. }
            | DerivedWeightRecipe::View { input, .. }
            | DerivedWeightRecipe::SubtractOne { input } => add(buffers(input, source)?, output),
        }
    }
    add(
        buffers(recipe, source)?,
        recipe
            .infer(source)
            .map_err(|error| error.to_string())?
            .byte_len(),
    )
}
