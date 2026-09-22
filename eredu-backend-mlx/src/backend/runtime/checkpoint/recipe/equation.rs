//! The recipe equation shared by native materialization and descriptive tracing.
use super::*;

/// Primitive realization and source custody for one recipe traversal. Descriptive
/// implementations record the same operations without acquiring payloads or
/// receiving execution authority. Native implementations retain the actual source
/// leases and settle validation before permitting their dependent operations.
pub(super) trait RecipeOperations: Sized {
    type Value;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, WeightRecipeError>;
    fn source(
        &mut self,
        key: &str,
        selection: &TensorSelection,
    ) -> Result<Self::Value, WeightRecipeError>;
    fn inputs(
        &mut self,
        recipes: &[DerivedWeightRecipe],
    ) -> Result<Vec<Self::Value>, WeightRecipeError>;
    fn indices(&mut self, values: &[i32]) -> Result<Self::Value, WeightRecipeError>;
    fn scalar(&mut self, value: f32) -> Result<Self::Value, WeightRecipeError>;
    fn take(
        &mut self,
        value: Self::Value,
        indices: Self::Value,
        axis: i32,
    ) -> Result<Self::Value, WeightRecipeError>;
    fn join(
        &mut self,
        values: Vec<Self::Value>,
        axis: i32,
        stack: bool,
    ) -> Result<Self::Value, WeightRecipeError>;
    fn reshape(
        &mut self,
        value: Self::Value,
        shape: &[i32],
    ) -> Result<Self::Value, WeightRecipeError>;
    fn transpose(
        &mut self,
        value: Self::Value,
        axes: &[i32],
    ) -> Result<Self::Value, WeightRecipeError>;
    fn cast(&mut self, value: Self::Value, dtype: Dtype) -> Result<Self::Value, WeightRecipeError>;
    fn view(&mut self, value: Self::Value, dtype: Dtype) -> Result<Self::Value, WeightRecipeError>;
    fn dtype(&self, value: &Self::Value) -> Result<Dtype, WeightRecipeError>;
    fn less(
        &mut self,
        left: &Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, WeightRecipeError>;
    fn all(&mut self, value: Self::Value) -> Result<Self::Value, WeightRecipeError>;
    fn require_true(&mut self, value: Self::Value) -> Result<(), WeightRecipeError>;
    fn multiply(
        &mut self,
        left: &Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, WeightRecipeError>;
    fn subtract(
        &mut self,
        left: Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, WeightRecipeError>;
    fn log(&mut self, value: Self::Value) -> Result<Self::Value, WeightRecipeError>;
    fn with_validation(
        &mut self,
        value: Self::Value,
        body: impl FnOnce(&mut Self, &Self::Value) -> Result<Self::Value, WeightRecipeError>,
    ) -> Result<Self::Value, WeightRecipeError>;
    fn contiguous(&mut self, value: Self::Value) -> Result<Self::Value, WeightRecipeError>;
}

pub(super) fn execute<W: RecipeOperations>(
    recipe: &DerivedWeightRecipe,
    worker: &mut W,
) -> Result<W::Value, WeightRecipeError> {
    match recipe {
        DerivedWeightRecipe::Source { key, selection } => worker.source(key, selection),
        DerivedWeightRecipe::Select { input, selection } => {
            let array = execute(input, worker)?;
            match selection {
                TensorSelection::Full => Ok(array),
                TensorSelection::Range { axis, start, end } => {
                    let mut indices =
                        worker
                            .vector(end.checked_sub(*start).ok_or(
                                WeightRecipeError::ArithmeticOverflow("selection range"),
                            )?)?;
                    for index in *start..*end {
                        indices.push(usize_to_i32(index, "selection index")?);
                    }
                    let indices = worker.indices(&indices)?;
                    worker.take(array, indices, usize_to_i32(*axis, "selection axis")?)
                }
                TensorSelection::Indices { axis, indices } => {
                    let mut converted = worker.vector(indices.len())?;
                    for index in indices {
                        converted.push(usize_to_i32(*index, "selection index")?);
                    }
                    let indices = worker.indices(&converted)?;
                    worker.take(array, indices, usize_to_i32(*axis, "selection axis")?)
                }
                TensorSelection::Contiguous {
                    offset_elements,
                    shape,
                } => {
                    let elements = shape.iter().try_fold(1usize, |count, dimension| {
                        count
                            .checked_mul(*dimension)
                            .ok_or(WeightRecipeError::ArithmeticOverflow(
                                "contiguous recipe selection size",
                            ))
                    })?;
                    let end = offset_elements.checked_add(elements).ok_or(
                        WeightRecipeError::ArithmeticOverflow("contiguous recipe selection end"),
                    )?;
                    let mut indices = worker.vector(elements)?;
                    for index in *offset_elements..end {
                        indices.push(usize_to_i32(index, "contiguous selection index")?);
                    }
                    let flattened = worker.reshape(array, &[-1])?;
                    let indices = worker.indices(&indices)?;
                    let selected = worker.take(flattened, indices, 0)?;
                    let shape = dimensions(shape, "contiguous selection shape", worker)?;
                    worker.reshape(selected, &shape)
                }
            }
        }
        DerivedWeightRecipe::Concatenate { axis, inputs }
        | DerivedWeightRecipe::Stack { axis, inputs } => {
            let arrays = worker.inputs(inputs)?;
            let stack = matches!(recipe, DerivedWeightRecipe::Stack { .. });
            worker.join(arrays, usize_to_i32(*axis, "recipe join axis")?, stack)
        }
        DerivedWeightRecipe::Reshape { input, shape } => {
            let array = execute(input, worker)?;
            let shape = dimensions(shape, "reshape dimension", worker)?;
            worker.reshape(array, &shape)
        }
        DerivedWeightRecipe::Transpose { input, axes } => {
            let array = execute(input, worker)?;
            let axes = dimensions(axes, "transpose axis", worker)?;
            worker.transpose(array, &axes)
        }
        DerivedWeightRecipe::Cast { input, dtype } => {
            let array = execute(input, worker)?;
            worker.cast(array, mlx_dtype(dtype)?)
        }
        DerivedWeightRecipe::View {
            input,
            dtype,
            shape,
        } => {
            let array = execute(input, worker)?;
            let viewed = worker.view(array, mlx_dtype(dtype)?)?;
            let shape = dimensions(shape, "view dimension", worker)?;
            worker.reshape(viewed, &shape)
        }
        DerivedWeightRecipe::NegLog { input } => {
            let array = execute(input, worker)?;
            worker.with_validation(array, |worker, array| {
                let zero = worker.scalar(0.0)?;
                let negative = worker.less(array, zero)?;
                let all_negative = worker.all(negative)?;
                worker.require_true(all_negative)?;
                let minus_one = worker.scalar(-1.0)?;
                let positive = worker.multiply(array, minus_one)?;
                worker.log(positive)
            })
        }
        DerivedWeightRecipe::SubtractOne { input } => {
            let array = execute(input, worker)?;
            let one = worker.scalar(1.0)?;
            let one = worker.cast(one, worker.dtype(&array)?)?;
            worker.subtract(array, one)
        }
    }
}

pub(super) fn materialize<W: RecipeOperations>(
    recipe: &DerivedWeightRecipe,
    worker: &mut W,
) -> Result<W::Value, WeightRecipeError> {
    let value = execute(recipe, worker)?;
    worker.contiguous(value)
}

fn dimensions<W: RecipeOperations>(
    values: &[usize],
    name: &'static str,
    worker: &mut W,
) -> Result<Vec<i32>, WeightRecipeError> {
    let mut converted = worker.vector(values.len())?;
    for value in values {
        converted.push(usize_to_i32(*value, name)?);
    }
    Ok(converted)
}
