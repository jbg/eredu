//! Native primitives and completion custody for the shared recipe equation.
use super::equation::RecipeOperations;
use super::*;

pub(super) struct NativeRecipe<'s, 'q, 'p, 'v, 'c, 'o, 'b, 'g> {
    pub(super) source: NativeRecipeSource<'s>,
    pub(super) stream: &'q Stream,
    pub(super) sources: &'p mut Vec<PendingWeightMaterialization>,
    pub(super) borrow_sources: bool,
    pub(super) context: &'v crate::backend::runtime::checkpoint::store::MaterializationView<'c>,
    pub(super) original: Option<(
        &'o mut OriginalMaterializationSlots<'b>,
        &'g OriginalScopeObserver,
    )>,
}

pub(crate) type RecipeLeafProducer<'a> =
    dyn FnMut(&str, &TensorSelection, &Stream) -> Result<Array, WeightRecipeError> + 'a;

pub(super) enum NativeRecipeSource<'a> {
    Checkpoint(&'a dyn CheckpointSource),
    Prepared {
        leaves: &'a mut RecipeLeafProducer<'a>,
        host: Option<&'a eredu_core::HostPreparationAuthority>,
    },
}

impl RecipeOperations for NativeRecipe<'_, '_, '_, '_, '_, '_, '_, '_> {
    type Value = Array;

    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, WeightRecipeError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(WeightRecipeError::MetadataAllocation)?;
        Ok(values)
    }

    fn source(
        &mut self,
        key: &str,
        selection: &TensorSelection,
    ) -> Result<Array, WeightRecipeError> {
        let store = match &mut self.source {
            NativeRecipeSource::Prepared { leaves, .. } => {
                return leaves(key, selection, self.stream);
            }
            NativeRecipeSource::Checkpoint(store) => *store,
        };
        let pending = match self.original.as_mut() {
            Some((slots, observer)) => slots
                .acquire_lease(store, key, selection, observer)?
                .prepare(
                    self.context,
                    self.stream,
                    self.stream,
                    slots,
                    observer,
                    self.borrow_sources,
                )?,
            None => {
                let lease =
                    self.context
                        .weight_lease(store.acquire_lease(TensorReadRequest {
                            key: key.to_owned(),
                            selection: selection.clone(),
                            policy: ReadPolicy::RequireBounded,
                        })?)?;
                if self.borrow_sources {
                    lease.prepare_borrowed_materialization(self.stream)?
                } else {
                    lease.prepare_materialization(self.stream, self.stream)?
                }
            }
        };
        let array = pending.output().clone();
        self.sources.push(pending);
        Ok(array)
    }

    fn inputs(&mut self, recipes: &[DerivedWeightRecipe]) -> Result<Vec<Array>, WeightRecipeError> {
        let store = match &self.source {
            NativeRecipeSource::Checkpoint(store) => *store,
            NativeRecipeSource::Prepared { .. } => {
                // Prepared leaves retain their admitted owners in the caller's
                // recovery payload and need no checkpoint-cache retry. Both
                // source mechanisms execute this same recursive equation.
                let mut values = self.vector(recipes.len())?;
                for recipe in recipes {
                    values.push(equation::execute(recipe, self)?);
                }
                return Ok(values);
            }
        };
        materialize_inputs(
            recipes,
            store,
            self.stream,
            self.sources,
            self.borrow_sources,
            self.context,
            self.original
                .as_mut()
                .map(|(slots, observer)| (&mut **slots, *observer)),
        )
    }

    fn indices(&mut self, values: &[i32]) -> Result<Array, WeightRecipeError> {
        Ok(Array::try_from_slice(
            values,
            &[usize_to_i32(values.len(), "selection index count")?],
        )?)
    }
    fn scalar(&mut self, value: f32) -> Result<Array, WeightRecipeError> {
        Ok(Array::try_from_f32(value)?)
    }
    fn take(
        &mut self,
        value: Array,
        indices: Array,
        axis: i32,
    ) -> Result<Array, WeightRecipeError> {
        Ok(value.take_axis(indices, axis, self.stream)?)
    }
    fn join(
        &mut self,
        values: Vec<Array>,
        axis: i32,
        stack: bool,
    ) -> Result<Array, WeightRecipeError> {
        let references = values.iter().collect::<Vec<_>>();
        Ok(if stack {
            stack_axis(&references, axis, self.stream)?
        } else {
            concatenate_axis(&references, axis, self.stream)?
        })
    }
    fn reshape(&mut self, value: Array, shape: &[i32]) -> Result<Array, WeightRecipeError> {
        Ok(value.reshape(shape, self.stream)?)
    }
    fn transpose(&mut self, value: Array, axes: &[i32]) -> Result<Array, WeightRecipeError> {
        Ok(value.transpose_axes(axes, self.stream)?)
    }
    fn cast(&mut self, value: Array, dtype: Dtype) -> Result<Array, WeightRecipeError> {
        Ok(value.as_dtype(dtype, self.stream)?)
    }
    fn view(&mut self, value: Array, dtype: Dtype) -> Result<Array, WeightRecipeError> {
        Ok(value.view_dtype(dtype, self.stream)?)
    }
    fn dtype(&self, value: &Array) -> Result<Dtype, WeightRecipeError> {
        Ok(value.dtype())
    }
    fn less(&mut self, left: &Array, right: Array) -> Result<Array, WeightRecipeError> {
        Ok(left.lt(right, self.stream)?)
    }
    fn all(&mut self, value: Array) -> Result<Array, WeightRecipeError> {
        Ok(value.all(false, self.stream)?)
    }
    fn require_true(&mut self, value: Array) -> Result<(), WeightRecipeError> {
        if value.try_item::<bool>(self.stream)? {
            Ok(())
        } else {
            Err(WeightRecipeError::NonNegativeNegLogInput)
        }
    }
    fn multiply(&mut self, left: &Array, right: Array) -> Result<Array, WeightRecipeError> {
        Ok(left.multiply(right, self.stream)?)
    }
    fn subtract(&mut self, left: Array, right: Array) -> Result<Array, WeightRecipeError> {
        Ok(left.subtract(right, self.stream)?)
    }
    fn log(&mut self, value: Array) -> Result<Array, WeightRecipeError> {
        Ok(value.log(self.stream)?)
    }
    fn with_validation(
        &mut self,
        value: Array,
        body: impl FnOnce(&mut Self, &Array) -> Result<Array, WeightRecipeError>,
    ) -> Result<Array, WeightRecipeError> {
        let prepared = match self.original.as_mut() {
            Some((slots, observer)) => WeightMaterialization::prepare_retained_with_operations(
                vec![value],
                std::mem::take(self.sources),
                slots,
                observer,
            )?,
            None => match &self.source {
                NativeRecipeSource::Prepared {
                    host: Some(host), ..
                } => WeightMaterialization::prepare_retained_with_host(
                    vec![value],
                    std::mem::take(self.sources),
                    (*host).clone(),
                )?,
                _ => WeightMaterialization::prepare_retained(
                    vec![value],
                    std::mem::take(self.sources),
                )?,
            },
        };
        let output = match body(self, &prepared.inputs()[0]) {
            Ok(output) => output,
            Err(WeightRecipeError::NonNegativeNegLogInput) => {
                prepared.finish()?;
                return Err(WeightRecipeError::NonNegativeNegLogInput);
            }
            Err(cause) => return Err(cause),
        };
        self.sources.extend(prepared.finish_preparation()?);
        Ok(output)
    }
    fn contiguous(&mut self, value: Array) -> Result<Array, WeightRecipeError> {
        Ok(contiguous(value, false, self.stream)?)
    }
}
