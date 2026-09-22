//! Descriptive realization of the ordinary recipe's shared primitive equation.
use super::{equation::RecipeOperations, *};
use crate::backend::nn::workspace::OrdinaryNativeControls;
use eredu_nn::{
    workspace::{
        WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType, WorkspaceOperationKind,
        WorkspaceTensor,
    },
    Tensor,
};
use safemlx::ops::OrdinaryRecipeCall;

#[derive(Clone, Copy, Default)]
struct RecipeWrapperPopulation {
    metadata_bytes: u64,
    observed: OrdinaryNativeControls,
}

/// The source roots and scalar validation roots remain visible to the enclosing
/// lifetime analysis. They neither certify checkpoint reads nor permit execution.
pub(crate) struct RecipeEquationTrace {
    pub(crate) output: WorkspaceTensor,
    pub(crate) sources: Vec<WorkspaceTensor>,
    pub(crate) validations: Vec<WorkspaceTensor>,
    wrappers: RecipeWrapperPopulation,
}

impl RecipeEquationTrace {
    /// Ordinary C/Rust operator transports and their actual native vector
    /// allocations, separate from the equation's descriptor/dispatch sources.
    pub(crate) fn ordinary_wrapper_population(&self) -> (u64, OrdinaryNativeControls) {
        (self.wrappers.metadata_bytes, self.wrappers.observed)
    }
    /// Finishes this equation's descriptive span through the shared native
    /// population reducer. The caller separately supplies leaf copies, final
    /// transfers, and their custody. The final root belongs to that caller's
    /// completion; scalar checks inside the equation retain their own frontiers.
    pub(crate) fn finish_native_population(
        &self,
        mechanism: crate::backend::nn::workspace::ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::SpeculativeNumericalRecipe, WeightRecipeError> {
        context.validate_values(
            std::iter::once(&self.output)
                .chain(&self.sources)
                .chain(&self.validations),
        )?;
        let report = context.finish_report(std::slice::from_ref(&self.output))?;
        Ok(
            crate::backend::nn::workspace::SpeculativeNumericalRecipe::inspect_completed_outputs(
                &report, 1, mechanism, context,
            )?,
        )
    }
}

pub(crate) fn trace_ordinary_recipe(
    recipe: &DerivedWeightRecipe,
    context: &WorkspaceContext,
    source: impl FnMut(
        &str,
        &TensorSelection,
        &WorkspaceContext,
    ) -> Result<WorkspaceTensor, WeightRecipeError>,
) -> Result<RecipeEquationTrace, WeightRecipeError> {
    let mut worker = RecipeTrace {
        context,
        source,
        sources: context.metadata_vec(0)?,
        validations: context.metadata_vec(0)?,
        wrappers: RecipeWrapperPopulation::default(),
    };
    let output = equation::materialize(recipe, &mut worker)?;
    Ok(RecipeEquationTrace {
        output,
        sources: worker.sources,
        validations: worker.validations,
        wrappers: worker.wrappers,
    })
}

struct RecipeTrace<'a, F> {
    context: &'a WorkspaceContext,
    source: F,
    sources: Vec<WorkspaceTensor>,
    validations: Vec<WorkspaceTensor>,
    wrappers: RecipeWrapperPopulation,
}
impl<F> RecipeTrace<'_, F> {
    fn wrapper(&mut self, call: OrdinaryRecipeCall) -> Result<(), WeightRecipeError> {
        let source = call.control_bytes().ok_or_else(|| {
            self.unsupported("ordinary recipe operator wrapper source is unqualified")
        })?;
        self.host_bytes(source.metadata_bytes())?;
        if let Some(observed) = source.observed_controls() {
            self.wrappers.observed.include(observed).ok_or(
                WeightRecipeError::ArithmeticOverflow("ordinary recipe observed controls"),
            )?;
        }
        Ok(())
    }
    fn host_bytes(&mut self, bytes: usize) -> Result<(), WeightRecipeError> {
        self.wrappers.metadata_bytes =
            self.wrappers
                .metadata_bytes
                .checked_add(u64::try_from(bytes).map_err(|_| {
                    WeightRecipeError::ArithmeticOverflow("recipe wrapper metadata")
                })?)
                .ok_or(WeightRecipeError::ArithmeticOverflow(
                    "recipe wrapper metadata",
                ))?;
        Ok(())
    }
    fn host_vector<T>(&mut self, capacity: usize) -> Result<(), WeightRecipeError> {
        let bytes = capacity
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<T>>()))
            .ok_or(WeightRecipeError::ArithmeticOverflow("recipe host vector"))?;
        self.host_bytes(bytes)
    }
    fn unsupported(&self, message: &'static str) -> WeightRecipeError {
        self.context
            .metadata_error(format_args!("{message}"))
            .into()
    }
    fn primitive(
        &self,
        name: &'static str,
        inputs: &[&WorkspaceTensor],
        shape: &[i32],
        dtype: WorkspaceDtype,
    ) -> Result<WorkspaceTensor, WeightRecipeError> {
        let mut outputs = self.context.metadata_vec(1)?;
        outputs.push(self.context.layout(shape, dtype)?);
        Ok(self
            .context
            .execute(WorkspaceOperationKind::Elementwise(name), inputs, outputs)?
            .pop()
            .expect("one declared recipe output"))
    }
}

impl<F> RecipeOperations for RecipeTrace<'_, F>
where
    F: FnMut(
        &str,
        &TensorSelection,
        &WorkspaceContext,
    ) -> Result<WorkspaceTensor, WeightRecipeError>,
{
    type Value = WorkspaceTensor;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, WeightRecipeError> {
        self.host_vector::<T>(capacity)?;
        Ok(self.context.metadata_vec(capacity)?)
    }
    fn source(
        &mut self,
        key: &str,
        selection: &TensorSelection,
    ) -> Result<Self::Value, WeightRecipeError> {
        let value = (self.source)(key, selection, self.context)?;
        self.context.validate_values([&value])?;
        self.context.reserve_metadata_vec(&mut self.sources, 1)?;
        self.sources.push(value.clone());
        Ok(value)
    }
    fn inputs(
        &mut self,
        recipes: &[DerivedWeightRecipe],
    ) -> Result<Vec<Self::Value>, WeightRecipeError> {
        self.host_vector::<Array>(recipes.len())?;
        let mut values = self.context.metadata_vec(recipes.len())?;
        for recipe in recipes {
            values.push(equation::execute(recipe, self)?);
        }
        Ok(values)
    }
    fn indices(&mut self, values: &[i32]) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::IndicesI32 {
            elements: values.len(),
        })?;
        Ok(crate::backend::nn::workspace::host_array::trace(
            &[usize_to_i32(values.len(), "selection index count")?],
            Dtype::Int32,
            self.context,
        )?)
    }
    fn scalar(&mut self, _value: f32) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::ScalarF32)?;
        Ok(crate::backend::nn::workspace::host_array::trace(
            &[],
            Dtype::Float32,
            self.context,
        )?)
    }
    fn take(
        &mut self,
        value: Self::Value,
        indices: Self::Value,
        axis: i32,
    ) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Take)?;
        Ok(value.take_axis(&indices, axis, self.context)?)
    }
    fn join(
        &mut self,
        values: Vec<Self::Value>,
        axis: i32,
        stack: bool,
    ) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Join {
            inputs: values.len(),
            stack,
        })?;
        self.host_vector::<&Array>(values.len())?;
        Ok(if stack {
            WorkspaceTensor::stack(&values, axis, self.context)?
        } else {
            WorkspaceTensor::concatenate(&values, axis, self.context)?
        })
    }
    fn reshape(
        &mut self,
        value: Self::Value,
        shape: &[i32],
    ) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Reshape { rank: shape.len() })?;
        Ok(value.reshape(shape, self.context)?)
    }
    fn transpose(
        &mut self,
        value: Self::Value,
        axes: &[i32],
    ) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Transpose { rank: axes.len() })?;
        Ok(value.transpose_axes(axes, self.context)?)
    }
    fn cast(&mut self, value: Self::Value, dtype: Dtype) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Cast)?;
        if self.dtype(&value)? == dtype {
            return Ok(value);
        }
        let floating = match dtype {
            Dtype::Float32 => Some(WorkspaceFloatingType::Float32),
            Dtype::Float16 => Some(WorkspaceFloatingType::Float16),
            Dtype::Bfloat16 => Some(WorkspaceFloatingType::Bfloat16),
            _ => None,
        };
        if let Some(floating) = floating {
            if value.layout().dtype() == WorkspaceDtype::Float32 {
                return Ok(value.cast_floating(floating, self.context)?);
            }
        }
        Err(self.unsupported("recipe cast lacks a selected scalar-conversion source"))
    }
    fn view(&mut self, value: Self::Value, dtype: Dtype) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::View)?;
        use crate::backend::nn::workspace::byte_view;
        let target = byte_view::Dtype::from_native(dtype)
            .ok_or_else(|| self.unsupported("recipe bit view lacks its selected scalar source"))?;
        Ok(byte_view::trace(&value, target, self.context)?)
    }
    fn dtype(&self, value: &Self::Value) -> Result<Dtype, WeightRecipeError> {
        Ok(match value.layout().dtype() {
            WorkspaceDtype::Bool => Dtype::Bool,
            WorkspaceDtype::Uint8 => Dtype::Uint8,
            WorkspaceDtype::Int32 => Dtype::Int32,
            WorkspaceDtype::Uint32 => Dtype::Uint32,
            WorkspaceDtype::Float32 => match value
                .layout()
                .representation()
                .ok_or_else(|| self.unsupported("recipe source lacks scalar representation"))?
                .dtype()
            {
                WorkspaceFloatingType::Float32 => Dtype::Float32,
                WorkspaceFloatingType::Float16 => Dtype::Float16,
                WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
            },
        })
    }
    fn less(
        &mut self,
        left: &Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Binary)?;
        self.primitive("less", &[left, &right], left.shape(), WorkspaceDtype::Bool)
    }
    fn all(&mut self, value: Self::Value) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::All)?;
        self.primitive("recipe_all", &[&value], &[], WorkspaceDtype::Bool)
    }
    fn require_true(&mut self, value: Self::Value) -> Result<(), WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::BoolScalarRead)?;
        if value.layout().dtype() != WorkspaceDtype::Bool || !value.shape().is_empty() {
            return Err(self.unsupported("recipe validation requires one Bool scalar"));
        }
        self.context
            .reserve_metadata_vec(&mut self.validations, 1)?;
        self.context.complete_values(&[&value])?;
        self.validations.push(value);
        Ok(())
    }
    fn multiply(
        &mut self,
        left: &Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Binary)?;
        Ok(left.multiply(&right, self.context)?)
    }
    fn subtract(
        &mut self,
        left: Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Binary)?;
        Ok(left.subtract(&right, self.context)?)
    }
    fn log(&mut self, value: Self::Value) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Unary)?;
        self.primitive("log", &[&value], value.shape(), value.layout().dtype())
    }
    fn with_validation(
        &mut self,
        value: Self::Value,
        body: impl FnOnce(&mut Self, &Self::Value) -> Result<Self::Value, WeightRecipeError>,
    ) -> Result<Self::Value, WeightRecipeError> {
        body(self, &value)
    }
    fn contiguous(&mut self, value: Self::Value) -> Result<Self::Value, WeightRecipeError> {
        self.wrapper(OrdinaryRecipeCall::Contiguous)?;
        Ok(value.contiguous(self.context)?)
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use crate::backend::nn::workspace::MlxCpuMatmulMechanism;
    use crate::backend::nn::workspace::{MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms};
    use eredu_nn::workspace::{WorkspaceExistingStorage, WorkspaceRepresentation};

    #[test]
    fn recipe_trace_retains_negative_validation_and_sources_without_execution_authority() {
        assert!(safemlx::OriginalScopeObserver::try_current()
            .unwrap()
            .is_none());
        let native = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(
            native.allocation(),
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
        );
        for shape in [&[][..], &[1][..], &[3][..], &[1, 3][..]] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(4096), &context).unwrap();
            let source = WorkspaceTensor::existing_with_storage(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &storage,
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let recipe = DerivedWeightRecipe::NegLog {
                input: Box::new(DerivedWeightRecipe::Source {
                    key: "rates".into(),
                    selection: TensorSelection::Full,
                }),
            };
            let trace = trace_ordinary_recipe(&recipe, &context, |key, selection, actual| {
                assert_eq!(key, "rates");
                assert_eq!(selection, &TensorSelection::Full);
                assert!(context.shares_trace(actual));
                Ok(source.clone())
            })
            .unwrap();
            assert_eq!(trace.sources.len(), 1);
            assert_eq!(trace.validations.len(), 1);
            assert_eq!(trace.output.shape(), shape);
            assert_eq!(trace.validations[0].layout().dtype(), WorkspaceDtype::Bool);
            assert!(trace.validations[0].shape().is_empty());
            let report = context.report(std::slice::from_ref(&trace.output)).unwrap();
            assert!(
                report.unpriced_operations.is_empty(),
                "{:?}",
                report.unpriced_operations
            );
            assert!(
                report.unpriced_host_operations.is_empty(),
                "{:?}",
                report.unpriced_host_operations
            );
            assert!(safemlx::OriginalScopeObserver::try_current()
                .unwrap()
                .is_none());
            assert_eq!(report.operations.len(), 8);
            assert_eq!(
                report
                    .operations
                    .iter()
                    .filter(|operation| matches!(
                        operation.kind,
                        WorkspaceOperationKind::Elementwise("recipe_all")
                    ))
                    .count(),
                1
            );
            assert_eq!(
                report
                    .operations
                    .iter()
                    .filter(|operation| matches!(
                        operation.kind,
                        WorkspaceOperationKind::ValueCompletion
                    ))
                    .count(),
                1
            );
            let population = trace
                .finish_native_population(
                    crate::backend::nn::workspace::ResidentExecutionMechanisms::Cpu {
                        ordinary: native,
                        cpu,
                    },
                    &context,
                )
                .unwrap();
            assert_eq!(population.completion.nested_completions, 1);
            assert!(population.storage.mutable_bytes() > 0);
            assert!(population.storage.maximum_births() > 0);
            let controls = population.ordinary_cpu_controls().unwrap();
            assert!(controls.observed_host_bytes > 0);
            assert!(controls.control_allocations > 0);
        }
    }
}
