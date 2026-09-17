//! Ordinary static target equations over their actual projected source.
use super::*;
use eredu_nn::workspace::{WorkspaceMetadataError, WorkspaceTraceReport};
use eredu_runtime::PredictionTargetOperation;
use std::mem::{size_of, size_of_val};

impl PreparedInferenceBlueprint {
    /// Quotes the selected target's ordinary embedding or vocabulary operation.
    /// The borrowed input retains the caller's exact layout/backing in context.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_external_target_static(
        &self,
        invocation: ExternalInvocation,
        operation: ExternalPredictionTargetOperation<'_, WorkspaceTensor>,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        trace: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        let geometry = invocation.geometry();
        let (input, valid) = match operation {
            ExternalPredictionTargetOperation::TokenEmbeddings(input) => (
                input,
                invocation.kind() == ExternalInvocationKind::TargetTokenEmbeddings
                    && matches!(
                        input.layout().dtype(),
                        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
                    )
                    && input.shape().len() == 2,
            ),
            ExternalPredictionTargetOperation::ProjectLogits(input) => (
                input,
                invocation.kind() == ExternalInvocationKind::TargetProjectLogits
                    && input.layout().dtype() == WorkspaceDtype::Float32
                    && input.shape().len() == 3,
            ),
        };
        context
            .charge_metadata(size_of::<(
                ExternalInvocation,
                ExternalTargetQuote<'_>,
                EquationVisitor<'_, '_, '_>,
                Result<EquationQuote, PreparedExecutionError<Error>>,
            )>())
            .map_err(|e| PreparedExecutionError::Metadata(e.into()))?;
        if !valid
            || geometry.cached_positions != 0
            || geometry.max_output_tokens != 0
            || geometry.prefill_chunk_positions != geometry.input_positions
            || input.shape()[0] as u64 != geometry.batch_size
            || input.shape()[1] as u64 != geometry.input_positions
        {
            return Err(PreparedExecutionError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        geometry
            .validate()
            .map_err(|e| preparation_message(context, format_args!("{e}")))?;
        state
            .validate_workspace_context(context)
            .map_err(PreparedExecutionError::Metadata)?;
        if self.selected().text_realization().state().policy() != &CacheResidencyPolicy::Device {
            return Err(PreparedExecutionError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        let trace = std::cell::RefCell::new(trace);
        EquationVisitor {
            geometry,
            state,
            context,
            sampling: None,
            parameters,
            unpriced_execution: None,
            observation: None,
            trace: Some(EquationTraceRef(&trace)),
            media: None,
            input_dtype: None,
            target_capture: false,
                routed_pass: None,
            external_target: Some(ExternalTargetQuote::Static(operation)),
        }
        .construct(self)
        .map(|(report, _)| report)
    }
}
struct StaticOperation<'a>(ExternalPredictionTargetOperation<'a, WorkspaceTensor>);
impl<A> PredictionTargetOperation<PreparedCompositeArchitecture<A>, WorkspaceBackend, ResidentState>
    for StaticOperation<'_>
where
    A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Output = WorkspaceTensor;
    fn preserves_architecture_declarations(&self) -> bool {
        true
    }
    fn apply(
        self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        _: &mut ResidentState,
        parallel: Option<&<WorkspaceBackend as eredu_nn::NeuralBackend>::ParallelContext>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if parallel.is_some() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        architecture
            .inner_mut()
            .external_prediction_target_operation(self.0, context)?
            .ok_or_else(|| {
                context.metadata_error(format_args!(
                    "target omitted its static assistant operation"
                ))
            })
    }
}
impl EquationVisitor<'_, '_, '_> {
    pub(in crate::prepared_execution::workspace) fn quote_external_static<A>(
        self,
        runtime: &mut EquationRuntime<'_, PreparedCompositeArchitecture<A>>,
        operation: ExternalPredictionTargetOperation<'_, WorkspaceTensor>,
    ) -> Result<EquationQuote, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error> + 'static,
        A::InputPartPlan: 'static,
    {
        let controls = [
            size_of::<StaticOperation<'_>>(),
            size_of::<WorkspaceTensor>(),
            size_of::<Result<WorkspaceTensor, Error>>(),
            size_of::<ResidentState>(),
            size_of::<Vec<WorkspaceTensor>>(),
            size_of::<WorkspaceTraceReport>(),
            size_of::<Result<WorkspaceTraceReport, Error>>(),
            size_of::<bool>(),
        ];
        self.context.charge_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let mut state = self.state.try_clone_workspace(self.context)?;
        let input = match operation {
            ExternalPredictionTargetOperation::TokenEmbeddings(input)
            | ExternalPredictionTargetOperation::ProjectLogits(input) => input,
        };
        let mut ran = false;
        let equations =
            quote_inference_workspace_with_context(self.geometry, self.context, |span| -> Result<WorkspaceTraceReport,Error> {
                if ran {
                    return Err(WorkspaceMetadataError::Unqualified.into());
                }
                ran = true;
                // The static equation leaves the actual cache frontier unchanged.
                let mut opening = self.context.metadata_vec(0)?;
                for value in state
                    .as_ref()
                    .iter()
                    .flat_map(|lane| lane.retained_values())
                {
                    self.context.reserve_metadata_vec(&mut opening, 1)?;
                    opening.push(value.clone());
                }
                self.context.reserve_metadata_vec(&mut opening, 1)?;
                opening.push(input.clone());
                self.context.begin_state_span(opening.iter())?;
                // Same checkpoint and possible failed-operation restore as the session.
                let mut checkpoint = self.context.metadata_vec(state.as_ref().len())?;
                for lane in state.as_ref() {
                    checkpoint.push(lane.checkpoint_for_transaction(self.context)?);
                }
                let output = runtime.prediction_operation(
                    &mut state,
                    StaticOperation(operation),
                    self.context,
                )?;
                let mut rollback = self.context.metadata_vec(checkpoint.len())?;
                for lane in &checkpoint {
                    rollback.push(lane.checkpoint_for_transaction(self.context)?);
                }
                let mut retained = self.context.metadata_vec(0)?;
                for value in state
                    .as_ref()
                    .iter()
                    .flat_map(|lane| lane.retained_values())
                {
                    self.context.reserve_metadata_vec(&mut retained, 1)?;
                    retained.push(value.clone());
                }
                let state_roots = retained.len();
                let output_population = self
                    .context
                    .report_scalars(std::slice::from_ref(&output))?
                    .closing_storage;
                self.context.reserve_metadata_vec(&mut retained, 1)?;
                retained.push(output);
                let report = self.context.finish_report(&retained)?;
                self.trace
                    .ok_or(WorkspaceMetadataError::Unqualified)?
                    .0
                    .borrow_mut()
                    .observe_prepared_with_storage(
                        span,
                        &report,
                        state_roots,
                        1,
                        None,
                        Some(output_population),
                    )?;
                Ok(report)
            })
            .map_err(|e| self.context.metadata_source(e))?;
        Ok((equations, None))
    }
}
