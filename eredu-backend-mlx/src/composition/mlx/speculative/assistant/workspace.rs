//! Actual typed assistant/input sources through one ordinary equation trace.
mod mechanism;
pub(crate) use mechanism::AssistantMechanisms;
use super::MlxExternalAssistant;
use crate::{
    MlxTensor,
    backend::nn::workspace::{
        ExistingArrayProjection, MlxMetalWorkspaceMechanisms, ProjectedNativeStorage,
        ResidentNativeRecipe, ResidentRecipeRecorder,
    },
};
use eredu_architectures::{
    ExternalAssistantArchitecture, external_assistant::invocation::ExternalAssistantOperation,
    prepared_execution::InferenceEquationTraceObserver,
};
use eredu_nn::{
    Error, ParameterId, ParameterMetadataView, ParameterSourceVisitor, Parameterized,
    workspace::{
        WorkspaceBackend, WorkspaceContext, WorkspaceFloatingType, WorkspaceMetadataError,
        HostMetadataFunding, WorkspaceParameterRepresentation, WorkspaceRepresentation,
        WorkspaceTensor, WorkspaceTraceReport,
    },
};
use eredu_runtime::{
    PreparedParameterBinding,
    working_memory::{
        InferenceWorkspaceReport, bind_prepared_workspace_parameters,
        quote_inference_workspace_with_context,
    },
};
use safemlx::Array;
use std::{
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

/// Immutable source loans remain live until the exact quote has been consumed.
/// Source binding and a report do not grant a native operation or a model role.
pub(crate) struct AssistantEquationQuote<'source, A: ExternalAssistantArchitecture, B, C = ()> {
    parts: AssistantEquationParts<B, C>,
    _source: PhantomData<&'source MlxExternalAssistant<A>>,
}
pub(crate) struct AssistantEquationParts<B, C = ()> {
    pub(crate) invocation: C,
    pub(crate) report: InferenceWorkspaceReport,
    pub(crate) recipe: ResidentNativeRecipe,
    pub(crate) storage: ProjectedNativeStorage,
    pub(crate) bindings: B,
    pub(crate) closing_roots: usize,
    pub(crate) context: WorkspaceContext,
    // Native projection and all report/source destinations retire before H.
    _funding: Option<HostMetadataFunding>,
}
impl<A: ExternalAssistantArchitecture, B, C> AssistantEquationQuote<'_, A, B, C> {
    pub(crate) fn into_parts(self) -> AssistantEquationParts<B, C> {
        self.parts
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Error,
    _funding: Option<HostMetadataFunding>,
}
fn controls(context: &WorkspaceContext, parts: &[usize]) -> Result<(), Error> {
    context.charge_metadata(
        parts
            .iter()
            .copied()
            .try_fold(size_of_val(parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    Ok(())
}
fn mismatch(context: &WorkspaceContext) -> Error {
    context.metadata_error(format_args!(
        "external assistant parameter differs from its retained materialization source"
    ))
}
fn parameter_mismatch(context: &WorkspaceContext, name: &str, reason: &str) -> Error {
    context.metadata_error(format_args!(
        "external assistant parameter {name}: {reason}"
    ))
}
struct Parameters<'source, 'context, A: ExternalAssistantArchitecture> {
    source: &'source MlxExternalAssistant<A>,
    projection: ExistingArrayProjection<'source>,
    rows: Vec<PreparedParameterBinding<'source, WorkspaceTensor>>,
    representations: Vec<WorkspaceParameterRepresentation>,
    names: Vec<&'source str>,
    context: &'context WorkspaceContext,
    failure: Option<Error>,
}
impl<'source, A: ExternalAssistantArchitecture> Parameters<'source, '_, A> {
    fn parameter(
        &mut self,
        metadata: ParameterMetadataView<'source>,
        value: &'source MlxTensor,
    ) -> Result<(), Error> {
        let name = metadata.id().as_str();
        let (key, retained) = self
            .source
            .source
            .values()
            .get_key_value(name)
            .ok_or_else(|| parameter_mismatch(self.context, name, "absent from retained materialization values"))?;
        if self.names.contains(&key.as_str())
            || self.rows.len() == self.source.source.bindings().len()
            || !self
                .source
                .source
                .bindings()
                .iter()
                .any(|binding| binding.name() == name)
        {
            return Err(parameter_mismatch(self.context, name, "duplicate or absent retained binding"));
        }
        controls(
            self.context,
            &[
                Array::descriptor_control_bytes()
                    .ok_or(WorkspaceMetadataError::Overflow)?
                    .checked_mul(2)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
                size_of::<ParameterId>(),
                size_of::<Result<ParameterId, eredu_nn::ParameterTopologyError>>(),
                size_of::<PreparedParameterBinding<'_, WorkspaceTensor>>(),
                size_of::<WorkspaceParameterRepresentation>(),
                size_of::<Result<(), Error>>(),
            ],
        )?;
        let current = value
            .as_array()
            .try_descriptor()
            .map_err(|cause| self.context.metadata_source(cause))?;
        let original = retained
            .try_descriptor()
            .map_err(|cause| self.context.metadata_source(cause))?;
        // Full allocation ownership is exact, independently of represented bytes.
        // Logical shape/type must also agree with the retained populated slot.
        if current.facts().allocation().is_none() {
            return Err(parameter_mismatch(self.context, name, "completed native backing is unavailable"));
        }
        if current.facts() != original.facts() || current.shape() != original.shape() {
            return Err(parameter_mismatch(self.context, name, "native backing, dtype, or shape differs from materialization"));
        }
        let floating = match current.facts().dtype() {
            safemlx::Dtype::Float32 => Some(WorkspaceFloatingType::Float32),
            safemlx::Dtype::Float16 => Some(WorkspaceFloatingType::Float16),
            safemlx::Dtype::Bfloat16 => Some(WorkspaceFloatingType::Bfloat16),
            _ => None,
        };
        let identity = current.facts().allocation().expect("validated completed source").identity();
        let representation = floating
            .zip(current.row_contiguous())
            .map(|(dtype, contiguous)| WorkspaceRepresentation::new(dtype, contiguous));
        drop(original);
        drop(current);
        let projected = self.projection.project(value.as_array())?;
        let id = ParameterId::new(self.context.metadata_string(format_args!("{name}"))?)
            .map_err(|cause| self.context.metadata_source(cause))?;
        let layout = projected.layout().clone().with_representation(representation);
        self.representations.push(WorkspaceParameterRepresentation::new(id, layout.clone()));
        let root = self.projection.storage_roots().find(|(actual, _, _)| *actual == identity)
            .map(|(_, _, root)| root.clone()).ok_or_else(|| parameter_mismatch(self.context, name, "projection omitted its native source root"))?;
        let represented = WorkspaceTensor::existing_with_storage(layout, &root, self.context)?;
        self.rows.push(PreparedParameterBinding::new(key, represented));
        self.names.push(key);
        Ok(())
    }
}
impl<'source, A: ExternalAssistantArchitecture> ParameterSourceVisitor<'source, MlxTensor>
    for Parameters<'source, '_, A>
{
    fn parameter(&mut self, metadata: ParameterMetadataView<'source>, value: &'source MlxTensor) {
        if self.failure.is_none() {
            self.failure = Parameters::parameter(self, metadata, value).err();
        }
    }
    fn retained(&mut self, _: &'source MlxTensor) {
        // An unnamed native numerical field requires its own equation source
        // projection. It cannot be inferred from named checkpoint bindings.
        if self.failure.is_none() {
            self.failure = Some(self.context.metadata_error(format_args!(
                "external assistant has an unnamed retained tensor without an equation source projection"
            )));
        }
    }
}

/// Quotes one actual assistant operation while the source and current arguments
/// remain immutably lent. `bind` receives their one combined storage inventory
/// before tracing starts and must retain the actual source/account pins.
pub(crate) fn quote<'source, A, I, B>(
    assistant: &'source MlxExternalAssistant<A>,
    arguments: &'source I::Arguments<'_, MlxTensor>,
    mechanism: MlxMetalWorkspaceMechanisms,
    context: &'source WorkspaceContext,
    bind: impl FnOnce(&ProjectedNativeStorage, &WorkspaceContext) -> Result<B, Error>,
) -> Result<AssistantEquationQuote<'source, A, B>, Error>
where
    A: ExternalAssistantArchitecture,
    I: ExternalAssistantOperation<A>,
{
    quote_with_invocation::<A,I,B,()>(assistant, arguments, AssistantMechanisms::Metal(mechanism), context, |_| Ok(()), bind)
}

/// Spends the source occurrence after exact argument projection and before
/// binding or constructing the unloaded module. A failure never refunds it.
pub(crate) fn quote_with_invocation<'source, A, I, B, C>(
    assistant: &'source MlxExternalAssistant<A>,
    arguments: &'source I::Arguments<'_, MlxTensor>,
    mechanism: AssistantMechanisms,
    context: &'source WorkspaceContext,
    claim: impl FnOnce(eredu_core::InferenceGeometry) -> Result<C, Error>,
    bind: impl FnOnce(&ProjectedNativeStorage, &WorkspaceContext) -> Result<B, Error>,
) -> Result<AssistantEquationQuote<'source, A, B, C>, Error>
where
    A: ExternalAssistantArchitecture,
    I: ExternalAssistantOperation<A>,
{
    controls(
        context,
        &[
            size_of::<Parameters<'_, '_, A>>(),
            size_of::<Option<usize>>(),
            size_of::<(usize, usize, bool)>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<(), eredu_nn::ParameterSourceError>>(),
            size_of::<Vec<WorkspaceTensor>>(),
            size_of::<std::collections::btree_map::Iter<'_, String, Array>>(),
            size_of::<std::slice::Iter<'_, eredu_runtime::WeightBinding>>(),
            size_of::<I::Projected>(),
            size_of::<A::Module<WorkspaceBackend>>(),
            size_of::<Result<A::Module<WorkspaceBackend>, Error>>(),
            size_of::<I::Output<WorkspaceTensor>>(),
            size_of::<Result<I::Output<WorkspaceTensor>, Error>>(),
            size_of::<AssistantEquationQuote<'_, A, B, C>>(),
            size_of::<Result<AssistantEquationQuote<'_, A, B, C>, Error>>(),
            size_of::<ResidentRecipeRecorder>(),
            size_of::<AssistantMechanisms>(),
            size_of::<Result<ResidentNativeRecipe, Error>>(),
            size_of_val(&bind),
            size_of_val(&claim),
            size_of::<C>(),
            size_of::<Result<C, Error>>(),
            size_of::<Result<B, Error>>(),
            WorkspaceContext::metadata_source_bytes::<Failure>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ],
    )?;
    let funding = context.metadata_funding();
    let work = || {
        let mut inputs = Some(0usize);
        I::visit_arguments(arguments, &mut |_| {
            inputs = inputs.and_then(|count| count.checked_add(1));
        });
        let parameters = assistant.source.bindings().len();
        let roots = parameters
            .checked_add(inputs.ok_or(WorkspaceMetadataError::Overflow)?)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut source = Parameters {
            source: assistant,
            projection: ExistingArrayProjection::with_source_count(context, roots)
                .map_err(|cause| context.metadata_source(cause))?,
            rows: context.metadata_vec(parameters)?,
            representations: context.metadata_vec(parameters)?,
            names: context.metadata_vec(parameters)?,
            context,
            failure: None,
        };
        assistant
            .module
            .inner
            .visit_parameter_sources(&mut source)
            .map_err(|cause| context.metadata_source(cause))?;
        if let Some(cause) = source.failure {
            return Err(cause);
        }
        if source.rows.len() != parameters || assistant.source.values().len() != parameters {
            return Err(context.metadata_error(format_args!(
                "external assistant parameter inventory differs: {} visited, {} bindings, {} materialized values",
                source.rows.len(), parameters, assistant.source.values().len()
            )));
        }
        let mut projected = I::project(
            arguments,
            |value| source.projection.project(value.as_array()),
            context,
        )?;
        let storage = source.projection.try_into_storage()?;
        if !storage.is_complete() {
            return Err(mismatch(context));
        }
        let geometry = I::geometry(&projected, context)?;
        let invocation = claim(geometry)?;
        let bindings = bind(&storage, context)?;
        // Source selection precedes every placeholder/module constructor.
        context.install_parameter_representations(source.representations)?;
        let mut module = I::workspace_module(&assistant.config, context)?;
        bind_prepared_workspace_parameters(&mut module, &mut source.rows, context, |_| false)?;

        let mut recorder = mechanism.recorder(geometry, context)?;
        let mut executed = false;
        let mut closing_roots = 0;
        let trace = |span: &eredu_runtime::working_memory::InferenceWorkspaceSpan| {
            if executed {
                return Err(mismatch(context));
            }
            executed = true;
            let mut opening =
                context.metadata_vec(inputs.ok_or(WorkspaceMetadataError::Overflow)?)?;
            I::visit_projected(&projected, &mut |value| opening.push(value.clone()));
            context.begin_state_span(opening.iter())?;
            // There is no artificial token constructor in this equation. All
            // inputs are the actual source projections installed above.
            let output = I::trace(&mut module, &mut projected, context)?;
            let mut outputs = Some(0usize);
            I::visit_output(&output, &mut |_| {
                outputs = outputs.and_then(|count| count.checked_add(1))
            });
            let outputs = outputs.ok_or(WorkspaceMetadataError::Overflow)?;
            let mut closing = context.metadata_vec(opening.len())?;
            I::visit_projected(&projected, &mut |value| closing.push(value.clone()));
            let mut returned = context.metadata_vec(outputs)?;
            I::visit_output(&output, &mut |value| returned.push(value.clone()));
            if closing.len() != opening.len() || returned.len() != outputs {
                return Err(mismatch(context));
            }
            closing_roots = closing
                .len()
                .checked_add(outputs)
                .ok_or(WorkspaceMetadataError::Overflow)?;
            let output_storage = context.report_scalars(&returned)?.closing_storage;
            let report = context.finish_report(&closing)?;
            recorder.observe_prepared_with_storage(
                span,
                &report,
                closing.len(),
                outputs,
                None,
                Some(output_storage),
            )?;
            Ok::<_, Error>(report)
        };
        controls(
            context,
            &[
                size_of_val(&trace),
                size_of::<Result<WorkspaceTraceReport, Error>>(),
                size_of::<
                    Result<
                        InferenceWorkspaceReport,
                        eredu_runtime::working_memory::InferenceWorkspaceError<Error>,
                    >,
                >(),
            ],
        )?;
        let report =
            quote_inference_workspace_with_context(geometry, context, trace).map_err(|cause| {
                match cause {
                    eredu_runtime::working_memory::InferenceWorkspaceError::Metadata(cause) => {
                        cause
                    }
                    cause => context.metadata_source(cause),
                }
            })?;
        if !executed {
            return Err(mismatch(context));
        }
        let recipe = recorder.finish_external_operation(report.span_workspace_plan())?;
        Ok(AssistantEquationQuote {
            parts: AssistantEquationParts {
                invocation,
                report,
                recipe,
                storage,
                bindings,
                closing_roots,
                context: context.clone(),
                _funding: funding.clone(),
            },
            _source: PhantomData,
        })
    };
    controls(context, &[size_of_val(&work)])?;
    work().map_err(|cause| {
        Error::backend_retained_source(Failure {
            cause,
            _funding: funding,
        })
    })
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod cpu_tests;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
