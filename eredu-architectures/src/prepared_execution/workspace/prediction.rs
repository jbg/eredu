//! Actual prediction equations through the retained target/profile construction.
use super::*;
use crate::prediction_extension::{
    MaterializedPredictionExecutor, MaterializedPredictionExtension, PredictionOperationInvoker,
    PredictionStateSourceLayout,
    equation::{PredictionEquation, PredictionEquationOutput, execute_prediction_equation},
    workspace::{
        WorkspacePredictionMaterializer, WorkspacePredictionParameterSource,
        WorkspacePredictionState,
    },
};
use eredu_core::speculative::SpeculativeActivationPhase;
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceMetadataError, WorkspaceTraceReport};
use eredu_runtime::{
    speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    working_memory::InferenceWorkspaceObserver,
};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};
mod construction;
mod roots;

/// Source-specific scalar input and actual post-equation readout. These are
/// native-mechanism companions, not architecture equations or source authority.
/// A literal host array must retain its real source receipt rather than using a
/// different tensor initializer to make a numerical quote appear complete.
pub trait WorkspacePredictionEquationTails {
    /// Models the same checked scalar token constructor as ordinary prediction.
    fn token(&mut self, id: u32, context: &WorkspaceContext) -> Result<WorkspaceTensor, Error>;
    /// Visits actual source values retained by the companion through completion.
    /// These belong to retained state, independently of semantic output roots.
    fn visit_retained(&self, _visitor: &mut dyn FnMut(&WorkspaceTensor)) {}
    /// Reports the exact post-equation state iterator used by physical
    /// completion. Output roots and explicit driver completion points remain
    /// separate; this scalar conveys neither storage nor native authority.
    fn completed_state_roots(&mut self, _count: usize, _context: &WorkspaceContext) -> Result<(), Error> { Ok(()) }
    /// Models the actual logits-row/output tail. Raw equation roots remain live
    /// independently. Reserve each exact row through `context` before appending;
    /// every appended view is retained through the completed report.
    fn readout(
        &mut self,
        output: &PredictionEquationOutput<WorkspaceTensor>,
        rows: &mut Vec<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<(), Error>;
}

#[derive(Debug, thiserror::Error)]
enum QuoteError {
    #[error("prediction equation differs from its actual invocation geometry")]
    Geometry,
    #[error("prediction quote source was consumed more than once")]
    Consumed,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct QuoteFailure {
    #[source]
    cause: PreparedExecutionError<Error>,
    _funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}
fn with_failure<T, F: FnOnce() -> Result<T, PreparedExecutionError<Error>>>(
    context: &WorkspaceContext,
    work: F,
) -> Result<T, PreparedExecutionError<Error>> {
    let parts = [
        size_of::<F>(),
        size_of::<Result<T, PreparedExecutionError<Error>>>(),
        size_of::<QuoteFailure>(),
        size_of::<Option<eredu_nn::workspace::HostMetadataFunding>>(),
        WorkspaceContext::metadata_source_bytes::<QuoteFailure>().ok_or_else(|| {
            PreparedExecutionError::Metadata(WorkspaceMetadataError::Overflow.into())
        })?,
    ];
    context
        .charge_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| {
                    PreparedExecutionError::Metadata(WorkspaceMetadataError::Overflow.into())
                })?,
        )
        .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
    let funding = context.metadata_funding();
    work().map_err(|cause| {
        PreparedExecutionError::Metadata(Error::backend_retained_source(QuoteFailure {
            cause,
            _funding: funding,
        }))
    })
}

fn value<T, F: FnOnce() -> T>(context: &WorkspaceContext, construct: F) -> Result<T, Error> {
    let parts = [
        size_of::<T>(),
        size_of::<F>(),
        size_of::<Result<T, Error>>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    Ok(construct())
}

/// Exact logical attribution for an already selected prediction observer.
/// The physical cache frontier belongs to the invocation descriptor separately.
pub struct EmbeddedPredictionWorkspaceObservation<'a> {
    observer: &'a mut dyn InferenceWorkspaceObserver,
    prediction: u64,
}
impl<'a> EmbeddedPredictionWorkspaceObservation<'a> {
    /// Borrows the selected observer without allocating or granting execution.
    pub fn new(observer: &'a mut dyn InferenceWorkspaceObserver, prediction: u64) -> Self {
        Self { observer, prediction }
    }
}

struct Quote<'a, 'observer, P: WorkspacePredictionParameterSource, Q> {
    blueprint: &'a PreparedInferenceBlueprint,
    workspace: EmbeddedInvocationWorkspace,
    equation: PredictionEquation<WorkspaceTensor>,
    target: &'a ResidentState,
    context: &'a WorkspaceContext,
    target_parameters: Option<&'a dyn WorkspaceLayerwiseParameters>,
    parameters: RefCell<Option<P::Context<'a>>>,
    project: RefCell<Option<Q>>,
    tails: RefCell<&'a mut dyn WorkspacePredictionEquationTails>,
    // Caller supplies its already selected logical phase observer. This quote
    // preserves that attribution; physical state position is not a prediction ID.
    // Its loan is independent of local parameter/context loans and never escapes
    // in the owned report returned by this quote.
    observer: RefCell<Option<&'observer mut dyn InferenceWorkspaceObserver>>,
    observation_prediction: Option<u64>,
    trace: RefCell<&'a mut dyn InferenceEquationTraceObserver>,
}

impl PreparedInferenceBlueprint {
    /// Quotes one actual selected prediction prefill, sequential/fused proposal,
    /// or replay. The extension is prepared exactly once after the total source
    /// agreement check; `project_state` receives that preparation's exact layout.
    /// Its returned state aliases the caller's actual native projection and stays
    /// live through parameter materialization, equations and trace consumption.
    ///
    /// The caller lends an already selected observation phase, exact scalar and
    /// readout companions, and native source owners. No placement is published;
    /// neither this report nor its descriptor grants native execution authority.
    /// Target transaction/copy/transport/native completion remain distinct owners.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_embedded_prediction_invocation<'a, 'observer, P, Q>(
        &'a self,
        workspace: EmbeddedInvocationWorkspace,
        equation: PredictionEquation<WorkspaceTensor>,
        target_state: &'a ResidentState,
        context: &'a WorkspaceContext,
        target_parameters: Option<&'a dyn WorkspaceLayerwiseParameters>,
        prediction_parameters: P::Context<'a>,
        project_state: Q,
        observation: Option<EmbeddedPredictionWorkspaceObservation<'observer>>,
        tails: &'a mut dyn WorkspacePredictionEquationTails,
        trace: &'a mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>>
    where
        P: WorkspacePredictionParameterSource,
        Q: for<'layout> FnOnce(
            PredictionStateSourceLayout<'layout>,
            &WorkspaceContext,
        ) -> Result<WorkspacePredictionState, Error>,
    {
        let parts = [
            size_of::<Quote<'a, 'observer, P, Q>>(),
            size_of::<Result<EquationQuote, PreparedExecutionError<Error>>>(),
            size_of::<QuoteError>(),
            size_of::<Option<EmbeddedPredictionWorkspaceObservation<'_>>>(),
            size_of::<Option<u64>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<PredictionEquation<WorkspaceTensor>>(),
            size_of::<EmbeddedInvocationWorkspace>(),
            size_of::<
                Result<
                    EmbeddedInvocationWorkspace,
                    eredu_runtime::speculative::embedded_occurrence::EmbeddedOccurrenceError,
                >,
            >(),
            size_of::<PredictionEquation<()>>(),
            size_of::<Result<PredictionEquation<()>, Error>>(),
            size_of::<Result<InferenceWorkspaceReport, PreparedExecutionError<Error>>>(),
        ];
        context
            .charge_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or_else(|| {
                        PreparedExecutionError::Metadata(WorkspaceMetadataError::Overflow.into())
                    })?,
            )
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        // The ordinary total constructor authenticates exact retained prediction
        // source/selection before any selected route invokes prepare/project.
        let (observation, observation_prediction) = match observation {
            Some(source) => (Some(source.observer), Some(source.prediction)),
            None => (None, None),
        };
        let quote = Quote::<P, Q> {
            blueprint: self,
            workspace,
            equation,
            target: target_state,
            context,
            target_parameters,
            parameters: RefCell::new(Some(prediction_parameters)),
            project: RefCell::new(Some(project_state)),
            tails: RefCell::new(tails),
            observer: RefCell::new(observation),
            observation_prediction,
            trace: RefCell::new(trace),
        };
        with_failure(context, || {
            quote
                .validate_equation()
                .map_err(PreparedExecutionError::Metadata)?;
            quote.construct().map(|(report, _)| report)
        })
    }
}

impl<'a, 'observer, P, Q> Quote<'a, 'observer, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    fn invalid(&self) -> Error {
        self.context.metadata_source(QuoteError::Geometry)
    }
    fn validate_equation(&self) -> Result<(), Error> {
        let invocation = self.workspace.invocation();
        let geometry = self.workspace.geometry();
        if EmbeddedInvocationWorkspace::prediction(
            invocation,
            geometry.cached_positions,
            geometry.output,
        )
        .map_err(|cause| self.context.metadata_source(cause))?
            != self.workspace
        {
            return Err(self.invalid());
        }
        // The target may already contain the completed prefill/replay whose
        // opening canonical coordinate this invocation retains. Authenticate
        // projection trace/layout without equating those frontiers or imposing a
        // common position across architecture-declared state segments.
        self.target.validate_workspace_context(self.context)?;
        let positions = i32::try_from(invocation.positions()).map_err(|_| self.invalid())?;
        let hidden = |value: &WorkspaceTensor, count: i32| {
            value.layout().dtype() == WorkspaceDtype::Float32
                && match value.shape() {
                    [1, n, width] => *n == count && *width > 0,
                    [1, n, streams, width] => *n == count && *streams > 0 && *width > 0,
                    _ => false,
                }
        };
        let tokens = |value: &WorkspaceTensor, count: i32| {
            matches!(
                value.layout().dtype(),
                WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
            ) && value.shape() == [1, count]
        };
        self.equation
            .as_ref()
            .try_map(|value| self.context.validate_values([value]))?;
        let matches = match (&self.equation, invocation.phase()) {
            (
                PredictionEquation::Prefill {
                    target_capture,
                    hidden: source,
                    tokens: ids,
                },
                SpeculativeActivationPhase::PredictionPrefill,
            ) => {
                hidden(source, positions)
                    && tokens(ids, positions)
                    && target_capture.shape().get(1).is_some_and(|&n| n > 0 && hidden(target_capture, n))
            }
            (
                PredictionEquation::Sequential {
                    hidden: source,
                    depth,
                    ..
                },
                SpeculativeActivationPhase::Proposal { depth: actual },
            ) => *depth == actual && positions == 1 && hidden(source, 1),
            (
                PredictionEquation::Fused { capacity, .. },
                SpeculativeActivationPhase::FusedProposal,
            ) => *capacity == invocation.positions(),
            (
                PredictionEquation::Replay {
                    captures,
                    tokens: ids,
                },
                SpeculativeActivationPhase::PredictionReplay,
            ) => hidden(captures, positions) && tokens(ids, positions),
            _ => false,
        };
        if !matches {
            return Err(self.invalid());
        }
        Ok(())
    }

    fn run<A>(
        &self,
        modules: crate::replicated_text::PreparedReplicatedTextModules<A>,
        mut extension: <A as crate::prediction_extension::MaterializedPredictionTarget<
            WorkspaceBackend,
        >>::Extension<WorkspacePredictionMaterializer<P>>,
        current: WorkspacePredictionState,
    ) -> Result<EquationQuote, Error>
    where
        A: eredu_runtime::LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
    {
        type WM<P> = WorkspacePredictionMaterializer<P>;
        let parts = [
            size_of::<EquationRuntime<'_, A>>(),
            size_of::<ResidentState>(),
            size_of::<WorkspacePredictionState>(),
            size_of::<PredictionEquationOutput<WorkspaceTensor>>(),
            size_of::<(
                Vec<WorkspaceTensor>,
                Vec<WorkspaceTensor>,
                Vec<WorkspaceTensor>,
                Vec<WorkspaceTensor>,
            )>(),
            size_of::<PredictionEquation<&WorkspaceTensor>>(),
            size_of::<Result<PredictionEquationOutput<WorkspaceTensor>, Error>>(),
            size_of::<Result<u64, crate::prediction_extension::equation::PredictionFrontierError>>(
            ),
            size_of::<std::cell::RefMut<'_, Option<&mut dyn InferenceWorkspaceObserver>>>(),
            size_of::<std::cell::Ref<'_, Option<&mut dyn InferenceWorkspaceObserver>>>(),
            size_of::<std::cell::RefMut<'_, &mut dyn WorkspacePredictionEquationTails>>(),
            size_of::<std::cell::Ref<'_, &mut dyn WorkspacePredictionEquationTails>>(),
            size_of::<std::cell::RefMut<'_, &mut dyn InferenceEquationTraceObserver>>(),
            size_of::<Result<(), Error>>(),
            size_of::<
                <A as crate::prediction_extension::MaterializedPredictionTarget<
                    WorkspaceBackend,
                >>::Extension<WM<P>>,
            >(),
            size_of::<Invoker<'_, '_, A>>(),
            size_of::<
                <<A as crate::prediction_extension::MaterializedPredictionTarget<
                    WorkspaceBackend,
                >>::Extension<WM<P>> as MaterializedPredictionExecutor<
                    A,
                    WorkspaceBackend,
                    WM<P>,
                >>::LaneState,
            >(),
            size_of::<Result<WorkspaceTraceReport, Error>>(),
        ];
        self.context.charge_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if self.target.layout() != modules.contract().selected().state().layout() {
            return Err(self.invalid());
        }
        let mut lane = current.prepare_current_lane::<A, _, P>(&extension, self.context)?;
        if extension
            .equation_frontier(&mut lane, &self.equation.as_ref())
            .map_err(|cause| self.context.metadata_source(cause))?
            != self.workspace.geometry().cached_positions
        {
            return Err(self.invalid());
        }
        match &self.equation {
            PredictionEquation::Sequential { depth, .. } if *depth >= extension.depth() => return Err(self.invalid()),
            PredictionEquation::Fused { .. } if !matches!(extension.occurrence_shape(), Some(eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape::Fused { .. })) => return Err(self.invalid()),
            _ => {}
        }
        let mut runtime =
            EquationRuntime::from_prepared(modules, self.target_parameters, self.context, None, false)?;
        let mut target = self.target.try_clone_workspace(self.context)?;
        let mut ran = false;
        let equations = quote_inference_workspace_with_context(
            self.workspace.geometry(),
            self.context,
            |span| {
                if ran {
                    return Err(self.context.metadata_source(QuoteError::Consumed));
                }
                ran = true;
                let observed = match self.observer.borrow_mut().as_mut() {
                    Some(observer) => observer.begin_span(
                        self.workspace.geometry(), span,
                        self.observation_prediction.expect("observer retains its logical coordinate"),
                        self.context,
                    )?,
                    None => false,
                };
                let mut opening = self.context.metadata_vec(0)?;
                roots::append_target(&target, &mut opening, self.context)?;
                roots::append_lane::<A, P, _>(&extension, &lane, &mut opening, self.context)?;
                if let Some(observer) = self.observer.borrow().as_ref() {
                    roots::append_observer(&**observer, &mut opening, self.context)?;
                }
                self.context.begin_state_span(opening.iter())?;
                let mut observer = self.observer.borrow_mut();
                let output =
                    execute_prediction_equation::<A, WorkspaceBackend, ResidentState, WM<P>, _, _>(
                        self.equation.as_ref(),
                        &mut extension,
                        &mut Invoker {
                            runtime: &mut runtime,
                            state: &mut target,
                            context: self.context,
                        },
                        &mut lane,
                        observer.as_deref_mut().filter(|_| observed).map(|o| {
                            o as &mut dyn eredu_runtime::ActivationObserver<WorkspaceTensor, Error>
                        }),
                        |id| self.tails.borrow_mut().token(id, self.context),
                    )?;
                drop(observer);
                let completion_controls = [
                    size_of::<usize>(), size_of::<Result<usize, Error>>(),
                    size_of::<&mut dyn Iterator<Item=&WorkspaceTensor>>(),
                    size_of::<fn(&mut dyn Iterator<Item=&WorkspaceTensor>)->Result<usize,Error>>(),
                ];
                self.context.charge_metadata(completion_controls.into_iter().try_fold(size_of_val(&completion_controls), usize::checked_add).ok_or(WorkspaceMetadataError::Overflow)?)?;
                let state_roots = extension.with_state_values(&mut lane, |values| {
                    let mut count=0usize;
                    for _ in values { count=count.checked_add(1).ok_or(WorkspaceMetadataError::Overflow)?; }
                    Ok::<_,Error>(count)
                })?;
                self.tails.borrow_mut().completed_state_roots(state_roots,self.context)?;
                let mut rows = self.context.metadata_vec(0)?;
                self.tails
                    .borrow_mut()
                    .readout(&output, &mut rows, self.context)?;
                let mut retained = self.context.metadata_vec(0)?;
                roots::append_target(&target, &mut retained, self.context)?;
                roots::append_lane::<A, P, _>(&extension, &lane, &mut retained, self.context)?;
                if let Some(observer) = self.observer.borrow().as_ref() {
                    roots::append_observer(&**observer, &mut retained, self.context)?;
                }
                roots::append_tails(&**self.tails.borrow(), &mut retained, self.context)?;
                let retained_roots = retained.len();
                let mut outputs = self.context.metadata_vec(2)?;
                output.visit_roots(|value| outputs.push(value.clone()));
                self.context
                    .reserve_metadata_vec(&mut outputs, rows.len())?;
                outputs.extend(rows);
                let output_roots = outputs.len();
                let population = self.context.report_scalars(&outputs)?.closing_storage;
                self.context
                    .reserve_metadata_vec(&mut retained, outputs.len())?;
                retained.extend(outputs);
                let report = self.context.finish_report(&retained)?;
                self.trace.borrow_mut().observe_prepared_with_storage(
                    span,
                    &report,
                    retained_roots,
                    output_roots,
                    None,
                    Some(population),
                )?;
                if observed {
                    self.observer.borrow_mut().as_mut().expect("active observer").end_span(span, self.context)?;
                }
                Ok(report)
            },
        )
        .map_err(|cause| self.context.metadata_source(cause))?;
        Ok((equations, None))
    }
}

struct Invoker<'a, 'parameters, A>
where
    A: eredu_runtime::LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
{
    runtime: &'a mut EquationRuntime<'parameters, A>,
    state: &'a mut ResidentState,
    context: &'a WorkspaceContext,
}
impl<A> PredictionOperationInvoker<A, WorkspaceBackend, ResidentState> for Invoker<'_, '_, A>
where
    A: eredu_runtime::LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
{
    type Error = Error;
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Error>
    where
        O: eredu_runtime::PredictionTargetOperation<A, WorkspaceBackend, ResidentState>,
    {
        self.runtime
            .prediction_operation(self.state, operation, self.context)
    }
    fn invalid_arguments(&self, message: std::fmt::Arguments<'_>) -> Error {
        self.context.metadata_error(message)
    }
    fn invalid(message: String) -> Error {
        Error::backend_message(message)
    }
}

#[cfg(test)]
mod tests;
