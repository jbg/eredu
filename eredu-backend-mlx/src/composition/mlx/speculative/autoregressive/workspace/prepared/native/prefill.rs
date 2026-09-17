//! The selected prefill callback runs the existing shared driver. Each actual
//! row receives the same native Scope/Graph/Record/recovery worker as decode.
use super::*;
use crate::composition::mlx::replicated_text::AutoregressiveSequenceCompletion;
use crate::composition::mlx::speculative::autoregressive::prefill_input::PreparedPrefillInput;
use eredu_core::{Completion, GenerationCancellationToken, SpeculativePrefillOutcome, Submission};
use eredu_runtime::prefill::{
    PrefillChunk, PrefillDriver, PrefillError, PrefillExecutor, PrefillOutcome,
};
use eredu_runtime::speculative::autoregressive::{AutoregressivePass, AutoregressivePrefill};
use eredu_runtime::working_memory::OriginalSpeculativePrefillSpan;
use std::convert::Infallible;

struct Program {
    plans: Vec<Plan>,
    addressable:Option<SpeculativeAddressableSources>,
    input: PreparedPrefillInput,
    recipe: AutoregressiveEquationRecipe,
    projected: ProjectedResidentState,
    layerwise: Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    report: InferenceWorkspaceReport,
    context: WorkspaceContext,
    source_origin: eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    roots_runtime: PrefillRootsRuntime,
    finished: Cell<bool>,
    role: OriginalSpeculativeRole,
    funding: WorkspaceMetadataFunding,
}
/// Private callback installation, never a reusable source or admission proof.
/// Its last Rc shell retires before the program and the program's funding.
pub(crate) struct ActiveSpeculativePrefill(Option<Rc<Program>>);
impl Clone for ActiveSpeculativePrefill {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.inner())))
    }
}
impl Drop for ActiveSpeculativePrefill {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl std::fmt::Debug for ActiveSpeculativePrefill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ActiveSpeculativePrefill")
    }
}
struct Installed<'a> {
    sources: &'a AutoregressiveSourcePair,
    active: ActiveSpeculativePrefill,
}
impl Drop for Installed<'_> {
    fn drop(&mut self) {
        let mut slot = self.sources.prefill.borrow_mut();
        if slot
            .as_ref()
            .is_some_and(|value| Rc::ptr_eq(value.inner(), self.active.inner()))
        {
            drop(slot.take());
        }
    }
}
impl AutoregressiveSourcePair {
    pub(crate) fn active_prefill(&self) -> Result<ActiveSpeculativePrefill, Error> {
        self.prefill
            .try_borrow()
            .ok()
            .and_then(|slot| slot.clone())
            .ok_or_else(|| retain_planning_error(Cause::Source, self.metadata_funding().clone()))
    }
}

impl PreparedAutoregressiveInvocation<'_> {
    pub(super) fn run_prefill<T, F>(
        mut self,
        environment: &OriginalCopyEnvironment<'_>,
        input: PreparedPrefillInput,
        run: F,
    ) -> Result<T, Error>
    where
        F: FnOnce(&mut Executable, &mut MlxAutoregressiveState) -> Result<T, Error>,
    {
        let cold = |cause| retain_planning_error(cause, self.funding.clone());
        let mechanism = self
            .model
            .workspace_mechanisms()
            .ok_or_else(|| cold(Cause::Source))?;
        if self.claim.invocation().execution_pass() != eredu_runtime::ExpertPass::Prefill
            || input.positions() != self.report.geometry().input_positions
        {
            return Err(cold(Cause::Source));
        }
        // The exact copied U32 source is complete before tracing. Only its
        // actual static index is added to each original equation row.
        self.recipe
            .bind_prefill_inputs(mechanism, &self.context, |chunk| {
                input.trace_span(chunk, &self.context)
            })
            .map_err(|cause| cold(Cause::Backend(cause)))?;
        let row_count = self.recipe.records().len();
        let mut addressable=SpeculativeAddressableSources::prepare(self.recipe.records(),self.group_source_facts,&self.funding)?;
        let view_controls =
            PreparedPrefillInput::view_control_bytes().ok_or_else(|| cold(Cause::Unknown))?;
        let mut plans = self
            .context
            .metadata_vec::<Plan>(row_count)
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        let mut physical = 0u64;
        let mut graph = 0u64;
        let mut record = 0u64;
        let mut controls = self.group_control_bytes.checked_add(addressable.as_ref().map_or(0,|source|source.controls()))
            .ok_or_else(||cold(Cause::Overflow))?;
        for ordinal in 0..row_count {
            let plan = Plan::inspect::<Option<Array>, SpanCall<'_>>(
                &self.recipe,
                self.state,
                ordinal,
                environment,
                None,
                view_controls,
            )
            .map_err(&cold)?;
            physical = physical
                .checked_add(
                    u64::try_from(plan.physical_capacity).map_err(|_| cold(Cause::Overflow))?,
                )
                .ok_or_else(|| cold(Cause::Overflow))?;
            graph = graph
                .checked_add(plan.graph_bytes)
                .ok_or_else(|| cold(Cause::Overflow))?;
            record = record
                .checked_add(plan.record_bytes)
                .ok_or_else(|| cold(Cause::Overflow))?;
            controls = controls
                .checked_add(plan.controls)
                .ok_or_else(|| cold(Cause::Overflow))?;
            if self.recipe.records()[ordinal].addressable().is_some() {
                controls=controls.checked_add(addressable_call_controls::<Option<Array>,SpanCall<'_>>()
                    .ok_or_else(||cold(Cause::Overflow))?).ok_or_else(||cold(Cause::Overflow))?;
            }
            plans.push(plan);
        }
        let roots_runtime = self
            .model
            .erased()
            .prefill_roots_runtime()
            .map_err(|cause| cold(Cause::Backend(cause)))?;
        // These are the actual controller/driver/callback objects. Native
        // per-row owners are admitted separately above, once per real row.
        let frames = [
            rc_bytes::<Program>().ok_or_else(|| cold(Cause::Overflow))?,
            size_of::<Program>(),
            size_of::<ActiveSpeculativePrefill>(),
            size_of::<Installed<'_>>(),
            size_of::<F>(),
            size_of::<T>(),
            size_of::<Result<T, Error>>(),
            size_of::<SpanCall<'_>>(),
            size_of::<Sequence<'_>>(),
            size_of::<Executor<'_, '_>>(),
            size_of::<PrefillDriver<Array, SettledSpan, OriginalSpeculativeRole>>(),
            size_of::<Submission<Option<Array>, SettledSpan>>(),
            size_of::<PrefillError<Error, Infallible>>(),
            size_of::<Result<(PrefillOutcome, Option<Array>), PrefillError<Error, Infallible>>>(),
            size_of::<SpeculativePrefillOutcome<AutoregressivePrefill<Array>>>(),
            size_of::<SpeculativePrefillOutcome<AutoregressivePrefill<crate::composition::mlx::speculative::sampling::numerical::OriginalNumericalValue>>>(),
            size_of::<Result<SpeculativePrefillOutcome<AutoregressivePrefill<crate::composition::mlx::speculative::sampling::numerical::OriginalNumericalValue>>, Error>>(),
            size_of::<crate::composition::mlx::speculative::autoregressive::CompletedNumericalPrefill>(),
            size_of::<Result<SpeculativePrefillOutcome<AutoregressivePrefill<Array>>, Error>>(),
            size_of::<OriginalSpeculativePrefillSpan>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| cold(Cause::Overflow))?;
        self.context
            .charge_metadata(bytes)
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        let requirements = SpeculativeInvocationRequirements::new(
            self.recipe.plan(),
            Some(physical),
            Some(graph),
            Some(record),
            Some(controls),
        )
        .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        let requirements = if let Some(source)=addressable.as_mut(){
            requirements.with_host_source_spans(source.take_facts()?)
                .map_err(|cause|retain_planning_error(cause,self.funding.clone()))?
        }else{match self.group_source_facts {
            Some(facts)=>requirements.with_host_source_constructions(facts)
                .map_err(|cause|retain_planning_error(cause,self.funding.clone()))?,
            None=>requirements,
        }};
        let role = self
            .sources.request()
            .reserve_role(self.claim, requirements)
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        let active = ActiveSpeculativePrefill(Some(Rc::new(Program {
            plans,
            addressable,
            input,
            recipe: self.recipe,
            projected: self.projected,
            layerwise: self.layerwise,
            report: self.report,
            context: self.context,
            source_origin: self.source_origin,
            roots_runtime,
            finished: Cell::new(false),
            role,
            funding: self.funding,
        })));
        {
            let mut slot = self
                .sources
                .prefill
                .try_borrow_mut()
                .map_err(|_| failure(Cause::Source, &active.inner().role))?;
            if slot.is_some() {
                return Err(failure(Cause::Source, &active.inner().role));
            }
            *slot = Some(active.clone());
        }
        let installed = Installed {
            sources: self.sources,
            active: active.clone(),
        };
        let result = run(self.model, self.state);
        drop(installed);
        let value = result?;
        if !active.inner().finished.get() {
            return Err(failure(Cause::Completion, &active.inner().role));
        }
        Ok(value)
    }
}

impl ActiveSpeculativePrefill {
    pub(super) fn layerwise(&self) -> Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace> {
        self.inner().layerwise.as_ref()
    }
    fn inner(&self) -> &Rc<Program> {
        self.0.as_ref().expect("live prefill view")
    }
    pub(crate) fn run(
        &self,
        model: &mut Executable,
        state: &mut MlxAutoregressiveState,
        pass: AutoregressivePass,
        cancellation: &GenerationCancellationToken,
        sources: &AutoregressiveSourcePair,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<SpeculativePrefillOutcome<AutoregressivePrefill<Array>>, Error> {
        self.run_with(
            model,
            state,
            pass,
            cancellation,
            sources,
            environment,
            |value, _, _, _| Ok(value),
        )
    }
    /// Same driver and settled-span proof, moving the final selected output
    /// directly into a numerical source. Draft state-only prefill stays None.
    pub(crate) fn run_numerical(
        &self,
        model: &mut Executable,
        state: &mut MlxAutoregressiveState,
        pass: AutoregressivePass,
        cancellation: &GenerationCancellationToken,
        sources: &AutoregressiveSourcePair,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<
        SpeculativePrefillOutcome<
            AutoregressivePrefill<
                crate::composition::mlx::speculative::sampling::numerical::OriginalNumericalValue,
            >,
        >,
        Error,
    > {
        self.run_with(model, state, pass, cancellation, sources, environment,
            |value, stream, role, funding| {
                if !matches!(stream, StateStream::Original(_)) {
                    return Err(failure(Cause::Source, role));
                }
                crate::composition::mlx::speculative::sampling::numerical::OriginalNumericalValue::from_prefill(
                    crate::composition::mlx::speculative::autoregressive::CompletedNumericalPrefill {
                        value, stream: stream.clone(), custody: role.budget_custody(), funding: funding.clone(),
                    },
                )
            })
    }
    fn run_with<T>(
        &self,
        model: &mut Executable,
        state: &mut MlxAutoregressiveState,
        pass: AutoregressivePass,
        cancellation: &GenerationCancellationToken,
        sources: &AutoregressiveSourcePair,
        environment: &OriginalCopyEnvironment<'_>,
        finish: fn(
            Array,
            &StateStream,
            &OriginalSpeculativeRole,
            &WorkspaceMetadataFunding,
        ) -> Result<T, Error>,
    ) -> Result<SpeculativePrefillOutcome<AutoregressivePrefill<T>>, Error> {
        let program = self.inner();
        if pass != program.role.invocation().pass()
            || program.finished.get()
            || !Rc::ptr_eq(sources.active_prefill()?.inner(), self.inner())
            || state.source_role != program.role.invocation().source()
            || !state
                .source_origin
                .as_ref()
                .is_some_and(|origin| origin.same_origin(&program.source_origin))
        {
            return Err(failure(Cause::Source, &program.role));
        }
        sources.validate_environment(environment)?;
        sources.validate_source(model, state.source_role)?;
        let mut driver =
            PrefillDriver::<Array, SettledSpan, OriginalSpeculativeRole>::new_original_speculative(
                sources.request().execution_identity(),
                program.role.clone(),
                program.report.geometry(),
                cancellation.clone(),
            )
            .map_err(|cause| retain_planning_error(cause, program.funding.clone()))?;
        let mut executor = Executor {
            active: self,
            sources,
            model,
            state,
            environment,
        };
        let result = driver
            .run_final(&mut executor)
            .map_err(|cause| match cause {
                PrefillError::Submission(cause) => cause,
                PrefillError::Completion(never) => match never {},
                PrefillError::OutputContract | PrefillError::Failed => {
                    failure(Cause::Completion, &program.role)
                }
            })?;
        let evaluated_tokens = usize::try_from(driver.completed_positions())
            .map_err(|_| failure(Cause::Overflow, &program.role))?;
        program.finished.set(true);
        Ok(match result {
            (PrefillOutcome::Complete, logits) => {
                SpeculativePrefillOutcome::Complete(AutoregressivePrefill {
                    logits: logits
                        .map(|value| finish(value, &state.stream, &program.role, &program.funding))
                        .transpose()?,
                    evaluated_tokens,
                })
            }
            (PrefillOutcome::Cancelled, _) => {
                SpeculativePrefillOutcome::Cancelled { evaluated_tokens }
            }
        })
    }
}

// Constructed only after run_span has completed and validated all state/output
// roots, settled recovery, and retired the exact native Record snapshot.
struct SettledSpan {
    _role: OriginalSpeculativeRole,
}
impl Completion for SettledSpan {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}
struct Executor<'a, 'env> {
    active: &'a ActiveSpeculativePrefill,
    sources: &'a AutoregressiveSourcePair,
    model: &'a mut Executable,
    state: &'a mut MlxAutoregressiveState,
    environment: &'a OriginalCopyEnvironment<'env>,
}
impl PrefillExecutor<OriginalSpeculativeRole> for Executor<'_, '_> {
    type Output = Array;
    type Completion = SettledSpan;
    type Error = Error;
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        role: OriginalSpeculativeRole,
    ) -> Result<Submission<Option<Array>, SettledSpan>, Error> {
        let program = self.active.inner();
        if !role.same_role(&program.role) {
            return Err(failure(Cause::Source, &program.role));
        }
        // The claim is consumed before the first copy/view/native constructor.
        // Cache rollback and cancellation cannot restore this ordinal.
        let span = role
            .claim_prefill_span(chunk)
            .map_err(|cause| retain_planning_error(cause, program.funding.clone()))?;
        self.sources
            .validate_source(self.model, role.invocation().source())?;
        if self.state.native.generation_fixed() != Some(chunk.position) {
            return Err(failure(Cause::Source, &role));
        }
        let mechanism = self
            .model
            .workspace_mechanisms()
            .ok_or_else(|| failure(Cause::Source, &role))?;
        let checkpoint = self.state.native.copy_original(
            self.environment,
            &program.roots_runtime,
            mechanism,
            &program.funding,
            self.sources.request().capacity_bytes(),
        )?;
        let plan = program
            .plans
            .get(span.ordinal())
            .ok_or_else(|| failure(Cause::Source, &role))?;
        let output = run_span(
            self.sources,
            self.model,
            self.state,
            &program.recipe,
            plan,
            Some(&span),
            self.environment,
            &program.roots_runtime,
            role.clone(),
            checkpoint,
            None,
            RetainedEquation::Prefill(self.active.clone()),
            program.funding.clone(),
            program.addressable.as_ref().map(|source|source.take(span.ordinal())).transpose()?.flatten(),
            SpanCall {
                input: &program.input,
                span: &span,
            },
            SpanCall::invoke,
        )?;
        Ok(Submission {
            output,
            completion: SettledSpan { _role: role },
        })
    }
}
struct SpanCall<'a> {
    input: &'a PreparedPrefillInput,
    span: &'a OriginalSpeculativePrefillSpan,
}
impl SpanCall<'_> {
    fn invoke(
        self,
        model: &mut Executable,
        state: &mut MlxAutoregressiveState,
        active: &ActiveSpeculativeInvocation,
    ) -> Result<Option<Array>, Error> {
        active.begin_equation_construction()?;
        let tokens = self
            .input
            .view_span(self.span.chunk(), active.observer(), &state.stream)?;
        model
            .erased_mut()
            .autoregressive_prefill_span_with_completion(
                &tokens,
                &mut state.native,
                self.span,
                &state.stream,
                &mut Sequence(active),
            )
    }
}
struct Sequence<'a>(&'a ActiveSpeculativeInvocation);
impl AutoregressiveSequenceCompletion for Sequence<'_> {
    fn role(&self) -> &OriginalSpeculativeRole {
        self.0.role()
    }
    fn metadata_funding(&self) -> WorkspaceMetadataFunding {
        self.0.metadata_funding()
    }
    fn take_checkpoint(&mut self) -> Result<MlxPredictionTargetState, Error> {
        self.0.take_checkpoint()
    }
    fn complete(
        &mut self,
        output: Option<&Array>,
        state: &dyn AutoregressiveStateRoots,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.0.complete_sequence_roots(output, state, stream)
    }
}
