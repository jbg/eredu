//! Existing shared prefill completion, with closed borrowed root filling.
use eredu_runtime::{speculative::external_occurrence::ExternalInvocation, working_memory::OriginalExternalSpeculativeRole};
use super::*;
use crate::backend::submission_recovery::prefill::nested::NestedCompletionProjection;
impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    pub(super) fn active_nested_completion(&self) -> Result<Option<NestedCompletionProjection>, Error> {
        let slot=self.nested_completion.try_borrow().map_err(|_|Error::PrefillScopeReentrant)?;
        Ok(slot.as_ref().filter(|projection|projection.is_active()).cloned())
    }
    pub(super) fn complete_nested_roots(&self, roots:&NestedCompletionProjection,
        output:Option<&MlxTensor>,state:&S,
        future:Option<&eredu_runtime::media_prefill::RetainedMediaRoots<'_,MlxTensor>>,
        stream:&Stream)->Result<(),Error> {
        use eredu_runtime::RuntimeState;
        roots.complete(stream, |visitor| {
            if let Some(output)=output { visitor(output); }
            state.visit_all_retained_values(visitor).map_err(Error::PrefillState)?;
            if let Some(future)=future { future.visit(visitor); }
            Ok(())
        })
    }

    pub(super) fn complete_prefill_roots(
        &self,
        output: Option<&MlxTensor>,
        state: &S,
        future: Option<&eredu_runtime::media_prefill::RetainedMediaRoots<'_, MlxTensor>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        use eredu_runtime::RuntimeState;
        let roots = self
            .prefill_roots
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?;
        // Ordinary completion sizes its own allocation from actual borrowed
        // values. Original completion already has its full cold maximum and
        // prepare_ordinary cannot allocate, replace or enlarge that population.
        let mut count = crate::backend::nn::tensor::active_token_validation_count()
            .map_err(|e| Error::PrefillRoots(e.into()))?
            .checked_add(usize::from(output.is_some()));
        state
            .visit_all_retained_values(&mut |_| {
                count = count.and_then(|n| n.checked_add(1));
            })
            .map_err(Error::PrefillState)?;
        if let Some(future) = future {
            future.visit(&mut |_| {
                count = count.and_then(|n| n.checked_add(1));
            });
        }
        let count = count.ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::Overflow,
        ))?;
        roots.prepare_ordinary(&self.prefill_roots_runtime, count)?;
        if let Some(output) = output {
            roots.append(output.as_array())?;
        }
        let mut result = Ok(());
        state
            .visit_all_retained_values(&mut |value| {
                if result.is_ok() {
                    result = roots.append(value.as_array());
                }
            })
            .map_err(Error::PrefillState)?;
        result?;
        roots.append_token_validations()?;
        if let Some(future) = future {
            let mut result = Ok(());
            future.visit(&mut |value| {
                if result.is_ok() {
                    result = roots.append(value.as_array());
                }
            });
            result?;
        }
        // Every state/source/TLS loan ended before native submission. The same
        // source's raw/compact/cut/prefix-context custody remains in the driver.
        #[cfg(test)]
        crate::tests::support::path_instrumentation::completion();
        let model = {
            let installed = self
                .prefill_controls
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            match installed.as_ref() {
                Some(
                    crate::backend::submission_recovery::prefill::PrefillControlProjection::Model(
                        model,
                    ),
                ) => Some(model.clone()),
                _ => None,
            }
        };
        if model.is_some() || roots.has_prepared_completion()? {
            roots.complete_on_stream(stream)?;
        } else {
            roots.complete()?;
        }
        #[cfg(test)]
        if let Some(future) = future {
            let mut n = 0;
            future.visit(&mut |_| n += 1);
            crate::backend::submission_recovery::prefill::test_trace::record(
                crate::backend::submission_recovery::prefill::test_trace::Event::Media {
                    future: n,
                },
            );
        }
        // Existing validation execution/error creation remains a separately
        // declared C obligation, never inferred funded from the fixed collector.
        validate_active_token_validations()?;
        if let Some(model) = model {
            model.mark_completed()?;
        }
        Ok(())
    }
}

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    /// Install one exact active target invocation without replacing outer
    /// prefill guards. Its separately owned guard deactivates even on unwind.
    pub(crate) fn install_session_nested_completion<D>(
        session:&ReplicatedTextSession<A,MlxNeuralBackend,Self,D>,
        projection:NestedCompletionProjection,
    )->Result<(),Error>
    where D:eredu_runtime::ReplicatedTextExecutionStrategy<A,MlxNeuralBackend,S,
        MlxArchitectureLayerwisePolicy<A,S>,MlxArchitectureLayerwisePolicy<A,S>>,
    {
        let previous=session.inspect_runtime_execution_fixed(|mechanisms,_,_| {
            let mut slot=mechanisms.nested_completion.try_borrow_mut().map_err(|_|Error::PrefillScopeReentrant)?;
            if slot.as_ref().is_some_and(|projection|projection.is_active()) {
                return Err(Error::PrefillScopeReentrant);
            }
            Ok(slot.replace(projection))
        }).map_err(Error::RuntimeInspection)??;
        drop(previous);
        Ok(())
    }

    pub(crate) fn bind_session_neural_recipe<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Result<(), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        let plan = session
            .inspect_runtime_execution_fixed(|mechanisms, _, runtime| {
                D::resident_policy(runtime)
                    .or_else(|| D::bounded_policy(runtime))
                    .map(|policy| {
                        policy
                            .original_operation_plan(recipe.plan().geometry(), None)?
                            .with_selected_stream(&mechanisms.stream)
                    })
                    .transpose()
            })
            .map_err(Error::RuntimeInspection)??;
        // Allocator/source queries execute after the native inspector and policy
        // loans end. The retained plan authenticates their immutable owners.
        if let Some(plan) = plan {
            plan.with_preparation_funding(funding).with_foreground_disk_reads(pool)?
                .bind_neural_recipe(recipe)?;
        }
        Ok(())
    }

    fn session_speculative_neural_plan<'source, D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        geometry: eredu_core::InferenceGeometry,
        source: Option<&'source crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    ) -> Result<crate::backend::runtime::execution::generic::SelectedOriginalOperationPlan<'source, A::Unit>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        session
            .inspect_runtime_execution_fixed(|mechanisms, _, runtime| {
                // This exact native bank has one selected execution stream and
                // local cancellation. Distributed agreement/transport must be
                // supplied by its own selected producer before it can join.
                if D::PARTITIONED_SESSION || D::DISTRIBUTED_PHASE_AGREEMENT
                    || mechanisms.parallel_layout.is_some()
                {
                    return Err(Error::PrefillScopeUnavailable);
                }
                D::resident_policy(runtime).or_else(|| D::bounded_policy(runtime))
                    .ok_or(Error::PrefillScopeUnavailable)?
                    .original_operation_plan(geometry, source)?
                    .with_selected_stream(&mechanisms.stream)
            })
            .map_err(Error::RuntimeInspection)?
    }
    pub(crate) fn bind_session_speculative_neural_recipe<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
        recipe: &mut crate::backend::nn::workspace::AutoregressiveEquationRecipe,
    ) -> Result<(u64, Option<eredu_runtime::working_memory::HostSourceConstructionFacts>), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        let plan = Self::session_speculative_neural_plan(session, recipe.plan().geometry(), source)
            .map_err(|cause| cause.at_speculative_stage("selected operation source"))?
            .prepare_speculative_source(pool, funding)
            .map_err(|cause| cause.at_speculative_stage("selected foreground source plan"))?;
        plan.bind_speculative_recipe(recipe)
            .map_err(|cause| cause.at_speculative_stage("selected operation recipe binding"))?;
        let source = plan.speculative_source_facts()?;
        plan.speculative_control_bytes(recipe).map(|controls| (controls, source))
    }

    pub(crate) fn prepare_session_speculative_neural_bank<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        role: eredu_runtime::working_memory::OriginalSpeculativeRole,
        scope: &safemlx::SubmissionScope,
        partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        Self::session_speculative_neural_plan(session, recipe.plan().geometry(), source)?
            .prepare_speculative(recipe, role, scope,partition)
    }

    /// Exact one-equation Embedded entry; the selected source plan and both
    /// residency bank workers are shared with ordinary/AR traversal.
    pub(crate) fn bind_session_embedded_neural_recipe<D>(
        session:&ReplicatedTextSession<A,MlxNeuralBackend,Self,D>,
        source:Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool,
        funding:&eredu_nn::workspace::WorkspaceMetadataFunding,
        recipe:&mut crate::backend::nn::workspace::EmbeddedEquationRecipe,
    )->Result<(u64,Option<eredu_runtime::working_memory::HostSourceConstructionFacts>),Error>
    where D:eredu_runtime::ReplicatedTextExecutionStrategy<A,MlxNeuralBackend,S,
        MlxArchitectureLayerwisePolicy<A,S>,MlxArchitectureLayerwisePolicy<A,S>>,
    {
        let plan=Self::session_speculative_neural_plan(session,recipe.workspace().geometry(),source)?
            .prepare_speculative_source(pool,funding)?;
        plan.bind_embedded_recipe(recipe)?;
        let source=plan.speculative_source_facts()?;
        plan.embedded_control_bytes(recipe).map(|controls|(controls,source))
    }
    pub(crate) fn prepare_session_embedded_neural_bank<D>(
        session:&ReplicatedTextSession<A,MlxNeuralBackend,Self,D>,
        source:Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        recipe:&crate::backend::nn::workspace::EmbeddedEquationRecipe,
        role:eredu_runtime::working_memory::OriginalEmbeddedSpeculativeRole,
        scope:&safemlx::SubmissionScope,
        partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    )->Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>,Error>
    where D:eredu_runtime::ReplicatedTextExecutionStrategy<A,MlxNeuralBackend,S,
        MlxArchitectureLayerwisePolicy<A,S>,MlxArchitectureLayerwisePolicy<A,S>>,
    {
        Self::session_speculative_neural_plan(session,recipe.workspace().geometry(),source)?
            .prepare_embedded(recipe,role,scope,partition)
    }

    pub(crate) fn bind_session_external_neural_recipe<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
    ) -> Result<(u64, Option<eredu_runtime::working_memory::HostSourceConstructionFacts>), Error>
    where D: eredu_runtime::ReplicatedTextExecutionStrategy<A, MlxNeuralBackend, S,
        MlxArchitectureLayerwisePolicy<A, S>, MlxArchitectureLayerwisePolicy<A, S>>,
    {
        let plan = Self::session_speculative_neural_plan(session, recipe.plan().geometry(), source)?
            .prepare_speculative_source(pool, funding)?;
        plan.bind_external_recipe(recipe)?;
        let source = plan.speculative_source_facts()?;
        plan.external_control_bytes(recipe).map(|controls| (controls, source))
    }
    pub(crate) fn prepare_session_external_neural_bank<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        recipe: &crate::backend::nn::workspace::ResidentNativeRecipe,
        invocation: ExternalInvocation,
        role: OriginalExternalSpeculativeRole,
        scope: &safemlx::SubmissionScope,
        partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>, Error>
    where D: eredu_runtime::ReplicatedTextExecutionStrategy<A, MlxNeuralBackend, S,
        MlxArchitectureLayerwisePolicy<A, S>, MlxArchitectureLayerwisePolicy<A, S>>,
    {
        Self::session_speculative_neural_plan(session, recipe.plan().geometry(), source)?
            .prepare_external(recipe, invocation, role, scope,partition)
    }

    pub(crate) fn prepare_session_speculative_span_neural_bank<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        scope: &safemlx::SubmissionScope,
        partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A, MlxNeuralBackend, S, MlxArchitectureLayerwisePolicy<A, S>, MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        Self::session_speculative_neural_plan(session, recipe.plan().geometry(), source)?
            .prepare_speculative_span(recipe, span, scope,partition)
    }

    pub(crate) fn session_prefill_control_facts<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        geometry: eredu_core::InferenceGeometry,
        graph_capacity: std::num::NonZeroU64,
        native_root_capacity: Option<u64>,
        retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        native_recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Result<Option<eredu_runtime::working_memory::TextPrefillScopeFacts>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        // Native parallel completion/control facts are supplied by the exact
        // retained per-equation recipe, not by the presence of a topology.
        let parallel_source=native_recipe.map(|recipe|recipe.parallel_control_source()).transpose()?.flatten();
        let selected = session
            .inspect_runtime_execution_fixed(|mechanisms, state, runtime| {
                if mechanisms.parallel_layout.is_some() && parallel_source.is_none() {
                    return Ok(None);
                }
                // Exact native recipe roots replace only the root population.
                // Bounded source/materialization controls still come from the
                // selected policy below, including any unresolved contribution.
                let capacity = if let Some(capacity) = native_root_capacity {
                    capacity
                } else {
                    let Some(counts) = state.retained_owner_slot_counts() else {
                        return Ok(None);
                    };
                    if counts.manager_roles != 0 {
                        return Ok(None);
                    }
                    let minimum = u64::try_from(
                        safemlx::PrefillRoots::layout(0)
                            .map_err(|e| Error::PrefillRoots(e.into()))?
                            .validation_descriptor_minimum,
                    )
                    .map_err(|_| {
                        Error::PrefillControl(
                            eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                        )
                    })?;
                    if minimum == 0 {
                        return Err(Error::PrefillControl(
                            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                        ));
                    }
                    // Original validation append checks arena membership and deduplicates
                    // descriptor aliases. Other arbitrary roots do not claim this bound.
                    let validations = graph_capacity.get() / minimum;
                    u64::try_from(counts.arrays)
                        .ok()
                        .and_then(|n| {
                            n.checked_add(u64::from(
                                geometry.output != eredu_core::OutputDemand::StateOnly,
                            ))
                        })
                        .and_then(|n| n.checked_add(validations))
                        .ok_or(Error::PrefillControl(
                            eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                        ))?
                };
                // This private quote is text-only. Prepared-media future/cut roots
                // require their source-derived contribution in complete request D.
                let operations = native_recipe
                    .and_then(|_| D::resident_policy(runtime))
                    .or_else(|| D::bounded_policy(runtime))
                    .map(|policy| {
                        policy
                            .original_operation_plan(geometry, retained_sources)?
                            .with_selected_stream(&mechanisms.stream)
                    })
                    .transpose()?;
                Ok(Some((
                    capacity,
                    operations,
                    mechanisms
                        .gguf_host_runtime
                        .as_ref()
                        .map(std::rc::Rc::clone)
                        .map_err(|cause| cause.retained()),
                )))
            })
            .map_err(Error::RuntimeInspection)??;
        let Some((capacity, operations, gguf_host_runtime)) = selected else {
            return Ok(None);
        };
        let operations = operations
            .map(|plan| {
                plan.with_preparation_funding(funding).with_foreground_disk_reads(pool)?
                    .with_neural_recipe(native_recipe)?
                    .with_gguf_host_runtime(&gguf_host_runtime)?
                    .with_source_arena_plan()
            })
            .transpose()?;
        let bytes = operations
            .as_ref()
            .map_or(Some(0), |plan| plan.control_bytes());
        let source = operations
            .as_ref()
            .and_then(|plan| plan.source_construction_facts());
        crate::backend::submission_recovery::prefill::facts(geometry, capacity)?
            .with_operation_controls(bytes)
            .map(|facts| {
                Some(
                    facts
                        .with_source_constructions(source)
                        .with_host_destinations(
                            operations
                                .as_ref()
                                .and_then(|plan| plan.host_destination_facts()),
                        ),
                )
            })
            .map_err(Error::PrefillControl)
    }
    pub(crate) fn prepare_session_original_operations<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        original: &eredu_runtime::working_memory::OriginalTextPrefillScopeSet,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        registration: crate::backend::runtime::execution::generic::OriginalOperationRegistration,
        controls: eredu_runtime::working_memory::OriginalTextControlGuard,
        host_destinations: Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
        retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        native_recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Result<
        Option<crate::backend::runtime::execution::generic::OriginalOperationBankOwner>,
        Error,
    >
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        original
            .validate_request(step.request())
            .map_err(Error::PrefillControl)?;
        original
            .validate_execution(session.inference_execution_identity())
            .map_err(Error::PrefillControl)?;
        let geometry = original.facts().plan().geometry();
        let (plan, gguf_host_runtime) = session
            .inspect_runtime_execution_fixed(|mechanisms, _, runtime| {
                let plan = native_recipe
                    .and_then(|_| D::resident_policy(runtime))
                    .or_else(|| D::bounded_policy(runtime))
                    .map(|policy| {
                        policy
                            .original_operation_plan(geometry, retained_sources)?
                            .with_selected_stream(&mechanisms.stream)
                    })
                    .transpose()?;
                Ok::<_, Error>((
                    plan,
                    mechanisms
                        .gguf_host_runtime
                        .as_ref()
                        .map(std::rc::Rc::clone)
                        .map_err(|cause| cause.retained()),
                ))
            })
            .map_err(Error::RuntimeInspection)??;
        // The native inspector, policy and state loans have all ended.
        let plan = plan
            .map(|plan| {
                plan.with_preparation_funding(funding).with_foreground_disk_reads(pool)?
                    .with_neural_recipe(native_recipe)?
                    .with_gguf_host_runtime(&gguf_host_runtime)?
                    .with_source_arena_plan()
            })
            .transpose()?;
        match plan {
            Some(plan) => {
                plan.prepare_install(original, step, registration, controls, host_destinations)
            }
            None if original.facts().operation_control_bytes() == Some(0)
                && host_destinations.is_none() =>
            {
                Ok(None)
            }
            None => Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )),
        }
    }
    pub(crate) fn session_native_storage<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
    ) -> Result<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        session.inspect_runtime_execution_fixed(|mechanisms, _, _| {
            Ok(crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage::new(
                &mechanisms.gguf_host_runtime, &mechanisms.native_storage_selection,
            ))
        }).map_err(Error::RuntimeInspection)?
    }
    pub(crate) fn session_prefill_roots_runtime<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
    ) -> Result<safemlx::PrefillRootsRuntime, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        // Clone only the completed ordinary initialization witness. No runtime
        // entry, handler initialization or allocation occurs under the inspector.
        session
            .inspect_runtime_execution_fixed(|mechanisms, _, _| {
                Ok(mechanisms.prefill_roots_runtime.clone())
            })
            .map_err(Error::RuntimeInspection)?
    }
    pub(crate) fn install_session_parallel_control<D>(
        session:&ReplicatedTextSession<A,MlxNeuralBackend,Self,D>,
        control:Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection>,
    )->Result<(),Error>
    where D:eredu_runtime::ReplicatedTextExecutionStrategy<A,MlxNeuralBackend,S,
        MlxArchitectureLayerwisePolicy<A,S>,MlxArchitectureLayerwisePolicy<A,S>>, {
        if let Some(control)=&control{control.validate_execution(session.inference_execution_identity())?;}
        let previous=session.inspect_runtime_execution_fixed(|mechanisms,_,_|{
            let mut slot=mechanisms.parallel_control.try_borrow_mut().map_err(|_|Error::PrefillScopeReentrant)?;
            Ok::<_,Error>(std::mem::replace(&mut *slot,control))
        }).map_err(Error::RuntimeInspection)??;
        drop(previous);
        Ok(())
    }

    pub(crate) fn install_session_prefill_controls<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        controls: Option<crate::backend::submission_recovery::prefill::PrefillControlProjection>,
    ) -> Result<(), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        if let Some(controls) = &controls {
            controls.validate_execution(session.inference_execution_identity())?;
        }
        let expired = session
            .inspect_runtime_execution_fixed(|mechanisms, _, _| {
                let mut slot = mechanisms
                    .prefill_controls
                    .try_borrow_mut()
                    .map_err(|_| Error::PrefillScopeReentrant)?;
                Ok::<_, Error>(std::mem::replace(&mut *slot, controls))
            })
            .map_err(Error::RuntimeInspection)??;
        // Both session and slot loans ended before potentially final weak/Q drop.
        drop(expired);
        Ok(())
    }
    pub(crate) fn retire_session_prefill_controls<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
    ) -> Result<(), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        let expired = session
            .inspect_runtime_execution_fixed(|mechanisms, _, _| {
                let mut slot = mechanisms
                    .prefill_controls
                    .try_borrow_mut()
                    .map_err(|_| Error::PrefillScopeReentrant)?;
                Ok::<_, Error>(if slot.as_ref().is_some_and(|view| !view.is_live()) {
                    slot.take()
                } else {
                    None
                })
            })
            .map_err(Error::RuntimeInspection)??;
        // Removing an expired weak view is housekeeping, not completion proof.
        // Native source/root owners were independently retired by Recovery.
        drop(expired);
        Ok(())
    }
    /// Tests the actual selected callback paths while their slot is borrowed.
    #[cfg(test)]
    pub(crate) fn inspection_adapters_for_test<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        session
            .inspect_runtime_execution_fixed(|_, _, runtime| {
                if let Some(policy) = D::bounded_policy(runtime) {
                    policy.inspect_operation_sources_for_test(geometry);
                }
                Ok::<_, Error>(())
            })
            .map_err(Error::RuntimeInspection)??;
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let graph = std::num::NonZeroU64::new(4 << 20).unwrap();
        assert!(Self::session_prefill_control_facts(
            session, &pool, geometry, graph, None, None, None, None
        )?
        .is_some());
        let overflow = Self::session_prefill_control_facts(
            session,
            &pool,
            eredu_core::InferenceGeometry {
                input_positions: u64::MAX / 2,
                prefill_chunk_positions: 1,
                ..geometry
            },
            graph,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            overflow,
            Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
        ));
        assert!(std::error::Error::source(&overflow)
            .unwrap()
            .is::<eredu_runtime::working_memory::WorkingMemoryError>());
        let invalid = Self::session_prefill_control_facts(
            session,
            &pool,
            eredu_core::InferenceGeometry {
                batch_size: 0,
                ..geometry
            },
            graph,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            invalid,
            Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch
            )
        ));
        let witness = Self::session_prefill_roots_runtime(session)?;
        drop(witness); // completed ordinary initialization witness, no new prepare
        session
            .inspect_runtime_execution_fixed(|mechanisms, _, _| {
                let slot = mechanisms.prefill_controls.borrow_mut();
                assert!(matches!(
                    Self::install_session_prefill_controls(session, None),
                    Err(Error::PrefillScopeReentrant)
                ));
                assert!(matches!(
                    Self::retire_session_prefill_controls(session),
                    Err(Error::PrefillScopeReentrant)
                ));
                drop(slot);
                Ok::<_, Error>(())
            })
            .map_err(Error::RuntimeInspection)??;
        Self::install_session_prefill_controls(session, None)?;
        Self::retire_session_prefill_controls(session)
    }
    #[cfg(test)]
    pub(crate) fn prefill_status_for_test<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
    ) -> Result<(bool, bool), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    {
        session
            .inspect_runtime_execution_fixed(|mechanisms, _, _| {
                let slot = mechanisms
                    .prefill_controls
                    .try_borrow()
                    .map_err(|_| Error::PrefillScopeReentrant)?;
                Ok((slot.is_some(), slot.as_ref().is_some_and(|v| v.is_live())))
            })
            .map_err(Error::RuntimeInspection)?
    }
}
