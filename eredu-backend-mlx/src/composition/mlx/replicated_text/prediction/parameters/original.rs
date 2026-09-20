//! Actual supplementary modules projected into one accepted occurrence bank.
use super::*;
use crate::backend::runtime::execution::generic::{
    PredictionModuleCall, PredictionModulePlan, PredictionModuleProjection,
};
use eredu_architectures::prediction_extension::{
    PredictionInvocation, PreparedPredictionInvocationRoots,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

fn identity() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}

pub(crate) fn inspect_prediction_module_plan<A, P>(
    extension: &P,
    parameters: &PredictionParameterStorage,
    validations: usize,
    stream: &Stream,
    pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    context: &WorkspaceContext,
) -> Result<Option<PredictionModulePlan>, Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Sources<'a> {
        manager: Option<ResidencyManager>,
        ordinals: Vec<(usize, usize)>,
        context: &'a WorkspaceContext,
    }
    impl PredictionResourceVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        for Sources<'_>
    {
        type Error = Error;
        fn module<U: Parameterized<MlxTensor>>(
            &mut self,
            _physical: usize,
            module: &MlxPredictionModule<U>,
        ) -> Result<(), Error> {
            let manager = module.manager.get().ok_or_else(|| identity().at_speculative_stage("prediction module manager"))?;
            let source = manager
                .supplementary_residency_source()
                .ok_or_else(|| identity().at_speculative_stage("prediction supplementary source"))?;
            if let Some(previous) = &self.manager {
                previous
                    .validate_supplementary_source(source)
                    .map_err(|_| identity().at_speculative_stage("prediction supplementary manager identity"))?;
            } else {
                self.manager = Some(manager.clone());
            }
            let ordinal = source
                .ordinal(module.id.as_ref().ok_or_else(|| identity().at_speculative_stage("prediction module physical identity"))?)
                .ok_or_else(|| identity().at_speculative_stage("prediction module source ordinal"))?;
            self.context
                .reserve_metadata_vec(&mut self.ordinals, 1)
                .map_err(Error::from)?;
            let copies = module.placeholders.len().checked_add(module.replacements.len())
                .ok_or_else(overflow)?;
            self.ordinals.push((ordinal, copies));
            Ok(())
        }
        fn pooling_state(
            &mut self,
            _: &OwnedPredictionCache<
                crate::backend::runtime::cache::state::MlxPoolingAttentionCache,
            >,
        ) -> Result<(), Error> {
            Ok(())
        }
        fn model_state(
            &mut self,
            _: &crate::backend::runtime::cache::state::MlxHybridState,
        ) -> Result<(), Error> {
            Ok(())
        }
    }
    let frames = [
        size_of::<Sources<'_>>(),
        size_of::<PredictionModuleCall>(),
        size_of::<Option<PredictionModulePlan>>(),
        size_of::<Result<Option<PredictionModulePlan>, Error>>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
    let mut source = Sources {
        manager: None,
        ordinals: Vec::new(),
        context,
    };
    extension.visit_retained_resources(&mut source)?;
    let calls = parameters
        .invocations()
        .with_completed(|calls| {
            let mut out = context.metadata_vec(calls.len())?;
            for call in calls {
                out.push(PredictionModuleCall {
                    source_ordinal: source.ordinals.get(call.module)
                        .ok_or(WorkspaceMetadataError::Unqualified)?.0,
                    retained_parameter_copies: source.ordinals.get(call.module)
                        .ok_or(WorkspaceMetadataError::Unqualified)?.1,
                    retained_roots: call
                        .retained_roots
                        .ok_or(WorkspaceMetadataError::Unqualified)?,
                    completion: call.completion.ok_or(WorkspaceMetadataError::Unqualified)?,
                });
            }
            Ok(out)
        })
        .map_err(Error::from)?;
    if calls.is_empty() {
        return Ok(None);
    }
    let manager = source.manager.ok_or_else(identity)?;
    let retained = manager
        .supplementary_residency_source()
        .ok_or_else(identity)?;
    PredictionModulePlan::inspect(
        &manager,
        retained,
        calls,
        validations,
        stream,
        pool,
        context,
    )
    .map(Some)
}

pub(crate) fn install_prediction_module_bank<A, P>(
    extension: &mut P,
    projection: &PredictionModuleProjection,
) -> Result<(), Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Install<'a>(&'a PredictionModuleProjection);
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Install<'_> {
        type Error = Error;
        fn visit<U: Parameterized<MlxTensor>>(
            &mut self,
            _: usize,
            module: &mut MlxPredictionModule<U>,
        ) -> Result<(), Error> {
            if module.original.is_some() {
                return Err(identity());
            }
            module.original = Some(self.0.clone());
            Ok(())
        }
    }
    extension.visit_modules(&mut Install(projection))
}
pub(crate) fn clear_prediction_module_bank<A, P>(
    extension: &mut P,
    projection: &PredictionModuleProjection,
) where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    struct Clear<'a>(&'a PredictionModuleProjection);
    impl PredictionModuleVisitor<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> for Clear<'_> {
        type Error = Infallible;
        fn visit<U: Parameterized<MlxTensor>>(
            &mut self,
            _: usize,
            module: &mut MlxPredictionModule<U>,
        ) -> Result<(), Infallible> {
            if module
                .original
                .as_ref()
                .is_some_and(|value| value.same_bank(self.0))
            {
                module.original = None;
            }
            Ok(())
        }
    }
    match extension.visit_modules(&mut Clear(projection)) {
        Ok(()) => {}
        Err(never) => match never {},
    }
}

impl<U: Parameterized<MlxTensor>> MlxPredictionModule<U> {
    pub(in crate::composition::mlx::replicated_text) fn invoke_with_roots<O>(
        &mut self,
        stream: &Stream,
        operation: impl FnOnce(
            &mut U,
            Option<&mut dyn PreparedPredictionInvocationRoots<MlxTensor>>,
        ) -> PredictionInvocation<MlxTensor, O>,
    ) -> Result<O, eredu_nn::Error> {
        if self.original.is_none() {
            if safemlx::OriginalScopeObserver::try_current()
                .map_err(eredu_nn::Error::backend_retained_source)?
                .is_some()
            {
                return Err(WorkspaceMetadataError::Unqualified.into());
            }
            return self
                .invoke(stream, |inner| {
                    let (outcome, roots) = operation(inner, None).into_parts();
                    (outcome.map_err(Error::from), roots)
                })
                .map_err(|cause| match cause {
                    Error::Neural(cause) => cause,
                    cause => eredu_nn::Error::backend_retained_source(cause),
                });
        }
        let projection = self
            .original
            .as_ref()
            .ok_or(WorkspaceMetadataError::Unqualified)?
            .clone();
        let manager = self
            .manager
            .get()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let id = self
            .id
            .as_ref()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let replacements = &self.replacements;
        let placeholders = &self.placeholders;
        let should_evict = !self.residency.is_fully_resident();
        let inner = &mut self.inner;
        projection.invoke(manager,id,stream,should_evict,|lease,rows,roots| {
            let result=crate::backend::runtime::checkpoint::binding::populate_module_from_original_lease_excluding(
                inner,lease,rows,|name|replacements.contains_key(name));
            let (outcome,retained)=match result {
                Ok(())=>{
                    match publish_original(inner,replacements,rows,roots) {
                        Ok(())=>{let (outcome,retained)=operation(inner,Some(&mut *roots)).into_parts();(outcome.map_err(Error::from),retained)}
                        Err(cause)=>(Err(cause),Vec::new()),
                    }
                }
                Err(cause)=>(Err(cause.into()),Vec::new()),
            };
            let restored=publish_original(inner,placeholders,rows,roots);
            (outcome.and_then(|value|restored.map(|_|value)),retained)
        }).map_err(|cause|match cause {Error::Neural(cause)=>cause,cause=>eredu_nn::Error::backend_retained_source(cause)})
    }
}

/// Same named slot replacement with fallible native handle sharing. An original
/// failure never escapes through the ordinary infallible Array::clone panic.
fn publish_original<U: Parameterized<MlxTensor>>(
    module: &mut U,
    values: &BTreeMap<String, MlxTensor>,
    binding_rows: usize,
    source: &dyn PreparedPredictionInvocationRoots<MlxTensor>,
) -> Result<(), Error> {
    struct PublishOriginal<'a> {
        values: &'a BTreeMap<String, MlxTensor>,
        failure: Option<safemlx::error::Exception>,
    }
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for PublishOriginal<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut MlxTensor) {
            if self.failure.is_some() {
                return;
            }
            if let Some(replacement) = self.values.get(metadata.id().as_str()) {
                match replacement.as_ref().try_clone_handle() {
                    Ok(replacement) => *value = MlxTensor::from(replacement),
                    Err(cause) => self.failure = Some(cause),
                }
            }
        }
    }
    let frames = [
        size_of::<PublishOriginal<'_>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<ParameterMetadata>(),
        size_of::<Option<safemlx::error::Exception>>(),
    ];
    let controls = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .and_then(|n| {
            n.checked_add(
                safemlx::Array::inspection_clone_handle_bytes().checked_mul(values.len())?,
            )
        })
        .and_then(|n| {
            n.checked_add(MlxNeuralBackend::prepared_binding_visit_control_bytes(
                binding_rows,
            )?)
        })
        .ok_or_else(overflow)?;
    source.controls(controls).map_err(Error::from)?;
    let mut publish = PublishOriginal {
        values,
        failure: None,
    };
    module.visit_parameters_mut(&mut publish);
    match publish.failure {
        Some(cause) => Err(cause.into()),
        None => Ok(()),
    }
}

pub(in crate::composition::mlx::replicated_text) fn invoke_shared<U, V, O, F>(
    module: &mut MlxPredictionModule<U>,
    shared: Option<&mut MlxPredictionModule<V>>,
    stream: &Stream,
    operation: F,
) -> Result<O, eredu_nn::Error>
where
    U: Parameterized<MlxTensor>,
    V: Parameterized<MlxTensor>,
    F: FnOnce(
        &mut U,
        Option<&mut V>,
        Option<&mut dyn PreparedPredictionInvocationRoots<MlxTensor>>,
    ) -> PredictionInvocation<MlxTensor, O>,
{
    module.invoke_with_roots(stream, |module, source| {
        let Some(source) = source else {
            return PredictionInvocation::from_prepared_parts(
                Err(WorkspaceMetadataError::Unqualified.into()),
                Vec::new(),
            );
        };
        let Some(shared) = shared else {
            return operation(module, None, Some(source));
        };
        let frames = [
            size_of::<F>(),
            size_of::<O>(),
            size_of::<Result<O, eredu_nn::Error>>(),
            size_of::<Result<Vec<MlxTensor>, eredu_nn::Error>>(),
            size_of::<PredictionInvocation<MlxTensor, O>>(),
        ];
        let paid = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)
            .map_err(eredu_nn::Error::from)
            .and_then(|bytes| source.controls(bytes));
        if let Err(cause) = paid {
            return PredictionInvocation::from_prepared_parts(Err(cause), Vec::new());
        }
        let mut retained = Ok(Vec::new());
        let outcome = shared.invoke_with_roots(stream, |shared, shared_source| {
            let invocation = operation(module, Some(shared), shared_source);
            retained = source.retain(&mut |visit| {
                for value in invocation.retained_values() {
                    visit(value);
                }
            });
            invocation
        });
        match retained {
            Ok(retained) => PredictionInvocation::from_prepared_parts(outcome, retained),
            Err(cause) => PredictionInvocation::from_prepared_parts(
                outcome.and_then(|_| Err(cause)),
                Vec::new(),
            ),
        }
    })
}
