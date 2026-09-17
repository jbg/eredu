mod evidence;
pub(crate) use evidence::{ExternalCompletedSource, retain_external_evidence,retain_external_evidence_for_roots,retain_external_evidence_for_placement};
use super::*;
mod original;
mod source;
pub(crate) mod workspace;
mod phase;
pub(crate) use phase::execute as execute_original_assistant;
pub(crate) use source::MlxExternalAssistantSource;
use original::DrafterRecovery;
use crate::backend::ordinary_retirement::{self, OrdinaryRetirement};
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// Reports only generic mechanisms available to neutral speculative selection.
///
/// Architecture identity, proposal mode, capture paths, and assistant family
/// never enter this report.
pub(crate) fn speculative_mechanism_capabilities() -> SpeculativeMechanismCapabilities {
    SpeculativeMechanismCapabilities::new([
        SpeculativeMechanism::TensorOperations,
        SpeculativeMechanism::NeuralOperations,
        SpeculativeMechanism::GroupedNeuralOperations,
        SpeculativeMechanism::HyperNeuralOperations,
        SpeculativeMechanism::PayloadMaterialization,
        SpeculativeMechanism::LogitsProcessing,
        SpeculativeMechanism::Sampling,
        SpeculativeMechanism::Randomness,
        SpeculativeMechanism::StateStorage,
        SpeculativeMechanism::StorageResidency,
        SpeculativeMechanism::ExactCompletion,
        SpeculativeMechanism::Observation,
        SpeculativeMechanism::Timing,
        SpeculativeMechanism::QueueBinding,
        SpeculativeMechanism::Communication,
        SpeculativeMechanism::Agreement,
        SpeculativeMechanism::Publication,
        SpeculativeMechanism::SameDeviceHandoff,
        SpeculativeMechanism::CrossDeviceTransfer,
    ])
}

pub(crate) struct MlxExternalAssistant<A: eredu_architectures::ExternalAssistantArchitecture> {
    pub(crate) config: A::Config,
    pub(crate) module: MlxModule<A::Module<MlxNeuralBackend>>,
    // Exact cold source and the actual populated values; no re-discovery or
    // module/configuration reconstruction is needed by a later managed quote.
    pub(crate) source: MlxExternalAssistantSource<A>,
    pub(crate) observers: eredu_architectures::external_assistant::ExternalAssistantObservers<
        MlxTensor,
        Array,
        Exception,
    >,
}

pub(crate) struct MlxAssistantPreparationVisitor {
    pub(super) stream: Stream,
    pub(super) weights_stream: Stream,
    pub(super) source_pool: Option<eredu_runtime::working_memory::WorkingMemoryPool>,
}

#[cfg(test)]
impl MlxAssistantPreparationVisitor {
    /// Uses the same ordinary exact lowering predicates in native source fixtures.
    pub(crate) fn lowering_for_native_source_test(
        descriptor: &eredu_runtime::WeightLoweringDescriptor,
        transforms: bool,
    ) -> Option<eredu_runtime::WeightLoweringKind> {
        if transforms && super::super::replicated_text::supports_transform(descriptor) {
            Some(eredu_runtime::WeightLoweringKind::Transform)
        } else if !transforms && super::super::replicated_text::supports_direct(descriptor) {
            Some(eredu_runtime::WeightLoweringKind::Direct)
        } else {
            None
        }
    }

    /// Retains ordinary source registration without exposing loader fields.
    pub(crate) fn for_native_source_test(
        stream: &Stream,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Self {
        Self { stream: stream.clone(), weights_stream: stream.clone(), source_pool: Some(pool.clone()) }
    }
}

impl eredu_architectures::ExternalAssistantPreparationVisitor for MlxAssistantPreparationVisitor {
    type Output<A: eredu_architectures::ExternalAssistantArchitecture> = MlxExternalAssistant<A>;
    type Error = Error;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    ) -> Result<Self::Output<A>, Self::Error> {
        materialize_external_assistant::<A>(prepared, &self.stream, &self.weights_stream, self.source_pool.as_ref())
    }
}

fn materialize_external_assistant<A: eredu_architectures::ExternalAssistantArchitecture>(
    prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    stream: &Stream,
    weights_stream: &Stream,
    source_pool: Option<&eredu_runtime::working_memory::WorkingMemoryPool>,
) -> Result<MlxExternalAssistant<A>, Error> {
    use crate::backend::runtime::{
        checkpoint::binding::{
            build_mlx_exact_replicated_text_bindings, materialize_module_bindings,
            populate_module_from_arrays_excluding,
        },
        execution::layerwise::quantize_exact_replicated_text_tasks,
    };
    let (store, checkpoint, artifact_identity, source_config, config, tasks) =
        prepared.into_parts();
    let mut store = store;
    let transformed = tasks
        .iter()
        .filter(|task| {
            matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            )
        })
        .collect::<Vec<_>>();
    if !transformed.is_empty() {
        let source = A::module::<MlxNeuralBackend>(source_config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let target = A::module::<MlxNeuralBackend>(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut groups = Vec::<(eredu_checkpoint::WeightQuantization, Vec<_>)>::new();
        for task in transformed {
            let format = task.executable().weight_quantization().ok_or_else(|| {
                Error::Quantization(format!(
                    "selected external assistant transform {:?} has no packed format",
                    task.name()
                ))
            })?;
            if let Some((_, grouped)) = groups.iter_mut().find(|(selected, _)| *selected == format)
            {
                grouped.push(task);
            } else {
                groups.push((format, vec![task]));
            }
        }
        for (format, grouped) in groups {
            store = quantize_exact_replicated_text_tasks(
                store,
                &source,
                &target,
                &[] as &[A::Module<MlxNeuralBackend>],
                &[],
                None,
                format,
                &grouped,
                stream,
            )?
            .0;
        }
    }
    let mut module = MlxModule::new(
        A::module::<MlxNeuralBackend>(config.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
    );
    let task_refs = tasks.iter().collect::<Vec<_>>();
    let bindings = build_mlx_exact_replicated_text_bindings(
        &module,
        store.as_ref(),
        &task_refs,
        &std::collections::BTreeSet::new(),
        None,
    )?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    // The materialization event settles native work; each array must also cross
    // its ordinary completion boundary before its descriptor is published as a
    // cold source. This remains inside the loading recovery owner and cannot
    // make a later workspace inspection evaluate an unfinished parameter.
    for array in arrays.values() {
        array.evaluated()?;
    }
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    // This is ordinary loading, before a request can quote these parameters.
    // Register their actual completed backing and the retained checkpoint source
    // in the selected backend pool. Later managed quotations only pin these
    // existing owners; they never invent source credit or materialize a weight.
    let registered_storage = source_pool.map(|pool| {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        for array in arrays.values() {
            storage.include_array(array)?;
        }
        storage.include_sources(store.source_storage()?)?;
        storage.register(pool)
    }).transpose()?;
    Ok(MlxExternalAssistant {
        config,
        module,
        source: MlxExternalAssistantSource::new(
            store, checkpoint, artifact_identity, source_config, tasks, bindings, arrays, registered_storage,
        ),
        observers: Default::default(),
    })
}

/// Architecture-dispatched MLX draft model with its fixed execution placement.
pub struct MlxDrafter {
    payload: Rc<OrdinaryRetirement<DrafterPayload>>,
    poisoned: Rc<Cell<bool>>,
}

struct DrafterPayload {
    execution: DrafterExecution,
    stream: Stream,
    // Model and raw stream aliases retire before their admitted placement.
    // Target placement continues borrowing the loaded target's backend.
    backend: Option<crate::backend::MlxBackend<'static>>,
}

enum DrafterExecution {
    Assistant(
        eredu_architectures::MaterializedExternalAssistantExecution<MlxAssistantPreparationVisitor>,
    ),
    Autoregressive {
        model: crate::backend::MlxModel,
        selected: SelectedSpeculativeRealization,
        tokenizer: TokenizerCompatibilityProof,
    },
}
impl DrafterExecution {
    fn selected(&self) -> &SelectedSpeculativeRealization {
        match self {
            Self::Assistant(a) => a.selected(),
            Self::Autoregressive { selected, .. } => selected,
        }
    }
    fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        match self {
            Self::Assistant(a) => a.tokenizer_compatibility(),
            Self::Autoregressive { tokenizer, .. } => *tokenizer,
        }
    }
    fn visit<
        W: eredu_architectures::MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor>,
    >(
        &mut self,
        visitor: W,
    ) -> W::Output {
        match self {
            Self::Assistant(a) => a.visit(visitor),
            Self::Autoregressive { .. } => {
                unreachable!("independent draft uses its selected ordinary executor")
            }
        }
    }
    fn capture(
        &self,
    ) -> &eredu_architectures::composite_execution::ExternalPredictionCaptureRequest {
        match self {
            Self::Assistant(a) => a.capture(),
            Self::Autoregressive { .. } => {
                unreachable!("independent draft consumes no target features")
            }
        }
    }
}

struct DrafterRetention {
    _payload: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload>>>>>,
    poisoned: Rc<Cell<bool>>,
    // During loading this owns the actual placement before a payload exists.
    // On success it moves into that payload; unresolved construction retains it.
    backend: Option<crate::backend::MlxBackend<'static>>,
}

impl Retention for DrafterRetention {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.poisoned.set(true);
        }
    }
}

struct DrafterOperation<'a> {
    drafter: &'a mut MlxDrafter,
    retained: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload>>>>>,
    recovery: DrafterRecovery,
    completed: bool,
}

impl DrafterOperation<'_> {
    fn payload(&mut self) -> &mut DrafterPayload {
        Rc::get_mut(&mut self.drafter.payload).expect("healthy drafter owns its native payload")
    }

    fn finish<T>(mut self, result: Result<T, Error>) -> Result<T, Error> {
        self.retained
            .replace(Some(Rc::clone(&self.drafter.payload)));
        self.recovery.seal();
        // One nonblocking poll may leave healthy CPU work or native recovery
        // records pending. A successful synchronous operation must wait for
        // completion; failures retain their nonblocking recovery path.
        let status = if result.is_ok() {
            self.recovery.wait()
        } else {
            self.recovery.progress()
        };
        if !status.settled || status.failed || status.blocked {
            self.drafter.poisoned.set(true);
            if let Some(funding) = self.recovery.funding() {
                return Err(original::failure(
                    original::Cause::Unresolved(result.err()), funding.clone(),
                ));
            }
            let message = match &result {
                Ok(_) => "native drafter work failed or remains unresolved".to_owned(),
                Err(error) => format!("native drafter work failed or remains unresolved: {error}"),
            };
            return Err(Error::Speculative(message));
        }
        let value = match result {
            Ok(value) => value,
            Err(cause) => return Err(match self.recovery.funding() {
                Some(funding) => original::failure(original::Cause::Operation(cause), funding.clone()),
                None => cause,
            }),
        };
        self.completed = true;
        self.retained.take();
        Ok(value)
    }
}

impl Drop for DrafterOperation<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        self.drafter.poisoned.set(true);
        self.retained
            .replace(Some(Rc::clone(&self.drafter.payload)));
        self.recovery.seal();
        let status = self.recovery.progress();
        if status.failed || status.blocked || std::thread::panicking() {
            self.drafter.poisoned.set(true);
        }
    }
}

impl MlxDrafter {
    fn begin_operation(&mut self) -> Result<DrafterOperation<'_>, Error> {
        crate::backend::submission_recovery::reap();
        ordinary_retirement::reclaim();
        if self.poisoned.get() {
            return Err(Error::Speculative(
                "native drafter is poisoned by failed or unresolved work".into(),
            ));
        }
        // A healthy terminal scope can still retain its payload while another
        // thread owns the native runtime. Terminal evidence is not exclusive
        // mutation authority: reject until retirement releases that owner.
        if Rc::get_mut(&mut self.payload).is_none() {
            return Err(Error::Speculative(
                "native drafter payload is still retained by prior work".into(),
            ));
        }
        let retained = Rc::new(RefCell::new(None));
        let recovery = Recovery::begin(DrafterRetention {
            _payload: Rc::clone(&retained),
            poisoned: Rc::clone(&self.poisoned),
            backend: None,
        })?;
        Ok(DrafterOperation {
            drafter: self,
            retained,
            recovery: DrafterRecovery::Ordinary(recovery),
            completed: false,
        })
    }

    /// Installs typed production observers for external-assistant tensors and logits.
    pub fn install_external_observers<TensorObserver, LogitsObserver>(
        &mut self,
        tensors: TensorObserver,
        logits: LogitsObserver,
    ) -> Result<(), Error>
    where
        TensorObserver: eredu_runtime::ActivationObserver<MlxTensor, Exception> + 'static,
        LogitsObserver: eredu_runtime::ActivationObserver<Array, Exception> + 'static,
    {
        if self.is_autoregressive() {
            return Err(Error::Speculative("independent drafts expose ordinary model observations, not target-feature assistant observations".into()));
        }
        let mut operation = self.begin_operation()?;
        operation
            .payload()
            .execution
            .visit(InstallExternalObservers {
                observers: Some(
                    eredu_architectures::external_assistant::ExternalAssistantObservers::new(
                        tensors, logits,
                    ),
                ),
            });
        operation.finish(Ok(()))
    }

    /// Materializes an architecture-inspected drafter with proven tokenizer compatibility.
    pub(crate) fn materialize(
        preparation: eredu_architectures::PreparedExternalDraft,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::materialize_impl(preparation, None, stream, weights_stream, None)
    }

    /// Same loader with the selected backend's actual source accounting pool.
    /// Source preparation precedes the ordinary drafter recovery and native
    /// model construction, matching the target model's provider handoff.
    pub(crate) fn materialize_with_source_pool(
        preparation: eredu_architectures::PreparedExternalDraft,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::materialize_impl(preparation, Some(pool), stream, weights_stream, None)
    }

    /// Same loader, retaining the backend selected for an explicit drafter
    /// placement. Its exact source stream/worker and execution registration are
    /// used together; they are never reconstructed from the raw Stream value.
    pub(crate) fn materialize_with_backend(
        preparation: eredu_architectures::PreparedExternalDraft,
        backend: crate::backend::MlxBackend<'static>,
    ) -> Result<Self, Error> {
        let stream = backend.stream().clone();
        let weights = backend.weights_stream().clone();
        let pool = backend.memory_pool().clone();
        Self::materialize_impl(preparation, Some(&pool), &stream, &weights, Some(backend))
    }

    fn materialize_impl(
        preparation: eredu_architectures::PreparedExternalDraft,
        pool: Option<&eredu_runtime::working_memory::WorkingMemoryPool>,
        stream: &Stream,
        weights_stream: &Stream,
        backend: Option<crate::backend::MlxBackend<'static>>,
    ) -> Result<Self, Error> {
        ordinary_retirement::reclaim();
        let layerwise_manager = match (&preparation, pool) {
            (eredu_architectures::PreparedExternalDraft::Autoregressive(prepared), Some(pool)) => {
                crate::backend::runtime::execution::generic::prepare_layerwise_manager(
                    prepared.sources(),
                    pool,
                    weights_stream,
                    stream,
                )?
            }
            _ => None,
        };
        let addressable_manager = match (&preparation, pool) {
            (eredu_architectures::PreparedExternalDraft::Autoregressive(prepared), Some(pool)) =>
                crate::composition::mlx::loading::prepare_addressable_source(
                    prepared.sources(), pool, weights_stream, stream)?,
            _ => None,
        };
        let poisoned = Rc::new(Cell::new(false));
        let retained = Rc::new(RefCell::new(None));
        let mut recovery = Recovery::begin(DrafterRetention {
            _payload: Rc::clone(&retained),
            poisoned: Rc::clone(&poisoned),
            backend,
        })?;
        let execution = match preparation {
            eredu_architectures::PreparedExternalDraft::Assistant(p) => {
                DrafterExecution::Assistant(p.materialize(MlxAssistantPreparationVisitor {
                    stream: stream.clone(),
                    weights_stream: weights_stream.clone(),
                    source_pool: pool.cloned(),
                })?)
            }
            eredu_architectures::PreparedExternalDraft::Autoregressive(p) => {
                let (sources, selected, tokenizer) = p.into_parts();
                let model = crate::composition::mlx::loading::materialize_model_plan_with_layerwise_manager(
                    sources,
                    None,
                    stream,
                    weights_stream,
                    layerwise_manager,
            addressable_manager,
        )?;
                DrafterExecution::Autoregressive {
                    model,
                    selected,
                    tokenizer,
                }
            }
        };
        let payload = Rc::new(OrdinaryRetirement::new(DrafterPayload {
            execution,
            stream: stream.clone(),
            backend: recovery.retention_mut().backend.take(),
        }));
        retained.replace(Some(Rc::clone(&payload)));
        recovery.seal();
        let status = recovery.wait();
        if !status.settled || status.failed || status.blocked {
            return Err(Error::Speculative(
                "native drafter materialization failed or remains unresolved".into(),
            ));
        }
        drop(recovery);
        retained.borrow_mut().take();
        Ok(Self { payload, poisoned })
    }

    pub(crate) fn visit<W, T>(&mut self, visitor: W) -> Result<T, Error>
    where
        W: eredu_architectures::MaterializedExternalAssistantVisitor<
            MlxAssistantPreparationVisitor,
            Output = Result<T, Error>,
        >,
    {
        let mut operation = self.begin_operation()?;
        let result = operation.payload().execution.visit(visitor);
        operation.finish(result)
    }

    pub(crate) fn is_autoregressive(&self) -> bool {
        matches!(
            self.payload.execution,
            DrafterExecution::Autoregressive { .. }
        )
    }
    pub(crate) fn with_autoregressive<T>(
        &mut self,
        run: impl FnOnce(&mut crate::backend::MlxModel) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut operation = self.begin_operation()?;
        let result = match &mut operation.payload().execution {
            DrafterExecution::Autoregressive { model, .. } => run(model),
            _ => Err(Error::Speculative(
                "draft requires target-feature execution".into(),
            )),
        };
        operation.finish(result)
    }

    /// Returns the portable proof established before this assistant was materialized.
    pub fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.payload.execution.tokenizer_compatibility()
    }

    /// Execution stream selected when this drafter was loaded.
    pub fn stream(&self) -> &Stream {
        &self.payload.stream
    }

    /// Returns topology selected by the portable execution plan before queue construction.
    pub fn topology(&self) -> SpeculativeExecutionTopology {
        self.payload.execution.selected().placement().topology()
    }

    /// Returns the one neutral realization retained from preconstruction selection.
    pub fn selected(&self) -> &SelectedSpeculativeRealization {
        self.payload.execution.selected()
    }

    pub(crate) fn capture(
        &self,
    ) -> &eredu_architectures::composite_execution::ExternalPredictionCaptureRequest {
        self.payload.execution.capture()
    }
}

struct InstallExternalObservers {
    observers: Option<
        eredu_architectures::external_assistant::ExternalAssistantObservers<
            MlxTensor,
            Array,
            Exception,
        >,
    >,
}

impl eredu_architectures::MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor>
    for InstallExternalObservers
{
    type Output = ();

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        mut self,
        assistant: &mut MlxExternalAssistant<A>,
    ) -> Self::Output {
        assistant.observers = self
            .observers
            .take()
            .expect("external observers are installed exactly once");
    }
}

#[cfg(test)]
#[path = "assistant_recovery_tests.rs"]
mod recovery_tests;
