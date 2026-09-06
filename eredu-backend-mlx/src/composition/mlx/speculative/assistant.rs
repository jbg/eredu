use super::*;
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
    pub(crate) observers: eredu_architectures::external_assistant::ExternalAssistantObservers<
        MlxTensor,
        Array,
        Exception,
    >,
}

pub(crate) struct MlxAssistantPreparationVisitor {
    pub(super) stream: Stream,
    pub(super) weights_stream: Stream,
}

impl eredu_architectures::ExternalAssistantPreparationVisitor for MlxAssistantPreparationVisitor {
    type Output<A: eredu_architectures::ExternalAssistantArchitecture> = MlxExternalAssistant<A>;
    type Error = Error;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    ) -> Result<Self::Output<A>, Self::Error> {
        materialize_external_assistant::<A>(prepared, &self.stream, &self.weights_stream)
    }
}

fn materialize_external_assistant<A: eredu_architectures::ExternalAssistantArchitecture>(
    prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxExternalAssistant<A>, Error> {
    use crate::backend::runtime::{
        checkpoint::binding::{
            build_mlx_exact_replicated_text_bindings, materialize_module_bindings,
            populate_module_from_arrays_excluding,
        },
        execution::layerwise::quantize_exact_replicated_text_tasks,
    };
    let (store, _checkpoint, _artifact_identity, source_config, config, tasks) =
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
        let source = A::module::<MlxNeuralBackend>(source_config, stream)
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
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(MlxExternalAssistant {
        config,
        module,
        observers: Default::default(),
    })
}

/// Architecture-dispatched MLX draft model with its fixed execution placement.
pub struct MlxDrafter {
    payload: Rc<OrdinaryRetirement<DrafterPayload>>,
    poisoned: Rc<Cell<bool>>,
}

struct DrafterPayload {
    execution:
        eredu_architectures::MaterializedExternalAssistantExecution<MlxAssistantPreparationVisitor>,
    stream: Stream,
}

struct DrafterRetention {
    _payload: Rc<RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload>>>>>,
    poisoned: Rc<Cell<bool>>,
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
    recovery: Recovery<DrafterRetention>,
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
            return Err(Error::Speculative(
                "native drafter work failed or remains unresolved".into(),
            ));
        }
        let value = result?;
        self.completed = true;
        Ok(value)
    }
}

impl Drop for DrafterOperation<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.drafter.poisoned.set(true);
        }
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
        })?;
        Ok(DrafterOperation {
            drafter: self,
            retained,
            recovery,
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
        preparation: eredu_architectures::PreparedExternalAssistantExecution,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        ordinary_retirement::reclaim();
        let poisoned = Rc::new(Cell::new(false));
        let retained = Rc::new(RefCell::new(None));
        let mut recovery = Recovery::begin(DrafterRetention {
            _payload: Rc::clone(&retained),
            poisoned: Rc::clone(&poisoned),
        })?;
        let execution = preparation.materialize(MlxAssistantPreparationVisitor {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
        })?;
        let payload = Rc::new(OrdinaryRetirement::new(DrafterPayload {
            execution,
            stream: stream.clone(),
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
