use super::*;

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
    assistant: eredu_architectures::MaterializedExternalAssistant<MlxAssistantPreparationVisitor>,
    tokenizer_compatibility: TokenizerCompatibilityProof,
    stream: Stream,
    selected: SelectedSpeculativeRealization,
    capture: eredu_architectures::composite_execution::ExternalPredictionCaptureRequest,
}

impl MlxDrafter {
    /// Installs typed production observers for external-assistant tensors and logits.
    pub fn install_external_observers<TensorObserver, LogitsObserver>(
        &mut self,
        tensors: TensorObserver,
        logits: LogitsObserver,
    ) where
        TensorObserver: eredu_runtime::ActivationObserver<MlxTensor, Exception> + 'static,
        LogitsObserver: eredu_runtime::ActivationObserver<Array, Exception> + 'static,
    {
        self.assistant.visit(InstallExternalObservers {
            observers: Some(
                eredu_architectures::external_assistant::ExternalAssistantObservers::new(
                    tensors, logits,
                ),
            ),
        });
    }

    /// Materializes an architecture-inspected drafter with proven tokenizer compatibility.
    pub(crate) fn materialize_with_compatibility(
        preparation: eredu_architectures::PreparedCompatibleExternalAssistant,
        tokenizer_compatibility: TokenizerCompatibilityProof,
        stream: &Stream,
        weights_stream: &Stream,
        selected: SelectedSpeculativeRealization,
    ) -> Result<Self, Error> {
        if !matches!(
            selected.requirements().strategy().class(),
            eredu_runtime::SpeculativeStrategyClass::External
        ) || selected.requirements().strategy().tokenizer_fingerprint()
            != Some(tokenizer_compatibility.fingerprint())
        {
            return Err(Error::ArchitectureModel(
                "external assistant materialization received a different neutral realization"
                    .into(),
            ));
        }
        let capture = preparation.capture().clone();
        let assistant = preparation.visit(MlxAssistantPreparationVisitor {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
        })?;
        Ok(Self {
            assistant,
            tokenizer_compatibility,
            stream: stream.clone(),
            selected,
            capture,
        })
    }

    pub(crate) fn visit<W>(
        &mut self,
        visitor: W,
    ) -> <W as eredu_architectures::MaterializedExternalAssistantVisitor<
        MlxAssistantPreparationVisitor,
    >>::Output
    where
        W: eredu_architectures::MaterializedExternalAssistantVisitor<
            MlxAssistantPreparationVisitor,
        >,
    {
        self.assistant.visit(visitor)
    }

    /// Returns the portable proof established before this assistant was materialized.
    pub const fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.tokenizer_compatibility
    }

    /// Execution stream selected when this drafter was loaded.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns topology selected by the portable execution plan before queue construction.
    pub const fn topology(&self) -> SpeculativeExecutionTopology {
        self.selected.placement().topology()
    }

    /// Returns the one neutral realization retained from preconstruction selection.
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        &self.selected
    }

    pub(crate) const fn capture(
        &self,
    ) -> &eredu_architectures::composite_execution::ExternalPredictionCaptureRequest {
        &self.capture
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
