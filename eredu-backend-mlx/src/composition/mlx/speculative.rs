//! MLX execution primitives for speculative model sessions.

/// External assistant adapters over neutral family equations and state.
pub mod external;
/// MLX semantic-generation resource adapter over the portable runtime driver.
pub mod scheduler;

pub use scheduler::SpeculativeComponentTimingGuard;

use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    CompletionCancellationMode, SamplingPlacement, SpeculativeDraftRandomPosition,
    SpeculativeExecutionTopology, SpeculativeSampling, TokenizerCompatibilityProof,
};
use eredu_runtime::{
    SelectedSpeculativeRealization, SpeculativeMechanism, SpeculativeMechanismCapabilities,
    SpeculativeSampler,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, maximum, softmax_axis},
    random,
    transforms::{async_eval_with_event, eval},
    Array, Event, Stream,
};
use std::cell::RefCell;

use crate::{
    backend::error::Error,
    backend::nn::shared::{MlxModule, MlxNeuralBackend},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    MlxTensor,
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
    stream: Stream,
    weights_stream: Stream,
    max_cached_shards: usize,
}

impl eredu_architectures::ExternalAssistantPreparationVisitor for MlxAssistantPreparationVisitor {
    type Output<A: eredu_architectures::ExternalAssistantArchitecture> = MlxExternalAssistant<A>;
    type Error = Error;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        prepared: eredu_architectures::SelectedExternalAssistant<A>,
    ) -> Result<Self::Output<A>, Self::Error> {
        materialize_external_assistant::<A>(
            prepared,
            &self.stream,
            &self.weights_stream,
            self.max_cached_shards,
        )
    }
}

fn materialize_external_assistant<A: eredu_architectures::ExternalAssistantArchitecture>(
    prepared: eredu_architectures::SelectedExternalAssistant<A>,
    stream: &Stream,
    weights_stream: &Stream,
    max_cached_shards: usize,
) -> Result<MlxExternalAssistant<A>, Error> {
    use crate::backend::runtime::{
        checkpoint::binding::{
            build_exact_replicated_text_bindings, materialize_module_bindings,
            populate_module_from_arrays_excluding,
        },
        execution::layerwise::quantize_exact_replicated_text_tasks,
    };
    use std::sync::Arc;

    let (checkpoint, source_config, config, tasks) = prepared.into_parts();
    let store = match checkpoint {
        eredu_architectures::ExternalAssistantCheckpoint::SafeTensors {
            source,
            catalog,
            plan,
            resolution,
        } => crate::composition::mlx::artifact::open_prepared_safetensors_checkpoint(
            &source,
            catalog,
            &plan,
            &resolution,
            max_cached_shards,
        )?,
        eredu_architectures::ExternalAssistantCheckpoint::Gguf {
            checkpoint,
            resolution,
            tensor_mapping,
        } => {
            let store: eredu_checkpoint::store::SharedCheckpointSource = Arc::new(
                eredu_checkpoint::gguf_store::GgufWeightStore::builder()
                    .max_cached_readers(max_cached_shards)?
                    .add_resolved_checkpoint(checkpoint, &resolution, &tensor_mapping)?
                    .build()?,
            );
            store
        }
    };
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
    let bindings = build_exact_replicated_text_bindings(
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
        preparation: eredu_architectures::CompatibleExternalAssistantPreparation,
        tokenizer_compatibility: TokenizerCompatibilityProof,
        max_cached_shards: usize,
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
            max_cached_shards,
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

/// Target and assistant streams assigned to one speculative session.
#[derive(Debug, Clone, Copy)]
pub struct SpeculativeExecutionStreams<'a> {
    target: &'a Stream,
    draft: &'a Stream,
    topology: SpeculativeExecutionTopology,
}

impl<'a> SpeculativeExecutionStreams<'a> {
    /// Binds queues to topology already selected by portable composition.
    pub fn bind(
        target: &'a Stream,
        draft: &'a Stream,
        topology: SpeculativeExecutionTopology,
    ) -> Result<Self, Exception> {
        let matches = match topology {
            SpeculativeExecutionTopology::Single => target == draft,
            SpeculativeExecutionTopology::SameDeviceSplit => {
                target != draft && target.get_device()? == draft.get_device()?
            }
            SpeculativeExecutionTopology::CrossDeviceSplit => {
                target.get_device()? != draft.get_device()?
            }
            _ => {
                return Err(Exception::custom(
                    "selected speculative topology is unsupported by the MLX queue binder",
                ))
            }
        };
        if !matches {
            return Err(Exception::custom(format!(
                "selected speculative topology {topology:?} does not match bound MLX queues"
            )));
        }
        Ok(Self {
            target,
            draft,
            topology,
        })
    }

    #[cfg(test)]
    fn for_test(target: &'a Stream, draft: &'a Stream) -> Result<Self, Exception> {
        let topology = if target == draft {
            SpeculativeExecutionTopology::Single
        } else if target.get_device()? == draft.get_device()? {
            SpeculativeExecutionTopology::SameDeviceSplit
        } else {
            SpeculativeExecutionTopology::CrossDeviceSplit
        };
        Self::bind(target, draft, topology)
    }

    /// Creates an assignment in which all speculative work uses one stream.
    pub const fn single(stream: &'a Stream) -> Self {
        Self {
            target: stream,
            draft: stream,
            topology: SpeculativeExecutionTopology::Single,
        }
    }

    /// Stream used for target prefill and verification.
    pub const fn target(self) -> &'a Stream {
        self.target
    }

    /// Stream used for proposal generation.
    pub const fn draft(self) -> &'a Stream {
        self.draft
    }

    /// Relationship between the target and assistant streams.
    pub const fn topology(self) -> SpeculativeExecutionTopology {
        self.topology
    }

    /// Whether target and assistant work use different streams.
    pub const fn is_split(self) -> bool {
        !matches!(self.topology, SpeculativeExecutionTopology::Single)
    }

    /// Whether values must be physically transferred between devices.
    pub const fn crosses_devices(self) -> bool {
        matches!(
            self.topology,
            SpeculativeExecutionTopology::CrossDeviceSplit
        )
    }

    /// Submits target outputs and orders subsequent assistant work after them.
    pub fn wait_for_target_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
    ) -> Result<Event, Exception> {
        self.wait_for_same_device_outputs(outputs, self.draft, "target-to-draft")
    }

    /// Submits assistant outputs and orders subsequent target work after them.
    pub fn wait_for_draft_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
    ) -> Result<Event, Exception> {
        self.wait_for_same_device_outputs(outputs, self.target, "draft-to-target")
    }

    fn wait_for_same_device_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
        consumer: &Stream,
        direction: &str,
    ) -> Result<Event, Exception> {
        if self.topology != SpeculativeExecutionTopology::SameDeviceSplit {
            return Err(Exception::custom(format!(
                "speculative {direction} event handoff requires distinct streams on one device, got {}",
                self.topology
            )));
        }
        let completion = async_eval_with_event(outputs)?;
        completion.wait_on(consumer)?;
        Ok(completion)
    }
}

/// Exact completion for one retained MLX speculative verification.
pub struct MlxSpeculativeCompletion {
    event: Event,
    _retained: Vec<Array>,
}

impl MlxSpeculativeCompletion {
    /// Submits all retained verification outputs as one exact completion.
    pub fn submit<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<Self, Exception> {
        let retained = outputs.into_iter().cloned().collect::<Vec<_>>();
        let event = async_eval_with_event(retained.iter())?;
        Ok(Self {
            event,
            _retained: retained,
        })
    }

    #[cfg(test)]
    fn retained(&self) -> &[Array] {
        &self._retained
    }
}

impl Completion for MlxSpeculativeCompletion {
    type Error = Exception;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete()
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize()
    }
}

#[derive(Default)]
struct SpeculativeCompletionQuarantine {
    work: Vec<MlxSpeculativeCompletion>,
}

impl SpeculativeCompletionQuarantine {
    fn reap(&mut self) {
        self.work
            .retain(|completion| !matches!(completion.is_complete(), Ok(true) | Err(_)));
    }
}

impl Drop for SpeculativeCompletionQuarantine {
    fn drop(&mut self) {
        self.reap();
        for completion in self.work.drain(..) {
            let _ = completion.wait();
        }
    }
}

thread_local! {
    static SPECULATIVE_COMPLETION_ORPHANS: RefCell<SpeculativeCompletionQuarantine> =
        RefCell::new(SpeculativeCompletionQuarantine::default());
}

fn reap_speculative_completion_orphans() {
    let empty = SPECULATIVE_COMPLETION_ORPHANS.try_with(|orphans| {
        if let Ok(mut orphans) = orphans.try_borrow_mut() {
            orphans.reap();
            return orphans.work.is_empty();
        }
        false
    });
    if matches!(empty, Ok(true)) {
        safemlx::unregister_thread_runtime_housekeeping(reap_speculative_completion_orphans);
    }
}

fn quarantine_speculative_completion(completion: MlxSpeculativeCompletion) {
    safemlx::register_thread_runtime_housekeeping(reap_speculative_completion_orphans);
    SPECULATIVE_COMPLETION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        orphans.work.push(completion);
    });
}

impl BoundedCompletion for MlxSpeculativeCompletion {
    fn supports_cancellation(cancellation: CompletionCancellationMode) -> bool {
        cancellation == CompletionCancellationMode::QuarantineUntilComplete
    }

    fn wait_bounded(
        self,
        policy: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            quarantine_speculative_completion(self);
            return Err(Exception::custom(
                "speculative completion deadline exceeds the host monotonic clock range; live work was quarantined safely",
            ));
        };
        loop {
            if self.is_complete()? {
                return Ok(BoundedCompletionOutcome::Completed);
            }
            if std::time::Instant::now() >= deadline {
                let selected = policy.cancellation();
                quarantine_speculative_completion(self);
                if selected != CompletionCancellationMode::QuarantineUntilComplete {
                    return Err(Exception::custom(
                        "MLX speculative execution has no native cancellation; timed-out work was quarantined safely",
                    ));
                }
                return Ok(BoundedCompletionOutcome::DeadlineExceeded {
                    cancellation: CompletionCancellationMode::QuarantineUntilComplete,
                });
            }
            std::thread::yield_now();
        }
    }
}

impl Drop for MlxSpeculativeCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

/// MLX implementation of opaque speculative sampling operations.
#[derive(Clone)]
pub struct MlxSpeculativeSampling<S> {
    inner: S,
}

impl<S> MlxSpeculativeSampling<S> {
    /// Wraps one runtime sampling policy for MLX execution.
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Returns the sampling policy after a test generation completes.
    #[cfg(test)]
    pub fn into_inner(self) -> S {
        self.inner
    }

    #[cfg(test)]
    pub const fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S> SpeculativeSampling for MlxSpeculativeSampling<S>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
{
    type Logits = Array;
    type Distribution = Array;
    type Seed = Array;
    type RandomState = RandomState;
    type DraftRandomness = Array;
    type RandomnessRoot = RandomState;
    type Context<'a>
        = SpeculativeExecutionStreams<'a>
    where
        Self: 'a;
    type Error = Exception;

    fn supports_exact_optimistic_promotion(&self) -> bool {
        SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(&self.inner)
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        SpeculativeSampler::<MlxSamplingBackend>::grammar_is_complete(&mut self.inner)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        SpeculativeSampler::<MlxSamplingBackend>::prefix_is_complete(&self.inner, history)
    }

    fn randomness_root<'a>(
        seed: Option<Self::Seed>,
        _: Self::Context<'a>,
    ) -> Result<Self::RandomnessRoot, Self::Error>
    where
        Self: 'a,
    {
        seed.map(RandomState::from_key)
            .ok_or_else(|| Exception::custom("random operations require an explicit PRNG key"))
    }

    fn target_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        root.next_key(context.target()).map(RandomState::from_key)
    }

    fn draft_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftRandomness, Self::Error>
    where
        Self: 'a,
    {
        let draft_key = root.next_key(context.target())?;
        let draft_key = if context.is_split() {
            if context.crosses_devices() {
                async_eval_with_event([&draft_key])?.synchronize()?;
                let copied = draft_key.copy(context.draft())?;
                async_eval_with_event([&copied])?.synchronize()?;
                copied
            } else {
                let _completion = context.wait_for_target_outputs([&draft_key])?;
                draft_key
            }
        } else {
            draft_key
        };
        Ok(draft_key)
    }

    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: SpeculativeDraftRandomPosition,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        Ok(RandomState::from_key(crate::backend::random::split_key_at(
            root,
            position.get(),
            context.draft(),
        )?))
    }

    fn process_logits<'a>(
        &mut self,
        logits: &Self::Logits,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Distribution, Self::Error>
    where
        Self: 'a,
    {
        SpeculativeSampler::<MlxSamplingBackend>::process_logits(
            &mut self.inner,
            &MlxTensor::from_array(logits.clone()),
            temperature,
            history,
            sampling_stream(placement, context)?,
        )
        .map(MlxTensor::into_array)
    }

    fn sample<'a>(
        &self,
        distribution: &Self::Distribution,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<u32, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_stream(placement, context)?;
        let token = SpeculativeSampler::<MlxSamplingBackend>::sample_processed(
            &self.inner,
            &MlxTensor::from_array(distribution.clone()),
            temperature,
            randomness,
            stream,
        )?;
        eval([token.as_array()])?;
        Ok(token.into_array().item::<u32>(stream))
    }

    fn probability_at<'a>(
        &self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_stream(placement, context)?;
        let probabilities = probabilities(distribution, stream)?;
        array_probability_at(&probabilities, token, stream)
    }

    fn sample_unit_interval<'a>(
        &self,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        uniform(randomness, context.target())
    }

    fn positive_probability_difference<'a>(
        &self,
        left: &Self::Distribution,
        right: &Self::Distribution,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::Distribution>, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_stream(placement, context)?;
        let left_probabilities = probabilities(left, stream)?;
        let right_probabilities = probabilities(right, stream)?;
        let difference = maximum(
            left_probabilities.subtract(&right_probabilities, stream)?,
            Array::from_f32(0.0),
            stream,
        )?;
        let mass = difference.sum(None, stream)?.item::<f32>(stream);
        if mass <= f32::EPSILON {
            Ok(None)
        } else {
            difference.log(stream).map(Some)
        }
    }

    fn update_sampler_state<'a>(
        &mut self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        SpeculativeSampler::<MlxSamplingBackend>::commit_token(
            &mut self.inner,
            &MlxTensor::from_array(distribution.clone()),
            token,
            sampling_stream(placement, context)?,
        )
    }

    fn prepare_verification<'a>(
        &self,
        distributions: &mut [&mut Self::Distribution],
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        if temperature == 0.0 || !context.is_split() {
            return Ok(());
        }
        if context.crosses_devices() {
            async_eval_with_event(distributions.iter().map(|distribution| &**distribution))?
                .synchronize()?;
            for distribution in distributions {
                **distribution = distribution.copy(context.target())?;
            }
        } else {
            let _completion = context
                .wait_for_draft_outputs(distributions.iter().map(|distribution| &**distribution))?;
        }
        Ok(())
    }
}

fn sampling_stream<'a>(
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'a>,
) -> Result<&'a Stream, Exception> {
    match placement {
        SamplingPlacement::Target => Ok(context.target()),
        SamplingPlacement::Draft => Ok(context.draft()),
        _ => Err(Exception::custom(
            "unsupported speculative sampling placement requires an explicit stream",
        )),
    }
}

fn probabilities(logits: &Array, stream: &Stream) -> Result<Array, Exception> {
    softmax_axis(&logits.as_type::<f32>(stream)?, -1, true, stream)
}

fn array_probability_at(
    probabilities: &Array,
    token: u32,
    stream: &Stream,
) -> Result<f32, Exception> {
    let vocabulary = probabilities.dim(-1);
    if vocabulary <= 0 || u64::from(token) >= vocabulary as u64 {
        return Err(Exception::custom(format!(
            "sampled token {token} exceeds vocabulary size {}",
            vocabulary
        )));
    }
    let token = i32::try_from(token)
        .map_err(|_| Exception::custom("sampled token exceeds the index domain"))?;
    let value = match probabilities.ndim() {
        2 => probabilities.try_index_device((0, token), stream)?,
        3 => probabilities.try_index_device((0, 0, token), stream)?,
        ndim => {
            return Err(Exception::custom(format!(
                "speculative distribution must be rank 2 or 3, got rank {ndim}"
            )))
        }
    };
    Ok(value.item::<f32>(stream))
}

fn uniform(state: Option<&mut RandomState>, stream: &Stream) -> Result<f32, Exception> {
    let state = state
        .ok_or_else(|| Exception::custom("stochastic speculative decoding requires a PRNG key"))?;
    let key = state.next_key(stream)?;
    Ok(random::uniform::<_, f32>(0.0, 1.0, &[1], &key, stream)?.item::<f32>(stream))
}

#[cfg(test)]
mod completion_tests {
    use eredu_core::Completion;
    use safemlx::{Array, Device, DeviceType, Stream};

    use super::{array_probability_at, MlxSpeculativeCompletion};

    #[test]
    fn probability_lookup_rejects_tokens_outside_the_i32_vocabulary_before_indexing() {
        let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
        let probabilities = Array::from_slice(&[0.25_f32, 0.75], &[1, 2]);

        let error = array_probability_at(&probabilities, u32::MAX, &stream).unwrap_err();
        assert!(error.to_string().contains("exceeds vocabulary size 2"));
    }

    #[test]
    fn native_speculative_completion_retains_every_submitted_array_handle() {
        let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
        let logits = Array::from_slice(&[1.0_f32], &[1]);
        let capture = Array::from_slice(&[3.0_f32], &[1]);
        let completion = MlxSpeculativeCompletion::submit([&logits, &capture]).unwrap();
        drop(logits);
        drop(capture);

        assert_eq!(completion.retained().len(), 2);
        completion.wait().unwrap();
        assert!(completion.is_complete().unwrap());
        assert_eq!(completion.retained()[0].clone().item::<f32>(&stream), 1.0);
        assert_eq!(completion.retained()[1].clone().item::<f32>(&stream), 3.0);
    }
}

#[cfg(test)]
mod external_materialization_tests {
    use eredu_architectures::{
        gemma4, ExternalAssistantArchitecture, ExternalAssistantTargetProfile,
        MaterializedExternalAssistantVisitor,
    };
    use eredu_checkpoint::schema::StoredDtypeConstraint;
    use safemlx::{Device, DeviceType, Stream};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    use super::{MlxAssistantPreparationVisitor, MlxExternalAssistant};

    const ASSISTANT_CONFIG: &str = r#"{
      "model_type":"gemma4_assistant","backbone_hidden_size":32,
      "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
      "text_config":{"model_type":"gemma4_text","hidden_size":32,
        "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
        "vocab_size":32,"max_position_embeddings":128,"tie_word_embeddings":false,
        "attention_k_eq_v":false,"layer_types":["full_attention"]}
    }"#;

    fn assistant_artifact() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), ASSISTANT_CONFIG).unwrap();
        let config = gemma4::AssistantConfig::from_json(ASSISTANT_CONFIG.as_bytes()).unwrap();
        let plan = gemma4::assistant_safetensors_plan(&config).unwrap();
        assert!(plan.layout_groups.is_empty());
        let tensors = plan
            .common_tensors
            .iter()
            .map(|tensor| {
                assert_eq!(tensor.dtype, StoredDtypeConstraint::Floating);
                let bytes = vec![0; tensor.shape.iter().product::<usize>() * 4];
                (tensor.key.clone(), tensor.shape.clone(), bytes)
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    fn target_profile() -> ExternalAssistantTargetProfile {
        let assistant = gemma4::AssistantConfig::from_json(ASSISTANT_CONFIG.as_bytes()).unwrap();
        let mut text = assistant.text_config;
        let mut publisher = *text.layer_schedule.get(0).unwrap();
        publisher.key_value = eredu_nn::AttentionStateSource::Publish {
            value: eredu_nn::AttentionValueSource::Projected,
        };
        text.layer_schedule = eredu_core::LayerSchedule::new(1, vec![publisher]).unwrap();
        ExternalAssistantTargetProfile::Gemma4(gemma4::FamilyConfig {
            model_type: "gemma4".into(),
            text,
            vision: None,
            image_token_id: None,
            video_token_id: None,
            audio: None,
            audio_token_id: None,
        })
    }

    fn visitor(stream: &Stream) -> MlxAssistantPreparationVisitor {
        MlxAssistantPreparationVisitor {
            stream: stream.clone(),
            weights_stream: stream.clone(),
            max_cached_shards: eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS,
        }
    }

    fn selected(
        preparation: eredu_architectures::ExternalAssistantPreparation,
        quantization: Option<eredu_core::QuantizationRequest>,
    ) -> eredu_architectures::SelectedExternalAssistantPreparation {
        preparation
            .select_materialization(quantization, |descriptor, transforms| {
                if transforms && super::super::replicated_text::supports_transform(descriptor) {
                    Some(eredu_runtime::WeightLoweringKind::Transform)
                } else if !transforms && super::super::replicated_text::supports_direct(descriptor)
                {
                    Some(eredu_runtime::WeightLoweringKind::Direct)
                } else {
                    None
                }
            })
            .unwrap()
    }

    struct InspectMaterialized;

    impl MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor> for InspectMaterialized {
        type Output = (String, Option<eredu_checkpoint::WeightQuantization>);

        fn visit<A: ExternalAssistantArchitecture>(
            self,
            assistant: &mut MlxExternalAssistant<A>,
        ) -> Self::Output {
            (
                A::configuration_model_type(&assistant.config).to_owned(),
                A::quantization(&assistant.config),
            )
        }
    }

    #[test]
    fn family_blind_mlx_materializer_revalidates_catalog_without_reloading_target() {
        let artifact = assistant_artifact();
        let compatible = selected(
            eredu_architectures::prepare_external_assistant(artifact.path()).unwrap(),
            None,
        )
        .prove_target_compatibility(&target_profile())
        .unwrap();
        let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
        crate::composition::mlx::path_instrumentation::reset();

        let mut materialized = compatible.visit(visitor(&stream)).unwrap();
        assert_eq!(
            materialized.visit(InspectMaterialized),
            ("gemma4_assistant".into(), None)
        );
        let counts = crate::composition::mlx::path_instrumentation::snapshot();
        assert_eq!(counts.payload_opens, 1);
        assert_eq!(counts.constructors, 0);
    }

    #[test]
    fn family_blind_mlx_materializer_applies_architecture_load_time_format() {
        let artifact = assistant_artifact();
        let compatible = selected(
            eredu_architectures::prepare_external_assistant(artifact.path()).unwrap(),
            Some(eredu_core::QuantizationRequest::MxFp4),
        )
        .prove_target_compatibility(&target_profile())
        .unwrap();
        let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
        crate::composition::mlx::path_instrumentation::reset();

        let mut materialized = compatible.visit(visitor(&stream)).unwrap();
        assert_eq!(
            materialized.visit(InspectMaterialized),
            (
                "gemma4_assistant".into(),
                Some(eredu_checkpoint::WeightQuantization::MxFp4)
            )
        );
        let counts = crate::composition::mlx::path_instrumentation::snapshot();
        assert_eq!(counts.payload_opens, 1);
        assert_eq!(counts.constructors, 0);
    }

    #[test]
    fn unsupported_target_lowering_fails_before_native_resources() {
        let plan = eredu_core::ExecutionPlan::fully_resident(
            eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
        )
        .with_weight_transformation(eredu_core::WeightTransformationPlan::Affine {
            bits: 4,
            group_size: 32_768,
        })
        .with_drafting(eredu_core::DraftingPlan::External {
            model: "unopened-assistant".into(),
            placement: eredu_core::DraftPlacementPlan::Target,
            max_draft_tokens: 2,
            lookahead: false,
            adaptive_lookahead: false,
        });
        let target = super::super::replicated_text::tests::tiny_artifact("llama", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(target.path())
            .expect("tiny target inspection");
        let factory = crate::composition::mlx::automatic::MlxBackendFactory::default();
        crate::composition::mlx::path_instrumentation::reset();

        let error = match eredu_core::select_execution_plan_target(&factory, &plan, inspection) {
            Ok(_) => panic!("invalid target packing geometry unexpectedly selected"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains("select_model_preparation"),
            "{error}"
        );
        assert_eq!(
            crate::composition::mlx::path_instrumentation::target_native_resource_realization_attempts(),
            0
        );
        let counts = crate::composition::mlx::path_instrumentation::snapshot();
        assert_eq!(counts.payload_opens, 0);
        assert_eq!(counts.constructors, 0);
        assert_eq!(counts.materializations, 0);
    }

    #[test]
    fn incompatible_target_assistant_pair_fails_before_native_target_resources() {
        let assistant = assistant_artifact();
        let target = super::super::replicated_text::tests::tiny_artifact("llama", false);
        let plan = eredu_core::ExecutionPlan::fully_resident(
            eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
        )
        .with_drafting(eredu_core::DraftingPlan::External {
            model: assistant.path().display().to_string(),
            placement: eredu_core::DraftPlacementPlan::Target,
            max_draft_tokens: 2,
            lookahead: false,
            adaptive_lookahead: false,
        });
        let factory = crate::composition::mlx::automatic::MlxBackendFactory::default();
        let inspection = eredu_architectures::configuration::inspect_artifact(target.path())
            .expect("tiny target inspection");
        let selected_target = eredu_core::select_execution_plan_target(&factory, &plan, inspection)
            .expect("ordinary target selection");
        let preparation = eredu_architectures::prepare_external_assistant(assistant.path())
            .expect("assistant inspection");
        crate::composition::mlx::path_instrumentation::reset();

        let error = eredu_core::select_execution_plan_drafting(
            &factory,
            &plan,
            &selected_target,
            Some(eredu_core::ExternalDraftArtifact {
                preparation,
                tokenizer_compatibility: eredu_core::TokenizerCompatibilityProof::prove(
                    [3; 32], [3; 32],
                )
                .unwrap(),
            }),
        )
        .expect_err("Llama has no external-assistant target contract");

        assert!(error
            .to_string()
            .contains("does not admit an external assistant"));
        assert_eq!(
            crate::composition::mlx::path_instrumentation::target_native_resource_realization_attempts(),
            0
        );
        let counts = crate::composition::mlx::path_instrumentation::snapshot();
        assert_eq!(counts.payload_opens, 0);
        assert_eq!(counts.constructors, 0);
        assert_eq!(counts.materializations, 0);
    }
}
