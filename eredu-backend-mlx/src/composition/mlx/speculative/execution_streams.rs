use super::autoregressive::{ActiveSpeculativeInvocation, AutoregressiveSourcePair};
use super::*;
use crate::backend::OriginalCopyEnvironment;
mod batch;
mod external_binding;

#[derive(Clone, Copy)]
struct OriginalExecution<'a> {
    sources: &'a AutoregressiveSourcePair,
    environment: &'a OriginalCopyEnvironment<'a>,
}
impl std::fmt::Debug for OriginalExecution<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalSpeculativeExecution")
    }
}

#[derive(Clone, Copy)]
struct OriginalNumericalExecution<'a> {
    sources: &'a OriginalSpeculativeNumericalSources,
    environment: &'a OriginalCopyEnvironment<'a>,
    draft_environment: &'a OriginalCopyEnvironment<'a>,
}
impl std::fmt::Debug for OriginalNumericalExecution<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalSpeculativeNumericalExecution")
    }
}

/// Target and assistant streams assigned to one speculative session.
#[derive(Debug, Clone, Copy)]
pub struct SpeculativeExecutionStreams<'a> {
    target: &'a Stream,
    draft: &'a Stream,
    topology: SpeculativeExecutionTopology,
    capture: Option<&'a super::super::session::SpeculativePartitionBinding>,
    memory_owner: Option<&'a crate::backend::managed_memory::NativeMemoryOwner>,
    memory_pool: Option<&'a eredu_runtime::working_memory::WorkingMemoryPool>,
    original: Option<OriginalExecution<'a>>,
    numerical: Option<OriginalNumericalExecution<'a>>,
    embedded: Option<&'a dyn EmbeddedInvocationSource>,
    external: Option<&'a dyn ExternalInvocationSource>,
    external_origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    embedded_invocation: Option<&'a dyn EmbeddedNumericalInvocation>,
    tensor_sources: Option<&'a [&'a eredu_architectures::speculative_execution::PreparedEmbeddedEvidence]>,
    prefill_input: Option<&'a super::super::prepared_speculative::OriginalEmbeddedPrefillInput>,
    cache_preparation: Option<&'a super::super::replicated_text::OriginalEmbeddedCachePreparation>,
    // A finite immutable table constructed before the shared visitor. Selected
    // contexts contain only their own ID, never a backedge into this table.
    batch_assignments: Option<&'a [SpeculativeExecutionStreams<'a>]>,
    batch_request: Option<eredu_core::SpeculativeRequestId>,
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
                ));
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
            capture: None,
            memory_owner: None,
            memory_pool: None,
            original: None,
            numerical: None,
            embedded: None,
            external: None,
            external_origin: None,
            embedded_invocation: None,
            tensor_sources: None,
            prefill_input: None,
            cache_preparation: None,
            batch_assignments: None,
            batch_request: None,
        })
    }

    #[cfg(test)]
    pub(super) fn for_test(target: &'a Stream, draft: &'a Stream) -> Result<Self, Exception> {
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
            capture: None,
            memory_owner: None,
            memory_pool: None,
            original: None,
            numerical: None,
            embedded: None,
            external: None,
            external_origin: None,
            embedded_invocation: None,
            tensor_sources: None,
            prefill_input: None,
            cache_preparation: None,
            batch_assignments: None,
            batch_request: None,
        }
    }

    /// Binds the actual selected pair and admitted native prerequisites. This
    /// borrowed handoff does not itself permit any invocation or construct work.
    pub(crate) fn with_original_sources(
        mut self,
        sources: &'a AutoregressiveSourcePair,
        environment: &'a OriginalCopyEnvironment<'a>,
    ) -> Result<Self, Error> {
        self = self.with_original_numerical_sources(sources.numerical_sources(), environment)?;
        self.original = Some(OriginalExecution { sources, environment });
        Ok(self)
    }
    /// Borrows the same admitted numerical source for any exact speculative
    /// request kind. This binds no model occurrence and creates no native scope.
    pub(crate) fn with_original_numerical_sources(
        self,
        sources: &'a OriginalSpeculativeNumericalSources,
        environment: &'a OriginalCopyEnvironment<'a>,
    ) -> Result<Self, Error> {
        self.with_original_numerical_environments(sources, environment, environment)
    }

    // Both lexical owners must authenticate against this same retained request.
    // Ordinary stream equality is never used to manufacture a missing owner.
    fn with_original_numerical_environments(
        mut self,
        sources: &'a OriginalSpeculativeNumericalSources,
        environment: &'a OriginalCopyEnvironment<'a>,
        draft_environment: &'a OriginalCopyEnvironment<'a>,
    ) -> Result<Self, Error> {
        let frames = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<OriginalNumericalExecution<'a>>(),
            // Existing single-environment forwarding and the selected common
            // binder own independent by-value context/result transports.
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<(&OriginalSpeculativeNumericalSources,
                &OriginalCopyEnvironment<'_>, &OriginalCopyEnvironment<'_>)>(),
            std::mem::size_of::<Result<(), Error>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<eredu_runtime::working_memory::WorkingMemoryError>()
                .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::HostMetadataFundingError::Overflow))?,
        ];
        let bytes=frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::HostMetadataFundingError::Overflow))?;
        sources.metadata_funding().reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
        sources.validate_environment(environment)?;
        if !std::ptr::eq(environment, draft_environment) {
            sources.validate_environment(draft_environment)?;
        }
        if self.target != environment.stream() || self.draft != draft_environment.stream()
            || !environment.pool().same_domain(draft_environment.pool())
            || self.memory_owner.is_some() || self.capture.is_some()
            || self.numerical.is_some() || self.original.is_some()
        {
            return Err(sources.retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.memory_pool=Some(environment.pool());
        self.numerical=Some(OriginalNumericalExecution { sources, environment, draft_environment });
        Ok(self)
    }
    /// Binds the same accepted Embedded request and its monotonic occurrence
    /// cursor. This does not open a scope or authorize an equation.
    pub(crate) fn with_original_embedded_sources(
        mut self,
        sources: &'a dyn EmbeddedInvocationSource,
        environment: &'a OriginalCopyEnvironment<'a>,
    ) -> Result<Self, Error> {
        self = self.with_original_numerical_sources(sources.numerical_sources(), environment)?;
        self.embedded = Some(sources);
        Ok(self)
    }
    pub(crate) fn original_embedded(self) -> Option<&'a dyn EmbeddedInvocationSource> {
        self.embedded
    }

    /// Borrows the exact separately materialized assistant request. No model
    /// invocation or native scope is created by this source handoff.
    pub(crate) fn with_original_external_sources(self,
        sources: &'a dyn ExternalInvocationSource,
        environment: &'a OriginalCopyEnvironment<'a>,
    ) -> Result<Self, Error> {
        self.with_original_external_environments(sources, environment, environment)
    }
    /// Lends the actual per-side prerequisites for the already selected topology.
    /// This grants no transfer, model occurrence, or new source completion.
    pub(crate) fn with_original_external_environments(mut self,
        sources: &'a dyn ExternalInvocationSource,
        target: &'a OriginalCopyEnvironment<'a>,
        draft: &'a OriginalCopyEnvironment<'a>,
    ) -> Result<Self, Error> {
        self = self.with_original_numerical_environments(sources.numerical_sources(), target, draft)?;
        self.external = Some(sources);
        Ok(self)
    }
    pub(crate) fn original_external(self) -> Option<&'a dyn ExternalInvocationSource> {
        self.external
    }
    /// Exact scheduler origin, supplied by the existing external executor hook.
    /// Reborrowing the same context preserves, rather than replaces, its origin.
    pub(crate) fn with_external_origin(mut self,
        origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    ) -> Result<Self, Error> {
        let Some(source) = self.external else { return Ok(self); };
        let funding = source.numerical_sources().metadata_funding();
        funding.reserve_metadata(std::mem::size_of::<(Self, Result<Self,Error>,
            Option<eredu_core::speculative::SpeculativeActivationOrigin>)>())
            .map_err(Error::WorkspacePlanning)?;
        if origin.is_none() || self.external_origin.is_some_and(|prior| Some(prior) != origin) {
            return Err(source.numerical_sources().retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        self.external_origin = origin;
        Ok(self)
    }
    pub(crate) fn external_origin(self)
        -> Option<eredu_core::speculative::SpeculativeActivationOrigin> { self.external_origin }

    /// Shortens the source context to one lexical native invocation. Comparing
    /// the actual borrowed source owner prevents a cross-request scalar producer.
    pub(crate) fn with_embedded_invocation<'scope>(
        self,
        active: &'scope dyn EmbeddedNumericalInvocation,
    ) -> Result<SpeculativeExecutionStreams<'scope>, Error>
    where 'a: 'scope,
    {
        let (sources, _) = self.original_numerical().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        if self.embedded.is_none() || self.embedded_invocation.is_some()
            || !std::ptr::eq(sources, active.sources())
        {
            return Err(sources.retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let controls = [
            std::mem::size_of::<SpeculativeExecutionStreams<'scope>>(),
            std::mem::size_of::<Result<SpeculativeExecutionStreams<'scope>, Error>>(),
            std::mem::size_of::<&dyn EmbeddedNumericalInvocation>(),
        ];
        sources.metadata_funding().reserve_metadata(
            controls.into_iter().try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::HostMetadataFundingError::Overflow))?,
        ).map_err(Error::WorkspacePlanning)?;
        let mut shortened: SpeculativeExecutionStreams<'scope> = self;
        shortened.embedded_invocation = Some(active);
        Ok(shortened)
    }
    /// Lends a finite authenticated inventory to one external model call.
    pub(crate) fn with_external_tensor_sources<'scope>(self,
        evidence: &'scope [&'scope eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
    ) -> Result<SpeculativeExecutionStreams<'scope>, Error> where 'a: 'scope {
        if self.external.is_none() {
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        self.with_tensor_sources(evidence)
    }
    /// Lends only the actual immutable packet evidence for one outer call.
    /// A model phase carries this loan forward; it cannot replace it in scope.
    pub(crate) fn with_embedded_tensor_sources<'scope>(
        self, evidence: &'scope [&'scope eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
    ) -> Result<SpeculativeExecutionStreams<'scope>, Error>
    where 'a: 'scope,
    {
        if self.embedded.is_none() {
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        self.with_tensor_sources(evidence)
    }
    fn with_tensor_sources<'scope>(self,
        evidence: &'scope [&'scope eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
    ) -> Result<SpeculativeExecutionStreams<'scope>, Error> where 'a: 'scope {
        let (sources, environment) = self.original_numerical().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        if (self.embedded.is_none() && self.external.is_none()) || self.embedded_invocation.is_some() {
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        let parts = [
            std::mem::size_of::<SpeculativeExecutionStreams<'scope>>(),
            std::mem::size_of::<Result<SpeculativeExecutionStreams<'scope>, Error>>(),
            std::mem::size_of_val(&evidence),
            std::mem::size_of::<std::slice::Iter<'_, &eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>>(),
        ];
        sources.metadata_funding().reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        if self.tensor_sources.is_some_and(|prior|prior.len()>evidence.len()
            || prior.iter().zip(evidence).any(|(a,b)|!std::ptr::eq(*a,*b))) {
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        if self.external.is_some() {
            super::tensor_sources::validate_external_tensor_sources(evidence,self)?;
        } else {
            super::tensor_sources::validate_tensor_sources(evidence,sources,environment)?;
        }
        let mut shortened: SpeculativeExecutionStreams<'scope> = self;
        shortened.tensor_sources = Some(evidence);
        Ok(shortened)
    }
    pub(crate) fn embedded_tensor_sources(self) -> &'a [&'a eredu_architectures::speculative_execution::PreparedEmbeddedEvidence] {
        self.tensor_sources.unwrap_or(&[])
    }

    /// Both model schedules lend the same completed B input source. The exact
    /// request/environment checks below remain independent of schedule kind.
    pub(crate) fn with_original_prefill_input<'scope>(self,
        input:&'scope super::super::prepared_speculative::OriginalEmbeddedPrefillInput,
    )->Result<SpeculativeExecutionStreams<'scope>,Error>
    where 'a:'scope,
    {
        let (sources,environment)=self.original_numerical().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
        sources.validate_environment(environment)?;
        input.validate_request(sources)?;
        if self.prefill_input.is_some() || (self.embedded.is_none() && self.external.is_none()) || self.embedded_invocation.is_some() {
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        sources.metadata_funding().reserve_metadata(std::mem::size_of::<(
            SpeculativeExecutionStreams<'scope>,Result<SpeculativeExecutionStreams<'scope>,Error>,
        )>()).map_err(Error::WorkspacePlanning)?;
        let mut scoped:SpeculativeExecutionStreams<'scope>=self;
        scoped.prefill_input=Some(input);
        Ok(scoped)
    }
    pub(crate) fn original_prefill_input(self)->Option<&'a super::super::prepared_speculative::OriginalEmbeddedPrefillInput> {
        self.prefill_input
    }

    pub(crate) fn embedded_invocation(self) -> Option<&'a dyn EmbeddedNumericalInvocation> {
        self.embedded_invocation
    }

    pub(crate) fn original_numerical(self) -> Option<(
        &'a OriginalSpeculativeNumericalSources,
        &'a OriginalCopyEnvironment<'a>,
    )> {
        self.original_numerical_for(SamplingPlacement::Target)
    }
    /// Selects a previously authenticated lexical owner; no stream is cloned.
    pub(crate) fn original_numerical_for(self, placement: SamplingPlacement) -> Option<(
        &'a OriginalSpeculativeNumericalSources,
        &'a OriginalCopyEnvironment<'a>,
    )> {
        self.numerical.and_then(|binding| Some((binding.sources, match placement {
            SamplingPlacement::Target => binding.environment,
            SamplingPlacement::Draft => binding.draft_environment,
            _ => return None,
        })))
    }
    /// Adds only a one-use cache constructor to the exact common request.
    /// Later target/prediction completion evidence belongs to each cache state.
    pub(crate) fn with_original_cache_preparation(mut self,
        preparation:&'a super::super::replicated_text::OriginalEmbeddedCachePreparation,
    )->Result<Self,eredu_core::BackendFailure> {
        let _controls=preparation.binding_controls()?;
        let Some((sources,_))=self.original_numerical() else {
            return Err(preparation.reject_source());
        };
        preparation.validate_sources(sources)?;
        if self.cache_preparation.is_some() || self.original.is_some() {
            return Err(preparation.reject_source());
        }
        self.cache_preparation=Some(preparation);
        Ok(self)
    }
    pub(crate) fn original_cache_preparation(self)->Option<&'a super::super::replicated_text::OriginalEmbeddedCachePreparation> {
        self.cache_preparation
    }
    pub(crate) fn original_request(
        self,
    ) -> Option<&'a eredu_runtime::working_memory::OriginalSpeculativeRequest> {
        self.numerical.map(|binding| binding.sources.request())
    }
    pub(crate) fn original_invocation(self) -> Result<Option<ActiveSpeculativeInvocation>, Error> {
        self.original.map_or(Ok(None), |binding| {
            binding.sources.active_invocation().map(Some)
        })
    }
    pub(crate) fn original_execution(
        self,
    ) -> Option<(
        &'a AutoregressiveSourcePair,
        &'a OriginalCopyEnvironment<'a>,
    )> {
        self.original
            .map(|binding| (binding.sources, binding.environment))
    }

    pub(in crate::composition::mlx) fn with_memory_owner(
        mut self,
        owner: &'a crate::backend::managed_memory::NativeMemoryOwner,
    ) -> Self {
        self.memory_owner = Some(owner);
        self.memory_pool = Some(owner.pool());
        self
    }

    pub(in crate::composition::mlx) fn memory_owner(
        self,
    ) -> Option<&'a crate::backend::managed_memory::NativeMemoryOwner> {
        self.memory_owner
    }

    /// Standalone native contexts use the same aggregate domain as production
    /// backends. Prepared execution binds its actual backend domain explicitly.
    pub(in crate::composition::mlx) fn memory_pool(
        self,
    ) -> eredu_runtime::working_memory::WorkingMemoryPool {
        self.memory_pool
            .cloned()
            .unwrap_or_else(crate::backend::managed_memory::domain)
    }

    #[cfg(test)]
    pub(in crate::composition::mlx) fn with_memory_pool(
        mut self,
        pool: &'a eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Self {
        self.memory_owner = None;
        self.memory_pool = Some(pool);
        self
    }

    pub(in crate::composition::mlx) fn with_capture_binding(
        mut self,
        binding: Option<&'a super::super::session::SpeculativePartitionBinding>,
    ) -> Self {
        self.capture = binding;
        self
    }
    pub(in crate::composition::mlx) fn capture_binding(
        self,
    ) -> Option<&'a super::super::session::SpeculativePartitionBinding> {
        self.capture
    }

    pub(in crate::composition::mlx) fn coordinate_speculative_step<B: AsRef<[eredu_core::SpeculativeScheduleState]> + AsMut<[eredu_core::SpeculativeScheduleState]>>(
        self,
        local: B,
    ) -> Result<B, eredu_core::BackendFailure> {
        match self.capture {
            Some(binding) => binding.coordinate_speculative_step(local),
            None => Ok(local),
        }
    }

    pub(in crate::composition::mlx) fn agree_text_preparation(
        self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        use eredu_core::run_preparation::{
            TextPreparationOutcome as O, TextPreparationStatus as S,
        };
        match self.capture {
            Some(binding) => binding.agree_text_preparation(stage, status),
            None => Ok(match status {
                S::Ready => O::Ready,
                S::Cancelled => O::Cancelled,
                S::Failed => O::Rejected { rank: 0 },
            }),
        }
    }

    pub(in crate::composition::mlx) fn finish_preparation<T, E>(
        self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        local: Result<T, E>,
        map_backend: impl FnOnce(eredu_core::BackendFailure) -> E,
    ) -> Result<T, E> {
        match self.capture {
            Some(binding) => binding.finish_preparation(stage, local, map_backend),
            None => local,
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
