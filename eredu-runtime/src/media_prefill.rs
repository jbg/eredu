//! Compact media ingress consumed by the shared selected prefill transaction.
//!
//! Sources retain one inference request or one accepted speculative occurrence.
//! Workspace-only sources retain neither; every native graph, storage and
//! completion authority remains with the calling transaction.

pub(crate) mod construction;
mod origin;
pub(crate) use origin::MediaPrefillOrigin;

use crate::layered::invocation::LayeredInvocation;
use crate::prefill::PrefillChunk;
use crate::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, InferenceStateRevision, WorkingMemoryError,
};
use crate::{
    ExecutionGraph, LayeredForwardState, LayeredTraversalHook,
    RuntimeState, SharedPreparedInputCacheIdentity,
};
use eredu_core::InferenceGeometry;
use eredu_nn::{NeuralBackend, Tensor};

/// Invalid retained-ingress topology or lifecycle; never permission to retry work.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum MediaIngressError {
    /// The selected graph has no declared primary decoder.
    #[error("media ingress has no selected primary decoder")]
    MissingDecoder,
    /// Work before the decoder is not its dependency closure.
    #[error("media ingress cut differs from the selected dependency prefix")]
    InvalidCut,
    /// A source was prepared for a different actual graph.
    #[error("media ingress source differs from the selected graph")]
    ForeignGraph,
    /// A supplied whole-input identity does not describe the retained plan input.
    #[error("media ingress cache identity differs from the retained input")]
    ForeignIdentity,
    /// The source cannot be replayed, skipped, or used after failure.
    #[error("media ingress source is not ready for this exact span")]
    InvalidSpan,
    /// Decoder ingress did not run or did not retain every required cut value.
    #[error("media ingress traversal did not produce its complete decoder cut")]
    IncompleteCut,
}

/// Family semantic operations around one selected encoder-to-decoder cut.
///
/// Implementations reuse their ordinary encoder/group/unit equations and retain
/// only media outputs and source metadata between decoder spans. Parallel entry
/// receives the same selected context as ordinary parallel execution.
pub trait PrefillIngressArchitecture<B, S>: crate::LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Owning exact prepared source and its architecture-derived semantic plan.
    type IngressPlan;
    /// Compact completed encoder outputs needed by later decoder spans.
    type Ingress;

    /// Original decoder geometry, including its admitted fixed chunk width.
    fn ingress_geometry(plan: &Self::IngressPlan) -> InferenceGeometry;
    /// Actual source/session binding for compiled semantics. Legacy plans carry
    /// none; this grants no native/input funding and performs no observation.
    fn ingress_session_binding(
        _plan: &Self::IngressPlan,
    ) -> Option<&crate::working_memory::MediaSessionBinding> {
        None
    }
    /// Original whole-input cache identity, published only after the final span.
    fn ingress_cache_identity(plan: &Self::IngressPlan)
    -> Option<SharedPreparedInputCacheIdentity>;
    /// Authenticates an externally retained cache identity against the actual
    /// plan input. Default rejects this optional capability before publication.
    fn validate_ingress_cache_identity(
        _plan: &Self::IngressPlan,
        _identity: &SharedPreparedInputCacheIdentity, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<(), Self::Error> {
        Err(Self::ingress_error(MediaIngressError::ForeignIdentity, metadata_context))
    }
/// Loans the actual graph after adapter admission checks. The destination
    /// funds diagnostic errors; it never selects a second graph producer.
    fn ingress_execution_graph(&self, context: Option<&eredu_nn::workspace::WorkspaceContext>)
        -> Result<crate::ArchitectureExecutionGraph<'_>, Self::Error>;
    /// Validates the plan against this actual selected architecture before work.
    fn validate_ingress_plan(&self, plan: &Self::IngressPlan, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<(), Self::Error>;
    /// Preserves a typed ingress contract cause within the architecture error.
    fn ingress_error(error: MediaIngressError, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Self::Error;
/// Retains the same typed cause through a participating error constructor.

    /// Starts only ingress work; full-prompt decoder embeddings/masks are deferred.
    fn begin_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Initializes semantic context from an actual selected encoder boundary,
    /// without repeating encoder projection. `encoder_continuation` selects the
    /// family-owned metadata needed by later encoder units.
    fn begin_ingress_received(
        &mut self,
        plan: &Self::IngressPlan,
        received: &B::Tensor,
        encoder_continuation: bool,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Retains compact outputs before primary decoder assembly changes context.
    fn retain_ingress(
        &mut self,
        plan: &Self::IngressPlan,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Ingress, Self::Error>;
    /// Builds the exact architecture context for a validated decoder span.
    fn begin_ingress_span(
        &mut self,
        plan: &Self::IngressPlan,
        ingress: &Self::Ingress,
        span: &PrefillChunk,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend;
    /// Visits all future-span roots, including outputs unused by the first span.
    fn visit_ingress_roots(ingress: &Self::Ingress, visitor: &mut dyn FnMut(&B::Tensor));
}

/// Validated dependency prefix of an actual selected architecture graph.
pub struct CompositePrefillCut {
    graph: ExecutionGraph,
    pub(crate) primary: usize,
    pub(crate) local_ingress_owner: bool,
    prefix: Vec<bool>,
    retained: Vec<bool>,
}

impl CompositePrefillCut {
    pub(crate) fn new(graph: ExecutionGraph, primary: &str) -> Result<Self, MediaIngressError> {
        Self::new_with_storage(
            graph,
            primary,
            |count| Ok(Vec::with_capacity(count)),
            |cause| cause,
        )
    }

    pub(crate) fn new_with_metadata(graph: ExecutionGraph, primary: &str,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        construction::Destination(Some(context)).cut(graph, primary)
    }

    fn new_with_storage<E>(
        graph: ExecutionGraph,
        primary: &str,
        allocate: impl Fn(usize) -> Result<Vec<bool>, E>,
        error: impl Fn(MediaIngressError) -> E,
    ) -> Result<Self, E> {
        let primary = graph
            .group_index(primary)
            .ok_or_else(|| error(MediaIngressError::MissingDecoder))?;
        let mut prefix = allocate(graph.groups().len())?;
        prefix.resize(graph.groups().len(), false);
        // The graph is already a validated DAG. Reverse topological propagation
        // computes the exact ancestor closure without a second execution loop.
        prefix[primary] = true;
        for &group in graph.execution_order().iter().rev() {
            if prefix[group] {
                for &dependency in graph.dependencies(group).expect("validated graph") {
                    prefix[dependency] = true;
                }
            }
        }
        prefix[primary] = false;
        let mut before_primary = true;
        for &group in graph.execution_order() {
            if group == primary {
                before_primary = false;
                continue;
            }
            if prefix[group] != before_primary {
                return Err(error(MediaIngressError::InvalidCut));
            }
        }
        let mut retained = allocate(graph.groups().len())?;
        retained.resize(graph.groups().len(), false);
        for &dependency in graph.dependencies(primary).expect("validated primary") {
            retained[dependency] = true;
        }
        Ok(Self {
            graph,
            primary,
            local_ingress_owner: true,
            prefix,
            retained,
        })
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Phase {
    Cold,
    Ready,
    InFlight,
    Failed,
    Complete,
}

// Closed outcomes of the actual first traversal, not a cold activity estimate.
enum CutValue<T> {
    Unseen,
    Inactive,
    Produced(T),
}
impl<T> CutValue<T> {
    fn value(&self) -> Option<&T> {
        if let Self::Produced(value) = self {
            Some(value)
        } else {
            None
        }
    }
    fn is_unseen(&self) -> bool {
        matches!(self, Self::Unseen)
    }
}

/// Persistent, non-cloneable ordinary media source for one exact request.
///
/// The selected session constructs this owner. A short `SessionPrefill` loan
/// can be recreated between advances without re-encoding or replacing its plan.
pub struct PreparedMediaPrefill<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    pub(crate) plan: A::IngressPlan,
    pub(crate) cut: CompositePrefillCut,
    origin: MediaPrefillOrigin,
    prompt_identity: Option<crate::SharedPreparedInputCacheIdentity>,
    geometry: InferenceGeometry,
    revision: InferenceStateRevision,
    phase: Phase,
    next: u64,
    current_end: u64,
    pub(crate) ingress: Option<A::Ingress>,
    cut_values: Vec<CutValue<B::Tensor>>,
    // This context can contain incomplete encoder temporaries. Retire it only
    // after the actual first-span completion/commit, or under failed custody.
    prefix_context: Option<A::ForwardContext>,
    marker: std::marker::PhantomData<fn() -> S>,
    // Real planning Context shared by the admitted native source, when present.
    // Retires after all source containers and native retained values.
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}

impl<A, B, S> PreparedMediaPrefill<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    pub(crate) fn semantic_binding(&self) -> Option<&crate::working_memory::MediaSessionBinding> {
        A::ingress_session_binding(&self.plan)
    }

    pub(crate) fn new(
        plan: A::IngressPlan,
        cut: CompositePrefillCut,
        request: InferenceRequest,
        execution: &InferenceExecutionIdentity,
        revision: InferenceStateRevision,
    ) -> Result<Self, WorkingMemoryError> {
        Self::new_with_storage(
            plan,
            cut,
            request,
            execution,
            revision,
            |count| Ok(Vec::with_capacity(count)),
            |cause| cause,
        )
    }

    fn new_with_storage<E>(
        plan: A::IngressPlan,
        cut: CompositePrefillCut,
        request: InferenceRequest,
        execution: &InferenceExecutionIdentity,
        revision: InferenceStateRevision,
        allocate: impl FnOnce(usize) -> Result<Vec<CutValue<B::Tensor>>, E>,
        error: impl FnOnce(WorkingMemoryError) -> E,
    ) -> Result<Self, E> {
        let geometry = A::ingress_geometry(&plan);
        request.validate(execution, geometry).map_err(error)?;
        Self::from_storage(plan, cut, MediaPrefillOrigin::Inference(request), revision, allocate)
    }

    fn from_storage<E>(
        plan: A::IngressPlan, cut: CompositePrefillCut, origin: MediaPrefillOrigin,
        revision: InferenceStateRevision,
        allocate: impl FnOnce(usize) -> Result<Vec<CutValue<B::Tensor>>, E>,
    ) -> Result<Self, E> {
        let geometry = A::ingress_geometry(&plan);
        let mut cut_values = allocate(cut.graph.groups().len())?;
        cut_values.resize_with(cut.graph.groups().len(), || CutValue::Unseen);
        Ok(Self {
            plan,
            cut,
            origin,
            prompt_identity: None,
            geometry,
            revision,
            phase: Phase::Cold,
            next: 0,
            current_end: 0,
            ingress: None,
            cut_values,
            prefix_context: None,
            marker: std::marker::PhantomData,
            metadata: None,
        })
    }

    pub(crate) fn new_speculative(
        plan: A::IngressPlan, cut: CompositePrefillCut,
        role: crate::working_memory::OriginalSpeculativeRole,
        revision: InferenceStateRevision,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        role.validate_prefill_geometry(A::ingress_geometry(&plan))
            .map_err(|cause| context.metadata_source(cause))?;
        context.charge_metadata(std::mem::size_of::<(
            Self, Result<Self, eredu_nn::Error>, A::IngressPlan, CompositePrefillCut,
            crate::working_memory::OriginalSpeculativeRole, InferenceStateRevision,
            Vec<CutValue<B::Tensor>>, CutValue<B::Tensor>,
        )>())?;
        let origin=MediaPrefillOrigin::speculative(role,A::ingress_geometry(&plan))
            .map_err(|cause|context.metadata_source(cause))?;
        let mut source = Self::from_storage(plan, cut, origin, revision,
            |count| context.metadata_vec(count))?;
        source.metadata = Some(context.clone());
        Ok(source)
    }

    pub(crate) fn validate_speculative_span(
        &self, span: &crate::working_memory::OriginalSpeculativePrefillSpan,
    ) -> Result<(), WorkingMemoryError> {
        self.origin.validate_span(span)?;
        self.validate_span(span.chunk()).map_err(|_| WorkingMemoryError::IdentityMismatch)
    }

    pub(crate) fn with_prompt_identity(
        mut self,
        identity: Option<crate::SharedPreparedInputCacheIdentity>,
    ) -> Result<Self, A::Error> {
        if let Some(identity) = &identity {
            match &self.metadata {
                Some(metadata) => A::validate_ingress_cache_identity(
                    &self.plan, identity, Some(metadata))?,
                None => A::validate_ingress_cache_identity(&self.plan, identity, None)?,
            }
        }
        self.prompt_identity = identity;
        Ok(self)
    }

    pub(crate) fn cache_identity(&self) -> Option<crate::SharedPreparedInputCacheIdentity> {
        self.prompt_identity
            .clone()
            .or_else(|| A::ingress_cache_identity(&self.plan))
    }

    /// Actual inference request, when this source was prepared for that driver.
    /// Speculative ingress carries its accepted role instead and cannot be
    /// converted into an inference request or replayed by the ordinary driver.
    pub fn request(&self) -> Result<&InferenceRequest, WorkingMemoryError> {
        self.origin.request()
    }

    /// Exact original decoder and chunk geometry.
    pub fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }

    /// Visits retained media roots without constructing a new owner vector.
    pub fn visit_retained_roots(&self, visitor: &mut dyn FnMut(&B::Tensor)) {
        if let Some(ingress) = &self.ingress {
            A::visit_ingress_roots(ingress, visitor);
        }
        for value in self.cut_values.iter().filter_map(CutValue::value) {
            visitor(value);
        }
    }

    pub(crate) fn validate_request(
        &self,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        self.request()?.validate_same_request(request)
    }

    pub(crate) fn validate_span(&self, chunk: &PrefillChunk) -> Result<(), MediaIngressError> {
        if !matches!(self.phase, Phase::Cold | Phase::Ready)
            || chunk.input.start != self.next
            || chunk.input.end <= self.next
            || chunk.input.end > self.geometry.input_positions
            || chunk.input.end
                != self
                    .next
                    .saturating_add(self.geometry.prefill_chunk_positions)
                    .min(self.geometry.input_positions)
            || self
                .geometry
                .cached_positions
                .checked_add(chunk.input.start)
                != Some(chunk.position)
            || chunk.output
                != self
                    .geometry
                    .output
                    .for_chunk(chunk.input.end == self.geometry.input_positions)
        {
            return Err(MediaIngressError::InvalidSpan);
        }
        Ok(())
    }

    pub(crate) fn validate_revision(
        &self,
        revision: &InferenceStateRevision,
    ) -> Result<(), WorkingMemoryError> {
        if &self.revision == revision {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// The closed span lifecycle already authenticated this source's exact request
    /// and old revision in the successful reservation vote. First admission changes
    /// the branch revision without executing input; retain only that actual result.
    /// A later span has the same admitted request and keeps its committed revision.
    pub(crate) fn admit_request(
        &mut self,
        retained: &mut crate::working_memory::InferenceRetention,
        request: &InferenceRequest,
    ) {
        retained.admit(request);
        self.revision = retained.revision().clone();
    }

    pub(crate) fn start(&mut self, chunk: &PrefillChunk) -> Result<bool, MediaIngressError> {
        self.validate_span(chunk)?;
        let initial = self.phase == Phase::Cold;
        self.phase = Phase::InFlight;
        self.current_end = chunk.input.end;
        Ok(initial)
    }

    pub(crate) fn validate_complete_cut(&self) -> Result<(), MediaIngressError> {
        if self.phase != Phase::InFlight
            || (self.cut.local_ingress_owner && self.ingress.is_none())
            || (self.cut.local_ingress_owner
                && self
                    .cut
                    .retained
                    .iter()
                    .zip(&self.cut_values)
                    .any(|(needed, value)| *needed && value.is_unseen()))
        {
            return Err(MediaIngressError::IncompleteCut);
        }
        Ok(())
    }

    pub(crate) fn committed(
        &mut self,
        revision: &InferenceStateRevision,
    ) -> Result<(), MediaIngressError> {
        if let Err(error) = self.validate_complete_cut() {
            self.phase = Phase::Failed;
            return Err(error);
        }
        self.origin.committed().map_err(|_|MediaIngressError::InvalidSpan)?;
        self.next = self.current_end;
        self.revision = revision.clone();
        self.phase = if self.next == self.geometry.input_positions {
            Phase::Complete
        } else {
            Phase::Ready
        };
        self.prefix_context = None;
        Ok(())
    }

    pub(crate) fn failed(&mut self) {
        self.phase = Phase::Failed;
    }
}

/// Borrowed invocation issued by the exact selected media source.
/// No public constructor; adapters can only consume its original cut and span.
pub struct MediaInvocation<'a, A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    pub(crate) source: &'a mut PreparedMediaPrefill<A, B, S>,
    pub(crate) span: &'a PrefillChunk,
    pub(crate) initial: bool,
}

impl<A, B, S> MediaInvocation<'_, A, B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    fn metadata_error(
        &self,
        error: MediaIngressError,
        context: &<B::Tensor as Tensor>::Context,
    ) -> A::Error {
        match self
            .source
            .metadata
            .as_ref()
            .or_else(|| B::construction_metadata(context))
        {
            Some(metadata) => A::ingress_error(error, Some(metadata)),
            None => A::ingress_error(error, None),
        }
    }
    fn matches_graph(
        &self,
        architecture: &A,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, A::Error> {
        let graph = architecture.ingress_execution_graph(self.source.metadata.as_ref().or_else(|| B::construction_metadata(context)))?;
        Ok(graph.matches(&self.source.cut.graph))
    }
    fn validate_plan(
        &self,
        architecture: &A,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), A::Error> {
        match self
            .source
            .metadata
            .as_ref()
            .or_else(|| B::construction_metadata(context))
        {
            Some(metadata) => {
                architecture.validate_ingress_plan(&self.source.plan, Some(metadata))
            }
            None => architecture.validate_ingress_plan(&self.source.plan, None),
        }
    }

    fn before_group_using(
        &mut self,
        architecture: &mut A,
        group: usize,
        initial: &mut B::Tensor,
        forward: &mut A::ForwardContext,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        mut retained: impl FnMut(&A::Ingress) -> Result<(), A::Error>,
    ) -> Result<bool, A::Error> {
        if !self.initial || group != self.source.cut.primary {
            return Ok(false);
        }
        let ingress = architecture.retain_ingress(&self.source.plan, forward, context)?;
        // Install the owner before a fallible span constructor can start work.
        self.source.ingress = Some(ingress);
        retained(self.source.ingress.as_ref().expect("installed ingress"))?;
        let next = architecture.begin_ingress_span(
            &self.source.plan,
            self.source.ingress.as_ref().expect("installed ingress"),
            self.span,
            state,
            parallel,
            context,
        )?;
        self.source.prefix_context = Some(std::mem::replace(forward, next.context));
        *initial = next.hidden;
        Ok(true)
    }

    /// Exact original architecture plan.
    pub fn plan(&self) -> &A::IngressPlan {
        &self.source.plan
    }
    /// Current validated decoder interval.
    pub fn span(&self) -> &PrefillChunk {
        self.span
    }
    /// Whether this is the first transaction, including encoder ancestors.
    pub fn is_initial(&self) -> bool {
        self.initial
    }
    /// Exact cut selected on this rank.
    pub fn primary_group(&self) -> usize {
        self.source.cut.primary
    }
    /// Whether this selected rank assembles the first decoder ingress.
    pub fn owns_decoder_ingress(&self) -> bool {
        self.source.cut.local_ingress_owner
    }
    /// A completed ancestor imported by a later decoder span.
    pub fn imported_group(&self, group: usize) -> Option<Option<B::Tensor>> {
        <Self as LayeredInvocation<A, B, S>>::retained_group(self, group)
    }
    /// Retains an actual group result under this invocation's original source.
    pub fn retain_group_result(&mut self, group: usize, value: &B::Tensor) {
        <Self as LayeredInvocation<A, B, S>>::after_group(self, group, value)
    }
    /// Records a disabled group only after the selected scheduler observes it.
    /// Imported later spans cannot change the first traversal's outcome.
    pub fn retain_inactive_group(&mut self, group: usize) {
        <Self as LayeredInvocation<A, B, S>>::after_inactive_group(self, group)
    }
    /// Installs the exact compact cut before primary decoder assembly.
    pub fn enter_group(
        &mut self,
        architecture: &mut A,
        group: usize,
        initial: &mut B::Tensor,
        forward: &mut A::ForwardContext,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, A::Error> {
        <Self as LayeredInvocation<A, B, S>>::before_group(
            self,
            architecture,
            group,
            initial,
            forward,
            state,
            parallel,
            context,
        )
    }
    /// Uses the same ingress transition while observing its exact retained cut.
    pub fn enter_group_with_cut(
        &mut self, architecture: &mut A, group: usize, initial: &mut B::Tensor,
        forward: &mut A::ForwardContext, state: &mut S, parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        cut: &mut dyn FnMut(&mut dyn FnMut(&mut dyn FnMut(&B::Tensor))) -> Result<(), A::Error>,
    ) -> Result<bool, A::Error> {
        self.before_group_using(architecture, group, initial, forward, state, parallel, context,
            |ingress| cut(&mut |visitor| A::visit_ingress_roots(ingress, visitor)))
    }

    /// Creates only family semantic context for the actual received boundary.
    pub fn begin_received(
        &mut self,
        architecture: &mut A,
        received: &B::Tensor,
        encoder_continuation: bool,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error> {
        if !self.initial || !self.matches_graph(architecture, context)? {
            return Err(self.metadata_error(MediaIngressError::ForeignGraph, context));
        }
        self.validate_plan(architecture, context)?;
        architecture.begin_ingress_received(
            &self.source.plan,
            received,
            encoder_continuation,
            state,
            parallel,
            context,
        )
    }
    /// Starts the selected ingress/span without changing request authority.
    pub fn begin_selected(
        &mut self,
        architecture: &mut A,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error> {
        if !self.matches_graph(architecture, context)? {
            return Err(self.metadata_error(MediaIngressError::ForeignGraph, context));
        }
        self.validate_plan(architecture, context)?;
        if self.initial {
            architecture.begin_ingress(&self.source.plan, state, parallel, context)
        } else {
            let ingress =
                self.source.ingress.as_ref().ok_or_else(|| {
                    self.metadata_error(MediaIngressError::IncompleteCut, context)
                })?;
            architecture.begin_ingress_span(
                &self.source.plan,
                ingress,
                self.span,
                state,
                parallel,
                context,
            )
        }
    }
}

impl<A, B, S> LayeredInvocation<A, B, S> for MediaInvocation<'_, A, B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    fn begin<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        _hook: &mut H,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error> {
        self.begin_selected(architecture, state, None, context)
    }

    fn begin_parallel<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        _hook: &mut H,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error> {
        self.begin_selected(architecture, state, Some(parallel), context)
    }

    fn is_retained_group(&self, group: usize) -> bool {
        !self.initial && self.source.cut.prefix[group]
    }
    fn retained_group(&self, group: usize) -> Option<Option<B::Tensor>> {
        self.is_retained_group(group)
            .then(|| self.source.cut_values[group].value().cloned())
    }

    fn before_group(
        &mut self,
        architecture: &mut A,
        group: usize,
        initial: &mut B::Tensor,
        forward: &mut A::ForwardContext,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, A::Error> {
        self.before_group_using(
            architecture,
            group,
            initial,
            forward,
            state,
            parallel,
            context,
            |_| Ok(()),
        )
    }
    fn before_group_with_hook<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        group: usize,
        initial: &mut B::Tensor,
        forward: &mut A::ForwardContext,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<bool, A::Error> {
        self.before_group_using(
            architecture,
            group,
            initial,
            forward,
            state,
            parallel,
            context,
            |ingress| {
                hook.retained_media_cut(
                    &mut |visitor| A::visit_ingress_roots(ingress, visitor),
                    context,
                )
            },
        )
    }

    fn after_inactive_group(&mut self, group: usize) {
        if self.initial && self.source.cut.prefix[group] {
            self.source.cut_values[group] = CutValue::Inactive;
        }
    }
    fn is_inactive_dependency(&self, group: usize) -> bool {
        !self.initial && matches!(self.source.cut_values[group], CutValue::Inactive)
    }
    fn after_group(&mut self, group: usize, value: &B::Tensor) {
        if self.initial && self.source.cut.retained[group] {
            self.source.cut_values[group] = CutValue::Produced(value.clone());
        }
    }
}

/// Selected strategies that can traverse a retained media cut through their
/// ordinary unit and transport driver. There is no fallback to whole input.
pub trait MediaTextExecutionStrategy<A, B, S, R, P>:
    crate::ReplicatedTextExecutionStrategy<A, B, S, R, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
    R: crate::LayerwisePolicy<B, A::Unit>,
    P: crate::LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    R::Error: std::fmt::Display,
{
    /// Validates the actual selected architecture and obtains its dependency cut.
    fn prepare_media_cut(
        runtime: &Self::Runtime,
        plan: &A::IngressPlan,
    ) -> Result<CompositePrefillCut, A::Error>;

    /// Builds the same actual cut with paid graph and table destinations.
    /// The default refuses before calling an uncounted architecture constructor.
    fn prepare_media_cut_with_metadata(
        _runtime: &Self::Runtime,
        _plan: &A::IngressPlan,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<CompositePrefillCut, eredu_nn::Error>
    where
        A: PrefillIngressArchitecture<B, S, Error = eredu_nn::Error>,
    {
        Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into())
    }

    /// Executes one already validated decoder span, including ingress on span one.
    /// Prepared observers borrow the actual retained runtime paths. The strategy
    /// must preserve that binding while using its selected ordinary unit equation.
    fn forward_media_span<O>(
        &mut self,
        runtime: &mut Self::Runtime,
        source: &mut PreparedMediaPrefill<A, B, S>,
        chunk: &PrefillChunk,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: Option<&crate::PreparedLayeredObservationPaths>,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        crate::ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized;
}

/// A borrowed traversal of the exact future media roots. Implemented only by
/// runtime sources; completion mechanisms cannot replace its owner or frontier.
pub struct RetainedMediaRoots<'a, T> {
    visit: &'a dyn MediaRootSource<T>,
}
trait MediaRootSource<T> {
    fn visit(&self, visitor: &mut dyn FnMut(&T));
}
impl<T> RetainedMediaRoots<'_, T> {
    /// Visits each retained root without collecting a library-owned vector.
    pub fn visit(&self, visitor: &mut dyn FnMut(&T)) {
        self.visit.visit(visitor)
    }
}
impl<A, B, S> MediaRootSource<B::Tensor> for PreparedMediaPrefill<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    fn visit(&self, visitor: &mut dyn FnMut(&B::Tensor)) {
        self.visit_retained_roots(visitor)
    }
}
impl<A, B, S> PreparedMediaPrefill<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    pub(crate) fn roots(&self) -> RetainedMediaRoots<'_, B::Tensor> {
        RetainedMediaRoots { visit: self }
    }
}

/// Ordinary metadata-only media traversal; this module cannot submit native work.
pub mod workspace;
