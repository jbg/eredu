//! Ownership of a bound text session and the resources attached to its execution.

use super::PreparedTextSessionFacts;
use crate::speculative::PredictionModule;
use crate::speculative::{
    MaterializedPredictionExecutor, PredictionExtensionMaterializer, PredictionModuleVisitor,
};
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    prefill::{ChunkedPrefillContextError, PrefillRequestDescriptor},
    replicated_session::{CompositePrompt, CompositeTextInvocation},
    ActivationObserver, PreparedInputCacheIdentity, ReplicatedTextArchitecture,
    ReplicatedTextInvocation, ReplicatedTextSessionError,
};
use crate::{
    replicated_session::PredictionModuleParameters, LayeredArchitecture,
    ReplicatedTextExecutionStrategy, ReplicatedTextSession, ReplicatedTextSessionMechanisms,
    SelectedProcessorExecution,
};
use eredu_core::{parameters::ParameterError, SubmissionBackend};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    NeuralBackend, Parameterized, Tensor,
};

/// No prediction extension is installed.
pub struct NoSelectedPrediction;

/// A materialized prediction extension and its selected execution realization.
pub struct SelectedPrediction<P> {
    /// Exact typed extension for the installed prediction target.
    pub extension: P,
    /// Retained speculative placement and capture requirements.
    pub selected: crate::SelectedSpeculativeRealization,
}

#[derive(Default)]
struct PredictionCatalog(Vec<PredictionModuleParameters>);

impl<B, X> PredictionModuleVisitor<B, X> for PredictionCatalog
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    X: PredictionExtensionMaterializer<B>,
{
    type Error = std::convert::Infallible;

    fn visit<M: Parameterized<B::Tensor>>(
        &mut self,
        ordinal: usize,
        module: &mut PredictionModule<B, X, M>,
    ) -> Result<(), Self::Error> {
        self.0.push(PredictionModuleParameters {
            ordinal,
            slots: module.parameters.clone(),
            tasks: module.tasks.clone(),
            report: module.materialization.clone(),
        });
        Ok(())
    }
}

/// Catalog of exact physical members supplied by the bound bank resources.
pub trait TextParameterBanks {
    /// Returns the physical bank members visible to this executable.
    fn bank_members(
        &self,
    ) -> Result<Vec<crate::parameter_operations::PreparedBankParameterMember>, ParameterError>;
}

use super::{session_erasure::BankOf, text_invocation::TextInvocationResources};
use std::collections::BTreeMap;

/// Weight pools retained and published by portable text composition.
pub(super) struct TextWeightPools<N: TextInvocationResources> {
    banks: BTreeMap<crate::RoutedBankId, BankOf<N>>,
    rows: Option<(BankOf<N>, crate::SelectedRowLookupRequirements)>,
}

impl<N: TextInvocationResources> Default for TextWeightPools<N> {
    fn default() -> Self {
        Self {
            banks: BTreeMap::new(),
            rows: None,
        }
    }
}

impl<N: TextInvocationResources> TextWeightPools<N> {
    pub(super) fn parameter_banks(&self) -> &BTreeMap<crate::RoutedBankId, BankOf<N>> {
        &self.banks
    }

    pub(super) fn row_pool(&self) -> Option<(&BankOf<N>, crate::SelectedRowLookupRequirements)> {
        self.rows
            .as_ref()
            .map(|(pool, requirements)| (pool, *requirements))
    }
}

/// The bound text executable held by the portable installation and retirement path.
pub struct BoundTextSession<S, I, N: TextInvocationResources, P> {
    /// Typed session that owns model execution and state.
    pub session: S,
    /// Selected prompt ingress, if the model has a separate input program.
    pub ingress: I,
    /// Native tensor, queue and memory bindings used by this executable.
    pub native: N,
    pub(super) banks: TextWeightPools<N>,
    /// Bound prediction extension, or the ordinary no-prediction selection.
    pub prediction: P,
    /// Selected architecture and residency facts for the installed session.
    pub facts: PreparedTextSessionFacts,
    /// Whether this rank publishes application scores.
    pub publishes_scores: bool,
}

impl<S, I, N: TextInvocationResources, P> BoundTextSession<S, I, N, P> {
    pub(super) fn with_weight_pools(
        mut self,
        banks: BTreeMap<crate::RoutedBankId, BankOf<N>>,
        rows: Option<(BankOf<N>, crate::SelectedRowLookupRequirements)>,
    ) -> Self {
        self.banks = TextWeightPools { banks, rows };
        self
    }
}

/// Retained composite ingress and placement facts.
pub struct CompositeSessionIngress<C> {
    /// Architecture-owned admission configuration.
    pub admission: C,
    /// Exact admitted input representation.
    pub processor: SelectedProcessorExecution,
    /// Retained placement of any external prediction lanes.
    pub external_target: crate::speculative::ExternalTargetPlacements,
}

impl<C> CompositeSessionIngress<C> {
    /// Couples the architecture configuration to its selected processor.
    pub fn new(admission: C, processor: SelectedProcessorExecution) -> Self {
        Self {
            admission,
            processor,
            external_target: Default::default(),
        }
    }
}

impl<A, B, M, D, I, N>
    BoundTextSession<ReplicatedTextSession<A, B, M, D>, I, N, NoSelectedPrediction>
where
    N: TextInvocationResources,
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Installs the native bank inventory against the session's parameter declarations.
    pub fn install_bank_catalog(mut self) -> Result<Self, ParameterError>
    where
        N: super::session_erasure::TextErasureResources,
    {
        let mut members = Vec::new();
        for bank in self.banks.parameter_banks().values() {
            members.extend(bank.bank_members()?);
        }
        self.session.install_bank_parameters(members)?;
        Ok(self)
    }

    /// Adopts a bound extension and publishes its parameter catalog before installation.
    pub fn install_prediction<X, Q>(
        mut self,
        mut prediction: SelectedPrediction<Q>,
        capability: crate::CapabilityEstimate,
    ) -> Result<
        BoundTextSession<ReplicatedTextSession<A, B, M, D>, I, N, SelectedPrediction<Q>>,
        ParameterError,
    >
    where
        B: BlockwiseAttentionBackend
            + DistributedNeuralBackend
            + GroupedNeuralBackend
            + HyperNeuralBackend,
        X: PredictionExtensionMaterializer<B>,
        Q: MaterializedPredictionExecutor<A, B, X>,
    {
        if prediction.extension.depth() == 0 || capability.speculative_draft_source().is_none() {
            return Err(ParameterError::Invalid(
                "prediction extension contract is missing executable draft depth".into(),
            ));
        }
        let mut catalog = PredictionCatalog::default();
        match prediction.extension.visit_modules(&mut catalog) {
            Ok(()) => {}
            Err(never) => match never {},
        }
        self.session.install_prediction_parameters(catalog.0)?;
        self.facts.capability = capability;
        Ok(BoundTextSession {
            session: self.session,
            ingress: self.ingress,
            native: self.native,
            banks: self.banks,
            prediction,
            facts: self.facts,
            publishes_scores: self.publishes_scores,
        })
    }
}

impl<A, B, M, D, N, P> BoundTextSession<ReplicatedTextSession<A, B, M, D>, (), N, P>
where
    N: TextInvocationResources,
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: ReplicatedTextArchitecture<B, M::State, Error = eredu_nn::Error>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Advances prepared text and establishes whether a failed invocation kept
    /// the installed model state intact. Native submission completion is tracked
    /// independently of this state evidence.
    pub fn execute_text<O>(
        &mut self,
        invocation: ReplicatedTextInvocation<B::Tensor, eredu_nn::Error>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match invocation {
            ReplicatedTextInvocation::FinalPositionPrefill(tokens) => {
                self.session.prefill_input_final_position(
                    match tokens {
                        Ok(ref tokens) => Ok(A::text_input(tokens, None)),
                        Err(error) => Err(error),
                    },
                    context,
                )
            }
            ReplicatedTextInvocation::Prefill {
                tokens,
                mask,
                capture,
            } => {
                let input = match tokens {
                    Ok(ref tokens) => Ok(tokens),
                    Err(error) => Err(error),
                }
                .and_then(|tokens| {
                    if let Some(request) = capture {
                        let [batch, sequence] = tokens.shape() else {
                            return Err(eredu_nn::Error::backend_source(
                                eredu_core::capture::CaptureError::Invalid(
                                    "prepared text input must have batch and sequence axes".into(),
                                ),
                            ));
                        };
                        let batch =
                            u64::try_from(*batch).map_err(eredu_nn::Error::backend_source)?;
                        let sequence =
                            u64::try_from(*sequence).map_err(eredu_nn::Error::backend_source)?;
                        request
                            .validate_prefill(batch, sequence)
                            .map_err(eredu_nn::Error::backend_source)?;
                    }
                    Ok(A::text_input(tokens, mask.as_ref()))
                });
                self.session
                    .prefill_input_result_with_observer(input, None, context, observer)
            }
            ReplicatedTextInvocation::Decode(tokens) => {
                self.session.decode_input_result_with_observer(
                    match tokens {
                        Ok(ref tokens) => Ok(A::text_input(tokens, None)),
                        Err(error) => Err(error),
                    },
                    context,
                    observer,
                )
            }
        }
    }
}

impl<A, B, M, D, N, P>
    BoundTextSession<
        ReplicatedTextSession<A, B, M, D>,
        CompositeSessionIngress<A::AdmissionConfig>,
        N,
        P,
    >
where
    N: TextInvocationResources,
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: CompositeArchitecture<B, M::State, Error = eredu_nn::Error>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Runs the selected composite invocation on the retained state.
    /// All ranks agree the prefill shape and chunking route before advancing it.
    pub fn execute_composite_text<O>(
        &mut self,
        invocation: CompositeTextInvocation<B::Tensor, A::InputPartPlan>,
        context: &B::Executor,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match invocation {
            CompositeTextInvocation::FinalPositionPrefill {
                prompt,
                fragment_identity,
            } => {
                let prompt = prompt.and_then(|mut prompt| {
                    let boundaries = self
                        .session
                        .resident_composite_prefill_boundaries(&self.ingress.admission, &prompt)?;
                    if let Some(identity) = fragment_identity {
                        prompt.2 = identity;
                    }
                    Ok((prompt, boundaries))
                });
                match prompt {
                    Ok((prompt, boundaries)) if !boundaries.is_empty() => self
                        .session
                        .prefill_composite_spans(prompt, boundaries, true, context, observer),
                    Ok((prompt, _)) => {
                        let (input, identity) = Self::composite_invocation_input(Ok(&prompt));
                        self.session.prefill_input_final_position_with_identity(
                            input,
                            identity.cloned(),
                            context,
                        )
                    }
                    Err(error) => self.session.prefill_input_final_position_with_identity(
                        Err(error),
                        None,
                        context,
                    ),
                }
            }
            CompositeTextInvocation::Prefill {
                prompt,
                capture,
                prediction,
            } => {
                let prompt = prompt.and_then(|prompt| {
                    let boundaries = self
                        .session
                        .resident_composite_prefill_boundaries(&self.ingress.admission, &prompt)?;
                    if !boundaries.is_empty() {
                        if prediction {
                            return Err(eredu_nn::Error::backend_source(
                                ChunkedPrefillContextError::Prediction,
                            ));
                        }
                        if capture.is_some() {
                            return Err(eredu_nn::Error::backend_source(
                                ChunkedPrefillContextError::CaptureGeometry,
                            ));
                        }
                    }
                    Ok((prompt, boundaries))
                });
                let prompt = match prompt {
                    Ok((prompt, boundaries)) if !boundaries.is_empty() => {
                        return self
                            .session
                            .prefill_composite_spans(prompt, boundaries, false, context, observer);
                    }
                    Ok((prompt, _)) => Ok(prompt),
                    Err(error) => Err(error),
                };
                let planned = prompt.and_then(|prompt| {
                    let admitted = &prompt.1;
                    if let Some(request) = capture {
                        let [batch, sequence] = admitted.decoder_shape();
                        request
                            .validate_prefill(batch, sequence)
                            .map_err(eredu_nn::Error::backend_source)?;
                    }
                    let tokens = usize::try_from(admitted.decoder_positions())
                        .map_err(eredu_nn::Error::backend_source)?;
                    let chunk_tokens =
                        A::prefill_chunk_limit(admitted)?.filter(|&limit| limit < tokens);
                    if chunk_tokens.is_some() {
                        if prediction {
                            return Err(eredu_nn::Error::backend_source(
                                ChunkedPrefillContextError::Prediction,
                            ));
                        }
                        if capture.is_some() {
                            return Err(eredu_nn::Error::backend_source(
                                ChunkedPrefillContextError::CaptureGeometry,
                            ));
                        }
                    }
                    Ok((
                        prompt,
                        PrefillRequestDescriptor {
                            tokens,
                            chunk_tokens,
                        },
                    ))
                });
                let (prompt, descriptor) = self
                    .session
                    .prepare_prefill(planned, |(_, descriptor)| *descriptor, context)
                    .map_err(|error| {
                        ReplicatedTextSessionError::BeforeStateMutation(Box::new(error))
                    })?;
                if let Some(chunk_tokens) = descriptor.chunk_tokens {
                    let (prepared, admitted, identity) = prompt;
                    let request = A::prepare_prefill_request(prepared, admitted, context);
                    let mut cursor = self
                        .session
                        .start_prefill(request, chunk_tokens, identity, context)
                        .map_err(|error| {
                            ReplicatedTextSessionError::BeforeStateMutation(Box::new(error))
                        })?;
                    self.session.finish_prefill(&mut cursor, context, observer)
                } else {
                    let (input, identity) = Self::composite_invocation_input(Ok(&prompt));
                    self.session.prefill_input_result_with_observer(
                        input,
                        identity.cloned(),
                        context,
                        observer,
                    )
                }
            }
            CompositeTextInvocation::Decode(prompt) => {
                let (input, _) = Self::composite_invocation_input(match prompt {
                    Ok(ref prompt) => Ok(prompt),
                    Err(error) => Err(error),
                });
                self.session
                    .decode_input_result_with_observer(input, context, observer)
            }
        }
    }

    fn composite_invocation_input<'a>(
        prompt: Result<&'a CompositePrompt<B::Tensor, A::InputPartPlan>, eredu_nn::Error>,
    ) -> (
        Result<A::Input<'a>, eredu_nn::Error>,
        Option<&'a PreparedInputCacheIdentity>,
    ) {
        match prompt {
            Ok((prepared, admitted, identity)) => (
                PreparedCompositeInput::new(prepared, admitted)
                    .map(A::prepared_input)
                    .map_err(eredu_nn::Error::backend),
                identity.as_ref(),
            ),
            Err(error) => (Err(error), None),
        }
    }
}
