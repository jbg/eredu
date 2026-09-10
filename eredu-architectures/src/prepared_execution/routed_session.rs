//! Selected grouped-provider construction; callbacks bind banks or wrap completed sessions.

use eredu_nn::{GroupedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    AddressableGroupedBank, IndexedMovement, LayeredArchitecture, ParameterBankResidency,
    ReplicatedTextSession, ReplicatedTextSessionMechanisms, RoutedLayeredArchitecture,
    RoutedReplicatedTextExecution, SubmissionBackend,
};

use super::PreparedExecutionError;
use crate::composite_execution::PreparedCompositeArchitecture;
use crate::routed_text::{
    PlannedAddressableBank, PlannedResidentBank, PreparedRoutedTextArchitecture, SelectedRoutedBank,
};

/// Retained architecture facts for final adaptation of a constructed text session.
#[derive(Debug, Clone)]
pub struct PreparedTextSessionFacts {
    prompt_cache_identity: eredu_core::cache::PromptCacheModelIdentity,
    capability: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    residency: eredu_runtime::LayerWeightResidency,
}

/// Exact ingress admission and adaptation facts retained with a constructed composite session.
pub struct PreparedCompositeSessionFacts<C> {
    text: PreparedTextSessionFacts,
    processor: eredu_runtime::SelectedProcessorExecution,
    admission: C,
}

impl<C> PreparedCompositeSessionFacts<C> {
    pub(super) fn new(
        text: PreparedTextSessionFacts,
        processor: eredu_runtime::SelectedProcessorExecution,
        admission: C,
    ) -> Self {
        Self {
            text,
            processor,
            admission,
        }
    }

    /// Consumes the facts after the routed session constructor has succeeded.
    pub fn into_parts(
        self,
    ) -> (
        PreparedTextSessionFacts,
        eredu_runtime::SelectedProcessorExecution,
        C,
    ) {
        (self.text, self.processor, self.admission)
    }
}

/// Constructs routed composite execution and retains exact processor/admission facts centrally.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn construct_selected_routed_composite_session<
    B,
    A,
    Admission,
    M,
    Bank,
    Movement,
    C,
    MakeBanks,
    R,
    I,
    O,
    E,
>(
    prepared: crate::replicated_text::PreparedRoutedCompositeTextArchitecture<A, Admission>,
    mechanisms: M,
    context: &<B::Tensor as Tensor>::Context,
    make_banks: MakeBanks,
    native: C,
    finish_resident: R,
    finish_addressable: I,
) -> Result<O, PreparedExecutionError<E>>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend,
    M: ReplicatedTextSessionMechanisms<PreparedCompositeArchitecture<A>, B>,
    PreparedCompositeArchitecture<A>:
        LayeredArchitecture<B, M::State> + RoutedLayeredArchitecture<B, M::State>,
    <PreparedCompositeArchitecture<A> as LayeredArchitecture<B, M::State>>::Error:
        std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    MakeBanks: FnOnce(
        &std::collections::BTreeMap<eredu_runtime::RoutedBankId, SelectedRoutedBank>,
        eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<
        std::collections::BTreeMap<eredu_runtime::RoutedBankId, (Bank, Movement)>,
        E,
    >,
    R: FnOnce(
        C,
        ReplicatedTextSession<
            PreparedCompositeArchitecture<A>,
            B,
            M,
            RoutedReplicatedTextExecution<eredu_runtime::RoutedBankProviders<PlannedResidentBank>>,
        >,
        PreparedCompositeSessionFacts<Admission>,
    ) -> Result<O, E>,
    I: FnOnce(
        C,
        ReplicatedTextSession<
            PreparedCompositeArchitecture<A>,
            B,
            M,
            RoutedReplicatedTextExecution<
                eredu_runtime::RoutedBankProviders<PlannedAddressableBank<B, Bank, Movement>>,
            >,
        >,
        PreparedCompositeSessionFacts<Admission>,
    ) -> Result<O, E>,
{
    let mut text = PreparedTextSessionFacts::from_prepared(prepared.routed().text());
    text.capability = prepared.capability_estimate().clone();
    text.effective_model_type = prepared.effective_model_type().to_owned();
    let (routed, processor, admission) = prepared.into_parts();
    let facts = PreparedCompositeSessionFacts {
        text,
        processor,
        admission,
    };
    construct_selected_routed_session(
        routed,
        mechanisms,
        context,
        make_banks,
        (native, facts),
        |(native, facts), session, _| finish_resident(native, session, facts),
        |(native, facts), session, _| finish_addressable(native, session, facts),
    )
}

impl PreparedTextSessionFacts {
    pub(super) fn from_parts(
        prompt_cache_identity: eredu_core::cache::PromptCacheModelIdentity,
        capability: crate::capability::CapabilityEstimate,
        effective_model_type: String,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Self {
        Self {
            prompt_cache_identity,
            capability,
            effective_model_type,
            residency,
        }
    }

    /// Copies immutable adaptation facts before consuming the selected modules.
    pub fn from_prepared<A>(
        prepared: &crate::replicated_text::PreparedReplicatedTextArchitecture<A>,
    ) -> Self {
        Self {
            prompt_cache_identity: prepared.prompt_cache_identity().clone(),
            capability: prepared.capability_estimate().clone(),
            effective_model_type: prepared.effective_model_type().to_owned(),
            residency: prepared.selected().residency(),
        }
    }

    /// Exact prompt-cache model identity retained during architecture preparation.
    pub const fn prompt_cache_identity(&self) -> &eredu_core::cache::PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    /// Capability estimate retained by the selected target architecture.
    pub const fn capability(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability
    }

    /// Normalized model-type label retained by the selected target architecture.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Exact selected execution-unit weight residency.
    pub const fn residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.residency
    }

    /// Moves immutable adaptation facts into a native outer session wrapper.
    pub fn into_parts(
        self,
    ) -> (
        eredu_core::cache::PromptCacheModelIdentity,
        crate::capability::CapabilityEstimate,
        String,
        eredu_runtime::LayerWeightResidency,
    ) {
        (
            self.prompt_cache_identity,
            self.capability,
            self.effective_model_type,
            self.residency,
        )
    }
}

/// Constructs the selected providers and session before native final adaptation.
///
/// Bank creation receives the complete collection and one shared residency budget. The
/// borrowed prepared value supplies admitted bank recipes and geometry, never
/// caller load options. Final callbacks receive already constructed sessions;
/// the shared native context is moved into only the selected final callback.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn construct_selected_routed_session<B, A, M, Bank, Movement, C, MakeBanks, R, I, O, E>(
    prepared: PreparedRoutedTextArchitecture<A>,
    mechanisms: M,
    context: &<B::Tensor as Tensor>::Context,
    make_banks: MakeBanks,
    native: C,
    finish_resident: R,
    finish_addressable: I,
) -> Result<O, PreparedExecutionError<E>>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State> + RoutedLayeredArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    MakeBanks: FnOnce(
        &std::collections::BTreeMap<eredu_runtime::RoutedBankId, SelectedRoutedBank>,
        eredu_runtime::ParameterBankLoadOptions,
    ) -> Result<
        std::collections::BTreeMap<eredu_runtime::RoutedBankId, (Bank, Movement)>,
        E,
    >,
    R: FnOnce(
        C,
        ReplicatedTextSession<
            A,
            B,
            M,
            RoutedReplicatedTextExecution<eredu_runtime::RoutedBankProviders<PlannedResidentBank>>,
        >,
        PreparedTextSessionFacts,
    ) -> Result<O, E>,
    I: FnOnce(
        C,
        ReplicatedTextSession<
            A,
            B,
            M,
            RoutedReplicatedTextExecution<
                eredu_runtime::RoutedBankProviders<PlannedAddressableBank<B, Bank, Movement>>,
            >,
        >,
        PreparedTextSessionFacts,
    ) -> Result<O, E>,
{
    let facts = PreparedTextSessionFacts::from_prepared(prepared.text());
    match prepared.bank_residency() {
        ParameterBankResidency::WithLayer => {
            let session = prepared
                .construct_resident_session::<B, M>(mechanisms, context)
                .map_err(PreparedExecutionError::Architecture)?;
            finish_resident(native, session, facts).map_err(PreparedExecutionError::Backend)
        }
        ParameterBankResidency::IndependentCache(options) => {
            let banks =
                make_banks(prepared.banks(), options).map_err(PreparedExecutionError::Backend)?;
            let session = prepared
                .construct_addressable_session::<B, M, Bank, Movement>(mechanisms, banks, context)
                .map_err(PreparedExecutionError::Architecture)?;
            finish_addressable(native, session, facts).map_err(PreparedExecutionError::Backend)
        }
        _ => Err(PreparedExecutionError::Architecture(
            "selected bank residency has no construction mechanism".into(),
        )),
    }
}
