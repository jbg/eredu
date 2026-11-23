//! Builds routed sessions from checked modules, bank plans and native mechanisms.

use eredu_core::SubmissionBackend;

use std::collections::BTreeMap;

use eredu_nn::{GroupedNeuralBackend, NeuralBackend, Tensor};

use crate::{
    AddressableGroupedBank, LayeredArchitecture, ParameterBankLoadOptions, ParameterBankResidency,
    ParameterProvider, ReplicatedTextSession, ReplicatedTextSessionMechanisms, RoutedBankId,
    RoutedLayeredArchitecture, RoutedReplicatedTextExecution,
};
use eredu_nn::tensor_movement::IndexedMovement;

use super::{
    routed_partition::RoutedTextExecutionError, PreparedCompositeSessionFacts,
    PreparedExecutionError, PreparedReplicatedTextArchitecture, PreparedTextSessionFacts,
    TextParameterBanks,
};

/// Realizes the selected grouped equation using resident modules or native storage.
pub trait RoutedBankBinding: Sized {
    /// Selected physical members eligible for independent acquisition.
    fn addressable_members(&self) -> &[crate::AddressableBankMember];

    /// Provider for experts that materialize with their layers.
    type Resident;
    /// Provider for independently acquired experts.
    type Addressable<B, Bank, Movement>: ParameterProvider<B> + RoutedBankObservation<B, Bank>
    where
        B: GroupedNeuralBackend,
        Bank: AddressableGroupedBank<B>,
        Movement: IndexedMovement<B>;

    /// Builds the resident provider with the retained route geometry.
    fn bind_resident(self) -> Result<Self::Resident, RoutedTextExecutionError>;

    /// Binds addressable storage and movement to the retained expert plan.
    fn bind_addressable<B, Bank, Movement>(
        self,
        bank: Bank,
        movement: Movement,
        options: ParameterBankLoadOptions,
    ) -> Result<Self::Addressable<B, Bank, Movement>, RoutedTextExecutionError>
    where
        B: GroupedNeuralBackend,
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
        Movement: IndexedMovement<B>,
        Movement::Error: std::fmt::Display;
}

/// Observes independently retained storage without knowing the bank equation.
pub trait RoutedBankObservation<B, Bank>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
{
    /// Whether this provider type uses independent storage.
    const ADDRESSABLE: bool;

    /// Borrows the independently retained storage owner, if present.
    fn addressable_storage(&self) -> Option<&Bank>;

    /// Observes the native bank report, when independently retained.
    fn addressable_report(&self) -> Result<Option<Bank::Report>, RoutedTextExecutionError>;
}

/// Family-blind backend visitor for an exact routed prediction target.
pub trait RoutedPredictionTargetVisitor<B, S, M, P>:
    Sized
    + RoutedSessionMechanisms<B, S>
    + RoutedTextResources<B, StorageError = Self::Error>
    + super::LocalPredictionResources
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::RuntimeState<B>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error: std::fmt::Display + From<eredu_core::parameters::ParameterError>;

    /// Called immediately before architecture construction begins.
    /// Supplies native residency data for the paired predictor.
    fn prediction_weight_options<A>(
        &self,
        extension: &mut <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
    ) -> Result<crate::ReplicatedWeightOptions<Self::WeightStorage>, Self::Error>
    where
        A: crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static;

    /// Concrete executable for this completed route.
    type BoundRoutedPrediction<A>: From<
            BoundRoutedText<
                B,
                A,
                P,
                Self::Mechanisms<A>,
                Self,
                (),
                super::SelectedPrediction<
                    <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
                >,
            >,
        > + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone;
}

/// Supplies exact state-profile visitors while architecture dispatch selects
/// the admitted routed prediction family.
pub trait RoutedPredictionProfileDispatcher<B, M, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
{
    /// Completed construction output shared by both admitted profiles.
    type Output;
    /// Mechanism-binding failure shared by both admitted profiles.
    type Error;
    /// State selected for mixed-attention and compressed-latent execution.
    type GatedState: crate::LayerRuntimeState<B>;
    /// State selected for pooling-attention execution.
    type PoolingState: crate::LayerRuntimeState<B>;
    /// Exact visitor for mixed-attention and compressed-latent profiles.
    type GatedVisitor: RoutedPredictionTargetVisitor<
        B,
        Self::GatedState,
        M,
        P,
        Output = Self::Output,
        Error = Self::Error,
    >;
    /// Exact visitor for the pooling-attention profile.
    type PoolingVisitor: RoutedPredictionTargetVisitor<
        B,
        Self::PoolingState,
        M,
        P,
        Output = Self::Output,
        Error = Self::Error,
    >;

    /// Consumes the dispatcher into its compressed-latent visitor.
    fn into_gated_visitor(self) -> Self::GatedVisitor;
    /// Consumes the dispatcher into its pooling-attention visitor.
    fn into_pooling_visitor(self) -> Self::PoolingVisitor;
}

/// Publishes an already constructed routed text architecture.
pub trait RoutedTextArchitectureVisitor<B, S, P>:
    Sized + RoutedSessionMechanisms<B, S> + RoutedTextResources<B, StorageError = Self::Error>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::GroupedNeuralBackend,
    S: crate::LayerRuntimeState<B>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error: std::fmt::Display;

    /// Called immediately before architecture construction begins.
    /// Concrete executable for this completed route.
    type BoundRouted<A>: From<BoundRoutedText<B, A, P, Self::Mechanisms<A>, Self, ()>>
        + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone;
}

/// Failure while architecture-owned routed dispatch invokes a mechanism visitor.
#[derive(Debug, thiserror::Error)]
pub enum RoutedTextDispatchError<E> {
    /// Architecture admission and selected realization disagreed.
    #[error("{0}")]
    Architecture(String),
    /// Backend mechanism binding failed.
    #[error("routed text mechanism binding failed: {0}")]
    Backend(E),
}

/// Physical inputs for binding a selected shared pool of experts and rows.
pub struct RoutedProviderBinding<'a> {
    /// Exact expert members to bind through the shared pool.
    pub members: Vec<crate::AddressableBankMember>,
    /// Scopes that use the shared pool for independently acquired experts.
    pub banks: Vec<RoutedBankId>,
    /// Selected row policy for the shared pool.
    pub rows: std::borrow::Cow<'a, crate::SelectedRowLookups>,
    /// Whether a row reader is required by the selected route.
    pub bind_rows: bool,
}

/// Projects the selected grouped and row providers onto one physical pool.
pub fn routed_provider_binding<'a, P: RoutedBankBinding>(
    banks: &BTreeMap<RoutedBankId, P>,
    residency: ParameterBankResidency,
    rows: Option<&'a crate::SelectedRowLookups>,
    row_support: &impl crate::RowLookupMechanismSupport,
) -> Result<RoutedProviderBinding<'a>, PreparedExecutionError<crate::RowLookupSelectionError>> {
    let addressable = matches!(residency, ParameterBankResidency::IndependentCache(_));
    if let (Some(rows), ParameterBankResidency::IndependentCache(options)) = (rows, residency) {
        if rows.options() != options {
            return Err(PreparedExecutionError::Backend(
                crate::RowLookupSelectionError::PoolMismatch,
            ));
        }
    }
    let bind_rows = rows.is_some();
    let rows = match rows {
        Some(rows) => std::borrow::Cow::Borrowed(rows),
        None => {
            let ParameterBankResidency::IndependentCache(options) = residency else {
                return Err(PreparedExecutionError::Architecture(
                    "provider binding has neither addressable banks nor rows".into(),
                ));
            };
            std::borrow::Cow::Owned(
                crate::SelectedRowLookups::select(Default::default(), options, 0, row_support)
                    .map_err(PreparedExecutionError::Backend)?,
            )
        }
    };
    let (members, banks) = if addressable {
        (
            banks
                .values()
                .flat_map(|bank| bank.addressable_members().iter().cloned())
                .collect(),
            banks.keys().copied().collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    Ok(RoutedProviderBinding {
        members,
        banks,
        rows,
        bind_rows,
    })
}

/// Checked text modules paired with every independently selected routed bank.
pub struct PreparedRoutedTextArchitecture<A, P> {
    text: PreparedReplicatedTextArchitecture<A>,
    bank_residency: crate::ParameterBankResidency,
    banks: BTreeMap<RoutedBankId, P>,
    rows: Option<crate::SelectedRowLookups>,
}

impl<A, P> PreparedRoutedTextArchitecture<A, P> {
    /// Couples checked text modules to admitted grouped banks and row lookups.
    pub fn new(
        text: PreparedReplicatedTextArchitecture<A>,
        bank_residency: ParameterBankResidency,
        banks: BTreeMap<RoutedBankId, P>,
        rows: Option<crate::SelectedRowLookups>,
    ) -> Self {
        Self {
            text,
            bank_residency,
            banks,
            rows,
        }
    }

    /// Exact row mechanisms selected before architecture/session construction.
    pub fn row_lookups(&self) -> Option<&crate::SelectedRowLookups> {
        self.rows.as_ref()
    }

    /// Shared text architecture and exact prepared sources.
    pub const fn text(&self) -> &PreparedReplicatedTextArchitecture<A> {
        &self.text
    }
    /// Placement selected for independently identified banks.
    pub const fn bank_residency(&self) -> crate::ParameterBankResidency {
        self.bank_residency
    }
    /// Complete bank facts consumed by reusable materialization mechanisms.
    pub const fn banks(&self) -> &BTreeMap<RoutedBankId, P> {
        &self.banks
    }
    /// Moves text modules and identified banks into execution.
    pub fn into_parts(
        self,
    ) -> (
        super::PreparedReplicatedTextModules<A>,
        crate::ParameterBankResidency,
        BTreeMap<RoutedBankId, P>,
        Option<crate::SelectedRowLookups>,
    ) {
        (
            self.text.into_modules(),
            self.bank_residency,
            self.banks,
            self.rows,
        )
    }
}

/// Checked composite architecture whose target decoder uses grouped providers.
pub struct PreparedRoutedCompositeTextArchitecture<A, C, P, R> {
    routed: PreparedRoutedTextArchitecture<A, P>,
    requirements: R,
    processor: crate::SelectedProcessorExecution,
    admission: C,
}

impl<A, C, P, R> PreparedRoutedCompositeTextArchitecture<A, C, P, R> {
    /// Couples the admitted processor and composite input policy to the routed target.
    pub fn new(
        routed: PreparedRoutedTextArchitecture<A, P>,
        requirements: R,
        processor: crate::SelectedProcessorExecution,
        admission: C,
    ) -> Self {
        Self {
            routed,
            requirements,
            processor,
            admission,
        }
    }

    /// Exact architecture, artifact, processor, and grouped-bank requirements.
    pub const fn requirements(&self) -> &R {
        &self.requirements
    }

    /// Authoritative processor mechanism selection.
    pub const fn processor(&self) -> &crate::SelectedProcessorExecution {
        &self.processor
    }

    /// Selected routed text, state, weight, and bank realization.
    pub const fn routed(&self) -> &PreparedRoutedTextArchitecture<A, P> {
        &self.routed
    }

    /// Architecture capability estimate presented by the shared session.
    pub const fn capability_estimate(&self) -> &crate::CapabilityEstimate {
        self.routed.text().capability_estimate()
    }

    /// Normalized model-type label presented by the shared session.
    pub fn effective_model_type(&self) -> &str {
        self.routed.text().effective_model_type()
    }

    /// Consumes the handoff into routed session-construction inputs.
    pub fn into_parts(
        self,
    ) -> (
        PreparedRoutedTextArchitecture<A, P>,
        crate::SelectedProcessorExecution,
        C,
    ) {
        (self.routed, self.processor, self.admission)
    }
}

impl<A, P: RoutedBankBinding> PreparedRoutedTextArchitecture<A, P> {
    fn validate_row_provider<B: eredu_nn::NeuralBackend, L: crate::RowLookupProvider<B>>(
        &self,
        provider: &Option<L>,
    ) -> Result<(), String> {
        match (&self.rows, provider) {
            (None, None) => Ok(()),
            (Some(selected), Some(provider)) => {
                for id in selected.prepared().entries().keys() {
                    if !provider.has_row_parameter(id) {
                        return Err(format!("missing selected row provider {id}"));
                    }
                }
                Ok(())
            }
            _ => Err("row provider presence differs from the cold selection".into()),
        }
    }
    /// Constructs resident execution through the shared session driver, returning
    /// moved auxiliary banks for prediction or other separately materialized roles.
    /// All providers are constructed once before splitting their ownership.
    pub fn construct_resident_session<B, M, L>(
        self,
        mechanisms: M,
        weights: crate::ReplicatedWeightOptions<M::WeightStorage>,
        rows: Option<L>,
        auxiliary_banks: &[RoutedBankId],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            crate::ReplicatedTextSession<
                A,
                B,
                M,
                crate::RoutedReplicatedTextExecution<
                    crate::ParameterProviders<crate::RoutedBankProviders<P::Resident>, Option<L>>,
                >,
            >,
            Option<crate::RoutedBankProviders<P::Resident>>,
        ),
        String,
    >
    where
        B: eredu_core::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
            > + GroupedNeuralBackend,
        M: crate::ReplicatedTextSessionMechanisms<A, B>,
        A: crate::LayeredArchitecture<B, M::State> + crate::RoutedLayeredArchitecture<B, M::State>,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
        P::Resident: crate::ParameterProvider<B>,
        L: crate::RowLookupProvider<B>,
    {
        self.validate_row_provider::<B, L>(&rows)?;
        let (mut modules, residency, banks, _) = self.into_parts();
        if residency != crate::ParameterBankResidency::WithLayer {
            return Err("selected banks are not resident with their layers".into());
        }
        let providers = banks
            .into_iter()
            .map(|(id, bank)| bank.bind_resident().map(|p| (id, p)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let mut providers =
            crate::RoutedBankProviders::new(providers).map_err(|e| e.to_string())?;
        let auxiliary = providers
            .split_off(auxiliary_banks)
            .map_err(|e| e.to_string())?;
        crate::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            weights,
            crate::RoutedReplicatedTextExecution::new(crate::ParameterProviders {
                grouped: providers,
                rows,
            }),
            context,
        )
        .map(|session| (session, auxiliary))
        .map_err(|e| e.to_string())
    }
    /// Constructs addressable execution and returns separately owned auxiliary banks.
    /// Native banks must cover the complete selected collection and share the admitted
    /// residency accounting before ownership is split; budgets are not reconstructed.
    pub fn construct_addressable_session<B, M, Bank, Movement, L>(
        self,
        mechanisms: M,
        weights: crate::ReplicatedWeightOptions<M::WeightStorage>,
        native_banks: BTreeMap<RoutedBankId, (Bank, Movement)>,
        rows: Option<L>,
        auxiliary_banks: &[RoutedBankId],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            crate::ReplicatedTextSession<
                A,
                B,
                M,
                crate::RoutedReplicatedTextExecution<
                    crate::ParameterProviders<
                        crate::RoutedBankProviders<P::Addressable<B, Bank, Movement>>,
                        Option<L>,
                    >,
                >,
            >,
            Option<crate::RoutedBankProviders<P::Addressable<B, Bank, Movement>>>,
        ),
        String,
    >
    where
        B: eredu_core::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
            > + GroupedNeuralBackend,
        M: crate::ReplicatedTextSessionMechanisms<A, B>,
        A: crate::LayeredArchitecture<B, M::State> + crate::RoutedLayeredArchitecture<B, M::State>,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
        L: crate::RowLookupProvider<B>,
        Bank: AddressableGroupedBank<B>,
        Bank::Error: std::fmt::Display,
        Movement: IndexedMovement<B>,
        Movement::Error: std::fmt::Display,
    {
        self.validate_row_provider::<B, L>(&rows)?;
        let (mut modules, residency, banks, _) = self.into_parts();
        let crate::ParameterBankResidency::IndependentCache(options) = residency else {
            return Err("selected banks are not independently addressable".into());
        };
        if native_banks.keys().ne(banks.keys()) {
            return Err("materialized bank identities differ from selected banks".into());
        }
        let mut native_banks = native_banks;
        let providers = banks
            .into_iter()
            .map(|(id, selected)| {
                let (bank, movement) = native_banks.remove(&id).expect("checked bank identities");
                selected
                    .bind_addressable::<B, _, _>(bank, movement, options)
                    .map(|p| (id, p))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let mut providers =
            crate::RoutedBankProviders::new(providers).map_err(|e| e.to_string())?;
        let auxiliary = providers
            .split_off(auxiliary_banks)
            .map_err(|e| e.to_string())?;
        crate::construct_replicated_text_session_with_execution(
            modules.take_architecture(),
            modules.take_source_architecture(),
            modules.take_contract(),
            mechanisms,
            weights,
            crate::RoutedReplicatedTextExecution::new(crate::ParameterProviders {
                grouped: providers,
                rows,
            }),
            context,
        )
        .map(|session| (session, auxiliary))
        .map_err(|e| e.to_string())
    }
}

pub use eredu_core::preparation::rows::NativeRoutedStorage;

/// Bank selected and installed by the portable routed driver.
type RoutedBank<B, N> = crate::expert::SharedAddressableParameterBank<
    <N as NativeRoutedStorage<B>>::BankMechanisms,
    <B as NeuralBackend>::Tensor,
>;

/// Row provider assembled by the portable routed driver.
type RoutedRows<B, N> = crate::row_lookup::BoundRowLookups<
    B,
    <N as NativeRoutedStorage<B>>::BankMechanisms,
    <N as NativeRoutedStorage<B>>::Decoder,
    <B as NeuralBackend>::ParallelContext,
>;

/// Portable binding of selected routed bank and row providers.
pub trait RoutedStorage<B: GroupedNeuralBackend> {
    /// Portable grouped bank owner.
    type Bank: AddressableGroupedBank<B, Error: std::fmt::Display>
        + TextParameterBanks
        + crate::expert::ScopedParameterBank
        + From<
            crate::expert::AddressableBankStorage<
                crate::residency::ResidencyDriver<Self::BankMechanisms>,
            >,
        > + 'static;
    /// Native grouped movement.
    type Movement: IndexedMovement<B, Error: std::fmt::Display> + 'static;
    /// Portable row lookup over the admitted pool.
    type Rows: crate::RowLookupProvider<B>
        + crate::row_lookup::BindLocalRows<Self::Bank, Self::RowBinding, Error = Self::StorageError>;
    /// Native context and decoder for already selected rows.
    type RowBinding;
    /// Native storage binding failure.
    type StorageError: std::fmt::Display + From<eredu_core::parameters::ParameterError>;

    /// Supplies exact native row lookup capabilities.
    fn row_lookup_support(&self) -> impl eredu_core::preparation::rows::NativeRowLookupSupport;

    /// Native checkpoint parameter format support for this pool.
    type BankParameters: eredu_core::checkpoint_mechanisms::CheckpointParameterBackend;

    /// Native storage and transfer mechanisms for this bank.
    type BankMechanisms: eredu_core::residency::ResidencyStorageMechanisms<
        Storage: eredu_core::residency::ResidencyStorage<Backend = Self::BankParameters>,
    >;

    /// Exact native placement for the admitted bank.
    fn bank_residency_context(
        &self,
    ) -> <Self::BankMechanisms as eredu_core::residency::ResidencyStorageMechanisms>::Context;

    /// Supplies native recipe lowering and bounded conversion.
    fn bank_materializer(
        &self,
    ) -> impl eredu_core::checkpoint_mechanisms::CheckpointEncodingMaterializer<
        Parameters = Self::BankParameters,
    >;

    /// Supplies the queue-bound indexed movement primitives.
    fn indexed_movement(&self) -> Self::Movement;

    /// Supplies tensor conversion and scalar materialization on native queues.
    fn row_binding(&self) -> Self::RowBinding;
}

impl<B, N> RoutedStorage<B> for N
where
    B: GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_core::checkpoint_mechanisms::CheckpointParameterBackend,
    B::ParallelContext: Clone + 'static,
    N: NativeRoutedStorage<B>,
    RoutedBank<B, N>: AddressableGroupedBank<B, Error: std::fmt::Display> + TextParameterBanks,
    RoutedRows<B, N>: crate::RowLookupProvider<B>
        + crate::row_lookup::BindLocalRows<RoutedBank<B, N>, N::RowBinding, Error = N::StorageError>,
{
    type Bank = RoutedBank<B, N>;
    type Movement = N::Movement;
    type Rows = RoutedRows<B, N>;
    type RowBinding = N::RowBinding;
    type StorageError = N::StorageError;
    type BankParameters = N::BankParameters;
    type BankMechanisms = N::BankMechanisms;

    fn row_lookup_support(&self) -> impl eredu_core::preparation::rows::NativeRowLookupSupport {
        NativeRoutedStorage::row_lookup_support(self)
    }

    fn bank_residency_context(
        &self,
    ) -> <Self::BankMechanisms as eredu_core::residency::ResidencyStorageMechanisms>::Context {
        NativeRoutedStorage::bank_residency_context(self)
    }

    fn bank_materializer(
        &self,
    ) -> impl eredu_core::checkpoint_mechanisms::CheckpointEncodingMaterializer<
        Parameters = Self::BankParameters,
    > {
        NativeRoutedStorage::bank_materializer(self)
    }

    fn indexed_movement(&self) -> Self::Movement {
        NativeRoutedStorage::indexed_movement(self)
    }

    fn row_binding(&self) -> Self::RowBinding {
        NativeRoutedStorage::row_binding(self)
    }
}

/// Installs each selected provider over one physical pool.
pub fn bind_routed_providers<B: GroupedNeuralBackend, V: RoutedStorage<B>>(
    visitor: &V,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    selection: RoutedProviderBinding<'_>,
) -> Result<
    (
        BTreeMap<RoutedBankId, (V::Bank, V::Movement)>,
        Option<V::Rows>,
    ),
    PreparedExecutionError<V::StorageError>,
> {
    let prepared = crate::expert::prepare_addressable_bank_members(
        &selection.members,
        source,
        &selection.rows,
        &visitor.row_lookup_support(),
        &visitor.bank_materializer(),
    )
    .map_err(PreparedExecutionError::Binding)?;
    let installed = prepared
        .install::<crate::residency::ResidencyDriver<V::BankMechanisms>>(
            visitor.bank_residency_context(),
        )
        .map_err(eredu_core::BackendFailure::from_error)
        .map_err(PreparedExecutionError::Binding)?;
    use crate::{expert::ScopedParameterBank as _, row_lookup::BindLocalRows as _};
    let pool = V::Bank::from(installed);
    let rows = selection
        .bind_rows
        .then(|| V::Rows::bind_local_rows(&pool, &selection.rows, &visitor.row_binding()))
        .transpose()
        .map_err(PreparedExecutionError::Backend)?;
    let banks = selection
        .banks
        .into_iter()
        .map(|id| {
            pool.scope_to(id.value() as usize)
                .map(|scope| (id, (scope, visitor.indexed_movement())))
        })
        .collect::<Result<_, _>>()
        .map_err(|error| PreparedExecutionError::Backend(error.into()))?;
    Ok((banks, rows))
}

type NativeRoutedBank<B, V> = <V as RoutedStorage<B>>::Bank;
type NativeRoutedMovement<B, V> = <V as RoutedStorage<B>>::Movement;
type NativeRoutedRows<B, V> = <V as RoutedStorage<B>>::Rows;

type RoutedBankExecution<P, Rows> = RoutedReplicatedTextExecution<
    crate::ParameterProviders<crate::RoutedBankProviders<P>, Option<Rows>>,
>;

/// Routed session with directly resident experts and its selected row provider.
pub type ResidentRoutedSession<B, A, P, M, V> = ReplicatedTextSession<
    A,
    B,
    M,
    RoutedBankExecution<<P as RoutedBankBinding>::Resident, NativeRoutedRows<B, V>>,
>;

/// Routed session with independently addressable experts and native movement.
pub type AddressableRoutedSession<B, A, P, M, V> = ReplicatedTextSession<
    A,
    B,
    M,
    RoutedBankExecution<AddressableRoutedProvider<B, P, V>, NativeRoutedRows<B, V>>,
>;

type AddressableRoutedProvider<B, P, V> =
    <P as RoutedBankBinding>::Addressable<B, NativeRoutedBank<B, V>, NativeRoutedMovement<B, V>>;

/// The session and separately owned auxiliary banks selected by portable construction.
pub enum BoundRoutedSession<B, A, P, M, V>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend,
    V: RoutedStorage<B>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    M: ReplicatedTextSessionMechanisms<A, B, PolicyError: std::fmt::Display>,
    A: LayeredArchitecture<B, M::State, Error = eredu_nn::Error>
        + RoutedLayeredArchitecture<B, M::State>,
{
    /// The provider owns experts with their layers.
    Resident(
        ResidentRoutedSession<B, A, P, M, V>,
        Option<crate::RoutedBankProviders<P::Resident>>,
    ),
    /// The provider separately acquires expert weights.
    Addressable(
        AddressableRoutedSession<B, A, P, M, V>,
        Option<crate::RoutedBankProviders<AddressableRoutedProvider<B, P, V>>>,
    ),
}

/// Native resources retained alongside a routed text executable.
pub trait RoutedTextResources<B: GroupedNeuralBackend>:
    RoutedStorage<B, Bank: Clone + Into<super::session_erasure::BankOf<Self::RoutedResources>>>
{
    /// Exact native execution queue and materialization resources.
    type RoutedResources: super::text_invocation::TextInvocationResources;

    /// Opens native resources independently of the selected weight pools.
    fn routed_resources(&self) -> Self::RoutedResources;
}

impl<B, N> RoutedTextResources<B> for N
where
    B: GroupedNeuralBackend,
    N: super::LocalTextResources<LocalResources: super::text_invocation::TextInvocationResources>
        + RoutedStorage<B, Bank: Clone + Into<super::session_erasure::BankOf<N::LocalResources>>>,
{
    type RoutedResources = N::LocalResources;

    fn routed_resources(&self) -> Self::RoutedResources {
        self.local_resources()
    }
}

/// A bound executable whose provider type follows the admitted bank residency.
pub enum RoutedExecutable<Resident, Addressable> {
    /// Grouped weights are retained with their layers.
    Resident(Resident),
    /// Grouped weights are independently addressable.
    Addressable(Addressable),
}

/// A routed executable with exact native resources and selected prompt ingress.
pub type BoundRoutedText<B, A, P, M, V, I, Q = super::NoSelectedPrediction> = RoutedExecutable<
    super::BoundTextSession<
        ResidentRoutedSession<B, A, P, M, V>,
        I,
        <V as RoutedTextResources<B>>::RoutedResources,
        Q,
    >,
    super::BoundTextSession<
        AddressableRoutedSession<B, A, P, M, V>,
        I,
        <V as RoutedTextResources<B>>::RoutedResources,
        Q,
    >,
>;

fn addressable_banks<'a, B, P, Bank>(
    providers: &'a crate::RoutedBankProviders<P>,
) -> impl Iterator<Item = (RoutedBankId, &'a Bank)>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B> + 'a,
    P: RoutedBankObservation<B, Bank>,
{
    providers
        .banks()
        .iter()
        .filter_map(|(id, provider)| provider.addressable_storage().map(|bank| (*id, bank)))
}

/// Binds an admitted routed executable to native queue and bank resources.
pub fn bind_routed_text<B, A, P, M, V, I>(
    visitor: &V,
    session: BoundRoutedSession<B, A, P, M, V>,
    facts: PreparedTextSessionFacts,
    ingress: I,
) -> BoundRoutedText<B, A, P, M, V, I>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    M: ReplicatedTextSessionMechanisms<
        A,
        B,
        PolicyError: std::fmt::Display,
        Error: std::fmt::Display,
    >,
    A: LayeredArchitecture<B, M::State, Error = eredu_nn::Error>
        + RoutedLayeredArchitecture<B, M::State>,
    V: RoutedTextResources<B>,
{
    match session {
        BoundRoutedSession::Resident(session, _) => {
            let resources = visitor.routed_resources();
            RoutedExecutable::Resident(facts.bind(
                session,
                ingress,
                resources,
                super::NoSelectedPrediction,
                true,
            ))
        }
        BoundRoutedSession::Addressable(session, _) => {
            let banks = addressable_banks::<B, _, V::Bank>(
                &session.execution_strategy().provider().grouped,
            )
            .map(|(id, bank)| (id, bank.clone().into()))
            .collect();
            let resources = visitor.routed_resources();
            RoutedExecutable::Addressable(
                facts
                    .bind(
                        session,
                        ingress,
                        resources,
                        super::NoSelectedPrediction,
                        true,
                    )
                    .with_weight_pools(banks, None),
            )
        }
    }
}

/// Installs a paired prediction extension before erasing the routed target.
pub fn install_routed_prediction<B, A, P, M, V, I, X, Q>(
    executable: BoundRoutedText<B, A, P, M, V, I>,
    extension: Q,
    binding: super::PredictionBinding,
) -> Result<
    BoundRoutedText<B, A, P, M, V, I, super::SelectedPrediction<Q>>,
    eredu_core::parameters::ParameterError,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    M: ReplicatedTextSessionMechanisms<
        A,
        B,
        PolicyError: std::fmt::Display,
        Error: std::fmt::Display,
    >,
    A: LayeredArchitecture<B, M::State, Error = eredu_nn::Error>
        + RoutedLayeredArchitecture<B, M::State>,
    V: RoutedTextResources<B>,
    X: crate::speculative::PredictionExtensionMaterializer<B>,
    Q: crate::speculative::MaterializedPredictionExecutor<A, B, X>,
{
    let (selected, capability) = binding.into_parts();
    let prediction = super::SelectedPrediction {
        extension,
        selected,
    };
    match executable {
        RoutedExecutable::Resident(executable) => executable
            .install_prediction::<X, _>(prediction, capability)
            .map(RoutedExecutable::Resident),
        RoutedExecutable::Addressable(executable) => executable
            .install_prediction::<X, _>(prediction, capability)
            .map(RoutedExecutable::Addressable),
    }
}

/// Constructs the selected routed session from its admitted sources and native operations.
pub fn construct_selected_routed_session<B, S, A, P, V>(
    visitor: &V,
    prepared: PreparedRoutedTextArchitecture<A, P>,
    mechanisms: V::Mechanisms<A>,
    weights: crate::ReplicatedWeightOptions<V::WeightStorage>,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    auxiliary: &[crate::RoutedBankId],
) -> Result<
    (
        BoundRoutedSession<B, A, P, V::Mechanisms<A>, V>,
        PreparedTextSessionFacts,
    ),
    PreparedExecutionError<V::StorageError>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend,
    S: crate::RuntimeState<B>,
    A: LayeredArchitecture<B, S, Error = eredu_nn::Error>
        + 'static
        + RoutedLayeredArchitecture<B, S>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    V: RoutedSessionMechanisms<B, S> + RoutedStorage<B>,
{
    let facts = PreparedTextSessionFacts::from_prepared(prepared.text());
    let tier = prepared.bank_residency();
    let (banks, rows) = if matches!(tier, ParameterBankResidency::IndependentCache(_))
        || prepared.row_lookups().is_some()
    {
        let selection = routed_provider_binding(
            prepared.banks(),
            tier,
            prepared.row_lookups(),
            &visitor.row_lookup_support(),
        )
        .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        bind_routed_providers(visitor, source, selection)?
    } else {
        (Default::default(), None)
    };
    let context = visitor.session_context();
    let session = match tier {
        ParameterBankResidency::WithLayer => {
            if !banks.is_empty() {
                return Err(PreparedExecutionError::Architecture(
                    "resident grouped weights received addressable banks".into(),
                ));
            }
            let (mut session, auxiliary) = prepared
                .construct_resident_session::<B, _, V::Rows>(
                    mechanisms, weights, rows, auxiliary, context,
                )
                .map_err(PreparedExecutionError::Architecture)?;
            session
                .install_bank_parameters([])
                .map_err(parameter_binding_error)?;
            BoundRoutedSession::Resident(session, auxiliary)
        }
        ParameterBankResidency::IndependentCache(_) => {
            let (mut session, auxiliary) = prepared
                .construct_addressable_session::<B, _, V::Bank, V::Movement, V::Rows>(
                    mechanisms, weights, banks, rows, auxiliary, context,
                )
                .map_err(PreparedExecutionError::Architecture)?;
            let mut members = Vec::new();
            for (_, bank) in
                addressable_banks::<B, _, V::Bank>(&session.execution_strategy().provider().grouped)
            {
                members.extend(bank.bank_members().map_err(parameter_binding_error)?);
            }
            session
                .install_bank_parameters(members)
                .map_err(parameter_binding_error)?;
            BoundRoutedSession::Addressable(session, auxiliary)
        }
    };
    Ok((session, facts))
}

fn parameter_binding_error<E>(
    error: eredu_core::parameters::ParameterError,
) -> PreparedExecutionError<E> {
    PreparedExecutionError::Binding(eredu_core::BackendFailure::from_error(error))
}

/// Constructs routed composite execution and retains exact processor and admission facts.
pub fn construct_selected_routed_composite_session<B, S, A, P, Req, V>(
    visitor: &V,
    prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig, P, Req>,
    mechanisms: V::Mechanisms<A>,
    weights: crate::ReplicatedWeightOptions<V::WeightStorage>,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    auxiliary: &[crate::RoutedBankId],
) -> Result<
    (
        BoundRoutedSession<B, A, P, V::Mechanisms<A>, V>,
        PreparedCompositeSessionFacts<A::AdmissionConfig>,
    ),
    PreparedExecutionError<V::StorageError>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend,
    S: crate::LayerRuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + RoutedLayeredArchitecture<B, S>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    V: RoutedSessionMechanisms<B, S> + RoutedStorage<B>,
{
    let (routed, processor, admission) = prepared.into_parts();
    let (session, facts) =
        construct_selected_routed_session(visitor, routed, mechanisms, weights, source, auxiliary)?;
    Ok((
        session,
        PreparedCompositeSessionFacts::new(facts, processor, admission),
    ))
}

/// Native resources supplied to portable session construction.
pub trait RoutedSessionMechanisms<B, S>:
    super::LocalTextResources<
    LocalResources: super::session_erasure::TextErasureResources<
        Types: super::session_erasure::TextErasureTypes<Tensor = B::Tensor, Context = B::Executor>,
    >,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: crate::RuntimeState<B>,
{
    /// Residency authority shared with prediction modules.
    type WeightStorage: crate::WeightResidencyStorage;

    /// Native parameter and state operations for the selected architecture.
    type Mechanisms<A>: crate::replicated_session::ReplicatedTextSnapshotMechanisms<
        A,
        B,
        State = S,
        WeightStorage = Self::WeightStorage,
        Error: std::error::Error + Send + Sync + 'static,
        PolicyError = <super::session_erasure::LocalTextTypes<Self> as super::TextExecutableTypes>::Error,
        StateReport: super::session_erasure::TextStateReport,
    > + 'static
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static;

    /// Borrows the exact execution context.
    fn session_context(&self) -> &B::Executor;

    /// Supplies native mechanisms for the retained target source.
    fn session_mechanisms<A>(
        &self,
        source: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Self::Mechanisms<A>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static;
}

impl<B, S, N> RoutedSessionMechanisms<B, S> for N
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    B::Executor: Sized,
    S: crate::LayerRuntimeState<B>
        + crate::state::PortableTextState<
            B,
            <N::Values as eredu_core::backend::NativeTextValues>::Cache,
        > + 'static,
    N: eredu_core::checkpoint_mechanisms::TextComponentResources<
            Values: eredu_core::checkpoint_mechanisms::NativeTextWeights<
                Neural = B,
                Tensor = B::Tensor,
                Context = B::Executor,
                Error: From<eredu_core::parameters::ParameterError>,
            > + super::TextPredictionPrimitives,
            Components: crate::ParameterUnitMechanisms<
                B,
                UnitError = <N::Values as eredu_core::backend::NativeTextValues>::Error,
            > + 'static,
        > + super::LocalTextResources<
            LocalResources: super::session_erasure::TextErasureResources<
                Types: super::session_erasure::TextErasureTypes<
                    Tensor = B::Tensor,
                    Context = B::Executor,
                    Error = <N::Values as eredu_core::backend::NativeTextValues>::Error,
                >,
            >,
        >,
{
    type WeightStorage = <N::Components as eredu_core::checkpoint_mechanisms::unit_binding::ParameterStorageBinding<B>>::ResidencyMechanisms;
    type Mechanisms<A>
        = super::route_assembly::weights::Weights<B, S, N>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static;

    fn session_context(&self) -> &B::Executor {
        self.component_execution()
    }

    fn session_mechanisms<A>(
        &self,
        source: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Self::Mechanisms<A>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    {
        super::route_assembly::weights::TextRouteWeights::new(source, self.text_components())
    }
}

/// Backend-generic visitor over one exact replicated composite architecture.
pub trait CompositeTextArchitectureVisitor<B, S, P, R, Q>:
    Sized
    + RoutedSessionMechanisms<B, S>
    + RoutedTextResources<B, StorageError = Self::Error>
    + super::LocalTextResources
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + Clone,
    S: crate::LayerRuntimeState<B>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error: std::fmt::Display;

    /// Concrete executable for this completed route.
    type BoundComposite<A>: From<
            super::BoundTextSession<
                ReplicatedTextSession<A, B, Self::Mechanisms<A>>,
                super::CompositeSessionIngress<A::AdmissionConfig>,
                Self::LocalResources,
                super::NoSelectedPrediction,
            >,
        > + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + 'static,
        A::InputPartPlan: 'static;

    /// Concrete executable for this completed route.
    type BoundCompositeRouted<A>: From<
            BoundRoutedText<
                B,
                A,
                P,
                Self::Mechanisms<A>,
                Self,
                super::CompositeSessionIngress<A::AdmissionConfig>,
            >,
        > + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::RoutedLayeredArchitecture<B, S>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone;
}

/// Binds portable composite ingress and the exact native routed resources.
pub fn bind_routed_composite<B, A, P, M, V>(
    visitor: &V,
    session: BoundRoutedSession<B, A, P, M, V>,
    facts: PreparedCompositeSessionFacts<A::AdmissionConfig>,
) -> BoundRoutedText<B, A, P, M, V, super::CompositeSessionIngress<A::AdmissionConfig>>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    M: ReplicatedTextSessionMechanisms<
        A,
        B,
        PolicyError: std::fmt::Display,
        Error: std::fmt::Display,
    >,
    A: crate::composite_execution::CompositeArchitecture<B, M::State, Error = eredu_nn::Error>
        + RoutedLayeredArchitecture<B, M::State>,
    V: RoutedTextResources<B>,
{
    bind_routed_text(
        visitor,
        session,
        facts.text,
        super::CompositeSessionIngress::new(facts.admission, facts.processor),
    )
}

/// Family-blind backend visitor for an exact composite prediction target.
pub trait CompositePredictionTargetVisitor<B, S, M, P, R, Q>:
    Sized
    + RoutedSessionMechanisms<B, S>
    + RoutedTextResources<B, StorageError = Self::Error>
    + super::LocalPredictionResources
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: crate::LayerRuntimeState<B>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error: std::fmt::Display + From<eredu_core::parameters::ParameterError>;

    /// Records that validated dispatch is about to construct architecture modules.
    /// Configures native resources for the paired prediction extension.
    fn prediction_weight_options<A>(
        &self,
        extension: &mut <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
    ) -> Result<crate::ReplicatedWeightOptions<Self::WeightStorage>, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static;

    /// Concrete executable for this completed route.
    type BoundCompositePrediction<A>: From<
            super::BoundTextSession<
                ReplicatedTextSession<A, B, Self::Mechanisms<A>>,
                super::CompositeSessionIngress<A::AdmissionConfig>,
                Self::LocalResources,
                super::SelectedPrediction<
                    <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
                >,
            >,
        > + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::InputPartPlan: 'static;

    /// Concrete executable for this completed route.
    type BoundCompositeRoutedPrediction<A>: From<
            BoundRoutedText<
                B,
                A,
                P,
                Self::Mechanisms<A>,
                Self,
                super::CompositeSessionIngress<A::AdmissionConfig>,
                super::SelectedPrediction<
                    <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
                >,
            >,
        > + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::RoutedLayeredArchitecture<B, S>
            + crate::speculative::PredictionTarget<B>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone;
}

/// Binds direct composite ingress and installs its paired prediction catalog.
pub fn bind_local_composite_prediction<B, S, M, P, R, Q, A, V>(
    visitor: V,
    session: ReplicatedTextSession<A, B, V::Mechanisms<A>>,
    extension: <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
    facts: PreparedCompositeSessionFacts<A::AdmissionConfig>,
) -> Result<V::Output, V::Error>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    S: crate::LayerRuntimeState<B>,
    P: RoutedBankBinding<Resident: crate::ParameterProvider<B>>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + crate::speculative::PredictionTarget<B>
        + 'static,
    A::InputPartPlan: 'static,
    V: CompositePredictionTargetVisitor<B, S, M, P, R, Q>,
{
    let resources = visitor.local_resources();
    let (selected, capability) = visitor.into_prediction_binding().into_parts();
    let executable = facts
        .bind(session, resources, true)
        .install_prediction::<M, _>(
            super::SelectedPrediction {
                extension,
                selected,
            },
            capability,
        )?;
    Ok(crate::executable_binding::install_text_route::<
        V::BoundCompositePrediction<A>,
        _,
        _,
    >(executable))
}

/// Native resources for materializing an independently prepared routed predictor.
pub trait RetainedRoutedPredictionMechanisms<B, S>:
    Sized + RoutedStorage<B, StorageError: From<eredu_core::BackendFailure>> + RoutedSessionMechanisms<B, S, WeightStorage = <Self::Materializer as eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<B>>::ResidencyMechanisms> + RoutedTextResources<B>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::LayerRuntimeState<B> + crate::PredictionModelState<B>,
{
    /// Native extension materialization and state.
    type Materializer: crate::speculative::PredictionExtensionMaterializer<
            B,
            ModelState = S,
            Error = Self::StorageError,
        > + 'static;

    /// Context for reading and staging the predictor's source weights.
    fn prediction_source_context(&self) -> &B::Executor;


}

impl<B, S, N, M> RetainedRoutedPredictionMechanisms<B, S> for N
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::LayerRuntimeState<B> + crate::PredictionModelState<B>,
    M: crate::speculative::PredictionExtensionMaterializer<
            B,
            ModelState = S,
            Error = N::StorageError,
        > + 'static,
    N: eredu_core::checkpoint_mechanisms::TextComponentResources<
            Values: super::TextPredictionPrimitives<Materializer = M, Context = B::Executor>,
        > + RoutedStorage<B, StorageError: From<eredu_core::BackendFailure>>
        + RoutedSessionMechanisms<B, S, WeightStorage = M::ResidencyMechanisms>
        + RoutedTextResources<B>,
{
    type Materializer = M;

    fn prediction_source_context(&self) -> &B::Executor {
        self.component_weights()
    }
}

/// A bound routed executable with the predictor specialized to the same bank placement.
pub type BoundRetainedRoutedPrediction<B, S, A, P, W, V, I> = RoutedExecutable<
    super::BoundTextSession<
        ResidentRoutedSession<B, A, P, <V as RoutedSessionMechanisms<B, S>>::Mechanisms<A>, V>,
        I,
        <V as RoutedTextResources<B>>::RoutedResources,
        super::SelectedPrediction<
            <W as crate::speculative::PreparedRoutedPrediction<B, A>>::Executor<
                <V as RetainedRoutedPredictionMechanisms<B, S>>::Materializer,
                crate::RoutedBankProviders<<P as RoutedBankBinding>::Resident>,
            >,
        >,
    >,
    super::BoundTextSession<
        AddressableRoutedSession<B, A, P, <V as RoutedSessionMechanisms<B, S>>::Mechanisms<A>, V>,
        I,
        <V as RoutedTextResources<B>>::RoutedResources,
        super::SelectedPrediction<
            <W as crate::speculative::PreparedRoutedPrediction<B, A>>::Executor<
                <V as RetainedRoutedPredictionMechanisms<B, S>>::Materializer,
                crate::RoutedBankProviders<AddressableRoutedProvider<B, P, V>>,
            >,
        >,
    >,
>;

/// Binds the target and its prediction after moving the selected auxiliary banks.
pub fn construct_retained_routed_prediction<B, S, A, P, W, V, I>(
    visitor: &V,
    prepared: PreparedRoutedTextArchitecture<A, P>,
    prediction: W,
    target_source: eredu_checkpoint::store::SharedCheckpointSource,
    provider_source: eredu_checkpoint::store::SharedCheckpointSource,
    ingress: I,
    binding: super::PredictionBinding,
) -> Result<
    BoundRetainedRoutedPrediction<B, S, A, P, W, V, I>,
    PreparedExecutionError<V::StorageError>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::LayerRuntimeState<B>
        + crate::PredictionModelState<B, LayerState: crate::RuntimeAppendStreams<B>>,
    A: LayeredArchitecture<B, S, Error = eredu_nn::Error>
        + 'static
        + RoutedLayeredArchitecture<B, S>,
    P: RoutedBankBinding<
        Resident: crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    >,
    AddressableRoutedProvider<B, P, V>:
        crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    W: crate::speculative::PreparedRoutedPrediction<B, A>,
    V: RetainedRoutedPredictionMechanisms<B, S>,
{
    use eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization;
    let auxiliary = prediction.banks();
    let mut context = V::Materializer::bind_source(
        prediction.source().clone(),
        visitor.session_context(),
        visitor.prediction_source_context(),
    );
    let mut extension = prediction
        .materialize::<V::Materializer>(&mut context, |_, selected| {
            crate::state::allocate_selected_prediction_state::<B, V::Materializer>(selected)
        })
        .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
    drop(context);
    let mechanisms = visitor.session_mechanisms(target_source);
    let residency = crate::speculative::prepare_prediction_residency::<A, B, V::Materializer, _>(
        &mut extension,
    )
    .map_err(PreparedExecutionError::Backend)?;

    let (session, facts) = construct_selected_routed_session(
        visitor,
        prepared,
        mechanisms,
        crate::ReplicatedWeightOptions {
            supplementary: residency,
            ..Default::default()
        },
        provider_source,
        &auxiliary,
    )?;
    let missing = || {
        PreparedExecutionError::Architecture(
            "prediction banks were not moved from target construction".into(),
        )
    };
    let (selected, capability) = binding.into_parts();
    match session {
        BoundRoutedSession::Resident(session, provider) => {
            let extension =
                W::with_provider::<V::Materializer, _>(extension, provider.ok_or_else(missing)?);
            let resources = visitor.routed_resources();
            facts
                .bind(
                    session,
                    ingress,
                    resources,
                    super::NoSelectedPrediction,
                    true,
                )
                .install_prediction::<V::Materializer, _>(
                    super::SelectedPrediction {
                        extension,
                        selected,
                    },
                    capability,
                )
                .map(RoutedExecutable::Resident)
        }
        BoundRoutedSession::Addressable(session, provider) => {
            let extension =
                W::with_provider::<V::Materializer, _>(extension, provider.ok_or_else(missing)?);
            let banks = addressable_banks::<B, _, V::Bank>(
                &session.execution_strategy().provider().grouped,
            )
            .map(|(id, bank)| (id, bank.clone().into()))
            .collect();
            let resources = visitor.routed_resources();
            facts
                .bind(
                    session,
                    ingress,
                    resources,
                    super::NoSelectedPrediction,
                    true,
                )
                .with_weight_pools(banks, None)
                .install_prediction::<V::Materializer, _>(
                    super::SelectedPrediction {
                        extension,
                        selected,
                    },
                    capability,
                )
                .map(RoutedExecutable::Addressable)
        }
    }
    .map_err(parameter_binding_error)
}

/// Native publication of a routed text target and its independently prepared predictor.
pub trait RetainedRoutedTextPrediction<B, S, P>:
    RetainedRoutedPredictionMechanisms<B, S> + RoutedTextArchitectureVisitor<B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::LayerRuntimeState<B>
        + crate::PredictionModelState<B, LayerState: crate::RuntimeAppendStreams<B>>,
    P: RoutedBankBinding<
        Resident: crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    >,
    AddressableRoutedProvider<B, P, Self>:
        crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
{
    /// Concrete executable for this completed route.
    type BoundRetainedPrediction<A, W>: From<BoundRetainedRoutedPrediction<B, S, A, P, W, Self, ()>>
        + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        W: crate::speculative::PreparedRoutedPrediction<B, A>;
}

/// Native publication of a composite target and its independently prepared predictor.
pub trait RetainedCompositePrediction<B, S, P>: RetainedRoutedPredictionMechanisms<B, S>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::LayerRuntimeState<B>
        + crate::PredictionModelState<B, LayerState: crate::RuntimeAppendStreams<B>>,
    P: RoutedBankBinding<
        Resident: crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    >,
    AddressableRoutedProvider<B, P, Self>:
        crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
{
    /// Completed native publication.
    type Published;

    /// Concrete executable for this completed route.
    type BoundCompositeRetainedPrediction<A, W>: From<
            BoundRetainedRoutedPrediction<
                B,
                S,
                A,
                P,
                W,
                Self,
                super::CompositeSessionIngress<A::AdmissionConfig>,
            >,
        > + crate::executable_binding::TextExecutableRoute<Self::Published>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        W: crate::speculative::PreparedRoutedPrediction<B, A>;
}

/// Materializes an independently prepared predictor, builds its composite target, and
/// moves the exact selected auxiliary banks into the predictor before publication.
pub fn construct_retained_composite_prediction<B, S, A, P, Req, W, V>(
    visitor: V,
    prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig, P, Req>,
    prediction: W,
    target_source: eredu_checkpoint::store::SharedCheckpointSource,
    provider_source: eredu_checkpoint::store::SharedCheckpointSource,
    binding: super::PredictionBinding,
) -> Result<V::Published, PreparedExecutionError<V::StorageError>>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::LayerRuntimeState<B>
        + crate::PredictionModelState<B, LayerState: crate::RuntimeAppendStreams<B>>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + RoutedLayeredArchitecture<B, S>,
    A::InputPartPlan: 'static,
    A::StaticModules: Clone,
    P: RoutedBankBinding<
        Resident: crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    >,
    AddressableRoutedProvider<B, P, V>:
        crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    W: crate::speculative::PreparedRoutedPrediction<B, A>,
    V: RetainedCompositePrediction<B, S, P>,
{
    let (prepared, processor, admission) = prepared.into_parts();
    let executable = construct_retained_routed_prediction(
        &visitor,
        prepared,
        prediction,
        target_source,
        provider_source,
        super::CompositeSessionIngress::new(admission, processor),
        binding,
    )?;
    Ok(crate::executable_binding::install_text_route::<
        V::BoundCompositeRetainedPrediction<A, W>,
        _,
        _,
    >(executable))
}
