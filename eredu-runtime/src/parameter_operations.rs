//! Shared accounting for loaded parameter queries and transactions.
use crate::{
    ArchitectureParameters, LayeredArchitecture, LayerwiseAcquireError, LayerwisePolicy,
    RuntimeState,
};
use eredu_core::{capture::*, parameters::ParameterError};
use eredu_nn::{NeuralBackend, Parameterized, Tensor};

mod assembly;
mod banks;
pub use banks::{prepare_bank_parameter_slots, PreparedBankParameter, PreparedBankParameterMember};
mod catalog;
mod coordination;
mod publication;
pub use assembly::{
    parameter_read_intent, PartitionParameterReadFragment, PartitionParameterReadPlan,
};
pub use catalog::{
    decode_parameter_catalog, encode_parameter_catalog, LocalParameterFacts,
    PartitionParameterCatalog,
};
pub use coordination::{
    ParameterModelIdentity, ParameterOperationBinding, ParameterOperationCoordinator,
    ParameterOperationKind, ParameterOperationTransport, ParameterReadPreparation,
    PARAMETER_CONTROL_WORDS,
};
pub use publication::{
    ParameterPublication, ParameterPublicationError, ParameterPublicationFailure,
    ParameterPublicationVisitor, ParameterReplacementValues, PreparedParameterPublication,
};

/// The architecture and residency policy actually retained by an execution.
/// Ordinary, routed and partitioned executors use the same local parameter
/// mechanism. This interface grants no global read, edit or communication authority;
/// callers must establish quiescence, exact prepared ownership and budgets first.
pub trait LayeredParameterOwner<B: NeuralBackend, S: RuntimeState<B>> {
    /// Architecture owning parameter identities and unit construction.
    type Architecture: LayeredArchitecture<B, S>;
    /// Policy owning populated units and their completion-safe residency loans.
    type Policy: LayerwisePolicy<B, <Self::Architecture as LayeredArchitecture<B, S>>::Unit>;

    /// Borrows both retained owners without selecting or constructing an execution.
    /// A custom executor with no parameter mechanism returns `None`; callers must
    /// report unavailable access instead of exposing incomplete values.
    fn parameter_parts(&mut self) -> Option<(&mut Self::Architecture, &mut Self::Policy)>;

    /// Borrows the same retained owners for cold inspection. No parameter
    /// construction, mutation, completion or source resolution is authorized.
    fn parameter_parts_ref(&self) -> Option<(&Self::Architecture, &Self::Policy)>;

    /// Checked slot ceiling for the exact static and idle policy topology.
    /// Retaining the scalar does not retain its source or replacement identity.
    fn retained_value_slot_bound(&self) -> Option<usize> {
        let (architecture, policy) = self.parameter_parts_ref()?;
        architecture
            .retained_static_value_slot_bound()?
            .checked_add(policy.retained_value_slot_bound()?)
    }

    /// Visits currently retained numerical values, including parameters,
    /// operator helpers, pinned modules and future-load overrides.
    /// This is not a catalog of unloaded parameters or a future workspace bound.
    /// False means the selected owner cannot provide a complete value traversal.
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        let Some((architecture, policy)) = self.parameter_parts_ref() else {
            return false;
        };
        let complete = policy.visit_retained_values(visitor);
        architecture.visit_retained_static_values(visitor) & complete
    }

    /// Visits actual resident slots without acquiring or rebuilding a unit.
    /// False means the selected policy cannot provide resident traversal.
    fn visit_loaded_parameters(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        let Some((architecture, policy)) = self.parameter_parts() else {
            return false;
        };
        visit_loaded_parameters_in_parts::<Self::Architecture, B, S, Self::Policy>(
            architecture,
            policy,
            visitor,
        )
    }

    /// Runs bounded work through the exact prepared static or unit owner.
    /// The policy validates the ordinal/address pair before construction and
    /// retains acquired resources until completion or safe terminal disposition.
    /// Pinned static values remain under the enclosing session completion owner.
    #[allow(clippy::type_complexity)]
    fn with_parameter_slots(
        &mut self,
        location: &PreparedParameterLocation,
        operation: &mut ParameterSlotOperation<
            '_,
            B::Tensor,
            <Self::Policy as LayerwisePolicy<
                B,
                <Self::Architecture as LayeredArchitecture<B, S>>::Unit,
            >>::Error,
        >,
        context: &<B::Tensor as Tensor>::Context,

        _preparation: Option<&B::ParameterPreparation<'_>>,
    ) -> Result<
        bool,
        LayerwiseAcquireError<
            <Self::Architecture as LayeredArchitecture<B, S>>::Error,
            <Self::Policy as LayerwisePolicy<
                B,
                <Self::Architecture as LayeredArchitecture<B, S>>::Unit,
            >>::Error,
        >,
    > {
        let Some((architecture, policy)) = self.parameter_parts() else {
            return Ok(false);
        };
        with_parameter_slots_in_parts::<Self::Architecture, B, S, Self::Policy>(
            architecture,
            policy,
            location,
            operation,
            context,
            _preparation,
        )
    }

    /// Invalidates observation/geometry declarations after successful publication.
    /// Release participant locks before retiring these displaced metadata owners.
    fn invalidate_parameter_observations(&mut self) {
        let _ = self.parameter_parts();
    }

    /// Lends the actual retained slots and future sources to a prepared publication pass.
    fn visit_parameter_publication(
        &mut self,
        publication: &mut dyn crate::parameter_operations::ParameterPublication<B::Tensor>,
    ) -> Result<
        bool,
        <Self::Policy as LayerwisePolicy<
            B,
            <Self::Architecture as LayeredArchitecture<B, S>>::Unit,
        >>::Error,
    > {
        let Some((architecture, policy)) = self.parameter_parts() else {
            return Ok(false);
        };
        visit_parameter_publication_in_parts::<Self::Architecture, B, S, Self::Policy>(
            architecture,
            policy,
            publication,
        )
    }
}

/// Visits an exact declared static aggregate and its selected idle policy.
/// Every row is borrowed from that owner; partial coverage cannot be installed
/// as complete source evidence. This constructs no parameter or native value.
pub(crate) fn visit_parameter_sources_in_parts<B, U, P, V>(
    static_modules: &(impl Parameterized<B::Tensor> + ?Sized),
    policy: &P,
    visitor: &mut V,
    context: &eredu_nn::workspace::WorkspaceContext,
) -> Result<bool, eredu_nn::Error>
where
    B: NeuralBackend,
    U: Parameterized<B::Tensor>,
    P: LayerwisePolicy<B, U>,
    V: for<'source> eredu_nn::ParameterSourceVisitor<'source, B::Tensor>,
{
    use eredu_nn::{workspace::WorkspaceMetadataError, ParameterSourceError};
    let controls = [
        std::mem::size_of_val(&static_modules),
        std::mem::size_of::<(&P, &mut V, &eredu_nn::workspace::WorkspaceContext)>(),
        std::mem::size_of::<Result<(), ParameterSourceError>>(),
        std::mem::size_of::<Result<bool, eredu_nn::Error>>(),
    ];
    context.charge_metadata(
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    match static_modules.visit_parameter_sources(visitor) {
        Ok(()) => {}
        Err(ParameterSourceError::UnclassifiedRetainedField) => return Ok(false),
        Err(cause) => return Err(context.metadata_source(cause)),
    }
    policy.visit_resident_parameter_sources(visitor, context)
}

/// Lends immutable loaded values from the actual architecture and policy.
/// The visitor cannot mutate slots or invalidate their observation declarations.
pub fn visit_loaded_parameters_in_parts<A, B, S, P>(
    architecture: &mut A,
    policy: &mut P,
    visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
) -> bool
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    if !policy.resident_parameters_available() {
        return false;
    }
    let mut adapter = LoadedSlotAdapter::<B>(visitor);
    match architecture.visit_static_parameters_mut(&mut adapter) {
        Ok(()) => {}
        Err(never) => match never {},
    }
    policy.visit_resident_units(&mut |unit| unit.visit_parameters_mut(&mut adapter))
}

/// Runs the shared prepared parameter loan through the retained source owners.
/// This does not expose mutable architecture or replace parameter slots.
pub fn with_parameter_slots_in_parts<A, B, S, P>(
    architecture: &mut A,
    policy: &mut P,
    location: &PreparedParameterLocation,
    operation: &mut ParameterSlotOperation<'_, B::Tensor, P::Error>,
    context: &<B::Tensor as Tensor>::Context,

    _preparation: Option<&B::ParameterPreparation<'_>>,
) -> Result<bool, LayerwiseAcquireError<A::Error, P::Error>>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    match location {
        PreparedParameterLocation::Bank { .. } | PreparedParameterLocation::Prediction { .. } => {
            Ok(false)
        }
        PreparedParameterLocation::Static { .. } => {
            operation(&mut |visitor| match architecture
                .visit_static_parameters_mut(&mut LoadedSlotAdapter::<B>(visitor))
            {
                Ok(()) => {}
                Err(never) => match never {},
            })
            .map_err(LayerwiseAcquireError::Policy)?;
            Ok(true)
        }
        PreparedParameterLocation::Unit { ordinal, address } => policy.inspect_unit(
            *ordinal,
            *address,
            |context| architecture.build_unit(address.group(), address.index(), context),
            |unit| {
                operation(&mut |visitor| {
                    unit.visit_parameters_mut(&mut LoadedSlotAdapter::<B>(visitor))
                })
            },
            context,
            _preparation,
        ),
    }
}

/// Visits publication participants without changing observation bindings.
/// The caller finalizes those bindings after successful committed publication.
pub fn visit_parameter_publication_in_parts<A, B, S, P>(
    architecture: &mut A,
    policy: &mut P,
    publication: &mut dyn ParameterPublication<B::Tensor>,
) -> Result<bool, P::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    if !policy.visit_parameter_publication(publication)? {
        return Ok(false);
    }
    struct Static<'a, T>(&'a mut dyn ParameterPublication<T>);
    impl<B: NeuralBackend> crate::StaticParameterVisitorMut<B> for Static<'_, B::Tensor> {
        type Error = std::convert::Infallible;
        fn visit_mut<M: Parameterized<B::Tensor>>(
            &mut self,
            _role: &str,
            module: &mut M,
        ) -> Result<(), Self::Error> {
            module.visit_parameters_mut(&mut ParameterPublicationVisitor(self.0));
            Ok(())
        }
    }
    match architecture.visit_static_parameters_mut(&mut Static(publication)) {
        Ok(()) => Ok(true),
        Err(never) => match never {},
    }
}
struct LoadedSlotAdapter<'a, B: NeuralBackend>(
    &'a mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
);
impl<'a, B: NeuralBackend> eredu_nn::ParameterVisitorMut<'a, B::Tensor>
    for LoadedSlotAdapter<'_, B>
{
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut B::Tensor,
    ) {
        self.0.visit_slot(metadata, value);
    }
}
impl<B: NeuralBackend> crate::StaticParameterVisitorMut<B> for LoadedSlotAdapter<'_, B> {
    type Error = std::convert::Infallible;
    fn visit_mut<M: Parameterized<B::Tensor>>(
        &mut self,
        _role: &str,
        module: &mut M,
    ) -> Result<(), Self::Error> {
        module.visit_parameters_mut(self);
        Ok(())
    }
}

/// A temporary traversal valid only within its owner's residency loan.
pub type ParameterSlotTraversal<'a, T> = dyn FnMut(&mut dyn eredu_nn::ParameterSlotVisitor<T>) + 'a;

/// Bounded work performed while all traversed native parameters remain retained.
/// The selected policy establishes completion before releasing the loan, including
/// on failure. Consumers must charge retained outputs before entering this call.
pub type ParameterSlotOperation<'a, T, E> =
    dyn FnMut(&mut ParameterSlotTraversal<'_, T>) -> Result<(), E> + 'a;

/// Architecture location of a prepared parameter slot, independent of residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedParameterLocation {
    /// A module retained by the selected typed prediction extension. The owning
    /// architecture supplies the module ordinal independently of proposal depth.
    Prediction {
        /// Stable physical parameter-owner ordinal within that extension.
        module: usize,
    },
    /// Independently acquired members of one architecture-declared bank and unit.
    Bank {
        /// Stable bank ordinal selected by the architecture.
        bank: usize,
        /// Architecture-global owning unit.
        unit: usize,
    },
    /// A pinned module outside the execution-unit residency window.
    Static {
        /// Architecture-declared static module role.
        role: String,
    },
    /// One unit acquired by the selected residency policy.
    Unit {
        /// Flat local policy ordinal.
        ordinal: usize,
        /// Architecture group and group-local unit identity.
        address: crate::ExecutionUnitAddress,
    },
}

/// Exact slot facts retained while preparing the selected binding plan.
///
/// The materialized geometry and dtype describe the prepared recipe output,
/// including selected load-time transforms. Reading this record never builds a
/// unit, reopens an artifact or allocates a backend tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedParameterSlot {
    /// Architecture-declared identity, sharing and companion relationships.
    pub parameter: eredu_nn::ParameterMetadata,
    /// Exact physical output metadata of the retained binding recipe.
    pub materialized: eredu_checkpoint::recipe::RecipeMetadata,
    /// Owner through which this slot is materialized.
    pub location: PreparedParameterLocation,
}

impl PreparedParameterSlot {
    /// Conservative additional work for sequential bank-member copies and their
    /// final concatenation. Ordinary unit materialization uses its residency budget.
    pub fn bank_loan_usage(&self) -> Result<CaptureUsage, ParameterError> {
        if !matches!(self.location, PreparedParameterLocation::Bank { .. }) {
            return Ok(CaptureUsage::default());
        }
        let members = *self
            .materialized
            .shape
            .first()
            .ok_or(ParameterError::Overflow)? as u64;
        let metadata = 256u64
            .checked_add(
                (self.parameter.id.as_str().len() as u64)
                    .checked_mul(4)
                    .ok_or(ParameterError::Overflow)?,
            )
            .and_then(|n| n.checked_add((self.materialized.shape.len() as u64).checked_mul(32)?))
            .ok_or(ParameterError::Overflow)?;
        Ok(CaptureUsage {
            retained_bytes: self
                .materialized
                .byte_len
                .checked_mul(2)
                .ok_or(ParameterError::Overflow)?,
            host_bytes: members
                .checked_mul(metadata)
                .and_then(|n| n.checked_add(512))
                .ok_or(ParameterError::Overflow)?,
            ..Default::default()
        })
    }
}

/// Reserves cumulative logical work before native evaluation, allocation or copies.
/// Charges survive failed work, model reset, overlay removal and snapshot restoration.
pub fn reserve_parameter_work(
    total: &mut CaptureUsage,
    cost: CaptureUsage,
    limit: CaptureUsage,
) -> Result<(), ParameterError> {
    let next = total.checked_add(cost)?;
    if let Some(budget) = next.exceeded(limit) {
        return Err(CaptureError::Limit {
            budget,
            cumulative: true,
        }
        .into());
    }
    *total = next;
    Ok(())
}
