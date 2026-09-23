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

    /// Visits actual resident slots without acquiring or rebuilding a unit.
    /// False means the selected policy cannot provide resident traversal.
    fn visit_loaded_parameters(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        let Some((architecture, policy)) = self.parameter_parts() else {
            return false;
        };
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
        match location {
            PreparedParameterLocation::Bank { .. }
            | PreparedParameterLocation::Prediction { .. } => Ok(false),
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
            ),
        }
    }

    /// Publishes already-completed replacements to loaded and future loaded units.
    /// The selected policy validates before changing handles. Its rejection leaves
    /// pinned modules untouched; after acceptance static publication is infallible.
    /// The live transaction owner separately handles caches, snapshots and peers.
    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, B::Tensor>,
        active: bool,
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
        if !policy.publish_parameter_replacements(values, active)? {
            return Ok(false);
        }
        struct Publish<'a, T>(&'a std::collections::BTreeMap<String, T>);
        impl<T: Tensor> eredu_nn::ParameterSlotVisitor<T> for Publish<'_, T> {
            fn visit_slot(&mut self, metadata: eredu_nn::ParameterMetadata, value: &mut T) {
                if let Some(replacement) = self.0.get(metadata.id.as_str()) {
                    value.publish_parameter(replacement);
                }
            }
        }
        match architecture
            .visit_static_parameters_mut(&mut LoadedSlotAdapter::<B>(&mut Publish(values)))
        {
            Ok(()) => {}
            Err(never) => match never {},
        }
        Ok(true)
    }
}

struct LoadedSlotAdapter<'a, B: NeuralBackend>(
    &'a mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
);
impl<'a, B: NeuralBackend> eredu_nn::ParameterVisitorMut<'a, B::Tensor>
    for LoadedSlotAdapter<'_, B>
{
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadata, value: &'a mut B::Tensor) {
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
    /// Canonical binding output within this materialization batch, after resolving
    /// actual binding aliases. This is not a checkpoint key or logical parameter
    /// alias. All static roles share one batch; units and prediction modules have
    /// independent batches. `None` means sharing/backing decomposition is unknown
    /// (including bank catalogs that aggregate independently acquired members).
    pub backing: Option<String>,
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
