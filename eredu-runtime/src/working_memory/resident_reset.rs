//! Original flat or nested resident table construction. Native publication is separate.
use super::{
    HostSlotStorageKey, InferenceExecutionIdentity, InferenceStateRetention, MemoryLedger,
    WorkingMemoryError, WorkingMemoryStorage,
};
use crate::{
    DenseHostSlotInitialization, HostSlotTable, SelectedStateRealization, SharedStateLayout,
};
use eredu_core::{SessionResetClaim, SessionResetRejection, cache::LayerCachePolicy};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};
mod account;
mod construction;
mod empty;
mod publication;
pub(crate) use account::ResetCustody;
pub(in crate::working_memory) use account::{Entry, Pending, capacity};
pub use empty::{PreparedResidentEmptyState, ResidentEmptyStateError};
pub use publication::{
    PreparedParameterStateReset, ResidentResetDisplaced, ResidentResetInstallation,
    ResidentResetProjection, ResidentResetPublicationCustody, ResidentResetPublicationProfile,
};

/// Concrete fixed-table reset representation. Every outer/child table is borrowed
/// from the actual state. Constructors may only move policy scalars, supplied
/// originally funded tables and empty numerical slots; they cannot allocate a
/// tensor or submit native work. Additional host managers or source owners use
/// the exact source plan and same-account construction context below.
pub trait ResidentTableResetState: InferenceStateRetention + Sized + Send + Sync + 'static {
    /// Actual element type in the outer state table.
    type Layer: Send + Sync + 'static;
    /// Actual inline child-table element; `()` for a representation without children.
    type Child: Send + Sync + 'static;
    /// Allocation-free exact source descriptor for additional host construction.
    type ResetPlan: Default + PartialEq;
    /// Prepared host-only source context; its outputs must retain their funding.
    type ResetContext: Default;
    /// Actual constructor controls beyond the existing fixed table worker.
    fn resident_reset_plan(&self) -> Result<(Self::ResetPlan, usize), WorkingMemoryError> {
        Ok((Self::ResetPlan::default(), 0))
    }
    /// Runs only after the same reset comparison. No native work is authorized.
    fn prepare_resident_reset_context(
        &self,
        _plan: &Self::ResetPlan,
        _funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<Self::ResetContext, eredu_core::BackendFailure> {
        Ok(Self::ResetContext::default())
    }
    /// Checks the actual component's source-qualified selected placement within its layer.
    fn validate_resident_reset_placement(
        _layer: &Self::Layer,
        _role: eredu_core::cache::StateComponentRole,
        placement: crate::StateComponentPlacement,
    ) -> bool {
        placement == crate::StateComponentPlacement::Device
    }
    /// Same empty layer worker, with optional source-derived host context.
    fn empty_resident_reset_layer_prepared(
        _context: &mut Self::ResetContext,
        _source: &Self::Layer,
        policy: &LayerCachePolicy,
        child: Option<HostSlotTable<Self::Child>>,
    ) -> Result<Self::Layer, eredu_core::BackendFailure> {
        Ok(Self::empty_resident_reset_layer_with_child(policy, child))
    }
    /// Actual fixed outer table, borrowed without constructing an inventory.
    fn resident_reset_layers(&self) -> &HostSlotTable<Self::Layer>;
    /// Retained architecture-declared layout for this exact state.
    fn resident_reset_layout(&self) -> &SharedStateLayout;
    /// Existing local-to-global layer origin.
    fn resident_reset_global_start(&self) -> usize;
    /// Rejects retained managers or other whole-state owners not covered by this producer.
    fn validate_resident_reset_state(&self) -> bool {
        true
    }
    /// True only when the actual state is identical to its empty policy worker:
    /// no retained numerical backing, nonzero frontier, manager or branch state.
    /// A reset may discard populated state; an independent startup may not.
    fn resident_fork_is_empty(&self) -> bool {
        false
    }
    /// Confirms this actual native representation implements the selected policy.
    fn validate_resident_reset_layer(layer: &Self::Layer, policy: &LayerCachePolicy) -> bool;
    /// Actual child table, including a zero-length table with its own metadata.
    fn resident_reset_child(_layer: &Self::Layer) -> Option<&HostSlotTable<Self::Child>> {
        None
    }
    /// Policy-only empty child element; never clones the source's numerical value.
    fn empty_resident_reset_child(_source: &Self::Child) -> Self::Child {
        unreachable!("a child-bearing representation supplies its empty element")
    }
    /// Existing flat policy worker. Child-bearing implementations use the next method.
    fn empty_resident_reset_layer(policy: &LayerCachePolicy) -> Self::Layer;
    /// Moves the actual funded child into an otherwise inline empty layer.
    fn empty_resident_reset_layer_with_child(
        policy: &LayerCachePolicy,
        child: Option<HostSlotTable<Self::Child>>,
    ) -> Self::Layer {
        assert!(child.is_none(), "flat reset has no child table");
        Self::empty_resident_reset_layer(policy)
    }
    /// Moves funded tables and the prepared host context into the same representation.
    /// No numerical allocation or execution is authorized.
    fn from_resident_reset(
        context: &mut Self::ResetContext,
        layout: SharedStateLayout,
        global_start: usize,
        layers: HostSlotTable<Self::Layer>,
    ) -> Self;
}

/// Exact current source, minted only by the shared session's checked inspection.
/// Its lexical borrow prevents replacement, revision changes and parameter
/// publication. It is not a native completion or admission capability.
pub struct ResidentResetSource<'a, S> {
    pub(crate) state: &'a S,
    pub(crate) selected: &'a SelectedStateRealization,
    pub(crate) execution: &'a InferenceExecutionIdentity,
    pub(crate) control: &'a crate::replicated_session::ParameterControlIdentity,
    pub(crate) revision: Option<&'a super::InferenceStateRevision>,
}
impl<'a, S> ResidentResetSource<'a, S> {
    /// Exact installed state, without mutable or owning extraction.
    pub fn state(&self) -> &'a S {
        self.state
    }
    /// Exact lexical state, selected realization, execution, parameter origin
    /// and branch revision. Equal contents are insufficient.
    pub fn same_source(&self, other: &Self) -> bool {
        std::ptr::eq(self.state, other.state)
            && std::ptr::eq(self.selected, other.selected)
            && Arc::ptr_eq(&self.execution.0, &other.execution.0)
            && self.control.matches(other.control)
            && self.revision == other.revision
    }
}

/// Concrete backend-session projection checked before original reset admission.
/// Implementations must compare against the actual current selected session
/// source through its ordinary idle inspection, without copying or allocating.
/// This bridge cannot accept a cached numeric report or an unrelated source.
pub trait ResidentResetSession<S: ResidentTableResetState> {
    /// Checks exact current source/execution/revision under the existing session
    /// borrow. Native providers additionally enforce their actual idle authority.
    fn validate_resident_reset_source(
        &self,
        source: &ResidentResetSource<'_, S>,
    ) -> Result<(), WorkingMemoryError>;
}

// Closed provenance alternatives. Ordinary keys are projected outside Usage;
// original tables borrow their actual constructor custody and independent pin.
enum TableSource<'a, K: HostSlotStorageKey> {
    Ordinary {
        registration: &'a WorkingMemoryStorage<K>,
        table: &'a K,
        layout: &'a K,
    },
    Original {
        metadata: &'a crate::HostSlotMetadata,
        custody: &'a ResetCustody,
    },
    Registered {
        table: K,
        layout: K,
    },
}

// Both entries construct only source-derived empty host state. The explicit
// reset entry carries the core claim; parameter publication supplies its current
// executable and the same selected session's exclusive source validation.
enum ResetRequest<'a> {
    Session(SessionResetClaim<'a>),
    Parameter {
        execution: &'a InferenceExecutionIdentity,
        limits: &'a eredu_core::MemoryLimits,
    },
}

/// Closed source-derived fixed destination plan.
///
/// ```compile_fail
/// use eredu_runtime::working_memory::{PreparedResidentKvReset, ResidentTableResetState, ResidentResetSession, HostSlotStorageKey, MemoryLedger};
/// use eredu_core::SessionResetClaim;
/// fn twice<S, K, T>(plan: PreparedResidentKvReset<'_, S, K>, session: &T, claim: SessionResetClaim<'_>, pool: &MemoryLedger)
/// where S: ResidentTableResetState, K: HostSlotStorageKey, T: ResidentResetSession<S> {
///     let _ = plan.construct(session, claim, pool);
///     let _ = plan.construct(session, claim, pool);
/// }
/// ```
/// No allocating initializer runs
/// until the selected session validates its actual source and the ledger accepts
/// the explicit reset claim or enclosing parameter-publication demand.
pub struct PreparedResidentKvReset<'a, S: ResidentTableResetState, K: HostSlotStorageKey> {
    source: ResidentResetSource<'a, S>,
    pool: &'a MemoryLedger,
    slots: DenseHostSlotInitialization<'a, S::Layer>,
    table_source: TableSource<'a, K>,
    bytes: u64,
    child_tables: usize,
    context_plan: S::ResetPlan,
    context_bytes: usize,
}
impl<'a, S: ResidentTableResetState, K: HostSlotStorageKey> PreparedResidentKvReset<'a, S, K> {
    /// Plans from the actual borrowed table/layout and two inline native keys.
    /// Key projections must identify those exact host sources. Existing
    /// canonical owners are acquired with the original comparison, never by
    /// constructing a grouped registration before admission.
    pub fn prepare_registered(
        source: ResidentResetSource<'a, S>,
        table: K,
        layout: K,
        pool: &'a MemoryLedger,
    ) -> Result<Self, ResidentResetError<S>> {
        let metadata = source.state.resident_reset_layers().metadata();
        if metadata.original_reset_custody().is_some()
            || table.host_slot_identity() != Some(metadata.identity().registry_key())
            || layout.host_slot_identity()
                != Some(
                    source
                        .state
                        .resident_reset_layout()
                        .identity()
                        .registry_key(),
                )
        {
            return Err(ResidentResetError::rejected(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        Self::prepare_source(source, TableSource::Registered { table, layout }, pool)
    }

    /// Uses the session-issued actual source and existing complete registration.
    /// Selection, every component, current representation and layout owner must
    /// agree before this plan can describe a fixed resident KV destination.
    pub fn prepare(
        source: ResidentResetSource<'a, S>,
        registration: &'a WorkingMemoryStorage<K>,
        pool: &'a MemoryLedger,
    ) -> Result<Self, ResidentResetError<S>> {
        let table = source.state.resident_reset_layers().metadata();
        if table.original_reset_custody().is_some() {
            return Err(ResidentResetError::rejected(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let table_source = TableSource::Ordinary {
            registration,
            table: registration
                .host_source_key(table.identity().registry_key())
                .map_err(ResidentResetError::rejected)?,
            layout: registration
                .host_source_key(
                    source
                        .state
                        .resident_reset_layout()
                        .identity()
                        .registry_key(),
                )
                .map_err(ResidentResetError::rejected)?,
        };
        Self::prepare_source(source, table_source, pool)
    }

    /// Plans from a table produced by an earlier genuine original reset. Its
    /// independent layout entry is reused; no grouped source pin is allocated
    /// before comparison, and no predecessor account survives successful fill.
    pub fn prepare_original(
        source: ResidentResetSource<'a, S>,
        pool: &'a MemoryLedger,
    ) -> Result<Self, ResidentResetError<S>> {
        let metadata = source.state.resident_reset_layers().metadata();
        let custody = metadata
            .original_reset_custody()
            .filter(|_| metadata.original_source_is_live())
            .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::IdentityMismatch))?;
        Self::prepare_source(source, TableSource::Original { metadata, custody }, pool)
    }

    fn prepare_source(
        source: ResidentResetSource<'a, S>,
        table_source: TableSource<'a, K>,
        pool: &'a MemoryLedger,
    ) -> Result<Self, ResidentResetError<S>> {
        let state = source.state;
        let _layout = state.resident_reset_layout();
        let table = state.resident_reset_layers();
        let mut child_tables = 0usize;
        let mut child_bytes = 0u64;
        validate_source_geometry(&source, ResidentResetError::rejected, |child| {
            let plan = child
                .prepare_copy_slots()
                .and_then(|p| p.for_dense_destination::<S::Child>())
                .map_err(|_| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
            child_tables = child_tables
                .checked_add(1)
                .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
            child_bytes = child_bytes
                .checked_add(plan.initialization_peak_bytes())
                .and_then(|n| n.checked_add(u64::try_from(child_control_bytes::<S, K>()?).ok()?))
                .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
            Ok(())
        })?;
        let slots = table
            .prepare_copy_slots()
            .and_then(|p| p.for_dense_destination())
            .map_err(|_| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        let (context_plan, context_bytes) = state
            .resident_reset_plan()
            .map_err(ResidentResetError::rejected)?;
        let context_controls = construction::control_bytes(context_bytes)
            .and_then(|n| n.checked_add(size_of::<ResetRequest<'_>>()))
            .and_then(|n| {
                n.checked_add(size_of::<(
                    eredu_core::DomainMemoryRequirements,
                    eredu_core::MemoryLimits,
                )>())
            })
            .and_then(|n| n.checked_add(size_of::<S::ResetPlan>()))
            .and_then(|n| n.checked_add(size_of::<S::ResetContext>()))
            .and_then(|n| n.checked_add(size_of::<&mut S::ResetContext>()))
            .and_then(|n| {
                n.checked_add(size_of::<Result<S::ResetContext, eredu_core::BackendFailure>>())
            })
            .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        let controls = control_bytes::<S, K>(matches!(&table_source, TableSource::Ordinary { .. }))
            .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        Layout::array::<account::ChildSourceCustody>(child_tables)
            .map_err(|_| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        let membership = account::table_membership_bytes(
            child_tables
                .checked_add(1)
                .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?,
        )
        .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        let bytes = slots
            .initialization_peak_bytes()
            .checked_add(
                account::domain_control_bytes(pool.topology())
                    .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?,
            )
            .and_then(|n| n.checked_add(child_bytes))
            .and_then(|n| n.checked_add(u64::try_from(context_controls).ok()?))
            .and_then(|n| n.checked_add(u64::try_from(membership).ok()?))
            .and_then(|n| n.checked_add(u64::try_from(controls).ok()?))
            .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        Ok(Self {
            source,
            pool,
            slots,
            table_source,
            bytes,
            child_tables,
            context_plan,
            context_bytes,
        })
    }
    /// Whether two borrowed plans name the exact same source and session
    /// revision. Equal policy/values alone cannot establish this identity.
    pub fn same_source(&self, other: &Self) -> bool {
        self.pool.same_ledger(other.pool)
            && self.context_plan == other.context_plan
            && self.context_bytes == other.context_bytes
            && std::ptr::eq(self.source.state, other.source.state)
            && std::ptr::eq(self.source.selected, other.source.selected)
            && Arc::ptr_eq(&self.source.execution.0, &other.source.execution.0)
            && self.source.control.matches(other.source.control)
            && self.source.revision == other.source.revision
            && self
                .slots
                .source_metadata()
                .same_storage(other.slots.source_metadata())
    }

    /// Complete fixed host construction envelope, excluding existing sources
    /// and the later native reset/publication/completion operation.
    pub fn required_bytes(&self) -> u64 {
        self.bytes
    }
    /// Consumes one real claim, source plan and original account. The source
    /// remains borrowed throughout fill; a second table cannot be extracted.
    /// `session` is the actual backend session supplied to the core reset hook.
    pub fn construct<T: ResidentResetSession<S>>(
        self,
        session: &T,
        claim: SessionResetClaim<'_>,
        pool: &MemoryLedger,
    ) -> Result<S, ResidentResetError<S>> {
        self.construct_prepared(session, ResetRequest::Session(claim), pool, |_, _| ())
            .map(|(state, ())| state)
    }

    fn construct_prepared<T: ResidentResetSession<S>, P>(
        self,
        session: &T,
        request: ResetRequest<'_>,
        pool: &MemoryLedger,
        prepare: impl FnOnce(&ResidentResetSource<'_, S>, &ResetCustody) -> P,
    ) -> Result<(S, P), ResidentResetError<S>> {
        if !self.pool.same_ledger(pool) {
            return Err(ResidentResetError::rejected(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        match &request {
            ResetRequest::Session(claim) => claim
                .validate_session(session)
                .map_err(ResidentResetError::claim)?,
            ResetRequest::Parameter { execution, limits } => {
                if !self.source.execution.same_execution(execution) {
                    return Err(ResidentResetError::rejected(
                        WorkingMemoryError::IdentityMismatch,
                    ));
                }
                limits
                    .validate(pool.topology())
                    .map_err(|cause| ResidentResetError::rejected(cause.into()))?;
            }
        }
        session
            .validate_resident_reset_source(&self.source)
            .map_err(ResidentResetError::rejected)?;
        let current = self
            .source
            .state
            .resident_reset_plan()
            .map_err(ResidentResetError::rejected)?;
        if current.0 != self.context_plan || current.1 != self.context_bytes {
            return Err(ResidentResetError::rejected(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let mut requirements = eredu_core::DomainMemoryRequirements::zero(pool.topology());
        requirements
            .add_allocation(
                self.bytes,
                &eredu_core::MemoryPlacement::fixed(pool.topology(), pool.topology().host_domain())
                    .map_err(|e| ResidentResetError::rejected(e.into()))?,
            )
            .map_err(|e| ResidentResetError::rejected(e.into()))?;
        let (requirements, limits) = match request {
            ResetRequest::Session(claim) => claim
                .compare(pool.topology(), requirements)
                .map_err(ResidentResetError::claim)?
                .into_parts(),
            ResetRequest::Parameter { limits, .. } => {
                for (domain, charge) in requirements.iter() {
                    limits
                        .get(domain)
                        .and_then(|limit| limit.check(domain, 0, charge.total()?))
                        .map_err(|cause| ResidentResetError::rejected(cause.into()))?;
                }
                (requirements, limits.clone())
            }
        };
        let state = self.source.state;
        let table = state.resident_reset_layers().metadata();
        let layout = state.resident_reset_layout();
        let admission = account::admit(
            pool,
            &self.table_source,
            (
                table.identity().registry_key(),
                table.capacity_bytes().ok_or_else(|| {
                    ResidentResetError::rejected(WorkingMemoryError::UnknownBound)
                })?,
            ),
            (
                layout.identity().registry_key(),
                layout.capacity_bytes().ok_or_else(|| {
                    ResidentResetError::rejected(WorkingMemoryError::UnknownBound)
                })?,
            ),
            requirements,
            limits,
        )
        .map_err(ResidentResetError::admission)?;
        // Both finite control vectors are allocated only under this account.
        // Source pins are existing-only and survive every later failed prefix.
        let mut source_children = Vec::new();
        let mut membership = Vec::new();
        let preparation = (|| -> Result<(), ResetCause> {
            source_children
                .try_reserve_exact(self.child_tables)
                .map_err(ResetCause::Allocation)?;
            membership
                .try_reserve_exact(self.child_tables + 1)
                .map_err(ResetCause::Allocation)?;
            if source_children.capacity() != self.child_tables
                || membership.capacity() != self.child_tables + 1
            {
                return Err(ResetCause::Memory(WorkingMemoryError::IdentityMismatch));
            }
            for layer in state.resident_reset_layers().slots() {
                if let Some(child) = S::resident_reset_child(layer) {
                    if source_children.len() == self.child_tables {
                        return Err(ResetCause::Memory(WorkingMemoryError::IdentityMismatch));
                    }
                    source_children.push(
                        account::pin_child::<K>(pool, child.metadata())
                            .map_err(ResetCause::Memory)?,
                    );
                }
            }
            if source_children.len() != self.child_tables {
                return Err(ResetCause::Memory(WorkingMemoryError::IdentityMismatch));
            }
            Ok(())
        })();
        if let Err(cause) = preparation {
            let _ = admission.custody.finish(Vec::new());
            return Err(ResidentResetError {
                cause,
                partial: Vec::new(),
                state: None,
                entry: None,
                source: Some(admission.source),
                source_children,
                custody: Some(admission.custody),
            });
        }
        // Original custody exists before any provider publication allocation.
        // Its prepared owner retires independently on every failed prefix.
        let prepared = prepare(&self.source, &admission.custody);
        // Infallible source clones occur only after original acceptance. The
        // inner worker owns all partial values before any fallible reserve/fill.
        let result = (|| {
            let funding = construction::prepare(self.context_bytes, &admission.custody)
                .map_err(ResidentResetError::construction)?;
            let mut context = state
                .prepare_resident_reset_context(&self.context_plan, funding.as_ref())
                .map_err(ResidentResetError::construction)?;
            construct_slots::<S>(
                self.slots,
                state.resident_reset_layers(),
                layout,
                state.resident_reset_global_start(),
                &admission.custody,
                &mut context,
            )
        })();
        match result {
            Ok(mut value) => {
                if let Err(cause) = value
                    .inference_retention_mut()
                    .install_original_reset_revision(admission.custody.clone())
                {
                    return Err(ResidentResetError {
                        cause: ResetCause::Memory(cause),
                        partial: Vec::new(),
                        state: Some(value),
                        entry: None,
                        source: Some(admission.source),
                        source_children,
                        custody: Some(admission.custody),
                    });
                }
                membership.push(table_member(value.resident_reset_layers().metadata()));
                for layer in value.resident_reset_layers().slots() {
                    if let Some(child) = S::resident_reset_child(layer) {
                        assert!(
                            membership.len() < membership.capacity(),
                            "validated child population"
                        );
                        membership.push(table_member(child.metadata()));
                    }
                }
                assert_eq!(
                    membership.len(),
                    self.child_tables + 1,
                    "validated child population"
                );
                if let Err(cause) = admission.custody.finish(membership) {
                    return Err(ResidentResetError {
                        cause: ResetCause::Memory(cause),
                        partial: Vec::new(),
                        state: Some(value),
                        entry: None,
                        source: Some(admission.source),
                        source_children,
                        custody: Some(admission.custody),
                    });
                }
                Ok((value, prepared))
            }
            Err(mut error) => {
                if let Err(cause) = admission.custody.finish(Vec::new()) {
                    error.cause = ResetCause::Memory(cause);
                }
                error.source = Some(admission.source);
                error.source_children = source_children;
                error.custody = Some(admission.custody);
                Err(error)
            }
        }
    }
}

fn source_geometry_control_bytes<S: ResidentTableResetState, E>(callback: usize) -> Option<usize> {
    let controls = [
        callback,
        size_of::<(
            &ResidentResetSource<'_, S>,
            &S,
            &SharedStateLayout,
            &HostSlotTable<S::Layer>,
        )>(),
        size_of::<
            std::iter::Enumerate<
                std::iter::Zip<
                    std::slice::Iter<'_, S::Layer>,
                    std::slice::Iter<'_, LayerCachePolicy>,
                >,
            >,
        >(),
        size_of::<Option<&HostSlotTable<S::Child>>>(),
        size_of::<&[eredu_core::cache::StateComponentPolicy]>(),
        size_of::<(
            eredu_core::cache::StateComponentRole,
            crate::StateComponentPlacement,
        )>(),
        size_of::<&[crate::replicated_text::SelectedStateComponentRealization]>(),
        size_of::<Result<(), E>>(),
        size_of::<Option<usize>>(),
        3 * size_of::<usize>(),
    ];
    controls
        .into_iter()
        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
}

fn validate_source_geometry<S: ResidentTableResetState, E>(
    source: &ResidentResetSource<'_, S>,
    error: impl Fn(WorkingMemoryError) -> E,
    mut inspect_child: impl FnMut(&HostSlotTable<S::Child>) -> Result<(), E>,
) -> Result<(), E> {
    let state = source.state;
    let layout = state.resident_reset_layout();
    let table = state.resident_reset_layers();
    if !state.validate_resident_reset_state()
        || source.selected.layout() != layout.layout()
        || table.len() != layout.layout().len()
    {
        return Err(error(WorkingMemoryError::IdentityMismatch));
    }
    let mut cursor = 0usize;
    for (index, (layer, policy)) in table
        .slots()
        .iter()
        .zip(layout.layout().layers().iter())
        .enumerate()
    {
        if !S::validate_resident_reset_layer(layer, policy) {
            return Err(error(WorkingMemoryError::UnknownBound));
        }
        if let Some(child) = S::resident_reset_child(layer) {
            inspect_child(child)?;
        }
        let expected = layout
            .layout()
            .components(index)
            .ok_or_else(|| error(WorkingMemoryError::IdentityMismatch))?;
        let end = cursor
            .checked_add(expected.len())
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        let realized = source
            .selected
            .components()
            .get(cursor..end)
            .ok_or_else(|| error(WorkingMemoryError::IdentityMismatch))?;
        if realized.iter().zip(expected).any(|(actual, expected)| {
            actual.layer() != index
                || actual.component() != expected
                || !S::validate_resident_reset_placement(
                    layer,
                    actual.component().role(),
                    actual.placement(),
                )
        }) {
            return Err(error(WorkingMemoryError::UnknownBound));
        }
        cursor = end;
    }
    if cursor != source.selected.components().len()
        || state
            .resident_reset_global_start()
            .checked_add(table.len())
            .is_none()
    {
        return Err(error(WorkingMemoryError::IdentityMismatch));
    }
    Ok(())
}

fn construct_slots<S: ResidentTableResetState>(
    plan: DenseHostSlotInitialization<'_, S::Layer>,
    source: &HostSlotTable<S::Layer>,
    layout: &SharedStateLayout,
    global_start: usize,
    custody: &ResetCustody,
    context: &mut S::ResetContext,
) -> Result<S, ResidentResetError<S>> {
    let (_, mut builder) = match plan.try_initialize_retaining_source() {
        Ok(value) => value,
        Err(cause) => {
            return Err(ResidentResetError {
                cause: ResetCause::Allocation(cause),
                partial: Vec::new(),
                state: None,
                entry: None,
                source: None,
                source_children: Vec::new(),
                custody: None,
            });
        }
    };
    for (layer, policy) in source.slots().iter().zip(layout.layout().layers().iter()) {
        #[cfg(test)]
        if tests::fail_at(builder.initialized_count()) {
            let cause = builder.capacity_overflow_for_test();
            return Err(ResidentResetError {
                cause: ResetCause::Allocation(cause),
                partial: builder.into_partial_values(),
                state: None,
                entry: None,
                source: None,
                source_children: Vec::new(),
                custody: None,
            });
        }

        let child = S::resident_reset_child(layer)
            .map(|source| construct_child::<S>(source, custody))
            .transpose();
        let child = match child {
            Ok(child) => child,
            Err(cause) => {
                return Err(ResidentResetError {
                    cause,
                    partial: builder.into_partial_values(),
                    state: None,
                    entry: None,
                    source: None,
                    source_children: Vec::new(),
                    custody: None,
                });
            }
        };
        let value = match S::empty_resident_reset_layer_prepared(context, layer, policy, child) {
            Ok(value) => value,
            Err(cause) => {
                return Err(ResidentResetError {
                    cause: ResetCause::Construction(cause),
                    partial: builder.into_partial_values(),
                    state: None,
                    entry: None,
                    source: None,
                    source_children: Vec::new(),
                    custody: None,
                });
            }
        };
        assert!(builder.push(value).is_ok(), "validated exact reset extent");
    }
    let values = builder.into_exact_values().expect("exact reset fill");
    let identity = match crate::HostMetadataIdentity::original_reset(custody.clone()) {
        Ok(identity) => identity,
        Err(cause) => {
            return Err(ResidentResetError {
                cause: ResetCause::Memory(cause),
                partial: values,
                state: None,
                entry: None,
                source: None,
                source_children: Vec::new(),
                custody: None,
            });
        }
    };
    let table = HostSlotTable::original_reset(values.into_boxed_slice(), identity, custody.clone());
    Ok(S::from_resident_reset(
        context,
        layout.clone(),
        global_start,
        table,
    ))
}

fn table_member(metadata: &crate::HostSlotMetadata) -> (crate::HostMetadataKey, u64) {
    (
        metadata.identity().registry_key().clone(),
        metadata.capacity_bytes().expect("validated table extent"),
    )
}
fn construct_child<S: ResidentTableResetState>(
    source: &HostSlotTable<S::Child>,
    custody: &ResetCustody,
) -> Result<HostSlotTable<S::Child>, ResetCause> {
    let plan = source
        .prepare_copy_slots()
        .and_then(|p| p.for_dense_destination())
        .map_err(|_| ResetCause::Memory(WorkingMemoryError::Overflow))?;
    let (_, mut builder) = plan
        .try_initialize_retaining_source()
        .map_err(ResetCause::Allocation)?;
    for slot in source.slots() {
        assert!(
            builder.push(S::empty_resident_reset_child(slot)).is_ok(),
            "validated child extent"
        );
    }
    let values = builder
        .into_exact_values()
        .ok()
        .expect("exact empty child fill");
    let identity =
        crate::HostMetadataIdentity::original_reset(custody.clone()).map_err(ResetCause::Memory)?;
    Ok(HostSlotTable::original_reset(
        values.into_boxed_slice(),
        identity,
        custody.clone(),
    ))
}
fn child_control_bytes<S: ResidentTableResetState, K: HostSlotStorageKey>() -> Option<usize> {
    [
        crate::HostMetadataIdentity::original_control_bytes()?,
        crate::HostSlotMetadata::original_control_bytes()?,
        crate::working_memory::storage::reset_layout::child_pin_control_bytes::<K>()?,
        size_of::<account::ChildSourceCustody>(),
        size_of::<Result<account::ChildSourceCustody, WorkingMemoryError>>(),
        size_of::<DenseHostSlotInitialization<'_, S::Child>>(),
        size_of::<crate::DenseHostSlotInitializationBuilder<S::Child>>(),
        size_of::<HostSlotTable<S::Child>>(),
        size_of::<Option<HostSlotTable<S::Child>>>(),
        size_of::<Result<HostSlotTable<S::Child>, ResetCause>>(),
        size_of::<(
            DenseHostSlotInitialization<'_, S::Child>,
            crate::DenseHostSlotInitializationBuilder<S::Child>,
        )>(),
        size_of::<
            Result<
                (
                    DenseHostSlotInitialization<'_, S::Child>,
                    crate::DenseHostSlotInitializationBuilder<S::Child>,
                ),
                std::collections::TryReserveError,
            >,
        >(),
        size_of::<Result<Vec<S::Child>, crate::DenseHostSlotInitializationBuilder<S::Child>>>(),
        size_of::<S::Child>(),
        size_of::<Result<crate::HostMetadataIdentity, WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

#[derive(Debug)]
enum ResetCause {
    Construction(eredu_core::BackendFailure),
    Memory(WorkingMemoryError),
    Claim(SessionResetRejection),
    Allocation(std::collections::TryReserveError),
}
/// Consuming failure owns every actual partial value and original allowance.
/// There is no cause/payload extraction, retry, or additional bank population.
pub struct ResidentResetError<S: ResidentTableResetState> {
    cause: ResetCause,
    partial: Vec<S::Layer>,
    state: Option<S>,
    entry: Option<Box<account::Entry>>,
    source: Option<account::TableSourceCustody>,
    source_children: Vec<account::ChildSourceCustody>,
    custody: Option<ResetCustody>,
}
impl<S: ResidentTableResetState> ResidentResetError<S> {
    fn construction(cause: eredu_core::BackendFailure) -> Self {
        Self {
            cause: ResetCause::Construction(cause),
            partial: Vec::new(),
            state: None,
            entry: None,
            source: None,
            source_children: Vec::new(),
            custody: None,
        }
    }
    fn admission(failure: account::AdmissionFailure) -> Self {
        Self {
            cause: ResetCause::Memory(failure.cause),
            partial: Vec::new(),
            state: None,
            entry: failure.entry,
            source: failure.source,
            source_children: Vec::new(),
            custody: failure.custody,
        }
    }

    fn rejected(cause: WorkingMemoryError) -> Self {
        Self {
            cause: ResetCause::Memory(cause),
            partial: Vec::new(),
            state: None,
            entry: None,
            source: None,
            source_children: Vec::new(),
            custody: None,
        }
    }
    fn claim(cause: SessionResetRejection) -> Self {
        Self {
            cause: ResetCause::Claim(cause),
            partial: Vec::new(),
            state: None,
            entry: None,
            source: None,
            source_children: Vec::new(),
            custody: None,
        }
    }
    /// Original held demand, or zero for rejection before admission.
    pub fn retained_bytes(&self) -> u64 {
        self.custody.as_ref().map_or(0, ResetCustody::bytes)
    }
    /// Actual number of retained partial layer values.
    pub fn initialized_count(&self) -> usize {
        self.partial.len()
    }
}
impl<S: ResidentTableResetState> fmt::Debug for ResidentResetError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResidentResetError")
            .field("cause", &self.cause)
            .field("partial", &self.partial.len())
            .field("retained", &self.retained_bytes())
            .finish()
    }
}
impl<S: ResidentTableResetState> fmt::Display for ResidentResetError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            ResetCause::Construction(e) => fmt::Display::fmt(e, f),
            ResetCause::Memory(e) => fmt::Display::fmt(e, f),
            ResetCause::Claim(e) => fmt::Display::fmt(e, f),
            ResetCause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<S: ResidentTableResetState> std::error::Error for ResidentResetError<S> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            ResetCause::Construction(e) => Some(e),
            ResetCause::Memory(e) => Some(e),
            ResetCause::Claim(e) => Some(e),
            ResetCause::Allocation(e) => Some(e),
        }
    }
}
fn arc_bytes<T>() -> Option<usize> {
    Some(
        Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}
fn control_bytes<S: ResidentTableResetState, K: HostSlotStorageKey>(
    grouped: bool,
) -> Option<usize> {
    [
        source_geometry_control_bytes::<S, ResidentResetError<S>>(
            size_of::<(&mut usize, &mut u64)>(),
        )?,
        account::control_bytes::<K>(grouped)?,
        super::InferenceStateRevision::reset_control_bytes()?,
        crate::HostMetadataIdentity::original_control_bytes()?,
        crate::HostSlotMetadata::original_control_bytes()?,
        size_of::<PreparedResidentKvReset<'_, S, K>>(),
        size_of::<Vec<account::ChildSourceCustody>>(),
        size_of::<Result<(), ResetCause>>(),
        size_of::<ResidentResetSource<'_, S>>(),
        size_of::<crate::DenseHostSlotInitializationBuilder<S::Layer>>(),
        size_of::<(
            &mut S::ResetContext,
            &S::Layer,
            &LayerCachePolicy,
            Option<HostSlotTable<S::Child>>,
            Result<S::Layer, eredu_core::BackendFailure>,
        )>(),
        size_of::<OriginalResidentResetSource>(),
        size_of::<HostSlotSource>(),
        size_of::<Result<HostSlotSource, WorkingMemoryError>>(),
        size_of::<Result<OriginalResidentResetSource, WorkingMemoryError>>(),
        size_of::<S>(),
        size_of::<Option<S>>(),
        size_of::<ResidentResetError<S>>(),
        size_of::<Result<S, ResidentResetError<S>>>(),
        size_of::<eredu_core::SessionResetClaim<'_>>(),
        size_of::<eredu_core::SessionResetLimits>(),
        size_of::<eredu_core::SessionResetAcceptance>(),
        size_of::<
            Result<
                (
                    DenseHostSlotInitialization<'_, S::Layer>,
                    crate::DenseHostSlotInitializationBuilder<S::Layer>,
                ),
                std::collections::TryReserveError,
            >,
        >(),
        size_of::<Result<Vec<S::Layer>, crate::DenseHostSlotInitializationBuilder<S::Layer>>>(),
        size_of::<Result<crate::HostMetadataIdentity, WorkingMemoryError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Result<eredu_core::SessionResetAcceptance, SessionResetRejection>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<ResidentResetError<S>>()?,
        size_of::<Result<(), eredu_core::BackendFailure>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

/// Policy-only layer constructor used by the portable DeviceState adapter.
/// Empty construction may move inline scalars/None values only; no allocation
/// or native work is permitted. This is a backend implementation contract.
pub trait ResidentResetLayer: Send + Sync + 'static {
    /// Exact source-derived descriptor for host-only construction.
    type ResetPlan: Default + PartialEq;
    /// Host-only context constructed after the reset admission succeeds.
    type ResetContext: Default;
    /// Inspects all actual layer owners without allocating or cloning them.
    fn reset_plan(_layers: &[Self]) -> Result<(Self::ResetPlan, usize), WorkingMemoryError>
    where
        Self: Sized,
    {
        Ok((Self::ResetPlan::default(), 0))
    }
    /// Constructs the inspected host context using the original reset funding.
    fn prepare_reset_context(
        _layers: &[Self],
        _plan: &Self::ResetPlan,
        _funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<Self::ResetContext, eredu_core::BackendFailure>
    where
        Self: Sized,
    {
        Ok(Self::ResetContext::default())
    }
    /// Validates the actual source placement of each selected component.
    fn matches_reset_placement(
        &self,
        _role: eredu_core::cache::StateComponentRole,
        placement: crate::StateComponentPlacement,
    ) -> bool {
        placement == crate::StateComponentPlacement::Device
    }
    /// Builds an empty layer from the exact source and prepared host context.
    fn empty_reset_prepared(
        &self,
        _context: &mut Self::ResetContext,
        policy: &LayerCachePolicy,
    ) -> Result<Self, eredu_core::BackendFailure>
    where
        Self: Sized,
    {
        Ok(Self::empty_resident_reset(policy))
    }
    /// Whether this is already the exact empty inline constructor state.
    fn reset_source_is_empty(&self) -> bool {
        false
    }

    /// Checks the existing physical policy, without reading/copying tensor data.
    fn matches_resident_reset(&self, policy: &LayerCachePolicy) -> bool;
    /// Constructs the validated empty layer through the existing policy worker.
    fn empty_resident_reset(policy: &LayerCachePolicy) -> Self;
}

#[cfg(test)]
mod tests;

/// Closed evidence that one fixed reset table remains charged in its original
/// domain. This owns metadata only; source access still needs the actual table.
/// It grants no construction, refill, generic registry insertion or native work.
#[derive(Clone, Debug)]
pub struct OriginalResidentResetSource {
    metadata: crate::HostSlotMetadata,
}
impl OriginalResidentResetSource {
    /// The exact closed token, including original custody and fixed extent.
    pub fn metadata(&self) -> &crate::HostSlotMetadata {
        &self.metadata
    }
    /// Compares constructor ownership only. Each candidate table must first be
    /// authenticated by `MemoryLedger::classify_host_slot_source`; this
    /// comparison cannot authorize an unpublished table or replace its capacity
    /// and liveness validation. It lets a checked inventory retain sibling tables
    /// under their existing single account without registering them again.
    pub fn same_constructor(&self, metadata: &crate::HostSlotMetadata) -> bool {
        metadata.original_reset_custody().is_some_and(|candidate| {
            self.metadata
                .original_reset_custody()
                .expect("validated source")
                .same_constructor(candidate)
        })
    }
    /// Entire original allowance retained by this source, not a new charge.
    pub fn original_bytes(&self) -> u64 {
        self.metadata
            .original_reset_custody()
            .expect("validated original source")
            .bytes()
    }
}
impl MemoryLedger {
    /// Pins the concrete reset source registry after successful fixed construction.
    /// No allocation, adoption, accounting increment or new budget is performed.
    /// A metadata token alone still cannot supply source values to a copy/reset.
    pub fn pin_original_reset_slots(
        &self,
        metadata: &crate::HostSlotMetadata,
    ) -> Result<OriginalResidentResetSource, WorkingMemoryError> {
        let custody = metadata
            .original_reset_custody()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        custody.validate_source(self, metadata)?;
        Ok(OriginalResidentResetSource {
            metadata: metadata.clone(),
        })
    }
}

/// Closed classification of an actual fixed table's accounting source. It does
/// not supply table contents or a new allocation authority.
#[derive(Debug)]
pub struct HostSlotSource {
    kind: HostSlotSourceKind,
}
#[derive(Debug)]
enum HostSlotSourceKind {
    Ordinary {
        key: crate::HostMetadataKey,
        bytes: u64,
    },
    Original(OriginalResidentResetSource),
}
impl HostSlotSource {
    /// Existing ordinary registration key and exact physical extent.
    pub fn registered(&self) -> Option<(&crate::HostMetadataKey, u64)> {
        match &self.kind {
            HostSlotSourceKind::Ordinary { key, bytes } => Some((key, *bytes)),
            HostSlotSourceKind::Original(_) => None,
        }
    }
    /// Original witness, authenticated in its actual pool; no registration is
    /// needed for that table. The witness is not execution or copy authority.
    pub fn into_original(self) -> Option<OriginalResidentResetSource> {
        match self.kind {
            HostSlotSourceKind::Ordinary { .. } => None,
            HostSlotSourceKind::Original(source) => Some(source),
        }
    }
}
impl MemoryLedger {
    /// Allocation-free classification. Original tables must still be live,
    /// completed, and match this pool's exact original entry and capacity.
    /// The caller retains its actual table borrow for any subsequent access.
    pub fn classify_host_slot_source(
        &self,
        metadata: &crate::HostSlotMetadata,
    ) -> Result<HostSlotSource, WorkingMemoryError> {
        let kind = if metadata.original_reset_custody().is_some() {
            HostSlotSourceKind::Original(self.pin_original_reset_slots(metadata)?)
        } else {
            HostSlotSourceKind::Ordinary {
                key: metadata.identity().registry_key().clone(),
                bytes: metadata
                    .capacity_bytes()
                    .ok_or(WorkingMemoryError::UnknownBound)?,
            }
        };
        Ok(HostSlotSource { kind })
    }
}
impl OriginalResidentResetSource {
    pub(super) fn validate_in(
        &self,
        pool: &MemoryLedger,
        usage: &super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.metadata
            .original_reset_custody()
            .expect("closed original source")
            .validate_source_in(pool, &self.metadata, usage)
    }
}

/// Library-owned ordinary source preparation. Every nonempty population retains
/// the genuine existing unquoted participant through its Vec and source teardown.
/// It provides neither original reset/copy admission nor a new byte allowance.
#[derive(Debug, Default)]
pub struct UnquotedOriginalSlotSources(Option<UnquotedSlotPopulation>);
#[derive(Debug)]
struct UnquotedSlotPopulation {
    sources: Vec<OriginalResidentResetSource>,
    lease: super::WorkingMemoryUnquotedLease,
    _host: eredu_core::HostPreparationAuthority,
}
impl UnquotedOriginalSlotSources {
    /// Exact bounded source-vector and owner controls, paid before construction.
    pub fn constructor_bytes(maximum: usize) -> Result<u64, WorkingMemoryError> {
        super::qualified_storage::array_bytes::<OriginalResidentResetSource>(maximum)?
            .checked_add(std::mem::size_of::<Self>() as u64)
            .and_then(|n| n.checked_add(std::mem::size_of::<UnquotedSlotPopulation>() as u64))
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Reuses the genuine participant and retains its already funded host envelope.
    pub fn prepare(
        lease: &super::WorkingMemoryUnquotedLease,
        maximum: usize,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self, WorkingMemoryError> {
        let sources = super::qualified_storage::vector(maximum, true)?;
        Ok(Self(Some(UnquotedSlotPopulation {
            sources,
            lease: lease.clone(),
            _host: host.clone(),
        })))
    }
    /// Retains one actual original table. Foreign, retired, or duplicate
    /// sources reject before Vec growth, preserving the existing population.
    pub fn push(&mut self, metadata: &crate::HostSlotMetadata) -> Result<(), WorkingMemoryError> {
        let population = self
            .0
            .as_mut()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let source = population
            .lease
            .inner()
            .pool
            .pin_original_reset_slots(metadata)?;
        if population
            .sources
            .iter()
            .any(|old| old.metadata.same_storage(metadata))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if population.sources.len() == population.sources.capacity() {
            return Err(WorkingMemoryError::CollectorCapacity {
                kind: super::CollectorCapacityKind::PublicationEntries,
                used: population.sources.len(),
                capacity: population.sources.capacity(),
            });
        }
        population.sources.push(source);
        Ok(())
    }
    /// Exact borrowed sources, with no owning iterator or lease extraction.
    pub fn sources(&self) -> &[OriginalResidentResetSource] {
        self.0.as_ref().map_or(&[], |p| p.sources.as_slice())
    }
    /// Whether a genuine ordinary participant is retained, including while
    /// preparing an empty population or preserving its first failure.
    pub fn has_custody(&self) -> bool {
        self.0.is_some()
    }
    /// No source has been retained, including the uninitialized default.
    pub fn is_empty(&self) -> bool {
        self.sources().is_empty()
    }
    pub(super) fn same_ledger(&self, pool: &MemoryLedger) -> bool {
        self.0
            .as_ref()
            .is_some_and(|p| p.lease.inner().pool.same_ledger(pool))
    }
    pub(super) fn take(&mut self) -> Self {
        Self(self.0.take())
    }
}
