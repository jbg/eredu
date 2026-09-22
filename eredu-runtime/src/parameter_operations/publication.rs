//! Prepared handle exchanges over the same borrowed parameter traversal.
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError};
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};

struct ReplacementRows<T> {
    rows: Vec<(String, T)>,
    // Rows, names and native wrappers retire before their preparation account.
    _funding: HostMetadataFunding,
}

/// Immutable completed replacement roots, shared by their actual future loaders.
/// Names address parameter slots; allocation sharing still requires storage facts.
pub struct ParameterReplacementValues<T>(Option<Arc<ReplacementRows<T>>>);

impl<T: std::fmt::Debug> std::fmt::Debug for ParameterReplacementValues<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}
impl<T> Default for ParameterReplacementValues<T> {
    fn default() -> Self {
        Self(None)
    }
}
impl<T> Clone for ParameterReplacementValues<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> Drop for ParameterReplacementValues<T> {
    fn drop(&mut self) {
        // No raw or weak owner escapes. Free the last shared shell before its
        // rows and retained funding, including concurrent alias retirement.
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl<T> ParameterReplacementValues<T> {
    /// Moves already funded rows and their payer into a separately priced shell.
    /// The producer funds vector capacity, names, values and their construction
    /// before creating `rows`; this function neither authenticates native storage
    /// nor makes an ordinary source complete.
    pub fn from_prepared_rows(
        rows: Vec<(String, T)>,
        funding: HostMetadataFunding,
        context: &WorkspaceContext,
    ) -> Result<Self, ParameterPublicationFailure> {
        if !context
            .metadata_funding()
            .as_ref()
            .is_some_and(|owner| owner.same_account(&funding))
        {
            return Err(ParameterPublicationFailure::Identity);
        }
        let frames = [
            size_of::<Self>(),
            size_of::<Option<ReplacementRows<T>>>(),
            size_of::<Result<Self, ParameterPublicationFailure>>(),
        ];
        context.charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(ParameterPublicationFailure::Overflow)?,
        )?;
        let mut owned = ReplacementRows {
            rows,
            _funding: funding,
        };
        owned.rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        if owned.rows.iter().any(|row| row.0.is_empty())
            || owned.rows.windows(2).any(|pair| pair[0].0 == pair[1].0)
        {
            return Err(ParameterPublicationFailure::Identity);
        }
        if owned.rows.is_empty() {
            return Ok(Self::default());
        }
        let rows = context
            .metadata_arc(owned)
            .map_err(ParameterPublicationFailure::Metadata)?;
        Ok(Self(Some(rows)))
    }
    /// Number of retained parameter roots.
    pub fn len(&self) -> usize {
        self.rows().len()
    }
    /// Whether this owner retains any override.
    pub fn is_empty(&self) -> bool {
        self.rows().is_empty()
    }
    /// Borrow the completed value addressed by this parameter identity.
    pub fn get(&self, id: &str) -> Option<&T> {
        self.index(id).map(|index| &self.rows()[index].1)
    }
    /// Whether this immutable owner contains the given parameter identity.
    pub fn contains_key(&self, id: &str) -> bool {
        self.index(id).is_some()
    }
    /// Borrow retained native values without creating handles.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.rows().iter().map(|row| &row.1)
    }
    /// Borrow exact names and values without reconstructing a map.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &T)> {
        self.rows().iter().map(|row| (row.0.as_str(), &row.1))
    }
    /// Immutable source identity, independent of value equality.
    pub fn same_source(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    fn rows(&self) -> &[(String, T)] {
        self.0.as_ref().map_or(&[], |rows| &rows.rows)
    }
    fn index(&self, id: &str) -> Option<usize> {
        self.rows()
            .binary_search_by(|row| row.0.as_str().cmp(id))
            .ok()
    }
}

/// The same actual slots and future source owners participate in every pass.
/// Implementations only lend their retained fields; a traversal must not create
/// storage, perform native work, or change participant membership.
pub trait ParameterPublication<T> {
    /// Lends one loaded slot. Unselected identities remain unchanged.
    fn slot(&mut self, id: &str, value: &mut T);
    /// Lends a counter whose proposal is computed only during preparation.
    /// Validation and exchange never call the producer.
    fn counter(
        &mut self,
        value: &mut u64,
        prepare: &mut dyn FnMut(
            &ParameterReplacementValues<T>,
            bool,
        ) -> Result<u64, eredu_nn::Error>,
    );
    /// Lends one future-loader source. Publication moves one prepared shared owner.
    fn replacement_source(&mut self, source: &mut ParameterReplacementValues<T>);
}

/// Adapter for the existing allocation-free mutable parameter traversal.
pub struct ParameterPublicationVisitor<'a, T>(pub &'a mut dyn ParameterPublication<T>);
impl<'a, T: 'a> eredu_nn::ParameterVisitorMut<'a, T> for ParameterPublicationVisitor<'_, T> {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut T) {
        self.0.slot(metadata.id().as_str(), value);
    }
}

/// Fixed failures detected before any prepared exchange.
#[derive(Debug, thiserror::Error)]
pub enum ParameterPublicationFailure {
    /// The participant, parameter identity or immutable source changed.
    #[error("parameter publication source or participant changed")]
    Identity,
    /// No complete traversal exists for the actual owner.
    #[error("parameter publication traversal is incomplete")]
    Unavailable,
    /// Counts and constructor sizes use checked arithmetic under every limit.
    #[error("parameter publication population overflow")]
    Overflow,
    /// Host storage could not be prepared.
    #[error(transparent)]
    Metadata(#[from] WorkspaceMetadataError),
}

/// Preparation keeps native producer errors distinct from participant refusals.
#[derive(Debug, thiserror::Error)]
pub enum ParameterPublicationError<E> {
    /// Fixed contract refusal; no participant was mutated.
    #[error(transparent)]
    Contract(#[from] ParameterPublicationFailure),
    /// The retained owner refused traversal before publication.
    #[error("parameter publication traversal failed")]
    Participant(#[source] E),
    /// The selected value constructor or identity observer refused preparation.
    #[error(transparent)]
    Value(eredu_nn::Error),
}

struct Slot<T> {
    address: usize,
    row: usize,
    expected: T,
    exchange: T,
}
struct CounterSlot {
    address: usize,
    before: u64,
    after: u64,
    exchange: u64,
}
struct Source<T> {
    address: usize,
    before: ParameterReplacementValues<T>,
    after: ParameterReplacementValues<T>,
    exchange: ParameterReplacementValues<T>,
}

/// Move-only prepared exchanges. Displaced values stay here until the caller
/// releases participant locks and retires this owner. An exchange may be reversed
/// with the same owner after validation, without allocating rollback handles.
pub struct PreparedParameterPublication<T> {
    values: ParameterReplacementValues<T>,
    slots: Vec<Slot<T>>,
    sources: Vec<Source<T>>,
    counters: Vec<CounterSlot>,
    exchanged: bool,
    _funding: HostMetadataFunding,
}

impl<T> PreparedParameterPublication<T> {
    /// Count and prepare every handle before mutating a participant. `clone_value`
    /// is the selected producer's funded handle constructor, never `T::clone`.
    pub fn prepare<E>(
        values: ParameterReplacementValues<T>,
        active: bool,
        mut visit: impl FnMut(&mut dyn ParameterPublication<T>) -> Result<bool, E>,
        mut clone_value: impl FnMut(&T) -> Result<T, eredu_nn::Error>,
        context: &WorkspaceContext,
        funding: HostMetadataFunding,
    ) -> Result<Self, ParameterPublicationError<E>> {
        if !context
            .metadata_funding()
            .as_ref()
            .is_some_and(|owner| owner.same_account(&funding))
        {
            return Err(ParameterPublicationFailure::Identity.into());
        }
        let frames = [
            size_of::<Self>(),
            size_of::<Counter<'_, T>>(),
            size_of::<Filler<'_, T>>(),
            size_of::<Validator<'_, T>>(),
            size_of::<Exchanger<'_, T>>(),
            size_of::<Result<Self, ParameterPublicationError<E>>>(),
            size_of::<ParameterPublicationFailure>(),
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(ParameterPublicationFailure::Overflow)?,
            )
            .map_err(ParameterPublicationFailure::Metadata)?;
        let mut count = Counter {
            values: &values,
            slots: Some(0),
            sources: Some(0),
            counters: Some(0),
        };
        if !visit(&mut count).map_err(ParameterPublicationError::Participant)? {
            return Err(ParameterPublicationFailure::Unavailable.into());
        }
        let slot_count = count.slots.ok_or(ParameterPublicationFailure::Overflow)?;
        let source_count = count.sources.ok_or(ParameterPublicationFailure::Overflow)?;
        let counter_count = count
            .counters
            .ok_or(ParameterPublicationFailure::Overflow)?;
        let mut out = Self {
            values,
            slots: context
                .metadata_vec(slot_count)
                .map_err(ParameterPublicationError::Value)?,
            sources: context
                .metadata_vec(source_count)
                .map_err(ParameterPublicationError::Value)?,
            counters: context
                .metadata_vec(counter_count)
                .map_err(ParameterPublicationError::Value)?,
            exchanged: false,
            _funding: funding,
        };
        let mut fill = Filler {
            out: &mut out,
            active,
            slots: slot_count,
            sources: source_count,
            counters: counter_count,
            clone_value: &mut clone_value,
            failure: None,
        };
        if !visit(&mut fill).map_err(ParameterPublicationError::Participant)? {
            return Err(ParameterPublicationFailure::Unavailable.into());
        }
        if let Some(cause) = fill.failure {
            return Err(ParameterPublicationError::Value(cause));
        }
        if out.slots.len() != slot_count
            || out.sources.len() != source_count
            || out.counters.len() != counter_count
        {
            return Err(ParameterPublicationFailure::Identity.into());
        }
        Ok(out)
    }

    /// Validate every participant and actual current value before an exchange.
    /// The caller separately authenticates the executable and holds its exclusive
    /// operation lease continuously through validation and the ensuing exchange.
    pub fn validate<E>(
        &self,
        mut visit: impl FnMut(&mut dyn ParameterPublication<T>) -> Result<bool, E>,
        mut same_value: impl FnMut(&T, &T) -> Result<bool, eredu_nn::Error>,
    ) -> Result<(), ParameterPublicationError<E>> {
        let mut check = Validator {
            publication: self,
            slots: 0,
            sources: 0,
            counters: 0,
            valid: true,
            same_value: &mut same_value,
            failure: None,
        };
        if !visit(&mut check).map_err(ParameterPublicationError::Participant)? {
            return Err(ParameterPublicationFailure::Unavailable.into());
        }
        if let Some(cause) = check.failure {
            return Err(ParameterPublicationError::Value(cause));
        }
        if !check.valid
            || check.slots != self.slots.len()
            || check.sources != self.sources.len()
            || check.counters != self.counters.len()
        {
            return Err(ParameterPublicationFailure::Identity.into());
        }
        Ok(())
    }

    /// Exchange already validated, exclusively borrowed participants. The visitor
    /// and this worker perform only moves and scalar comparisons. No displaced
    /// payload is destroyed here. A changed traversal is an invariant violation,
    /// and the enclosing execution owner must poison or quarantine on unwinding.
    pub fn exchange(&mut self, visit: impl FnOnce(&mut dyn ParameterPublication<T>)) {
        let mut swap = Exchanger {
            publication: self,
            slots: 0,
            sources: 0,
            counters: 0,
        };
        visit(&mut swap);
        assert_eq!(
            swap.slots,
            swap.publication.slots.len(),
            "validated parameter slots"
        );
        assert_eq!(
            swap.sources,
            swap.publication.sources.len(),
            "validated parameter sources"
        );
        assert_eq!(
            swap.counters,
            swap.publication.counters.len(),
            "validated parameter counters"
        );
        self.exchanged = !self.exchanged;
    }
}

struct Counter<'a, T> {
    values: &'a ParameterReplacementValues<T>,
    slots: Option<usize>,
    sources: Option<usize>,
    counters: Option<usize>,
}
impl<T> ParameterPublication<T> for Counter<'_, T> {
    fn counter(
        &mut self,
        _value: &mut u64,
        _prepare: &mut dyn FnMut(
            &ParameterReplacementValues<T>,
            bool,
        ) -> Result<u64, eredu_nn::Error>,
    ) {
        self.counters = self.counters.and_then(|n| n.checked_add(1));
    }
    fn slot(&mut self, id: &str, _: &mut T) {
        if self.values.index(id).is_some() {
            self.slots = self.slots.and_then(|n| n.checked_add(1));
        }
    }
    fn replacement_source(&mut self, _: &mut ParameterReplacementValues<T>) {
        self.sources = self.sources.and_then(|n| n.checked_add(1));
    }
}
struct Filler<'a, T> {
    out: &'a mut PreparedParameterPublication<T>,
    active: bool,
    slots: usize,
    sources: usize,
    counters: usize,
    clone_value: &'a mut dyn FnMut(&T) -> Result<T, eredu_nn::Error>,
    failure: Option<eredu_nn::Error>,
}
impl<T> ParameterPublication<T> for Filler<'_, T> {
    fn counter(
        &mut self,
        value: &mut u64,
        prepare: &mut dyn FnMut(
            &ParameterReplacementValues<T>,
            bool,
        ) -> Result<u64, eredu_nn::Error>,
    ) {
        if self.failure.is_some() {
            return;
        }
        if self.out.counters.len() == self.counters {
            self.failure = Some(WorkspaceMetadataError::Unqualified.into());
            return;
        }
        match prepare(&self.out.values, self.active) {
            Ok(after) => self.out.counters.push(CounterSlot {
                address: std::ptr::from_mut(value).addr(),
                before: *value,
                after,
                exchange: after,
            }),
            Err(cause) => self.failure = Some(cause),
        }
    }
    fn slot(&mut self, id: &str, value: &mut T) {
        if self.failure.is_some() {
            return;
        }
        let Some(row) = self.out.values.index(id) else {
            return;
        };
        if self.out.slots.len() == self.slots {
            self.failure = Some(WorkspaceMetadataError::Unqualified.into());
            return;
        }
        let values = (|| {
            Ok::<_, eredu_nn::Error>((
                (self.clone_value)(value)?,
                (self.clone_value)(&self.out.values.rows()[row].1)?,
            ))
        })();
        match values {
            Ok((expected, exchange)) => self.out.slots.push(Slot {
                address: std::ptr::from_mut(value).addr(),
                row,
                expected,
                exchange,
            }),
            Err(cause) => self.failure = Some(cause),
        }
    }
    fn replacement_source(&mut self, source: &mut ParameterReplacementValues<T>) {
        if self.failure.is_some() {
            return;
        }
        if self.out.sources.len() == self.sources {
            self.failure = Some(WorkspaceMetadataError::Unqualified.into());
            return;
        }
        let after = if self.active {
            self.out.values.clone()
        } else {
            ParameterReplacementValues::default()
        };
        self.out.sources.push(Source {
            address: std::ptr::from_mut(source).addr(),
            before: source.clone(),
            exchange: after.clone(),
            after,
        });
    }
}
struct Validator<'a, T> {
    publication: &'a PreparedParameterPublication<T>,
    slots: usize,
    sources: usize,
    counters: usize,
    valid: bool,
    same_value: &'a mut dyn FnMut(&T, &T) -> Result<bool, eredu_nn::Error>,
    failure: Option<eredu_nn::Error>,
}
impl<T> ParameterPublication<T> for Validator<'_, T> {
    fn counter(
        &mut self,
        value: &mut u64,
        _prepare: &mut dyn FnMut(
            &ParameterReplacementValues<T>,
            bool,
        ) -> Result<u64, eredu_nn::Error>,
    ) {
        let Some(slot) = self.publication.counters.get(self.counters) else {
            self.valid = false;
            return;
        };
        self.counters = self
            .counters
            .checked_add(1)
            .expect("bounded parameter counters");
        let expected = if self.publication.exchanged {
            slot.after
        } else {
            slot.before
        };
        self.valid &= slot.address == std::ptr::from_mut(value).addr() && *value == expected;
    }
    fn slot(&mut self, id: &str, value: &mut T) {
        let Some(row) = self.publication.values.index(id) else {
            return;
        };
        let Some(slot) = self.publication.slots.get(self.slots) else {
            self.valid = false;
            return;
        };
        self.slots = self.slots.checked_add(1).expect("bounded parameter slots");
        self.valid &= slot.row == row && slot.address == std::ptr::from_mut(value).addr();
        if !self.valid || self.failure.is_some() {
            return;
        }
        let expected = if self.publication.exchanged {
            &self.publication.values.rows()[row].1
        } else {
            &slot.expected
        };
        match (self.same_value)(value, expected) {
            Ok(valid) => self.valid &= valid,
            Err(cause) => self.failure = Some(cause),
        }
    }
    fn replacement_source(&mut self, source: &mut ParameterReplacementValues<T>) {
        let Some(slot) = self.publication.sources.get(self.sources) else {
            self.valid = false;
            return;
        };
        self.sources = self
            .sources
            .checked_add(1)
            .expect("bounded parameter sources");
        let expected = if self.publication.exchanged {
            &slot.after
        } else {
            &slot.before
        };
        self.valid &=
            slot.address == std::ptr::from_mut(source).addr() && source.same_source(expected);
    }
}
struct Exchanger<'a, T> {
    publication: &'a mut PreparedParameterPublication<T>,
    slots: usize,
    sources: usize,
    counters: usize,
}
impl<T> ParameterPublication<T> for Exchanger<'_, T> {
    fn counter(
        &mut self,
        value: &mut u64,
        _prepare: &mut dyn FnMut(
            &ParameterReplacementValues<T>,
            bool,
        ) -> Result<u64, eredu_nn::Error>,
    ) {
        let slot = &mut self.publication.counters[self.counters];
        assert_eq!(
            std::ptr::from_mut(value).addr(),
            slot.address,
            "validated parameter counter"
        );
        self.counters = self
            .counters
            .checked_add(1)
            .expect("bounded parameter counters");
        std::mem::swap(value, &mut slot.exchange);
    }
    fn slot(&mut self, id: &str, value: &mut T) {
        let Some(row) = self.publication.values.index(id) else {
            return;
        };
        let slot = &mut self.publication.slots[self.slots];
        assert_eq!(
            (row, std::ptr::from_mut(value).addr()),
            (slot.row, slot.address),
            "validated parameter slot"
        );
        self.slots = self.slots.checked_add(1).expect("bounded parameter slots");
        std::mem::swap(value, &mut slot.exchange);
    }
    fn replacement_source(&mut self, source: &mut ParameterReplacementValues<T>) {
        let slot = &mut self.publication.sources[self.sources];
        assert_eq!(
            std::ptr::from_mut(source).addr(),
            slot.address,
            "validated parameter source"
        );
        self.sources = self
            .sources
            .checked_add(1)
            .expect("bounded parameter sources");
        std::mem::swap(source, &mut slot.exchange);
    }
}

#[cfg(test)]
mod tests;
