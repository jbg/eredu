//! Concrete context-owned allocation requests, shared by census and construction.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

mod copy;
mod query;
pub use copy::visit_isolated_copy_operations;
pub use query::{WorkspaceContextMetadataBuilder, WorkspaceContextMetadataError};

/// Immutable cumulative capacities for the represented context constructors.
/// This contains no source, execution or funding authority. Shape, state,
/// equation temporaries and source geometry require their own complete census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkspaceMetadataEnvelope {
    pub(super) facts: usize,
    pub(super) context: usize,
    pub(super) reports: usize,
}
impl WorkspaceMetadataEnvelope {
    /// Adds complete observed spans without refunding retired metadata.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            facts: self.facts.checked_add(other.facts)?,
            context: self.context.checked_add(other.context)?,
            reports: self.reports.checked_add(other.reports)?,
        })
    }
    /// Separate fact buffer and typed-error construction allowance.
    pub fn fact_bytes(self) -> usize {
        self.facts
    }
    /// Allowance for the participating context-owned constructors.
    pub fn context_bytes(self) -> usize {
        self.context
    }
    /// Actual scalar reports constructed by the participating source program.
    pub fn report_count(self) -> usize {
        self.reports
    }

    /// Sum of the represented domains, not a whole request memory bound.
    pub fn represented_bytes(self) -> Option<usize> {
        self.facts.checked_add(self.context)
    }
}

impl WorkspaceContext {
    /// Whether this context records and checks participating metadata producers.
    /// This does not depend on a census result, which can be absent on overflow.
    pub fn uses_checked_metadata(&self) -> bool {
        self.facts.is_some()
    }

    /// Debits one concrete participating metadata producer before allocation.
    /// Legacy contexts have no finite allowance. This is not an admission grant.
    pub fn charge_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataError> {
        if self.facts.is_none() {
            return Ok(());
        }
        let available = self.identity.metadata_remaining.get();
        let remaining = available
            .checked_sub(bytes)
            .ok_or(WorkspaceMetadataError::Capacity {
                required: bytes,
                available,
            })?;
        let total = self
            .identity
            .metadata_bytes
            .get()
            .and_then(|v| v.checked_add(bytes))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if let Some(funding) = &self.funding {
            funding.reserve_metadata(bytes)?;
        }
        self.identity.metadata_remaining.set(remaining);
        self.identity.metadata_bytes.set(Some(total));
        Ok(())
    }

    /// Exact requested shared shell plus fixed constructor transports. Nested
    /// payload ownership remains with the concrete caller's separate query.
    pub fn metadata_rc_bytes<T>() -> Option<usize> {
        let layout = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align();
        let parts = [
            layout.size(),
            size_of::<T>(),
            size_of::<Rc<T>>(),
            size_of::<Result<Rc<T>, WorkspaceMetadataError>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Constructs one ordinary Rc after its exact shell debit. This contains no
    /// authority erasure; a caller retaining funding must close its own drop order.
    pub fn metadata_rc<T>(&self, value: T) -> Result<Rc<T>, WorkspaceMetadataError> {
        self.charge_metadata(
            Self::metadata_rc_bytes::<T>().ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        Ok(Rc::new(value))
    }

    /// Exact atomic shared shell and constructor transports. Nested payloads
    /// and their retirement custody remain the concrete caller's responsibility.
    pub fn metadata_arc_bytes<T>() -> Option<usize> {
        use std::sync::{Arc, atomic::AtomicUsize};
        let layout = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<T>()).ok()?.0.pad_to_align();
        let parts = [layout.size(), size_of::<T>(), size_of::<Arc<T>>(),
            size_of::<Result<Arc<T>, WorkspaceMetadataError>>()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Constructs the ordinary Arc after reserving its exact shared shell.
    /// This supplies storage only, never source or execution permission.
    pub fn metadata_arc<T>(&self, value: T) -> Result<std::sync::Arc<T>, WorkspaceMetadataError> {
        self.charge_metadata(Self::metadata_arc_bytes::<T>().ok_or(WorkspaceMetadataError::Overflow)?)?;
        Ok(std::sync::Arc::new(value))
    }

    /// Reserves the next actual vector backing before a metadata producer can
    /// grow. Source recording and finite construction use the same geometric
    /// growth. The old and new requests are charged cumulatively; no refund is
    /// inferred from a reallocation or a completed span.
    pub fn reserve_metadata_vec<T>(
        &self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), Error> {
        let required = values
            .len()
            .checked_add(additional)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if required <= values.capacity() {
            return Ok(());
        }
        if self.facts.is_none() {
            return values
                .try_reserve(additional)
                .map_err(Error::backend_retained_source);
        }
        // The requested length, not allocator-provided spare capacity, fixes
        // the next request. A source query can therefore enumerate the same
        // powers of two even when try_reserve_exact returns extra capacity.
        let target = required
            .checked_next_power_of_two()
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.reserve_metadata_exact(values, target)
    }

    /// One requested vector backing and the same fixed reserve/error controls
    /// used by metadata_vec. Empty and zero-sized vectors request no allocation.
    pub fn metadata_vec_bytes<T>(capacity: usize) -> Option<usize> {
        if capacity == 0 || size_of::<T>() == 0 {
            return Some(0);
        }
        let parts = [
            Layout::array::<T>(capacity).ok()?.size(),
            Error::retained_source_construction_bytes::<TryReserveError>()?,
            size_of::<Vec<T>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Vec<T>, Error>>(),
            size_of::<(&mut Vec<T>, usize)>(),
            size_of::<MetadataDestination<'_>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Cumulative requests for growing a vector from an already-priced initial
    /// capacity to this final length. Extra allocator capacity may skip requests,
    /// but cannot introduce a request absent from this powers-of-two sequence.
    pub fn metadata_vec_growth_bytes<T>(initial: usize, final_length: usize) -> Option<usize> {
        if final_length <= initial || size_of::<T>() == 0 {
            return Some(0);
        }
        let mut target = initial.checked_add(1)?.checked_next_power_of_two()?;
        let final_target = final_length.checked_next_power_of_two()?;
        let mut bytes = 0usize;
        loop {
            bytes = bytes.checked_add(Self::metadata_vec_bytes::<T>(target)?)?;
            if target == final_target {
                return Some(bytes);
            }
            target = target.checked_mul(2)?;
        }
    }

    /// Creates one exact-capacity vector using the same debit as growth.
    pub fn metadata_vec<T>(&self, capacity: usize) -> Result<Vec<T>, Error> {
        let mut values = Vec::new();
        self.reserve_metadata_exact(&mut values, capacity)?;
        Ok(values)
    }

    fn reserve_metadata_exact<T>(&self, values: &mut Vec<T>, capacity: usize) -> Result<(), Error> {
        MetadataDestination::Context(self).reserve_vec(values, capacity)
    }

    pub(super) fn try_new_storage(
        &self,
        bytes: Option<u64>,
        possible_aliases: Vec<Rc<Storage>>,
    ) -> Result<Rc<Storage>, Error> {
        let layout = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Storage>())
            .map_err(|_| WorkspaceMetadataError::Overflow)?
            .0
            .pad_to_align();
        let controls = [
            layout.size(),
            size_of::<Storage>(),
            size_of::<Rc<Storage>>(),
            size_of::<Result<Rc<Storage>, Error>>(),
        ];
        let amount = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.charge_metadata(amount)?;
        Ok(self.new_storage(bytes, possible_aliases))
    }

    pub(super) fn insert_assumption(
        &self,
        values: &mut Vec<String>,
        value: String,
    ) -> Result<(), Error> {
        if let Err(index) = values.binary_search(&value) {
            self.reserve_metadata_vec(values, 1)?;
            values.insert(index, value);
        }
        Ok(())
    }

    /// Census since the last consuming report, including trailing report or
    /// plan construction. This value carries no allocation authority.
    pub fn metadata_census(&self) -> Option<WorkspaceMetadataEnvelope> {
        Some(WorkspaceMetadataEnvelope {
            facts: self.identity.fact_bytes.get()?,
            context: self.identity.metadata_bytes.get()?,
            reports: self.identity.metadata_reports.get()?,
        })
    }
}

struct TextCount(usize);
impl fmt::Write for TextCount {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.0 = self.0.checked_add(value.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}
struct TextOutput(Vec<u8>);
impl fmt::Write for TextOutput {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        if value.len() > self.0.capacity() - self.0.len() {
            return Err(fmt::Error);
        }
        self.0.extend_from_slice(value.as_bytes());
        Ok(())
    }
}
impl WorkspaceContext {
    fn metadata_string_control_bytes() -> Option<usize> {
        let controls = [
            size_of::<MetadataDestination<'_>>(),
            size_of::<TextCount>(),
            size_of::<TextOutput>(),
            size_of::<fmt::Arguments<'_>>(),
            size_of::<String>(),
            size_of::<Result<String, Error>>(),
            size_of::<fmt::Error>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }

    /// Exact deterministic UTF-8 buffer and formatter controls used by
    /// metadata_string. The caller supplies its already-counted byte length.
    pub fn metadata_string_bytes(length: usize) -> Option<usize> {
        Self::metadata_string_control_bytes()?.checked_add(Self::metadata_vec_bytes::<u8>(length)?)
    }

    /// Counts deterministic borrowed formatting, admits its exact UTF-8 buffer,
    /// then emits through a capacity-checked writer. Callers must supply only
    /// formatting whose own Display implementations do not allocate or mutate;
    /// this does not qualify arbitrary Display callbacks or prebuilt strings.
    pub fn metadata_string(&self, arguments: fmt::Arguments<'_>) -> Result<String, Error> {
        MetadataDestination::Context(self).string(arguments)
    }
}

impl WorkspaceContext {
    fn metadata_error_control_bytes() -> Option<usize> {
        let controls = [
            size_of::<fmt::Arguments<'_>>(),
            size_of::<Error>(),
            size_of::<Result<String, Error>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }

    /// Exact paid diagnostic producer, including its consumed UTF-8 buffer.
    pub fn metadata_error_bytes(length: usize) -> Option<usize> {
        Self::metadata_error_control_bytes()?.checked_add(Self::metadata_string_bytes(length)?)
    }

    /// Constructs a counted deterministic diagnostic and moves its buffer into
    /// the existing neutral Error. Capacity/format refusal itself stays inline;
    /// no fallback formatter or second String/Arc allocation is attempted.
    pub fn metadata_error(&self, arguments: fmt::Arguments<'_>) -> Error {
        MetadataDestination::Context(self).error(arguments)
    }
}

impl WorkspaceContext {
    /// Exact closed typed-error owner and the same fixed transports consumed by
    /// metadata_source. The source is not formatted, cloned, or constructed here.
    pub fn metadata_source_bytes<E: std::error::Error + Send + Sync + 'static>() -> Option<usize> {
        let controls = [
            size_of::<MetadataDestination<'_>>(),
            Error::retained_source_construction_bytes::<E>()?,
            size_of::<E>(),
            size_of::<Error>(),
            size_of::<Result<(), WorkspaceMetadataError>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }

    /// Retains an exact typed construction cause after admitting its closed
    /// source/control owner. No eager diagnostic formatting occurs on a finite
    /// context. On debit refusal, the owned cause retires before the returned
    /// inline metadata error; no new wrapper allocation is attempted.
    pub fn metadata_source<E: std::error::Error + Send + Sync + 'static>(&self, cause: E) -> Error {
        MetadataDestination::Context(self).source(cause)
    }
}

// One producer, with either Context census/finite checks or the exact retained
// host account. Funding never creates a synthetic Context or a second allowance.
#[derive(Clone, Copy)]
enum MetadataDestination<'a> {
    Context(&'a WorkspaceContext),
    Funding(&'a HostMetadataFunding),
}
impl MetadataDestination<'_> {
    fn checked(self) -> bool {
        match self {
            Self::Context(context) => context.uses_checked_metadata(),
            Self::Funding(_) => true,
        }
    }
    fn charge(self, bytes: usize) -> Result<(), WorkspaceMetadataError> {
        match self {
            Self::Context(context) => context.charge_metadata(bytes),
            Self::Funding(funding) => funding.reserve_metadata(bytes).map_err(Into::into),
        }
    }
    fn reserve_vec<T>(self, values: &mut Vec<T>, capacity: usize) -> Result<(), Error> {
        if capacity <= values.capacity() {
            return Ok(());
        }
        let bytes = WorkspaceContext::metadata_vec_bytes::<T>(capacity)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.charge(bytes)?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(Error::backend_retained_source)
    }
    fn vector<T>(self, capacity: usize) -> Result<Vec<T>, Error> {
        let mut values = Vec::new();
        self.reserve_vec(&mut values, capacity)?;
        Ok(values)
    }
    fn string(self, arguments: fmt::Arguments<'_>) -> Result<String, Error> {
        let mut count = TextCount(0);
        fmt::write(&mut count, arguments).map_err(|_| WorkspaceMetadataError::Text)?;
        let amount = WorkspaceContext::metadata_string_control_bytes()
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.charge(amount)?;
        let mut output = TextOutput(self.vector(count.0)?);
        fmt::write(&mut output, arguments).map_err(|_| WorkspaceMetadataError::Text)?;
        if output.0.len() != count.0 {
            return Err(WorkspaceMetadataError::Text.into());
        }
        // The private writer only appends complete UTF-8 str slices.
        Ok(String::from_utf8(output.0).expect("metadata writer emits valid UTF-8"))
    }
    fn source<E: std::error::Error + Send + Sync + 'static>(self, cause: E) -> Error {
        if !self.checked() {
            return Error::backend_retained_source(cause);
        }
        let Some(amount) = WorkspaceContext::metadata_source_bytes::<E>() else {
            return WorkspaceMetadataError::Overflow.into();
        };
        match self.charge(amount) {
            Ok(()) => Error::backend_retained_source(cause),
            Err(error) => error.into(),
        }
    }
}
impl MetadataDestination<'_> {
    fn error(self, arguments: fmt::Arguments<'_>) -> Error {
        let Some(amount) = WorkspaceContext::metadata_error_control_bytes() else {
            return WorkspaceMetadataError::Overflow.into();
        };
        if let Err(cause) = self.charge(amount) {
            return cause.into();
        }
        match self.string(arguments) {
            Ok(message) => Error::backend_message(message),
            Err(cause) => cause,
        }
    }
}
/// NN metadata allocation helpers backed by the canonical core account.
/// These helpers apply workspace growth and error policy; they introduce no
/// account wrapper, additional allocation authority, or retirement owner.
pub trait WorkspaceMetadataAllocation {
    /// Builds a counted NN diagnostic; an escaping owner must retain funding.
    fn metadata_error(&self, arguments: fmt::Arguments<'_>) -> Error;
    /// Allocates an exact-capacity vector after the cumulative account debit.
    fn metadata_vec<T>(&self, capacity: usize) -> Result<Vec<T>, Error>;
    /// Grows a vector using the workspace's counted powers-of-two policy.
    fn reserve_metadata_vec<T>(&self, values: &mut Vec<T>, additional: usize) -> Result<(), Error>;
    /// Builds counted UTF-8 text, retaining no independent account alias.
    fn metadata_string(&self, arguments: fmt::Arguments<'_>) -> Result<String, Error>;
    /// Constructs a counted NN source error; its owner must retain funding.
    fn metadata_source<E: std::error::Error + Send + Sync + 'static>(&self, cause: E) -> Error;
}
impl WorkspaceMetadataAllocation for HostMetadataFunding {
    /// Same counted diagnostic producer, without introducing a Context. The
    /// enclosing returned/error owner must retain an independent funding alias.
    fn metadata_error(&self, arguments: fmt::Arguments<'_>) -> Error {
        MetadataDestination::Funding(self).error(arguments)
    }
    /// Builds the same exact-capacity metadata vector against this retained
    /// account. Keep an independent funding alias with any escaping result/error.
    fn metadata_vec<T>(&self, capacity: usize) -> Result<Vec<T>, Error> {
        MetadataDestination::Funding(self).vector(capacity)
    }
    /// Grows a metadata vector through the same exact backing/error producer as
    /// checked Context growth. Each request remains cumulatively charged.
    fn reserve_metadata_vec<T>(&self, values: &mut Vec<T>, additional: usize) -> Result<(), Error> {
        let required = values.len().checked_add(additional).ok_or(WorkspaceMetadataError::Overflow)?;
        if required <= values.capacity() { return Ok(()); }
        let target = required.checked_next_power_of_two().ok_or(WorkspaceMetadataError::Overflow)?;
        MetadataDestination::Funding(self).reserve_vec(values, target)
    }
    /// Uses the Context's counted UTF-8 producer with this actual account.
    /// Formatting must be deterministic and allocation-free. Retain funding
    /// independently through the returned text or error's complete lifetime.
    fn metadata_string(&self, arguments: fmt::Arguments<'_>) -> Result<String, Error> {
        MetadataDestination::Funding(self).string(arguments)
    }
    /// Uses the same paid closed error-source producer without creating a Context.
    /// The returned cause does not itself retain funding; its enclosing owner must.
    fn metadata_source<E: std::error::Error + Send + Sync + 'static>(&self, cause: E) -> Error {
        MetadataDestination::Funding(self).source(cause)
    }
}
