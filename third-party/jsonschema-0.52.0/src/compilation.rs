//! Prospective storage for the ordinary schema compiler.
use crate::ValidationError;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    error::Error,
    fmt,
    mem::size_of,
    rc::Rc,
    sync::{atomic::AtomicUsize, Arc},
};

/// The original account used by a schema compilation.
///
/// Accepted reservations remain charged until the compiled validator or error
/// releases this source. A refusal must preserve its concrete cause in the
/// source so [`CompilationError`] can borrow it without allocating an error.
pub trait CompilationFunding: Send + Sync {
    /// Reserve the reached allocation before its producer runs.
    fn reserve(&self, bytes: usize) -> Result<(), CompilationAllocationError>;
    /// The original failure retained by this account, when present.
    fn source_error(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        None
    }
}

/// A fixed refusal from a reached compiler storage operation.
#[derive(Debug)]
pub enum CompilationAllocationError {
    /// The original account refused the allocation.
    Refused,
    /// The requested layout cannot be represented.
    Overflow,
    /// The host allocator refused the requested vector allocation.
    Allocation(TryReserveError),
    /// The owning hash table refused its requested allocation.
    Table(hashbrown::TryReserveError),
    /// The actual destination did not match its requested allocation.
    Capacity,
    /// The underlying host producer refused allocation.
    HostAllocation,
    /// An extension has no prospective construction contract.
    Unqualified(&'static str),
}
impl fmt::Display for CompilationAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused => f.write_str("schema compilation allocation refused"),
            Self::Overflow => f.write_str("schema compilation allocation extent overflow"),
            Self::Allocation(error) => error.fmt(f),
            Self::Table(error) => error.fmt(f),
            Self::HostAllocation => f.write_str("schema compilation host allocation failed"),
            Self::Capacity => f.write_str("schema compilation allocation destination changed"),
            Self::Unqualified(component) => {
                write!(f, "schema compilation source is unqualified: {component}")
            }
        }
    }
}
impl Error for CompilationAllocationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::Table(error) => Some(error),
            _ => None,
        }
    }
}

/// A compiler failure retaining the original allocation source.
#[derive(Debug)]
pub struct CompilationError {
    cause: Cause,
    // Keep the account alive until every owned diagnostic has been retired.
    funding: Funding,
}
#[derive(Debug)]
enum Cause {
    Allocation(CompilationAllocationError),
    Schema(ValidationError<'static>),
    JsonSource(serde_json::bounded_events::ValueError),
    Reference(referencing::Error),
    Entropy(getrandom::Error),
}
impl CompilationError {
    /// The reached allocation refusal, if compilation failed on storage.
    pub fn allocation_error(&self) -> Option<&CompilationAllocationError> {
        match &self.cause {
            Cause::Allocation(error) => Some(error),
            _ => None,
        }
    }
    /// The actual reference-resolution diagnostic, when that worker failed.
    /// Callers must distinguish its allocation and unqualified-producer variants
    /// before treating an unresolved local reference as a semantic condition.
    pub fn reference_error(&self) -> Option<&referencing::Error> {
        match &self.cause {
            Cause::Reference(error) => Some(error),
            _ => None,
        }
    }
    /// The schema diagnostic, if compilation failed on schema semantics.
    pub fn schema_error(&self) -> Option<&ValidationError<'static>> {
        match &self.cause {
            Cause::Schema(error) => Some(error),
            _ => None,
        }
    }
}
impl fmt::Display for CompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Allocation(error) => error.fmt(f),
            Cause::Schema(error) => error.fmt(f),
            Cause::JsonSource(error) => error.fmt(f),
            Cause::Reference(error) => error.fmt(f),
            Cause::Entropy(error) => error.fmt(f),
        }
    }
}
impl Error for CompilationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.cause {
            Cause::Allocation(CompilationAllocationError::Refused) => {
                self.funding.0.as_ref().and_then(|source| {
                    source
                        .source_error()
                        .map(|error| error as &(dyn Error + 'static))
                })
            }
            Cause::Allocation(error) => Some(error),
            Cause::Schema(error) => Some(error),
            Cause::JsonSource(error) => Some(error),
            Cause::Reference(error) => Some(error),
            Cause::Entropy(error) => Some(error),
        }
    }
}

/// Internal propagation keeps borrowed schema diagnostics borrowed until the
/// public boundary; storage failures already retain their original account.
#[derive(Debug)]
pub(crate) enum CompileError<'a> {
    Schema(ValidationError<'a>),
    Storage(CompilationError),
    Reference(referencing::Error),
}
impl<'a> From<ValidationError<'a>> for CompileError<'a> {
    fn from(error: ValidationError<'a>) -> Self {
        Self::Schema(error)
    }
}
impl From<CompilationError> for CompileError<'_> {
    fn from(error: CompilationError) -> Self {
        Self::Storage(error)
    }
}
impl From<referencing::Error> for CompileError<'_> {
    fn from(error: referencing::Error) -> Self {
        Self::Reference(error)
    }
}
impl fmt::Display for CompileError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(error) => error.fmt(f),
            Self::Storage(error) => error.fmt(f),
            Self::Reference(error) => error.fmt(f),
        }
    }
}
impl Error for CompileError<'_> {}
impl<'a> CompileError<'a> {
    pub(crate) fn to_owned(self) -> CompileError<'static> {
        self.to_owned_with_funding(&Funding::default())
    }
    pub(crate) fn to_owned_with_funding(self, funding: &Funding) -> CompileError<'static> {
        match self {
            Self::Schema(error) => match error.to_owned_with_funding(funding) {
                Ok(error) => CompileError::Schema(error),
                Err(error) => CompileError::Storage(error),
            },
            Self::Storage(error) => CompileError::Storage(error),
            Self::Reference(error) => funding.reference_error(error),
        }
    }
    /// Reference-only ordinary entry points keep their existing public error.
    pub(crate) fn into_ordinary_reference(self) -> referencing::Error {
        match self {
            Self::Reference(error) => error,
            Self::Storage(error) => panic!("{error}"),
            Self::Schema(_) => {
                unreachable!("reference-only worker cannot return schema validation errors")
            }
        }
    }
    pub(crate) fn into_ordinary(self) -> ValidationError<'a> {
        match self {
            Self::Schema(error) => error,
            Self::Reference(error) => error.into(),
            // The ordinary API historically aborts on storage failure. It
            // never converts a refusal into a schema diagnostic or retries.
            Self::Storage(error) => panic!("{error}"),
        }
    }
}

impl CompileError<'static> {
    pub(crate) fn into_compilation(self, funding: &Funding) -> CompilationError {
        match self {
            Self::Schema(error) => funding.schema_error(error),
            Self::Storage(error) => error,
            Self::Reference(error) => match funding.reference_error(error) {
                Self::Storage(error) => error,
                Self::Reference(error) => CompilationError {
                    cause: Cause::Reference(error),
                    funding: funding.clone(),
                },
                Self::Schema(_) => unreachable!("reference conversion preserves its error kind"),
            },
        }
    }
}

pub(crate) enum PatternError {
    Syntax,
    Storage(CompilationError),
}
impl From<CompilationError> for PatternError {
    fn from(error: CompilationError) -> Self {
        Self::Storage(error)
    }
}
impl PatternError {
    pub(crate) fn diagnostic<'a>(
        self,
        funding: &Funding,
        location: &crate::paths::Location,
        instance: &'a serde_json::Value,
    ) -> CompileError<'a> {
        match self {
            Self::Storage(error) => CompileError::Storage(error),
            Self::Syntax => {
                let error = (|| {
                    crate::ValidationError::format_with_funding(
                        location.clone(),
                        crate::paths::LazyEvaluationPath::SameAsSchemaPath,
                        crate::paths::Location::new_with_funding(funding)?,
                        std::borrow::Cow::Borrowed(instance),
                        funding.copy_str("regex")?,
                        funding,
                    )
                })();
                match error {
                    Ok(error) => CompileError::Schema(error),
                    Err(error) => CompileError::Storage(error),
                }
            }
        }
    }
}

/// Explicit policy carried through the existing compiler. `None` is the
/// ordinary unenforced entry; it does not manufacture an account or a cache.
#[derive(Clone, Default)]
pub(crate) struct Funding(Option<Arc<dyn CompilationFunding>>);
impl fmt::Debug for Funding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompilationFunding")
            .field("enforced", &self.0.is_some())
            .finish()
    }
}
impl Funding {
    pub(crate) fn is_enforced(&self) -> bool {
        self.0.is_some()
    }
    pub(crate) fn reference_allocation(
        &self,
        error: referencing::allocation::AllocationError,
    ) -> CompilationError {
        self.error(match error {
            referencing::allocation::AllocationError::Refused => {
                CompilationAllocationError::Refused
            }
            referencing::allocation::AllocationError::SizeOverflow => {
                CompilationAllocationError::Overflow
            }
            referencing::allocation::AllocationError::HostAllocation => {
                CompilationAllocationError::HostAllocation
            }
        })
    }
    pub(crate) fn reference_error(&self, error: referencing::Error) -> CompileError<'static> {
        match error {
            referencing::Error::Allocation(error) => self
                .error(match error {
                    referencing::allocation::AllocationError::Refused => {
                        CompilationAllocationError::Refused
                    }
                    referencing::allocation::AllocationError::SizeOverflow => {
                        CompilationAllocationError::Overflow
                    }
                    referencing::allocation::AllocationError::HostAllocation => {
                        CompilationAllocationError::HostAllocation
                    }
                })
                .into(),
            referencing::Error::Unqualified(component) => self
                .error(CompilationAllocationError::Unqualified(component))
                .into(),
            error => CompileError::Reference(error),
        }
    }
    pub(crate) fn json_source(&self, bytes: &[u8]) -> Result<serde_json::Value, CompilationError> {
        serde_json::bounded_events::from_slice_with_allocations(bytes, self).map_err(|error| {
            match error {
                serde_json::bounded_events::ValueError::Allocation(error) => self.json_error(error),
                error => CompilationError {
                    cause: Cause::JsonSource(error),
                    funding: self.clone(),
                },
            }
        })
    }
    pub(crate) fn value(
        &self,
        value: &serde_json::Value,
    ) -> Result<serde_json::Value, CompilationError> {
        value
            .try_clone_with_allocations(self)
            .map_err(|error| self.json_error(error))
    }
    pub(crate) fn json_insert(
        &self,
        map: &mut serde_json::Map<String, serde_json::Value>,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<(), CompilationError> {
        let key = self.copy_str(key)?;
        let value = self.value(value)?;
        map.try_insert_with_allocations(key, value, self)
            .map_err(|error| self.json_error(error))?;
        Ok(())
    }
    /// Lend this source to the existing validation scratch workers. All scratch
    /// retires before the original compiler failure is returned.
    pub(crate) fn with_validation_context<F: crate::validator::original::OriginalJson, T>(
        &self,
        run: impl FnOnce(&mut crate::validator::ValidationContext<'_>) -> T,
    ) -> Result<T, CompilationError> {
        use crate::validator::{workspace, ValidationContext};
        struct Loan<'a> {
            funding: &'a Funding,
            failure: Option<CompilationError>,
        }
        impl workspace::Funding for Loan<'_> {
            fn reserve(&mut self, bytes: usize) -> bool {
                if self.failure.is_some() {
                    return false;
                }
                match self.funding.reserve(bytes) {
                    Ok(()) => true,
                    Err(error) => {
                        self.failure = Some(error);
                        false
                    }
                }
            }
        }
        if !self.is_enforced() {
            let mut context = ValidationContext::default();
            context.workspace.use_invocation_regex_cache();
            return Ok(run(&mut context));
        }
        let controls = crate::validator::original::OriginalValidationSource::<F>::control_bytes()
            .and_then(|bytes| bytes.checked_add(size_of::<Loan<'_>>()));
        self.reserve(controls.ok_or_else(|| self.error(CompilationAllocationError::Overflow))?)?;
        let seed = self.random_state()?;
        let mut loan = Loan {
            funding: self,
            failure: None,
        };
        let mut context = ValidationContext::with_workspace(&mut loan, &seed);
        context.workspace.set_input_controls(
            F::original_input_controls()
                .ok_or_else(|| self.error(CompilationAllocationError::Overflow))?,
        );
        context
            .workspace
            .set_string_controls(F::original_string_controls());
        context.set_diagnostic_funding(self.clone());
        let result = run(&mut context);
        let diagnostic_failure = context.take_diagnostic_failure();
        let failure = context.workspace_failure();
        if let Some(error) = loan.failure {
            return Err(error);
        }
        if let Some(error) = diagnostic_failure {
            return Err(error);
        }
        if let Some(error) = failure {
            return Err(self.workspace_error(error));
        }
        Ok(result)
    }
    pub(crate) fn unique(&self, items: &[serde_json::Value]) -> Result<bool, CompilationError> {
        self.with_validation_context::<crate::SerdeJson, _>(|context| {
            context.unique_items::<crate::SerdeJson>(&items)
        })
    }

    pub(crate) fn copy_vec<T, U>(
        &self,
        values: &[T],
        mut copy: impl FnMut(&T) -> Result<U, CompilationError>,
    ) -> Result<Vec<U>, CompilationError> {
        let mut output = Vec::new();
        self.grow(&mut output, values.len())?;
        for value in values {
            output.push(copy(value)?);
        }
        Ok(output)
    }
    pub(crate) fn format(&self, arguments: fmt::Arguments<'_>) -> Result<String, CompilationError> {
        struct Counter(usize);
        impl fmt::Write for Counter {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                self.0 = self.0.checked_add(text.len()).ok_or(fmt::Error)?;
                Ok(())
            }
        }
        let mut counter = Counter(0);
        fmt::write(&mut counter, arguments)
            .map_err(|_| self.error(CompilationAllocationError::Overflow))?;
        let mut output = self.string(counter.0)?;
        struct Fixed<'a>(&'a mut String);
        impl fmt::Write for Fixed<'_> {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                if text.len() > self.0.capacity() - self.0.len() {
                    return Err(fmt::Error);
                }
                self.0.push_str(text);
                Ok(())
            }
        }
        fmt::write(&mut Fixed(&mut output), arguments)
            .map_err(|_| self.error(CompilationAllocationError::Capacity))?;
        Ok(output)
    }
    pub(crate) fn syntax_allocation(
        &self,
        error: jsonschema_regex::allocation::AllocationError,
    ) -> CompilationError {
        use jsonschema_regex::allocation::AllocationError;
        self.error(match error {
            AllocationError::Refused => CompilationAllocationError::Refused,
            AllocationError::SizeOverflow => CompilationAllocationError::Overflow,
            AllocationError::HostAllocation => CompilationAllocationError::HostAllocation,
        })
    }
    pub(crate) fn arc_slice<T>(&self, values: Vec<T>) -> Result<Arc<[T]>, CompilationError> {
        let layout = Layout::array::<T>(values.len())
            .map_err(|_| self.error(CompilationAllocationError::Overflow))?;
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(layout)
            .map_err(|_| self.error(CompilationAllocationError::Overflow))?;
        self.reserve(layout.pad_to_align().size())?;
        Ok(Arc::from(values))
    }
    pub(crate) fn annotation(
        &self,
        value: &serde_json::Value,
    ) -> Result<Arc<serde_json::Value>, CompilationError> {
        self.arc(self.value(value)?)
    }
    pub(crate) fn number(
        &self,
        value: &serde_json::Number,
    ) -> Result<serde_json::Number, CompilationError> {
        value
            .try_clone_with_allocations(self)
            .map_err(|error| self.json_error(error))
    }
    pub(crate) fn literal_value(
        &self,
        literal: &jsonschema_value::literal::Literal,
    ) -> Result<serde_json::Value, CompilationError> {
        literal
            .to_value_with_allocations(self)
            .map_err(|error| self.json_error(error))
    }
    pub(crate) fn literal(
        &self,
        value: &serde_json::Value,
    ) -> Result<jsonschema_value::literal::Literal, CompilationError> {
        jsonschema_value::literal::Literal::from_value_with_allocations(value, self)
            .map_err(|error| self.json_error(error))
    }
    pub(crate) fn literal_object(
        &self,
        value: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<jsonschema_value::literal::Literal, CompilationError> {
        jsonschema_value::literal::Literal::from_object_with_allocations(value, self)
            .map_err(|error| self.json_error(error))
    }
    pub(crate) fn literals(
        &self,
        values: &[serde_json::Value],
    ) -> Result<Vec<jsonschema_value::literal::Literal>, CompilationError> {
        let mut output = Vec::new();
        self.grow(&mut output, values.len())?;
        for value in values {
            output.push(self.literal(value)?);
        }
        Ok(output)
    }
    pub(crate) fn boxed_str(&self, value: &str) -> Result<Box<str>, CompilationError> {
        let copied = self.copy_str(value)?;
        debug_assert_eq!(copied.len(), copied.capacity());
        Ok(copied.into_boxed_str())
    }
    pub(crate) fn insert_set<K: Eq + std::hash::Hash, S: std::hash::BuildHasher>(
        &self,
        set: &mut hashbrown::HashSet<K, S>,
        value: K,
    ) -> Result<bool, CompilationError> {
        if !set.contains(&value) {
            if let Some(layout) = set
                .try_reserve_layout(1)
                .map_err(|error| self.error(CompilationAllocationError::Table(error)))?
            {
                self.reserve(layout.size())?;
                set.try_reserve(1)
                    .map_err(|error| self.error(CompilationAllocationError::Table(error)))?;
            }
        }
        Ok(set.insert(value))
    }
    pub(crate) fn enforced(source: Arc<dyn CompilationFunding>) -> Self {
        Self(Some(source))
    }
    pub(crate) fn error(&self, cause: CompilationAllocationError) -> CompilationError {
        CompilationError {
            cause: Cause::Allocation(cause),
            funding: self.clone(),
        }
    }
    pub(crate) fn schema_error(&self, cause: ValidationError<'static>) -> CompilationError {
        CompilationError {
            cause: Cause::Schema(cause),
            funding: self.clone(),
        }
    }
    pub(crate) fn workspace_error(
        &self,
        error: crate::validator::workspace::Error,
    ) -> CompilationError {
        use crate::validator::workspace;
        let cause = match error {
            workspace::Error::Funding => CompilationAllocationError::Refused,
            workspace::Error::Overflow => CompilationAllocationError::Overflow,
            workspace::Error::Allocation(error) => CompilationAllocationError::Allocation(error),
            workspace::Error::Table(error) => CompilationAllocationError::Table(error),
            workspace::Error::Capacity => CompilationAllocationError::Capacity,
            workspace::Error::HostAllocation => CompilationAllocationError::HostAllocation,
            workspace::Error::Unqualified(component) => {
                CompilationAllocationError::Unqualified(match component {
                    workspace::Component::Source(name) | workspace::Component::Validator(name) => {
                        name
                    }
                    workspace::Component::RegexSyntax => "regex syntax validation",
                })
            }
        };
        self.error(cause)
    }
    pub(crate) fn random_state(&self) -> Result<ahash::RandomState, CompilationError> {
        self.reserve(size_of::<(
            [u8; 32],
            [u64; 4],
            ahash::RandomState,
            Result<(), getrandom::Error>,
            &Self,
        )>())?;
        let mut bytes = [0; 32];
        getrandom::fill(&mut bytes).map_err(|error| CompilationError {
            cause: Cause::Entropy(error),
            funding: self.clone(),
        })?;
        let keys: [u64; 4] = std::array::from_fn(|index| {
            u64::from_le_bytes(
                bytes[index * 8..index * 8 + 8]
                    .try_into()
                    .expect("fixed seed word"),
            )
        });
        Ok(ahash::RandomState::with_seeds(
            keys[0], keys[1], keys[2], keys[3],
        ))
    }
    pub(crate) fn reserve(&self, bytes: usize) -> Result<(), CompilationError> {
        self.0
            .as_ref()
            .map_or(Ok(()), |source| source.reserve(bytes))
            .map_err(|error| self.error(error))
    }
    pub(crate) fn json_error(
        &self,
        error: serde_json::allocation::AllocationError,
    ) -> CompilationError {
        use serde_json::allocation::AllocationError;
        self.error(match error {
            AllocationError::Refused => CompilationAllocationError::Refused,
            AllocationError::SizeOverflow => CompilationAllocationError::Overflow,
            AllocationError::HostAllocation => CompilationAllocationError::HostAllocation,
        })
    }
    pub(crate) fn error_context(
        &self,
        branches: Vec<Vec<ValidationError<'_>>>,
    ) -> Result<Vec<Vec<ValidationError<'static>>>, CompilationError> {
        let mut owned = Vec::new();
        self.grow(&mut owned, branches.len())?;
        for branch in branches {
            owned.push(self.owned_errors(branch)?);
        }
        Ok(owned)
    }
    pub(crate) fn owned_errors(
        &self,
        errors: Vec<ValidationError<'_>>,
    ) -> Result<Vec<ValidationError<'static>>, CompilationError> {
        let mut owned = Vec::new();
        self.grow(&mut owned, errors.len())?;
        for error in errors {
            owned.push(error.to_owned_with_funding(self)?);
        }
        Ok(owned)
    }
    pub(crate) fn input_value<'a, F: crate::Json>(
        &self,
        input: &F::Node<'a>,
    ) -> Result<std::borrow::Cow<'a, serde_json::Value>, CompilationError> {
        F::prepare_value_with_allocations(input, self).map_err(|error| match error {
            jsonschema_value::KeyPreparationError::Allocation(error) => self.json_error(error),
            jsonschema_value::KeyPreparationError::Unqualified(source) => {
                self.error(CompilationAllocationError::Unqualified(source))
            }
        })
    }
    pub(crate) fn key<F: crate::Json>(
        &self,
        key: &str,
    ) -> Result<F::PreparedKey, CompilationError> {
        F::prepare_key_with_allocations(key, self).map_err(|error| match error {
            jsonschema_value::KeyPreparationError::Allocation(error) => self.json_error(error),
            jsonschema_value::KeyPreparationError::Unqualified(source) => {
                self.error(CompilationAllocationError::Unqualified(source))
            }
        })
    }
    pub(crate) fn grow<T>(
        &self,
        values: &mut Vec<T>,
        required: usize,
    ) -> Result<(), CompilationError> {
        if required <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(required))
            .ok_or_else(|| self.error(CompilationAllocationError::Overflow))?;
        let layout = Layout::array::<T>(capacity)
            .map_err(|_| self.error(CompilationAllocationError::Overflow))?;
        self.reserve(layout.size())?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|error| self.error(CompilationAllocationError::Allocation(error)))?;
        if values.capacity() != capacity {
            return Err(self.error(CompilationAllocationError::Capacity));
        }
        Ok(())
    }
    pub(crate) fn push<T>(&self, values: &mut Vec<T>, value: T) -> Result<(), CompilationError> {
        let required = values
            .len()
            .checked_add(1)
            .ok_or_else(|| self.error(CompilationAllocationError::Overflow))?;
        self.grow(values, required)?;
        values.push(value);
        Ok(())
    }
    pub(crate) fn copy_str(&self, value: &str) -> Result<String, CompilationError> {
        let mut bytes = Vec::new();
        self.grow(&mut bytes, value.len())?;
        bytes.extend_from_slice(value.as_bytes());
        Ok(String::from_utf8(bytes).expect("copied UTF-8 source"))
    }
    pub(crate) fn string(&self, capacity: usize) -> Result<String, CompilationError> {
        let mut bytes = Vec::new();
        self.grow(&mut bytes, capacity)?;
        Ok(String::from_utf8(bytes).expect("empty UTF-8 destination"))
    }
    pub(crate) fn boxed<T>(&self, value: T) -> Result<Box<T>, CompilationError> {
        self.reserve(size_of::<T>())?;
        Ok(Box::new(value))
    }
    fn shared_layout<T>(&self) -> Result<Layout, CompilationError> {
        Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<T>())
            .map(|(layout, _)| layout.pad_to_align())
            .map_err(|_| self.error(CompilationAllocationError::Overflow))
    }
    pub(crate) fn arc<T>(&self, value: T) -> Result<Arc<T>, CompilationError> {
        self.reserve(self.shared_layout::<T>()?.size())?;
        Ok(Arc::new(value))
    }
    pub(crate) fn rc<T>(&self, value: T) -> Result<Rc<T>, CompilationError> {
        self.reserve(self.shared_layout::<T>()?.size())?;
        Ok(Rc::new(value))
    }
    pub(crate) fn arc_str(&self, value: &str) -> Result<Arc<str>, CompilationError> {
        let payload = Layout::array::<u8>(value.len())
            .map_err(|_| self.error(CompilationAllocationError::Overflow))?;
        let bytes = Layout::new::<[AtomicUsize; 2]>()
            .extend(payload)
            .map_err(|_| self.error(CompilationAllocationError::Overflow))?
            .0
            .pad_to_align()
            .size();
        self.reserve(bytes)?;
        Ok(Arc::from(value))
    }
    pub(crate) fn insert<K, V, S>(
        &self,
        map: &mut hashbrown::HashMap<K, V, S>,
        key: K,
        value: V,
    ) -> Result<Option<V>, CompilationError>
    where
        K: Eq + std::hash::Hash,
        S: std::hash::BuildHasher,
    {
        if !map.contains_key(&key) {
            self.reserve_map(map, 1)?;
        }
        Ok(map.insert(key, value))
    }
    pub(crate) fn reserve_map<K, V, S>(
        &self,
        map: &mut hashbrown::HashMap<K, V, S>,
        additional: usize,
    ) -> Result<(), CompilationError>
    where
        K: Eq + std::hash::Hash,
        S: std::hash::BuildHasher,
    {
        let layout = map
            .try_reserve_layout(additional)
            .map_err(|error| self.error(CompilationAllocationError::Table(error)))?;
        if let Some(layout) = layout {
            self.reserve(layout.size())?;
        }
        map.try_reserve(additional)
            .map_err(|error| self.error(CompilationAllocationError::Table(error)))?;
        if layout.is_some_and(|layout| layout.size() != map.allocation_size()) {
            return Err(self.error(CompilationAllocationError::Capacity));
        }
        Ok(())
    }
}

impl serde_json::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), serde_json::allocation::AllocationError> {
        self.0
            .as_ref()
            .map_or(Ok(()), |source| source.reserve(bytes))
            .map_err(|error| match error {
                CompilationAllocationError::Refused => {
                    serde_json::allocation::AllocationError::Refused
                }
                CompilationAllocationError::Overflow => {
                    serde_json::allocation::AllocationError::SizeOverflow
                }
                _ => serde_json::allocation::AllocationError::HostAllocation,
            })
    }
    fn is_enforced(&self) -> bool {
        self.0.is_some()
    }
}

impl referencing::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), referencing::allocation::AllocationError> {
        self.0
            .as_ref()
            .map_or(Ok(()), |source| source.reserve(bytes))
            .map_err(|error| match error {
                CompilationAllocationError::Refused => {
                    referencing::allocation::AllocationError::Refused
                }
                CompilationAllocationError::Overflow => {
                    referencing::allocation::AllocationError::SizeOverflow
                }
                _ => referencing::allocation::AllocationError::HostAllocation,
            })
    }
    fn is_enforced(&self) -> bool {
        self.is_enforced()
    }
}

impl jsonschema_regex::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), jsonschema_regex::allocation::AllocationError> {
        use jsonschema_regex::allocation::AllocationError;
        self.0
            .as_ref()
            .map_or(Ok(()), |source| source.reserve(bytes))
            .map_err(|error| match error {
                CompilationAllocationError::Refused => AllocationError::Refused,
                CompilationAllocationError::Overflow => AllocationError::SizeOverflow,
                _ => AllocationError::HostAllocation,
            })
    }
}

impl regex::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), regex::allocation::AllocationError> {
        <Self as jsonschema_regex::allocation::Allocation>::reserve(self, bytes).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct Refusal;
    impl fmt::Display for Refusal {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("test source reservation refused")
        }
    }
    impl Error for Refusal {}
    struct Source {
        calls: AtomicUsize,
        refuse_at: usize,
        cause: Refusal,
    }
    impl Source {
        fn new(refuse_at: usize) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                refuse_at,
                cause: Refusal,
            })
        }
    }
    impl CompilationFunding for Source {
        fn reserve(&self, _: usize) -> Result<(), CompilationAllocationError> {
            let index = self.calls.fetch_add(1, Ordering::Relaxed);
            assert!(
                index <= self.refuse_at,
                "producer continued after first refusal"
            );
            if index == self.refuse_at {
                Err(CompilationAllocationError::Refused)
            } else {
                Ok(())
            }
        }
        fn source_error(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
            Some(&self.cause)
        }
    }

    #[test]
    fn reached_source_refusals_stop_without_retry_and_keep_the_original_cause() {
        let schemas = [
            serde_json::json!({"type":"object", "properties":{"a":{"enum":["one","two"]}}, "required":["a"], "additionalProperties":false}),
            serde_json::json!({"$id":"https://example.test/root", "$defs":{"a/b":{"type":"string"}}, "properties":{"a":{"$ref":"#/$defs/a~1b"}}, "dependentRequired":{"a":["b","c"]}, "unevaluatedProperties":false}),
            serde_json::json!({"$id":"https://example.test/cycle", "type":"object", "$ref":"#", "properties":{"name":{"type":"string"}}, "unevaluatedProperties":false}),
        ];
        let options = crate::ValidationOptions::default().without_schema_validation();
        for schema in &schemas {
            let source = Source::new(usize::MAX);
            let funding = Funding::enforced(source.clone());
            let validator = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                &options, schema, &funding,
            )
            .expect("qualified retained producers");
            let calls = source.calls.load(Ordering::Relaxed);
            assert!(calls > 0);
            let ordinary =
                crate::compiler::build_validator::<crate::SerdeJson>(&options, schema).unwrap();
            for input in [
                serde_json::json!({}),
                serde_json::json!({"a":"one","b":true,"c":null}),
                serde_json::json!({"name":"child"}),
            ] {
                assert_eq!(validator.is_valid(&input), ordinary.is_valid(&input));
            }
            let weak = Arc::downgrade(&source);
            drop(source);
            drop(funding);
            assert!(
                weak.upgrade().is_some(),
                "graph keeps its construction source"
            );
            drop(validator);
            assert!(weak.upgrade().is_none(), "source retires after graph owner");
            for refuse_at in 0..calls {
                let source = Source::new(refuse_at);
                let weak = Arc::downgrade(&source);
                let funding = Funding::enforced(source.clone());
                let error = match crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                    &options, schema, &funding,
                ) {
                    Err(CompileError::Storage(error)) => error,
                    other => {
                        panic!("expected original allocation refusal at {refuse_at}, got {other:?}")
                    }
                };
                assert!(matches!(
                    error.allocation_error(),
                    Some(CompilationAllocationError::Refused)
                ));
                assert!(error.source().unwrap().downcast_ref::<Refusal>().is_some());
                assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
                drop(source);
                drop(funding);
                assert!(weak.upgrade().is_some(), "failure keeps original source");
                drop(error);
                assert!(weak.upgrade().is_none());
            }
        }
    }

    #[test]
    fn diagnostic_dispatch_preserves_shape_and_every_reached_refusal() {
        use crate::validator::Validate;
        let options = crate::ValidationOptions::default()
            .without_schema_validation()
            .should_validate_formats(true);
        for (schema, input) in [
            (
                serde_json::json!({"pattern":"(?=a+)(a+)\\1"}),
                serde_json::json!("b"),
            ),
            (
                serde_json::json!({"pattern":"^word$"}),
                serde_json::json!("b"),
            ),
            (
                serde_json::json!({"patternProperties":{"^x-[a-z]+$":{"type":"string"}}, "additionalProperties":false}),
                serde_json::json!({"other":1}),
            ),
            (
                serde_json::json!({"properties":{"x-name":{"type":"string"}},"patternProperties":{"^x-[a-z]+$":{"minLength":2}}, "additionalProperties":false}),
                serde_json::json!({"other":1}),
            ),
            (
                serde_json::json!({"patternProperties":{"^x-[a-z]+$":{"type":"string"}}, "unevaluatedProperties":false}),
                serde_json::json!({"other":1}),
            ),
            (serde_json::json!({"type":"string"}), serde_json::json!(42)),
            (serde_json::json!({"multipleOf":3}), serde_json::json!(7)),
            (
                serde_json::json!({"multipleOf":0.1}),
                serde_json::json!(0.15),
            ),
            #[cfg(feature = "idna")]
            (
                serde_json::json!({"format":"idn-email"}),
                serde_json::json!("用户@a\u{200d}b.example"),
            ),
            #[cfg(feature = "idna")]
            (
                serde_json::json!({"format":"idn-hostname"}),
                serde_json::json!("a\u{200d}b.example"),
            ),
            (
                serde_json::json!({"$schema":"http://json-schema.org/draft-07/schema#", "contentEncoding":"base64"}),
                serde_json::json!("invalid!"),
            ),
            (
                serde_json::json!({"$schema":"http://json-schema.org/draft-07/schema#", "contentMediaType":"application/json"}),
                serde_json::json!("{invalid}"),
            ),
            (
                serde_json::json!({"$schema":"http://json-schema.org/draft-07/schema#", "contentEncoding":"base64", "contentMediaType":"application/json"}),
                serde_json::json!("invalid!"),
            ),
            (
                serde_json::json!({"$schema":"http://json-schema.org/draft-07/schema#", "contentEncoding":"base64", "contentMediaType":"application/json"}),
                serde_json::json!("Zm9vYmFy"),
            ),
            (
                serde_json::json!({"$schema":"http://json-schema.org/draft-07/schema#", "contentEncoding":"base64", "contentMediaType":"application/json"}),
                serde_json::json!("//4="),
            ),
            (
                serde_json::json!({"$defs":{"t":{"type":"string"}}, "$ref":"#/$defs/t"}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"anyOf":[{"type":"string"},{"type":"array"}]}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"oneOf":[{"type":"number"},{"minimum":0}]}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"oneOf":[{"type":"string"}]}),
                serde_json::json!(42),
            ),
            (serde_json::json!(false), serde_json::json!(42)),
            (
                serde_json::json!({"not":{"type":"number"}}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"propertyNames":{"maxLength":2}}),
                serde_json::json!({"long":1}),
            ),
            (
                serde_json::json!({"propertyNames":false}),
                serde_json::json!({"long":1}),
            ),
            (
                serde_json::json!({"additionalProperties":false}),
                serde_json::json!({"long":1}),
            ),
            (
                serde_json::json!({"properties":{"name":{"type":"string"}},"additionalProperties":false}),
                serde_json::json!({"long":1}),
            ),
            (
                serde_json::json!({"properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}),
                serde_json::json!({}),
            ),
            (
                serde_json::json!({"if":{"type":"number"},"then":{"minimum":50}}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"dependentRequired":{"name":["other"]}}),
                serde_json::json!({"name":1}),
            ),
            (
                serde_json::json!({"items":{"type":"string"}}),
                serde_json::json!([42]),
            ),
            (
                serde_json::json!({"contains":{"type":"string"}}),
                serde_json::json!([42]),
            ),
            (
                serde_json::json!({"const":["a",{"b":2}]}),
                serde_json::json!([42]),
            ),
            (
                serde_json::json!({"const":{"b":2}}),
                serde_json::json!([42]),
            ),
            (
                serde_json::json!({"enum":["a",{"b":2}]}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"format":"email"}),
                serde_json::json!("not an email"),
            ),
            (
                serde_json::json!({"format":"hostname"}),
                serde_json::json!("xn--invalid-.example"),
            ),
            (
                serde_json::json!({"format":"regex"}),
                serde_json::json!("[unterminated"),
            ),
            (
                serde_json::json!({"unevaluatedProperties":false}),
                serde_json::json!({"extra":1}),
            ),
            (
                serde_json::json!({"unevaluatedItems":false}),
                serde_json::json!([1]),
            ),
        ] {
            let validator =
                crate::compiler::build_validator::<crate::SerdeJson>(&options, &schema).unwrap();
            let expected = validator.validate(&input).unwrap_err();
            let source = Source::new(usize::MAX);
            let funding = Funding::enforced(source.clone());
            let actual = funding
                .with_validation_context::<crate::SerdeJson, _>(|context| {
                    validator.root.validate(
                        &&input,
                        &crate::paths::LazyLocation::new(),
                        None,
                        context,
                    )
                })
                .unwrap()
                .unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert_eq!(actual.schema_path(), expected.schema_path());
            assert_eq!(actual.instance_path(), expected.instance_path());
            assert_eq!(actual.evaluation_path(), expected.evaluation_path());
            let calls = source.calls.load(Ordering::Relaxed);
            let weak = Arc::downgrade(&source);
            drop(source);
            drop(funding);
            assert!(weak.upgrade().is_some());
            drop(actual);
            assert!(weak.upgrade().is_none());
            for refuse_at in 0..calls {
                let source = Source::new(refuse_at);
                let error = Funding::enforced(source.clone())
                    .with_validation_context::<crate::SerdeJson, _>(|context| {
                        validator.root.validate(
                            &&input,
                            &crate::paths::LazyLocation::new(),
                            None,
                            context,
                        )
                    })
                    .unwrap_err();
                assert!(matches!(
                    error.allocation_error(),
                    Some(CompilationAllocationError::Refused)
                ));
                assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
            }
        }
    }

    #[test]
    fn floating_multiple_uses_invocation_funds_for_original_rational_scratch() {
        use crate::validator::workspace::{Error as WorkspaceError, Funding as InvocationFunding};
        struct Invocation {
            calls: usize,
            refuse_at: usize,
        }
        impl InvocationFunding for Invocation {
            fn reserve(&mut self, _: usize) -> bool {
                let call = self.calls;
                assert!(
                    call <= self.refuse_at,
                    "numeric producer continued after refusal"
                );
                self.calls += 1;
                call != self.refuse_at
            }
        }
        let options = crate::ValidationOptions::default().without_schema_validation();
        for (value, divisor) in [
            (0.15, 0.1),
            (1e300, 0.7),
            (f64::MAX, 0.123456789),
            (1e-15, 1e-17),
        ] {
            let schema = serde_json::json!({"multipleOf":divisor});
            let input = serde_json::json!(value);
            let ordinary =
                crate::compiler::build_validator::<crate::SerdeJson>(&options, &schema).unwrap();
            let source = Source::new(usize::MAX);
            let graph = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                &options,
                &schema,
                &Funding::enforced(source.clone()),
            )
            .unwrap();
            let prepared = crate::OriginalValidationSource::new(graph).unwrap();
            assert!(prepared.retained_bytes().unwrap() > 0);
            let cold_calls = source.calls.load(Ordering::Relaxed);
            let mut invocation = Invocation {
                calls: 0,
                refuse_at: usize::MAX,
            };
            assert_eq!(
                prepared.is_valid(&input, &mut invocation).unwrap(),
                ordinary.is_valid(&input)
            );
            for refuse_at in 0..invocation.calls {
                let mut refused = Invocation {
                    calls: 0,
                    refuse_at,
                };
                assert!(matches!(
                    prepared.is_valid(&input, &mut refused),
                    Err(WorkspaceError::Funding)
                ));
                assert_eq!(refused.calls, refuse_at + 1);
            }
            assert_eq!(
                source.calls.load(Ordering::Relaxed),
                cold_calls,
                "numeric invocation cannot spend source funding"
            );
        }
    }

    #[test]
    fn content_sources_keep_invocation_funding_and_callback_qualification() {
        use crate::validator::workspace::{Error as WorkspaceError, Funding as InvocationFunding};
        struct Invocation {
            calls: usize,
            refuse_at: usize,
        }
        impl InvocationFunding for Invocation {
            fn reserve(&mut self, _: usize) -> bool {
                let call = self.calls;
                assert!(
                    call <= self.refuse_at,
                    "content producer continued after refusal"
                );
                self.calls += 1;
                call != self.refuse_at
            }
        }
        let options = crate::ValidationOptions::default()
            .with_draft(crate::Draft::Draft7)
            .without_schema_validation();
        for (schema, text) in [
            (serde_json::json!({"contentEncoding":"base64"}), "Zm9vYmFy"),
            (
                serde_json::json!({"contentEncoding":"base64url"}),
                "PDw_Pz4-",
            ),
            (
                serde_json::json!({"contentEncoding":"base32"}),
                "MZXW6YTBOI======",
            ),
            (
                serde_json::json!({"contentEncoding":"base32hex"}),
                "CPNMUOJ1E8======",
            ),
            (
                serde_json::json!({"contentEncoding":"base16"}),
                "666f6f626172",
            ),
            (
                serde_json::json!({"contentMediaType":"application/json"}),
                "{\"number\":3.5,\"escaped\":\"\\u0041\"}",
            ),
            (
                serde_json::json!({"contentEncoding":"base64","contentMediaType":"application/json"}),
                "e30=",
            ),
        ] {
            let input = serde_json::json!(text);
            let source = Source::new(usize::MAX);
            let graph = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                &options,
                &schema,
                &Funding::enforced(source.clone()),
            )
            .unwrap();
            let prepared = crate::OriginalValidationSource::new(graph).unwrap();
            assert!(prepared.retained_bytes().unwrap() > 0);
            let cold_calls = source.calls.load(Ordering::Relaxed);
            let mut invocation = Invocation {
                calls: 0,
                refuse_at: usize::MAX,
            };
            assert!(
                prepared.is_valid(&input, &mut invocation).unwrap(),
                "{schema}: {text}"
            );
            for refuse_at in 0..invocation.calls {
                let mut refused = Invocation {
                    calls: 0,
                    refuse_at,
                };
                assert!(matches!(
                    prepared.is_valid(&input, &mut refused),
                    Err(WorkspaceError::Funding)
                ));
                assert_eq!(refused.calls, refuse_at + 1);
            }
            assert_eq!(
                source.calls.load(Ordering::Relaxed),
                cold_calls,
                "invocation cannot spend cold source funds"
            );
        }
        static CUSTOM_CALLS: AtomicUsize = AtomicUsize::new(0);
        fn custom(_: &str) -> bool {
            CUSTOM_CALLS.fetch_add(1, Ordering::Relaxed);
            true
        }
        fn convert(_: &str) -> Result<Option<String>, crate::ValidationError<'static>> {
            CUSTOM_CALLS.fetch_add(1, Ordering::Relaxed);
            Ok(Some("{}".to_owned()))
        }
        for (schema, options) in [
            (
                serde_json::json!({"contentMediaType":"application/json"}),
                options
                    .clone()
                    .with_content_media_type("application/json", custom),
            ),
            (
                serde_json::json!({"contentEncoding":"base64"}),
                options
                    .clone()
                    .with_content_encoding("base64", custom, convert),
            ),
            (
                serde_json::json!({"contentMediaType":"application/json", "contentEncoding":"base64"}),
                options
                    .clone()
                    .with_content_encoding("base64", custom, convert),
            ),
        ] {
            let graph =
                crate::compiler::build_validator::<crate::SerdeJson>(&options, &schema).unwrap();
            assert!(graph.is_valid(&serde_json::json!("e30=")));
            let prior = CUSTOM_CALLS.load(Ordering::Relaxed);
            let prepared = crate::OriginalValidationSource::new(graph).unwrap();
            assert!(matches!(
                prepared.retained_bytes().unwrap_err().allocation_error(),
                Some(CompilationAllocationError::Unqualified(_))
            ));
            let mut invocation = Invocation {
                calls: 0,
                refuse_at: usize::MAX,
            };
            assert!(matches!(
                prepared.is_valid(&serde_json::json!("e30="), &mut invocation),
                Err(WorkspaceError::Unqualified(_))
            ));
            assert_eq!(
                CUSTOM_CALLS.load(Ordering::Relaxed),
                prior,
                "opaque callbacks cannot run under an unrelated bound"
            );
        }
    }

    #[test]
    fn source_initialization_and_census_fund_each_reached_producer() {
        let schema = serde_json::json!({"type":"object", "properties":{"name":{"type":"string","format":"email"},"other":{"enum":["one","two"]}}});
        let options = crate::ValidationOptions::default()
            .without_schema_validation()
            .should_validate_formats(true);
        let source = Source::new(usize::MAX);
        let funding = Funding::enforced(source.clone());
        let graph = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
            &options, &schema, &funding,
        )
        .unwrap();
        let first = source.calls.load(Ordering::Relaxed);
        let prepared = crate::OriginalValidationSource::new(graph).unwrap();
        let bytes = prepared.retained_bytes().unwrap();
        assert!(bytes > 0);
        let last = source.calls.load(Ordering::Relaxed);
        struct Invocation {
            calls: usize,
        }
        impl crate::validator::workspace::Funding for Invocation {
            fn reserve(&mut self, _: usize) -> bool {
                self.calls += 1;
                true
            }
        }
        let mut invocation = Invocation { calls: 0 };
        assert!(prepared
            .is_valid(
                &serde_json::json!({"name":"user@xn--bcher-kva.example"}),
                &mut invocation
            )
            .unwrap());
        assert!(invocation.calls > 0);
        assert_eq!(
            source.calls.load(Ordering::Relaxed),
            last,
            "invocation scratch cannot use the cold source account"
        );
        drop(prepared);
        for refuse_at in first..last {
            let source = Source::new(refuse_at);
            let weak = Arc::downgrade(&source);
            let funding = Funding::enforced(source.clone());
            let graph = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                &options, &schema, &funding,
            )
            .unwrap();
            let result = crate::OriginalValidationSource::new(graph)
                .and_then(|prepared| prepared.retained_bytes());
            let error = result.unwrap_err();
            assert!(matches!(
                error.allocation_error(),
                Some(CompilationAllocationError::Refused)
            ));
            assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
            drop(source);
            drop(funding);
            assert!(weak.upgrade().is_some());
            drop(error);
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn public_source_bridge_keeps_policy_and_error_custody_explicit() {
        let schema = serde_json::json!({"$schema":"http://json-schema.org/draft-07/schema#", "type":"string"});
        let ordinary =
            crate::Validator::<crate::SerdeJson>::build_with_funding(&schema, None).unwrap();
        let source = Source::new(usize::MAX);
        let weak = Arc::downgrade(&source);
        let paid =
            crate::Validator::<crate::SerdeJson>::build_with_funding(&schema, Some(source.clone()))
                .unwrap();
        for input in [serde_json::json!("valid"), serde_json::json!(42)] {
            assert_eq!(ordinary.is_valid(&input), paid.is_valid(&input));
        }
        drop(source);
        assert!(weak.upgrade().is_some());
        drop(paid);
        assert!(weak.upgrade().is_none());

        let source = Source::new(0);
        let weak = Arc::downgrade(&source);
        let error =
            crate::Validator::<crate::SerdeJson>::build_with_funding(&schema, Some(source.clone()))
                .unwrap_err();
        assert!(error.source().unwrap().downcast_ref::<Refusal>().is_some());
        assert_eq!(source.calls.load(Ordering::Relaxed), 1);
        drop(source);
        assert!(weak.upgrade().is_some());
        drop(error);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn cold_meta_validation_uses_the_actual_funded_graph() {
        for draft in [
            crate::Draft::Draft4,
            crate::Draft::Draft6,
            crate::Draft::Draft7,
            crate::Draft::Draft201909,
            crate::Draft::Draft202012,
        ] {
            let options = crate::ValidationOptions::default().with_draft(draft);
            let source = Source::new(usize::MAX);
            let funding = Funding::enforced(source.clone());
            let schema = serde_json::json!({"type":"object", "properties":{"name":{"type":"string", "minLength":2}}, "required":["name"]});
            let validator = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                &options, &schema, &funding,
            )
            .unwrap();
            assert!(validator.is_valid(&serde_json::json!({"name":"okay"})));
            let invalid = serde_json::json!({"type":17});
            let expected = crate::compiler::build_validator::<crate::SerdeJson>(&options, &invalid)
                .unwrap_err();
            let actual = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                &options, &invalid, &funding,
            )
            .unwrap_err();
            match (expected, actual) {
                (CompileError::Schema(expected), CompileError::Schema(actual)) => {
                    assert_eq!(expected.to_string(), actual.to_string());
                    assert_eq!(expected.instance_path(), actual.instance_path());
                    assert_eq!(expected.schema_path(), actual.schema_path());
                    assert_eq!(expected.evaluation_path(), actual.evaluation_path());
                }
                other => panic!("meta schema diagnostic mismatch: {other:?}"),
            }
        }
    }

    #[test]
    fn regex_source_census_and_scoped_invocations_preserve_first_refusal() {
        use crate::validator::workspace::{Error as WorkspaceError, Funding as InvocationFunding};
        struct Invocation {
            calls: usize,
            refuse_at: usize,
        }
        impl InvocationFunding for Invocation {
            fn reserve(&mut self, _: usize) -> bool {
                let call = self.calls;
                assert!(
                    call <= self.refuse_at,
                    "regex invocation continued after refusal"
                );
                self.calls += 1;
                call != self.refuse_at
            }
        }
        for standard in [false, true] {
            let options = crate::ValidationOptions::default().without_schema_validation();
            let options = if standard {
                options.with_pattern_options(crate::PatternOptions::regex())
            } else {
                options
            };
            for (schema, good, bad) in [
                (
                    serde_json::json!({"pattern":if standard { "a{2,4}" } else { "(?=a+)(a+)\\1" }}),
                    serde_json::json!("aaaa"),
                    serde_json::json!("b"),
                ),
                (
                    serde_json::json!({"pattern":"[α-ω]+[0-9]+"}),
                    serde_json::json!("αβ12"),
                    serde_json::json!("none"),
                ),
                (
                    serde_json::json!({"patternProperties":{"^x-[a-z]+$":{"type":"string"}},"additionalProperties":false}),
                    serde_json::json!({"x-name":"okay"}),
                    serde_json::json!({"other":0}),
                ),
                (
                    serde_json::json!({"properties":{"x-name":{"minLength":2}},"patternProperties":{"^x-[a-z]+$":{"type":"string"}},"additionalProperties":false}),
                    serde_json::json!({"x-name":"okay"}),
                    serde_json::json!({"other":0}),
                ),
                (
                    serde_json::json!({"properties":{"x-name":{"minLength":2}},"patternProperties":{"^x-[a-z]+$":{"type":"string"}},"additionalProperties":{"type":"boolean"}}),
                    serde_json::json!({"x-name":"okay","other":true}),
                    serde_json::json!({"other":0}),
                ),
                (
                    serde_json::json!({"patternProperties":{"^x-[a-z]+$":{"type":"string"}},"unevaluatedProperties":false}),
                    serde_json::json!({"x-name":"okay"}),
                    serde_json::json!({"other":0}),
                ),
            ] {
                let cold = Source::new(usize::MAX);
                let graph = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                    &options,
                    &schema,
                    &Funding::enforced(cold.clone()),
                )
                .unwrap();
                let prepared = crate::OriginalValidationSource::new(graph).unwrap();
                let bytes = prepared.retained_bytes().unwrap();
                assert!(bytes > 0);
                let calls = cold.calls.load(Ordering::Relaxed);
                for (input, expected) in [(&good, true), (&bad, false)] {
                    let mut invocation = Invocation {
                        calls: 0,
                        refuse_at: usize::MAX,
                    };
                    assert_eq!(
                        prepared.is_valid(input, &mut invocation).unwrap(),
                        expected,
                        "{schema:?}"
                    );
                    for refuse_at in 0..invocation.calls {
                        let mut refused = Invocation {
                            calls: 0,
                            refuse_at,
                        };
                        assert!(matches!(
                            prepared.is_valid(input, &mut refused),
                            Err(WorkspaceError::Funding)
                        ));
                        assert_eq!(refused.calls, refuse_at + 1);
                    }
                }
                assert_eq!(
                    cold.calls.load(Ordering::Relaxed),
                    calls,
                    "invocations cannot spend source funding"
                );
                assert_eq!(
                    prepared.retained_bytes().unwrap(),
                    bytes,
                    "scoped searches cannot mutate retained source"
                );
                for refuse_at in 0..calls {
                    let source = Source::new(refuse_at);
                    let result = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                        &options,
                        &schema,
                        &Funding::enforced(source.clone()),
                    )
                    .map_err(|error| match error {
                        CompileError::Storage(error) => error,
                        other => panic!("unexpected {other:?}"),
                    })
                    .and_then(crate::OriginalValidationSource::new)
                    .and_then(|prepared| prepared.retained_bytes());
                    let error = match result {
                        Err(error) => error,
                        Ok(bytes) => {
                            // Each fresh construction receives fresh hash entropy.
                            // An unreached reservation ordinal cannot refuse; all
                            // actually reached ordinals must still preserve refusal.
                            assert!(source.calls.load(Ordering::Relaxed) <= refuse_at);
                            assert!(bytes > 0);
                            continue;
                        }
                    };
                    assert!(
                        matches!(
                            error.allocation_error(),
                            Some(CompilationAllocationError::Refused)
                        ),
                        "schema={schema:?} refusal={refuse_at}/{calls} error={error:?}"
                    );
                    assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
                }
            }
        }
    }

    #[test]
    fn deferred_diagnostic_aliases_share_one_prepaid_path_and_source() {
        use crate::paths::{capture_evaluation_path_with_funding, Location, RefTracker};
        let suffix = Location::from_escaped("/properties/item/$ref");
        let base = Location::from_escaped("/$defs/entry");
        let location = Location::from_escaped("/$defs/entry/type");
        let tracker = RefTracker::new(&suffix, &base, None);
        let source = Source::new(usize::MAX);
        let weak = Arc::downgrade(&source);
        let funding = Funding::enforced(source.clone());
        let path =
            capture_evaluation_path_with_funding(Some(&tracker), &location, &funding).unwrap();
        let alias = path.clone();
        let calls = source.calls.load(Ordering::Relaxed);
        assert_eq!(
            path.resolve(&location).as_str(),
            "/properties/item/$ref/type"
        );
        assert!(std::ptr::eq(
            path.resolve(&location),
            alias.resolve(&location)
        ));
        assert_eq!(
            source.calls.load(Ordering::Relaxed),
            calls,
            "one construction was already reserved"
        );
        drop(source);
        drop(funding);
        drop(path);
        assert!(weak.upgrade().is_some());
        drop(alias);
        assert!(weak.upgrade().is_none());
        for refuse_at in 0..calls {
            let source = Source::new(refuse_at);
            let error = capture_evaluation_path_with_funding(
                Some(&tracker),
                &location,
                &Funding::enforced(source.clone()),
            )
            .unwrap_err();
            assert!(matches!(
                error.allocation_error(),
                Some(CompilationAllocationError::Refused)
            ));
            assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
        }
    }

    #[test]
    fn ecma_validation_cache_uses_the_same_parser_with_scoped_storage() {
        for pattern in [
            r"(?<word>\p{L}+)\k<word>",
            r"[unterminated",
            r"^(?:one|two)[0-9]{2,4}$",
        ] {
            let expected = jsonschema_regex::is_valid_ecma_regex(pattern);
            let source = Source::new(usize::MAX);
            let funding = Funding::enforced(source.clone());
            let actual = funding
                .with_validation_context::<crate::SerdeJson, _>(|context| {
                    let first = context.is_valid_ecma_regex(pattern);
                    let calls = source.calls.load(Ordering::Relaxed);
                    assert_eq!(context.is_valid_ecma_regex(pattern), first);
                    assert_eq!(
                        source.calls.load(Ordering::Relaxed),
                        calls,
                        "cached answer reuses admitted source"
                    );
                    first
                })
                .unwrap();
            assert_eq!(actual, expected);
            let calls = source.calls.load(Ordering::Relaxed);
            for refuse_at in 0..calls {
                let source = Source::new(refuse_at);
                let error = Funding::enforced(source.clone())
                    .with_validation_context::<crate::SerdeJson, _>(|context| {
                        context.is_valid_ecma_regex(pattern)
                    })
                    .unwrap_err();
                assert!(matches!(
                    error.allocation_error(),
                    Some(CompilationAllocationError::Refused)
                ));
                assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
            }
        }
    }

    #[test]
    fn cold_uniqueness_scratch_and_diagnostic_keep_refusal_precedence() {
        let values = vec![serde_json::Value::from(1); 20];
        let source = Source::new(usize::MAX);
        let funding = Funding::enforced(source.clone());
        assert!(!funding.unique(&values).unwrap());
        let calls = source.calls.load(Ordering::Relaxed);
        for refuse_at in 0..calls {
            let source = Source::new(refuse_at);
            let error = Funding::enforced(source.clone())
                .unique(&values)
                .unwrap_err();
            assert!(matches!(
                error.allocation_error(),
                Some(CompilationAllocationError::Refused)
            ));
            assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
        }
        let options = crate::ValidationOptions::default().without_schema_validation();
        let schema = serde_json::json!({"dependentRequired":{"a":["b","b"]}});
        let source = Source::new(usize::MAX);
        let funding = Funding::enforced(source.clone());
        let error = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
            &options, &schema, &funding,
        )
        .unwrap_err();
        assert!(matches!(error, CompileError::Schema(_)));
        let calls = source.calls.load(Ordering::Relaxed);
        for refuse_at in 0..calls {
            let source = Source::new(refuse_at);
            let error = crate::compiler::build_validator_with_funding::<crate::SerdeJson>(
                &options,
                &schema,
                &Funding::enforced(source.clone()),
            )
            .unwrap_err();
            assert!(matches!(error, CompileError::Storage(_)), "{error:?}");
            assert_eq!(source.calls.load(Ordering::Relaxed), refuse_at + 1);
        }
    }
}
