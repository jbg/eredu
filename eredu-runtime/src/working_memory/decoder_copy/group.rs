//! Closed two-level tables; native providers bind actual child-of-parent semantics.
use super::*;
use crate::HostSlotTable;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

mod dense;
pub use dense::*;

fn check_vector<T>(count: usize) -> Result<(), WorkingMemoryError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or(WorkingMemoryError::Overflow)
        .map(|_| ())
}
fn sum(mut values: impl Iterator<Item = u64>) -> Result<u64, WorkingMemoryError> {
    values.try_fold(0u64, |a, b| {
        a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
    })
}

#[derive(Debug)]
pub(super) struct GroupProgress {
    ready: Box<[AtomicBool]>,
}

#[derive(Debug)]
pub(super) struct GroupProgressOwner {
    inner: Option<Arc<GroupProgress>>,
    preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl Clone for GroupProgressOwner {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            preparation: self.preparation.clone(),
        }
    }
}
impl Drop for GroupProgressOwner {
    fn drop(&mut self) {
        // Final Arc allocation and its ready buffer retire before the grant.
        if let Some(inner) = self.inner.take() {
            drop(Arc::into_inner(inner));
        }
    }
}
impl std::ops::Deref for GroupProgressOwner {
    type Target = GroupProgress;
    fn deref(&self) -> &GroupProgress {
        self.inner.as_deref().expect("live progress")
    }
}
impl GroupProgress {
    fn new(count: usize) -> Result<GroupProgressOwner, WorkingMemoryError> {
        Self::new_prepared(count, None)
    }
    fn new_prepared(
        count: usize,
        authority: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<GroupProgressOwner, WorkingMemoryError> {
        check_vector::<AtomicBool>(count)?;
        let mut ready = super::super::qualified_storage::vector(count, authority.is_some())?;
        for _ in 0..count {
            ready.push(AtomicBool::new(false));
        }
        Ok(GroupProgressOwner {
            inner: Some(Arc::new(Self {
                ready: ready.into_boxed_slice(),
            })),
            preparation: authority.cloned(),
        })
    }
    fn complete(&self, index: usize) {
        self.ready[index].store(true, Ordering::Release);
    }
    fn is_complete(&self) -> bool {
        self.ready.iter().all(|x| x.load(Ordering::Acquire))
    }
}

/// A failed group/child finish retains its entire partial owner for recovery.
/// It never exposes the underlying unfunded primitive or a prompt claim.
#[must_use = "recover or retire the entire funded owner"]
pub struct DecoderGroupFinishError<O> {
    owner: O,
    error: WorkingMemoryError,
}
impl<O> DecoderGroupFinishError<O> {
    /// Original typed state/count rejection.
    pub fn error(&self) -> &WorkingMemoryError {
        &self.error
    }
    /// Recover the same group or child with every payload and hold intact.
    pub fn into_owner(self) -> O {
        self.owner
    }
}
impl<O> fmt::Debug for DecoderGroupFinishError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DecoderGroupFinishError")
            .field(&self.error)
            .finish()
    }
}
impl<O> fmt::Display for DecoderGroupFinishError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl<O> std::error::Error for DecoderGroupFinishError<O> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl<'a, S, D, K: Ord + Send + 'static> RegisteredDecoderHostCopy<'a, S, K, D> {
    /// Change only destination inline type, preserving actual source values and
    /// registered/funded origin. This grants no nested allocation or conversion.
    pub fn for_destination<N>(
        self,
    ) -> Result<RegisteredDecoderHostCopy<'a, S, K, N>, DecoderCopyAdmissionError> {
        Ok(RegisteredDecoderHostCopy {
            plan: self.plan.for_destination()?,
            source: self.source,
            destination_identity: self.destination_identity,
        })
    }
    fn registered_pin(&self) -> Option<RegisteredStoragePin>
    where
        K: Clone + Sync,
    {
        match &self.source {
            DecoderSource::Registered(s) => Some(RegisteredStoragePin::new(s.clone())),
            DecoderSource::Funded { .. } => None,
        }
    }
    fn initialize_funded(
        self,
        execution: InferenceExecutionIdentity,
        custody: WorkingMemoryDecoderHostScope,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> InitializedDecoderSlots<D> {
        let source_identity = self.plan.source_metadata().identity().clone();
        let source_capacity = self
            .plan
            .source_metadata()
            .capacity_bytes()
            .expect("prepared source extent");
        let retained = self.retained_bytes();
        let protected = self.initialization_peak_bytes();
        let slots = match preparation {
            Some(authority) => self.plan.initialize_prepared(
                self.destination_identity
                    .expect("identity prepared before admission"),
                authority,
            ),
            None => self.plan.initialize(),
        };
        InitializedDecoderSlots {
            slots,
            source_identity,
            source_capacity,
            execution,
            retained,
            protected,
            custody,
        }
    }
}

/// Exact outer source plus one ordered actual child source per outer slot.
/// The native closed provider must prove child i belongs to actual outer i.
/// Runtime verifies count, exact table identities/capacities and every origin;
/// this generic group alone is not a transitive source-completeness certificate.
#[must_use = "join the actual group to one admitted copy"]
pub struct RegisteredDecoderTableGroup<'a, OS, OD, CS, CD, K: Ord + Send + 'static> {
    outer: RegisteredDecoderHostCopy<'a, OS, K, OD>,
    children: Vec<RegisteredDecoderHostCopy<'a, CS, K, CD>>,
    retained: u64,
    protected: u64,
    // Saved-only child plans still allocate this vector. Keep its grant even
    // when no source binding owns a prepared registration.
    preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl<'a, OS, OD, CS, CD, K: Ord + Send + 'static>
    RegisteredDecoderTableGroup<'a, OS, OD, CS, CD, K>
{
    /// Child index is its unique vector position. Equal source identities do
    /// not deduplicate distinct destination tables. No byte term is accepted.
    pub fn new(
        outer: RegisteredDecoderHostCopy<'a, OS, K, OD>,
        children: Vec<RegisteredDecoderHostCopy<'a, CS, K, CD>>,
    ) -> Result<Self, DecoderCopyAdmissionError> {
        Self::new_with_preparation(outer, children, None)
    }
    /// Shared counted owner retaining the enclosing host preparation after all
    /// source-plan storage. It creates no capacity or copy-account authority.
    pub fn new_with_preparation(
        outer: RegisteredDecoderHostCopy<'a, OS, K, OD>,
        children: Vec<RegisteredDecoderHostCopy<'a, CS, K, CD>>,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<Self, DecoderCopyAdmissionError> {
        if children.len() != outer.len() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let count = children
            .len()
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        check_vector::<u64>(count)?;
        check_vector::<DecoderCopySource<'_, K>>(count)?;
        check_vector::<Option<InitializedDecoderSlots<CD>>>(children.len())?;
        let retained = sum(std::iter::once(outer.retained_bytes())
            .chain(children.iter().map(|p| p.retained_bytes())))?;
        let protected = sum(std::iter::once(outer.initialization_peak_bytes())
            .chain(children.iter().map(|p| p.initialization_peak_bytes())))?;
        Ok(Self {
            outer,
            children,
            retained,
            protected,
            preparation: preparation.cloned(),
        })
    }
    /// Actual outer extent and exact number of child tables.
    pub fn len(&self) -> usize {
        self.outer.len()
    }
    /// Whether the actual outer table is empty.
    pub fn is_empty(&self) -> bool {
        self.outer.is_empty()
    }
    /// Sum of the actual outer/child destination payloads.
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }
    /// Sum of all exact outer/child construction envelopes.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.protected
    }
}
impl<OS, OD, CS, CD, K: Ord + Send + 'static> fmt::Debug
    for RegisteredDecoderTableGroup<'_, OS, OD, CS, CD, K>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredDecoderTableGroup")
            .field("outer", &self.outer)
            .field("children", &self.children.len())
            .field("protected", &self.protected)
            .finish()
    }
}

/// Sampler, native program and all actual grouped tables under one account.
#[must_use = "admit the complete grouped operation before copying"]
pub struct RegisteredTextComponentsGroupCopy<'a, OS, OD, CS, CD, K: Ord + Send + 'static> {
    sampling: RegisteredSamplingCopy<'a, K>,
    group: RegisteredDecoderTableGroup<'a, OS, OD, CS, CD, K>,
    complete_source: WorkingMemoryStorage<K>,
    bytes: u64,
}
impl<'a, K: Clone + Ord + Send + Sync + 'static> RegisteredSamplingCopy<'a, K> {
    /// Joins actual two-level tables and mandatory complete registered native
    /// inventory. Source completeness and nested native work remain provider duties.
    pub fn with_decoder_group<OS, OD, CS, CD>(
        self,
        group: RegisteredDecoderTableGroup<'a, OS, OD, CS, CD, K>,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<RegisteredTextComponentsGroupCopy<'a, OS, OD, CS, CD, K>, DecoderCopyAdmissionError>
    {
        let bytes = self
            .required_bytes()
            .checked_add(group.protected)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(RegisteredTextComponentsGroupCopy {
            sampling: self,
            group,
            complete_source,
            bytes,
        })
    }
}
impl<OS, OD, CS, CD, K: Ord + Send + 'static>
    RegisteredTextComponentsGroupCopy<'_, OS, OD, CS, CD, K>
{
    /// Combined destination demand, excluding independently charged source/safety.
    pub fn required_bytes(&self) -> u64 {
        self.bytes
    }
}

/// Partial outer and untaken children, with each exact table hold independent.
/// Taken children remain funded if this coordinator retires first.
#[must_use = "finish or retire the complete group and all taken children"]
pub struct InitializedDecoderTableGroup<OD, CD> {
    outer: InitializedDecoderSlots<OD>,
    children: Vec<Option<InitializedDecoderSlots<CD>>>,
    progress: GroupProgressOwner,
    retained: u64,
    protected: u64,
}
impl<OD, CD> InitializedDecoderTableGroup<OD, CD> {
    /// Validate the actual separately retained outer native source before work.
    pub fn validate_source<S>(
        &self,
        source: &HostSlotInitialization<'_, S, OD>,
    ) -> Result<(), WorkingMemoryError> {
        self.outer.validate_source(source)
    }
    /// Exact outer slot count.
    pub fn len(&self) -> usize {
        self.outer.len()
    }
    /// Whether the actual outer source is empty.
    pub fn is_empty(&self) -> bool {
        self.outer.is_empty()
    }
    /// Installed outer values.
    pub fn initialized_count(&self) -> usize {
        self.outer.initialized_count()
    }
    /// Sum of actual destination table payloads.
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }
    /// Original sum of independent table holds.
    pub fn protected_bytes(&self) -> u64 {
        self.protected
    }
    /// Take the corresponding child at most once; no new allocation occurs.
    pub fn take_child(
        &mut self,
        index: usize,
    ) -> Result<InitializedDecoderGroupChild<CD>, WorkingMemoryError> {
        let slots = self
            .children
            .get_mut(index)
            .and_then(Option::take)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        Ok(InitializedDecoderGroupChild {
            slots,
            progress: self.progress.clone(),
            index,
        })
    }
    /// Move one already-created outer value. Independent nested resources remain
    /// the native worker's responsibility and are preserved by full rejection.
    pub fn push(&mut self, value: OD) -> Result<(), HostSlotPushError<OD>> {
        self.outer.push(value)
    }
    /// Only all finished children plus an exact outer fill can finish the group.
    pub fn finish(self) -> Result<FundedDecoderSlots<OD>, DecoderGroupFinishError<Self>> {
        if !self.progress.is_complete() || self.initialized_count() != self.len() {
            return Err(DecoderGroupFinishError {
                owner: self,
                error: WorkingMemoryError::PreparationNotReady,
            });
        }
        Ok(self
            .outer
            .finish()
            .expect("checked exact grouped outer fill"))
    }
}
impl<OD, CD> fmt::Debug for InitializedDecoderTableGroup<OD, CD> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedDecoderTableGroup")
            .field("outer", &self.outer)
            .field("children", &self.children.len())
            .finish_non_exhaustive()
    }
}

/// One taken grouped child; it has no runnable or prompt completion authority.
pub struct InitializedDecoderGroupChild<D> {
    slots: InitializedDecoderSlots<D>,
    progress: GroupProgressOwner,
    index: usize,
}
impl<D> InitializedDecoderGroupChild<D> {
    /// Validate the exact original child plan before nested numerical work.
    pub fn validate_source<S>(
        &self,
        source: &HostSlotInitialization<'_, S, D>,
    ) -> Result<(), WorkingMemoryError> {
        self.slots.validate_source(source)
    }
    /// Exact child slot count.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Whether the actual child has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Number of values already installed.
    pub fn initialized_count(&self) -> usize {
        self.slots.initialized_count()
    }
    /// Exact child retained payload.
    pub fn retained_bytes(&self) -> u64 {
        self.slots.retained_bytes()
    }
    /// Original child host hold.
    pub fn protected_bytes(&self) -> u64 {
        self.slots.protected_bytes()
    }
    /// Move one independently constructed value without growth.
    pub fn push(&mut self, value: D) -> Result<(), HostSlotPushError<D>> {
        self.slots.push(value)
    }
    /// Finish the actual child once; its funded owner carries no prompt claim.
    pub fn finish(self) -> Result<FundedDecoderSlots<D>, DecoderGroupFinishError<Self>> {
        if self.initialized_count() != self.len() {
            return Err(DecoderGroupFinishError {
                owner: self,
                error: WorkingMemoryError::PreparationNotReady,
            });
        }
        let slots = self
            .slots
            .finish()
            .expect("checked exact grouped child fill");
        self.progress.complete(self.index);
        Ok(slots)
    }
}
impl<D> fmt::Debug for InitializedDecoderGroupChild<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedDecoderGroupChild")
            .field("slots", &self.slots)
            .field("index", &self.index)
            .finish()
    }
}

impl WorkingMemoryPool {
    /// One same-lock source check/account commit for sampler, outer, all children,
    /// operands and full source inventory. Only native scope owns the pin bundle.
    /// All table P holds precede the first initializer; native adoption cannot
    /// consume them. Neither child finish nor host retirement certifies native work.
    pub fn copy_text_components_group<OS, OD, CS, CD, K: Clone + Ord + Send + Sync + 'static>(
        &self,
        mut copy: RegisteredTextComponentsGroupCopy<'_, OS, OD, CS, CD, K>,
        limits: WorkspaceCopyLimits,
    ) -> Result<
        (
            FundedSamplerCopy,
            InitializedDecoderTableGroup<OD, CD>,
            AdmittedWorkspaceCopy,
        ),
        DecoderCopyAdmissionError,
    > {
        let bytes = copy
            .bytes
            .checked_add(limits.safety_reserve_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        if let Some(budget_bytes) = limits.application_memory_budget_bytes {
            if bytes > budget_bytes {
                return Err(DecoderCopyAdmissionError::ApplicationBudgetExceeded {
                    required_bytes: bytes,
                    budget_bytes,
                });
            }
        }
        let preparation = copy.complete_source.source_preparation().cloned();
        copy.group
            .outer
            .prepare_destination_identity(preparation.as_ref())?;
        for child in &mut copy.group.children {
            child.prepare_destination_identity(preparation.as_ref())?;
        }
        let exact = preparation.is_some();
        let execution = match preparation.as_ref() {
            Some(preparation) => {
                let tables = copy
                    .group
                    .len()
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                super::super::WorkspaceCopyAccountLayout::decoder_group(tables)?
                    .execution(preparation)
            }
            None => InferenceExecutionIdentity::default(),
        };
        let progress = GroupProgress::new_prepared(copy.group.len(), preparation.as_ref())?;
        let mut children = super::super::qualified_storage::vector(copy.group.len(), exact)?;
        let count = copy
            .group
            .len()
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let mut holds = super::super::qualified_storage::vector(count, exact)?;
        let mut sources = super::super::qualified_storage::vector(count, exact)?;
        holds.push(copy.group.outer.initialization_peak_bytes());
        sources.push(copy.group.outer.source());
        for child in &copy.group.children {
            holds.push(child.initialization_peak_bytes());
            sources.push(child.source());
        }
        let operands = copy.sampling.arrays.source().registration();
        let registered = std::iter::once(&copy.group.outer.source)
            .chain(copy.group.children.iter().map(|child| &child.source))
            .filter(|source| matches!(source, DecoderSource::Registered(_)))
            .count();
        let count = 2usize
            .checked_add(registered)
            .ok_or(WorkingMemoryError::Overflow)?;
        let pins = [
            Some(RegisteredStoragePin::new(operands.clone())),
            Some(RegisteredStoragePin::new(copy.complete_source.clone())),
        ]
        .into_iter()
        .chain(std::iter::once(copy.group.outer.registered_pin()))
        .chain(copy.group.children.iter().map(|p| p.registered_pin()))
        .flatten();
        let pin = if copy.complete_source.has_source_preparation() {
            RegisteredStoragePin::aggregate_counted(pins, count)?
        } else {
            RegisteredStoragePin::aggregate(pins)
        };
        let host = copy.sampling.host_bytes;
        let (funding, host_scope, scopes, scope) = self.open_grouped_text_components_account(
            copy.sampling.sampler.source(),
            copy.sampling.sampler.execution(),
            &sources,
            &holds,
            operands,
            &copy.complete_source,
            pin,
            &execution,
            bytes,
            host,
            limits.capacity_bytes,
        )?;
        drop(sources);
        // The native custody is created before any fallible payload work; unwind
        // keeps complete source pins quarantined while actual payload fields retire.
        let native = AdmittedWorkspaceCopy::from_account(execution.clone(), bytes, funding, scope);
        #[cfg(test)]
        tests::before_initialize();
        let sampler = copy.sampling.sampler_plan.copy();
        let sampler =
            FundedSamplerCopy::from_shared_account(sampler, execution.clone(), host, host_scope);
        let mut scopes = scopes.into_iter();
        let outer = copy.group.outer.initialize_funded(
            execution.clone(),
            scopes.next().expect("outer scope"),
            preparation.as_ref(),
        );
        for child in copy.group.children {
            children.push(Some(child.initialize_funded(
                execution.clone(),
                scopes.next().expect("child scope"),
                preparation.as_ref(),
            )));
        }
        Ok((
            sampler,
            InitializedDecoderTableGroup {
                outer,
                children,
                progress,
                retained: copy.group.retained,
                protected: copy.group.protected,
            },
            native,
        ))
    }
}

#[cfg(test)]
mod tests;
