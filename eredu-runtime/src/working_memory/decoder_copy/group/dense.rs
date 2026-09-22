//! One fresh prompt claim around a complete dense outer/child group.
use super::*;
use crate::working_memory::{
    InferencePreparationStage, WorkingMemoryFundingRun, WorkingMemoryFundingScope,
    WorkingMemoryReservation,
};
use crate::{DenseHostSlotInitialization, HostSlotAttachmentError};

/// Type-changing actual source plans for one dense outer and every child table.
/// Child i must be the native provider's actual child of outer i. Runtime checks
/// their ordered count, exact identities/capacities, origin health and checked P.
#[must_use = "construct only through the original fresh prompt stage"]
pub struct RegisteredDenseDecoderTableGroup<'a, OS, OD, CS, CD, K: HostSlotStorageKey> {
    outer: RegisteredDenseDecoderInitialization<'a, OS, OD, K>,
    children: Vec<RegisteredDenseDecoderInitialization<'a, CS, CD, K>>,
    retained: u64,
    protected: u64,
    preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl<'a, OS, OD, CS, CD, K: HostSlotStorageKey>
    RegisteredDenseDecoderTableGroup<'a, OS, OD, CS, CD, K>
{
    /// Join actual source plans; no arbitrary count, byte term or initializer.
    pub fn new(
        outer: RegisteredDenseDecoderInitialization<'a, OS, OD, K>,
        children: Vec<RegisteredDenseDecoderInitialization<'a, CS, CD, K>>,
    ) -> Result<Self, DecoderCopyAdmissionError> {
        Self::new_with_preparation(outer, children, None)
    }
    /// Retain the admitted constructor after the actual child-plan vector.
    pub fn new_with_preparation(
        outer: RegisteredDenseDecoderInitialization<'a, OS, OD, K>,
        children: Vec<RegisteredDenseDecoderInitialization<'a, CS, CD, K>>,
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
        check_vector::<Option<InitializedDenseDecoderSlots<'_, CS, CD, K>>>(children.len())?;
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
    /// Actual outer count, also the exact number of child plans.
    pub fn len(&self) -> usize {
        self.outer.len()
    }
    /// Whether the actual outer source has no slots.
    pub fn is_empty(&self) -> bool {
        self.outer.is_empty()
    }
    /// Sum of all retained dense destination payloads.
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }
    /// Sum of all exact dense initialization/fill envelopes.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.protected
    }

    /// Counted group constructors; per-table binding/metadata/transfer and the
    /// actual source pin topology are separate additive owning contributions.
    pub fn preparation_control_bytes(
        children: usize,
    ) -> Result<usize, preparation::DecoderHostPreparationError> {
        use std::mem::size_of;
        let tables = children
            .checked_add(1)
            .ok_or(preparation::DecoderHostPreparationError::Overflow)?;
        preparation::add([
            preparation::vector_bytes::<RegisteredDenseDecoderInitialization<'_, CS, CD, K>>(
                children,
            )?,
            preparation::vector_bytes::<Option<InitializedDenseDecoderSlots<'_, CS, CD, K>>>(
                children,
            )?,
            preparation::vector_bytes::<AtomicBool>(children)?,
            preparation::vector_bytes::<u64>(tables)?,
            preparation::vector_bytes::<DecoderCopySource<'_, K>>(tables)?,
            preparation::vector_bytes::<WorkingMemoryDecoderHostScope>(tables)?,
            usize::try_from(preparation::memory(
                crate::working_memory::qualified_storage::shared_bytes::<GroupProgress>(),
            )?)
            .map_err(|_| preparation::DecoderHostPreparationError::Overflow)?,
            size_of::<Self>(),
            size_of::<InitializedDenseDecoderTableGroup<'_, OS, OD, CS, CD, K>>(),
            size_of::<GroupProgress>(),
            size_of::<GroupProgressOwner>(),
            size_of::<Option<GroupProgress>>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
            size_of::<std::vec::IntoIter<WorkingMemoryDecoderHostScope>>(),
            size_of::<std::vec::IntoIter<RegisteredDenseDecoderInitialization<'_, CS, CD, K>>>(),
            size_of::<Result<Self, DecoderCopyAdmissionError>>(),
        ])
    }

    pub(in crate::working_memory) fn construct(
        mut self,
        stage: InferencePreparationStage,
        reservation: &WorkingMemoryReservation,
        funding: &WorkingMemoryFundingRun,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<
        (
            InitializedDenseDecoderTableGroup<'a, OS, OD, CS, CD, K>,
            WorkingMemoryFundingScope,
        ),
        DecoderCopyAdmissionError,
    > {
        let host = complete_source.source_preparation().cloned();
        self.outer.prepare_destination(host.as_ref())?;
        for child in &mut self.children {
            child.prepare_destination(host.as_ref())?;
        }
        let count = self
            .len()
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let progress = GroupProgress::new_prepared(self.len(), host.as_ref())?;
        let mut children =
            crate::working_memory::qualified_storage::vector(self.len(), host.is_some())?;
        let mut holds = crate::working_memory::qualified_storage::vector(count, host.is_some())?;
        holds.extend(
            std::iter::once(self.outer.initialization_peak_bytes())
                .chain(self.children.iter().map(|p| p.initialization_peak_bytes())),
        );
        let mut sources = crate::working_memory::qualified_storage::vector(count, host.is_some())?;
        sources.extend(
            std::iter::once(self.outer.source()).chain(self.children.iter().map(|p| p.source())),
        );
        let pin_count = sources
            .iter()
            .filter(|source| matches!(source, DecoderCopySource::Registered(_)))
            .count()
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let pins = std::iter::once(Some(RegisteredStoragePin::new(complete_source.clone())))
            .chain(std::iter::once(self.outer.registered_pin()))
            .chain(self.children.iter().map(|p| p.registered_pin()))
            .flatten();
        let pins = if host.is_some() {
            RegisteredStoragePin::aggregate_counted(pins, pin_count)?
        } else {
            RegisteredStoragePin::aggregate(pins)
        };
        let (execution, scopes, native) = funding.open_grouped_dense_prompt_scopes(
            reservation,
            &sources,
            &holds,
            &complete_source,
            pins,
        )?;
        drop(sources);
        let prepared = (|| {
            self.outer
                .prepare_before_allocation(&scopes[0], &execution)?;
            for (child, scope) in self.children.iter_mut().zip(scopes.iter().skip(1)) {
                child.prepare_before_allocation(scope, &execution)?;
            }
            Ok::<_, WorkingMemoryError>(())
        })();
        if let Err(error) = prepared {
            // No table or native operation has yet been constructed or exposed.
            native.certify()?;
            return Err(error.into());
        }
        #[cfg(test)]
        super::tests::before_initialize();
        let mut scopes = scopes.into_iter();
        let outer = self.outer.initialize_funded(
            Some(stage),
            execution.clone(),
            scopes.next().expect("outer host scope"),
        );
        for child in self.children {
            children.push(Some(child.initialize_funded(
                None,
                execution.clone(),
                scopes.next().expect("child host scope"),
            )));
        }
        Ok((
            InitializedDenseDecoderTableGroup {
                outer,
                children,
                progress,
                retained: self.retained,
                protected: self.protected,
            },
            native,
        ))
    }
}
impl<OS, OD, CS, CD, K: HostSlotStorageKey> fmt::Debug
    for RegisteredDenseDecoderTableGroup<'_, OS, OD, CS, CD, K>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredDenseDecoderTableGroup")
            .field("outer", &self.outer)
            .field("children", &self.children.len())
            .field("protected", &self.protected)
            .finish()
    }
}

/// The sole original prompt claim remains in the outer owner. Child builders
/// can publish actual tables, but never produce or finish a prompt completion.
#[must_use = "retain every partial table through native failure recovery"]
pub struct InitializedDenseDecoderTableGroup<'a, OS, OD, CS, CD, K: HostSlotStorageKey> {
    outer: InitializedDenseDecoderSlots<'a, OS, OD, K>,
    children: Vec<Option<InitializedDenseDecoderSlots<'a, CS, CD, K>>>,
    progress: GroupProgressOwner,
    retained: u64,
    protected: u64,
}
impl<'a, OS, OD, CS, CD, K: HostSlotStorageKey>
    InitializedDenseDecoderTableGroup<'a, OS, OD, CS, CD, K>
{
    /// Match the exact separately borrowed outer native source before any work.
    pub fn validate_source(
        &self,
        source: &DenseHostSlotInitialization<'_, OS, OD>,
    ) -> Result<(), WorkingMemoryError> {
        self.outer.validate_source(source)
    }
    /// Exact outer count.
    pub fn len(&self) -> usize {
        self.outer.len()
    }
    /// Whether the actual outer has no values.
    pub fn is_empty(&self) -> bool {
        self.outer.is_empty()
    }
    /// Already installed outer values.
    pub fn initialized_count(&self) -> usize {
        self.outer.initialized_count()
    }
    /// Sum of all exact dense destination payloads.
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }
    /// Original sum of separately protected table envelopes.
    pub fn protected_bytes(&self) -> u64 {
        self.protected
    }
    /// Take the corresponding already allocated child at most once.
    pub fn take_child(
        &mut self,
        index: usize,
    ) -> Result<InitializedDenseDecoderGroupChild<'a, CS, CD, K>, WorkingMemoryError> {
        let inner = self
            .children
            .get_mut(index)
            .and_then(Option::take)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        Ok(InitializedDenseDecoderGroupChild {
            inner,
            progress: self.progress.clone(),
            index,
        })
    }
    /// Move an actual outer value whose nested resources have independent custody.
    pub fn push(&mut self, value: OD) -> Result<(), HostSlotPushError<OD>> {
        self.outer.push(value)
    }
    /// Only all published child tables plus an exact outer fill return the
    /// completed outer owner. Its existing publish method yields the sole claim.
    pub fn finish(
        self,
    ) -> Result<FundedDenseHostSlots<'a, OS, OD, K>, DecoderGroupFinishError<Self>> {
        if !self.progress.is_complete() || self.initialized_count() != self.len() {
            return Err(DecoderGroupFinishError {
                owner: self,
                error: WorkingMemoryError::PreparationNotReady,
            });
        }
        Ok(self
            .outer
            .finish()
            .expect("checked exact grouped dense outer"))
    }
}
impl<OS, OD, CS, CD, K: HostSlotStorageKey> fmt::Debug
    for InitializedDenseDecoderTableGroup<'_, OS, OD, CS, CD, K>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedDenseDecoderTableGroup")
            .field("outer", &self.outer)
            .field("children", &self.children.len())
            .finish_non_exhaustive()
    }
}

/// A taken dense child with protected payload but no prompt claim.
pub struct InitializedDenseDecoderGroupChild<'a, S, D, K: HostSlotStorageKey> {
    inner: InitializedDenseDecoderSlots<'a, S, D, K>,
    progress: GroupProgressOwner,
    index: usize,
}
impl<'a, S, D, K: HostSlotStorageKey> InitializedDenseDecoderGroupChild<'a, S, D, K> {
    /// Match exact actual child identity/geometry before any native fill.
    pub fn validate_source(
        &self,
        source: &DenseHostSlotInitialization<'_, S, D>,
    ) -> Result<(), WorkingMemoryError> {
        self.inner.validate_source(source)
    }
    /// Required child slots.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Whether this actual child has no slots.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Already installed values.
    pub fn initialized_count(&self) -> usize {
        self.inner.initialized_count()
    }
    /// Exact retained dense child payload.
    pub fn retained_bytes(&self) -> u64 {
        self.inner.retained_bytes()
    }
    /// Exact independently protected child envelope.
    pub fn protected_bytes(&self) -> u64 {
        self.inner.protected_bytes()
    }
    /// Move an already independently constructed nested value without growth.
    pub fn push(&mut self, value: D) -> Result<(), HostSlotPushError<D>> {
        self.inner.push(value)
    }
    /// Complete only this child's fixed buffer, never the enclosing prompt.
    pub fn finish(
        self,
    ) -> Result<FundedDenseDecoderGroupChild<'a, S, D, K>, DecoderGroupFinishError<Self>> {
        if self.initialized_count() != self.len() {
            return Err(DecoderGroupFinishError {
                owner: self,
                error: WorkingMemoryError::PreparationNotReady,
            });
        }
        let inner = self
            .inner
            .finish()
            .expect("checked exact grouped dense child");
        Ok(FundedDenseDecoderGroupChild {
            inner,
            progress: self.progress,
            index: self.index,
        })
    }
}
impl<S, D, K: HostSlotStorageKey> fmt::Debug for InitializedDenseDecoderGroupChild<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedDenseDecoderGroupChild")
            .field("inner", &self.inner)
            .field("index", &self.index)
            .finish()
    }
}

/// Exact completed child whose protected hold can transfer only to its real table.
/// No standalone parent-stage owner or completion can be extracted.
pub struct FundedDenseDecoderGroupChild<'a, S, D, K: HostSlotStorageKey> {
    inner: FundedDenseHostSlots<'a, S, D, K>,
    progress: GroupProgressOwner,
    index: usize,
}
impl<'a, S, D, K: HostSlotStorageKey> FundedDenseDecoderGroupChild<'a, S, D, K> {
    /// Actual destination identity used by the existing exact storage handoff.
    pub fn metadata(&self) -> &crate::HostSlotMetadata {
        self.inner.metadata()
    }
    /// Completed child slots.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Whether the actual completed child has no slots.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Readonly actual child value.
    pub fn get(&self, index: usize) -> Option<&D> {
        self.inner.get(index)
    }
    /// Actual retained child payload.
    pub fn retained_bytes(&self) -> u64 {
        self.inner.retained_bytes()
    }
    /// Original child host envelope.
    pub fn protected_bytes(&self) -> u64 {
        self.inner.protected_bytes()
    }
    /// Publish and attach the exact child charge before returning the table.
    /// Success advances private group progress but returns no prompt completion.
    pub fn publish(self, key: K) -> Result<HostSlotTable<D>, DecoderGroupHandoffError<Self>> {
        let Self {
            inner,
            progress,
            index,
        } = self;
        match inner.publish_table(key) {
            Ok((table, stage)) => {
                debug_assert!(stage.is_none());
                progress.complete(index);
                Ok(table)
            }
            Err(error) => {
                let (inner, error) = error.into_parts();
                Err(DecoderGroupHandoffError {
                    owner: Self {
                        inner,
                        progress,
                        index,
                    },
                    error,
                })
            }
        }
    }
}
impl<S, D, K: HostSlotStorageKey> fmt::Debug for FundedDenseDecoderGroupChild<'_, S, D, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FundedDenseDecoderGroupChild")
            .field("inner", &self.inner)
            .field("index", &self.index)
            .finish()
    }
}

/// Failed child publication preserves its exact complete owner and original cause.
#[must_use = "recover or retire the completed child with its host custody"]
pub struct DecoderGroupHandoffError<O> {
    owner: O,
    error: HostSlotAttachmentError<WorkingMemoryError>,
}
impl<O> DecoderGroupHandoffError<O> {
    /// Original typed attachment/accounting failure.
    pub fn error(&self) -> &HostSlotAttachmentError<WorkingMemoryError> {
        &self.error
    }
    /// Recover the complete child and its error without raw table extraction.
    pub fn into_parts(self) -> (O, HostSlotAttachmentError<WorkingMemoryError>) {
        (self.owner, self.error)
    }
}
impl<O> fmt::Debug for DecoderGroupHandoffError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DecoderGroupHandoffError")
            .field(&self.error)
            .finish()
    }
}
impl<O> fmt::Display for DecoderGroupHandoffError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl<O> std::error::Error for DecoderGroupHandoffError<O> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
