//! Consumed native preparation for the shared isolated-copy program.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment, OriginalCopyEnvironmentError,
    nn::workspace::IsolatedCopyNativeLayout,
    submission_recovery::{PreparedRecovery, Recovery, Retention},
};
use eredu_core::BackendFailure;
use eredu_runtime::working_memory::{
    WorkingMemoryPool, WorkspaceCopyCustody, WorkspaceCopyRetention,
};
use safemlx::{
    ArrayDescriptorError, DeviceType, Dtype, InitializedInputAllocator, OperationEvent,
    OriginalBufferBudget, OriginalBufferCause, OriginalNativeControlError, OriginalScopeObserver,
    PipelineCacheCause, PrefillFailureCause, PrefillRootsRuntime, PreparedOriginalBufferBudget,
    PreparedPipelineCache, PreparedPipelineCachePlan, PreparedPrefillFailure,
    PreparedResidentGraph, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    RetainedPrefillFailure, ScopedSubmissionProgress, StreamCopyCause, StreamCopyPlan,
    SubmissionGraphQuota, SubmissionGraphQuotaCause, SubmissionRecordQuota,
    SubmissionRecordQuotaCause, SubmissionScopeOwnerCause,
};
use std::mem::{size_of, size_of_val};
#[path = "original/realtime.rs"]
mod realtime;
pub(crate) use realtime::{RealtimeCopyPlan,RealtimeCopyContext};
#[path = "original/host_store.rs"]
mod host_store;
pub(crate) use host_store::{PreparedSavedHostCopy, SavedHostCopyPlan};

#[derive(Debug, thiserror::Error)]
pub(crate) enum OriginalCopyCause {
    #[error("native isolated-copy layout is not qualified")]
    UnknownLayout,
    #[error("native isolated-copy layout overflows")]
    Overflow,
    #[error("native isolated-copy source has no completed backing")]
    UnsettledSource,
    #[error("native isolated-copy request belongs to a different pool")]
    ForeignPool,
    #[error("native isolated-copy physical bound exceeds the admitted copy envelope")]
    Capacity,
    #[error(transparent)]
    Environment(#[from] OriginalCopyEnvironmentError),
    #[error(transparent)]
    Descriptor(#[from] ArrayDescriptorError),
    #[error(transparent)]
    HostDescriptor(#[from] safemlx::HostTransferMetadataError),
    #[error(transparent)]
    HostInput(#[from] safemlx::PreparedInputCause),
    #[error(transparent)]
    HostStore(#[from] safemlx::PreparedHostCopyError),
    #[error(transparent)]
    Pending(#[from] super::PendingTokenSourceCause),
    #[error(transparent)]
    Stream(#[from] StreamCopyCause),
    #[error(transparent)]
    Buffer(#[from] OriginalBufferCause),
    #[error(transparent)]
    Graph(#[from] SubmissionGraphQuotaCause),
    #[error(transparent)]
    Record(#[from] SubmissionRecordQuotaCause),
    #[error(transparent)]
    Pipeline(#[from] PipelineCacheCause),
    #[error(transparent)]
    Failure(#[from] PrefillFailureCause),
    #[error(transparent)]
    Scope(#[from] SubmissionScopeOwnerCause),
    #[error(transparent)]
    Controls(#[from] OriginalNativeControlError),
    #[error("native isolated-copy observation unavailable: {0:?}")]
    Observation(ScopedSubmissionProgress),
    #[error(transparent)]
    Native(#[from] Exception),
}

/// Fixed cause plus the same accepted account; no preparation is retried and
/// no failed native prefix loses its independently retained quota ownership.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct OriginalCopyFailure {
    #[source]
    cause: OriginalCopyCause,
    _custody: WorkspaceCopyRetention,
}
impl From<OriginalCopyFailure> for crate::backend::error::Error {
    fn from(value: OriginalCopyFailure) -> Self {
        Self::StorageSource(BackendFailure::from_error(value))
    }
}

/// Allocation-free census. The enclosing registered copy proof still owns
/// source identity, exclusion and immutability; this builder grants none.
#[derive(Default)]
pub(crate) struct OriginalCopyLayoutBuilder {
    operands: usize,
    source_clones: usize,
    logical_bytes: usize,
    maximum_rank: usize,
    host: HostPopulation,
    host_stores: host_store::Population,
}
#[derive(Default)]
struct HostPopulation {
    operands: usize,
    logical_bytes: usize,
    direct_extents: usize,
    controls: usize,
}
impl OriginalCopyLayoutBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// One actual initial source handle retained in the recovery collector.
    pub(crate) fn push_retained_source(&mut self, source: &Array) -> Result<(), OriginalCopyCause> {
        let descriptor = source.try_descriptor()?;
        if descriptor.facts().allocation().is_none() {
            return Err(OriginalCopyCause::UnsettledSource);
        }
        self.source_clones = self
            .source_clones
            .checked_add(1)
            .ok_or(OriginalCopyCause::Overflow)?;
        Ok(())
    }

    /// One actual invocation of contiguous + eager deep_copy, including aliases
    /// and repeated logical views. Shapes remain borrowed under the descriptor.
    pub(crate) fn push_operand(&mut self, source: &Array) -> Result<(), OriginalCopyCause> {
        let descriptor = source.try_descriptor()?;
        let facts = descriptor.facts();
        if facts.allocation().is_none() {
            return Err(OriginalCopyCause::UnsettledSource);
        }
        // These are exactly the source representations supported by the shared
        // saved-copy projection. Other typed deep-copy APIs remain ordinary.
        if !matches!(
            facts.dtype(),
            Dtype::Float32
                | Dtype::Float16
                | Dtype::Bfloat16
                | Dtype::Int32
                | Dtype::Uint32
                | Dtype::Uint8
                | Dtype::Bool
        ) {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        self.logical_bytes = self
            .logical_bytes
            .checked_add(facts.logical_bytes())
            .ok_or(OriginalCopyCause::Overflow)?;
        self.operands = self
            .operands
            .checked_add(1)
            .ok_or(OriginalCopyCause::Overflow)?;
        self.maximum_rank = self.maximum_rank.max(facts.rank());
        Ok(())
    }

    /// One actual immutable Host source followed by the same isolated-copy
    /// leaf. The enclosing source inventory still authenticates and owns it;
    /// this allocation-free census reads no payload and creates no grant.
    pub(crate) fn push_host_operand(
        &mut self,
        source: &safemlx::ImmutableHostTransferBuffer,
    ) -> Result<(), OriginalCopyCause> {
        let descriptor = source.try_fixed_descriptor::<4>()?;
        if descriptor.shape().is_empty()
            || descriptor.shape().iter().any(|n| *n <= 0)
            || descriptor.policy() != safemlx::HostTransferPolicy::Transfer
            || descriptor.storage_kind() != safemlx::HostTransferStorageKind::MetalShared
            || !matches!(
                descriptor.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let rank = descriptor.shape().len();
        let (controls, direct) =
            safemlx::ImmutableHostTransferBuffer::original_copy_layout(rank, descriptor.dtype())
                .ok_or(OriginalCopyCause::UnknownLayout)?;
        let controls = controls
            .checked_add(
                safemlx::HostTransferDescriptor::<4>::control_bytes()
                    .ok_or(OriginalCopyCause::UnknownLayout)?,
            )
            .and_then(|n| n.checked_add(size_of::<HostPopulation>()))
            .ok_or(OriginalCopyCause::Overflow)?;
        self.host.operands = self
            .host
            .operands
            .checked_add(1)
            .ok_or(OriginalCopyCause::Overflow)?;
        self.host.logical_bytes = self
            .host
            .logical_bytes
            .checked_add(descriptor.nbytes())
            .ok_or(OriginalCopyCause::Overflow)?;
        self.host.direct_extents = self
            .host
            .direct_extents
            .checked_add(direct)
            .ok_or(OriginalCopyCause::Overflow)?;
        self.host.controls = self
            .host
            .controls
            .checked_add(controls)
            .ok_or(OriginalCopyCause::Overflow)?;
        self.logical_bytes = self
            .logical_bytes
            .checked_add(descriptor.nbytes())
            .ok_or(OriginalCopyCause::Overflow)?;
        self.operands = self
            .operands
            .checked_add(1)
            .ok_or(OriginalCopyCause::Overflow)?;
        self.maximum_rank = self.maximum_rank.max(rank);
        Ok(())
    }

    /// One finite destination of the same copied operand; source matching and
    /// manager policy remain with the closed saved-state producer.
    pub(crate) fn push_host_destination(
        &mut self,
        plan: &SavedHostCopyPlan,
    ) -> Result<(), OriginalCopyCause> {
        self.host_stores.add(plan)?;
        self.maximum_rank = self.maximum_rank.max(4);
        Ok(())
    }

    fn layout(&self, pending_tail: usize) -> Option<IsolatedCopyNativeLayout> {
        IsolatedCopyNativeLayout::with_host_transfers(
            self.operands,
            self.source_clones,
            self.maximum_rank,
            pending_tail,
            self.host.operands,
            self.host.direct_extents,
            self.host.controls,
            self.host_stores.count,
            self.host_stores.direct,
            self.host_stores.controls,
        )
    }

    pub(crate) fn resume_control_bytes() -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<OriginalResumeCopyPopulation>(),
            size_of::<Option<OriginalResumeCopyPopulation>>(),
            size_of::<Result<OriginalResumeCopyPopulation, OriginalCopyCause>>(),
            size_of::<Option<OriginalCopyCause>>(),
            size_of::<Result<(), OriginalCopyCause>>(),
            size_of::<OriginalCopyCause>(),
            size_of::<safemlx::ArrayDescriptorFacts>(),
            size_of::<usize>() * 4,
            size_of::<bool>(),
            size_of::<DeviceType>(),
            size_of::<StreamCopyPlan<()>>()*2,
            size_of::<Result<StreamCopyPlan<()>,StreamCopyCause>>(),
            size_of::<StreamCopyCause>(),size_of::<(&Stream,)>(),
            size_of::<(Self, &super::PreparedPendingTokenInput<'_>, &Stream)>(),
            size_of::<(Self, u64, &Stream)>(),
            size_of::<(Self, &super::PreparedPendingTokenInput<'_>, DeviceType)>(),
            size_of::<Result<OriginalResumeCopyPopulation, OriginalCopyCause>>(),
            Array::descriptor_control_bytes()?,
            super::PreparedPendingTokenInput::source_control_bytes()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Derives the decoder/key/pending program for a fresh original request.
    /// The caller enumerates the actual retained source arrays first. The
    /// pending operand is validated here and appended exactly once; its reshape
    /// and optional signed-to-unsigned cast form the final completed frontier.
    pub(crate) fn finish_resume(
        self,
        pending: &Array,
    ) -> Result<OriginalResumeCopyPopulation, OriginalCopyCause> {
        let prepared = super::PreparedPendingTokenInput::new_fixed(pending)
            .map_err(OriginalCopyCause::Pending)?;
        self.finish_resume_input(&prepared)
    }

    /// Appends the exact saved scalar or complete pending-prefill program. Its
    /// bound source is revalidated before any shared census is consumed.
    pub(crate) fn finish_resume_input(
        self,
        prepared: &super::PreparedPendingTokenInput<'_>,
    ) -> Result<OriginalResumeCopyPopulation, OriginalCopyCause> {
        self.finish_resume_input_for(prepared, DeviceType::Gpu)
    }

    pub(crate) fn finish_resume_input_on(self,prepared:&super::PreparedPendingTokenInput<'_>,stream:&Stream)
        ->Result<OriginalResumeCopyPopulation,OriginalCopyCause> {
        let source=StreamCopyPlan::<()>::capture(stream)?;
        self.finish_resume_input_for(prepared,source.device_type())
    }

    fn finish_resume_input_for(
        mut self,
        prepared: &super::PreparedPendingTokenInput<'_>,
        device: DeviceType,
    ) -> Result<OriginalResumeCopyPopulation, OriginalCopyCause> {
        prepared
            .validate_fixed()
            .map_err(OriginalCopyCause::Pending)?;
        if !self.host_stores.empty() {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let pending = prepared.source();
        let facts = pending.try_descriptor()?.facts();
        self.push_operand(pending)?;
        if self.source_clones == 0 {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let cast = facts.dtype() == Dtype::Int32;
        let layout = match device {
            DeviceType::Gpu => self.layout(1 + usize::from(cast)),
            DeviceType::Cpu if self.host.operands==0 => IsolatedCopyNativeLayout::cpu_resume(
                self.operands,self.source_clones,self.maximum_rank,facts.rank(),
                usize::try_from(prepared.positions()).map_err(|_|OriginalCopyCause::Overflow)?,cast),
            _ => None,
        }.ok_or(OriginalCopyCause::UnknownLayout)?;
        let logical_bytes = self
            .logical_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.host.logical_bytes))
            .and_then(|n| n.checked_add(if cast { facts.logical_bytes() } else { 0 }))
            .ok_or(OriginalCopyCause::Overflow)?;
        let births = self
            .operands
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.host.operands))
            .and_then(|n| n.checked_add(usize::from(cast)))
            .ok_or(OriginalCopyCause::Overflow)?;
        let roots = self
            .source_clones
            .checked_add(
                self.operands
                    .checked_mul(2)
                    .ok_or(OriginalCopyCause::Overflow)?,
            )
            .and_then(|n| n.checked_add(self.host.operands))
            .and_then(|n| n.checked_add(1 + usize::from(cast)))
            .ok_or(OriginalCopyCause::Overflow)?;
        Ok(OriginalResumeCopyPopulation {
            layout,
            logical_bytes,
            births,
            roots,
            pending_positions: prepared.positions(),
        })
    }

    /// No token tail is constructed for a completed media input or a terminal
    /// restore (zero positions).
    /// Counts are solely the actual saved decoder/key descriptor census. Empty
    /// state/key therefore has zero births/roots/completions, not a fake token.
    pub(crate) fn finish_resume_completed_input(
        self,
        positions: u64,
    ) -> Result<OriginalResumeCopyPopulation, OriginalCopyCause> {
        self.finish_resume_completed_input_for(positions,DeviceType::Gpu)
    }
    pub(crate) fn finish_resume_completed_input_on(self,positions:u64,stream:&Stream)
        ->Result<OriginalResumeCopyPopulation,OriginalCopyCause> {
        let source=StreamCopyPlan::<()>::capture(stream)?;
        self.finish_resume_completed_input_for(positions,source.device_type())
    }
    fn finish_resume_completed_input_for(self,positions:u64,device:DeviceType)
        ->Result<OriginalResumeCopyPopulation,OriginalCopyCause> {
        if !self.host_stores.empty() {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let layout = match device {
            DeviceType::Gpu=>self.layout(0),
            DeviceType::Cpu if self.host.operands==0=>IsolatedCopyNativeLayout::cpu_completed(self.operands,self.source_clones,self.maximum_rank),
            _=>None,
        }.ok_or(OriginalCopyCause::UnknownLayout)?;
        let logical_bytes = self
            .logical_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.host.logical_bytes))
            .ok_or(OriginalCopyCause::Overflow)?;
        let births = self
            .operands
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.host.operands))
            .ok_or(OriginalCopyCause::Overflow)?;
        let roots = self
            .source_clones
            .checked_add(births)
            .ok_or(OriginalCopyCause::Overflow)?;
        Ok(OriginalResumeCopyPopulation {
            layout,
            logical_bytes,
            births,
            roots,
            pending_positions: positions,
        })
    }

    /// Empty copy programs require no native execution preparation. Every
    /// nonempty plan retains the actual backend stream/pool/allocator loans.
    pub(crate) fn finish<'a>(
        self,
        environment: &'a OriginalCopyEnvironment<'_>,
    ) -> Result<Option<OriginalCopyPlan<'a>>, OriginalCopyCause> {
        if self.operands == 0 && self.source_clones == 0 && self.host_stores.empty() {
            return Ok(None);
        }
        if self.operands == 0
            || !cfg!(all(
                target_vendor = "apple",
                not(feature = "cuda")
            ))
        {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let stream = StreamCopyPlan::<()>::capture(environment.stream())?;
        let layout = match stream.device_type() {
            DeviceType::Gpu => self.layout(0),
            DeviceType::Cpu if self.host.operands == 0 && self.host_stores.empty() => {
                IsolatedCopyNativeLayout::cpu(self.operands, self.source_clones, self.maximum_rank)
            }
            _ => None,
        }
        .ok_or(OriginalCopyCause::UnknownLayout)?;
        let logical_bytes = self
            .logical_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.host.logical_bytes))
            .ok_or(OriginalCopyCause::Overflow)?;
        let births = self
            .operands
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.host.operands))
            .ok_or(OriginalCopyCause::Overflow)?;
        self.finish_population(environment, stream, layout, logical_bytes, births)
    }

    /// Same source-bound copy plan with the shared pending-input tail. This
    /// retains the existing separate copy account and native completion engine.
    pub(crate) fn finish_pending_input<'a>(
        self,
        prepared: &super::PreparedPendingTokenInput<'_>,
        environment: &'a OriginalCopyEnvironment<'_>,
    ) -> Result<OriginalCopyPlan<'a>, OriginalCopyCause> {
        if self.operands != 0
            || self.source_clones != 1
            || !cfg!(all(
                target_vendor = "apple",
                not(feature = "cuda")
            ))
        {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let stream = StreamCopyPlan::<()>::capture(environment.stream())?;
        // The scalar/complete [1,N] source is checked and appended exactly
        // once. Its actual stream selects the existing CPU or GPU worker facts
        // before the separate copy account can admit the population.
        let source_clones = self.source_clones;
        let population = self.finish_resume_input_for(prepared, stream.device_type())?;
        let completed = Self {
            operands: 1,
            source_clones,
            logical_bytes: 0,
            maximum_rank: prepared.source().try_descriptor()?.facts().rank().max(2),
            host: HostPopulation::default(),
            host_stores: host_store::Population::default(),
        };
        completed
            .finish_population(
                environment,
                stream,
                population.layout(),
                population.logical_bytes(),
                population.births(),
            )?
            .ok_or(OriginalCopyCause::UnknownLayout)
    }

    fn finish_population<'a>(
        self,
        environment: &'a OriginalCopyEnvironment<'_>,
        stream: StreamCopyPlan<()>,
        layout: IsolatedCopyNativeLayout,
        logical_bytes: usize,
        births: usize,
    ) -> Result<Option<OriginalCopyPlan<'a>>, OriginalCopyCause> {
        let runtime = environment.input_runtime()?;
        let logical_bytes = logical_bytes
            .checked_add(self.host_stores.compaction)
            .ok_or(OriginalCopyCause::Overflow)?;
        let births = births
            .checked_add(self.host_stores.count)
            .ok_or(OriginalCopyCause::Overflow)?;
        let population =
            OriginalBufferBudget::population_layout(&runtime, logical_bytes, births)?;
        let buffer_bytes = population.capacity();
        let physical_bytes = buffer_bytes
            .checked_add(self.host_stores.backing)
            .ok_or(OriginalCopyCause::Overflow)?;
        let pipeline = (layout.kernel_attempts != 0)
            .then(|| PreparedPipelineCachePlan::new(layout.kernel_attempts));
        let pipeline_bytes = match pipeline {
            Some(plan) => plan
                .layout::<WorkspaceCopyRetention>()?
                .required_bytes()
                .ok_or(OriginalCopyCause::Overflow)?,
            None => 0,
        };
        let parts = [
            layout.controls,
            population.control_bytes(),
            PreparedSubmissionGraphQuota::<WorkspaceCopyRetention>::layout(layout.graph_capacity)?
                .total_bytes()
                .ok_or(OriginalCopyCause::Overflow)?,
            PreparedSubmissionRecordQuota::<WorkspaceCopyRetention>::layout(
                layout.record_capacity,
            )?
            .total_bytes()
            .ok_or(OriginalCopyCause::Overflow)?,
            PreparedOriginalBufferBudget::<WorkspaceCopyRetention>::layout(&runtime, buffer_bytes)?
                .total_owner_bytes()
                .ok_or(OriginalCopyCause::Overflow)?,
            PreparedPrefillFailure::<WorkspaceCopyRetention>::layout()?
                .total_bytes()
                .ok_or(OriginalCopyCause::Overflow)?,
            pipeline_bytes,
            safemlx::OriginalNativeControlLayout::inspect()?.fixed_control_bytes,
            OperationEvent::nested_completion_control_bytes::<1>()
                .ok_or(OriginalCopyCause::UnknownLayout)?,
            safemlx::original_scoped_evaluation_control_bytes()
                .ok_or(OriginalCopyCause::UnknownLayout)?,
            safemlx::original_scoped_deep_copy_control_bytes(self.maximum_rank)
                .ok_or(OriginalCopyCause::UnknownLayout)?,
            OriginalCopyEnvironment::control_bytes().ok_or(OriginalCopyCause::UnknownLayout)?,
            stream
                .control_bytes()
                .ok_or(OriginalCopyCause::UnknownLayout)?,
            Array::descriptor_control_bytes().ok_or(OriginalCopyCause::UnknownLayout)?,
            size_of::<Self>(),
            size_of::<super::NativeCopy<'_>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<OriginalCopyPlan<'_>>(),
            size_of::<PreparedOriginalCopy>(),
            size_of::<OriginalCopyExecution>(),
            size_of::<OriginalCopyCause>(),
            size_of::<OriginalCopyFailure>(),
            size_of::<Result<Option<OriginalCopyPlan<'_>>, OriginalCopyCause>>(),
            size_of::<Result<PreparedOriginalCopy, OriginalCopyCause>>(),
            size_of::<Result<PreparedOriginalCopy, OriginalCopyFailure>>(),
            size_of::<Option<OriginalScopeObserver>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalCopyFailure>()
                .ok_or(OriginalCopyCause::UnknownLayout)?,
        ];
        let host_bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(OriginalCopyCause::Overflow)?;
        Ok(Some(OriginalCopyPlan {
            stream: environment.stream(),
            pool: environment.pool(),
            allocator: environment.allocator(),
            layout,
            pipeline,
            physical_bytes,
            buffer_bytes,
            host_stores: self.host_stores,
            host_bytes,
        }))
    }
}

/// Closed population produced only by actual completed source descriptors.
/// It binds no scope or grant; the fresh request's recipe consumes it before
/// selecting shared Graph/Record/physical storage and validates the source pair.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OriginalResumeCopyPopulation {
    layout: IsolatedCopyNativeLayout,
    logical_bytes: usize,
    births: usize,
    roots: usize,
    pending_positions: u64,
}
impl OriginalResumeCopyPopulation {
    pub(crate) fn pending_positions(self) -> u64 {
        self.pending_positions
    }
    pub(crate) fn layout(self) -> IsolatedCopyNativeLayout {
        self.layout
    }
    pub(crate) fn logical_bytes(self) -> usize {
        self.logical_bytes
    }
    pub(crate) fn births(self) -> usize {
        self.births
    }
    pub(crate) fn roots(self) -> usize {
        self.roots
    }
}

pub(crate) struct OriginalCopyPlan<'a> {
    stream: &'a Stream,
    pool: &'a WorkingMemoryPool,
    allocator: &'static InitializedInputAllocator,
    layout: IsolatedCopyNativeLayout,
    pipeline: Option<PreparedPipelineCachePlan>,
    physical_bytes: usize,
    buffer_bytes: usize,
    host_stores: host_store::Population,
    host_bytes: usize,
}
impl OriginalCopyPlan<'_> {
    /// H only. The shared account Arc, collector, source/table/returned owners
    /// and publication producer are separate consumed components.
    pub(crate) fn control_bytes<R: Retention>(&self) -> Option<usize> {
        let recovery =
            usize::try_from(PreparedRecovery::<R, WorkspaceCopyRetention>::control_bytes()?)
                .ok()?;
        let parts = [
            recovery,
            size_of::<Result<(Recovery<R>, OriginalCopyExecution), OriginalCopyFailure>>(),
            size_of::<(Recovery<R>, OriginalCopyExecution)>(),
            size_of::<Result<(), OriginalCopyCause>>(),
        ];
        parts.into_iter().try_fold(
            self.host_bytes.checked_add(size_of_val(&parts))?,
            usize::checked_add,
        )
    }
    /// Whole physical native bound, not an additive B charge. Compose with the
    /// existing shared copy program as max(shared numerical B, this bound).
    pub(crate) fn physical_bytes(&self) -> usize {
        self.physical_bytes
    }
    pub(crate) fn stream(&self) -> &Stream {
        self.stream
    }

    /// Additional internal completion roots from the selected Host stores.
    /// They join the same final copy root collector, never another carrier.
    pub(crate) fn host_destination_roots(&self) -> usize {
        self.host_stores.count
    }

    /// All native storage is born after the existing copy admission. The
    /// initializer marker is the model's actual completed initialization witness;
    /// no ordinary runtime initialization is performed by this worker.
    pub(crate) fn prepare(
        self,
        custody: &WorkspaceCopyCustody,
        _initialized: &PrefillRootsRuntime,
    ) -> Result<PreparedOriginalCopy, OriginalCopyFailure> {
        let retained = custody.retention();
        let result = (|| -> Result<PreparedOriginalCopy, OriginalCopyCause> {
            if !custody.pool().same_domain(self.pool) {
                return Err(OriginalCopyCause::ForeignPool);
            }
            if u64::try_from(self.physical_bytes).map_err(|_| OriginalCopyCause::Overflow)?
                > custody.bytes()
            {
                return Err(OriginalCopyCause::Capacity);
            }
            let runtime = self
                .allocator
                .try_borrow_runtime()
                .map_err(OriginalCopyEnvironmentError::from)?;
            let graph =
                PreparedSubmissionGraphQuota::try_new(self.layout.graph_capacity, retained.clone())
                    .map_err(|e| e.into_parts().0)?
                    .try_allocate()
                    .map_err(|e| e.into_parts().0)?;
            let record = PreparedSubmissionRecordQuota::try_new(
                self.layout.record_capacity,
                retained.clone(),
            )
            .map_err(|e| e.into_parts().0)?
            .try_allocate()
            .map_err(|e| e.into_parts().0)?;
            let pipeline = self
                .pipeline
                .map(|plan| plan.realize(retained.clone()).map_err(|e| e.into_parts().0))
                .transpose()?;
            if let Some(pipeline) = &pipeline {
                pipeline.install(&graph)?;
            }
            let buffer = PreparedOriginalBufferBudget::try_new(
                &runtime,
                self.buffer_bytes,
                retained.clone(),
            )
            .map_err(|e| e.into_parts().0)?
            .try_allocate()
            .map_err(|e| e.into_parts().0)?;
            let failure = PreparedPrefillFailure::try_new(retained.clone())
                .map_err(|e| e.into_parts().0)?
                .try_allocate()
                .map_err(|e| e.into_parts().0)?;
            Ok(PreparedOriginalCopy {
                graph,
                record,
                buffer,
                failure,
                _pipeline: pipeline,
                layout: self.layout,
                custody: retained.clone(),
                host_stores: self.host_stores,
            })
        })();
        result.map_err(|cause| OriginalCopyFailure {
            cause,
            _custody: retained,
        })
    }
}

/// Completed constructors; no Scope or copy work has been accepted yet.
pub(crate) struct PreparedOriginalCopy {
    graph: SubmissionGraphQuota,
    record: SubmissionRecordQuota,
    buffer: OriginalBufferBudget,
    failure: RetainedPrefillFailure,
    _pipeline: Option<PreparedPipelineCache<WorkspaceCopyRetention>>,
    layout: IsolatedCopyNativeLayout,
    custody: WorkspaceCopyRetention,
    host_stores: host_store::Population,
}
impl PreparedOriginalCopy {
    /// Accounting-only alias for an escaped error. Native Scope and birth
    /// owners independently retain the actual thread-bound allocator budget.
    pub(crate) fn retention(&self) -> WorkspaceCopyRetention {
        self.custody.clone()
    }

    /// Same admitted budget before Scope entry, for preparing finite publication.
    pub(crate) fn budget(&self) -> &OriginalBufferBudget {
        &self.buffer
    }
    pub(crate) fn begin<R: Retention>(
        self,
        retention: R,
    ) -> Result<(Recovery<R>, OriginalCopyExecution), OriginalCopyFailure> {
        let custody = self.custody.clone();
        let result = (|| -> Result<(Recovery<R>, OriginalCopyExecution), OriginalCopyCause> {
            if !self.host_stores.empty() {
                return Err(OriginalCopyCause::UnknownLayout);
            }
            let pending = PreparedRecovery::new(retention, self.custody.clone())
                .map_err(|error| error.cause)?
                .with_record_quota(Some(self.record.clone()))
                .with_graph_quota(Some(self.graph.clone()));
            let mut recovery = pending.try_begin().map_err(|error| error.cause)?;
            recovery.configure_scope(|scope| -> Result<(), OriginalCopyCause> {
                scope
                    .enable_scoped_observation()
                    .map_err(OriginalCopyCause::Observation)?;
                scope.require_original_native_controls()?;
                self.failure.bind_original_scope(scope)?;
                scope.enable_original_native_controls()?;
                scope.bind_original_buffer_budget(&self.buffer)?;
                Ok(())
            })?;
            let observer =
                OriginalScopeObserver::try_current()?.ok_or(OriginalCopyCause::UnknownLayout)?;
            let mut bank = OperationEvent::prepare_resident_graph(self.layout.graph, &observer)?;
            bank.configure_nested_completions(
                &self.layout.traversal,
                self.layout.completion_attempts,
            )?;
            Ok((
                recovery,
                OriginalCopyExecution {
                    bank: Some(bank),
                    prepared: self,
                },
            ))
        })();
        result.map_err(|cause| OriginalCopyFailure {
            cause,
            _custody: custody,
        })
    }
}

/// Lives alongside the existing SessionOperation recovery. The caller finishes
/// construction before sealing that recovery; failure unwinding drops the bank
/// first. Consumed Graph blocks and all native births retain their own owners.
pub(crate) struct OriginalCopyExecution {
    bank: Option<PreparedResidentGraph>,
    prepared: PreparedOriginalCopy,
}
impl OriginalCopyExecution {
    pub(crate) fn finish_construction(&mut self) {
        drop(self.bank.take());
    }
    /// Actual mutable-copy birth witness for finite publication, no grant API.
    pub(crate) fn budget(&self) -> &OriginalBufferBudget {
        &self.prepared.buffer
    }
}
