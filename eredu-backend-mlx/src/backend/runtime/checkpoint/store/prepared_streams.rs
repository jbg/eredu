//! Two explicitly immutable wrapper copies through the existing source account.
use eredu_runtime::working_memory::{
    InitializedSharedNative, MemoryLedger, OriginalHostMetadataCustody,
    SharedNativeInitializationCustody, SharedNativeInitializationError, SharedNativeInitializer,
    WorkingMemoryError,
};
use safemlx::{PreparedStreamCopy, Stream, StreamCopyCause, StreamCopyError, StreamCopyPlan};
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct Initializer(StreamCopyPlan<SharedNativeInitializationCustody>);
impl SharedNativeInitializer for Initializer {
    type Output = PreparedStreamCopy<SharedNativeInitializationCustody>;
    type Error = StreamCopyError<SharedNativeInitializationCustody>;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        // The owning safe module supplies actual Body/OwnedNode layouts; the
        // neutral host producer qualifies Arc/Box requested extents. Neither
        // layer infers these from observed capacity or the available budget.
        let shared =
            OriginalHostMetadataCustody::shared_storage_bytes(self.0.shared_body_layout())?;
        let node = OriginalHostMetadataCustody::boxed_storage_bytes(self.0.owner_node_layout())?;
        let controls = [
            size_of::<&MemoryLedger>(),
            size_of::<StreamCopyPlan<SharedNativeInitializationCustody>>(),
            size_of::<
                Result<
                    InitializedSharedNative<PreparedStreamCopy<SharedNativeInitializationCustody>>,
                    PreparedMaterializationStreamError,
                >,
            >(),
            size_of::<
                Result<
                    PreparedStreamCopy<SharedNativeInitializationCustody>,
                    PreparedMaterializationStreamError,
                >,
            >(),
            size_of::<PreparedMaterializationStreams>(),
            size_of::<PreparedMaterializationStreamError>(),
            size_of::<crate::backend::runtime::residency::manager::ResidencyError>(),
            size_of::<
                Result<
                    crate::backend::runtime::residency::manager::ResidencyManager,
                    crate::backend::runtime::residency::manager::ResidencyError,
                >,
            >(),
            size_of::<Option<PreparedMaterializationStreams>>(),
            size_of::<Result<PreparedMaterializationStreams, PreparedMaterializationStreamError>>(),
        ];
        let allocation = usize::try_from(
            shared
                .checked_add(node)
                .ok_or(WorkingMemoryError::Overflow)?,
        )
        .map_err(|_| WorkingMemoryError::Overflow)?;
        controls
            .into_iter()
            .try_fold(allocation, usize::checked_add)
            .and_then(|n| n.checked_add(size_of_val(&controls)))
            .and_then(|n| n.checked_add(self.0.native_wrapper_bytes()))
            .and_then(|n| n.checked_add(self.0.control_bytes()?))
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        self.0.realize(custody)
    }
}
/// Typed stream-copy refusal retaining every actual failed constructor owner.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct PreparedMaterializationStreamError(Failure);
#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("materialization stream accounting: {0}")]
    Accounting(#[source] WorkingMemoryError),
    #[error("materialization stream recipe: {0}")]
    Layout(#[source] StreamCopyCause),
    #[error("materialization stream admission: {0}")]
    Initialization(#[source] SharedNativeInitializationError<Initializer>),
}
impl PreparedMaterializationStreamError {
    fn layout(value: StreamCopyCause) -> Self {
        Self(Failure::Layout(value))
    }
    fn initialization(value: SharedNativeInitializationError<Initializer>) -> Self {
        Self(Failure::Initialization(value))
    }
}
/// The same admitted immutable-copy producer, for a single shared mechanism
/// stream. No ordinary Stream clone or independent wrapper protocol is added.
pub(crate) fn prepare_one_materialization_stream(
    pool: &MemoryLedger,
    source: &Stream,
) -> Result<PreparedStreamCopy<SharedNativeInitializationCustody>, PreparedMaterializationStreamError>
{
    let plan =
        StreamCopyPlan::capture(source).map_err(PreparedMaterializationStreamError::layout)?;
    prepare_materialization_stream_from_plan(pool, plan).map(|owned| owned.output().clone())
}
/// The same counted immutable-wrapper constructor with an already captured
/// exact context plan. Selection cannot create a new queue/worker or grant.
pub(crate) fn prepare_materialization_stream_from_plan(
    pool: &MemoryLedger,
    plan: StreamCopyPlan<SharedNativeInitializationCustody>,
) -> Result<
    InitializedSharedNative<PreparedStreamCopy<SharedNativeInitializationCustody>>,
    PreparedMaterializationStreamError,
> {
    pool.initialize_shared_native(Initializer(plan))
        .map_err(PreparedMaterializationStreamError::initialization)
}

/// Private original constructor output. Public ordinary materialization contexts
/// continue owning independent mutable Stream wrappers.
#[derive(Debug, Clone)]
pub(crate) struct PreparedMaterializationStreams {
    source: PreparedStreamCopy<SharedNativeInitializationCustody>,
    execution: PreparedStreamCopy<SharedNativeInitializationCustody>,
}
impl PreparedMaterializationStreams {
    pub(crate) fn required_bytes(
        source: &Stream,
        execution: &Stream,
    ) -> Result<u64, PreparedMaterializationStreamError> {
        let source = Initializer(
            StreamCopyPlan::capture(source).map_err(PreparedMaterializationStreamError::layout)?,
        );
        let execution = Initializer(
            StreamCopyPlan::capture(execution)
                .map_err(PreparedMaterializationStreamError::layout)?,
        );
        let query = || {
            MemoryLedger::shared_native_initialization_required_bytes(&source)?
                .checked_add(MemoryLedger::shared_native_initialization_required_bytes(
                    &execution,
                )?)
                .ok_or(WorkingMemoryError::Overflow)
        };
        query().map_err(|cause| PreparedMaterializationStreamError(Failure::Accounting(cause)))
    }
    pub(crate) fn prepare(
        pool: &MemoryLedger,
        source: &Stream,
        execution: &Stream,
    ) -> Result<Self, PreparedMaterializationStreamError> {
        let source = Initializer(
            StreamCopyPlan::capture(source).map_err(PreparedMaterializationStreamError::layout)?,
        );
        let execution = Initializer(
            StreamCopyPlan::capture(execution)
                .map_err(PreparedMaterializationStreamError::layout)?,
        );
        // Each exact request has its own disjoint comparison. A failed second
        // request retires the unused first output; no context was published.
        let source = pool
            .initialize_shared_native(source)
            .map_err(PreparedMaterializationStreamError::initialization)?;
        let execution = pool
            .initialize_shared_native(execution)
            .map_err(PreparedMaterializationStreamError::initialization)?;
        Ok(Self {
            source: source.output().clone(),
            execution: execution.output().clone(),
        })
    }
    pub(crate) fn source_stream(&self) -> &Stream {
        self.source.as_stream()
    }
    pub(crate) fn execution_stream(&self) -> &Stream {
        self.execution.as_stream()
    }
    pub(crate) fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        self.source.owner().validate_pool(pool)?;
        self.execution.owner().validate_pool(pool)
    }
}

/// Private manager-only context; no public constructor/getter exposes the new
/// immutable wrapper through the old public context/raw-interop API.
pub(crate) struct ManagerMaterializationContext(ContextStorage);
enum ContextStorage {
    Ordinary(super::MlxParameterMaterializationContext),
    Prepared {
        streams: PreparedMaterializationStreams,
        cache: super::CacheHandle,
    },
}
impl ManagerMaterializationContext {
    pub(crate) fn ordinary(context: super::MlxParameterMaterializationContext) -> Self {
        Self(ContextStorage::Ordinary(context))
    }
    pub(crate) fn prepared(
        streams: PreparedMaterializationStreams,
        cache: super::CacheHandle,
        pool: &MemoryLedger,
    ) -> Result<Self, WorkingMemoryError> {
        streams.validate_pool(pool)?;
        cache.validate_pool(pool)?;
        Ok(Self(ContextStorage::Prepared { streams, cache }))
    }
    pub(crate) fn view(&self) -> super::MaterializationView<'_> {
        match &self.0 {
            ContextStorage::Ordinary(context) => context.into(),
            ContextStorage::Prepared { streams, cache } => super::MaterializationView {
                source: streams.source_stream(),
                execution: streams.execution_stream(),
                converted_groups: cache,
            },
        }
    }
    pub(crate) fn cache_handle(&self) -> super::CacheHandle {
        self.view().converted_groups.clone()
    }
    pub(crate) fn matches_cache(&self, other: &super::CacheHandle) -> bool {
        self.view().converted_groups.same(other)
    }
}

#[cfg(test)]
mod tests;
