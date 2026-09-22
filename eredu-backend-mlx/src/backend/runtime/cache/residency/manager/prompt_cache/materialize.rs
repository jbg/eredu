//! Authenticated fixed-state payloads through the shared native numerical worker.
use super::*;
use crate::backend::{
    Error,
    nn::workspace::{MlxMetalWorkspaceMechanisms, ResidentExecutionMechanisms, host_array},
    submission_recovery::native_role::physical,
};
use eredu_core::cache::SharedPromptCacheManifest;
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceContext, WorkspaceMetadataAllocation, WorkspaceMetadataError,
};
use eredu_runtime::{
    cache::{PersistentCacheStateTensor, PromptCachePersistenceFunding},
    working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError},
};
use safemlx::{ArrayElement, PreparedInputRuntime};
use std::{
    mem::{size_of, size_of_val},
    rc::Rc,
};

#[path = "materialize/staging.rs"]
mod staging;
use staging::Staging;
#[path = "materialize/publication.rs"]
mod publication;
#[cfg(test)]
#[path = "materialize/tests.rs"]
mod tests;

struct NativeSource {
    runtime: PreparedInputRuntime,
    ledger: MemoryLedger,
    execution: InferenceExecutionIdentity,
    context: WorkspaceContext,
    funding: HostMetadataFunding,
}
/// A retained allocator, ledger and executable loan, without a numerical bank.
/// Each authenticated tensor independently admits its actual constructor.
#[derive(Clone)]
pub(crate) struct PromptCacheMaterialization(Option<Rc<NativeSource>>);
impl Drop for PromptCacheMaterialization {
    fn drop(&mut self) {
        if let Some(body) = self.0.take() {
            // The final Rc shell retires before its payload releases funding.
            drop(Rc::into_inner(body));
        }
    }
}
impl PromptCacheMaterialization {
    fn body(&self) -> &NativeSource {
        self.0
            .as_deref()
            .expect("live cache materialization source")
    }
    pub(crate) fn new(
        ledger: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        runtime: PreparedInputRuntime,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let frames = [
            size_of::<Self>(),
            size_of::<NativeSource>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<(
                &MemoryLedger,
                &InferenceExecutionIdentity,
                PreparedInputRuntime,
                &WorkspaceContext,
            )>(),
            WorkspaceContext::metadata_source_bytes::<WorkingMemoryError>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ];
        context.charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        crate::backend::managed_memory::input_allocator::admitted_initializer(ledger)
            .map_err(|cause| context.metadata_source(cause))?;
        Ok(Self(Some(context.metadata_rc(NativeSource {
            runtime,
            ledger: ledger.clone(),
            execution: execution.clone(),
            context: context.clone(),
            funding,
        })?)))
    }
    pub(crate) fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        persistence: &PromptCachePersistenceFunding,
    ) -> Result<(), Error> {
        let body = self.body();
        body.context
            .charge_metadata(size_of::<(
                &Self,
                &InferenceExecutionIdentity,
                &PromptCachePersistenceFunding,
                Result<(), Error>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        if !body.execution.same_execution(execution)
            || !body.context.shares_trace(persistence.context())
            || persistence
                .context()
                .metadata_funding()
                .is_none_or(|funding| !funding.same_account(&body.funding))
        {
            return Err(Error::Neural(
                body.context
                    .metadata_source(WorkingMemoryError::IdentityMismatch),
            ));
        }
        Ok(())
    }
}

pub(crate) fn load_prompt_cache_state_tensors_funded(
    directory: &Path,
    manifest: &SharedPromptCacheManifest,
    stream: &Stream,
    funding: &PromptCachePersistenceFunding,
    materialization: &PromptCacheMaterialization,
) -> Result<Vec<LoadedPromptCacheStateTensor>, CacheResidencyError> {
    let context = funding.context();
    materialization
        .validate(&materialization.body().execution, funding)
        .map_err(|cause| failure(context, cause))?;
    let root = funding
        .resolve_root(directory)
        .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
    let frames = [
        size_of::<(
            &Path,
            &SharedPromptCacheManifest,
            &Stream,
            &PromptCachePersistenceFunding,
            &PromptCacheMaterialization,
        )>(),
        size_of::<PersistentCacheStateTensor<Staging<u8>>>(),
        size_of::<
            Result<
                PersistentCacheStateTensor<Staging<u8>>,
                eredu_runtime::cache::PersistentCacheReadFailure,
            >,
        >(),
        size_of::<File>(),
        size_of::<Result<File, std::io::Error>>(),
        size_of::<Vec<LoadedPromptCacheStateTensor>>(),
        size_of::<Result<Vec<LoadedPromptCacheStateTensor>, CacheResidencyError>>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| {
                    CacheResidencyError::Preparation(WorkspaceMetadataError::Overflow.into())
                })?,
        )
        .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
    let mut output = context.metadata_vec(manifest.state_tensors.len())?;
    for declaration in &manifest.state_tensors {
        let path = funding
            .shard_path(&root, &declaration.shard)
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let file = File::open(&path)
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let source =
            PersistentCacheStateTensor::read_with(file, &path, declaration, funding, |bytes| {
                Staging::new(materialization.body(), bytes, |_| 0)
            })?;
        use safetensors::tensor::Dtype as Stored;
        let array = match source.dtype() {
            Stored::F32 => construct::<f32, 4>(&source, materialization, stream, |bytes| {
                f32::from_le_bytes(bytes)
            }),
            Stored::F16 => construct::<half::f16, 2>(&source, materialization, stream, |bytes| {
                half::f16::from_bits(u16::from_le_bytes(bytes))
            }),
            Stored::BF16 => construct::<half::bf16, 2>(&source, materialization, stream, |bytes| {
                half::bf16::from_bits(u16::from_le_bytes(bytes))
            }),
            Stored::I32 => {
                construct::<i32, 4>(&source, materialization, stream, i32::from_le_bytes)
            }
            Stored::U32 => {
                construct::<u32, 4>(&source, materialization, stream, u32::from_le_bytes)
            }
            _ => Err(Error::OriginalSourceContract {
                stage: "fixed-state native constructor source",
                cause: WorkingMemoryError::UnknownBound,
            }),
        }
        .map_err(|cause| failure(context, cause))?;
        let array = publication::publish(array, materialization.body())
            .map_err(|cause| failure(context, cause))?;
        output.push(LoadedPromptCacheStateTensor {
            owner: declaration.owner,
            role: declaration.role,
            array,
        });
    }
    Ok(output)
}
fn failure(context: &WorkspaceContext, cause: Error) -> CacheResidencyError {
    CacheResidencyError::Preparation(context.metadata_source(cause.into_backend_failure()))
}
fn construct<T: StateScalar, const N: usize>(
    source: &PersistentCacheStateTensor<Staging<u8>>,
    materialization: &PromptCacheMaterialization,
    stream: &Stream,
    decode: impl Fn([u8; N]) -> T,
) -> Result<physical::CompletedNumerical<Array>, Error> {
    let body = materialization.body();
    let fail = || Error::Neural(WorkspaceMetadataError::Overflow.into());
    let frames = [
        size_of::<(
            &PersistentCacheStateTensor<Staging<u8>>,
            &PromptCacheMaterialization,
            &Stream,
        )>(),
        size_of::<Staging<T>>(),
        size_of::<[u8; N]>(),
        size_of::<std::slice::ChunksExact<'_, u8>>(),
        size_of::<MlxMetalWorkspaceMechanisms>(),
        size_of::<Result<MlxMetalWorkspaceMechanisms, eredu_nn::Error>>(),
        size_of::<Result<Array, Error>>(),
        size_of::<physical::CompletedNumerical<Array>>(),
        crate::backend::runtime::cache::completed_borrow_control_bytes().ok_or_else(fail)?,
    ];
    let controls = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or_else(fail)?;
    body.context
        .charge_metadata(controls)
        .map_err(|cause| Error::Neural(cause.into()))?;
    if size_of::<T>() != N || source.payload().len() % N != 0 {
        return Err(Error::Neural(
            body.context
                .metadata_source(WorkingMemoryError::IdentityMismatch),
        ));
    }
    let common = MlxMetalWorkspaceMechanisms::current_host()
        .map_err(Error::Neural)?
        .original_storage();
    let mechanism = ResidentExecutionMechanisms::from_stream(common, stream, &body.funding)
        .map_err(Error::Neural)?;
    let context = mechanism
        .context(body.funding.clone())
        .map_err(|cause| Error::Neural(cause.into()))?;
    let values = Staging::new(body, source.payload().len() / N, |index| {
        let begin = index * N;
        decode(
            source.payload()[begin..begin + N]
                .try_into()
                .expect("validated fixed width"),
        )
    })
    .map_err(Error::Neural)?;
    context.begin_span();
    let output = host_array::trace(source.shape(), T::DTYPE, &context).map_err(Error::Neural)?;
    context.complete_values(&[&output]).map_err(Error::Neural)?;
    let report = context.finish_report(&[output]).map_err(Error::Neural)?;
    let completed = physical::execute_numerical_with_runtime(
        &report,
        &[],
        1,
        mechanism,
        &context,
        &body.runtime,
        &body.ledger,
        &body.execution,
        0,
        || {
            let array = T::construct(values.as_ref(), source.shape())?;
            crate::backend::runtime::cache::complete_and_borrow(&array, stream)?;
            Ok(array)
        },
    )?;
    Ok(completed)
}

trait StateScalar: ArrayElement {
    fn construct(values: &[Self], shape: &[i32]) -> Result<Array, safemlx::error::Exception>;
}
macro_rules! scalar {
    ($($ty:ty),+) => { $(impl StateScalar for $ty {
        fn construct(values:&[Self], shape:&[i32])->Result<Array,safemlx::error::Exception> {
            Array::try_from_slice(values,shape)
        }
    })+ };
}
scalar!(f32, half::f16, half::bf16, i32, u32);
