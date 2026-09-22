//! Typed optional entry retained before the selected composite is erased.
use super::*;
pub(super) mod speculative;
use eredu_architectures::composite_execution::CompositeMediaIngressArchitecture;
use eredu_runtime::media_prefill::MediaTextExecutionStrategy;

type NativeSession<A, D> = ReplicatedTextSession<
    PreparedCompositeArchitecture<A>,
    MlxNeuralBackend,
    MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
    D,
>;
type PreparedInput<A> = Result<
    (
        eredu_runtime::PreparedModelInput<MlxTensor>,
        eredu_architectures::media_plan::AdmittedCompositeInput<
            <A as CompositeArchitecture<MlxNeuralBackend, MlxHybridState>>::InputPartPlan,
        >,
        Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    ),
    Error,
>;
pub(super) type NativeMediaCaptureValidation<A> = fn(
    &<A as CompositeArchitecture<MlxNeuralBackend, MlxHybridState>>::AdmissionConfig,
    PreparedInput<A>,
    eredu_core::InferenceGeometry,
) -> Result<(), Error>;

type SessionError = eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>;

impl<A, D, P> CompletedComposite<A, D, P>
where
    A: CompositeMediaIngressArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::InputPartPlan: 'static,
    A::IngressPlan: 'static,
    A::Ingress: 'static,
    D: MediaTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    pub(in crate::composition::mlx::replicated_text::session) fn with_media_prefill(
        mut self,
    ) -> Self {
        self.speculative_media_prepare = Some(speculative::prepare::<A, D>);
        self.speculative_media_run = Some(speculative::run::<A, D>);
        self.bind_original_media = Some(A::bind_original_media_semantics);
        self.original_media_prefill = Some(run_original::<A, D>);
        self.media_capture_validation = Some(validate::<A>);
        self
    }
}

fn validate<A>(
    admission: &A::AdmissionConfig,
    prepared: PreparedInput<A>,
    geometry: eredu_core::InferenceGeometry,
) -> Result<(), Error>
where
    A: CompositeMediaIngressArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::InputPartPlan: 'static,
{
    let (input, _, _) = prepared?;
    // This constructs only the architecture's host plan. No encoder is executed.
    // Source-specific masks, freshness and retained-media semantics are checked
    // by the same total constructor later consumed by the shared driver.
    let plan = A::prepare_ingress_plan(admission, input, &input::MlxTensorInputInspector, geometry)
        .map_err(|error| Error::Other(Box::new(error)))?;
    drop(plan);
    Ok(())
}

pub(super) type NativeOriginalMediaBinding<A> = fn(
    &<A as CompositeArchitecture<MlxNeuralBackend, MlxHybridState>>::AdmissionConfig,
    eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
    &eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
    &eredu_runtime::working_memory::OriginalPreparedHostInput,
    eredu_runtime::working_memory::MediaSessionBinding,
) -> Result<
    eredu_architectures::media_plan::BoundPreparedMediaSemantics,
    eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
>;
pub(super) type NativeOriginalMediaPrefillEntry<A, D> = fn(
    &mut NativeSession<A, D>,
    &input::OriginalMediaPacket,
    Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    Option<&eredu_runtime::working_memory::InferenceRequest>,
    Option<&eredu_nn::workspace::WorkspaceContext>,
    Option<&eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
    Option<std::num::NonZeroU64>,
    &eredu_core::GenerationCancellationToken,
    &Stream,
    &mut dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>,
) -> Result<
    eredu_runtime::replicated_session::PrefillSourceProgress<MlxTensor>,
    SessionError,
>;
fn run_original<A, D>(
    session: &mut NativeSession<A, D>,
    packet: &input::OriginalMediaPacket,
    identity: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    request: Option<&eredu_runtime::working_memory::InferenceRequest>,
    metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    copied_semantics: Option<&eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
    chunk: Option<std::num::NonZeroU64>,
    cancellation: &eredu_core::GenerationCancellationToken,
    stream: &Stream,
    observer: &mut dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>,
) -> Result<eredu_runtime::replicated_session::PrefillSourceProgress<MlxTensor>, SessionError>
where
    A: CompositeMediaIngressArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::InputPartPlan: 'static,
    D: MediaTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    let metadata = metadata.ok_or(SessionError::WorkingMemory(
        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    ))?;
    let make_plan = |geometry| {
        A::prepare_bound_original_ingress_plan_with_metadata(
            packet.clone_lowered(),
            copied_semantics
                .cloned()
                .unwrap_or_else(|| packet.semantics()),
            geometry,
            metadata,
        )
    };
    {
        // Pay the error's closed transport before constructing a source or plan.
        let funding = metadata
            .metadata_funding()
            .ok_or(SessionError::WorkingMemory(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))?;
        let bytes = eredu_nn::Error::retained_source_construction_bytes::<MediaExecutionFailure>()
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<(
                    MediaExecutionFailure,
                    Result<(), SessionError>,
                    Option<&eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
                )>())
            })
            .ok_or(SessionError::WorkingMemory(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        metadata
            .charge_metadata(bytes)
            .map_err(|cause| SessionError::Architecture(cause.into()))?;
        // Only the accepted source-bound prompt supplies this existing ledger.
        session
            .try_prefill_media_source_with_metadata(
                request,
                Some(packet.shape()),
                identity,
                chunk,
                make_plan,
                metadata,
                cancellation,
                stream,
                observer,
            )
            .map_err(|cause| {
                let retained = |cause| {
                    SessionError::Architecture(eredu_nn::Error::backend_retained_source(
                        MediaExecutionFailure {
                            cause,
                            _funding: funding,
                        },
                    ))
                };
                // Reuse the existing before-mutation Box and preserve its classification.
                match cause {
                    SessionError::BeforeStateMutation(mut source) => {
                        let cause = std::mem::replace(
                            &mut *source,
                            SessionError::WorkingMemory(
                                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                            ),
                        );
                        *source = retained(cause);
                        SessionError::BeforeStateMutation(source)
                    }
                    cause => retained(cause),
                }
            })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("original media binding: {cause}")]
pub(super) struct OriginalBindingFailure {
    #[source]
    pub(super) cause: Error,
    pub(super) original: eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct MediaExecutionFailure {
    #[source]
    cause: SessionError,
    // Account only: no native source/completion backedge and no Context Rc.
    _funding: eredu_nn::workspace::HostMetadataFunding,
}
