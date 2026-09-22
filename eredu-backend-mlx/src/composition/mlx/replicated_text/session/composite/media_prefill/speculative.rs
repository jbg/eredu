//! Typed retained ingress installed before erasing the selected composite.
use super::*;
use crate::composition::mlx::replicated_text::{
    AutoregressiveSequenceCompletion, AutoregressiveStateRoots, OriginalAutoregressiveMediaPrefill,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataAllocation};
use eredu_runtime::working_memory::{
    OriginalSpeculativePrefillSpan, OriginalSpeculativeRole, WorkingMemoryError,
};

type Source<A> = eredu_runtime::media_prefill::PreparedMediaPrefill<
    PreparedCompositeArchitecture<A>,
    MlxNeuralBackend,
    MlxHybridState,
>;
pub(in crate::composition::mlx::replicated_text::session::composite) type Prepare<A, D> =
    fn(
        &NativeSession<A, D>,
        &input::OriginalMediaPacket,
        &eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        eredu_core::InferenceGeometry,
        &OriginalSpeculativeRole,
        &WorkspaceContext,
    ) -> Result<OriginalAutoregressiveMediaPrefill, Error>;
pub(in crate::composition::mlx::replicated_text::session::composite) type Run<A, D> =
    fn(
        &mut NativeSession<A, D>,
        &mut OriginalAutoregressiveMediaPrefill,
        &OriginalSpeculativePrefillSpan,
        &Stream,
        &mut dyn AutoregressiveSequenceCompletion,
        &mut dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>,
    ) -> Result<Option<MlxTensor>, Error>;

pub(super) fn prepare<A, D>(
    session: &NativeSession<A, D>,
    packet: &input::OriginalMediaPacket,
    semantics: &eredu_architectures::media_plan::BoundPreparedMediaSemantics,
    geometry: eredu_core::InferenceGeometry,
    role: &OriginalSpeculativeRole,
    metadata: &WorkspaceContext,
) -> Result<OriginalAutoregressiveMediaPrefill, Error>
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
    let plan = A::prepare_bound_original_ingress_plan_with_metadata(
        packet.clone_lowered(),
        semantics.clone(),
        geometry,
        metadata,
    )?;
    let source = session
        .prepare_speculative_media_prefill(plan, role, metadata)
        .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
    OriginalAutoregressiveMediaPrefill::new(
        source,
        |source, visitor| {
            source
                .downcast_ref::<Source<A>>()
                .expect("visitor installed with the typed media source")
                .visit_retained_roots(&mut |tensor| visitor(tensor.as_array()));
        },
        metadata,
    )
}

pub(super) fn run<A, D>(
    session: &mut NativeSession<A, D>,
    source: &mut OriginalAutoregressiveMediaPrefill,
    span: &OriginalSpeculativePrefillSpan,
    stream: &Stream,
    completion: &mut dyn AutoregressiveSequenceCompletion,
    observer: &mut dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>,
) -> Result<Option<MlxTensor>, Error>
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
    let funding = completion.metadata_funding();
    funding
        .reserve_metadata(std::mem::size_of::<(
            MlxPredictionTargetState,
            Result<MlxPredictionTargetState, Error>,
            super::super::super::mechanisms::StateCheckpoint<MlxHybridState>,
            Result<Option<MlxTensor>, SessionError>,
            Roots<'_>,
        )>())
        .map_err(Error::WorkspacePlanning)?;
    if !span.role().same_role(completion.role()) {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    let source = source
        .downcast_mut::<Source<A>>()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let mut checkpoint = completion.take_checkpoint()?;
    if !checkpoint.is::<MlxHybridState>() {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    let checkpoint = super::super::super::mechanisms::StateCheckpoint::independent(
        checkpoint.take_state::<MlxHybridState>()?,
    );
    session
        .prefill_media_span_with_checkpoint_completion_and_observer(
            source,
            span,
            stream,
            checkpoint,
            &funding,
            |output, state, media, stream| {
                completion.complete(
                    output.map(MlxTensor::as_array),
                    &Roots { state, media },
                    stream,
                )
            },
            observer,
        )
        .map_err(|cause| Error::Neural(funding.metadata_source(cause)))
}
struct Roots<'a> {
    state: &'a MlxHybridState,
    media: &'a eredu_runtime::media_prefill::RetainedMediaRoots<'a, MlxTensor>,
}
impl AutoregressiveStateRoots for Roots<'_> {
    fn visit_roots(&self, visitor: &mut dyn FnMut(&Array)) -> Result<(), Error> {
        self.state.visit_roots(visitor)?;
        self.media.visit(&mut |tensor| visitor(tensor.as_array()));
        Ok(())
    }
}
