//! Portable erasure and invocation of a typed retained prefill session.

use eredu_checkpoint::quantization::WeightMaterializationReport;
use eredu_core::{
    capture::CaptureRequestShape,
    communication::CommunicationAuthority,
    input::PreparedInputInspector,
    observation::ObserverTensorMapping,
    residency::{
        ParameterConversionRetentionReport, ParameterConversionRetentionTrimReport,
        ParameterConversionTrimError,
    },
    BackendFailure, BackendFailureKind, CapabilityError, CollectiveGroupId, Observed,
    SessionCapabilities,
};

use crate::{
    prefill::ErasedPrefillCursor,
    text_session_driver::{TextExecutable, TextPromptView, TextSnapshotStorage},
    ActivationObserver, CacheResidencyReport, CapabilityEstimate, DenseDiskStreamReport,
    ResidencyReport,
};
use eredu_core::PreparedModelInput;

use super::{
    erased_text::{ErasedTextExecutable, InstalledTextExecutable},
    retained_session::{
        PreparedInputPartPlan, PreparedRetainedTextSession, RetainedTextBindingError,
    },
    PreparedTextSessionFacts,
};
use eredu_core::backend::TextExecutableTypes;

/// Native conversions used by portable text invocation.
pub trait TextTensorMechanisms: TextExecutableTypes {
    /// Tensor metadata for the decoder ingress representation.
    type TensorInputInspector: PreparedInputInspector<Self::Tensor>;
    /// Converts decoder observation handles for the selected invocation.
    type TensorObserver: ObserverTensorMapping<
        InvocationTensor = Self::Tensor,
        ObserverTensor = Self::InvocationTensor,
        InvocationError = eredu_nn::Error,
        ObserverError = Self::Error,
    >;

    /// Converts already prepared tensor handles for decoder ingress.
    fn prepared_input(
        input: TextPromptView<'_, Self::InputTensor>,
    ) -> Result<PreparedModelInput<Self::Tensor>, Self::Error>;

    /// Supplies side-effect-free identity inspection for ingress.
    fn tensor_input_inspector() -> Self::TensorInputInspector;
    /// Converts a prepared decode tensor into the decoder representation.
    fn decode_tensor(tokens: &Self::Decode) -> Self::Tensor;
    /// Converts scores to the submission's native representation.
    fn scores(scores: Self::Tensor) -> Self::Scores;
}

/// Native state and communication observations needed by retained execution.
pub trait RetainedTextMechanisms:
    TextTensorMechanisms<Tensor: eredu_nn::Tensor<Context = Self::Context>>
{
    /// State backing the installed decoder.
    type State: TextSnapshotStorage<Self::Context> + 'static;
    /// Exact realized communication handle.
    type Communication: 'static;
    /// Borrows a realized collective by its exact identity.
    fn bound_group(
        communication: &Self::Communication,
        group: CollectiveGroupId,
    ) -> Option<&Self::Group>;
    /// Borrows the authority shared by the retained communication handle.
    fn authority(communication: &Self::Communication) -> &CommunicationAuthority;
    /// Reads the installed cache offset.
    fn state_offset(state: &Self::State) -> i32;
    /// Reads native cache residency without advancing the model.
    fn cache_residency(
        state: &Self::State,
    ) -> Result<Option<CacheResidencyReport>, Self::StateError>;
}

struct RetainedExecutable<R, M: RetainedTextMechanisms> {
    session: R,
    communication: Option<M::Communication>,
    stream: M::Context,
    facts: PreparedTextSessionFacts,
    publishes_scores: bool,
}

/// Installs and erases a retained session with the selected native resources.
pub fn install<M, R>(
    session: R,
    communication: Option<M::Communication>,
    stream: M::Context,
) -> Result<InstalledTextExecutable<M>, RetainedTextBindingError>
where
    M: RetainedTextMechanisms,
    M::Communication: 'static,
    M::Context: 'static,
    R: PreparedRetainedTextSession<M::Tensor, M::State> + 'static,
{
    let publication = session.publication();
    if communication.is_some() != publication.is_some() {
        return Err(RetainedTextBindingError::Publication);
    }
    let admitted = session.session_capabilities();
    let realized = SessionCapabilities::new(false, admitted.output_observation(), false);
    eredu_core::SessionAdmission::new(admitted).validate(realized)?;
    let facts = PreparedTextSessionFacts::from_parts(
        session.capabilities().clone(),
        session
            .capabilities()
            .capabilities()
            .effective_model_type
            .clone(),
        session.decoder_residency(),
    );
    Ok(super::erased_text::erase_text_executable(
        RetainedExecutable::<R, M> {
            session,
            communication,
            stream,
            facts,
            publishes_scores: publication
                .is_none_or(|(_, authority)| authority.local_public_output()),
        },
    ))
}

fn unsupported(feature: &str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::Unsupported,
        std::io::Error::other(format!("retained executable does not support {feature}")),
    )
}

impl<R, M> RetainedExecutable<R, M>
where
    M: RetainedTextMechanisms,
    R: PreparedRetainedTextSession<M::Tensor, M::State>,
{
    fn start_retained_prefill(
        &mut self,
        input: Result<TextPromptView<'_, M::InputTensor>, M::Error>,
        max_tokens: usize,
        geometry: Option<CaptureRequestShape>,
        stream: &M::Context,
    ) -> Result<R::Cursor, M::Error> {
        let identity = input
            .as_ref()
            .ok()
            .and_then(|input| input.cache_identity())
            .cloned();
        let prepared = input
            .and_then(M::prepared_input)
            .map_err(BackendFailure::from_error);
        self.session
            .start_prefill(
                prepared,
                &M::tensor_input_inspector(),
                identity.as_ref(),
                max_tokens,
                geometry,
                stream,
            )
            .map_err(|error| {
                if self.communication.is_some() {
                    error.into()
                } else {
                    error.with_model_state_preserved(true).into()
                }
            })
    }
}

impl<R, M> ErasedTextExecutable<M> for RetainedExecutable<R, M>
where
    M: RetainedTextMechanisms,
    R: PreparedRetainedTextSession<M::Tensor, M::State>,
{
    fn realized_session_capabilities(&self) -> SessionCapabilities {
        self.session.session_capabilities()
    }

    fn forecast_state_offset(&self) -> Result<Option<i32>, M::StateError> {
        Ok(Some(M::state_offset(self.session.state().get())))
    }

    fn prepared_input_token_ids(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
        _: &M::Context,
    ) -> Result<Vec<u32>, M::Error> {
        let prepared = M::prepared_input(input)?;
        self.session
            .input_tokens(
                &prepared,
                &M::tensor_input_inspector(),
                input.cache_identity(),
            )
            .map_err(M::Error::from)
    }

    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &M::Group,
        &eredu_core::communication::CommunicationAuthority,
        &M::Context,
        usize,
    )> {
        let (publication, authority) = self.session.publication()?;
        let communication = self.communication.as_ref()?;
        Some((
            M::bound_group(communication, publication.group)?,
            M::authority(communication),
            &self.stream,
            authority.owner_group_rank(),
        ))
    }

    fn control_state(
        &self,
    ) -> Option<&dyn crate::text_session_driver::ErasedTextStateAccess<M::Context>> {
        Some(self.session.state())
    }

    fn control_state_mut(
        &mut self,
    ) -> Option<(
        &mut dyn crate::text_session_driver::ErasedTextStateAccess<M::Context>,
        &M::Context,
    )> {
        Some((self.session.state_mut(), &self.stream))
    }

    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.session.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        Ok(match self.session.residency()? {
            Some(report) => report.offload().parameter_conversion_retention().clone(),
            None => Observed::Unsupported {
                reason: "selected session does not select conversion retention".into(),
            },
        })
    }

    fn component_residency_reports(&self) -> Result<Vec<ResidencyReport>, M::Error> {
        self.session.component_residency().map_err(M::Error::from)
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, M::Error> {
        self.session.residency().map_err(M::Error::from)
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, M::Error> {
        self.session.dense_stream().map_err(M::Error::from)
    }

    fn materialization_report(&self) -> Option<&WeightMaterializationReport> {
        None
    }

    fn parameter_bank_report(
        &self,
    ) -> Result<Option<crate::expert::ParameterBanksResidencyReport>, M::Error> {
        Ok(None)
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, M::StateError> {
        M::cache_residency(self.session.state().get())
    }
}

impl<R, M> TextExecutable<M> for RetainedExecutable<R, M>
where
    M: RetainedTextMechanisms,
    R: PreparedRetainedTextSession<M::Tensor, M::State>,
{
    type Decode = M::Decode;
    type ObservationTensor = M::Tensor;
    type ObserverMapping = M::ObserverMapping;
    type Scores = M::Scores;
    type Context = M::Context;
    type Error = M::Error;
    type AttentionState = M::PredictionState;

    fn prepared_input_plans(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
    ) -> Result<Vec<PreparedInputPartPlan>, CapabilityError> {
        let invalid = |error: M::Error| CapabilityError::UnsupportedInput {
            architecture: self.effective_model_type().into(),
            reason: error.to_string(),
        };
        let prepared = M::prepared_input(input).map_err(invalid)?;
        self.session
            .input_plans(&prepared, &M::tensor_input_inspector())
    }

    fn prepared_input_positions(&self, input: TextPromptView<'_, M::InputTensor>) -> Option<u64> {
        self.prepared_input_plans(input)
            .ok()?
            .iter()
            .try_fold(0_u64, |total, part| {
                total.checked_add(part.decoder_positions())
            })
    }

    fn effective_model_type(&self) -> &str {
        self.facts.effective_model_type()
    }

    fn capability_estimate(&self) -> &CapabilityEstimate {
        self.facts.capability()
    }

    fn accepts_split_prefill(&self) -> bool {
        true
    }

    fn retains_prefill_cursor(&self) -> bool {
        true
    }

    fn start_prefill_cursor(
        &mut self,
        input: TextPromptView<'_, M::InputTensor>,
        max_tokens: std::num::NonZeroUsize,
        geometry: Option<CaptureRequestShape>,
        stream: &M::Context,
    ) -> Result<Option<ErasedPrefillCursor>, M::Error> {
        self.start_retained_prefill(Ok(input), max_tokens.get(), geometry, stream)
            .map(|cursor| Some(ErasedPrefillCursor::new(cursor)))
    }

    fn advance_prefill_ingress(
        &mut self,
        cursor: &mut ErasedPrefillCursor,
        stream: &M::Context,
        observer: &mut dyn ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<(), M::Error> {
        self.session
            .advance_retained_ingress(
                cursor,
                stream,
                &mut crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(
                    observer,
                ),
            )
            .map_err(M::Error::from)
    }

    fn advance_prefill_cursor(
        &mut self,
        cursor: &mut ErasedPrefillCursor,
        stream: &M::Context,
        observer: &mut dyn ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        self.session
            .advance_retained_decoder(
                cursor,
                stream,
                &mut crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(
                    observer,
                ),
            )
            .map(M::scores)
            .map_err(M::Error::from)
    }

    fn prefill_result_with_observer(
        &mut self,
        input: Result<TextPromptView<'_, M::InputTensor>, M::Error>,
        mask: Option<&M::InvocationTensor>,
        geometry: Option<CaptureRequestShape>,
        stream: &M::Context,
        observer: &mut dyn ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        if mask.is_some() {
            return Err(unsupported("explicit prefill mask")
                .with_model_state_preserved(true)
                .into());
        }
        let mut cursor = self.start_retained_prefill(input, usize::MAX, geometry, stream)?;
        self.session
            .complete_prefill(
                &mut cursor,
                stream,
                &mut crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(
                    observer,
                ),
            )
            .map(M::scores)
            .map_err(M::Error::from)
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&M::Decode, M::Error>,
        stream: &M::Context,
        observer: &mut dyn ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        self.session
            .decode(
                tokens
                    .map(M::decode_tensor)
                    .map_err(BackendFailure::from_error),
                stream,
                &mut crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(
                    observer,
                ),
            )
            .map(M::scores)
            .map_err(M::Error::from)
    }

    fn has_partition_control(&self) -> bool {
        self.communication.is_some()
    }

    fn partition_public_output(&self) -> bool {
        self.publishes_scores
    }

    fn reset_local(&mut self) -> Result<(), M::Error> {
        *self.session.state_mut().get_mut() = self.session.fresh_state()?;
        Ok(())
    }

    fn reset_partitioned(&mut self) -> Result<(), M::Error> {
        self.session.reset(&self.stream).map_err(M::Error::from)
    }
}
