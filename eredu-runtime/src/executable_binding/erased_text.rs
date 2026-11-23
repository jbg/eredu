//! Installed text executable and the backend representations carried by its operations.

use eredu_core::backend::TextExecutableTypes;
use eredu_core::{
    speculative::erasure::ErasedGenerationExecutor, BackendFailure, BackendFailureKind,
};

use crate::{
    speculative::{
        ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation,
    },
    text_session_driver::{TextExecutable, TextPromptView},
    CacheResidencyReport, DenseDiskStreamReport, ResidencyReport,
};
use eredu_core::PreparedInputPayload;

/// Owning handle to a complete, installed text executable.
pub type InstalledTextExecutable<M> = Box<dyn ErasedTextExecutable<M>>;

/// Erases an already constructed executable at the portable binding boundary.
pub(super) fn erase_text_executable<M, E>(executable: E) -> InstalledTextExecutable<M>
where
    M: TextExecutableTypes,
    E: ErasedTextExecutable<M> + 'static,
{
    Box::new(executable)
}

fn unsupported(message: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::Unsupported,
        std::io::Error::new(std::io::ErrorKind::Unsupported, message),
    )
}

#[doc(hidden)]
pub trait ErasedExternalPredictionExecutable<M: TextExecutableTypes>: 'static {
    /// Observes the installed lane and its parameter owners without evaluating work.
    fn continuation_memory_observation(
        &self,
        state: &M::PredictionState,
        additional: u64,
    ) -> Result<
        eredu_core::speculative::SpeculativeModelMemoryObservation,
        eredu_core::BackendFailure,
    >;

    /// Exposes the portable placement cache for the installed target.
    fn target_placements(&mut self) -> &mut crate::speculative::ExternalTargetPlacements;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    >;

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    >;
    fn prepare_external_prediction_target_cache(&mut self) -> Result<M::PredictionState, M::Error>;
    fn capture_external_prediction_target(
        &mut self,
        input: Result<TextPromptView<'_, M::InputTensor>, M::Error>,
        pass: crate::ExpertPass,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut M::PredictionState,
    ) -> Result<(M::Tensor, ExternalPredictionTargetCapture<M::Tensor>), M::Error>;
    fn apply_external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, M::Tensor>,
    ) -> Result<M::Tensor, M::Error>;
}

pub(crate) fn text_token_ids<M: TextExecutableTypes>(
    input: TextPromptView<'_, M::InputTensor>,
    context: &M::Context,
) -> Result<Vec<u32>, M::Error> {
    let mut ids = Vec::new();
    for part in input.parts {
        let PreparedInputPayload::TokenIds(tokens) = part.payload() else {
            return Err(unsupported(
                "text executable requires token-ID input for decoder alignment",
            )
            .into());
        };
        ids.extend(M::read_token_ids(tokens, context)?);
    }
    Ok(ids)
}

pub(crate) fn text_input_plans<M: TextExecutableTypes>(
    model_type: &str,
    input: TextPromptView<'_, M::InputTensor>,
) -> Result<Vec<super::retained_session::PreparedInputPartPlan>, eredu_core::CapabilityError> {
    input
        .parts
        .iter()
        .map(|part| {
            crate::executable_binding::input_admission::text_only_input_part(
                model_type,
                part,
                &M::input_inspector(),
            )
        })
        .collect()
}

/// Erased operations of an installed text executable, retained by the portable driver.
#[allow(missing_docs)]
pub trait ErasedTextExecutable<M: TextExecutableTypes>:
    TextExecutable<
    M,
    Decode = M::Decode,
    ObservationTensor = M::Tensor,
    ObserverMapping = M::ObserverMapping,
    Scores = M::Scores,
    Context = M::Context,
    Error = M::Error,
    AttentionState = M::PredictionState,
>
{
    fn realized_session_capabilities(&self) -> eredu_core::SessionCapabilities;

    fn forecast_state_offset(&self) -> Result<Option<i32>, M::StateError> {
        Ok(None)
    }

    fn prepared_input_token_ids(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
        context: &M::Context,
    ) -> Result<Vec<u32>, M::Error> {
        text_token_ids::<M>(input, context)
    }

    fn partition_observation_hooks(&self) -> Option<crate::inspection::ObservationHookSupport> {
        None
    }

    fn partition_parameter_description(
        &self,
    ) -> Option<&std::sync::Arc<crate::ArchitectureParameterDescription>> {
        None
    }

    fn visit_loaded_parameters(
        &mut self,
        _visitor: &mut dyn eredu_nn::ParameterSlotVisitor<M::Tensor>,
    ) -> bool {
        false
    }

    fn parameter_materialization_tasks(&self) -> &[crate::ReplicatedTextMaterializationTask] {
        &[]
    }

    fn prepared_parameter_slots(&self) -> &[crate::parameter_operations::PreparedParameterSlot] {
        &[]
    }

    fn with_parameter_slots(
        &mut self,
        _location: &crate::parameter_operations::PreparedParameterLocation,
        _ids: &std::collections::BTreeSet<String>,
        _operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            M::Tensor,
            M::Error,
        >,
        _stream: &M::Context,
    ) -> Result<bool, M::Error> {
        Ok(false)
    }

    fn publish_parameter_replacements(
        &mut self,
        _values: &std::collections::BTreeMap<String, M::Tensor>,
        _active: bool,
    ) -> Result<bool, M::Error> {
        Ok(false)
    }

    fn control_state(
        &self,
    ) -> Option<&dyn crate::text_session_driver::ErasedTextStateAccess<M::Context>> {
        None
    }

    fn control_state_mut(
        &mut self,
    ) -> Option<(
        &mut dyn crate::text_session_driver::ErasedTextStateAccess<M::Context>,
        &M::Context,
    )> {
        None
    }

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    >;

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    >;

    fn component_residency_reports(&self) -> Result<Vec<ResidencyReport>, M::Error> {
        Ok(Vec::new())
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, M::Error>;

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, M::Error>;

    fn materialization_report(
        &self,
    ) -> Option<&eredu_checkpoint::quantization::WeightMaterializationReport>;

    fn parameter_bank_report(
        &self,
    ) -> Result<Option<crate::expert::ParameterBanksResidencyReport>, M::Error>;

    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &M::Group,
        &eredu_core::communication::CommunicationAuthority,
        &M::Context,
        usize,
    )> {
        None
    }

    fn bind_embedded_prediction(
        &mut self,
    ) -> Option<
        Box<
            dyn ErasedGenerationExecutor<
                    M::EmbeddedExecutor,
                    crate::text_session_driver::PreparedTextPrompt<M::InputTensor>,
                > + '_,
        >,
    > {
        None
    }

    fn install_embedded_prediction_observers(&mut self, _observers: M::EmbeddedObservers) -> bool {
        false
    }

    fn take_speculative_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        None
    }

    fn take_speculative_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        None
    }

    fn external_prediction_mut(
        &mut self,
    ) -> Option<&mut dyn ErasedExternalPredictionExecutable<M>> {
        None
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, M::StateError>;
}

mod route_seal {
    pub trait Sealed {}

    impl<S, I, N: super::super::text_invocation::TextInvocationResources, P> Sealed
        for super::super::BoundTextSession<S, I, N, P>
    {
    }
    impl<L, R> Sealed for super::super::routed_session::RoutedExecutable<L, R> {}
    impl<L, P> Sealed for super::super::routed_partition_binding::BoundPartitionExecutable<L, P> {}
}

/// Converts a fully assembled route into its installed executable.
pub trait TextExecutableRoute<Output>: route_seal::Sealed {
    /// Erases the typed route after portable construction has completed.
    fn install(self) -> Output;
}

impl<M, S, I, N, P> TextExecutableRoute<InstalledTextExecutable<M>>
    for super::BoundTextSession<S, I, N, P>
where
    M: TextExecutableTypes,
    N: super::text_invocation::TextInvocationResources,
    Self: ErasedTextExecutable<M> + 'static,
{
    fn install(self) -> InstalledTextExecutable<M> {
        erase_text_executable(self)
    }
}

impl<L, R, O> TextExecutableRoute<O> for super::routed_session::RoutedExecutable<L, R>
where
    L: TextExecutableRoute<O>,
    R: TextExecutableRoute<O>,
{
    fn install(self) -> O {
        match self {
            Self::Resident(executable) => executable.install(),
            Self::Addressable(executable) => executable.install(),
        }
    }
}

impl<L, P, O> TextExecutableRoute<O>
    for super::routed_partition_binding::BoundPartitionExecutable<L, P>
where
    L: TextExecutableRoute<O>,
    P: TextExecutableRoute<O>,
{
    fn install(self) -> O {
        match self {
            Self::Local(executable) => executable.install(),
            Self::Pipeline(executable) => executable.install(),
        }
    }
}

/// Installs the portable route using the executable representation selected by composition.
pub fn install_text_route<Ready, Bound, Output>(bound: Bound) -> Output
where
    Ready: From<Bound> + TextExecutableRoute<Output>,
{
    Ready::from(bound).install()
}

impl<M: TextExecutableTypes> crate::text_session_driver::InstalledTextState
    for InstalledTextExecutable<M>
{
    type Context = M::Context;

    fn control_state(
        &self,
    ) -> Option<&dyn crate::text_session_driver::ErasedTextStateAccess<M::Context>> {
        self.as_ref().control_state()
    }

    fn control_state_mut(
        &mut self,
    ) -> Option<(
        &mut dyn crate::text_session_driver::ErasedTextStateAccess<M::Context>,
        &M::Context,
    )> {
        self.as_mut().control_state_mut()
    }

    fn has_embedded_prediction(&self) -> bool {
        self.as_ref().has_embedded_prediction()
    }

    fn snapshot_retirement(&self) -> Option<Box<dyn std::any::Any>> {
        M::snapshot_retirement()
    }
}

impl<M: TextExecutableTypes> crate::text_session_driver::SpeculativeObservationModel
    for InstalledTextExecutable<M>
{
    type Observers = M::EmbeddedObservers;

    fn attach_prediction_observers(&mut self, observers: Self::Observers) -> bool {
        self.as_mut()
            .install_embedded_prediction_observers(observers)
    }

    fn take_speculative_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        self.as_mut().take_speculative_activation_capture()
    }

    fn take_speculative_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.as_mut().take_speculative_activation_error()
    }
}
