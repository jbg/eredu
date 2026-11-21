//! Portable type composition for bound text execution.

use eredu_core::{
    backend::{NativeTextValues, TextExecutableTypes},
    communication::CommunicationAuthority,
    input::PreparedInputInspector,
    speculative::embedded::EmbeddedPredictionOperators,
    BackendFailure, CollectiveGroupId, PreparedInputPart, PreparedInputPayload, PreparedModelInput,
};
use eredu_nn::Tensor;

use crate::{
    parameter_operations::BankParameterReplacementFacts,
    speculative::{
        EmbeddedPredictionObservers, PortablePredictionTargetState, SpeculativeTensorMechanisms,
    },
    state::{HybridCacheStorage, HybridState},
    text_session_driver::{TextPromptView, TextSnapshotStorage},
    CacheResidencyReport,
};

use super::{
    embedded_erasure::EmbeddedTextBinding,
    retained_executable::{RetainedTextMechanisms, TextTensorMechanisms},
    session_erasure::TextErasureTypes,
    text_invocation::TextInvocationMechanisms,
};

/// Native text capabilities whose embedded predictor can use portable capture.
pub trait TextPredictionPrimitives:
    NativeTextValues<
    Embedded: SpeculativeTensorMechanisms<
        Tensor = Self::Tensor,
        Logits = <Self::Embedded as EmbeddedPredictionOperators>::Logits,
        Error = <Self::Embedded as EmbeddedPredictionOperators>::Error,
    >,
>
{
}

impl<N> TextPredictionPrimitives for N
where
    N: NativeTextValues,
    N::Embedded: SpeculativeTensorMechanisms<
        Tensor = N::Tensor,
        Logits = <N::Embedded as EmbeddedPredictionOperators>::Logits,
        Error = <N::Embedded as EmbeddedPredictionOperators>::Error,
    >,
{
}

type PredictionError<N> = <<N as NativeTextValues>::Embedded as EmbeddedPredictionOperators>::Error;

type Observers<N> = EmbeddedPredictionObservers<
    <N as NativeTextValues>::Tensor,
    <<N as NativeTextValues>::Embedded as EmbeddedPredictionOperators>::Logits,
    PredictionError<N>,
>;

/// Text representations and portable installed state derived from native capabilities.
pub struct PortableTextTypes<N>(std::marker::PhantomData<fn() -> N>);

impl<N: NativeTextValues> TextExecutableTypes for PortableTextTypes<N> {
    type InputTensor = N::InputTensor;
    type Decode = N::Decode;
    type Scores = N::Scores;
    type Tensor = N::Tensor;
    type InvocationTensor = N::InvocationTensor;
    type ObserverMapping = N::ObserverMapping;
    type Context = N::Context;
    type Error = N::Error;
    type StateError = N::StateError;
    type PredictionState = PortablePredictionTargetState<N::Neural, N::Cache, N::Error>;
    type Group = N::Group;
    type EmbeddedExecutor =
        crate::speculative::embedded_mechanisms::PortableEmbeddedExecutor<N::Embedded>;
    type EmbeddedObservers = Observers<N>;
    type InputInspector = N::InputInspector;

    fn snapshot_retirement() -> Option<Box<dyn std::any::Any>> {
        N::snapshot_retirement()
    }

    fn input_inspector() -> Self::InputInspector {
        N::input_inspector()
    }

    fn read_token_ids(value: &N::InputTensor, queue: &N::Context) -> Result<Vec<u32>, N::Error> {
        N::read_token_ids(value, queue)
    }
}

impl<N: NativeTextValues> TextTensorMechanisms for PortableTextTypes<N> {
    type TensorInputInspector = N::TensorInputInspector;
    type TensorObserver = N::TensorObserver;

    fn prepared_input(
        input: TextPromptView<'_, N::InputTensor>,
    ) -> Result<PreparedModelInput<N::Tensor>, N::Error> {
        let inspector = N::input_inspector();
        let parts = input
            .parts
            .iter()
            .map(|part| {
                let payload = part.payload();
                let identity = inspector.identity(payload.value())?;
                payload
                    .kind()
                    .validate_decoder_tensor(identity.dtype(), identity.shape())?;
                let value = N::input_tensor(payload.value());
                let payload = match payload {
                    PreparedInputPayload::TokenIds(_) => PreparedInputPayload::TokenIds(value),
                    PreparedInputPayload::Tensor(_) => PreparedInputPayload::Tensor(value),
                    PreparedInputPayload::Embeddings(_) => PreparedInputPayload::Embeddings(value),
                };
                PreparedInputPart::new_with_extents(
                    part.modality(),
                    payload,
                    part.metadata()
                        .iter()
                        .map(|(key, value)| (*key, N::input_tensor(value))),
                    part.extents().iter().copied(),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(BackendFailure::from_error)?;
        let inspector = N::tensor_input_inspector();
        PreparedModelInput::new(parts, |value| inspector.identity(value))
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }

    fn tensor_input_inspector() -> Self::TensorInputInspector {
        N::tensor_input_inspector()
    }
    fn decode_tensor(value: &N::Decode) -> N::Tensor {
        N::decode_tensor(value)
    }
    fn scores(value: N::Tensor) -> N::Scores {
        N::scores(value)
    }
}

impl<N> RetainedTextMechanisms for PortableTextTypes<N>
where
    N: TextPredictionPrimitives,
    N::Cache: HybridCacheStorage,
    HybridState<N::Cache>: TextSnapshotStorage<N::Context>
        + crate::state::PortableTextState<N::Neural, N::Cache>
        + 'static,
{
    type State = HybridState<N::Cache>;
    type Communication = N::Communication;

    fn bound_group(
        communication: &Self::Communication,
        group: CollectiveGroupId,
    ) -> Option<&N::Group> {
        N::sampling_group(communication, group)
    }

    fn authority(communication: &Self::Communication) -> &CommunicationAuthority {
        N::authority(communication)
    }

    fn state_offset(state: &Self::State) -> i32 {
        state.offset()
    }

    fn cache_residency(state: &Self::State) -> Result<Option<CacheResidencyReport>, N::StateError> {
        state
            .residency_report()
            .map_err(BackendFailure::from_error)
            .map_err(N::state_error)
    }
}

impl<N: TextPredictionPrimitives> TextInvocationMechanisms for PortableTextTypes<N> {
    type PredictionMaterializer = N::Materializer;

    fn concatenate_token_parts(
        parts: &[&N::InputTensor],
        queue: &N::Context,
    ) -> Result<N::Tensor, N::Error> {
        N::concatenate_token_parts(parts, queue)
    }
    fn prompt_tokens(value: &N::InvocationTensor) -> N::InputTensor {
        N::prompt_tokens(value)
    }
    fn decode_prompt_tokens(value: &N::Decode) -> N::InputTensor {
        N::decode_prompt_tokens(value)
    }
}

impl<N: TextPredictionPrimitives> TextErasureTypes for PortableTextTypes<N> {
    fn state_error(error: BackendFailure) -> N::StateError {
        N::state_error(error)
    }
    fn replacement_facts(value: &N::Tensor) -> BankParameterReplacementFacts {
        BankParameterReplacementFacts {
            leading_dimension: value.shape().first().copied(),
            element_type: value.element_type(),
            byte_len: N::tensor_bytes(value),
        }
    }
    fn continuation_state(
        state: &Self::PredictionState,
        additional: u64,
    ) -> Result<(u64, Option<u64>, Option<u64>), BackendFailure> {
        state.memory_bounds(additional)
    }
    fn decoder_token_ids(value: &N::Tensor) -> Result<Vec<u32>, N::Error> {
        N::decoder_token_ids(value)
    }
}

impl<N: TextPredictionPrimitives> EmbeddedTextBinding for PortableTextTypes<N> {
    type Mechanisms = N::Embedded;

    fn slice_tokens(
        tokens: &N::Tensor,
        range: std::ops::Range<usize>,
        queue: &N::Context,
    ) -> Result<N::Tensor, PredictionError<N>> {
        N::slice_tokens(tokens, range, queue)
    }
    fn settle_input(value: &N::Tensor) -> Result<(), PredictionError<N>> {
        N::settle_input(value)
    }
    fn input_failure(error: BackendFailure) -> PredictionError<N> {
        N::input_failure(error)
    }
    fn embedded_observers(value: &mut Observers<N>) -> &mut Observers<N> {
        value
    }
}
