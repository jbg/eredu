//! Native binding visitor for the architecture-owned speech provider aggregate.

use std::collections::BTreeMap;

use eredu_architectures::voicechat::{
    assembly::LearnedSpeechComponents,
    function_head_preparation::PreparedFunctionHeadBindings,
    perception_preparation::PreparedPerceptionBindings,
    provider_construction::{
        PreparedSpeechProviders, ProviderConstructionError, SpeechProviderBinder,
    },
    provider_residency::{SpeechProvider, SpeechResidencyPolicy, SpeechResidencyReport},
    rnnt_preparation::PreparedRnntBindings,
    tts_conditioning_preparation::PreparedConditioningBindings,
    tts_mog_preparation::PreparedMogBindings,
    tts_rvq_preparation::PreparedRvqBindings,
    tts_transformer_preparation::{PreparedTtsTransformerBindings, TtsTransformerKind},
};
use eredu_codec::{
    ear_decoder_preparation::PreparedDecoderBindings,
    ear_encoder_preparation::PreparedEncoderBindings, prvq_preparation::PreparedQuantizerBindings,
};
use eredu_core::BackendFailure;
use eredu_runtime::{LayerWeightResidency, LayerwiseModelMetadata};
use safemlx::{Array, Stream};

use super::{
    materialize_ear_decoder, materialize_ear_encoder, materialize_prvq,
    materialize_voicechat_conditioning, materialize_voicechat_function_head,
    materialize_voicechat_mog, materialize_voicechat_perception, materialize_voicechat_rnnt,
    materialize_voicechat_rvq, materialize_voicechat_tts_transformer, MlxConditioning,
    MlxEarDecoder, MlxEarEncoder, MlxFunctionHead, MlxMog, MlxPerception, MlxPrvq, MlxRnnt, MlxRvq,
    MlxTtsTransformer,
};
use crate::backend::{nn::shared::MlxNeuralBackend, submission_recovery::detached_preparation};

/// Independently selected speech providers for the ordinary coupled executor.
/// The Nemotron runtime and all conversation state remain separate owners.
pub type MlxSpeechProviders = LearnedSpeechComponents<
    MlxNeuralBackend,
    MlxPerception,
    MlxTtsTransformer,
    MlxMog,
    MlxRvq,
    MlxConditioning,
    MlxEarDecoder,
    MlxPrvq,
    MlxEarEncoder,
    MlxFunctionHead,
    MlxRnnt,
>;

/// Completed speech construction and its admitted parameter-pool accounting.
pub struct MaterializedMlxSpeechProviders {
    /// Learned providers consumed by the existing architecture executor.
    pub speech: MlxSpeechProviders,
    /// Summed independent pool allowances and conservative materialization reserve.
    /// This excludes the backbone, live state and activation workspaces.
    pub residency: SpeechResidencyReport,
    /// Exact native metadata for each independently selected provider.
    pub providers: BTreeMap<SpeechProvider, LayerwiseModelMetadata>,
}

pub(super) struct Binder<'a> {
    weights: &'a Stream,
    execution: &'a Stream,
    metadata: BTreeMap<SpeechProvider, LayerwiseModelMetadata>,
}
impl<'a> Binder<'a> {
    pub(super) fn new(weights: &'a Stream, execution: &'a Stream) -> Self {
        Self {
            weights,
            execution,
            metadata: BTreeMap::new(),
        }
    }
}
impl SpeechProviderBinder<MlxNeuralBackend> for Binder<'_> {
    type Perception = MlxPerception;
    type Transformer = MlxTtsTransformer;
    type Mog = MlxMog;
    type Rvq = MlxRvq;
    type Conditioning = MlxConditioning;
    type Encoder = MlxEarEncoder;
    type Decoder = MlxEarDecoder;
    type Quantizer = MlxPrvq;
    type Function = MlxFunctionHead;
    type Rnnt = MlxRnnt;
    fn perception(
        &mut self,
        prepared: PreparedPerceptionBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Perception, BackendFailure> {
        let (provider, metadata) =
            materialize_voicechat_perception(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::Perception, metadata);
        Ok(provider)
    }
    fn function(
        &mut self,
        prepared: PreparedFunctionHeadBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Function, BackendFailure> {
        let (provider, metadata) =
            materialize_voicechat_function_head(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::FunctionHead, metadata);
        Ok(provider)
    }
    fn mog(
        &mut self,
        prepared: PreparedMogBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Mog, BackendFailure> {
        let (provider, metadata) =
            materialize_voicechat_mog(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::Mog, metadata);
        Ok(provider)
    }
    fn rvq(
        &mut self,
        prepared: PreparedRvqBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Rvq, BackendFailure> {
        let (provider, metadata) =
            materialize_voicechat_rvq(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::Rvq, metadata);
        Ok(provider)
    }
    fn conditioning(
        &mut self,
        prepared: PreparedConditioningBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Conditioning, BackendFailure> {
        let (provider, metadata) =
            materialize_voicechat_conditioning(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::Conditioning, metadata);
        Ok(provider)
    }
    fn encoder(
        &mut self,
        prepared: PreparedEncoderBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Encoder, BackendFailure> {
        let (provider, metadata) =
            materialize_ear_encoder(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::CodecEncoder, metadata);
        Ok(provider)
    }
    fn decoder(
        &mut self,
        prepared: PreparedDecoderBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Decoder, BackendFailure> {
        let (provider, metadata) =
            materialize_ear_decoder(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::CodecDecoder, metadata);
        Ok(provider)
    }
    fn quantizer(
        &mut self,
        prepared: PreparedQuantizerBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Quantizer, BackendFailure> {
        let (provider, metadata) =
            materialize_prvq(prepared, residency, self.weights, self.execution)?;
        self.metadata
            .insert(SpeechProvider::CodecQuantizer, metadata);
        Ok(provider)
    }
    fn transformer(
        &mut self,
        kind: TtsTransformerKind,
        prepared: PreparedTtsTransformerBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Transformer, BackendFailure> {
        let (provider, metadata) = materialize_voicechat_tts_transformer(
            prepared,
            residency,
            self.weights,
            self.execution,
        )?;
        let id = match kind {
            TtsTransformerKind::CharacterEncoder => SpeechProvider::CharacterEncoder,
            TtsTransformerKind::Causal => SpeechProvider::TtsTransformer,
        };
        self.metadata.insert(id, metadata);
        Ok(provider)
    }
    fn rnnt(
        &mut self,
        prepared: PreparedRnntBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Rnnt, BackendFailure> {
        let (provider, metadata) =
            materialize_voicechat_rnnt(prepared, residency, self.weights, self.execution)?;
        self.metadata.insert(SpeechProvider::Rnnt, metadata);
        Ok(provider)
    }
}

/// Binds all selected speech providers under the architecture's aggregate admission.
///
/// Exact preparation and summed independent-pool budget validation precede the
/// first binding callback. The architecture owns role order and consuming assembly;
/// this visitor supplies only the existing native parameter realizations.
/// Native completion protects successful construction and failure cleanup.
pub fn materialize_voicechat_speech_providers(
    prepared: PreparedSpeechProviders,
    residency: &SpeechResidencyPolicy,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<MaterializedMlxSpeechProviders, ProviderConstructionError> {
    crate::initialize_allocator_cache_policy(false).map_err(|error| {
        ProviderConstructionError::Backend(eredu_core::BackendFailure::from_error(error))
    })?;
    let result = detached_preparation(
        Vec::<Array>::new(),
        || {
            let mut binder = Binder::new(weights_stream, execution_stream);
            let (speech, residency) = prepared.materialize::<MlxNeuralBackend, _>(
                residency,
                &mut binder,
                execution_stream,
            )?;
            Ok(MaterializedMlxSpeechProviders {
                speech,
                residency,
                providers: binder.metadata,
            })
        },
        |error| ProviderConstructionError::Backend(BackendFailure::from_error(error)),
    );
    // One-use construction buffers have no inference reuse value. Active roots
    // and unresolved submissions remain protected by native ownership.
    let trim = crate::trim_allocator_cache();
    result.and_then(|value| {
        trim.map_err(|error| {
            ProviderConstructionError::Backend(eredu_core::BackendFailure::from_error(error))
        })?;
        Ok(value)
    })
}

impl
    eredu_architectures::voicechat::components::BackboneProviderBinder<
        MlxNeuralBackend,
        crate::backend::runtime::cache::state::MlxHybridState,
    > for Binder<'_>
{
    type Backbone = super::MlxVoiceChatBackbone;

    fn backbone(
        &mut self,
        prepared: &eredu_architectures::voicechat::backbone::PreparedBackboneParameters,
        residency: &LayerWeightResidency,
    ) -> Result<Self::Backbone, BackendFailure> {
        super::materialize_voicechat_backbone(prepared, residency, self.weights, self.execution)
    }
}
