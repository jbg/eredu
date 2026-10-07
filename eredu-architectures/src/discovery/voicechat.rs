//! Coupled speech graph, with distinct transcription and conversational frontiers.

use super::*;
use crate::voicechat::config::VoiceChatConfig;

pub(super) fn describe(g: &mut Builder, c: &VoiceChatConfig) {
    use ArchitectureNodeKind as K;
    g.node("realtime", K::Realtime, None, None, None);
    for (id, kind, prefix, width) in [
        (
            "perception.features",
            K::Processor,
            "stt_model.perception.preprocessor",
            Some(c.perception.preprocessor.features),
        ),
        (
            "perception.encoder",
            K::Encoder,
            "stt_model.perception.encoder",
            Some(c.perception.encoder.d_model),
        ),
        (
            "perception.projection",
            K::Projector,
            "stt_model.perception.proj",
            Some(c.perception.output_dim),
        ),
        (
            "transcription.prediction",
            K::Mixer,
            "stt_model.rnnt_decoder",
            Some(c.rnnt.pred_hidden),
        ),
        (
            "transcription.joint",
            K::OutputHead,
            "stt_model.rnnt_joint",
            Some(c.rnnt.vocab_size + 1),
        ),
        (
            "backbone",
            K::DecoderBlock,
            "stt_model.llm",
            Some(c.perception.output_dim),
        ),
        (
            "agent_text",
            K::OutputHead,
            "stt_model.lm_head",
            Some(crate::voicechat::channels::TEXT_VOCABULARY_SIZE),
        ),
        (
            "function",
            K::OutputHead,
            "stt_model.function_head",
            Some(crate::voicechat::channels::TEXT_VOCABULARY_SIZE),
        ),
        (
            "tts.characters",
            K::Encoder,
            "tts_model.tts_model.embed_subword.backbone.encoder",
            Some(c.tts.character_encoder.hidden_size),
        ),
        (
            "tts.backbone",
            K::DecoderBlock,
            "tts_model.tts_model.backbone",
            Some(c.tts.backbone.hidden_size),
        ),
        (
            "tts.mixture",
            K::Prediction,
            "tts_model.tts_model.mog_head",
            Some(c.tts.latent_size),
        ),
        (
            "codec.quantizer",
            K::Prediction,
            "tts_model.audio_codec.prvq",
            Some(c.codec.latent_size),
        ),
        (
            "codec.encoder",
            K::Encoder,
            "tts_model.audio_codec.encoder",
            Some(c.codec.latent_size),
        ),
        (
            "codec.decoder",
            K::DecoderBlock,
            "tts_model.audio_codec.decoder",
            None,
        ),
        (
            "speaker",
            K::Embedding,
            "tts_model.audio_prompt_latents",
            Some(c.tts.latent_size),
        ),
    ] {
        g.node(id, kind, Some("realtime"), Some(prefix), width);
    }
    for (from, to) in [
        ("perception.features", "perception.encoder"),
        ("perception.encoder", "perception.projection"),
        ("perception.encoder", "transcription.joint"),
        ("transcription.prediction", "transcription.joint"),
        ("perception.projection", "backbone"),
        ("backbone", "agent_text"),
        ("backbone", "function"),
        ("agent_text", "tts.characters"),
        ("tts.characters", "tts.backbone"),
        ("speaker", "tts.backbone"),
        ("tts.backbone", "tts.mixture"),
        ("tts.mixture", "codec.quantizer"),
        ("codec.encoder", "codec.quantizer"),
        ("codec.quantizer", "codec.decoder"),
    ] {
        g.edge(from, to, ArchitectureEdgeKind::Data);
    }
    for (from, to) in [
        ("agent_text", "backbone"),
        ("function", "backbone"),
        ("transcription.joint", "transcription.prediction"),
        ("codec.quantizer", "tts.backbone"),
    ] {
        g.edge(from, to, ArchitectureEdgeKind::State);
    }
    g.partial("Coupled speech components and feedback are described; internal block equations and realtime invocation clocks are not enumerated by this descriptor");
    g.descriptor.observations.completeness = DescriptionCompleteness::Unsupported(vec![
        "Prefill/decode capture discovery does not represent realtime speech and transcription publication identities".into(),
    ]);
}
