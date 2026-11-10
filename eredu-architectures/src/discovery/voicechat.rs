//! Coupled speech graph, with distinct transcription and conversational frontiers.

use super::*;
use crate::voicechat::config::VoiceChatConfig;

fn attach_identity(g: &mut Builder, node: &str, identity: &str, owner: Option<&str>) {
    let prefix = identity.strip_suffix(".weight").unwrap_or(identity);
    g.attach_parameters(node, prefix);
    if let Some(owner) = owner {
        let shared = g.parameter_group(owner.strip_suffix(".weight").unwrap_or(owner));
        g.descriptor
            .parameter_groups
            .iter_mut()
            .find(|group| group.canonical_prefix == prefix)
            .expect("attached declared parameter")
            .shared_with = Some(shared);
    }
}

fn attach(g: &mut Builder, node: &str, spec: &(impl eredu_nn::DeclareLogicalParameters + ?Sized)) {
    for declared in spec
        .logical_parameters()
        .expect("admitted VoiceChat declarations")
    {
        let parameter = declared.parameter;
        attach_identity(
            g,
            node,
            parameter.id.as_str(),
            parameter.alias_of.as_ref().map(|id| id.as_str()),
        );
    }
}

pub(super) fn describe(g: &mut Builder, c: &VoiceChatConfig) {
    use crate::voicechat::{
        perception_specs,
        rnnt_layered::{RnntPhase, RnntSelection},
        rnnt_specs::{self, UnitSpec},
        tts_conditioning_specs, tts_mog_specs, tts_rvq_specs, tts_transformer_specs,
    };
    use eredu_codec::ear_specs::EncoderParameters;
    use ArchitectureNodeKind as K;
    let perception = perception_specs::released_declaration()
        .expect("admitted perception declarations")
        .spec();
    let rnnt = rnnt_specs::released_declaration().expect("admitted RNNT declarations");
    let rnnt = rnnt.spec();
    let UnitSpec::Embedding(predictor) = rnnt
        .unit(RnntPhase::Embedding, 0)
        .expect("declared predictor embedding")
    else {
        unreachable!("RNNT embedding phase declares an embedding");
    };
    let UnitSpec::Projection(joint) = rnnt
        .unit(RnntPhase::Output, 0)
        .expect("declared RNNT output")
    else {
        unreachable!("RNNT output phase declares a projection");
    };
    let conditioning = tts_conditioning_specs::released_declaration()
        .expect("admitted TTS conditioning declarations")
        .spec();
    let transformers =
        tts_transformer_specs::released_declarations().expect("admitted transformer declarations");
    let characters = transformers.character.spec();
    let speech = transformers.causal.spec();
    let mog = tts_mog_specs::released_declaration().expect("admitted mixture declarations");
    let rvq = tts_rvq_specs::released_declaration().expect("admitted RVQ declarations");
    let codec =
        crate::voicechat::codec::released_declarations().expect("admitted codec declarations");
    let encoder = codec.encoder.spec();
    let decoder = codec.decoder.spec();
    let quantizer = codec.quantizer.spec();
    let encoder_width = match encoder
        .units()
        .last()
        .expect("nonempty codec encoder")
        .parameters()
    {
        EncoderParameters::Input(spec) | EncoderParameters::Downsample(spec) => {
            spec.geometry().output_channels
        }
        EncoderParameters::Block(spec) => spec.project().geometry().output_channels,
    };
    let backbone =
        crate::voicechat::backbone::released_declaration().expect("admitted backbone declaration");
    let backbone_statics = &backbone.statics;
    let text_output = backbone_statics.output_projection_spec();
    let backbone_parameters = backbone.definition().parameters();
    let function = crate::voicechat::function_head_layered::released_declaration()
        .expect("admitted function declaration");

    g.node("realtime", K::Realtime, None, None, None);
    // Host processor geometry is independent of neural parameter ownership.
    g.node(
        "perception.features",
        K::Processor,
        Some("realtime"),
        Some("stt_model.perception.preprocessor"),
        Some(c.perception.preprocessor.features),
    );
    for (id, kind, width) in [
        (
            "perception.encoder",
            K::Encoder,
            Some(perception.pinned.projection.input as usize),
        ),
        (
            "perception.projection",
            K::Projector,
            Some(perception.pinned.projection.output as usize),
        ),
        (
            "transcription.prediction",
            K::Mixer,
            Some(predictor.dimensions as usize),
        ),
        (
            "transcription.joint",
            K::OutputHead,
            Some(joint.output as usize),
        ),
        (
            "backbone",
            K::DecoderBlock,
            Some(backbone_statics.embedding_spec().dimensions as usize),
        ),
        (
            "agent_text",
            K::OutputHead,
            Some(text_output.output as usize),
        ),
        (
            "function",
            K::OutputHead,
            Some(function.dimensions().vocabulary as usize),
        ),
        (
            "tts.characters",
            K::Encoder,
            Some(characters.parameters.normalization.dimensions as usize),
        ),
        (
            "tts.backbone",
            K::DecoderBlock,
            Some(speech.parameters.normalization.dimensions as usize),
        ),
        (
            "tts.mixture",
            K::Prediction,
            Some(mog.spec().components.residual.output as usize),
        ),
        (
            "codec.quantizer",
            K::Prediction,
            Some(quantizer.geometry().channels as usize),
        ),
        ("codec.encoder", K::Encoder, Some(encoder_width as usize)),
        ("codec.decoder", K::DecoderBlock, None),
        (
            "speaker",
            K::Embedding,
            Some(
                *conditioning
                    .ingress
                    .aria
                    .shape
                    .last()
                    .expect("declared prompt feature axis") as usize,
            ),
        ),
    ] {
        g.node(id, kind, Some("realtime"), None, width);
    }
    attach(g, "perception.encoder", &perception.pinned.subsampling);
    attach(g, "perception.encoder", &perception.blocks[..]);
    attach(g, "perception.projection", &perception.pinned.projection);
    for phase in RnntSelection::Complete.phases() {
        let node = match phase {
            RnntPhase::Embedding | RnntPhase::Predictor => "transcription.prediction",
            _ => "transcription.joint",
        };
        for unit in rnnt.phase(*phase).expect("declared RNNT phase") {
            attach(g, node, unit);
        }
    }
    for group in backbone_parameters.groups() {
        for member in group.group().members() {
            if member.linear_companion().is_some() {
                continue;
            }
            let metadata = member
                .parameter_metadata()
                .expect("declared backbone identity");
            let canonical = |id: &eredu_nn::ParameterId| {
                crate::voicechat::backbone::checkpoint_parameter_name(backbone.args(), id.as_str())
                    .expect("declared backbone source extension")
            };
            let identity = canonical(&metadata.id);
            let owner = metadata.alias_of.as_ref().map(canonical);
            let node = if metadata.id == text_output.weight.id {
                "agent_text"
            } else {
                "backbone"
            };
            attach_identity(g, node, &identity, owner.as_deref());
        }
    }
    attach(g, "function", function.spec());
    attach(g, "tts.characters", &conditioning.character);
    attach(g, "tts.characters", characters);
    attach(g, "tts.backbone", speech);
    attach(g, "tts.backbone", &conditioning.fusion);
    for slot in [
        &conditioning.ingress.bos,
        &conditioning.ingress.null,
        &conditioning.ingress.projection,
    ] {
        attach(g, "tts.backbone", slot);
    }
    attach(g, "tts.backbone", rvq.spec());
    attach(g, "tts.mixture", mog.spec());
    attach(g, "codec.quantizer", quantizer);
    attach(g, "codec.encoder", encoder);
    attach(g, "codec.decoder", decoder);
    attach(g, "speaker", &conditioning.ingress.aria);
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
