//! Exact portable frame-policy contracts retained across selection and sessions.

use eredu_core::{RealtimeChannelLayout, RealtimeChannelPayload, RealtimeSpeechConfig};

use crate::RealtimeIdentity;

/// Canonical architecture-owned schedule identity for admission and consensus.
///
/// Descriptors include a schema/version discriminator and every inference-relevant
/// policy field. Equal descriptors must mean identical scheduling semantics;
/// addresses, Debug formatting, native resources and process-local hashes are
/// excluded. The descriptor represents immutable policy, not mutable frontiers.
pub trait RealtimeScheduleDescriptor {
    /// Encodes complete immutable scheduling policy as deterministic u32 words.
    fn schedule_descriptor(&self) -> Vec<u32>;
}

impl RealtimeScheduleDescriptor for RealtimeSpeechConfig {
    fn schedule_descriptor(&self) -> Vec<u32> {
        // Domain RTSC, schema version 1. Integers use low/high u64 limbs.
        let mut words = vec![0x5254_5343, 1];
        for count in [
            self.total_audio_codebooks(),
            self.input_audio_codebooks(),
            self.generated_audio_codebooks(),
            self.depth_audio_codebooks(),
        ] {
            push_usize(&mut words, count);
        }
        words.push(self.text_padding_token() as u32);
        words.push(self.audio_padding_token() as u32);
        // The enum is non-exhaustive. Its explicit serde wire discriminant
        // preserves future variants without a lossy catch-all numeric value.
        let convention = serde_json::to_string(&self.frame_convention())
            .expect("the portable frame convention has an infallible serde representation");
        push_bytes(&mut words, convention.as_bytes());
        push_usize(&mut words, self.delays().len());
        for &delay in self.delays() {
            push_usize(&mut words, delay);
        }
        words
    }
}

/// Complete cold scheduling contract without a concrete family or native state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeFrameContract {
    policy_identity: RealtimeIdentity,
    policy_descriptor: Vec<u32>,
    channels: RealtimeChannelLayout,
    sampler_count: usize,
}

impl RealtimeFrameContract {
    /// Retains exact policy, typed channels and categorical sampler ownership.
    /// Dense-only execution may declare zero sampler states.
    pub fn new(
        policy_identity: RealtimeIdentity,
        policy_descriptor: Vec<u32>,
        channels: RealtimeChannelLayout,
        sampler_count: usize,
    ) -> Result<Self, RealtimeFrameContractError> {
        if policy_descriptor.is_empty() {
            return Err(RealtimeFrameContractError::EmptyPolicyDescriptor);
        }
        Ok(Self {
            policy_identity,
            policy_descriptor,
            channels,
            sampler_count,
        })
    }

    /// Architecture policy identity used by construction proofs.
    pub const fn policy_identity(&self) -> &RealtimeIdentity {
        &self.policy_identity
    }
    /// Complete versioned immutable schedule descriptor.
    pub fn policy_descriptor(&self) -> &[u32] {
        &self.policy_descriptor
    }
    /// Exact categorical/dense channel declarations.
    pub const fn channels(&self) -> &RealtimeChannelLayout {
        &self.channels
    }
    /// Explicit categorical sampler-bank cardinality.
    pub const fn sampler_count(&self) -> usize {
        self.sampler_count
    }

    /// Canonical complete frame contract for topology-wide exact consensus.
    ///
    /// Schema 1 retains declaration order, payload kinds and dimensions, cadence,
    /// delay and every feedback dependency. UTF-8 strings are byte-length-prefixed
    /// and packed into little-endian words; lengths and usize values use u64 limbs.
    pub fn consensus_descriptor(&self) -> Vec<u32> {
        // Domain RTFC, schema version 1.
        let mut words = vec![0x5254_4643, 1];
        push_bytes(&mut words, self.policy_identity.as_str().as_bytes());
        push_usize(&mut words, self.policy_descriptor.len());
        words.extend_from_slice(&self.policy_descriptor);
        push_bytes(&mut words, self.channels.policy().as_bytes());
        push_usize(&mut words, self.channels.channels().len());
        for channel in self.channels.channels() {
            push_usize(&mut words, channel.id.index());
            match &channel.payload {
                RealtimeChannelPayload::Categorical { width, cardinality } => {
                    words.push(0);
                    push_usize(&mut words, *width);
                    push_usize(&mut words, *cardinality);
                }
                RealtimeChannelPayload::Dense { shape } => {
                    words.push(1);
                    push_usize(&mut words, shape.len());
                    for &dimension in shape {
                        push_usize(&mut words, dimension);
                    }
                }
            }
            push_usize(&mut words, channel.cadence.period);
            push_usize(&mut words, channel.cadence.phase);
            push_usize(&mut words, channel.delay);
            push_usize(&mut words, channel.feedback.len());
            for dependency in &channel.feedback {
                push_usize(&mut words, dependency.source.index());
                push_usize(&mut words, dependency.lag);
            }
        }
        push_usize(&mut words, self.sampler_count);
        words
    }
}

/// A cold frame contract lacks the data required for exact schedule admission.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealtimeFrameContractError {
    /// An identity label alone cannot establish complete policy equivalence.
    #[error("realtime frame contract requires a nonempty versioned policy descriptor")]
    EmptyPolicyDescriptor,
}

fn push_usize(words: &mut Vec<u32>, value: usize) {
    let value = u64::try_from(value).expect("supported host usize fits u64");
    words.extend([value as u32, (value >> 32) as u32]);
}
fn push_bytes(words: &mut Vec<u32>, bytes: &[u8]) {
    push_usize(words, bytes.len());
    for chunk in bytes.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(word));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{
        RealtimeChannelCadence, RealtimeChannelDeclaration, RealtimeChannelFeedback,
        RealtimeChannelId, RealtimeFrameConvention,
    };

    fn speech(convention: RealtimeFrameConvention, generated: usize) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(
            3,
            3 - generated,
            generated,
            2,
            12,
            8,
            convention,
            vec![0, 1, 2, 1],
        )
        .unwrap()
    }

    #[test]
    fn schedule_descriptor_distinguishes_coordinate_semantics_and_channel_ownership() {
        let history = speech(RealtimeFrameConvention::FeedbackAlignedHistory, 1);
        let absolute = speech(RealtimeFrameConvention::AbsoluteDelayedSlots, 1);
        let ownership = speech(RealtimeFrameConvention::FeedbackAlignedHistory, 2);
        assert_eq!(
            history.schedule_descriptor(),
            history.clone().schedule_descriptor()
        );
        assert_ne!(
            history.schedule_descriptor(),
            absolute.schedule_descriptor()
        );
        assert_ne!(
            history.schedule_descriptor(),
            ownership.schedule_descriptor()
        );
    }

    fn channels(lag: usize) -> RealtimeChannelLayout {
        let category = RealtimeChannelId::new(7);
        let dense = RealtimeChannelId::new(42);
        RealtimeChannelLayout::new(
            "coupled_cadence_v1".into(),
            vec![
                RealtimeChannelDeclaration {
                    id: category,
                    payload: RealtimeChannelPayload::Categorical {
                        width: 1,
                        cardinality: 32,
                    },
                    cadence: RealtimeChannelCadence {
                        period: 1,
                        phase: 0,
                    },
                    delay: 0,
                    feedback: vec![],
                },
                RealtimeChannelDeclaration {
                    id: dense,
                    payload: RealtimeChannelPayload::Dense { shape: vec![2, 3] },
                    cadence: RealtimeChannelCadence {
                        period: 2,
                        phase: 0,
                    },
                    delay: 2,
                    feedback: vec![RealtimeChannelFeedback { source: dense, lag }],
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn complete_contract_descriptor_binds_feedback_history_and_sampler_ownership() {
        let identity = RealtimeIdentity::new("coupled_policy_v1").unwrap();
        let base =
            RealtimeFrameContract::new(identity.clone(), vec![1, 80, 16000, 22050], channels(2), 1)
                .unwrap();
        let longer_history =
            RealtimeFrameContract::new(identity.clone(), vec![1, 80, 16000, 22050], channels(4), 1)
                .unwrap();
        let dense_only =
            RealtimeFrameContract::new(identity.clone(), vec![1, 80, 16000, 22050], channels(2), 0)
                .unwrap();
        assert_eq!(
            base.consensus_descriptor(),
            base.clone().consensus_descriptor()
        );
        assert_ne!(
            base.consensus_descriptor(),
            longer_history.consensus_descriptor()
        );
        assert_ne!(
            base.consensus_descriptor(),
            dense_only.consensus_descriptor()
        );
        assert_eq!(
            RealtimeFrameContract::new(identity, vec![], channels(2), 1),
            Err(RealtimeFrameContractError::EmptyPolicyDescriptor)
        );
    }
}
