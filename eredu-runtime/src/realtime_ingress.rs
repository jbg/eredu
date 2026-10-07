//! Portable realtime-frame validation before opaque token materialization.

use eredu_core::{
    RealtimeChannelCadence, RealtimeChannelDeclaration, RealtimeChannelFeedback, RealtimeChannelId,
    RealtimeChannelLayout, RealtimeChannelPayload, RealtimeFrameForcing, RealtimeInputFrame,
    RealtimeSpeechConfig,
};

use crate::{RealtimeChannelIngressError, TokenDomain};

/// Exact schedule and token domains used to validate portable realtime input.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeIngressContract {
    schedule: RealtimeSpeechConfig,
    channels: RealtimeChannelLayout,
    text: TokenDomain,
    audio: TokenDomain,
}

impl RealtimeIngressContract {
    /// Creates a contract whose padding IDs are inside the admitted domains.
    pub fn new(
        schedule: RealtimeSpeechConfig,
        text: TokenDomain,
        audio: TokenDomain,
    ) -> Result<Self, RealtimeIngressError> {
        validate_token(schedule.text_padding_token(), text, RealtimeTokenKind::Text)?;
        validate_token(
            schedule.audio_padding_token(),
            audio,
            RealtimeTokenKind::Audio,
        )?;
        let channels = RealtimeChannelLayout::new(
            serde_json::to_string(&schedule).expect("validated schedule serializes"),
            schedule
                .delays()
                .iter()
                .enumerate()
                .map(|(index, &delay)| RealtimeChannelDeclaration {
                    id: RealtimeChannelId::new(index),
                    payload: RealtimeChannelPayload::Categorical {
                        width: 1,
                        cardinality: if index == 0 {
                            text.cardinality()
                        } else {
                            audio.cardinality()
                        },
                    },
                    cadence: RealtimeChannelCadence {
                        period: 1,
                        phase: 0,
                    },
                    delay,
                    feedback: {
                        let mut dependencies = vec![RealtimeChannelFeedback {
                            source: RealtimeChannelId::new(index),
                            lag: 1,
                        }];
                        if index > 0 && index <= schedule.generated_audio_codebooks() {
                            dependencies.push(RealtimeChannelFeedback {
                                source: RealtimeChannelId::new(index - 1),
                                lag: 0,
                            });
                        }
                        dependencies
                    },
                })
                .collect(),
        )
        .map_err(RealtimeIngressError::Channels)?;
        Ok(Self {
            schedule,
            channels,
            text,
            audio,
        })
    }

    /// Returns the exact normalized speech schedule.
    pub const fn schedule(&self) -> &RealtimeSpeechConfig {
        &self.schedule
    }

    /// Exact typed channels used by retained payloads and transaction identity.
    pub const fn channels(&self) -> &RealtimeChannelLayout {
        &self.channels
    }

    /// Returns the complete admitted text-token domain.
    pub const fn text_domain(&self) -> TokenDomain {
        self.text
    }

    /// Returns the complete admitted audio-token domain.
    pub const fn audio_domain(&self) -> TokenDomain {
        self.audio
    }

    /// Validates host shape, forcing, and selected token values atomically.
    pub fn validate<'a>(
        &'a self,
        frame: &'a RealtimeInputFrame,
    ) -> Result<ValidatedRealtimeInput<'a>, RealtimeIngressError> {
        let batch = frame.batch();
        if batch == 0 {
            return Err(RealtimeIngressError::EmptyBatch);
        }
        let input_columns = self.schedule.input_audio_codebooks();
        validate_shape(
            RealtimePayloadKind::InputAudio,
            frame.input_audio_tokens().len(),
            batch,
            input_columns,
        )?;
        for &token in frame.input_audio_tokens() {
            validate_token(token, self.audio, RealtimeTokenKind::Audio)?;
        }

        let generated = self.schedule.generated_audio_codebooks();
        let forced_audio = frame.forced_generated_audio_tokens();
        let forcing = match (forced_audio, frame.forced_generated_audio_codebooks()) {
            (None, None) => vec![false; generated],
            (None, Some(_)) => return Err(RealtimeIngressError::ForcingMaskWithoutPayload),
            (Some(tokens), mask) => {
                validate_shape(
                    RealtimePayloadKind::ForcedAudio,
                    tokens.len(),
                    batch,
                    generated,
                )?;
                let mask = mask.map_or_else(|| vec![true; generated], <[bool]>::to_vec);
                if mask.len() != generated {
                    return Err(RealtimeIngressError::ForcingMaskCount {
                        expected: generated,
                        actual: mask.len(),
                    });
                }
                for row in tokens.chunks_exact(generated) {
                    for (token, selected) in row.iter().zip(&mask) {
                        if *selected {
                            validate_token(*token, self.audio, RealtimeTokenKind::Audio)?;
                        }
                    }
                }
                mask
            }
        };

        let forced_text = frame.forced_text_tokens();
        if let Some(tokens) = forced_text {
            validate_shape(RealtimePayloadKind::ForcedText, tokens.len(), batch, 1)?;
            for &token in tokens {
                validate_token(token, self.text, RealtimeTokenKind::Text)?;
            }
        }
        Ok(ValidatedRealtimeInput {
            contract: self,
            frame,
            forcing: RealtimeFrameForcing::new(forced_text.is_some(), forcing),
        })
    }
}

/// A portable frame proven valid before any opaque/native allocation.
pub struct ValidatedRealtimeInput<'a> {
    contract: &'a RealtimeIngressContract,
    frame: &'a RealtimeInputFrame,
    forcing: RealtimeFrameForcing,
}

impl ValidatedRealtimeInput<'_> {
    /// Returns the validated portable frame.
    pub const fn frame(&self) -> &RealtimeInputFrame {
        self.frame
    }

    /// Returns the exact schedule forcing mask derived from validated payloads.
    pub const fn forcing(&self) -> &RealtimeFrameForcing {
        &self.forcing
    }

    /// Converts validated host arrays through one family-blind mechanism.
    pub fn materialize<M: RealtimeHostTokenMaterializer>(
        &self,
        materializer: &mut M,
    ) -> Result<MaterializedRealtimeInput<M::Tensor>, M::Error> {
        let batch = self.frame.batch();
        let generated = self.contract.schedule.generated_audio_codebooks();
        let input_columns = self.contract.schedule.input_audio_codebooks();
        let mut inputs = Vec::with_capacity(input_columns);
        for column in 0..input_columns {
            let values = self
                .frame
                .input_audio_tokens()
                .chunks_exact(input_columns)
                .map(|row| row[column])
                .collect::<Vec<_>>();
            inputs.push((
                RealtimeChannelId::new(1 + generated + column),
                materializer.materialize_i32(&values, [batch, 1])?,
            ));
        }
        let mut forced = Vec::new();
        if let Some(tokens) = self.frame.forced_text_tokens() {
            forced.push((
                RealtimeChannelId::new(0),
                materializer.materialize_i32(tokens, [batch, 1])?,
            ));
        }
        if let Some(tokens) = self.frame.forced_generated_audio_tokens() {
            for (column, selected) in self.forcing.generated_audio().iter().enumerate() {
                if !selected {
                    continue;
                }
                let values = tokens
                    .chunks_exact(generated)
                    .map(|row| row[column])
                    .collect::<Vec<_>>();
                forced.push((
                    RealtimeChannelId::new(1 + column),
                    materializer.materialize_i32(&values, [batch, 1])?,
                ));
            }
        }
        Ok(MaterializedRealtimeInput::from_validated(
            self.contract.channels.clone(),
            batch,
            inputs,
            forced,
            self.frame.retains_diagnostics(),
        ))
    }
}

/// Narrow backend mechanism for copying one validated host token matrix.
pub trait RealtimeHostTokenMaterializer {
    /// Opaque/native token tensor.
    type Tensor;
    /// Materialization failure.
    type Error;

    /// Copies one already validated row-major i32 matrix.
    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<Self::Tensor, Self::Error>;
}

/// Host materialization capability for mixed categorical and dense channels.
pub trait RealtimeHostFrameMaterializer: RealtimeHostTokenMaterializer {
    /// Copies a validated finite f32 frame with its leading batch dimension.
    fn materialize_f32(
        &mut self,
        values: &[f32],
        shape: &[usize],
    ) -> Result<Self::Tensor, Self::Error>;
}

/// Opaque categorical or dense channel inputs and independently selected overrides.
///
/// A channel may occur once in each group. Architecture policy determines which
/// channels are required and how an override participates in execution; this
/// container has no text/audio matrix convention or schedule state.
#[derive(Debug, Clone)]
pub struct MaterializedRealtimeInput<T> {
    channels: RealtimeChannelLayout,
    batch: usize,
    inputs: Vec<(RealtimeChannelId, T)>,
    forced: Vec<(RealtimeChannelId, T)>,
    retain_diagnostics: bool,
}

impl<T> MaterializedRealtimeInput<T> {
    /// Validates channel identities and batch-expanded geometry for opaque values.
    /// Native shape and dtype validation belongs to the materialization mechanism;
    /// host values use [`crate::ValidatedRealtimeFrameInput`] before allocation.
    pub fn new(
        channels: RealtimeChannelLayout,
        batch: usize,
        inputs: Vec<(RealtimeChannelId, T)>,
        forced: Vec<(RealtimeChannelId, T)>,
        retain_diagnostics: bool,
    ) -> Result<Self, RealtimeChannelIngressError> {
        crate::realtime_channel_ingress::validate_channel_metadata(
            &channels,
            batch,
            inputs.iter().map(|(id, _)| *id),
        )?;
        crate::realtime_channel_ingress::validate_channel_metadata(
            &channels,
            batch,
            forced.iter().map(|(id, _)| *id),
        )?;
        Ok(Self::from_validated(
            channels,
            batch,
            inputs,
            forced,
            retain_diagnostics,
        ))
    }

    pub(crate) fn from_validated(
        channels: RealtimeChannelLayout,
        batch: usize,
        inputs: Vec<(RealtimeChannelId, T)>,
        forced: Vec<(RealtimeChannelId, T)>,
        retain_diagnostics: bool,
    ) -> Self {
        Self {
            channels,
            batch,
            inputs,
            forced,
            retain_diagnostics,
        }
    }

    /// Exact channels admitted before materialization.
    pub const fn channels(&self) -> &RealtimeChannelLayout {
        &self.channels
    }
    /// Validated positive batch dimension.
    pub const fn batch(&self) -> usize {
        self.batch
    }
    /// Ordinary inputs in submission order, keyed by channel identity.
    pub fn inputs(&self) -> &[(RealtimeChannelId, T)] {
        &self.inputs
    }
    /// Selected overrides only; unselected sentinel values are never materialized.
    pub fn forced_values(&self) -> &[(RealtimeChannelId, T)] {
        &self.forced
    }
    /// Looks up an ordinary input by declared channel identity.
    pub fn input(&self, channel: RealtimeChannelId) -> Option<&T> {
        self.inputs
            .iter()
            .find(|(id, _)| *id == channel)
            .map(|(_, value)| value)
    }
    /// Looks up a selected override by declared channel identity.
    pub fn forced(&self, channel: RealtimeChannelId) -> Option<&T> {
        self.forced
            .iter()
            .find(|(id, _)| *id == channel)
            .map(|(_, value)| value)
    }
    /// Every materialized payload requiring completion ownership.
    pub fn retained_payloads(&self) -> impl Iterator<Item = &T> {
        self.inputs
            .iter()
            .chain(&self.forced)
            .map(|(_, value)| value)
    }
    /// Whether ordered categorical diagnostics were requested.
    pub const fn retains_diagnostics(&self) -> bool {
        self.retain_diagnostics
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Portable host payload whose row-major geometry failed validation.
pub enum RealtimePayloadKind {
    /// Live input-side audio matrix.
    InputAudio,
    /// Optional generated-audio forcing matrix.
    ForcedAudio,
    /// Optional text forcing column.
    ForcedText,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Architecture-selected token domain used for a portable value.
pub enum RealtimeTokenKind {
    /// Text token domain.
    Text,
    /// Audio token domain.
    Audio,
}

fn validate_shape(
    payload: RealtimePayloadKind,
    actual: usize,
    rows: usize,
    columns: usize,
) -> Result<(), RealtimeIngressError> {
    let expected = rows
        .checked_mul(columns)
        .ok_or(RealtimeIngressError::ShapeOverflow { rows, columns })?;
    if actual == expected {
        Ok(())
    } else {
        Err(RealtimeIngressError::PayloadShape {
            payload,
            expected,
            actual,
        })
    }
}

fn validate_token(
    token: i32,
    domain: TokenDomain,
    kind: RealtimeTokenKind,
) -> Result<(), RealtimeIngressError> {
    let token = usize::try_from(token).map_err(|_| RealtimeIngressError::TokenDomain {
        kind,
        token,
        cardinality: domain.cardinality(),
    })?;
    if token < domain.cardinality() {
        Ok(())
    } else {
        Err(RealtimeIngressError::TokenDomain {
            kind,
            token: i32::try_from(token).unwrap_or(i32::MAX),
            cardinality: domain.cardinality(),
        })
    }
}

/// Invalid portable realtime input detected before opaque materialization.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeIngressError {
    /// Invalid channel declaration.
    #[error(transparent)]
    Channels(eredu_core::RealtimeChannelError),
    /// Realtime input must contain at least one batch row.
    #[error("realtime input batch must be positive")]
    EmptyBatch,
    /// A row-major payload shape overflowed.
    #[error("realtime payload shape {rows}x{columns} overflowed")]
    ShapeOverflow {
        /// Batch rows.
        rows: usize,
        /// Payload columns.
        columns: usize,
    },
    /// A row-major payload has the wrong number of values.
    #[error("realtime {payload:?} payload has {actual} values, expected {expected}")]
    PayloadShape {
        /// Affected payload.
        payload: RealtimePayloadKind,
        /// Required value count.
        expected: usize,
        /// Supplied value count.
        actual: usize,
    },
    /// A partial forcing mask was supplied without audio payloads.
    #[error("realtime generated-audio forcing mask has no payload")]
    ForcingMaskWithoutPayload,
    /// A generated-audio forcing mask has the wrong codebook count.
    #[error("realtime forcing mask has {actual} entries, expected {expected}")]
    ForcingMaskCount {
        /// Expected generated-codebook count.
        expected: usize,
        /// Supplied mask count.
        actual: usize,
    },
    /// A selected token is outside its exact zero-based domain.
    #[error("realtime {kind:?} token {token} is outside 0..{cardinality}")]
    TokenDomain {
        /// Text or audio domain.
        kind: RealtimeTokenKind,
        /// Invalid token value.
        token: i32,
        /// Exclusive domain end.
        cardinality: usize,
    },
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use eredu_core::{RealtimeFrameConvention, RealtimeInputFrame};

    use super::*;

    fn contract() -> RealtimeIngressContract {
        RealtimeIngressContract::new(
            RealtimeSpeechConfig::new(
                4,
                2,
                2,
                2,
                16,
                15,
                RealtimeFrameConvention::FeedbackAlignedHistory,
                vec![0, 1, 2, 1, 2],
            )
            .unwrap(),
            TokenDomain::new(17),
            TokenDomain::new(16),
        )
        .unwrap()
    }

    #[derive(Default)]
    struct Recorder(Vec<(Vec<i32>, [usize; 2])>);

    impl RealtimeHostTokenMaterializer for Recorder {
        type Tensor = usize;
        type Error = Infallible;

        fn materialize_i32(
            &mut self,
            values: &[i32],
            shape: [usize; 2],
        ) -> Result<Self::Tensor, Self::Error> {
            self.0.push((values.to_vec(), shape));
            Ok(self.0.len() - 1)
        }
    }

    #[test]
    fn validation_precedes_every_opaque_materialization() {
        let contract = contract();
        let invalid = RealtimeInputFrame::new(2, vec![1, 2, 3, 99]);
        let mut materializer = Recorder::default();
        assert!(matches!(
            contract.validate(&invalid),
            Err(RealtimeIngressError::TokenDomain { .. })
        ));
        assert!(materializer.0.is_empty());

        let valid = RealtimeInputFrame::new(2, vec![1, 2, 3, 4])
            .with_partially_forced_generated_audio(vec![5, 99, 6, 99], vec![true, false])
            .with_forced_text(vec![7, 8])
            .with_diagnostics();
        let validated = contract.validate(&valid).unwrap();
        assert_eq!(validated.forcing().generated_audio(), &[true, false]);
        let input = validated.materialize(&mut materializer).unwrap();
        assert_eq!(
            materializer.0,
            vec![
                (vec![1, 3], [2, 1]),
                (vec![2, 4], [2, 1]),
                (vec![7, 8], [2, 1]),
                (vec![5, 6], [2, 1]),
            ]
        );
        assert_eq!(input.input(RealtimeChannelId::new(3)), Some(&0));
        assert_eq!(input.input(RealtimeChannelId::new(4)), Some(&1));
        assert_eq!(input.forced(RealtimeChannelId::new(0)), Some(&2));
        assert_eq!(input.forced(RealtimeChannelId::new(1)), Some(&3));
        assert_eq!(input.forced(RealtimeChannelId::new(2)), None);
        assert_eq!(
            input.retained_payloads().copied().collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(input.retains_diagnostics());
    }

    #[test]
    fn shapes_masks_and_selected_domains_fail_closed() {
        let contract = contract();
        assert_eq!(
            contract.validate(&RealtimeInputFrame::new(0, vec![])).err(),
            Some(RealtimeIngressError::EmptyBatch)
        );
        assert!(matches!(
            contract.validate(&RealtimeInputFrame::new(2, vec![1, 2, 3])),
            Err(RealtimeIngressError::PayloadShape { .. })
        ));
        assert!(matches!(
            contract.validate(
                &RealtimeInputFrame::new(1, vec![1, 2])
                    .with_partially_forced_generated_audio(vec![3, 4], vec![true])
            ),
            Err(RealtimeIngressError::ForcingMaskCount { .. })
        ));
        assert!(matches!(
            contract.validate(&RealtimeInputFrame::new(1, vec![1, 2]).with_forced_text(vec![17])),
            Err(RealtimeIngressError::TokenDomain { .. })
        ));
    }
}
