//! Portable realtime-frame validation before opaque token materialization.

use eredu_core::{RealtimeFrameForcing, RealtimeInputFrame, RealtimeSpeechConfig};

use crate::TokenDomain;

/// Exact schedule and token domains used to validate portable realtime input.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeIngressContract {
    schedule: RealtimeSpeechConfig,
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
        Ok(Self {
            schedule,
            text,
            audio,
        })
    }

    /// Returns the exact normalized speech schedule.
    pub const fn schedule(&self) -> &RealtimeSpeechConfig {
        &self.schedule
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
        let input_audio = materializer.materialize_i32(
            self.frame.input_audio_tokens(),
            [batch, self.contract.schedule.input_audio_codebooks()],
        )?;
        let forced_audio = self
            .frame
            .forced_generated_audio_tokens()
            .map(|tokens| {
                materializer.materialize_i32(
                    tokens,
                    [batch, self.contract.schedule.generated_audio_codebooks()],
                )
            })
            .transpose()?;
        let forced_text = self
            .frame
            .forced_text_tokens()
            .map(|tokens| materializer.materialize_i32(tokens, [batch, 1]))
            .transpose()?;
        Ok(MaterializedRealtimeInput {
            schedule: self.contract.schedule.clone(),
            batch,
            input_audio,
            forced_audio,
            forced_text,
            forcing: self.forcing.clone(),
            retain_diagnostics: self.frame.retains_diagnostics(),
        })
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

/// Opaque input tensors paired with the neutral forcing and diagnostic policy.
pub struct MaterializedRealtimeInput<T> {
    schedule: RealtimeSpeechConfig,
    batch: usize,
    input_audio: T,
    forced_audio: Option<T>,
    forced_text: Option<T>,
    forcing: RealtimeFrameForcing,
    retain_diagnostics: bool,
}

impl<T> MaterializedRealtimeInput<T> {
    /// Returns the exact schedule under which host validation completed.
    pub const fn schedule(&self) -> &RealtimeSpeechConfig {
        &self.schedule
    }

    /// Returns the validated positive batch dimension.
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Returns input-side audio tokens in batch-by-input-codebook shape.
    pub const fn input_audio(&self) -> &T {
        &self.input_audio
    }

    /// Returns optional forced generated-audio tokens.
    pub const fn forced_audio(&self) -> Option<&T> {
        self.forced_audio.as_ref()
    }

    /// Returns optional forced text tokens.
    pub const fn forced_text(&self) -> Option<&T> {
        self.forced_text.as_ref()
    }

    /// Returns the neutral forcing mask.
    pub const fn forcing(&self) -> &RealtimeFrameForcing {
        &self.forcing
    }

    /// Returns whether ordered logits diagnostics were requested.
    pub const fn retains_diagnostics(&self) -> bool {
        self.retain_diagnostics
    }

    /// Consumes opaque tensors and neutral policy.
    pub fn into_parts(
        self,
    ) -> (
        RealtimeSpeechConfig,
        usize,
        T,
        Option<T>,
        Option<T>,
        RealtimeFrameForcing,
        bool,
    ) {
        (
            self.schedule,
            self.batch,
            self.input_audio,
            self.forced_audio,
            self.forced_text,
            self.forcing,
            self.retain_diagnostics,
        )
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
        assert_eq!(materializer.0.len(), 3);
        assert_eq!(materializer.0[0].1, [2, 2]);
        assert_eq!(materializer.0[1].1, [2, 2]);
        assert_eq!(materializer.0[2].1, [2, 1]);
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
