//! Request-bounded media preparation over shared Qwen ingress and rotary utilities.
//! This produces ordinary target inputs; it does not own a generation/state driver.
use super::{
    prepared::PreparedVision,
    target::{TargetInput, TargetSpec},
};
use crate::{
    media_plan::{AdmittedCompositeInput, QwenInputPartPlan},
    qwen::{ingress, vl::InputPart},
};
use eredu_nn::{Error, Index, RotaryPosition, Tensor};
use eredu_runtime::OriginalTokenIds;

/// Invalid provenance, bounded geometry, or unsupported prepared-input contract.
#[derive(Debug, thiserror::Error)]
pub enum MediaInputError {
    /// Shared shape and modality admission failed.
    #[error(transparent)]
    Admission(#[from] eredu_core::CapabilityError),
    /// Required explicit metadata is absent or malformed.
    #[error(transparent)]
    Metadata(#[from] eredu_core::PreparedInputError),
    /// Original integer identities are invalid.
    #[error(transparent)]
    Tokens(#[from] eredu_runtime::TokenInputError),
    /// Request differs from the retained preparation contract.
    #[error("invalid qwen4_exp media input: {0}")]
    Geometry(&'static str),
    /// Native tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] Error),
}

pub(crate) mod admission;
pub mod processor;
pub use admission::{AdmittedMediaInput, MediaAdmissionConfig, MediaInputPartPlan};

/// Retained tower source coupled to a source-independent media admission policy.
#[derive(Clone)]
pub struct MediaIngress {
    vision: PreparedVision,
    policy: MediaAdmissionConfig,
}
impl MediaIngress {
    pub(crate) fn new(spec: &TargetSpec, vision: PreparedVision) -> Result<Self, MediaInputError> {
        let policy = MediaAdmissionConfig::new(spec, vision.config(), vision.media_tokens())?;
        Ok(Self { vision, policy })
    }
    /// Retained shared tower parameters, including independently addressable blocks.
    pub fn vision(&self) -> &PreparedVision {
        &self.vision
    }
    /// Source-free geometry, identity and limits for processor admission.
    pub fn admission_config(&self) -> &MediaAdmissionConfig {
        &self.policy
    }
    pub(crate) fn identity(&self) -> &str {
        &self.policy.identity
    }
}

impl MediaAdmissionConfig {
    /// Derives source-independent media admission from exact retained geometry.
    /// No checkpoint, payload reader, device or backend context is required.
    pub fn new(
        spec: &TargetSpec,
        vision: &crate::qwen::vision::VisionConfig,
        media: &crate::qwen4_exp::config::MediaTokens,
    ) -> Result<Self, MediaInputError> {
        spec.limits
            .validate_for_config(&spec.config)
            .map_err(|_| MediaInputError::Geometry("invalid target invocation/history limits"))?;
        vision
            .validate_for(crate::qwen::vision::VisionMode::DeepStack)
            .map_err(|_| MediaInputError::Geometry("invalid vision geometry"))?;
        if vision.deepstack_layer_count() != 0 {
            return Err(MediaInputError::Geometry(
                "DeepStack injection is not declared",
            ));
        }
        if vision.out_hidden_size != spec.config.hidden_size {
            return Err(MediaInputError::Geometry(
                "vision output width differs from target",
            ));
        }
        if let Some(expected) = &spec.config.vision {
            let mut expected = expected.clone();
            let mut actual = vision.clone();
            expected.linear_formats.clear();
            actual.linear_formats.clear();
            if expected != actual {
                return Err(MediaInputError::Geometry(
                    "vision geometry differs from target",
                ));
            }
        }
        let ids = media;
        let token_ids = [ids.image, ids.video, ids.start, ids.end];
        if token_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != 4
            || token_ids
                .iter()
                .any(|id| *id >= spec.config.vocabulary as u32)
            || spec.config.media.as_ref().is_some_and(|m| {
                [m.image, m.video, m.start, m.end] != [ids.image, ids.video, ids.start, ids.end]
            })
        {
            return Err(MediaInputError::Geometry(
                "media token identities differ from target",
            ));
        }
        let rope = &spec.config.attention;
        if rope.rotary.algorithm != eredu_nn::RotaryAlgorithm::Default {
            return Err(MediaInputError::Geometry(
                "media rotary scaling is not implemented",
            ));
        }
        let sections: [i32; 3] = rope
            .rope
            .get("mrope_section")
            .and_then(|s| serde_json::from_value(s.clone()).ok())
            .unwrap_or([11, 11, 10]);
        if sections.iter().any(|s| *s <= 0)
            || sections.iter().map(|s| i64::from(*s)).sum::<i64>() * 2
                != i64::from(rope.rotary.dimensions)
        {
            return Err(MediaInputError::Geometry(
                "media rotary sections do not cover the rotary width",
            ));
        }
        let mut vision_config = vision.clone();
        vision_config.linear_formats.clear();
        Ok(Self {
            identity: eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
                "qwen4_exp.media.v1",
                [
                    ("target", spec.geometry_fingerprint()),
                    (
                        "vision",
                        crate::qwen::vision::prompt_cache_architecture_fingerprint(vision),
                    ),
                    ("tokens", format!("{:?}", media)),
                ],
            ),
            vision: vision_config,
            media: media.clone(),
            hidden: spec.config.hidden_size,
            vocabulary: spec.config.vocabulary,
            // Media ingress has one sequence lane. Its owned IDs, rotary products,
            // and encoder output span the full request; target work is chunked.
            max_tokens: spec.limits.history_tokens as usize,
            maximum_chunk_tokens: (spec.limits.qsa.tokens as usize)
                .min(spec.limits.invocation_tokens),
            rotary_width: rope.rotary.dimensions,
            theta: rope.rotary.base,
            sections,
        })
    }
    /// Geometry identity used by requests and conditional execution plans.
    pub fn identity(&self) -> &str {
        &self.identity
    }
}

/// Exact admitted request, with original IDs independent from projected media.
#[derive(Clone)]
pub struct PreparedMediaInput<T> {
    policy: String,
    prepared: ingress::PreparedInput<T>,
    admitted: AdmittedCompositeInput<QwenInputPartPlan>,
    ids: T,
    cosine: T,
    sine: T,
    delta: i32,
    hidden: i32,
    maximum_chunk_tokens: usize,
}

/// Semantic roots owned by the prepared Flash-Next media request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedMediaTensorRole {
    /// Shared ingress assembly, including patches and per-part token IDs.
    Ingress(ingress::PreparedInputTensorRole),
    /// Canonical original IDs across the complete request.
    TokenIds,
    /// Target rotary cosine products across the complete request.
    RotaryCosine,
    /// Target rotary sine products across the complete request.
    RotarySine,
}
impl<T: Tensor> PreparedMediaInput<T> {
    pub(crate) fn validate_policy(&self, ingress: &MediaIngress) -> Result<(), MediaInputError> {
        ingress.policy.validate_request_bounds(
            self.admitted.decoder_positions(),
            self.maximum_chunk_tokens,
        )?;
        if self.policy != ingress.policy.identity {
            return Err(MediaInputError::Geometry(
                "prepared media input belongs to a different admission policy",
            ));
        }
        Ok(())
    }
    /// All native request values needed by deferred vision/target execution.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.storage_values().map(|(_, value)| value)
    }
    /// Named request roots for a combined storage survey. An enclosing cursor
    /// also retains the original processor input and any completed encoder output.
    pub fn storage_values(&self) -> impl Iterator<Item = (PreparedMediaTensorRole, &T)> {
        self.prepared
            .storage_values()
            .map(|(role, value)| (PreparedMediaTensorRole::Ingress(role), value))
            .chain([
                (PreparedMediaTensorRole::TokenIds, &self.ids),
                (PreparedMediaTensorRole::RotaryCosine, &self.cosine),
                (PreparedMediaTensorRole::RotarySine, &self.sine),
            ])
    }
    /// Original request IDs before embedding replacement.
    pub fn token_ids(&self) -> &T {
        &self.ids
    }
    /// Admitted geometry and original-input identity for the exact request.
    pub fn admission(&self) -> &AdmittedCompositeInput<QwenInputPartPlan> {
        &self.admitted
    }
    /// Supplies flattened patches and grids to the existing shared vision tower.
    /// Projected-only and text-only requests have no tower work.
    pub fn with_vision_input<R>(
        &self,
        apply: impl FnOnce(Option<crate::qwen::vision::VisionInput<'_, T>>) -> R,
    ) -> R {
        self.prepared.with_parts(|parts, pixels| {
            let grids: Vec<_> = parts
                .iter()
                .flat_map(|p| match p {
                    InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => *grid,
                    _ => &[],
                })
                .copied()
                .collect();
            apply(pixels.map(|pixels| crate::qwen::vision::VisionInput {
                pixels,
                grid: &grids,
            }))
        })
    }
    /// Assembles one bounded target span from an encoder result retained once for
    /// the request. Only intersecting text IDs are embedded; media and rotary
    /// products are sliced at exact absolute positions, including inside a frame.
    pub fn assemble(
        &self,
        range: std::ops::Range<i32>,
        vision: Option<&T>,
        mut embed: impl FnMut(&T) -> Result<T, Error>,
        context: &T::Context,
    ) -> Result<MediaTargetInput<T>, MediaInputError> {
        if range.start < 0 || range.start >= range.end || range.end > self.ids.dim(1) {
            return Err(MediaInputError::Geometry(
                "target span is outside prepared request",
            ));
        }
        if (range.end - range.start) as usize > self.maximum_chunk_tokens {
            return Err(MediaInputError::Geometry(
                "target span exceeds admitted assembly bound",
            ));
        }
        self.prepared.with_parts(|parts, _| {
            let media_tokens = parts
                .iter()
                .map(|p| match p {
                    InputPart::Image { tokens, .. } | InputPart::Video { tokens, .. } => {
                        tokens.dim(1)
                    }
                    _ => 0,
                })
                .sum::<i32>();
            if match vision {
                Some(v) => v.shape() != [1, media_tokens, self.hidden],
                None => media_tokens != 0,
            } {
                return Err(MediaInputError::Geometry(
                    "projected tower output differs from admitted media span",
                ));
            }
            let mut offset = 0;
            let mut token_offset = 0;
            let capacity = parts.len().min((range.end - range.start) as usize);
            let mut values = Vec::with_capacity(capacity);
            let mut tokens = Vec::with_capacity(capacity);
            for part in parts {
                let ids = match part {
                    InputPart::Text(ids)
                    | InputPart::Projected { tokens: ids, .. }
                    | InputPart::Image { tokens: ids, .. }
                    | InputPart::Video { tokens: ids, .. } => *ids,
                };
                let part_end = token_offset + ids.dim(1);
                let start = range.start.max(token_offset);
                let end = range.end.min(part_end);
                if start < end {
                    let local_start = start - token_offset;
                    let local_end = end - token_offset;
                    let ids = self
                        .ids
                        .index(&[Index::Full, Index::Range(start, end)], context)?;
                    let value = match part {
                        InputPart::Text(_) => embed(&ids)?,
                        InputPart::Projected { embeddings, .. } => embeddings.index(
                            &[
                                Index::Full,
                                Index::Range(local_start, local_end),
                                Index::Full,
                            ],
                            context,
                        )?,
                        InputPart::Image { .. } | InputPart::Video { .. } => {
                            vision.expect("validated output").index(
                                &[
                                    Index::Full,
                                    Index::Range(offset + local_start, offset + local_end),
                                    Index::Full,
                                ],
                                context,
                            )?
                        }
                    };
                    tokens.push(ids);
                    values.push(value);
                }
                if matches!(part, InputPart::Image { .. } | InputPart::Video { .. }) {
                    offset += ids.dim(1);
                }
                token_offset = part_end;
            }
            let ordered = tokens
                .iter()
                .zip(&values)
                .map(|(ids, value)| eredu_nn::multimodal::OrderedInputPart {
                    token_ids: ids,
                    embeddings: value,
                })
                .collect::<Vec<_>>();
            let assembled =
                eredu_nn::multimodal::assemble_ordered_inputs(&ordered, self.hidden, context)?;
            Ok(MediaTargetInput {
                // A one-segment concatenation may preserve a view on some backends.
                // Returned target spans must not retain the whole media allocation.
                ids: assembled.token_ids.compact(context)?,
                embeddings: assembled.embeddings.compact(context)?,
                cosine: self
                    .cosine
                    .index(
                        &[
                            Index::Full,
                            Index::Range(range.start, range.end),
                            Index::Full,
                        ],
                        context,
                    )?
                    .compact(context)?,
                sine: self
                    .sine
                    .index(
                        &[
                            Index::Full,
                            Index::Range(range.start, range.end),
                            Index::Full,
                        ],
                        context,
                    )?
                    .compact(context)?,
                delta: self.delta,
            })
        })
    }
}

/// Owned original IDs, replaced embeddings and absolute rotary products for one span.
/// The target commits the declared rotary delta in its ordinary fixed state,
/// so subsequent decode, snapshots and restore do not depend on this object.
#[derive(Clone)]
pub struct MediaTargetInput<T> {
    ids: T,
    embeddings: T,
    cosine: T,
    sine: T,
    delta: i32,
}
impl<T: Tensor> MediaTargetInput<T> {
    /// Persisted difference between next rotary position and causal token count.
    pub fn rotary_delta(&self) -> i32 {
        self.delta
    }
    /// Exact concatenated original token IDs.
    pub fn token_ids(&self) -> &T {
        &self.ids
    }
    /// Borrows through the existing target driver without re-embedding media.
    pub fn with_target_input<R>(&self, apply: impl FnOnce(TargetInput<'_, T>) -> R) -> R {
        apply(TargetInput {
            position_delta: Some(self.delta),
            ids: Some(OriginalTokenIds::Tensor(&self.ids)),
            batch: 1,
            tokens: self.ids.dim(1),
            embeddings: Some(&self.embeddings),
            visible: None,
            rotary: Some(RotaryPosition::Embeddings {
                cosine: &self.cosine,
                sine: &self.sine,
            }),
        })
    }
    /// Copies one bounded contiguous prefill span, retaining its absolute media products.
    pub fn slice(
        &self,
        range: std::ops::Range<i32>,
        context: &T::Context,
    ) -> Result<Self, MediaInputError> {
        if range.start < 0 || range.start >= range.end || range.end > self.ids.dim(1) {
            return Err(MediaInputError::Geometry(
                "prefill slice is outside prepared request",
            ));
        }
        Ok(Self {
            ids: self
                .ids
                .index(
                    &[Index::Full, Index::Range(range.start, range.end)],
                    context,
                )?
                .compact(context)?,
            embeddings: self
                .embeddings
                .index(
                    &[
                        Index::Full,
                        Index::Range(range.start, range.end),
                        Index::Full,
                    ],
                    context,
                )?
                .compact(context)?,
            cosine: self
                .cosine
                .index(
                    &[
                        Index::Full,
                        Index::Range(range.start, range.end),
                        Index::Full,
                    ],
                    context,
                )?
                .compact(context)?,
            sine: self
                .sine
                .index(
                    &[
                        Index::Full,
                        Index::Range(range.start, range.end),
                        Index::Full,
                    ],
                    context,
                )?
                .compact(context)?,
            delta: self.delta,
        })
    }
}
