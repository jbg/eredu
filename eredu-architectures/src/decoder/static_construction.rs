//! Shared static-module declarations, ordering and explicit construction storage.
//!
//! These borrowed declarations create no native module, source grant or admission
//! certificate. An arbitrary construction sink can allocate and run callbacks;
//! implementing the trait is never a fixed-storage or completeness assertion.
use super::{module_metadata::ModuleMetadata, StaticModuleSpec, StaticModules};
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_nn::{
    DistributedNeuralBackend, EmbeddingSpec, Error, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationScale, Tensor, VocabularyParallelRange,
    VocabularyRangeError,
};

pub(crate) mod format;
mod owned;
pub(super) use owned::from_config;
pub use format::{
    static_linear_format, StaticDeclarationError, StaticLinearFormatDeclaration,
    StaticParameterDeclaration,
};

/// An actual retained static-module specification with its names borrowed.
///
/// This view does not account for its owner's three Strings or for generated
/// configuration names/lookups. Callers must retain and bind that source.
#[derive(Debug, Clone, Copy)]
pub struct StaticModuleSpecView<'a> {
    /// Independent final normalization groups.
    pub normalization_groups: Option<i32>,
    /// Exact embedding primary identity.
    pub embedding_weight: &'a str,
    /// Exact final-normalization identity.
    pub normalization_weight: &'a str,
    /// Exact untied-head primary identity.
    pub head_weight: &'a str,
    /// Complete logical vocabulary rows.
    pub vocabulary: i32,
    /// Hidden feature width.
    pub hidden_size: i32,
    /// Final normalization epsilon.
    pub normalization_epsilon: f32,
    /// Scalar added to the learned normalization weight.
    pub normalization_offset: f32,
    /// Existing ordinary embedding encoding policy.
    pub embedding_quantization: Option<WeightQuantization>,
    /// Existing complete head encoding policy.
    pub head_format: LinearFormat,
    /// Whether readout reuses the embedding.
    pub tied_head: bool,
}

impl StaticModuleSpec {
    /// Borrows the actual complete specification without copying its names.
    pub fn borrowed(&self) -> StaticModuleSpecView<'_> {
        StaticModuleSpecView {
            normalization_groups: self.normalization_groups,
            embedding_weight: &self.embedding_weight,
            normalization_weight: &self.normalization_weight,
            head_weight: &self.head_weight,
            vocabulary: self.vocabulary,
            hidden_size: self.hidden_size,
            normalization_epsilon: self.normalization_epsilon,
            normalization_offset: self.normalization_offset,
            embedding_quantization: self.embedding_quantization,
            head_format: self.head_format,
            tied_head: self.tied_head,
        }
    }
}

/// Retained vocabulary ownership selected for this construction.
#[derive(Debug, Clone, Copy)]
pub enum StaticModulePlacement<'a> {
    /// Replicated ordinary embedding and head.
    Replicated,
    /// Exact existing embedding and optional output ownership.
    Vocabulary {
        /// Actual embedding range.
        embedding: &'a VocabularyParallelRange,
        /// Actual untied output range; tied outputs must omit it.
        output: Option<&'a VocabularyParallelRange>,
    },
}

/// Borrowed embedding declaration at its actual construction position.
#[derive(Debug, Clone, Copy)]
pub struct StaticEmbeddingDeclaration<'a> {
    /// Complete logical vocabulary rows.
    pub vocabulary: i32,
    /// Hidden width.
    pub dimensions: i32,
    /// Exact primary parameter identity.
    pub weight: &'a str,
    /// Exact checkpoint encoding selected by the architecture.
    pub format: LinearFormat,
    /// Validated local vocabulary ownership, if distributed.
    pub range: Option<&'a VocabularyParallelRange>,
}
impl StaticEmbeddingDeclaration<'_> {
    fn into_with(self, metadata: ModuleMetadata<'_>) -> Result<EmbeddingSpec, Error> {
        metadata.controls::<EmbeddingSpec>()?;
        Ok(EmbeddingSpec {
            vocabulary: self.vocabulary,
            dimensions: self.dimensions,
            weight: metadata.plain_parameter(self.weight)?,
            format: metadata.format(self.weight, self.format)?,
        })
    }
}

/// Borrowed final normalization declaration at its actual construction position.
#[derive(Debug, Clone, Copy)]
pub struct StaticNormalizationDeclaration<'a> {
    /// Independent normalization groups.
    pub groups: Option<i32>,
    /// Hidden width.
    pub dimensions: i32,
    /// Normalization epsilon.
    pub epsilon: f32,
    /// Exact learned scale identity.
    pub weight: &'a str,
    /// Learned scale offset; both signs of zero select ordinary Learned.
    pub offset: f32,
}
impl StaticNormalizationDeclaration<'_> {
    #[cfg(test)]
    fn into_ordinary(self) -> Result<NormalizationConstructionSpec, Error> {
        self.into_with(ModuleMetadata::ordinary())
    }
    fn into_with(
        self,
        metadata: ModuleMetadata<'_>,
    ) -> Result<NormalizationConstructionSpec, Error> {
        metadata.controls::<NormalizationConstructionSpec>()?;
        let weight = metadata.plain_parameter(self.weight)?;
        Ok(NormalizationConstructionSpec {
            groups: self.groups,
            dimensions: self.dimensions,
            epsilon: self.epsilon,
            scale: if self.offset == 0.0 {
                NormalizationScale::Learned(weight)
            } else {
                NormalizationScale::LearnedOffset {
                    weight,
                    offset: self.offset,
                }
            },
        })
    }
}

/// Borrowed untied output declaration at its actual construction position.
#[derive(Debug, Clone, Copy)]
pub struct StaticHeadDeclaration<'a> {
    /// Hidden input width.
    pub input: i32,
    /// Complete logical output vocabulary.
    pub output: i32,
    /// Exact primary parameter identity.
    pub weight: &'a str,
    /// Exact checkpoint encoding selected by the architecture.
    pub format: LinearFormat,
    /// Validated local output vocabulary ownership, if distributed.
    pub range: Option<&'a VocabularyParallelRange>,
}
impl StaticHeadDeclaration<'_> {
    fn into_with(self, metadata: ModuleMetadata<'_>) -> Result<LinearSpec, Error> {
        metadata.controls::<LinearSpec>()?;
        Ok(LinearSpec {
            input: self.input,
            output: self.output,
            weight: metadata.plain_parameter(self.weight)?,
            bias: None,
            format: metadata.format(self.weight, self.format)?,
        })
    }
}

/// Fixed failure selected by the shared construction-order worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StaticConstructionError {
    /// Actual range geometry or operator rows are invalid.
    #[error(transparent)]
    Vocabulary(#[from] VocabularyRangeError),
    /// A tied output declared independent ownership.
    #[error("tied decoder output must not declare separate vocabulary ownership")]
    TiedOutputRange,
    /// An untied distributed output omitted its ownership.
    #[error("untied decoder output is missing vocabulary ownership")]
    MissingOutputRange,
}
impl From<StaticConstructionError> for Error {
    fn from(cause: StaticConstructionError) -> Self {
        // Preserve the ordinary error-storage policy as well as its exact text.
        match cause {
            StaticConstructionError::Vocabulary(cause) => cause.into(),
            StaticConstructionError::TiedOutputRange => {
                Error::backend("tied decoder output must not declare separate vocabulary ownership")
            }
            StaticConstructionError::MissingOutputRange => {
                Error::backend("untied decoder output is missing vocabulary ownership")
            }
        }
    }
}

/// Storage/construction policy called by the one architecture declaration worker.
///
/// Implementations may allocate, call a backend or fail. Neither this public
/// trait nor a successful traversal grants original source, Q/B or fixed-storage
/// authority. A future strict consumer must bind its concrete storage policy.
pub trait StaticModuleConstruction<'a> {
    /// Retained embedding or concrete declaration result.
    type Embedding;
    /// Retained normalization or concrete declaration result.
    type Normalization;
    /// Retained head or concrete declaration result.
    type Linear;
    /// Owning failure with an explicit conversion for shared order checks.
    type Error: From<StaticConstructionError>;
    /// Stores a shared validation failure using this destination's error policy.
    fn construction_error(&mut self, cause: StaticConstructionError) -> Self::Error {
        cause.into()
    }
    /// Constructs the embedding at the first module position.
    fn embedding(
        &mut self,
        declaration: StaticEmbeddingDeclaration<'a>,
    ) -> Result<Self::Embedding, Self::Error>;
    /// Constructs final normalization after successful embedding construction.
    fn normalization(
        &mut self,
        declaration: StaticNormalizationDeclaration<'a>,
    ) -> Result<Self::Normalization, Self::Error>;
    /// Constructs the untied head after all applicable output-range checks.
    fn head(&mut self, declaration: StaticHeadDeclaration<'a>)
        -> Result<Self::Linear, Self::Error>;
}

/// Results in the same successful field-destruction order as StaticModules.
#[derive(Debug)]
pub struct StaticModuleParts<E, N, L> {
    /// Embedding owner, retired first when the complete result is dropped.
    pub embeddings: E,
    /// Final normalization owner, retired second.
    pub norm: N,
    /// Optional head owner, retired last.
    pub lm_head: Option<L>,
}

impl<'a> StaticModuleSpecView<'a> {
    /// Declares and constructs the selected modules in the actual ordinary order.
    /// Previous local owners retire in reverse order on failure. No request
    /// storage or backend completeness is inferred from sink success.
    pub fn construct_with<S: StaticModuleConstruction<'a>>(
        self,
        placement: StaticModulePlacement<'a>,
        sink: &mut S,
    ) -> Result<StaticModuleParts<S::Embedding, S::Normalization, S::Linear>, S::Error> {
        let embedding_range = match placement {
            StaticModulePlacement::Replicated => None,
            StaticModulePlacement::Vocabulary { embedding, .. } => {
                embedding
                    .validate_global_rows_fixed(self.vocabulary)
                    .map_err(|cause| sink.construction_error(cause.into()))?;
                Some(embedding)
            }
        };
        let embeddings = sink.embedding(StaticEmbeddingDeclaration {
            vocabulary: self.vocabulary,
            dimensions: self.hidden_size,
            weight: self.embedding_weight,
            format: self.embedding_quantization.into(),
            range: embedding_range,
        })?;
        let norm = sink.normalization(StaticNormalizationDeclaration {
            groups: self.normalization_groups,
            dimensions: self.hidden_size,
            epsilon: self.normalization_epsilon,
            weight: self.normalization_weight,
            offset: self.normalization_offset,
        })?;
        // Deliberately after both module constructions, as in from_parallel_spec.
        let output_range = match (placement, self.tied_head) {
            (StaticModulePlacement::Replicated, _) => None,
            (StaticModulePlacement::Vocabulary { output: None, .. }, true) => None,
            (
                StaticModulePlacement::Vocabulary {
                    output: Some(_), ..
                },
                true,
            ) => {
                return Err(sink.construction_error(StaticConstructionError::TiedOutputRange));
            }
            (StaticModulePlacement::Vocabulary { output: None, .. }, false) => {
                return Err(sink.construction_error(StaticConstructionError::MissingOutputRange));
            }
            (
                StaticModulePlacement::Vocabulary {
                    output: Some(range),
                    ..
                },
                false,
            ) => {
                range
                    .validate_global_rows_fixed(self.vocabulary)
                    .map_err(|cause| sink.construction_error(cause.into()))?;
                Some(range)
            }
        };
        let lm_head = if self.tied_head {
            None
        } else {
            Some(sink.head(StaticHeadDeclaration {
                input: self.hidden_size,
                output: self.vocabulary,
                weight: self.head_weight,
                format: self.head_format,
                range: output_range,
            })?)
        };
        Ok(StaticModuleParts {
            embeddings,
            norm,
            lm_head,
        })
    }
}

struct Ordinary<'a, B: NeuralBackend> {
    context: &'a <B::Tensor as Tensor>::Context,
    metadata: ModuleMetadata<'a>,
}
impl<'a, B: NeuralBackend> StaticModuleConstruction<'a> for Ordinary<'_, B> {
    type Embedding = B::Embedding;
    type Normalization = B::Normalization;
    type Linear = B::Linear;
    type Error = Error;
    fn construction_error(&mut self, cause: StaticConstructionError) -> Error {
        self.metadata.construction_error(cause)
    }
    fn embedding(
        &mut self,
        declaration: StaticEmbeddingDeclaration<'a>,
    ) -> Result<B::Embedding, Error> {
        B::embedding(declaration.into_with(self.metadata)?, self.context)
    }
    fn normalization(
        &mut self,
        declaration: StaticNormalizationDeclaration<'a>,
    ) -> Result<B::Normalization, Error> {
        B::normalization(declaration.into_with(self.metadata)?, self.context)
    }
    fn head(&mut self, declaration: StaticHeadDeclaration<'a>) -> Result<B::Linear, Error> {
        B::linear(declaration.into_with(self.metadata)?, self.context)
    }
}
struct Parallel<'a, B: DistributedNeuralBackend> {
    context: &'a <B::Tensor as Tensor>::Context,
    metadata: ModuleMetadata<'a>,
}
impl<'a, B: DistributedNeuralBackend> StaticModuleConstruction<'a> for Parallel<'_, B> {
    type Embedding = B::Embedding;
    type Normalization = B::Normalization;
    type Linear = B::Linear;
    type Error = Error;
    fn construction_error(&mut self, cause: StaticConstructionError) -> Error {
        self.metadata.construction_error(cause)
    }
    fn embedding(
        &mut self,
        declaration: StaticEmbeddingDeclaration<'a>,
    ) -> Result<B::Embedding, Error> {
        let spec = declaration.into_with(self.metadata)?;
        match declaration.range {
            Some(range) => B::vocabulary_parallel_embedding(spec, range.clone(), self.context),
            None => B::embedding(spec, self.context),
        }
    }
    fn normalization(
        &mut self,
        declaration: StaticNormalizationDeclaration<'a>,
    ) -> Result<B::Normalization, Error> {
        B::normalization(declaration.into_with(self.metadata)?, self.context)
    }
    fn head(&mut self, declaration: StaticHeadDeclaration<'a>) -> Result<B::Linear, Error> {
        let spec = declaration.into_with(self.metadata)?;
        match declaration.range {
            Some(range) => B::vocabulary_parallel_linear(spec, range.clone(), self.context),
            None => B::linear(spec, self.context),
        }
    }
}

pub(super) fn ordinary<B: NeuralBackend>(
    spec: &StaticModuleSpec,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<StaticModules<B>, Error> {
    let metadata = ModuleMetadata::new::<B>(context);
    metadata.controls::<(
        StaticModules<B>,
        StaticModuleParts<B::Embedding, B::Normalization, B::Linear>,
        StaticModuleSpecView<'_>,
        StaticModulePlacement<'_>,
        StaticEmbeddingDeclaration<'_>,
        StaticNormalizationDeclaration<'_>,
        StaticHeadDeclaration<'_>,
        Result<StaticModuleParts<B::Embedding, B::Normalization, B::Linear>, Error>,
        StaticConstructionError,
        Ordinary<'_, B>,
    )>()?;
    let parts = spec.borrowed().construct_with(
        StaticModulePlacement::Replicated,
        &mut Ordinary::<B> { context, metadata },
    )?;
    Ok(StaticModules {
        embeddings: parts.embeddings,
        norm: parts.norm,
        lm_head: parts.lm_head,
    })
}
pub(super) fn parallel<B: DistributedNeuralBackend>(
    spec: &StaticModuleSpec,
    embedding: &VocabularyParallelRange,
    output: Option<&VocabularyParallelRange>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<StaticModules<B>, Error> {
    let metadata = ModuleMetadata::new::<B>(context);
    metadata.controls::<(
        StaticModules<B>,
        StaticModuleParts<B::Embedding, B::Normalization, B::Linear>,
        StaticModuleSpecView<'_>,
        StaticModulePlacement<'_>,
        StaticEmbeddingDeclaration<'_>,
        StaticNormalizationDeclaration<'_>,
        StaticHeadDeclaration<'_>,
        Result<StaticModuleParts<B::Embedding, B::Normalization, B::Linear>, Error>,
        StaticConstructionError,
        Parallel<'_, B>,
        VocabularyParallelRange,
        Option<VocabularyParallelRange>,
    )>()?;
    let parts = spec.borrowed().construct_with(
        StaticModulePlacement::Vocabulary { embedding, output },
        &mut Parallel::<B> { context, metadata },
    )?;
    Ok(StaticModules {
        embeddings: parts.embeddings,
        norm: parts.norm,
        lm_head: parts.lm_head,
    })
}

#[cfg(test)]
mod tests;
