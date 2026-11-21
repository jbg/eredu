//! Portable target capture and operation contracts for external drafting.

use eredu_core::AttentionPolicy;

/// Exact external-assistant capture requested from one ordinary target pass.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExternalPredictionCaptureRequest {
    /// Final decoder hidden state and every architecture-published shared K/V class.
    SharedAttention {
        /// Exact final decoder unit-output observation path.
        final_hidden_path: String,
    },
    /// Ordered decoder-unit outputs consumed by an assistant.
    OrderedLayerOutputs {
        /// Exact zero-based decoder layers, in assistant encoder order.
        target_layers: Box<[usize]>,
        /// Exact unit-output observation paths in the same order.
        target_paths: Box<[String]>,
    },
}

/// Architecture-owned values captured from one committed ordinary target pass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExternalPredictionTargetCapture<T> {
    /// Target hidden state plus shared attention publications.
    HiddenWithSharedAttention {
        /// Final decoder activation before vocabulary projection.
        hidden: T,
        /// Shared K/V values keyed by their exact attention policy.
        shared_kv: Vec<(AttentionPolicy, T, T)>,
    },
    /// Decoder outputs in the requested target-layer order.
    OrderedLayerOutputs {
        /// Exact ordered target-layer activations.
        target_states: Vec<T>,
    },
}

/// Target-owned static operation needed by an external assistant.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExternalPredictionTargetOperation<'a, T> {
    /// Applies the ordinary target token embedding to assistant proposal IDs.
    TokenEmbeddings(&'a T),
    /// Applies the ordinary target vocabulary projection to assistant states.
    ProjectLogits(&'a T),
    /// Looks up the target token table without input normalization or scaling.
    RawTokenEmbeddings(&'a T),
    /// Applies the target vocabulary head without final normalization, scaling or capping.
    /// Tensor-parallel vocabulary shards are gathered before returning.
    RawVocabularyProjection(&'a T),
}

/// Semantic identity of a target-owned static equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExternalPredictionTargetOperationKind {
    /// Ordinary normalized token embeddings.
    TokenEmbeddings,
    /// Ordinary final-normalized and transformed readout.
    ProjectLogits,
    /// Raw token table lookup.
    RawTokenEmbeddings,
    /// Raw vocabulary matrix product.
    RawVocabularyProjection,
}
impl ExternalPredictionTargetOperationKind {
    /// Whether this equation consumes token IDs rather than decoder hidden states.
    pub const fn is_lookup(self) -> bool {
        matches!(self, Self::TokenEmbeddings | Self::RawTokenEmbeddings)
    }
}
impl<T> ExternalPredictionTargetOperation<'_, T> {
    /// Stable equation identity independent of native tensor representation.
    pub const fn kind(&self) -> ExternalPredictionTargetOperationKind {
        match self {
            Self::TokenEmbeddings(_) => ExternalPredictionTargetOperationKind::TokenEmbeddings,
            Self::ProjectLogits(_) => ExternalPredictionTargetOperationKind::ProjectLogits,
            Self::RawTokenEmbeddings(_) => {
                ExternalPredictionTargetOperationKind::RawTokenEmbeddings
            }
            Self::RawVocabularyProjection(_) => {
                ExternalPredictionTargetOperationKind::RawVocabularyProjection
            }
        }
    }
    /// Request tensor whose side-effect-free geometry resolves the static plan.
    pub fn input(&self) -> &T {
        match self {
            Self::TokenEmbeddings(v)
            | Self::ProjectLogits(v)
            | Self::RawTokenEmbeddings(v)
            | Self::RawVocabularyProjection(v) => v,
        }
    }
}

impl<T> Copy for ExternalPredictionTargetOperation<'_, T> {}
impl<T> Clone for ExternalPredictionTargetOperation<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
