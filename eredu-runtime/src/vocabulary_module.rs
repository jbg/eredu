use eredu_nn::{
    DistributedNeuralBackend, EmbeddingSpec, Error, LinearSpec, Tensor, VocabularyParallelRange,
};

/// Independent ownership and placement of a declared vocabulary resource.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum VocabularyModulePlacement {
    /// The selected component does not own this resource.
    Omitted,
    /// Materialize the complete resource on this owner.
    Local,
    /// Materialize the declared contiguous vocabulary rows.
    VocabularyParallel(VocabularyParallelRange),
}

impl VocabularyModulePlacement {
    /// Checks selected ownership against the operator's global row count.
    ///
    /// Components can validate all interacting resources before allocating any.
    pub fn validate_rows(&self, rows: i32) -> Result<(), Error> {
        if let Self::VocabularyParallel(range) = self {
            range.validate_global_rows(rows)?;
        }
        Ok(())
    }

    /// Materializes only the selected embedding resource through the backend.
    pub fn materialize_embedding<B: DistributedNeuralBackend>(
        self,
        spec: EmbeddingSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Embedding>, Error> {
        self.validate_rows(spec.vocabulary)?;
        match self {
            Self::Omitted => Ok(None),
            Self::Local => B::embedding(spec, context).map(Some),
            Self::VocabularyParallel(range) => {
                B::vocabulary_parallel_embedding(spec, range, context).map(Some)
            }
        }
    }

    /// Materializes only the selected output projection through the backend.
    pub fn materialize_linear<B: DistributedNeuralBackend>(
        self,
        spec: LinearSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Linear>, Error> {
        self.validate_rows(spec.output)?;
        match self {
            Self::Omitted => Ok(None),
            Self::Local => B::linear(spec, context).map(Some),
            Self::VocabularyParallel(range) => {
                B::vocabulary_parallel_linear(spec, range, context).map(Some)
            }
        }
    }
}
