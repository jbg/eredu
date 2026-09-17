//! Ordinary sampler ownership; speculative policies keep their existing owner.

use super::*;
use eredu_runtime::working_memory::{
    BorrowedFundedSampler, PreparedRunSample, RunOwnedTextSampler, WorkingMemoryError,
};

pub(in crate::composition::mlx::session) enum MlxOrdinarySampler {
    Unquoted(MlxTextSampler),
    Funded(RunOwnedTextSampler),
}

impl MlxOrdinarySampler {
    pub(in crate::composition::mlx::session) fn as_sampler(&self) -> &MlxTextSampler {
        match self {
            Self::Unquoted(sampler) => sampler,
            Self::Funded(sampler) => sampler.as_sampler(),
        }
    }

    pub(in crate::composition::mlx::session) fn is_funded(&self) -> bool {
        matches!(self, Self::Funded(_))
    }

    pub(in crate::composition::mlx::session) fn borrow_funded(
        &self,
    ) -> Result<BorrowedFundedSampler<'_>, WorkingMemoryError> {
        match self {
            Self::Unquoted(_) => Err(WorkingMemoryError::UnknownBound),
            Self::Funded(sampler) => Ok(sampler.borrow_funded()),
        }
    }

    pub(in crate::composition::mlx::session) fn prepare_copy(
        &self,
    ) -> Result<
        eredu_runtime::generation::SamplerCopyPlan<'_>,
        eredu_runtime::generation::SamplerCopyError,
    > {
        self.as_sampler().prepare_copy()
    }

    pub(in crate::composition::mlx::session) fn prepare_sample(
        &mut self,
    ) -> Result<PreparedOrdinarySampler<'_>, Error> {
        match self {
            Self::Unquoted(sampler) => Ok(PreparedOrdinarySampler::Unquoted(sampler)),
            Self::Funded(sampler) => sampler
                .prepare_sample()
                .map(|prepared| PreparedOrdinarySampler::Funded(prepared))
                .map_err(|error| Error::Other(Box::new(error))),
        }
    }

    #[cfg(test)]
    pub(in crate::composition::mlx::session) fn history_capacity(&self) -> usize {
        self.as_sampler().history_capacity()
    }

    #[cfg(test)]
    pub(in crate::composition::mlx::session) fn history_len(&self) -> usize {
        self.as_sampler().history_len()
    }
}

pub(in crate::composition::mlx::session) enum PreparedOrdinarySampler<'a> {
    Unquoted(&'a mut MlxTextSampler),
    Funded(PreparedRunSample<'a>),
}

impl PreparedOrdinarySampler<'_> {
    pub(in crate::composition::mlx::session) fn sample(
        self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        self.sample_with::<MlxSamplingBackend>(logits, temperature, random, stream)
    }
    pub(in crate::composition::mlx::session) fn sample_with<B>(
        self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        context: &B::Context,
    ) -> Result<MlxTensor, B::Error>
    where
        B: eredu_runtime::SamplingBackend<
            Logits = MlxTensor,
            Token = MlxTensor,
            RandomState = RandomState,
        >,
    {
        match self {
            Self::Unquoted(sampler) => {
                Sampler::<B>::sample(sampler, logits, temperature, random, context)
            }
            Self::Funded(prepared) => prepared.sample::<B>(logits, temperature, random, context),
        }
    }
}
