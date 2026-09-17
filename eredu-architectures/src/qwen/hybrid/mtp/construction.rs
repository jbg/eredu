//! Prediction declarations retained once beside their exact source task rows.
use super::*;
use crate::decoder::ModuleMetadata;
#[derive(Debug, Clone, Copy)]
pub(crate) struct PredictionSharedSpec {
    hidden_size: i32,
    rms_norm_eps: f32,
}
impl PredictionSharedSpec {
    pub(crate) const fn new(hidden_size: i32, rms_norm_eps: f32) -> Self {
        Self {
            hidden_size,
            rms_norm_eps,
        }
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionShared<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, &Self, PredictionShared<B>)>()?;
        PredictionShared::from_dimensions(self.hidden_size, self.rms_norm_eps, context)
    }
}
#[derive(Debug)]
pub(crate) struct PredictionUnitSpec {
    block: super::super::block::PredictionBlockSpec,
    experts: i32,
}
impl PredictionUnitSpec {
    pub(crate) fn new(config: &HybridConfig, depth: usize) -> Result<Self, Error> {
        Ok(Self {
            block: super::super::block::PredictionBlockSpec::new(config, depth)?,
            experts: config.num_experts,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionUnit<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, &Self, PredictionUnit<B>)>()?;
        Ok(PredictionUnit {
            block: self.block.instantiate::<B>(context)?,
            experts: self.experts,
        })
    }
}

#[cfg(test)]
mod tests;
