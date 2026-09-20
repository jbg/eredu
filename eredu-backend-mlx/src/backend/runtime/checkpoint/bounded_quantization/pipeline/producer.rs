//! Submission ownership for the common bounded tile window.
use super::*;
use crate::backend::submission_recovery::native_role::cold;
use std::convert::Infallible;

pub(in super::super) trait TileCompletion {
    fn materialization(&self) -> &WeightMaterialization;
    /// Called only after successful wait and writeback. Error and abandoned
    /// queue entries retain their resources through their ordinary Drop path.
    fn retire(self) -> Result<(), Error>;
}

impl TileCompletion for WeightMaterialization {
    fn materialization(&self) -> &WeightMaterialization {
        self
    }
    fn retire(self) -> Result<(), Error> {
        // Ordinary tile teardown continues through its existing recovery owner.
        drop(self);
        Ok(())
    }
}

impl<I: 'static> TileCompletion for cold::Submission<I, WeightMaterialization, Infallible> {
    fn materialization(&self) -> &WeightMaterialization {
        match self.result() {
            Ok(value) => value,
            Err(never) => match *never {},
        }
    }
    fn retire(self) -> Result<(), Error> {
        let materialization = match self.finish().map_err(Error::StorageSource)? {
            Ok(value) => value,
            Err(never) => match never {},
        };
        materialization.finish()?;
        // Native owner destruction can defer account-only Rust payloads under
        // the runtime guard. Retire those on the host before reusing a tile slot.
        safemlx::reclaim_allocation_owners();
        Ok(())
    }
}

pub(in super::super) trait TileProducer {
    type Completion: TileCompletion;
    type Error: From<Error> + From<eredu_checkpoint::recipe::RecipeError>;

    fn submit(
        &mut self,
        source: &dyn CheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<Self::Completion, Self::Error>;
}

pub(super) struct OrdinaryTileProducer(
    pub(super) [MlxParameterMaterializationContext; BOUNDED_QUANTIZATION_TILE_BUFFERS],
);

impl TileProducer for OrdinaryTileProducer {
    type Completion = WeightMaterialization;
    type Error = Error;

    fn submit(
        &mut self,
        source: &dyn CheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<Self::Completion, Error> {
        let context = &self.0[slot];
        let pending = recipe.prepare_borrowed_materialization(source, context)?;
        let (dense, source_leases) = pending.into_parts();
        let mut prepared = WeightMaterialization::prepare_retained(vec![dense], source_leases)?;
        prepare_quantized_outputs(&mut prepared, quantization, target, context.source_stream())?;
        Ok(prepared.submit_prepared_outputs()?)
    }
}

mod cpu_resources;
mod encoded_affine;

#[cfg(test)]
mod tests;
