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

    /// Return the completion owner and its validated logical input byte count.
    fn submit(
        &mut self,
        source: &eredu_checkpoint::store::RetainedCheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<(Self::Completion, u64), Self::Error>;
}

pub(super) struct OrdinaryTileProducer(
    pub(super) [MlxParameterMaterializationContext; BOUNDED_QUANTIZATION_TILE_BUFFERS],
);

impl TileProducer for OrdinaryTileProducer {
    type Completion = WeightMaterialization;
    type Error = Error;

    fn submit(
        &mut self,
        source: &eredu_checkpoint::store::RetainedCheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<(Self::Completion, u64), Error> {
        let source = source.as_ref();
        // Ordinary lease preflight runs only after the shared tile budget fits.
        recipe.preflight_bounded(source)?;
        let catalog = eredu_checkpoint::recipe::UncachedRecipeCatalog::new(source);
        let source_bytes = recipe.infer(&catalog)?.byte_len();
        let context = &self.0[slot];
        let pending = recipe.prepare_borrowed_materialization(source, context)?;
        let (dense, source_leases) = pending.into_parts();
        let mut prepared = WeightMaterialization::prepare_retained(vec![dense], source_leases)?;
        prepare_quantized_outputs(&mut prepared, quantization, target, context.source_stream())?;
        Ok((prepared.submit_prepared_outputs()?, source_bytes))
    }
}

mod cpu_resources;
mod encoded_affine;
mod metadata;
mod admission;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod planning_tests;

impl super::super::preparation::PreparedQuantization {
    /// Submit encoded CPU affine tiles through their admitted runtime, streams,
    /// read constructors and native owners. Cold declarations and output/overlay
    /// construction remain separately owned prerequisites of this prepared value.
    fn materialize_cpu_encoded(
        self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        resources: &cpu_resources::CpuTileResources,
    ) -> Result<
        (QuantizedCheckpoint, BoundedQuantizationPlan),
        PipelineAdmissionError<encoded_affine::TileError<()>>,
    > {
        resources.validate_pool(pool).map_err(|cause| {
            PipelineAdmissionError::Producer(encoded_affine::TileError::Resources(cause))
        })?;
        self.materialize_with_admitted_producer(
            pool,
            safemlx::DeviceType::Cpu,
            &mut encoded_affine::CpuEncodedAffineProducer { pool, resources },
        )
    }
}
