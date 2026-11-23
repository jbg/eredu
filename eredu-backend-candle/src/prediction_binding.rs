//! Native resources supplied to the portable prepared prediction route.

use std::collections::BTreeMap;

use eredu_core::{
    backend::local_text::NoTextTransport,
    checkpoint_mechanisms::{
        CheckpointEncodingMaterializer, ModelBindingResources, TextBindingContext,
        TextBindingResources,
    },
    preparation::{rows::*, NativePreparation},
    BackendFailure, BackendFailureKind, ObservationMechanisms,
};
use eredu_nn::{tensor_movement::IndexedMovement, Tensor, TensorElementType};

use crate::{
    prediction::CandlePredictionMaterializer,
    residency::CandleWeightResidency,
    rows::{CandleRowDecoder, CandleRowLookupSupport},
    CandleCacheStorage, CandleContext, CandleNeuralBackend, CandleTensor, CandleTextValues,
};

fn unsupported(message: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::Unsupported,
        std::io::Error::other(message),
    )
}

impl ModelBindingResources for CandleContext {
    type Placement = crate::CandlePlacement;
    type CommunicationSetup<'a> = NoTextTransport;
    type Communication = NoTextTransport;
    type TextValues = CandleTextValues;
    type Target = Self;
    type BindingError = BackendFailure;

    fn has_native_rank(_: &Self::Placement) -> bool {
        false
    }

    fn open_communication_setup(
        &self,
        _: &Self::Placement,
    ) -> Result<NoTextTransport, BackendFailure> {
        Err(unsupported(
            "local text placement has no communication setup",
        ))
    }

    fn realize_model_communication<'a>(
        &'a self,
        _: eredu_core::communication::WorldCommunicationBinding<NoTextTransport>,
    ) -> Result<NoTextTransport, BackendFailure> {
        Err(unsupported(
            "local text placement has no communication world",
        ))
    }

    fn bound_communication_manifest(
        value: &NoTextTransport,
    ) -> &eredu_core::communication::CommunicationManifest {
        match *value {}
    }

    fn reclaim_retired_bindings(&self) {
        Self::reclaim_retired();
    }

    fn open_binding_target(
        &self,
        placement: &Self::Placement,
        _: Option<&NoTextTransport>,
    ) -> Result<Self, BackendFailure> {
        if self.placement() != *placement {
            return Err(crate::text::invalid(
                "Candle context differs from the selected placement",
            ));
        }
        self.check_healthy()?;
        Ok(self.clone())
    }

    fn floating_state_element(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<TensorElementType> {
        crate::CandlePreparation::new(self.placement())
            .floating_state_dtype(source)
            .map(eredu_core::preparation::StateStorageDtype::element_type)
    }

    fn observation_mechanisms(&self) -> ObservationMechanisms {
        crate::CandlePreparation::new(self.placement()).observation_mechanisms()
    }

    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        crate::CandlePreparation::new(self.placement()).capture_capabilities()
    }

    fn finish_binding(&self) -> Result<(), BackendFailure> {
        self.synchronize()
    }
}

impl TextBindingResources for CandleContext {
    type Neural = CandleNeuralBackend;
    type CacheStorage = CandleCacheStorage;
    type PredictionMaterializer = CandlePredictionMaterializer;
    type RankPlacement = eredu_core::checkpoint_mechanisms::LocalTextPlacement;
}

impl<'a> TextBindingContext<'a> for CandleContext {
    type TextResources = Self;

    fn text_execution(&'a self) -> &'a Self {
        self
    }
    fn text_weights(&'a self) -> &'a Self {
        self
    }
    fn text_resources(&'a self) -> Self {
        self.clone()
    }
}

impl CheckpointEncodingMaterializer for CandleContext {
    type Parameters = CandleNeuralBackend;

    fn lower_mxfp4(
        &self,
        recipe: eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<eredu_checkpoint::recipe::DerivedWeightRecipe, BackendFailure> {
        use eredu_core::checkpoint_mechanisms::ExactParameterMaterializer;
        CandleNeuralBackend::lower_mxfp4(recipe, source)
    }

    fn weights_quantizer(
        &self,
    ) -> Result<
        Box<
            dyn eredu_core::checkpoint_mechanisms::bounded_quantization::BoundedQuantizationBackend
                + '_,
        >,
        BackendFailure,
    > {
        use eredu_core::checkpoint_mechanisms::ExactParameterMaterializer;
        CandleNeuralBackend::quantizer(self)
    }
}

impl RowDecodeBinding<CandleNeuralBackend, CandleWeightResidency, CandleRowDecoder>
    for CandleContext
{
    type ParameterReads = Self;

    fn support(&self) -> impl NativeRowLookupSupport {
        CandleRowLookupSupport::new(self.placement())
    }

    fn parameter_reads(&self) -> Self {
        self.clone()
    }

    fn decoder(
        &self,
        scale: Option<(eredu_checkpoint::recipe::RecipeDtype, CandleTensor)>,
    ) -> Result<CandleRowDecoder, BackendFailure> {
        CandleRowDecoder::new(scale)
    }
}

pub struct CandleIndexedMovement;

impl IndexedMovement<CandleNeuralBackend> for CandleIndexedMovement {
    type Error = BackendFailure;

    fn index_demands(
        &mut self,
        indices: &CandleTensor,
        upper_bound: usize,
        context: &CandleContext,
    ) -> Result<Vec<(usize, u64)>, BackendFailure> {
        let mut counts = BTreeMap::new();
        for id in indices
            .to_i32_vec(context)
            .map_err(BackendFailure::from_error)?
        {
            let id = usize::try_from(id).map_err(BackendFailure::from_error)?;
            if id >= upper_bound {
                return Err(BackendFailure::from_error(std::io::Error::other(
                    "group index exceeds bank extent",
                )));
            }
            *counts.entry(id).or_insert(0) += 1;
        }
        Ok(counts.into_iter().collect())
    }

    fn remap_indices(
        &mut self,
        indices: &CandleTensor,
        mapping: &[(usize, usize)],
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let mapping = mapping.iter().copied().collect::<BTreeMap<_, _>>();
        let ids = indices
            .to_i32_vec(context)
            .map_err(BackendFailure::from_error)?
            .into_iter()
            .map(|id| {
                let id = usize::try_from(id).map_err(BackendFailure::from_error)?;
                let compact = mapping.get(&id).ok_or_else(|| {
                    BackendFailure::from_error(std::io::Error::other(
                        "compact group index is missing",
                    ))
                })?;
                i32::try_from(*compact).map_err(BackendFailure::from_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        CandleTensor::from_i32_slice(&ids, indices.shape(), context)
            .map_err(BackendFailure::from_error)
    }

    fn select_rows(
        &mut self,
        value: &CandleTensor,
        start: usize,
        end: usize,
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let start = i32::try_from(start).map_err(BackendFailure::from_error)?;
        let end = i32::try_from(end).map_err(BackendFailure::from_error)?;
        value
            .index(&[eredu_nn::Index::Range(start, end)], context)
            .map_err(BackendFailure::from_error)
    }

    fn concatenate_rows(
        &mut self,
        values: &[CandleTensor],
        context: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        CandleTensor::concatenate(values, 0, context).map_err(BackendFailure::from_error)
    }
}

impl NativeRoutedStorage<CandleNeuralBackend> for CandleContext {
    type Movement = CandleIndexedMovement;
    type Decoder = CandleRowDecoder;
    type RowBinding = Self;
    type StorageError = BackendFailure;
    type BankParameters = CandleNeuralBackend;
    type BankMechanisms = CandleWeightResidency;

    fn row_lookup_support(&self) -> impl NativeRowLookupSupport {
        CandleRowLookupSupport::new(self.placement())
    }
    fn bank_residency_context(&self) -> Self {
        self.clone()
    }
    fn bank_materializer(
        &self,
    ) -> impl CheckpointEncodingMaterializer<Parameters = CandleNeuralBackend> {
        self.clone()
    }
    fn indexed_movement(&self) -> Self::Movement {
        CandleIndexedMovement
    }
    fn row_binding(&self) -> Self {
        self.clone()
    }
}
