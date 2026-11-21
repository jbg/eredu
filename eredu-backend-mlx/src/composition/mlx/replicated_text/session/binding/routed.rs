use super::*;
use crate::backend::runtime::residency::parameter_bank::{
    MlxIndexedMovement, MlxRowDecoder, MlxRowLookupSupport,
};
use eredu_core::preparation::rows::NativeRoutedStorage;

impl NativeRoutedStorage<MlxNeuralBackend> for MlxLocalTextBinding<'_> {
    type Movement = MlxIndexedMovement;
    type Decoder = MlxRowDecoder;
    type RowBinding = crate::backend::runtime::residency::parameter_bank::MlxRowDecodeBinding;
    type StorageError = Error;

    fn row_lookup_support(&self) -> impl eredu_core::preparation::rows::NativeRowLookupSupport {
        MlxRowLookupSupport
    }

    type BankParameters = MlxNeuralBackend;

    type BankMechanisms = crate::backend::runtime::residency::manager::MlxResidencyResources;

    fn bank_residency_context(&self) -> (Stream, Stream) {
        (self.weights_stream.clone(), self.stream.clone())
    }

    fn bank_materializer(
        &self,
    ) -> impl eredu_core::checkpoint_mechanisms::CheckpointEncodingMaterializer<
        Parameters = Self::BankParameters,
    > {
        crate::composition::mlx::component_binding::MlxComponentBinder::new(
            self.weights_stream,
            self.stream,
        )
    }

    fn indexed_movement(&self) -> MlxIndexedMovement {
        MlxIndexedMovement
    }

    fn row_binding(&self) -> Self::RowBinding {
        crate::backend::runtime::residency::parameter_bank::MlxRowDecodeBinding::new(
            self.weights_stream,
            self.stream,
        )
    }
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct MlxLocalTextBinding<'a> {
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
}
