use super::*;

mod composite;
#[cfg(feature = "distributed")]
mod rank;
mod routed;

pub(in crate::composition::mlx) use routed::MlxLocalTextBinding;

impl eredu_core::checkpoint_mechanisms::TextPublicationResources for MlxLocalTextBinding<'_> {
    type Values = MlxTextValues;
    type BankStorage = crate::backend::runtime::residency::manager::MlxResidencyResources;

    fn publication_context(&self) -> &Stream {
        self.stream
    }
}
