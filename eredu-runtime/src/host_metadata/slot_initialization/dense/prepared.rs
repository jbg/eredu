//! Exact source-borrowed dense host copying under accepted preparation custody.
use super::*;
use crate::HostMetadataIdentity;
use eredu_core::HostPreparationAuthority;

#[derive(Debug, thiserror::Error)]
enum Cause<E> {
    #[error("{0}")]
    Metadata(#[source] crate::working_memory::WorkingMemoryError),
    #[error("prepared dense host destination allocation failed")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("{0}")]
    Value(#[source] E),
}
/// A failed fixed destination retains its original preparation authority after
/// its typed cause. Nested resources stay with their producer/recovery owners.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedDenseHostCopyError<E: std::error::Error + 'static> {
    #[source]
    cause: Cause<E>,
    host: HostPreparationAuthority,
}
impl<S, D> DenseHostSlotInitialization<'_, S, D> {
    /// Exact inline destination, shared metadata constructor and typed callback
    /// transport. Nested value allocations and source ownership are separate.
    /// The enclosing provider admits this amount before calling the constructor.
    pub fn prepared_copy_bytes<E: std::error::Error + 'static, F>(&self, _copy: &F) -> Option<u64> {
        let parts = [
            Self::preparation_control_bytes()?,
            size_of::<F>(),
            size_of::<E>(),
            size_of::<Cause<E>>(),
            size_of::<PreparedDenseHostCopyError<E>>(),
            size_of::<Result<D, E>>(),
            size_of::<Result<HostSlotTable<D>, PreparedDenseHostCopyError<E>>>(),
            size_of::<Result<HostMetadataIdentity, crate::working_memory::WorkingMemoryError>>(),
            size_of::<
                Result<
                    (Self, DenseHostSlotInitializationBuilder<D>),
                    std::collections::TryReserveError,
                >,
            >(),
            size_of::<HostPreparationAuthority>(),
            size_of::<(&S, &HostPreparationAuthority)>(),
            size_of::<std::ops::Range<usize>>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
        self.initialization_peak_bytes()
            .checked_add(u64::try_from(controls).ok()?)
    }

    /// Fills the actual source-selected table through its existing fixed-buffer
    /// worker. This authorizes only the priced host destination: the callback's
    /// native copy, recovery and completion must have independent admission.
    /// Every published table alias retains the same preparation authority.
    pub fn copy_with_preparation<E, F>(
        self,
        host: &HostPreparationAuthority,
        mut copy: F,
    ) -> Result<HostSlotTable<D>, PreparedDenseHostCopyError<E>>
    where
        E: std::error::Error + 'static,
        F: FnMut(usize, &S) -> Result<D, E>,
    {
        let result = (|| {
            let identity = HostMetadataIdentity::prepared_host(host).map_err(Cause::Metadata)?;
            let (source, mut builder) = self
                .try_initialize_retaining_source()
                .map_err(Cause::Allocation)?;
            for index in 0..source.len() {
                let value = copy(index, source.source_at(index).expect("fixed source count"))
                    .map_err(Cause::Value)?;
                assert!(
                    builder.push(value).is_ok(),
                    "fixed source-selected destination"
                );
            }
            Ok(builder
                .finish_with_preparation(Some((identity, host)))
                .ok()
                .expect("complete source-selected destination")
                .into_table())
        })();
        result.map_err(|cause| PreparedDenseHostCopyError {
            cause,
            host: host.clone(),
        })
    }
}
