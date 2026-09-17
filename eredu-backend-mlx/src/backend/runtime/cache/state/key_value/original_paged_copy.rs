//! Same paid state slots and registered numerical worker for full paged copies.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment, error::Error, nn::workspace::MlxMetalWorkspaceMechanisms,
    runtime::cache::state::CompletedResidentSource,
};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::{InferenceRetention, WorkingMemoryError};
use safemlx::PrefillRootsRuntime;
impl MlxKeyValueState {
    /// Copies the actual selected paged table. A resident-only table returns
    /// None so its existing registered whole-state program remains unchanged.
    /// This is native construction, not admission of a subsequent text request.
    pub(crate) fn copy_original_paged(
        &self,
        completed: Option<&CompletedResidentSource>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
        host: &HostPreparationAuthority,
        capacity: u64,
    ) -> Result<Option<Self>, Error> {
        let Some(manager) = self.layers.slots().iter().find_map(|layer| match layer {
            MlxKeyValueLayerState::Paged(cache) => Some(cache.manager()),
            _ => None,
        }) else {
            return Ok(None);
        };
        let funding = context
            .metadata_funding()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let controls = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Option<Self>, Error>>(),
            std::mem::size_of::<(
                &Self,
                Option<&CompletedResidentSource>,
                &OriginalCopyEnvironment<'_>,
                &PrefillRootsRuntime,
                MlxMetalWorkspaceMechanisms,
                &WorkspaceContext,
                &HostPreparationAuthority,
                u64,
            )>(),
            std::mem::size_of::<std::slice::Iter<'_, MlxKeyValueLayerState>>(),
            std::mem::size_of::<(usize, usize)>(),
            std::mem::size_of::<HostPreparationAuthority>(),
        ];
        context
            .charge_metadata(
                controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        let mut tails = 0usize;
        for (index, layer) in self.layers.slots().iter().enumerate() {
            match layer {
                MlxKeyValueLayerState::Stateless => {}
                MlxKeyValueLayerState::Paged(cache) => {
                    if !manager.same_catalog(cache.manager())
                        || self.global_layer_start.checked_add(index) != Some(cache.global_layer())
                    {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    }
                    tails = tails
                        .checked_add(
                            cache
                                .original_tail_operands()
                                .map_err(Error::PrefillControl)?,
                        )
                        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
                }
                MlxKeyValueLayerState::Device(_) => {
                    return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
                }
            }
        }
        let slots = self
            .layers
            .prepare_copy_slots()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            .for_dense_destination::<MlxKeyValueLayerState>()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let copied = manager.copy_completed_paged_with(
            completed,
            environment,
            initialized,
            mechanisms,
            context,
            capacity,
            tails,
            |mut destination, worker| {
                let layers = super::super::resident_copy::host_copy::copy_slots(
                    slots,
                    host,
                    &funding,
                    |_, layer| match layer {
                        MlxKeyValueLayerState::Stateless => Ok(MlxKeyValueLayerState::Stateless),
                        MlxKeyValueLayerState::Paged(cache) => cache
                            .copy_registered_tail(&mut destination, worker, context)
                            .map(MlxKeyValueLayerState::Paged),
                        MlxKeyValueLayerState::Device(_) => {
                            Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
                        }
                    },
                )?;
                Ok((layers, destination))
            },
        )?;
        // The host capsule contains the original copy receipts. Keep it after
        // all local arrays until the final manager owns that same capsule.
        let ((layers, mut destination), copy_host) = copied.into_parts();
        if let Err(cause) = destination.finish_copy(&copy_host, context) {
            drop(layers);
            drop(destination);
            drop(copy_host);
            return Err(Error::Neural(context.metadata_source(cause)));
        }
        let state = Self {
            layers,
            layout: self.layout.clone(),
            global_layer_start: self.global_layer_start,
            paged_transaction_branch: false,
            inference_retention: InferenceRetention::new(),
        };
        drop((destination, copy_host));
        Ok(Some(state))
    }
}
