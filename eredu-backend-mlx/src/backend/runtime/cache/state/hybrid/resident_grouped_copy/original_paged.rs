//! Independent paged manager and exact grouped role tables under one source loan.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    nn::workspace::MlxMetalWorkspaceMechanisms,
    runtime::cache::{
        residency::PreparedIndependentCacheManager,
        state::{CompletedResidentSource, resident_copy::host_copy::copy_slots},
    },
};
use eredu_core::{HostPreparationAuthority, MemoryLimits};
use eredu_runtime::{DenseHostSlotInitialization, HostSlotTable};
use safemlx::PrefillRootsRuntime;
use std::mem::{size_of, size_of_val};

impl PreparedHybridGroupedCopy<'_> {
    /// The inspected source owns the actual manager identity and every fixed
    /// child table. The existing page/tail worker admits and completes each
    /// numerical copy; the same host-slot worker preserves all role slots.
    pub(in crate::backend::runtime::cache::state) fn copy_original_paged(
        &self,
        completed: Option<&CompletedResidentSource>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
        host: &HostPreparationAuthority,
        capacity: &MemoryLimits,
    ) -> Result<MlxHybridState, Error> {
        let funding = context
            .metadata_funding()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let controls = [
            size_of::<Self>(),
            size_of::<MlxHybridState>(),
            size_of::<Result<MlxHybridState, Error>>(),
            size_of::<Layer<'_>>(),
            size_of::<Option<Layer<'_>>>(),
            size_of::<Option<MlxHybridAttentionState>>(),
            size_of::<Slot>(),
            size_of::<Result<Slot, Error>>(),
            size_of::<MlxTensor>(),
            size_of::<Result<MlxTensor, Error>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<HostSlotTable<MlxHybridLayerState>>(),
            size_of::<PreparedIndependentCacheManager>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<(&Self, &WorkspaceContext, &HostPreparationAuthority)>(),
            size_of::<(
                Option<&CompletedResidentSource>,
                &OriginalCopyEnvironment<'_>,
                &PrefillRootsRuntime,
                MlxMetalWorkspaceMechanisms,
                &MemoryLimits,
            )>(),
        ];
        context
            .charge_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        self.validate_fixed()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        let manager = self
            .paged
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
            .snapshot()
            .manager();
        let mut operands = 0usize;
        for index in 0..self.len() {
            let layer = self
                .layer(index)
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            match layer.attention {
                None
                | Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Stateless)) => {}
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) => {
                    if !manager.same_catalog(cache.manager())
                        || self.global_layer_start().checked_add(index)
                            != Some(cache.global_layer())
                    {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    }
                    operands = operands
                        .checked_add(
                            cache
                                .original_tail_operands()
                                .map_err(Error::PrefillControl)?,
                        )
                        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
                }
                // The retained paged snapshot source rejects these mechanisms;
                // it cannot authenticate them as paged KV tails.
                Some(_) => return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound)),
            }
            operands = operands
                .checked_add(
                    layer
                        .fixed
                        .iter()
                        .filter(|(_, value)| value.is_some())
                        .count(),
                )
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        }
        let copy = manager.copy_completed_paged_with(
            completed,
            environment,
            initialized,
            mechanisms,
            context,
            capacity,
            operands,
            |mut destination, worker| {
                let layers = match self.source {
                    Source::Live(source) => self.copy_original_group(
                        source
                            .layers
                            .prepare_copy_slots()
                            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
                            .for_dense_destination()
                            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
                        &mut destination,
                        worker,
                        context,
                        host,
                    )?,
                    Source::Saved(source) => self.copy_original_group(
                        source
                            .layers
                            .prepare_copy_slots()
                            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
                            .for_dense_destination()
                            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
                        &mut destination,
                        worker,
                        context,
                        host,
                    )?,
                };
                Ok((layers, destination))
            },
        )?;
        let ((layers, mut destination), copy_host) = copy.into_parts();
        if let Err(cause) = destination.finish_copy(&copy_host, context) {
            drop(layers);
            drop(destination);
            drop(copy_host);
            return Err(Error::Neural(context.metadata_source(cause)));
        }
        // The manager retains the completed numerical custody before the
        // temporary capsule retires, including all copied fixed-slot arrays.
        let manager = destination.destination().clone();
        let state = MlxHybridState {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
            manager: Some(manager),
            inference_retention: InferenceRetention::new(),
        };
        drop((destination, copy_host));
        Ok(state)
    }

    fn copy_original_group<S>(
        &self,
        source: DenseHostSlotInitialization<'_, S, MlxHybridLayerState>,
        destination: &mut PreparedIndependentCacheManager,
        worker: &mut crate::backend::runtime::cache::original_copy::Worker<'_, '_>,
        context: &WorkspaceContext,
        host: &HostPreparationAuthority,
    ) -> Result<HostSlotTable<MlxHybridLayerState>, Error> {
        let funding = context
            .metadata_funding()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        copy_slots(source, host, &funding, |index, _| {
            let layer = self
                .layer(index)
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            let child = match layer.fixed {
                Fixed::Live(source) => source.prepare_slots(),
                Fixed::Saved(source) => source.prepare_copy_slots(),
            }
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
            .for_dense_destination()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
            let attention = match layer.attention {
                None => None,
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Stateless)) => Some(
                    MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Stateless),
                ),
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) => {
                    Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Paged(cache.copy_registered_tail(
                            destination,
                            worker,
                            context,
                        )?),
                    ))
                }
                Some(_) => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
            };
            let fixed = copy_slots(child, host, &funding, |_, value| {
                super::super::fixed_slots::copy_slot_with(value, |tensor| {
                    worker
                        .copy(IsolatedArrayCopy::new(tensor.as_array()))
                        .map(MlxTensor::from_array)
                })
            })?;
            Ok(MlxHybridLayerState {
                attention,
                fixed: FixedStateSlots::from_published_slots(fixed),
                fixed_offset: layer.fixed_offset,
            })
        })
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
