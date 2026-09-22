//! Ordinary transfer recovery and publication use the same prepared destinations
//! as controller acquisition, without an original-operation observer.
use super::super::host_acquisition::{HostAcquisitionCause, HostAcquisitionFailure};
use super::*;
use crate::backend::submission_recovery::PreparedRecovery;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext};

impl ResidentTransfer {
    pub(in crate::backend::runtime::residency::manager) fn host_consumer_control_bytes()
    -> Option<usize> {
        let fields = [
            usize::try_from(
                PreparedRecovery::<TransferObservation, HostMetadataFunding>::control_bytes()?,
            )
            .ok()?,
            usize::try_from(ResidentRecovery::<TransferObservation>::control_bytes()?).ok()?,
            Stream::ordinary_clone_control_bytes()?,
            Event::ordinary_wait_wrapper_control_bytes()?,
            Event::wait_record_layout(1)?.named_control_bytes(),
            HostAcquisitionFailure::control_bytes()?,
            size_of::<TransferObservation>(),
            size_of::<(&Self, &Stream)>(),
            size_of::<Option<&HostMetadataFunding>>(),
            size_of::<Result<(), ResidencyError>>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
}

pub(super) fn prepare_host_consumer(
    value: TransferObservation,
    funding: &HostMetadataFunding,
) -> Result<ResidentRecovery<TransferObservation>, ResidencyError> {
    let fail = |cause| HostAcquisitionFailure::new(HostAcquisitionCause::Scope(cause), funding);
    let ready = PreparedRecovery::new(value, funding.clone()).map_err(|error| fail(error.cause))?;
    let recovery = ready.try_begin().map_err(|error| fail(error.cause))?;
    Ok(ResidentRecovery::ordinary(recovery))
}

pub(in crate::backend::runtime::residency::manager) struct PreparedHostTransfer {
    value: Rc<ResidentTransferResources>,
    publication: publication::TransferPublication,
    funding: HostMetadataFunding,
}

impl PreparedHostTransfer {
    /// Ordinary C handles are paid by the host metadata account; their shared
    /// graph, root-vector buffers and dispatch allocations are native facts.
    pub(in crate::backend::runtime::residency::manager) fn native_wrapper_control_bytes(
        population: WindowPopulation,
    ) -> Option<usize> {
        safemlx::ImmutableHostTransferBuffer::ordinary_copy_wrapper_control_bytes()?
            .checked_add(Array::ordinary_clone_control_bytes()?)?
            .checked_mul(population.physical_bindings)?
            .checked_add(Event::ordinary_submission_wrapper_control_bytes()?)
    }

    pub(in crate::backend::runtime::residency::manager) fn control_bytes(
        population: WindowPopulation,
    ) -> Option<usize> {
        let shape = TransferPayloadShape::window(population)?;
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, HostAcquisitionFailure>>(),
            WorkspaceContext::metadata_rc_bytes::<ResidentTransferResources>()?,
            WorkspaceContext::metadata_rc_bytes::<OrdinaryRetirement<TransferApplication>>()?,
            usize::try_from(OrdinaryRetirement::<TransferApplication>::control_bytes()?).ok()?,
            usize::try_from(PreparedRecovery::<
                Rc<ResidentTransferResources>,
                HostMetadataFunding,
            >::control_bytes()?)
            .ok()?,
            usize::try_from(ResidentRecovery::<Rc<ResidentTransferResources>>::control_bytes()?)
                .ok()?,
            usize::try_from(publication::TransferPublication::control_bytes(shape)?).ok()?,
            shape.resource_requested_bytes()?,
            population.named.catalog_requested_bytes,
            population.named.destination_requested_bytes,
            super::super::named_arrays::NameCatalogOwner::custody_transport_bytes()?,
        ];
        fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)
    }

    pub(in crate::backend::runtime::residency::manager) fn prepare(
        manager: &ResidencyManager,
        roots: &[OffloadUnitId],
        scratch: &mut [eredu_runtime::residency::ResidencyClosureSlot],
        population: WindowPopulation,
        funding: &HostMetadataFunding,
    ) -> Result<Self, HostAcquisitionFailure> {
        let fail = |cause| HostAcquisitionFailure::new(cause, funding);
        let shape = TransferPayloadShape::window(population)
            .ok_or_else(|| fail(HostAcquisitionCause::Overflow))?;
        funding
            .reserve_metadata(
                Self::control_bytes(population)
                    .and_then(|bytes| {
                        bytes.checked_add(Self::native_wrapper_control_bytes(population)?)
                    })
                    .ok_or_else(|| fail(HostAcquisitionCause::Overflow))?,
            )
            .map_err(|cause| fail(HostAcquisitionCause::Funding(cause)))?;
        let catalog = manager
            .prepare_name_catalog_with_custody(roots, scratch, funding.clone().into())
            .map_err(|error| fail(HostAcquisitionCause::Named(error.into_source())))?;
        let mut value = Rc::new(ResidentTransferResources {
            sources: Vec::new(),
            retained_arrays: Vec::new(),
            retained_host: Vec::new(),
            retained_events: Vec::new(),
            event: None,
            application: Rc::new(OrdinaryRetirement::new(TransferApplication::host(
                funding.clone(),
            ))),
            named: None,
        });
        let resources = Rc::get_mut(&mut value).expect("unissued ordinary transfer");
        let reserve = (|| {
            resources.sources.try_reserve_exact(shape.pending_sources)?;
            resources
                .retained_arrays
                .try_reserve_exact(shape.retained_arrays)?;
            resources
                .retained_host
                .try_reserve_exact(shape.retained_host)?;
            resources
                .retained_events
                .try_reserve_exact(shape.retained_events)?;
            Rc::get_mut(&mut resources.application)
                .expect("unissued application")
                .ids
                .try_reserve_exact(shape.unit_ids)
        })();
        reserve.map_err(|cause| fail(HostAcquisitionCause::Allocation(cause)))?;
        resources.named = Some(
            catalog
                .prepare_window(manager, roots, scratch)
                .map_err(|error| fail(HostAcquisitionCause::Named(error.into_source())))?,
        );
        let mut publication =
            publication::TransferPublication::with_custody(funding.clone().into());
        publication
            .prepare(manager, roots, scratch, shape)
            .map_err(|cause| fail(HostAcquisitionCause::Named(cause)))?;
        Ok(Self {
            value,
            publication,
            funding: funding.clone(),
        })
    }

    pub(super) fn activate(
        mut self,
        ids: &[OffloadUnitId],
        missing: &[bool],
        tier: MemoryTier,
        failed: FailureFlag,
    ) -> Result<
        (
            ResidentRecovery<Rc<ResidentTransferResources>>,
            publication::TransferPublication,
        ),
        ResidencyError,
    > {
        let fail = |cause| HostAcquisitionFailure::new(cause, &self.funding);
        let value = Rc::get_mut(&mut self.value).expect("unissued ordinary transfer");
        let application = Rc::get_mut(&mut value.application).expect("unissued application");
        self.publication
            .select_application(ids, missing, &mut application.ids)
            .map_err(|cause| fail(HostAcquisitionCause::Lease(cause)))?;
        application.tier = tier;
        application.failed_transfer = Some(failed);
        let ready = PreparedRecovery::new(self.value, self.funding.clone())
            .map_err(|error| fail(HostAcquisitionCause::Scope(error.cause)))?;
        let recovery = ready
            .try_begin()
            .map_err(|error| fail(HostAcquisitionCause::Scope(error.cause)))?;
        Ok((ResidentRecovery::ordinary(recovery), self.publication))
    }
}
