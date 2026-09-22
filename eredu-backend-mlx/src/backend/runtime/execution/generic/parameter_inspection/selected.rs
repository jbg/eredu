//! A standalone parameter loan through the existing selected residency worker.
use super::super::*;
use super::{HostMetadataFunding, MlxParameterPreparation};
use crate::backend::runtime::residency::manager::{
    ForegroundDiskSubsetCeiling, ForegroundDiskWindowPlan, OriginalResidencySource,
    WindowPopulation,
};
use crate::backend::{
    nn::workspace::SpeculativeNumericalRecipe,
    submission_recovery::native_role::{NativeRoleCapacity, physical},
};
use eredu_core::DomainMemoryRequirements;
use eredu_nn::{
    Tensor,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use eredu_runtime::{
    residency::ResidencyClosureSlot,
    working_memory::{
        HostSourceConstructionFacts, HostSourceConstructionProgram, NumericalSourceRequirements,
        OriginalHostSourceBank, OriginalNumericalBudgetCustody, WorkingMemoryError,
    },
};
use safemlx::{Array, OperationEvent, OriginalBufferBudget, PreparedPipelineCachePlan};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};

fn unknown() -> Error {
    Error::OriginalSourceContract {
        stage: "selected parameter source preparation",
        cause: WorkingMemoryError::UnknownBound,
    }
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}
fn identity() -> Error {
    Error::OriginalSourceContract {
        stage: "selected parameter source identity",
        cause: WorkingMemoryError::IdentityMismatch,
    }
}
fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn metadata(cause: WorkspaceMetadataError) -> Error {
    Error::Neural(cause.into())
}

struct Accepted {
    bank: OriginalHostSourceBank,
    disk: Option<crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity>,
    funding: HostMetadataFunding,
    custody: OriginalNumericalBudgetCustody,
}

#[derive(Debug, thiserror::Error)]
enum ParameterLoanFailure {
    #[error("parameter source construction or completion failed: {0}")]
    Source(#[source] Error),
    #[error("parameter numerical consumer failed: {0}")]
    Consumer(#[source] Error),
    #[error("completed parameter source retirement failed: {0}")]
    Retirement(#[source] Error),
}

impl MlxParameterPreparation<'_> {
    pub(in super::super) fn inspect<U, P, E, F, V>(
        &self,
        policy: &mut MlxLayerwisePolicy<U, P>,
        ordinal: usize,
        build: F,
        operation: V,
        stream: &Stream,
    ) -> Result<bool, LayerwiseAcquireError<E, Error>>
    where
        U: Parameterized<MlxTensor> + 'static,
        P: MlxUnitPopulator<U>,
        F: FnOnce(&Stream) -> Result<U, E>,
        V: FnOnce(&mut U) -> Result<(), Error>,
    {
        let result = (|| {
            let context = self
                .mechanism
                .context(self.funding.clone())
                .map_err(metadata)?;
            context
                .charge_metadata(Self::control_bytes().ok_or_else(overflow)?)
                .map_err(metadata)?;
            if !policy.populator.preserves_prepared_parameter_source() {
                return Err(unknown());
            }
            let retained = policy.operation_source.as_ref().map_err(|_| unknown())?;
            let source = policy
                .residency
                .selected_target_source(retained)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
            let workspace =
                policy.layerwise_workspace_with_metadata(self.mechanism.allocation(), &context)?;
            let constructors = policy
                .residency
                .parameter_constructors(ordinal)
                .ok_or_else(unknown)?;
            let id = policy.unit_ids.get(ordinal).ok_or_else(identity)?;
            let range = policy
                .layout
                .window_range(
                    ordinal,
                    std::num::NonZeroUsize::new(policy.window_depth).ok_or_else(identity)?,
                )
                .ok_or_else(identity)?;
            let manager = &policy.residency;
            let ids = &policy.unit_ids;
            let populator = &mut policy.populator;
            self.inspect_source_prepared(
                manager,
                &source,
                &workspace,
                id,
                constructors,
                || {
                    manager
                        .trim_device_units(ids, &ids[range])
                        .map_err(Error::from)
                },
                build,
                |unit, transfer, rows| populator.populate_original(unit, transfer, rows),
                operation,
                stream,
            )
        })();
        result.map_err(LayerwiseAcquireError::Policy)?
    }

    /// The actual target or supplementary owner lends its exact constructor,
    /// source and binding worker. This creates no model/token occurrence.
    pub(crate) fn inspect_source<U, E, F, P, V>(
        &self,
        manager: &ResidencyManager,
        source: &crate::backend::runtime::residency::manager::SelectedResidencySource,
        parameter_source: &super::super::LayerwiseWorkspace,
        id: &OffloadUnitId,
        constructors: super::super::ParameterConstructors,
        build: F,
        populate: P,
        operation: V,
        stream: &Stream,
    ) -> Result<Result<bool, LayerwiseAcquireError<E, Error>>, Error>
    where
        U: 'static,
        F: FnOnce(&Stream) -> Result<U, E>,
        P: FnOnce(&mut MlxModule<U>, &ResidentUnitLease, usize) -> Result<(), Error>,
        V: FnOnce(&mut U) -> Result<(), Error>,
    {
        self.inspect_source_prepared(
            manager,
            source,
            parameter_source,
            id,
            constructors,
            || Ok(()),
            build,
            populate,
            operation,
            stream,
        )
    }

    fn inspect_source_prepared<U, E, A, F, P, V>(
        &self,
        manager: &ResidencyManager,
        source: &crate::backend::runtime::residency::manager::SelectedResidencySource,
        parameter_source: &super::super::LayerwiseWorkspace,
        id: &OffloadUnitId,
        constructors: super::super::ParameterConstructors,
        prepare: A,
        build: F,
        populate: P,
        operation: V,
        stream: &Stream,
    ) -> Result<Result<bool, LayerwiseAcquireError<E, Error>>, Error>
    where
        U: 'static,
        A: FnOnce() -> Result<(), Error>,
        F: FnOnce(&Stream) -> Result<U, E>,
        P: FnOnce(&mut MlxModule<U>, &ResidentUnitLease, usize) -> Result<(), Error>,
        V: FnOnce(&mut U) -> Result<(), Error>,
    {
        let context = self
            .mechanism
            .context(self.funding.clone())
            .map_err(metadata)?;
        let fixed = [
            Self::control_bytes().ok_or_else(overflow)?,
            size_of::<A>(),
            size_of::<F>(),
            size_of::<P>(),
            size_of::<V>(),
            size_of::<U>(),
            size_of::<E>(),
            size_of::<Result<U, E>>(),
            size_of::<LayerwiseAcquireError<E, Error>>(),
            size_of::<Accepted>(),
            size_of::<RefCell<Option<Accepted>>>(),
            size_of::<OriginalResidencySource>(),
            size_of::<WindowPopulation>(),
            size_of::<(OffloadUnitId, usize)>(),
            size_of::<SpeculativeNumericalRecipe>(),
            size_of::<HostSourceConstructionProgram>(),
            HostSourceConstructionProgram::plan_control_bytes().ok_or_else(overflow)?,
            ForegroundDiskSubsetCeiling::control_bytes().ok_or_else(overflow)?,
        ];
        context
            .charge_metadata(
                fixed
                    .into_iter()
                    .try_fold(size_of_val(&fixed), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(metadata)?;
        if !safemlx::StreamCopyPlan::<()>::capture(self.environment.stream())
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            .matches_source(stream)
        {
            return Err(Error::OriginalSourceContract {
                stage: "selected parameter source stream",
                cause: WorkingMemoryError::IdentityMismatch,
            });
        }
        let mut closure = context
            .metadata_vec(source.source().controller_units)
            .map_err(Error::Neural)?;
        closure.resize(
            source.source().controller_units,
            ResidencyClosureSlot::default(),
        );
        let window = manager
            .selected_source_population(&source, std::slice::from_ref(id), &mut closure)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let pool = self.environment.pool();
        let disk_plan = if source.foreground().is_some() {
            Some(ForegroundDiskWindowPlan::new_with_metadata(
                manager,
                pool,
                window,
                std::slice::from_ref(id),
                &mut closure,
                Some(self.funding),
            )?)
        } else {
            None
        };
        let disk = match &disk_plan {
            Some(plan) => Some(plan.subset_ceiling(window.units).ok_or_else(overflow)?),
            None => None,
        };
        let materialized = disk
            .as_ref()
            .map_or(Default::default(), |disk| disk.population().materialized);
        let source_bytes =
            OriginalSelectedResidencyAttempt::source_construction_bytes(&source, window)
                .and_then(|n| {
                    n.checked_add(
                        disk_plan
                            .as_ref()
                            .map(ForegroundDiskWindowPlan::attempt_control_bytes)
                            .unwrap_or(Some(0))?,
                    )
                })
                .ok_or_else(overflow)?;
        let mut components = context
            .metadata_vec(1 + usize::from(disk.is_some()))
            .map_err(Error::Neural)?;
        components.push(
            HostSourceConstructionFacts::new(
                source_bytes,
                OriginalSelectedResidencyAttempt::construction_attempts(),
                0,
            )
            .map_err(memory)?,
        );
        if let Some(disk) = &disk {
            components.push(disk.source_facts(1).ok_or_else(overflow)?);
        }
        let program = HostSourceConstructionProgram::from_components(components).map_err(memory)?;
        // The source marker orders after this loan's actual copies and binding
        // constructors. Numerical consumers run after this scope has closed.
        context.begin_span();
        let marker = eredu_nn::workspace::WorkspaceTensor::from_f32_slice(&[0.0], &[1], &context)
            .map_err(Error::Neural)?;
        context.complete_values(&[&marker]).map_err(Error::Neural)?;
        let report = context
            .finish_report(std::slice::from_ref(&marker))
            .map_err(Error::Neural)?;
        let recipe = SpeculativeNumericalRecipe::inspect_completed_outputs(
            &report,
            1,
            self.mechanism,
            &context,
        )
        .and_then(|recipe| {
            recipe
                .with_selected_layerwise_source(parameter_source, window, constructors, &context)?
                .with_materialized_storage(materialized, &context)
        })
        .map_err(Error::Neural)?;
        drop(marker);
        let allocator = self
            .environment
            .input_runtime()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let population = OriginalBufferBudget::population_layout(
            &allocator,
            usize::try_from(recipe.storage.mutable_bytes()).map_err(|_| overflow())?,
            recipe.storage.maximum_births(),
        )
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let capacity = NativeRoleCapacity {
            graph: recipe.graph_capacity,
            records: recipe.record_capacity,
            backing: population.capacity(),
        };
        let request_bytes = id.as_str().len();
        let unit_controls = PreparedUnit::<U>::control_bytes().ok_or_else(overflow)?;
        let binding_controls =
            crate::backend::runtime::checkpoint::binding::original_parameter_binding_control_bytes(
                window.bindings,
            )
            .ok_or_else(overflow)?;
        let controls = [
            usize::try_from(unit_controls).map_err(|_| overflow())?,
            binding_controls,
            request_bytes,
            usize::try_from(recipe.controls).map_err(|_| overflow())?,
            size_of::<PreparedUnit<U>>(),
            size_of::<MlxUnitLease<U>>(),
            size_of::<super::super::submission::CompletedUnitLease<U>>(),
            size_of::<Array>(),
            size_of::<Result<Array, safemlx::error::Exception>>(),
            size_of::<(A, F, P)>(),
            size_of::<V>(),
            size_of::<U>(),
            size_of::<Result<bool, LayerwiseAcquireError<E, Error>>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<safemlx::PreparedResidentGraph>(),
            ForegroundDiskSubsetCeiling::control_bytes().ok_or_else(overflow)?,
        ];
        let control_bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(overflow)?;
        // The selected disk worker charges its exact descriptor constructor
        // itself. Keep that capacity available after the fixed controls debit.
        let dynamic_controls = if disk_plan.is_some() {
            ForegroundDiskWindowPlan::construction_control_bytes(window.units)
                .ok_or_else(overflow)?
        } else {
            0
        };
        let admitted_controls = control_bytes
            .checked_add(materialized.controls)
            .and_then(|n| n.checked_add(dynamic_controls))
            .ok_or_else(overflow)?;
        let mut completed_retention =
            crate::backend::runtime::residency::manager::PreparedCompletedSourceRetention::prepare(
                window.units,
                window.bindings,
                &context,
            )?;
        let accepted = RefCell::new(None::<Accepted>);
        let plan = physical::Plan::new(
            &allocator,
            capacity,
            Some(PreparedPipelineCachePlan::new(recipe.kernels)),
            (),
            |_, role| {
                let Accepted {
                    mut bank,
                    disk,
                    funding,
                    custody,
                } = accepted.borrow_mut().take().ok_or_else(identity)?;
                funding
                    .reserve_metadata(control_bytes)
                    .map_err(Error::WorkspacePlanning)?;
                let access =
                    OriginalSelectedResidencyAccess::numerical(custody.clone(), role, &funding)?;
                let prepared = PreparedUnit::new(custody.into());
                let mut attempt = access.prepare_source(
                    manager,
                    source,
                    std::slice::from_ref(id),
                    &mut bank,
                    disk.as_ref().map(|capacity| (pool, capacity, &funding)),
                )?;
                let mut graph = OperationEvent::prepare_resident_graph(
                    recipe.completion.graph,
                    role.observer(),
                )?;
                if recipe.completion.nested_completions != 0 {
                    graph.configure_nested_completions(
                        &recipe.completion.nested_traversal().ok_or_else(unknown)?,
                        recipe.completion.nested_completions,
                    )?;
                }
                prepare()?;
                let unit = match build(stream) {
                    Ok(unit) => unit,
                    Err(cause) => return Ok(Err(LayerwiseAcquireError::Architecture(cause))),
                };
                let requests = [(id.clone(), 1)];
                let transfer = access.with_residency(&mut attempt, |slots, current| {
                    let transfer = manager.acquire_many_with_original_transfer(
                        &requests,
                        MemoryTier::Device,
                        slots,
                        current,
                    )?;
                    transfer.order_after_original(stream, slots.observations, current)?;
                    Ok(transfer)
                })?;
                let mut lease = MlxUnitLease::from_prepared(
                    prepared,
                    MlxModule::new(unit),
                    MlxUnitTransfer::Ordinary {
                        _transfer: transfer,
                    },
                    role.observer().clone(),
                );
                let (unit, transfer) = lease.population_parts();
                populate(unit, transfer, window.bindings)?;
                let completed = lease.complete_original_retaining(|_, _| {
                    let seed = Array::try_from_slice(&[0.0f32], &[1])?;
                    let marker = seed.copy(stream)?;
                    crate::backend::runtime::cache::complete_values([&marker], stream)?;
                    Ok(())
                });
                drop(graph);
                completed.map(Ok)
            },
        )
        .with_parent(safemlx::OriginalScopeObserver::try_current()?);
        let native_metadata =
            u64::try_from(plan.metadata_bytes().map_err(memory)?).map_err(|_| overflow())?;
        let placement = self
            .environment
            .buffer_placement()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let domains = pool.topology();
        let descriptor_bytes = (|| {
            DomainMemoryRequirements::construction_backing_bytes(domains, 1)?
                .checked_add(placement.clone_backing_bytes()?)
                .and_then(|bytes| bytes.checked_add(pool.configured_limits().backing_bytes().ok()?))
                .ok_or(eredu_core::MemoryDomainError::Overflow)
        })()
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        context
            .charge_metadata(usize::try_from(descriptor_bytes).map_err(|_| overflow())?)
            .map_err(metadata)?;
        let mut native = DomainMemoryRequirements::zero_with_allowance_capacity(domains, 1);
        native
            .add_allocation(
                u64::try_from(capacity.backing).map_err(|_| overflow())?,
                &placement,
            )
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let requirements = NumericalSourceRequirements::new(
            native,
            Some(native_metadata),
            Some(u64::try_from(admitted_controls).map_err(|_| overflow())?),
            program.facts(),
        )
        .map_err(memory)?;
        let mut account = pool
            .reserve_numerical_source(
                self.execution,
                requirements,
                pool.configured_limits().clone(),
            )
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let custody = account.budget_custody();
        let mut banks = account
            .take_source_constructions()
            .map_err(memory)?
            .partition_program(&program)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let bank = banks.take(0).map_err(memory)?;
        let disk = match disk {
            Some(source) => Some(
                source
                    .prepare_capacity(
                        1,
                        banks.take(1).map_err(memory)?,
                        custody.clone().into(),
                        None,
                    )
                    .map_err(memory)?,
            ),
            None => None,
        };
        let funding = account
            .metadata_funding()
            .map_err(Error::WorkspacePlanning)?;
        *accepted.borrow_mut() = Some(Accepted {
            bank,
            disk,
            funding,
            custody,
        });
        let source = plan
            .run_prepaid_with_source(pool, account.claim_native().map_err(memory)?)
            .map_err(|cause| {
                let cause = match cause {
                    physical::Failure::Admission(cause) => memory(cause),
                    physical::Failure::Native(cause) => Error::StorageSource(cause),
                };
                Error::Neural(context.metadata_source(ParameterLoanFailure::Source(cause)))
            })?;
        let completed_source = source.source;
        let mut source = match source.value {
            Ok(source) => source,
            Err(cause) => return Ok(Err(cause)),
        };
        for lease in source.retained_leases() {
            completed_retention.retain(lease.storage(), &completed_source, &context)?;
        }
        context.charge_metadata(
            crate::backend::runtime::residency::manager::ResidentLeaseStorage::host_receipt_control_bytes()
                .and_then(|n| n.checked_add(std::mem::size_of::<(
                    &super::MlxParameterPreparation<'_>,
                    &[crate::backend::runtime::residency::manager::ResidentUnitLease],
                    crate::backend::runtime::residency::storage::RetainedAllocationReceipt<'_>,
                    crate::backend::runtime::residency::storage::RetainedAllocationSource,
                    Result<(), Error>,
                )>()))
                .ok_or_else(overflow)?
        ).map_err(metadata)?;
        for lease in source.retained_leases() {
            context.charge_metadata(
                crate::backend::runtime::residency::manager::ResidentLeaseStorage::completed_numerical_source_control_bytes()
                    .ok_or_else(overflow)?
            ).map_err(metadata)?;
            for previous in lease.storage().completed_numerical_sources() {
                self.retain_source(previous.clone(), &context)?;
            }
            for receipt in lease.storage().host_receipts() {
                self.retain_host_source(receipt, &context)?;
            }
        }
        self.retain_source(completed_source, &context)?;
        // The source role has closed. The same completed loan still holds its
        // unit, transfer pins and native/source accounts throughout the consumer.
        let outcome = operation(source.unit_mut()).map_err(|cause| {
            LayerwiseAcquireError::Policy(Error::Neural(
                context.metadata_source(ParameterLoanFailure::Consumer(cause)),
            ))
        });
        let retirement = source.retire().map_err(|cause| {
            Error::Neural(context.metadata_source(ParameterLoanFailure::Retirement(cause)))
        });
        match outcome {
            Err(cause) => Ok(Err(cause)),
            Ok(()) => retirement.map(|_| Ok(true)),
        }
    }
}
