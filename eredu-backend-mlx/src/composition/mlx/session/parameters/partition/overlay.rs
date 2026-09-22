//! All-owner publication with reversible native state and shared provenance.
use super::*;
use std::cell::RefCell;

struct LocalEdit {
    region: ParameterRegion,
    update: ParameterUpdate,
}
struct LocalTarget {
    id: String,
    edits: Vec<LocalEdit>,
}
struct PreparedOverlay {
    original: ParameterReplacementValues<MlxTensor>,
    replacement: ParameterReplacementValues<MlxTensor>,
    original_sources: super::super::CompletedParameterSources,
    replacement_sources: super::super::CompletedParameterSources,
    state: super::super::PreparedParameterReset,
    publication: super::super::publication::PreparedNativeParameterPublication,
    transforms: Vec<(String, ProjectionInputTransform)>,
    active: Option<String>,
    state_exchanged: bool,
    funding: eredu_nn::workspace::HostMetadataFunding,
}

impl MlxModelSession {
    pub(in super::super) fn activate_partition_parameter_overlay(
        &mut self,
        overlay: &AdmittedParameterOverlay,
        limits: CaptureUsage,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<ParameterDiscovery, ParameterError> {
        let mut catalog = self.partition_parameter_catalog(Some(limits))?;
        let transport = self
            .payload
            .distributed
            .clone()
            .expect("partition catalogue");
        let owner = transport.parameter_operations()?;
        let prepared_transport = self.prepared_parameter_transport(&transport)?;
        let binding = self.parameter_operation_binding()?;
        let mut budget = NativeParameterBudget {
            total: Rc::clone(&self.payload.parameter_state.usage),
            limit: limits,
            operation: None,
        };
        let local = (|| {
            overlay.validate(&catalog.discovery)?;
            if catalog.discovery.overlay_identity.is_some() {
                return Err(ParameterError::Unsupported(
                    "remove the active overlay before installing another transaction".into(),
                ));
            }
            let epoch = self
                .payload
                .parameter_state
                .epoch
                .checked_add(1)
                .ok_or(ParameterError::Overflow)?;
            let estimate = self
                .payload
                .model
                .erased()
                .estimate_parameter_reset_state()
                .ok_or_else(|| {
                    ParameterError::Unsupported(
                        "bounded fresh parameter state is unavailable".into(),
                    )
                })?;
            budget.reserve_quota(CaptureUsage {
                captures: 1,
                retained_bytes: add(4096, estimate.retained_bytes)?,
                host_bytes: add(4096, estimate.copy_bytes)?,
                encoded_bytes: 4096,
            })?;
            let targets =
                prepare_targets(&catalog, overlay, transport.parameter_rank(), &mut budget)?;
            Ok((targets, estimate, epoch))
        })();
        let intent: [u8; 32] = Sha256::digest(overlay.intent_identity().as_bytes()).into();
        let session = RefCell::new(&mut *self);
        let result = owner.transaction(
            &prepared_transport,
            binding,
            &intent,
            ParameterOperationKind::Activation,
            local,
            |(targets, estimate, epoch)| {
                let mut session = session.borrow_mut();
                let operation =
                    super::super::numerical::PreparedOperation::new(&session, environment)
                        .map_err(failure)?;
                let context = &operation.context;
                let mut original = context
                    .metadata_vec(targets.len())
                    .map_err(|cause| failure(Error::Neural(cause)))?;
                let mut replacement = context
                    .metadata_vec(targets.len())
                    .map_err(|cause| failure(Error::Neural(cause)))?;
                for target in &targets {
                    let mut edits = context
                        .metadata_vec(target.edits.len())
                        .map_err(|cause| failure(Error::Neural(cause)))?;
                    edits.extend(target.edits.iter().map(|edit| (&edit.region, &edit.update)));
                    let (before, after) = session
                        .prepare_parameter_update(
                            &target.id,
                            &catalog.layouts[&target.id],
                            &edits,
                            &operation,
                        )
                        .map_err(failure)?;
                    context
                        .charge_metadata(
                            target
                                .id
                                .len()
                                .checked_mul(2)
                                .ok_or(ParameterError::Overflow)?,
                        )
                        .map_err(|cause| failure(Error::Neural(cause.into())))?;
                    original.push((target.id.clone(), before));
                    replacement.push((target.id.clone(), after));
                }
                let original = super::super::parameter_rows(original, &operation)?;
                let replacement = super::super::parameter_rows(replacement, &operation)?;
                let sources = super::super::CompletedParameterSources::from_prepared(
                    operation.take_sources(),
                    context,
                    operation.funding.clone(),
                )
                .map_err(failure)?;
                let original_sources = sources
                    .select(&original, context, operation.funding.clone())
                    .map_err(failure)?;
                let replacement_sources = sources
                    .select(&replacement, context, operation.funding.clone())
                    .map_err(failure)?;
                let (publication, state) = session
                    .prepare_parameter_publication(
                        replacement.clone(),
                        replacement_sources.clone(),
                        true,
                        &operation,
                    )
                    .map_err(failure)?;
                let mut transforms = context
                    .metadata_vec(catalog.discovery.parameters.len())
                    .map_err(|cause| failure(Error::Neural(cause)))?;
                for parameter in &catalog.discovery.parameters {
                    if overlay.shared_targets().contains(&parameter.shared_id) {
                        context
                            .charge_metadata(parameter.id.len())
                            .map_err(|cause| failure(Error::Neural(cause.into())))?;
                        transforms.push((parameter.id.clone(), parameter.input_transform.clone()));
                    }
                }
                context
                    .charge_metadata(overlay.intent_identity().len())
                    .map_err(|cause| failure(Error::Neural(cause.into())))?;
                let active = Some(overlay.intent_identity().to_owned());
                let prepared = PreparedOverlay {
                    original,
                    replacement,
                    original_sources,
                    replacement_sources,
                    state,
                    publication,
                    transforms,
                    active,
                    funding: operation.funding.clone(),
                    state_exchanged: false,
                };
                Ok((prepared, estimate, epoch))
            },
            |(prepared, _, _)| {
                session
                    .borrow_mut()
                    .publish_partition_overlay(prepared, true, false)
            },
            |(prepared, _, _)| {
                session
                    .borrow_mut()
                    .publish_partition_overlay(prepared, false, true)
            },
        );
        drop(session);
        let (prepared, estimate, epoch) = self.finish_parameter_control(&transport, result)?;
        let payload = self
            .payload
            .get_mut()
            .expect("completed parameter transaction");
        // Only completed peer publication invalidates reusable snapshots. These
        // final metadata changes cannot submit native work or fail semantically.
        payload.model.erased_mut().finalize_parameter_publication();
        let state = &mut payload.parameter_state;
        state.originals = prepared.original;
        state.published = prepared.replacement;
        state.original_sources = prepared.original_sources;
        state.published_sources = prepared.replacement_sources;
        state.metadata = Some(prepared.funding);
        state.reset_estimate = Some(estimate);
        state.active = prepared.active;
        state.baseline_transforms = prepared.transforms;
        state.epoch = epoch;
        state.floating_state_dtype_bytes = None;
        for parameter in &mut catalog.discovery.parameters {
            if overlay.shared_targets().contains(&parameter.shared_id) {
                parameter.input_transform = ProjectionInputTransform::Identity;
                let width = match parameter.dtype {
                    Some(InterventionDtype::Float32) => 4,
                    Some(InterventionDtype::Float16 | InterventionDtype::Bfloat16) => 2,
                    None => 0,
                };
                state.floating_state_dtype_bytes = state
                    .floating_state_dtype_bytes
                    .max(std::num::NonZeroU8::new(width));
            }
        }
        catalog.discovery.identity =
            format!("{}:parameters:{epoch}", self.intervention_session_identity);
        catalog.discovery.overlay_identity = state.active.clone();
        catalog.discovery.usage = state.usage.get();
        catalog.discovery.coordination_usage = owner.usage()?;
        Ok(catalog.discovery)
    }

    pub(in super::super) fn remove_partition_parameter_overlay(
        &mut self,
        identity: &str,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<ParameterDiscovery, ParameterError> {
        let mut catalog = self.partition_parameter_catalog(None)?;
        let transport = self
            .payload
            .distributed
            .clone()
            .expect("partition catalogue");
        let owner = transport.parameter_operations()?;
        let prepared_transport = self.prepared_parameter_transport(&transport)?;
        let binding = self.parameter_operation_binding()?;
        // Removal has no caller payload or new weight materialization. Its known
        // state/handle costs are charged monotonically, like loaded discovery.
        let local = (|| {
            let state = &self.payload.parameter_state;
            if catalog.discovery.identity != identity || state.active.is_none() {
                return Err(ParameterError::StaleIdentity);
            }
            let epoch = state.epoch.checked_add(1).ok_or(ParameterError::Overflow)?;
            let estimate = state.reset_estimate.ok_or_else(|| {
                ParameterError::Invalid("active overlay has no reset bound".into())
            })?;
            let mut cost = CaptureUsage {
                captures: 1,
                retained_bytes: add(4096, estimate.retained_bytes)?,
                host_bytes: add(4096, estimate.copy_bytes)?,
                encoded_bytes: 4096,
            };
            for (id, _) in state.originals.iter() {
                cost.host_bytes = add(cost.host_bytes, add(512, mul(id.len() as u64, 4)?)?)?;
            }
            NativeParameterBudget {
                total: Rc::clone(&state.usage),
                limit: UNLIMITED,
                operation: None,
            }
            .reserve_quota(cost)?;
            Ok((
                state.originals.clone(),
                state.published.clone(),
                state.original_sources.clone(),
                state.published_sources.clone(),
                epoch,
            ))
        })();
        let mut digest = Sha256::new();
        digest.update(b"eredu-remove-parameter-overlay-v1\0");
        if let Some(active) = &catalog.discovery.overlay_identity {
            digest.update(active.as_bytes());
        }
        let intent = digest.finalize().into();
        let session = RefCell::new(&mut *self);
        let result = owner.transaction(
            &prepared_transport,
            binding,
            &intent,
            ParameterOperationKind::Removal,
            local,
            |(replacement, original, replacement_sources, original_sources, epoch)| {
                let mut session = session.borrow_mut();
                let operation =
                    super::super::numerical::PreparedOperation::new(&session, environment)
                        .map_err(failure)?;
                let (publication, state) = session
                    .prepare_parameter_publication(
                        replacement.clone(),
                        replacement_sources.clone(),
                        false,
                        &operation,
                    )
                    .map_err(failure)?;
                Ok((
                    PreparedOverlay {
                        original,
                        replacement,
                        original_sources,
                        replacement_sources,
                        state,
                        publication,
                        funding: operation.funding.clone(),
                        transforms: Vec::new(),
                        active: None,
                        state_exchanged: false,
                    },
                    epoch,
                ))
            },
            |(prepared, _)| {
                session
                    .borrow_mut()
                    .publish_partition_overlay(prepared, false, false)
            },
            |(prepared, _)| {
                session
                    .borrow_mut()
                    .publish_partition_overlay(prepared, true, true)
            },
        );
        drop(session);
        let (_, epoch) = self.finish_parameter_control(&transport, result)?;
        let payload = self
            .payload
            .get_mut()
            .expect("completed parameter restoration");
        payload.model.erased_mut().finalize_parameter_publication();
        let state = &mut payload.parameter_state;
        for parameter in &mut catalog.discovery.parameters {
            if let Some((_, transform)) = state
                .baseline_transforms
                .iter()
                .find(|(id, _)| id == &parameter.id)
            {
                parameter.input_transform = transform.clone();
            }
        }
        state.baseline_transforms.clear();
        state.originals = Default::default();
        state.published = Default::default();
        state.original_sources = Default::default();
        state.published_sources = Default::default();
        state.reset_estimate = None;
        state.active = None;
        state.floating_state_dtype_bytes = None;
        state.epoch = epoch;
        catalog.discovery.identity =
            format!("{}:parameters:{epoch}", self.intervention_session_identity);
        catalog.discovery.overlay_identity = None;
        catalog.discovery.usage = state.usage.get();
        catalog.discovery.coordination_usage = owner.usage()?;
        Ok(catalog.discovery)
    }

    fn publish_partition_overlay(
        &mut self,
        prepared: &mut PreparedOverlay,
        _active: bool,
        restore: bool,
    ) -> Result<(), ParameterError> {
        if !restore || prepared.state_exchanged {
            self.original_model_source()
                .map_err(|cause| failure(Error::PrefillControl(cause)))?;
            let payload = self.payload.get_mut().ok_or_else(|| {
                failure(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::ReservedWorkActive,
                ))
            })?;
            prepared
                .publication
                .exchange_reset(payload.model.erased_mut(), prepared.state.native.as_mut())
                .map_err(failure)?;
            crate::composition::mlx::replicated_text::exchange_parameter_reset_memory(
                prepared.state.native.as_mut(),
                &mut payload.state_memory,
                payload.nonstate_publication.get_mut(),
                prepared.state.covered_publication,
            )
            .expect("validated prepared reset");
            payload
                .model
                .exchange_parameter_sources(&mut prepared.state.sources);
            prepared.state_exchanged = !prepared.state_exchanged;
        }
        #[cfg(test)]
        if !restore
            && std::mem::take(
                &mut self
                    .payload
                    .get_mut()
                    .expect("completed publication")
                    .parameter_state
                    .reject_publication,
            )
        {
            return Err(ParameterError::Invalid(
                "injected rejection after completed parameter publication".into(),
            ));
        }
        Ok(())
    }
}

fn prepare_targets(
    catalog: &GlobalParameterCatalog,
    overlay: &AdmittedParameterOverlay,
    rank: usize,
    budget: &mut NativeParameterBudget,
) -> Result<Vec<LocalTarget>, ParameterError> {
    let mut targets = Vec::new();
    for target in &catalog.discovery.parameters {
        if !overlay.shared_targets().contains(&target.shared_id) {
            continue;
        }
        if !target.access().replacement {
            return Err(ParameterError::Unsupported(target.condition.clone()));
        }
        // Global baseline transforms and return metadata are retained even on
        // nonowning ranks. Native storage is charged only to actual local owners.
        budget.reserve_quota(CaptureUsage {
            host_bytes: add(512, mul(target.id.len() as u64, 4)?)?,
            ..Default::default()
        })?;
        for (edit, shared) in overlay.plan().edits.iter().zip(overlay.shared_targets()) {
            if shared == &target.shared_id
                && (edit.parameter_shape != target.shape || Some(edit.dtype) != target.dtype)
            {
                return Err(ParameterError::Invalid(
                    "shared parameter slots have incompatible geometry or dtype".into(),
                ));
            }
        }
        let map = catalog
            .global
            .coordinates(&target.id)
            .and_then(|maps| maps.get(rank))
            .and_then(Option::as_ref);
        let Some(map) = map else {
            continue;
        };
        if map.local_shape().contains(&0) {
            continue;
        }
        let layout = catalog.layouts.get(&target.id).ok_or_else(|| {
            ParameterError::Incomplete("local edit owner has no effective layout".into())
        })?;
        let n = elements(&layout.shape)?;
        if n > i32::MAX as u64 {
            return Err(ParameterError::Unsupported(
                "local edited parameter exceeds native i32 indexing".into(),
            ));
        }
        budget.reserve_quota(CaptureUsage {
            retained_bytes: add(mul(n, 48)?, layout.conversion_and_loan_bytes()?)?,
            host_bytes: add(
                layout.host_conversion_bytes()?,
                layout.selection_metadata_bytes(&target.id)?,
            )?,
            ..Default::default()
        })?;
        let mut edits = Vec::new();
        for (edit, shared) in overlay.plan().edits.iter().zip(overlay.shared_targets()) {
            if shared != &target.shared_id {
                continue;
            }
            let projected = map.project_region(&edit.region, 65_536, budget)?;
            for (index, fragment) in projected.fragments().iter().enumerate() {
                let values = elements(&fragment.local().shape)?;
                budget.reserve_quota(CaptureUsage {
                    retained_bytes: add(mul(n, 8)?, mul(values, 16)?)?,
                    host_bytes: add(
                        512,
                        add(
                            mul(values, 8)?,
                            mul(fragment.local().shape.len() as u64, 32)?,
                        )?,
                    )?,
                    ..Default::default()
                })?;
                edits.push(LocalEdit {
                    region: fragment.local().clone(),
                    update: projected.project_update(index, &edit.update, budget)?,
                });
            }
        }
        targets.push(LocalTarget {
            id: target.id.clone(),
            edits,
        });
    }
    Ok(targets)
}
