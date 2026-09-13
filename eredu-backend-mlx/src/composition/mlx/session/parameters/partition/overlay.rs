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
    original: BTreeMap<String, MlxTensor>,
    replacement: BTreeMap<String, MlxTensor>,
    state: Box<dyn std::any::Any>,
    state_exchanged: bool,
}

impl MlxModelSession {
    pub(in super::super) fn activate_partition_parameter_overlay(
        &mut self,
        overlay: &AdmittedParameterOverlay,
        limits: CaptureUsage,
        stream: &Stream,
    ) -> Result<ParameterDiscovery, ParameterError> {
        let mut catalog = self.partition_parameter_catalog(Some(limits))?;
        let transport = self
            .payload
            .distributed
            .clone()
            .expect("partition catalogue");
        let owner = transport.parameter_operations()?;
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
            &transport,
            binding,
            &intent,
            ParameterOperationKind::Activation,
            local,
            |(targets, estimate, epoch)| {
                let prepared = session
                    .borrow_mut()
                    .with_model_operation(|model| {
                        let mut original = BTreeMap::new();
                        let mut replacement = BTreeMap::new();
                        for target in &targets {
                            let layout = &catalog.layouts[&target.id];
                            let mut ids = BTreeSet::from([target.id.clone()]);
                            layout.extend_dependencies(&mut ids);
                            let candidate = with_selected_parameter_values(
                                model.erased_mut(),
                                &ids,
                                stream,
                                |selected| {
                                    let source = &selected[&target.id];
                                    source.as_array().evaluated()?;
                                    let mut value =
                                        layout.effective(&target.id, selected, stream)?;
                                    for edit in &target.edits {
                                        if let Err(error) = apply_effective_update(
                                            &mut value,
                                            &edit.region,
                                            &edit.update,
                                            stream,
                                        )? {
                                            return Ok(Err(error));
                                        }
                                    }
                                    // Every owner promotes an affected packed parameter,
                                    // even if the selected edit misses this owner's shard.
                                    value.evaluated()?;
                                    Ok(Ok((source.clone(), MlxTensor::from_array(value))))
                                },
                            )?;
                            let (before, after) = match candidate {
                                Ok(pair) => pair,
                                Err(error) => return Ok(Err(error)),
                            };
                            original.insert(target.id.clone(), before);
                            replacement.insert(target.id.clone(), after);
                        }
                        let state = model.erased_mut().prepare_parameter_reset_state()?;
                        Ok(Ok(PreparedOverlay {
                            original,
                            replacement,
                            state,
                            state_exchanged: false,
                        }))
                    })
                    .map_err(failure)??;
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
        let payload = Rc::get_mut(&mut self.payload).expect("completed parameter transaction");
        // Only completed peer publication invalidates reusable snapshots. These
        // final metadata changes cannot submit native work or fail semantically.
        payload.model.erased_mut().invalidate_parameter_snapshots();
        let state = &mut payload.parameter_state;
        state.originals = prepared.original;
        state.published = prepared.replacement;
        state.reset_estimate = Some(estimate);
        state.active = Some(overlay.intent_identity().into());
        state.epoch = epoch;
        state.floating_state_dtype_bytes = None;
        for parameter in &mut catalog.discovery.parameters {
            if overlay.shared_targets().contains(&parameter.shared_id) {
                state
                    .baseline_transforms
                    .insert(parameter.id.clone(), parameter.input_transform.clone());
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
    ) -> Result<ParameterDiscovery, ParameterError> {
        let mut catalog = self.partition_parameter_catalog(None)?;
        let transport = self
            .payload
            .distributed
            .clone()
            .expect("partition catalogue");
        let owner = transport.parameter_operations()?;
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
            for id in state.originals.keys() {
                cost.host_bytes = add(cost.host_bytes, add(512, mul(id.len() as u64, 4)?)?)?;
            }
            NativeParameterBudget {
                total: Rc::clone(&state.usage),
                limit: UNLIMITED,
                operation: None,
            }
            .reserve_quota(cost)?;
            Ok((state.originals.clone(), state.published.clone(), epoch))
        })();
        let mut digest = Sha256::new();
        digest.update(b"eredu-remove-parameter-overlay-v1\0");
        if let Some(active) = &catalog.discovery.overlay_identity {
            digest.update(active.as_bytes());
        }
        let intent = digest.finalize().into();
        let session = RefCell::new(&mut *self);
        let result = owner.transaction(
            &transport,
            binding,
            &intent,
            ParameterOperationKind::Removal,
            local,
            |(replacement, original, epoch)| {
                let state = session
                    .borrow_mut()
                    .with_model_operation(|model| {
                        model.erased_mut().prepare_parameter_reset_state()
                    })
                    .map_err(failure)?;
                Ok((
                    PreparedOverlay {
                        original,
                        replacement,
                        state,
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
        let payload = Rc::get_mut(&mut self.payload).expect("completed parameter restoration");
        payload.model.erased_mut().invalidate_parameter_snapshots();
        let state = &mut payload.parameter_state;
        for parameter in &mut catalog.discovery.parameters {
            if let Some(transform) = state.baseline_transforms.remove(&parameter.id) {
                parameter.input_transform = transform;
            }
        }
        state.originals.clear();
        state.published.clear();
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
        active: bool,
        restore: bool,
    ) -> Result<(), ParameterError> {
        self.with_model_operation(|model| {
            // The outer parameter coordinator supplies the only peer agreement.
            // No nested collective may strand peers on a local publication error.
            if !restore {
                model
                    .erased_mut()
                    .exchange_parameter_reset_state(prepared.state.as_mut())?;
                prepared.state_exchanged = true;
            }
            if !model.erased_mut().publish_parameter_replacements(
                if restore {
                    &prepared.original
                } else {
                    &prepared.replacement
                },
                active,
            )? {
                return Err(Error::ArchitectureModel(
                    "prepared parameter publication is unavailable".into(),
                ));
            }
            if restore && prepared.state_exchanged {
                model
                    .erased_mut()
                    .exchange_parameter_reset_state(prepared.state.as_mut())?;
                prepared.state_exchanged = false;
            }
            Ok(())
        })
        .map_err(failure)?;
        #[cfg(test)]
        if !restore
            && std::mem::take(
                &mut Rc::get_mut(&mut self.payload)
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
