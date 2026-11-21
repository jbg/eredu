//! Plans exact checkpoint reads for admitted host and device residency copies.

use std::collections::BTreeMap;

use eredu_core::{
    residency::{
        parameter_binding::WeightBinding, MemoryTier, RealizedResidency, ResidencyCopyRequest,
        ResidencyCopyScope, ResidencyStorage, TransferDirection,
    },
    BackendFailure,
};

use crate::exact_parameter_materialization::reads::{
    classify_reads, materialize_classified_reads, ordinary_input, DirectReads,
};

pub(super) fn realize<M: ResidencyStorage>(
    storage: &M,
    requests: &[ResidencyCopyRequest<'_>],
    tier: MemoryTier,
    submit: bool,
) -> Result<RealizedResidency<M::Copy, M::Transfer>, M::Error> {
    let mut scope = storage.begin_realization(tier)?;
    let mut copies = Vec::with_capacity(requests.len());
    if tier == MemoryTier::Host {
        for request in requests {
            let mut buffers = BTreeMap::new();
            for binding in request
                .bindings
                .iter()
                .filter(|binding| !binding.is_alias())
            {
                let input = ordinary_input(request.source, binding)?;
                let value = scope.read_host(request.id, binding, input)?;
                buffers.insert(binding.name().to_owned(), value);
            }
            add_aliases(request, &mut buffers, |alias| scope.host_alias(alias))?;
            copies.push(scope.host_owner(request, buffers)?);
        }
        return Ok(RealizedResidency {
            copies,
            transfer: None,
        });
    }

    // Admit all direct reads for a unit together. A host copy, alias or conversion
    // sends the entire unit through its ordinary path.
    let mut direct = BTreeMap::new();
    let mut ordinary = BTreeMap::new();
    let mut batch = DirectReads::new();
    for request in requests {
        if scope.has_host_copy(request.id) {
            continue;
        }
        let classified = classify_reads(
            request.source,
            request
                .bindings
                .iter()
                .filter(|binding| !binding.is_alias()),
            &scope,
        )?;
        if classified.is_empty()
            || !request.aliases.is_empty()
            || classified.iter().any(|(_, read)| read.is_none())
        {
            ordinary.insert(request.id, classified);
            continue;
        }
        for (binding, read) in classified {
            let read = read.expect("all direct reads were admitted");
            if let Some(reads) = batch.push(binding.expected_bytes(), (request.id, binding, read)) {
                stage_direct(&mut scope, reads, &mut direct)?;
            }
        }
    }
    if let Some(reads) = batch.take() {
        stage_direct(&mut scope, reads, &mut direct)?;
    }

    for request in requests {
        let (mut arrays, direction) = if let Some(arrays) = direct.remove(request.id) {
            (arrays, TransferDirection::DiskToDevice)
        } else if scope.has_host_copy(request.id) {
            (
                scope.copy_from_host(
                    request.id,
                    &request
                        .bindings
                        .iter()
                        .filter(|binding| !binding.is_alias())
                        .map(WeightBinding::name)
                        .collect::<Vec<_>>(),
                )?,
                TransferDirection::HostToDevice,
            )
        } else {
            let classified = ordinary
                .remove(request.id)
                .expect("disk bindings were classified");
            let owners = materialize_classified_reads(request.source, classified, &mut scope)?;
            (owners, TransferDirection::DiskToDevice)
        };
        add_aliases(request, &mut arrays, |alias| scope.device_alias(alias))?;
        copies.push(scope.device_owner(request, arrays, direction)?);
    }

    let transfer = scope.finish_device(&copies, submit)?;
    Ok(RealizedResidency { copies, transfer })
}

fn add_aliases<V: Clone, E: From<BackendFailure>>(
    request: &ResidencyCopyRequest<'_>,
    values: &mut BTreeMap<String, V>,
    mut external: impl FnMut(&eredu_core::residency::ResidentBindingAlias) -> Result<V, E>,
) -> Result<(), E> {
    for alias in &request.aliases {
        let value = if &alias.owner == request.id {
            values.get(&alias.binding).cloned().ok_or_else(|| {
                BackendFailure::from_error(std::io::Error::other(format!(
                    "alias {:?} has no materialized owner {:?}",
                    alias.target, alias.binding
                )))
            })?
        } else {
            external(alias)?
        };
        values.insert(alias.target.clone(), value);
    }
    Ok(())
}

type DirectOwner<'a, R> = (
    &'a eredu_core::residency::OffloadUnitId,
    &'a WeightBinding,
    R,
);
type DirectValues<V> = BTreeMap<eredu_core::residency::OffloadUnitId, BTreeMap<String, V>>;

fn stage_direct<S: ResidencyCopyScope>(
    scope: &mut S,
    reads: Vec<DirectOwner<'_, S::DirectRead>>,
    direct: &mut DirectValues<S::Value>,
) -> Result<(), S::Error> {
    let (owners, reads): (Vec<_>, Vec<_>) = reads
        .into_iter()
        .map(|(id, binding, read)| ((id, binding.name()), (read, binding.storage_layout())))
        .unzip();
    let pending = scope.submit_direct(reads)?;
    let values = scope.take_values(pending)?;
    if owners.len() != values.len() {
        return Err(BackendFailure::from_error(std::io::Error::other(
            "native residency read returned an incorrect number of values",
        ))
        .into());
    }
    for ((id, name), value) in owners.into_iter().zip(values) {
        if direct
            .entry(id.clone())
            .or_default()
            .insert(name.into(), value)
            .is_some()
        {
            return Err(BackendFailure::from_error(std::io::Error::other(format!(
                "duplicate physical parameter {name:?} in unit {id}"
            )))
            .into());
        }
    }
    Ok(())
}
