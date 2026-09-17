//! Shared canonical-first, external-alias-second acquisition semantics.
//! The neutral controller supplies the finite visited closure; this layer only
//! binds actual native storage without recursive unit acquisition.
use super::*;

type PreparedHost = (OffloadUnitId, ResidentHostBuffers, u64, u64, u64);

pub(super) fn bind_host(
    state: &ManagerState,
    prepared: &mut [PreparedHost],
) -> Result<(), ResidencyError> {
    let mut aliases = Vec::new();
    for (index, (id, _, _, _, _)) in prepared.iter().enumerate() {
        let unit = state
            .control
            .unit(id)
            .ok_or(ResidencyError::StatePoisoned)?;
        for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
            let (owner, _) = state
                .control
                .binding_owner_borrowed(id, binding)
                .ok_or(ResidencyError::StatePoisoned)?;
            if owner == id {
                continue;
            }
            let source = host_binding(state, id, binding, |owner, name| {
                prepared
                    .iter()
                    .find(|(candidate, ..)| candidate == owner)
                    .and_then(|(_, buffers, ..)| buffers.buffers.get(name))
                    .cloned()
            })?;
            aliases.push((index, binding.name().to_owned(), source));
        }
    }
    for (index, name, source) in aliases {
        if prepared[index].1.buffers.insert(name, source).is_some() {
            return Err(ResidencyError::StatePoisoned);
        }
    }
    Ok(())
}

/// Same canonical owner lookup for ordinary aliases and prepared full rows.
/// A physical buffer handle is cloned; there is no per-unit owner backedge.
pub(in crate::backend::runtime::residency::manager) fn host_binding(
    state: &ManagerState,
    id: &OffloadUnitId,
    binding: &WeightBinding,
    prepared: impl FnOnce(&OffloadUnitId, &str) -> Option<RetainedHostBuffer>,
) -> Result<RetainedHostBuffer, ResidencyError> {
    let (owner, canonical) = if binding.is_alias() {
        state
            .control
            .binding_owner_borrowed(id, binding)
            .ok_or(ResidencyError::StatePoisoned)?
    } else {
        (id, binding)
    };
    prepared(owner, canonical.name())
        .or_else(|| {
            state
                .storage
                .get(owner)?
                .host
                .as_ref()?
                .buffers
                .get(canonical.name())
                .cloned()
        })
        .ok_or(ResidencyError::StatePoisoned)
}

pub(super) fn bind_device(
    state: &ManagerState,
    prepared: &mut [(OffloadUnitId, PreparedResidentArrays)],
) -> Result<(), ResidencyError> {
    if prepared
        .iter()
        .any(|(_, values)| values.arrays.is_prepared())
    {
        if prepared
            .iter()
            .any(|(_, values)| !values.arrays.is_prepared())
        {
            return Err(NamedArrayError::InvalidSource.into());
        }
        return bind_prepared_device(state, prepared);
    }
    let mut aliases = Vec::new();
    for (index, (id, _)) in prepared.iter().enumerate() {
        let unit = state
            .control
            .unit(id)
            .ok_or(ResidencyError::StatePoisoned)?;
        for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
            let (owner, canonical) = state
                .control
                .binding_owner_borrowed(id, binding)
                .ok_or(ResidencyError::StatePoisoned)?;
            if owner == id {
                continue;
            }
            let source = prepared
                .iter()
                .find(|(candidate, _)| candidate == owner)
                .and_then(|(_, values)| values.arrays.get(canonical.name()))
                .or_else(|| {
                    state
                        .storage
                        .get(owner)?
                        .device
                        .as_ref()?
                        .arrays
                        .get(canonical.name())
                })
                .ok_or(ResidencyError::StatePoisoned)?;
            aliases.push((index, binding.name().to_owned(), source.clone()));
        }
    }
    for (index, name, source) in aliases {
        if prepared[index]
            .1
            .arrays
            .insert_ordinary(name, source)?
            .is_some()
        {
            return Err(ResidencyError::StatePoisoned);
        }
    }
    Ok(())
}

fn bind_prepared_device(
    state: &ManagerState,
    prepared: &mut [(OffloadUnitId, PreparedResidentArrays)],
) -> Result<(), ResidencyError> {
    // Every canonical row exists before this phase. A direct physical-cell
    // handle makes even A <-> B unit aliases acyclic in storage ownership.
    for index in 0..prepared.len() {
        let unit = state
            .control
            .unit(&prepared[index].0)
            .ok_or(ResidencyError::StatePoisoned)?;
        let catalog = prepared[index]
            .1
            .arrays
            .catalog()
            .ok_or(NamedArrayError::InvalidSource)?
            .clone();
        for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
            let (owner, canonical) = state
                .control
                .binding_owner_borrowed(unit.id(), binding)
                .ok_or(ResidencyError::StatePoisoned)?;
            let alias = if let Some((_, values)) = prepared.iter().find(|(id, _)| id == owner) {
                values.arrays.alias_value(canonical.name())
            } else {
                state
                    .storage
                    .get(owner)
                    .and_then(|unit| unit.device.as_ref())
                    .and_then(|values| values.alias_value(canonical.name(), owner, &catalog))
            }
            .ok_or(NamedArrayError::MissingValue)?;
            prepared[index].1.arrays.put_alias(binding.name(), alias)?;
        }
    }
    Ok(())
}

pub(in crate::backend::runtime::residency::manager) fn prepare_owner_pins(
    state: &ManagerState,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
) -> Result<(), ResidencyError> {
    for id in ids {
        let unit = state
            .control
            .unit(id)
            .ok_or(ResidencyError::StatePoisoned)?;
        for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
            let (owner, _) = state
                .control
                .binding_owner_borrowed(id, binding)
                .ok_or(ResidencyError::StatePoisoned)?;
            if owner == id || state.alias_owner_pins.contains(owner, tier) {
                continue;
            }
            // The complete finite domain was constructed from these same units.
            state
                .alias_owner_pins
                .position(owner, tier)
                .ok_or(ResidencyError::StatePoisoned)?;
            if state
                .control
                .ledger()
                .copy_status(owner, tier)?
                .is_some_and(|copy| copy.pins() == u64::MAX)
            {
                return Err(
                    eredu_core::residency::ResidencyLedgerError::ArithmeticOverflow {
                        context: "resident lease count",
                    }
                    .into(),
                );
            }
        }
    }
    Ok(())
}

pub(in crate::backend::runtime::residency::manager) fn pin_owners(
    state: &mut ManagerState,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
) -> Result<(), ResidencyError> {
    for id in ids {
        let bindings = state
            .control
            .unit(id)
            .ok_or(ResidencyError::StatePoisoned)?
            .bindings()
            .len();
        for ordinal in 0..bindings {
            let unit = state
                .control
                .unit(id)
                .ok_or(ResidencyError::StatePoisoned)?;
            let binding = &unit.bindings()[ordinal];
            if !binding.is_alias() {
                continue;
            }
            let (owner, _) = state
                .control
                .binding_owner_borrowed(id, binding)
                .ok_or(ResidencyError::StatePoisoned)?;
            if owner == id {
                continue;
            }
            let index = state
                .alias_owner_pins
                .position(owner, tier)
                .ok_or(ResidencyError::StatePoisoned)?;
            if state.alias_owner_pins.is_pinned(index) {
                continue;
            }
            // Borrow the constructor-owned key from a disjoint field. The
            // same locked preflight/commit never clones IDs or grows an index.
            state
                .control
                .ledger_mut()
                .pin(state.alias_owner_pins.id(index), tier, 1)?;
            state.alias_owner_pins.pin(index);
        }
    }
    Ok(())
}
