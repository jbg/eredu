//! Hash the retained physical update list, without rebuilding a source or payload.
use super::*;
use sha2::{Digest, Sha256};
pub(super) fn identity(
    owner: &PreparedPartitionModelIntervention,
) -> std::result::Result<[u8; 32], Failure> {
    let mut hash = Sha256::new();
    hash.update(b"eredu-original-partition-native-intervention-v1\0");
    hash.update(owner.projection.geometry_identity());
    hash.update(
        owner
            .projection
            .source()
            .plan()
            .admission()
            .intent_identity()
            .as_bytes(),
    );
    vector(&mut hash, &owner.shape);
    dtype(&mut hash, owner.dtype);
    hash.update([u8::from(owner.window.is_some())]);
    if let Some(window) = owner.window {
        vector(&mut hash, &window.range());
        hash.update(window.logical_positions().to_le_bytes());
    }
    hash.update((owner.updates.len() as u64).to_le_bytes());
    for update in &owner.updates {
        hash.update((update.region as u64).to_le_bytes());
        for values in [
            &update.slice.starts,
            &update.slice.ends,
            &update.slice.strides,
            &update.slice.shape,
        ] {
            vector(&mut hash, values);
        }
        let original = owner
            .projection
            .update(update.region)
            .ok_or(Failure::ClaimMismatch)?;
        let selected = update
            .payload
            .as_ref()
            .and_then(PreparedWindowInterventionPayload::projected_action)
            .unwrap_or(original.action);
        action(&mut hash, selected)?;
    }
    Ok(hash.finalize().into())
}
fn vector(hash: &mut Sha256, values: &[u64]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
}
fn indices(hash: &mut Sha256, values: &[u32]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
}
fn dtype(hash: &mut Sha256, value: InterventionDtype) {
    hash.update([match value {
        InterventionDtype::Float32 => 0,
        InterventionDtype::Float16 => 1,
        InterventionDtype::Bfloat16 => 2,
    }]);
}
fn action(hash: &mut Sha256, value: &InterventionAction) -> std::result::Result<(), Failure> {
    match value {
        InterventionAction::Zero { dtype: value } => {
            hash.update([0]);
            dtype(hash, *value);
        }
        InterventionAction::Scale {
            dtype: value,
            factor,
        } => {
            hash.update([1]);
            dtype(hash, *value);
            hash.update(factor.to_bits().to_le_bytes());
        }
        InterventionAction::Mask {
            dtype: value,
            shape,
            keep,
        } => {
            hash.update([2]);
            dtype(hash, *value);
            vector(hash, shape);
            hash.update((keep.len() as u64).to_le_bytes());
            for value in keep {
                hash.update([u8::from(*value)]);
            }
        }
        InterventionAction::MaskComponents {
            dtype: value,
            indices: values,
            keep_selected,
        } => {
            hash.update([3]);
            dtype(hash, *value);
            indices(hash, values);
            hash.update([u8::from(*keep_selected)]);
        }
        InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => {
            hash.update([if matches!(value, InterventionAction::Replace { .. }) {
                4
            } else {
                5
            }]);
            dtype(hash, tensor.values.dtype());
            vector(hash, &tensor.shape);
            match &tensor.values {
                InterventionValues::Float32(values) => {
                    hash.update((values.len() as u64).to_le_bytes());
                    for value in values {
                        hash.update(value.to_bits().to_le_bytes());
                    }
                }
                InterventionValues::Float16(values) | InterventionValues::Bfloat16(values) => {
                    hash.update((values.len() as u64).to_le_bytes());
                    for value in values {
                        hash.update(value.to_le_bytes());
                    }
                }
            }
        }
        InterventionAction::MaskLogits {
            dtype: value,
            token_ids,
        } => {
            hash.update([6]);
            dtype(hash, *value);
            indices(hash, token_ids);
        }
        _ => return Err(Failure::ClaimMismatch),
    }
    Ok(())
}

pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Sha256>(),
        size_of::<[u8; 32]>(),
        size_of::<[u8; 8]>(),
        size_of::<(
            &PreparedPartitionModelIntervention,
            &Update,
            &InterventionAction,
        )>(),
        size_of::<(&mut Sha256, &InterventionAction)>(),
        size_of::<(&mut Sha256, &[u64])>(),
        size_of::<(&mut Sha256, &[u32])>(),
        size_of::<std::result::Result<[u8; 32], Failure>>(),
        size_of::<std::result::Result<(), Failure>>(),
        size_of::<[&Vec<u64>; 4]>(),
        size_of::<std::slice::Iter<'_, u64>>(),
        size_of::<std::slice::Iter<'_, f32>>(),
        size_of::<std::slice::Iter<'_, u16>>(),
        size_of::<std::slice::Iter<'_, bool>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
