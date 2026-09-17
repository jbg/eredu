//! One exact geometry digest for both owning projection constructors.
use super::*;
pub(super) fn identity<'a>(plan: &AdmittedInterventionPlan, operation: usize, local_columns: bool,
    sum_offset_owner: Option<bool>, shape: &[u64], coordinates: &ComponentCoordinateMap,
    count: usize, mut region: impl FnMut(usize) -> (&'a ResolvedCaptureSlice, Option<&'a ResolvedCaptureSlice>)) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"eredu-partition-activation-geometry-v2\0");
    digest.update(plan.intent_identity().as_bytes());
    digest.update((operation as u64).to_le_bytes());
    digest.update([u8::from(local_columns)]);
    digest.update([match sum_offset_owner { None => 0, Some(false) => 1, Some(true) => 2 }]);
    let vector = |digest: &mut Sha256, values: &[u64]| {
        digest.update((values.len() as u64).to_le_bytes());
        for value in values { digest.update(value.to_le_bytes()); }
    };
    vector(&mut digest, shape);
    digest.update((coordinates.global_count() as u64).to_le_bytes());
    digest.update((coordinates.local_count() as u64).to_le_bytes());
    for local in 0..coordinates.local_count() {
        digest.update((coordinates.local_to_global(local).expect("retained coordinate") as u64).to_le_bytes());
    }
    digest.update((count as u64).to_le_bytes());
    for index in 0..count {
        let (local, destination) = region(index);
        digest.update([u8::from(destination.is_some())]);
        for slice in std::iter::once(local).chain(destination) {
            for values in [&slice.starts, &slice.ends, &slice.strides, &slice.shape] { vector(&mut digest, values); }
        }
    }
    digest.finalize().into()
}
