use super::*;

pub(super) fn preflight_source_collisions(
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    plan: &BoundedQuantizationPlan,
) -> Result<(), Error> {
    let source_keys = source.source_keys().into_iter().collect::<BTreeSet<_>>();
    for target in plan.targets() {
        if source_keys.contains(target.scales_name())
            || target
                .biases_name()
                .is_some_and(|name| source_keys.contains(name))
        {
            return Err(quantization_error(format!(
                "bounded quantization target {:?} already has checkpoint quantization companions; implicit transcoding is unsupported",
                target.weight_name()
            )));
        }
    }
    Ok(())
}

pub(super) fn checked_product(dimensions: &[usize], context: &'static str) -> Result<usize, Error> {
    dimensions.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| quantization_error(format!("{context} overflow")))
    })
}

pub(super) fn report_source_bytes(rows: usize, one_row_bytes: u64) -> Result<u64, Error> {
    one_row_bytes
        .checked_mul(rows as u64)
        .ok_or_else(|| quantization_error("source byte count overflow"))
}

pub(super) fn quantization_error(message: impl Into<String>) -> Error {
    Error::Quantization(message.into())
}
