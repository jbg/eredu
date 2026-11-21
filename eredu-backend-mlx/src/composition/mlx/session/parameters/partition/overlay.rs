//! Reversible local tensor and execution-state replacement.
use super::*;

pub(super) fn base_overlay_cost(
    layout: &EffectiveParameterLayout,
    n: u64,
) -> Result<(CaptureUsage, u64), ParameterError> {
    if n > i32::MAX as u64 {
        return Err(ParameterError::Unsupported(
            "local edited parameter exceeds native i32 indexing".into(),
        ));
    }
    let base_cost = CaptureUsage {
        retained_bytes: add(mul(n, 48)?, encoding::conversion_bytes(layout)?)?,
        host_bytes: encoding::host_conversion_bytes(layout)?,
        ..Default::default()
    };
    Ok((base_cost, mul(n, 8)?))
}

pub(super) fn prepare_replacement<'a>(
    selected: &BTreeMap<String, MlxTensor>,
    layout: &EffectiveParameterLayout,
    id: &str,
    edits: impl Iterator<Item = (&'a ParameterRegion, &'a ParameterUpdate)>,
    stream: &Stream,
) -> Result<Result<(MlxTensor, MlxTensor), ParameterError>, Error> {
    let source = &selected[id];
    source.as_array().evaluated()?;
    let mut value = encoding::effective(layout, id, selected, stream)?;
    for (region, update) in edits {
        if let Err(error) = apply_effective_update(&mut value, region, update, stream)? {
            return Ok(Err(error));
        }
    }
    // Packed parameters can be promoted even on a shard with no edited cells.
    value.evaluated()?;
    Ok(Ok((source.clone(), MlxTensor::from_array(value))))
}
