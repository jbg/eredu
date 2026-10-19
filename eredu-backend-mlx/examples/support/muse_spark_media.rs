//! Digest-checked released media comparison records.
use super::*;

pub(super) fn read_tensor(
    args: &Args,
    record: &Value,
) -> Result<(Vec<f32>, Vec<i32>), Box<dyn Error>> {
    let raw = std::fs::read(
        args.reference
            .join(record["file"].as_str().ok_or("missing tensor file")?),
    )?;
    if sha256(&raw) != record["sha256"] || raw.len() % 4 != 0 {
        return Err("media reference tensor differs from its retained digest".into());
    }
    let shape = record["shape"]
        .as_array()
        .ok_or("missing media tensor shape")?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or("invalid media tensor dimension")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let values = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let count = shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v as usize));
    if count != Some(values.len()) || !values.iter().all(|v| v.is_finite()) {
        return Err("invalid media reference tensor size or values".into());
    }
    Ok((values, shape))
}
