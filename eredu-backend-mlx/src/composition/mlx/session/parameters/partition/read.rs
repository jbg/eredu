//! Effective local reads and contractions over admitted tensor regions.
use super::*;

pub(super) fn parameter_read_cost(
    layout: &EffectiveParameterLayout,
    fragments: &[ParameterReadFragment],
) -> Result<CaptureUsage, ParameterError> {
    let mut cost = CaptureUsage::default();
    for ParameterReadFragment {
        region,
        projection,
        output_elements,
    } in fragments
    {
        let selected = elements(&region.shape)?;
        let count = *output_elements;
        let directions = projection
            .as_ref()
            .map_or(0, |p| p.projection().coefficients.len() as u64);
        cost.retained_bytes = add(
            cost.retained_bytes,
            add(
                mul(selected, 32)?,
                add(mul(count, 16)?, mul(directions, 8)?)?,
            )?,
        )?;
        cost.host_bytes = add(cost.host_bytes, add(mul(count, 16)?, mul(directions, 8)?)?)?;
    }
    let source = elements(&layout.shape)?;
    if source > i32::MAX as u64 {
        return Err(ParameterError::Unsupported(
            "local native parameter exceeds i32 indexing".into(),
        ));
    }
    cost.retained_bytes = add(
        cost.retained_bytes,
        add(
            mul(
                source,
                if layout.format == eredu_checkpoint::LinearFormat::Dense {
                    8
                } else {
                    32
                },
            )?,
            encoding::conversion_bytes(layout)?,
        )?,
    )?;
    cost.host_bytes = add(cost.host_bytes, encoding::host_conversion_bytes(layout)?)?;

    Ok(cost)
}

pub(super) fn read_parameter_fragments(
    selected: &BTreeMap<String, MlxTensor>,
    layout: &EffectiveParameterLayout,
    parameter: &str,
    fragments: &[ParameterReadFragment],
    words: usize,
    stream: &Stream,
) -> Result<Vec<u32>, Error> {
    let tensor = encoding::effective(layout, parameter, selected, stream)?;
    let mut output = Vec::with_capacity(words);
    for ParameterReadFragment {
        region, projection, ..
    } in fragments
    {
        let values = match projection {
            None => encoding::read_effective(&tensor, region, stream)?,
            Some(projection) => {
                encoding::project_effective(&tensor, projection.projection(), stream)?
            }
        };
        output.extend(values.into_iter().map(f32::to_bits));
    }
    Ok(output)
}

pub(super) fn local_read_cost(
    layout: &EffectiveParameterLayout,
    source: u64,
    selected: u64,
    output: u64,
    directions: Option<u64>,
) -> Result<CaptureUsage, ParameterError> {
    if source > i32::MAX as u64 || output > i32::MAX as u64 {
        return Err(ParameterError::Unsupported(if directions.is_some() {
            "native projection exceeds i32 indexing".into()
        } else {
            "native indexing exceeds i32".into()
        }));
    }
    let (work, host) = match directions {
        None => (mul(output, 32)?, mul(output, 16)?),
        Some(directions) => (
            add(
                mul(selected, 16)?,
                add(mul(directions, 8)?, mul(output, 16)?)?,
            )?,
            add(mul(directions, 8)?, mul(output, 16)?)?,
        ),
    };
    let width = if layout.format == eredu_checkpoint::LinearFormat::Dense {
        8
    } else {
        32
    };
    Ok(CaptureUsage {
        retained_bytes: add(
            add(4096, encoding::conversion_bytes(layout)?)?,
            add(mul(source, width)?, work)?,
        )?,
        host_bytes: add(add(256, host)?, encoding::host_conversion_bytes(layout)?)?,
        ..Default::default()
    })
}

pub(super) fn read_parameter_values(
    selected: &BTreeMap<String, MlxTensor>,
    layout: &EffectiveParameterLayout,
    parameter: &str,
    region: &ParameterRegion,
    projection: Option<&ParameterProjection>,
    stream: &Stream,
) -> Result<Vec<f32>, Error> {
    let tensor = encoding::effective(layout, parameter, selected, stream)?;
    match projection {
        Some(projection) => encoding::project_effective(&tensor, projection, stream),
        None => encoding::read_effective(&tensor, region, stream),
    }
}
