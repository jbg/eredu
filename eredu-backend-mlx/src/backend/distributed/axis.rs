use super::*;

/// Gathers equal-shaped shards along an arbitrary existing tensor axis.
pub fn all_gather_axis(
    input: &Array,
    axis: i32,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    all_gather_axis_for(
        eredu_runtime::CommunicationOperation::AllGatherEven,
        input,
        axis,
        group,
        stream,
    )
}

fn all_gather_axis_for(
    operation: eredu_runtime::CommunicationOperation,
    input: &Array,
    axis: i32,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let _setup = group.begin_bounded_setup()?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Exception::custom(
            "axis all-gather requires a non-scalar input",
        ));
    }
    let ndim_i32 =
        i32::try_from(ndim).map_err(|_| Exception::custom("input rank does not fit in i32"))?;
    let axis = if axis < 0 { axis + ndim_i32 } else { axis };
    if !(0..ndim_i32).contains(&axis) {
        return Err(Exception::custom(format!(
            "all-gather axis {axis} is outside input rank {ndim}"
        )));
    }
    if axis == 0 {
        return distributed::all_gather_for(operation, input, group, stream);
    }
    let gathered = distributed::all_gather_for(operation, input, group, stream)?;
    let rank_height = input.shape()[0];
    let mut shards = Vec::with_capacity(group.size());
    for rank in 0..group.size() {
        let start = i32::try_from(rank)
            .ok()
            .and_then(|rank| rank.checked_mul(rank_height))
            .ok_or_else(|| Exception::custom("gathered rank offset exceeds i32"))?;
        let end = start
            .checked_add(rank_height)
            .ok_or_else(|| Exception::custom("gathered rank end exceeds i32"))?;
        shards.push(gathered.try_index_device(start..end, stream)?);
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    let output = concatenate_axis(&shard_refs, axis, stream)?;
    let mut expected = input.shape().to_vec();
    expected[axis as usize] = expected[axis as usize]
        .checked_mul(
            i32::try_from(group.size())
                .map_err(|_| Exception::custom("group size does not fit in i32"))?,
        )
        .ok_or_else(|| Exception::custom("axis all-gather output shape exceeds i32"))?;
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "axis all-gather completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(output)
}

fn all_gather_axis_unchecked(
    input: &Array,
    axis: i32,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Exception> {
    if axis == 0 {
        return distributed::all_gather_unchecked(input, group, stream);
    }
    let gathered = distributed::all_gather_unchecked(input, group, stream)?;
    let rank_height = input.shape()[0];
    let mut shards = Vec::with_capacity(group.size());
    for rank in 0..group.size() {
        let start = i32::try_from(rank)
            .ok()
            .and_then(|rank| rank.checked_mul(rank_height))
            .ok_or_else(|| Exception::custom("gathered rank offset exceeds i32"))?;
        let end = start
            .checked_add(rank_height)
            .ok_or_else(|| Exception::custom("gathered rank end exceeds i32"))?;
        shards.push(gathered.try_index_device(start..end, stream)?);
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    concatenate_axis(&shard_refs, axis, stream)
}

/// Gathers unequal contiguous shards along an arbitrary tensor axis.
pub fn all_gather_uneven_axis(
    input: &Array,
    axis: i32,
    widths: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let _setup = group.begin_bounded_setup()?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    if widths.len() != group.size() {
        return Err(Exception::custom(format!(
            "uneven all-gather received {} widths for group size {}",
            widths.len(),
            group.size()
        )));
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Exception::custom(
            "uneven axis all-gather requires a non-scalar input",
        ));
    }
    let ndim_i32 =
        i32::try_from(ndim).map_err(|_| Exception::custom("input rank does not fit in i32"))?;
    let axis = if axis < 0 { axis + ndim_i32 } else { axis };
    if !(0..ndim_i32).contains(&axis) {
        return Err(Exception::custom(format!(
            "uneven all-gather axis {axis} is outside input rank {ndim}"
        )));
    }
    let rank = group.rank();
    let local_width = usize::try_from(input.shape()[axis as usize])
        .map_err(|_| Exception::custom("input shape contains a negative dimension"))?;
    if local_width != widths[rank] {
        return Err(Exception::custom(format!(
            "rank {rank} local width {local_width} does not match declared width {}",
            widths[rank]
        )));
    }
    let max_width = widths.iter().copied().max().unwrap_or(0);
    if max_width == 0 {
        return Err(Exception::custom(
            "uneven all-gather requires at least one non-empty shard",
        ));
    }
    let operation = eredu_runtime::CommunicationOperation::AllGatherUneven;
    group.validate_tensor(operation, input, false)?;
    let output_width = widths.iter().try_fold(0usize, |total, width| {
        total
            .checked_add(*width)
            .ok_or_else(|| Exception::custom("uneven all-gather output width overflowed usize"))
    })?;
    let non_axis_elements = input
        .shape()
        .iter()
        .enumerate()
        .filter(|(dimension, _)| *dimension != axis as usize)
        .try_fold(1usize, |total, (_, dimension)| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| Exception::custom("input shape contains a negative dimension"))?;
            total.checked_mul(dimension).ok_or_else(|| {
                Exception::custom("uneven all-gather output elements overflowed usize")
            })
        })?;
    let output_elements = non_axis_elements
        .checked_mul(output_width)
        .ok_or_else(|| Exception::custom("uneven all-gather output elements overflowed usize"))?;
    group.validate_expected_output(operation, input.dtype(), input.ndim(), output_elements)?;
    let padded = if local_width == max_width {
        input.clone()
    } else {
        let mut padding_shape = input.shape().to_vec();
        padding_shape[axis as usize] = i32::try_from(max_width - local_width)
            .map_err(|_| Exception::custom("padding width does not fit in i32"))?;
        let padding = zeros_dtype(&padding_shape, input.dtype(), stream)?;
        concatenate_axis(&[input, &padding], axis, stream)?
    };
    let gathered = all_gather_axis_unchecked(&padded, axis, group, stream)?;
    let group_size = i32::try_from(group.size())
        .map_err(|_| Exception::custom("distributed group size does not fit in i32"))?;
    let padded_shards = gathered.split(group_size, Some(axis), stream)?;
    let mut shards = Vec::with_capacity(widths.len());
    for (padded, &width) in padded_shards.into_iter().zip(widths) {
        if width == max_width {
            shards.push(padded);
        } else {
            let width = i32::try_from(width)
                .map_err(|_| Exception::custom("shard width does not fit in i32"))?;
            shards.push(
                padded
                    .split_axis(&[width], Some(axis), stream)?
                    .into_iter()
                    .next()
                    .expect("one split index produces a leading shard"),
            );
        }
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    let output = concatenate_axis(&shard_refs, axis, stream)?;
    group.validate_tensor(operation, &output, true)?;
    let mut expected = input.shape().to_vec();
    expected[axis as usize] = i32::try_from(output_width)
        .map_err(|_| Exception::custom("uneven all-gather output width exceeds i32"))?;
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "uneven all-gather completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(output)
}

/// Exchanges variable-sized blocks along an arbitrary existing tensor axis.
pub fn all_to_all_v_axis(
    input: &Array,
    axis: usize,
    send_counts: &[usize],
    receive_counts: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array, Exception> {
    let _setup = group.begin_bounded_setup()?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    if input.ndim() == 0 {
        return Err(Exception::custom(
            "variable all-to-all requires a non-scalar input",
        ));
    }
    if axis >= input.ndim() {
        return Err(Exception::custom(format!(
            "variable all-to-all axis {axis} is outside input rank {}",
            input.ndim()
        )));
    }
    if axis == 0 {
        return distributed::all_to_all_v(input, send_counts, receive_counts, group, stream);
    }

    let ndim = input.ndim();
    let axis = i32::try_from(axis)
        .map_err(|_| Exception::custom("variable all-to-all axis exceeds i32"))?;
    let mut to_front = Vec::with_capacity(ndim);
    to_front.push(axis);
    for current in 0..ndim {
        let current = i32::try_from(current)
            .map_err(|_| Exception::custom("variable all-to-all rank exceeds i32"))?;
        if current != axis {
            to_front.push(current);
        }
    }
    let transposed = input.transpose_axes(&to_front, stream)?;
    let exchanged =
        distributed::all_to_all_v(&transposed, send_counts, receive_counts, group, stream)?;
    let mut restore = vec![0i32; ndim];
    for (position, original_axis) in to_front.into_iter().enumerate() {
        restore[original_axis as usize] = i32::try_from(position)
            .map_err(|_| Exception::custom("variable all-to-all rank exceeds i32"))?;
    }
    exchanged.transpose_axes(&restore, stream)
}
