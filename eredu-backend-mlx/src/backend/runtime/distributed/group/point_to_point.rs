use super::collectives::{logical_direct_exchange, pack_logical_value};
use super::*;

fn checked_peer(peer: usize, group: &Group, role: &str) -> Result<()> {
    if group.size() == 1 {
        return Err(Exception::custom(format!(
            "cannot use a {role} rank with a singleton distributed group"
        )));
    }
    if peer >= group.size() {
        return Err(Exception::custom(format!(
            "invalid {role} rank {peer} for distributed group of size {}",
            group.size()
        )));
    }
    Ok(())
}

/// Sends `input` to a rank in a native or logical group.
pub fn send(
    input: &Array,
    destination: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.ensure_available()?;
    checked_peer(destination, group, "destination")?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    if group.logical.is_none() {
        return native::send(input, destination, &group.native, stream);
    }
    let stream = stream.as_ref();
    if let Some(exchanged) = logical_direct_exchange(input, group, stream)? {
        return Ok(exchanged);
    }
    let logical = group.logical.as_ref().expect("logical group");
    if !logical.world_collective_wave {
        return Err(Exception::custom(
            "logical send cannot use a world collective without a consensus-proven participation wave",
        ));
    }
    let source_global = logical.global_ranks[logical.rank];
    let packed = pack_logical_value(input, source_global, group.native.size(), stream)?;
    native::all_sum(&packed, &group.native, stream)?.try_index_device(
        i32::try_from(source_global)
            .map_err(|_| Exception::custom("logical source rank does not fit in i32"))?,
        stream,
    )
}

/// Receives an array from a rank in a native or logical group.
pub fn recv(
    shape: &[i32],
    dtype: Dtype,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.ensure_available()?;
    checked_peer(source, group, "source")?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    if group.logical.is_none() {
        return native::recv(shape, dtype, source, &group.native, stream);
    }
    let stream = stream.as_ref();
    let empty = zeros_dtype(shape, dtype, stream)?;
    recv_like(&empty, source, group, stream)
}

/// Receives from a rank using `like` for shape and dtype.
pub(crate) fn recv_like(
    like: &Array,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.ensure_available()?;
    checked_peer(source, group, "source")?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    if group.logical.is_none() {
        return native::recv_like(like, source, &group.native, stream);
    }
    let stream = stream.as_ref();
    if let Some(exchanged) = logical_direct_exchange(like, group, stream)? {
        return Ok(exchanged);
    }
    let logical = group.logical.as_ref().expect("logical group");
    if !logical.world_collective_wave {
        return Err(Exception::custom(
            "logical receive cannot use a world collective without a consensus-proven participation wave",
        ));
    }
    let source_global = logical.global_ranks[source];
    let packed = pack_logical_value(like, source_global, group.native.size(), stream)?;
    native::all_sum(&packed, &group.native, stream)?.try_index_device(
        i32::try_from(source_global)
            .map_err(|_| Exception::custom("logical source rank does not fit in i32"))?,
        stream,
    )
}
