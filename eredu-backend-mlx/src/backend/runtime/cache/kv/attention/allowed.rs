//! Actual integer-coordinate page mask, shared by ordinary and original work.
use super::*;
use eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry;

pub(super) fn mask(
    state: &BlockwiseAttentionAccumulator,
    start: i64,
    end: i64,
    stream: &Stream,
) -> Result<Array, Exception> {
    if state.causal {
        match AbsoluteAttentionMaskGeometry::new(
            state.query_start,
            state.query_len,
            start,
            end,
            state.sliding_window,
            state.prefix_tokens,
        ) {
            Ok(geometry) => return integer_mask(geometry, stream),
            Err(cause) => {
                // Keep the existing ordinary wide-coordinate worker. Original
                // constructors require the represented exact integer profile.
                if safemlx::OriginalScopeObserver::try_current()?.is_some() {
                    return Err(Exception::from_source(cause));
                }
            }
        }
    } else if safemlx::OriginalScopeObserver::try_current()?.is_some() {
        return safemlx::ops::full::<bool>(
            &[state.query_len, (end - start) as i32],
            Array::try_from_bool(true)?,
            stream,
        );
    }
    let allowed = if state.causal {
        absolute_attention_mask(
            state.query_start,
            state.query_len,
            start,
            end,
            state.sliding_window,
            state.prefix_tokens,
        )
    } else {
        vec![true; state.query_len as usize * (end - start) as usize]
    };
    Array::try_from_slice(&allowed, &[state.query_len, (end - start) as i32])
}
fn integer_mask(g: AbsoluteAttentionMaskGeometry, stream: &Stream) -> Result<Array, Exception> {
    let query = Array::arange::<_, i32>(g.query_start(), g.query_end(), None, stream)?
        .reshape(&[g.queries(), 1], stream)?;
    let key = Array::arange::<_, i32>(g.key_start(), g.key_end(), None, stream)?
        .reshape(&[1, g.keys()], stream)?;
    let mut mask = query.ge(&key, stream)?;
    if let Some(window) = g.window() {
        let first = query.subtract(Array::try_from_int(window - 1)?, stream)?;
        let mut visible = key.ge(&first, stream)?;
        if g.prefix() > 0 {
            visible =
                visible.logical_or(&key.lt(Array::try_from_int(g.prefix())?, stream)?, stream)?;
        }
        mask = mask.logical_and(&visible, stream)?;
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires native CPU integer mask execution"]
    fn page_mask_preserves_large_integer_coordinates_prefix_and_window_edges() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        for (query, keys, end, window, prefix) in [
            (16_777_217, 16_777_216, 16_777_221, Some(2), 0),
            (16_777_217, 0, 3, Some(2), 2),
            (
                i64::from(i32::MAX) - 3,
                i64::from(i32::MAX) - 5,
                i64::from(i32::MAX),
                Some(i32::MAX),
                0,
            ),
            (8, 5, 11, None, 0),
        ] {
            let geometry =
                AbsoluteAttentionMaskGeometry::new(query, 3, keys, end, window, prefix).unwrap();
            let expected = absolute_attention_mask(query, 3, keys, end, window, prefix);
            let actual = integer_mask(geometry, &stream).unwrap();
            assert_eq!(
                actual.evaluated().unwrap().as_slice::<bool>(),
                expected.as_slice()
            );
        }
    }
}
