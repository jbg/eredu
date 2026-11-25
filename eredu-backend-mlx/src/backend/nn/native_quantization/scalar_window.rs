use super::*;
use std::cell::Cell;
thread_local! {
    static ACTIVE_STREAM: Cell<Option<(i32, i32)>> = const { Cell::new(None) };
    static Q4_WINDOW: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q6_WINDOW: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn scope_scalar_projections(
    stream: &Stream,
    rows: i32,
) -> Option<eredu_nn::ScalarProjectionScope> {
    if !(2..=8).contains(&rows) || !uses_metal_device(stream.get_device().ok()?.get_type().ok()?) {
        return None;
    }
    let index = stream.get_index().ok()?;
    let previous = ACTIVE_STREAM.replace(Some((index, rows)));
    Some(eredu_nn::ScalarProjectionScope::new(move || {
        ACTIVE_STREAM.set(previous)
    }))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn has_scalar_projection_scope(stream_index: i32) -> bool {
    ACTIVE_STREAM
        .get()
        .is_some_and(|(index, _)| index == stream_index)
}

/// Shares each resident packed block across a small window of activation rows.
/// Each row keeps its own baseline matrix-vector accumulator and lane reduction.
pub(super) fn project(
    input: &Array,
    view: &NativeQuantizedTensor,
    tokens: i32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    if dtype != Dtype::Float32
        || !ACTIVE_STREAM
            .get()
            .is_some_and(|(index, rows)| rows == tokens && stream.get_index().ok() == Some(index))
    {
        return Ok(None);
    }

    let (slot, name, source, rows) = match view.format() {
        NativeQuantizationFormat::GgufQ4K => (
            &Q4_WINDOW,
            "native_q4k_scalar_window",
            include_str!("q4k_scalar_window.metal"),
            if view.columns <= 2048 { 4 } else { 2 },
        ),
        NativeQuantizationFormat::GgufQ6K => (
            &Q6_WINDOW,
            "native_q6k_scalar_window",
            include_str!("q6k_scalar_window.metal"),
            if view.rows >= 65_536 { 1 } else { 2 },
        ),
        _ => return Ok(None),
    };
    const GROUPS: i32 = 2;
    let workgroups = (view.rows - 1) / (rows * GROUPS) + 1;
    let config = CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / 256)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("ROWS_PER_SIMD", rows)
        .with_template_arg_int("TOKENS", tokens)
        .with_grid([REDUCTION_TILE, workgroups * GROUPS, 1])
        .with_thread_group([REDUCTION_TILE, GROUPS, 1])
        .with_output_arg([tokens, view.rows], dtype);

    slot.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(MetalKernel::new(
                name,
                ["input", "weight"],
                ["out"],
                source,
                "",
                true,
                false,
            )?);
        }
        cell.borrow()
            .as_ref()
            .expect("packed window kernel initialized")
            .apply_one_device([input, view.storage.bytes()], &config, stream)
            .map(Some)
    })
}
