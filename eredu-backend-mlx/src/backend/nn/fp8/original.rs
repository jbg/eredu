//! Checked original construction profiles of the shared FP8 workers.
use super::*;
use safemlx::OriginalScopeObserver;
use std::mem::size_of;

pub(crate) fn geometry(input: &[i32], weight: &[i32]) -> Option<(i32, i32, i32, i32)> {
    let (&width, prefix) = input.split_last()?;
    if weight.len() != 2 || width <= 0 || weight[0] <= 0 || weight[1] != width {
        return None;
    }
    let rows = prefix
        .iter()
        .try_fold(1i32, |n, &d| if d > 0 { n.checked_mul(d) } else { None })?;
    let scales = width / 128 + i32::from(width % 128 != 0);
    rows.checked_mul(width)?;
    rows.checked_mul(scales)?.checked_mul(128)?;
    if rows <= TILED_ROW_THRESHOLD {
        (weight[0] / 16 + i32::from(weight[0] % 16 != 0)).checked_mul(16)?;
        rows.checked_mul(16)?;
    } else {
        rows.checked_mul(weight[0])?;
    }
    Some((rows, width, weight[0], scales))
}

pub(super) fn validate(
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Option<i32>, Exception> {
    let Some(observer) = OriginalScopeObserver::try_current()? else {
        return Ok(None);
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        // Whole and observed projection consume the same prepared kernels;
        // the recorder separately includes the actual reconstruction program.
        if stream.device_type()? != DeviceType::Gpu
            || !matches!(
                input.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
            || weight.dtype() != Dtype::Uint8
        {
            return Err(observer.invalid_input_error());
        }
        let (rows, _, _, _) = geometry(input.shape(), weight.shape())
            .ok_or_else(|| observer.invalid_input_error())?;
        crate::backend::nn::fp8::kernel::validate_call()?;
        Ok(Some(rows))
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, stream);
        Err(observer.capacity_error())
    }
}

pub(super) fn invalid(message: std::fmt::Arguments<'_>) -> Exception {
    match OriginalScopeObserver::try_current() {
        Ok(Some(observer)) => observer.invalid_input_error(),
        Err(cause) => cause,
        Ok(None) => Exception::custom(message.to_string()),
    }
}

pub(crate) fn control_bytes() -> Option<usize> {
    let native = crate::backend::nn::fp8::kernel::control_bytes()?
        .checked_add(Stream::device_type_control_bytes()?)?
        .checked_add(safemlx::ops::reshape_like_prefix_control_bytes()?)?;
    [
        size_of::<QuantizedActivations>(),
        size_of::<Result<QuantizedActivations, Exception>>(),
        size_of::<[Array; 2]>(),
        size_of::<Result<[Array; 2], Exception>>(),
        size_of::<[&Array; 4]>(),
        size_of::<[i32; 2]>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<Option<i32>, Exception>>(),
        size_of::<Option<OriginalScopeObserver>>(),
        size_of::<&[i32]>(),
        size_of::<Option<&mut dyn super::super::linear::NativeProjectionInputObserver>>(),
        14 * size_of::<Array>(),
        10 * size_of::<i32>(),
        3 * size_of::<bool>(),
        2 * size_of::<Dtype>(),
        size_of::<std::fmt::Arguments<'_>>(),
        OriginalScopeObserver::control_bytes()?,
    ]
    .into_iter()
    .try_fold(native, usize::checked_add)
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ReconstructionFailure {
    #[source]
    cause: eredu_nn::ProjectionObservationError,
    _custody: Exception,
}
/// Preserve the actual fixed semantic error and the accepted native failure
/// carrier. The observed prepare receipt pays the closed erasure beforehand.
pub(super) fn reconstruction_error(cause: eredu_nn::ProjectionObservationError) -> Exception {
    let custody = match OriginalScopeObserver::try_current() {
        Ok(None) => return Exception::from_source(cause),
        Ok(Some(scope)) => scope.invalid_input_error(),
        Err(cause) => cause,
    };
    Exception::from_retained_source(ReconstructionFailure {
        cause,
        _custody: custody,
    })
}
pub(crate) fn reconstruction_control_bytes() -> Option<usize> {
    let controls = [
        Exception::retained_source_control_bytes::<ReconstructionFailure>()?,
        safemlx::ops::indexing::inline_basic_index_control_bytes()?,
        size_of::<InputReconstruction<'_>>(),
        size_of::<eredu_nn::BlockFp8InputReconstructionPlan<'_>>(),
        size_of::<
            Result<
                eredu_nn::BlockFp8InputReconstructionPlan<'_>,
                eredu_nn::ProjectionObservationError,
            >,
        >(),
        size_of::<eredu_nn::GeneratedTensorSource>(),
        size_of::<Result<eredu_nn::GeneratedTensorSource, eredu_nn::ProjectionObservationError>>(),
        size_of::<Result<(), eredu_nn::ProjectionObservationError>>(),
        size_of::<&mut dyn eredu_nn::RetainedGeneratedTensorFactory<Array, Exception>>(),
        size_of::<&mut dyn FnMut(&Array) -> Result<(), Exception>>(),
        size_of::<
            &mut dyn FnMut(eredu_nn::GeneratedTensorSourceRole, &Array) -> Result<(), Exception>,
        >(),
        7 * size_of::<Array>(),
        size_of::<[i32; 2]>(),
        size_of::<[i32; 3]>(),
        size_of::<Result<Array, Exception>>(),
    ];
    controls
        .into_iter()
        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
}

/// Cold and runtime validation of the same contiguous grouped shader indices.
pub(crate) fn grouped_geometry(
    input: &[i32],
    weight: &[i32],
    scale: &[i32],
    ids: &[i32],
) -> Option<(i32, i32, i32, i32)> {
    grouped_geometry_with_layout(
        input,
        weight,
        scale,
        ids,
        eredu_nn::LinearRowLayout::Contiguous,
    )
}

pub(crate) fn grouped_geometry_with_layout(
    input: &[i32],
    weight: &[i32],
    scale: &[i32],
    ids: &[i32],
    layout: eredu_nn::LinearRowLayout,
) -> Option<(i32, i32, i32, i32)> {
    if input.len() != 2
        || weight.len() != 3
        || scale.len() != 3
        || ids != [input[0]]
        || weight[0] <= 0
        || weight[2] != input[1]
    {
        return None;
    }
    let geometry = geometry(input, &weight[1..])?;
    let (routes, width, output, columns) = geometry;
    if scale
        != [
            weight[0],
            i32::try_from(layout.scale_rows_fixed(output as usize, 128).ok()?).ok()?,
            columns,
        ]
    {
        return None;
    }
    // Kernels use uint for flattened source addresses. Check the actual full
    // banks as well as route/grid products, independently of selected IDs.
    (routes as u32).checked_mul(output as u32)?;
    u32::try_from(weight[0])
        .ok()?
        .checked_mul(output as u32)?
        .checked_mul(width as u32)?;
    scale
        .iter()
        .try_fold(1u32, |n, &d| n.checked_mul(u32::try_from(d).ok()?))?;
    Some(geometry)
}
pub(super) fn validate_grouped(
    input: &Array,
    weight: &Array,
    scale: &Array,
    ids: &Array,
    layout: eredu_nn::LinearRowLayout,
    stream: &Stream,
) -> Result<(), Exception> {
    let Some(observer) = OriginalScopeObserver::try_current()? else {
        return Ok(());
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        if stream.device_type()? != DeviceType::Gpu
            || !matches!(
                input.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
            || weight.dtype() != Dtype::Uint8
            || ids.dtype() != Dtype::Int32
            || !matches!(
                scale.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16 | Dtype::Uint8
            )
            || grouped_geometry_with_layout(
                input.shape(),
                weight.shape(),
                scale.shape(),
                ids.shape(),
                layout,
            )
            .is_none()
        {
            return Err(observer.invalid_input_error());
        }
        crate::backend::nn::fp8::kernel::validate_call()?;
        Ok(())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, scale, ids, layout, stream);
        Err(observer.capacity_error())
    }
}
pub(crate) fn grouped_control_bytes() -> Option<usize> {
    let controls = [
        crate::backend::nn::fp8::kernel::grouped_control_bytes()?,
        Stream::device_type_control_bytes()?,
        OriginalScopeObserver::control_bytes()?,
        size_of::<QuantizedActivations>(),
        size_of::<Result<QuantizedActivations, Exception>>(),
        size_of::<[&Array; 5]>(),
        size_of::<[i32; 2]>(),
        size_of::<[i32; 3]>(),
        4 * size_of::<&[i32]>(),
        size_of::<Option<(i32, i32, i32, i32)>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        size_of::<eredu_nn::LinearRowLayout>(),
        size_of::<std::fmt::Arguments<'_>>(),
        12 * size_of::<i32>(),
        3 * size_of::<Dtype>(),
        6 * size_of::<Array>(),
    ];
    controls
        .into_iter()
        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
}
