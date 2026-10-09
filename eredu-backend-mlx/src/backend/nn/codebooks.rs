//! Native ordered lookup/sum over generic stacked codebooks.

use crate::MlxTensor;
use eredu_nn::Error;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use eredu_nn::Tensor;
use safemlx::Stream;

pub(crate) fn ordered_lookup_sum(
    table: &MlxTensor,
    codes: &MlxTensor,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    eredu_nn::codebooks::validate(table, codes)?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if stream
        .get_device()
        .map_err(Error::backend_source)?
        .get_type()
        .map_err(Error::backend_source)?
        == safemlx::DeviceType::Gpu
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            Dtype,
        };
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        let depth = table.dim(0);
        let vocabulary = table.dim(1);
        let width = table.dim(2);
        let shape = [codes.dim(0), codes.dim(1), width];
        let elements = i32::try_from(
            i64::from(shape[0])
                .checked_mul(i64::from(shape[1]))
                .and_then(|n| n.checked_mul(i64::from(width)))
                .ok_or_else(|| Error::backend("ordered codebook sum grid overflow"))?,
        )
        .map_err(|_| Error::backend("ordered codebook sum grid overflow"))?;
        // Match ordinary gather's signed wrap-from-end indices plus PAD.
        let ids = codes.as_array();
        let valid = ids
            .ge(safemlx::Array::from_int(-vocabulary), stream)
            .and_then(|valid| {
                valid.logical_and(
                    &ids.le(safemlx::Array::from_int(vocabulary), stream)?,
                    stream,
                )
            })
            .map_err(Error::backend_source)?;
        super::tensor::register_device_validation(
            valid
                .logical_not(stream)
                .and_then(|invalid| invalid.any(false, stream))
                .map_err(Error::backend_source)?,
            "ordered codebook index is outside its signed gather domain and PAD",
        )
        .map_err(Error::backend_source)?;
        let table = table
            .as_array()
            .contiguous(false, stream)
            .map_err(Error::backend_source)?;
        let ids = ids
            .contiguous(false, stream)
            .map_err(Error::backend_source)?;
        let config = CustomKernelConfig::new()
            .with_template_arg_int("DEPTH", depth)
            .with_template_arg_int("VOCABULARY", vocabulary)
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("ELEMENTS", elements)
            .with_grid([elements, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg(shape, Dtype::Float32);
        let output = KERNEL
            .with(|cell| {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(MetalKernel::new(
                        "ordered_codebook_lookup_sum",
                        ["table", "codes"],
                        ["output"],
                        include_str!("codebooks.metal"),
                        "",
                        true,
                        false,
                    )?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("initialized codebook kernel")
                    .apply_one_device([&table, &ids], &config, stream)
            })
            .map_err(Error::backend_source)?;
        return Ok(MlxTensor::from_array(output));
    }
    eredu_nn::codebooks::ordered_codebook_lookup_sum(table, codes, stream)
}
