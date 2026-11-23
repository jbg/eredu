//! Bounded Metal reductions that preserve contiguous BF16 attention order.

use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

/// The retained normalization and value roots of a three-sweep attention scan.
#[derive(Default)]
pub(crate) struct OrderedBf16Attention {
    pub(crate) mass: Option<Array>,
    pub(crate) values: Option<Array>,
}

impl OrderedBf16Attention {
    pub(crate) fn supported(stream: &Stream) -> Result<bool, Exception> {
        Ok(cfg!(all(feature = "metal", not(feature = "cuda")))
            && stream.get_device()?.get_type()? == safemlx::DeviceType::Gpu)
    }

    pub(crate) fn roots(&self) -> impl Iterator<Item = &Array> {
        self.mass.iter().chain(&self.values)
    }

    /// Owned FP32 roots and overlapping accumulator updates.
    pub(crate) fn workspace_bytes(&self) -> u64 {
        self.mass
            .as_ref()
            .map_or(0, |array| (array.nbytes() as u64).saturating_mul(2))
            .saturating_add(
                self.values
                    .as_ref()
                    .map_or(0, |array| (array.nbytes() as u64).saturating_mul(2)),
            )
    }

    pub(crate) fn accumulate_mass(
        &mut self,
        scores: &Array,
        maximum: &Array,
        start: i32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            use safemlx::{fast::CustomKernelConfig, Dtype};

            let width = scores.dim(-1);
            let rows = row_count(scores)?;
            let lanes = match &self.mass {
                Some(lanes) => lanes.clone(),
                None => safemlx::ops::zeros::<f32>(&[rows, 4], stream)?,
            };
            let geometry = Array::from_slice(&[rows, width, start], &[3]);
            let threads = rows
                .checked_mul(4)
                .ok_or_else(|| Exception::custom("BF16 attention dispatch size overflows"))?;
            let config = CustomKernelConfig::new()
                .with_grid([threads, 1, 1])
                .with_thread_group([128, 1, 1])
                .with_output_arg(&[rows, 4], Dtype::Float32);
            self.mass = Some(mass_kernel::apply(
                [scores, maximum, &lanes, &geometry],
                &config,
                stream,
            )?);
            Ok(())
        }
        #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
        {
            let _ = (scores, maximum, start, stream);
            Err(unsupported())
        }
    }

    pub(crate) fn accumulate_values(
        &mut self,
        scores: &Array,
        maximum: &Array,
        values: &Array,
        start: i32,
        end: i32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            use eredu_nn::arithmetic::Bf16DotGeometry;
            use safemlx::{fast::CustomKernelConfig, Dtype};

            let mass = self.mass.as_ref().ok_or_else(unsupported)?;
            let rows = row_count(scores)?;
            let probability_geometry = Array::from_slice(&[rows, scores.dim(3)], &[2]);
            let threads = i32::try_from(scores.size())
                .map_err(|_| Exception::custom("BF16 probability dispatch size overflows"))?;
            let probability_config = CustomKernelConfig::new()
                .with_grid([threads, 1, 1])
                .with_thread_group([128, 1, 1])
                .with_output_arg(scores.shape(), Dtype::Bfloat16);
            let probabilities = probability_kernel::apply(
                [scores, maximum, mass, &probability_geometry],
                &probability_config,
                stream,
            )?;
            let features = values.dim(3);
            let queries = scores.dim(2);
            let reduction = Bf16DotGeometry {
                rows: u64::try_from(queries).map_err(|_| unsupported())?,
                width: u64::try_from(end).map_err(|_| unsupported())?,
                columns: u64::try_from(features).map_err(|_| unsupported())?,
            }
            .value_reduction();
            let page = reduction.page(
                u64::try_from(start).map_err(|_| unsupported())?,
                usize::try_from(scores.dim(3)).map_err(|_| unsupported())?,
            );
            let lanes = reduction.lanes() as i32;
            let shape = [scores.dim(0), scores.dim(1), queries, features, lanes];
            let previous = match &self.values {
                Some(values) => values.clone(),
                None => safemlx::ops::zeros::<f32>(&shape, stream)?,
            };
            let geometry = Array::from_slice(
                &[
                    rows,
                    scores.dim(3),
                    page.first_lane as i32,
                    page.interleaved as i32,
                    queries,
                    scores.dim(1) / values.dim(1),
                    features,
                ],
                &[7],
            );
            let config = CustomKernelConfig::new()
                .with_template_arg_int("LANES", lanes)
                .with_grid([features, rows, lanes])
                .with_thread_group([32, 4, 1])
                .with_output_arg(&shape, Dtype::Float32);
            self.values = Some(values_kernel::apply(
                [&probabilities, values, &previous, &geometry],
                &config,
                stream,
            )?);
            Ok(())
        }
        #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
        {
            let _ = (scores, maximum, values, start, end, stream);
            Err(unsupported())
        }
    }

    pub(crate) fn finish(self, stream: &Stream) -> Result<Array, Exception> {
        let values = self
            .values
            .ok_or_else(|| Exception::custom("BF16 attention has no completed value sweep"))?;
        let lane = |index| values.try_index_device((.., .., .., .., index), stream);
        let mut output = lane(0)?;
        for index in 1..values.dim(-1) {
            output = output.add(lane(index)?, stream)?;
        }
        output.as_dtype(safemlx::Dtype::Bfloat16, stream)
    }
}

fn unsupported() -> Exception {
    Exception::custom("ordered BF16 attention requires a Metal scan with established normalization")
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn row_count(scores: &Array) -> Result<i32, Exception> {
    i32::try_from(scores.size() / scores.dim(-1) as usize)
        .map_err(|_| Exception::custom("BF16 attention row count overflows"))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
macro_rules! kernel {
    ($module:ident, $name:literal, $source:literal, [$($input:literal),+], $output:literal) => {
        mod $module {
            use super::*;
            use safemlx::fast::{CustomKernelConfig, MetalKernel};
            use std::cell::RefCell;

            thread_local! {
                static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
            }

            pub(super) fn apply<const N: usize>(
                inputs: [&Array; N],
                config: &CustomKernelConfig,
                stream: &Stream,
            ) -> Result<Array, Exception> {
                KERNEL.with(|cell| {
                    let mut kernel = cell.borrow_mut();
                    if kernel.is_none() {
                        *kernel = Some(MetalKernel::new(
                            $name,
                            [$($input),+],
                            [$output],
                            include_str!($source),
                            crate::backend::nn::arithmetic::metal_arithmetic_header(),
                            true,
                            false,
                        )?);
                    }
                    kernel
                        .as_ref()
                        .expect("initialized attention kernel")
                        .apply_device(inputs, config, stream)?
                        .pop()
                        .ok_or_else(|| Exception::custom("missing ordered attention output"))
                })
            }
        }
    };
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
kernel!(
    mass_kernel,
    "paged_bf16_softmax_mass",
    "paged_bf16_mass.metal",
    ["scores", "maximum", "previous", "geometry"],
    "mass"
);
#[cfg(all(feature = "metal", not(feature = "cuda")))]
kernel!(
    values_kernel,
    "paged_bf16_weighted_values",
    "paged_bf16_values.metal",
    ["probabilities", "values", "previous", "geometry"],
    "output"
);

#[cfg(all(feature = "metal", not(feature = "cuda")))]
kernel!(
    probability_kernel,
    "paged_bf16_probabilities",
    "paged_bf16_probabilities.metal",
    ["scores", "maximum", "mass", "geometry"],
    "probabilities"
);
