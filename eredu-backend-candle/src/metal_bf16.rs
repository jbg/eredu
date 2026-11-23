//! Native Metal execution for the BF16 reduction profiles.

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use eredu_nn::arithmetic::bf16_row_block_ends;
use objc2_metal::MTLSize;

use crate::metal::{size, Launch};

pub(crate) fn product(
    lhs: &Tensor,
    rhs: &Tensor,
    transpose_rhs: bool,
    sequential: bool,
    interleaved: usize,
    shape: Shape,
) -> Result<Tensor> {
    lhs.contiguous()?.apply_op2_no_bwd(
        &rhs.contiguous()?,
        &Product {
            shape,
            transpose_rhs,
            sequential,
            interleaved,
        },
    )
}

struct Product {
    shape: Shape,
    transpose_rhs: bool,
    sequential: bool,
    interleaved: usize,
}

impl CustomOp2 for Product {
    fn name(&self) -> &'static str {
        "ordered BF16 matrix product"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ordered Metal product requires Metal storage")
    }

    fn metal_fwd(
        &self,
        lhs: &MetalStorage,
        left: &Layout,
        rhs: &MetalStorage,
        right: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let [.., rows, input_width] = left.dims() else {
            candle_core::bail!("BF16 product requires a matrix");
        };
        let columns = self.shape.dims()[self.shape.rank() - 1];
        let count = self.shape.elem_count();
        let threads_per_column = if self.sequential || !self.transpose_rhs {
            1
        } else {
            32
        };
        let width = columns
            .checked_mul(threads_per_column)
            .ok_or_else(|| candle_core::Error::Msg("Metal BF16 product grid overflows".into()))?;
        let grid = MTLSize {
            width,
            height: count / columns,
            depth: 1,
        };
        let [row_end, tail_end] = bf16_row_block_ends(*input_width);
        let args = [
            u32::try_from(*rows)?,
            u32::try_from(*input_width)?,
            u32::try_from(columns)?,
            u32::from(self.sequential),
            u32::from(self.transpose_rhs),
            u32::try_from(self.interleaved)?,
            u32::try_from(row_end)?,
            u32::try_from(tail_end)?,
        ];
        Launch {
            kernel: "bf16_product",
            shape: &self.shape,
            dtype: DType::BF16,
            grid,
            group: 128,
        }
        .run(&[(lhs, left), (rhs, right)], args)
    }
}

pub(crate) fn softmax(scores: &Tensor) -> Result<Tensor> {
    let width = scores.dims().last().copied().unwrap_or(0);
    if width == 0 {
        candle_core::bail!("BF16 softmax requires a nonempty score axis");
    }
    unary(
        scores,
        "bf16_softmax",
        scores.shape().clone(),
        DType::BF16,
        scores.elem_count() / width,
        [u32::try_from(width)?],
    )
}

pub(crate) fn rms_inverse(input: &Tensor, epsilon: f32, cascaded: bool) -> Result<Tensor> {
    let mut shape = input.dims().to_vec();
    let width = shape
        .last_mut()
        .ok_or_else(|| candle_core::Error::Msg("RMS requires a feature axis".into()))?;
    if *width == 0 {
        candle_core::bail!("RMS requires a nonempty feature axis");
    }
    let rows = input.elem_count() / *width;
    let grid = rows
        .checked_mul(32)
        .ok_or_else(|| candle_core::Error::Msg("Metal RMS grid overflows".into()))?;
    let features = u32::try_from(*width)?;
    *width = 1;
    unary(
        input,
        if cascaded {
            "cascaded_rms_inverse"
        } else {
            "ordered_rms_inverse"
        },
        shape.into(),
        DType::F32,
        grid,
        [features, epsilon.to_bits()],
    )
}

pub(crate) fn silu(input: &Tensor) -> Result<Tensor> {
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    unary(
        input,
        "bf16_silu",
        input.shape().clone(),
        DType::BF16,
        input.elem_count(),
        [],
    )
}

fn unary<const N: usize>(
    input: &Tensor,
    name: &'static str,
    shape: Shape,
    dtype: DType,
    grid: usize,
    args: [u32; N],
) -> Result<Tensor> {
    input.contiguous()?.apply_op1_no_bwd(&Unary {
        name,
        shape,
        dtype,
        grid,
        args,
    })
}

struct Unary<const N: usize> {
    name: &'static str,
    shape: Shape,
    dtype: DType,
    grid: usize,
    args: [u32; N],
}

impl<const N: usize> CustomOp1 for Unary<N> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ordered Metal operation requires Metal storage")
    }

    fn metal_fwd(&self, input: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        Launch {
            kernel: self.name,
            shape: &self.shape,
            dtype: self.dtype,
            grid: size(self.grid),
            group: 64,
        }
        .run(&[(input, layout)], self.args)
    }
}

pub(crate) fn page_pass<const N: usize>(
    name: &'static str,
    inputs: [&Tensor; 3],
    shape: Shape,
    dtype: DType,
    args: [u32; N],
) -> Result<Tensor> {
    let [a, b, c] = inputs;
    a.contiguous()?.apply_op3_no_bwd(
        &b.contiguous()?,
        &c.contiguous()?,
        &PagePass {
            name,
            shape,
            dtype,
            args,
        },
    )
}

struct PagePass<const N: usize> {
    name: &'static str,
    shape: Shape,
    dtype: DType,
    args: [u32; N],
}

impl<const N: usize> CustomOp3 for PagePass<N> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ordered Metal attention pass requires Metal storage")
    }

    fn metal_fwd(
        &self,
        a: &MetalStorage,
        la: &Layout,
        b: &MetalStorage,
        lb: &Layout,
        c: &MetalStorage,
        lc: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Launch {
            kernel: self.name,
            shape: &self.shape,
            dtype: self.dtype,
            grid: size(self.shape.elem_count()),
            group: 64,
        }
        .run(&[(a, la), (b, lb), (c, lc)], self.args)
    }
}
