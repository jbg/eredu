//! Packed grouped operations compose the shared native CPU leaf census.
use super::*;
use safemlx::CpuUnaryOperation;
impl Program {
    pub(super) fn selection(&mut self, tokens: usize, routes: usize, index: Dtype) -> Option<()> {
        let count = tokens.checked_mul(routes)?;
        self.native.maximum_captures = self.native.maximum_captures.max(3); // argsort output + two weak captures
        self.reshape(2, 1)?;
        self.cast(index, Dtype::Int32, 1, count)?;
        self.copy(
            OperationEvent::cpu_argsort_layout(Dtype::Int32, 1, count, 1, false)?,
            1,
            count,
            Dtype::Uint32,
        )?;
        self.reshape(1, 1)?; // take() flattens its source
        self.gather(Dtype::Int32, Dtype::Uint32, 1, count, count, 1)?;
        self.cast(Dtype::Uint32, Dtype::Int32, 1, count)?;
        // Integer floor_divide lowers to the ordinary Divide primitive.
        self.scalar_binary(CpuBinaryOperation::Divide, Dtype::Int32, 1, count)
    }
    pub(super) fn input_rows(&mut self, tokens: usize, routes: usize, width: usize) -> Option<()> {
        self.gather(
            Dtype::Float32,
            Dtype::Int32,
            2,
            tokens.checked_mul(width)?,
            tokens.checked_mul(routes)?,
            width,
        )
    }
    pub(super) fn projection(&mut self, rows: usize, groups: usize, p: Projection) -> Option<()> {
        let geometry = self
            .mechanism
            .matmul
            .selected()
            .geometry(
                3,
                1,
                u32::try_from(p.output).ok()?,
                u32::try_from(p.input).ok()?,
                u32::try_from(rows).ok()?,
            )
            .ok()?;
        let _ = geometry;
        // The supplied transposed bank and the BF16-only probe's inverse view
        // are both ordinary aliases. CPU F32 exits that probe before kernels.
        for _ in 0..2 {
            self.copy(
                OperationEvent::cpu_transpose_alias_layout(3, false)?,
                1,
                0,
                Dtype::Float32,
            )?;
        }
        self.controls(crate::backend::nn::matrix::row_projection_probe_control_bytes()?)?;
        self.controls(safemlx::Stream::device_type_control_bytes()?)?;
        self.shells = self.shells.checked_add(2)?;
        self.reshape(2, 3)?;
        self.cast(
            Dtype::Float32,
            Dtype::Float32,
            3,
            rows.checked_mul(p.input)?,
        )?;
        self.cast(
            Dtype::Float32,
            Dtype::Float32,
            3,
            groups.checked_mul(p.input)?.checked_mul(p.output)?,
        )?;
        self.copy(
            OperationEvent::cpu_arange_int_layout(Dtype::Uint32, rows, false)?,
            0,
            rows,
            Dtype::Uint32,
        )?;
        self.reshape(1, 1)?;
        self.cast(Dtype::Uint32, Dtype::Uint32, 1, rows)?;
        self.cast(Dtype::Int32, Dtype::Uint32, 1, rows)?;
        self.broadcast(1, 1)?;
        self.broadcast(1, 1)?;
        self.copy(
            OperationEvent::cpu_tiled_gather_mm_layout(3, 3, 1, 1, p.output, p.input, rows, false)?,
            4,
            rows.checked_mul(p.output)?,
            Dtype::Float32,
        )?;
        self.native.maximum_captures = self.native.maximum_captures.max(1);
        self.reshape(3, 2)?;
        if p.bias {
            self.gather(
                Dtype::Float32,
                Dtype::Int32,
                2,
                groups.checked_mul(p.output)?,
                rows,
                p.output,
            )?;
            let elements = rows.checked_mul(p.output)?;
            self.binary(
                CpuBinaryOperation::Add,
                Dtype::Float32,
                2,
                elements,
                2,
                2,
                elements,
                elements,
            )?;
        }
        self.controls(size_of::<(
            &mut Self,
            usize,
            usize,
            Projection,
            eredu_nn::CpuMatmulGeometry,
            Result<eredu_nn::CpuMatmulGeometry, eredu_nn::CpuMatmulError>,
            Option<()>,
        )>())
    }
    fn silu(&mut self, elements: usize) -> Option<()> {
        self.controls(crate::backend::nn::arithmetic::cpu_pointwise_probe_control_bytes()?)?;
        self.unary(CpuUnaryOperation::Negative, Dtype::Float32, 2, elements)?;
        self.unary(CpuUnaryOperation::Exponential, Dtype::Float32, 2, elements)?;
        self.scalar_binary(CpuBinaryOperation::Add, Dtype::Float32, 2, elements)?;
        self.binary(
            CpuBinaryOperation::Divide,
            Dtype::Float32,
            2,
            elements,
            2,
            2,
            elements,
            elements,
        )?;
        self.cast(Dtype::Float32, Dtype::Float32, 2, elements)
    }
    pub(super) fn activation(&mut self, d: Descriptor, rows: usize) -> Option<()> {
        let elements = rows.checked_mul(d.units)?;
        match d.activation {
            Activation::Linear(eredu_nn::GroupedLinearActivation::Identity) => {}
            Activation::Linear(eredu_nn::GroupedLinearActivation::Silu) => self.silu(elements)?,
            Activation::Relu2 => {
                self.scalar()?;
                self.binary(
                    CpuBinaryOperation::Maximum,
                    Dtype::Float32,
                    2,
                    elements,
                    2,
                    0,
                    elements,
                    1,
                )?;
                self.unary(CpuUnaryOperation::Square, Dtype::Float32, 2, elements)?;
            }
            Activation::Gated(policy) => {
                self.slice(2)?;
                self.slice(2)?;
                if policy.gate_upper_bound().is_some() {
                    self.scalar_binary(CpuBinaryOperation::Minimum, Dtype::Float32, 2, elements)?;
                }
                if policy.up_absolute_bound().is_some() {
                    self.scalar_binary(CpuBinaryOperation::Maximum, Dtype::Float32, 2, elements)?;
                    self.scalar_binary(CpuBinaryOperation::Minimum, Dtype::Float32, 2, elements)?;
                }
                if policy.up_offset() != 0.0 {
                    self.scalar_binary(CpuBinaryOperation::Add, Dtype::Float32, 2, elements)?;
                }
                match policy.activation() {
                    eredu_nn::GatedProductActivation::Silu
                        if policy.sigmoid_multiplier() == 1.0 =>
                    {
                        self.silu(elements)?
                    }
                    eredu_nn::GatedProductActivation::Silu => {
                        self.scalar_binary(
                            CpuBinaryOperation::Multiply,
                            Dtype::Float32,
                            2,
                            elements,
                        )?;
                        self.controls(
                            crate::backend::nn::arithmetic::cpu_pointwise_probe_control_bytes()?,
                        )?;
                        self.cast(Dtype::Float32, Dtype::Float32, 2, elements)?; // native sigmoid frontend
                        self.unary(CpuUnaryOperation::Sigmoid, Dtype::Float32, 2, elements)?;
                        self.cast(Dtype::Float32, Dtype::Float32, 2, elements)?;
                        self.binary(
                            CpuBinaryOperation::Multiply,
                            Dtype::Float32,
                            2,
                            elements,
                            2,
                            2,
                            elements,
                            elements,
                        )?;
                    }
                    eredu_nn::GatedProductActivation::GeluApproximate => {
                        // layers::gelu_approximate, preserving its scalar sources
                        // and polynomial order rather than substituting exact GELU.
                        self.scalar()?; // F32 half
                        self.binary(
                            CpuBinaryOperation::Multiply,
                            Dtype::Float32,
                            2,
                            elements,
                            0,
                            2,
                            1,
                            elements,
                        )?;
                        self.scalar()?; // I32 exponent three
                        self.cast(Dtype::Int32, Dtype::Float32, 0, 1)?;
                        self.binary(
                            CpuBinaryOperation::Power,
                            Dtype::Float32,
                            2,
                            elements,
                            2,
                            0,
                            elements,
                            1,
                        )?;
                        self.scalar_binary(
                            CpuBinaryOperation::Multiply,
                            Dtype::Float32,
                            2,
                            elements,
                        )?;
                        self.binary(
                            CpuBinaryOperation::Add,
                            Dtype::Float32,
                            2,
                            elements,
                            2,
                            2,
                            elements,
                            elements,
                        )?;
                        self.scalar()?; // F32 2/pi, then actual scalar sqrt
                        self.unary(CpuUnaryOperation::Sqrt, Dtype::Float32, 0, 1)?;
                        self.binary(
                            CpuBinaryOperation::Multiply,
                            Dtype::Float32,
                            2,
                            elements,
                            0,
                            2,
                            1,
                            elements,
                        )?;
                        self.unary(CpuUnaryOperation::Tanh, Dtype::Float32, 2, elements)?;
                        self.scalar_binary(CpuBinaryOperation::Add, Dtype::Float32, 2, elements)?;
                        self.binary(
                            CpuBinaryOperation::Multiply,
                            Dtype::Float32,
                            2,
                            elements,
                            2,
                            2,
                            elements,
                            elements,
                        )?;
                    }
                    _ => return None,
                }
                self.binary(
                    CpuBinaryOperation::Multiply,
                    Dtype::Float32,
                    2,
                    elements,
                    2,
                    2,
                    elements,
                    elements,
                )?;
            }
        }
        self.controls(size_of::<(&mut Self, Descriptor, usize, usize, Option<()>)>())
    }
    pub(super) fn weighted(
        &mut self,
        tokens: usize,
        routes: usize,
        width: usize,
        reduction: eredu_nn::GroupReduction,
    ) -> Option<()> {
        let count = tokens.checked_mul(routes)?;
        let elements = count.checked_mul(width)?;
        self.reshape(2, 1)?;
        self.reshape(1, 1)?; // Array::take flattens the explicit coefficient view
        self.gather(Dtype::Float32, Dtype::Int32, 1, count, count, 1)?;
        self.reshape(1, 2)?;
        self.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            2,
            elements,
            2,
            2,
            elements,
            count,
        )?;
        self.full(Dtype::Float32, 2, elements)?;
        self.reshape(2, 3)?;
        self.scatter(Dtype::Float32, 2, count, elements)?;
        self.reshape(2, 3)?;
        if reduction != eredu_nn::GroupReduction::Sum {
            self.full(Dtype::Int32, 1, count)?;
            self.reshape(1, 2)?;
            self.scatter(Dtype::Int32, 1, count, count)?;
            self.reshape(1, 2)?;
            self.copy(
                OperationEvent::cpu_argsort_layout(Dtype::Int32, 2, routes, tokens, false)?,
                1,
                count,
                Dtype::Uint32,
            )?;
            self.reshape(2, 3)?;
            self.broadcast(3, 3)?;
            // take_along_axis broadcasts both sources ignoring the selected
            // axis; its actual broadcast U32 index remains readable by GatherAxis.
            self.broadcast(3, 3)?;
            self.broadcast(3, 3)?;
            self.native.maximum_captures = self.native.maximum_captures.max(4);
            self.copy(
                OperationEvent::cpu_gather_axis_row_layout(3, elements, false)?,
                2,
                elements,
                Dtype::Float32,
            )?;
            let output = tokens.checked_mul(width)?;
            self.full(Dtype::Float32, 2, output)?;
            for _ in 0..routes {
                self.slice(3)?;
                self.copy(
                    OperationEvent::cpu_squeeze_layout(3, false)?,
                    1,
                    0,
                    Dtype::Float32,
                )?;
                self.binary(
                    CpuBinaryOperation::Add,
                    Dtype::Float32,
                    2,
                    output,
                    2,
                    2,
                    output,
                    output,
                )?;
                self.cast(Dtype::Float32, Dtype::Float32, 2, output)?;
            }
            return Some(());
        }
        if routes > 1 {
            self.native.maximum_captures = self.native.maximum_captures.max(3);
            self.copy(
                OperationEvent::cpu_strided_sum_layout(3, 1, routes, tokens, width, false)?,
                1,
                tokens.checked_mul(width)?,
                Dtype::Float32,
            )?;
        } else {
            self.cast(Dtype::Float32, Dtype::Float32, 3, elements)?;
        }
        self.copy(
            OperationEvent::cpu_squeeze_layout(3, false)?,
            1,
            0,
            Dtype::Float32,
        )
    }
    fn scatter(
        &mut self,
        dtype: Dtype,
        rank: usize,
        selections: usize,
        elements: usize,
    ) -> Option<()> {
        // Non-donating copy output + two copy captures + index/update/output.
        self.native.maximum_captures = self.native.maximum_captures.max(6);
        self.broadcast(1, 1)?;
        self.cast(Dtype::Int32, Dtype::Int32, 1, selections)?;
        self.cast(dtype, dtype, rank + 1, elements)?;
        self.copy(
            OperationEvent::cpu_scatter_layout(
                dtype,
                Dtype::Int32,
                rank,
                elements,
                elements,
                false,
            )?,
            3,
            elements,
            dtype,
        )
    }
    pub(super) fn bias_tail(&mut self, d: Descriptor) -> Option<()> {
        self.selection(d.tokens, d.routes, d.index)?;
        self.gather(
            Dtype::Float32,
            Dtype::Int32,
            2,
            d.groups.checked_mul(d.output)?,
            d.tokens.checked_mul(d.routes)?,
            d.output,
        )?;
        self.weighted(d.tokens, d.routes, d.output, d.reduction)?;
        let count = d.tokens.checked_mul(d.output)?;
        self.binary(
            CpuBinaryOperation::Subtract,
            Dtype::Float32,
            2,
            count,
            2,
            2,
            count,
            count,
        )
    }
    pub(super) fn concatenate(&mut self, chunks: usize, elements: usize) -> Option<()> {
        if chunks <= 1 {
            return Some(());
        }
        let source = OperationEvent::cpu_concatenate_many_layout(
            Dtype::Float32,
            2,
            chunks,
            elements,
            false,
        )?;
        self.bytes = self.bytes.checked_add(
            self.capacity(elements, Dtype::Float32)?
                .checked_mul(source.backing_births() as u64)?,
        )?;
        self.native.concatenate(source, chunks)?;
        self.controls(safemlx::ops::concatenate_axis_control_bytes()?)
    }
}
