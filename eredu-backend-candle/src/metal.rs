//! Shared shader compilation and dispatch for Candle Metal operators.

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp1, CustomOp2, DType, Layout, MetalDevice,
    MetalStorage, Result, Shape, Tensor,
};
use candle_metal_kernels::metal::{ComputePipeline, Library};
use objc2_metal::{MTLCompileOptions, MTLSize};
use std::{cell::RefCell, collections::HashMap};

struct Kernels {
    library: Library,
    pipelines: HashMap<&'static str, ComputePipeline>,
}

thread_local! {
    static KERNELS: RefCell<Option<(candle_core::metal_backend::DeviceId, Kernels)>> =
        const { RefCell::new(None) };
}

fn pipeline(device: &MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    KERNELS.with(|entry| {
        let mut entry = entry.borrow_mut();
        if entry.as_ref().is_none_or(|(id, _)| *id != device.id()) {
            let options = MTLCompileOptions::new();
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let library = device
                .device()
                .new_library_with_source(
                    &[
                        "#include <metal_stdlib>\nusing namespace metal;\n",
                        eredu_nn::arithmetic::STABLE_EXP_SOURCE,
                        include_str!("metal_bf16.metal"),
                        include_str!("metal_quantized.metal"),
                        include_str!("metal_f32.metal"),
                    ]
                    .join("\n"),
                    Some(&options),
                )
                .map_err(candle_core::Error::wrap)?;
            *entry = Some((
                device.id(),
                Kernels {
                    library,
                    pipelines: HashMap::new(),
                },
            ));
        }
        let (_, kernels) = entry.as_mut().unwrap();
        if let Some(pipeline) = kernels.pipelines.get(name) {
            return Ok(pipeline.clone());
        }
        let function = kernels
            .library
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        let pipeline = device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        kernels.pipelines.insert(name, pipeline.clone());
        Ok(pipeline)
    })
}

pub(crate) fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Launch<'a> {
    pub(crate) kernel: &'static str,
    pub(crate) shape: &'a Shape,
    pub(crate) dtype: DType,
    pub(crate) grid: MTLSize,
    pub(crate) group: usize,
}

impl Launch<'_> {
    pub(crate) fn unary<const N: usize>(self, input: &Tensor, args: [u32; N]) -> Result<Tensor> {
        input
            .contiguous()?
            .apply_op1_no_bwd(&Operator { launch: self, args })
    }

    pub(crate) fn unary_strided<const N: usize>(
        self,
        input: &Tensor,
        args: [u32; N],
    ) -> Result<Tensor> {
        input.apply_op1_no_bwd(&Operator { launch: self, args })
    }

    pub(crate) fn binary<const N: usize>(
        self,
        lhs: &Tensor,
        rhs: &Tensor,
        args: [u32; N],
    ) -> Result<Tensor> {
        lhs.contiguous()?
            .apply_op2_no_bwd(&rhs.contiguous()?, &Operator { launch: self, args })
    }

    pub(crate) fn run<const N: usize>(
        self,
        inputs: &[(&MetalStorage, &Layout)],
        args: [u32; N],
    ) -> Result<(MetalStorage, Shape)> {
        let _ = u32::try_from(self.grid.width)?;
        let _ = u32::try_from(self.grid.height)?;
        let device = inputs[0].0.device();
        let count = self.shape.elem_count();
        let pipeline = pipeline(device, self.kernel)?;
        let output = device
            .new_buffer_builder()
            .with_size_for(count, self.dtype)
            .build()?;
        let command = device.command_encoder()?;
        let encoder = command.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        for (slot, (input, layout)) in inputs.iter().enumerate() {
            let offset = layout.start_offset() * input.dtype().size_in_bytes();
            encoder.set_input_buffer(slot, Some(input.buffer()), offset);
        }
        encoder.set_output_buffer(inputs.len(), Some(&output), 0);
        if N != 0 {
            encoder.set_bytes(inputs.len() + 1, &args);
        }
        encoder.dispatch_threads(self.grid, size(self.group));
        Ok((
            MetalStorage::new(output, device.clone(), count, self.dtype),
            self.shape.clone(),
        ))
    }
}

struct Operator<'a, const N: usize> {
    launch: Launch<'a>,
    args: [u32; N],
}

impl<const N: usize> CustomOp1 for Operator<'_, N> {
    fn name(&self) -> &'static str {
        self.launch.kernel
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{} requires Metal storage", self.launch.kernel)
    }
    fn metal_fwd(&self, input: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        self.launch.run(&[(input, layout)], self.args)
    }
}

impl<const N: usize> CustomOp2 for Operator<'_, N> {
    fn name(&self) -> &'static str {
        self.launch.kernel
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{} requires Metal storage", self.launch.kernel)
    }
    fn metal_fwd(
        &self,
        lhs: &MetalStorage,
        left: &Layout,
        rhs: &MetalStorage,
        right: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        self.launch.run(&[(lhs, left), (rhs, right)], self.args)
    }
}
