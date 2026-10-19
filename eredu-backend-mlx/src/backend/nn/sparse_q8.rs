//! Exact-zero-aware affine Q8 matrix-vector projection over shared transposed storage.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use safemlx::Dtype;
use safemlx::{error::Exception, Array, Stream};

pub(crate) fn project(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType,
        };
        use std::cell::RefCell;
        if stream.get_device()?.get_type()? != DeviceType::Gpu
            || input.dtype() != Dtype::Float32
            || weight.dtype() != Dtype::Uint32
            || scales.dtype() != Dtype::Float32
            || biases.dtype() != Dtype::Float32
            || weight.ndim() != 2
            || scales.ndim() != 2
            || weight.dim(0) <= 0
            || weight.dim(0) > 65536
            || weight.dim(1) <= 0
            || weight.dim(1) > i32::MAX / 4
            || biases.shape() != scales.shape()
            || input.size() != weight.dim(1) as usize * 4
            || input.ndim() == 0
            || input.dim(-1) != weight.dim(1) * 4
            || weight.dim(1) % 16 != 0
            || scales.shape() != [weight.dim(0), weight.dim(1) / 16]
        {
            return Ok(None);
        }
        // Layout metadata is authoritative only after parameter materialization.
        // Parameter edits and host-restored canonical weights use ordinary QMV.
        if !weight.is_available()?
            || !scales.is_available()?
            || !biases.is_available()?
            || weight.strides() != [1, weight.dim(0) as usize]
            || scales.strides() != [1, scales.dim(0) as usize]
            || biases.strides() != scales.strides()
        {
            return Ok(None);
        }
        let input = input.contiguous(false, stream)?;
        // The validated logical views already point at column-major physical
        // storage. Disable automatic contiguity conversion and index it directly.
        let rows = weight.dim(0);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("N", rows)
            .with_template_arg_int("WORDS", weight.dim(1))
            .with_grid([((rows + 31) / 32) * 1024, 1, 1])
            .with_thread_group([1024, 1, 1])
            .with_output_arg([rows], Dtype::Float32);
        thread_local! {static KERNEL: RefCell<Option<MetalKernel>> = const {RefCell::new(None)};}
        let output = KERNEL.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "exact_zero_affine_q8",
                    ["x", "w", "s", "b"],
                    ["out"],
                    r#"
                    uint lane=thread_index_in_simdgroup;
                    uint part=simdgroup_index_in_threadgroup;
                    uint base=threadgroup_position_in_grid.x*32;
                    uint row=base+lane;
                    threadgroup float partials[1024];
                    float result=0;
                    if(row<N) {
                        for(uint word=part;word<WORDS;word+=32) {
                            float4 v=float4(x[word*4],x[word*4+1],x[word*4+2],x[word*4+3]);
                            float scale=s[(word/16)*N+row];
                            float bias=b[(word/16)*N+row];
                            if(all(v==float4(0))) {
                                // Finite quantization companions permit an exact skip.
                                // Preserve zero times nonfinite rather than hiding NaNs.
                                if(!isfinite(scale) || !isfinite(bias)) result=NAN;
                                continue;
                            }
                            float sum=0;sum+=v.x;sum+=v.y;sum+=v.z;sum+=v.w;
                            uint q=w[word*N+row];
                            float accum=0;
                            accum+=v.x*float(q&255);
                            accum+=v.y*float((q>>8)&255);
                            accum+=v.z*float((q>>16)&255);
                            accum+=v.w*float(q>>24);
                            result+=scale*accum+sum*bias;
                        }
                    }
                    partials[part*32+lane]=result;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float value=partials[lane*32+part];
                    float total=simd_sum(value);
                    if(lane==0 && base+part<N) out[base+part]=total;
                "#,
                    "",
                    false,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("sparse Q8 kernel initialized")
                .apply_one_device([&input, weight, scales, biases], &config, stream)
        })?;
        let mut shape = input.shape().to_vec();
        *shape.last_mut().expect("non-scalar input") = rows;
        Ok(Some(output.reshape(&shape, stream)?))
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, scales, biases, stream);
        Ok(None)
    }
}
