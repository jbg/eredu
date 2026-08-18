#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use safemlx::{
        module::Module, nn::QuantizedLinearBuilder, ops::QuantizationMode, Array, Device,
        DeviceType, ExecutionContext,
    };

    if !safemlx::cuda::is_available()? {
        return Err("MLX was built with CUDA, but no CUDA device is available".into());
    }

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let mut projection = QuantizedLinearBuilder::new(64, 8)
        .group_size(32)
        .bits(4)
        .mode(QuantizationMode::MxFp4)
        .bias(false)
        .build(gpu.stream())?;
    let input = Array::ones::<f32>(&[2, 64], gpu.stream())?;
    let output = projection.forward(&input, gpu.stream())?.into_evaluated()?;
    let values: &[f32] = output.as_slice();
    if values.len() != 16 || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("invalid MXFP4 projection output: {values:?}").into());
    }

    println!("CUDA MXFP4 smoke test passed: shape=[2, 8]");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("re-run with `--no-default-features --features cuda`");
    std::process::exit(2);
}
