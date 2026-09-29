//! Isolate native recurrent mechanism allocations, excluding checkpoint weights.
//! Usage: recurrent_workspace_probe TOKENS HEADS KEY_WIDTH VALUE_WIDTH KERNEL
use safemlx::{memory, transforms::eval, Array, Device, DeviceType, Stream};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<i32> = std::env::args()
        .skip(1)
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let [tokens, heads, key, value, kernel] = args.as_slice() else {
        panic!("TOKENS HEADS KEY_WIDTH VALUE_WIDTH KERNEL")
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let make = |shape: &[i32], value| {
        Array::from_slice(&vec![value; shape.iter().product::<i32>() as usize], shape)
    };
    let q = make(&[1, *tokens, *heads, *key], 0.01_f32);
    let v = make(&[1, *tokens, *heads, *value], 0.02_f32);
    let g = make(&[1, *tokens, *heads], -0.1_f32);
    let b = make(&[1, *tokens, *heads], 0.5_f32);
    eval([&q, &v, &g, &b])?;
    memory::reset_peak_memory()?;
    let baseline = memory::active_memory()?;
    let (state, output) = eredu_backend_mlx::backend::nn::gated_delta::gated_delta_scan(
        &q, &q, &v, &g, &b, None, &stream,
    )?;
    eval([&state, &output])?;
    println!(
        "scan baseline={baseline} peak={} output={} state={} dtype={:?}",
        memory::peak_memory()?,
        output.nbytes(),
        state.nbytes(),
        state.dtype()
    );
    drop((q, v, g, b, state, output));
    let channels = heads * (2 * key + value);
    let input = make(&[1, tokens + kernel - 1, channels], 0.01_f32);
    let weight = make(&[channels, *kernel, 1], 0.02_f32);
    eval([&input, &weight])?;
    memory::reset_peak_memory()?;
    let baseline = memory::active_memory()?;
    let output = safemlx::ops::conv1d(&input, &weight, 1, 0, 1, channels, &stream)?;
    eval([&output])?;
    println!(
        "convolution baseline={baseline} peak={} output={} dtype={:?}",
        memory::peak_memory()?,
        output.nbytes(),
        output.dtype()
    );
    Ok(())
}
