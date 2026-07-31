//! Isolated native-Q4_K versus converted-affine Metal benchmark.
//!
//! The Q4_K block generator is deterministic and intentionally synthetic so
//! this benchmark can run without a model checkpoint. Shapes default to the
//! Gemma 4 projection dimensions used by the 26B-A4B checkpoint.

use std::time::{Duration, Instant};

use half::f16;
use safemlx::{
    memory,
    native_quantization::NativeQuantizedTensor,
    ops::{quantized_matmul_with_mode, QuantizationMode},
    transforms::eval,
    Array, Device, DeviceType, Dtype, Stream,
};
use safemlx_gguf::{ConvertedTensor, GgmlType, Reader, TensorInput, Writer};

fn q4k_block(seed: usize) -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&f16::from_f32(0.015625).to_bits().to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(0.0078125).to_bits().to_le_bytes());
    for (index, value) in block[4..].iter_mut().enumerate() {
        *value = (seed as u8)
            .wrapping_mul(31)
            .wrapping_add((index as u8).wrapping_mul(17))
            .wrapping_add(7);
    }
    block
}

fn q4k_matrix(rows: i32, columns: i32) -> Vec<u8> {
    let blocks = (rows * columns / 256) as usize;
    let mut raw = Vec::with_capacity(blocks * 144);
    for block in 0..blocks {
        raw.extend(q4k_block(block));
    }
    raw
}

fn converted_affine(
    raw: &[u8],
    rows: i32,
    columns: i32,
    stream: &Stream,
) -> (Array, Array, Array, usize) {
    let mut file = Vec::new();
    Writer::default()
        .write(
            std::io::Cursor::new(&mut file),
            &std::collections::BTreeMap::new(),
            &[TensorInput {
                name: "weight",
                dimensions: &[columns as u64, rows as u64],
                ggml_type: GgmlType::Q4K,
                data: raw,
            }],
        )
        .unwrap();
    let mut reader = Reader::new(std::io::Cursor::new(file)).unwrap();
    let descriptor = reader.tensors()[0].clone();
    let ConvertedTensor::Affine(affine) = reader.read_tensor(&descriptor).unwrap() else {
        panic!("Q4_K conversion did not return affine storage")
    };
    let packed = affine.weights.len() * 4 + affine.scales.len() * 2 + affine.biases.len() * 2;
    let weights = Array::from_slice(
        &affine.weights,
        &[
            rows,
            i32::try_from(affine.weight_shape[1]).expect("packed dimension"),
        ],
    )
    .copy(stream)
    .unwrap();
    let scales = Array::from_slice(
        &affine
            .scales
            .iter()
            .copied()
            .map(f16::from_bits)
            .collect::<Vec<_>>(),
        &[rows, columns / 32],
    )
    .copy(stream)
    .unwrap();
    let biases = Array::from_slice(
        &affine
            .biases
            .iter()
            .copied()
            .map(f16::from_bits)
            .collect::<Vec<_>>(),
        &[rows, columns / 32],
    )
    .copy(stream)
    .unwrap();
    eval([&weights, &scales, &biases]).unwrap();
    (weights, scales, biases, packed)
}

fn time_average(
    iterations: usize,
    stream: &Stream,
    mut operation: impl FnMut() -> Array,
) -> Duration {
    for _ in 0..3 {
        let output = operation();
        eval([&output]).unwrap();
    }
    stream.synchronize().unwrap();
    let started = Instant::now();
    for _ in 0..iterations {
        let output = operation();
        eval([&output]).unwrap();
    }
    stream.synchronize().unwrap();
    started.elapsed() / iterations as u32
}

fn benchmark_shape(rows: i32, columns: i32, batch: i32, iterations: usize, stream: &Stream) {
    let raw = q4k_matrix(rows, columns);
    let native = NativeQuantizedTensor::from_q4k_bytes(&raw, &[rows, columns], stream).unwrap();
    let (affine_weight, affine_scales, affine_biases, affine_bytes) =
        converted_affine(&raw, rows, columns, stream);
    let input = Array::from_slice(
        &(0..batch * columns)
            .map(|index| (index as f32 % 41.0 - 20.0) / 21.0)
            .collect::<Vec<_>>(),
        &[batch, columns],
    )
    .copy(stream)
    .unwrap();

    let native_output = native.linear(&input, true, stream).unwrap();
    let affine_output = quantized_matmul_with_mode(
        &input,
        &affine_weight,
        &affine_scales,
        Some(&affine_biases),
        true,
        32,
        4,
        QuantizationMode::Affine,
        stream,
    )
    .unwrap();
    eval([&native_output, &affine_output]).unwrap();
    let native_values = native_output.evaluated().unwrap();
    let affine_float = affine_output.as_dtype(Dtype::Float32, stream).unwrap();
    let affine_values = affine_float.evaluated().unwrap();
    let (max_abs, rms) = native_values
        .as_slice::<f32>()
        .iter()
        .zip(affine_values.as_slice::<f32>())
        .fold((0.0f32, 0.0f64), |(max_abs, sum_sq), (native, affine)| {
            let error = (native - affine).abs();
            (max_abs.max(error), sum_sq + f64::from(error * error))
        });
    let rms = (rms / native_output.size() as f64).sqrt();

    let native_time = time_average(iterations, stream, || {
        native.linear(&input, true, stream).unwrap()
    });
    let affine_time = time_average(iterations, stream, || {
        quantized_matmul_with_mode(
            &input,
            &affine_weight,
            &affine_scales,
            Some(&affine_biases),
            true,
            32,
            4,
            QuantizationMode::Affine,
            stream,
        )
        .unwrap()
    });
    let native_gbps = raw.len() as f64 / native_time.as_secs_f64() / 1e9;
    let affine_gbps = affine_bytes as f64 / affine_time.as_secs_f64() / 1e9;
    println!(
        "{rows:5}x{columns:5} batch={batch:3} native={:8.3} ms {:6.1} GB/s affine={:8.3} ms {:6.1} GB/s speedup={:5.2}x raw={:8} affine={:8} max_abs={max_abs:.6} rms={rms:.6}",
        native_time.as_secs_f64() * 1e3,
        native_gbps,
        affine_time.as_secs_f64() * 1e3,
        affine_gbps,
        affine_time.as_secs_f64() / native_time.as_secs_f64(),
        raw.len(),
        affine_bytes,
    );
}

fn main() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    memory::clear_cache().unwrap();
    memory::reset_peak_memory().unwrap();
    println!("shape             native latency/bandwidth       affine latency/bandwidth");
    for (rows, columns, batch, iterations) in [
        (2816, 2816, 1, 40),
        (2112, 2816, 1, 40),
        (704, 2816, 1, 60),
        (2816, 2816, 32, 10),
        (2112, 2816, 32, 10),
        (704, 2816, 32, 15),
    ] {
        benchmark_shape(rows, columns, batch, iterations, &stream);
    }
    stream.synchronize().unwrap();
    println!(
        "mlx_active_bytes={} mlx_peak_bytes={}",
        memory::active_memory().unwrap(),
        memory::peak_memory().unwrap()
    );
}
