//! Independent Q4_K projection validation on the released LFM3 GGUF.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use anyhow::{ensure, Context};
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use eredu_checkpoint::LinearFormat;
use eredu_gguf::{BlockDecoder, Endian, GgmlType, Reader};
use eredu_nn::{
    publish_parameter_replacements, LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend,
    ParameterSpec,
};
use half::f16;
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use sha2::{Digest, Sha256};

const SHA256: &str = "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4";
const WEIGHT: &str = "blk.0.shortconv.in_proj.weight";
const INPUTS: usize = 2048;
const OUTPUTS: usize = 6144;
const BLOCK_VALUES: usize = 256;
const BLOCK_BYTES: usize = 144;
const ROW_BYTES: usize = INPUTS / BLOCK_VALUES * BLOCK_BYTES;
const PROMPT_ROWS: usize = 17;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .context("usage: lfm3_q4k_reference CHECKPOINT.gguf")?;
    ensure!(args.next().is_none(), "expected one checkpoint path");
    let encoded = read_weights(Path::new(&path))?;
    let reference: Vec<f64> = encoded.chunks_exact(BLOCK_BYTES).flat_map(decode).collect();
    ensure!(
        reference.len() == OUTPUTS * INPUTS,
        "unexpected weight count"
    );

    let decoder = BlockDecoder::new(GgmlType::Q4K, Endian::Little)?;
    let mut decoded = vec![0.0; reference.len()];
    decoder.decode_into(&encoded, &mut decoded)?;
    ensure!(
        decoded.iter().zip(&reference).all(|(&a, &b)| a == b as f32),
        "portable GGUF decoding differs from the independently decoded weights"
    );
    println!(
        "portable FP32 decode: {} weights match exactly",
        reference.len()
    );

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mut linear = MlxNeuralBackend::linear(
        LinearSpec {
            input: INPUTS as i32,
            output: OUTPUTS as i32,
            weight: ParameterSpec::trainable(WEIGHT)?,
            bias: None,
            format: LinearFormatSpec::unscaled(LinearFormat::GgufIQuant {
                ggml_type: GgmlType::Q4K,
                endian: Endian::Little,
            })?,
        },
        &stream,
    )?;
    let packed = Array::from_slice(&encoded, &[OUTPUTS as i32, ROW_BYTES as i32]);
    publish_parameter_replacements(
        &mut linear,
        &[(WEIGHT.into(), MlxTensor::from_array(packed))].into(),
    );
    let inputs: Vec<f32> = (0..PROMPT_ROWS).flat_map(activation_row).collect();
    let expected: Vec<f64> = inputs
        .chunks_exact(INPUTS)
        .flat_map(|input| {
            reference.chunks_exact(INPUTS).map(move |weights| {
                input
                    .iter()
                    .zip(weights)
                    .map(|(&x, &w)| f64::from(x) * w)
                    .sum()
            })
        })
        .collect();

    for (rows, dtype, limit) in [
        (1, Dtype::Float32, 0.00001),
        (PROMPT_ROWS, Dtype::Float32, 0.001),
        (PROMPT_ROWS, Dtype::Float16, 0.001),
    ] {
        let input = Array::from_slice(&inputs[..rows * INPUTS], &[rows as i32, INPUTS as i32]);
        let input = MlxTensor::from_array(input.as_dtype(dtype, &stream)?);
        let output = linear.forward(&input, &stream)?;
        ensure!(
            output.as_array().shape() == [rows as i32, OUTPUTS as i32],
            "unexpected projection shape"
        );
        let output = output.as_array().as_dtype(Dtype::Float32, &stream)?;
        let output = output.evaluated()?;
        let mut worst: f64 = 0.0;
        for (actual, expected) in output
            .as_slice::<f32>()
            .chunks_exact(OUTPUTS)
            .zip(expected.chunks_exact(OUTPUTS))
        {
            let error = actual
                .iter()
                .zip(expected)
                .map(|(&a, &b)| (f64::from(a) - b).powi(2))
                .sum::<f64>();
            let norm = expected.iter().map(|b| b.powi(2)).sum::<f64>();
            let relative_l2 = (error / norm).sqrt();
            ensure!(relative_l2.is_finite(), "non-finite projection error");
            worst = worst.max(relative_l2);
        }
        println!("Metal {dtype:?}, {rows} rows: maximum relative L2 {worst:.9} (limit {limit})");
        ensure!(
            worst <= limit,
            "projection differs from the independent Q4_K contraction"
        );
    }
    Ok(())
}

fn read_weights(path: &Path) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    let digest: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    ensure!(
        digest == SHA256,
        "expected the pinned LiquidAI Q4_K_M artifact"
    );
    let mut gguf = Reader::new(BufReader::new(file))?;
    ensure!(
        gguf.endian() == Endian::Little,
        "expected little-endian GGUF"
    );
    let weight = gguf
        .tensors()
        .iter()
        .find(|tensor| tensor.name == WEIGHT)
        .context("missing projection")?
        .clone();
    ensure!(
        weight.ggml_type == GgmlType::Q4K && weight.dimensions == [INPUTS as u64, OUTPUTS as u64],
        "unexpected Q4_K projection layout"
    );
    Ok(gguf.read_raw(&weight)?)
}

// Q4_K stores eight 32-value groups. The first eight table bytes hold the
// complete low groups and the high bits of the remaining groups.
fn decode(block: &[u8]) -> [f64; BLOCK_VALUES] {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f64();
    let d_min = f16::from_le_bytes([block[2], block[3]]).to_f64();
    let table = &block[4..16];
    let mut values = [0.0; BLOCK_VALUES];
    for group in 0..8 {
        let (scale, min) = if group < 4 {
            (table[group] % 64, table[group + 4] % 64)
        } else {
            let low = table[group + 4];
            (
                16 * (table[group - 4] / 64) + low % 16,
                16 * (table[group] / 64) + low / 16,
            )
        };
        for lane in 0..32 {
            let packed = block[16 + (group / 2) * 32 + lane];
            let code = if group % 2 == 0 {
                packed % 16
            } else {
                packed / 16
            };
            values[group * 32 + lane] =
                d * f64::from(scale) * f64::from(code) - d_min * f64::from(min);
        }
    }
    values
}

// Exactly representable in both FP16 and the signed 8-bit activation blocks.
// Each block has a unique peak of -127/128, fixing the activation scale at 1/128.
fn activation_row(row: usize) -> impl Iterator<Item = f32> {
    (0..INPUTS).map(move |column| {
        let code = if column % BLOCK_VALUES == (row * 11) % BLOCK_VALUES {
            -127
        } else {
            ((column * 53 + row * 29 + column / 7 * 17) % 253) as i32 - 126
        };
        code as f32 / 128.0
    })
}
