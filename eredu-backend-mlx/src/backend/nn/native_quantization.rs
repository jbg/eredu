//! Checkpoint-native quantized storage and device execution.
//!
//! Native tensors keep their physical checkpoint blocks intact and describe
//! logical matrices as zero-copy row views. Backends select kernels from the
//! format, operation, shape, and device; callers remain independent of the
//! originating model architecture. Metal linear kernels reuse each decoded
//! block across a small activation-row tile. CPU execution decodes one packed
//! weight row at a time into bounded scratch space; it never materializes the
//! complete dense matrix or a second persistent affine copy.
//!
//! The bundled MLX C API does not wrap an mmap region as a guaranteed
//! device-accessible buffer. Native loading therefore stores raw bytes in one
//! persistent MLX-owned allocation, while logical views share it through `Arc`.

use std::{cell::RefCell, collections::HashMap, fmt::Write, sync::Arc};

use eredu_gguf::{Endian as GgufEndian, GgmlType, IQuantCodebook};
#[cfg(any(test, not(feature = "cuda")))]
use safemlx::DeviceType;
use safemlx::{
    error::Exception,
    fast::{CustomKernelConfig, MetalKernel},
    ops::matmul,
    transforms::eval,
    Array, Dtype, Stream,
};

const Q4_K_BLOCK_VALUES: i32 = 256;
const Q4_K_BLOCK_BYTES: i32 = 144;
const Q5_K_BLOCK_VALUES: i32 = 256;
const Q5_K_BLOCK_BYTES: i32 = 176;
const Q6_K_BLOCK_VALUES: i32 = 256;
const Q6_K_BLOCK_BYTES: i32 = 210;
const Q5_1_BLOCK_VALUES: i32 = 32;
const Q5_1_BLOCK_BYTES: i32 = 24;
const Q8_0_BLOCK_VALUES: i32 = 32;
const Q8_0_BLOCK_BYTES: i32 = 34;
const OUT_TILE: i32 = 4;
const REDUCTION_TILE: i32 = 32;
const Q4K_DECODE_ROWS_PER_SIMD: i32 = 2;
const Q4K_DECODE_SIMD_GROUPS: i32 = 2;
const NATIVE_BATCH_TILE: i32 = 8;
// A single packed-weight decode is faster for the small tail just above the
// normal tile. Wider per-thread accumulator arrays lose occupancy, so 11+
// rows remain split across the regular eight-row tiles.
const NATIVE_SINGLE_TILE_MAX: i32 = 10;
const Q8_LARGE_OUT_TILE: i32 = 8;
const Q8_LARGE_OUTPUT_ROWS: i32 = 65_536;

thread_local! {
    static Q4K_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_MATMUL_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q4K_EMBEDDING_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5K_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5K_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5K_MATMUL_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q6K_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q6K_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q6K_MATMUL_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q5_1_EMBEDDING_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_BATCH_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_GROUPED_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static Q8_0_EMBEDDING_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static IQ_LINEAR_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_BATCH_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_GROUPED_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
    static IQ_EMBEDDING_KERNEL: RefCell<HashMap<(NativeQuantizationFormat, bool), MetalKernel>> =
        RefCell::new(HashMap::new());
}

fn native_batch_tile(rows: i32) -> i32 {
    if rows > NATIVE_BATCH_TILE && rows <= NATIVE_SINGLE_TILE_MAX {
        rows
    } else {
        NATIVE_BATCH_TILE
    }
}

// Match llama.cpp's small-batch K-quant tiling. Four or five activation rows
// per thread keep accumulator pressure low while a threadgroup shares each
// packed weight chunk across eight output rows.
fn qk_small_batch_tile(rows: i32) -> i32 {
    match rows {
        1..=5 => rows,
        6 => 3,
        7..=8 => 4,
        9 => 3,
        10 => 5,
        _ => NATIVE_BATCH_TILE,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KQuantLinearKernelClass {
    MatrixVector,
    SmallBatch,
    MatrixMatrix,
}

fn k_quant_linear_kernel_class(rows: i32) -> KQuantLinearKernelClass {
    match rows {
        1 => KQuantLinearKernelClass::MatrixVector,
        2..=8 => KQuantLinearKernelClass::SmallBatch,
        _ => KQuantLinearKernelClass::MatrixMatrix,
    }
}

fn k_quant_supports_tiled_matmul(view: &NativeQuantizedTensor) -> bool {
    view.matrix_count == 1 && view.row_start == 0 && view.rows == view.physical_rows
}

mod decode;
mod dispatch;
mod kernels;
mod storage;

pub use dispatch::native_grouped_linear;
pub use storage::{NativeQuantizationFormat, NativeQuantizedTensor};

#[cfg(test)]
mod tests;
