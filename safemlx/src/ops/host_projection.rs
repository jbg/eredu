//! Bounded host compatibility operations.
use crate::error::{Exception, Result};

/// FP32 row projection with 64-byte-aligned BLAS operands.
///
/// Accelerate's one-row dot reduction depends on the input address alignment.
/// Keeping both operands aligned gives a stable reduction independently of the
/// tensor allocator. This operation owns only the selected projection's buffers.
#[cfg(target_vendor = "apple")]
pub fn aligned_f32_row_projection(input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    #[repr(align(64))]
    #[derive(Clone, Copy)]
    struct Line([f32; 16]);
    fn aligned(values: &[f32]) -> Vec<Line> {
        values
            .chunks(16)
            .map(|chunk| {
                let mut line = Line([0.0; 16]);
                line.0[..chunk.len()].copy_from_slice(chunk);
                line
            })
            .collect()
    }
    if input.is_empty() || weight.is_empty() || weight.len() % input.len() != 0 {
        return Err(Exception::custom(
            "invalid aligned FP32 row projection geometry",
        ));
    }
    let width = i32::try_from(input.len())
        .map_err(|_| Exception::custom("aligned FP32 projection width exceeds BLAS limits"))?;
    let rows = i32::try_from(weight.len() / input.len())
        .map_err(|_| Exception::custom("aligned FP32 projection rows exceed BLAS limits"))?;
    let left = aligned(input);
    let right = aligned(weight);
    let mut output = vec![0.0; rows as usize];
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            layout: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
    // SAFETY: 64-byte padding retains contiguous FP32 elements. The checked
    // positive geometry bounds every BLAS read and write within these owned,
    // disjoint buffers; the synchronous call cannot retain their pointers.
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            1,
            rows,
            width,
            1.0,
            left.as_ptr().cast(),
            width,
            right.as_ptr().cast(),
            width,
            0.0,
            output.as_mut_ptr(),
            rows,
        );
    }
    Ok(output)
}
