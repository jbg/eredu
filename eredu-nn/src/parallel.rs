//! Portable readiness sequencing for tensor-parallel neural operators.

use super::{DistributedNeuralBackend, Error, NeuralBackend, Tensor, TensorElementType};

/// Native graph inputs for a row-parallel reduction and its local epilogue.
pub struct RowParallelProjection<T> {
    /// Rank-local partial with the selected reduction precision.
    pub partial: T,
    /// Replicated bias to add after the reduction.
    pub bias: Option<T>,
    /// Output precision to restore after the epilogue.
    pub output_cast: Option<TensorElementType>,
}

pub(super) fn tensor_geometry<T: Tensor>(value: &T, words: &mut Vec<u64>) {
    words.extend([
        value.element_type().map_or(u64::MAX, |kind| kind as u64),
        value.shape().len() as u64,
    ]);
    words.extend(value.shape().iter().map(|&extent| extent as u64));
}

pub(super) fn agree_parallel_setup<B: NeuralBackend + ?Sized, T>(
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
    protocol: u64,
    phase: u64,
    local: Result<T, Error>,
    geometry: impl FnOnce(&T, &mut Vec<u64>),
) -> Result<T, Error> {
    if !B::coordinates_parallel_readiness(parallel) {
        return local;
    }
    let mut words = Vec::new();
    if let Ok(value) = &local {
        geometry(value, &mut words);
    }
    let accepted = B::complete_parallel_readiness(
        parallel,
        [protocol, phase, words.len() as u64, 0, 0, 0, 0, 0],
        local.is_ok(),
        context,
    );
    let value = local?;
    if !accepted? {
        return Err(Error::backend(
            "a tensor participant failed local preparation or selected different geometry",
        ));
    }
    for (index, chunk) in words.chunks(6).enumerate() {
        let mut descriptor = [u64::MAX; 8];
        descriptor[0] = protocol;
        descriptor[1] = index as u64;
        descriptor[2..2 + chunk.len()].copy_from_slice(chunk);
        if !B::complete_parallel_readiness(parallel, descriptor, true, context)? {
            return Err(Error::backend("tensor participant geometry differs"));
        }
    }
    Ok(value)
}

pub(super) fn vocabulary_project<B: DistributedNeuralBackend + ?Sized>(
    prepared: Result<(B::Tensor, Vec<usize>), Error>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
    phase: u64,
) -> Result<B::Tensor, Error> {
    let prepared = prepared.and_then(|(local, widths)| {
        B::validate_vocabulary_gather(&local, &widths, parallel)?;
        Ok((local, widths))
    });
    let (local, widths) = agree_parallel_setup::<B, _>(
        parallel,
        context,
        0x564f434142,
        phase,
        prepared,
        |(value, widths), words| {
            words.extend([
                value.element_type().map_or(u64::MAX, |kind| kind as u64),
                value.shape().len() as u64,
            ]);
            words.extend(
                value
                    .shape()
                    .iter()
                    .take(value.shape().len().saturating_sub(1))
                    .map(|&extent| extent as u64),
            );
            words.extend(widths.iter().map(|&width| width as u64));
        },
    )?;
    B::gather_vocabulary(local, &widths, parallel, context)
}
