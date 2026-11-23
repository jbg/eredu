//! MLX kernels for the blocks selected by portable paged attention.

use super::MlxCacheBackend;
use crate::{
    backend::runtime::cache::kv::{BlockwiseAttentionAccumulator, KeyValueAttentionBlock},
    MlxTensor,
};
use eredu_core::{
    cache::mechanisms::{
        AttentionScan, CacheAttentionAccumulator, PagedAttentionGeometry, PagedAttentionOperands,
        PagedKeyValueStorage,
    },
    BackendFailure,
};
use safemlx::{error::Exception, Array, Stream};

struct MlxCacheAttention {
    accumulator: BlockwiseAttentionAccumulator,
    relative: Option<crate::backend::nn::relative_attention::RelativeAttentionKernel>,
    score_bytes_per_position: u64,
}

impl eredu_core::cache::mechanisms::CacheAttentionAccumulator<Array, Stream> for MlxCacheAttention {
    fn scan(&self) -> AttentionScan {
        self.accumulator.scan()
    }

    fn begin_pass(&mut self, pass: usize, stream: &Stream) -> Result<(), BackendFailure> {
        self.accumulator
            .begin_pass(pass, stream)
            .map_err(BackendFailure::from_error)
    }

    fn accumulate(
        &mut self,
        source: eredu_core::cache::mechanisms::CacheAttentionInput,
        positions: std::ops::Range<i64>,
        [keys, values]: [&Array; 2],
        stream: &Stream,
        retention: impl Sized + 'static,
    ) -> Result<u64, eredu_core::BackendFailure> {
        use eredu_core::cache::mechanisms::CacheAttentionInput;
        let block = KeyValueAttentionBlock::unleased(
            positions.start,
            positions.end,
            keys.clone(),
            values.clone(),
        );
        let score_bytes = self.score_bytes_per_position * (positions.end - positions.start) as u64;
        let mut scratch = score_bytes;
        let mut accumulate = || {
            let bias = match (source, self.relative.as_ref()) {
                (CacheAttentionInput::Transient, _) | (_, None) => None,
                (_, Some(relative)) => {
                    let bias = relative.bias(block.start, block.end, stream)?;
                    scratch = scratch.max(relative.prepared_bytes() + 3 * bias.nbytes() as u64);
                    Some(bias)
                }
            };
            self.accumulator
                .accumulate_with_bias(&block, bias.as_ref(), stream)?;
            if source == CacheAttentionInput::SealedPage {
                self.accumulator.submit()?;
            }
            Ok(())
        };
        if source == CacheAttentionInput::SealedPage {
            super::with_cache_read(stream, retention, accumulate)?;
        } else {
            accumulate().map_err(eredu_core::BackendFailure::from_error)?;
        }
        Ok(scratch.saturating_add(self.accumulator.ordered_workspace_bytes(score_bytes)))
    }

    fn finish(self, stream: &Stream) -> Result<Array, eredu_core::BackendFailure> {
        let output = self
            .accumulator
            .finish(stream)
            .map_err(eredu_core::BackendFailure::from_error)?;
        safemlx::transforms::eval([&output]).map_err(eredu_core::BackendFailure::from_error)?;
        Ok(output)
    }
}

impl MlxCacheAttention {
    fn new(
        operands: PagedAttentionOperands<'_, Array>,
        geometry: PagedAttentionGeometry,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let queries = operands.queries;
        let mut accumulator = BlockwiseAttentionAccumulator::new(
            queries,
            operands.scale,
            operands.mask,
            geometry.query_start,
            geometry.window,
            geometry.prefix,
            operands.sinks,
            geometry.context_end,
            stream,
        )?;
        if let Some(start) = geometry.transient_context_start {
            accumulator.use_retained_context(start)?;
        }
        accumulator.set_softcap(operands.softcap)?;
        accumulator.set_arithmetic(operands.arithmetic, stream)?;
        Ok(Self {
            accumulator,
            relative: None,
            score_bytes_per_position: queries.dim(0) as u64
                * queries.dim(1) as u64
                * queries.dim(2) as u64
                * 4,
        })
    }
}

impl PagedKeyValueStorage for MlxCacheBackend {
    fn describe_paged_append(
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_nn::Error> {
        crate::backend::nn::memory::describe(invocation)
    }

    fn paged_attention(
        operands: PagedAttentionOperands<'_, Array>,
        geometry: PagedAttentionGeometry,
        stream: &Stream,
    ) -> Result<impl CacheAttentionAccumulator<Array, Stream>, BackendFailure> {
        MlxCacheAttention::new(operands, geometry, stream).map_err(BackendFailure::from_error)
    }

    fn paged_relative_attention(
        input: &eredu_nn::RelativeAttentionInput<'_, MlxTensor>,
        geometry: PagedAttentionGeometry,
        stream: &Stream,
    ) -> Result<impl CacheAttentionAccumulator<Array, Stream>, BackendFailure> {
        let relative =
            crate::backend::nn::relative_attention::RelativeAttentionKernel::new(input, stream)
                .map_err(BackendFailure::from_error)?;
        let operands = PagedAttentionOperands {
            queries: &relative.queries,
            scale: 1.0 / relative.queries.dim(3) as f32,
            mask: None,
            sinks: None,
            softcap: None,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
        };
        let mut kernel = MlxCacheAttention::new(operands, geometry, stream)
            .map_err(BackendFailure::from_error)?;
        kernel.relative = Some(relative);
        Ok(kernel)
    }

    fn indexed_positions(
        input: &eredu_nn::IndexedAttentionInput<'_, MlxTensor>,
        batch: i32,
        query: i32,
        stream: &Stream,
    ) -> Result<Vec<Option<i32>>, BackendFailure> {
        crate::backend::nn::attention::indexed_positions(input, batch, query, stream)
            .map_err(BackendFailure::from_error)
    }

    fn attend_indexed_row(
        input: &eredu_nn::IndexedAttentionInput<'_, MlxTensor>,
        batch: i32,
        query: i32,
        slots: &[Option<i32>],
        gathered: Option<[Array; 2]>,
        stream: &Stream,
    ) -> Result<Array, BackendFailure> {
        crate::backend::nn::attention::attend_indexed_row(
            input, batch, query, slots, gathered, stream,
        )
        .map_err(BackendFailure::from_error)
    }
}
