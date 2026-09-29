//! One table owner publishes compact row results to a tensor-parallel group.
use std::{borrow::Borrow, marker::PhantomData};

use eredu_nn::{DistributedNeuralBackend, ParameterId, Tensor};

use super::{admit, RowLookupError, RowLookupProvider, RowLookupSpec};
use crate::ParameterBankAccess;

/// Row lookup whose retained table provider exists on exactly one group member.
///
/// All participants must call with identical specifications, row IDs, ordering
/// and access mode. Preparation supplies group-local rank/owner identities and
/// the same request bound on every rank. Only the owner acquires or decodes table
/// rows; peers allocate the compact result shape. A sum broadcasts that result.
///
/// A status reduction precedes the data reduction so a local admission, source,
/// or allocation failure does not leave another rank waiting for table values.
/// Its separately retained status group includes every participant in a scheduled
/// collective wave, including idle pipeline stages; the data group stays local.
/// Source leases follow the underlying provider's completed-row contract; the
/// returned compact tensor is retained by the collective graph independently.
pub struct TensorParallelRowLookup<B, L, G>
where
    B: DistributedNeuralBackend,
{
    spec: RowLookupSpec,
    maximum_requests: usize,
    provider: Option<L>,
    parallel: G,
    status_parallel: G,
    marker: PhantomData<fn() -> B>,
}

impl<B, L, G> TensorParallelRowLookup<B, L, G>
where
    B: DistributedNeuralBackend,
    L: RowLookupProvider<B>,
    G: Borrow<B::ParallelContext>,
{
    /// Retains one exact table declaration and group ownership. Nonowners pass
    /// `None` and never need a table source or residency cache. `status_parallel`
    /// covers all ranks whose scheduled next operation depends on lookup success;
    /// ordinary tensor-only execution supplies the same context for both groups.
    pub fn new(
        spec: RowLookupSpec,
        maximum_requests: usize,
        provider: Option<L>,
        parallel: G,
        status_parallel: G,
        rank: usize,
        owner: usize,
    ) -> Result<Self, RowLookupError> {
        spec.validate()?;
        let size = B::parallel_size(parallel.borrow());
        if size == 0
            || B::parallel_size(status_parallel.borrow()) == 0
            || B::parallel_size(status_parallel.borrow()) > i32::MAX as usize
            || B::parallel_rank(status_parallel.borrow())
                >= B::parallel_size(status_parallel.borrow())
            || size > i32::MAX as usize
            || rank >= size
            || rank != B::parallel_rank(parallel.borrow())
            || owner >= size
            || provider.is_some() != (rank == owner)
            || maximum_requests == 0
            || maximum_requests > i32::MAX as usize
        {
            return Err(RowLookupError::Geometry);
        }
        if provider
            .as_ref()
            .is_some_and(|provider| !provider.has_row_parameter(&spec.parameter))
        {
            return Err(RowLookupError::Missing(spec.parameter.clone()));
        }
        Ok(Self {
            spec,
            maximum_requests,
            provider,
            parallel,
            status_parallel,
            marker: PhantomData,
        })
    }
}

impl<B, L, G> RowLookupProvider<B> for TensorParallelRowLookup<B, L, G>
where
    B: DistributedNeuralBackend,
    L: RowLookupProvider<B>,
    G: Borrow<B::ParallelContext>,
{
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool {
        parameter == &self.spec.parameter
    }

    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RowLookupError> {
        let local = (|| {
            if spec != &self.spec {
                return Err(RowLookupError::Specification(spec.parameter.clone()));
            }
            admit("requests", rows.len() as u64, self.maximum_requests as u64)?;
            if rows.is_empty() {
                return Err(RowLookupError::Geometry);
            }
            for &row in rows {
                if row >= spec.rows {
                    return Err(RowLookupError::OutOfRange {
                        row,
                        rows: spec.rows,
                    });
                }
            }
            let shape = [rows.len() as i32, spec.dimensions];
            let value = if let Some(provider) = self.provider.as_mut() {
                provider.lookup_rows(spec, rows, access, context)?
            } else {
                B::Tensor::full_f32(0., &shape, context)?.cast_float(spec.output_type, context)?
            };
            if value.shape() != shape || value.element_type() != Some(spec.output_type) {
                return Err(RowLookupError::Specification(spec.parameter.clone()));
            }
            Ok(value)
        })();
        let failed = B::Tensor::full_i32(i32::from(local.is_err()), &[1], context)?;
        let failed = B::sum_parallel(failed, self.status_parallel.borrow(), context)?;
        let failed = failed.to_i32_vec(context)?;
        if failed.len() != 1 || failed[0] < 0 {
            return Err(RowLookupError::Geometry);
        }
        if failed[0] != 0 {
            return match local {
                Err(error) => Err(error),
                Ok(_) => Err(RowLookupError::ParallelPeerFailure),
            };
        }
        // A singleton data group has no peers. In a pipeline wave, idle stages
        // participate in the status sum but do not schedule a singleton data sum.
        if B::parallel_size(self.parallel.borrow()) == 1 {
            local
        } else {
            B::sum_parallel(local?, self.parallel.borrow(), context).map_err(RowLookupError::from)
        }
    }
}
