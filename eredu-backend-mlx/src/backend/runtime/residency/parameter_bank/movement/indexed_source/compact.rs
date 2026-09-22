//! Acquired member rows feed the existing counted compact-parameter binder.
use super::*;
use crate::backend::nn::shared::PreparedCompactBindings;
use eredu_nn::Tensor;

impl OriginalIndexedChunkSource {
    /// Fills exact named destinations from this source's real selected leases.
    /// The caller supplies names from the retained compact constructor. Native
    /// Slice/Concatenate Graph and completion resources belong to that enclosing
    /// invocation; this method funds its actual host/handle producer controls.
    pub(crate) fn with_compact_bindings<'names,T,F>(&self,
        acquisition:&AcquiredParameterGroups,names:&'names[&'names str],stream:&Stream,run:F)
        ->Result<T,Error>
    where F:FnOnce(PreparedCompactBindings<'names>)->Result<T,Error> {
        self.validate_parent(stream)?;
        let b=self.body();
        let same=acquisition.original.as_ref().is_some_and(|source|
            source.0.as_ref().zip(self.0.as_ref()).is_some_and(|(left,right)|Rc::ptr_eq(left,right)));
        if !same || !b.acquired.get() || b.binding_attempted.get() || b.completion_attempted.get()
            || names.is_empty() || names.iter().any(|name|name.is_empty())
            || names.windows(2).any(|pair|pair[0]>=pair[1])
            || acquisition.identities.len()!=acquisition.transfer.leases().len()
            || acquisition.identities.is_empty() {
            return Err(self.failure(Cause::Identity));
        }
        let count=acquisition.identities.len();
        let operands=count.checked_mul(names.len()).ok_or_else(||self.failure(Cause::Overflow))?;
        let clone=safemlx::PreparedArrayClone::control_bytes()
            .and_then(|bytes|bytes.checked_add(Array::inspection_clone_handle_bytes()))
            .ok_or_else(||self.failure(Cause::Overflow))?;
        let fixed=[size_of::<T>(),size_of::<F>(),size_of::<Result<T,Error>>(),
            size_of::<Result<PreparedCompactBindings<'names>,Error>>(),
            size_of::<(&AcquiredParameterGroups,&[&str],&Stream)>(),
            size_of::<Vec<Array>>(),size_of::<Result<Vec<Array>,Error>>(),
            size_of::<[usize;8]>(),size_of::<[i32;2]>(),
            size_of::<safemlx::PreparedArrayClone>(),
            size_of::<Option<(&MlxTensor,usize)>>(),
            size_of::<crate::backend::runtime::residency::manager::ResidentBindingNames<'_>>(),
            size_of::<Result<MlxTensor,eredu_nn::Error>>(),
            size_of::<Result<Array,safemlx::error::Exception>>(),
            size_of::<(Instant,Duration)>(),
            size_of::<std::sync::MutexGuard<'_,ParameterBankStatisticsTable>>(),
            size_of::<(&ParameterBankStatisticsTable,&mut ParameterBankStatistics,usize,u64)>(),
            size_of::<std::sync::MutexGuard<'_,AddressableParameterBank>>(),
            size_of::<std::sync::TryLockError<std::sync::MutexGuard<'_,AddressableParameterBank>>>(),
            size_of::<Result<(),AddressableParameterBankError>>(),
            Layout::array::<Array>(count).map_err(|_|self.failure(Cause::Overflow))?.size(),
            clone.checked_mul(operands).ok_or_else(||self.failure(Cause::Overflow))?,
            safemlx::ops::concatenate_axis_control_bytes().and_then(|bytes|bytes.checked_mul(names.len()))
                .ok_or_else(||self.failure(Cause::Overflow))?,
            Array::descriptor_comparison_control_bytes().and_then(|bytes|bytes.checked_mul(operands))
                .ok_or_else(||self.failure(Cause::Overflow))?,
            eredu_nn::Error::retained_source_construction_bytes::<Failure>().ok_or_else(||self.failure(Cause::Overflow))?];
        let bytes=fixed.into_iter().try_fold(size_of_val(&fixed),usize::checked_add)
            .ok_or_else(||self.failure(Cause::Overflow))?;
        b.funding.reserve_metadata(bytes).map_err(|cause|self.failure(Cause::Funding(cause)))?;
        if b.binding_attempted.replace(true){return Err(self.failure(Cause::Spent));}
        let started=Instant::now();
        let mut bindings=PreparedCompactBindings::new(names.len(),&b.funding)
            .map_err(|cause|self.failure(Cause::Consumer(Error::Neural(cause))))?;
        let mut rows=Vec::new();
        rows.try_reserve_exact(count).map_err(|cause|self.failure(Cause::Allocation(cause)))?;
        // The lexical source loan fixes parameter publication while the exact
        // replacement descriptor or acquired lease is projected. No callback,
        // blocking acquisition or completion runs while this cache is locked.
        b.identity.bank.with_workspace_source(&b.funding,|source| {
            if !source.same_source(&b.identity.bank)
                || source.parameter_revision()!=b.identity.parameter_revision {
                return Err(self.failure(Cause::Identity));
            }
            crate::backend::runtime::residency::parameter_bank::parameters::fill_compact_rows(
                &source, acquisition, names, stream, &b.funding, &mut bindings, &mut rows)
                .map_err(|cause| self.failure(Cause::Compact(cause)))?;
            Ok::<_,Error>(())
        }).map_err(|cause|self.failure(Cause::Bank(cause)))??;
        drop(rows);
        let result=run(bindings).map_err(|cause|self.failure(Cause::Consumer(cause)))?;
        b.bound.set(true);
        // Construction updates only an existing catalog-selected counter row.
        // No callback or native completion runs while this pool is locked.
        let bank=b.identity.bank.inner.try_lock().map_err(|cause|self.failure(Cause::CounterLock {
            busy:matches!(cause,std::sync::TryLockError::WouldBlock),
        }))?;
        bank.record_compact_bank(b.identity.census.bank(),acquisition.pass(),acquisition.scratch_bytes(),started.elapsed())
            .map_err(|cause|self.failure(Cause::Telemetry(cause)))?;
        drop(bank);
        Ok(result)
    }
}
