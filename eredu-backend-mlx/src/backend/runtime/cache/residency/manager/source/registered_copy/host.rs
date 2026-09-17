//! Original Host rows consume the same copy plan, source pins and destination.
use super::*;
use safemlx::OriginalBufferBudget;
use std::cell::RefCell;

impl PreparedPagedArrayCopy {
    /// Fresh prompt copies mutable arrays while preserving the actual sealed
    /// Host source owner. The canonical destination is still independent; this
    /// branch does not create or reclassify physical backing.
    pub(crate) fn copy_retained_for_resume(
        &mut self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), NativeError>,
    ) -> Result<(), NativeError> {
        self.pending.copy_rows(
            &mut |array| {
                let copy = IsolatedArrayCopy::new(array).copy_retained(stream, roots)?;
                observe(&copy)?;
                Ok(copy)
            },
            &mut |row, _work, context| {
                let source = row.host.as_ref().ok_or_else(|| {
                    NativeError::Neural(context.metadata_source(CacheSourceError::Identity))
                })?;
                let [first, second] = source.each_ref().map(Arc::clone);
                Ok(match row.id.representation {
                    CacheRepresentation::KeyValue => HostCacheBlock::KeyValue {
                        keys: first,
                        values: second,
                    },
                    CacheRepresentation::CompressedLatentRotary => {
                        HostCacheBlock::CompressedLatentRotary {
                            latent: first,
                            rotary_key: second,
                        }
                    }
                })
            },
        )
    }
    /// After the same B admission, before opening its native Scope. Each exact
    /// source row consumes its two declared Host destination slots once.
    pub(crate) fn prepare_host_destinations(
        &mut self,
        copy: &mut PreparedOriginalCopy,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<(), NativeError> {
        let pending = &mut self.pending;
        if !pending.rows.iter().any(|row| row.host.is_some()) {
            return Ok(());
        }
        if pending.copy_budget.is_some() {
            return Err(NativeError::PrefillControl(
                WorkingMemoryError::PreparationAlreadyStarted,
            ));
        }
        pending
            .context
            .charge_metadata(
                control_bytes().ok_or(NativeError::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| NativeError::Neural(cause.into()))?;
        pending.copy_budget = Some(copy.budget().clone());
        pending.copy_custody = Some(copy.retention());
        for (row, work) in pending.rows.iter_mut().zip(&mut pending.host_work) {
            let Some(sources) = &row.host else {
                continue;
            };
            for (index, source) in sources.iter().enumerate() {
                work.loads[index] = Some(PreparedHostArrayCopy::prepare(source, &pending.context)?);
                let plan = SavedHostCopyPlan::from_host(source, environment)
                    .map_err(|cause| NativeError::Neural(pending.context.metadata_source(cause)))?;
                work.stores[index] = Some(copy.prepare_host_destination(plan, environment)?);
            }
        }
        Ok(())
    }
    pub(crate) fn copy_retained_with_host(
        &mut self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), NativeError>,
        observe_host: &mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), NativeError>,
    ) -> Result<(), NativeError> {
        // Borrow the two callback lanes through one cell only while the shared
        // worker is in the corresponding leaf; no borrow crosses native work.
        let observe = RefCell::new(observe);
        let budget = self.pending.copy_budget.clone();
        self.pending.copy_rows(
            &mut |array| {
                let copy = IsolatedArrayCopy::new(array).copy_retained(stream, roots)?;
                (*observe.borrow_mut())(&copy)?;
                Ok(copy)
            },
            &mut |row, work, context| {
                let budget = budget.as_ref().ok_or(NativeError::PrefillControl(
                    WorkingMemoryError::UnknownBound,
                ))?;
                let mut copied = [None, None];
                for index in 0..2 {
                    let load = work.loads[index]
                        .as_mut()
                        .ok_or(NativeError::PrefillControl(
                            WorkingMemoryError::UnknownBound,
                        ))?;
                    let value = load.copy_retained(stream, roots)?;
                    (*observe.borrow_mut())(&value)?;
                    let store = work.stores[index]
                        .as_mut()
                        .ok_or(NativeError::PrefillControl(
                            WorkingMemoryError::UnknownBound,
                        ))?;
                    store.run(&value, stream, roots)?;
                    observe_host(store)?;
                    copied[index] = Some(Arc::clone(store.completed_for(budget).ok_or_else(
                        || NativeError::Neural(context.metadata_source(CacheSourceError::Identity)),
                    )?));
                }
                let [first, second] =
                    copied.map(|value| value.expect("two completed same-budget Host copies"));
                Ok(match row.id.representation {
                    CacheRepresentation::KeyValue => HostCacheBlock::KeyValue {
                        keys: first,
                        values: second,
                    },
                    CacheRepresentation::CompressedLatentRotary => {
                        HostCacheBlock::CompressedLatentRotary {
                            latent: first,
                            rotary_key: second,
                        }
                    }
                })
            },
        )
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<(
            &mut PreparedPagedArrayCopy,
            &mut PreparedOriginalCopy,
            &OriginalCopyEnvironment<'_>,
        )>(),
        size_of::<(&mut PreparedPagedArrayCopy, &Stream, &RefCell<Vec<Array>>)>(),
        size_of::<RefCell<&mut dyn FnMut(&Array) -> Result<(), NativeError>>>(),
        size_of::<std::cell::RefMut<'_, &mut dyn FnMut(&Array) -> Result<(), NativeError>>>(),
        size_of::<&mut dyn FnMut(&PreparedSavedHostCopy) -> Result<(), NativeError>>(),
        size_of::<Option<OriginalBufferBudget>>(),
        size_of::<[Option<Arc<ImmutableHostTransferBuffer>>; 2]>(),
        size_of::<Result<HostCacheBlock, NativeError>>(),
        size_of::<[Arc<ImmutableHostTransferBuffer>; 2]>(),
        size_of::<Result<(), NativeError>>(),
        size_of::<SavedHostCopyPlan>(),
        size_of::<Result<SavedHostCopyPlan, crate::backend::array_copy::OriginalCopyCause>>(),
        size_of::<std::slice::IterMut<'_, Row>>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, Arc<ImmutableHostTransferBuffer>>>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
