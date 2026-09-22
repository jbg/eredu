//! Borrowed, completed mutable tails from the actual paged state owner.
use super::*;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::CacheBlockSelection;

pub(crate) struct PromptCacheTail<'a> {
    pub(super) manager: &'a CacheResidencyManager,
    pub(super) id: CacheBlockId,
    pub(super) arrays: [&'a Array; 2],
}
impl<'a> PromptCacheTail<'a> {
    pub(crate) fn from_source(
        manager: &'a CacheResidencyManager,
        id: CacheBlockId,
        arrays: [Option<&'a Array>; 2],
        context: &WorkspaceContext,
    ) -> Result<Option<Self>, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(std::mem::size_of::<(
                Self,
                [Option<&Array>; 2],
                Result<Option<Self>, CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let arrays = match arrays {
            [None, None] if id.start == id.end => return Ok(None),
            [Some(first), Some(second)] if id.start < id.end => [first, second],
            _ => return Err(fail(CacheSourceError::Identity)),
        };
        let bytes = arrays
            .iter()
            .try_fold(0u64, |sum, array| {
                sum.checked_add(u64::try_from(array.nbytes()).ok()?)
            })
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        if id.session_id != manager.session_id() {
            return Err(fail(CacheSourceError::Identity));
        }
        manager.with_source_loan(
            CacheBlockSelection::new(id.global_layer, id.representation, id.start, id.end, 0),
            context,
            |source| {
                if source.tail()
                    != Some(eredu_runtime::cache::MutableCacheTail { bytes, end: id.end })
                {
                    return Err(fail(CacheSourceError::Identity));
                }
                Ok(())
            },
        )?;
        Ok(Some(Self {
            manager,
            id,
            arrays,
        }))
    }
}
