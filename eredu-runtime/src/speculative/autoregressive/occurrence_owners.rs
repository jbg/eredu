//! The existing request cursors, held outside all mutable cache checkpoints.
use super::*;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeRequestId};
use std::{cell::RefCell, mem::{size_of, size_of_val}, num::NonZeroUsize};

pub(super) enum OccurrenceOwners<'a> {
    Uninstalled,
    Single(RefCell<AutoregressiveOccurrenceCursor<'a>>),
    Table {
        cursors: SpeculativeBuffer<RefCell<AutoregressiveOccurrenceCursor<'a>>>,
        // Table/caller transports retire after the actual cursor destination.
        _host: HostPreparationAuthority,
    },
}
impl<'a> OccurrenceOwners<'a> {
    pub(super) fn prepare<M: AutoregressiveMechanisms>(
        plans: SpeculativeBuffer<AutoregressiveSchedulePlan<'a>>,
        capacity: NonZeroUsize,
        context: M::Context<'_>,
    ) -> Result<Self, M::Error> {
        let controls = [size_of::<Self>(), size_of::<Result<Self, M::Error>>(),
            size_of::<(NonZeroUsize, M::Context<'_>)>(), size_of::<HostPreparationAuthority>(),
            size_of::<SpeculativeBuffer<AutoregressiveSchedulePlan<'a>>>(),
            size_of::<eredu_core::SpeculativeBufferIntoIter<AutoregressiveSchedulePlan<'a>>>(),
            size_of::<SpeculativeBuffer<RefCell<AutoregressiveOccurrenceCursor<'a>>>>(),
            size_of::<Result<SpeculativeBuffer<RefCell<AutoregressiveOccurrenceCursor<'a>>>, M::Error>>(),
            size_of::<Option<SpeculativeRequestId>>(),
            size_of::<Result<Option<&RefCell<AutoregressiveOccurrenceCursor<'a>>>, AutoregressiveOccurrenceError>>(),
            size_of::<Result<(), eredu_core::GenerationError>>(),
        ];
        let host = M::driver_host_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add), context)?;
        if plans.is_empty() || plans.iter().any(|plan|
            plan.selected().requirements().strategy().proposal_capacity() != capacity) {
            return Err(M::occurrence_error(AutoregressiveOccurrenceError::Selection));
        }
        let mut cursors = M::driver_buffer(plans.len(), context)?;
        for plan in plans {
            cursors.try_push(RefCell::new(plan.into_cursor()))
                .map_err(|_| M::occurrence_error(AutoregressiveOccurrenceError::Geometry))?;
        }
        Ok(Self::Table { cursors, _host: host })
    }

    pub(super) fn select(&self, request: Option<SpeculativeRequestId>)
        -> Result<Option<&RefCell<AutoregressiveOccurrenceCursor<'a>>>, AutoregressiveOccurrenceError>
    {
        match self {
            Self::Uninstalled => Ok(None),
            Self::Single(cursor) => Ok(Some(cursor)),
            Self::Table { cursors, .. } => request.and_then(|request| cursors.get(request.index()))
                .map(Some).ok_or(AutoregressiveOccurrenceError::Selection),
        }
    }
}
