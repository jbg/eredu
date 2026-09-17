//! Actual logical subgroup forwarding source, sharing the paired native worker.
use super::*;

pub(crate) struct OriginalGroupExchange<'a> {
    plan: LogicalExchangePlan<'a>,
    order: usize,
    source: RetainedCommunicationSource,
    funding: WorkspaceMetadataFunding,
}
impl OriginalCommunicationSource<'_> {
    pub(crate) fn group_exchange(
        &self,
        order: usize,
    ) -> Result<Option<OriginalGroupExchange<'_>>, Error> {
        let controls = [
            size_of::<OriginalGroupExchange<'_>>(),
            size_of::<Result<Option<OriginalGroupExchange<'_>>, Error>>(),
            size_of::<Option<(&Group, &CommunicationGroupDescriptor, bool)>>(),
            size_of::<
                Result<
                    Option<LogicalExchangePlan<'_>>,
                    crate::backend::runtime::distributed::group::LogicalExchangeCause,
                >,
            >(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?,
        ];
        self.funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        WorkspaceMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        self.validate()?;
        let (group, _, _) = self
            .group(order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        if !self.matches_group(order, group) {
            return Err(failure(Cause::Identity, &self.source, &self.funding));
        }
        if !group.is_logical() {
            return Ok(None);
        }
        let Some(plan) = group
            .logical_collective_exchange_plan()
            .map_err(|cause| failure(Cause::LogicalExchange(cause), &self.source, &self.funding))?
        else {return Ok(None);};
        // Forwarding needs the actual retained all-world participation proof.
        if plan.rounds() > 1 && !self.source.group(order).is_some_and(|(_,wave)|wave) {
            return Err(failure(
                Cause::LogicalWorldTransport,
                &self.source,
                &self.funding,
            ));
        }
        Ok(Some(OriginalGroupExchange {
            plan,
            order,
            source: self.source.clone(),
            funding: self.funding.clone(),
        }))
    }
}
impl OriginalGroupExchange<'_> {
    pub(crate) fn group(&self) -> &Group {
        self.plan.group()
    }
    pub(crate) fn order(&self) -> usize { self.order }
    pub(crate) fn plan(&self)->LogicalExchangePlan<'_>{self.plan}
    pub(crate) fn rounds(&self)->usize{self.plan.rounds()}
    pub(crate) fn round_layout_storage<'a>(
        &'a self,
        source: &'a OriginalCommunicationSource<'_>,
        round: usize,
        shape: &'a [i32],
        dtype: safemlx::Dtype,
    ) -> Result<OriginalRouteLayoutRound<'a>, Error> {
        let order=if self.plan.sends_first(){SubmissionOrder::SendFirst}else{SubmissionOrder::ReceiveFirst};
        OriginalRouteLayoutRound::query(
            source,
            &self.source,
            &self.funding,
            self.plan,
            ExchangeSelection::Group(self.order),
            order,
            round,
            shape,
            dtype,
        )
    }
}

impl OriginalCommunicationSource<'_> {
    /// The exact retained explicit-route step shares the existing native paired
    /// source and completion; an actual no-op step creates no native operation.
    pub(crate) fn group_routed_exchange_layout<'a>(&'a self, order: usize,
        value: usize, step: usize, shape: &'a [i32], dtype: safemlx::Dtype)
        -> Result<Option<super::layout::OriginalRouteLayoutRound<'a>>, Error> {
        reserve(&self.funding, &[
            size_of::<(usize, usize, usize)>(),
            size_of::<crate::backend::runtime::distributed::group::LogicalRoutedPlan<'_>>(),
            size_of::<crate::backend::runtime::distributed::group::LogicalRoutedValue<'_>>(),
            size_of::<Option<LogicalExchangePlan<'_>>>(),
            size_of::<Result<Option<super::layout::OriginalRouteLayoutRound<'_>>, Error>>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        self.validate()?;
        let (group, _, wave) = self.group(order).ok_or_else(|| failure(Cause::Identity, &self.source, &self.funding))?;
        if !self.matches_group(order, group) { return Err(failure(Cause::Identity, &self.source, &self.funding)); }
        let plan = group.logical_routed_plan().map_err(|cause| failure(Cause::LogicalExchange(cause), &self.source, &self.funding))?
            .ok_or_else(|| failure(Cause::Identity, &self.source, &self.funding))?;
        if !std::ptr::eq(plan.group(), group) { return Err(failure(Cause::Identity, &self.source, &self.funding)); }
        let selected = plan.value(value).ok_or_else(|| failure(Cause::LogicalRound, &self.source, &self.funding))?;
        if step >= selected.steps() { return Err(failure(Cause::LogicalRound, &self.source, &self.funding)); }
        let Some(exchange) = selected.exchange(step) else { return Ok(None); };
        if !wave { return Err(failure(Cause::LogicalWorldTransport, &self.source, &self.funding)); }
        let submission = if exchange.sends_first() { SubmissionOrder::SendFirst } else { SubmissionOrder::ReceiveFirst };
        super::layout::OriginalRouteLayoutRound::query(self, &self.source, &self.funding,
            exchange, ExchangeSelection::Group(order), submission, 0, shape, dtype).map(Some)
    }
}

fn overflow() -> Error { Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow) }
fn reserve(funding: &WorkspaceMetadataFunding, parts: &[usize]) -> Result<(), Error> {
    funding.reserve_metadata(parts.iter().copied().try_fold(size_of_val(parts), usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}


impl OriginalCommunicationSource<'_> {
    /// An actual local variable route may have different send/receive extents.
    /// Its two prototypes retain independent source layouts; the selected pair
    /// and completion are still the ordinary group's exact reached exchange.
    pub(crate) fn group_variable_route_layout<'a>(&'a self, order: usize, value: usize,
        step: usize, send_shape: &'a [i32], receive_shape: &'a [i32], dtype: safemlx::Dtype)
        -> Result<Option<super::layout::OriginalRouteLayoutRound<'a>>, Error> {
        reserve(&self.funding, &[
            size_of::<(usize, usize, usize)>(),
            size_of::<crate::backend::runtime::distributed::group::LogicalVariableRoutePlan<'_>>(),
            size_of::<crate::backend::runtime::distributed::group::LogicalVariableRoute<'_>>(),
            size_of::<Option<LogicalExchangePlan<'_>>>(),
            size_of::<Result<Option<super::layout::OriginalRouteLayoutRound<'_>>, Error>>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        self.validate()?;
        let (group, _, _) = self.group(order).ok_or_else(|| failure(Cause::Identity, &self.source, &self.funding))?;
        if !self.matches_group(order, group) { return Err(failure(Cause::Identity, &self.source, &self.funding)); }
        let plan = group.logical_variable_route_plan()
            .map_err(|cause| failure(Cause::LogicalExchange(cause), &self.source, &self.funding))?
            .ok_or_else(|| failure(Cause::Identity, &self.source, &self.funding))?;
        if !std::ptr::eq(plan.group(), group) { return Err(failure(Cause::Identity, &self.source, &self.funding)); }
        let selected = plan.values().nth(value).ok_or_else(|| failure(Cause::LogicalRound, &self.source, &self.funding))?;
        if step >= selected.steps() { return Err(failure(Cause::LogicalRound, &self.source, &self.funding)); }
        let Some(exchange) = selected.exchange(step) else { return Ok(None); };
        let submission = if exchange.sends_first() { SubmissionOrder::SendFirst } else { SubmissionOrder::ReceiveFirst };
        super::layout::OriginalRouteLayoutRound::query_asymmetric(self, &self.source, &self.funding,
            exchange, ExchangeSelection::Group(order), submission, 0, send_shape, receive_shape, dtype).map(Some)
    }
}
