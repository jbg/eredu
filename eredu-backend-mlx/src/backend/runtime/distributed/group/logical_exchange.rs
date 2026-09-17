//! One borrowed selection for ordinary logical exchange and its original source.
use super::*;

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub(crate) enum LogicalExchangeCause {
    #[error("non-neighbor logical pair requires a consensus-proven world participation wave")]
    Participation,
}

/// Minted only from the actual logical group. The private fields cannot be
/// supplied from equal scalar geometry or an unrelated native communicator.
#[derive(Clone, Copy)]
pub(crate) struct LogicalExchangePlan<'a> {
    group: &'a Group,
    rounds: usize,
    destination: usize,
    source: usize,
}
impl<'a> LogicalExchangePlan<'a> {
    pub(crate) fn group(&self) -> &'a Group { self.group }
    pub(crate) fn rounds(&self) -> usize { self.rounds }
    pub(crate) fn peers(&self) -> (usize, usize) { (self.destination, self.source) }
    /// Shared ordered forwarding loop; every iteration consumes its predecessor.
    /// Source callers price the concrete value/callback/error frames before entry.
    pub(crate) fn fold_rounds<T,E,F>(&self,mut value:T,mut step:F)->std::result::Result<T,E>
    where F:FnMut(usize,T)->std::result::Result<T,E> {
        for ordinal in 0..self.rounds { value=step(ordinal,value)?; }
        Ok(value)
    }
    pub(crate) fn iteration_control_bytes<T,E,F>(&self,_step:&F)->Option<usize> {
        let parts=[std::mem::size_of::<&Self>(),std::mem::size_of::<T>(),
            std::mem::size_of::<F>(),std::mem::size_of::<std::result::Result<T,E>>(),
            std::mem::size_of::<std::ops::Range<usize>>(),std::mem::size_of::<usize>()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
    }
    /// Break each actual blocking send/receive cycle using selected physical
    /// endpoints. Direct pairs are complementary; a forwarding ring keeps its
    /// wraparound endpoint on receive-first, with no ordinary fallback.
    pub(crate) fn sends_first(&self)->bool {self.group.native_group().rank()<self.destination}

}
impl Group {
    /// Exactly the existing logical pair choice. None leaves the existing
    /// packed-world equation selected; it does not authorize a native fallback.
    pub(crate) fn logical_exchange_plan(&self)
        -> std::result::Result<Option<LogicalExchangePlan<'_>>, LogicalExchangeCause>
    {
        let Some(logical) = &self.logical else { return Ok(None); };
        if logical.global_ranks.len() != 2 { return Ok(None); }
        let peer = logical.global_ranks[1 - logical.rank];
        let rank = self.native.rank();
        let size = self.native.size();
        let direct = (rank + 1) % size == peer || (peer + 1) % size == rank;
        let (rounds, destination, source) = if direct {
            (1, peer, peer)
        } else if size.is_multiple_of(2) && (rank + size / 2) % size == peer {
            if !logical.world_collective_wave { return Err(LogicalExchangeCause::Participation); }
            (size / 2, (rank + 1) % size, (rank + size - 1) % size)
        } else { return Ok(None); };
        Ok(Some(LogicalExchangePlan { group: self, rounds, destination, source }))
    }
}

impl Group {
    /// The ordinary collective selects explicit routed values before trying a
    /// direct pair. A prepared pair may qualify only that same selected branch.
    pub(crate) fn logical_collective_exchange_plan(&self)
        -> std::result::Result<Option<LogicalExchangePlan<'_>>,LogicalExchangeCause>{
        if self.logical.as_ref().is_some_and(|logical|logical.routes.is_some()) {return Ok(None);}
        self.logical_exchange_plan()
    }
}

/// The actual ordinary packed-world choice. Membership and ordering are borrowed
/// from the selected logical group, never recreated from a rank/size equation.
#[derive(Clone,Copy)]
pub(crate) struct LogicalPackedWorldPlan<'a> {group:&'a Group}
impl LogicalPackedWorldPlan<'_> {
    pub(crate) fn group(&self)->&Group {self.group}
    pub(crate) fn world_size(&self)->usize {self.group.native.size()}
    pub(crate) fn world_rank(&self)->usize {self.group.native.rank()}
    pub(crate) fn members(&self)->&[usize] {&self.group.logical.as_ref().expect("packed logical source").global_ranks}
    pub(crate) fn representative(&self)->usize {self.members()[0]}
}
impl Group {
    pub(crate) fn logical_packed_world_plan(&self)
        ->std::result::Result<Option<LogicalPackedWorldPlan<'_>>,LogicalExchangeCause> {
        let Some(logical)=&self.logical else{return Ok(None);};
        if logical.routes.is_some() || self.logical_exchange_plan()?.is_some(){return Ok(None);}
        if !logical.world_collective_wave {return Err(LogicalExchangeCause::Participation);}
        Ok(Some(LogicalPackedWorldPlan{group:self}))
    }
}

/// The existing variable exchange's physical-world branch. Unlike packed Sum
/// and Gather selection, this branch applies whenever its retained participation
/// wave is present, including groups with explicit logical routes.
#[derive(Clone, Copy)]
pub(crate) struct LogicalVariableWorldPlan<'a> { group: &'a Group }
impl LogicalVariableWorldPlan<'_> {
    pub(crate) fn group(&self) -> &Group { self.group }
    pub(crate) fn world_size(&self) -> usize { self.group.native.size() }
    pub(crate) fn world_rank(&self) -> usize { self.group.native.rank() }
    pub(crate) fn members(&self) -> &[usize] {
        &self.group.logical.as_ref().expect("selected logical variable source").global_ranks
    }
    pub(crate) fn canonical_order(&self) -> bool {
        self.members().windows(2).all(|members| members[0] < members[1])
    }
    pub(crate) fn input_order(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.world_size()).filter_map(|rank| self.members().iter().position(|member| *member == rank))
    }
    pub(crate) fn output_order(&self) -> impl Iterator<Item = usize> + '_ {
        self.members().iter().copied()
    }
}
impl Group {
    pub(crate) fn logical_variable_world_plan(&self) -> Option<LogicalVariableWorldPlan<'_>> {
        self.logical.as_ref().filter(|logical| logical.world_collective_wave)
            .map(|_| LogicalVariableWorldPlan { group: self })
    }
}

/// One existing explicit-route collective itinerary. Every borrowed exchange is
/// minted from the retained route table, including its actual no-op positions.
#[derive(Clone, Copy)]
pub(crate) struct LogicalRoutedPlan<'a> { group: &'a Group }
#[derive(Clone, Copy)]
pub(crate) struct LogicalRoutedValue<'a> { group: &'a Group, route: &'a handle::LogicalRoute }
impl<'a> LogicalRoutedPlan<'a> {
    pub(crate) fn group(&self) -> &'a Group { self.group }
    fn routes(&self) -> &'a [handle::LogicalRoute] {
        self.group.logical.as_ref().and_then(|logical| logical.routes.as_deref())
            .expect("retained explicit-route plan")
    }
    pub(crate) fn len(&self) -> usize { self.routes().len() }
    pub(crate) fn value(&self, index: usize) -> Option<LogicalRoutedValue<'a>> {
        self.routes().get(index).map(|route| LogicalRoutedValue { group: self.group, route })
    }
    pub(crate) fn values(&self) -> impl Iterator<Item = LogicalRoutedValue<'a>> + '_ {
        self.routes().iter().map(|route| LogicalRoutedValue { group: self.group, route })
    }
}
impl<'a> LogicalRoutedValue<'a> {
    pub(crate) fn source_rank(&self) -> usize { self.route.source_rank }
    pub(crate) fn steps(&self) -> usize { self.route.exchanges.len() }
    pub(crate) fn exchange(&self, step: usize) -> Option<LogicalExchangePlan<'a>> {
        self.route.exchanges.get(step).copied().flatten().map(|peer| LogicalExchangePlan {
            group: self.group, rounds: 1, destination: peer, source: peer,
        })
    }
    pub(crate) fn fold_exchanges<T, E, F>(&self, mut value: T, mut step: F) -> std::result::Result<T, E>
    where F: FnMut(usize, LogicalExchangePlan<'a>, T) -> std::result::Result<T, E> {
        for ordinal in 0..self.steps() {
            if let Some(exchange) = self.exchange(ordinal) { value = step(ordinal, exchange, value)?; }
        }
        Ok(value)
    }
    pub(crate) fn iteration_control_bytes<T, E, F>(&self, _step: &F) -> Option<usize> {
        let frames = [std::mem::size_of::<Self>(), std::mem::size_of::<T>(), std::mem::size_of::<F>(),
            std::mem::size_of::<std::result::Result<T, E>>(), std::mem::size_of::<std::ops::Range<usize>>(),
            std::mem::size_of::<Option<LogicalExchangePlan<'_>>>()];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
impl Group {
    pub(crate) fn logical_routed_plan(&self)
        -> std::result::Result<Option<LogicalRoutedPlan<'_>>, LogicalExchangeCause> {
        let Some(logical) = self.logical.as_ref() else { return Ok(None); };
        let Some(routes) = logical.routes.as_ref() else { return Ok(None); };
        if routes.iter().any(|route| route.exchanges.iter().any(Option::is_some))
            && !logical.world_collective_wave { return Err(LogicalExchangeCause::Participation); }
        Ok(Some(LogicalRoutedPlan { group: self }))
    }
}


/// The ordinary variable exchange's local route branch. A world participation
/// wave remains a distinct earlier selection; this cannot invent one.
#[derive(Clone, Copy)]
pub(crate) struct LogicalVariableRoutePlan<'a> { group: &'a Group, routed: bool }
#[derive(Clone, Copy)]
pub(crate) struct LogicalVariableRoute<'a> {
    plan: LogicalVariableRoutePlan<'a>,
    index: usize,
}
impl<'a> LogicalVariableRoutePlan<'a> {
    pub(crate) fn group(&self) -> &'a Group { self.group }
    pub(crate) fn len(&self) -> usize {
        if self.routed {
            self.group.logical.as_ref().and_then(|logical| logical.routes.as_deref())
                .expect("retained variable route selection").len()
        } else { 2 }
    }
    pub(crate) fn values(&self) -> impl ExactSizeIterator<Item = LogicalVariableRoute<'a>> + '_ {
        (0..self.len()).map(move |index| LogicalVariableRoute { plan: *self, index })
    }
}
impl<'a> LogicalVariableRoute<'a> {
    fn route(&self) -> Option<&'a handle::LogicalRoute> {
        self.plan.routed.then(|| &self.plan.group.logical.as_ref()
            .and_then(|logical| logical.routes.as_deref())
            .expect("retained variable route selection")[self.index])
    }
    pub(crate) fn source_rank(&self) -> usize {
        self.route().map_or_else(|| if self.index == 0 { self.plan.group.rank() }
            else { 1 - self.plan.group.rank() }, |route| route.source_rank)
    }
    pub(crate) fn destination_rank(&self) -> usize {
        let rank = self.plan.group.rank(); let size = self.plan.group.size();
        let source = self.source_rank();
        let shift = if rank >= source { rank - source } else { size - (source - rank) };
        // Modular addition without rank + shift overflow.
        if shift >= size - rank { shift - (size - rank) } else { rank + shift }
    }
    pub(crate) fn steps(&self) -> usize {
        self.route().map_or(usize::from(self.index != 0), |route| route.exchanges.len())
    }
    pub(crate) fn exchange(&self, step: usize) -> Option<LogicalExchangePlan<'a>> {
        if step >= self.steps() { return None; }
        let peer = match self.route() {
            Some(route) => route.exchanges[step]?,
            None => self.plan.group.logical.as_ref().expect("retained variable pair")
                .global_ranks[1 - self.plan.group.rank()],
        };
        Some(LogicalExchangePlan { group: self.plan.group, rounds: 1, destination: peer, source: peer })
    }
    pub(crate) fn fold_exchanges<T, E, F>(&self, mut value: T, mut step: F) -> std::result::Result<T, E>
    where F: FnMut(usize, LogicalExchangePlan<'a>, T) -> std::result::Result<T, E> {
        for ordinal in 0..self.steps() {
            if let Some(exchange) = self.exchange(ordinal) { value = step(ordinal, exchange, value)?; }
        }
        Ok(value)
    }
    pub(crate) fn iteration_control_bytes<T, E, F>(&self, _step: &F) -> Option<usize> {
        let frames = [std::mem::size_of::<Self>(), std::mem::size_of::<T>(), std::mem::size_of::<F>(),
            std::mem::size_of::<std::result::Result<T, E>>(), std::mem::size_of::<std::ops::Range<usize>>(),
            std::mem::size_of::<Option<LogicalExchangePlan<'_>>>()];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
impl Group {
    /// Borrows exactly the local route choice used after physical-world variable
    /// exchange selection. Unsupported topology is not a substitute world wave.
    pub(crate) fn logical_variable_route_plan(&self)
        -> std::result::Result<Option<LogicalVariableRoutePlan<'_>>, LogicalExchangeCause> {
        let Some(logical) = &self.logical else { return Ok(None); };
        if logical.world_collective_wave || logical.global_ranks.len() == 1 { return Ok(None); }
        if let Some(routes) = &logical.routes {
            if routes.iter().any(|route| route.exchanges.iter().any(Option::is_some)) {
                return Err(LogicalExchangeCause::Participation);
            }
            return Ok(Some(LogicalVariableRoutePlan { group: self, routed: true }));
        }
        let Some(exchange) = self.logical_exchange_plan()? else { return Ok(None); };
        if exchange.rounds() != 1 { return Err(LogicalExchangeCause::Participation); }
        Ok(Some(LogicalVariableRoutePlan { group: self, routed: false }))
    }
}
