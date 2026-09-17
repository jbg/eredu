//! Status reduction through the retained connected-chain or world-wave pair.
//! Model tensor collectives retain their independently selected wave protocol.
use super::*;

fn chain_start(world: usize, members: &[usize]) -> Option<usize> {
    if world == 0
        || members.is_empty()
        || members.iter().any(|rank| *rank >= world)
        || members.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let mut start = None;
    for index in 0..members.len() {
        let next = (index + 1) % members.len();
        if (members[index] + 1) % world != members[next] {
            if start.is_some() {
                return None;
            }
            start = Some(next);
        }
    }
    Some(start.unwrap_or(0))
}

/// Exact cold membership fact used by both admission and native execution.
pub(crate) fn independent_status_members(world: usize, members: &[usize]) -> bool {
    chain_start(world, members).is_some()
}

/// Immutable coordinates of the actual connected Ring status protocol.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct StatusChainPlan { pub(crate) left:Option<usize>,pub(crate) right:Option<usize> }
impl StatusChainPlan {
    fn prepare(world:usize,members:&[usize],rank:usize)->Option<Self>{
        let start=chain_start(world,members)?;let count=members.len();
        if rank>=count{return None;}
        let ordinal=(rank+count-start)%count;
        Some(Self{left:(ordinal>0).then(||members[(start+ordinal-1)%count]),
            right:(ordinal+1<count).then(||members[(start+ordinal+1)%count])})
    }
    pub(crate) fn control_bytes<O:StatusChainOperations>(self)->Option<usize>{
        use std::mem::size_of;
        let controls=[size_of::<Self>(),size_of::<O::Value>()*2,size_of::<std::result::Result<O::Value,O::Error>>(),size_of::<std::result::Result<(),O::Error>>(),
            size_of::<(&Self,&O::Value,&mut O)>(),size_of::<Option<usize>>(),size_of::<usize>()];
        controls.into_iter().try_fold(std::mem::size_of_val(&controls),usize::checked_add)
    }
    pub(crate) fn sends(self)->usize{usize::from(self.left.is_some())+usize::from(self.right.is_some())}
    pub(crate) fn receives(self)->usize{self.sends()}
    pub(crate) fn accepts(self,operation:safemlx::distributed::GroupWorkerOperation)->bool{
        match operation {safemlx::distributed::GroupWorkerOperation::Send{peer}|safemlx::distributed::GroupWorkerOperation::Receive{peer}=>
            usize::try_from(peer).is_ok_and(|peer|self.left==Some(peer)||self.right==Some(peer)),_=>false}
    }
    pub(crate) fn execute<O:StatusChainOperations>(self,input:&O::Value,ops:&mut O)->std::result::Result<O::Value,O::Error>{
        let mut total=match self.left {Some(peer)=>{let received=ops.receive(input,peer)?;ops.add(input,&received)?},None=>ops.alias(input)?};
        if let Some(peer)=self.right {ops.send(&total,peer)?;total=ops.receive(input,peer)?;}
        if let Some(peer)=self.left {ops.send(&total,peer)?;}
        Ok(total)
    }
}
/// Selected status itinerary. A non-neighbor pair is available only through
/// the ordinary logical exchange plan's already retained world-wave proof.
#[derive(Clone,Copy)]
pub(crate) enum StatusPlan<'a> {
    Chain(StatusChainPlan),
    Exchange { rounds:usize,destination:usize,source:usize,sends_first:bool },
    Routed(LogicalRoutedPlan<'a>),
    Packed(LogicalPackedWorldPlan<'a>),
}
impl PartialEq for StatusPlan<'_> {
    fn eq(&self,other:&Self)->bool {match (self,other) {
        (Self::Chain(a),Self::Chain(b))=>a==b,
        (Self::Exchange{rounds:a,destination:b,source:c,sends_first:d},
            Self::Exchange{rounds:e,destination:f,source:g,sends_first:h})=>(a,b,c,d)==(e,f,g,h),
        (Self::Routed(a),Self::Routed(b))=>std::ptr::eq(a.group(),b.group()),
        (Self::Packed(a),Self::Packed(b))=>std::ptr::eq(a.group(),b.group()),
        _=>false,
    }}
}
impl Eq for StatusPlan<'_>{}
impl std::fmt::Debug for StatusPlan<'_>{
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {match self {
        Self::Chain(plan)=>f.debug_tuple("Chain").field(plan).finish(),
        Self::Exchange{rounds,destination,source,sends_first}=>f.debug_tuple("Exchange").field(&(rounds,destination,source,sends_first)).finish(),
        Self::Routed(plan)=>f.debug_tuple("Routed").field(&plan.len()).finish_non_exhaustive(),
        Self::Packed(plan)=>f.debug_tuple("Packed").field(&plan.world_size()).field(&plan.representative()).finish_non_exhaustive(),
    }}
}
/// Allocation-free traversal of every actual route step and its native leaves.
/// Completed peers are borrowed from the retained topology table, not rebuilt.
pub(crate) struct StatusOperations<'a>{plan:StatusPlan<'a>,route:usize,step:usize,send:bool}
impl Iterator for StatusOperations<'_>{
    type Item=(bool,usize);
    fn next(&mut self)->Option<Self::Item>{
        match self.plan {
            StatusPlan::Routed(plan)=>loop {
                let route=plan.value(self.route)?;
                if self.step>=route.steps(){self.route+=1;self.step=0;continue;}
                let Some(exchange)=route.exchange(self.step) else{self.step+=1;continue};
                let peers=exchange.peers();let sending=self.send;
                if !sending{self.step+=1;}self.send=!sending;
                return Some((sending,if sending{peers.0}else{peers.1}));
            },
            _=>{
                let peer=if self.send{self.plan.send_peer(self.step)}else{self.plan.receive_peer(self.step)}?;
                let sending=self.send;if !sending{self.step+=1;}self.send=!sending;
                Some((sending,peer))
            }
        }
    }
}
impl<'a> StatusPlan<'a> {
    pub(crate) fn operations(self)->StatusOperations<'a>{StatusOperations{plan:self,route:0,step:0,send:true}}
    pub(crate) fn sends(self)->usize {match self {Self::Chain(plan)=>plan.sends(),Self::Exchange{rounds,..}=>rounds,Self::Packed(_)=>0,Self::Routed(_)=>self.operations().filter(|(send,_)|*send).count()}}
    pub(crate) fn receives(self)->usize {self.sends()}
    pub(crate) fn additions(self)->usize {match self {Self::Chain(plan)=>usize::from(plan.left.is_some()),Self::Exchange{..}=>1,Self::Packed(_)=>0,Self::Routed(plan)=>plan.len()-1}}
    pub(crate) fn send_peer(self,index:usize)->Option<usize> {match self {
        Self::Packed(_)=>None,
        Self::Chain(plan)=>[plan.left,plan.right].into_iter().flatten().nth(index),
        Self::Exchange{rounds,destination,..}=>(index<rounds).then_some(destination),
        Self::Routed(_)=>self.operations().filter(|(send,_)|*send).nth(index).map(|(_,peer)|peer),
    }}
    pub(crate) fn receive_peer(self,index:usize)->Option<usize> {match self {
        Self::Packed(_)=>None,
        Self::Chain(plan)=>[plan.left,plan.right].into_iter().flatten().nth(index),
        Self::Exchange{rounds,source,..}=>(index<rounds).then_some(source),
        Self::Routed(_)=>self.operations().filter(|(send,_)|!send).nth(index).map(|(_,peer)|peer),
    }}
    pub(crate) fn accepts(self,operation:safemlx::distributed::GroupWorkerOperation)->bool {match self {
        Self::Packed(_)=>false,
        Self::Chain(plan)=>plan.accepts(operation),
        Self::Routed(_)=>self.operations().any(|(sending,peer)|match operation {
            safemlx::distributed::GroupWorkerOperation::Send{peer:actual}=>sending&&usize::try_from(actual)==Ok(peer),
            safemlx::distributed::GroupWorkerOperation::Receive{peer:actual}=>!sending&&usize::try_from(actual)==Ok(peer),
            _=>false,
        }),
        Self::Exchange{destination,source,..}=>match operation {
            safemlx::distributed::GroupWorkerOperation::Send{peer}=>usize::try_from(peer)==Ok(destination),
            safemlx::distributed::GroupWorkerOperation::Receive{peer}=>usize::try_from(peer)==Ok(source),
            _=>false,
        },
    }}
    pub(crate) fn control_bytes<O:StatusChainOperations>(self)->Option<usize>{
        use std::mem::size_of;
        let parts=[size_of::<Self>(),size_of::<O::Value>()*3,size_of::<std::result::Result<O::Value,O::Error>>(),
            size_of::<std::result::Result<(),O::Error>>(),size_of::<(&Self,&O::Value,&mut O)>(),
            size_of::<std::ops::Range<usize>>(),size_of::<usize>()*4,size_of::<bool>(),
            size_of::<[Option<usize>;2]>(),size_of::<std::array::IntoIter<Option<usize>,2>>(),size_of::<StatusOperations<'_>>(),
            size_of::<Option<O::Value>>(),size_of::<LogicalRoutedPlan<'_>>(),size_of::<LogicalRoutedValue<'_>>(),
            size_of::<Option<LogicalExchangePlan<'_>>>(),size_of::<LogicalPackedWorldPlan<'_>>(),size_of::<[usize;2]>()];
        let controls=parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)?;
        match self {Self::Chain(plan)=>controls.checked_add(plan.control_bytes::<O>()?),Self::Exchange{..}|Self::Routed(_)|Self::Packed(_)=>Some(controls)}
    }
    /// Each primitive settles before returning. The physical wraparound endpoint
    /// receives first, breaking the Ring's blocking send cycle. Every round then
    /// forwards the actual prior received word; no absent participant is invented.
    pub(crate) fn execute<O:StatusChainOperations>(self,input:&O::Value,ops:&mut O)->std::result::Result<O::Value,O::Error>{
        match self {
            Self::Chain(plan)=>plan.execute(input,ops),
            Self::Packed(plan)=>ops.packed_sum(input,plan),
            Self::Routed(plan)=>{
                let mut total=None;
                for index in 0..plan.len() {
                    let route=plan.value(index).expect("retained route ordinal");
                    let mut value=ops.alias(input)?;
                    for step in 0..route.steps() {
                        let Some(exchange)=route.exchange(step)else{continue};
                        let (destination,source)=exchange.peers();
                        if exchange.sends_first(){ops.send(&value,destination)?;value=ops.receive(input,source)?;}
                        else {let received=ops.receive(input,source)?;ops.send(&value,destination)?;value=received;}
                    }
                    total=Some(match total {None=>value,Some(previous)=>ops.add(&previous,&value)?});
                }
                Ok(total.expect("selected routed plan covers its nonempty group"))
            },
            Self::Exchange{rounds,destination,source,sends_first}=>{
                let mut peer=ops.alias(input)?;
                for _ in 0..rounds {
                    if sends_first {ops.send(&peer,destination)?;peer=ops.receive(input,source)?;}
                    else {let received=ops.receive(input,source)?;ops.send(&peer,destination)?;peer=received;}
                }
                ops.add(input,&peer)
            }
        }
    }
}
pub(crate) trait StatusChainOperations {
    type Value;type Error;
    fn packed_sum(&mut self,input:&Self::Value,plan:LogicalPackedWorldPlan<'_>)->std::result::Result<Self::Value,Self::Error>;
    fn alias(&mut self,input:&Self::Value)->std::result::Result<Self::Value,Self::Error>;
    fn receive(&mut self,input:&Self::Value,peer:usize)->std::result::Result<Self::Value,Self::Error>;
    fn add(&mut self,left:&Self::Value,right:&Self::Value)->std::result::Result<Self::Value,Self::Error>;
    fn send(&mut self,value:&Self::Value,peer:usize)->std::result::Result<(),Self::Error>;
}
impl Group {
    pub(crate) fn selected_status_plan(&self)->std::result::Result<Option<StatusPlan<'_>>,LogicalExchangeCause>{
        if let Some(plan)=self.independent_status_plan(){return Ok(Some(StatusPlan::Chain(plan)));}
        // Preserve the ordinary routed-before-pair selection, with its exact
        // route order and inactive steps. The Group constructor proves coverage.
        if let Some(plan)=self.logical_routed_plan()? {
            if plan.len()==0{return Err(LogicalExchangeCause::Participation);}
            return Ok(Some(StatusPlan::Routed(plan)));
        }
        if let Some(plan)=self.logical_collective_exchange_plan()? {
            let (destination,source)=plan.peers();
            return Ok(Some(StatusPlan::Exchange{rounds:plan.rounds(),destination,source,sends_first:plan.sends_first()}));
        }
        Ok(self.logical_packed_world_plan()?.map(StatusPlan::Packed))
    }
    pub(crate) fn independent_status_plan(&self)->Option<StatusChainPlan>{
        let logical=self.logical.as_ref()?;
        StatusChainPlan::prepare(self.native.size(),&logical.global_ranks,logical.rank)
    }
}
pub(super) fn independent_status_sum(input:&Array,group:&Group,stream:&Stream)->Result<Option<Array>>{
    let Some(plan)=group.selected_status_plan().map_err(Exception::from_source)? else{return Ok(None);};
    let policy=if plan.sends()==0{None}else{Some(group.completion_policy().ok_or_else(||Exception::custom("independent status agreement requires bounded completion"))?)};
    let mut ops=OrdinaryStatus{group,stream,policy,started:std::time::Instant::now()};
    plan.execute(input,&mut ops).map(Some)
}
struct OrdinaryStatus<'a>{group:&'a Group,stream:&'a Stream,policy:Option<CommunicationCompletionPolicy>,started:std::time::Instant}
impl StatusChainOperations for OrdinaryStatus<'_>{
    type Value=Array;type Error=Exception;
    fn packed_sum(&mut self,input:&Array,plan:LogicalPackedWorldPlan<'_>)->Result<Array>{
        use crate::backend::nn::logical_collective::{Native,packed};
        let value=packed::pack(&Native(self.stream),input,plan.representative(),plan.world_size())?;
        let completed=super::collectives::ordered_world_sum(&value,self.group,self.stream)?;
        packed::sum_result(&Native(self.stream),&completed,plan.representative())
    }
    fn alias(&mut self,input:&Array)->Result<Array>{Ok(input.clone())}
    fn receive(&mut self,input:&Array,peer:usize)->Result<Array>{
        let received=native::recv_like(input,peer,&self.group.native,self.group.communication_stream(self.stream.as_ref())?)?;
        finish_send(&received,self.group,self.stream,self.policy.expect("nontrivial status receive"),self.started)?;
        Ok(received)
    }
    fn add(&mut self,left:&Array,right:&Array)->Result<Array>{left.add(right,self.stream)}
    fn send(&mut self,value:&Array,peer:usize)->Result<()>{
        let sent=native::send(value,peer,&self.group.native,self.group.communication_stream(self.stream.as_ref())?)?;
        finish_send(&sent,self.group,self.stream,self.policy.expect("nontrivial status chain"),self.started)
    }
}

fn finish_send(
    sent: &Array,
    group: &Group,
    stream: &Stream,
    policy: CommunicationCompletionPolicy,
    started: std::time::Instant,
) -> Result<()> {
    use eredu_core::{BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait};
    let remaining = policy.timeout().saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(Exception::custom(
            "independent status setup deadline exceeded before submission",
        ));
    }
    let completion =
        crate::backend::runtime::distributed::completion::MlxCommunicationCompletion::submit(
            [sent],
            vec![sent.clone()],
            vec![],
            vec![group.clone()],
            vec![],
            vec![stream.clone()],
        )?;
    let remaining = policy
        .timeout()
        .saturating_sub(started.elapsed())
        .max(std::time::Duration::from_nanos(1));
    let wait = BoundedCompletionWait::new(remaining, policy.cancellation())
        .map_err(|error| Exception::custom(error.to_string()))?;
    match completion.wait_bounded(wait)? {
        BoundedCompletionOutcome::Completed => Ok(()),
        BoundedCompletionOutcome::DeadlineExceeded { .. } => Err(Exception::custom("independent status send exceeded its deadline; native resources retained until completion")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires native MLX CPU execution; run explicitly"]
    fn status_intermediate_timeout_retains_and_fences_native_group() {
        use crate::backend::runtime::distributed::completion::{
            ensure_group_available, force_next_communication_pending,
            release_forced_pending_orphans,
        };
        let native = native::init(false, native::Backend::Ring).unwrap();
        let group = Group::uncontracted(&native);
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let value = Array::ones::<i32>(&[1], &stream).unwrap();
        let policy = CommunicationCompletionPolicy::new(
            std::time::Duration::from_millis(5),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        force_next_communication_pending();
        let result = finish_send(&value, &group, &stream, policy, std::time::Instant::now());
        assert!(result
            .unwrap_err()
            .what()
            .contains("native resources retained"));
        assert!(ensure_group_available(&group).is_err());
        release_forced_pending_orphans();
        assert!(ensure_group_available(&group).is_ok());
        finish_send(&value, &group, &stream, policy, std::time::Instant::now()).unwrap();
    }
    #[test]
    fn independent_status_fact_matches_connected_native_ring_membership() {
        for world in 1..=10 {
            for mask in 1usize..1 << world {
                let members = (0..world)
                    .filter(|rank| mask & (1 << rank) != 0)
                    .collect::<Vec<_>>();
                let mut reached = std::collections::BTreeSet::from([members[0]]);
                loop {
                    let previous = reached.len();
                    for rank in reached.clone() {
                        for neighbor in [(rank + 1) % world, (rank + world - 1) % world] {
                            if members.contains(&neighbor) {
                                reached.insert(neighbor);
                            }
                        }
                    }
                    if reached.len() == previous {
                        break;
                    }
                }
                assert_eq!(
                    independent_status_members(world, &members),
                    reached.len() == members.len(),
                    "{world}: {members:?}"
                );
            }
        }
        for (world, members) in [
            (0, vec![0]),
            (4, vec![]),
            (4, vec![0, 4]),
            (4, vec![1, 1]),
            (4, vec![2, 1]),
        ] {
            assert!(!independent_status_members(world, &members));
        }
    }
}

#[cfg(test)]
mod chain_source_tests {
    use super::*;
    #[derive(Debug,PartialEq,Eq,Clone,Copy)]enum Action{Alias,Receive(usize),Add,Send(usize,i32)}
    struct Probe{plan:StatusChainPlan,prefix:i32,total:i32,events:Vec<Action>,fail:Option<usize>}
    impl Probe{fn event(&mut self,action:Action)->std::result::Result<(),usize>{let ordinal=self.events.len();self.events.push(action);if self.fail==Some(ordinal){Err(ordinal)}else{Ok(())}}}
    impl StatusChainOperations for Probe{
        type Value=i32;type Error=usize;
        fn packed_sum(&mut self,_:&i32,_:LogicalPackedWorldPlan<'_>)->std::result::Result<i32,usize>{panic!("chain fixture never selects packed world")}
        fn alias(&mut self,value:&i32)->std::result::Result<i32,usize>{self.event(Action::Alias)?;Ok(*value)}
        fn receive(&mut self,_:&i32,peer:usize)->std::result::Result<i32,usize>{self.event(Action::Receive(peer))?;Ok(if self.plan.left==Some(peer){self.prefix}else{assert_eq!(self.plan.right,Some(peer));self.total})}
        fn add(&mut self,left:&i32,right:&i32)->std::result::Result<i32,usize>{self.event(Action::Add)?;Ok(left+right)}
        fn send(&mut self,value:&i32,peer:usize)->std::result::Result<(),usize>{self.event(Action::Send(peer,*value))}
    }
    #[test]
    fn member_status_chain_preserves_peer_order_payloads_and_first_failure(){
        for world in 1..=8usize{for mask in 1usize..1usize<<world{
            let members=(0..world).filter(|rank|mask&(1<<rank)!=0).collect::<Vec<_>>();
            let Some(start)=chain_start(world,&members) else{continue;};
            let order=(0..members.len()).map(|n|members[(start+n)%members.len()]).collect::<Vec<_>>();
            let total=members.iter().map(|rank|*rank as i32+1).sum::<i32>();
            for rank in 0..members.len(){
                let plan=StatusChainPlan::prepare(world,&members,rank).unwrap();
                let ordinal=order.iter().position(|global|*global==members[rank]).unwrap();
                let input=members[rank] as i32+1;let prefix=order[..ordinal].iter().map(|rank|*rank as i32+1).sum::<i32>();
                let mut probe=Probe{plan,prefix,total,events:vec![],fail:None};
                assert_eq!(plan.execute(&input,&mut probe),Ok(total));
                let mut expected=vec![];
                if let Some(peer)=plan.left{expected.extend([Action::Receive(peer),Action::Add]);}else{expected.push(Action::Alias);}
                if let Some(peer)=plan.right{expected.extend([Action::Send(peer,prefix+input),Action::Receive(peer)]);}
                if let Some(peer)=plan.left{expected.push(Action::Send(peer,total));}
                assert_eq!(probe.events,expected);
                assert_eq!(plan.sends(),expected.iter().filter(|event|matches!(event,Action::Send(..))).count());
                assert_eq!(plan.receives(),expected.iter().filter(|event|matches!(event,Action::Receive(..))).count());
                for failed in 0..expected.len(){let mut probe=Probe{plan,prefix,total,events:vec![],fail:Some(failed)};
                    assert_eq!(plan.execute(&input,&mut probe),Err(failed));assert_eq!(probe.events,&expected[..=failed]);}
            }
        }}
    }
}

#[cfg(test)]
mod world_exchange_tests {
    use super::*;
    use std::sync::mpsc::{Receiver,SyncSender,TrySendError,sync_channel};
    use std::time::{Duration,Instant};
    struct Peer {
        source:usize,destination:usize,send:SyncSender<i32>,receive:Receiver<i32>,
        sends:usize,receives:usize,
    }
    impl StatusChainOperations for Peer {
        type Value=i32;type Error=&'static str;
        fn packed_sum(&mut self,_:&i32,_:LogicalPackedWorldPlan<'_>)->std::result::Result<i32,Self::Error>{Err("pair fixture never selects packed world")}
        fn alias(&mut self,input:&i32)->std::result::Result<i32,Self::Error>{Ok(*input)}
        fn add(&mut self,left:&i32,right:&i32)->std::result::Result<i32,Self::Error>{Ok(left+right)}
        fn receive(&mut self,_:&i32,peer:usize)->std::result::Result<i32,Self::Error>{
            assert_eq!(peer,self.source);self.receives+=1;
            self.receive.recv_timeout(Duration::from_secs(2)).map_err(|_|"receive timeout")
        }
        fn send(&mut self,value:&i32,peer:usize)->std::result::Result<(),Self::Error>{
            assert_eq!(peer,self.destination);self.sends+=1;
            let started=Instant::now();
            loop {match self.send.try_send(*value){
                Ok(())=>return Ok(()),
                Err(TrySendError::Disconnected(_))=>return Err("receiver retired"),
                Err(TrySendError::Full(_))=>{
                    if started.elapsed()>Duration::from_secs(2){return Err("send timeout")}
                    std::thread::yield_now();
                }
            }}
        }
    }
    #[test]
    fn status_world_pair_forwards_every_rank_and_preserves_each_subgroup_result() {
        for world in [4usize,6,8] {for failed in [None,Some(0),Some(world-1)] {
            // Zero-capacity physical links make sends truly rendezvous with
            // receives. The wraparound receive-first rule must break the cycle.
            let (send,mut receive):(Vec<_>,Vec<_>)=(0..world).map(|_|{
                let (send,receive)=sync_channel(0);(send,Some(receive))
            }).unzip();
            let handles:Vec<_>=send.into_iter().enumerate().map(|(rank,send)|{
                let source=(rank+world-1)%world;let destination=(rank+1)%world;
                let receive=receive[source].take().unwrap();
                std::thread::spawn(move ||{
                    let plan=StatusPlan::Exchange{rounds:world/2,destination,source,sends_first:rank<destination};
                    let mut peer=Peer{source,destination,send,receive,sends:0,receives:0};
                    let result=plan.execute(&i32::from(failed!=Some(rank)),&mut peer).unwrap();
                    assert_eq!(peer.sends,plan.sends());assert_eq!(peer.receives,plan.receives());
                    let opposite=(rank+world/2)%world;
                    assert_eq!(result,i32::from(failed!=Some(rank))+i32::from(failed!=Some(opposite)));
                })
            }).collect();
            for thread in handles{thread.join().unwrap();}
        }}
    }
}
