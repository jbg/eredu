//! Model-independent sampling coordination using architecture-selected ranks.
//! The driver supplies actual native/source workers; this adds no sampler engine.
use std::mem::{size_of,size_of_val};

/// Invalid selected rank geometry, before any local sample or communication.
#[derive(Clone,Copy,Debug,Eq,PartialEq,thiserror::Error)]
pub enum SamplingSynchronizationError {
    /// The selected/native local rank is outside its actual group.
    #[error("distributed sampling rank is outside the selected group")]
    Rank,
    /// The shared sampling contract requires a positive batch size.
    #[error("distributed sampling batch size must be positive")]
    Batch,
}
/// Checked rank selection supplied by the architecture's retained mechanism.
/// It supplies no group identity, native source or communication admission.
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct SamplingSynchronizationPlan { rank:usize,owner:usize,batch:i32 }
impl SamplingSynchronizationPlan {
    /// Validate existing rank selection against the actual group membership.
    pub fn new(size:usize,rank:usize,owner:usize,batch:i32)->Result<Self,SamplingSynchronizationError>{
        if rank>=size||owner>=size{return Err(SamplingSynchronizationError::Rank);}
        if batch<=0{return Err(SamplingSynchronizationError::Batch);}
        Ok(Self{rank,owner,batch})
    }
    /// Whether this rank runs the shared sampler and advances its RNG/history.
    pub fn is_sampling_rank(self)->bool{self.rank==self.owner}
    /// Native group-local canonical owner, selected by the architecture.
    pub fn sampling_rank(self)->usize{self.owner}
    /// Positive batch extent carried by token synchronization.
    pub fn batch_size(self)->i32{self.batch}
    /// Exact generic driver frames; native/host destinations stay producer-owned.
    pub fn control_bytes<D:SamplingSynchronizationDriver>()->Option<usize>{
        let parts=[size_of::<Self>(),size_of::<&mut D>(),size_of::<D::LocalToken>(),
            size_of::<Option<D::LocalToken>>(),size_of::<D::Contribution>(),size_of::<D::Token>(),
            size_of::<SynchronizedSampling<D::Token>>(),size_of::<D::Error>(),size_of::<bool>(),
            size_of::<Result<D::LocalToken,D::Error>>(),size_of::<Result<D::Contribution,D::Error>>(),
            size_of::<Result<D::Token,D::Error>>(),size_of::<Result<bool,D::Error>>(),
            size_of::<Result<SynchronizedSampling<D::Token>,D::Error>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
/// Native/materialization callbacks for the one shared coordination sequence.
/// Prepared implementations retain exact sources and nonrefunding budgets.
pub trait SamplingSynchronizationDriver {
    /// Actual completed or lazy local result of the existing shared sampler.
    type LocalToken;
    /// One rank's native token/stop contribution in the selected wire type.
    type Contribution;
    /// Final native token after synchronized wire conversion.
    type Token;
    /// Original source-preserving failure, including communication disposition.
    type Error;
    /// Called only on the canonical rank. Other ranks must not advance sampling.
    fn sample_local(&mut self,plan:SamplingSynchronizationPlan)->Result<Self::LocalToken,Self::Error>;
    /// Some(local token) on the owner; None produces the ordinary zero contribution.
    fn token_contribution(&mut self,local:Option<Self::LocalToken>,plan:SamplingSynchronizationPlan)
        ->Result<Self::Contribution,Self::Error>;
    /// First actual Broadcast, resolved through the selected communication authority.
    fn exchange_token(&mut self,value:Self::Contribution,plan:SamplingSynchronizationPlan)
        ->Result<Self::Contribution,Self::Error>;
    /// Preserve the existing wire-to-token conversion after successful completion.
    fn finish_token(&mut self,value:Self::Contribution,plan:SamplingSynchronizationPlan)
        ->Result<Self::Token,Self::Error>;
    /// The owner contributes its stop state; every other rank contributes false.
    fn finished_contribution(&mut self,finished:bool,plan:SamplingSynchronizationPlan)
        ->Result<Self::Contribution,Self::Error>;
    /// Second actual Broadcast and completed scalar read through the same authority.
    fn exchange_finished(&mut self,value:Self::Contribution,plan:SamplingSynchronizationPlan)
        ->Result<bool,Self::Error>;
}
/// The two synchronized results; no native completion assertion is added here.
#[derive(Debug)]
pub struct SynchronizedSampling<T> {
    /// Selected token in the backend's ordinary output representation.
    pub token:T,
    /// Canonical owner's stop state, resolved on every rank.
    pub finished:bool,
}
/// Execute the existing rank/order protocol. Any producer failure stops before
/// the next collective; the backend owns failure agreement/quarantine policy.
pub fn synchronize_sampling<D:SamplingSynchronizationDriver>(plan:SamplingSynchronizationPlan,
    finished:bool,driver:&mut D)->Result<SynchronizedSampling<D::Token>,D::Error>{
    let local=if plan.is_sampling_rank(){Some(driver.sample_local(plan)?)}else{None};
    let contribution=driver.token_contribution(local,plan)?;
    let received=driver.exchange_token(contribution,plan)?;
    let token=driver.finish_token(received,plan)?;
    let contribution=driver.finished_contribution(plan.is_sampling_rank()&&finished,plan)?;
    let finished=driver.exchange_finished(contribution,plan)?;
    Ok(SynchronizedSampling{token,finished})
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Driver { calls:Vec<&'static str>,sampled:usize,fail:bool,stop:Option<bool> }
    impl SamplingSynchronizationDriver for Driver {
        type LocalToken=u32;type Contribution=f32;type Token=u32;type Error=&'static str;
        fn sample_local(&mut self,_:SamplingSynchronizationPlan)->Result<u32,Self::Error>{
            self.calls.push("sample");self.sampled+=1;Ok(37)
        }
        fn token_contribution(&mut self,value:Option<u32>,_:SamplingSynchronizationPlan)->Result<f32,Self::Error>{
            self.calls.push("token contribution");Ok(value.unwrap_or(0) as f32)
        }
        fn exchange_token(&mut self,value:f32,plan:SamplingSynchronizationPlan)->Result<f32,Self::Error>{
            self.calls.push("token exchange");
            assert_eq!(value,if plan.is_sampling_rank(){37.0}else{0.0});
            if self.fail{Err("retained communication failure")}else{Ok(37.0)}
        }
        fn finish_token(&mut self,value:f32,_:SamplingSynchronizationPlan)->Result<u32,Self::Error>{
            self.calls.push("token decode");Ok(value as u32)
        }
        fn finished_contribution(&mut self,value:bool,_:SamplingSynchronizationPlan)->Result<f32,Self::Error>{
            self.calls.push("stop contribution");self.stop=Some(value);Ok(u8::from(value) as f32)
        }
        fn exchange_finished(&mut self,_:f32,_:SamplingSynchronizationPlan)->Result<bool,Self::Error>{
            self.calls.push("stop exchange");Ok(true)
        }
    }
    #[test]
    fn selected_sampling_rank_preserves_history_and_order_and_stops_after_token_failure(){
        for rank in 0..2 {
            let plan=SamplingSynchronizationPlan::new(2,rank,1,1).unwrap();
            let mut driver=Driver{calls:Vec::new(),sampled:0,fail:false,stop:None};
            let output=synchronize_sampling(plan,true,&mut driver).unwrap();
            assert_eq!((output.token,output.finished),(37,true));
            assert_eq!(driver.sampled,usize::from(rank==1));assert_eq!(driver.stop,Some(rank==1));
            assert_eq!(&driver.calls[usize::from(rank==1)..],
                &["token contribution","token exchange","token decode","stop contribution","stop exchange"]);
            assert!(SamplingSynchronizationPlan::control_bytes::<Driver>().unwrap()>0);
            let mut refused=Driver{calls:Vec::new(),sampled:0,fail:true,stop:None};
            assert_eq!(synchronize_sampling(plan,false,&mut refused).unwrap_err(),"retained communication failure");
            assert_eq!(refused.calls.last(),Some(&"token exchange"));assert_eq!(refused.stop,None);
        }
        assert_eq!(SamplingSynchronizationPlan::new(2,2,1,1),Err(SamplingSynchronizationError::Rank));
        assert_eq!(SamplingSynchronizationPlan::new(2,0,1,0),Err(SamplingSynchronizationError::Batch));
    }
}
