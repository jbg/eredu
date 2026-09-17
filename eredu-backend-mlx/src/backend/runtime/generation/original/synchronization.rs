//! Native callbacks for the shared architecture-selected sampling rank driver.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::control::BoundSamplingSource;
use eredu_runtime::generation::{SamplingSynchronizationDriver,SamplingSynchronizationPlan,synchronize_sampling};

enum Wire { Input(Array), Completed(f32) }
struct Driver<'context,'stream,'random,F> {
    context:&'context OriginalSamplingContext<'stream>,
    sample:Option<F>,
    random:Option<&'random mut RandomState>,
    source:BoundSamplingSource,
}
impl OriginalSamplingContext<'_> {
    pub(crate) fn sample_synchronized<'random,F>(&self,random:Option<&'random mut RandomState>,sample:F)
        ->Result<Array,Error>
    where F:FnOnce(Option<&'random mut RandomState>)->Result<MlxTensor,Error> {
        self.constructing()?;
        let source=self.synchronization.try_borrow_mut().map_err(|_|Error::PredictionScopeReentrant)?
            .take().ok_or(Error::PredictionScopeUnavailable)?;
        let plan=source.plan();
        let frames=[size_of::<Driver<'_,'_,'random,F>>(),size_of::<F>(),size_of::<Wire>(),
            size_of::<BoundSamplingSource>(),size_of::<Result<Array,Error>>(),
            size_of::<(&Self,Option<&mut RandomState>)>(),
            SamplingSynchronizationPlan::control_bytes::<Driver<'_,'_,'random,F>>()
                .ok_or(Error::PredictionScopeUnavailable)?];
        source.fund(frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
            .ok_or(Error::PredictionScopeUnavailable)?)?;
        if !plan.is_sampling_rank() {
            // Ordinary coordination never calls the sampler on this rank.
            // Consume its already-selected SamplingEvent with a completed zero
            // source and the unchanged key; this neither samples nor refunds.
            self.retain_random(random.as_deref())?;
            let placeholder=source.token_source(0)?;
            let random=self.random_root.borrow_mut().take();
            self.complete(&placeholder,random)?;
            self.phase.set(ReadPhase::Complete);
        }
        let result=synchronize_sampling(plan,false,&mut Driver{
            context:self,sample:Some(sample),random,source,
        })?;
        Ok(result.token)
    }
}
impl<'random,F> SamplingSynchronizationDriver for Driver<'_,'_,'random,F>
where F:FnOnce(Option<&'random mut RandomState>)->Result<MlxTensor,Error> {
    type LocalToken=MlxTensor;
    type Contribution=Wire;
    type Token=Array;
    type Error=Error;
    fn sample_local(&mut self,_:SamplingSynchronizationPlan)->Result<MlxTensor,Error>{
        self.sample.take().ok_or(Error::PredictionScopeUnavailable)?(self.random.take())
    }
    fn token_contribution(&mut self,local:Option<MlxTensor>,plan:SamplingSynchronizationPlan)->Result<Wire,Error>{
        if self.context.phase.get()!=ReadPhase::Complete || local.is_some()!=plan.is_sampling_rank() {
            return Err(Error::PredictionScopeUnavailable);
        }
        // Only the existing completed u32 sampler readout supplies this word.
        // Preserve ordinary Broadcast's u32 -> F32 wire conversion exactly.
        let value=if local.is_some(){self.context.readout.get().ok_or(Error::PredictionScopeUnavailable)? as f32}else{0.0};
        Ok(Wire::Input(self.source.contribution(value,true)?))
    }
    fn exchange_token(&mut self,value:Wire,_:SamplingSynchronizationPlan)->Result<Wire,Error>{
        let Wire::Input(input)=value else{return Err(Error::PredictionScopeUnavailable)};
        Ok(Wire::Completed(self.source.exchange(input,true)?))
    }
    fn finish_token(&mut self,value:Wire,_:SamplingSynchronizationPlan)->Result<Array,Error>{
        let Wire::Completed(value)=value else{return Err(Error::PredictionScopeUnavailable)};
        if value<0.0 || value.fract()!=0.0 || f64::from(value)>f64::from(u32::MAX) {
            return Err(Error::PredictionScopeUnavailable);
        }
        // The native consumer uses the actual completed wire value. This is a
        // source-only scalar copy, with no new lazy graph or cast after sealing.
        self.source.token_source(value as u32)
    }
    fn finished_contribution(&mut self,finished:bool,_:SamplingSynchronizationPlan)->Result<Wire,Error>{
        Ok(Wire::Input(self.source.contribution(if finished{1.0}else{0.0},false)?))
    }
    fn exchange_finished(&mut self,value:Wire,_:SamplingSynchronizationPlan)->Result<bool,Error>{
        let Wire::Input(input)=value else{return Err(Error::PredictionScopeUnavailable)};
        Ok(self.source.exchange(input,false)? != 0.0)
    }
}
