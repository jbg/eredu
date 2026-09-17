//! Host source of the existing coordinator, specialized only at the native boundary.
use super::*;
use crate::{GenerationSampler, RealtimeIngressSource, PredictionDirective};
use crate::host_metadata::funded_vec_bytes as vector;
use crate::realtime_payload::{HistoryHostSource,source_vec_bytes};
use eredu_core::{HostMetadataFundingError,RealtimeFrameScheduleState,RealtimeFrameTransition,
    RealtimeSlotCoordinate,RealtimeTargetSource,RealtimeTemporalSource,RealtimeForcedSource,RealtimeFrameSlot};
use eredu_core::realtime::RealtimeSlotTable;
use std::mem::{size_of,size_of_val};

/// Immutable host population projected from the actual validated transition.
/// This owns no tensor, account, source grant, sampler state or execution authority.
#[derive(Debug)]
pub struct RealtimeCoordinatorHostSource {
    history:HistoryHostSource,
    prepared_history_rows:usize,
    schedule_copies:usize,
    schedule_advance:usize,
    prepared_schedule:usize,
    insertion_capacity:usize,
    insertions:usize,
    temporal:usize,
    targets:usize,
    resolved_targets:usize,
    output:Option<usize>,
    generated:usize,
    model:bool,
    diagnostics:bool,
    forced_tail:usize,
    sampler_bytes:usize,
    random_bytes:usize,
    alias_bytes:usize,
}
impl RealtimeCoordinatorHostSource {
    /// Reads the real prebranch history/schedule and the transition validated by
    /// the same executor. Backend alias/random costs come from their actual copy workers.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<P>(ingress:RealtimeIngressSource<'_>,payload:&RealtimePayloadContract,
        history:&RealtimePayloadHistory<P>,schedule:&RealtimeFrameScheduleState,
        transition:&RealtimeFrameTransition,decisions:&RealtimeDecisionExecution,
        samplers:&[GenerationSampler],native_alias_bytes:usize,random_copy_bytes:Option<usize>)
        ->Result<Self,HostMetadataFundingError> {
        let calculate=||->Option<Self> {
            schedule.validate_schedule(ingress.contract().schedule()).ok()?;
            let history_source=history.coordinator_host_source(payload).ok()?;
            if transition.frontier()!=schedule.frontier(){return None;}
            let placements=||transition.input_placements().iter().chain(transition.forced_placements())
                .chain(transition.warmup_padding()).copied();
            let insertion_capacity=transition.input_placements().len()
                .checked_add(transition.forced_placements().len())?
                .checked_add(transition.warmup_padding().len())?;
            // The exact shared sorted insertion replaces duplicate coordinates.
            // Count the same distinct source keys without building a second table.
            let insertions=placements().enumerate().filter(|(i,key)|!placements().take(*i).any(|prior|prior==*key)).count();
            let additional=placements().enumerate().filter(|(i,key)|
                !placements().take(*i).any(|prior|prior==*key) && !history.entries().any(|(prior,_)|prior==*key)).count();
            let prepared_history_rows=history_source.rows.checked_add(additional)?;
            let frame=ingress.frame();let config=ingress.contract().schedule();
            let forced_audio=if frame.forced_generated_audio_tokens().is_some() {
                frame.forced_generated_audio_codebooks().map_or(config.generated_audio_codebooks(),|mask|mask.iter().filter(|v|**v).count())
            }else{0};
            let model=transition.model_call_required();
            let targets=transition.targets().len();
            if model && targets!=samplers.len(){return None;}
            let diagnostics=frame.retains_diagnostics();
            let mut sampler_bytes=0usize;
            if model {for (sampler,target) in samplers.iter().zip(transition.targets()) {
                sampler_bytes=sampler_bytes.checked_add(sampler.host_clone_bytes()?)?;
                if matches!(target.source(),RealtimeTargetSource::Sampled) {
                    sampler_bytes=sampler_bytes.checked_add(sampler.host_sample_bytes()?)?;
                }
            }}
            let forced=transition.targets().iter().filter(|target|!matches!(target.source(),RealtimeTargetSource::Sampled)).count();
            let forced_tail=if model && !diagnostics && decisions.allow_fully_forced_tail_skip {
                transition.targets().iter().rev().take_while(|target|!matches!(target.source(),RealtimeTargetSource::Sampled)).count()
            }else{0};
            let resolved_targets=transition.targets().iter().filter(|target|target.coordinate().is_some()).count();
            let output=transition.output().map(<[RealtimeSlotCoordinate]>::len);
            let mut aliases=history_source.rows.checked_mul(2)?;
            aliases=aliases.checked_add(transition.temporal_inputs().iter().filter(|source|
                matches!(source,RealtimeTemporalSource::Occupied{..})).count())?;
            // Interpreter forced text placement and retained/current-input directives.
            aliases=aliases.checked_add(transition.forced_placements().iter().filter(|key|key.slot()==RealtimeFrameSlot::Text).count())?;
            aliases=aliases.checked_add(transition.targets().iter().filter(|target|matches!(target.source(),
                RealtimeTargetSource::Existing(_)|RealtimeTargetSource::Forced(RealtimeForcedSource::Retained))
                || matches!(target.source(),RealtimeTargetSource::Forced(RealtimeForcedSource::CurrentInput))
                    && target.slot()==RealtimeFrameSlot::Text).count())?;
            if model {
                // Native callback may decline a permitted tail skip; retain its
                // finite executed branch as well as the one terminal skip directory.
                aliases=aliases.checked_add(prepared_history_rows)?
                    .checked_add(forced.checked_mul(2)?)?
                    .checked_add(targets.checked_mul(2)?)?
                    .checked_add(if diagnostics {targets.checked_mul(2)?}else{0})?
                    .checked_add(1)?.checked_add(resolved_targets)?.checked_add(output.unwrap_or(0))?;
            }else{aliases=aliases.checked_add(config.generated_audio_codebooks())?;}
            Some(Self{history:history_source,prepared_history_rows,
                schedule_copies:schedule.host_clone_bytes()?.checked_mul(2)?,
                schedule_advance:schedule.advance_host_bytes(forced_audio,frame.forced_text_tokens().is_some(),transition)?,
                prepared_schedule:config.host_clone_bytes()?,insertion_capacity,insertions,
                temporal:transition.temporal_inputs().len(),targets,resolved_targets,output,
                generated:config.generated_audio_codebooks(),model,diagnostics,forced_tail,sampler_bytes,
                random_bytes:if model {random_copy_bytes.unwrap_or(0)}else{0},
                alias_bytes:aliases.checked_mul(native_alias_bytes)?})
        };
        calculate().ok_or(HostMetadataFundingError::Unavailable)
    }
    /// Specializes only concrete coordinator/driver/descriptor layouts. Native
    /// tensor data, scheduler branch and numerical operation quotas remain separate.
    pub fn control_bytes<B,S,M,T,C,H,F,E,K>(&self)->Option<usize>
    where B:SamplingBackend<Token=T,Logits=T>,B::RandomState:Clone,C:Completion+Clone,
        S:Sampler<B>+Clone,T:Clone,H:RealtimeHostTokenMaterializer<Tensor=T>,
        F:RealtimeFrameTensorMechanisms<Tensor=T>,E:PreparedRealtimeFrameExecutor<B,S,M>,
        K:RealtimeFrameCompletionMechanism<T,M,E::Retained,Completion=C> {
        let bound=self.history.bound();
        let prepared=bound.with_rows(self.prepared_history_rows);
        let parts=[coordinator_header_bytes::<B,S,M,T,C,H,F,E,K>()?,
            self.history.clone_bytes::<T>()?,self.history.bind_bytes(),bound.clone_bytes::<T>()?,
            self.schedule_copies,self.schedule_advance,self.prepared_schedule,
            crate::realtime_interpreter::prepared_host_header_bytes::<T,F::Error>(),
            crate::realtime_interpreter::completed_host_header_bytes::<T,T,F::Error>(),
            RealtimeSlotTable::<T>::construction_bytes(self.insertion_capacity)?,
            bound.overwrite_bytes::<T>(self.insertions)?,vector::<T>(self.temporal)?,
            vector::<PredictionDirective<T>>(self.targets)?,self.sampler_bytes,self.random_bytes,self.alias_bytes];
        let mut bytes=parts.into_iter().try_fold(0usize,usize::checked_add)?;
        if self.model {
            let count=if self.diagnostics {self.targets}else{0};
            for part in [vector::<PredictionDirective<T>>(self.targets)?,vector::<f32>(self.targets)?,
                vector::<S>(self.targets)?,SequentialDecisionDriver::<B,S>::host_source_bytes(self.targets,self.diagnostics,self.forced_tail)?,
                vector::<T>(self.targets)?,vector::<T>(count)?,prepared.clone_bytes::<T>()?,
                vector::<(RealtimeSlotCoordinate,T)>(self.resolved_targets)?,prepared.overwrite_bytes::<T>(self.resolved_targets)?] {
                bytes=bytes.checked_add(part)?;
            }
            if let Some(count)=self.output {
                bytes=bytes.checked_add(source_vec_bytes::<RealtimeSlotCoordinate>(count)?)?
                    .checked_add(source_vec_bytes::<&T>(count)?)?.checked_add(vector::<T>(count)?)?;
            }
        }else{bytes=bytes.checked_add(vector::<T>(self.generated)?)?;}
        Some(bytes)
    }
}
pub(super) fn coordinator_header_bytes<B,S,M,T,C,H,F,E,K>()->Option<usize>
where B:SamplingBackend<Token=T,Logits=T>,B::RandomState:Clone,C:Completion+Clone,
    S:Sampler<B>+Clone,T:Clone,H:RealtimeHostTokenMaterializer<Tensor=T>,
    F:RealtimeFrameTensorMechanisms<Tensor=T>,E:PreparedRealtimeFrameExecutor<B,S,M>,
    K:RealtimeFrameCompletionMechanism<T,M,E::Retained,Completion=C> {
    let parts=[size_of::<RealtimeFrameExecutionView<'_,M,T,S,B::RandomState>>(),
        size_of::<RealtimeFrameExecutionUpdates<T,S,B::RandomState,C>>(),
        size_of::<MaterializedRealtimeInput<T>>(),size_of::<SubmittedRealtimeFrame<T,C>>(),
        size_of::<Option<E::Retained>>(),size_of::<(&H,&F,&E,&K,&B::Context)>(),
        size_of::<Result<SubmittedRealtimeFrame<T,C>,RealtimeFrameCoordinatorError<H::Error,F::Error,E::Error,B::Error,K::Error>>>(),
        eredu_core::HostMetadataFunding::reservation_control_bytes()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{RealtimeSpeechConfig,RealtimeFrameConvention,RealtimeFrameForcing};
    use crate::{TokenDomain,RealtimePayloadGeneration,RealtimePayloadOwnerIdentity};
    // No Clone implementation: a cold source must borrow the actual payload.
    struct Payload;
    #[test]
    fn coordinator_source_borrows_payloads_and_prices_actual_forcing_and_replacements() {
        let config=RealtimeSpeechConfig::new(2,1,1,1,9,8,
            RealtimeFrameConvention::FeedbackAlignedHistory,vec![0,0,1]).unwrap();
        let contract=RealtimePayloadContract::new(config.clone(),1,TokenDomain::new(10),TokenDomain::new(9),
            RealtimePayloadGeneration::new(1).unwrap(),RealtimePayloadOwnerIdentity::new(1).unwrap()).unwrap();
        let mut history=RealtimePayloadHistory::with_contract(contract.clone());
        history.insert(&config,RealtimeSlotCoordinate::new(0,RealtimeFrameSlot::Audio(1)),Payload).unwrap();
        let schedule=RealtimeFrameScheduleState::new(config.clone());
        let mut next=schedule.clone();
        let transition=next.advance(&config,&RealtimeFrameForcing::new(true,vec![false])).unwrap();
        let ingress=RealtimeIngressContract::new(config,TokenDomain::new(10),TokenDomain::new(9)).unwrap();
        let frame=eredu_core::RealtimeInputFrame::new(1,vec![4]).with_forced_text(vec![3]);
        let input=ingress.inspect(&frame).unwrap();
        let samplers=[GenerationSampler::new(),GenerationSampler::new()];
        let decisions=RealtimeDecisionExecution{allow_fully_forced_tail_skip:true};
        let source=RealtimeCoordinatorHostSource::prepare(input,&contract,&history,&schedule,&transition,
            &decisions,&samplers,11,Some(17)).unwrap();
        assert_eq!(source.insertions,2);
        assert_eq!(source.prepared_history_rows,2);
        assert_eq!(source.history.rows,1);
        assert_eq!(source.sampler_bytes,samplers[0].host_clone_bytes().unwrap()
            +samplers[1].host_clone_bytes().unwrap()+samplers[1].host_sample_bytes().unwrap());
        assert_eq!(source.random_bytes,17);
        assert_eq!(source.alias_bytes,15*11);
        assert_eq!(history.len(),1);
        assert!(RealtimeCoordinatorHostSource::prepare(input,&contract,&history,&next,&transition,
            &decisions,&samplers,11,None).is_err());
        assert!(RealtimeCoordinatorHostSource::prepare(input,&contract,&history,&schedule,&transition,
            &decisions,&samplers[..1],11,None).is_err());
    }
}
