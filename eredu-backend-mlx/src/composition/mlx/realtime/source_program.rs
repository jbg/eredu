//! Original frame ingress and selected parameter reads share one source program.
use crate::backend::error::Error;
use eredu_nn::workspace::{WorkspaceContext,HostMetadataFunding};
use eredu_runtime::working_memory::{HostSourceConstructionFacts,HostSourceConstructionProgram,
    OriginalHostSourceBank,OriginalHostSourceProgramBanks,OriginalHostSourceProgramError,WorkingMemoryError};
use std::mem::{size_of,size_of_val};
fn invalid()->Error {Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn overflow()->Error {Error::PrefillControl(WorkingMemoryError::Overflow)}

pub(super) struct FrameSourceProgram {
    input:HostSourceConstructionFacts,
    operation:Option<HostSourceConstructionFacts>,
    program:Option<HostSourceConstructionProgram>,
}
pub(super) struct FrameSourceBanks {
    pub(super) input:OriginalHostSourceBank,
    pub(super) operation:Option<OriginalHostSourceBank>,
    pub(super) program:Option<OriginalHostSourceProgramBanks>,
}
impl FrameSourceProgram {
    pub(super) fn prepare(input:HostSourceConstructionFacts,operation:Option<HostSourceConstructionFacts>,
        context:&WorkspaceContext)->Result<Self,Error> {
        context.charge_metadata(size_of::<(Self,Result<Self,Error>,Option<HostSourceConstructionProgram>)>()
            .checked_add(HostSourceConstructionProgram::plan_control_bytes().ok_or_else(overflow)?)
            .ok_or_else(overflow)?).map_err(|cause|Error::Neural(cause.into()))?;
        let program=if let Some(operation)=operation {
            let mut components=context.metadata_vec(2).map_err(Error::Neural)?;
            components.push(input);components.push(operation);
            Some(HostSourceConstructionProgram::from_components(components).map_err(Error::PrefillControl)?)
        }else{None};
        Ok(Self{input,operation,program})
    }
    pub(super) fn facts(&self)->HostSourceConstructionFacts {
        self.program.as_ref().map_or(self.input,HostSourceConstructionProgram::facts)
    }
    pub(super) fn control_bytes()->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<FrameSourceBanks>(),size_of::<OriginalHostSourceBank>(),
            size_of::<Option<OriginalHostSourceBank>>(),size_of::<Option<OriginalHostSourceProgramBanks>>(),
            size_of::<Result<FrameSourceBanks,Error>>(),size_of::<(&Self,&HostMetadataFunding)>(),
            WorkspaceContext::metadata_source_bytes::<OriginalHostSourceProgramError>()?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    pub(super) fn accept(&self,bank:OriginalHostSourceBank,funding:&HostMetadataFunding)
        ->Result<FrameSourceBanks,Error> {
        funding.reserve_metadata(Self::control_bytes().ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
        if !bank.matches_facts(self.facts()){return Err(invalid());}
        match &self.program {
            None if self.operation.is_none()=>Ok(FrameSourceBanks{input:bank,operation:None,program:None}),
            Some(plan)=>{
                let mut program=bank.partition_program(plan)
                    .map_err(|cause|Error::Neural(funding.metadata_source(cause)))?;
                let input=program.take(0).map_err(Error::PrefillControl)?;
                let operation=program.take(1).map_err(Error::PrefillControl)?;
                if !input.matches_facts(self.input)||!self.operation.is_some_and(|facts|operation.matches_facts(facts)) {
                    return Err(invalid());
                }
                Ok(FrameSourceBanks{input,operation:Some(operation),program:Some(program)})
            },
            _=>Err(invalid()),
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
