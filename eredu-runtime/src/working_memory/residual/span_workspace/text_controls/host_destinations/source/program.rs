//! Exactly admitted composition of independent original source roots.
use super::*;
use std::mem::{size_of,size_of_val};

/// Immutable component table. The caller supplies its already paid owned table;
/// this plan creates no source bank, native capacity or construction permission.
#[derive(Debug)]
pub struct HostSourceConstructionProgram {
    components:Vec<HostSourceConstructionFacts>,
    facts:HostSourceConstructionFacts,
}
impl HostSourceConstructionProgram {
    /// Compose the actual source facts once. Nested programs must be flattened
    /// by their retained producer before this boundary, never recursively split.
    pub fn from_components(components:Vec<HostSourceConstructionFacts>)->Result<Self,WorkingMemoryError> {
        if components.is_empty()||components.iter().any(|part|part.program.is_some()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut capacity=0u64;let mut attempts=0usize;let mut protected=0u64;
        for part in &components {
            capacity=capacity.checked_add(part.capacity).ok_or(WorkingMemoryError::Overflow)?;
            attempts=attempts.checked_add(part.attempts).ok_or(WorkingMemoryError::Overflow)?;
            protected=protected.checked_add(part.protected).ok_or(WorkingMemoryError::Overflow)?;
        }
        let root=HostSourceConstructionFacts::new(0,0,0)?;
        protected=protected.checked_add(root.protected).and_then(|n|n.checked_add(Self::partition_control_bytes(components.len()).ok()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        use std::sync::atomic::{AtomicU64,Ordering};
        static NEXT:AtomicU64=AtomicU64::new(1);
        let identity=NEXT.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|n|n.checked_add(1))
            .map_err(|_|WorkingMemoryError::Overflow)?;
        Ok(Self{components,facts:HostSourceConstructionFacts{capacity,attempts,partitions:0,protected,program:Some(identity),peak:None}})
    }
    /// Complete selected aggregate, including every source's peak and controls.
    pub const fn facts(&self)->HostSourceConstructionFacts {self.facts}
    /// Read-only source facts in the original declared order.
    pub fn components(&self)->&[HostSourceConstructionFacts] {&self.components}
    /// Named planning frames; the caller additionally pays the actual input Vec.
    pub fn plan_control_bytes()->Option<usize> {
        let parts=[size_of::<Self>(),size_of::<Result<Self,WorkingMemoryError>>(),size_of::<Vec<HostSourceConstructionFacts>>(),
            size_of::<std::slice::Iter<'_,HostSourceConstructionFacts>>(),size_of::<HostSourceConstructionFacts>()*2,
            size_of::<[u64;4]>(),size_of::<usize>(),size_of::<WorkingMemoryError>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    fn partition_control_bytes(count:usize)->Result<u64,WorkingMemoryError> {
        let parts=[size_of::<OriginalHostSourceProgramBanks>(),size_of::<OriginalHostSourceProgramError>(),
            size_of::<Result<OriginalHostSourceProgramBanks,OriginalHostSourceProgramError>>(),
            size_of::<(OriginalHostSourceBank,&Self)>(),size_of::<Option<OriginalHostSourceBank>>(),
            size_of::<(&mut OriginalHostSourceProgramBanks,usize)>(),size_of::<Result<OriginalHostSourceBank,WorkingMemoryError>>(),
            size_of::<std::slice::Iter<'_,HostSourceConstructionFacts>>()];
        let fixed=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add).ok_or(WorkingMemoryError::Overflow)?;
        qualified_storage::array_bytes::<Option<OriginalHostSourceBank>>(count)?
            .checked_add(qualified_storage::vector_control_bytes::<Option<OriginalHostSourceBank>>()?)
            .and_then(|n|n.checked_add(u64::try_from(fixed).ok()?)).ok_or(WorkingMemoryError::Overflow)
    }
}
/// One paid directory of the original declared roots. Each position is moved
/// out once; unused slots and the directory keep their original source account.
#[derive(Debug)]
pub struct OriginalHostSourceProgramBanks {
    banks:Vec<Option<OriginalHostSourceBank>>,
    identity:u64,
    _source:OriginalHostSourceBank,
}
impl OriginalHostSourceProgramBanks {
    /// Authenticate the immutable program, not equal scalar capacities.
    pub fn matches(&self,program:&HostSourceConstructionProgram)->bool {Some(self.identity)==program.facts.program}
    /// Move one exact declared source root without allocating or refunding it.
    pub fn take(&mut self,index:usize)->Result<OriginalHostSourceBank,WorkingMemoryError> {
        self._source.controls.validate_account(self._source.reservation.as_ref())?;
        self.banks.get_mut(index).and_then(Option::take).ok_or(WorkingMemoryError::AlreadyStarted)
    }
}
/// Failed partition keeps the complete actual source account and unspent root.
#[derive(Debug,thiserror::Error)]
#[error("original source program: {cause}")]
pub struct OriginalHostSourceProgramError {
    #[source] cause:WorkingMemoryError,
    _source:OriginalHostSourceBank,
}
impl OriginalHostSourceBank {
    /// The sole transition from an admitted composite root to its declared
    /// components. Components inherit the account but keep their own source
    /// peaks and partition populations; ordinary split children cannot do this.
    pub fn partition_program(mut self,program:&HostSourceConstructionProgram)
        ->Result<OriginalHostSourceProgramBanks,OriginalHostSourceProgramError> {
        let result=(|| {
            self.controls.validate_account(self.reservation.as_ref())?;
            if !self.matches_facts(program.facts)||self.program.is_none()||self.peak_owner.is_some() {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            let mut banks=qualified_storage::vector(program.components.len(),true)?;
            for &facts in &program.components {
                banks.push(Some(Self::new_with_custody(facts,self.reservation.clone(),self.controls.clone())));
            }
            Ok(banks)
        })();
        match result {
            Err(cause)=>Err(OriginalHostSourceProgramError{cause,_source:self}),
            Ok(banks)=>{
                let identity=self.program.take().expect("validated source program");
                self.remaining=0;self.attempts=0;self.partitions=0;
                Ok(OriginalHostSourceProgramBanks{banks,identity,_source:self})
            }
        }
    }
}
