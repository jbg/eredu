//! Source-bound initialized inputs for the canonical realtime frame coordinator.
use crate::{backend::{error::Error,initialized_input},MlxTensor};
use eredu_nn::workspace::{WorkspaceMetadataFunding,WorkspaceMetadataFundingError};
use eredu_runtime::{RealtimeIngressSource,RealtimeInputMatrix,RealtimeHostTokenMaterializer,
    RealtimePayloadKind,MaterializedRealtimeInput,working_memory::{HostSourceConstructionFacts,
    OriginalHostSourceBank,OriginalHostSourceCustody,WorkingMemoryError}};
use safemlx::{PreparedInputRuntime,PreparedInputPlan,PreparedInputCause,Array};
use std::mem::{size_of,size_of_val};

const KINDS:[RealtimePayloadKind;3]=[RealtimePayloadKind::InputAudio,
    RealtimePayloadKind::ForcedAudio,RealtimePayloadKind::ForcedText];

#[derive(Debug,thiserror::Error)]
pub(crate) enum RealtimeInputSourceError {
    #[error(transparent)] Input(#[from] PreparedInputCause),
    #[error(transparent)] Memory(#[from] WorkingMemoryError),
    #[error(transparent)] Metadata(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)] Backend(#[from] Error),
}
fn overflow()->RealtimeInputSourceError {WorkingMemoryError::Overflow.into()}
fn identity()->RealtimeInputSourceError {WorkingMemoryError::IdentityMismatch.into()}
fn sum(parts:&[usize])->Option<usize> {
    parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
}
/// Describes only the actual initialized source producer. It is neither a full
/// frame quote nor an accepted bank, and remains tied to the borrowed frame.
pub(crate) struct RealtimeInputSource<'a> {
    runtime:&'a PreparedInputRuntime,
    ingress:RealtimeIngressSource<'a>,
    facts:HostSourceConstructionFacts,
}
impl<'a> RealtimeInputSource<'a> {
    pub(crate) fn inspect(runtime:&'a PreparedInputRuntime,ingress:RealtimeIngressSource<'a>)
        ->Result<Self,RealtimeInputSourceError> {
        let mut bytes=0u64;
        let controls=matrix_control_bytes().ok_or_else(overflow)?;
        for kind in KINDS {
            if let Some(matrix)=ingress.matrix(kind) {
                let shape=matrix.shape();
                let plan=runtime.i32(matrix.values(),&shape)?;
                let (storage,_)=initialized_input::storage_bytes(&plan,controls)?;
                bytes=bytes.checked_add(storage).ok_or_else(overflow)?;
            }
        }
        let facts=HostSourceConstructionFacts::new(bytes,ingress.matrices(),0)?;
        Ok(Self{runtime,ingress,facts})
    }
    pub(crate) fn facts(&self)->HostSourceConstructionFacts {self.facts}
    /// The real adapter/policy frames, separate from each source constructor.
    pub(crate) fn control_bytes(&self)->Result<usize,RealtimeInputSourceError> {
        sum(&[size_of::<Self>(),size_of::<RealtimeInputMaterializer<'_>>(),
            size_of::<(Self,OriginalHostSourceBank,OriginalHostSourceCustody,&WorkspaceMetadataFunding)>(),
            size_of::<Result<RealtimeInputMaterializer<'_>,RealtimeInputSourceError>>(),
            size_of::<(&mut RealtimeInputMaterializer<'_>,&RealtimeIngressSource<'_>)>(),
            size_of::<Result<(),RealtimeInputSourceError>>(),
            size_of::<MaterializedRealtimeInput<MlxTensor>>(),
            size_of::<Result<MaterializedRealtimeInput<MlxTensor>,RealtimeInputSourceError>>(),
            size_of::<[RealtimePayloadKind;3]>(),size_of::<usize>(),
            self.ingress.policy_storage_bytes().ok_or_else(overflow)?,
            eredu_core::HostMetadataFunding::reservation_control_bytes(),
        ]).ok_or_else(overflow)
    }
    /// Consumes a direct partition of the enclosing admitted frame's bank.
    /// Equal facts alone are insufficient: the original account must also match.
    pub(crate) fn accept(self,bank:OriginalHostSourceBank,custody:OriginalHostSourceCustody,
        funding:&'a WorkspaceMetadataFunding)->Result<RealtimeInputMaterializer<'a>,RealtimeInputSourceError> {
        if !bank.belongs_to_source(&custody) || !bank.matches_facts(self.facts) {return Err(identity());}
        funding.reserve_metadata(self.control_bytes()?)?;
        Ok(RealtimeInputMaterializer{source:self,bank,custody,started:false,cursor:0,failed:false})
    }
}
fn matrix_control_bytes()->Option<usize> {
    sum(&[size_of::<(&mut RealtimeInputMaterializer<'_>,RealtimeInputMatrix<'_>)>(),
        size_of::<RealtimeInputMatrix<'_>>(),size_of::<Option<RealtimeInputMatrix<'_>>>(),
        size_of::<[usize;2]>(),size_of::<PreparedInputPlan<'_>>(),
        size_of::<Result<PreparedInputPlan<'_>,PreparedInputCause>>(),
        size_of::<Result<Array,Error>>(),size_of::<Result<MlxTensor,RealtimeInputSourceError>>(),
        size_of::<RealtimeInputSourceError>(),size_of::<usize>()])
}
/// Finite sequential source use, with the original bank retained on refusals.
pub(crate) struct RealtimeInputMaterializer<'a> {
    source:RealtimeInputSource<'a>,
    bank:OriginalHostSourceBank,
    custody:OriginalHostSourceCustody,
    started:bool,
    cursor:usize,
    failed:bool,
}
impl RealtimeInputMaterializer<'_> {
    pub(crate) fn finish(self)->Result<(),RealtimeInputSourceError> {
        if !self.started || self.failed || KINDS[self.cursor..].iter()
            .any(|kind|self.source.ingress.matrix(*kind).is_some()) {Err(identity())} else {Ok(())}
    }
}
impl RealtimeHostTokenMaterializer for RealtimeInputMaterializer<'_> {
    type Tensor=MlxTensor;
    type Error=RealtimeInputSourceError;
    fn prepare_source(&mut self,source:&RealtimeIngressSource<'_>)->Result<(),Self::Error> {
        if self.started || self.failed || !self.source.ingress.same_source(source) {
            self.failed=true;return Err(identity());
        }
        self.started=true;
        Ok(())
    }
    fn materialize_matrix(&mut self,matrix:RealtimeInputMatrix<'_>)->Result<MlxTensor,Self::Error> {
        if !self.started || self.failed {self.failed=true;return Err(identity());}
        while self.cursor<KINDS.len() && self.source.ingress.matrix(KINDS[self.cursor]).is_none() {
            self.cursor+=1;
        }
        let expected=KINDS.get(self.cursor).and_then(|kind|self.source.ingress.matrix(*kind));
        let same=expected.is_some_and(|value|value.kind()==matrix.kind() && value.shape()==matrix.shape()
            && std::ptr::eq(value.values(),matrix.values()));
        if !same {self.failed=true;return Err(identity());}
        // A refused actual attempt remains spent; it cannot replay this source.
        self.cursor+=1;self.failed=true;
        let shape=matrix.shape();
        let plan=self.source.runtime.i32(matrix.values(),&shape)?;
        let value=initialized_input::construct(&mut self.bank,&self.custody,plan,
            matrix_control_bytes().ok_or_else(overflow)?)?;
        self.failed=false;
        Ok(MlxTensor::from_array(value))
    }
    fn materialize_i32(&mut self,_:&[i32],_:[usize;2])->Result<MlxTensor,Self::Error> {
        self.failed=true;Err(identity())
    }
}
