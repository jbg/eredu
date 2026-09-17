//! Exact retained host account for public frame output values.
use super::{RealtimeOutputFrame,RealtimeDecisionDiagnostics};
use crate::{HostMetadataFunding,HostMetadataFundingError};
use std::mem::{size_of,size_of_val};
impl PartialEq for RealtimeOutputFrame {
    fn eq(&self,other:&Self)->bool {
        self.batch==other.batch && self.text_tokens==other.text_tokens
            && self.decision_audio_tokens==other.decision_audio_tokens
            && self.sampled_audio_tokens==other.sampled_audio_tokens
            && self.output_audio_tokens==other.output_audio_tokens && self.diagnostics==other.diagnostics
    }
}
impl Clone for RealtimeOutputFrame {
    fn clone(&self)->Self {self.try_clone().expect("funded frame output clone requires admitted host capacity")}
}
fn vector_bytes<T>(count:usize)->Option<usize> {
    std::alloc::Layout::array::<T>(count).ok()?.size()
        .checked_add(size_of::<Vec<T>>())?.checked_add(size_of::<std::collections::TryReserveError>())
}
fn copy<T:Copy>(source:&[T])->Result<Vec<T>,HostMetadataFundingError> {
    let mut output=Vec::new();
    output.try_reserve_exact(source.len()).map_err(|_|HostMetadataFundingError::Unavailable)?;
    output.extend_from_slice(source);Ok(output)
}
impl RealtimeOutputFrame {
    /// Fixed handoff controls. Payload vectors must already have been built by
    /// the same accepted host source; attaching this account grants no bytes.
    pub fn host_source_handoff_bytes()->Option<usize> {
        let parts=[size_of::<Self>(),size_of::<HostMetadataFunding>(),size_of::<Option<HostMetadataFunding>>(),
            size_of::<Result<Self,HostMetadataFundingError>>(),HostMetadataFunding::reservation_control_bytes()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Retains the source that paid for these output payloads. This is account
    /// custody only, never native completion or an additional allocation grant.
    pub fn with_host_source(mut self,funding:HostMetadataFunding)->Result<Self,HostMetadataFundingError> {
        if self.host_funding.is_some() {return Err(HostMetadataFundingError::Unavailable);}
        // Keep custody inside the output before a fallible reservation so a
        // refused handoff still drops payload vectors before its final account.
        self.host_funding=Some(funding);
        self.host_funding.as_ref().expect("installed host custody").reserve_metadata(
            Self::host_source_handoff_bytes().ok_or(HostMetadataFundingError::Overflow)?)?;
        Ok(self)
    }
    /// Actual finite destination of a deep host clone, including diagnostics.
    pub fn host_clone_bytes(&self)->Option<usize> {
        let parts=[size_of::<Self>(),size_of::<Result<Self,HostMetadataFundingError>>(),
            size_of::<RealtimeDecisionDiagnostics>(),size_of::<HostMetadataFunding>(),
            HostMetadataFunding::reservation_control_bytes()];
        let mut bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?;
        for tokens in [self.text_tokens.as_slice(),self.decision_audio_tokens.as_slice(),self.sampled_audio_tokens.as_slice()]
            .into_iter().chain(self.output_audio_tokens.as_deref()) {
            bytes=bytes.checked_add(vector_bytes::<i32>(tokens.len())?)?;
        }
        bytes=bytes.checked_add(vector_bytes::<RealtimeDecisionDiagnostics>(self.diagnostics.len())?)?;
        for diagnostic in &self.diagnostics {
            bytes=bytes.checked_add(vector_bytes::<usize>(diagnostic.shape().len())?)?
                .checked_add(vector_bytes::<f32>(diagnostic.logits().len())?)?;
        }
        Some(bytes)
    }
    /// Fallible deep clone preserving the actual account and cumulative spend.
    /// Ordinary unaccounted values retain their existing allocation behavior.
    pub fn try_clone(&self)->Result<Self,HostMetadataFundingError> {
        if let Some(funding)=&self.host_funding {
            funding.reserve_metadata(self.host_clone_bytes().ok_or(HostMetadataFundingError::Overflow)?)?;
        }
        let mut diagnostics=Vec::new();
        diagnostics.try_reserve_exact(self.diagnostics.len()).map_err(|_|HostMetadataFundingError::Unavailable)?;
        for diagnostic in &self.diagnostics {
            diagnostics.push(RealtimeDecisionDiagnostics::new(diagnostic.prediction(),copy(diagnostic.shape())?,
                copy(diagnostic.logits())?).map_err(|_|HostMetadataFundingError::Unavailable)?);
        }
        Ok(Self{batch:self.batch,text_tokens:copy(&self.text_tokens)?,
            decision_audio_tokens:copy(&self.decision_audio_tokens)?,sampled_audio_tokens:copy(&self.sampled_audio_tokens)?,
            output_audio_tokens:self.output_audio_tokens.as_deref().map(copy).transpose()?,diagnostics,
            host_funding:self.host_funding.clone()})
    }
}
