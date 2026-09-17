//! Control-only provider waves reuse the selected vote source constructor.
use super::*;
pub(crate) struct ExpertProviderWaveQuote {
    pub(crate) declaration:eredu_nn::workspace::WorkspaceExpertProviderWave,
    pub(crate) provider:ExpertProviderQuote,
    source:OriginalParallelSource,
}
impl ExpertProviderWaveQuote {
    pub(crate) fn prepare(source:&OriginalParallelSource,declaration:eredu_nn::workspace::WorkspaceExpertProviderWave,
        mechanism:ResidentExecutionMechanisms)->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,Result<Self,Error>,WorkspaceContext)>())?;
        let provider=ExpertProviderQuote::prepare_for(source,mechanism,declaration.groups())?;
        Ok(Self{declaration,provider,source:source.clone()})
    }
    pub(crate) fn source(&self)->&OriginalParallelSource{&self.source}
}
