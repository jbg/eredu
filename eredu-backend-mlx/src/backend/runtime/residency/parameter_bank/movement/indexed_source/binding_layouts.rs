//! Actual member-major source descriptors for the independent bank quote.
use super::*;
use eredu_nn::{Tensor,workspace::{WorkspaceAddressableRegionView,WorkspaceContext}};
use eredu_runtime::working_memory::{WorkspaceParameterRow,WorkspaceParameterRows};
use eredu_runtime::parameter_operations::PreparedBankParameterMember;
use crate::backend::runtime::execution::generic::LayerwiseWorkspace;

/// Borrowed exact numerical source; neither alternative grants execution.
pub(crate) enum IndexedBindingStorage<'a> {
    Copy(WorkspaceParameterRow<'a>),
    Replacement { value:&'a MlxTensor, row:usize },
}
pub(crate) struct IndexedBindingLayout<'a> {
    pub(crate) member:&'a PreparedBankParameterMember,
    pub(crate) storage:IndexedBindingStorage<'a>,
}
/// Retains the actual source against which the quoted descriptors were checked.
pub(crate) struct IndexedBindingIdentity {
    source:IndexedBankSource,
    first:Option<AddressableChunkCensus>,
    revision:u64,
    funding:WorkspaceMetadataFunding,
}
impl IndexedBindingIdentity {
    pub(crate) fn source(&self)->&IndexedBankSource{&self.source}
    pub(crate) fn first(&self)->Option<AddressableChunkCensus>{self.first}
    pub(crate) fn revision(&self)->u64{self.revision}
    pub(crate) fn funding(&self)->&WorkspaceMetadataFunding{&self.funding}
}
fn first_from_source(binding:&IndexedBankSource,bank:&AddressableBankSourceLoan<'_>,
    region:WorkspaceAddressableRegionView<'_>,funding:&WorkspaceMetadataFunding,
)->Result<Option<AddressableChunkCensus>,Error> {
    let fail=|cause|failed(cause,&binding.bank,funding,None);
    region.validate().map_err(|_|fail(Cause::Geometry))?;
    let id=usize::try_from(region.bank).map_err(|_|fail(Cause::Geometry))?;
    let (members,maximum)=bank.unit_population(id,region.unit).ok_or_else(||fail(Cause::Geometry))?;
    let access=if region.prefill{ParameterBankAccess::Bulk}else{ParameterBankAccess::Incremental};
    let plan=AddressableChunkPlan::new(region.chunks.rows,region.chunks.routes,members,access,Some(maximum),
        binding.options.prefill_compact_bank_target_bytes()).map_err(|_|fail(Cause::Geometry))?;
    if !bank.same_source(&binding.bank)||plan.workspace_source()!=region.chunks
        || region.compact_scratch_bytes!=bank.scratch_bytes()
        || region.compact_scratch_bytes!=binding.options.compact_bank_scratch_bytes()
        || region.bulk_target_bytes!=binding.options.prefill_compact_bank_target_bytes() {
        return Err(fail(Cause::IdentityAt("selected chunk plan or scratch policy")));
    }
    for (ordinal,(key,_)) in bank.unit_members(id,region.unit).enumerate() {
        let declared=region.local_members.map_or(Some(ordinal),|values|values.get(ordinal).copied());
        if declared!=Some(key.member()){return Err(fail(Cause::IdentityAt("local member order")));}
    }
    Ok(AddressableChunkCensus::new(id,region.unit,plan,0,access))
}
impl IndexedBankSource {
    /// Authenticates the architecture's descriptive chunk plan against this bank.
    pub(crate) fn first_census(&self,region:WorkspaceAddressableRegionView<'_>,funding:&WorkspaceMetadataFunding)
        ->Result<Option<AddressableChunkCensus>,Error> {
        let bytes=AddressableChunkPlan::control_bytes().checked_add(size_of::<(WorkspaceAddressableRegionView<'_>,
            Option<AddressableChunkCensus>,Result<Option<AddressableChunkCensus>,Error>)>())
            .ok_or_else(||failed(Cause::Overflow,&self.bank,funding,None))?;
        funding.reserve_metadata(bytes).map_err(|cause|failed(Cause::Funding(cause),&self.bank,funding,None))?;
        self.with_workspace_source(funding,|bank|first_from_source(self,&bank,region,funding))
            .map_err(|cause|failed(Cause::Bank(cause),&self.bank,funding,None))?
    }
    /// The visitor may copy descriptors into paid storage, but must not reenter
    /// this bank. No acquisition, tensor operation, completion or user callback
    /// is performed while the lexical immutable source loan is held.
    pub(crate) fn with_layouts<F>(&self,region:WorkspaceAddressableRegionView<'_>,context:&WorkspaceContext,
        mut visit:F)->Result<IndexedBindingIdentity,Error>
    where F:for<'a> FnMut(IndexedBindingLayout<'a>)->Result<(),Error> {
        let funding=context.metadata_funding().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
        let fail=|cause|failed(cause,&self.bank,&funding,None);
        let frames=[size_of::<F>(),size_of::<Self>(),size_of::<IndexedBindingIdentity>(),
            size_of::<IndexedBindingLayout<'_>>(),size_of::<Result<IndexedBindingIdentity,Error>>(),
            size_of::<[u8;ParameterBankKey::unit_id_buffer_bytes()]>(),size_of::<[usize;8]>(),
            size_of::<LayerwiseWorkspace>(),size_of::<Result<LayerwiseWorkspace,Error>>(),
            size_of::<WorkspaceParameterRow<'_>>(),size_of::<Option<WorkspaceParameterRow<'_>>>(),
            size_of::<Option<(&MlxTensor,usize)>>(),AddressableChunkPlan::control_bytes()];
        funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||fail(Cause::Overflow))?).map_err(|cause|fail(Cause::Funding(cause)))?;
        self.with_workspace_source(&funding,|bank| {
            let first=first_from_source(self,&bank,region,&funding)?;
            let source=bank.manager().supplementary_residency_source().ok_or_else(||fail(Cause::IdentityAt("selected manager source")))?;
            let workspace=LayerwiseWorkspace::from_supplementary_source(bank.manager(),source,context)?;
            let id=usize::try_from(region.bank).map_err(|_|fail(Cause::Geometry))?;
            for (key,_) in bank.unit_members(id,region.unit) {
                let mut name=[0u8;ParameterBankKey::unit_id_buffer_bytes()];
                let name=key.write_unit_id(&mut name[..key.unit_id_length()]).ok_or_else(||fail(Cause::IdentityAt("unit name")))?;
                let mut unit=None;
                for index in 0..workspace.unit_count() {
                    if workspace.unit(index).map_err(|cause|fail(Cause::Consumer(Error::Neural(context.metadata_source(cause)))))?.id.as_str()==name {
                        if unit.replace(index).is_some(){return Err(fail(Cause::IdentityAt("duplicate unit identity")));}
                    }
                }
                let unit=unit.ok_or_else(||fail(Cause::IdentityAt("missing unit identity")))?;
                let count=workspace.unit(unit).map_err(|cause|fail(Cause::Consumer(Error::Neural(context.metadata_source(cause)))))?.rows;
                let mut seen=0usize;
                for member in bank.members().filter(|member|member.key==key) {
                    let storage=if let Some((value,row))=bank.replacement(member) {
                        if value.shape().first().and_then(|n|usize::try_from(*n).ok()).is_none_or(|rows|row>=rows) {
                            return Err(fail(Cause::Geometry));
                        }
                        IndexedBindingStorage::Replacement{value,row}
                    }else{
                        let mut selected=None;
                        for index in 0..count {
                            let row=workspace.row(unit,index).map_err(|cause|fail(Cause::Consumer(Error::Neural(context.metadata_source(cause)))))?;
                            if row.binding.name()==member.binding {
                                if selected.replace(row).is_some(){return Err(fail(Cause::IdentityAt("duplicate physical binding")));}
                            }
                        }
                        let row=selected.ok_or_else(||fail(Cause::IdentityAt("missing physical binding")))?;
                        if row.shape.first()!=Some(&1)||row.physical_bytes!=member.materialized.byte_len {
                            return Err(fail(Cause::Geometry));
                        }
                        IndexedBindingStorage::Copy(row)
                    };
                    visit(IndexedBindingLayout{member,storage})?;
                    seen=seen.checked_add(1).ok_or_else(||fail(Cause::Overflow))?;
                }
                if seen!=count{return Err(fail(Cause::IdentityAt("physical binding count")));}
            }
            Ok(IndexedBindingIdentity{source:self.clone(),first,revision:bank.parameter_revision(),funding:funding.clone()})
        }).map_err(|cause|fail(Cause::Bank(cause)))?
    }
}
