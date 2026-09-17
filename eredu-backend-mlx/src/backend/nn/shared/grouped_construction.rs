//! One native constructor, with ordinary placeholders or exact acquired rows.
use super::*;
use super::parameters::{compact::{self, CompactBindingCause}, *};
use crate::backend::nn::grouped::ParameterFactory;
use eredu_nn::{Parameter, workspace::WorkspaceMetadataFunding};
use std::mem::{size_of,size_of_val};

#[derive(Clone,Copy)]
pub(super) struct Declaration<'a> {
    pub name: &'static str,
    pub source: &'a ParameterSpec,
    pub weight: Option<&'a ParameterSpec>,
}
impl<'a> Declaration<'a> {
    pub fn parameter(name:&'static str,source:&'a ParameterSpec)->Self {Self{name,source,weight:None}}
    pub fn companion(name:&'static str,weight:&'a ParameterSpec,source:&'a ParameterSpec)->Self {
        Self{name,source,weight:Some(weight)}
    }
}
pub(super) enum Constructor<'stream,'names> {
    Ordinary(&'stream Stream),
    Prepared { values: PreparedCompactBindings<'names>, funding: WorkspaceMetadataFunding },
}
impl<'stream,'names> Constructor<'stream,'names> {
    pub fn ordinary(stream:&'stream Stream)->Self {Self::Ordinary(stream)}
    /// Host-only constructor frames. Acquired values already own their exact
    /// Slice/Concatenate producer; no scalar, Full or Eval is constructed here.
    pub fn control_bytes<T>(rows:usize)->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<T>(),size_of::<Result<T,ComputeError>>(),
            size_of::<[Option<Declaration<'_>>;8]>(),size_of::<Declaration<'_>>(),
            size_of::<[Option<&str>;4]>(),size_of::<[i32;3]>(),size_of::<[i32;2]>(),
            size_of::<[Option<Parameter<MlxTensor>>;4]>(),
            size_of::<(Array,Option<Array>,Option<Array>)>(),
            size_of::<(&str,&[i32],Dtype,Option<&[i32]>)>(),
            size_of::<(usize,usize,bool)>(),size_of::<Option<WorkspaceMetadataFunding>>(),
            size_of::<Result<Option<WorkspaceMetadataFunding>,ComputeError>>(),
            crate::backend::nn::grouped::parameter_factory_control_bytes()?,
            compact::linear_control_bytes()?,
            Array::descriptor_comparison_control_bytes()?.checked_mul(rows)?,
            ComputeError::retained_source_control_bytes::<CompactBindingCause>()?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    pub fn prepared<T>(mut values:PreparedCompactBindings<'names>)->Result<Self,ComputeError> {
        let controls=Self::control_bytes::<T>(values.len());
        values.begin(controls)?;
        if !values.ready() {return Err(values.failure(CompactBindingCause::Identity));}
        let funding=values.funding().clone();
        Ok(Self::Prepared{values,funding})
    }
    pub fn clone_parameter(&self,source:&ParameterSpec,weight:Option<&ParameterSpec>)
        ->Result<ParameterSpec,ComputeError> {
        match self {
            Self::Ordinary(_)=>Ok(match weight {
                Some(weight)=>bind_linear_companion(weight,source.clone()),None=>source.clone(),
            }),
            Self::Prepared{funding,..}=>match weight {
                Some(weight)=>funding.clone_parameter_companion(weight,source),
                None=>funding.clone_parameter_spec(source),
            },
        }
    }
    pub fn finish(self)->Result<Option<WorkspaceMetadataFunding>,ComputeError> {
        match self {
            Self::Ordinary(_)=>Ok(None),
            Self::Prepared{values,funding}=>{
                if !values.consumed(){return Err(values.failure(CompactBindingCause::Identity));}
                Ok(Some(funding))
            }
        }
    }
    pub fn named<M:NativeRetainedValues>(self,module:M,rows:&[Option<Declaration<'_>>])
        ->Result<MlxNamedModule<M>,ComputeError> {
        match &self {
            Self::Ordinary(_)=>{
                let mut topology=Vec::with_capacity(rows.iter().flatten().count());
                for row in rows.iter().flatten() {
                    topology.push((row.name,self.clone_parameter(row.source,row.weight)?));
                }
                self.finish()?;
                MlxNamedModule::with_exact_topology(module,topology)
            }
            Self::Prepared{funding,..}=>{
                let mut topology=PreparedParameterTopology::prepare(rows.iter().flatten().count(),funding)?;
                for row in rows.iter().flatten() {
                    topology.push(row.name,self.clone_parameter(row.source,row.weight)?)?;
                }
                let topology=topology.finish()?;
                self.finish()?;
                MlxNamedModule::with_prepared_topology(module,topology)
            }
        }
    }
}
impl ParameterFactory for Constructor<'_,'_> {
    type Error=ComputeError;
    fn array(&mut self,name:&str,shape:&[i32],dtype:Dtype,floating_shape:Option<&[i32]>)
        ->Result<Array,ComputeError> {
        match self {
            Self::Ordinary(stream)=>compute(safemlx::ops::zeros_dtype(shape,dtype,*stream)),
            Self::Prepared{values,..}=>{
                let value=values.value(name).ok_or_else(||values.failure(CompactBindingCause::Identity))?;
                compact::validate_shape(shape,dtype,value,floating_shape).map_err(|cause|values.failure(cause))?;
                values.take(name).ok_or_else(||values.failure(CompactBindingCause::Identity))
            }
        }
    }
    fn geometry(&self,message:&'static str)->ComputeError {
        match self {Self::Ordinary(_)=>ComputeError::backend(message),
            Self::Prepared{values,..}=>values.failure(CompactBindingCause::Geometry)}
    }
}
