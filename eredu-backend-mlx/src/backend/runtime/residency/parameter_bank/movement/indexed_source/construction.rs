//! Exact acquired rows feed the existing grouped constructor declarations.
use super::*;
use eredu_nn::{GroupedGatedProductSpec,GroupedLinearSpec,GroupedRelu2Spec};

impl OriginalIndexedChunkSource {
    fn names<'a,const N:usize>(&self,rows:[Option<&'a str>;N])->Result<([&'a str;N],usize),Error> {
        let frames=[size_of::<[Option<&str>;N]>(),size_of::<[&str;N]>(),
            size_of::<Result<([&str;N],usize),Error>>(),size_of::<(usize,usize,&str)>(),
            size_of::<Option<&str>>(),size_of::<(Instant,Duration)>(),
            eredu_nn::Error::retained_source_construction_bytes::<Failure>().ok_or_else(||self.failure(Cause::Overflow))?];
        let bytes=frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||self.failure(Cause::Overflow))?;
        self.body().funding.reserve_metadata(bytes).map_err(|cause|self.failure(Cause::Funding(cause)))?;
        let mut names=["";N];let mut count=0;
        // At most eight borrowed native field names. Stable insertion requires
        // neither string copies nor a sort scratch allocation.
        for name in rows.into_iter().flatten() {
            if name.is_empty(){return Err(self.failure(Cause::Geometry));}
            let mut index=count;
            while index>0 && names[index-1]>name {names[index]=names[index-1];index-=1;}
            if index>0 && names[index-1]==name{return Err(self.failure(Cause::Identity));}
            names[index]=name;count+=1;
        }
        if count==0{return Err(self.failure(Cause::Geometry));}
        Ok((names,count))
    }
    fn validate_groups(&self,acquisition:&AcquiredParameterGroups,groups:i32)->Result<(),Error> {
        if usize::try_from(groups).ok()!=Some(acquisition.identities.len()) {
            return Err(self.failure(Cause::Geometry));
        }
        Ok(())
    }
    pub(crate) fn gated_product_groups(&self,acquisition:&AcquiredParameterGroups,
        spec:&GroupedGatedProductSpec,stream:&Stream)
        ->Result<<MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups,Error> {
        self.validate_groups(acquisition,spec.group_count())?;
        let names=MlxNeuralBackend::gated_compact_parameter_names(spec)
            .map_err(|cause|self.failure(Cause::Consumer(Error::Neural(cause))))?;
        let (names,count)=self.names(names)?;
        self.with_compact_bindings(acquisition,&names[..count],stream,|bindings|{
            let spec=self.body().funding.clone_grouped_gated_product(spec).map_err(Error::Neural)?;
            MlxNeuralBackend::grouped_gated_product_from_bindings(spec,bindings).map_err(Error::Neural)
        })
    }
    pub(crate) fn linear_groups(&self,acquisition:&AcquiredParameterGroups,
        spec:&GroupedLinearSpec,stream:&Stream)
        ->Result<<MlxNeuralBackend as GroupedNeuralBackend>::LinearGroups,Error> {
        self.validate_groups(acquisition,spec.group_count())?;
        let (names,count)=self.names(MlxNeuralBackend::linear_compact_parameter_names(spec))?;
        self.with_compact_bindings(acquisition,&names[..count],stream,|bindings|{
            let spec=self.body().funding.clone_grouped_linear(spec).map_err(Error::Neural)?;
            MlxNeuralBackend::grouped_linear_from_bindings(spec,bindings).map_err(Error::Neural)
        })
    }
    pub(crate) fn relu2_groups(&self,acquisition:&AcquiredParameterGroups,
        spec:&GroupedRelu2Spec,stream:&Stream)
        ->Result<<MlxNeuralBackend as GroupedNeuralBackend>::Relu2Groups,Error> {
        self.validate_groups(acquisition,spec.group_count())?;
        let (names,count)=self.names(MlxNeuralBackend::relu2_compact_parameter_names(spec))?;
        self.with_compact_bindings(acquisition,&names[..count],stream,|bindings|{
            let spec=self.body().funding.clone_grouped_relu2(spec).map_err(Error::Neural)?;
            MlxNeuralBackend::grouped_relu2_from_bindings(spec,bindings).map_err(Error::Neural)
        })
    }
}

use eredu_nn::workspace::ParameterMetadataAllocation;
